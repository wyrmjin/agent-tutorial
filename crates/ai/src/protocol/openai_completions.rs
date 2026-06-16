//! OpenAI-compatible `/chat/completions` protocol (codec).
//!
//! Migrated from the former `deepseek.rs`.

use serde::Serialize;

use crate::error::AiError;
use crate::message::{Message, ToolSpec};
use crate::protocol::{Protocol, ProtocolKind, SamplingParams};
use crate::stream::StreamDecoder;

pub struct OpenAiCompletionsProtocol;

impl OpenAiCompletionsProtocol {
    pub fn new() -> Self {
        OpenAiCompletionsProtocol
    }
}

impl Default for OpenAiCompletionsProtocol {
    fn default() -> Self {
        Self::new()
    }
}

// ── request body types (Serialize) ──────────────────────────────────────────

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<ApiToolOwned>,
    messages: Vec<ApiMessage>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
}

#[derive(Serialize)]
struct ApiMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ToolCallDelta>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Serialize, Clone)]
struct ToolCallDelta {
    index: u32,
    id: String,
    #[serde(rename = "type")]
    call_type: String,
    function: FunctionDelta,
}

#[derive(Serialize, Clone)]
struct FunctionDelta {
    name: String,
    arguments: String,
}

#[derive(Serialize)]
struct ApiToolOwned {
    #[serde(rename = "type")]
    tool_type: String,
    function: ApiFunctionOwned,
}

#[derive(Serialize)]
struct ApiFunctionOwned {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

fn build_api_messages(messages: &[Message]) -> Vec<ApiMessage> {
    messages
        .iter()
        .map(|msg| match msg {
            Message::System { content } => ApiMessage {
                role: "system".to_string(),
                content: if content.is_empty() { None } else { Some(content.clone()) },
                tool_calls: None,
                tool_call_id: None,
            },
            Message::User { content } => ApiMessage {
                role: "user".to_string(),
                content: if content.is_empty() { None } else { Some(content.clone()) },
                tool_calls: None,
                tool_call_id: None,
            },
            Message::Assistant { content, tool_calls } => {
                let tc_deltas = tool_calls.as_ref().map(|tcs| {
                    tcs.iter()
                        .enumerate()
                        .map(|(i, tc)| ToolCallDelta {
                            index: i as u32,
                            id: tc.id.clone(),
                            call_type: "function".to_string(),
                            function: FunctionDelta {
                                name: tc.name.clone(),
                                arguments: tc.input.to_string(),
                            },
                        })
                        .collect()
                });
                ApiMessage {
                    role: "assistant".to_string(),
                    content: if content.is_empty() { None } else { Some(content.clone()) },
                    tool_calls: tc_deltas,
                    tool_call_id: None,
                }
            }
            Message::Tool { content, tool_call_id } => ApiMessage {
                role: "tool".to_string(),
                content: if content.is_empty() { None } else { Some(content.clone()) },
                tool_calls: None,
                tool_call_id: Some(tool_call_id.clone()),
            },
        })
        .collect()
}

fn build_api_tools(tools: &[ToolSpec]) -> Vec<ApiToolOwned> {
    tools
        .iter()
        .map(|t| ApiToolOwned {
            tool_type: "function".to_string(),
            function: ApiFunctionOwned {
                name: t.name.clone(),
                description: t.description.clone(),
                parameters: t.parameters.clone(),
            },
        })
        .collect()
}

impl Protocol for OpenAiCompletionsProtocol {
    fn kind(&self) -> ProtocolKind {
        ProtocolKind::OpenAiCompletions
    }

    fn endpoint_path(&self) -> &str {
        "/chat/completions"
    }

    fn protocol_headers(&self) -> Vec<(String, String)> {
        // content-type 由 transport 的 .json() 设置, 无需在此重复。
        Vec::new()
    }

    fn build_body(
        &self,
        model_id: &str,
        params: &SamplingParams,
        messages: &[Message],
        tools: &[ToolSpec],
    ) -> Result<serde_json::Value, AiError> {
        let request = ChatRequest {
            model: model_id.to_string(),
            tools: build_api_tools(tools),
            messages: build_api_messages(messages),
            stream: true,
            max_tokens: params.max_tokens,
            temperature: params.temperature,
        };
        serde_json::to_value(&request).map_err(|e| AiError::Encode(e.to_string()))
    }

    fn new_decoder(&self) -> Box<dyn StreamDecoder> {
        // 在 Task 7 实现, 暂时 unimplemented 以便先验证 build_body。
        unimplemented!("decoder added in Task 7")
    }
}

#[cfg(test)]
mod build_body_tests {
    use super::OpenAiCompletionsProtocol;
    use crate::message::Message;
    use crate::protocol::{Protocol, SamplingParams};

    #[test]
    fn encodes_basic_chat_request() {
        let p = OpenAiCompletionsProtocol::new();
        let messages = vec![Message::system("sys"), Message::user("hi")];
        let body = p
            .build_body("deepseek-v4-flash", &SamplingParams::default(), &messages, &[])
            .unwrap();

        assert_eq!(body["model"], "deepseek-v4-flash");
        assert_eq!(body["stream"], true);
        assert_eq!(body["messages"][0]["role"], "system");
        assert_eq!(body["messages"][0]["content"], "sys");
        assert_eq!(body["messages"][1]["role"], "user");
        // tools 为空时不应出现该字段
        assert!(body.get("tools").is_none());
        // 默认采样参数不附加
        assert!(body.get("max_tokens").is_none());
        assert!(body.get("temperature").is_none());
    }

    #[test]
    fn includes_sampling_params_when_set() {
        let p = OpenAiCompletionsProtocol::new();
        let params = SamplingParams {
            max_tokens: Some(256),
            temperature: Some(0.7),
            thinking_level: None,
        };
        let body = p
            .build_body("m", &params, &[Message::user("x")], &[])
            .unwrap();
        assert_eq!(body["max_tokens"], 256);
        assert_eq!(body["temperature"], 0.7);
    }
}
