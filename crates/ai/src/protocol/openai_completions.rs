//! OpenAI-compatible `/chat/completions` protocol (codec).
//!
//! Migrated from the former `deepseek.rs`.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use logger::trace;

use crate::error::AiError;
use crate::message::{Message, ToolSpec};
use crate::protocol::{Protocol, ProtocolKind, SamplingParams, ThinkingLevel};
use crate::stream::{SseFrameReader, StopReason, StreamChunk, StreamDecoder, Usage};

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
    /// 思考强度(DeepSeek/OpenAI reasoning_effort)。仅当 thinking_level 非 None 时设置。
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<String>,
}

/// 把统一的 ThinkingLevel 翻译成 DeepSeek/OpenAI 线上的 reasoning_effort 字符串。
/// (参考 DeepSeek thinking_mode 文档: low/medium→high, xhigh→max。这里直传原值,
///  让服务端做兼容映射, 与文档语义一致。)
fn reasoning_effort_for(level: ThinkingLevel) -> &'static str {
    match level {
        ThinkingLevel::None => "low",
        ThinkingLevel::Low => "low",
        ThinkingLevel::Medium => "medium",
        ThinkingLevel::High => "high",
        ThinkingLevel::Xhigh => "xhigh",
        ThinkingLevel::Max => "max",
    }
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
            reasoning_effort: params
                .thinking_level
                .map(|level| reasoning_effort_for(level).to_string()),
        };
        serde_json::to_value(&request).map_err(|e| AiError::Encode(e.to_string()))
    }

    fn new_decoder(&self) -> Box<dyn StreamDecoder> {
        Box::new(OpenAiCompletionsDecoder::new())
    }
}

// ── SSE response types (Deserialize) ─────────────────────────────────────────

#[derive(Deserialize, Debug)]
struct ChatChunk {
    choices: Vec<ChoiceDelta>,
    usage: Option<ApiUsage>,
}

#[derive(Deserialize, Debug)]
struct ChoiceDelta {
    delta: DeltaContent,
    finish_reason: Option<String>,
}

#[derive(Deserialize, Debug, Default)]
struct DeltaContent {
    content: Option<String>,
    tool_calls: Option<Vec<ToolCallChunk>>,
}

#[derive(Deserialize, Debug)]
struct ToolCallChunk {
    index: u32,
    id: Option<String>,
    function: FunctionChunk,
}

#[derive(Deserialize, Debug, Default)]
struct FunctionChunk {
    name: Option<String>,
    arguments: Option<String>,
}

#[derive(Deserialize, Debug)]
struct PromptTokensDetails {
    cached_tokens: u64,
}

#[derive(Deserialize, Debug)]
struct ApiUsage {
    prompt_tokens: u64,
    completion_tokens: u64,
    #[serde(default)]
    prompt_tokens_details: Option<PromptTokensDetails>,
    // 以下为 DeepSeek 专有字段, 其它 OpenAI 兼容服务不会返回, 故 default=0。
    #[serde(default)]
    prompt_cache_hit_tokens: u64,
    #[serde(default)]
    prompt_cache_miss_tokens: u64,
}

// ── decoder ──────────────────────────────────────────────────────────────────

/// Tool-call accumulation entry: (id, name, arguments-so-far).
type PendingToolCall = (Option<String>, Option<String>, String);

pub struct OpenAiCompletionsDecoder {
    frames: SseFrameReader,
    pending_tool_calls: HashMap<u32, PendingToolCall>,
    done: bool,
}

impl OpenAiCompletionsDecoder {
    fn new() -> Self {
        Self {
            frames: SseFrameReader::new(),
            pending_tool_calls: HashMap::new(),
            done: false,
        }
    }

    /// Decode a single SSE line into zero or more chunks.
    fn process_line(&mut self, line: &str) -> Result<Vec<StreamChunk>, AiError> {
        trace!(%line, "sse_line");
        if self.done {
            return Ok(Vec::new());
        }
        if line == "data: [DONE]" {
            self.done = true;
            return Ok(Vec::new());
        }
        let Some(data) = line.strip_prefix("data: ") else {
            return Ok(Vec::new());
        };

        let chunk: ChatChunk = serde_json::from_str(data).map_err(|e| AiError::Parse(e.to_string()))?;

        let mut out = Vec::new();
        for choice in &chunk.choices {
            let delta = &choice.delta;

            if let Some(content) = &delta.content
                && !content.is_empty()
            {
                out.push(StreamChunk::Text(content.clone()));
            }

            if let Some(tool_calls) = &delta.tool_calls {
                for tc in tool_calls {
                    let entry = self
                        .pending_tool_calls
                        .entry(tc.index)
                        .or_insert_with(|| (None, None, String::new()));
                    if let Some(id) = &tc.id {
                        entry.0 = Some(id.clone());
                    }
                    if let Some(name) = &tc.function.name {
                        entry.1 = Some(name.clone());
                    }
                    if let Some(args) = &tc.function.arguments {
                        entry.2.push_str(args);
                    }
                }
            }

            if let Some(reason) = &choice.finish_reason {
                // Flush ALL pending tool calls, ordered by index.
                let mut indices: Vec<u32> = self.pending_tool_calls.keys().copied().collect();
                indices.sort_unstable();
                for idx in indices {
                    let (id, name, args) = self.pending_tool_calls.remove(&idx).unwrap();
                    if let (Some(id), Some(name)) = (id, name) {
                        let input: serde_json::Value = if args.is_empty() {
                            // 无参工具: arguments 流式累积为空, 发空对象。
                            serde_json::Value::Object(serde_json::Map::new())
                        } else {
                            // 非空 arguments 必须是合法 JSON; 畸形(如被 max_tokens 截断)
                            // 应明确报错, 而非静默吞成 Null 让工具收到误导性入参。
                            serde_json::from_str(&args).map_err(|e| {
                                AiError::Parse(format!("tool {name} arguments: {e}"))
                            })?
                        };
                        out.push(StreamChunk::ToolUse { id, name, input });
                    }
                }

                let stop_reason = match reason.as_str() {
                    "stop" | "end_turn" => StopReason::EndTurn,
                    "tool_calls" => StopReason::ToolUse,
                    "length" => StopReason::MaxTokens,
                    _ => StopReason::EndTurn,
                };

                let usage = chunk
                    .usage
                    .as_ref()
                    .map(|u| {
                        let mut extra = HashMap::new();
                        extra.insert(
                            "prompt_cache_hit_tokens".to_string(),
                            serde_json::Value::Number(u.prompt_cache_hit_tokens.into()),
                        );
                        extra.insert(
                            "prompt_cache_miss_tokens".to_string(),
                            serde_json::Value::Number(u.prompt_cache_miss_tokens.into()),
                        );
                        if let Some(details) = &u.prompt_tokens_details {
                            extra.insert(
                                "cached_tokens".to_string(),
                                serde_json::Value::Number(details.cached_tokens.into()),
                            );
                        }
                        Usage {
                            input_tokens: u.prompt_tokens,
                            output_tokens: u.completion_tokens,
                            extra,
                        }
                    })
                    .unwrap_or_default();

                out.push(StreamChunk::Finished { stop_reason, usage });
                self.done = true;
                // 终态已产出, 同一帧里后续 choice 不应再产生 chunk(否则 Text 会排在 Finished 之后)。
                break;
            }
        }
        Ok(out)
    }
}

impl StreamDecoder for OpenAiCompletionsDecoder {
    fn feed(&mut self, bytes: &[u8]) -> Result<Vec<StreamChunk>, AiError> {
        let mut out = Vec::new();
        for line in self.frames.push(bytes) {
            out.extend(self.process_line(&line)?);
        }
        Ok(out)
    }

    fn finish(&mut self) -> Result<Vec<StreamChunk>, AiError> {
        let mut out = Vec::new();
        if let Some(line) = self.frames.flush() {
            out.extend(self.process_line(&line)?);
        }
        Ok(out)
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

    #[test]
    fn thinking_level_emits_reasoning_effort() {
        let p = OpenAiCompletionsProtocol::new();
        // None → 不发 reasoning_effort
        let none_body = p
            .build_body(
                "m",
                &SamplingParams {
                    thinking_level: None,
                    ..Default::default()
                },
                &[Message::user("x")],
                &[],
            )
            .unwrap();
        assert!(none_body.get("reasoning_effort").is_none());

        // 各级别 → 对应 effort 字符串
        for (level, effort) in [
            (crate::protocol::ThinkingLevel::Low, "low"),
            (crate::protocol::ThinkingLevel::Medium, "medium"),
            (crate::protocol::ThinkingLevel::High, "high"),
            (crate::protocol::ThinkingLevel::Xhigh, "xhigh"),
            (crate::protocol::ThinkingLevel::Max, "max"),
        ] {
            let body = p
                .build_body(
                    "m",
                    &SamplingParams {
                        thinking_level: Some(level),
                        ..Default::default()
                    },
                    &[Message::user("x")],
                    &[],
                )
                .unwrap();
            assert_eq!(
                body["reasoning_effort"], effort,
                "ThinkingLevel {level:?} 应映射到 reasoning_effort={effort:?}"
            );
        }
    }
}

#[cfg(test)]
mod decoder_tests {
    use super::OpenAiCompletionsProtocol;
    use crate::protocol::Protocol;
    use crate::stream::{StopReason, StreamChunk};

    fn decode_all(bytes_chunks: &[&[u8]]) -> Vec<StreamChunk> {
        let p = OpenAiCompletionsProtocol::new();
        let mut d = p.new_decoder();
        let mut out = Vec::new();
        for b in bytes_chunks {
            out.extend(d.feed(b).unwrap());
        }
        out.extend(d.finish().unwrap());
        out
    }

    #[test]
    fn decodes_text_then_finish() {
        let chunks = decode_all(&[
            b"data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"},\"finish_reason\":null}],\"usage\":null}\n",
            b"data: {\"choices\":[{\"delta\":{\"content\":\" world\"},\"finish_reason\":null}],\"usage\":null}\n",
            b"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5,\"prompt_cache_hit_tokens\":0,\"prompt_cache_miss_tokens\":10}}\n",
            b"data: [DONE]\n",
        ]);

        let texts: Vec<_> = chunks
            .iter()
            .filter_map(|c| match c {
                StreamChunk::Text(t) => Some(t.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(texts, vec!["Hello".to_string(), " world".to_string()]);

        match chunks.last().unwrap() {
            StreamChunk::Finished { stop_reason, usage } => {
                assert_eq!(*stop_reason, StopReason::EndTurn);
                assert_eq!(usage.input_tokens, 10);
                assert_eq!(usage.output_tokens, 5);
            }
            other => panic!("expected Finished, got {other:?}"),
        }
    }

    #[test]
    fn decodes_tool_call_accumulated_across_frames() {
        // arguments 跨两帧累积; 故意把第一帧切成两次 feed。
        let chunks = decode_all(&[
            b"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"function\":{\"name\":\"echo\",\"arguments\":\"{\\\"msg\\\"\"}}]},\"finish_reason\":null}],\"usage\":null}\n",
            b"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\":\\\"hi\\\"}\"}}]},\"finish_reason\":null}],\"usage\":null}\n",
            b"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1,\"prompt_cache_hit_tokens\":0,\"prompt_cache_miss_tokens\":1}}\n",
        ]);

        let tool_uses: Vec<_> = chunks
            .iter()
            .filter_map(|c| match c {
                StreamChunk::ToolUse { id, name, input } => {
                    Some((id.clone(), name.clone(), input.clone()))
                }
                _ => None,
            })
            .collect();
        assert_eq!(tool_uses.len(), 1);
        assert_eq!(tool_uses[0].0, "call_1");
        assert_eq!(tool_uses[0].1, "echo");
        assert_eq!(tool_uses[0].2["msg"], "hi");

        assert!(matches!(
            chunks.last().unwrap(),
            StreamChunk::Finished { stop_reason: StopReason::ToolUse, .. }
        ));
    }

    #[test]
    fn decodes_finish_with_openai_style_usage_without_cache_fields() {
        // 非 DeepSeek 的 OpenAI 兼容服务通常只返回 prompt_tokens/completion_tokens/total_tokens,
        // 不含 prompt_cache_hit_tokens 等 DeepSeek 专有字段。decoder 必须能解析, 不能 Parse 错误。
        let chunks = decode_all(&[b"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5,\"total_tokens\":15}}\n"]);

        match chunks.last().unwrap() {
            StreamChunk::Finished { stop_reason, usage } => {
                assert_eq!(*stop_reason, StopReason::EndTurn);
                assert_eq!(usage.input_tokens, 10);
                assert_eq!(usage.output_tokens, 5);
                // DeepSeek 专有字段缺失时按 0 记入 extra(关键是不能 Parse 失败中断流)。
                assert_eq!(usage.extra["prompt_cache_hit_tokens"], 0);
                assert_eq!(usage.extra["prompt_cache_miss_tokens"], 0);
            }
            other => panic!("expected Finished, got {other:?}"),
        }
    }

    #[test]
    fn no_chunks_after_finished_in_multi_choice_frame() {
        // 同一帧里 finish_reason 出现在前一个 choice, 后续 choice 的 Text 不应在 Finished 之后产出。
        // 期望: Text("x") → Finished, 不会有 Text("y")。
        let chunks = decode_all(&[
            b"data: {\"choices\":[{\"delta\":{\"content\":\"x\"},\"finish_reason\":\"stop\"},{\"delta\":{\"content\":\"y\"}}],\"usage\":null}\n",
        ]);
        let finished_idx = chunks
            .iter()
            .position(|c| matches!(c, StreamChunk::Finished { .. }))
            .expect("应有 Finished");
        assert_eq!(
            finished_idx,
            chunks.len() - 1,
            "Finished 必须是最后一个 chunk, 但实际序列: {:?}",
            chunks
        );
    }

    #[test]
    fn empty_tool_arguments_become_empty_object() {
        // 工具调用 arguments 为空(无参工具常见)→ input 应是空对象 {}, 而非 Value::Null。
        let chunks = decode_all(&[
            b"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c1\",\"function\":{\"name\":\"ping\",\"arguments\":\"\"}}]},\"finish_reason\":null}],\"usage\":null}\n",
            b"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1}}\n",
        ]);
        let tu = chunks
            .iter()
            .find_map(|c| match c {
                StreamChunk::ToolUse { input, .. } => Some(input.clone()),
                _ => None,
            })
            .expect("应有 ToolUse");
        assert!(tu.is_object(), "空参应是 JSON 对象, 实际: {tu}");
        assert!(tu.as_object().unwrap().is_empty(), "应是空对象 {{}}");
    }

    #[test]
    fn malformed_tool_arguments_surface_parse_error() {
        // arguments 非空但畸形(如被 max_tokens 截断)→ 不应静默吞成 Null, 应返回 Parse 错误。
        let p = OpenAiCompletionsProtocol::new();
        let mut d = p.new_decoder();
        let res = d.feed(b"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c1\",\"function\":{\"name\":\"echo\",\"arguments\":\"{not json\"}}]},\"finish_reason\":\"tool_calls\"}],\"usage\":null}\n");
        assert!(
            matches!(res, Err(crate::error::AiError::Parse(_))),
            "畸形 arguments 应返回 Parse 错误, 实际: {res:?}"
        );
    }
}