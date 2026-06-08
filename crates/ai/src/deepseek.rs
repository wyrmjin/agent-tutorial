//! DeepSeek provider implementation.
//!
//! DeepSeek API is OpenAI-compatible, using the chat completions endpoint.
//! API docs: https://api-docs.deepseek.com/

use std::collections::HashMap;
use std::pin::Pin;

use bytes::Bytes;
use futures::StreamExt;
use futures::stream::Stream;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use logger::{debug, error, info, trace};

use crate::{
    ApiProtocol, Message, Provider, ProviderConfig, ProviderError, ProviderType,
    StopReason, StreamChunk, StreamChunkIterator, ToolSpec, Usage,
};

const DEFAULT_BASE_URL: &str = "https://api.deepseek.com";
const DEFAULT_MODEL: &str = "deepseek-v4-flash";

pub struct DeepSeekProvider {
    client: Client,
    config: ProviderConfig,
}

impl DeepSeekProvider {
    /// Create a new DeepSeek provider.
    ///
    /// Fills in DeepSeek-specific defaults for any empty fields in `config`:
    /// - `base_url` → `https://api.deepseek.com`
    /// - `model` → `deepseek-v4-flash`
    /// - `protocol` → `ApiProtocol::OpenAI`
    pub fn new(config: ProviderConfig) -> Self {
        let config = ProviderConfig {
            base_url: if config.base_url.is_empty() {
                DEFAULT_BASE_URL.to_string()
            } else {
                config.base_url
            },
            model: if config.model.is_empty() {
                DEFAULT_MODEL.to_string()
            } else {
                config.model
            },
            protocol: Some(config.protocol.unwrap_or(ApiProtocol::OpenAI)),
            ..config
        };
        Self {
            client: Client::new(),
            config,
        }
    }
}

// ── OpenAI-compatible API types ────────────────────────────────────────────

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
struct ChatRequest {
    model: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<ApiToolOwned>,
    messages: Vec<ApiMessage>,
    stream: bool,
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

// ── SSE response types ─────────────────────────────────────────────────────

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
    prompt_tokens_details: Option<PromptTokensDetails>,
    prompt_cache_hit_tokens: u64,
    prompt_cache_miss_tokens: u64,
}

// ── SSE stream iterator ────────────────────────────────────────────────────

type ByteStream = Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send>>;

pub struct DeepSeekStreamIterator {
    byte_stream: ByteStream,
    buffer: String,
    pending_tool_calls: HashMap<u32, (Option<String>, Option<String>, String)>,
    done: bool,
}

impl DeepSeekStreamIterator {
    fn new(response: reqwest::Response) -> Self {
        Self {
            byte_stream: Box::pin(response.bytes_stream()),
            buffer: String::new(),
            pending_tool_calls: HashMap::new(),
            done: false,
        }
    }

    /// Read the next complete SSE line (minus the "data: " prefix).
    async fn next_sse_line(&mut self) -> Result<Option<String>, ProviderError> {
        loop {
            // Try to extract a line from the buffer
            if let Some(pos) = self.buffer.find('\n') {
                let line = self.buffer[..pos].trim().to_string();
                self.buffer = self.buffer[pos + 1..].to_string();
                if line.is_empty() {
                    continue;
                }
                info!(line, "sse line");
                return Ok(Some(line));
            }

            // Read more data
            match self.byte_stream.next().await {
                Some(Ok(bytes)) => {
                    self.buffer.push_str(&String::from_utf8_lossy(&bytes));
                }
                Some(Err(e)) => return Err(e.into()),
                None => {
                    // Stream ended, flush remaining buffer
                    if self.buffer.is_empty() {
                        return Ok(None);
                    }
                    let line = std::mem::take(&mut self.buffer).trim().to_string();
                    return Ok(if line.is_empty() { None } else { Some(line) });
                }
            }
        }
    }
}

#[async_trait::async_trait]
impl StreamChunkIterator for DeepSeekStreamIterator {
    async fn next(&mut self) -> Result<Option<StreamChunk>, ProviderError> {
        if self.done {
            return Ok(None);
        }

        loop {
            let Some(line) = self.next_sse_line().await? else {
                self.done = true;
                return Ok(None);
            };

            trace!(%line, "sse_line");
            if line == "data: [DONE]" {
                self.done = true;
                return Ok(None);
            }

            let Some(data) = line.strip_prefix("data: ") else {
                continue;
            };

            let chunk: ChatChunk =
                serde_json::from_str(data).map_err(|e| ProviderError::Parse(e.to_string()))?;

            for choice in &chunk.choices {
                let delta = &choice.delta;

                // Text content
                if let Some(ref content) = delta.content {
                    if !content.is_empty() {
                        return Ok(Some(StreamChunk::Text(content.clone())));
                    }
                }

                // Tool calls — accumulate by index
                if let Some(tool_calls) = &delta.tool_calls {
                    for tc in tool_calls {
                        let entry = self
                            .pending_tool_calls
                            .entry(tc.index)
                            .or_insert_with(|| (None, None, String::new()));

                        if let Some(ref id) = tc.id {
                            entry.0 = Some(id.clone());
                        }
                        if let Some(ref name) = tc.function.name {
                            entry.1 = Some(name.clone());
                        }
                        if let Some(ref args) = tc.function.arguments {
                            entry.2.push_str(args);
                        }
                    }
                }

                // Finish reason
                if let Some(ref reason) = choice.finish_reason {
                    // Flush pending tool calls first
                    if !self.pending_tool_calls.is_empty() {
                        let indices: Vec<u32> = self.pending_tool_calls.keys().copied().collect();
                        if let Some(&idx) = indices.first() {
                            let (id, name, args) = self.pending_tool_calls.remove(&idx).unwrap();
                            if let (Some(id), Some(name)) = (id, name) {
                                let input: serde_json::Value = if args.is_empty() {
                                    serde_json::Value::Object(serde_json::Map::new())
                                } else {
                                    serde_json::from_str(&args).unwrap_or_default()
                                };
                                // Don't mark done if there are more tool calls to flush
                                if self.pending_tool_calls.is_empty() {
                                    self.done = true;
                                }
                                return Ok(Some(StreamChunk::ToolUse { id, name, input }));
                            }
                        }
                    }

                    let stop_reason = match reason.as_str() {
                        "stop" | "end_turn" => StopReason::EndTurn,
                        "tool_calls" => StopReason::ToolUse,
                        "length" => StopReason::MaxTokens,
                        _ => StopReason::EndTurn,
                    };

                    let usage = chunk.usage.as_ref().map(|u| {
                        let mut extra = std::collections::HashMap::new();
                        extra.insert(
                            "prompt_cache_hit_tokens".to_string(),
                            serde_json::Value::Number(u.prompt_cache_hit_tokens.into()),
                        );
                        extra.insert(
                            "prompt_cache_miss_tokens".to_string(),
                            serde_json::Value::Number(u.prompt_cache_miss_tokens.into()),
                        );
                        if let Some(ref details) = u.prompt_tokens_details {
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
                    });

                    self.done = true;
                    return Ok(Some(StreamChunk::Finished {
                        stop_reason,
                        usage: usage.unwrap_or(Usage {
                            input_tokens: 0,
                            output_tokens: 0,
                            extra: HashMap::new(),
                        }),
                    }));
                }
            }
        }
    }
}

// ── Provider trait implementation ──────────────────────────────────────────

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
            },
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

#[async_trait::async_trait]
impl Provider for DeepSeekProvider {
    async fn stream_chat(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
    ) -> Result<Box<dyn StreamChunkIterator>, ProviderError> {
        let api_messages = build_api_messages(messages);
        let api_tools = build_api_tools(tools);

        let request = ChatRequest {
            model: self.config.model.clone(),
            tools: api_tools,
            messages: api_messages,
            stream: true,
        };

        let url = format!("{}/chat/completions", self.config.base_url);
        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            error!(%status, %body, "deepseek api error");
            return Err(ProviderError::Api {
                status: status.as_u16(),
                message: body,
            });
        }
        debug!(?response, "api_response");
        Ok(Box::new(DeepSeekStreamIterator::new(response)))
    }

    fn provider_type(&self) -> ProviderType {
        ProviderType::DeepSeek
    }
}
