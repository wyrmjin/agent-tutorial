//! DeepSeek provider implementation.
//!
//! DeepSeek API is OpenAI-compatible, using the chat completions endpoint.
//! API docs: https://api-docs.deepseek.com/

use std::collections::HashMap;
use std::pin::Pin;

use bytes::Bytes;
use futures::stream::Stream;
use futures::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use logger::{debug, error, trace};

use crate::{
    Message, Provider, ProviderType, Role, StopReason, StreamChunk, StreamChunkIterator,
    ToolSpec, Usage,
};

const DEFAULT_BASE_URL: &str = "https://api.deepseek.com";
const DEFAULT_MODEL: &str = "deepseek-v4-flash";

/// Configuration for creating a DeepSeek provider.
#[derive(Clone)]
pub struct DeepSeekConfig {
    pub api_key: String,
    pub base_url: String,
    pub model: String,
}

impl DeepSeekConfig {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: DEFAULT_BASE_URL.to_string(),
            model: DEFAULT_MODEL.to_string(),
        }
    }

    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }
}

pub struct DeepSeekProvider {
    client: Client,
    config: DeepSeekConfig,
}

impl DeepSeekProvider {
    pub fn new(config: DeepSeekConfig) -> Self {
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
    #[serde(rename = "type")]
    call_type: Option<String>,
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

type ByteStream = Pin<Box<dyn Stream<Item=Result<Bytes, reqwest::Error>> + Send>>;

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
    async fn next_sse_line(&mut self) -> anyhow::Result<Option<String>> {
        loop {
            // Try to extract a line from the buffer
            if let Some(pos) = self.buffer.find('\n') {
                let line = self.buffer[..pos].trim().to_string();
                self.buffer = self.buffer[pos + 1..].to_string();
                if line.is_empty() {
                    continue;
                }
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
    async fn next(&mut self) -> anyhow::Result<Option<StreamChunk>> {
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

            let chunk: ChatChunk = serde_json::from_str(data)?;

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
                        let indices: Vec<u32> =
                            self.pending_tool_calls.keys().copied().collect();
                        if let Some(&idx) = indices.first() {
                            let (id, name, args) =
                                self.pending_tool_calls.remove(&idx).unwrap();
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

fn role_str(role: &Role) -> &str {
    match role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    }
}

fn build_api_messages(messages: &[Message]) -> Vec<ApiMessage> {
    let mut api_messages: Vec<ApiMessage> = Vec::new();


    for msg in messages {
        let tool_calls: Option<Vec<ToolCallDelta>> = msg.tool_calls.as_ref().map(|tcs| {
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

        api_messages.push(ApiMessage {
            role: role_str(&msg.role).to_string(),
            content: if msg.content.is_empty() { None } else { Some(msg.content.clone()) },
            tool_calls,
            tool_call_id: msg.tool_call_id.clone(),
        });
    }

    api_messages
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
    ) -> anyhow::Result<Box<dyn StreamChunkIterator>> {
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
            anyhow::bail!("DeepSeek API error ({status}): {body}");
        }
        debug!(?response, "api_response");
        Ok(Box::new(DeepSeekStreamIterator::new(response)))
    }


    fn provider_type(&self) -> ProviderType {
        ProviderType::DeepSeek
    }
}
