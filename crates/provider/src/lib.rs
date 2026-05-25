//! LLM Provider abstraction layer.
//!
//! Defines the [`Provider`] trait for interacting with different LLM backends
//! (Anthropic, OpenAI, Ollama, etc.).

pub mod deepseek;

pub use deepseek::{DeepSeekConfig, DeepSeekProvider};

use std::fmt;

/// Supported LLM provider types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderType {
    Anthropic,
    OpenAI,
    Ollama,
    DeepSeek,
}

impl fmt::Display for ProviderType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Anthropic => write!(f, "anthropic"),
            Self::OpenAI => write!(f, "openai"),
            Self::Ollama => write!(f, "ollama"),
            Self::DeepSeek => write!(f, "deepseek"),
        }
    }
}

/// A single message in a conversation.
#[derive(Debug, Clone)]
pub struct Message {
    pub role: Role,
    pub content: String,
    /// For assistant messages with tool calls.
    pub tool_calls: Option<Vec<ToolCallRequest>>,
    /// For tool result messages.
    pub tool_call_id: Option<String>,
}

impl Message {
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    pub fn tool_result(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: content.into(),
            tool_calls: None,
            tool_call_id: Some(tool_call_id.into()),
        }
    }
}

/// A tool call request from the assistant.
#[derive(Debug, Clone)]
pub struct ToolCallRequest {
    pub id: String,
    pub name: String,
    pub input: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// Tool definition passed to the LLM.
#[derive(Debug, Clone)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

/// Result of a tool execution that gets sent back to the LLM.
#[derive(Debug, Clone)]
pub struct ToolResult {
    pub tool_use_id: String,
    pub content: String,
    pub is_error: bool,
}

/// A streaming chunk from the LLM.
#[derive(Debug, Clone)]
pub enum StreamChunk {
    Text(String),
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    ToolUseEnd,
    Finished {
        stop_reason: StopReason,
        usage: Usage,
    },
}

#[derive(Debug, Clone)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub extra: std::collections::HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopReason {
    EndTurn,
    ToolUse,
    MaxTokens,
    StopSequence,
}

/// The core abstraction over LLM backends.
#[async_trait::async_trait]
pub trait Provider: Send + Sync {
    /// Send a conversation and get a streaming response.
    async fn stream_chat(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
        system_prompt: &str,
    ) -> anyhow::Result<Box<dyn StreamChunkIterator>>;

    /// Non-streaming chat for simple requests.
    async fn chat(
        &self,
        messages: &[Message],
        system_prompt: &str,
    ) -> anyhow::Result<String>;

    /// Which provider this is.
    fn provider_type(&self) -> ProviderType;
}

/// Iterator over streaming chunks.
#[async_trait::async_trait]
pub trait StreamChunkIterator: Send {
    async fn next(&mut self) -> anyhow::Result<Option<StreamChunk>>;
}
