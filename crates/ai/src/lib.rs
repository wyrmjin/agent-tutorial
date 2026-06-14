//! LLM Provider abstraction layer.
//!
//! Defines the [`Provider`] trait for interacting with different LLM backends
//! (Anthropic, OpenAI, Ollama, etc.).

pub mod deepseek;
pub mod error;
pub mod registry;
pub mod types;

pub use deepseek::DeepSeekProvider;
pub use error::ProviderError;
pub use registry::ProviderRegistry;

use std::fmt;

/// The API protocol used to communicate with an LLM provider.
///
/// Different providers may support multiple protocols. For example, DeepSeek
/// supports both OpenAI-compatible chat completions and Anthropic-compatible
/// messages APIs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApiProtocol {
    /// OpenAI-compatible `/chat/completions` API.
    OpenAI,
    /// Anthropic-compatible `/v1/messages` API.
    Anthropic,
}

impl fmt::Display for ApiProtocol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OpenAI => write!(f, "openai"),
            Self::Anthropic => write!(f, "anthropic"),
        }
    }
}

/// Generic configuration for creating an LLM provider.
///
/// Fields left empty (empty string for `base_url`/`model`, or no explicit
/// `protocol`) are filled with provider-specific defaults inside each
/// provider's constructor.
#[derive(Clone)]
pub struct ProviderConfig {
    /// API key for authentication.
    pub api_key: String,
    /// Base URL of the LLM API endpoint.
    pub base_url: String,
    /// Model name to use.
    pub model: String,
    /// API protocol to use.
    pub protocol: Option<ApiProtocol>,
}

impl ProviderConfig {
    /// Create a new config with the given API key. All other fields use
    /// empty defaults that the provider will fill in.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: String::new(),
            model: String::new(),
            protocol: None,
        }
    }

    /// Set the base URL.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Set the model name.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Set the API protocol.
    pub fn with_protocol(mut self, protocol: ApiProtocol) -> Self {
        self.protocol = Some(protocol);
        self
    }
}

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
pub enum Message {
    /// A system prompt message.
    System { content: String },
    /// A user message.
    User { content: String },
    /// An assistant (LLM) response message.
    Assistant {
        content: String,
        /// Tool calls requested by the assistant.
        tool_calls: Option<Vec<ToolCallRequest>>,
    },
    /// A tool execution result message.
    Tool {
        content: String,
        /// ID of the tool call this result corresponds to.
        tool_call_id: String,
    },
}

impl Message {
    /// Returns the role of this message.
    pub fn role(&self) -> Role {
        match self {
            Message::System { .. } => Role::System,
            Message::User { .. } => Role::User,
            Message::Assistant { .. } => Role::Assistant,
            Message::Tool { .. } => Role::Tool,
        }
    }

    /// Returns the content of this message, regardless of variant.
    pub fn content(&self) -> &str {
        match self {
            Message::System { content }
            | Message::User { content }
            | Message::Assistant { content, .. }
            | Message::Tool { content, .. } => content,
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Message::User {
            content: content.into(),
        }
    }

    pub fn system(content: impl Into<String>) -> Self {
        Message::System {
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Message::Assistant {
            content: content.into(),
            tool_calls: None,
        }
    }

    pub fn tool_result(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Message::Tool {
            content: content.into(),
            tool_call_id: tool_call_id.into(),
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

/// A streaming chunk from the LLM.
#[derive(Debug, Clone)]
pub enum StreamChunk {
    Text(String),
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    Finished {
        stop_reason: StopReason,
        usage: Usage,
    },
}

#[derive(Debug, Clone, Default)]
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
}

/// The core abstraction over LLM backends.
#[async_trait::async_trait]
pub trait Provider: Send + Sync {
    /// Send a conversation and get a streaming response.
    async fn stream_chat(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
    ) -> Result<Box<dyn StreamChunkIterator>, ProviderError>;

    /// Which provider this is.
    fn provider_type(&self) -> ProviderType;
}

/// Iterator over streaming chunks.
#[async_trait::async_trait]
pub trait StreamChunkIterator: Send {
    async fn next(&mut self) -> Result<Option<StreamChunk>, ProviderError>;
}
