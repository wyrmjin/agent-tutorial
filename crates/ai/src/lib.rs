//! AI 通信层(重构进行中)。

pub mod deepseek;
pub mod error;
pub mod message;
pub mod protocol;
pub mod registry;
pub mod stream;

pub use deepseek::DeepSeekProvider;
pub use error::AiError;
/// 临时别名 —— 让尚未迁移的旧代码继续编译, 后续 Task 切换调用方后删除。
pub use error::AiError as ProviderError;
pub use message::{Message, Role, ToolCallRequest, ToolSpec};
pub use protocol::{Protocol, ProtocolKind, SamplingParams, ThinkingLevel};
pub use registry::ProviderRegistry;
pub use stream::{
    ByteStream, DecodingStream, SseFrameReader, StopReason, StreamChunk, StreamChunkIterator,
    StreamDecoder, Usage,
};

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
