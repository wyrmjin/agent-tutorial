//! API protocol abstraction (pure codec — no network IO).

pub mod openai_completions;

use std::sync::Arc;

use crate::error::AiError;
use crate::message::{Message, ToolSpec};
use crate::stream::StreamDecoder;
use crate::usage::UsageNormalizer;

/// Which wire protocol a model speaks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProtocolKind {
    OpenAiCompletions,
    OpenAiResponses,
    AnthropicMessages,
    OpenAiCodexResponses,
}

/// Extended-thinking effort level. Each protocol's `build_body`
/// translates this into its own wire representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkingLevel {
    None,
    Low,
    Medium,
    High,
    Xhigh,
    Max,
}

/// Unified sampling parameters, protocol-agnostic.
#[derive(Debug, Clone, Default)]
pub struct SamplingParams {
    pub max_tokens: Option<u32>,
    pub temperature: Option<f64>,
    pub thinking_level: Option<ThinkingLevel>,
}

/// A wire protocol = pure codec. No network IO.
pub trait Protocol: Send + Sync {
    fn kind(&self) -> ProtocolKind;

    /// Request path relative to the endpoint base_url, e.g. "/chat/completions".
    fn endpoint_path(&self) -> &str;

    /// Protocol-intrinsic headers (NOT auth). Empty is fine when the transport's
    /// `.json()` already sets content-type.
    fn protocol_headers(&self) -> Vec<(String, String)>;

    /// Build the request body. Pure function — unit-testable without network.
    fn build_body(
        &self,
        model_id: &str,
        params: &SamplingParams,
        messages: &[Message],
        tools: &[ToolSpec],
    ) -> Result<serde_json::Value, AiError>;

    /// New decoder holding this response's streaming accumulation state.
    ///
    /// `normalizer` 由调用方(Transport, 来自 Provider)注入, decoder 把原始
    /// usage 字段交给它归一化——协议因此不感知具体供应商。
    fn new_decoder(&self, normalizer: Arc<dyn UsageNormalizer>) -> Box<dyn StreamDecoder>;
}
