use std::sync::Arc;
use crate::Provider;

pub enum Api {
    OpenaiCompletions,
    OpenaiResponses,
    OpenaiCodexResponses,
    AnthropicMessages,
}

pub enum ThinkingLevel {
    /// No extended thinking.
    None,
    /// Low effort thinking.
    Low,
    /// Medium effort thinking.
    Medium,
    /// High effort / deep thinking.
    High,

    Xhigh,

    Max,
}
pub struct Model {
    /// Unique model identifier (e.g., "deepseek-chat", "gpt-4o").
    pub id: String,
    /// Human-readable display name.
    pub name: String,
    /// Which provider offers this model.
    pub provider: Arc<dyn Provider>,
    /// API base URL.
    pub base_url: String,
    /// The API protocol to use for communication.
    pub api: Api,
    /// Maximum output tokens.
    pub max_tokens: Option<u32>,
    /// Whether this model supports reasoning/thinking mode.
    pub reasoning: bool,
    /// Thinking level (only relevant when `reasoning` is true).
    pub thinking_level: Option<ThinkingLevel>,
    /// Temperature for sampling.
    pub temperature: Option<f64>,
}
