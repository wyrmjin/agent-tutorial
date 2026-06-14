//! Streaming response model and the chunk iterator the agent consumes.

use crate::error::AiError;

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

/// Iterator over streaming chunks (pull model: the agent calls `next`).
#[async_trait::async_trait]
pub trait StreamChunkIterator: Send {
    async fn next(&mut self) -> Result<Option<StreamChunk>, AiError>;
}
