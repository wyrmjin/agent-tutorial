//! Events and structured result produced during an agent run.

use ai::Usage;

/// Events emitted during an agent run.
#[derive(Debug, Clone)]
pub enum AgentEvent {
    /// The LLM produced text content.
    Text(String),
    /// The LLM wants to invoke a tool.
    ToolRequest {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    /// A tool execution result.
    ToolResponse {
        id: String,
        content: String,
        is_error: bool,
    },
    /// A tool needs user approval before it can proceed.
    ApprovalRequired {
        tool_call_id: String,
        tool_name: String,
        path: String,
        message: String,
    },
    /// The conversation turn is complete.
    TurnEnd { usage: Usage },
    /// An error occurred during execution.
    Error {
        message: String,
        /// If true, the agent can continue; if false, the agent must stop.
        recoverable: bool,
    },
    /// The agent has finished processing. This is always the last event.
    Done { messages: Vec<ai::Message> },
}

/// Structured result returned after the agent finishes.
#[derive(Debug, Clone)]
pub struct AgentResult {
    pub messages: Vec<ai::Message>,
    pub total_turns: usize,
    pub usage: Usage,
}

impl Default for AgentResult {
    fn default() -> Self {
        Self {
            messages: Vec::new(),
            total_turns: 0,
            usage: Usage::default(),
        }
    }
}
