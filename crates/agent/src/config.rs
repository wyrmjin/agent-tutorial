//! Configuration for an agent run.

/// Configuration for an agent run.
#[derive(Debug, Clone)]
pub struct AgentConfig {
    pub max_tool_rounds: usize,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_tool_rounds: 25,
        }
    }
}
