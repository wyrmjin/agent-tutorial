//! Tool system — trait, registry, and built-in tools.

mod bash;
mod read_file;
mod write_file;

pub use bash::BashTool;
pub use read_file::ReadFileTool;
pub use write_file::WriteFileTool;

use std::collections::HashMap;

/// Result of executing a tool.
#[derive(Debug, Clone)]
pub struct ToolOutput {
    pub content: String,
    pub is_error: bool,
}

/// A tool that the agent can invoke.
#[async_trait::async_trait]
pub trait Tool: Send + Sync {
    /// Unique name, e.g. "bash", "read", "write".
    fn name(&self) -> &str;
    /// Human-readable description, shown to the LLM.
    fn description(&self) -> &str;
    /// JSON Schema for the tool's parameters.
    fn parameters(&self) -> serde_json::Value;
    /// Execute the tool with the given input.
    async fn execute(&self, input: serde_json::Value) -> ToolOutput;
}

/// Holds all registered tools and dispatches by name.
pub struct ToolRegistry {
    tools: HashMap<String, Box<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    pub fn register<T: Tool + 'static>(&mut self, tool: T) {
        self.tools.insert(tool.name().to_string(), Box::new(tool));
    }

    pub fn get(&self, name: &str) -> Option<&dyn Tool> {
        self.tools.get(name).map(|t| t.as_ref())
    }

    /// Execute a tool by name. Returns None if the tool isn't registered.
    pub async fn execute(&self, name: &str, input: serde_json::Value) -> Option<ToolOutput> {
        match self.get(name) {
            Some(tool) => Some(tool.execute(input).await),
            None => None,
        }
    }

    /// Build provider ToolSpecs for all registered tools.
    pub fn to_tool_specs(&self) -> Vec<provider::ToolSpec> {
        self.tools
            .iter()
            .map(|(name, tool)| provider::ToolSpec {
                name: name.clone(),
                description: tool.description().to_string(),
                parameters: tool.parameters(),
            })
            .collect()
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}
