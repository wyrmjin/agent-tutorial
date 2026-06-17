//! Tool system — trait, registry, and built-in tools.

mod bash;
mod read_file;
mod write_file;

pub use bash::BashTool;
pub use read_file::ReadFileTool;
pub use write_file::WriteFileTool;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tracing::warn;

/// Result of executing a tool.
#[derive(Debug, Clone)]
pub struct ToolOutput {
    pub content: String,
    pub is_error: bool,
    /// 是否需要用户审批才能继续执行
    pub needs_approval: bool,
    /// 需要审批的文件路径（仅当 needs_approval 为 true 时有意义）
    pub approval_path: Option<String>,
}

impl ToolOutput {
    /// 创建一个成功的工具输出。
    pub fn ok(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: false,
            needs_approval: false,
            approval_path: None,
        }
    }

    /// 创建一个错误的工具输出。
    pub fn error(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: true,
            needs_approval: false,
            approval_path: None,
        }
    }

    /// 创建一个成功输出，但允许指定 `is_error` 标志（用于命令执行退出码非零等场景）。
    pub fn ok_with_status(content: impl Into<String>, is_error: bool) -> Self {
        Self {
            content: content.into(),
            is_error,
            needs_approval: false,
            approval_path: None,
        }
    }

    /// 创建一个需要用户审批的工具输出。
    pub fn approval(content: impl Into<String>, path: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: false,
            needs_approval: true,
            approval_path: Some(path.into()),
        }
    }
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
    ///
    /// **Approval contract**: If the tool returns a `ToolOutput` with
    /// `needs_approval = true`, the tool MUST NOT have performed any
    /// side effects. The caller will request user approval and, if
    /// granted, call `approve()` and then `execute()` again. The second
    /// call should perform the actual operation.
    async fn execute(&self, input: serde_json::Value) -> ToolOutput;

    /// 批准工具的待审批操作。默认空实现，由需要审批的工具覆盖。
    fn approve(&self, _input: &serde_json::Value) {}
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

    /// 批准指定工具的待审批操作。由 main.rs 在用户同意后调用。
    pub fn approve(&self, tool_name: &str, input: &serde_json::Value) {
        if let Some(tool) = self.get(tool_name) {
            tool.approve(input);
        } else {
            warn!(%tool_name, "approve called for unknown tool");
        }
    }

    /// Build ai ToolSpecs for all registered tools.
    pub fn to_tool_specs(&self) -> Vec<ai::ToolSpec> {
        self.tools
            .iter()
            .map(|(name, tool)| ai::ToolSpec {
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

/// 检查给定路径是否在当前工作目录内。
/// 对拼接后的路径尝试 canonicalize（解析符号链接和 `..`），
/// 不存在的路径则手动规范化 `..` 组件。
pub(crate) fn is_within_cwd(path: &Path) -> bool {
    let cwd = match std::env::current_dir() {
        Ok(d) => d,
        Err(_) => return false,
    };

    let resolved_cwd = match cwd.canonicalize() {
        Ok(d) => d,
        Err(_) => return false,
    };

    let absolute: PathBuf = if path.is_absolute() {
        path.to_path_buf()
    } else {
        cwd.join(path)
    };

    // 路径已存在：canonicalize 解析符号链接和 ..
    if let Ok(canonical) = absolute.canonicalize() {
        return canonical.starts_with(&resolved_cwd);
    }

    // 路径不存在：手动规范化 .. 组件
    normalize_path(&absolute).starts_with(&resolved_cwd)
}

/// 手动规范化路径中的 `.` 和 `..` 组件（不解析符号链接）。
fn normalize_path(path: &Path) -> PathBuf {
    let mut components = Vec::new();
    for comp in path.components() {
        match comp {
            std::path::Component::ParentDir => {
                components.pop();
            }
            std::path::Component::CurDir => {}
            comp => components.push(comp),
        }
    }
    components.iter().collect()
}
