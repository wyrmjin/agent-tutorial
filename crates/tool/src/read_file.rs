//! Read file tool — read file contents with line limit and timeout.
//!
//! 路径限制：默认只允许读取当前工作目录内的文件。
//! 读取外部文件需要用户审批，审批通过后该路径会被加入白名单。

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::{Tool, ToolOutput, is_within_cwd};
use logger::{debug, error, warn};

pub struct ReadFileTool {
    timeout: Duration,
    /// 已批准读取的外部路径白名单
    approved_external_paths: Arc<Mutex<HashSet<PathBuf>>>,
}

impl ReadFileTool {
    pub fn new(timeout_secs: u64) -> Self {
        Self {
            timeout: Duration::from_secs(timeout_secs),
            approved_external_paths: Arc::new(Mutex::new(HashSet::new())),
        }
    }

    /// 检查路径是否已经被批准。
    fn is_path_approved(&self, path: &Path) -> bool {
        if let Ok(paths) = self.approved_external_paths.lock() {
            // 同时匹配原始路径和规范化路径，避免 canonicalize 竞态
            if paths.contains(path) {
                return true;
            }
            if let Ok(canonical) = path.canonicalize() {
                return paths.contains(&canonical);
            }
            false
        } else {
            false
        }
    }
}

impl Default for ReadFileTool {
    fn default() -> Self {
        Self::new(30)
    }
}

#[async_trait::async_trait]
impl Tool for ReadFileTool {
    fn name(&self) -> &str {
        "read_file"
    }

    fn description(&self) -> &str {
        "Read file contents from the filesystem. Returns the file text with line numbers. \
         Use to inspect source code, configuration, logs, or any text file. \
         Supports an optional limit parameter to read only the first N lines. \
         Only files within the current working directory can be read directly; \
         reading files outside requires user approval."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the file to read"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to read (optional)"
                }
            },
            "required": ["path"]
        })
    }

    fn approve(&self, input: &serde_json::Value) {
        if let Some(path_str) = input.get("path").and_then(|v| v.as_str()) {
            if let Ok(mut paths) = self.approved_external_paths.lock() {
                let path = PathBuf::from(path_str);
                // 同时存入原始路径和规范化路径，避免竞态条件
                paths.insert(path.clone());
                if let Ok(canonical) = path.canonicalize() {
                    paths.insert(canonical);
                }
            }
        }
    }

    async fn execute(&self, input: serde_json::Value) -> ToolOutput {
        let path_str = input.get("path").and_then(|v| v.as_str()).unwrap_or("");

        if path_str.is_empty() {
            return ToolOutput::error("Error: path parameter is required");
        }

        let path = PathBuf::from(path_str);
        let limit = input
            .get("limit")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize);

        debug!(%path_str, ?limit, "read_file execute start");

        // 路径安全检查：必须在工作目录内或已获批准
        if !is_within_cwd(&path) {
            if self.is_path_approved(&path) {
                debug!(%path_str, "external path already approved");
            } else {
                warn!(%path_str, "read_file blocked: path outside working directory");
                return ToolOutput::approval(
                    format!(
                        "路径限制：文件 `{path_str}` 不在当前工作目录内。\n\
                         需要用户批准才能读取此文件。\n\
                         请询问用户是否同意读取，用户同意后系统会批准该路径并重新执行读取。"
                    ),
                    path_str,
                );
            }
        }

        match tokio::time::timeout(self.timeout, tokio::fs::read_to_string(&path)).await {
            Err(_) => {
                warn!(%path_str, timeout = ?self.timeout, "read_file timed out");
                ToolOutput::error(format!("读取文件超时 ({:?})", self.timeout))
            }
            Ok(Err(e)) => {
                error!(%path_str, error = %e, "read_file failed");
                let msg = match e.kind() {
                    std::io::ErrorKind::NotFound => format!("文件不存在: {path_str}"),
                    std::io::ErrorKind::PermissionDenied => format!("没有权限读取: {path_str}"),
                    std::io::ErrorKind::InvalidData => {
                        format!("无法以文本方式读取，文件可能是二进制或非 UTF-8 编码: {path_str}")
                    }
                    _ => format!("读取文件失败: {e}"),
                };
                ToolOutput::error(msg)
            }
            Ok(Ok(content)) => {
                let line_count = content.lines().count();
                let result = match limit {
                    Some(n) if n < line_count => {
                        let truncated: String =
                            content.lines().take(n).collect::<Vec<&str>>().join("\n");
                        let remaining = line_count - n;
                        let hint = format!(
                            "\n\n--- 文件内容被截断 (已显示 {n} 行 / 共 {line_count} 行，剩余 {remaining} 行) ---"
                        );
                        debug!(
                            %path_str,
                            total_lines = line_count,
                            read_lines = n,
                            remaining,
                            bytes = content.len(),
                            "read_file completed (truncated)"
                        );
                        truncated + &hint
                    }
                    _ => {
                        debug!(
                            %path_str,
                            lines = line_count,
                            bytes = content.len(),
                            "read_file completed"
                        );
                        content
                    }
                };

                ToolOutput::ok(result)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rt() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .unwrap()
    }

    #[test]
    fn execute_empty_path_returns_error() {
        let tool = ReadFileTool::new(5);
        let result = rt().block_on(tool.execute(serde_json::json!({})));
        assert!(result.is_error);
        assert!(result.content.contains("required"));
    }

    #[test]
    fn execute_file_not_found() {
        let tool = ReadFileTool::new(5);
        let result =
            rt().block_on(tool.execute(serde_json::json!({"path": "/nonexistent/path/file.txt"})));
        assert!(!result.is_error);
        assert!(result.content.contains("路径限制"));
        assert!(result.needs_approval);
    }

    #[test]
    fn execute_cwd_internal_file_not_found() {
        let tool = ReadFileTool::new(5);
        let path = std::env::current_dir()
            .unwrap()
            .join("nonexistent_test_file_12345.txt");
        let result =
            rt().block_on(tool.execute(serde_json::json!({"path": path.to_str().unwrap()})));
        assert!(result.is_error);
        assert!(result.content.contains("文件不存在"));
        assert!(!result.needs_approval);
    }

    #[test]
    fn execute_read_file_success() {
        let tool = ReadFileTool::new(5);
        let manifest = std::env::current_dir().unwrap().join("Cargo.toml");
        let result =
            rt().block_on(tool.execute(serde_json::json!({"path": manifest.to_str().unwrap()})));
        assert!(!result.is_error);
        assert!(result.content.contains("[package]") || result.content.contains("[workspace"));
    }

    #[test]
    fn execute_with_limit_truncates() {
        let tool = ReadFileTool::new(5);
        let manifest = std::env::current_dir().unwrap().join("Cargo.toml");
        let result = rt().block_on(
            tool.execute(serde_json::json!({"path": manifest.to_str().unwrap(), "limit": 1})),
        );
        assert!(!result.is_error);
        assert!(result.content.contains("文件内容被截断"));
    }

    #[test]
    fn execute_with_limit_larger_than_file() {
        let tool = ReadFileTool::new(5);
        let manifest = std::env::current_dir().unwrap().join("Cargo.toml");
        let result = rt().block_on(
            tool.execute(serde_json::json!({"path": manifest.to_str().unwrap(), "limit": 1000})),
        );
        assert!(!result.is_error);
        assert!(!result.content.contains("文件内容被截断"));
    }

    #[test]
    fn name_and_description() {
        let tool = ReadFileTool::default();
        assert_eq!(tool.name(), "read_file");
        assert!(tool.description().contains("file"));
    }

    #[test]
    fn parameters_requires_path() {
        let tool = ReadFileTool::default();
        let params = tool.parameters();
        assert_eq!(params["type"], "object");
        let required: Vec<_> = params["required"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert!(required.contains(&"path"));
    }

    #[test]
    fn execute_external_path_blocked() {
        let tool = ReadFileTool::new(5);
        let result = rt().block_on(tool.execute(serde_json::json!({"path": "/etc/passwd"})));
        assert!(!result.is_error);
        assert!(result.needs_approval);
        assert!(result.approval_path.is_some());
        assert!(result.content.contains("路径限制"));
    }

    #[test]
    fn approve_and_read_external_file() {
        let tool = ReadFileTool::new(5);
        let path = std::env::temp_dir().join("agent-tutorial-test-read.txt");
        std::fs::write(&path, "hello external world").unwrap();

        // 第一次读取：被拒绝
        let result =
            rt().block_on(tool.execute(serde_json::json!({"path": path.to_str().unwrap()})));
        assert!(result.needs_approval);

        // 批准路径
        let input = serde_json::json!({"path": path.to_str().unwrap()});
        tool.approve(&input);

        // 第二次读取：成功
        let result =
            rt().block_on(tool.execute(serde_json::json!({"path": path.to_str().unwrap()})));
        assert!(!result.is_error);
        assert!(result.content.contains("hello external world"));

        // 清理
        std::fs::remove_file(&path).ok();
    }
}
