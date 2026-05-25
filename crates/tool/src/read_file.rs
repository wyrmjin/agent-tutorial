//! Read file tool — read file contents with line limit and timeout.

use std::path::PathBuf;
use std::time::Duration;

use crate::{Tool, ToolOutput};
use logger::{debug, error, warn};

pub struct ReadFileTool {
    timeout: Duration,
}

impl ReadFileTool {
    pub fn new(timeout_secs: u64) -> Self {
        Self {
            timeout: Duration::from_secs(timeout_secs),
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
         Supports an optional limit parameter to read only the first N lines."
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

    async fn execute(&self, input: serde_json::Value) -> ToolOutput {
        let path_str = input
            .get("path")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        if path_str.is_empty() {
            return ToolOutput {
                content: "Error: path parameter is required".to_string(),
                is_error: true,
            };
        }

        let path = PathBuf::from(path_str);
        let limit = input
            .get("limit")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize);

        debug!(%path_str, ?limit, "read_file execute start");

        match tokio::time::timeout(self.timeout, tokio::fs::read_to_string(&path)).await {
            Err(_) => {
                warn!(%path_str, timeout = ?self.timeout, "read_file timed out");
                ToolOutput {
                    content: format!("读取文件超时 ({:?})", self.timeout),
                    is_error: true,
                }
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
                ToolOutput {
                    content: msg,
                    is_error: true,
                }
            }
            Ok(Ok(content)) => {
                let line_count = content.lines().count();
                let result = match limit {
                    Some(n) if n < line_count => {
                        let truncated: String = content
                            .lines()
                            .take(n)
                            .collect::<Vec<&str>>()
                            .join("\n");
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

                ToolOutput {
                    content: result,
                    is_error: false,
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rt() -> tokio::runtime::Runtime {
        tokio::runtime::Runtime::new().unwrap()
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
        assert!(result.is_error);
        assert!(result.content.contains("文件不存在"));
    }

    #[test]
    fn execute_read_file_success() {
        let tool = ReadFileTool::new(5);
        // 读取项目的 Cargo.toml（必定存在）
        let manifest = std::env::current_dir()
            .unwrap()
            .join("Cargo.toml");
        let result =
            rt().block_on(tool.execute(serde_json::json!({"path": manifest.to_str().unwrap()})));
        assert!(!result.is_error);
        assert!(result.content.contains("[package]") || result.content.contains("[workspace"));
    }

    #[test]
    fn execute_with_limit_truncates() {
        let tool = ReadFileTool::new(5);
        let manifest = std::env::current_dir()
            .unwrap()
            .join("Cargo.toml");
        let result = rt().block_on(
            tool.execute(serde_json::json!({"path": manifest.to_str().unwrap(), "limit": 1})),
        );
        assert!(!result.is_error);
        assert!(result.content.contains("文件内容被截断"));
    }

    #[test]
    fn execute_with_limit_larger_than_file() {
        let tool = ReadFileTool::new(5);
        let manifest = std::env::current_dir()
            .unwrap()
            .join("Cargo.toml");
        let result = rt().block_on(
            tool.execute(serde_json::json!({"path": manifest.to_str().unwrap(), "limit": 1000})),
        );
        assert!(!result.is_error);
        // 不应有截断提示
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
}
