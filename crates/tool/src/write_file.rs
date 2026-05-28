//! Write file tool — write content to files with path safety check.

use std::path::PathBuf;
use std::time::Duration;

use crate::{is_within_cwd, Tool, ToolOutput};
use logger::{debug, error, warn};

pub struct WriteFileTool {
    timeout: Duration,
}

impl WriteFileTool {
    pub fn new(timeout_secs: u64) -> Self {
        Self {
            timeout: Duration::from_secs(timeout_secs),
        }
    }
}

impl Default for WriteFileTool {
    fn default() -> Self {
        Self::new(30)
    }
}

#[async_trait::async_trait]
impl Tool for WriteFileTool {
    fn name(&self) -> &str {
        "write_file"
    }

    fn description(&self) -> &str {
        "Write content to a file"
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]})
    }

    async fn execute(&self, input: serde_json::Value) -> ToolOutput {
        let path_str = input.get("path").and_then(|v| v.as_str()).unwrap_or("");

        if path_str.is_empty() {
            return ToolOutput::error("Error: path parameter is required");
        }
        let content_str = input.get("content").and_then(|v| v.as_str()).unwrap_or("");
        if content_str.is_empty() {
            return ToolOutput::error("Error: content parameter is required");
        }
        let path = PathBuf::from(path_str);

        debug!(%path_str, %content_str, "write_file execute start");

        // 路径安全检查：只允许写入当前工作目录内
        if !is_within_cwd(&path) {
            warn!(%path_str, "write_file blocked: path outside working directory");
            return ToolOutput::error(format!(
                "路径限制：文件 `{path_str}` 不在当前工作目录内，不允许写入。"
            ));
        }

        // 确保父目录存在
        if let Some(parent) = path.parent() {
            if let Err(e) = tokio::fs::create_dir_all(parent).await {
                error!(dir = %parent.display(), error = %e, "创建目录失败");
                return ToolOutput::error(format!("创建目录失败: {e}"));
            }
        }

        match tokio::time::timeout(self.timeout, tokio::fs::write(&path, &content_str)).await {
            Err(_) => {
                warn!(%path_str, timeout = ?self.timeout, "write_file timed out");
                ToolOutput::error(format!("写入文件超时 ({:?})", self.timeout))
            }
            Ok(Err(e)) => {
                error!(%path_str, error = %e, "write_file failed");
                let msg = match e.kind() {
                    std::io::ErrorKind::NotFound => format!("文件不存在: {path_str}"),
                    std::io::ErrorKind::PermissionDenied => format!("没有权限写入: {path_str}"),
                    std::io::ErrorKind::InvalidData => {
                        format!("无法以文本方式写入，文件可能是二进制或非 UTF-8 编码: {path_str}")
                    }
                    _ => format!("写入文件失败: {e}"),
                };
                ToolOutput::error(msg)
            }
            Ok(Ok(())) => {
                debug!(%path_str, bytes = content_str.len(), "write_file completed");
                ToolOutput::ok(format!("成功写入文件: {path_str}"))
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
        let tool = WriteFileTool::new(5);
        let result = rt().block_on(
            tool.execute(serde_json::json!({"content": "hello"})),
        );
        assert!(result.is_error);
        assert!(result.content.contains("required"));
    }

    #[test]
    fn execute_empty_content_returns_error() {
        let tool = WriteFileTool::new(5);
        let result =
            rt().block_on(tool.execute(serde_json::json!({"path": "/tmp/test.txt"})));
        assert!(result.is_error);
        assert!(result.content.contains("required"));
    }

    #[test]
    fn execute_write_file_success() {
        let tool = WriteFileTool::new(5);
        let tmp = std::env::current_dir().unwrap().join("target/test-tmp/tool_test_write_success.txt");
        let path_str = tmp.to_str().unwrap();

        let result = rt().block_on(
            tool.execute(serde_json::json!({"path": path_str, "content": "hello world"})),
        );
        assert!(!result.is_error);
        assert!(result.content.contains("成功写入"));

        // 验证内容正确写入
        let written = std::fs::read_to_string(&tmp).unwrap();
        assert_eq!(written, "hello world");

        // 清理
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn execute_creates_parent_directories() {
        let tool = WriteFileTool::new(5);
        let tmp = std::env::current_dir().unwrap().join("target/test-tmp/nested/subdir/test.txt");
        let path_str = tmp.to_str().unwrap();

        let result = rt().block_on(
            tool.execute(serde_json::json!({"path": path_str, "content": "nested content"})),
        );
        assert!(!result.is_error);

        let written = std::fs::read_to_string(&tmp).unwrap();
        assert_eq!(written, "nested content");

        // 清理
        std::fs::remove_file(&tmp).ok();
        std::fs::remove_dir(tmp.parent().unwrap()).ok();
        std::fs::remove_dir(tmp.parent().unwrap().parent().unwrap()).ok();
    }

    #[test]
    fn execute_overwrite_existing_file() {
        let tool = WriteFileTool::new(5);
        let tmp = std::env::current_dir().unwrap().join("target/test-tmp/tool_test_overwrite.txt");
        let path_str = tmp.to_str().unwrap();

        // 先写一次
        rt().block_on(
            tool.execute(serde_json::json!({"path": path_str, "content": "first"})),
        );
        // 覆盖写入
        let result = rt().block_on(
            tool.execute(serde_json::json!({"path": path_str, "content": "second"})),
        );
        assert!(!result.is_error);

        let written = std::fs::read_to_string(&tmp).unwrap();
        assert_eq!(written, "second");

        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn name_and_description() {
        let tool = WriteFileTool::default();
        assert_eq!(tool.name(), "write_file");
        assert!(tool.description().contains("Write"));
    }

    #[test]
    fn parameters_requires_path_and_content() {
        let tool = WriteFileTool::default();
        let params = tool.parameters();
        assert_eq!(params["type"], "object");
        let required: Vec<_> = params["required"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert!(required.contains(&"path"));
        assert!(required.contains(&"content"));
    }
}
