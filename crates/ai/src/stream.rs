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

/// Splits a byte stream into trimmed, non-empty SSE lines.
/// Stateless w.r.t. transport — feed it bytes, get back complete lines.
#[derive(Default)]
pub struct SseFrameReader {
    buffer: String,
}

impl SseFrameReader {
    pub fn new() -> Self {
        Self { buffer: String::new() }
    }

    /// Append bytes and return all complete lines now available
    /// (trimmed; empty lines skipped).
    pub fn push(&mut self, bytes: &[u8]) -> Vec<String> {
        self.buffer.push_str(&String::from_utf8_lossy(bytes));
        let mut lines = Vec::new();
        while let Some(pos) = self.buffer.find('\n') {
            let line = self.buffer[..pos].trim().to_string();
            self.buffer = self.buffer[pos + 1..].to_string();
            if !line.is_empty() {
                lines.push(line);
            }
        }
        lines
    }

    /// At stream end, return the trailing buffered line if non-empty.
    pub fn flush(&mut self) -> Option<String> {
        let line = std::mem::take(&mut self.buffer).trim().to_string();
        if line.is_empty() { None } else { Some(line) }
    }
}

#[cfg(test)]
mod sse_tests {
    use super::SseFrameReader;

    #[test]
    fn splits_complete_lines_and_skips_blanks() {
        let mut r = SseFrameReader::new();
        let lines = r.push(b"data: a\n\ndata: b\n");
        assert_eq!(lines, vec!["data: a".to_string(), "data: b".to_string()]);
    }

    #[test]
    fn joins_half_line_across_pushes() {
        let mut r = SseFrameReader::new();
        assert!(r.push(b"data: hel").is_empty());
        let lines = r.push(b"lo\n");
        assert_eq!(lines, vec!["data: hello".to_string()]);
    }

    #[test]
    fn flush_returns_trailing_nonempty() {
        let mut r = SseFrameReader::new();
        assert!(r.push(b"data: tail").is_empty());
        assert_eq!(r.flush(), Some("data: tail".to_string()));
        assert_eq!(r.flush(), None);
    }
}
