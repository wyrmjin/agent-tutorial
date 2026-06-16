//! Streaming response model and the chunk iterator the agent consumes.

use std::collections::VecDeque;
use std::pin::Pin;

use bytes::Bytes;
use futures::stream::Stream;
use futures::StreamExt;

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
///
/// Buffers raw bytes (not a String) so a multi-byte UTF-8 character split
/// across two byte chunks is reassembled correctly: decoding happens once a
/// whole line is available, never on a partial chunk.
#[derive(Default)]
pub struct SseFrameReader {
    buffer: Vec<u8>,
}

impl SseFrameReader {
    pub fn new() -> Self {
        Self { buffer: Vec::new() }
    }

    /// Append bytes and return all complete lines now available
    /// (trimmed; empty lines skipped).
    pub fn push(&mut self, bytes: &[u8]) -> Vec<String> {
        self.buffer.extend_from_slice(bytes);
        let mut lines = Vec::new();
        while let Some(pos) = self.buffer.iter().position(|&b| b == b'\n') {
            let line_bytes = self.buffer.drain(..=pos).collect::<Vec<_>>();
            // 按完整行解码(可能含末尾的 \n 以及 \r\n 的 \r), 解码后再 trim。
            let line = String::from_utf8_lossy(&line_bytes).trim().to_string();
            if !line.is_empty() {
                lines.push(line);
            }
        }
        lines
    }

    /// At stream end, return the trailing buffered line if non-empty.
    pub fn flush(&mut self) -> Option<String> {
        if self.buffer.is_empty() {
            return None;
        }
        let line = String::from_utf8_lossy(&self.buffer).trim().to_string();
        self.buffer.clear();
        if line.is_empty() { None } else { Some(line) }
    }
}

/// A protocol-specific streaming decoder. Owns accumulation state
/// (e.g. partially-streamed tool calls). Does its own framing.
pub trait StreamDecoder: Send {
    /// Feed raw bytes; return any chunks now decodable.
    fn feed(&mut self, bytes: &[u8]) -> Result<Vec<StreamChunk>, AiError>;
    /// Called once the byte stream ends; flush any pending state.
    fn finish(&mut self) -> Result<Vec<StreamChunk>, AiError>;
}

/// Boxed byte stream as produced by `reqwest::Response::bytes_stream`.
pub type ByteStream = Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send>>;

/// Generic adapter: drives a byte stream through a `StreamDecoder`,
/// exposing the pull-based `StreamChunkIterator` the agent consumes.
/// Reused across all protocols.
pub struct DecodingStream {
    byte_stream: ByteStream,
    decoder: Box<dyn StreamDecoder>,
    pending: VecDeque<StreamChunk>,
    finished: bool,
}

impl DecodingStream {
    pub fn new(byte_stream: ByteStream, decoder: Box<dyn StreamDecoder>) -> Self {
        Self {
            byte_stream,
            decoder,
            pending: VecDeque::new(),
            finished: false,
        }
    }
}

#[async_trait::async_trait]
impl StreamChunkIterator for DecodingStream {
    async fn next(&mut self) -> Result<Option<StreamChunk>, AiError> {
        loop {
            if let Some(c) = self.pending.pop_front() {
                return Ok(Some(c));
            }
            if self.finished {
                return Ok(None);
            }
            match self.byte_stream.next().await {
                Some(Ok(bytes)) => self.pending.extend(self.decoder.feed(&bytes)?),
                Some(Err(e)) => return Err(e.into()),
                None => {
                    self.pending.extend(self.decoder.finish()?);
                    self.finished = true;
                }
            }
        }
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

    #[test]
    fn multibyte_char_split_across_chunks_is_preserved() {
        // "data: 你" 的 UTF-8 字节: d a t a :   <space> 你=E4 BD A0
        // 故意把第一块切在 "你" 的中间(E4 BD),第二块补 A0 + 换行。
        // 若每块都 lossy 解码, "你" 会被拆成两个 U+FFFD; 用字节 buffer 则能完整恢复。
        let mut r = SseFrameReader::new();
        assert!(r.push(b"data: \xe4\xbd").is_empty()); // 半个 "你"
        let lines = r.push(b"\xa0\n"); // 补全 "你" + 换行
        assert_eq!(lines, vec!["data: 你".to_string()]);
    }

    #[test]
    fn handles_crlf_line_endings() {
        // SSE 帧常以 \r\n 结尾; trim 应剥掉 \r, 不残留。
        let mut r = SseFrameReader::new();
        let lines = r.push(b"data: a\r\ndata: b\r\n");
        assert_eq!(lines, vec!["data: a".to_string(), "data: b".to_string()]);
    }
}

#[cfg(test)]
mod decoding_stream_tests {
    use super::*;
    use bytes::Bytes;
    use futures::stream;

    /// Decoder that turns each fed byte-chunk into one Text chunk,
    /// and emits a Finished on finish().
    struct EchoDecoder;
    impl StreamDecoder for EchoDecoder {
        fn feed(&mut self, bytes: &[u8]) -> Result<Vec<StreamChunk>, AiError> {
            Ok(vec![StreamChunk::Text(String::from_utf8_lossy(bytes).into_owned())])
        }
        fn finish(&mut self) -> Result<Vec<StreamChunk>, AiError> {
            Ok(vec![StreamChunk::Finished {
                stop_reason: StopReason::EndTurn,
                usage: Usage::default(),
            }])
        }
    }

    #[test]
    fn yields_decoded_chunks_then_finish() {
        // 用 futures 自带的 block_on 跑异步, 避免给 ai crate 引入 tokio dev-dep。
        futures::executor::block_on(async {
            let byte_stream = stream::iter(vec![
                Ok(Bytes::from_static(b"foo")),
                Ok(Bytes::from_static(b"bar")),
            ]);
            let mut ds = DecodingStream::new(Box::pin(byte_stream), Box::new(EchoDecoder));

            let mut texts = Vec::new();
            let mut finished = false;
            while let Some(chunk) = ds.next().await.unwrap() {
                match chunk {
                    StreamChunk::Text(t) => texts.push(t),
                    StreamChunk::Finished { .. } => finished = true,
                    _ => {}
                }
            }
            assert_eq!(texts, vec!["foo".to_string(), "bar".to_string()]);
            assert!(finished);
        });
    }
}
