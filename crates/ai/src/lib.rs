//! AI 通信层:协议(Protocol)/传输(Transport)/供应商(Provider)三层正交抽象,
//! 由 Model 组合、ModelRegistry 管理。
//!
//! 一次请求的数据流:
//! Model → Protocol::build_body(纯函数) → Transport(HTTP) → Protocol::new_decoder()(解码) → StreamChunk

pub mod error;
pub mod message;
pub mod model;
pub mod protocol;
pub mod provider;
pub mod registry;
pub mod stream;
pub mod transport;

pub use error::AiError;
pub use message::{Message, Role, ToolCallRequest, ToolSpec};
pub use model::{Capabilities, LanguageModel, Model};
pub use protocol::openai_completions::OpenAiCompletionsProtocol;
pub use protocol::{Protocol, ProtocolKind, SamplingParams, ThinkingLevel};
pub use provider::{AuthStyle, Endpoint, GenericProvider, Provider};
pub use registry::ModelRegistry;
pub use stream::{
    ByteStream, DecodingStream, SseFrameReader, StopReason, StreamChunk, StreamChunkIterator,
    StreamDecoder, Usage,
};
pub use transport::Transport;
