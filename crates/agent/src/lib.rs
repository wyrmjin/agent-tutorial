//! Agent loop: orchestrates the conversation between user, LLM, and tools.
//!
//! The [`Agent`] drives the core loop:
//! 1. Send user message + history to the LLM
//! 2. If LLM responds with text -> yield to caller
//! 3. If LLM requests tool use -> execute tool, append result, loop back to 1
//! 4. If LLM ends turn -> done

pub mod agent;
mod agent_loop;
mod approval;
pub mod config;
pub mod event;
pub mod event_stream;
pub mod handle;
#[cfg(test)]
mod tests;

pub use agent::Agent;
pub use config::AgentConfig;
pub use event::{AgentEvent, AgentResult};
pub use handle::{AgentHandle, ApprovalDecision, SteeringMessage};
