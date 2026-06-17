//! The main agent that runs the conversation loop.

use std::panic::AssertUnwindSafe;
use std::sync::Arc;

use ai::{LanguageModel, Message};
use futures::FutureExt;
use tool::ToolRegistry;
use tracing::error;

use crate::config::AgentConfig;
use crate::event_stream::EventStream;
use crate::handle::AgentHandle;

/// The main agent that runs the conversation loop.
///
/// Conversation history is stored in a shared slot (`Arc<Mutex<Vec<Message>>>`)
/// so the background agent loop can automatically write it back when finished,
/// eliminating the need for manual `set_history` calls.
pub struct Agent {
    model: Arc<dyn LanguageModel>,
    history: Arc<std::sync::Mutex<Vec<Message>>>,
    /// Guard flag to prevent concurrent `run_stream` calls.
    running: Arc<std::sync::Mutex<bool>>,
}

impl Agent {
    /// Create a new agent with the given ai and system prompt.
    pub fn new(model: Arc<dyn LanguageModel>, system_prompt: String) -> Self {
        let mut history = Vec::new();
        if !system_prompt.is_empty() {
            history.push(Message::system(system_prompt));
        }
        Self {
            model,
            history: Arc::new(std::sync::Mutex::new(history)),
            running: Arc::new(std::sync::Mutex::new(false)),
        }
    }

    /// Access the conversation history.
    ///
    /// Returns a cloned snapshot of the current history.
    /// The history is automatically updated after each `run_stream` completes.
    pub fn history(&self) -> Vec<Message> {
        self.history.lock().unwrap().clone()
    }

    /// Manually set the conversation history.
    ///
    /// This is rarely needed — `run_stream` automatically writes back the
    /// final history on completion. Use this only when you need to override
    /// or restore a previous conversation.
    pub fn set_history(&self, history: Vec<Message>) {
        *self.history.lock().unwrap() = history;
    }

    /// Run one turn: spawn the agent loop in a background task and return
    /// a handle for streaming consumption and control.
    ///
    /// When the background task finishes, it automatically writes the final
    /// conversation history back into this agent, so the next call to
    /// `run_stream` (or `history()`) sees the updated state.
    ///
    /// # Panics
    ///
    /// Panics if called while a previous `run_stream` is still running.
    /// `run_stream` must be called sequentially — wait for the previous
    /// stream to complete before starting a new one.
    pub fn run_stream(
        &self,
        user_input: &str,
        config: &AgentConfig,
        tools: Arc<ToolRegistry>,
    ) -> AgentHandle {
        let (sender, event_stream) = EventStream::new();
        let (abort_tx, abort_rx) = tokio::sync::mpsc::unbounded_channel();
        let (steer_tx, steer_rx) = tokio::sync::mpsc::unbounded_channel();

        // Prevent concurrent runs — the history is moved out via take
        // and must not be taken again until finalize() writes it back.
        {
            let mut running = self.running.lock().unwrap();
            assert!(
                !*running,
                "run_stream called while a previous run is still active"
            );
            *running = true;
        }

        // Take current history out of the shared slot and add user message.
        let mut history = std::mem::take(&mut *self.history.lock().unwrap());
        history.push(ai::Message::user(user_input));

        let model = Arc::clone(&self.model);
        let tool_specs = tools.to_tool_specs();
        let max_rounds = config.max_tool_rounds;
        let history_slot = Arc::clone(&self.history);
        let running_flag = Arc::clone(&self.running);
        let loop_steer_tx = steer_tx.clone();

        // Record pre-spawn history length so we can truncate on panic
        // instead of cloning the entire history.
        // Save a backup of the original history (before the user message)
        // so we can restore it if the background task panics.
        let original_history = history.clone();

        tokio::spawn(async move {
            let result = AssertUnwindSafe(crate::agent_loop::run_loop_bg(
                model,
                history,
                tool_specs,
                tools,
                max_rounds,
                sender,
                abort_rx,
                steer_rx,
                loop_steer_tx,
                history_slot.clone(),
            ))
            .catch_unwind()
            .await;

            if let Err(panic_info) = result {
                // Background task panicked — restore the original history
                // to prevent data loss.
                error!("Agent background task panicked: {:?}", panic_info);
                *history_slot.lock().unwrap() = original_history;
            }

            // Clear the running flag so the next run_stream can proceed.
            *running_flag.lock().unwrap() = false;
        });

        AgentHandle::new(event_stream, abort_tx, steer_tx)
    }
}
