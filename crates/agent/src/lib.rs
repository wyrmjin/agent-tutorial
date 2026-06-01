//! Agent loop: orchestrates the conversation between user, LLM, and tools.
//!
//! The [`Agent`] drives the core loop:
//! 1. Send user message + history to the LLM
//! 2. If LLM responds with text -> yield to caller
//! 3. If LLM requests tool use -> execute tool, append result, loop back to 1
//! 4. If LLM ends turn -> done

pub mod event_stream;

use std::sync::Arc;

use logger::{debug, info, warn};
use provider::{Message, Provider, ProviderError, Role, StopReason, StreamChunk, ToolCallRequest};
use tool::ToolRegistry;

/// Errors that can occur during agent execution.
#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    /// An error from the LLM provider.
    #[error("Provider error: {0}")]
    Provider(#[from] ProviderError),

    /// Maximum tool rounds exceeded.
    #[error("Maximum tool rounds exceeded ({max})")]
    MaxRoundsExceeded { max: usize },

    /// A tool was not found in the registry.
    #[error("Unknown tool: {name}")]
    UnknownTool { name: String },

    /// User denied the approval.
    #[error("User denied approval: {reason}")]
    ApprovalDenied { reason: String },
}

/// Configuration for an agent run.
#[derive(Debug, Clone)]
pub struct AgentConfig {
    pub max_tool_rounds: usize,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_tool_rounds: 25,
        }
    }
}

/// Events emitted during an agent run.
#[derive(Debug, Clone)]
pub enum AgentEvent {
    /// The LLM produced text content.
    Text(String),
    /// The LLM wants to invoke a tool.
    ToolRequest {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    /// A tool execution result.
    ToolResponse {
        id: String,
        content: String,
        is_error: bool,
    },
    /// A tool needs user approval before it can proceed.
    ApprovalRequired {
        tool_call_id: String,
        tool_name: String,
        path: String,
        message: String,
    },
    /// The conversation turn is complete.
    TurnEnd { usage: provider::Usage },
    /// An error occurred during execution.
    Error {
        message: String,
        /// If true, the agent can continue; if false, the agent must stop.
        recoverable: bool,
    },
    /// The agent has finished processing. This is always the last event.
    Done {
        messages: Vec<provider::Message>,
    },
}

/// Structured result returned after the agent finishes.
#[derive(Debug, Clone)]
pub struct AgentResult {
    pub messages: Vec<provider::Message>,
    pub total_turns: usize,
    pub usage: provider::Usage,
}

impl Default for AgentResult {
    fn default() -> Self {
        Self {
            messages: Vec::new(),
            total_turns: 0,
            usage: provider::Usage::default(),
        }
    }
}

/// Control messages that can be sent to a running agent.
pub enum SteeringMessage {
    /// Abort the agent execution.
    Abort,
    /// Inject a message into the conversation history.
    InjectMessage(provider::Message),
}

/// Handle to a running agent. Provides the event stream plus control methods.
pub struct AgentHandle {
    stream: event_stream::EventStream<AgentEvent, AgentResult>,
    abort_tx: tokio::sync::mpsc::UnboundedSender<()>,
    steer_tx: tokio::sync::mpsc::UnboundedSender<SteeringMessage>,
}

impl AgentHandle {
    /// Get a mutable reference to the event stream for async iteration.
    pub fn stream_mut(&mut self) -> &mut event_stream::EventStream<AgentEvent, AgentResult> {
        &mut self.stream
    }

    /// Request the agent to abort.
    pub fn abort(&self) {
        let _ = self.abort_tx.send(());
    }

    /// Inject a message into the agent's conversation.
    pub fn inject(&self, msg: provider::Message) {
        let _ = self.steer_tx.send(SteeringMessage::InjectMessage(msg));
    }

    /// Check if the stream is done.
    pub fn is_done(&self) -> bool {
        self.stream.is_done()
    }
}

/// Saved state for a tool call waiting for user approval.
struct PendingApproval {
    id: String,
    name: String,
    input: serde_json::Value,
    round: usize,
    /// Tool calls queued after this one, to be executed after approval is resolved.
    remaining_calls: Vec<ToolCallRequest>,
}

/// The main agent that runs the conversation loop.
pub struct Agent {
    provider: Arc<dyn Provider>,
    history: Vec<Message>,
    /// Tool call waiting for user approval. When set, the agent loop is paused.
    pending_approval: Option<PendingApproval>,
}

impl Agent {
    /// Create a new agent with the given provider and system prompt.
    pub fn new(provider: Arc<dyn Provider>, system_prompt: String) -> Self {
        let mut history = Vec::new();
        if !system_prompt.is_empty() {
            history.push(Message {
                role: Role::System,
                content: system_prompt,
                tool_calls: None,
                tool_call_id: None,
            });
        }
        Self {
            provider,
            history,
            pending_approval: None,
        }
    }

    /// Access the conversation history.
    pub fn history(&self) -> &[Message] {
        &self.history
    }

    /// Check if the agent is waiting for user approval.
    pub fn has_pending_approval(&self) -> bool {
        self.pending_approval.is_some()
    }

    /// Run one turn: given user input, loop through LLM <-> tool calls
    /// and yield events to the caller.
    pub async fn run(
        &mut self,
        user_input: &str,
        config: &AgentConfig,
        tools: &ToolRegistry,
    ) -> Result<Vec<AgentEvent>, AgentError> {
        let mut events = Vec::new();

        self.history.push(Message {
            role: Role::User,
            content: user_input.to_string(),
            tool_calls: None,
            tool_call_id: None,
        });

        debug!(
            user_input,
            turn = self.history.len() / 2,
            "agent start turn"
        );

        let tool_specs = tools.to_tool_specs();
        self.run_loop(config, tools, &tool_specs, 0, &mut events)
            .await?;
        Ok(events)
    }

    /// Resolve a pending approval and continue the agent loop.
    /// Call this after the user has approved or denied the file read request.
    pub async fn resolve_approval(
        &mut self,
        approved: bool,
        user_input: &str,
        config: &AgentConfig,
        tools: &ToolRegistry,
    ) -> Result<Vec<AgentEvent>, AgentError> {
        let pending = self
            .pending_approval
            .take()
            .expect("resolve_approval called without pending approval");

        // 如果用户批准，通知工具完成审批
        if approved {
            tools.approve(&pending.name, &pending.input);
        }

        let output = if approved {
            // 重新执行工具，这次路径已在白名单中
            match tools.execute(&pending.name, pending.input.clone()).await {
                Some(out) => out,
                None => tool::ToolOutput::error(format!("未知工具: {}", pending.name)),
            }
        } else {
            tool::ToolOutput::error(format!("用户拒绝了此操作: {user_input}"))
        };

        self.history
            .push(Message::tool_result(&pending.id, &output.content));

        let mut events = vec![AgentEvent::ToolResponse {
            id: pending.id.clone(),
            content: output.content.clone(),
            is_error: output.is_error,
        }];

        // Execute remaining tool calls that were queued behind the approved one
        let round = pending.round;
        if let Some(pending_approval) = self
            .execute_tool_batch(&pending.remaining_calls, round, tools, &mut events)
            .await
        {
            self.pending_approval = Some(pending_approval);
            return Ok(events);
        }

        // Continue the LLM loop from the current round
        let tool_specs = tools.to_tool_specs();
        self.run_loop(config, tools, &tool_specs, round, &mut events)
            .await?;
        Ok(events)
    }

    /// Core LLM <-> tool loop. Can be called from `run` (starting round 0)
    /// or `resolve_approval` (continuing from a paused round).
    async fn run_loop(
        &mut self,
        config: &AgentConfig,
        tools: &ToolRegistry,
        tool_specs: &[provider::ToolSpec],
        mut round: usize,
        events: &mut Vec<AgentEvent>,
    ) -> Result<(), AgentError> {
        loop {
            if round >= config.max_tool_rounds {
                warn!(
                    max_rounds = config.max_tool_rounds,
                    "reached maximum tool rounds"
                );
                return Err(AgentError::MaxRoundsExceeded {
                    max: config.max_tool_rounds,
                });
            }
            round += 1;
            info!("starting round {}", round);

            let mut stream = self
                .provider
                .stream_chat(&self.history, &tool_specs)
                .await?;

            let mut assistant_content = String::new();
            let mut tool_requests: Vec<(String, String, serde_json::Value)> = Vec::new();
            let mut stop_reason: Option<StopReason> = None;
            let mut usage = None;

            while let Some(chunk) = stream.next().await? {
                match chunk {
                    StreamChunk::Text(text) => {
                        assistant_content.push_str(&text);
                        events.push(AgentEvent::Text(text));
                    }
                    StreamChunk::ToolUse { id, name, input } => {
                        tool_requests.push((id, name, input));
                    }
                    StreamChunk::ToolUseEnd => {}
                    StreamChunk::Finished {
                        stop_reason: sr,
                        usage: u,
                    } => {
                        stop_reason = Some(sr);
                        usage = Some(u);
                    }
                }
            }

            match stop_reason {
                Some(StopReason::EndTurn) if tool_requests.is_empty() => {
                    if !assistant_content.is_empty() {
                        self.history.push(Message {
                            role: Role::Assistant,
                            content: assistant_content,
                            tool_calls: None,
                            tool_call_id: None,
                        });
                    }
                    if let Some(u) = usage {
                        info!(
                            round = round,
                            input_tokens = u.input_tokens,
                            output_tokens = u.output_tokens,
                            "turn complete"
                        );
                        events.push(AgentEvent::TurnEnd { usage: u });
                    }
                    return Ok(());
                }
                _ => {
                    let tool_calls: Vec<ToolCallRequest> = tool_requests
                        .iter()
                        .map(|(id, name, input)| ToolCallRequest {
                            id: id.clone(),
                            name: name.clone(),
                            input: input.clone(),
                        })
                        .collect();

                    if !tool_calls.is_empty() {
                        self.history.push(Message {
                            role: Role::Assistant,
                            content: assistant_content,
                            tool_calls: Some(tool_calls.clone()),
                            tool_call_id: None,
                        });

                        if let Some(pending_approval) = self
                            .execute_tool_batch(&tool_calls, round, tools, events)
                            .await
                        {
                            self.pending_approval = Some(pending_approval);
                            return Ok(());
                        }
                        continue;
                    }

                    // No tool calls but not EndTurn (e.g. MaxTokens)
                    if !assistant_content.is_empty() {
                        self.history.push(Message {
                            role: Role::Assistant,
                            content: assistant_content,
                            tool_calls: None,
                            tool_call_id: None,
                        });
                    }
                    if let Some(u) = usage {
                        info!(
                            round = round,
                            input_tokens = u.input_tokens,
                            output_tokens = u.output_tokens,
                            "turn complete"
                        );
                        events.push(AgentEvent::TurnEnd { usage: u });
                    }
                    return Ok(());
                }
            }
        }
    }

    /// Execute a batch of tool calls. Returns `Some(PendingApproval)` if any tool
    /// needs user approval, in which case the remaining calls are saved for later.
    async fn execute_tool_batch(
        &mut self,
        tool_calls: &[ToolCallRequest],
        round: usize,
        tools: &ToolRegistry,
        events: &mut Vec<AgentEvent>,
    ) -> Option<PendingApproval> {
        for (i, tc) in tool_calls.iter().enumerate() {
            events.push(AgentEvent::ToolRequest {
                id: tc.id.clone(),
                name: tc.name.clone(),
                input: tc.input.clone(),
            });

            info!(tool = %tc.name, tool_id = %tc.id, "executing tool");

            let output = match tools.execute(&tc.name, tc.input.clone()).await {
                Some(out) => out,
                None => tool::ToolOutput::error(format!("未知工具: {}", tc.name)),
            };

            if output.needs_approval {
                info!(
                    tool = %tc.name,
                    path = ?output.approval_path,
                    "tool needs user approval"
                );
                events.push(AgentEvent::ApprovalRequired {
                    tool_call_id: tc.id.clone(),
                    tool_name: tc.name.clone(),
                    path: output.approval_path.clone().unwrap_or_default(),
                    message: output.content.clone(),
                });
                return Some(PendingApproval {
                    id: tc.id.clone(),
                    name: tc.name.clone(),
                    input: tc.input.clone(),
                    round,
                    remaining_calls: tool_calls[i + 1..].to_vec(),
                });
            }

            self.history
                .push(Message::tool_result(&tc.id, &output.content));
            events.push(AgentEvent::ToolResponse {
                id: tc.id.clone(),
                content: output.content,
                is_error: output.is_error,
            });
        }
        None
    }
}
