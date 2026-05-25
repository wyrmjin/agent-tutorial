//! Agent loop: orchestrates the conversation between user, LLM, and tools.
//!
//! The [`Agent`] drives the core loop:
//! 1. Send user message + history to the LLM
//! 2. If LLM responds with text → yield to caller
//! 3. If LLM requests tool use → execute tool, append result, loop back to 1
//! 4. If LLM ends turn → done

use logger::{debug, info, warn};
use provider::{Message, Provider, Role, StopReason, StreamChunk, ToolCallRequest};
use tool::ToolRegistry;

/// Configuration for an agent run.
#[derive(Debug, Clone)]
pub struct AgentConfig {
    pub system_prompt: String,
    pub max_tool_rounds: usize,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            system_prompt: String::new(),
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
    /// The conversation turn is complete.
    TurnEnd { usage: provider::Usage },
}

/// The main agent that runs the conversation loop.
pub struct Agent<P: Provider> {
    provider: P,
    history: Vec<Message>,
}

impl<P: Provider> Agent<P> {
    pub fn new(provider: P, system_prompt: String) -> Self {
        let mut history = Vec::new();
        if !system_prompt.is_empty() {
            history.push(Message {
                role: Role::System,
                content: system_prompt,
                tool_calls: None,
                tool_call_id: None,
            });
        }
        Self { provider, history }
    }

    /// Access the conversation history.
    pub fn history(&self) -> &[Message] {
        &self.history
    }

    /// Run one turn: given user input, loop through LLM ↔ tool calls
    /// and yield events to the caller.
    pub async fn run(
        &mut self,
        user_input: &str,
        config: &AgentConfig,
        tools: &ToolRegistry,
    ) -> anyhow::Result<Vec<AgentEvent>> {
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

        let mut round = 0;
        loop {
            if round >= config.max_tool_rounds {
                warn!(
                    max_rounds = config.max_tool_rounds,
                    "reached maximum tool rounds"
                );
                break;
            }
            round += 1;
            info!("starting round {}", round);

            let mut stream = self
                .provider
                .stream_chat(&self.history, &tool_specs, &config.system_prompt)
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
                    // 纯文本回复 — 结束
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
                    return Ok(events);
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
                        // 记录 assistant 的 tool_calls
                        self.history.push(Message {
                            role: Role::Assistant,
                            content: assistant_content,
                            tool_calls: Some(tool_calls.clone()),
                            tool_call_id: None,
                        });

                        // 通过 ToolRegistry 实际执行每个工具
                        for tc in &tool_calls {
                            events.push(AgentEvent::ToolRequest {
                                id: tc.id.clone(),
                                name: tc.name.clone(),
                                input: tc.input.clone(),
                            });

                            info!(tool = %tc.name, tool_id = %tc.id, "executing tool");

                            let output = match tools.execute(&tc.name, tc.input.clone()).await {
                                Some(out) => out,
                                None => tool::ToolOutput {
                                    content: format!("未知工具: {}", tc.name),
                                    is_error: true,
                                },
                            };

                            self.history
                                .push(Message::tool_result(&tc.id, &output.content));
                            events.push(AgentEvent::ToolResponse {
                                id: tc.id.clone(),
                                content: output.content,
                                is_error: output.is_error,
                            });
                        }
                        // 继续循环，让 LLM 处理 tool results
                        continue;
                    }

                    // 无 tool calls 但也不是 EndTurn（如 MaxTokens）
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
                    return Ok(events);
                }
            }
        }

        Ok(events)
    }
}
