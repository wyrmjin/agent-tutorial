//! Background agent loop — runs inside a tokio::spawn task.

use ai::{LanguageModel, Message, StopReason, StreamChunk};
use std::sync::Arc;
use tool::ToolRegistry;
use tracing::info;

use crate::approval::{ApprovalAction, wait_for_approval};
use crate::event::{AgentEvent, AgentResult};
use crate::event_stream::EventStreamSender;
use crate::handle::SteeringMessage;

// ---------------------------------------------------------------------------
// Private types
// ---------------------------------------------------------------------------

/// Outcome of consuming all chunks from one LLM streaming call.
enum StreamOutcome {
    /// The LLM ended its turn with text only (no tool calls).
    EndTurn { content: String, usage: ai::Usage },
    /// The LLM requested one or more tool calls.
    ToolUse {
        content: String,
        tool_calls: Vec<ai::ToolCallRequest>,
        usage: ai::Usage,
    },
    /// The LLM hit the max token limit before finishing.
    MaxTokens { content: String, usage: ai::Usage },
    /// The stream ended with an error or without a proper stop reason.
    StreamError {
        content: String,
        error_message: String,
        usage: ai::Usage,
    },
}

/// Result of executing a single tool call.
struct ToolExecResult {
    content: String,
    is_error: bool,
}

/// Sentinel indicating the agent was aborted during execution.
struct AbortSignal;

impl StreamOutcome {
    /// Extract a reference to the usage stats (every variant carries them).
    fn usage(&self) -> &ai::Usage {
        match self {
            StreamOutcome::EndTurn { usage, .. }
            | StreamOutcome::MaxTokens { usage, .. }
            | StreamOutcome::ToolUse { usage, .. }
            | StreamOutcome::StreamError { usage, .. } => usage,
        }
    }
}

// ---------------------------------------------------------------------------
// Main loop
// ---------------------------------------------------------------------------

/// Run the agent conversation loop in the background.
///
/// This function is spawned into a tokio task and drives the round-based
/// conversation: call LLM → process response → execute tools → repeat.
pub(crate) async fn run_loop_bg(
    model: Arc<dyn LanguageModel>,
    history: Vec<Message>,
    tool_specs: Vec<ai::ToolSpec>,
    tools: Arc<ToolRegistry>,
    max_rounds: usize,
    sender: EventStreamSender<AgentEvent, AgentResult>,
    mut abort_rx: tokio::sync::mpsc::UnboundedReceiver<()>,
    mut steer_rx: tokio::sync::mpsc::UnboundedReceiver<SteeringMessage>,
    steer_tx: tokio::sync::mpsc::UnboundedSender<SteeringMessage>,
    history_slot: Arc<std::sync::Mutex<Vec<Message>>>,
) {
    let mut history = history;
    let mut total_turns = 0;
    let mut total_usage = ai::Usage::default();
    let mut loop_exhausted = false;

    for round in 0..max_rounds {
        // 1. Check abort signal
        if abort_rx.try_recv().is_ok() {
            sender.push(AgentEvent::Error {
                message: "Agent aborted by user".to_string(),
                recoverable: false,
            });
            break;
        }

        // 2. Drain injected messages
        drain_injected_messages(&mut steer_rx, &mut history);

        info!("starting round {}", round + 1);

        // 3. Call LLM (streaming)
        let mut stream = match model.stream_chat(&history, &tool_specs).await {
            Ok(s) => s,
            Err(e) => {
                sender.push(AgentEvent::Error {
                    message: format!("Provider error: {e}"),
                    recoverable: false,
                });
                break;
            }
        };

        // 4. Consume stream
        let outcome = consume_stream(&mut *stream, &sender, &mut abort_rx).await;

        // 5. Accumulate usage (every variant carries it)
        let usage = outcome.usage();
        total_usage.input_tokens += usage.input_tokens;
        total_usage.output_tokens += usage.output_tokens;
        // 单轮缓存命中率(DeepSeek prompt_cache_*), 用于诊断哪一轮 miss。
        if let Some((hit, miss)) = usage.cache_tokens() {
            // cache_tokens 为 Some 时 cache_hit_percent 必为 Some。
            let pct = usage.cache_hit_percent().unwrap_or(0.0);
            info!(
                round = round + 1,
                cache_hit_tokens = hit,
                cache_miss_tokens = miss,
                cache_hit_pct = %format_args!("{:.1}", pct),
                "cache stats"
            );
        }

        // 6. Dispatch on outcome
        match outcome {
            StreamOutcome::EndTurn { content, .. } | StreamOutcome::MaxTokens { content, .. } => {
                push_assistant_text(&mut history, &content);
                emit_turn_end(&sender, round, &total_usage);
                total_turns += 1;
                break;
            }
            StreamOutcome::ToolUse {
                content,
                tool_calls,
                ..
            } => {
                let aborted = process_tool_calls(
                    &tool_calls,
                    &content,
                    &mut history,
                    &tools,
                    &sender,
                    &mut steer_rx,
                    &steer_tx,
                    &mut abort_rx,
                )
                .await;
                if aborted {
                    break;
                }
                total_turns += 1;
                // If this was the last round and the loop will naturally end,
                // mark as exhausted (tool loop never reached EndTurn).
                if round == max_rounds - 1 {
                    loop_exhausted = true;
                }
                // continue loop for next round
            }
            StreamOutcome::StreamError {
                content,
                error_message,
                ..
            } => {
                // Preserve partial content in history for debugging
                push_assistant_text(&mut history, &content);
                sender.push(AgentEvent::Error {
                    message: error_message,
                    recoverable: false,
                });
                break;
            }
        }
    }

    // 6. Finalization
    finalize(
        &sender,
        history,
        total_turns,
        max_rounds,
        loop_exhausted,
        total_usage,
        &history_slot,
    );
}

// ---------------------------------------------------------------------------
// Stream consumption
// ---------------------------------------------------------------------------

/// Read all chunks from the stream, accumulating text and tool requests.
///
/// Emits `AgentEvent::Text` for each text chunk and returns the accumulated
/// [`StreamOutcome`] based on the stop reason.
///
/// If the stream returns an error or ends without a `Finished` chunk,
/// returns [`StreamOutcome::StreamError`].
///
/// Also checks the abort channel during stream consumption so that abort
/// signals are responded to promptly, not just at round boundaries.
async fn consume_stream(
    stream: &mut dyn ai::StreamChunkIterator,
    sender: &EventStreamSender<AgentEvent, AgentResult>,
    abort_rx: &mut tokio::sync::mpsc::UnboundedReceiver<()>,
) -> StreamOutcome {
    let mut content = String::new();
    let mut tool_requests: Vec<(String, String, serde_json::Value)> = Vec::new();
    let mut stop_reason: Option<StopReason> = None;
    let mut usage = ai::Usage::default();
    let mut stream_error: Option<String> = None;

    loop {
        tokio::select! {
            chunk_result = stream.next() => {
                match chunk_result {
                    Ok(Some(chunk)) => match chunk {
                        StreamChunk::Text(text) => {
                            content.push_str(&text);
                            sender.push(AgentEvent::Text(text));
                        }
                        StreamChunk::ToolUse { id, name, input } => {
                            tool_requests.push((id, name, input));
                        }
                        StreamChunk::Finished {
                            stop_reason: sr,
                            usage: u,
                        } => {
                            stop_reason = Some(sr);
                            usage = u;
                        }
                    },
                    Ok(None) => break,
                    Err(e) => {
                        stream_error = Some(format!("Provider stream error: {e}"));
                        break;
                    }
                }
            }
            _ = abort_rx.recv() => {
                // Abort signal received during stream consumption.
                // Return as a stream error so the caller can break out.
                return StreamOutcome::StreamError {
                    content,
                    error_message: "Agent aborted by user".to_string(),
                    usage,
                };
            }
        }
    }

    // If the stream errored, report it
    if let Some(error_message) = stream_error {
        return StreamOutcome::StreamError {
            content,
            error_message,
            usage,
        };
    }

    // If stop_reason is missing (stream ended without Finished), treat as error
    if stop_reason.is_none() {
        return StreamOutcome::StreamError {
            content,
            error_message: "Stream ended without a stop reason".to_string(),
            usage,
        };
    }

    match stop_reason {
        Some(StopReason::EndTurn) if tool_requests.is_empty() => {
            StreamOutcome::EndTurn { content, usage }
        }
        Some(StopReason::MaxTokens) if tool_requests.is_empty() => {
            StreamOutcome::MaxTokens { content, usage }
        }
        _ => {
            let tool_calls: Vec<ai::ToolCallRequest> = tool_requests
                .into_iter()
                .map(|(id, name, input)| ai::ToolCallRequest { id, name, input })
                .collect();
            StreamOutcome::ToolUse {
                content,
                tool_calls,
                usage,
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tool execution
// ---------------------------------------------------------------------------

/// Execute all tool calls in sequence, handling approvals.
///
/// Pushes the assistant message and each tool result into `history`.
/// Returns `true` if the agent was aborted during execution.
async fn process_tool_calls(
    tool_calls: &[ai::ToolCallRequest],
    assistant_content: &str,
    history: &mut Vec<Message>,
    tools: &Arc<ToolRegistry>,
    sender: &EventStreamSender<AgentEvent, AgentResult>,
    steer_rx: &mut tokio::sync::mpsc::UnboundedReceiver<SteeringMessage>,
    steer_tx: &tokio::sync::mpsc::UnboundedSender<SteeringMessage>,
    abort_rx: &mut tokio::sync::mpsc::UnboundedReceiver<()>,
) -> bool {
    // Push assistant message with tool calls into history
    history.push(ai::Message::Assistant {
        content: assistant_content.to_string(),
        tool_calls: Some(tool_calls.to_vec()),
    });

    for (idx, tc) in tool_calls.iter().enumerate() {
        match execute_tool_with_approval(tc, tools, sender, steer_rx, steer_tx, abort_rx).await {
            Ok(result) => {
                history.push(ai::Message::tool_result(&tc.id, &result.content));
                sender.push(AgentEvent::ToolResponse {
                    id: tc.id.clone(),
                    content: result.content,
                    is_error: result.is_error,
                });
            }
            Err(AbortSignal) => {
                sender.push(AgentEvent::Error {
                    message: "Agent aborted while waiting for approval".to_string(),
                    recoverable: false,
                });
                // Backfill error tool_results for the current and all remaining
                // tool calls so that tool_calls and tool_results stay paired.
                for remaining_tc in &tool_calls[idx..] {
                    history.push(ai::Message::tool_result(
                        &remaining_tc.id,
                        "Aborted: agent was stopped before this tool could complete",
                    ));
                }
                return true;
            }
        }
    }

    false
}

/// Execute a single tool call, handling the approval flow if needed.
///
/// **Design note**: We call `tools.execute()` once first. If the tool returns
/// `needs_approval=true`, the tool implementation MUST NOT have performed any
/// side effects — it should only return an approval request message. After the
/// user approves, we call `tools.approve()` to whitelist the path and then
/// `tools.execute()` again to get the actual result.
///
/// Returns `Ok(ToolExecResult)` after execution (possibly post-approval
/// re-execution), or `Err(AbortSignal)` if the agent was aborted while
/// waiting for approval.
async fn execute_tool_with_approval(
    tool_call: &ai::ToolCallRequest,
    tools: &Arc<ToolRegistry>,
    sender: &EventStreamSender<AgentEvent, AgentResult>,
    steer_rx: &mut tokio::sync::mpsc::UnboundedReceiver<SteeringMessage>,
    steer_tx: &tokio::sync::mpsc::UnboundedSender<SteeringMessage>,
    abort_rx: &mut tokio::sync::mpsc::UnboundedReceiver<()>,
) -> Result<ToolExecResult, AbortSignal> {
    sender.push(AgentEvent::ToolRequest {
        id: tool_call.id.clone(),
        name: tool_call.name.clone(),
        input: tool_call.input.clone(),
    });

    info!(tool = %tool_call.name, tool_id = %tool_call.id, "executing tool");

    let output = execute_tool(tools, &tool_call.name, tool_call.input.clone()).await;

    if !output.needs_approval {
        return Ok(ToolExecResult {
            content: output.content,
            is_error: output.is_error,
        });
    }

    // Tool requests approval — the tool MUST NOT have performed side effects
    // at this point. We emit the approval request and wait for the user.
    sender.push(AgentEvent::ApprovalRequired {
        tool_call_id: tool_call.id.clone(),
        tool_name: tool_call.name.clone(),
        path: output.approval_path.clone().unwrap_or_default(),
        message: output.content.clone(),
    });

    let action = wait_for_approval(&tool_call.id, steer_rx, steer_tx, abort_rx).await;

    match action {
        ApprovalAction::Approved => {
            // Whitelist the path and re-execute to get the actual result.
            tools.approve(&tool_call.name, &tool_call.input);
            let output = execute_tool(tools, &tool_call.name, tool_call.input.clone()).await;
            Ok(ToolExecResult {
                content: output.content,
                is_error: output.is_error,
            })
        }
        ApprovalAction::Denied { reason } => {
            let msg = if reason.is_empty() {
                "用户拒绝了此操作".to_string()
            } else {
                format!("用户拒绝了此操作: {reason}")
            };
            Ok(ToolExecResult {
                content: msg,
                is_error: true,
            })
        }
        ApprovalAction::Aborted => Err(AbortSignal),
    }
}

// ---------------------------------------------------------------------------
// Small synchronous helpers
// ---------------------------------------------------------------------------

/// Execute a tool by name, falling back to an "unknown tool" error if not found.
async fn execute_tool(
    tools: &Arc<ToolRegistry>,
    name: &str,
    input: serde_json::Value,
) -> tool::ToolOutput {
    match tools.execute(name, input).await {
        Some(out) => out,
        None => tool::ToolOutput::error(format!("未知工具: {}", name)),
    }
}

/// Drain `InjectMessage` variants from the steering channel into history.
fn drain_injected_messages(
    steer_rx: &mut tokio::sync::mpsc::UnboundedReceiver<SteeringMessage>,
    history: &mut Vec<Message>,
) {
    while let Ok(SteeringMessage::InjectMessage(msg)) = steer_rx.try_recv() {
        history.push(msg);
    }
}

/// Push assistant text into history if non-empty.
fn push_assistant_text(history: &mut Vec<Message>, content: &str) {
    if !content.is_empty() {
        history.push(ai::Message::assistant(content));
    }
}

/// Emit `TurnEnd` event and log usage for the round.
fn emit_turn_end(
    sender: &EventStreamSender<AgentEvent, AgentResult>,
    round: usize,
    usage: &ai::Usage,
) {
    info!(
        round = round + 1,
        input_tokens = usage.input_tokens,
        output_tokens = usage.output_tokens,
        "turn complete"
    );
    sender.push(AgentEvent::TurnEnd {
        usage: usage.clone(),
    });
}

/// Emit final `Done` event, write history back into the shared slot, and end the stream.
///
/// Note: Uses `std::sync::Mutex` for the history slot. This is acceptable because:
/// - `finalize` is a synchronous function (no `.await` while holding the lock).
/// - The critical section is extremely short (a single Vec assignment).
/// - Tokio's official guidance states `std::sync::Mutex` is fine for short
///   non-async critical sections.
fn finalize(
    sender: &EventStreamSender<AgentEvent, AgentResult>,
    messages: Vec<Message>,
    total_turns: usize,
    max_rounds: usize,
    exhausted: bool,
    total_usage: ai::Usage,
    history_slot: &Arc<std::sync::Mutex<Vec<Message>>>,
) {
    if exhausted {
        sender.push(AgentEvent::Error {
            message: format!("Maximum tool rounds exceeded ({max_rounds})"),
            recoverable: false,
        });
    }
    let history_for_result = messages.clone();
    sender.push(AgentEvent::Done {
        messages: messages.clone(),
    });
    // Write final history back into the shared slot for the next run_stream call.
    *history_slot.lock().unwrap() = messages;
    sender.end(AgentResult {
        messages: history_for_result,
        total_turns,
        usage: total_usage,
    });
}
