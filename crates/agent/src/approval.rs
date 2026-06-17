//! Approval handling for tool calls that require user confirmation.

use crate::handle::SteeringMessage;
use tracing::warn;

/// Result of waiting for an approval decision from the consumer.
pub(crate) enum ApprovalAction {
    /// User approved the tool call.
    Approved,
    /// User denied the tool call.
    Denied { reason: String },
    /// Agent was aborted while waiting for approval.
    Aborted,
}

/// Wait for an approval decision from the consumer via the steering channel.
/// Also checks the abort channel so Ctrl+C works during approval wait.
///
/// Any `InjectMessage` variants received while waiting are re-sent back to
/// `steer_tx` before returning so they are not lost.
pub(crate) async fn wait_for_approval(
    expected_id: &str,
    steer_rx: &mut tokio::sync::mpsc::UnboundedReceiver<SteeringMessage>,
    steer_tx: &tokio::sync::mpsc::UnboundedSender<SteeringMessage>,
    abort_rx: &mut tokio::sync::mpsc::UnboundedReceiver<()>,
) -> ApprovalAction {
    let mut injected: Vec<ai::Message> = Vec::new();

    let result = loop {
        tokio::select! {
            msg = steer_rx.recv() => {
                match msg {
                    Some(SteeringMessage::ApprovalResponse { tool_call_id, decision }) => {
                        if tool_call_id == expected_id {
                            break match decision {
                                crate::handle::ApprovalDecision::Approved => ApprovalAction::Approved,
                                crate::handle::ApprovalDecision::Denied { reason } => {
                                    ApprovalAction::Denied { reason }
                                }
                            };
                        }
                        // Mismatched ID — log and keep waiting
                        warn!(
                            expected = %expected_id,
                            received = %tool_call_id,
                            "approval response ID mismatch, ignoring"
                        );
                    }
                    Some(SteeringMessage::InjectMessage(msg)) => {
                        // recv() already removed the message from the channel.
                        // Buffer it and re-inject before returning so it isn't lost.
                        injected.push(msg);
                    }
                    None => {
                        break ApprovalAction::Aborted;
                    }
                }
            }
            _ = abort_rx.recv() => {
                break ApprovalAction::Aborted;
            }
        }
    };

    // Re-inject buffered messages back into the steering channel so they
    // can be drained at the top of the next round.
    for msg in injected {
        let _ = steer_tx.send(SteeringMessage::InjectMessage(msg));
    }

    result
}
