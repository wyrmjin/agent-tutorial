//! Handle to a running agent plus steering control messages.

use crate::event::{AgentEvent, AgentResult};
use crate::event_stream::EventStream;

/// Decision from the consumer about a pending approval.
#[derive(Debug, Clone)]
pub enum ApprovalDecision {
    /// User approved the tool call.
    Approved,
    /// User denied the tool call.
    Denied { reason: String },
}

/// Control messages that can be sent to a running agent.
pub enum SteeringMessage {
    /// Inject a message into the conversation history.
    InjectMessage(ai::Message),
    /// Response to an approval request from the consumer.
    ApprovalResponse {
        tool_call_id: String,
        decision: ApprovalDecision,
    },
}

/// Handle to a running agent. Provides the event stream plus control methods.
pub struct AgentHandle {
    stream: EventStream<AgentEvent, AgentResult>,
    abort_tx: tokio::sync::mpsc::UnboundedSender<()>,
    steer_tx: tokio::sync::mpsc::UnboundedSender<SteeringMessage>,
}

impl AgentHandle {
    pub(crate) fn new(
        stream: EventStream<AgentEvent, AgentResult>,
        abort_tx: tokio::sync::mpsc::UnboundedSender<()>,
        steer_tx: tokio::sync::mpsc::UnboundedSender<SteeringMessage>,
    ) -> Self {
        Self {
            stream,
            abort_tx,
            steer_tx,
        }
    }

    /// Get a mutable reference to the event stream for async iteration.
    pub fn stream_mut(&mut self) -> &mut EventStream<AgentEvent, AgentResult> {
        &mut self.stream
    }

    /// Request the agent to abort.
    pub fn abort(&self) {
        let _ = self.abort_tx.send(());
    }

    /// Inject a message into the agent's conversation.
    pub fn inject(&self, msg: ai::Message) {
        let _ = self.steer_tx.send(SteeringMessage::InjectMessage(msg));
    }

    /// Approve a pending tool call by its ID.
    pub fn approve(&self, tool_call_id: String) {
        let _ = self.steer_tx.send(SteeringMessage::ApprovalResponse {
            tool_call_id,
            decision: ApprovalDecision::Approved,
        });
    }

    /// Deny a pending tool call by its ID.
    pub fn deny(&self, tool_call_id: String, reason: String) {
        let _ = self.steer_tx.send(SteeringMessage::ApprovalResponse {
            tool_call_id,
            decision: ApprovalDecision::Denied { reason },
        });
    }

    /// Check if the stream is done.
    pub fn is_done(&self) -> bool {
        self.stream.is_done()
    }
}
