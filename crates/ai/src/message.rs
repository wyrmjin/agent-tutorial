//! Conversation data model: messages, roles, tool calls and tool specs.

/// A single message in a conversation.
#[derive(Debug, Clone)]
pub enum Message {
    /// A system prompt message.
    System { content: String },
    /// A user message.
    User { content: String },
    /// An assistant (LLM) response message.
    Assistant {
        content: String,
        /// Tool calls requested by the assistant.
        tool_calls: Option<Vec<ToolCallRequest>>,
    },
    /// A tool execution result message.
    Tool {
        content: String,
        /// ID of the tool call this result corresponds to.
        tool_call_id: String,
    },
}

impl Message {
    /// Returns the role of this message.
    pub fn role(&self) -> Role {
        match self {
            Message::System { .. } => Role::System,
            Message::User { .. } => Role::User,
            Message::Assistant { .. } => Role::Assistant,
            Message::Tool { .. } => Role::Tool,
        }
    }

    /// Returns the content of this message, regardless of variant.
    pub fn content(&self) -> &str {
        match self {
            Message::System { content }
            | Message::User { content }
            | Message::Assistant { content, .. }
            | Message::Tool { content, .. } => content,
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Message::User { content: content.into() }
    }

    pub fn system(content: impl Into<String>) -> Self {
        Message::System { content: content.into() }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Message::Assistant { content: content.into(), tool_calls: None }
    }

    pub fn tool_result(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Message::Tool { content: content.into(), tool_call_id: tool_call_id.into() }
    }
}

/// A tool call request from the assistant.
#[derive(Debug, Clone)]
pub struct ToolCallRequest {
    pub id: String,
    pub name: String,
    pub input: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// Tool definition passed to the LLM.
#[derive(Debug, Clone)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}
