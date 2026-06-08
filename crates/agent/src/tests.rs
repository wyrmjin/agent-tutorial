use std::sync::Arc;

use futures::StreamExt;

use crate::agent::Agent;
use crate::config::AgentConfig;
use crate::event::AgentEvent;

// ---------------------------------------------------------------------------
// Mock Provider
// ---------------------------------------------------------------------------

/// A mock ai that returns a fixed set of stream chunks.
struct MockProvider {
    response: Vec<ai::StreamChunk>,
}

impl MockProvider {
    fn new(chunks: Vec<ai::StreamChunk>) -> Self {
        Self { response: chunks }
    }
}

#[async_trait::async_trait]
impl ai::Provider for MockProvider {
    async fn stream_chat(
        &self,
        _messages: &[ai::Message],
        _tools: &[ai::ToolSpec],
    ) -> Result<Box<dyn ai::StreamChunkIterator>, ai::ProviderError> {
        Ok(Box::new(MockStream {
            chunks: self.response.clone(),
            index: 0,
        }))
    }

    fn provider_type(&self) -> ai::ProviderType {
        ai::ProviderType::DeepSeek
    }
}

struct MockStream {
    chunks: Vec<ai::StreamChunk>,
    index: usize,
}

#[async_trait::async_trait]
impl ai::StreamChunkIterator for MockStream {
    async fn next(&mut self) -> Result<Option<ai::StreamChunk>, ai::ProviderError> {
        if self.index < self.chunks.len() {
            let chunk = self.chunks[self.index].clone();
            self.index += 1;
            Ok(Some(chunk))
        } else {
            Ok(None)
        }
    }
}

fn make_mock_provider(chunks: Vec<ai::StreamChunk>) -> Arc<dyn ai::Provider> {
    Arc::new(MockProvider::new(chunks))
}

fn make_default_provider() -> Arc<dyn ai::Provider> {
    make_mock_provider(vec![
        ai::StreamChunk::Text("Hello, ".to_string()),
        ai::StreamChunk::Text("world!".to_string()),
        ai::StreamChunk::Finished {
            stop_reason: ai::StopReason::EndTurn,
            usage: ai::Usage::default(),
        },
    ])
}

fn make_tools() -> Arc<tool::ToolRegistry> {
    Arc::new(tool::ToolRegistry::new())
}

// ---------------------------------------------------------------------------
// Multi-round mock ai
// ---------------------------------------------------------------------------

/// A mock ai that returns different responses for each call.
/// This allows testing multi-round tool-call loops.
struct MultiRoundMockProvider {
    responses: Vec<Vec<ai::StreamChunk>>,
    call_count: std::sync::Mutex<usize>,
}

impl MultiRoundMockProvider {
    fn new(responses: Vec<Vec<ai::StreamChunk>>) -> Self {
        Self {
            responses,
            call_count: std::sync::Mutex::new(0),
        }
    }
}

#[async_trait::async_trait]
impl ai::Provider for MultiRoundMockProvider {
    async fn stream_chat(
        &self,
        _messages: &[ai::Message],
        _tools: &[ai::ToolSpec],
    ) -> Result<Box<dyn ai::StreamChunkIterator>, ai::ProviderError> {
        let idx = {
            let mut count = self.call_count.lock().unwrap();
            let i = *count;
            *count += 1;
            i
        };
        let chunks = self
            .responses
            .get(idx)
            .cloned()
            .unwrap_or_else(|| vec![ai::StreamChunk::Finished {
                stop_reason: ai::StopReason::EndTurn,
                usage: ai::Usage::default(),
            }]);
        Ok(Box::new(MockStream {
            chunks,
            index: 0,
        }))
    }

    fn provider_type(&self) -> ai::ProviderType {
        ai::ProviderType::DeepSeek
    }
}

// ---------------------------------------------------------------------------
// Mock tool for testing
// ---------------------------------------------------------------------------

/// A simple echo tool that returns its input as the result.
struct EchoTool;

#[async_trait::async_trait]
impl tool::Tool for EchoTool {
    fn name(&self) -> &str {
        "echo"
    }

    fn description(&self) -> &str {
        "Echoes the input back as output"
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "message": { "type": "string" }
            },
            "required": ["message"]
        })
    }

    async fn execute(&self, input: serde_json::Value) -> tool::ToolOutput {
        let msg = input
            .get("message")
            .and_then(|v| v.as_str())
            .unwrap_or("no message");
        tool::ToolOutput::ok(format!("Echo: {msg}"))
    }
}

fn make_tools_with_echo() -> Arc<tool::ToolRegistry> {
    let mut registry = tool::ToolRegistry::new();
    registry.register(EchoTool);
    Arc::new(registry)
}

// ---------------------------------------------------------------------------
// Helper: collect all events from a handle
// ---------------------------------------------------------------------------

async fn collect_events(handle: &mut crate::handle::AgentHandle) -> Vec<AgentEvent> {
    let mut events = Vec::new();
    while let Some(event) = handle.stream_mut().next().await {
        events.push(event);
    }
    events
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_run_stream_returns_text_events() {
    let provider = make_default_provider();
    let agent = Agent::new(provider, "test system prompt".to_string());
    let tools = make_tools();
    let config = AgentConfig::default();

    let mut handle = agent.run_stream("hello", &config, tools);

    let mut texts = Vec::new();
    let mut got_done = false;
    while let Some(event) = handle.stream_mut().next().await {
        match event {
            AgentEvent::Text(t) => texts.push(t),
            AgentEvent::Done { .. } => got_done = true,
            _ => {}
        }
    }
    assert_eq!(texts, vec!["Hello, ", "world!"]);
    assert!(got_done);
}

#[tokio::test]
async fn test_run_stream_done_contains_messages() {
    let provider = make_default_provider();
    let agent = Agent::new(provider, "test".to_string());
    let tools = make_tools();
    let config = AgentConfig::default();

    let mut handle = agent.run_stream("hello", &config, tools);

    let mut done_messages = None;
    while let Some(event) = handle.stream_mut().next().await {
        if let AgentEvent::Done { messages } = event {
            done_messages = Some(messages);
        }
    }

    let messages = done_messages.expect("should have received Done event");
    // Should contain: system prompt + user message + assistant response
    assert!(
        messages.len() >= 2,
        "expected at least 2 messages, got {}",
        messages.len()
    );
}

#[tokio::test]
async fn test_run_stream_abort() {
    let provider = make_mock_provider(vec![
        ai::StreamChunk::Text("starting...".to_string()),
        ai::StreamChunk::Finished {
            stop_reason: ai::StopReason::EndTurn,
            usage: ai::Usage::default(),
        },
    ]);
    let agent = Agent::new(provider, "test".to_string());
    let tools = make_tools();
    let config = AgentConfig::default();

    let mut handle = agent.run_stream("hello", &config, tools);

    // Read first event then abort
    let first = handle.stream_mut().next().await;
    assert!(first.is_some());

    handle.abort();

    // Stream should close cleanly — either we get an abort error event
    // or the stream already completed before the abort signal was processed.
    // Either outcome is acceptable; the key invariant is that the stream
    // terminates without panicking.
    let mut got_done = false;
    while let Some(event) = handle.stream_mut().next().await {
        if let AgentEvent::Done { .. } = &event {
            got_done = true;
        }
    }
    // The stream must always finish with a Done event.
    assert!(got_done, "stream should always end with a Done event");
}

#[tokio::test]
async fn test_history_auto_writeback_after_stream() {
    // Verify that history is automatically written back after run_stream completes,
    // without requiring a manual set_history call.
    let provider = make_default_provider();
    let agent = Agent::new(provider, "test system prompt".to_string());
    let tools = make_tools();
    let config = AgentConfig::default();

    // History before run_stream should contain only system prompt
    let before = agent.history();
    assert_eq!(before.len(), 1);
    assert_eq!(before[0].role(), ai::Role::System);

    // Run the stream to completion
    let mut handle = agent.run_stream("hello", &config, tools);
    while let Some(event) = handle.stream_mut().next().await {
        match event {
            AgentEvent::Done { .. } => {} // no manual set_history needed!
            _ => {}
        }
    }

    // History should now contain: system prompt + user message + assistant response
    let after = agent.history();
    assert!(
        after.len() >= 3,
        "expected at least 3 messages after auto-writeback, got {}",
        after.len()
    );
    assert_eq!(after[0].role(), ai::Role::System);
    assert_eq!(after[1].role(), ai::Role::User);
    assert_eq!(after[2].role(), ai::Role::Assistant);
}

#[tokio::test]
async fn test_history_accumulates_across_turns() {
    // Verify that history accumulates correctly across multiple run_stream calls.
    let provider = make_default_provider();
    let agent = Agent::new(provider, "test".to_string());
    let tools = make_tools();
    let config = AgentConfig::default();

    // First turn
    let mut handle = agent.run_stream("first message", &config, Arc::clone(&tools));
    while let Some(_) = handle.stream_mut().next().await {}

    let after_first = agent.history();
    let first_len = after_first.len();
    assert!(first_len >= 2, "expected >= 2 messages after first turn");

    // Second turn
    let mut handle = agent.run_stream("second message", &config, tools);
    while let Some(_) = handle.stream_mut().next().await {}

    let after_second = agent.history();
    assert!(
        after_second.len() > first_len,
        "expected more messages after second turn ({} > {})",
        after_second.len(),
        first_len
    );
    // Should contain both user messages
    let user_messages: Vec<_> = after_second
        .iter()
        .filter(|m| m.role() == ai::Role::User)
        .collect();
    assert_eq!(user_messages.len(), 2);
}

// ---------------------------------------------------------------------------
// NEW: ToolUse tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_tool_use_executes_tool_and_loops() {
    // Round 1: LLM requests tool use → tool executes → round 2: LLM ends turn.
    let provider = Arc::new(MultiRoundMockProvider::new(vec![
        // First call: request tool use
        vec![
            ai::StreamChunk::Text("Let me check...".to_string()),
            ai::StreamChunk::ToolUse {
                id: "call_1".to_string(),
                name: "echo".to_string(),
                input: serde_json::json!({"message": "hello"}),
            },
            ai::StreamChunk::Finished {
                stop_reason: ai::StopReason::ToolUse,
                usage: ai::Usage::default(),
            },
        ],
        // Second call: end turn with tool result
        vec![
            ai::StreamChunk::Text("Done!".to_string()),
            ai::StreamChunk::Finished {
                stop_reason: ai::StopReason::EndTurn,
                usage: ai::Usage::default(),
            },
        ],
    ]));

    let agent = Agent::new(provider, "test".to_string());
    let tools = make_tools_with_echo();
    let config = AgentConfig::default();

    let mut handle = agent.run_stream("use the echo tool", &config, tools);
    let events = collect_events(&mut handle).await;

    // Should see: Text, ToolRequest, ToolResponse, Text, TurnEnd, Done
    let tool_requests: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            AgentEvent::ToolRequest { id, name, .. } => Some((id.clone(), name.clone())),
            _ => None,
        })
        .collect();
    assert_eq!(tool_requests.len(), 1);
    assert_eq!(tool_requests[0].0, "call_1");
    assert_eq!(tool_requests[0].1, "echo");

    let tool_responses: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            AgentEvent::ToolResponse { id, content, .. } => Some((id.clone(), content.clone())),
            _ => None,
        })
        .collect();
    assert_eq!(tool_responses.len(), 1);
    assert_eq!(tool_responses[0].0, "call_1");
    assert!(tool_responses[0].1.contains("Echo: hello"));

    // Should have a Done event
    assert!(
        events.iter().any(|e| matches!(e, AgentEvent::Done { .. })),
        "should have a Done event"
    );

    // History should contain assistant + tool_result messages
    let history = agent.history();
    let assistant_msgs: Vec<_> = history
        .iter()
        .filter(|m| m.role() == ai::Role::Assistant)
        .collect();
    assert!(
        assistant_msgs.len() >= 2,
        "expected >= 2 assistant messages, got {}",
        assistant_msgs.len()
    );
}

#[tokio::test]
async fn test_multiple_tool_calls_in_single_round() {
    // LLM requests two tool calls in one response.
    let provider = Arc::new(MultiRoundMockProvider::new(vec![
        vec![
            ai::StreamChunk::ToolUse {
                id: "call_1".to_string(),
                name: "echo".to_string(),
                input: serde_json::json!({"message": "first"}),
            },
            ai::StreamChunk::ToolUse {
                id: "call_2".to_string(),
                name: "echo".to_string(),
                input: serde_json::json!({"message": "second"}),
            },
            ai::StreamChunk::Finished {
                stop_reason: ai::StopReason::ToolUse,
                usage: ai::Usage::default(),
            },
        ],
        vec![
            ai::StreamChunk::Text("All done!".to_string()),
            ai::StreamChunk::Finished {
                stop_reason: ai::StopReason::EndTurn,
                usage: ai::Usage::default(),
            },
        ],
    ]));

    let agent = Agent::new(provider, "test".to_string());
    let tools = make_tools_with_echo();
    let config = AgentConfig::default();

    let mut handle = agent.run_stream("use two tools", &config, tools);
    let events = collect_events(&mut handle).await;

    let tool_requests: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            AgentEvent::ToolRequest { id, .. } => Some(id.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(tool_requests.len(), 2);

    let tool_responses: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            AgentEvent::ToolResponse { id, .. } => Some(id.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(tool_responses.len(), 2);
}

#[tokio::test]
async fn test_multi_round_tool_loop() {
    // LLM calls tool 3 times before ending — tests the loop mechanism.
    let provider = Arc::new(MultiRoundMockProvider::new(vec![
        // Round 1: tool call
        vec![
            ai::StreamChunk::ToolUse {
                id: "call_1".to_string(),
                name: "echo".to_string(),
                input: serde_json::json!({"message": "round1"}),
            },
            ai::StreamChunk::Finished {
                stop_reason: ai::StopReason::ToolUse,
                usage: ai::Usage::default(),
            },
        ],
        // Round 2: another tool call
        vec![
            ai::StreamChunk::ToolUse {
                id: "call_2".to_string(),
                name: "echo".to_string(),
                input: serde_json::json!({"message": "round2"}),
            },
            ai::StreamChunk::Finished {
                stop_reason: ai::StopReason::ToolUse,
                usage: ai::Usage::default(),
            },
        ],
        // Round 3: end turn
        vec![
            ai::StreamChunk::Text("Finished after 2 tool calls.".to_string()),
            ai::StreamChunk::Finished {
                stop_reason: ai::StopReason::EndTurn,
                usage: ai::Usage::default(),
            },
        ],
    ]));

    let agent = Agent::new(provider, "test".to_string());
    let tools = make_tools_with_echo();
    let config = AgentConfig::default();

    let mut handle = agent.run_stream("multi-round", &config, tools);
    let events = collect_events(&mut handle).await;

    let tool_request_count = events
        .iter()
        .filter(|e| matches!(e, AgentEvent::ToolRequest { .. }))
        .count();
    assert_eq!(tool_request_count, 2, "expected 2 tool requests");

    let turn_end_count = events
        .iter()
        .filter(|e| matches!(e, AgentEvent::TurnEnd { .. }))
        .count();
    // TurnEnd is emitted only for the final EndTurn, not for tool rounds
    assert_eq!(turn_end_count, 1, "expected 1 TurnEnd event");

    // Verify history has all the tool calls and results
    let history = agent.history();
    // system + user + assistant(tool_use) + tool_result + assistant(tool_use) + tool_result + assistant(text)
    assert!(
        history.len() >= 7,
        "expected >= 7 messages in history, got {}",
        history.len()
    );
}

#[tokio::test]
async fn test_unknown_tool_returns_error() {
    // LLM requests a tool that doesn't exist.
    let provider = Arc::new(MultiRoundMockProvider::new(vec![
        vec![
            ai::StreamChunk::ToolUse {
                id: "call_1".to_string(),
                name: "nonexistent_tool".to_string(),
                input: serde_json::json!({}),
            },
            ai::StreamChunk::Finished {
                stop_reason: ai::StopReason::ToolUse,
                usage: ai::Usage::default(),
            },
        ],
        vec![
            ai::StreamChunk::Text("OK understood.".to_string()),
            ai::StreamChunk::Finished {
                stop_reason: ai::StopReason::EndTurn,
                usage: ai::Usage::default(),
            },
        ],
    ]));

    let agent = Agent::new(provider, "test".to_string());
    let tools = make_tools(); // empty registry — no tools registered
    let config = AgentConfig::default();

    let mut handle = agent.run_stream("use unknown tool", &config, tools);
    let events = collect_events(&mut handle).await;

    // ToolResponse should contain an error about unknown tool
    let tool_responses: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            AgentEvent::ToolResponse {
                content, is_error, ..
            } if *is_error => Some(content.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(tool_responses.len(), 1);
    assert!(tool_responses[0].contains("未知工具"));
}

#[tokio::test]
async fn test_stream_error_reports_error_event() {
    // Provider returns a stream that yields an error.
    struct ErrorStream;
    #[async_trait::async_trait]
    impl ai::StreamChunkIterator for ErrorStream {
        async fn next(
            &mut self,
        ) -> Result<Option<ai::StreamChunk>, ai::ProviderError> {
            Err(ai::ProviderError::Api { status: 500, message: "connection lost".to_string() })
        }
    }
    struct ErrorProvider;
    #[async_trait::async_trait]
    impl ai::Provider for ErrorProvider {
        async fn stream_chat(
            &self,
            _messages: &[ai::Message],
            _tools: &[ai::ToolSpec],
        ) -> Result<Box<dyn ai::StreamChunkIterator>, ai::ProviderError> {
            Ok(Box::new(ErrorStream))
        }
        fn provider_type(&self) -> ai::ProviderType {
            ai::ProviderType::DeepSeek
        }
    }

    let agent = Agent::new(Arc::new(ErrorProvider), "test".to_string());
    let tools = make_tools();
    let config = AgentConfig::default();

    let mut handle = agent.run_stream("hello", &config, tools);
    let events = collect_events(&mut handle).await;

    let error_events: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            AgentEvent::Error { message, .. } => Some(message.clone()),
            _ => None,
        })
        .collect();
    assert!(
        error_events.iter().any(|m| m.contains("Provider stream error")),
        "expected a stream error event, got errors: {:?}",
        error_events
    );
    assert!(
        events.iter().any(|e| matches!(e, AgentEvent::Done { .. })),
        "should still have a Done event after error"
    );
}
