//! Model: the composition hub — combines provider + protocol + transport + params.

use std::sync::Arc;

use crate::error::AiError;
use crate::message::{Message, ToolSpec};
use crate::protocol::{Protocol, SamplingParams};
use crate::provider::Provider;
use crate::stream::StreamChunkIterator;
use crate::transport::Transport;

/// Static model capabilities.
#[derive(Debug, Clone, Default)]
pub struct Capabilities {
    pub reasoning: bool,
}

/// The narrow interface the agent depends on (dependency inversion).
#[async_trait::async_trait]
pub trait LanguageModel: Send + Sync {
    async fn stream_chat(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
    ) -> Result<Box<dyn StreamChunkIterator>, AiError>;

    fn model_id(&self) -> &str;
}

/// A concrete model: combines a provider, a protocol and a shared transport.
pub struct Model {
    pub id: String,
    pub display_name: String,
    provider: Arc<dyn Provider>,
    protocol: Arc<dyn Protocol>,
    transport: Arc<Transport>,
    params: SamplingParams,
    pub capabilities: Capabilities,
}

impl Model {
    pub fn new(
        id: impl Into<String>,
        provider: Arc<dyn Provider>,
        protocol: Arc<dyn Protocol>,
        transport: Arc<Transport>,
        params: SamplingParams,
        capabilities: Capabilities,
    ) -> Self {
        let id = id.into();
        Self {
            display_name: id.clone(),
            id,
            provider,
            protocol,
            transport,
            params,
            capabilities,
        }
    }

    /// Override the human-readable display name (defaults to `id`).
    pub fn with_display_name(mut self, name: impl Into<String>) -> Self {
        self.display_name = name.into();
        self
    }
}

#[async_trait::async_trait]
impl LanguageModel for Model {
    async fn stream_chat(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
    ) -> Result<Box<dyn StreamChunkIterator>, AiError> {
        let endpoint = self.provider.endpoint();
        let normalizer = self.provider.usage_normalizer();
        self.transport
            .stream(
                &endpoint,
                self.protocol.as_ref(),
                normalizer,
                &self.id,
                &self.params,
                messages,
                tools,
            )
            .await
    }

    fn model_id(&self) -> &str {
        &self.id
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::ProtocolKind;
    use crate::protocol::openai_completions::OpenAiCompletionsProtocol;
    use crate::provider::{AuthStyle, GenericProvider};

    #[test]
    fn model_id_and_display_name_default_to_id() {
        let provider: Arc<dyn crate::provider::Provider> = Arc::new(GenericProvider::new(
            "deepseek",
            "https://api.deepseek.com",
            "k",
            AuthStyle::Bearer,
            vec![ProtocolKind::OpenAiCompletions],
        ));
        let model = Model::new(
            "deepseek-v4-flash",
            provider,
            Arc::new(OpenAiCompletionsProtocol::new()),
            Arc::new(Transport::new()),
            SamplingParams::default(),
            Capabilities::default(),
        );
        assert_eq!(model.model_id(), "deepseek-v4-flash");
        assert_eq!(model.display_name, "deepseek-v4-flash");
    }
}
