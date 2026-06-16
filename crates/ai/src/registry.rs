//! Runtime provider registry.
//!
//! Allows registering multiple LLM providers by name and looking them up
//! dynamically, instead of hard-coding a single provider type.

use std::collections::HashMap;
use std::sync::Arc;

use crate::{LanguageModel, Provider, ProviderType};

/// A registry of LLM providers, keyed by name.
pub struct ProviderRegistry {
    providers: HashMap<String, Arc<dyn Provider>>,
    default_name: Option<String>,
}

impl ProviderRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            providers: HashMap::new(),
            default_name: None,
        }
    }

    /// Register a provider under the given name.
    ///
    /// If this is the first provider registered, it becomes the default.
    pub fn register(&mut self, name: impl Into<String>, provider: Arc<dyn Provider>) {
        let name = name.into();
        if self.default_name.is_none() {
            self.default_name = Some(name.clone());
        }
        self.providers.insert(name, provider);
    }

    /// Look up a provider by name.
    pub fn get(&self, name: &str) -> Option<Arc<dyn Provider>> {
        self.providers.get(name).cloned()
    }

    /// Get the default provider (the first one registered).
    pub fn default_provider(&self) -> Option<Arc<dyn Provider>> {
        self.default_name
            .as_ref()
            .and_then(|name| self.providers.get(name).cloned())
    }

    /// List all registered provider names.
    pub fn provider_names(&self) -> Vec<&str> {
        self.providers.keys().map(|s| s.as_str()).collect()
    }

    /// Get the provider type for a named provider.
    pub fn provider_type(&self, name: &str) -> Option<ProviderType> {
        self.providers.get(name).map(|p| p.provider_type())
    }
}

impl Default for ProviderRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// A registry of language models, keyed by name.
pub struct ModelRegistry {
    models: HashMap<String, Arc<dyn LanguageModel>>,
    default_name: Option<String>,
}

impl ModelRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
            default_name: None,
        }
    }

    /// Register a model under the given name.
    /// The first model registered becomes the default.
    pub fn register(&mut self, name: impl Into<String>, model: Arc<dyn LanguageModel>) {
        let name = name.into();
        if self.default_name.is_none() {
            self.default_name = Some(name.clone());
        }
        self.models.insert(name, model);
    }

    /// Look up a model by registry name.
    pub fn get(&self, name: &str) -> Option<Arc<dyn LanguageModel>> {
        self.models.get(name).cloned()
    }

    /// Get the default model (the first one registered).
    pub fn default_model(&self) -> Option<Arc<dyn LanguageModel>> {
        self.default_name
            .as_ref()
            .and_then(|name| self.models.get(name).cloned())
    }

    /// List all registered model names.
    pub fn model_names(&self) -> Vec<&str> {
        self.models.keys().map(|s| s.as_str()).collect()
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ProviderType;

    /// A minimal mock provider for testing the registry.
    struct MockProvider {
        ptype: ProviderType,
    }

    #[async_trait::async_trait]
    impl Provider for MockProvider {
        async fn stream_chat(
            &self,
            _messages: &[crate::Message],
            _tools: &[crate::ToolSpec],
        ) -> Result<Box<dyn crate::StreamChunkIterator>, crate::ProviderError> {
            unimplemented!("mock")
        }

        fn provider_type(&self) -> ProviderType {
            self.ptype
        }
    }

    #[test]
    fn register_and_lookup() {
        let mut registry = ProviderRegistry::new();
        let provider = Arc::new(MockProvider {
            ptype: ProviderType::DeepSeek,
        });
        registry.register("deepseek", provider.clone());

        assert!(registry.get("deepseek").is_some());
        assert!(registry.get("unknown").is_none());
    }

    #[test]
    fn first_registered_is_default() {
        let mut registry = ProviderRegistry::new();
        registry.register(
            "deepseek",
            Arc::new(MockProvider {
                ptype: ProviderType::DeepSeek,
            }),
        );
        registry.register(
            "openai",
            Arc::new(MockProvider {
                ptype: ProviderType::OpenAI,
            }),
        );

        let default = registry.default_provider().unwrap();
        assert_eq!(default.provider_type(), ProviderType::DeepSeek);
    }

    #[test]
    fn empty_registry_returns_none() {
        let registry = ProviderRegistry::new();
        assert!(registry.default_provider().is_none());
        assert!(registry.get("anything").is_none());
    }

    #[test]
    fn provider_names_lists_all() {
        let mut registry = ProviderRegistry::new();
        registry.register(
            "deepseek",
            Arc::new(MockProvider {
                ptype: ProviderType::DeepSeek,
            }),
        );
        registry.register(
            "openai",
            Arc::new(MockProvider {
                ptype: ProviderType::OpenAI,
            }),
        );

        let mut names = registry.provider_names();
        names.sort();
        assert_eq!(names, vec!["deepseek", "openai"]);
    }
}

#[cfg(test)]
mod model_registry_tests {
    use super::ModelRegistry;
    use std::sync::Arc;

    struct MockModel {
        id: String,
    }

    #[async_trait::async_trait]
    impl crate::LanguageModel for MockModel {
        async fn stream_chat(
            &self,
            _messages: &[crate::Message],
            _tools: &[crate::ToolSpec],
        ) -> Result<Box<dyn crate::StreamChunkIterator>, crate::AiError> {
            unimplemented!("mock")
        }
        fn model_id(&self) -> &str {
            &self.id
        }
    }

    fn mock(id: &str) -> Arc<dyn crate::LanguageModel> {
        Arc::new(MockModel { id: id.to_string() })
    }

    #[test]
    fn register_and_lookup() {
        let mut r = ModelRegistry::new();
        r.register("deepseek", mock("deepseek-v4-flash"));
        assert!(r.get("deepseek").is_some());
        assert!(r.get("unknown").is_none());
    }

    #[test]
    fn first_registered_is_default() {
        let mut r = ModelRegistry::new();
        r.register("a", mock("a"));
        r.register("b", mock("b"));
        assert_eq!(r.default_model().unwrap().model_id(), "a");
    }

    #[test]
    fn empty_registry_returns_none() {
        let r = ModelRegistry::new();
        assert!(r.default_model().is_none());
        assert!(r.get("x").is_none());
    }

    #[test]
    fn model_names_lists_all() {
        let mut r = ModelRegistry::new();
        r.register("a", mock("a"));
        r.register("b", mock("b"));
        let mut names = r.model_names();
        names.sort();
        assert_eq!(names, vec!["a", "b"]);
    }
}
