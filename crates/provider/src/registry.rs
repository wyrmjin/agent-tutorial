//! Runtime provider registry.
//!
//! Allows registering multiple LLM providers by name and looking them up
//! dynamically, instead of hard-coding a single provider type.

use std::collections::HashMap;
use std::sync::Arc;

use crate::{Provider, ProviderType};

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
        ) -> anyhow::Result<Box<dyn crate::StreamChunkIterator>> {
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
