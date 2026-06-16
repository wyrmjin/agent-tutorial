//! Runtime model registry.
//!
//! Allows registering multiple language models by name and looking them up
//! dynamically, instead of hard-coding a single model.

use std::collections::HashMap;
use std::sync::Arc;

use crate::LanguageModel;

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
