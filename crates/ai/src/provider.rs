//! Provider abstraction: who we talk to (endpoint + auth + supported protocols).

use crate::protocol::ProtocolKind;

/// Endpoint with auth already baked into headers; transport uses it directly.
#[derive(Debug, Clone)]
pub struct Endpoint {
    pub base_url: String,
    pub headers: Vec<(String, String)>,
}

/// How the API key is injected as a header.
#[derive(Debug, Clone)]
pub enum AuthStyle {
    /// `Authorization: Bearer {key}`
    Bearer,
    /// `{header_name}: {key}`, e.g. "x-api-key"
    ApiKeyHeader(String),
}

/// A provider describes who we talk to — not how (that's `Protocol`).
pub trait Provider: Send + Sync {
    fn name(&self) -> &str;
    fn endpoint(&self) -> Endpoint;
    fn supported_protocols(&self) -> &[ProtocolKind];
}

/// Generic config-driven provider; covers DeepSeek / OpenAI / Anthropic etc.
pub struct GenericProvider {
    name: String,
    base_url: String,
    api_key: String,
    auth: AuthStyle,
    protocols: Vec<ProtocolKind>,
    extra_headers: Vec<(String, String)>,
}

impl GenericProvider {
    pub fn new(
        name: impl Into<String>,
        base_url: impl Into<String>,
        api_key: impl Into<String>,
        auth: AuthStyle,
        protocols: Vec<ProtocolKind>,
    ) -> Self {
        Self {
            name: name.into(),
            base_url: base_url.into(),
            api_key: api_key.into(),
            auth,
            protocols,
            extra_headers: Vec::new(),
        }
    }

    /// Add a provider-specific header (e.g. OpenRouter's referer).
    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.extra_headers.push((key.into(), value.into()));
        self
    }
}

impl Provider for GenericProvider {
    fn name(&self) -> &str {
        &self.name
    }

    fn endpoint(&self) -> Endpoint {
        let mut headers = self.extra_headers.clone();
        match &self.auth {
            AuthStyle::Bearer => {
                headers.push(("Authorization".to_string(), format!("Bearer {}", self.api_key)));
            }
            AuthStyle::ApiKeyHeader(name) => {
                headers.push((name.clone(), self.api_key.clone()));
            }
        }
        Endpoint { base_url: self.base_url.clone(), headers }
    }

    fn supported_protocols(&self) -> &[ProtocolKind] {
        &self.protocols
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bearer_auth_injects_authorization_header() {
        let p = GenericProvider::new(
            "deepseek",
            "https://api.deepseek.com",
            "sk-123",
            AuthStyle::Bearer,
            vec![ProtocolKind::OpenAiCompletions],
        );
        let ep = p.endpoint();
        assert_eq!(ep.base_url, "https://api.deepseek.com");
        assert!(ep
            .headers
            .iter()
            .any(|(k, v)| k == "Authorization" && v == "Bearer sk-123"));
        assert_eq!(p.supported_protocols(), &[ProtocolKind::OpenAiCompletions]);
    }

    #[test]
    fn api_key_header_auth_injects_named_header() {
        let p = GenericProvider::new(
            "anthropic",
            "https://api.anthropic.com",
            "sk-ant",
            AuthStyle::ApiKeyHeader("x-api-key".to_string()),
            vec![ProtocolKind::AnthropicMessages],
        );
        let ep = p.endpoint();
        assert!(ep.headers.iter().any(|(k, v)| k == "x-api-key" && v == "sk-ant"));
    }
}
