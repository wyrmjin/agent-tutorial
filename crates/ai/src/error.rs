//! Structured error types for the provider layer.

/// Errors that can occur in the provider layer.
#[derive(Debug, thiserror::Error)]
pub enum ProviderError {
    /// API request returned a non-success status code.
    #[error("API error ({status}): {message}")]
    Api { status: u16, message: String },

    /// Failed to parse a response from the API.
    #[error("Failed to parse API response: {0}")]
    Parse(String),

    /// Network or connection error.
    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),

    /// SSE stream ended unexpectedly.
    #[error("Stream ended unexpectedly: {0}")]
    StreamUnexpected(String),

    /// Configuration error (e.g. missing API key).
    #[error("Configuration error: {0}")]
    Config(String),
}
