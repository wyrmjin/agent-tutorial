//! Structured error types for the ai layer.

/// Errors that can occur in the ai layer.
#[derive(Debug, thiserror::Error)]
pub enum AiError {
    /// API request returned a non-success status code.
    #[error("API error ({status}): {message}")]
    Api { status: u16, message: String },

    /// Failed to encode a request body (e.g. parameters not serializable).
    #[error("Failed to encode request: {0}")]
    Encode(String),

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
