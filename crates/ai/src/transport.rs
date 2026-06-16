//! Generic HTTP transport. Knows nothing about specific protocols or providers.

use crate::error::AiError;
use crate::message::{Message, ToolSpec};
use crate::protocol::{Protocol, SamplingParams};
use crate::provider::Endpoint;
use crate::stream::{DecodingStream, StreamChunkIterator};

pub struct Transport {
    client: reqwest::Client,
}

impl Transport {
    pub fn new() -> Self {
        Self { client: reqwest::Client::new() }
    }

    /// Encode via `protocol`, POST to the provider `endpoint`, and return a
    /// decoding stream of `StreamChunk`s.
    pub async fn stream(
        &self,
        endpoint: &Endpoint,
        protocol: &dyn Protocol,
        model_id: &str,
        params: &SamplingParams,
        messages: &[Message],
        tools: &[ToolSpec],
    ) -> Result<Box<dyn StreamChunkIterator>, AiError> {
        let body = protocol.build_body(model_id, params, messages, tools)?;
        let url = format!("{}{}", endpoint.base_url, protocol.endpoint_path());

        let mut req = self.client.post(&url).json(&body);
        for (k, v) in protocol.protocol_headers() {
            req = req.header(k.as_str(), v.as_str());
        }
        for (k, v) in &endpoint.headers {
            req = req.header(k.as_str(), v.as_str());
        }

        let response = req.send().await?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let message = response.text().await.unwrap_or_default();
            logger::error!(%status, %message, "ai api error");
            return Err(AiError::Api { status, message });
        }
        logger::debug!("ai api response ok");

        Ok(Box::new(DecodingStream::new(
            Box::pin(response.bytes_stream()),
            protocol.new_decoder(),
        )))
    }
}

impl Default for Transport {
    fn default() -> Self {
        Self::new()
    }
}
