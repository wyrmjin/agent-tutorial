//! Generic HTTP transport. Knows nothing about specific protocols or providers.

use std::sync::Arc;

use crate::error::AiError;
use crate::message::{Message, ToolSpec};
use crate::protocol::{Protocol, SamplingParams};
use crate::provider::Endpoint;
use crate::stream::{DecodingStream, StreamChunkIterator};
use crate::usage::UsageNormalizer;
use tracing::{debug, error};

pub struct Transport {
    client: reqwest::Client,
}

impl Transport {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
        }
    }

    /// Encode via `protocol`, POST to the provider `endpoint`, and return a
    /// decoding stream of `StreamChunk`s.
    ///
    /// `normalizer` 来自调用方的 provider, 透传给 decoder 用于 usage 归一化。
    #[allow(clippy::too_many_arguments)] // 正交三层的接线 + 请求载荷, 难以进一步收敛。
    pub async fn stream(
        &self,
        endpoint: &Endpoint,
        protocol: &dyn Protocol,
        normalizer: Arc<dyn UsageNormalizer>,
        model_id: &str,
        params: &SamplingParams,
        messages: &[Message],
        tools: &[ToolSpec],
    ) -> Result<Box<dyn StreamChunkIterator>, AiError> {
        let body = protocol.build_body(model_id, params, messages, tools)?;
        let url = join_url(&endpoint.base_url, protocol.endpoint_path());

        let mut req = self.client.post(&url).json(&body);
        for (k, v) in protocol.protocol_headers() {
            req = req.header(k.as_str(), v.as_str());
        }
        for (k, v) in &endpoint.headers {
            req = req.header(k.as_str(), v.as_str());
        }
        debug!(%url, body = %body, "ai api request");

        let response = req.send().await?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let message = response.text().await.unwrap_or_default();
            error!(%status, %message, "ai api error");
            return Err(AiError::Api { status, message });
        }
        debug!(status = %response.status(), "ai api response ok");

        Ok(Box::new(DecodingStream::new(
            Box::pin(response.bytes_stream()),
            protocol.new_decoder(normalizer),
        )))
    }
}

impl Default for Transport {
    fn default() -> Self {
        Self::new()
    }
}

/// 拼接 base_url 与 endpoint_path, 规范化交界处多余的斜杠:
/// base 带/不带尾斜杠、path 带/不带前导斜杠, 都恰好用一个 '/' 连接。
fn join_url(base_url: &str, path: &str) -> String {
    let base = base_url.trim_end_matches('/');
    let path = path.trim_start_matches('/');
    format!("{base}/{path}")
}

#[cfg(test)]
mod tests {
    use super::join_url;

    #[test]
    fn join_url_handles_trailing_slash_on_base() {
        assert_eq!(
            join_url("https://api.deepseek.com/", "/chat/completions"),
            "https://api.deepseek.com/chat/completions"
        );
    }

    #[test]
    fn join_url_handles_no_trailing_slash() {
        assert_eq!(
            join_url("https://api.deepseek.com", "/chat/completions"),
            "https://api.deepseek.com/chat/completions"
        );
    }

    #[test]
    fn join_url_handles_path_without_leading_slash() {
        assert_eq!(
            join_url("https://api.deepseek.com", "chat/completions"),
            "https://api.deepseek.com/chat/completions"
        );
    }

    #[test]
    fn join_url_handles_both_slashes() {
        assert_eq!(
            join_url("https://api.openai.com/v1/", "/chat/completions"),
            "https://api.openai.com/v1/chat/completions"
        );
    }
}
