//! 用量统计 — 协议/供应商无关的通用模型。
//!
//! 采用"互斥分量"口径: `input` / `cache_read` / `cache_write` 两两不重叠,
//! `total_tokens` 即三者与 `output` 之和。`UsageNormalizer` 由各供应商提供,
//! 负责把原始响应归一化成互斥语义(详见下方各实现)。

use serde_json::Value;

/// 一次 LLM 响应的用量统计。字段为互斥分量。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Usage {
    /// 未命中/常规输入 token 数(已扣除缓存命中部分)。
    pub input: u64,
    /// 输出 token 数。
    pub output: u64,
    /// 命中缓存读到的 token 数。
    pub cache_read: u64,
    /// 本轮写入缓存的 token 数(Anthropic `cache_creation`;DeepSeek 为 0)。
    pub cache_write: u64,
    /// = `input + output + cache_read + cache_write`。
    pub total_tokens: u64,
}

impl Usage {
    /// 由互斥分量构造, 自动计算 `total_tokens`。
    ///
    /// decoder 用它构造 `Usage`, 避免手算 `total_tokens` 出错。
    pub fn from_components(input: u64, output: u64, cache_read: u64, cache_write: u64) -> Self {
        Self {
            input,
            output,
            cache_read,
            cache_write,
            total_tokens: input + output + cache_read + cache_write,
        }
    }
}

/// 把原始 API 响应中的 usage 字段归一化为互斥 [`Usage`]。
///
/// 协议层解码时只提取公共字段(`prompt_tokens` / `completion_tokens`),
/// 其余供应商特有字段(如缓存命中等)随 `extra` 透传, 由各供应商的
/// normalizer 自行提取——协议因此与具体供应商解耦。
pub trait UsageNormalizer: Send + Sync {
    /// `extra` 包含响应中除 `prompt_tokens` / `completion_tokens` 外的全部字段
    /// (DeepSeek 的 `prompt_cache_hit_tokens`、OpenAI 的 `prompt_tokens_details` 等)。
    fn normalize(&self, prompt_tokens: u64, completion_tokens: u64, extra: &Value) -> Usage;
}

/// OpenAI 标准口径: 缓存命中来自 `prompt_tokens_details.cached_tokens`。
#[derive(Default)]
pub struct OpenAiUsageNormalizer;

impl UsageNormalizer for OpenAiUsageNormalizer {
    fn normalize(&self, prompt_tokens: u64, completion_tokens: u64, extra: &Value) -> Usage {
        let cache_read = extra
            .get("prompt_tokens_details")
            .and_then(|d| d.get("cached_tokens"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        Usage::from_components(
            prompt_tokens.saturating_sub(cache_read),
            completion_tokens,
            cache_read,
            0,
        )
    }
}

/// DeepSeek 口径: 缓存命中来自 `prompt_cache_hit_tokens`。
#[derive(Default)]
pub struct DeepSeekUsageNormalizer;

impl UsageNormalizer for DeepSeekUsageNormalizer {
    fn normalize(&self, prompt_tokens: u64, completion_tokens: u64, extra: &Value) -> Usage {
        let cache_read = extra
            .get("prompt_cache_hit_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        Usage::from_components(
            prompt_tokens.saturating_sub(cache_read),
            completion_tokens,
            cache_read,
            0,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{DeepSeekUsageNormalizer, OpenAiUsageNormalizer, Usage, UsageNormalizer};
    use serde_json::json;

    #[test]
    fn from_components_sums_total() {
        let u = Usage::from_components(80, 20, 10, 5);
        assert_eq!(u.input, 80);
        assert_eq!(u.output, 20);
        assert_eq!(u.cache_read, 10);
        assert_eq!(u.cache_write, 5);
        assert_eq!(u.total_tokens, 80 + 20 + 10 + 5);
    }

    #[test]
    fn default_is_all_zero() {
        let u = Usage::default();
        assert_eq!(u, Usage::from_components(0, 0, 0, 0));
        assert_eq!(u.total_tokens, 0);
    }

    #[test]
    fn total_is_simple_sum_of_components() {
        // 互斥不变量: total 永远是四项之和, 无重叠。
        let u = Usage::from_components(100, 50, 30, 0);
        assert_eq!(u.total_tokens, 180);
    }

    #[test]
    fn openai_normalizer_reads_details_cached() {
        // OpenAI 风格: prompt_tokens_details.cached_tokens = 25。
        // prompt_tokens=100 已含命中部分, 扣除后 input=75。
        // total = 75(input) + 5(output) + 25(cache_read) + 0 = 105。
        let extra = json!({"prompt_tokens_details": {"cached_tokens": 25}});
        let u = OpenAiUsageNormalizer.normalize(100, 5, &extra);
        assert_eq!(u.input, 75);
        assert_eq!(u.cache_read, 25);
        assert_eq!(u.total_tokens, 105);
    }

    #[test]
    fn openai_normalizer_ignores_deepseek_field() {
        // 即便 extra 带有 DeepSeek 的 prompt_cache_hit_tokens, OpenAI normalizer 也忽略。
        let extra = json!({"prompt_cache_hit_tokens": 30});
        let u = OpenAiUsageNormalizer.normalize(100, 5, &extra);
        assert_eq!(u.cache_read, 0);
        assert_eq!(u.input, 100);
    }

    #[test]
    fn deepseek_normalizer_reads_cache_hit() {
        // DeepSeek 风格: prompt_cache_hit_tokens = 30。
        let extra = json!({"prompt_cache_hit_tokens": 30, "prompt_cache_miss_tokens": 70});
        let u = DeepSeekUsageNormalizer.normalize(100, 5, &extra);
        assert_eq!(u.input, 70);
        assert_eq!(u.cache_read, 30);
        assert_eq!(u.total_tokens, 105);
    }

    #[test]
    fn deepseek_normalizer_ignores_openai_field() {
        // 即便 extra 带有 OpenAI 的 prompt_tokens_details, DeepSeek normalizer 也忽略。
        let extra = json!({"prompt_tokens_details": {"cached_tokens": 25}});
        let u = DeepSeekUsageNormalizer.normalize(100, 5, &extra);
        assert_eq!(u.cache_read, 0);
        assert_eq!(u.input, 100);
    }

    #[test]
    fn normalizer_zero_cache_when_field_absent() {
        // 两者在缺少缓存字段时 cache_read 均为 0。
        let extra = json!({});
        assert_eq!(
            OpenAiUsageNormalizer.normalize(50, 10, &extra).cache_read,
            0
        );
        assert_eq!(
            DeepSeekUsageNormalizer.normalize(50, 10, &extra).cache_read,
            0
        );
    }
}
