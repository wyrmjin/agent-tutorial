//! 用量统计 — 协议/供应商无关的通用模型。
//!
//! 采用"互斥分量"口径: `input` / `cache_read` / `cache_write` 两两不重叠,
//! `total_tokens` 即三者与 `output` 之和。各家 decoder 负责把原始响应
//! 归一化成互斥语义(参见各 Protocol 实现)。

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

#[cfg(test)]
mod tests {
    use super::Usage;

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
}
