//! 统一日志系统 — 封装 tracing 配置，提供控制台+文件双输出、双格式。

use std::io;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::prelude::*;

/// 日志配置结构体，通过 Builder 模式构建。
#[derive(Debug)]
pub struct Logger {
    level: String,
    log_dir: String,
    file_prefix: String,
}

impl Default for Logger {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            log_dir: "./logs".to_string(),
            file_prefix: "agent-tutorial".to_string(),
        }
    }
}

impl Logger {
    pub fn builder() -> Self {
        Self::default()
    }

    pub fn level(mut self, level: impl Into<String>) -> Self {
        self.level = level.into();
        self
    }

    pub fn log_dir(mut self, dir: impl Into<String>) -> Self {
        self.log_dir = dir.into();
        self
    }

    pub fn file_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.file_prefix = prefix.into();
        self
    }

    /// 初始化日志系统。返回的 WorkerGuard 必须被调用方持有，否则文件写入会丢失。
    #[must_use = "WorkerGuard must be held, otherwise file logs are silently dropped"]
    pub fn init(self) -> anyhow::Result<tracing_appender::non_blocking::WorkerGuard> {
        // 控制台 Layer：stderr，人类可读，带颜色
        let console_layer = tracing_subscriber::fmt::layer()
            .with_target(true)
            .with_timer(tracing_subscriber::fmt::time::LocalTime::rfc_3339())
            .with_writer(io::stderr)
            .pretty();

        // 文件 Layer：JSON 格式，按天滚动
        let file_appender = tracing_appender::rolling::daily(&self.log_dir, &self.file_prefix);
        let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);
        let file_layer = tracing_subscriber::fmt::layer()
            .json()
            .with_writer(non_blocking);

        // 日志级别过滤：优先读环境变量 LOG_LEVEL，否则用配置值
        let filter =
            EnvFilter::try_from_env("LOG_LEVEL").unwrap_or_else(|_| EnvFilter::new(&self.level));

        tracing_subscriber::registry()
            .with(filter)
            .with(console_layer)
            .with(file_layer)
            .try_init()?;

        Ok(guard)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn try_init_twice_returns_err() {
        // First init should succeed
        let _guard = Logger::builder()
            .level("debug")
            .log_dir("/tmp/test-logs")
            .init()
            .expect("first init must succeed");

        // Second init should return Err because a global subscriber is already set
        let result = Logger::builder()
            .level("debug")
            .log_dir("/tmp/test-logs")
            .init();

        assert!(result.is_err(), "second try_init() should return Err");
    }
}
