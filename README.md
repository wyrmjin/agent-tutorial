# Agent Tutorial

基于 Rust 构建的编码智能体（Coding Agent）教程项目。通过 DeepSeek API 驱动的 LLM 对话循环，智能体可以调用 Bash、文件读写等工具，在终端中自动完成编码任务。

## 架构概览

```
crates/
├── provider/   # LLM Provider 抽象层
├── agent/      # Agent 主循环（用户 ↔ LLM ↔ 工具）
├── tool/       # 工具系统（Bash、ReadFile、WriteFile）
└── logger/     # 统一日志系统（基于 tracing）
src/
└── main.rs     # CLI 入口
```

## 工作流程

1. 用户在终端输入任务描述
2. Agent 将用户输入 + 对话历史发送给 LLM
3. LLM 返回文本回复，或发起工具调用（Shell 命令/读写文件）
4. Agent 执行工具调用，将结果返回给 LLM
5. 循环直到 LLM 给出最终文本回复

## 快速开始

### 前置要求

- Rust 工具链（1.80+）
- DeepSeek API Key（[获取地址](https://platform.deepseek.com)）

### 安装与运行

```bash
# 1. 克隆仓库
git clone git@github.com:wyrmjin/agent-tutorial.git
cd agent-tutorial

# 2. 配置 API Key
cp .env.example .env
# 编辑 .env，填入你的 DEEPSEEK_API_KEY

# 3. 编译并运行
cargo run --release
```

运行后进入交互模式，直接输入任务即可：

```
Agent 已就绪，输入提示词（输入 exit / quit 退出）:

> 帮我创建一个 hello world 的 Python 脚本
```

### 日志

日志输出到控制台（stderr，可读格式）和 `./logs/` 目录（JSON 格式，按天滚动）。日志级别通过 `.env` 中的 `LOG_LEVEL` 配置（`TRACE` / `DEBUG` / `INFO` / `WARN` / `ERROR`）。

## 内置工具

| 工具 | 名称 | 描述 |
|------|------|------|
| `bash` | Shell 执行 | 执行 bash 命令，带超时和危险命令拦截 |
| `read_file` | 文件读取 | 读取文件内容，支持行数限制 |
| `write_file` | 文件写入 | 写入文件内容，自动创建父目录 |

### Bash 安全机制

Bash 工具内置 10 类危险命令检测，拦截以下操作：

- `rm -rf /` 等递归删除系统关键目录
- `mkfs.*` / `mke2fs` 等磁盘格式化命令
- `dd` 写入块设备
- 输出重定向到 `/dev/sd*` / `/dev/nvme*` 等块设备
- `shutdown` / `reboot` / `halt` 等系统关机命令
- Fork bomb 模式（`:(){ :|:& };:` 等）
- `chmod -R` / `chown -R` 作用于系统关键路径
- `curl ... | sh` 等下载内容直接管道给解释器
- `git push --force` 到 `main` / `master` 等受保护分支

## Provider 支持

当前已实现 DeepSeek Provider（兼容 OpenAI API 协议）。`Provider` trait 定义了统一的 LLM 后端抽象，预留了对 Anthropic、OpenAI、Ollama 等后端的扩展接口。

切换模型：

```rust
DeepSeekConfig::new(api_key)
    .with_model("deepseek-chat")     // 或 "deepseek-reasoner"
    .with_base_url("https://your-proxy.com")  // 可选：自定义 API 地址
```

## 项目结构扩展

- 添加新工具：实现 `tool::Tool` trait 并在 `main.rs` 中注册到 `ToolRegistry`
- 添加新 Provider：实现 `provider::Provider` trait
- 修改 Agent 行为：调整 `AgentConfig` 参数（如 `max_tool_rounds`）

## 许可证

MIT License © 2026 Wyrm.J
