use std::sync::Arc;

use agent::{Agent, AgentConfig};
use futures::StreamExt;
use logger::{Logger, debug, error, info};
use ai::{DeepSeekProvider, ProviderConfig, ProviderRegistry};
use tool::{BashTool, ReadFileTool, ToolRegistry, WriteFileTool};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();
    let _log_guard = Logger::builder().init()?;

    let api_key =
        std::env::var("DEEPSEEK_API_KEY").expect("DEEPSEEK_API_KEY must be set in .env file");

    // 使用 ProviderRegistry 注册 ai
    let mut provider_registry = ProviderRegistry::new();
    provider_registry.register(
        "deepseek",
        Arc::new(DeepSeekProvider::new(
            ProviderConfig::new(api_key).with_model("deepseek-v4-flash"),
        )),
    );

    let provider = provider_registry
        .default_provider()
        .expect("no ai registered");

    // 注册工具
    let mut tools = ToolRegistry::new();
    tools.register(BashTool::default());
    tools.register(ReadFileTool::default());
    tools.register(WriteFileTool::default());

    let tools = Arc::new(tools);

    let os = os_info::get();
    info!(%os, "agent starting");

    let cwd = std::env::current_dir()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| "unknown".to_string());
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_else(|_| "unknown".to_string());
    let shell = std::env::var("SHELL")
        .or_else(|_| std::env::var("COMSPEC"))
        .unwrap_or_else(|_| "unknown".to_string());
    let user = std::env::var("USER")
        .or_else(|_| std::env::var("USERNAME"))
        .unwrap_or_else(|_| "unknown".to_string());

    let system_prompt = format!(
        "you are a coding agent at {os}. Use bash and read_file tools to solve tasks. Act, don't explain.\n\n\
         当前用户环境信息:\n\
         - 操作系统: {os}\n\
         - 当前工作目录: {cwd}\n\
         - 用户主目录: {home}\n\
         - 默认 Shell: {shell}\n\
         - 当前用户: {user}"
    );
    debug!(%system_prompt, "system prompt configured");
    let agent = Agent::new(provider, system_prompt);

    let config = AgentConfig::default();

    println!("Agent 已就绪，输入提示词（输入 exit / quit 退出）:\n");

    loop {
        // 读取用户输入
        let mut input = String::new();
        print!("> ");
        use std::io::Write;
        if std::io::stdout().flush().is_err() {
            break;
        }
        match std::io::stdin().read_line(&mut input) {
            Ok(0) => break,
            Ok(_) => {}
            Err(e) => {
                error!("读取输入失败: {e}");
                continue;
            }
        }
        let input = input.trim().to_string();

        if input.is_empty() {
            continue;
        }

        if input.eq_ignore_ascii_case("exit") || input.eq_ignore_ascii_case("quit") {
            println!("再见！");
            break;
        }

        // 执行 agent（流式）
        let mut handle = agent.run_stream(&input, &config, Arc::clone(&tools));

        while let Some(event) = handle.stream_mut().next().await {
            match &event {
                agent::AgentEvent::ApprovalRequired {
                    tool_call_id,
                    path,
                    message,
                    ..
                } => {
                    println!("\n⚠️  {message}");
                    println!("   文件路径: {path}");
                    println!("   是否同意读取？(同意/拒绝)");

                    match tokio::task::spawn_blocking(|| {
                        let mut buf = String::new();
                        std::io::stdin().read_line(&mut buf).map(|n| (n, buf))
                    })
                    .await
                    {
                        Ok(Ok((0, _))) => {
                            handle.deny(tool_call_id.clone(), "EOF".to_string());
                        }
                        Ok(Ok((_, buf))) => {
                            if is_approved(&buf) {
                                handle.approve(tool_call_id.clone());
                            } else {
                                handle.deny(tool_call_id.clone(), buf.trim().to_string());
                            }
                        }
                        Ok(Err(e)) => {
                            handle.deny(tool_call_id.clone(), format!("输入错误: {e}"));
                        }
                        Err(e) => {
                            handle.deny(tool_call_id.clone(), format!("输入任务失败: {e}"));
                        }
                    }
                }
                _ => display_event(event),
            }
        }

        println!();
    }

    Ok(())
}

/// 判断用户输入是否表示同意。采用白名单模式：只有明确肯定的词才视为同意。
fn is_approved(input: &str) -> bool {
    let lower = input.trim().to_lowercase();
    let trimmed = lower.trim();
    matches!(
        trimmed,
        "同意"
            | "允许"
            | "可以"
            | "yes"
            | "y"
            | "ok"
            | "批准"
            | "sure"
            | "go ahead"
            | "好的"
            | "没问题"
            | "当然"
    )
}

fn display_event(event: agent::AgentEvent) {
    match event {
        agent::AgentEvent::Text(text) => {
            print!("{}", text);
        }
        agent::AgentEvent::ToolRequest { name, .. } => {
            println!("\n[调用工具: {name}]");
        }
        agent::AgentEvent::ToolResponse {
            content, is_error, ..
        } => {
            if is_error {
                println!(
                    "[工具错误: {}]",
                    content.chars().take(200).collect::<String>()
                );
            } else {
                println!(
                    "[工具结果: {}]",
                    content.chars().take(200).collect::<String>()
                );
            }
        }
        agent::AgentEvent::TurnEnd { usage } => {
            println!("\n\n--- 回合结束 ---");
            println!(
                "输入 tokens: {}, 输出 tokens: {}",
                usage.input_tokens, usage.output_tokens
            );
        }
        agent::AgentEvent::ApprovalRequired { .. } => {
            // 已在主循环中内联处理，不会到达这里
        }
        agent::AgentEvent::Error {
            message,
            recoverable,
        } => {
            if recoverable {
                println!("\n⚠️  可恢复错误: {message}");
            } else {
                println!("\n❌ 错误: {message}");
            }
        }
        agent::AgentEvent::Done { .. } => {
            // Done 事件在调用处处理，这里不显示
        }
    }
}
