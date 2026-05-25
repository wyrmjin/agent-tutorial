use agent::{Agent, AgentConfig};
use provider::{DeepSeekConfig, DeepSeekProvider};
use tool::{BashTool, ReadFileTool, Tool, ToolRegistry, WriteFileTool};
use logger::{debug, error, info, Logger};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();
    let _log_guard = Logger::builder().init()?;
    let api_key =
        std::env::var("DEEPSEEK_API_KEY").expect("DEEPSEEK_API_KEY must be set in .env file");

    let provider = DeepSeekProvider::new(DeepSeekConfig::new(api_key).with_model("deepseek-chat"));

    // 注册工具
    let mut tools = ToolRegistry::new();
    tools.register(BashTool::default());
    tools.register(ReadFileTool::default());

    let os = os_info::get();
    info!(%os, "agent starting");
    let system_prompt = format!(
        "you are a coding agent at {}. Use bash and read_file tools to solve tasks. Act, don't explain.",
        os
    );
    debug!(%system_prompt, "system prompt configured");
    let mut agent = Agent::new(provider, system_prompt);

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
            Ok(0) => break, // EOF
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

        // 执行 agent
        let events = agent.run(&input, &config, &tools).await?;

        for event in events {
            match event {
                agent::AgentEvent::Text(text) => {
                    print!("{}", text);
                }
                agent::AgentEvent::TurnEnd { usage } => {
                    println!("\n\n--- 回合结束 ---");
                    println!(
                        "输入 tokens: {}, 输出 tokens: {}",
                        usage.input_tokens, usage.output_tokens
                    );
                }
                other => {
                    println!("[{other:?}]");
                }
            }
        }

        println!();
    }

    Ok(())
}
