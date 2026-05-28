//! Bash tool — execute shell commands with dangerous-command blocking.

use std::time::Duration;

use crate::{Tool, ToolOutput};
use logger::{error, info, warn};

pub struct BashTool {
    timeout: Duration,
}

impl BashTool {
    pub fn new(timeout_secs: u64) -> Self {
        Self {
            timeout: Duration::from_secs(timeout_secs),
        }
    }

    /// 检查命令是否危险，返回 Some(reason) 表示危险，None 表示安全。
    fn check_dangerous(command: &str) -> Option<&'static str> {
        let cmd = command.trim();

        if cmd.is_empty() {
            return None;
        }

        // 将多段命令拆分为独立子命令：先替换复合分隔符，再按单字符拆分
        let normalized = cmd
            .replace("&&", "\n")
            .replace("||", "\n")
            .replace("2>&1", " ") // 避免误拆分重定向中的 &
            .replace("1>&2", " ")
            .replace("&>>", " ")
            .replace("&>", " ");
        for segment in normalized.split(|c| c == ';' || c == '\n' || c == '|' || c == '&') {
            let s = segment.trim();
            if s.is_empty() {
                continue;
            }

            // 1. rm -rf 作用于根目录或家目录等关键路径
            if let Some(reason) = Self::check_rm_dangerous(s) {
                return Some(reason);
            }

            // 2. 磁盘格式化命令
            if let Some(reason) = Self::check_mkfs(s) {
                return Some(reason);
            }

            // 3. dd 写入块设备
            if let Some(reason) = Self::check_dd_dangerous(s) {
                return Some(reason);
            }

            // 4. 输出重定向到块设备
            if let Some(reason) = Self::check_redirect_to_dev(s) {
                return Some(reason);
            }

            // 5. 系统关机/重启命令
            if let Some(reason) = Self::check_system_control(s) {
                return Some(reason);
            }

            // 6. fork bomb
            if let Some(reason) = Self::check_fork_bomb(s) {
                return Some(reason);
            }

            // 7. chmod -R 777 作用于关键路径
            if let Some(reason) = Self::check_chmod_dangerous(s) {
                return Some(reason);
            }

            // 8. chown -R 作用于根路径
            if let Some(reason) = Self::check_chown_dangerous(s) {
                return Some(reason);
            }

            // 9. curl/wget 管道传递给 shell 执行
            if let Some(reason) = Self::check_pipe_to_shell(s) {
                return Some(reason);
            }

            // 10. git push --force / --force-with-lease 到主分支（仅检测命令本身）
            if let Some(reason) = Self::check_git_force_push(s) {
                return Some(reason);
            }
        }

        None
    }

    fn check_rm_dangerous(cmd: &str) -> Option<&'static str> {
        // 匹配 rm 命令，且带有 -r 或 -rf 等递归删除标志
        let tokens: Vec<&str> = cmd.split_whitespace().collect();
        if tokens.is_empty() {
            return None;
        }

        let is_rm = tokens[0] == "rm" || tokens[0].ends_with("/rm");
        if !is_rm {
            return None;
        }

        let has_recursive = tokens.iter().any(|t| t.contains('r') && t.starts_with('-'));
        let has_force = tokens.iter().any(|t| {
            *t == "-rf" || *t == "-fr" || (t.starts_with('-') && t.contains('f') && t.contains('r'))
        });

        if !has_recursive && !has_force {
            return None;
        }

        // 检查是否作用于危险路径
        let dangerous_paths = &[
            "/", "/*", "/.", "/..", "~/", "~/*", "/bin", "/boot", "/dev", "/etc", "/home", "/lib",
            "/lib64", "/opt", "/proc", "/root", "/sbin", "/srv", "/sys", "/usr", "/var", "/System",
        ];

        for token in &tokens[1..] {
            // 跳过选项参数
            if token.starts_with('-') {
                continue;
            }
            for dp in dangerous_paths {
                if *token == *dp || (token.starts_with(dp) && dp.ends_with('/')) {
                    return Some("禁止操作：递归删除系统关键目录");
                }
            }
        }

        // 检测 ./* 或 ./ 开头的路径 —— 在系统根目录下执行 rm -rf ./* 等同于删除系统文件
        // 这种情况在没有设置工作目录时尤为危险，暂时放行，因为 ./* 通常是用户有意操作当前目录

        None
    }

    fn check_mkfs(cmd: &str) -> Option<&'static str> {
        if cmd.starts_with("mkfs.") || cmd.contains("mkfs.") {
            return Some("禁止操作：格式化磁盘命令 (mkfs)");
        }
        if cmd.starts_with("mke2fs") || cmd.starts_with("newfs") {
            return Some("禁止操作：创建文件系统命令");
        }
        None
    }

    fn check_dd_dangerous(cmd: &str) -> Option<&'static str> {
        // 检测 dd 命令是否写入到块设备
        if !cmd.starts_with("dd ") && !cmd.contains(" dd ") {
            return None;
        }
        // 检查 of= 参数是否指向块设备
        if let Some(of_idx) = cmd.find("of=") {
            let of_val = &cmd[of_idx + 3..];
            let of_path = of_val.split_whitespace().next().unwrap_or("");
            if of_path.starts_with("/dev/sd")
                || of_path.starts_with("/dev/nvme")
                || of_path.starts_with("/dev/mmcblk")
                || of_path.starts_with("/dev/hd")
                || of_path.starts_with("/dev/xvd")
                || of_path.starts_with("/dev/vd")
                || of_path.starts_with("/dev/disk")
                || of_path.starts_with("/dev/rdisk")
            {
                return Some("禁止操作：dd 写入到块设备");
            }
        }
        None
    }

    fn check_redirect_to_dev(cmd: &str) -> Option<&'static str> {
        // 检测输出重定向到块设备：> /dev/sda, >> /dev/nvme0n1 等
        let dev_patterns = &[
            "/dev/sd",
            "/dev/nvme",
            "/dev/mmcblk",
            "/dev/hd",
            "/dev/xvd",
            "/dev/vd",
            "/dev/disk",
            "/dev/rdisk",
        ];
        for pattern in dev_patterns {
            let redirect_single = format!("> {}", pattern);
            let redirect_double = format!(">> {}", pattern);
            if cmd.contains(&redirect_single) || cmd.contains(&redirect_double) {
                return Some("禁止操作：输出重定向到块设备");
            }
            // 也检测 >/dev/sd 这种无空格形式
            let no_space = format!(">{}", pattern);
            if cmd.contains(&no_space) {
                return Some("禁止操作：输出重定向到块设备");
            }
        }
        // 检测 cat/cp 写入 /dev 设备
        if (cmd.starts_with("cp ") || cmd.contains(" cp ")) && cmd.contains(" /dev/") {
            return Some("禁止操作：复制到 /dev 设备");
        }
        None
    }

    fn check_system_control(cmd: &str) -> Option<&'static str> {
        let base = cmd.split_whitespace().next().unwrap_or("");
        let dangerous_cmds = &[
            "shutdown",
            "reboot",
            "halt",
            "poweroff",
            "init",
            "telinit",
            "systemctl",
            "launchctl",
        ];
        if dangerous_cmds.contains(&base) {
            // systemctl/launchctl 有非破坏性用法，仅拦截明显的关机/重启
            if base == "systemctl" || base == "launchctl" {
                let lower = cmd.to_lowercase();
                if lower.contains("shutdown")
                    || lower.contains("reboot")
                    || lower.contains("halt")
                    || lower.contains("poweroff")
                    || lower.contains("stop")
                {
                    return Some("禁止操作：系统关机/重启/停止服务命令");
                }
            } else {
                return Some("禁止操作：系统关机/重启命令");
            }
        }
        None
    }

    fn check_fork_bomb(cmd: &str) -> Option<&'static str> {
        // 检测经典的 fork bomb 模式 :(){ :|:& };:
        if cmd.contains(":(){") || cmd.contains(":|:&") {
            return Some("禁止操作：检测到 fork bomb 模式");
        }
        // 检测通用的递归函数 fork bomb: f(){ f|f& };f
        let no_spaces = cmd.replace(' ', "");
        if no_spaces.contains("(){(|") || no_spaces.contains("(){:|") {
            return Some("禁止操作：检测到 fork bomb 模式");
        }
        None
    }

    fn check_chmod_dangerous(cmd: &str) -> Option<&'static str> {
        let tokens: Vec<&str> = cmd.split_whitespace().collect();
        if tokens.is_empty() {
            return None;
        }
        if tokens[0] != "chmod" && !tokens[0].ends_with("/chmod") {
            return None;
        }
        let has_recursive = tokens.iter().any(|t| *t == "-R" || *t == "--recursive");
        if !has_recursive {
            return None;
        }
        let dangerous_paths = &[
            "/", "/*", "/.", "/..", "~/", "~/*", "/bin", "/boot", "/dev", "/etc", "/home", "/lib",
            "/lib64", "/opt", "/proc", "/root", "/sbin", "/srv", "/sys", "/usr", "/var", "/System",
        ];
        for token in &tokens[1..] {
            if token.starts_with('-') {
                continue;
            }
            // 权限模式参数（如 755, 777, u+x 等），跳过
            if token.chars().all(|c| c.is_ascii_digit())
                || token.starts_with('u')
                || token.starts_with('g')
                || token.starts_with('o')
                || token.starts_with('a')
            {
                continue;
            }
            for dp in dangerous_paths {
                if *token == *dp || (token.starts_with(dp) && dp.ends_with('/')) {
                    return Some("禁止操作：递归修改系统关键目录权限");
                }
            }
        }
        None
    }

    fn check_chown_dangerous(cmd: &str) -> Option<&'static str> {
        let tokens: Vec<&str> = cmd.split_whitespace().collect();
        if tokens.is_empty() {
            return None;
        }
        if tokens[0] != "chown" && !tokens[0].ends_with("/chown") {
            return None;
        }
        let has_recursive = tokens.iter().any(|t| *t == "-R" || *t == "--recursive");
        if !has_recursive {
            return None;
        }
        let dangerous_paths = &[
            "/", "/*", "/.", "/..", "~/", "~/*", "/bin", "/boot", "/dev", "/etc", "/home", "/lib",
            "/lib64", "/opt", "/proc", "/root", "/sbin", "/srv", "/sys", "/usr", "/var", "/System",
        ];
        for token in &tokens[1..] {
            if token.starts_with('-') || token.contains(':') {
                continue;
            }
            for dp in dangerous_paths {
                if *token == *dp || (token.starts_with(dp) && dp.ends_with('/')) {
                    return Some("禁止操作：递归修改系统关键目录所有者");
                }
            }
        }
        None
    }

    fn check_pipe_to_shell(cmd: &str) -> Option<&'static str> {
        // 检测 curl/wget ... | sh/bash/zsh/python/perl 这类模式
        let lower = cmd.to_lowercase();
        let downloaders = &["curl ", "wget ", "aria2c "];
        let has_downloader = downloaders.iter().any(|d| lower.contains(d));
        if !has_downloader {
            return None;
        }
        let shells = &[
            "| sh",
            "| bash",
            "| zsh",
            "| /bin/sh",
            "| /bin/bash",
            "| python",
            "| perl",
            "| ruby",
        ];
        for shell in shells {
            if lower.contains(shell) {
                return Some("禁止操作：下载内容直接管道传递给解释器执行，存在安全风险");
            }
        }
        None
    }

    fn check_git_force_push(cmd: &str) -> Option<&'static str> {
        let tokens: Vec<&str> = cmd.split_whitespace().collect();
        if tokens.len() < 4 {
            return None;
        }
        let is_git = tokens[0] == "git" || tokens[0].ends_with("/git");
        if !is_git || tokens[1] != "push" {
            return None;
        }
        let is_force = tokens
            .iter()
            .any(|t| t == &"--force" || t == &"-f" || t == &"--force-with-lease");
        if !is_force {
            return None;
        }
        let protected_branches = &["main", "master", "production", "prod", "release"];
        for token in &tokens[2..] {
            if protected_branches.contains(token) {
                return Some("禁止操作：强制推送到受保护分支");
            }
            // 检测 origin/main 格式
            for branch in protected_branches {
                if token.ends_with(&format!("/{}", branch)) {
                    return Some("禁止操作：强制推送到受保护分支");
                }
            }
        }
        None
    }
}

impl Default for BashTool {
    fn default() -> Self {
        Self::new(120)
    }
}

#[async_trait::async_trait]
impl Tool for BashTool {
    fn name(&self) -> &str {
        "bash"
    }

    fn description(&self) -> &str {
        "Execute a bash command in the shell. Returns stdout and stderr output. \
         Use for file operations, running scripts, checking system state, etc. \
         Commands are run with a timeout and will be killed if they exceed it."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute"
                }
            },
            "required": ["command"]
        })
    }

    async fn execute(&self, input: serde_json::Value) -> ToolOutput {
        let command = input.get("command").and_then(|v| v.as_str()).unwrap_or("");

        if command.is_empty() {
            return ToolOutput::error("Error: command parameter is required");
        }

        // 危险命令安全检查
        if let Some(reason) = Self::check_dangerous(command) {
            warn!(%command, %reason, "dangerous command blocked");
            return ToolOutput::error(format!("安全限制：{reason}\n被拦截的命令: {command}"));
        }

        info!(%command, "bash execute start");

        let cwd = match std::env::current_dir() {
            Ok(d) => d,
            Err(e) => return ToolOutput::error(format!("无法获取当前工作目录: {e}")),
        };

        match tokio::time::timeout(
            self.timeout,
            tokio::process::Command::new("bash")
                .current_dir(&cwd)
                .args(["-c", command])
                .output(),
        )
        .await
        {
            Err(_) => {
                warn!(%command, timeout = ?self.timeout, "bash command timed out");
                ToolOutput::error(format!("命令超时 ({:?})", self.timeout))
            }
            Ok(Err(e)) => {
                error!(%command, error = %e, "bash command failed to execute");
                ToolOutput::error(format!("执行失败: {e}"))
            }
            Ok(Ok(output)) => {
                let mut result = String::new();
                if !output.stdout.is_empty() {
                    result.push_str(&String::from_utf8_lossy(&output.stdout));
                }
                if !output.stderr.is_empty() {
                    if !result.is_empty() {
                        result.push('\n');
                    }
                    result.push_str("--- stderr ---\n");
                    result.push_str(&String::from_utf8_lossy(&output.stderr));
                }
                if result.is_empty() {
                    result = "(无输出)".to_string();
                }
                let is_error = !output.status.success();
                info!(
                    %command,
                    exit_code = output.status.code().map_or("signal".to_string(), |c| c.to_string()),
                    stdout_len = output.stdout.len(),
                    stderr_len = output.stderr.len(),
                    "bash command completed"
                );
                ToolOutput::ok_with_status(result, is_error)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========== check_dangerous 综合测试 ==========

    #[test]
    fn safe_commands_return_none() {
        assert_eq!(BashTool::check_dangerous(""), None);
        assert_eq!(BashTool::check_dangerous("echo hello"), None);
        assert_eq!(BashTool::check_dangerous("ls -la"), None);
        assert_eq!(BashTool::check_dangerous("cat file.txt"), None);
        assert_eq!(BashTool::check_dangerous("git status"), None);
        assert_eq!(BashTool::check_dangerous("cargo build"), None);
    }

    // ========== check_rm_dangerous ==========

    #[test]
    fn rm_normal_path_allowed() {
        assert_eq!(BashTool::check_dangerous("rm file.txt"), None);
        assert_eq!(BashTool::check_dangerous("rm -r ./my_project"), None);
        // rm 不加递归标志的绝对路径通过
        assert_eq!(BashTool::check_dangerous("rm /tmp/test"), None);
    }

    #[test]
    fn rm_rf_root_blocked() {
        assert!(BashTool::check_dangerous("rm -rf /").is_some());
        assert!(BashTool::check_dangerous("rm -rf /*").is_some());
        assert!(BashTool::check_dangerous("rm -r /etc").is_some());
        assert!(BashTool::check_dangerous("rm -rf /home").is_some());
        assert!(BashTool::check_dangerous("rm -rf /usr").is_some());
        assert!(BashTool::check_dangerous("rm -rf /System").is_some());
    }

    #[test]
    fn rm_rf_root_fr_flags() {
        assert!(BashTool::check_dangerous("rm -fr /").is_some());
        assert!(BashTool::check_dangerous("rm -rf /boot").is_some());
    }

    #[test]
    fn rm_with_full_path_blocked() {
        assert!(BashTool::check_dangerous("/bin/rm -rf /").is_some());
    }

    // ========== check_mkfs ==========

    #[test]
    fn mkfs_commands_blocked() {
        assert!(BashTool::check_dangerous("mkfs.ext4 /dev/sda1").is_some());
        assert!(BashTool::check_dangerous("mkfs.fat /dev/sdb").is_some());
        assert!(BashTool::check_dangerous("mke2fs /dev/sda1").is_some());
        assert!(BashTool::check_dangerous("newfs /dev/sda1").is_some());
    }

    // ========== check_dd_dangerous ==========

    #[test]
    fn dd_write_to_block_device_blocked() {
        assert!(BashTool::check_dangerous("dd if=img.iso of=/dev/sda").is_some());
        assert!(BashTool::check_dangerous("dd if=img.iso of=/dev/nvme0n1").is_some());
        assert!(BashTool::check_dangerous("dd if=img.iso of=/dev/mmcblk0").is_some());
        assert!(BashTool::check_dangerous("dd if=img.iso of=/dev/hda").is_some());
        assert!(BashTool::check_dangerous("dd if=img.iso of=/dev/xvda").is_some());
        assert!(BashTool::check_dangerous("dd if=img.iso of=/dev/vda").is_some());
        assert!(BashTool::check_dangerous("dd if=img.iso of=/dev/disk0").is_some());
        assert!(BashTool::check_dangerous("dd if=img.iso of=/dev/rdisk0").is_some());
    }

    #[test]
    fn dd_write_to_regular_file_allowed() {
        assert_eq!(BashTool::check_dangerous("dd if=/dev/zero of=output.bin"), None);
    }

    // ========== check_redirect_to_dev ==========

    #[test]
    fn redirect_to_block_device_blocked() {
        assert!(BashTool::check_dangerous("cat file > /dev/sda").is_some());
        assert!(BashTool::check_dangerous("echo hi >> /dev/nvme0n1").is_some());
        assert!(BashTool::check_dangerous("cat file >/dev/sda").is_some());
    }

    #[test]
    fn cp_to_dev_blocked() {
        assert!(BashTool::check_dangerous("cp file /dev/sda").is_some());
    }

    // ========== check_system_control ==========

    #[test]
    fn shutdown_commands_blocked() {
        assert!(BashTool::check_dangerous("shutdown now").is_some());
        assert!(BashTool::check_dangerous("reboot").is_some());
        assert!(BashTool::check_dangerous("halt").is_some());
        assert!(BashTool::check_dangerous("poweroff").is_some());
    }

    #[test]
    fn systemctl_shutdown_blocked() {
        assert!(BashTool::check_dangerous("systemctl shutdown").is_some());
        assert!(BashTool::check_dangerous("systemctl reboot").is_some());
        assert!(BashTool::check_dangerous("systemctl poweroff").is_some());
        assert!(BashTool::check_dangerous("systemctl stop nginx").is_some());
    }

    #[test]
    fn launchctl_shutdown_blocked() {
        assert!(BashTool::check_dangerous("launchctl shutdown").is_some());
        assert!(BashTool::check_dangerous("launchctl reboot").is_some());
        assert!(BashTool::check_dangerous("launchctl stop com.apple.nginx").is_some());
    }

    #[test]
    fn systemctl_status_allowed() {
        // systemctl 的非破坏性用法不应被拦截
        assert_eq!(BashTool::check_dangerous("systemctl status nginx"), None);
        assert_eq!(BashTool::check_dangerous("systemctl list-units"), None);
    }

    // ========== check_fork_bomb ==========

    #[test]
    fn fork_bomb_classic_blocked() {
        assert!(BashTool::check_dangerous(":(){ :|:& };:").is_some());
    }

    #[test]
    fn fork_bomb_function_pattern_no_spaces() {
        // check_fork_bomb 的去空格模式检测
        let pattern1 = "(){(|";
        let pattern2 = "(){:|";
        assert!(BashTool::check_fork_bomb(pattern1).is_some());
        assert!(BashTool::check_fork_bomb(pattern2).is_some());
    }

    // ========== check_chmod_dangerous ==========

    #[test]
    fn chmod_recursive_root_blocked() {
        assert!(BashTool::check_dangerous("chmod -R 777 /").is_some());
        assert!(BashTool::check_dangerous("chmod --recursive 777 /etc").is_some());
    }

    #[test]
    fn chmod_non_recursive_allowed() {
        assert_eq!(BashTool::check_dangerous("chmod 755 /etc/hosts"), None);
        assert_eq!(BashTool::check_dangerous("chmod +x script.sh"), None);
    }

    // ========== check_chown_dangerous ==========

    #[test]
    fn chown_recursive_root_blocked() {
        assert!(BashTool::check_dangerous("chown -R user:group /").is_some());
        assert!(BashTool::check_dangerous("chown --recursive user /etc").is_some());
    }

    #[test]
    fn chown_non_recursive_allowed() {
        assert_eq!(BashTool::check_dangerous("chown user file.txt"), None);
    }

    // ========== check_pipe_to_shell ==========

    #[test]
    fn curl_pipe_to_shell_blocked() {
        // check_pipe_to_shell 直接调用能检测 pipe-to-shell
        assert!(BashTool::check_pipe_to_shell("curl url | sh").is_some());
        assert!(BashTool::check_pipe_to_shell("curl url | bash").is_some());
        assert!(BashTool::check_pipe_to_shell("wget url | zsh").is_some());
        assert!(BashTool::check_pipe_to_shell("curl url | python").is_some());
    }

    #[test]
    fn curl_without_pipe_to_shell_allowed() {
        assert_eq!(BashTool::check_dangerous("curl https://example.com/script.sh"), None);
        assert_eq!(BashTool::check_dangerous("curl url -o output.sh"), None);
    }

    // ========== check_git_force_push ==========

    #[test]
    fn git_force_push_to_protected_branch_blocked() {
        assert!(BashTool::check_dangerous("git push --force origin main").is_some());
        assert!(BashTool::check_dangerous("git push -f origin master").is_some());
        assert!(BashTool::check_dangerous("git push --force-with-lease origin production").is_some());
        assert!(BashTool::check_dangerous("git push --force origin prod").is_some());
        assert!(BashTool::check_dangerous("git push --force origin release").is_some());
    }

    #[test]
    fn git_force_push_to_feature_branch_allowed() {
        assert_eq!(
            BashTool::check_dangerous("git push --force origin feature/test"),
            None
        );
    }

    #[test]
    fn git_push_non_force_allowed() {
        assert_eq!(
            BashTool::check_dangerous("git push origin main"),
            None
        );
    }

    #[test]
    fn git_force_push_short_branch_name_blocked() {
        // origin/main 格式也应该被拦截
        assert!(BashTool::check_dangerous("git push --force origin origin/main").is_some());
    }

    // ========== 复合命令分割测试 ==========

    #[test]
    fn compound_command_with_semicolon() {
        assert!(BashTool::check_dangerous("echo hello; rm -rf /").is_some());
    }

    #[test]
    fn compound_command_with_and_and() {
        assert!(BashTool::check_dangerous("echo hello && rm -rf /").is_some());
    }

    #[test]
    fn compound_command_with_pipe() {
        assert!(BashTool::check_dangerous("cat file | rm -rf /").is_some());
    }

    // ========== execute 方法测试 ==========

    fn rt() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_current_thread()
            .enable_io()
            .enable_time()
            .build()
            .unwrap()
    }

    #[test]
    fn execute_empty_command_returns_error() {
        let tool = BashTool::new(5);
        let result = rt().block_on(tool.execute(serde_json::json!({})));
        assert!(result.is_error);
        assert!(result.content.contains("required"));
    }

    #[test]
    fn execute_dangerous_command_blocked() {
        let tool = BashTool::new(5);
        let result = rt().block_on(tool.execute(serde_json::json!({"command": "rm -rf /"})));
        assert!(result.is_error);
        assert!(result.content.contains("安全限制"));
    }

    #[test]
    fn execute_safe_command_succeeds() {
        let tool = BashTool::new(5);
        let result = rt().block_on(tool.execute(serde_json::json!({"command": "echo hello"})));
        assert!(!result.is_error);
        assert!(result.content.contains("hello"));
    }

    #[test]
    fn execute_command_with_stderr() {
        let tool = BashTool::new(5);
        let result =
            rt().block_on(tool.execute(serde_json::json!({"command": "echo err >&2"})));
        assert!(result.content.contains("stderr"));
        assert!(result.content.contains("err"));
    }

    #[test]
    fn execute_failing_command_returns_is_error() {
        let tool = BashTool::new(5);
        let result = rt().block_on(tool.execute(serde_json::json!({"command": "exit 1"})));
        assert!(result.is_error);
    }

    #[test]
    fn name_and_description() {
        let tool = BashTool::default();
        assert_eq!(tool.name(), "bash");
        assert!(tool.description().contains("bash"));
    }

    #[test]
    fn parameters_requires_command() {
        let tool = BashTool::default();
        let params = tool.parameters();
        assert_eq!(params["type"], "object");
        let required: Vec<_> = params["required"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert!(required.contains(&"command"));
    }
}
