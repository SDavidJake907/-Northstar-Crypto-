//! Prompt loading — system prompts for entry, exit, scan, review, and optimizer.

use std::path::PathBuf;

pub const QWEN_FALLBACK: &str = r#"You are a crypto trading AI. Output ONLY JSON: {"action":"HOLD","confidence":0.5,"trend_score":0,"score":0.0,"lane":"rejected","reason":"fallback"}"#;

/// Load a prompt file with 4-tier fallback:
/// 1. Env override (e.g. AI_ENTRY_PROMPT_PATH=C:\...\prompt.txt)
/// 2. Exe-relative (works from Task Scheduler / services)
/// 3. CWD-relative (current behavior)
/// 4. CWD/data/... (explicit current_dir join)
pub(crate) fn load_prompt_from_paths(env_key: &str, rel_path: &str) -> Option<String> {
    // 1) Env override — explicit path wins
    if let Ok(p) = std::env::var(env_key) {
        if !p.is_empty() {
            let pb = PathBuf::from(&p);
            if let Ok(content) = std::fs::read_to_string(&pb) {
                let trimmed = content.trim();
                if !trimmed.is_empty() {
                    tracing::info!("[AI] Loaded {env_key} from env: {}", pb.display());
                    return Some(trimmed.to_string());
                }
            }
            tracing::warn!("[AI] {env_key} set but unreadable: {}", pb.display());
        }
    }

    // 2) Exe-relative — where the binary lives
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let pb = dir.join(rel_path);
            if let Ok(content) = std::fs::read_to_string(&pb) {
                let trimmed = content.trim();
                if !trimmed.is_empty() {
                    return Some(trimmed.to_string());
                }
            }
        }
    }

    // 3) CWD-relative — current behavior
    let pb = PathBuf::from(rel_path);
    if let Ok(content) = std::fs::read_to_string(&pb) {
        let trimmed = content.trim();
        if !trimmed.is_empty() {
            return Some(trimmed.to_string());
        }
    }

    // 4) CWD/data/... (explicit current_dir join)
    if let Ok(cwd) = std::env::current_dir() {
        let pb = cwd.join(rel_path);
        if let Ok(content) = std::fs::read_to_string(&pb) {
            let trimmed = content.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
    }

    None
}

pub(crate) fn load_nemo_prompt() -> String {
    load_prompt_from_paths("AI_PROMPT_PATH", "data/nemo_prompt.txt")
        .unwrap_or_else(|| QWEN_FALLBACK.to_string())
}

pub fn load_nemo_exit_prompt() -> String {
    load_prompt_from_paths("AI_EXIT_PROMPT_PATH", "data/nemo_exit_prompt.txt")
        .unwrap_or_else(load_nemo_prompt)
}

#[allow(dead_code)]
pub fn load_nemo_scan_prompt() -> String {
    load_prompt_from_paths("AI_SCAN_PROMPT_PATH", "data/nemo_scan_prompt.txt")
        .unwrap_or_else(|| "You are a crypto market analyst. Output: MARKET|direction|reason then TOP1-3|symbol|conf|reason".to_string())
}

pub fn load_nemo_review_prompt() -> String {
    load_prompt_from_paths("AI_REVIEW_PROMPT_PATH", "data/nemo_review_prompt.txt")
        .unwrap_or_else(|| "You are a trade reviewer. Output: REVIEW|grade|lesson".to_string())
}

#[allow(dead_code)]
pub fn load_nemo_boss_prompt() -> String {
    load_prompt_from_paths("AI_BOSS_PROMPT_PATH", "data/nemo_boss_prompt.txt")
        .unwrap_or_else(|| "You are the Chief Trading AI. Output: REGIME|direction|conf|reason then ENTRY|symbol|CONFIRM/VETO|conf|reason".to_string())
}

pub fn load_nemo_entry_prompt() -> String {
    load_prompt_from_paths("AI_ENTRY_PROMPT_PATH", "data/nemo_entry_prompt.txt")
        .unwrap_or_else(load_nemo_prompt)
}

/// Meme-specific entry prompt for L4 lane coins.
/// Falls back to standard entry prompt if meme prompt file not found.
pub fn load_nemo_meme_entry_prompt() -> String {
    load_prompt_from_paths("AI_MEME_ENTRY_PROMPT_PATH", "data/nemo_meme_entry_prompt.txt")
        .unwrap_or_else(load_nemo_entry_prompt)
}

/// Meme-specific exit prompt for L4 lane positions.
/// Falls back to standard exit prompt if meme prompt file not found.
pub fn load_nemo_meme_exit_prompt() -> String {
    load_prompt_from_paths("AI_MEME_EXIT_PROMPT_PATH", "data/nemo_meme_exit_prompt.txt")
        .unwrap_or_else(load_nemo_exit_prompt)
}

/// Load NPU scanner prompt
pub(crate) fn load_npu_scan_prompt() -> String {
    load_prompt_from_paths("NPU_SCAN_PROMPT_PATH", "data/npu_scan_prompt.txt")
        .or_else(|| load_prompt_from_paths("DEEPSEEK_VERIFY_PROMPT_PATH", "data/deepseek_verify_prompt.txt"))
        .unwrap_or_else(|| "You are a lane verification AI. Output JSON with verdict AGREE or DISAGREE.".to_string())
}
