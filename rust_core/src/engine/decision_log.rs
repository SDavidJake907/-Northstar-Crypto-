//! Decision Logger — writes every AI model decision to structured JSONL files.
//! The 49B optimizer reads these to understand what each model is doing,
//! identify failure patterns, and update prompts accordingly.
//!
//! Files:
//!   data/ai1_decisions.jsonl  — Nemotron 9B entry confirmations
//!   data/ai2_decisions.jsonl  — OpenReasoning 1.5B exit advisories
//!   data/ai3_decisions.jsonl  — Phi-3 NPU scan verdicts

use serde::Serialize;
use std::io::Write;

const AI1_LOG: &str = "data/ai1_decisions.jsonl";
const AI2_LOG: &str = "data/ai2_decisions.jsonl";
const AI3_LOG: &str = "data/ai3_decisions.jsonl";
const MAX_LINES: usize = 500; // rotate after 500 entries

// ── AI_1 Entry Decision ───────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct Ai1Decision {
    pub ts:          f64,
    pub symbol:      String,
    pub action:      String,   // "BUY" | "HOLD" | "VETO"
    pub confidence:  f64,
    pub regime:      String,
    pub lane:        String,
    pub tools_used:  Vec<String>,
    pub tool_rounds: u8,
    pub reason:      String,
    pub source:      String,   // "batch" | "tool_call"
    pub latency_ms:  u64,
}

pub fn log_ai1(entry: &Ai1Decision) {
    append_jsonl(AI1_LOG, entry);
}

// ── AI_2 Exit Decision ────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct Ai2Decision {
    pub ts:          f64,
    pub symbol:      String,
    pub action:      String,   // "SELL" | "HOLD"
    pub confidence:  f64,
    pub reason:      String,
    pub pnl_pct:     f64,      // current PnL when exit was evaluated
    pub hold_min:    f64,      // minutes held so far
    pub regime:      String,
    pub latency_ms:  u64,
}

pub fn log_ai2(entry: &Ai2Decision) {
    append_jsonl(AI2_LOG, entry);
}

// ── AI_3 Phi-3 Scan Decision ──────────────────────────────────────────────────

#[derive(Serialize)]
pub struct Ai3Decision {
    pub ts:          f64,
    pub symbol:      String,
    pub verdict:     String,   // "PASS" | "REJECT"
    pub lane:        String,   // "L1" | "L2" | "L3" | "NONE"
    pub action:      String,   // recomputed_action: "BUY" | "HOLD"
    pub reason:      String,
    pub latency_ms:  u64,
}

pub fn log_ai3(entry: &Ai3Decision) {
    append_jsonl(AI3_LOG, entry);
}

// ── Summary for 49B Optimizer ─────────────────────────────────────────────────

/// Build a compact summary of recent decisions from all 3 models.
/// Called by the 49B optimizer prompt builder.
pub fn build_model_summary() -> String {
    let mut out = String::with_capacity(2048);

    out.push_str("=== AI MODEL DECISION LOGS (recent) ===\n\n");

    // AI_1 summary
    out.push_str("AI_1 (9B Entry):\n");
    out.push_str(&summarise_ai1(50));
    out.push('\n');

    // AI_2 summary
    out.push_str("AI_2 (1.5B Exit):\n");
    out.push_str(&summarise_ai2(50));
    out.push('\n');

    // AI_3 summary
    out.push_str("AI_3 (Phi-3 Scan):\n");
    out.push_str(&summarise_ai3(50));
    out.push('\n');

    out
}

fn summarise_ai1(n: usize) -> String {
    let entries = read_last_n::<serde_json::Value>(AI1_LOG, n);
    if entries.is_empty() { return "  No data yet.\n".to_string(); }

    let total = entries.len();
    let buys  = entries.iter().filter(|e| e["action"] == "BUY").count();
    let holds = entries.iter().filter(|e| e["action"] == "HOLD" || e["action"] == "VETO").count();

    // Tools used frequency
    let mut tool_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for e in &entries {
        if let Some(tools) = e["tools_used"].as_array() {
            for t in tools {
                if let Some(name) = t.as_str() {
                    *tool_counts.entry(name.to_string()).or_insert(0) += 1;
                }
            }
        }
    }
    let mut top_tools: Vec<_> = tool_counts.iter().collect();
    top_tools.sort_by(|a, b| b.1.cmp(a.1));
    let tools_str: Vec<String> = top_tools.iter().take(5)
        .map(|(k, v)| format!("{}({}x)", k, v))
        .collect();

    // Recent BUY symbols
    let recent_buys: Vec<String> = entries.iter()
        .filter(|e| e["action"] == "BUY")
        .take(10)
        .filter_map(|e| e["symbol"].as_str().map(|s| s.to_string()))
        .collect();

    format!(
        "  Last {} decisions: {}B / {}H  ({:.0}% BUY rate)\n\
         Top tools: {}\n\
         Recent BUYs: {}\n",
        total, buys, holds,
        (buys as f64 / total as f64) * 100.0,
        if tools_str.is_empty() { "none".to_string() } else { tools_str.join(", ") },
        if recent_buys.is_empty() { "none".to_string() } else { recent_buys.join(", ") },
    )
}

fn summarise_ai2(n: usize) -> String {
    let entries = read_last_n::<serde_json::Value>(AI2_LOG, n);
    if entries.is_empty() { return "  No data yet.\n".to_string(); }

    let total = entries.len();
    let sells = entries.iter().filter(|e| e["action"] == "SELL").count();
    let holds = entries.iter().filter(|e| e["action"] == "HOLD").count();

    // Coins it keeps holding despite losses
    let mut hold_losers: Vec<String> = entries.iter()
        .filter(|e| e["action"] == "HOLD" && e["pnl_pct"].as_f64().unwrap_or(0.0) < -1.0)
        .filter_map(|e| e["symbol"].as_str().map(|s| s.to_string()))
        .collect();
    hold_losers.dedup();

    format!(
        "  Last {} decisions: {}S / {}H  ({:.0}% SELL rate)\n\
         HOLDing losers (pnl<-1%%): {}\n",
        total, sells, holds,
        (sells as f64 / total as f64) * 100.0,
        if hold_losers.is_empty() { "none".to_string() } else { hold_losers.join(", ") },
    )
}

fn summarise_ai3(n: usize) -> String {
    let entries = read_last_n::<serde_json::Value>(AI3_LOG, n);
    if entries.is_empty() { return "  No data yet.\n".to_string(); }

    let total  = entries.len();
    let passes = entries.iter().filter(|e| e["verdict"] == "PASS").count();
    let rejects= entries.iter().filter(|e| e["verdict"] == "REJECT").count();

    // Most rejected coins
    let mut reject_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for e in entries.iter().filter(|e| e["verdict"] == "REJECT") {
        if let Some(sym) = e["symbol"].as_str() {
            *reject_counts.entry(sym.to_string()).or_insert(0) += 1;
        }
    }
    let mut top_rejects: Vec<_> = reject_counts.iter().collect();
    top_rejects.sort_by(|a, b| b.1.cmp(a.1));
    let reject_str: Vec<String> = top_rejects.iter().take(5)
        .map(|(k, v)| format!("{}({}x)", k, v))
        .collect();

    format!(
        "  Last {} scans: {}P / {}R  ({:.0}% pass rate)\n\
         Most rejected: {}\n",
        total, passes, rejects,
        (passes as f64 / total as f64) * 100.0,
        if reject_str.is_empty() { "none".to_string() } else { reject_str.join(", ") },
    )
}

// ── File helpers ──────────────────────────────────────────────────────────────

fn append_jsonl<T: Serialize>(path: &str, item: &T) {
    let Ok(line) = serde_json::to_string(item) else { return };
    let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true).append(true).open(path)
    else { return };
    let _ = writeln!(f, "{}", line);

    // Rotate if too large
    if let Ok(meta) = std::fs::metadata(path) {
        if meta.len() > 1_000_000 {
            rotate_log(path);
        }
    }
}

fn rotate_log(path: &str) {
    // Keep last MAX_LINES lines
    let Ok(content) = std::fs::read_to_string(path) else { return };
    let lines: Vec<&str> = content.lines().collect();
    if lines.len() > MAX_LINES {
        let kept = lines[lines.len() - MAX_LINES..].join("\n");
        let _ = std::fs::write(path, kept + "\n");
    }
}

fn read_last_n<T: serde::de::DeserializeOwned>(path: &str, n: usize) -> Vec<T> {
    let Ok(content) = std::fs::read_to_string(path) else { return vec![] };
    content.lines()
        .rev()
        .take(n)
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect()
}
