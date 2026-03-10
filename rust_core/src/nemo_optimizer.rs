//! Nemo Self-Optimizer — AI-driven parameter tuning (FULL AUTO).
//!
//! Every NEMO_OPTIMIZER_INTERVAL_SEC (default 1800s), Nemo reviews recent
//! trading performance and proposes parameter adjustments via structured
//! JSON tool calls. Changes are validated against a safety whitelist with
//! hard min/max bounds, written to .env, and logged to data/nemo_optimizer.jsonl.
//! The existing hot_reload() picks up changes within 10s.
//!
//! Auto-revert: if portfolio drops >5% after changes, revert to previous values.

use crate::ai_bridge::AiBridge;
use crate::journal::Journal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::Write;

const ERROR_FLAGS_FILE: &str = "data/error_flags.jsonl";

/// Append a runtime error to the error flags file so the 49B optimizer
/// can diagnose and suggest fixes on its next cycle.
pub fn flag_error(source: &str, message: &str) {
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0);
    let line = format!(
        "{{\"ts\":{ts:.0},\"source\":\"{source}\",\"message\":{}}}\n",
        serde_json::Value::String(message.to_string())
    );
    if let Ok(mut f) = fs::OpenOptions::new().create(true).append(true).open(ERROR_FLAGS_FILE) {
        let _ = f.write_all(line.as_bytes());
    }
}

// ── Safety Whitelist ─────────────────────────────────────────────

struct ParamBound {
    key: &'static str,
    min: f64,
    max: f64,
}

const PARAM_WHITELIST: &[ParamBound] = &[
    // ── Entry / Exit thresholds ──────────────────────────────────────
    ParamBound { key: "ENTRY_THRESHOLD",           min: 0.40, max: 0.75 },
    ParamBound { key: "EXIT_THRESHOLD",            min: 0.20, max: 0.60 },
    ParamBound { key: "NEMO_EXIT_MIN_CONF",        min: 0.45, max: 0.90 },
    ParamBound { key: "NEMO_EXIT_INTERVAL_SEC",    min: 30.0, max: 300.0 },
    // ── Adaptive confidence floor ────────────────────────────────────
    ParamBound { key: "HF_CONF_BASE",              min: 0.30, max: 0.70 },
    ParamBound { key: "HF_CONF_MIN",               min: 0.25, max: 0.60 },
    ParamBound { key: "HF_CONF_MAX",               min: 0.50, max: 0.85 },
    ParamBound { key: "NEMO_EXIT_MIN_HOLD_SEC",    min: 60.0, max: 3600.0 },
    // ── Position sizing ──────────────────────────────────────────────
    ParamBound { key: "BASE_USD",                  min: 15.0, max: 60.0 },
    ParamBound { key: "MIN_ORDER_USD",             min: 5.0,  max: 20.0 },
    // ── Profit / Loss targets ────────────────────────────────────────
    ParamBound { key: "TAKE_PROFIT_PCT",           min: 1.0,  max: 6.0  },
    ParamBound { key: "STOP_LOSS_PCT",             min: 0.5,  max: 3.0  },
    ParamBound { key: "TRAILING_STOP_PCT",         min: 0.15, max: 1.5  },
    ParamBound { key: "BREAKEVEN_ACTIVATE_PCT",    min: 0.3,  max: 1.5  },
    ParamBound { key: "TRAILING_STOP_ACTIVATE_PCT",min: 0.5,  max: 3.0  },
    // ── Timing / Cooldowns ───────────────────────────────────────────
    ParamBound { key: "MIN_HOLD_MINUTES",          min: 5.0,  max: 120.0 },
    ParamBound { key: "TRADE_COOLDOWN_SEC",        min: 15.0, max: 300.0 },
    ParamBound { key: "AGGR_COOLDOWN_SEC",         min: 30.0, max: 600.0 },
    ParamBound { key: "DEF_COOLDOWN_SEC",          min: 30.0, max: 600.0 },
    ParamBound { key: "PRICE_COOLDOWN_PCT",        min: 0.0,  max: 1.0  },
    ParamBound { key: "LIMIT_OFFSET_PCT",          min: 0.01, max: 2.0  },
    ParamBound { key: "ENTRY_ORDER_TIMEOUT_SEC",   min: 60.0, max: 900.0 },
    // ── Risk limits ──────────────────────────────────────────────────
    ParamBound { key: "AGGR_MAX_POSITIONS",        min: 1.0,  max: 8.0  },
    ParamBound { key: "DEF_MAX_POSITIONS",         min: 1.0,  max: 8.0  },
    ParamBound { key: "MAX_DAILY_LOSS_USD",        min: 3.0,  max: 30.0 },
    ParamBound { key: "MAX_CONFIDENCE_FOR_SIZING", min: 0.50, max: 0.95 },
    ParamBound { key: "AGGR_MAX_SPREAD_PCT",       min: 0.2,  max: 3.0  },
    ParamBound { key: "AGGR_MIN_VOL_RATIO",        min: 0.05, max: 1.0  },
    // ── Meme coin controls ───────────────────────────────────────────
    ParamBound { key: "MEME_BASE_USD",             min: 5.0,  max: 30.0 },
    ParamBound { key: "MEME_MAX_POSITIONS",        min: 1.0,  max: 4.0  },
    ParamBound { key: "MEME_TAKE_PROFIT_PCT",      min: 2.0,  max: 12.0 },
    ParamBound { key: "MEME_STOP_LOSS_PCT",        min: 1.0,  max: 5.0  },
    ParamBound { key: "MEME_TRAILING_STOP_PCT",    min: 0.3,  max: 3.0  },
    ParamBound { key: "MEME_LIMIT_OFFSET_PCT",     min: 0.3,  max: 3.0  },
];

const MAX_CHANGES_PER_CYCLE: usize = 3;
const MAX_APPLIES_PER_24H: usize = 6;

// ── Data Structures ──────────────────────────────────────────────

#[allow(dead_code)]
#[derive(Serialize, Deserialize, Debug, Clone)]
#[allow(dead_code)] // fields populated via serde deserialization from LLM JSON
pub struct Proposal {
    pub id: String,
    pub param: String,
    pub from: f64,
    pub to: f64,
    pub scope: String,
    pub reason: String,
    pub expected_metric: String,
    pub risk_estimate: String,
    pub timestamp: f64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ParamChange {
    key: String,
    old_value: f64,
    new_value: f64,
    clamped: bool,
    reason: String,
    risk_estimate: String,
    expected_metric: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct RejectedChange {
    key: String,
    attempted_value: String,
    reject_reason: String,
}

#[derive(Serialize, Deserialize, Debug)]
struct AuditEntry {
    timestamp: f64,
    cycle: u64,
    event: String,
    changes: Vec<ParamChange>,
    rejected: Vec<RejectedChange>,
    performance_summary: String,
    llm_raw_response: String,
    equity_at_change: f64,
}

// ── NemoOptimizer ────────────────────────────────────────────────

pub struct NemoOptimizer {
    enabled: bool,
    interval_sec: f64,
    last_run_ts: f64,
    cycle_count: u64,
    applies_today: usize,
    last_reset_day: u32,
    max_applies_per_24h: usize,
    strict: bool,
    // Auto-revert tracking
    pre_change_equity: f64,
    revert_drawdown_pct: f64,
    last_changes: Vec<ParamChange>,
}

impl NemoOptimizer {
    /// Check if portfolio has dropped enough since last change to trigger auto-revert.
    fn check_auto_revert(&mut self, equity: f64) {
        if self.last_changes.is_empty() || self.pre_change_equity <= 0.0 {
            return;
        }
        let drawdown_pct = (self.pre_change_equity - equity) / self.pre_change_equity * 100.0;
        if drawdown_pct >= self.revert_drawdown_pct {
            tracing::warn!(
                "[AI-OPT] Auto-revert triggered: equity dropped {:.1}% (threshold={:.1}%)",
                drawdown_pct, self.revert_drawdown_pct
            );
            // Build reverse changes and write them
            let reverts: Vec<ParamChange> = self.last_changes.iter().map(|c| ParamChange {
                key: c.key.clone(),
                old_value: c.new_value,
                new_value: c.old_value,
                clamped: false,
                reason: "auto_revert".into(),
                risk_estimate: "low".into(),
                expected_metric: "recovery".into(),
            }).collect();
            if let Err(e) = write_env_changes(&reverts) {
                tracing::error!("[AI-OPT] Auto-revert write failed: {e}");
            } else {
                tracing::info!("[AI-OPT] Auto-revert applied {} params", reverts.len());
                flag_error("optimizer", &format!("auto_revert_triggered:drawdown={drawdown_pct:.1}%"));
            }
            self.last_changes.clear();
            self.pre_change_equity = 0.0;
        }
    }

    pub fn new() -> Self {
        let enabled = std::env::var("NEMO_OPTIMIZER_ENABLED")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let interval_sec = std::env::var("NEMO_OPTIMIZER_INTERVAL_SEC")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1800.0);
        let max_applies_per_24h = std::env::var("NEMO_OPTIMIZER_MAX_APPLIES_24H")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(MAX_APPLIES_PER_24H);
        let strict = std::env::var("NEMO_OPT_STRICT")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let revert_drawdown_pct = std::env::var("NEMO_OPTIMIZER_REVERT_DRAWDOWN_PCT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(5.0);

        Self {
            enabled,
            interval_sec,
            last_run_ts: 0.0,
            cycle_count: 0,
            applies_today: 0,
            last_reset_day: 0,
            max_applies_per_24h,
            strict,
            pre_change_equity: 0.0,
            revert_drawdown_pct,
            last_changes: Vec::new(),
        }
    }

    /// Called every tick from the trading loop.
    pub async fn maybe_run(
        &mut self,
        now: f64,
        journal: &Journal,
        ai: &AiBridge,
        equity: f64,
    ) {
        // Re-read enabled flag (hot-reloadable)
        self.enabled = std::env::var("NEMO_OPTIMIZER_ENABLED")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        if !self.enabled {
            return;
        }

        self.interval_sec = std::env::var("NEMO_OPTIMIZER_INTERVAL_SEC")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1800.0);
        self.max_applies_per_24h = std::env::var("NEMO_OPTIMIZER_MAX_APPLIES_24H")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(MAX_APPLIES_PER_24H);
        self.strict = std::env::var("NEMO_OPT_STRICT")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        self.revert_drawdown_pct = std::env::var("NEMO_OPTIMIZER_REVERT_DRAWDOWN_PCT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(5.0);

        // Check auto-revert before anything else
        self.check_auto_revert(equity);

        if (now - self.last_run_ts) < self.interval_sec {
            return;
        }
        // Defer first run — don't fire immediately on startup.
        // WS reconnects during first minutes cause the channel to drop while optimizer blocks.
        if self.last_run_ts == 0.0 {
            self.last_run_ts = now;
            tracing::info!("[AI-OPT] First startup — deferring optimizer by {} sec", self.interval_sec as u32);
            return;
        }
        self.last_run_ts = now;

        // Reset daily counter
        let today = (now / 86400.0) as u32;
        if today != self.last_reset_day {
            self.applies_today = 0;
            self.last_reset_day = today;
        }

        if self.applies_today >= self.max_applies_per_24h {
            tracing::info!(
                "[AI-OPT] Daily limit reached ({}/{}), skipping",
                self.applies_today,
                self.max_applies_per_24h
            );
            return;
        }

        self.cycle_count += 1;
        tracing::info!("[AI-OPT] Starting optimization cycle {}", self.cycle_count);

        self.run_cycle(journal, ai, equity).await;
    }

    async fn run_cycle(&mut self, journal: &Journal, ai: &AiBridge, equity: f64) {
    // Step 1: Build performance summary
    let perf = journal.performance_snapshot(20);
    if perf.trade_count < 3 {
        tracing::info!("[AI-OPT] Only {} trades -- need >= 3, skipping", perf.trade_count);
        return;
    }

    // Step 2: Read current params
    let current_params = read_current_params();

    // Step 3: Build prompts
    let system_prompt = load_optimizer_prompt();
    let user_prompt = build_user_prompt(&perf, &current_params, journal);

    // Step 4+5: Call LLM with calculator loop (max 3 turns)
    let mut calc_context = String::new();
    let mut final_changes: Vec<ParamChange> = Vec::new();
    let mut final_rejected: Vec<RejectedChange> = Vec::new();
    let mut final_llm_response = String::new();
    let mut opt_error: Option<String> = None;

    tracing::info!("[AI-OPT] Using {}", ai.optimizer_model_label());

    for turn in 0..3u8 {
        let full_prompt = if calc_context.is_empty() {
            user_prompt.clone()
        } else {
            format!("{}\n{}", user_prompt, calc_context)
        };

        let start = std::time::Instant::now();
        let llm_response = match ai.infer_for_optimizer(&system_prompt, &full_prompt).await {
            Ok(text) => text,
            Err(e) => {
                tracing::warn!("[AI-OPT] LLM inference failed: {e}");
                return;
            }
        };
        let ms = start.elapsed().as_millis();
        tracing::info!("[AI-OPT] LLM responded in {ms}ms ({} chars) turn={}", llm_response.len(), turn + 1);
        if llm_response.trim().is_empty() {
            let msg = "empty_response".to_string();
            tracing::error!("[AI-OPT] LLM returned empty response, aborting cycle");
            flag_error("optimizer", "empty LLM response");
            opt_error = Some(msg);
            break;
        }

        let (changes, rejected, calc_result, parse_error) =
            parse_and_validate(&llm_response, &current_params, journal, self.strict);
        if let Some(err) = parse_error {
            tracing::error!("[AI-OPT] Strict validation failed: {err}");
            flag_error("optimizer", &format!("strict_validation:{err}"));
            opt_error = Some(format!("strict_validation:{err}"));
            break;
        }

        if let Some(result_text) = calc_result {
            calc_context.push_str(&result_text);
            continue; // re-prompt with calculator result
        }

        final_changes = changes;
        final_rejected = rejected;
        final_llm_response = llm_response;
        break;
    }

    if let Some(err) = opt_error {
        let perf_summary = format!(
            "{}T/{}W/{}L WR={:.0}% PnL=${:.2} hold={:.0}min",
            perf.trade_count, perf.win_count, perf.loss_count,
            perf.win_rate * 100.0, perf.total_pnl_usd, perf.avg_hold_minutes
        );
        write_audit_log(AuditEntry {
            timestamp: crate::ai_bridge::now_ts(),
            cycle: self.cycle_count,
            event: format!("error:{err}"),
            changes: Vec::new(),
            rejected: Vec::new(),
            performance_summary: perf_summary,
            llm_raw_response: final_llm_response,
            equity_at_change: equity,
        });
        return;
    }

    // Step 6: Apply
    if !final_changes.is_empty() {
        if let Err(e) = write_env_changes(&final_changes) {
            tracing::error!("[AI-OPT] Failed to write .env: {e}");
            return;
        }
        clear_error_flags();
        self.pre_change_equity = equity;
        self.last_changes = final_changes.clone();
        self.applies_today += final_changes.len();

        for c in &final_changes {
            tracing::info!(
                "[AI-OPT] APPLIED {} : {:.4} ? {:.4}{} -- {} (risk={}, expect={})",
                c.key, c.old_value, c.new_value,
                if c.clamped { " [CLAMPED]" } else { "" },
                c.reason, c.risk_estimate, c.expected_metric
            );
        }
    } else {
        tracing::info!("[AI-OPT] No changes -- AI says params are fine");
    }

    // Step 7: Audit log
    let perf_summary = format!(
        "{}T/{}W/{}L WR={:.0}% PnL=${:.2} hold={:.0}min",
        perf.trade_count, perf.win_count, perf.loss_count,
        perf.win_rate * 100.0, perf.total_pnl_usd, perf.avg_hold_minutes
    );
    write_audit_log(AuditEntry {
        timestamp: crate::ai_bridge::now_ts(),
        cycle: self.cycle_count,
        event: if final_changes.is_empty() { "no_change".into() } else { "applied".into() },
        changes: final_changes,
        rejected: final_rejected,
        performance_summary: perf_summary,
        llm_raw_response: final_llm_response,
        equity_at_change: equity,
    });
}
}

// ── Helper free functions ─────────────────────────────────────────

/// Read current whitelisted param values from .env into a HashMap.
fn read_current_params() -> HashMap<String, f64> {
    let mut map = HashMap::new();
    let env_path = ".env";
    let content = fs::read_to_string(env_path).unwrap_or_default();
    for bound in PARAM_WHITELIST {
        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with('#') { continue; }
            if let Some(rest) = trimmed.strip_prefix(bound.key) {
                if let Some(val_str) = rest.strip_prefix('=') {
                    if let Ok(v) = val_str.trim().parse::<f64>() {
                        map.insert(bound.key.to_string(), v);
                    }
                }
            }
        }
    }
    map
}

/// Load the optimizer system prompt from file.
fn load_optimizer_prompt() -> String {
    for path in &["data/nemo_optimizer_prompt.txt", "data/nemo9b_optimizer_prompt.txt"] {
        if let Ok(s) = fs::read_to_string(path) {
            if !s.trim().is_empty() { return s; }
        }
    }
    "You are NEMOTRON PRIME, a trading parameter optimizer. Analyze performance and propose JSON parameter changes.".to_string()
}

/// Build the user message for the optimizer LLM call.
fn build_user_prompt(
    perf: &crate::journal::PerformanceSnapshot,
    current_params: &HashMap<String, f64>,
    journal: &Journal,
) -> String {
    let mut params_str = String::new();
    let mut keys: Vec<&String> = current_params.keys().collect();
    keys.sort();
    for k in keys {
        params_str.push_str(&format!("{}={:.4}\n", k, current_params[k]));
    }

    let errors = fs::read_to_string(ERROR_FLAGS_FILE).unwrap_or_default();
    let error_section = if errors.is_empty() {
        String::new()
    } else {
        format!("\nRECENT_ERRORS:\n{}\n", errors.lines().take(10).collect::<Vec<_>>().join("\n"))
    };

    let recent: String = perf.recent_trades.iter().take(10).map(|t| {
        format!("{} {} pnl={:.2}% ${:.2} hold={:.0}min exit={}\n",
            t.symbol, t.result, t.pnl_percent, t.pnl_usd, t.hold_minutes, t.exit_reason)
    }).collect();

    let cancel_section = {
        let s = journal.cancel_summary();
        if s.is_empty() { String::new() } else { format!("\nCANCELED_ORDERS:\n{}\n", s) }
    };

    // Include AI model decision summaries — 49B sees what each model is doing
    let model_summary = crate::engine::decision_log::build_model_summary();

    format!(
        "PERFORMANCE (last {} trades):\n\
         win_rate={:.1}% wins={} losses={} avg_pnl={:.2}% pnl=${:.2} avg_hold={:.0}min\n\n\
         CURRENT_PARAMS:\n{}\n\
         RECENT_TRADES:\n{}\n{}{}\n{}",
        perf.trade_count,
        perf.win_rate * 100.0, perf.win_count, perf.loss_count,
        perf.avg_pnl_pct * 100.0, perf.total_pnl_usd, perf.avg_hold_minutes,
        params_str,
        recent,
        error_section,
        cancel_section,
        model_summary,
    )
}

/// Clear the error flags file after a successful optimizer cycle.
pub fn clear_error_flags() {
    let _ = fs::write(ERROR_FLAGS_FILE, "");
}

/// Parse LLM response: extract JSON tool call and validate against whitelist.
fn parse_and_validate(
    llm_response: &str,
    current_params: &HashMap<String, f64>,
    journal: &Journal,
    strict: bool,
) -> (Vec<ParamChange>, Vec<RejectedChange>, Option<String>, Option<String>) {
    let mut changes = Vec::new();
    let mut rejected = Vec::new();

    // Extract JSON from response (handle markdown fences, reasoning blocks)
    let json_str = extract_json(llm_response);
    let json_str = match json_str {
        Some(s) => s,
        None => {
            if strict {
                return (changes, rejected, None, Some("no_json".into()));
            }
            tracing::info!("[AI-OPT] No JSON found in response, treating as no_change");
            return (changes, rejected, None, None);
        }
    };

    // Parse JSON
    let parsed: serde_json::Value = match serde_json::from_str(&json_str) {
        Ok(v) => v,
        Err(e) => {
            if strict {
                return (changes, rejected, None, Some("json_parse".into()));
            }
            tracing::warn!("[AI-OPT] JSON parse failed: {e} -- raw: {}", &json_str[..json_str.len().min(200)]);
            return (changes, rejected, None, None);
        }
    };

    // Check tool type
    let tool = parsed.get("tool").and_then(|v| v.as_str()).unwrap_or("");
    if tool == "no_change" {
        let reason = parsed.pointer("/args/reason").and_then(|v| v.as_str()).unwrap_or("no reason");
        tracing::info!("[AI-OPT] AI says no_change: {}", reason);
        return (changes, rejected, None, None);
    }

    if tool == "diagnostics" {
        let focus = parsed.pointer("/args/focus").and_then(|v| v.as_str()).unwrap_or("all");
        tracing::info!("[AI-OPT] AI requested diagnostics (focus={})", focus);
        // Log current params as diagnostics dump
        for bound in PARAM_WHITELIST {
            if let Some(val) = current_params.get(bound.key) {
                tracing::info!(
                    "[AI-DIAG] {}={} (bounds: {}-{})",
                    bound.key, format_value(*val), format_value(bound.min), format_value(bound.max),
                );
            }
        }
        return (changes, rejected, None, None);
    }

    // Calculator tool -- evaluate expression and signal re-prompt
    if tool == "calculate" {
        let expression = parsed.pointer("/args/expression")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let label = parsed.pointer("/args/label")
            .and_then(|v| v.as_str())
            .unwrap_or("result");
        if expression.is_empty() {
            tracing::warn!("[AI-OPT] Empty calculator expression");
            return (changes, rejected, None, None);
        }
        match eval_math(expression) {
            Ok(result) => {
                let result_text = format!(
                    "\n[CALCULATOR] {} = {:.6} ({})",
                    expression, result, label
                );
                tracing::info!("[AI-OPT] Calculator: {} = {:.6} ({})", expression, result, label);
                return (changes, rejected, Some(result_text), None);
            }
            Err(e) => {
                let result_text = format!(
                    "\n[CALCULATOR] ERROR: '{}' -- {}",
                    expression, e
                );
                tracing::warn!("[AI-OPT] Calculator error: '{}' -- {}", expression, e);
                return (changes, rejected, Some(result_text), None);
            }
        }
    }

    // Journal query tool -- look up trade history and signal re-prompt
    if tool == "journal_query" {
        let query_type = parsed.pointer("/args/type")
            .and_then(|v| v.as_str()).unwrap_or("trades");
        let symbol = parsed.pointer("/args/symbol")
            .and_then(|v| v.as_str());
        let result_filter = parsed.pointer("/args/result")
            .and_then(|v| v.as_str());
        let limit = parsed.pointer("/args/limit")
            .and_then(|v| v.as_u64()).unwrap_or(10) as usize;
        let label = parsed.pointer("/args/label")
            .and_then(|v| v.as_str()).unwrap_or("query");

        let result_text = if query_type == "cancellations" {
            let cancels = journal.recent_cancellations(limit.min(20));
            if cancels.is_empty() {
                format!("\n[JOURNAL] {} -- no canceled orders this session", label)
            } else {
                let mut out = format!("\n[JOURNAL] {} ({} canceled orders):\n", label, cancels.len());
                for c in &cancels {
                    out.push_str(&format!(
                        "  {} ${:.2} limit={:.6} age={:.0}s reason={}\n",
                        c.symbol, c.reserved_usd, c.limit_price,
                        c.age_sec, c.reason
                    ));
                }
                out
            }
        } else if query_type == "stats" {
            let sym = symbol.unwrap_or("ALL");
            if sym.eq_ignore_ascii_case("ALL") {
                format!("\n[JOURNAL] All coin stats:\n{}", journal.all_coin_stats())
            } else {
                format!("\n[JOURNAL] {}", journal.coin_stats(sym))
            }
        } else {
            let trades = journal.query_trades(symbol, result_filter, limit.min(20));
            if trades.is_empty() {
                format!("\n[JOURNAL] {} -- no matching trades found", label)
            } else {
                let mut out = format!("\n[JOURNAL] {} ({} trades):\n", label, trades.len());
                for t in &trades {
                    out.push_str(&format!(
                        "  {} {} {:.2}% ${:.2} hold={:.0}min exit={}\n",
                        t.symbol, t.result,
                        t.pnl_percent * 100.0, t.pnl_usd,
                        t.hold_minutes, t.exit_reason
                    ));
                }
                out
            }
        };

        tracing::info!("[AI-OPT] Journal query ({}): {}", label,
            result_text.lines().count().saturating_sub(1));
        return (changes, rejected, Some(result_text), None);
    }

    // ── update_ai_prompt: 49B can rewrite any model's prompt file ────────────
    if tool == "update_ai_prompt" {
        let model = parsed.pointer("/args/model").and_then(|v| v.as_str()).unwrap_or("");
        let content = parsed.pointer("/args/content").and_then(|v| v.as_str()).unwrap_or("");
        let reason = parsed.pointer("/args/reason").and_then(|v| v.as_str()).unwrap_or("49B update");

        let path = match model {
            "ai1" | "entry"    => Some("data/nemo9b_entry_prompt.txt"),
            "ai2" | "exit"     => Some("data/nemo9b_exit_prompt.txt"),
            "ai3" | "phi3"     => Some("data/npu_scan_prompt.txt"),
            "meme"             => Some("data/nemo_meme_entry_prompt.txt"),
            _ => None,
        };

        if let Some(p) = path {
            if !content.is_empty() && content.len() > 50 {
                if let Err(e) = std::fs::write(p, content) {
                    tracing::warn!("[AI-OPT] Failed to update prompt {}: {}", p, e);
                } else {
                    tracing::info!("[AI-OPT] Updated {} prompt ({} chars) — {}", model, content.len(), reason);
                    flag_error("prompt_update", &format!("49B updated {} prompt: {}", model, reason));
                }
            } else {
                tracing::warn!("[AI-OPT] Prompt update rejected: too short or empty ({} chars)", content.len());
            }
        } else {
            tracing::warn!("[AI-OPT] Unknown model '{}' for prompt update", model);
        }
        return (changes, rejected, None, None);
    }

    if tool != "propose_param_change" {
        if strict {
            return (changes, rejected, None, Some(format!("unknown_tool:{tool}")));
        }
        tracing::warn!("[AI-OPT] Unknown tool '{}' , ignoring", tool);
        return (changes, rejected, None, None);
    }

    let args = match parsed.get("args") {
        Some(a) => a,
        None => {
            if strict {
                return (changes, rejected, None, Some("missing_args".into()));
            }
            return (changes, rejected, None, None);
        }
    };

    let key = args.get("param").and_then(|v| v.as_str()).unwrap_or("").to_uppercase();
    let to_val = args.get("to").and_then(|v| v.as_f64()).unwrap_or(0.0);
    let reason = args.get("reason").and_then(|v| v.as_str()).unwrap_or("no_reason").to_string();
    let expected = args.get("expected_metric").and_then(|v| v.as_str()).unwrap_or("").to_string();
    let risk = args.get("risk_estimate").and_then(|v| v.as_str()).unwrap_or("unknown").to_string();

    // Whitelist check (supports per-coin variants like LIMIT_OFFSET_PCT_SOL)
    let bound = match PARAM_WHITELIST.iter().find(|b| b.key == key) {
        Some(b) => b,
        None => {
            // Check for per-coin param variants (e.g., LIMIT_OFFSET_PCT_BTC uses LIMIT_OFFSET_PCT bounds)
            let base_key = if let Some(pos) = key.rfind('_') {
                Some(&key[..pos])
            } else {
                None
            };
            match base_key.and_then(|bk| PARAM_WHITELIST.iter().find(|b| b.key == bk)) {
                Some(b) => b,
                None => {
                    rejected.push(RejectedChange {
                        key: key.clone(),
                        attempted_value: format!("{}", to_val),
                        reject_reason: "not_in_whitelist".to_string(),
                    });
                    return (changes, rejected, None, None);
                }
            }
        }
    };

    // Clamp to bounds
    if !to_val.is_finite() {
        rejected.push(RejectedChange {
            key,
            attempted_value: format!("{}", to_val),
            reject_reason: "invalid_value".to_string(),
        });
        return (changes, rejected, None, None);
    }

    let clamped = to_val.clamp(bound.min, bound.max);
    let was_clamped = (clamped - to_val).abs() > 1e-9;
    let old_value = current_params.get(key.as_str()).copied().unwrap_or(0.0);

    if (clamped - old_value).abs() < 1e-9 {
        tracing::info!("[AI-OPT] {} unchanged ({:.4}), skipping", key, old_value);
        return (changes, rejected, None, None);
    }

    if changes.len() >= MAX_CHANGES_PER_CYCLE {
        rejected.push(RejectedChange {
            key,
            attempted_value: format!("{}", to_val),
            reject_reason: "max_changes_exceeded".to_string(),
        });
        return (changes, rejected, None, None);
    }

    changes.push(ParamChange {
        key,
        old_value,
        new_value: clamped,
        clamped: was_clamped,
        reason,
        risk_estimate: risk,
        expected_metric: expected,
    });

    (changes, rejected, None, None)
}

/// Extract first JSON object from LLM text (handles markdown fences, reasoning).
fn extract_json(text: &str) -> Option<String> {
    // Strip reasoning content (some models use <think> blocks)
    let text = if let Some(pos) = text.find("</think>") {
        &text[pos + 8..]
    } else {
        text
    };

    // Try to find JSON object
    let trimmed = text.trim();

    // Handle markdown code fences
    let content = if trimmed.contains("```json") {
        let start = trimmed.find("```json")? + 7;
        let end = trimmed[start..].find("```").map(|i| start + i)?;
        &trimmed[start..end]
    } else if trimmed.contains("```") {
        let start = trimmed.find("```")? + 3;
        let end = trimmed[start..].find("```").map(|i| start + i)?;
        &trimmed[start..end]
    } else {
        trimmed
    };

    // Find first { ... } pair
    let content = content.trim();
    let start = content.find('{')?;
    let mut depth = 0;
    let mut end = start;
    for (i, ch) in content[start..].char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    end = start + i;
                    break;
                }
            }
            _ => {}
        }
    }
    if depth == 0 && end > start {
        Some(content[start..=end].to_string())
    } else {
        None
    }
}

// ── Simple math evaluator for Nemo's calculator tool ──
// Recursive descent parser: +, -, *, /, parentheses, unary minus, decimals.
fn eval_math(expr: &str) -> Result<f64, String> {
    let tokens: Vec<char> = expr.chars().filter(|c| !c.is_whitespace()).collect();
    let mut pos = 0;
    let result = eval_expr(&tokens, &mut pos)?;
    if pos < tokens.len() {
        return Err(format!("unexpected char '{}' at position {}", tokens[pos], pos));
    }
    if !result.is_finite() {
        return Err("result is infinite or NaN".into());
    }
    Ok(result)
}

fn eval_expr(tokens: &[char], pos: &mut usize) -> Result<f64, String> {
    let mut left = eval_term(tokens, pos)?;
    while *pos < tokens.len() && (tokens[*pos] == '+' || tokens[*pos] == '-') {
        let op = tokens[*pos];
        *pos += 1;
        let right = eval_term(tokens, pos)?;
        left = if op == '+' { left + right } else { left - right };
    }
    Ok(left)
}

fn eval_term(tokens: &[char], pos: &mut usize) -> Result<f64, String> {
    let mut left = eval_unary(tokens, pos)?;
    while *pos < tokens.len() && (tokens[*pos] == '*' || tokens[*pos] == '/') {
        let op = tokens[*pos];
        *pos += 1;
        let right = eval_unary(tokens, pos)?;
        if op == '/' && right == 0.0 {
            return Err("division by zero".into());
        }
        left = if op == '*' { left * right } else { left / right };
    }
    Ok(left)
}

fn eval_unary(tokens: &[char], pos: &mut usize) -> Result<f64, String> {
    if *pos < tokens.len() && tokens[*pos] == '-' {
        *pos += 1;
        let val = eval_primary(tokens, pos)?;
        return Ok(-val);
    }
    eval_primary(tokens, pos)
}

fn eval_primary(tokens: &[char], pos: &mut usize) -> Result<f64, String> {
    if *pos >= tokens.len() {
        return Err("unexpected end of expression".into());
    }
    if tokens[*pos] == '(' {
        *pos += 1;
        let val = eval_expr(tokens, pos)?;
        if *pos >= tokens.len() || tokens[*pos] != ')' {
            return Err("missing closing parenthesis".into());
        }
        *pos += 1;
        return Ok(val);
    }
    // Parse number
    let start = *pos;
    while *pos < tokens.len() && (tokens[*pos].is_ascii_digit() || tokens[*pos] == '.') {
        *pos += 1;
    }
    if *pos == start {
        return Err(format!("expected number at position {}", start));
    }
    let num_str: String = tokens[start..*pos].iter().collect();
    num_str.parse::<f64>().map_err(|e| format!("invalid number '{}': {}", num_str, e))
}

/// Surgically update .env file — only change matched keys, preserve everything else.
fn write_env_changes(changes: &[ParamChange]) -> Result<(), String> {
    let env_path = std::path::Path::new(".env");
    let content = fs::read_to_string(env_path)
        .map_err(|e| format!("failed to read .env: {e}"))?;

    let change_map: HashMap<&str, f64> = changes.iter()
        .map(|c| (c.key.as_str(), c.new_value))
        .collect();

    let mut output = String::with_capacity(content.len());
    let mut applied: std::collections::HashSet<&str> = std::collections::HashSet::new();

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            output.push_str(line);
            output.push('\n');
            continue;
        }

        if let Some(eq_pos) = trimmed.find('=') {
            let key = trimmed[..eq_pos].trim();
            if let Some(&new_val) = change_map.get(key) {
                output.push_str(&format!("{}={}\n", key, format_value(new_val)));
                applied.insert(key);
                continue;
            }
        }

        output.push_str(line);
        output.push('\n');
    }

    // Write directly (Windows doesn't support atomic rename well with open files)
    fs::write(env_path, &output)
        .map_err(|e| format!("failed to write .env: {e}"))?;

    Ok(())
}

/// Smart value formatting: integers stay integers, decimals get appropriate precision.
fn format_value(val: f64) -> String {
    if (val - val.round()).abs() < 1e-9 && val.abs() < 1e9 {
        format!("{}", val as i64)
    } else {
        let s = format!("{:.4}", val);
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

/// Public API: apply a single param change to .env — used by entry AI tool calls.
/// Validates against whitelist bounds before writing.
pub fn apply_single_param_change(key: &str, new_value: f64, reason: &str) -> Result<(), String> {
    // Check whitelist
    let bound = PARAM_WHITELIST.iter().find(|b| b.key == key)
        .ok_or_else(|| format!("{} not in optimizer whitelist", key))?;

    if new_value < bound.min || new_value > bound.max {
        return Err(format!("{} = {:.3} out of bounds [{:.3}, {:.3}]",
            key, new_value, bound.min, bound.max));
    }

    let old_value = std::env::var(key)
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(new_value);

    let change = ParamChange {
        key:             key.to_string(),
        old_value,
        new_value,
        clamped:         false,
        reason:          reason.to_string(),
        risk_estimate:   "tool_call".to_string(),
        expected_metric: "manual".to_string(),
    };

    write_env_changes(&[change])
}

fn write_audit_log(entry: AuditEntry) {
    let path = std::path::Path::new("data/nemo_optimizer.jsonl");
    if let Ok(line) = serde_json::to_string(&entry) {
        if let Ok(mut f) = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
        {
            let _ = writeln!(f, "{}", line);
        } else {
            tracing::warn!("[AI-OPT] Failed to write audit log");
        }
    }
}
