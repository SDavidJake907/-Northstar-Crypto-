//! Trade journaling — crash-safe JSONL logging + pattern learning.
//!
//! Ported from python_orch/memory/journal.py
//! Features:
//!   - Crash-safe JSONL trade log (one JSON object per line)
//!   - Pattern tracking (win rate, avg P&L per entry condition set)
//!   - Thought logging (daily log files)
//!   - Context builder for AI memory prompts
//!   - Thread-safe (uses Mutex internally)

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

// ── Data Structures ──────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TradeRecord {
    pub symbol: String,
    pub entry_time: f64,
    pub exit_time: f64,
    pub entry_price: f64,
    pub exit_price: f64,
    pub quantity: f64,
    pub pnl_percent: f64,
    pub pnl_usd: f64,
    pub result: String, // "WIN", "LOSS", "BREAKEVEN"
    pub hold_minutes: f64,
    pub entry_reasons: Vec<String>,
    pub exit_reason: String,
    pub entry_context: String,
    pub points: i32,
    pub points_reason: String,
    /// Feature snapshot at entry time (44 f64 values) for GBDT training.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub feature_snapshot: Option<Vec<f64>>,
    /// Multi-timeframe trend alignment at entry (0=none, 1=1d, 2=1d+7d, 3=all)
    #[serde(default)]
    pub trend_alignment: u8,
    /// 7-day trend % at entry
    #[serde(default)]
    pub trend_7d_pct: f64,
    /// 30-day trend % at entry
    #[serde(default)]
    pub trend_30d_pct: f64,
}

impl TradeRecord {
    pub fn classify_result(pnl_pct: f64) -> &'static str {
        if pnl_pct > 0.001 {
            "WIN"
        } else if pnl_pct < -0.001 {
            "LOSS"
        } else {
            "BREAKEVEN"
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Pattern {
    pub conditions: Vec<String>,
    pub win_count: u32,
    pub loss_count: u32,
    pub total_pnl: f64, // accumulated pnl_percent (fraction), NOT USD
    pub last_seen: f64,
}


/// Record of a canceled (unfilled) limit order — visible to Nemo optimizer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CancelRecord {
    pub symbol: String,
    pub limit_price: f64,
    pub reserved_usd: f64,
    pub age_sec: f64,
    pub reason: String,
    pub timestamp: f64,
}

// ── Performance Snapshot (for optimizer) ─────────────────────────

pub struct PerformanceSnapshot {
    pub trade_count: usize,
    pub win_count: usize,
    pub loss_count: usize,
    pub win_rate: f64,
    pub avg_pnl_pct: f64,
    pub total_pnl_usd: f64,
    pub avg_hold_minutes: f64,
    pub recent_trades: Vec<TradeSummary>,
    pub top_patterns: Vec<PatternSummary>,
}

pub struct TradeSummary {
    pub symbol: String,
    pub pnl_percent: f64,
    pub pnl_usd: f64,
    pub result: String,
    pub hold_minutes: f64,
    pub exit_reason: String,
}

pub struct PatternSummary {
    pub conditions: String,
    pub win_rate: f64,
    pub avg_pnl: f64,
    pub total_trades: u32,
}

fn env_usize(key: &str, def: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(def)
}

impl Pattern {
    pub fn win_rate(&self) -> f64 {
        let total = self.win_count + self.loss_count;
        if total == 0 {
            return 0.0;
        }
        self.win_count as f64 / total as f64
    }

    pub fn avg_pnl(&self) -> f64 {
        let total = self.win_count + self.loss_count;
        if total == 0 {
            return 0.0;
        }
        self.total_pnl / total as f64
    }
}

// ── Journal (thread-safe) ────────────────────────────────────────

pub struct Journal {
    base_dir: PathBuf,
    trades: Mutex<Vec<TradeRecord>>,
    patterns: Mutex<HashMap<String, Pattern>>,
    total_points: Mutex<i32>,
    cancellations: Mutex<Vec<CancelRecord>>,
}

impl Journal {
    fn max_trades() -> usize {
        env_usize("JOURNAL_MAX_TRADES", 2000).clamp(100, 100_000)
    }

    fn max_patterns() -> usize {
        env_usize("JOURNAL_MAX_PATTERNS", 1000).clamp(100, 50_000)
    }

    fn trim_trades_locked(trades: &mut Vec<TradeRecord>) {
        let max_trades = Self::max_trades();
        if trades.len() > max_trades {
            let drop_n = trades.len() - max_trades;
            trades.drain(0..drop_n);
        }
    }

    fn trim_patterns_locked(patterns: &mut HashMap<String, Pattern>) {
        let max_patterns = Self::max_patterns();
        let len = patterns.len();
        if len > max_patterns {
            let mut keys: Vec<(String, f64)> = patterns
                .iter()
                .map(|(k, v)| (k.clone(), v.last_seen))
                .collect();
            keys.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
            for (k, _) in keys.into_iter().take(len - max_patterns) {
                patterns.remove(&k);
            }
        }
    }

    /// Create or load journal from a directory.
    pub fn new(base_dir: &str) -> Self {
        let base = PathBuf::from(base_dir);
        let _ = fs::create_dir_all(&base);
        let _ = fs::create_dir_all(base.join("thoughts"));

        let raw_trades = Self::load_trades(&base);
        let patterns = Self::load_patterns(&base);

        // Auto-clean: purge dust sweeps and auto-adopted holdings on startup
        let before = raw_trades.len();
        let trades: Vec<TradeRecord> = raw_trades
            .into_iter()
            .filter(|t| {
                if t.exit_reason.contains("dust_sweep") {
                    return false;
                }
                if t.entry_reasons.iter().any(|r| r.contains("auto_adopted")) {
                    return false;
                }
                if t.entry_context.contains("pre-existing") {
                    return false;
                }
                true
            })
            .collect();
        let purged = before - trades.len();
        if purged > 0 {
            // Rewrite clean journal to disk
            let jsonl_path = base.join("trades.jsonl");
            if let Ok(mut f) = fs::File::create(&jsonl_path) {
                use std::io::Write;
                for t in &trades {
                    if let Ok(line) = serde_json::to_string(t) {
                        let _ = writeln!(f, "{}", line);
                    }
                }
            }
            tracing::info!("[JOURNAL] Auto-clean: purged {} dust/adopted entries", purged);
        }

        let total_points: i32 = trades.iter().map(|t| t.points).sum();
        let mut trades = trades;
        let mut patterns = patterns;
        Self::trim_trades_locked(&mut trades);
        Self::trim_patterns_locked(&mut patterns);

        tracing::info!(
            "[JOURNAL] Loaded {} trades, {} patterns, {} points",
            trades.len(),
            patterns.len(),
            total_points,
        );

        Self {
            base_dir: base,
            trades: Mutex::new(trades),
            patterns: Mutex::new(patterns),
            total_points: Mutex::new(total_points),
            cancellations: Mutex::new(Vec::new()),
        }
    }

    /// Record a completed trade (crash-safe: appends to JSONL + updates JSON).
    pub fn record_trade(&self, trade: TradeRecord) {
        // Append to JSONL (crash-safe — one line per trade)
        let jsonl_path = self.base_dir.join("trades.jsonl");
        if let Ok(line) = serde_json::to_string(&trade) {
            if let Ok(mut f) = fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&jsonl_path)
            {
                let _ = writeln!(f, "{}", line);
            }
        }

        // Update pattern tracker
        if !trade.entry_reasons.is_empty() {
            let key = Self::pattern_key(&trade.entry_reasons);
            let mut patterns = self.patterns.lock().unwrap_or_else(|e| {
            tracing::warn!("[JOURNAL] patterns mutex poisoned; continuing");
            e.into_inner()
        });
            let pattern = patterns.entry(key).or_insert_with(|| Pattern {
                conditions: trade.entry_reasons.clone(),
                win_count: 0,
                loss_count: 0,
                total_pnl: 0.0,
                last_seen: 0.0,
            });

            if trade.pnl_percent > 0.001 {
                pattern.win_count += 1;
            } else if trade.pnl_percent < -0.001 {
                pattern.loss_count += 1;
            }
            // BREAKEVEN (|pnl| <= 0.001) — neither win nor loss
            pattern.total_pnl += trade.pnl_percent;
            pattern.last_seen = trade.exit_time;
            Self::trim_patterns_locked(&mut patterns);
        }

        // Update points
        {
            let mut pts = self.total_points.lock().unwrap_or_else(|e| {
            tracing::warn!("[JOURNAL] points mutex poisoned; continuing");
            e.into_inner()
        });
            *pts += trade.points;
        }

        // Add to in-memory list
        {
            let mut trades = self.trades.lock().unwrap_or_else(|e| {
            tracing::warn!("[JOURNAL] trades mutex poisoned; continuing");
            e.into_inner()
        });
            trades.push(trade);
            Self::trim_trades_locked(&mut trades);
        }

        // Save full state (trades.json + patterns.json)
        self.save_all();
    }

    /// Log a thought/decision to the daily log file.
    pub fn log_thought(&self, symbol: &str, action: &str, confidence: f64, reasoning: &str) {
        let now = now_ts();
        let date = date_str(now);
        let path = self
            .base_dir
            .join("thoughts")
            .join(format!("{date}_thoughts.log"));

        let ts_str = timestamp_str(now);
        let line = format!("[{ts_str}] {symbol} {action} conf={confidence:.0}% — {reasoning}\n");

        if let Ok(mut f) = fs::OpenOptions::new().create(true).append(true).open(&path) {
            let _ = f.write_all(line.as_bytes());
        }

        // Console output for BUY/SELL decisions
        if action == "BUY" || action == "SELL" {
            tracing::info!("[THOUGHT] {symbol} {action} {confidence:.0}% — {reasoning}");
        }
    }

    /// Build context string for AI memory prompts.
    /// Build a structured performance snapshot for the optimizer.
    pub fn performance_snapshot(&self, last_n: usize) -> PerformanceSnapshot {
        let trades = self.trades.lock().unwrap_or_else(|e| e.into_inner());
        let patterns = self.patterns.lock().unwrap_or_else(|e| e.into_inner());

        let recent: Vec<&TradeRecord> = trades.iter().rev().take(last_n).collect();
        let count = recent.len();
        let wins = recent.iter().filter(|t| t.result == "WIN").count();
        let losses = recent.iter().filter(|t| t.result == "LOSS").count();
        let wr = if count > 0 { wins as f64 / count as f64 } else { 0.0 };
        let avg_pnl = if count > 0 {
            recent.iter().map(|t| t.pnl_percent * 100.0).sum::<f64>() / count as f64
        } else { 0.0 };
        let total_pnl = recent.iter().map(|t| t.pnl_usd).sum();
        let avg_hold = if count > 0 {
            recent.iter().map(|t| t.hold_minutes).sum::<f64>() / count as f64
        } else { 0.0 };

        let trade_summaries: Vec<TradeSummary> = recent.iter().map(|t| TradeSummary {
            symbol: t.symbol.clone(),
            pnl_percent: t.pnl_percent * 100.0,
            pnl_usd: t.pnl_usd,
            result: t.result.clone(),
            hold_minutes: t.hold_minutes,
            exit_reason: t.exit_reason.clone(),
        }).collect();

        let mut pat_list: Vec<PatternSummary> = patterns.values()
            .filter(|p| (p.win_count + p.loss_count) >= 3)
            .map(|p| PatternSummary {
                conditions: p.conditions.join("+"),
                win_rate: p.win_rate(),
                avg_pnl: p.avg_pnl(),
                total_trades: p.win_count + p.loss_count,
            })
            .collect();
        pat_list.sort_by(|a, b| b.win_rate.partial_cmp(&a.win_rate).unwrap_or(std::cmp::Ordering::Equal));

        PerformanceSnapshot {
            trade_count: count,
            win_count: wins,
            loss_count: losses,
            win_rate: wr,
            avg_pnl_pct: avg_pnl,
            total_pnl_usd: total_pnl,
            avg_hold_minutes: avg_hold,
            recent_trades: trade_summaries,
            top_patterns: pat_list,
        }
    }

    /// Record a canceled/unfilled limit order (visible to Nemo optimizer).
    pub fn record_cancellation(&self, record: CancelRecord) {
        // Append to JSONL
        let path = self.base_dir.join("cancellations.jsonl");
        if let Ok(line) = serde_json::to_string(&record) {
            if let Ok(mut f) = fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
            {
                let _ = writeln!(f, "{}", line);
            }
        }
        tracing::info!(
            "[JOURNAL] Order canceled: {} ${:.2} limit={:.6} age={:.0}s reason={}",
            record.symbol, record.reserved_usd, record.limit_price,
            record.age_sec, record.reason
        );
        let mut cancels = self.cancellations.lock().unwrap_or_else(|e| e.into_inner());
        cancels.push(record);
        // Keep last 100
        if cancels.len() > 100 {
            let drop_n = cancels.len() - 100;
            cancels.drain(0..drop_n);
        }
    }

    /// Summary of recent cancellations for Nemo optimizer prompt.
    pub fn cancel_summary(&self) -> String {
        let cancels = self.cancellations.lock().unwrap_or_else(|e| e.into_inner());
        if cancels.is_empty() {
            return String::new();
        }
        let total = cancels.len();
        let timeout_count = cancels.iter().filter(|c| c.reason == "timeout").count();
        let funds_count = cancels.iter().filter(|c| c.reason == "insufficient_funds").count();
        let avg_age = cancels.iter().map(|c| c.age_sec).sum::<f64>() / total as f64;

        // Per-coin breakdown
        let mut coin_counts: HashMap<String, usize> = HashMap::new();
        for c in cancels.iter() {
            *coin_counts.entry(c.symbol.clone()).or_insert(0) += 1;
        }
        let mut coins: Vec<_> = coin_counts.into_iter().collect();
        coins.sort_by(|a, b| b.1.cmp(&a.1));
        let coin_str: Vec<String> = coins.iter().take(5)
            .map(|(sym, n)| format!("{}:{}", sym, n))
            .collect();

        format!(
            "ORDER CANCELLATIONS (this session): {} total ({} timeout, {} insufficient_funds) avg_age={:.0}s coins=[{}]",
            total, timeout_count, funds_count, avg_age, coin_str.join(", ")
        )
    }

    /// Recent cancellation records for journal_query tool.
    pub fn recent_cancellations(&self, limit: usize) -> Vec<CancelRecord> {
        let cancels = self.cancellations.lock().unwrap_or_else(|e| e.into_inner());
        cancels.iter().rev().take(limit).cloned().collect()
    }

    /// Query trades filtered by symbol and/or result (for Nemo optimizer tool).
    pub fn query_trades(&self, symbol: Option<&str>, result_filter: Option<&str>, limit: usize) -> Vec<TradeRecord> {
        let trades = self.trades.lock().unwrap_or_else(|e| e.into_inner());
        trades.iter().rev()
            .filter(|t| symbol.map_or(true, |s| t.symbol.eq_ignore_ascii_case(s)))
            .filter(|t| result_filter.map_or(true, |r| t.result.eq_ignore_ascii_case(r)))
            .take(limit)
            .cloned()
            .collect()
    }

    /// Kelly Criterion: optimal fraction of capital to bet on a coin.
    /// f* = W - (1-W)/b where W = win rate, b = avg_win / avg_loss (payoff ratio).
    /// Returns half-Kelly clamped to [0.0, 0.5] for safety. 0.25 = default when no data.
    pub fn kelly_fraction(&self, symbol: &str) -> f64 {
        let trades = self.trades.lock().unwrap_or_else(|e| e.into_inner());
        let coin_trades: Vec<&TradeRecord> = trades.iter()
            .filter(|t| t.symbol.eq_ignore_ascii_case(symbol))
            .collect();
        let total = coin_trades.len();
        if total < 3 {
            return 0.25; // not enough data, use default
        }
        let wins: Vec<f64> = coin_trades.iter()
            .filter(|t| t.result == "WIN")
            .map(|t| t.pnl_percent.abs())
            .collect();
        let losses: Vec<f64> = coin_trades.iter()
            .filter(|t| t.result == "LOSS")
            .map(|t| t.pnl_percent.abs())
            .collect();
        if wins.is_empty() {
            return 0.0; // never won, don't bet
        }
        if losses.is_empty() {
            return 0.5; // never lost, max bet
        }
        let w = wins.len() as f64 / total as f64;
        let avg_win = wins.iter().sum::<f64>() / wins.len() as f64;
        let avg_loss = losses.iter().sum::<f64>() / losses.len() as f64;
        if avg_loss < 1e-9 {
            return 0.5;
        }
        let b = avg_win / avg_loss; // payoff ratio
        let kelly_raw = w - (1.0 - w) / b;
        // Half-Kelly for safety, clamped
        (kelly_raw * 0.5).clamp(0.0, 0.5)
    }

    /// Per-coin stats summary (for Nemo optimizer tool).
    pub fn coin_stats(&self, symbol: &str) -> String {
        let trades = self.trades.lock().unwrap_or_else(|e| e.into_inner());
        let coin_trades: Vec<&TradeRecord> = trades.iter()
            .filter(|t| t.symbol.eq_ignore_ascii_case(symbol))
            .collect();
        let total = coin_trades.len();
        if total == 0 {
            return format!("{}: 0 trades", symbol.to_uppercase());
        }
        let wins = coin_trades.iter().filter(|t| t.result == "WIN").count();
        let losses = coin_trades.iter().filter(|t| t.result == "LOSS").count();
        let avg_pnl = coin_trades.iter().map(|t| t.pnl_percent).sum::<f64>() / total as f64;
        let avg_hold = coin_trades.iter().map(|t| t.hold_minutes).sum::<f64>() / total as f64;
        let total_pnl_usd = coin_trades.iter().map(|t| t.pnl_usd).sum::<f64>();
        // Compute kelly inline to avoid deadlock (we already hold the lock)
        let kelly = {
            let win_trades: Vec<f64> = coin_trades.iter()
                .filter(|t| t.result == "WIN")
                .map(|t| t.pnl_percent.abs())
                .collect();
            let loss_trades: Vec<f64> = coin_trades.iter()
                .filter(|t| t.result == "LOSS")
                .map(|t| t.pnl_percent.abs())
                .collect();
            if total < 3 || win_trades.is_empty() {
                0.25
            } else if loss_trades.is_empty() {
                0.5
            } else {
                let w = win_trades.len() as f64 / total as f64;
                let avg_win = win_trades.iter().sum::<f64>() / win_trades.len() as f64;
                let avg_loss = loss_trades.iter().sum::<f64>() / loss_trades.len() as f64;
                if avg_loss < 1e-9 { 0.5 } else {
                    let b = avg_win / avg_loss;
                    (( w - (1.0 - w) / b ) * 0.5).clamp(0.0, 0.5)
                }
            }
        };
        format!("{}|{}t|{}W/{}L|WR{:.0}%|K{:.2}|avg{:+.2}%|${:+.2}|{:.0}m",
            symbol.to_uppercase(), total, wins, losses,
            wins as f64 / total as f64 * 100.0,
            kelly, avg_pnl * 100.0, total_pnl_usd, avg_hold)
    }

    /// All-coins stats summary (for Nemo optimizer tool).
    pub fn all_coin_stats(&self) -> String {
        let trades = self.trades.lock().unwrap_or_else(|e| e.into_inner());
        let mut symbols: Vec<String> = trades.iter().map(|t| t.symbol.to_uppercase()).collect();
        symbols.sort();
        symbols.dedup();
        drop(trades); // release lock before calling coin_stats

        let mut out = String::new();
        for sym in &symbols {
            out.push_str(&self.coin_stats(sym));
            out.push('\n');
        }
        if out.is_empty() {
            out.push_str("No trades recorded yet.");
        }
        out
    }

    pub fn build_context(&self) -> String {
        let trades = self.trades.lock().unwrap_or_else(|e| {
            tracing::warn!("[JOURNAL] trades mutex poisoned; continuing");
            e.into_inner()
        });
        let patterns = self.patterns.lock().unwrap_or_else(|e| {
            tracing::warn!("[JOURNAL] patterns mutex poisoned; continuing");
            e.into_inner()
        });
        let pts = *self.total_points.lock().unwrap_or_else(|e| {
            tracing::warn!("[JOURNAL] points mutex poisoned; continuing");
            e.into_inner()
        });

        let total = trades.len();
        if total == 0 {
            return String::new();
        }

        let wins = trades.iter().filter(|t| t.result == "WIN").count();
        let losses = trades.iter().filter(|t| t.result == "LOSS").count();
        let wr = if total > 0 {
            wins as f64 / total as f64 * 100.0
        } else {
            0.0
        };
        let total_pnl: f64 = trades.iter().map(|t| t.pnl_usd).sum();

        // Recent lessons (last 5 trades)
        let recent: Vec<&TradeRecord> = trades.iter().rev().take(5).collect();
        let mut lessons = String::new();
        for t in &recent {
            let emoji = if t.result == "WIN" { "+" } else { "-" };
            lessons.push_str(&format!(
                "  {emoji} {}: {:.1}% ({})\n",
                t.symbol,
                t.pnl_percent * 100.0,
                t.exit_reason
            ));
        }

        // Winning patterns
        let mut win_patterns: Vec<&Pattern> = patterns
            .values()
            .filter(|p| p.win_rate() > 0.6 && (p.win_count + p.loss_count) >= 3)
            .collect();
        win_patterns.sort_by(|a, b| {
            b.win_rate()
                .partial_cmp(&a.win_rate())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut win_strs = String::new();
        for p in win_patterns.iter().take(3) {
            win_strs.push_str(&format!(
                "  REPEAT: {} (WR={:.0}%, avg={:+.2}%)\n",
                p.conditions.join("+"),
                p.win_rate() * 100.0,
                p.avg_pnl() * 100.0,
            ));
        }

        // Losing patterns
        let mut loss_patterns: Vec<&Pattern> = patterns
            .values()
            .filter(|p| p.win_rate() < 0.4 && (p.win_count + p.loss_count) >= 3)
            .collect();
        loss_patterns.sort_by(|a, b| {
            a.win_rate()
                .partial_cmp(&b.win_rate())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut loss_strs = String::new();
        for p in loss_patterns.iter().take(3) {
            loss_strs.push_str(&format!(
                "  AVOID: {} (WR={:.0}%, avg={:+.2}%)\n",
                p.conditions.join("+"),
                p.win_rate() * 100.0,
                p.avg_pnl() * 100.0,
            ));
        }

        format!(
            "MEMORY:\n\
             Stats: {total} trades, {wins}W/{losses}L, WR={wr:.0}%, PnL=${total_pnl:.2}, Points={pts}\n\
             Recent:\n{lessons}\
             {win_strs}\
             {loss_strs}\n"
        )
    }

    // ── Internal ─────────────────────────────────────────────────

    fn pattern_key(reasons: &[String]) -> String {
        let mut sorted = reasons.to_vec();
        sorted.sort();
        sorted.join("|")
    }

    fn save_all(&self) {
        // Snapshot under lock, then perform serialization/I/O without holding mutexes.
        let trades_snapshot = if let Ok(trades) = self.trades.lock() {
            Some(trades.clone())
        } else {
            None
        };
        if let Some(trades) = trades_snapshot {
            let path = self.base_dir.join("trades.json");
            if let Ok(data) = serde_json::to_string_pretty(&trades) {
                let _ = fs::write(&path, data);
            }
        }

        let patterns_snapshot = if let Ok(patterns) = self.patterns.lock() {
            Some(patterns.clone())
        } else {
            None
        };
        if let Some(patterns) = patterns_snapshot {
            let path = self.base_dir.join("patterns.json");
            if let Ok(data) = serde_json::to_string_pretty(&patterns) {
                let _ = fs::write(&path, data);
            }
        }
    }

    fn load_trades(base: &Path) -> Vec<TradeRecord> {
        let path = base.join("trades.json");
        if !path.exists() {
            return Vec::new();
        }
        match fs::read_to_string(&path) {
            Ok(data) => serde_json::from_str(&data).unwrap_or_default(),
            Err(_) => Vec::new(),
        }
    }

    fn load_patterns(base: &Path) -> HashMap<String, Pattern> {
        let path = base.join("patterns.json");
        if !path.exists() {
            return HashMap::new();
        }
        match fs::read_to_string(&path) {
            Ok(data) => serde_json::from_str(&data).unwrap_or_default(),
            Err(_) => HashMap::new(),
        }
    }
}

// ── Utilities ────────────────────────────────────────────────────

fn now_ts() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

fn date_str(ts: f64) -> String {
    let secs = ts as i64;
    let mut days = secs / 86400;

    // Compute year accounting for leap years
    let mut y = 1970i64;
    loop {
        let days_in_year = if is_leap(y) { 366 } else { 365 };
        if days < days_in_year {
            break;
        }
        days -= days_in_year;
        y += 1;
    }

    // Compute month and day
    let leap = is_leap(y);
    let mdays: [i64; 12] = [
        31,
        if leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    let mut m = 0usize;
    while m < 12 && days >= mdays[m] {
        days -= mdays[m];
        m += 1;
    }
    let month = (m + 1) as i64;
    let day = days + 1;

    format!("{y:04}-{month:02}-{day:02}")
}

fn is_leap(y: i64) -> bool {
    (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0)
}

fn timestamp_str(ts: f64) -> String {
    let secs = ts as u64;
    let h = (secs % 86400) / 3600;
    let m = (secs % 3600) / 60;
    let s = secs % 60;
    format!("{h:02}:{m:02}:{s:02}")
}

#[cfg(test)]
mod tests {
    use super::{Journal, TradeRecord};
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir(name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        std::env::temp_dir().join(format!("hk_journal_{name}_{nonce}"))
    }

    #[test]
    fn record_trade_persists_json_files() {
        let dir = temp_dir("persist");
        fs::create_dir_all(&dir).expect("mkdir");
        let j = Journal::new(dir.to_str().expect("dir str"));

        j.record_trade(TradeRecord {
            symbol: "BTC".into(),
            entry_time: 1.0,
            exit_time: 2.0,
            entry_price: 100.0,
            exit_price: 101.0,
            quantity: 1.0,
            pnl_percent: 0.01,
            pnl_usd: 1.0,
            result: "WIN".into(),
            hold_minutes: 1.0,
            entry_reasons: vec!["test".into()],
            exit_reason: "tp".into(),
            entry_context: "ctx".into(),
            points: 2,
            points_reason: "win".into(),
            feature_snapshot: None,
        });

        let trades_path = dir.join("trades.json");
        let patterns_path = dir.join("patterns.json");
        let trades_data = fs::read_to_string(&trades_path).expect("trades.json");
        let patterns_data = fs::read_to_string(&patterns_path).expect("patterns.json");

        assert!(trades_data.contains("\"symbol\": \"BTC\""));
        assert!(patterns_data.contains("test"));

        let _ = fs::remove_file(dir.join("trades.jsonl"));
        let _ = fs::remove_file(trades_path);
        let _ = fs::remove_file(patterns_path);
        let _ = fs::remove_dir_all(dir.join("thoughts"));
        let _ = fs::remove_dir_all(dir);
    }
}
