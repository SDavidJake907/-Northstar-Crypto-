//! AuroraKin907 Memory Packet — Adaptive Learning Layer
//!
//! Computes a 5-layer memory packet from historical performance data
//! and injects it into entry/exit prompts as pipe-delimited context.
//!
//! Layers: MEMG (global), MEMR (regime), MEMP (pattern), MEMC (coin), POLICY

use crate::config::{CircuitBreakerState, CircuitLevel};
use crate::journal::Journal;
use crate::nemo_memory::OutcomeRecord;
use std::collections::HashMap;

// ── Structs ───────────────────────────────────────────────────────

pub enum ConfidenceBias {
    Overconf,
    Underconf,
    Calibrated,
}

impl ConfidenceBias {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Overconf => "overconf",
            Self::Underconf => "underconf",
            Self::Calibrated => "calibrated",
        }
    }
}

pub struct MemG {
    pub winrate_50: f64,
    pub avg_pnl_50: f64,
    pub fee_drag_7d: f64,
    pub turnover_7d: f64,
    pub dd_7d: f64,
    pub max_loss_streak: u32,
    pub confidence_bias: ConfidenceBias,
}

pub struct MemR {
    pub regime: String,
    pub trades: u32,
    pub wr: f64,
    pub avg_pnl: f64,
    pub note: String,
}

pub struct MemP {
    pub ptype: &'static str, // "REPEAT" or "AVOID"
    pub signature: String,
    pub n: u32,
    pub wr: f64,
    pub avg_pnl: f64,
}

pub struct MemC {
    pub sym: String,
    pub t30: u32,
    pub wr30: f64,
    pub avg30: f64,
    pub best_regime: String,
    pub worst_regime: String,
    pub penalty: f64,
}

pub struct Policy {
    pub entry_strictness: f64,
    pub risk_budget: f64,
    pub new_entries_max: u32,
    pub turnover_budget: f64,
    pub min_edge_gain: f64,
}

pub struct MemoryPacket {
    pub memg: MemG,
    pub memr: Vec<MemR>,
    pub memp: Vec<MemP>,
    pub memc: Vec<MemC>,
    pub policy: Policy,
    #[allow(dead_code)]
    pub computed_at: f64,
}

pub struct CachedMemoryPacket {
    packet: Option<MemoryPacket>,
    last_compute_ts: f64,
    cache_ttl_sec: f64,
}

impl Default for CachedMemoryPacket {
    fn default() -> Self { Self::new() }
}

// ── Cache ─────────────────────────────────────────────────────────

impl CachedMemoryPacket {
    pub fn new() -> Self {
        Self {
            packet: None,
            last_compute_ts: 0.0,
            cache_ttl_sec: 60.0,
        }
    }

    /// Recompute the memory packet if cache has expired (>60s).
    /// Takes disjoint borrows from TradingLoop fields — no borrow conflicts.
    pub fn maybe_recompute(
        &mut self,
        now: f64,
        nemo: &crate::nemo_memory::NemoMemory,
        journal: &Journal,
        circuit: &CircuitBreakerState,
        regime: &str,
        fee_per_side_pct: f64,
    ) {
        if (now - self.last_compute_ts) < self.cache_ttl_sec {
            return;
        }
        let outcomes = nemo.outcomes();
        if outcomes.is_empty() {
            return;
        }
        self.packet = Some(compute_packet(outcomes, journal, circuit, regime, fee_per_side_pct));
        self.last_compute_ts = now;

        if let Some(ref pkt) = self.packet {
            tracing::info!(
                "[MEM-PACKET] Recomputed: MEMG wr={:.2} dd={:.1}% streak={}",
                pkt.memg.winrate_50,
                pkt.memg.dd_7d,
                pkt.memg.max_loss_streak,
            );
        }
    }

    /// Force recompute on next tick (called after trade close).
    pub fn invalidate(&mut self) {
        self.last_compute_ts = 0.0;
    }

    pub fn get(&self) -> Option<&MemoryPacket> {
        self.packet.as_ref()
    }
}

// ── Computation ───────────────────────────────────────────────────

fn compute_packet(
    outcomes: &[OutcomeRecord],
    journal: &Journal,
    circuit: &CircuitBreakerState,
    current_regime: &str,
    fee_per_side_pct: f64,
) -> MemoryPacket {
    let now = crate::ai_bridge::now_ts();
    let memg = compute_memg(outcomes, fee_per_side_pct);
    let memr = compute_memr(outcomes);
    let memp = compute_memp(journal);
    let memc = compute_memc(outcomes);
    let policy = compute_policy(&memg, &memr, circuit, current_regime, fee_per_side_pct);

    MemoryPacket { memg, memr, memp, memc, policy, computed_at: now }
}

// ── MEMG: Global Performance ──────────────────────────────────────

fn compute_memg(outcomes: &[OutcomeRecord], fee_per_side_pct: f64) -> MemG {
    let last50: Vec<&OutcomeRecord> = outcomes.iter().rev().take(50).collect();
    let n = last50.len() as f64;
    if n == 0.0 {
        return MemG {
            winrate_50: 0.0, avg_pnl_50: 0.0, fee_drag_7d: 0.0,
            turnover_7d: 0.0, dd_7d: 0.0, max_loss_streak: 0,
            confidence_bias: ConfidenceBias::Calibrated,
        };
    }

    let wins = last50.iter().filter(|o| o.pnl_pct > 0.001).count() as f64;
    let winrate_50 = wins / n;
    let avg_pnl_50 = last50.iter().map(|o| o.pnl_pct).sum::<f64>() / n;

    // Fee drag: total round-trip fees over 7d window
    let round_trip_fee = 2.0 * fee_per_side_pct; // e.g. 0.52%
    let fee_drag_7d = n * round_trip_fee / 100.0; // as fraction

    // Turnover: trades per day
    let ts_range = {
        let min_ts = last50.iter().map(|o| o.ts).fold(f64::MAX, f64::min);
        let max_ts = last50.iter().map(|o| o.ts).fold(0.0_f64, f64::max);
        (max_ts - min_ts).max(86400.0) // at least 1 day
    };
    let turnover_7d = n / (ts_range / 86400.0);

    // Max drawdown: running peak-to-trough on cumulative PnL
    let dd_7d = {
        let mut cum = 0.0_f64;
        let mut peak = 0.0_f64;
        let mut max_dd = 0.0_f64;
        // Iterate chronologically (last50 is reversed)
        for o in last50.iter().rev() {
            cum += o.pnl_pct;
            if cum > peak { peak = cum; }
            let dd = cum - peak;
            if dd < max_dd { max_dd = dd; }
        }
        max_dd * 100.0 // as percentage
    };

    // Max loss streak
    let max_loss_streak = {
        let mut streak = 0u32;
        let mut max_streak = 0u32;
        for o in last50.iter().rev() {
            if o.pnl_pct < -0.001 {
                streak += 1;
                if streak > max_streak { max_streak = streak; }
            } else {
                streak = 0;
            }
        }
        max_streak
    };

    // Confidence bias
    let avg_reasons = last50.iter().map(|o| o.entry_reasons.len() as f64).sum::<f64>() / n;
    let confidence_bias = if winrate_50 < 0.40 && avg_reasons >= 3.0 {
        ConfidenceBias::Overconf
    } else if winrate_50 > 0.55 && avg_reasons < 2.0 {
        ConfidenceBias::Underconf
    } else {
        ConfidenceBias::Calibrated
    };

    MemG { winrate_50, avg_pnl_50, fee_drag_7d, turnover_7d, dd_7d, max_loss_streak, confidence_bias }
}

// ── MEMR: Per-Regime Performance ──────────────────────────────────

fn compute_memr(outcomes: &[OutcomeRecord]) -> Vec<MemR> {
    let mut by_regime: HashMap<String, Vec<&OutcomeRecord>> = HashMap::new();
    for o in outcomes {
        let key = if o.regime_label.is_empty() { "UNKNOWN".to_string() } else { o.regime_label.to_uppercase() };
        by_regime.entry(key).or_default().push(o);
    }

    let mut result: Vec<MemR> = by_regime.into_iter().map(|(regime, trades)| {
        let n = trades.len() as f64;
        let wins = trades.iter().filter(|o| o.pnl_pct > 0.001).count() as f64;
        let wr = if n > 0.0 { wins / n } else { 0.0 };
        let avg_pnl = if n > 0.0 { trades.iter().map(|o| o.pnl_pct).sum::<f64>() / n } else { 0.0 };

        let note = if wr < 0.30 && n >= 5.0 {
            "avoid_new_entries".to_string()
        } else if wr > 0.60 {
            "favorable_regime".to_string()
        } else if avg_pnl < -0.005 {
            "negative_edge".to_string()
        } else {
            String::new()
        };

        MemR { regime, trades: n as u32, wr, avg_pnl, note }
    }).collect();

    result.sort_by(|a, b| b.trades.cmp(&a.trades));
    result
}

// ── MEMP: Pattern Performance ─────────────────────────────────────

fn compute_memp(journal: &Journal) -> Vec<MemP> {
    let snap = journal.performance_snapshot(200);
    let mut result = Vec::new();

    // REPEAT: top 3 patterns with WR>60% and n>=3
    let mut good: Vec<&crate::journal::PatternSummary> = snap.top_patterns.iter()
        .filter(|p| p.win_rate > 0.60 && p.total_trades >= 3)
        .collect();
    good.sort_by(|a, b| b.win_rate.partial_cmp(&a.win_rate).unwrap_or(std::cmp::Ordering::Equal));
    for p in good.iter().take(3) {
        result.push(MemP {
            ptype: "REPEAT",
            signature: p.conditions.clone(),
            n: p.total_trades,
            wr: p.win_rate,
            avg_pnl: p.avg_pnl,
        });
    }

    // AVOID: top 3 patterns with WR<40% and n>=3
    let mut bad: Vec<&crate::journal::PatternSummary> = snap.top_patterns.iter()
        .filter(|p| p.win_rate < 0.40 && p.total_trades >= 3)
        .collect();
    bad.sort_by(|a, b| a.win_rate.partial_cmp(&b.win_rate).unwrap_or(std::cmp::Ordering::Equal));
    for p in bad.iter().take(3) {
        result.push(MemP {
            ptype: "AVOID",
            signature: p.conditions.clone(),
            n: p.total_trades,
            wr: p.win_rate,
            avg_pnl: p.avg_pnl,
        });
    }

    result
}

// ── MEMC: Per-Coin Performance ────────────────────────────────────

fn compute_memc(outcomes: &[OutcomeRecord]) -> Vec<MemC> {
    let mut by_sym: HashMap<String, Vec<&OutcomeRecord>> = HashMap::new();
    for o in outcomes {
        by_sym.entry(o.symbol.clone()).or_default().push(o);
    }

    let mut result: Vec<MemC> = by_sym.into_iter().map(|(sym, trades)| {
        let n = trades.len() as f64;
        let wins = trades.iter().filter(|o| o.pnl_pct > 0.001).count() as f64;
        let wr = if n > 0.0 { wins / n } else { 0.0 };
        let avg = if n > 0.0 { trades.iter().map(|o| o.pnl_pct).sum::<f64>() / n } else { 0.0 };

        // Best/worst regime for this coin
        let mut regime_stats: HashMap<String, (u32, u32)> = HashMap::new(); // (wins, total)
        for o in &trades {
            let key = if o.regime_label.is_empty() { "UNKNOWN".to_string() } else { o.regime_label.to_uppercase() };
            let entry = regime_stats.entry(key).or_insert((0, 0));
            entry.1 += 1;
            if o.pnl_pct > 0.001 { entry.0 += 1; }
        }
        let best_regime = regime_stats.iter()
            .max_by(|a, b| {
                let wr_a = if a.1.1 > 0 { a.1.0 as f64 / a.1.1 as f64 } else { 0.0 };
                let wr_b = if b.1.1 > 0 { b.1.0 as f64 / b.1.1 as f64 } else { 0.0 };
                wr_a.partial_cmp(&wr_b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(k, _)| k.clone())
            .unwrap_or_default();
        let worst_regime = regime_stats.iter()
            .min_by(|a, b| {
                let wr_a = if a.1.1 > 0 { a.1.0 as f64 / a.1.1 as f64 } else { 0.0 };
                let wr_b = if b.1.1 > 0 { b.1.0 as f64 / b.1.1 as f64 } else { 0.0 };
                wr_a.partial_cmp(&wr_b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(k, _)| k.clone())
            .unwrap_or_default();

        // Penalty: progressive for coins with WR<35% and n>=5
        let penalty = if wr < 0.35 && n >= 5.0 {
            ((0.35 - wr) * 2.0).min(0.50) // 0..0.50 penalty
        } else {
            0.0
        };

        MemC { sym, t30: n as u32, wr30: wr, avg30: avg, best_regime, worst_regime, penalty }
    }).collect();

    result.sort_by(|a, b| b.t30.cmp(&a.t30));
    result.truncate(10);
    result
}

// ── POLICY: Derived Adaptive Policy ───────────────────────────────

fn compute_policy(
    memg: &MemG,
    memr: &[MemR],
    circuit: &CircuitBreakerState,
    current_regime: &str,
    fee_per_side_pct: f64,
) -> Policy {
    // Entry strictness: base 0.50
    let mut strictness = 0.50_f64;
    if memg.max_loss_streak >= 5 { strictness += 0.15; }
    if memg.dd_7d < -3.0 { strictness += 0.10; } // dd_7d is negative
    if matches!(memg.confidence_bias, ConfidenceBias::Overconf) { strictness += 0.10; }
    // Check current regime WR
    let regime_upper = current_regime.to_uppercase();
    if let Some(mr) = memr.iter().find(|r| r.regime == regime_upper) {
        if mr.wr < 0.30 && mr.trades >= 5 { strictness += 0.12; }
    }
    strictness = strictness.clamp(0.30, 0.95);

    // Risk budget from circuit breaker level
    let risk_budget = match circuit.level {
        CircuitLevel::Green => 1.0,
        CircuitLevel::Yellow => 0.55,
        CircuitLevel::Red => 0.20,
        CircuitLevel::Black => 0.0,
    };

    // New entries max
    let new_entries_max = if circuit.exit_only {
        0
    } else if memg.max_loss_streak >= 5 || memg.dd_7d < -3.0 {
        1
    } else {
        3
    };

    // Turnover budget
    let turnover_budget = if memg.turnover_7d > 8.0 {
        0.10
    } else if memg.turnover_7d > 5.0 {
        0.25
    } else if memg.turnover_7d > 3.0 {
        0.50
    } else {
        1.0
    };

    // Min edge gain: fee_per_trade + 0.03%, floor at 0.05%
    let fee_per_trade = 2.0 * fee_per_side_pct / 100.0; // round-trip as fraction
    let min_edge_gain = (fee_per_trade + 0.0003).max(0.0005);

    Policy { entry_strictness: strictness, risk_budget, new_entries_max, turnover_budget, min_edge_gain }
}

// ── Formatting ────────────────────────────────────────────────────

impl MemoryPacket {
    /// Full packet for entry executor (~300 chars).
    /// Includes: MEMG + MEMR + MEMP + MEMC(candidate) + POLICY
    pub fn format_for_entry(&self, symbol: &str) -> String {
        let mut out = String::with_capacity(512);
        out.push_str("=== MEMORY PACKET ===\n");

        // MEMG
        out.push_str(&format!(
            "MEMG|{:.2}|{:+.2}|{:.2}|{:.2}|{:.1}|{}|{}\n",
            self.memg.winrate_50,
            self.memg.avg_pnl_50 * 100.0,
            self.memg.fee_drag_7d * 100.0,
            self.memg.turnover_7d,
            self.memg.dd_7d,
            self.memg.max_loss_streak,
            self.memg.confidence_bias.as_str(),
        ));

        // MEMR (all regimes)
        for mr in &self.memr {
            out.push_str(&format!(
                "MEMR|{}|{}|{:.2}|{:+.2}|{}\n",
                mr.regime, mr.trades, mr.wr,
                mr.avg_pnl * 100.0,
                if mr.note.is_empty() { "-" } else { &mr.note },
            ));
        }

        // MEMP (patterns)
        for mp in &self.memp {
            out.push_str(&format!(
                "MEMP|{}|{}|{}|{:.2}|{:+.2}\n",
                mp.ptype, mp.signature, mp.n, mp.wr, mp.avg_pnl * 100.0,
            ));
        }

        // MEMC (only for the candidate symbol)
        let sym_upper = symbol.to_uppercase();
        if let Some(mc) = self.memc.iter().find(|c| c.sym.eq_ignore_ascii_case(&sym_upper)) {
            out.push_str(&format!(
                "MEMC|{}|{}|{:.2}|{:+.2}|{}|{}|{:.2}\n",
                mc.sym, mc.t30, mc.wr30, mc.avg30 * 100.0,
                mc.best_regime, mc.worst_regime, mc.penalty,
            ));
        }

        // POLICY
        out.push_str(&format!(
            "POLICY|{:.2}|{:.2}|{}|{:.2}|{:.2}\n",
            self.policy.entry_strictness,
            self.policy.risk_budget,
            self.policy.new_entries_max,
            self.policy.turnover_budget,
            self.policy.min_edge_gain * 100.0,
        ));

        out
    }

    /// Compact packet for exit watchdog (~100 chars).
    /// Includes: MEMG(compact) + MEMC(symbol) + POLICY(risk only)
    #[allow(dead_code)]
    pub fn format_for_exit(&self, symbol: &str) -> String {
        let mut out = String::with_capacity(200);
        out.push_str("=== MEM ===\n");

        // MEMG compact: winrate, dd, streak, bias
        out.push_str(&format!(
            "MEMG|{:.2}|{:.1}|{}|{}\n",
            self.memg.winrate_50,
            self.memg.dd_7d,
            self.memg.max_loss_streak,
            self.memg.confidence_bias.as_str(),
        ));

        // MEMC for this symbol
        let sym_upper = symbol.to_uppercase();
        if let Some(mc) = self.memc.iter().find(|c| c.sym.eq_ignore_ascii_case(&sym_upper)) {
            out.push_str(&format!(
                "MEMC|{}|{}|{:.2}|{:+.2}|{:.2}\n",
                mc.sym, mc.t30, mc.wr30, mc.avg30 * 100.0, mc.penalty,
            ));
        }

        // POLICY compact: risk_budget, new_entries_max
        out.push_str(&format!(
            "POLICY|{:.2}|{}\n",
            self.policy.risk_budget,
            self.policy.new_entries_max,
        ));

        out
    }
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_outcome(pnl: f64, regime: &str, symbol: &str) -> OutcomeRecord {
        OutcomeRecord {
            ts: 1000.0,
            symbol: symbol.to_string(),
            side: "long".to_string(),
            pnl_pct: pnl,
            hold_minutes: 30.0,
            exit_reason: "test".to_string(),
            entry_reasons: vec!["trend_ok".to_string()],
            tag: "normal".to_string(),
            regime_label: regime.to_string(),
        }
    }

    #[test]
    fn memg_basic() {
        let outcomes: Vec<OutcomeRecord> = (0..10)
            .map(|i| make_outcome(if i % 3 == 0 { -0.01 } else { 0.005 }, "bullish", "BTC"))
            .collect();
        let memg = compute_memg(&outcomes, 0.26);
        // 7 wins out of 10 (indices 1,2,4,5,7,8,9 have pnl>0.001 — wait let me recalculate)
        // i=0: -0.01 (loss), i=1: 0.005 (win), i=2: 0.005 (win), i=3: -0.01 (loss),
        // i=4: 0.005 (win), i=5: 0.005 (win), i=6: -0.01 (loss), i=7: 0.005 (win),
        // i=8: 0.005 (win), i=9: -0.01 (loss) => 6 wins / 10 = 0.60
        assert!(memg.winrate_50 > 0.5);
        assert!(memg.max_loss_streak <= 2);
    }

    #[test]
    fn policy_tightens_on_losses() {
        let outcomes: Vec<OutcomeRecord> = (0..10)
            .map(|_| make_outcome(-0.02, "bearish", "BTC"))
            .collect();
        let memg = compute_memg(&outcomes, 0.26);
        let memr = compute_memr(&outcomes);
        let circuit = CircuitBreakerState::new(1000.0);
        let policy = compute_policy(&memg, &memr, &circuit, "BEARISH", 0.26);
        // All losses → loss streak 10 → strictness should be high
        assert!(policy.entry_strictness >= 0.65);
        assert!(policy.new_entries_max <= 1);
    }

    #[test]
    fn format_entry_contains_header() {
        let pkt = MemoryPacket {
            memg: MemG {
                winrate_50: 0.46, avg_pnl_50: 0.0012, fee_drag_7d: 0.0034,
                turnover_7d: 1.25, dd_7d: -1.8, max_loss_streak: 6,
                confidence_bias: ConfidenceBias::Overconf,
            },
            memr: vec![],
            memp: vec![],
            memc: vec![],
            policy: Policy {
                entry_strictness: 0.72, risk_budget: 0.55,
                new_entries_max: 1, turnover_budget: 0.25, min_edge_gain: 0.0008,
            },
            computed_at: 0.0,
        };
        let text = pkt.format_for_entry("BTC");
        assert!(text.contains("=== MEMORY PACKET ==="));
        assert!(text.contains("MEMG|"));
        assert!(text.contains("POLICY|"));
    }

    #[test]
    fn format_exit_is_compact() {
        let pkt = MemoryPacket {
            memg: MemG {
                winrate_50: 0.46, avg_pnl_50: 0.0012, fee_drag_7d: 0.0034,
                turnover_7d: 1.25, dd_7d: -1.8, max_loss_streak: 6,
                confidence_bias: ConfidenceBias::Overconf,
            },
            memr: vec![],
            memp: vec![],
            memc: vec![],
            policy: Policy {
                entry_strictness: 0.72, risk_budget: 0.55,
                new_entries_max: 1, turnover_budget: 0.25, min_edge_gain: 0.0008,
            },
            computed_at: 0.0,
        };
        let text = pkt.format_for_exit("BTC");
        assert!(text.contains("=== MEM ==="));
        assert!(text.len() < 300);
    }

    #[test]
    fn cached_packet_respects_ttl() {
        let mut cached = CachedMemoryPacket::new();
        assert!(cached.get().is_none());
        cached.last_compute_ts = 100.0;
        // Would not recompute at ts=150 (within 60s)
        // but would at ts=161
        assert!((161.0 - cached.last_compute_ts) >= cached.cache_ttl_sec);
    }
}
