//! Unified lane assignment — replaces scattered inline tier/lane logic in tick.rs.
//! Single function: regime × tier × signals → Lane | BLOCKED.
//! Lane is write-once at entry and immutable during hold.

use super::regime_sm::Regime;

// ── Lane enum ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lane { L1, L2, L3, L4 }

impl Lane {
    pub fn as_str(&self) -> &'static str {
        match self {
            Lane::L1 => "L1",
            Lane::L2 => "L2",
            Lane::L3 => "L3",
            Lane::L4 => "L4",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "L1" => Lane::L1,
            "L2" => Lane::L2,
            "L3" => Lane::L3,
            _    => Lane::L4,
        }
    }
}

impl std::fmt::Display for Lane {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// ── Lane result ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum LaneResult {
    Assigned(Lane),
    Blocked(String),
}

impl LaneResult {
    pub fn is_blocked(&self) -> bool { matches!(self, LaneResult::Blocked(_)) }
    pub fn lane(&self) -> Option<Lane> {
        match self { LaneResult::Assigned(l) => Some(*l), _ => None }
    }
    pub fn reason(&self) -> String {
        match self { LaneResult::Blocked(r) => r.clone(), _ => String::new() }
    }
}

// ── Inputs ────────────────────────────────────────────────────────────────────

pub struct LaneInputs<'a> {
    pub symbol:             &'a str,
    pub regime:             Regime,
    pub tier:               &'a str,    // "large" | "mid" | "meme" | "small"
    pub is_behavioral_meme: bool,
    pub mtf_7d:             f64,        // -99.0 = missing
    pub vol_ratio:          f64,
    pub momentum:           f64,
    pub zscore:             f64,
    pub imbalance:          f64,
    pub spread_pct:         f64,        // already defaulted by caller
}

// ── Spread caps per lane (fraction) ──────────────────────────────────────────

pub fn spread_cap(lane: Lane) -> f64 {
    match lane {
        Lane::L1 => 0.0030,
        Lane::L2 => 0.0020,
        Lane::L3 => 0.0025,
        Lane::L4 => 0.0050,
    }
}

// ── Main function ─────────────────────────────────────────────────────────────

/// Deterministic lane assignment.
/// Called AFTER data quality checks, BEFORE fee_gate() and AI call.
pub fn assign_lane(i: &LaneInputs) -> LaneResult {

    // ── G1: Hard regime blocks ────────────────────────────────────────
    // BEARISH: hard block — exit-only mode, no new entries
    if i.regime == Regime::Bearish {
        return block(i.symbol, "regime=BEARISH", "LANE-BLOCK");
    }
    // UNKNOWN: advisory only — treat as SIDEWAYS, AI sees the warm-up label
    // (was hard block — caused missing every move after restart)

    // ── G2: VOLATILE — only L4 with vol + momentum gate ──────────────
    if i.regime == Regime::Volatile {
        if i.vol_ratio >= 3.0 && i.momentum > 0.10 {
            return fee_gate_result(i.symbol, Lane::L4, i.spread_pct);
        }
        let reason = if i.vol_ratio < 3.0 { "volatile_low_vol" } else { "volatile_low_momentum" };
        return block(i.symbol, reason, "LANE-BLOCK");
    }

    // ── G3: Behavioral meme override (large-tier exempt) ─────────────
    if i.is_behavioral_meme {
        if i.tier == "large" {
            tracing::info!(
                "[BEHAVIORAL-MEME] {} large_tier_exempt — skipping L4 upgrade",
                i.symbol
            );
            // fall through to matrix
        } else {
            tracing::info!(
                "[BEHAVIORAL-MEME] {} tier={} upgraded→L4 vol={:.1}x",
                i.symbol, i.tier, i.vol_ratio
            );
            return fee_gate_result(i.symbol, Lane::L4, i.spread_pct);
        }
    }

    // ── G4: L4 vol floor (pre-check before matrix) ───────────────────
    // Applied to meme tier coins even without behavioral signal
    let _is_meme_tier = matches!(i.tier, "meme" | "small");

    // ── G5: Regime × tier matrix ──────────────────────────────────────
    let raw_lane: Lane = match i.regime {

        Regime::Bullish => match i.tier {
            "large" | "mid" => {
                if i.vol_ratio >= 1.2 && i.momentum > 0.0 {
                    Lane::L1
                } else if i.vol_ratio >= 0.5 {
                    Lane::L3
                } else {
                    return block(i.symbol, "bullish_low_vol", "LANE-BLOCK");
                }
            }
            _ => {
                // meme / small
                if i.vol_ratio < 2.0 {
                    return block(i.symbol, "l4_vol_floor", "LANE-BLOCK");
                }
                Lane::L4
            }
        },

        Regime::Sideways => match i.tier {
            "large" | "mid" => {
                let reversion = i.zscore <= -0.90 || i.imbalance >= 0.25;
                if reversion { Lane::L2 } else { Lane::L3 }
            }
            _ => {
                if i.vol_ratio < 2.0 {
                    return block(i.symbol, "l4_vol_floor", "LANE-BLOCK");
                }
                Lane::L4
            }
        },

        // Fallback for UNKNOWN or any unhandled regime — treat as SIDEWAYS advisory
        _ => Lane::L3,
    };

    // ── G6: MTF demotion ──────────────────────────────────────────────
    let lane = match apply_mtf(i.symbol, raw_lane, i.mtf_7d) {
        Ok(l)   => l,
        Err(r)  => return r,
    };

    // ── G7: Fee gate (spread cap) ─────────────────────────────────────
    fee_gate_result(i.symbol, lane, i.spread_pct)
}

// ── MTF demotion ──────────────────────────────────────────────────────────────

fn apply_mtf(symbol: &str, lane: Lane, mtf_7d: f64) -> Result<Lane, LaneResult> {
    // L2 exception: missing data (-99.0) → allow (mean reversion is regime-agnostic)
    if lane == Lane::L2 && mtf_7d == -99.0 {
        return Ok(Lane::L2);
    }

    match lane {
        Lane::L1 => {
            if mtf_7d < -10.0 {
                Err(block(symbol, "mtf_-10_floor: L1 blocked", "LANE-BLOCK"))
            } else if mtf_7d < -5.0 {
                tracing::info!("[LANE-DEMOTE] {} L1→L3 mtf_7d={:.1}%", symbol, mtf_7d);
                Ok(Lane::L3)
            } else {
                Ok(Lane::L1)
            }
        }
        Lane::L3 => {
            if mtf_7d < -10.0 {
                Err(block(symbol, "mtf_-10_floor: L3 blocked", "LANE-BLOCK"))
            } else {
                Ok(Lane::L3)
            }
        }
        Lane::L2 => {
            if mtf_7d < -10.0 {
                Err(block(symbol, "L2_mtf_-10", "LANE-BLOCK"))
            } else {
                Ok(Lane::L2)
            }
        }
        Lane::L4 => Ok(Lane::L4), // no MTF restriction on L4
    }
}

// ── Fee gate (spread cap only — net-edge handled by fee_filter.rs) ────────────

fn fee_gate_result(symbol: &str, lane: Lane, spread_pct: f64) -> LaneResult {
    let cap = spread_cap(lane);
    if spread_pct > cap {
        tracing::info!(
            "[LANE-BLOCK] {} fee_gate: spread={:.4}% > cap={:.4}% lane={}",
            symbol, spread_pct * 100.0, cap * 100.0, lane
        );
        LaneResult::Blocked(format!("fee_gate:spread_{:.3}pct_gt_{:.3}pct", spread_pct * 100.0, cap * 100.0))
    } else {
        tracing::info!(
            "[LANE-ASSIGN] {} lane={} spread={:.4}%",
            symbol, lane, spread_pct * 100.0
        );
        LaneResult::Assigned(lane)
    }
}

fn block(symbol: &str, reason: &str, tag: &str) -> LaneResult {
    tracing::info!("[{}] {} {}", tag, symbol, reason);
    LaneResult::Blocked(reason.to_string())
}
