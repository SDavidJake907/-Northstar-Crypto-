//! Unified lane assignment â€” replaces scattered inline tier/lane logic in tick.rs.
//! Single function: regime Ã— tier Ã— signals â†’ Lane | BLOCKED.
//! Lane is write-once at entry and immutable during hold.

use super::regime_sm::Regime;

// â”€â”€ Lane enum â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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

// â”€â”€ Lane result â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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

// â”€â”€ Inputs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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
    pub atr_norm:           f64,        // atr/price — used for dynamic spread cap
}

// â”€â”€ Spread caps per lane (fraction) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Dynamic spread cap: base + k * atr_norm
/// High-ATR coins get more room since their moves are bigger.
/// Formula from operator config — mirrors real Kraken spread behavior.
pub fn spread_cap(lane: Lane, atr_norm: f64) -> f64 {
    let env_f64 = |key: &str, default: f64| -> f64 {
        std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
    };
    let (base, k) = match lane {
        Lane::L1 => (env_f64("LANE_SPREAD_CAP_L1", 0.005), 0.25_f64),  // base 0.5%
        Lane::L2 => (env_f64("LANE_SPREAD_CAP_L2", 0.010), 0.25_f64),  // base 1.0%
        Lane::L3 => (env_f64("LANE_SPREAD_CAP_L3", 0.020), 0.25_f64),  // base 2.0%
        Lane::L4 => (env_f64("LANE_SPREAD_CAP_L4", 0.050), 0.50_f64),  // base 5.0%
    };
    base + k * atr_norm
}

// â”€â”€ Main function â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Deterministic lane assignment.
/// Called AFTER data quality checks, BEFORE fee_gate() and AI call.
pub fn assign_lane(i: &LaneInputs) -> LaneResult {
    let env_f64 = |key: &str, default: f64| -> f64 {
        std::env::var(key).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
    };

    let vol_l4_volatile = env_f64("LANE_VOLATILE_L4_VOL", 2.0);
    let mom_l4_volatile = env_f64("LANE_VOLATILE_L4_MOM", 0.05);
    let vol_l1_bullish  = env_f64("LANE_BULLISH_L1_VOL", 1.2);
    let vol_l3_bullish  = env_f64("LANE_BULLISH_L3_VOL", 0.5);
    let vol_l4_sideways = env_f64("LANE_SIDEWAYS_L4_VOL", 1.6);


    // â”€â”€ G1: Hard regime blocks â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    // BEARISH: hard block â€” exit-only mode, no new entries
    if i.regime == Regime::Bearish {
        return block(i.symbol, "regime=BEARISH", "LANE-BLOCK");
    }
    // UNKNOWN: advisory only â€” treat as SIDEWAYS, AI sees the warm-up label
    // (was hard block â€” caused missing every move after restart)

    // â”€â”€ G2: VOLATILE â€” only L4 with vol + momentum gate â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if i.regime == Regime::Volatile {
        if i.vol_ratio >= vol_l4_volatile && i.momentum > mom_l4_volatile {
            return fee_gate_result(i.symbol, Lane::L4, i.spread_pct, i.atr_norm);
        }
        let reason = if i.vol_ratio < vol_l4_volatile { "volatile_low_vol" } else { "volatile_low_momentum" };
        return block(i.symbol, reason, "LANE-BLOCK");
    }

    // â”€â”€ G3: Behavioral meme override (large-tier exempt) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if i.is_behavioral_meme {
        if i.tier == "large" {
            tracing::info!(
                "[BEHAVIORAL-MEME] {} large_tier_exempt â€” skipping L4 upgrade",
                i.symbol
            );
            // fall through to matrix
        } else {
            tracing::info!(
                "[BEHAVIORAL-MEME] {} tier={} upgradedâ†’L4 vol={:.1}x",
                i.symbol, i.tier, i.vol_ratio
            );
            return fee_gate_result(i.symbol, Lane::L4, i.spread_pct, i.atr_norm);
        }
    }

    // â”€â”€ G4: L4 vol floor (pre-check before matrix) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    // Applied to meme tier coins even without behavioral signal
    let _is_meme_tier = matches!(i.tier, "meme" | "small");

    // â”€â”€ G5: Regime Ã— tier matrix â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    let raw_lane: Lane = match i.regime {

        Regime::Bullish => match i.tier {
            "large" | "mid" => {
                if i.vol_ratio >= vol_l1_bullish && i.momentum > 0.0 {
                    Lane::L1
                } else if i.vol_ratio >= vol_l3_bullish {
                    Lane::L3
                } else {
                    return block(i.symbol, "bullish_low_vol", "LANE-BLOCK");
                }
            }
            _ => {
                // meme / small
                if i.vol_ratio < vol_l4_sideways {
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
                if i.vol_ratio < vol_l4_sideways {
                    return block(i.symbol, "l4_vol_floor", "LANE-BLOCK");
                }
                Lane::L4
            }
        },

        // Fallback for UNKNOWN or any unhandled regime â€” treat as SIDEWAYS advisory
        _ => Lane::L3,
    };

    // â”€â”€ G6: MTF demotion â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    let lane = match apply_mtf(i.symbol, raw_lane, i.mtf_7d) {
        Ok(l)   => l,
        Err(r)  => return r,
    };

    // â”€â”€ G7: Fee gate (spread cap) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    fee_gate_result(i.symbol, lane, i.spread_pct, i.atr_norm)
}

// â”€â”€ MTF demotion â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

fn apply_mtf(symbol: &str, lane: Lane, mtf_7d: f64) -> Result<Lane, LaneResult> {
    // L2 exception: missing data (-99.0) â†’ allow (mean reversion is regime-agnostic)
    if lane == Lane::L2 && mtf_7d == -99.0 {
        return Ok(Lane::L2);
    }

    match lane {
        Lane::L1 => {
            if mtf_7d < -10.0 {
                Err(block(symbol, "mtf_-10_floor: L1 blocked", "LANE-BLOCK"))
            } else if mtf_7d < -5.0 {
                tracing::info!("[LANE-DEMOTE] {} L1â†’L3 mtf_7d={:.1}%", symbol, mtf_7d);
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

// â”€â”€ Fee gate (spread cap only â€” net-edge handled by fee_filter.rs) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

fn fee_gate_result(symbol: &str, lane: Lane, spread_pct: f64, atr_norm: f64) -> LaneResult {
    let cap = spread_cap(lane, atr_norm);
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
