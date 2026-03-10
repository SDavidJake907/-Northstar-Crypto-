//! Schedule-based parameter overlay — auto-adjusts margin gate, tool rounds,
//! concurrency, and ATR tightening by AKST time block.
//!
//! Gated behind `SCHEDULE_ENABLED=1` — when disabled, returns full-mode defaults.

use crate::config::CachedEnv;

/// Active parameter profile for the current time block.
#[derive(Clone, Debug)]
pub struct ScheduleOverlay {
    pub margin_gate: f64,
    pub max_tool_rounds: u8,
    pub atr_tighten: f64,       // 1.0 = normal, 0.8 = 20% tighter
    pub max_concurrent: usize,
    #[allow(dead_code)]
    pub l2_aggression: f64,     // 1.0 = normal, 0.5 = halved
    pub mode: &'static str,     // "full" | "conservative" | "minimal" | "overnight"
    pub block_name: &'static str,
}

impl Default for ScheduleOverlay {
    fn default() -> Self {
        Self {
            margin_gate: 0.15,
            max_tool_rounds: 2,
            atr_tighten: 1.0,
            max_concurrent: 3,
            l2_aggression: 1.0,
            mode: "full",
            block_name: "DEFAULT",
        }
    }
}

/// Get the current AKST hour (UTC-9, simple offset — no crate needed).
fn akst_hour() -> u32 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    // AKST = UTC - 9 hours
    let akst_secs = secs.wrapping_sub(9 * 3600);
    let day_secs = (akst_secs % 86400) as u32;
    day_secs / 3600
}

/// Compute the current schedule overlay based on AKST time and env config.
///
/// Time blocks (AKST):
///   05:30–09:30  HIGH VOL    — full size, margin ≥0.15, 2 rounds, 3 concurrent
///   09:30–12:00  MIDDAY      — conservative, margin ≥0.17, ATR ×0.8, 2 concurrent
///   14:00–20:00  AFTERNOON   — minimal, margin ≥0.18, 1 round, fewer L2
///   20:00–05:30  OVERNIGHT   — conservative, margin ≥0.20, 1 round, 1 concurrent
pub fn current_overlay(env: &CachedEnv) -> ScheduleOverlay {
    if !env.get_bool("SCHEDULE_ENABLED", false) {
        return ScheduleOverlay::default();
    }

    let hour = akst_hour();

    // Time blocks use half-hour boundaries approximated to full hours
    // 05:30-09:30 ≈ hours 5-9, 09:30-12:00 ≈ hours 9-11,
    // 14:00-20:00 ≈ hours 14-19, 20:00-05:30 ≈ hours 20-5
    match hour {
        5..=8 => ScheduleOverlay {
            margin_gate: env.get_f64("SCHED_HIGH_VOL_MARGIN", 0.15),
            max_tool_rounds: env.get_f64("SCHED_HIGH_VOL_TOOL_ROUNDS", 2.0) as u8,
            atr_tighten: 1.0,
            max_concurrent: env.get_f64("SCHED_HIGH_VOL_CONCURRENT", 3.0) as usize,
            l2_aggression: 1.0,
            mode: "full",
            block_name: "HIGH_VOL",
        },
        9..=11 => ScheduleOverlay {
            margin_gate: env.get_f64("SCHED_MIDDAY_MARGIN", 0.17),
            max_tool_rounds: env.get_f64("SCHED_MIDDAY_TOOL_ROUNDS", 2.0) as u8,
            atr_tighten: env.get_f64("SCHED_MIDDAY_ATR_TIGHTEN", 0.8),
            max_concurrent: env.get_f64("SCHED_MIDDAY_CONCURRENT", 2.0) as usize,
            l2_aggression: 1.0,
            mode: "conservative",
            block_name: "MIDDAY",
        },
        14..=19 => ScheduleOverlay {
            margin_gate: env.get_f64("SCHED_AFTERNOON_MARGIN", 0.18),
            max_tool_rounds: env.get_f64("SCHED_AFTERNOON_TOOL_ROUNDS", 1.0) as u8,
            atr_tighten: env.get_f64("SCHED_AFTERNOON_ATR_TIGHTEN", 0.8),
            max_concurrent: env.get_f64("SCHED_AFTERNOON_CONCURRENT", 2.0) as usize,
            l2_aggression: 0.5,
            mode: "minimal",
            block_name: "AFTERNOON",
        },
        // 12-13 gap + 20-4 = overnight
        _ => ScheduleOverlay {
            margin_gate: env.get_f64("SCHED_OVERNIGHT_MARGIN", 0.20),
            max_tool_rounds: env.get_f64("SCHED_OVERNIGHT_TOOL_ROUNDS", 1.0) as u8,
            atr_tighten: env.get_f64("SCHED_OVERNIGHT_ATR_TIGHTEN", 0.8),
            max_concurrent: env.get_f64("SCHED_OVERNIGHT_CONCURRENT", 1.0) as usize,
            l2_aggression: 0.5,
            mode: "overnight",
            block_name: "OVERNIGHT",
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_overlay_is_full_mode() {
        let overlay = ScheduleOverlay::default();
        assert_eq!(overlay.mode, "full");
        assert!((overlay.margin_gate - 0.15).abs() < 1e-9);
        assert_eq!(overlay.max_concurrent, 3);
    }
}
