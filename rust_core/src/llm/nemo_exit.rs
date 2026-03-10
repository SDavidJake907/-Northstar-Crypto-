//! Deterministic Nemo exit scoring with validity masks.
//!
//! 4 buckets: TrendBreak, MomentumBreak, OrderflowBreak, WhalePressure
//! Each bucket has a severity [0,1] AND a validity flag.
//! Dead sensors (stuck at zero, saturated, NaN) are marked invalid and
//! excluded from confirmation counting.
//!
//! Confirmation rule: "3 of valid" (conservative — let winners run).
//!   valid_votes >= 3 → need 3 confirmations (all must agree)
//!   valid_votes == 2 → need 2 confirmations
//!   valid_votes <  2 → need 1 confirmation
//! Whale is always an override channel (not counted in votes).

use std::fmt::Write;

// ── Bucket (severity + validity) ────────────────────────────────

#[derive(Debug, Clone, Copy)]
pub struct Bucket {
    pub sev: f64,    // 0..1 severity (how bad is this signal?)
    pub valid: bool,  // is this sensor actually producing usable data?
}

// ── Input / Output ──────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
pub struct NemoExitInput {
    pub pnl_pct: f64,
    pub hold_minutes: f64,
    pub min_hold_minutes: f64,
    pub min_conf: f64,
    pub trend_score: f64,
    pub momentum_score: f64,
    pub book_imbalance: f64,
    pub buy_ratio: f64,
    pub whale_score: f64,
    pub macd_hist: f64,
    pub zscore: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NemoAction {
    Hold,
    Sell,
}

impl NemoAction {
    pub fn as_str(&self) -> &'static str {
        match self {
            NemoAction::Hold => "HOLD",
            NemoAction::Sell => "SELL",
        }
    }
}

#[derive(Debug, Clone)]
pub struct NemoExitOutput {
    pub action: NemoAction,
    pub confidence: f64,
    pub confirmations: u8,
    pub valid_votes: u8,
    pub required: u8,
    pub bucket_trend: Bucket,
    pub bucket_momentum: Bucket,
    pub bucket_orderflow: Bucket,
    pub bucket_whale: Bucket,
    pub z_penalty: f64,
    pub score: f64,
    pub reason: String,
}

// ── Helpers ─────────────────────────────────────────────────────

#[inline]
fn clamp01(x: f64) -> f64 {
    x.max(0.0).min(1.0)
}

#[inline]
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

/// Dead sensor heuristic: value is finite and not stuck at zero.
#[inline]
fn nonzero_or_unknown(x: f64, eps: f64) -> bool {
    x.is_finite() && x.abs() > eps
}

// ── Bucket builder with validity detection ──────────────────────

fn build_buckets(x: &NemoExitInput) -> (Bucket, Bucket, Bucket, Bucket, f64) {
    // ── Validity detection (heuristic — no per-feature timestamps) ──

    // Trend: valid if finite (trend_score has real spread [-5..+3])
    let trend_valid = x.trend_score.is_finite();

    // Momentum: valid if finite (has real spread [-0.6..+0.7])
    let mom_valid = x.momentum_score.is_finite();

    // Orderflow: TWO sub-sensors, both can be dead independently.
    //   book_imbalance: mostly 0 and unsigned → dead until we see negatives
    //   buy_ratio: saturated at 1.0 → dead (no information)
    let book_live = x.book_imbalance.is_finite()
        && nonzero_or_unknown(x.book_imbalance, 1e-9);
    let buy_live = x.buy_ratio.is_finite()
        && x.buy_ratio >= 0.0
        && x.buy_ratio <= 1.0
        && x.buy_ratio < 0.999; // saturated = dead
    let order_valid = book_live || buy_live;

    // Whale: has real spread [-0.68..+1.0], valid when finite
    let whale_valid = x.whale_score.is_finite();

    // MACD: currently all zero → dead
    let _macd_valid = x.macd_hist.is_finite()
        && nonzero_or_unknown(x.macd_hist, 1e-12);

    // Z-score: currently all zero → dead
    let z_valid = x.zscore.is_finite()
        && nonzero_or_unknown(x.zscore, 1e-12);

    // ── Bucket severities (only when valid, else 0.0) ──

    // A: Trend break — fires when trend_score < -0.5
    let trend_sev = if trend_valid {
        clamp01((-x.trend_score - 0.5) / 2.0)
    } else {
        0.0
    };

    // B: Momentum break — fires when momentum_score < -0.1
    let mom_sev = if mom_valid {
        clamp01((-x.momentum_score - 0.1) / 0.5)
    } else {
        0.0
    };
    // MACD disabled until it warms up (would be: max(mom_sev, macd_sev))

    // C: Orderflow break — only from valid sub-sensors
    let book_sev = if book_live {
        clamp01((-x.book_imbalance - 0.05) / 0.15)
    } else {
        0.0
    };
    let buy_sev = if buy_live {
        clamp01((0.5 - x.buy_ratio) / 0.2)
    } else {
        0.0
    };
    let order_sev = if order_valid {
        book_sev.max(buy_sev)
    } else {
        0.0
    };

    // D: Whale override — negative whale_score = adverse pressure
    let whale_sev = if whale_valid {
        clamp01((-x.whale_score - 0.5) / 0.3)
    } else {
        0.0
    };

    // Z-score penalty: don't puke bottoms unless other signals align
    let z_penalty = if z_valid && x.zscore <= -1.5 { 0.6 } else { 0.0 };

    (
        Bucket { sev: trend_sev, valid: trend_valid },
        Bucket { sev: mom_sev, valid: mom_valid },
        Bucket { sev: order_sev, valid: order_valid },
        Bucket { sev: whale_sev, valid: whale_valid },
        z_penalty,
    )
}

// ── Core scoring function ───────────────────────────────────────

pub fn nemo_exit_score(x: NemoExitInput) -> NemoExitOutput {
    let min_hold_ok = x.hold_minutes >= x.min_hold_minutes;

    let (a, b, c, d, z_penalty) = build_buckets(&x);

    // Weighted confirmations (Nemo suggestion #5):
    // Strong confirms (sev >= 0.80) count as full vote
    // Moderate confirms (sev >= 0.60) count as half vote
    // This avoids unnecessary exits in choppy markets
    let c_a_strong = (a.valid && a.sev >= 0.80) as u8;
    let c_b_strong = (b.valid && b.sev >= 0.80) as u8;
    let c_c_strong = (c.valid && c.sev >= 0.80) as u8;
    let c_a_mod = (a.valid && a.sev >= 0.60 && a.sev < 0.80) as u8;
    let c_b_mod = (b.valid && b.sev >= 0.60 && b.sev < 0.80) as u8;
    let c_c_mod = (c.valid && c.sev >= 0.60 && c.sev < 0.80) as u8;
    let strong_count = c_a_strong + c_b_strong + c_c_strong;
    let mod_count = c_a_mod + c_b_mod + c_c_mod;
    // Weighted: 2 strong + 1 moderate = exit, or 3 strong = exit
    let confirmations = strong_count + mod_count; // for logging compatibility
    let weighted_ok = strong_count >= 2 || (strong_count >= 1 && mod_count >= 2);

    // Valid votes: how many channels are actually producing data
    let valid_votes = a.valid as u8 + b.valid as u8 + c.valid as u8;

    // Required: kept for logging, actual decision uses weighted_ok
    let required = if valid_votes >= 3 { 3 } else if valid_votes == 2 { 2 } else { 1 };

    // Weighted score
    let score =
        0.90 * (confirmations as f64)
        + 1.20 * a.sev
        + 1.50 * b.sev
        + 1.80 * c.sev
        + 2.50 * d.sev
        - z_penalty;

    let confidence = sigmoid(score);

    // Decision rules
    let whale_override = d.valid && d.sev >= 0.95; // was 0.90 — raised to avoid choppy market over-triggers
    let sell_ok = min_hold_ok
        && confidence >= x.min_conf
        && (weighted_ok || whale_override);

    let action = if sell_ok { NemoAction::Sell } else { NemoAction::Hold };

    let reason = build_reason(
        action, confidence, confirmations, valid_votes, required,
        &a, &b, &c, &d,
        x.min_hold_minutes,
        min_hold_ok, whale_override, x.pnl_pct,
    );

    NemoExitOutput {
        action,
        confidence,
        confirmations,
        valid_votes,
        required,
        bucket_trend: a,
        bucket_momentum: b,
        bucket_orderflow: c,
        bucket_whale: d,
        z_penalty,
        score,
        reason,
    }
}

// ── Reason builder ──────────────────────────────────────────────

fn build_reason(
    action: NemoAction,
    confidence: f64,
    confirmations: u8,
    valid_votes: u8,
    required: u8,
    a: &Bucket,
    b: &Bucket,
    c: &Bucket,
    d: &Bucket,
    min_hold_minutes: f64,
    min_hold_ok: bool,
    whale_override: bool,
    pnl_pct: f64,
) -> String {
    let mut reason = String::with_capacity(200);

    if !min_hold_ok {
        let _ = write!(reason, "anti-churn: hold < {:.0}min", min_hold_minutes);
        return reason;
    }

    let _ = write!(reason, "{}: ", action.as_str());

    // Active drivers
    let mut drivers: Vec<&str> = Vec::new();
    if a.valid && a.sev >= 0.5 { drivers.push("trend_break"); }
    if b.valid && b.sev >= 0.5 { drivers.push("momentum_break"); }
    if c.valid && c.sev >= 0.5 { drivers.push("orderflow_weak"); }
    if whale_override {
        drivers.push("whale_override");
    } else if d.valid && d.sev >= 0.3 {
        drivers.push("whale_pressure");
    }

    if drivers.is_empty() {
        reason.push_str("no thesis break");
    } else {
        reason.push_str(&drivers.join(" + "));
    }

    // Stats
    let _ = write!(reason, " (conf={:.2} confirms={}/{} valid={}/3",
        confidence, confirmations, required, valid_votes);

    // Dead sensors
    let mut dead: Vec<&str> = Vec::new();
    if !a.valid { dead.push("T"); }
    if !b.valid { dead.push("M"); }
    if !c.valid { dead.push("O"); }
    if !d.valid { dead.push("W"); }
    if !dead.is_empty() {
        let _ = write!(reason, " dead={}", dead.join(","));
    }

    reason.push(')');

    // PnL zone
    if pnl_pct > 0.5 {
        reason.push_str(" [PROFIT]");
    } else if pnl_pct < -0.5 {
        reason.push_str(" [LOSS]");
    } else {
        reason.push_str(" [EVEN]");
    }

    reason
}

/// Build a validity summary string for LLM prompt injection.
/// e.g. "valid: trend=1 mom=1 order=0 whale=1 confirms=2/required=2"
pub fn validity_summary(out: &NemoExitOutput) -> String {
    format!(
        "valid: trend={} mom={} order={} whale={} confirms={}/required={}",
        out.bucket_trend.valid as u8,
        out.bucket_momentum.valid as u8,
        out.bucket_orderflow.valid as u8,
        out.bucket_whale.valid as u8,
        out.confirmations,
        out.required,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // Default test config: 30min hold, 0.60 min confidence
    fn test_input(pnl_pct: f64, hold_minutes: f64, trend_score: f64,
                  momentum_score: f64, book_imbalance: f64, buy_ratio: f64,
                  whale_score: f64) -> NemoExitInput {
        NemoExitInput {
            pnl_pct, hold_minutes,
            min_hold_minutes: 30.0, min_conf: 0.60,
            trend_score, momentum_score, book_imbalance, buy_ratio,
            whale_score, macd_hist: 0.0, zscore: 0.0,
        }
    }

    #[test]
    fn test_hold_early() {
        let out = nemo_exit_score(test_input(-2.0, 3.0, -5.0, -0.6, 0.0, 0.0, -0.8));
        assert_eq!(out.action, NemoAction::Hold, "must HOLD before 30 min");
    }

    #[test]
    fn test_sell_trend_momentum() {
        let out = nemo_exit_score(test_input(-1.5, 30.0, -3.0, -0.5, 0.0, 1.0, 0.0));
        // orderflow invalid (book=0, buy_ratio=1.0 saturated)
        // valid_votes = 2 (trend + momentum), required = 2
        // 2 confirmations >= 2 required → SELL
        assert_eq!(out.action, NemoAction::Sell, "trend+momentum break = SELL");
        assert!(out.confirmations >= 2);
        assert!(!out.bucket_orderflow.valid, "orderflow should be invalid");
    }

    #[test]
    fn test_dead_sensors_lower_threshold() {
        let out = nemo_exit_score(test_input(-1.0, 35.0, -3.0, 0.2, 0.0, 1.0, 0.0));
        // valid_votes = 2 (trend + momentum), required = 2
        // only trend fires (1 confirmation < 2 required) → HOLD
        assert_eq!(out.valid_votes, 2);
        assert_eq!(out.required, 2);
    }

    #[test]
    fn test_hold_mixed() {
        let out = nemo_exit_score(test_input(0.3, 35.0, 1.0, 0.2, 0.05, 0.8, 0.3));
        assert_eq!(out.action, NemoAction::Hold, "positive signals = HOLD");
        assert_eq!(out.confirmations, 0);
    }

    #[test]
    fn test_whale_override() {
        let out = nemo_exit_score(test_input(-0.5, 35.0, 0.0, 0.0, 0.0, 1.0, -0.95));
        assert_eq!(out.action, NemoAction::Sell, "whale override = SELL");
        assert!(out.bucket_whale.valid);
        assert!(out.bucket_whale.sev >= 0.90);
    }

    #[test]
    fn test_orderflow_valid_when_buy_ratio_moves() {
        let out = nemo_exit_score(test_input(-1.0, 35.0, -2.0, -0.3, 0.0, 0.3, 0.0));
        assert!(out.bucket_orderflow.valid, "buy_ratio=0.3 should make orderflow valid");
        assert_eq!(out.valid_votes, 3);
        assert_eq!(out.required, 3);
    }
}
