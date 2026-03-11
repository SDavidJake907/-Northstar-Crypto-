//! Fee-aware entry filter — blocks negative-expectancy trades before AI call.
//! Runs AFTER assign_lane(), BEFORE any AI inference.
//! Two checks: (1) spread cap per lane, (2) net_edge = expected_move - total_cost.

use super::lane::Lane;

// ── Constants ─────────────────────────────────────────────────────────────────

const FEE_ROUND_TRIP:    f64 = 0.0052;  // 0.26% × 2 sides taker
const SLIPPAGE_FIXED:    f64 = 0.0015;  // 0.15% floor
const EXPECTED_MOVE_CAP: f64 = 0.20;    // 20% ceiling — prevents meme inflation

// ── Inputs ────────────────────────────────────────────────────────────────────

pub struct FeeInputs<'a> {
    pub symbol:      &'a str,
    pub lane:        Lane,
    pub atr_norm:    f64,    // floor = 0.010 applied by caller
    pub spread_pct:  f64,    // default = atr_norm × 0.10 if 0.0
    pub trend_score: f64,
    pub zscore:      f64,
    pub momentum:    f64,
    pub vol_ratio:   f64,
}

pub struct FeeResult {
    pub passed:             bool,
    pub reason:             String,
    pub expected_move_pct:  f64,
    pub total_cost_pct:     f64,
    pub net_edge_pct:       f64,
}

// ── Main function ─────────────────────────────────────────────────────────────

pub fn fee_gate(i: &FeeInputs) -> FeeResult {

    // ── 1. Estimate expected move ─────────────────────────────────────
    let raw_expected: f64 = match i.lane {
        Lane::L1 => {
            let ts = i.trend_score.max(0.0);  // negative trend → zero expected
            i.atr_norm * ts * 2.0
        }
        Lane::L2 => {
            // zscore alone goes to 0 in sideways — add momentum + vol boost so
            // real setups aren't zeroed out by a flat z-score
            let zscore_part   = i.zscore.abs() * i.atr_norm * 0.80;
            let momentum_part = i.momentum.max(0.0) * 0.05;
            let vol_part      = (i.vol_ratio - 1.0).max(0.0) * i.atr_norm * 0.30;
            zscore_part + momentum_part + vol_part
        }
        Lane::L3 => {
            (i.atr_norm * 1.5) + (i.momentum.max(0.0) * 0.10)
        }
        Lane::L4 => {
            i.vol_ratio * i.atr_norm * 0.50
        }
    };
    let expected_move = raw_expected.min(EXPECTED_MOVE_CAP);

    // ── 2. Total cost ─────────────────────────────────────────────────
    let slippage_vol: f64 = i.atr_norm * match i.lane {
        Lane::L1 => 0.30,
        Lane::L2 => 0.20,
        Lane::L3 => 0.25,
        Lane::L4 => 0.40,
    };
    let total_cost  = FEE_ROUND_TRIP + SLIPPAGE_FIXED + slippage_vol;
    let net_edge    = expected_move - total_cost;

    // ── 3. Threshold per lane ─────────────────────────────────────────
    // Minimum net edge required — let AI make final call on marginal setups
    let threshold: f64 = match i.lane {
        Lane::L1 => 0.0015,  // 0.15% — strong momentum signal, low bar
        Lane::L2 => 0.0010,  // 0.10% — compression plays, very low bar
        Lane::L3 => 0.0015,  // 0.15% — trend rides
        Lane::L4 => 0.0020,  // 0.20% — hot movers have huge expected moves
    };

    // ── 4. Block check ────────────────────────────────────────────────
    if net_edge < threshold {
        let reason = block_reason(i.lane, net_edge, i.trend_score, i.zscore, i.vol_ratio);
        tracing::info!(
            "[FEE-GATE] {} BLOCKED lane={} expected={:.3}% cost={:.3}% net={:.3}% threshold={:.3}% reason={}",
            i.symbol, i.lane,
            expected_move * 100.0, total_cost * 100.0,
            net_edge * 100.0, threshold * 100.0,
            reason
        );
        crate::nemo_optimizer::flag_error(
            "fee_gate",
            &format!("{} {} net={:.3}%<{:.3}%", i.symbol, i.lane, net_edge * 100.0, threshold * 100.0),
        );
        return FeeResult {
            passed: false,
            reason: format!("fee_gate:{}", reason),
            expected_move_pct: expected_move * 100.0,
            total_cost_pct:    total_cost   * 100.0,
            net_edge_pct:      net_edge     * 100.0,
        };
    }

    // ── 5. Pass ───────────────────────────────────────────────────────
    tracing::debug!(
        "[FEE-PASS] {} lane={} expected={:.3}% cost={:.3}% net={:.3}%",
        i.symbol, i.lane,
        expected_move * 100.0, total_cost * 100.0, net_edge * 100.0
    );

    FeeResult {
        passed: true,
        reason: String::new(),
        expected_move_pct: expected_move * 100.0,
        total_cost_pct:    total_cost   * 100.0,
        net_edge_pct:      net_edge     * 100.0,
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn block_reason(lane: Lane, net_edge: f64, trend_score: f64, zscore: f64, vol_ratio: f64) -> &'static str {
    let _ = net_edge;
    match lane {
        Lane::L1 if trend_score <= 1.0 => "weak_trend_low_edge",
        Lane::L1                        => "l1_insufficient_edge",
        Lane::L2 if zscore.abs() < 0.5 => "l2_zscore_too_shallow",
        Lane::L2                        => "l2_insufficient_edge",
        Lane::L3                        => "l3_insufficient_edge",
        Lane::L4 if vol_ratio < 2.0    => "l4_low_vol_spike",
        Lane::L4                        => "l4_insufficient_edge",
    }
}
