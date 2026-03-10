use std::collections::{HashMap, VecDeque};

#[derive(Clone, Debug)]
pub struct Candle {
    pub close: f64,
    pub high: f64,
    pub low: f64,
    pub volume: f64,
}

#[derive(Clone, Debug, Default)]
pub struct OrderbookMetrics {
    pub spread_pct: f64,
    pub imbalance: f64,
}

#[derive(Clone, Debug, Default)]
pub struct TradeFlow {
    pub buy_ratio: f64,
    pub sell_ratio: f64,
}

#[derive(Clone, Debug, Default)]
pub struct BookReversalMetrics {
    pub trend: i32,
    pub reversal: bool,
    pub reversal_dir: i32,
    pub strength: f64,
    pub avg_imb: f64,
}

#[allow(dead_code)]
#[derive(Clone, Debug, Default)]
pub struct Features {
    pub price: f64,
    pub ema9: Option<f64>,
    pub ema21: Option<f64>,
    pub ema50: Option<f64>,
    pub ema55: Option<f64>,
    pub ema200: Option<f64>,
    pub rsi: f64,
    pub macd_hist: f64,
    pub atr: f64,
    pub bb_upper: Option<f64>,
    pub bb_lower: Option<f64>,
    pub zscore: f64,
    pub quant_vol: f64,
    pub quant_momo: f64,
    pub quant_regime: f64,
    pub quant_meanrev: f64,
    pub quant_liquidity: f64,
    pub quant_risk: f64,
    pub quant_corr_btc: f64,
    pub quant_corr_eth: f64,
    pub vol_ratio: f64,
    pub trend_score: i32,
    pub market_state: f64,
    pub momentum_score: f64,
    pub trend_strength: i32,
    pub consecutive_candles: i32,
    pub tud_up: i32,
    pub tud_down: i32,
    pub spread_pct: f64,
    pub book_imbalance: f64,
    pub buy_ratio: f64,
    pub sell_ratio: f64,
    pub flow_imbalance: f64,
    pub flow_book_div: f64,
    pub flow_book_div_abs: f64,
    pub book_trend: i32,
    pub book_reversal: bool,
    pub book_reversal_dir: i32,
    pub book_strength: f64,
    pub book_avg_imb: f64,
    // Quantum math indicators
    pub hurst_exp: f64,
    pub shannon_entropy: f64,
    pub autocorr_lag1: f64,
    pub adx: f64,
}

fn safe_float(v: f64, default: f64) -> f64 {
    if v.is_finite() {
        v
    } else {
        default
    }
}

#[inline]
fn clamp01(v: f64) -> f64 {
    v.clamp(0.0, 1.0)
}

// ── Pure-Math Scoring Engine ────────────────────────────────────
//
// Converts discrete if/else trading rules into continuous math:
//   - Logistic gates replace boolean conditions
//   - Products replace AND, soft-union replaces OR
//   - Softmax over (S_buy, S_sell, S_hold) gives action + confidence
//
// The result is a smooth, differentiable decision surface with no branching.

/// Action scores from the pure-math scoring engine.
#[derive(Clone, Debug, Default)]
pub struct ActionScores {
    pub s_buy: f64,
    pub s_sell: f64,
    pub s_hold: f64,
    pub action: String,
    pub confidence: f64,
    /// Individual signal strengths (for transparency in the AI prompt)
    pub crash: f64,
    pub hunt: f64,
    pub buy_pressure: f64,
    pub sell_pressure: f64,
    pub trend_buy: f64,
    pub momentum_buy: f64,
    /// L2 mean-reversion signals
    pub mean_rev_buy: f64,
    pub mean_rev_sell: f64,
    pub is_l2: bool,
    /// Dynamic threshold: fluctuates with EMA tunnels + RSI
    /// Strong trend + good RSI = lower threshold (easier entry)
    /// Choppy + extreme RSI = higher threshold (harder entry)
    pub dynamic_threshold: f64,
    /// Why this decision was made — tracks missing indicators, clamped gates, etc.
    /// Empty = all data present. Non-empty during HOLD = likely broken pipeline, not market uncertainty.
    pub reasons: Vec<String>,
}

/// Check if STRICT_FEATURES is enabled (log/block on missing indicator data).
fn strict_enabled() -> bool {
    matches!(
        std::env::var("STRICT_FEATURES").ok().as_deref(),
        Some("1") | Some("true") | Some("TRUE") | Some("on") | Some("ON")
    )
}

/// Extract a required indicator value. Returns None if missing/zero/non-finite.
/// Pushes a reason string when the value is absent.
fn require_indicator(name: &str, v: Option<f64>, reasons: &mut Vec<String>) -> Option<f64> {
    match v.filter(|x| x.is_finite() && *x > 0.0) {
        Some(x) => Some(x),
        None => {
            reasons.push(format!("missing_{name}"));
            if strict_enabled() {
                tracing::warn!("[FEATURE-MISSING] {name} missing/invalid — signals degraded");
            }
            None
        }
    }
}

/// Logistic gate: g(x; k) = 1 / (1 + e^(-k*x))
/// Smooth replacement for boolean threshold checks.
/// k controls sharpness: large k ≈ hard step, small k ≈ soft transition.
#[inline]
fn gate(x: f64, k: f64) -> f64 {
    let exp_arg = (-k * x).clamp(-500.0, 500.0); // prevent overflow
    1.0 / (1.0 + exp_arg.exp())
}

/// Soft OR: 1 - (1-a)(1-b)
/// Replaces boolean OR with a continuous [0,1] value.
#[inline]
fn soft_or(a: f64, b: f64) -> f64 {
    1.0 - (1.0 - a) * (1.0 - b)
}

/// Softplus: t * ln(1 + e^(x/t))
/// Smooth replacement for max(x, 0).
#[inline]
fn softplus(x: f64, t: f64) -> f64 {
    let arg = (x / t).clamp(-500.0, 500.0);
    t * (1.0 + arg.exp()).ln()
}

/// Normalization: maps [lo, hi] → [-1, +1]
#[inline]
fn norm(x: f64, lo: f64, hi: f64) -> f64 {
    if (hi - lo).abs() < 1e-12 {
        return 0.0;
    }
    ((x - lo) / (hi - lo)).clamp(0.0, 1.0) * 2.0 - 1.0
}

/// Compute the 7-dimensional weighted SCORE from raw features.
/// This is the "energy" that drives buy/sell tendency.
fn compute_weighted_score(f: &Features) -> f64 {
    let trend = norm(f.trend_score as f64, -5.0, 5.0);
    let momentum = (norm(f.macd_hist, -1.0, 1.0) + norm(f.rsi, 20.0, 80.0)) / 2.0;
    let volatility = -norm(f.atr, 0.0, f.price * 0.05);
    let volume = norm(f.vol_ratio, 0.5, 2.0);
    let mean_rev = -norm(f.zscore, -2.5, 2.5);
    let orderflow = (norm(f.book_imbalance, -1.0, 1.0) + norm(f.buy_ratio, 0.3, 0.7)) / 2.0;
    let quant = norm(f.quant_momo, -0.03, 0.03);

    // Regime-aware L1/L3 weights: momentum-dominant for trending markets
    // Momentum drives entries, trend confirms direction, orderflow validates
    // Volume/volatility demoted — they punish good trending setups unfairly
    let score = 0.25 * trend        // direction confirmation
        + 0.40 * momentum           // PRIMARY driver — MACD + RSI momentum
        + 0.05 * volatility         // minor — don't penalize trending ATR
        + 0.05 * volume             // minor — good trends often start on low vol
        + 0.02 * mean_rev           // near-zero — mean-rev fights trends
        + 0.20 * orderflow          // orderflow validates (book_imb + buy_ratio)
        + 0.03 * quant;             // quant momentum tiebreaker

    score.clamp(-1.0, 1.0)
}

/// Detect L2_COMPRESSION (sideways / range-bound regime).
///
/// Conditions:
///   - EMAs braided: |EMA9 - EMA21| < ε and |EMA21 - EMA50| < ε (relative to price)
///   - MACD_hist ≈ 0 (within tolerance)
///   - RSI in [40, 60]
///   - Low ATR (relative to price)
///   - |Zscore| < 1.0 (no extremes)
///   - No directional orderflow dominance: |Book_Imb| < 0.2
fn detect_l2(f: &Features) -> bool {
    if f.price <= 0.0 { return false; }

    // All 3 EMAs must be present + positive — otherwise can't detect compression
    let (ema9, ema21, ema50) = match (
        f.ema9.filter(|v| v.is_finite() && *v > 0.0),
        f.ema21.filter(|v| v.is_finite() && *v > 0.0),
        f.ema50.filter(|v| v.is_finite() && *v > 0.0),
    ) {
        (Some(a), Some(b), Some(c)) => (a, b, c),
        _ => return false, // missing EMAs → can't detect L2
    };

    // Braided EMAs: all within 1.5% of each other (relative to price)
    let eps = f.price * 0.015;
    let emas_braided = (ema9 - ema21).abs() < eps && (ema21 - ema50).abs() < eps;

    // MACD_hist near zero (relative to price, tolerance 0.15%)
    let macd_flat = (f.macd_hist / f.price).abs() < 0.0015;

    // RSI in neutral zone [40, 60]
    let rsi_neutral = f.rsi >= 40.0 && f.rsi <= 60.0;

    // Low ATR (< 2% of price)
    let atr_low = f.atr < f.price * 0.02;

    // Z-score not extreme
    let z_contained = f.zscore.abs() < 1.0;

    // No directional orderflow dominance
    let flow_neutral = f.book_imbalance.abs() < 0.20;

    // Need at least 5 of 6 conditions (allow one to be borderline)
    let hits = [emas_braided, macd_flat, rsi_neutral, atr_low, z_contained, flow_neutral]
        .iter().filter(|&&x| x).count();
    hits >= 5
}

/// Weighted score for L2_COMPRESSION regime.
/// Shifts weights: reduce trend/momentum, boost mean_rev/orderflow/zscore.
fn compute_l2_weighted_score(f: &Features) -> f64 {
    let trend = norm(f.trend_score as f64, -5.0, 5.0);
    let momentum = (norm(f.macd_hist, -1.0, 1.0) + norm(f.rsi, 20.0, 80.0)) / 2.0;
    let volatility = -norm(f.atr, 0.0, f.price * 0.05);
    let volume = norm(f.vol_ratio, 0.5, 2.0);
    let mean_rev = -norm(f.zscore, -2.5, 2.5);  // negative Z = bullish in mean-rev
    let orderflow = (norm(f.book_imbalance, -1.0, 1.0) + norm(f.buy_ratio, 0.3, 0.7)) / 2.0;
    let quant = norm(f.quant_momo, -0.03, 0.03);

    // L2 weights: mean_rev + orderflow dominant, trend/momentum reduced
    let score = 0.08 * trend        // was 0.30
        + 0.07 * momentum           // was 0.22
        + 0.10 * volatility         // same
        + 0.10 * volume             // was 0.20
        + 0.35 * mean_rev           // was 0.10 — now primary driver
        + 0.25 * orderflow          // was 0.05 — major increase
        + 0.05 * quant;             // was 0.03

    score.clamp(-1.0, 1.0)
}

/// Compute pure-math action scores from features.
///
/// Pipeline: features → gates → signal strengths → action scores → softmax → decision
///
/// v2: Trend-following + momentum signals added. Dynamic threshold fluctuates
///     with EMA tunnels and RSI — no fixed stone threshold.
/// v3: L2_COMPRESSION mean-reversion signals for sideways markets.
///
/// No if/else. All continuous functions.
pub fn compute_action_scores(f: &Features) -> ActionScores {
    let entry_thresh: f64 = std::env::var("ENTRY_THRESHOLD")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.60);
    compute_action_scores_with_threshold(f, entry_thresh)
}

pub fn compute_action_scores_with_threshold(f: &Features, entry_threshold: f64) -> ActionScores {
    let mut reasons: Vec<String> = Vec::new();

    let is_l2 = detect_l2(f);
    let score = if is_l2 { compute_l2_weighted_score(f) } else { compute_weighted_score(f) };

    // Gate sharpness parameters
    let k = 20.0;  // main signal gates (fairly sharp)

    // Extract required EMAs safely — no more unwrap_or(0.0) fake math
    let ema21_opt = require_indicator("ema21", f.ema21, &mut reasons);
    let ema50_opt = require_indicator("ema50", f.ema50, &mut reasons);

    // ── Crash signal ───────────────────────────────────────────
    // SELL if EMA regime gap <= -0.005 OR realized vol >= 0.02
    // With missing EMAs: regime_gap=0.0 (neutral), crash relies on quant_vol only
    let regime_gap = match (ema21_opt, ema50_opt) {
        (Some(e21), Some(e50)) if f.price > 0.0 => (e21 - e50) / f.price,
        _ => 0.0, // neutral — don't fake a crash signal from missing data
    };
    let c1 = gate(-0.005 - regime_gap, k);
    let c2 = gate(f.quant_vol - 0.02, k);
    let crash = soft_or(c1, c2);

    // ── Hunt signal (orderbook-driven buy) ────────────────────
    // Book_Imb >= 0.25 AND BuyRatio >= 0.52 AND Momentum > 0
    // (lowered from 0.50 — catches whale activity earlier)
    let h1 = gate(f.book_imbalance - 0.25, k);
    let h2 = gate(f.buy_ratio - 0.52, k);
    let h3 = gate(f.momentum_score, k);
    let hunt = h1 * h2 * h3;

    // ── Trend-buy signal ──────────────────────────────────────
    // Catches rallies even when orderbook is flat:
    //   trend_score >= 3 AND price > EMA21 AND RSI in [40,75] AND momentum > 0.1
    // If EMA21 missing → trend_buy = 0.0 (disabled, not faked)
    let trend_norm = (f.trend_score as f64 - 3.0) / 2.0; // 3→0, 5→1
    let t1 = gate(trend_norm, k);
    let t2 = match ema21_opt {
        Some(e21) => gate((f.price - e21) / e21 * 100.0, k),
        None => 0.0, // missing EMA21 → disable trend gate (not fake it)
    };
    let t3 = gate(f.rsi - 40.0, k) * gate(75.0 - f.rsi, k); // RSI sweet spot [40,75]
    let t4 = gate(f.momentum_score - 0.1, k);
    let trend_buy = t1 * t2 * t3 * t4;

    // ── Momentum-buy signal ───────────────────────────────────
    // Strong momentum run even without trend confirmation:
    //   momentum > 0.35 AND buy_ratio > 0.48 AND RSI > 35 AND RSI < 78
    let m1 = gate(f.momentum_score - 0.35, k);
    let m2 = gate(f.buy_ratio - 0.48, k);
    let m3 = gate(f.rsi - 35.0, k) * gate(78.0 - f.rsi, k);
    let momentum_buy = m1 * m2 * m3;

    // ── L2 Mean-Reversion BUY (sideways markets) ──────────────
    // Z < -1.5, RSI < 35, price near BB_lower, book flips bullish, low ATR
    // If BB missing → mr_buy = 0.0 (disabled, not computed from fake bands)
    let mr_buy = if is_l2 {
        match (
            f.bb_lower.filter(|v| v.is_finite() && *v > 0.0),
            f.bb_upper.filter(|v| v.is_finite() && *v > 0.0),
        ) {
            (Some(bb_lower), Some(bb_upper)) if bb_upper > bb_lower => {
                let bb_range = bb_upper - bb_lower;
                let bb_low_prox = gate((bb_lower + bb_range * 0.15 - f.price) / bb_range * 10.0, k);
                let z_deep = gate(-1.5 - f.zscore, k);
                let rsi_low = gate(35.0 - f.rsi, k);
                let book_flip = gate(f.book_imbalance, k);
                let atr_ok = gate(0.02 - f.atr / f.price.max(1e-9), k);
                z_deep * rsi_low * bb_low_prox * book_flip * atr_ok
            }
            _ => {
                reasons.push("missing_bb_lower_upper".into());
                0.0
            }
        }
    } else { 0.0 };

    // ── L2 Mean-Reversion SELL (sideways markets) ─────────────
    // Z > +1.5, RSI > 65, price near BB_upper, book flips bearish, low ATR
    let mr_sell = if is_l2 {
        match (
            f.bb_lower.filter(|v| v.is_finite() && *v > 0.0),
            f.bb_upper.filter(|v| v.is_finite() && *v > 0.0),
        ) {
            (Some(bb_lower), Some(bb_upper)) if bb_upper > bb_lower => {
                let bb_range = bb_upper - bb_lower;
                let bb_high_prox = gate((f.price - bb_upper + bb_range * 0.15) / bb_range * 10.0, k);
                let z_high = gate(f.zscore - 1.5, k);
                let rsi_high = gate(f.rsi - 65.0, k);
                let book_bear = gate(-f.book_imbalance, k);
                let atr_ok = gate(0.02 - f.atr / f.price.max(1e-9), k);
                z_high * rsi_high * bb_high_prox * book_bear * atr_ok
            }
            _ => 0.0, // already logged in mr_buy branch
        }
    } else { 0.0 };

    // ── Moderate buy pressure ──────────────────────────────────
    // Book_Imb >= 0.10 AND BuyRatio >= 0.50 AND Momentum >= 0
    let b1 = gate(f.book_imbalance - 0.10, k);
    let b2 = gate(f.buy_ratio - 0.50, k);
    let b3 = gate(f.momentum_score, k);
    let buy_pressure = b1 * b2 * b3;

    // ── Sell pressure ──────────────────────────────────────────
    // Book_Imb <= -0.25 OR (Momentum < -0.002 AND BuyRatio < 0.45)
    let s1 = gate(-0.25 - f.book_imbalance, k);
    let s2 = gate(-0.002 - f.momentum_score, k);
    let s3 = gate(0.45 - f.buy_ratio, k);
    let sell_pressure = soft_or(s1, s2 * s3);

    // ── Composite confidence ─────────────────────────────────
    // Uses best of: trend strength, momentum, and book imbalance
    // Capped at 0.90 (was 0.75 — too restrictive)
    let conf_trend = (f.trend_score as f64).clamp(0.0, 5.0) / 5.0;
    let conf_momo = f.momentum_score.clamp(0.0, 1.0);
    let conf_book = f.book_imbalance.clamp(0.0, 1.0);
    let conf_buy = (0.45 + 0.20 * conf_trend + 0.15 * conf_momo + 0.15 * conf_book).min(0.90);
    let conf_sell = (0.5 + 0.5 * f.book_imbalance.abs().clamp(0.0, 1.0)).min(0.85);

    // ── Score decomposition (smooth max(x,0) via softplus) ────
    let t = 0.1;          // softplus temperature
    let lambda = 0.30;    // weight of SCORE on action scores (was 0.15)
    let pos_score = softplus(score, t);
    let neg_score = softplus(-score, t);

    // ── Action scores: S_buy, S_sell, S_hold ───────────────────
    // v3: L2 mean-reversion signals contribute in sideways markets,
    //     while trend/momentum signals are damped.
    let l2_damp = if is_l2 { 0.25 } else { 1.0 }; // dampen trend signals in L2
    let l2_boost = if is_l2 { 1.0 } else { 0.0 };  // only active in L2

    let s_buy  = l2_damp * 0.70 * hunt
              + l2_damp * 0.80 * trend_buy
              + l2_damp * 0.50 * momentum_buy
              + conf_buy * buy_pressure
              + l2_boost * 0.90 * mr_buy    // L2 mean-reversion BUY
              + lambda * pos_score
              - 2.0 * crash;

    let s_sell = 0.90 * crash
              + conf_sell * sell_pressure
              + l2_boost * 0.85 * mr_sell   // L2 mean-reversion SELL
              + lambda * neg_score;

    // HOLD floor: slightly higher in L2 (require stronger signal for sideways scalps)
    let s_hold = if is_l2 { 0.50 } else { 0.45 } + 0.25 * (1.0 - score.abs().tanh());

    // ── Softmax → probabilities ────────────────────────────────
    let tau = 0.35; // temperature (was 0.5 — sharper decisions now)
    let max_s = s_buy.max(s_sell).max(s_hold);
    let exp_buy  = ((s_buy  - max_s) / tau).exp();
    let exp_sell = ((s_sell - max_s) / tau).exp();
    let exp_hold = ((s_hold - max_s) / tau).exp();
    let sum = exp_buy + exp_sell + exp_hold;

    let p_buy  = exp_buy  / sum;
    let p_sell = exp_sell / sum;
    let p_hold = exp_hold / sum;

    let (action, confidence) = if p_buy >= p_sell && p_buy >= p_hold {
        ("BUY", p_buy)
    } else if p_sell >= p_buy && p_sell >= p_hold {
        ("SELL", p_sell)
    } else {
        ("HOLD", p_hold)
    };

    // ── Dynamic threshold ──────────────────────────────────────
    // Fluctuates with EMA tunnels + RSI — NOT set in stone.
    //
    // Base: 0.58 (from .env ENTRY_THRESHOLD)
    // Strong trend (3+):    -0.08  (easier to enter trending markets)
    // Full alignment (5):   -0.04 extra
    // RSI sweet spot [45,65]: -0.05 (ideal buy zone)
    // Good momentum (>0.2): -0.03
    // RSI overbought (>75): +0.10 (don't chase)
    // RSI oversold (<30):   +0.05 (wait for bounce confirmation)
    // Weak/negative trend:  +0.05 (harder in choppy)
    let base_thresh = entry_threshold;
    let mut thresh_adj: f64 = 0.0;

    // Trend adjustments
    if f.trend_score >= 3 { thresh_adj -= 0.08; }
    if f.trend_score >= 5 { thresh_adj -= 0.04; } // extra for full alignment
    if f.trend_score <= 0 { thresh_adj += 0.05; }

    // RSI adjustments
    if f.rsi >= 45.0 && f.rsi <= 65.0 { thresh_adj -= 0.05; }
    if f.rsi > 75.0 { thresh_adj += 0.10; }
    if f.rsi < 30.0 { thresh_adj += 0.05; }

    // Momentum adjustment
    if f.momentum_score > 0.2 { thresh_adj -= 0.03; }

    // ADX adjustment: low trend strength = choppy/noisy, raise threshold
    if f.adx > 0.0 && f.adx < 15.0 { thresh_adj += 0.05; }

    // L2 mean-reversion: lower threshold when Z-score is extreme (good mean-rev setup)
    if is_l2 {
        thresh_adj -= 0.05; // L2 baseline easier (small scalps are the point)
        if f.zscore.abs() > 1.5 { thresh_adj -= 0.05; } // extreme Z = strong mean-rev signal
    }

    let dynamic_threshold = (base_thresh + thresh_adj).clamp(0.30, 0.75);

    // Log when HOLD is caused by missing data (not market uncertainty)
    if action == "HOLD" && !reasons.is_empty() && strict_enabled() {
        tracing::warn!(
            "[STRICT] HOLD due to missing data: {:?} — not a market signal",
            reasons,
        );
    }

    ActionScores {
        s_buy: safe_float(s_buy, 0.0),
        s_sell: safe_float(s_sell, 0.0),
        s_hold: safe_float(s_hold, 0.0),
        action: action.to_string(),
        confidence: safe_float(confidence, 0.0),
        crash: safe_float(crash, 0.0),
        hunt: safe_float(hunt, 0.0),
        buy_pressure: safe_float(buy_pressure, 0.0),
        sell_pressure: safe_float(sell_pressure, 0.0),
        trend_buy: safe_float(trend_buy, 0.0),
        momentum_buy: safe_float(momentum_buy, 0.0),
        mean_rev_buy: safe_float(mr_buy, 0.0),
        mean_rev_sell: safe_float(mr_sell, 0.0),
        is_l2,
        dynamic_threshold: safe_float(dynamic_threshold, entry_threshold),
        reasons,
    }
}

fn mean_std(slice: &[f64]) -> (f64, f64) {
    if slice.is_empty() {
        return (0.0, 0.0);
    }
    let mean = slice.iter().sum::<f64>() / slice.len() as f64;
    let mut var = 0.0;
    for x in slice {
        let d = x - mean;
        var += d * d;
    }
    var /= slice.len() as f64;
    (mean, var.sqrt())
}

/// EMA that returns None when there's not enough data (empty prices).
/// With < period candles, uses simple average (valid approximation).
/// Only returns None for truly missing data (0 candles).
fn ema_opt(prices: &[f64], period: usize) -> Option<f64> {
    if prices.is_empty() {
        return None;
    }
    if prices.len() < period {
        // SMA fallback — valid approximation, better than None
        return Some(prices.iter().sum::<f64>() / prices.len() as f64);
    }
    let k = 2.0 / (period as f64 + 1.0);
    // Seed with SMA of first 'period' prices (standard EMA initialization)
    let mut ema_val = prices[0..period].iter().sum::<f64>() / period as f64;
    for p in &prices[period..] {
        ema_val = (p * k) + (ema_val * (1.0 - k));
    }
    Some(ema_val)
}

fn rsi(prices: &[f64], period: usize) -> f64 {
    if prices.len() <= period {
        return 50.0;
    }
    // Initial average gains/losses over the first 'period' changes
    let mut gains = 0.0;
    let mut losses = 0.0;
    for i in 1..=period {
        let delta = prices[i] - prices[i - 1];
        if delta > 0.0 {
            gains += delta;
        } else {
            losses += delta.abs();
        }
    }
    let mut avg_gain = gains / period as f64;
    let mut avg_loss = losses / period as f64;

    // Wilder's smoothing for remaining prices
    for i in (period + 1)..prices.len() {
        let delta = prices[i] - prices[i - 1];
        let gain = if delta > 0.0 { delta } else { 0.0 };
        let loss = if delta < 0.0 { delta.abs() } else { 0.0 };
        avg_gain = (avg_gain * (period - 1) as f64 + gain) / period as f64;
        avg_loss = (avg_loss * (period - 1) as f64 + loss) / period as f64;
    }

    if avg_loss == 0.0 {
        return 100.0;
    }
    let rs = avg_gain / avg_loss;
    100.0 - (100.0 / (1.0 + rs))
}

fn macd_hist(prices: &[f64]) -> f64 {
    if prices.len() < 26 + 9 {
        return 0.0;
    }
    // Compute MACD line series incrementally, then EMA(9) for signal line
    let k12 = 2.0 / 13.0;
    let k26 = 2.0 / 27.0;
    let k9 = 2.0 / 10.0;

    let mut ema12 = prices[0];
    let mut ema26 = prices[0];
    let mut signal = 0.0;
    let mut signal_seeded = false;

    for (i, &p) in prices.iter().enumerate().skip(1) {
        ema12 = p * k12 + ema12 * (1.0 - k12);
        ema26 = p * k26 + ema26 * (1.0 - k26);
        let macd_line = ema12 - ema26;
        if i >= 26 && !signal_seeded {
            signal = macd_line;
            signal_seeded = true;
        } else if signal_seeded {
            signal = macd_line * k9 + signal * (1.0 - k9);
        }
    }

    let final_macd = ema12 - ema26;
    final_macd - signal // histogram = MACD line - signal line
}

fn atr(highs: &[f64], lows: &[f64], closes: &[f64], period: usize) -> f64 {
    if closes.len() < period + 1 || highs.is_empty() || lows.is_empty() {
        if !highs.is_empty() && !lows.is_empty() {
            return highs[highs.len() - 1] - lows[lows.len() - 1];
        }
        return 0.0;
    }
    let mut trs = Vec::with_capacity(closes.len().saturating_sub(1));
    for i in 1..closes.len() {
        let tr = (highs[i] - lows[i])
            .max((highs[i] - closes[i - 1]).abs())
            .max((lows[i] - closes[i - 1]).abs());
        trs.push(tr);
    }
    let start = trs.len().saturating_sub(period);
    let (mean, _) = mean_std(&trs[start..]);
    mean
}

/// Bollinger bands that return None when there's not enough data.
fn bollinger_opt(prices: &[f64], period: usize, mult: f64) -> Option<(f64, f64, f64)> {
    if prices.len() < period {
        return None;
    }
    let window = &prices[prices.len() - period..];
    let (mean, std) = mean_std(window);
    Some((mean + mult * std, mean, mean - mult * std))
}

fn zscore(prices: &[f64], period: usize) -> f64 {
    if prices.len() < period {
        return 0.0;
    }
    let window = &prices[prices.len() - period..];
    let (mean, std) = mean_std(window);
    if std == 0.0 {
        0.0
    } else {
        (prices[prices.len() - 1] - mean) / std
    }
}

fn corr_tail(a: &[f64], b: &[f64], tail: usize) -> f64 {
    if a.len() < 3 || b.len() < 3 {
        return 0.0;
    }
    let n = std::cmp::min(tail, std::cmp::min(a.len(), b.len()));
    let a = &a[a.len() - n..];
    let b = &b[b.len() - n..];
    if a.iter().all(|x| (x - a[0]).abs() < 1e-12) || b.iter().all(|x| (x - b[0]).abs() < 1e-12) {
        return 0.0;
    }
    let (ma, _) = mean_std(a);
    let (mb, _) = mean_std(b);
    let mut num = 0.0;
    let mut da = 0.0;
    let mut db = 0.0;
    for i in 0..n {
        let xa = a[i] - ma;
        let xb = b[i] - mb;
        num += xa * xb;
        da += xa * xa;
        db += xb * xb;
    }
    if da <= 0.0 || db <= 0.0 {
        return 0.0;
    }
    let c = num / (da.sqrt() * db.sqrt());
    c.clamp(-1.0, 1.0)
}

fn momentum_metrics(closes: &[f64]) -> (f64, i32, i32, i32, i32) {
    if closes.len() < 2 {
        return (0.0, 0, 0, 0, 0);
    }

    let levels = [5usize, 4, 3, 2, 1];
    let weights = [0.35, 0.25, 0.20, 0.12, 0.08];
    let mut level_scores = [0.0_f64; 5];
    let mut trend_strength: i32 = 0;

    for (li, level) in levels.iter().enumerate() {
        if closes.len() < level + 1 {
            continue;
        }
        let start = closes.len() - level;
        let mut ups = 0;
        let mut downs = 0;
        for idx in start..closes.len() {
            if idx == 0 {
                continue;
            }
            let a = closes[idx - 1];
            let b = closes[idx];
            if b > a {
                ups += 1;
            } else if b < a {
                downs += 1;
            }
        }
        let diff = ups as i32 - downs as i32;
        level_scores[li] = diff as f64 / (*level as f64);
        if trend_strength == 0 {
            if ups == *level {
                trend_strength = *level as i32;
            } else if downs == *level {
                trend_strength = -(*level as i32);
            }
        }
    }

    let mut momentum_score = 0.0;
    for i in 0..5 {
        momentum_score += weights[i] * level_scores[i];
    }

    // Consecutive candles
    let mut cons_up = 0;
    let mut cons_down = 0;
    for idx in (1..closes.len()).rev() {
        let a = closes[idx - 1];
        let b = closes[idx];
        if b > a {
            if cons_down > 0 {
                break;
            }
            cons_up += 1;
        } else if b < a {
            if cons_up > 0 {
                break;
            }
            cons_down += 1;
        } else {
            break;
        }
        if cons_up >= 5 || cons_down >= 5 {
            break;
        }
    }
    let consecutive = if cons_up > 0 { cons_up } else { -cons_down };

    // 3-up / 3-down legacy
    let mut tud_up = 0;
    let mut tud_down = 0;
    if closes.len() >= 4 {
        let start = closes.len() - 3;
        for idx in start..closes.len() {
            let a = closes[idx - 1];
            let b = closes[idx];
            if b > a {
                tud_up += 1;
            } else if b < a {
                tud_down += 1;
            }
        }
    }

    (
        momentum_score,
        trend_strength,
        consecutive,
        tud_up,
        tud_down,
    )
}

/// Hurst Exponent via Rescaled Range (R/S) method.
/// H > 0.5 = trending (momentum), H < 0.5 = mean-reverting, H ≈ 0.5 = random walk.
fn hurst_exponent(closes: &[f64]) -> f64 {
    if closes.len() < 20 {
        return 0.5; // not enough data
    }
    // Compute log returns
    let mut log_rets = Vec::with_capacity(closes.len() - 1);
    for i in 1..closes.len() {
        if closes[i - 1] > 0.0 && closes[i] > 0.0 {
            log_rets.push((closes[i] / closes[i - 1]).ln());
        }
    }
    if log_rets.len() < 20 {
        return 0.5;
    }

    // R/S for multiple sub-period sizes
    let mut log_ns = Vec::new();
    let mut log_rs = Vec::new();
    let sizes: &[usize] = &[8, 16, 32, 64, 128];

    for &n in sizes {
        if n > log_rets.len() {
            break;
        }
        let num_blocks = log_rets.len() / n;
        if num_blocks == 0 {
            continue;
        }
        let mut rs_sum = 0.0;
        let mut valid_blocks = 0;
        for b in 0..num_blocks {
            let block = &log_rets[b * n..(b + 1) * n];
            let (mean, std) = mean_std(block);
            if std < 1e-15 {
                continue;
            }
            // Cumulative deviation from mean
            let mut cum_dev = Vec::with_capacity(n);
            let mut running = 0.0;
            for &r in block {
                running += r - mean;
                cum_dev.push(running);
            }
            let range = cum_dev.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
                - cum_dev.iter().cloned().fold(f64::INFINITY, f64::min);
            rs_sum += range / std;
            valid_blocks += 1;
        }
        if valid_blocks > 0 {
            let avg_rs = rs_sum / valid_blocks as f64;
            if avg_rs > 0.0 {
                log_ns.push((n as f64).ln());
                log_rs.push(avg_rs.ln());
            }
        }
    }

    // Linear regression: slope of log(R/S) vs log(n) = Hurst exponent
    if log_ns.len() < 2 {
        return 0.5;
    }
    let n_pts = log_ns.len() as f64;
    let sum_x: f64 = log_ns.iter().sum();
    let sum_y: f64 = log_rs.iter().sum();
    let sum_xy: f64 = log_ns.iter().zip(log_rs.iter()).map(|(x, y)| x * y).sum();
    let sum_x2: f64 = log_ns.iter().map(|x| x * x).sum();
    let denom = n_pts * sum_x2 - sum_x * sum_x;
    if denom.abs() < 1e-15 {
        return 0.5;
    }
    let slope = (n_pts * sum_xy - sum_x * sum_y) / denom;
    slope.clamp(0.0, 1.0)
}

/// Shannon Entropy of return distribution, normalized to [0, 1].
/// Low entropy = predictable patterns. High entropy = noise/chaos.
fn shannon_entropy(closes: &[f64]) -> f64 {
    if closes.len() < 10 {
        return 0.5;
    }
    let mut rets = Vec::with_capacity(closes.len() - 1);
    for i in 1..closes.len() {
        let denom = closes[i - 1].max(1e-9);
        rets.push((closes[i] - closes[i - 1]) / denom);
    }
    if rets.len() < 10 {
        return 0.5;
    }

    // Find min/max for binning
    let min_r = rets.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_r = rets.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = max_r - min_r;
    if range < 1e-15 {
        return 0.0; // all returns identical = perfectly predictable
    }

    // Bin into 10 buckets
    let n_bins = 10usize;
    let mut bins = vec![0u32; n_bins];
    for &r in &rets {
        let idx = ((r - min_r) / range * (n_bins as f64 - 1.0)).round() as usize;
        bins[idx.min(n_bins - 1)] += 1;
    }

    // Compute entropy: H = -Σ p_i * log2(p_i)
    let total = rets.len() as f64;
    let mut entropy = 0.0;
    for &count in &bins {
        if count > 0 {
            let p = count as f64 / total;
            entropy -= p * p.log2();
        }
    }
    // Normalize by max entropy (log2 of n_bins)
    let max_entropy = (n_bins as f64).log2();
    if max_entropy > 0.0 {
        (entropy / max_entropy).clamp(0.0, 1.0)
    } else {
        0.5
    }
}

/// Autocorrelation at lag 1. Positive = momentum persistence, Negative = mean-reverting.
fn autocorr_lag1(closes: &[f64]) -> f64 {
    if closes.len() < 12 {
        return 0.0;
    }
    let mut rets = Vec::with_capacity(closes.len() - 1);
    for i in 1..closes.len() {
        let denom = closes[i - 1].max(1e-9);
        rets.push((closes[i] - closes[i - 1]) / denom);
    }
    if rets.len() < 10 {
        return 0.0;
    }
    let (mean, std) = mean_std(&rets);
    if std < 1e-15 {
        return 0.0;
    }
    let n = rets.len();
    let mut cov = 0.0;
    for i in 1..n {
        cov += (rets[i] - mean) * (rets[i - 1] - mean);
    }
    cov /= (n - 1) as f64;
    (cov / (std * std)).clamp(-1.0, 1.0)
}

/// ADX (Average Directional Index) — trend strength 0-100.
/// ADX > 25 = trending, ADX < 20 = choppy/ranging.
fn compute_adx(highs: &[f64], lows: &[f64], closes: &[f64], period: usize) -> f64 {
    let n = closes.len();
    if n < period + 1 || highs.len() < n || lows.len() < n {
        return 0.0;
    }
    let mut plus_dm = Vec::with_capacity(n - 1);
    let mut minus_dm = Vec::with_capacity(n - 1);
    let mut tr = Vec::with_capacity(n - 1);
    for i in 1..n {
        let up = highs[i] - highs[i - 1];
        let down = lows[i - 1] - lows[i];
        plus_dm.push(if up > down && up > 0.0 { up } else { 0.0 });
        minus_dm.push(if down > up && down > 0.0 { down } else { 0.0 });
        tr.push(
            (highs[i] - lows[i])
                .max((highs[i] - closes[i - 1]).abs())
                .max((lows[i] - closes[i - 1]).abs()),
        );
    }
    if plus_dm.len() < period {
        return 0.0;
    }
    let mut s_pdm: f64 = plus_dm[..period].iter().sum();
    let mut s_mdm: f64 = minus_dm[..period].iter().sum();
    let mut s_tr: f64 = tr[..period].iter().sum();
    let mut dx_vals = Vec::new();
    for i in period..plus_dm.len() {
        s_pdm = s_pdm - s_pdm / period as f64 + plus_dm[i];
        s_mdm = s_mdm - s_mdm / period as f64 + minus_dm[i];
        s_tr = s_tr - s_tr / period as f64 + tr[i];
        if s_tr > 0.0 {
            let pdi = 100.0 * s_pdm / s_tr;
            let mdi = 100.0 * s_mdm / s_tr;
            let sum = pdi + mdi;
            if sum > 0.0 {
                dx_vals.push(100.0 * (pdi - mdi).abs() / sum);
            }
        }
    }
    if dx_vals.is_empty() {
        return 0.0;
    }
    if dx_vals.len() < period {
        return dx_vals.iter().sum::<f64>() / dx_vals.len() as f64;
    }
    let mut adx = dx_vals[..period].iter().sum::<f64>() / period as f64;
    for i in period..dx_vals.len() {
        adx = (adx * (period as f64 - 1.0) + dx_vals[i]) / period as f64;
    }
    adx.clamp(0.0, 100.0)
}

/// Reusable buffers for feature computation — avoids heap allocation per tick.
#[derive(Default)]
pub struct FeatureBuffers {
    pub closes: Vec<f64>,
    pub highs: Vec<f64>,
    pub lows: Vec<f64>,
    pub volumes: Vec<f64>,
}

impl FeatureBuffers {
    pub fn new() -> Self {
        Self {
            closes: Vec::with_capacity(512),
            highs: Vec::with_capacity(512),
            lows: Vec::with_capacity(512),
            volumes: Vec::with_capacity(512),
        }
    }

    fn fill_from(&mut self, candles: &[Candle]) {
        self.closes.clear();
        self.highs.clear();
        self.lows.clear();
        self.volumes.clear();
        for c in candles {
            self.closes.push(safe_float(c.close, 0.0));
            self.highs.push(safe_float(c.high, 0.0));
            self.lows.push(safe_float(c.low, 0.0));
            self.volumes.push(safe_float(c.volume, 0.0));
        }
    }
}

pub fn compute_features_buffered(
    candles: &[Candle],
    orderbook: Option<&OrderbookMetrics>,
    trade_flow: Option<&TradeFlow>,
    book_reversal: Option<&BookReversalMetrics>,
    ref_closes: Option<&HashMap<String, VecDeque<f64>>>,
    bufs: &mut FeatureBuffers,
) -> Features {
    if candles.is_empty() {
        return Features {
            rsi: 50.0,
            buy_ratio: 0.5,
            sell_ratio: 0.5,
            ..Features::default()
        };
    }

    bufs.fill_from(candles);
    let closes = &bufs.closes;
    let highs = &bufs.highs;
    let lows = &bufs.lows;
    let volumes = &bufs.volumes;

    let price = *closes.last().unwrap_or(&0.0);

    let ema9_v = ema_opt(&closes, 9);
    let ema21_v = ema_opt(&closes, 21);
    let ema50_v = ema_opt(&closes, 50);
    let ema55_v = ema_opt(&closes, 55);
    let ema200_v = ema_opt(&closes, 200);
    let rsi14 = rsi(&closes, 14);
    let macd_h = macd_hist(&closes);
    let atr14 = atr(&highs, &lows, &closes, 14);
    let bb_opt = bollinger_opt(&closes, 20, 2.0);
    let z = zscore(&closes, 30);

    let mut rets = Vec::new();
    if closes.len() >= 2 {
        for i in 1..closes.len() {
            let denom = closes[i - 1].max(1e-9);
            rets.push((closes[i] - closes[i - 1]) / denom);
        }
    }

    let quant_vol = if rets.len() >= 5 {
        let start = rets.len().saturating_sub(30);
        let (_, std) = mean_std(&rets[start..]);
        std
    } else {
        0.0
    };

    let quant_momo = if closes.len() >= 10 && closes[closes.len() - 10] > 0.0 {
        (closes[closes.len() - 1] / closes[closes.len() - 10]) - 1.0
    } else {
        0.0
    };

    let quant_regime = match (ema21_v, ema50_v) {
        (Some(e21), Some(e50)) if price > 0.0 => (e21 - e50) / price,
        _ => 0.0,
    };
    let quant_meanrev = -z.abs();

    let mut vol_ratio = 1.0;
    if volumes.len() >= 2 {
        let lb = std::cmp::min(20, volumes.len() - 1);
        let start = volumes.len() - lb - 1;
        let slice = &volumes[start..volumes.len() - 1];
        let (avg, _) = mean_std(slice);
        if avg > 0.0 {
            vol_ratio = volumes[volumes.len() - 1] / avg;
        }
    }

    let atr_pct = if price > 0.0 { atr14 / price } else { 0.0 };
    let quant_risk = (atr_pct / 0.05).clamp(0.0, 1.0);
    let stability = 1.0 - quant_risk;
    let vol_norm = (vol_ratio.min(2.0) / 2.0).max(0.0);
    let quant_liquidity = 0.5 * vol_norm + 0.5 * stability;

    let mut quant_corr_btc = 0.0;
    let mut quant_corr_eth = 0.0;
    if let Some(refs) = ref_closes {
        if let Some(btc) = refs.get("BTC") {
            if btc.len() >= 2 {
                let mut btc_ret = Vec::new();
                for i in 1..btc.len() {
                    let denom = btc[i - 1].max(1e-9);
                    btc_ret.push((btc[i] - btc[i - 1]) / denom);
                }
                quant_corr_btc = corr_tail(&rets, &btc_ret, 60);
            }
        }
        if let Some(eth) = refs.get("ETH") {
            if eth.len() >= 2 {
                let mut eth_ret = Vec::new();
                for i in 1..eth.len() {
                    let denom = eth[i - 1].max(1e-9);
                    eth_ret.push((eth[i] - eth[i - 1]) / denom);
                }
                quant_corr_eth = corr_tail(&rets, &eth_ret, 60);
            }
        }
    }

    // Trend scoring: compare EMAs when available, neutral (0) when missing
    let mut trend_score: i32 = 0;
    let e9 = ema9_v.unwrap_or(0.0);
    let e21 = ema21_v.unwrap_or(0.0);
    let e50 = ema50_v.unwrap_or(0.0);
    if ema9_v.is_some() && ema21_v.is_some() {
        trend_score += if e9 > e21 { 1 } else { -1 };
    }
    if ema21_v.is_some() && ema50_v.is_some() {
        trend_score += if e21 > e50 { 1 } else { -1 };
    }
    if ema9_v.is_some() { trend_score += if price > e9 { 1 } else { -1 }; }
    if ema21_v.is_some() { trend_score += if price > e21 { 1 } else { -1 }; }
    if ema50_v.is_some() { trend_score += if price > e50 { 1 } else { -1 }; }

    let spread_pct = orderbook.map(|o| o.spread_pct).unwrap_or(0.0);
    let book_imbalance = orderbook.map(|o| o.imbalance).unwrap_or(0.0);
    let buy_ratio = trade_flow
        .map(|t| t.buy_ratio)
        .unwrap_or(0.5)
        .clamp(0.0, 1.0);
    let sell_ratio = trade_flow
        .map(|t| t.sell_ratio)
        .unwrap_or(0.5)
        .clamp(0.0, 1.0);
    let flow_imbalance = (buy_ratio - 0.5) * 2.0;
    let flow_book_div = flow_imbalance - book_imbalance;
    let flow_book_div_abs = flow_book_div.abs();
    let (momentum_score, trend_strength, consecutive_candles, tud_up, tud_down) =
        momentum_metrics(&closes);

    let trend_norm = (trend_score as f64 / 5.0).clamp(-1.0, 1.0);
    let momo_norm = (momentum_score / 0.5).clamp(-1.0, 1.0);
    let vol_penalty = (quant_vol.abs() / 0.03).clamp(0.0, 1.0);
    let atr_penalty = if price > 0.0 {
        (atr14 / (price * 0.05)).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let market_state = clamp01(0.5 + 0.2 * trend_norm + 0.2 * momo_norm - 0.2 * vol_penalty - 0.1 * atr_penalty);


    // Quantum math indicators
    let hurst = hurst_exponent(&closes);
    let entropy = shannon_entropy(&closes);
    let autocorr = autocorr_lag1(&closes);
    let adx_val = compute_adx(&highs, &lows, &closes, 14);

    let (book_trend, book_reversal_flag, book_reversal_dir, book_strength, book_avg_imb) =
        if let Some(br) = book_reversal {
            (
                br.trend,
                br.reversal,
                br.reversal_dir,
                br.strength,
                br.avg_imb,
            )
        } else {
            (0, false, 0, 0.0, 0.0)
        };

    Features {
        price: safe_float(price, 0.0),
        ema9: ema9_v.map(|v| safe_float(v, 0.0)),
        ema21: ema21_v.map(|v| safe_float(v, 0.0)),
        ema50: ema50_v.map(|v| safe_float(v, 0.0)),
        ema55: ema55_v.map(|v| safe_float(v, 0.0)),
        ema200: ema200_v.map(|v| safe_float(v, 0.0)),
        rsi: safe_float(rsi14, 50.0),
        macd_hist: safe_float(macd_h, 0.0),
        atr: safe_float(atr14, 0.0),
        bb_upper: bb_opt.map(|(u, _, _)| safe_float(u, 0.0)),
        bb_lower: bb_opt.map(|(_, _, l)| safe_float(l, 0.0)),
        zscore: safe_float(z, 0.0),
        quant_vol: safe_float(quant_vol, 0.0),
        quant_momo: safe_float(quant_momo, 0.0),
        quant_regime: safe_float(quant_regime, 0.0),
        quant_meanrev: safe_float(quant_meanrev, 0.0),
        quant_liquidity: safe_float(quant_liquidity, 0.0),
        quant_risk: safe_float(quant_risk, 0.0),
        quant_corr_btc: safe_float(quant_corr_btc, 0.0),
        quant_corr_eth: safe_float(quant_corr_eth, 0.0),
        vol_ratio: safe_float(vol_ratio, 0.0),
        trend_score,
        market_state: safe_float(market_state, 0.5),
        momentum_score: safe_float(momentum_score, 0.0),
        trend_strength,
        consecutive_candles,
        tud_up,
        tud_down,
        spread_pct,
        book_imbalance,
        buy_ratio,
        sell_ratio,
        flow_imbalance: safe_float(flow_imbalance, 0.0),
        flow_book_div: safe_float(flow_book_div, 0.0),
        flow_book_div_abs: safe_float(flow_book_div_abs, 0.0),
        book_trend,
        book_reversal: book_reversal_flag,
        book_reversal_dir,
        book_strength: safe_float(book_strength, 0.0),
        book_avg_imb: safe_float(book_avg_imb, 0.0),
        hurst_exp: safe_float(hurst, 0.5),
        shannon_entropy: safe_float(entropy, 0.5),
        autocorr_lag1: safe_float(autocorr, 0.0),
        adx: safe_float(adx_val, 0.0),
    }
}
