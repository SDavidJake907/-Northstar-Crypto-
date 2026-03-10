//! Gate checks for trade entry decisions.

/// Gate check features extracted from the snapshot.
pub struct GateFeatures {
    pub quant_liquidity: f64,
    pub vol_ratio: f64,
    pub spread_pct: f64,
    pub buy_slip_pct_100k: Option<f64>,
    pub buy_fill_100k: Option<f64>,
    pub atr_ratio: f64,
}

impl GateFeatures {
    pub fn from_json(feats: &serde_json::Value) -> Self {
        let f = |key: &str, default: f64| -> f64 {
            feats.get(key).and_then(|v| v.as_f64()).unwrap_or(default)
        };
        Self {
            quant_liquidity: f("quant_liquidity", 0.0),
            vol_ratio: f("vol_ratio", 0.0),
            spread_pct: f("spread_pct", 0.0),
            buy_slip_pct_100k: feats.get("buy_slip_pct_100k").and_then(|v| v.as_f64()),
            buy_fill_100k: feats.get("buy_fill_100k").and_then(|v| v.as_f64()),
            // Use ATR as % of price (not raw $), so BTC ATR=$500 at $100k = 0.5%
            atr_ratio: {
                let raw_atr = f("atr", 0.0);
                let price = f("price", 1.0);
                if price > 0.0 { (raw_atr / price) * 100.0 } else { 0.0 }
            },
        }
    }
}

/// Gate check configuration.
pub struct GateConfig {
    pub min_liquidity: f64,
    pub min_vol_ratio: f64,
    pub max_spread_pct: f64,
    pub max_slip_pct: f64,
    pub min_fill_100k: f64,
    pub require_liq_metrics: bool,
    pub max_atr_ratio: f64,
    /// Enable composite scoring — only allow trades with score > 0.0
    pub use_scoring: bool,
    /// Scales the composite score (1.0 = normal, >1.0 = more selective)
    pub profit_target_multiplier: f64,
}

impl Default for GateConfig {
    fn default() -> Self {
        Self {
            min_liquidity: 30_000.0,      // lowered 50K→30K: lets more coins through
            min_vol_ratio: 1.2,
            max_spread_pct: 0.10,         // tightened 0.15→0.10: better fill quality
            max_slip_pct: 0.20,           // tightened 0.25→0.20: better fill quality
            min_fill_100k: 100_000.0,
            require_liq_metrics: false,
            max_atr_ratio: 2.5,
            use_scoring: false,           // off by default — safe to ship
            profit_target_multiplier: 1.0,
        }
    }
}

/// Compute a continuous trade quality score in [0.0, 1.0] × multiplier.
/// Higher = better quality setup. Used when gate.use_scoring = true.
pub fn score_trade(feats: &GateFeatures, config: &GateConfig) -> f64 {
    // Each sub-score in [0.0, 1.0]
    let liq_score = (feats.quant_liquidity / config.min_liquidity.max(1.0))
        .min(2.0) / 2.0;

    let vol_score = (feats.vol_ratio / config.min_vol_ratio.max(0.01))
        .min(2.0) / 2.0;

    let spread_score = if config.max_spread_pct > 0.0 {
        1.0 - (feats.spread_pct / config.max_spread_pct).clamp(0.0, 1.0)
    } else {
        1.0
    };

    let slip_score = feats.buy_slip_pct_100k
        .map(|s| {
            if config.max_slip_pct > 0.0 {
                1.0 - (s / config.max_slip_pct).clamp(0.0, 1.0)
            } else {
                1.0
            }
        })
        .unwrap_or(0.5); // unknown slip → neutral

    let atr_score = if config.max_atr_ratio > 0.0 {
        1.0 - (feats.atr_ratio / config.max_atr_ratio).clamp(0.0, 1.0)
    } else {
        1.0
    };

    // Weighted composite: liquidity + volume matter most
    let raw = liq_score    * 0.25
            + vol_score    * 0.25
            + spread_score * 0.20
            + slip_score   * 0.20
            + atr_score    * 0.10;

    raw * config.profit_target_multiplier
}

/// Check if a trade passes all gate checks (liquidity, volume, spread, slippage, fill).
///
/// Returns (passed, list_of_rejection_reasons).
pub fn passes_gate_checks(
    feats: &GateFeatures,
    gate: &GateConfig,
    tier: &str,
    slip_mult: f64,
) -> (bool, Vec<String>) {
    let mut reasons: Vec<String> = Vec::new();

    let max_spread = gate.max_spread_pct * slip_mult;
    let max_slip = gate.max_slip_pct * slip_mult;

    // Tier-based fill multiplier
    let tier_fill_mult = match tier {
        "large" => 1.0,
        "mid" => 0.5,
        "meme" => 0.2,
        _ => 0.5,
    };
    let min_fill_adjusted = gate.min_fill_100k * tier_fill_mult;

    if feats.quant_liquidity < gate.min_liquidity {
        reasons.push("liq_low".into());
    }
    if feats.vol_ratio < gate.min_vol_ratio {
        reasons.push("vol_low".into());
    }
    if feats.spread_pct > max_spread {
        reasons.push(format!(
            "spread_high({:.2}>{:.2})",
            feats.spread_pct, max_spread
        ));
    }

    if gate.require_liq_metrics {
        if feats.buy_slip_pct_100k.is_none() {
            reasons.push("slip_missing".into());
        }
        if feats.buy_fill_100k.is_none() {
            reasons.push("fill_missing".into());
        }
    }

    if let Some(slip) = feats.buy_slip_pct_100k {
        if slip > max_slip {
            reasons.push(format!("slip_high({slip:.2}>{max_slip:.2})"));
        }
    }
    if let Some(fill) = feats.buy_fill_100k {
        if fill < min_fill_adjusted {
            reasons.push(format!("illiquid({fill:.0}<{min_fill_adjusted:.0})"));
        }
    }

    if gate.max_atr_ratio > 0.0 && feats.atr_ratio > gate.max_atr_ratio {
        reasons.push(format!(
            "atr_spike({:.2}>{:.2})",
            feats.atr_ratio, gate.max_atr_ratio
        ));
    }

    // Optional scoring layer — only runs when explicitly enabled
    if gate.use_scoring && reasons.is_empty() {
        let score = score_trade(feats, gate);
        if score <= 0.0 {
            reasons.push(format!("score_low({score:.2})"));
        }
    }

    (reasons.is_empty(), reasons)
}
