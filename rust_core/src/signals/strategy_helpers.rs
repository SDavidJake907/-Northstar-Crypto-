use super::strategy::StrategyConfig;
use serde_json::Value;
use std::collections::HashMap;

pub fn compute_softmax_weights(
    cfg: &StrategyConfig,
    features_map: &HashMap<String, Value>,
) -> HashMap<String, f64> {
    let mut scores: Vec<(String, f64)> = Vec::new();
    for (symbol, feats) in features_map {
        if symbol.starts_with("__") {
            continue;
        }
        let m = feats
            .get("momentum_score")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let t = feats
            .get("trend_score")
            .and_then(|v| v.as_i64())
            .map(|v| v as f64)
            .unwrap_or(0.0);
        let v = feats
            .get("vol_ratio")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0)
            .max(1e-4);
        let score = cfg.alpha * m + cfg.beta * t + cfg.gamma / (v + 1e-4);
        scores.push((symbol.clone(), score));
    }

    if scores.is_empty() {
        return HashMap::new();
    }

    let tau = cfg.tau.max(1e-3);
    let max_score = scores
        .iter()
        .map(|(_, s)| *s)
        .fold(f64::NEG_INFINITY, f64::max);

    let mut exp_sum = 0.0;
    let mut exp_values: Vec<f64> = Vec::with_capacity(scores.len());
    for (_, score) in &scores {
        let exp = ((score - max_score) / tau).exp();
        exp_values.push(exp);
        exp_sum += exp;
    }

    let norm_sum = exp_sum.max(1e-12);
    let mut weights = HashMap::new();
    for ((symbol, _), exp) in scores.iter().zip(exp_values.iter()) {
        weights.insert(symbol.clone(), exp / norm_sum);
    }

    cap_and_normalize(weights, cfg.w_max)
}

pub fn cap_and_normalize(mut weights: HashMap<String, f64>, cap: f64) -> HashMap<String, f64> {
    if cap.is_sign_negative() || cap.is_nan() {
        return weights;
    }
    let cap = cap.max(0.0);
    let mut sum = 0.0;
    for value in weights.values_mut() {
        if *value > cap {
            *value = cap;
        }
        sum += *value;
    }
    if sum == 0.0 {
        weights.clear();
        return weights;
    }
    for value in weights.values_mut() {
        *value /= sum;
    }
    weights
}

