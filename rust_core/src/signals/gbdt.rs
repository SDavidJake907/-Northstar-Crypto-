//! GBDT (Gradient Boosted Decision Trees) — learns from trade outcomes.
//!
//! Uses forust-ml (pure Rust, XGBoost-compatible) to predict P(profitable trade)
//! from the 44-field Features struct. Blends with existing math scoring.
//!
//! 3-phase warm-up:
//!   Cold  (<100 samples): math-only, collect feature snapshots
//!   Warm  (100–500):      30% GBDT + 70% math
//!   Hot   (500+):         60% GBDT + 40% math

use crate::signals::features::Features;
use forust_ml::{GradientBooster, Matrix};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};

pub type SharedGbdt = std::sync::Arc<tokio::sync::RwLock<GbdtEngine>>;

// ── Feature Names (deterministic order, matches features_to_vec) ──────

pub const NUM_FEATURES: usize = 44;

pub const FEATURE_NAMES: [&str; NUM_FEATURES] = [
    "price", "ema9", "ema21", "ema50", "ema55", "ema200",
    "rsi", "macd_hist", "atr", "bb_upper", "bb_lower",
    "zscore", "quant_vol", "quant_momo", "quant_regime", "quant_meanrev",
    "quant_liquidity", "quant_risk", "quant_corr_btc", "quant_corr_eth",
    "vol_ratio", "trend_score", "market_state", "momentum_score",
    "trend_strength", "consecutive_candles", "tud_up", "tud_down",
    "spread_pct", "book_imbalance", "buy_ratio", "sell_ratio",
    "flow_imbalance", "flow_book_div", "flow_book_div_abs", "book_trend",
    "book_reversal", "book_reversal_dir", "book_strength", "book_avg_imb",
    "hurst_exp", "shannon_entropy", "autocorr_lag1", "adx",
];

// ── Training Sample ───────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainingSample {
    pub features: Vec<f64>,
    pub pnl_percent: f64,
    pub label: u8, // 1 = WIN, 0 = LOSS/BREAKEVEN
}

// ── GBDT Engine ───────────────────────────────────────────────────────

pub struct GbdtEngine {
    model: Option<GradientBooster>,
    sample_count: usize,
    samples_since_retrain: usize,
    blend_weight: f64,
    data_dir: PathBuf,
}

const WARM_THRESHOLD: usize = 100;
const HOT_THRESHOLD: usize = 500;
const RETRAIN_INTERVAL: usize = 50;

impl GbdtEngine {
    /// Initialize: load existing model + count samples.
    pub fn init(data_dir: &str) -> Self {
        let data_dir = PathBuf::from(data_dir);
        let model_path = data_dir.join("gbdt_model.json");
        let samples_path = data_dir.join("gbdt_samples.jsonl");

        // Count existing samples
        let sample_count = count_lines(&samples_path);

        // Load model if exists
        let model = if model_path.exists() {
            match std::fs::read_to_string(&model_path) {
                Ok(json) => match GradientBooster::from_json(&json) {
                    Ok(m) => {
                        tracing::info!(
                            "[GBDT] Loaded model from {} ({} samples)",
                            model_path.display(),
                            sample_count
                        );
                        Some(m)
                    }
                    Err(e) => {
                        tracing::warn!("[GBDT] Failed to load model: {e}");
                        None
                    }
                },
                Err(e) => {
                    tracing::warn!("[GBDT] Failed to read model file: {e}");
                    None
                }
            }
        } else {
            None
        };

        let blend_weight = compute_blend_weight(sample_count, model.is_some());

        let phase = if sample_count >= HOT_THRESHOLD {
            "hot"
        } else if sample_count >= WARM_THRESHOLD {
            "warm"
        } else {
            "cold"
        };

        tracing::info!(
            "[GBDT] Initialized — {} ({} samples, blend {:.0}%, model {})",
            phase,
            sample_count,
            blend_weight * 100.0,
            if model.is_some() { "loaded" } else { "none" }
        );

        Self {
            model,
            sample_count,
            samples_since_retrain: 0,
            blend_weight,
            data_dir,
        }
    }

    /// Predict P(win) for a single feature vector. Returns 0.5 if no model.
    pub fn predict(&self, features: &[f64]) -> f64 {
        let model = match &self.model {
            Some(m) => m,
            None => return 0.5,
        };
        if features.len() != NUM_FEATURES {
            return 0.5;
        }
        let matrix = Matrix::new(features, 1, NUM_FEATURES);
        let preds = model.predict(&matrix, false);
        if preds.is_empty() {
            return 0.5;
        }
        // LogLoss objective → apply sigmoid to get probability
        sigmoid(preds[0])
    }

    /// Record a training sample (called when a trade closes).
    pub fn record_sample(&mut self, features: Vec<f64>, pnl_percent: f64) {
        if features.len() != NUM_FEATURES {
            tracing::warn!("[GBDT] Bad feature vec length: {} (expected {})", features.len(), NUM_FEATURES);
            return;
        }

        let label = if pnl_percent > 0.001 { 1u8 } else { 0u8 };
        let sample = TrainingSample {
            features,
            pnl_percent,
            label,
        };

        // Append to JSONL file
        let samples_path = self.data_dir.join("gbdt_samples.jsonl");
        match std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&samples_path)
        {
            Ok(mut f) => {
                if let Ok(json) = serde_json::to_string(&sample) {
                    let _ = writeln!(f, "{}", json);
                }
            }
            Err(e) => {
                tracing::error!("[GBDT] Failed to write sample: {e}");
                return;
            }
        }

        self.sample_count += 1;
        self.samples_since_retrain += 1;

        let need = if self.sample_count < WARM_THRESHOLD {
            format!("need {} for warm", WARM_THRESHOLD - self.sample_count)
        } else if self.sample_count < HOT_THRESHOLD {
            format!("need {} for hot", HOT_THRESHOLD - self.sample_count)
        } else {
            "hot".into()
        };

        tracing::info!(
            "[GBDT] Sample recorded ({} total, {}, pnl={:.3}% label={})",
            self.sample_count, need, pnl_percent * 100.0, label
        );

        // Auto-retrain when threshold reached
        if self.samples_since_retrain >= RETRAIN_INTERVAL && self.sample_count >= WARM_THRESHOLD {
            self.retrain();
        }
    }

    /// Retrain model from all accumulated samples.
    pub fn retrain(&mut self) {
        let samples_path = self.data_dir.join("gbdt_samples.jsonl");
        let samples = match load_samples(&samples_path) {
            Ok(s) => s,
            Err(e) => {
                tracing::error!("[GBDT] Failed to load training samples: {e}");
                return;
            }
        };

        if samples.len() < 20 {
            tracing::warn!("[GBDT] Not enough samples to train ({})", samples.len());
            return;
        }

        // Build column-major flat data for forust Matrix
        let n = samples.len();
        let mut col_major = vec![0.0f64; n * NUM_FEATURES];
        let mut labels = Vec::with_capacity(n);

        for (row_idx, sample) in samples.iter().enumerate() {
            labels.push(sample.label as f64);
            for col_idx in 0..NUM_FEATURES {
                col_major[col_idx * n + row_idx] = sample.features[col_idx];
            }
        }

        let matrix = Matrix::new(&col_major, n, NUM_FEATURES);

        // Configure booster for binary classification
        let mut booster = GradientBooster::default()
            .set_objective_type(forust_ml::objective::ObjectiveType::LogLoss)
            .set_iterations(200)
            .set_learning_rate(0.05)
            .set_max_depth(5)
            .set_max_leaves(32)
            .set_l2(1.0)
            .set_subsample(0.8)
            .set_colsample_bytree(0.8)
            .set_min_leaf_weight(5.0)
            .set_missing(f64::NAN)
            .set_seed(42);

        match booster.fit_unweighted(&matrix, &labels, None) {
            Ok(()) => {}
            Err(e) => {
                tracing::error!("[GBDT] Training failed: {e}");
                return;
            }
        }

        // Evaluate: simple accuracy on training data (we don't have enough for holdout yet)
        let pred_matrix = {
            let mut cm = vec![0.0f64; n * NUM_FEATURES];
            for (row_idx, sample) in samples.iter().enumerate() {
                for col_idx in 0..NUM_FEATURES {
                    cm[col_idx * n + row_idx] = sample.features[col_idx];
                }
            }
            cm
        };
        let pm = Matrix::new(&pred_matrix, n, NUM_FEATURES);
        let preds = booster.predict(&pm, false);
        let correct = preds
            .iter()
            .zip(labels.iter())
            .filter(|(p, l)| {
                let predicted = if sigmoid(**p) >= 0.5 { 1.0 } else { 0.0 };
                (predicted - *l).abs() < 0.01
            })
            .count();
        let accuracy = correct as f64 / n as f64;

        // Feature importance
        let importance = booster.calculate_feature_importance(
            forust_ml::gradientbooster::ImportanceMethod::Gain,
            true,
        );
        let top5 = top_n_importances(&importance, 5);

        // Save model
        let model_path = self.data_dir.join("gbdt_model.json");
        match booster.json_dump() {
            Ok(json) => {
                if let Err(e) = std::fs::write(&model_path, json) {
                    tracing::error!("[GBDT] Failed to save model: {e}");
                }
            }
            Err(e) => {
                tracing::error!("[GBDT] Failed to serialize model: {e}");
            }
        }

        self.model = Some(booster);
        self.samples_since_retrain = 0;
        self.blend_weight = compute_blend_weight(self.sample_count, true);

        let top_str: Vec<String> = top5
            .iter()
            .map(|(name, imp)| format!("{}:{:.2}", name, imp))
            .collect();

        tracing::info!(
            "[GBDT] Retrained — {} samples, accuracy {:.1}%, blend {:.0}%, top: [{}]",
            n,
            accuracy * 100.0,
            self.blend_weight * 100.0,
            top_str.join(", ")
        );
    }

    /// Get top N feature importances (name, normalized importance).
    #[allow(dead_code)]
    pub fn top_importances(&self, n: usize) -> Vec<(String, f64)> {
        let model = match &self.model {
            Some(m) => m,
            None => return Vec::new(),
        };
        let importance = model.calculate_feature_importance(
            forust_ml::gradientbooster::ImportanceMethod::Gain,
            true,
        );
        top_n_importances(&importance, n)
    }

    /// Current blend weight: 0.0 (cold), 0.3 (warm), 0.6 (hot).
    pub fn blend_weight(&self) -> f64 {
        self.blend_weight
    }

    pub fn sample_count(&self) -> usize {
        self.sample_count
    }

    pub fn has_model(&self) -> bool {
        self.model.is_some()
    }
}

// ── Feature Extraction ────────────────────────────────────────────────

/// Convert Features struct to a deterministic 44-element f64 vector.
/// Options → 0.0 when None, bools → 0/1, integers → f64.
pub fn features_to_vec(f: &Features) -> Vec<f64> {
    vec![
        f.price,
        f.ema9.unwrap_or(0.0),
        f.ema21.unwrap_or(0.0),
        f.ema50.unwrap_or(0.0),
        f.ema55.unwrap_or(0.0),
        f.ema200.unwrap_or(0.0),
        f.rsi,
        f.macd_hist,
        f.atr,
        f.bb_upper.unwrap_or(0.0),
        f.bb_lower.unwrap_or(0.0),
        f.zscore,
        f.quant_vol,
        f.quant_momo,
        f.quant_regime,
        f.quant_meanrev,
        f.quant_liquidity,
        f.quant_risk,
        f.quant_corr_btc,
        f.quant_corr_eth,
        f.vol_ratio,
        f.trend_score as f64,
        f.market_state,
        f.momentum_score,
        f.trend_strength as f64,
        f.consecutive_candles as f64,
        f.tud_up as f64,
        f.tud_down as f64,
        f.spread_pct,
        f.book_imbalance,
        f.buy_ratio,
        f.sell_ratio,
        f.flow_imbalance,
        f.flow_book_div,
        f.flow_book_div_abs,
        f.book_trend as f64,
        if f.book_reversal { 1.0 } else { 0.0 },
        f.book_reversal_dir as f64,
        f.book_strength,
        f.book_avg_imb,
        f.hurst_exp,
        f.shannon_entropy,
        f.autocorr_lag1,
        f.adx,
    ]
}

// ── Helpers ───────────────────────────────────────────────────────────

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn compute_blend_weight(sample_count: usize, has_model: bool) -> f64 {
    if !has_model || sample_count < WARM_THRESHOLD {
        0.0
    } else if sample_count < HOT_THRESHOLD {
        0.30
    } else {
        0.60
    }
}

fn count_lines(path: &Path) -> usize {
    match std::fs::read_to_string(path) {
        Ok(content) => content.lines().filter(|l| !l.trim().is_empty()).count(),
        Err(_) => 0,
    }
}

fn load_samples(path: &Path) -> anyhow::Result<Vec<TrainingSample>> {
    let content = std::fs::read_to_string(path)?;
    let mut samples = Vec::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        match serde_json::from_str::<TrainingSample>(line) {
            Ok(s) if s.features.len() == NUM_FEATURES => samples.push(s),
            Ok(_) => {} // skip malformed
            Err(_) => {} // skip parse errors
        }
    }
    Ok(samples)
}

fn top_n_importances(importance: &HashMap<usize, f32>, n: usize) -> Vec<(String, f64)> {
    let mut pairs: Vec<(usize, f32)> = importance.iter().map(|(&k, &v)| (k, v)).collect();
    pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    pairs.truncate(n);
    pairs
        .into_iter()
        .map(|(idx, imp)| {
            let name = if idx < NUM_FEATURES {
                FEATURE_NAMES[idx].to_string()
            } else {
                format!("feat_{}", idx)
            };
            (name, imp as f64)
        })
        .collect()
}
