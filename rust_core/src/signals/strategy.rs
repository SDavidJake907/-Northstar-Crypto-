// strategy.rs — Hot-reloadable strategy parameters from data/strategy.toml
//
// Single source of truth for all trading math: scoring weights, normalization
// ranges, lane filters, signal cascade thresholds, meme coin list.
// On reload, auto-regenerates data/nemo_prompt.txt so Nemo sees the same math.
#![allow(dead_code)]

use anyhow::Result;
use serde::Deserialize;
use std::sync::Arc;
use std::time::{Instant, SystemTime};

// ── Top-Level TOML Structure ────────────────────────────────────

#[derive(Clone, Debug, Deserialize)]
pub struct StrategyToml {
    pub meta: Meta,
    pub scoring: Scoring,
    pub mlp: Mlp,
    pub signals: Signals,
    pub filter: Filter,
    pub memes: Memes,
    pub strategy: StrategyConfig,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Meta {
    pub version: u32,
}

// ── Scoring ─────────────────────────────────────────────────────

#[derive(Clone, Debug, Deserialize)]
pub struct Scoring {
    pub bias: f64,
    pub weights: ScoringWeights,
    pub norm: ScoringNorm,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ScoringWeights {
    pub trend: f64,
    pub momentum: f64,
    pub volatility: f64,
    pub volume: f64,
    pub mean_revert: f64,
    pub orderflow: f64,
    pub quant: f64,
    #[serde(default)]
    pub universe: Option<f64>,
    #[serde(default)]
    pub whale: Option<f64>,
    #[serde(default)]
    pub cross_radar: Option<f64>,
    #[serde(default)]
    pub ml_signal: Option<f64>,
}

impl ScoringWeights {
    pub fn as_array(&self) -> [f64; 7] {
        [
            self.trend,
            self.momentum,
            self.volatility,
            self.volume,
            self.mean_revert,
            self.orderflow,
            self.quant,
        ]
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct ScoringNorm {
    pub trend_score: [f64; 2],
    pub macd_hist: [f64; 2],
    pub rsi: [f64; 2],
    pub atr_pct_mult: f64,
    pub vol_ratio: [f64; 2],
    pub zscore: [f64; 2],
    pub book_imbalance: [f64; 2],
    pub buy_ratio: [f64; 2],
    pub quant_momo: [f64; 2],
    pub quant_regime: [f64; 2],
}

// ── MLP (24D Feature Vector) ────────────────────────────────────

#[derive(Clone, Debug, Deserialize)]
pub struct Mlp {
    #[allow(dead_code)] // deserialized from TOML but not accessed at runtime
    pub norm: MlpNorm,
    pub crash_flag: CrashFlag,
}

#[allow(dead_code)] // deserialized from TOML but fields not individually accessed
#[derive(Clone, Debug, Deserialize)]
pub struct MlpNorm {
    pub trend_score: [f64; 2],
    pub macd_hist: [f64; 2],
    pub rsi: [f64; 2],
    pub atr_pct: [f64; 2],
    pub vol_ratio: [f64; 2],
    pub zscore: [f64; 2],
    pub market_state: [f64; 2],
    pub book_imbalance: [f64; 2],
    pub buy_ratio: [f64; 2],
    pub spread_pct: [f64; 2],
    pub buy_slip_pct_100k: [f64; 2],
    pub quant_momo: [f64; 2],
    pub quant_regime: [f64; 2],
    pub quant_vol: [f64; 2],
    pub quant_liquidity: [f64; 2],
    pub rsi_p20: [f64; 2],
    pub rsi_p80: [f64; 2],
    pub trend_strength: [f64; 2],
    pub markov_score: [f64; 2],
    pub markov_prob: [f64; 2],
}

#[allow(dead_code)] // deserialized from TOML, passed as opaque struct
#[derive(Clone, Debug, Deserialize)]
pub struct CrashFlag {
    pub quant_regime_le: f64,
    pub quant_vol_ge: f64,
}

// ── Signal Cascade ──────────────────────────────────────────────

#[derive(Clone, Debug, Deserialize)]
pub struct Signals {
    pub crash: SignalCrash,
    pub hunt: SignalHunt,
    pub buy: SignalBuy,
    pub sell: SignalSell,
    pub hold: SignalHold,
}

#[derive(Clone, Debug, Deserialize)]
pub struct SignalCrash {
    pub quant_regime_le: f64,
    pub quant_vol_ge: f64,
    pub confidence: f64,
}

#[derive(Clone, Debug, Deserialize)]
pub struct SignalHunt {
    pub imb_ge: f64,
    pub buy_ratio_ge: f64,
    pub momo_gt: f64,
    pub confidence: f64,
}

#[derive(Clone, Debug, Deserialize)]
pub struct SignalBuy {
    pub imb_ge: f64,
    pub buy_ratio_ge: f64,
    pub momo_ge: f64,
    pub conf_base: f64,
    pub conf_mult: f64,
    pub conf_cap: f64,
}

#[derive(Clone, Debug, Deserialize)]
pub struct SignalSell {
    pub imb_le: f64,
    pub momo_lt: f64,
    pub buy_ratio_lt: f64,
    pub conf_base: f64,
    pub conf_mult: f64,
    pub conf_cap: f64,
}

#[derive(Clone, Debug, Deserialize)]
pub struct SignalHold {
    pub confidence: f64,
}

// ── Filters ─────────────────────────────────────────────────────

#[derive(Clone, Debug, Deserialize)]
pub struct Filter {
    pub global: GlobalFilter,
    pub lane1: Lane1,
    pub lane2: Lane2,
    pub lane3: Lane3,
    #[serde(default)]
    pub markov: MarkovFilter,
}

#[derive(Clone, Debug, Deserialize)]
pub struct MarkovFilter {
    #[serde(default = "default_markov_min_bull_prob")]
    pub min_bull_prob: f64,
    #[serde(default = "default_markov_min_score")]
    pub min_score: f64,
    #[serde(default = "default_true")]
    pub enabled: bool,
}
fn default_markov_min_bull_prob() -> f64 {
    0.35
}
fn default_markov_min_score() -> f64 {
    -0.30
}
fn default_true() -> bool {
    true
}
impl Default for MarkovFilter {
    fn default() -> Self {
        Self {
            min_bull_prob: 0.35,
            min_score: -0.30,
            enabled: true,
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct GlobalFilter {
    pub min_liquidity: f64,
    pub max_spread_pct: f64,
    pub min_vol_ratio: f64,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Lane1 {
    pub min_vol_ratio: f64,
    pub max_rsi: f64,
    pub min_trend_score: i64,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Lane2 {
    pub max_bb_width: f64,
    pub max_quant_vol: f64,
    pub max_ema50_dist: f64,
    pub max_atr_pct: f64,
    pub min_book_imb: f64,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Lane3 {
    pub min_rsi: f64,
    pub max_rsi: f64,
    pub max_quant_risk: f64,
    pub min_trend_score: i64,
    pub min_momo: f64,
}

// ── Memes ───────────────────────────────────────────────────────

#[derive(Clone, Debug, Deserialize)]
pub struct Memes {
    pub symbols: Vec<String>,
}

#[allow(dead_code)] // fields deserialized from TOML, used by strategy_helpers at runtime
#[derive(Clone, Debug, Deserialize)]
pub struct StrategyConfig {
    #[serde(rename = "type", default = "default_strategy_type")]
    pub strategy_type: String,
    #[serde(rename = "H", default = "default_horizon")]
    pub horizon: usize,
    #[serde(rename = "K", default = "default_rebalance_k")]
    pub rebalance_k: usize,
    #[serde(default = "default_w_max")]
    pub w_max: f64,
    #[serde(default = "default_delta")]
    pub delta: f64,
    #[serde(default = "default_tau")]
    pub tau: f64,
    #[serde(default = "default_alpha")]
    pub alpha: f64,
    #[serde(default = "default_beta")]
    pub beta: f64,
    #[serde(default = "default_gamma")]
    pub gamma: f64,
}

fn default_strategy_type() -> String {
    "softmax".to_string()
}

fn default_horizon() -> usize {
    21
}

fn default_rebalance_k() -> usize {
    12
}

fn default_w_max() -> f64 {
    0.15
}

fn default_delta() -> f64 {
    0.01
}

fn default_tau() -> f64 {
    1.0
}

fn default_alpha() -> f64 {
    1.0
}

fn default_beta() -> f64 {
    0.5
}

fn default_gamma() -> f64 {
    0.2
}

// ── Strategy Runtime Wrapper ────────────────────────────────────

pub struct Strategy {
    inner: Arc<StrategyToml>,
    last_reload: Instant,
    last_modified: Option<SystemTime>,
    toml_path: String,
    prompt_path: String,
}

impl Strategy {
    /// Load strategy from TOML file. Generates nemo_prompt.txt on first load.
    pub fn load(toml_path: &str, prompt_path: &str) -> Result<Self> {
        let content = std::fs::read_to_string(toml_path).map_err(|e| {
            tracing::error!("[STRATEGY] cannot read {toml_path}: {e}");
            anyhow::anyhow!("strategy.toml not found at {toml_path}")
        })?;
        let parsed: StrategyToml = toml::from_str(&content).map_err(|e| {
            tracing::error!("[STRATEGY] parse error in {toml_path}: {e}");
            anyhow::anyhow!("strategy.toml parse error: {e}")
        })?;

        let mtime = std::fs::metadata(toml_path)
            .ok()
            .and_then(|m| m.modified().ok());

        tracing::info!(
            "[STRATEGY] loaded strategy.toml v{} ({} meme coins)",
            parsed.meta.version,
            parsed.memes.symbols.len()
        );

        let s = Self {
            inner: Arc::new(parsed),
            last_reload: Instant::now(),
            last_modified: mtime,
            toml_path: toml_path.to_string(),
            prompt_path: prompt_path.to_string(),
        };
        s.regenerate_prompt();
        Ok(s)
    }

    /// Get an Arc clone for passing to TradingLoop.
    pub fn get_arc(&self) -> Arc<StrategyToml> {
        Arc::clone(&self.inner)
    }

    /// Check file mtime; if changed, re-parse and regenerate nemo_prompt.txt.
    pub fn hot_reload(&mut self) {
        if self.last_reload.elapsed().as_secs() < 10 {
            return;
        }
        self.last_reload = Instant::now();

        let current_mtime = std::fs::metadata(&self.toml_path)
            .ok()
            .and_then(|m| m.modified().ok());

        if current_mtime == self.last_modified {
            return;
        }

        match std::fs::read_to_string(&self.toml_path) {
            Ok(content) => match toml::from_str::<StrategyToml>(&content) {
                Ok(parsed) => {
                    tracing::info!("[STRATEGY] reloaded strategy.toml v{}", parsed.meta.version);
                    self.inner = Arc::new(parsed);
                    self.last_modified = current_mtime;
                    self.regenerate_prompt();
                }
                Err(e) => {
                    tracing::error!("[STRATEGY] parse error (keeping old config): {e}");
                }
            },
            Err(e) => {
                tracing::error!("[STRATEGY] read error (keeping old config): {e}");
            }
        }
    }

    /// Regenerate data/nemo_prompt.txt from current strategy values.
    /// Skips if file already exists (allows manual edits to survive restarts).
    fn regenerate_prompt(&self) {
        if std::path::Path::new(&self.prompt_path).exists() {
            tracing::info!("[STRATEGY] nemo_prompt.txt exists, preserving manual edits (delete file to regenerate)");
            return;
        }
        let prompt = r#"You are a crypto trading AI. A math engine has already scored each coin using logistic gates and softmax. You review the scores and raw data, then agree or override.

WHAT YOU SEE:
- MATH line: the engine's recommendation (BUY/SELL/HOLD), confidence %, and raw scores S_buy, S_sell, S_hold
- Market data: Trend (-5 to +5), RSI, MACD, Momentum, Volume, Zscore
- Orderbook: Book_Imb (-1 to +1), BuyRatio (0 to 1), Spread
- Portfolio: total value, free USD, held positions

HOW THE MATH WORKS:
- S_buy, S_sell, S_hold are continuous scores from logistic gates over market signals
- The highest score wins (argmax). Confidence comes from softmax probability
- Crash detection, buyer pressure, and sell pressure are all smooth gate functions — no hard thresholds

YOUR JOB:
1. If the math recommendation makes sense given the raw data, AGREE with it
2. If something looks off (e.g. math says BUY but RSI is 85+, or spread is huge), OVERRIDE
3. You can adjust confidence up or down based on how many signals align

WHEN TO OVERRIDE:
- RSI > 80 and math says BUY → too overbought, switch to HOLD
- RSI < 25 and math says SELL → might be oversold bounce, switch to HOLD
- Spread > 0.5% → never BUY, fees will eat profit
- All three scores are close together → low conviction, prefer HOLD

Reply with ONLY one JSON object:
{"action":"BUY","confidence":0.72,"reason":"agree math — trend+3 buyers dominant"}"#;

        match std::fs::write(&self.prompt_path, &prompt) {
            Ok(_) => tracing::info!(
                "[STRATEGY] regenerated {} ({} bytes)",
                self.prompt_path,
                prompt.len()
            ),
            Err(e) => tracing::error!("[STRATEGY] failed to write {}: {e}", self.prompt_path),
        }
    }
}
