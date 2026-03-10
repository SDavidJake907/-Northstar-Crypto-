//! Market Brain — macro awareness from cross-coin features.
//!
//! Reads the full `features_map` once per tick to produce:
//! - BTC gravity (trend, momentum, crash detection)
//! - Breadth indicators (green/red count, avg RSI, volume, book pressure)
//! - Sector stats (majors, L1, DeFi, memes, etc.)
//! - Market regime classification (Crash/Bearish/Sideways/Bullish/Euphoric)
//! - Funding rate intelligence from Kraken Futures (dynamic confidence base)
//!
//! No AI calls, no disk I/O — purely computational (except async funding fetch).

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;

// ── Regime ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarketRegime {
    Crash,
    Bearish,
    Sideways,
    Bullish,
    Euphoric,
}

impl fmt::Display for MarketRegime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl MarketRegime {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Crash => "CRASH",
            Self::Bearish => "BEARISH",
            Self::Sideways => "SIDEWAYS",
            Self::Bullish => "BULLISH",
            Self::Euphoric => "EUPHORIC",
        }
    }
}

// ── Sub-Structs ────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct BtcGravity {
    pub trend_score: i32,
    pub momentum: f64,
    pub rsi: f64,
    pub is_crashing: bool,
    pub dominance_hint: String,
}

#[derive(Debug, Clone)]
pub struct Breadth {
    pub green_count: usize,
    pub red_count: usize,
    pub total_count: usize,
    pub avg_rsi: f64,
    pub avg_momentum: f64,
    pub total_vol_ratio: f64,
    pub net_book_pressure: f64,
}

#[allow(dead_code)] // avg_trend stored for future sector trend display
#[derive(Debug, Clone)]
pub struct SectorStats {
    pub name: String,
    pub coin_count: usize,
    pub avg_momentum: f64,
    pub avg_trend: f64,
    pub avg_vol_ratio: f64,
    pub green_pct: f64,
}

// ── Funding Rate Intelligence ─────────────────────────────────────

#[derive(Debug, Clone)]
pub struct CoinFunding {
    pub symbol: String,
    pub rate: f64,
    pub open_interest: f64,
    pub crowd_bias: CrowdBias,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CrowdBias {
    ExtremeLong,   // funding > 0.001 — fade opportunity
    LongCrowded,   // funding > 0.0001
    SlightLong,    // funding > 0.00001
    Neutral,       // near zero
    SlightShort,   // funding < -0.00001
    ShortCrowded,  // funding < -0.0001
    ExtremeShort,  // funding < -0.001 — squeeze opportunity
}

impl CrowdBias {
    fn from_rate(rate: f64) -> Self {
        if rate > 0.001 { Self::ExtremeLong }
        else if rate > 0.0001 { Self::LongCrowded }
        else if rate > 0.00001 { Self::SlightLong }
        else if rate < -0.001 { Self::ExtremeShort }
        else if rate < -0.0001 { Self::ShortCrowded }
        else if rate < -0.00001 { Self::SlightShort }
        else { Self::Neutral }
    }

}

#[derive(Debug, Clone)]
pub struct FundingSnapshot {
    pub coins: Vec<CoinFunding>,
    pub market_avg_rate: f64,
    pub long_count: usize,
    pub short_count: usize,
    pub crowd_sentiment: String,   // "BEARISH", "BULLISH", "MIXED"
}

impl Default for FundingSnapshot {
    fn default() -> Self {
        Self {
            coins: Vec::new(),
            market_avg_rate: 0.0,
            long_count: 0,
            short_count: 0,
            crowd_sentiment: "UNKNOWN".to_string(),
        }
    }
}

/// Shared handle for async funding updates.
pub type SharedFunding = Arc<RwLock<FundingSnapshot>>;

pub fn new_shared_funding() -> SharedFunding {
    Arc::new(RwLock::new(FundingSnapshot::default()))
}

// ── Multi-Scale Market Fingerprint ────────────────────────────────
//
// 4 horizons × 8 metrics = 32-number dense market state vector.
// Each horizon captures the same 8 metrics at different timescales,
// giving Qwen a "trajectory" of market state — like a token sequence.
//
// Horizons (in ticks, base tick ~2s):
//   10s =  5 ticks
//   1m  = 30 ticks
//   5m  = 150 ticks
//   30m = 900 ticks
//
// We sample at 10s resolution (every 5 ticks) to keep buffers small:
//   10s = 1 sample lookback
//   1m  = 6 samples
//   5m  = 30 samples
//   30m = 180 samples
//
// 8 metrics per horizon:
//   r_mkt   = equal-weighted market log return
//   r_btc   = BTC log return
//   r_eth   = ETH log return
//   breadth = fraction of coins with positive return
//   med_ret = median cross-section return
//   iqr_ret = IQR of cross-section returns
//   rv_mkt  = realized vol (RMS of market returns over window)
//   corr_avg = avg BTC correlation across alts

/// Horizon lookback in 10s-samples.
const H_10S: usize = 1;
const H_1M: usize = 6;
const H_5M: usize = 30;
const H_30M: usize = 180;
/// 24h horizon in 10s-samples (8640 = 86400s / 10s).
const H_1D: usize = 8640;
/// Max ring buffer size (24h worth of 10s samples + margin).
const FP_BUF_CAP: usize = 8700;
/// How many base ticks between samples (sample at 10s resolution).
const FP_SAMPLE_INTERVAL: u32 = 5;

/// One 10s snapshot of per-coin prices (for computing log returns later).
#[derive(Clone, Debug)]
struct PriceSnapshot {
    /// symbol → price at this snapshot
    prices: HashMap<String, f64>,
}

/// Rolling price history for multi-scale fingerprint computation.
pub struct PriceHistory {
    /// Ring buffer of 10s price snapshots.
    snapshots: VecDeque<PriceSnapshot>,
    /// Tick counter to know when to sample.
    tick_count: u32,
}

impl PriceHistory {
    pub fn new() -> Self {
        Self {
            snapshots: VecDeque::with_capacity(FP_BUF_CAP),
            tick_count: 0,
        }
    }

    /// Push a new price snapshot if it's time to sample (every 5 ticks = ~10s).
    /// Returns true if a new sample was added.
    pub fn maybe_push(&mut self, features_map: &HashMap<String, serde_json::Value>) -> bool {
        self.tick_count += 1;
        if self.tick_count < FP_SAMPLE_INTERVAL {
            return false;
        }
        self.tick_count = 0;

        let mut prices = HashMap::new();
        for (sym, feats) in features_map {
            if sym.starts_with("__") { continue; }
            if let Some(p) = feats.get("price").and_then(|v| v.as_f64()) {
                if p > 0.0 {
                    prices.insert(sym.clone(), p);
                }
            }
        }
        self.snapshots.push_back(PriceSnapshot { prices });
        if self.snapshots.len() > FP_BUF_CAP {
            self.snapshots.pop_front();
        }
        true
    }

    /// Compute log returns at a given horizon (in samples) for all coins.
    /// Returns (per_coin_returns, btc_return, eth_return).
    fn returns_at_horizon(&self, horizon: usize) -> (Vec<f64>, f64, f64) {
        let n = self.snapshots.len();
        if n < 2 || horizon == 0 {
            return (Vec::new(), 0.0, 0.0);
        }
        let lookback = horizon.min(n - 1);
        let now = &self.snapshots[n - 1];
        let then = &self.snapshots[n - 1 - lookback];

        let mut returns = Vec::new();
        let mut r_btc = 0.0f64;
        let mut r_eth = 0.0f64;

        for (sym, &price_now) in &now.prices {
            if let Some(&price_then) = then.prices.get(sym) {
                if price_then > 0.0 && price_now > 0.0 {
                    let lr = (price_now / price_then).ln();
                    returns.push(lr);
                    if sym == "BTC" { r_btc = lr; }
                    if sym == "ETH" { r_eth = lr; }
                }
            }
        }
        (returns, r_btc, r_eth)
    }

    /// Compute realized vol of market returns over a window of samples.
    fn realized_vol(&self, horizon: usize, window: usize) -> f64 {
        let n = self.snapshots.len();
        if n < 3 {
            return 0.0;
        }
        let mut mkt_returns = Vec::new();
        let steps = window.min(n - 1);
        for i in 1..=steps {
            let idx = n - 1 - i;
            if idx == 0 { break; }
            let lookback = horizon.min(idx);
            let now = &self.snapshots[idx];
            let then = &self.snapshots[idx - lookback.min(idx)];
            let mut r_sum = 0.0f64;
            let mut count = 0usize;
            for (sym, &p_now) in &now.prices {
                if let Some(&p_then) = then.prices.get(sym) {
                    if p_then > 0.0 && p_now > 0.0 {
                        r_sum += (p_now / p_then).ln();
                        count += 1;
                    }
                }
            }
            if count > 0 {
                mkt_returns.push(r_sum / count as f64);
            }
        }
        if mkt_returns.is_empty() {
            return 0.0;
        }
        // RMS
        let rms = (mkt_returns.iter().map(|r| r * r).sum::<f64>() / mkt_returns.len() as f64).sqrt();
        rms
    }
}

/// Single-horizon 8-metric fingerprint slice.
#[derive(Debug, Clone, Copy, Default)]
pub struct FpSlice {
    pub values: [f64; 8],
}

/// Multi-scale fingerprint: 5 horizons × 8 metrics = 40 numbers.
/// FP_1d gives Qwen 24h regime memory — the "daily anchor" in x_{1:t}.
#[derive(Debug, Clone)]
pub struct MarketFingerprint {
    pub fp_1d: FpSlice,
    pub fp_30m: FpSlice,
    pub fp_5m: FpSlice,
    pub fp_1m: FpSlice,
    pub fp_10s: FpSlice,
}

impl Default for MarketFingerprint {
    fn default() -> Self {
        Self {
            fp_1d: FpSlice::default(),
            fp_30m: FpSlice::default(),
            fp_5m: FpSlice::default(),
            fp_1m: FpSlice::default(),
            fp_10s: FpSlice::default(),
        }
    }
}

impl MarketFingerprint {
    /// Compute multi-scale fingerprint from price history + current features.
    pub fn compute(
        history: &PriceHistory,
        features_map: &HashMap<String, serde_json::Value>,
    ) -> Self {
        let fp_10s = Self::compute_slice(history, features_map, H_10S, 60);
        let fp_1m = Self::compute_slice(history, features_map, H_1M, 60);
        let fp_5m = Self::compute_slice(history, features_map, H_5M, 72);
        let fp_30m = Self::compute_slice(history, features_map, H_30M, 48);
        let fp_1d = Self::compute_slice(history, features_map, H_1D, 180);
        Self { fp_1d, fp_30m, fp_5m, fp_1m, fp_10s }
    }

    /// Compute one 8-metric slice at a given horizon.
    fn compute_slice(
        history: &PriceHistory,
        features_map: &HashMap<String, serde_json::Value>,
        horizon: usize,
        rv_window: usize,
    ) -> FpSlice {
        let (coin_returns, r_btc, r_eth) = history.returns_at_horizon(horizon);

        if coin_returns.is_empty() {
            return FpSlice::default();
        }

        let n = coin_returns.len() as f64;

        // 1. r_mkt: equal-weighted market return
        let r_mkt = coin_returns.iter().sum::<f64>() / n;

        // 4. breadth: fraction with positive return
        let breadth = coin_returns.iter().filter(|&&r| r > 0.0).count() as f64 / n;

        // 5-6. median + IQR
        let (med_ret, iqr_ret) = median_iqr(&coin_returns);

        // 7. realized vol (RMS of market returns over rolling window)
        let rv_mkt = history.realized_vol(horizon, rv_window);

        // 8. avg BTC correlation proxy (from current features, not historical)
        let mut corr_sum = 0.0f64;
        let mut corr_count = 0usize;
        for (sym, feats) in features_map {
            if sym.starts_with("__") || sym == "BTC" || sym == "ETH" { continue; }
            let c = feats.get("quant_corr_btc").and_then(|v| v.as_f64()).unwrap_or(0.0);
            corr_sum += c;
            corr_count += 1;
        }
        let corr_avg = if corr_count > 0 { corr_sum / corr_count as f64 } else { 0.0 };

        FpSlice {
            values: [r_mkt, r_btc, r_eth, breadth, med_ret, iqr_ret, rv_mkt, corr_avg],
        }
    }

    /// 5-line compact string for prompt injection (1d → 10s, big picture first).
    pub fn to_prompt_block(&self) -> String {
        format!(
            "FP_1d:[{}]\nFP_30m:[{}]\nFP_5m:[{}]\nFP_1m:[{}]\nFP_10s:[{}]",
            Self::fmt_slice(&self.fp_1d),
            Self::fmt_slice(&self.fp_30m),
            Self::fmt_slice(&self.fp_5m),
            Self::fmt_slice(&self.fp_1m),
            Self::fmt_slice(&self.fp_10s),
        )
    }

    fn fmt_slice(s: &FpSlice) -> String {
        s.values.iter().map(|v| format!("{:.4}", v)).collect::<Vec<_>>().join(",")
    }
}

/// Compute median and IQR of a slice.
fn median_iqr(data: &[f64]) -> (f64, f64) {
    if data.is_empty() {
        return (0.0, 0.0);
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    let median = if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    };
    let q1 = sorted[n / 4];
    let q3 = sorted[3 * n / 4];
    (median, q3 - q1)
}

// ── Greed Index (Fear & Greed 0–100) ─────────────────────────────
//
// 4 z-scored components from FP data, weighted sum → sigmoid → 0–100.
// No external APIs needed — purely from WS-derived fingerprint.
//
// Components (all z-scored via exponential moving stats):
//   M = z(r_mkt_1d)        momentum   (greed+)  weight 0.40
//   B = z(breadth_30m)     breadth    (greed+)  weight 0.25
//   V = -z(rv_mkt_1d)      volatility (fear+)   weight 0.20
//   C = -z(corr_avg_30m)   correlation(fear+)   weight 0.15
//
// Greed = 100 · σ(0.40M + 0.25B + 0.20V + 0.15C)

/// Exponential moving z-score tracker for one component.
struct ExpStat {
    mean: f64,
    var: f64,
    count: u64,
    alpha: f64,
}

impl ExpStat {
    fn new(alpha: f64) -> Self {
        Self { mean: 0.0, var: 0.01, count: 0, alpha }
    }

    fn update(&mut self, x: f64) {
        self.count += 1;
        if self.count == 1 {
            self.mean = x;
            self.var = 0.01;
            return;
        }
        let a = self.alpha;
        let diff = x - self.mean;
        self.mean += a * diff;
        self.var = (1.0 - a) * (self.var + a * diff * diff);
    }

    fn zscore(&self, x: f64) -> f64 {
        if self.count < 60 { return 0.0; } // need ~10 min warm-up
        let std = self.var.sqrt().max(1e-10);
        ((x - self.mean) / std).clamp(-3.0, 3.0)
    }
}

/// Tracks rolling z-scores for greed index computation.
struct GreedTracker {
    momentum: ExpStat,
    breadth: ExpStat,
    volatility: ExpStat,
    correlation: ExpStat,
}

/// EMA alpha ≈ 2/(N+1), N=10000 → ~28h effective window at 10s sampling.
const GREED_ALPHA: f64 = 0.0002;

impl GreedTracker {
    fn new() -> Self {
        Self {
            momentum: ExpStat::new(GREED_ALPHA),
            breadth: ExpStat::new(GREED_ALPHA),
            volatility: ExpStat::new(GREED_ALPHA),
            correlation: ExpStat::new(GREED_ALPHA),
        }
    }

    /// Update with current FP values and return greed index (0–100).
    /// During warm-up (< 60 samples), returns 50.0 (neutral).
    fn update(&mut self, fp_1d: &FpSlice, fp_30m: &FpSlice) -> f64 {
        let r_mkt = fp_1d.values[0];      // 24h market return
        let breadth = fp_30m.values[3];    // 30m breadth (fraction green)
        let rv_mkt = fp_1d.values[6];      // 24h realized vol
        let corr_avg = fp_30m.values[7];   // 30m avg BTC correlation

        self.momentum.update(r_mkt);
        self.breadth.update(breadth);
        self.volatility.update(rv_mkt);
        self.correlation.update(corr_avg);

        let m = self.momentum.zscore(r_mkt);       // greed+
        let b = self.breadth.zscore(breadth);       // greed+
        let v = -self.volatility.zscore(rv_mkt);    // fear+ inverted
        let c = -self.correlation.zscore(corr_avg);  // fear+ inverted

        let s = 0.40 * m + 0.25 * b + 0.20 * v + 0.15 * c;

        // Sigmoid → 0–100
        let greed = 100.0 / (1.0 + (-s).exp());
        (greed * 10.0).round() / 10.0
    }
}

// ── Market State ─────────────────────────────────────────────────

#[derive(Debug, Clone)]
#[allow(dead_code)] // fields stored for telemetry + future prompt enrichment
pub struct MarketState {
    pub regime: MarketRegime,
    pub btc: BtcGravity,
    pub breadth: Breadth,
    pub sectors: Vec<SectorStats>,
    pub correlation_note: String,
    pub fingerprint: MarketFingerprint,
    pub greed_index: f64,
    pub ts: f64,
}

// ── MarketBrain ────────────────────────────────────────────────────

pub struct MarketBrain {
    current_state: Option<MarketState>,
    last_regime: Option<MarketRegime>,
    price_history: PriceHistory,
    greed_tracker: GreedTracker,
}

impl MarketBrain {
    pub fn new() -> Self {
        Self {
            current_state: None,
            last_regime: None,
            price_history: PriceHistory::new(),
            greed_tracker: GreedTracker::new(),
        }
    }

    /// Called once per tick, BEFORE candidate selection.
    pub fn update(
        &mut self,
        features_map: &HashMap<String, serde_json::Value>,
        strategy: &crate::strategy::StrategyToml,
    ) {
        let now = crate::ai_bridge::now_ts();
        let btc = Self::compute_btc_gravity(features_map);
        let breadth = Self::compute_breadth(features_map);
        let sectors = Self::compute_sectors(features_map, &strategy.memes.symbols);
        let regime = Self::classify_regime(&btc, &breadth);
        let correlation_note = Self::build_correlation_note(features_map);

        // Push price snapshot + compute multi-scale fingerprint + greed index
        self.price_history.maybe_push(features_map);
        let fingerprint = MarketFingerprint::compute(&self.price_history, features_map);
        let greed_index = self.greed_tracker.update(&fingerprint.fp_1d, &fingerprint.fp_30m);

        // Log regime changes
        if self.last_regime != Some(regime) {
            if let Some(prev) = self.last_regime {
                tracing::info!(
                    "[MARKET-BRAIN] Regime change: {} -> {} (green={}/{} avgRSI={:.0})",
                    prev, regime,
                    breadth.green_count, breadth.total_count, breadth.avg_rsi,
                );
            } else {
                tracing::info!(
                    "[MARKET-BRAIN] Initial regime: {} (green={}/{} avgRSI={:.0})",
                    regime, breadth.green_count, breadth.total_count, breadth.avg_rsi,
                );
            }
            self.last_regime = Some(regime);
        }

        self.current_state = Some(MarketState {
            regime,
            btc,
            breadth,
            sectors,
            correlation_note,
            fingerprint,
            greed_index,
            ts: now,
        });
    }

    // ── Public Query API ───────────────────────────────────────────

    pub fn is_btc_crashing(&self) -> bool {
        self.current_state
            .as_ref()
            .map(|s| s.btc.is_crashing)
            .unwrap_or(false)
    }

    pub fn regime(&self) -> MarketRegime {
        self.current_state
            .as_ref()
            .map(|s| s.regime)
            .unwrap_or(MarketRegime::Sideways)
    }

    /// Read-only access to full market state (for portfolio allocator, etc.)
    pub fn state(&self) -> Option<&MarketState> {
        self.current_state.as_ref()
    }

    /// Clock line for AI prompts — gives Qwen time awareness.
    fn build_clock_line() -> String {
        let secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let days_since_epoch = secs / 86400;
        let time_of_day = secs % 86400;
        let hour = (time_of_day / 3600) as u8;
        let minute = ((time_of_day % 3600) / 60) as u8;
        let dow_num = ((days_since_epoch + 4) % 7) as u8; // 0=Sun..6=Sat
        let dow = match dow_num {
            0 => "Sun", 1 => "Mon", 2 => "Tue", 3 => "Wed",
            4 => "Thu", 5 => "Fri", 6 => "Sat", _ => "?",
        };
        let z = days_since_epoch as i64 + 719468;
        let era = z / 146097;
        let doe = z - era * 146097;
        let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
        let y = yoe + era * 400;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        let mp = (5 * doy + 2) / 153;
        let d = doy - (153 * mp + 2) / 5 + 1;
        let m = if mp < 10 { mp + 3 } else { mp - 9 };
        let y = if m <= 2 { y + 1 } else { y };
        let session = match hour {
            0..=6   => "ASIA",
            7..=12  => "EUR",
            13..=15 => "EUR+US",
            16..=21 => "US",
            22..=23 => "ASIA",
            _ => "?",
        };
        let weekend = if dow_num == 0 || dow_num == 6 { "|WEEKEND" } else { "" };
        format!("CLOCK:{}-{:02}-{:02}T{:02}:{:02}Z|{}|{}{}", y, m, d, hour, minute, dow, session, weekend)
    }

    /// Full market context block for entry prompts.
    pub fn build_market_block(&self) -> String {
        let state = match &self.current_state {
            Some(s) => s,
            None => return String::new(),
        };

        let btc = &state.btc;
        let br = &state.breadth;

        // Compact market context — matrix format
        let crash_tag = if btc.is_crashing { "CRASH" }
            else if btc.dominance_hint == "risk_on" { "ON" }
            else if btc.dominance_hint == "risk_off" { "OFF" }
            else { "N" };
        let pressure_tag = if br.net_book_pressure > 0.1 { "B" }
            else if br.net_book_pressure < -0.1 { "S" }
            else { "N" };

        format!(
            "{}\nGREED:{:.1}\n{}\nMKT:{}|BTC:T{:+},R{:.0},M{:+.2},{}|{}/{} green|avgR{:.0}|book{:+.2}({})|vol{:.1}x",
            state.fingerprint.to_prompt_block(),
            state.greed_index,
            Self::build_clock_line(),
            state.regime,
            btc.trend_score, btc.rsi, btc.momentum, crash_tag,
            br.green_count, br.total_count, br.avg_rsi,
            br.net_book_pressure, pressure_tag, br.total_vol_ratio,
        )
    }

    /// Slim market context for entry prompts (greed + regime, no fingerprint).
    pub fn build_market_slim(&self) -> String {
        let state = match &self.current_state {
            Some(s) => s,
            None => return String::new(),
        };
        let btc = &state.btc;
        let br = &state.breadth;
        let crash_tag = if btc.is_crashing { "CRASH" }
            else if btc.dominance_hint == "risk_on" { "ON" }
            else if btc.dominance_hint == "risk_off" { "OFF" }
            else { "N" };
        let pressure_tag = if br.net_book_pressure > 0.1 { "B" }
            else if br.net_book_pressure < -0.1 { "S" }
            else { "N" };
        format!(
            "GREED:{:.1}\n{}\nMKT:{}|BTC:T{:+},R{:.0},M{:+.2},{}|{}/{} green|avgR{:.0}|book{:+.2}({})|vol{:.1}x",
            state.greed_index,
            Self::build_clock_line(),
            state.regime,
            btc.trend_score, btc.rsi, btc.momentum, crash_tag,
            br.green_count, br.total_count, br.avg_rsi,
            br.net_book_pressure, pressure_tag, br.total_vol_ratio,
        )
    }

    /// Short market context for exit prompts (one line).
    #[allow(dead_code)]
    pub fn build_market_short(&self) -> String {
        let state = match &self.current_state {
            Some(s) => s,
            None => return String::new(),
        };
        format!(
            "{}\nGREED:{:.1}\n{}\nMARKET: {} {}/{} green, BTC trend={:+} RSI={:.0}",
            state.fingerprint.to_prompt_block(),
            state.greed_index,
            Self::build_clock_line(),
            state.regime,
            state.breadth.green_count,
            state.breadth.total_count,
            state.btc.trend_score,
            state.btc.rsi,
        )
    }

    // ── Private Computation ────────────────────────────────────────

    fn compute_btc_gravity(
        features_map: &HashMap<String, serde_json::Value>,
    ) -> BtcGravity {
        let btc = features_map.get("BTC");
        let trend_score = btc
            .and_then(|v| v.get("trend_score"))
            .and_then(|v| v.as_i64())
            .unwrap_or(0) as i32;
        let momentum = btc
            .and_then(|v| v.get("momentum_score"))
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let rsi = btc
            .and_then(|v| v.get("rsi"))
            .and_then(|v| v.as_f64())
            .unwrap_or(50.0);
        let quant_regime = btc
            .and_then(|v| v.get("quant_regime"))
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let quant_vol = btc
            .and_then(|v| v.get("quant_vol"))
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);

        // Crash detection: EMA spread collapsing + high volatility
        let is_crashing = quant_regime < -0.005 && quant_vol > 0.02;

        // Dominance hint from momentum direction
        let dominance_hint = if momentum > 0.005 && trend_score >= 2 {
            "risk_on".to_string()
        } else if momentum < -0.005 && trend_score <= -2 {
            "risk_off".to_string()
        } else {
            "neutral".to_string()
        };

        BtcGravity {
            trend_score,
            momentum,
            rsi,
            is_crashing,
            dominance_hint,
        }
    }

    fn compute_breadth(
        features_map: &HashMap<String, serde_json::Value>,
    ) -> Breadth {
        let mut green = 0usize;
        let mut red = 0usize;
        let mut total = 0usize;
        let mut rsi_sum = 0.0f64;
        let mut momentum_sum = 0.0f64;
        let mut vol_sum = 0.0f64;
        let mut imb_sum = 0.0f64;

        for (sym, feats) in features_map {
            if sym.starts_with("__") {
                continue;
            }
            total += 1;

            let momentum = feats
                .get("momentum_score")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let rsi = feats
                .get("rsi")
                .and_then(|v| v.as_f64())
                .unwrap_or(50.0);
            let vol_ratio = feats
                .get("vol_ratio")
                .and_then(|v| v.as_f64())
                .unwrap_or(1.0);
            let book_imb = feats
                .get("book_imbalance")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            if momentum > 0.0 {
                green += 1;
            } else {
                red += 1;
            }

            rsi_sum += rsi;
            momentum_sum += momentum;
            vol_sum += vol_ratio;
            imb_sum += book_imb;
        }

        let n = total.max(1) as f64;
        Breadth {
            green_count: green,
            red_count: red,
            total_count: total,
            avg_rsi: rsi_sum / n,
            avg_momentum: momentum_sum / n,
            total_vol_ratio: vol_sum / n,
            net_book_pressure: imb_sum / n,
        }
    }

    fn compute_sectors(
        features_map: &HashMap<String, serde_json::Value>,
        meme_list: &[String],
    ) -> Vec<SectorStats> {
        let mut buckets: HashMap<&str, Vec<(f64, f64, f64)>> = HashMap::new(); // (momentum, trend, vol)

        for (sym, feats) in features_map {
            if sym.starts_with("__") {
                continue;
            }
            let sector = sector_for(sym, meme_list);
            let momentum = feats
                .get("momentum_score")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let trend = feats
                .get("trend_score")
                .and_then(|v| v.as_i64())
                .unwrap_or(0) as f64;
            let vol = feats
                .get("vol_ratio")
                .and_then(|v| v.as_f64())
                .unwrap_or(1.0);
            buckets.entry(sector).or_default().push((momentum, trend, vol));
        }

        let mut sectors: Vec<SectorStats> = buckets
            .into_iter()
            .map(|(name, coins)| {
                let n = coins.len().max(1) as f64;
                let avg_momentum = coins.iter().map(|c| c.0).sum::<f64>() / n;
                let avg_trend = coins.iter().map(|c| c.1).sum::<f64>() / n;
                let avg_vol = coins.iter().map(|c| c.2).sum::<f64>() / n;
                let green_count = coins.iter().filter(|c| c.0 > 0.0).count();
                SectorStats {
                    name: name.to_string(),
                    coin_count: coins.len(),
                    avg_momentum,
                    avg_trend,
                    avg_vol_ratio: avg_vol,
                    green_pct: green_count as f64 / n,
                }
            })
            .collect();

        // Sort by avg_momentum descending (hot sectors first)
        sectors.sort_by(|a, b| b.avg_momentum.partial_cmp(&a.avg_momentum).unwrap_or(std::cmp::Ordering::Equal));
        sectors
    }

    fn classify_regime(btc: &BtcGravity, breadth: &Breadth) -> MarketRegime {
        let total = breadth.total_count.max(1) as f64;
        let red_pct = breadth.red_count as f64 / total;
        let green_pct = breadth.green_count as f64 / total;

        if btc.is_crashing && red_pct > 0.70 {
            return MarketRegime::Crash;
        }
        if red_pct > 0.70 && breadth.avg_momentum < -0.01 {
            return MarketRegime::Bearish;
        }
        if green_pct > 0.80 && breadth.avg_rsi > 70.0 {
            return MarketRegime::Euphoric;
        }
        if green_pct > 0.60 && breadth.avg_momentum > 0.005 {
            return MarketRegime::Bullish;
        }
        MarketRegime::Sideways
    }

    fn build_correlation_note(
        features_map: &HashMap<String, serde_json::Value>,
    ) -> String {
        let mut corr_btc_sum = 0.0f64;
        let mut corr_eth_sum = 0.0f64;
        let mut count = 0usize;

        for (sym, feats) in features_map {
            if sym.starts_with("__") || sym == "BTC" || sym == "ETH" {
                continue;
            }
            let corr_btc = feats
                .get("quant_corr_btc")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let corr_eth = feats
                .get("quant_corr_eth")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            corr_btc_sum += corr_btc;
            corr_eth_sum += corr_eth;
            count += 1;
        }

        if count == 0 {
            return String::new();
        }

        let avg_btc = corr_btc_sum / count as f64;
        let avg_eth = corr_eth_sum / count as f64;

        if avg_btc > 0.7 {
            format!("Correlation: alts tightly coupled to BTC (avg {:.2})", avg_btc)
        } else if avg_btc < 0.3 {
            format!(
                "Correlation: alts decoupling from BTC (avg {:.2}), ETH avg {:.2}",
                avg_btc, avg_eth,
            )
        } else {
            format!("Correlation: moderate BTC coupling (avg {:.2})", avg_btc)
        }
    }
}

// ── Funding Rate Intelligence ─────────────────────────────────────

/// Regime-based confidence base (from market physics balance ratios).
pub fn regime_confidence_base(regime: MarketRegime) -> f64 {
    match regime {
        MarketRegime::Euphoric => 0.70,
        MarketRegime::Bullish  => 0.70,
        MarketRegime::Sideways => 0.55, // reversion base (actual reversion conf=0.85 handled in prompt)
        MarketRegime::Bearish  => 0.35, // low conf for longs in bearish
        MarketRegime::Crash    => 0.10, // almost no long confidence in crash
    }
}

/// Fetch funding rates from Kraken Futures API.
/// Returns parsed snapshot or None on error.
pub async fn fetch_funding_rates(client: &reqwest::Client) -> Option<FundingSnapshot> {
    let url = "https://futures.kraken.com/derivatives/api/v3/tickers";
    let resp = client.get(url).timeout(std::time::Duration::from_secs(10)).send().await.ok()?;
    let body: serde_json::Value = resp.json().await.ok()?;
    let tickers = body.get("tickers")?.as_array()?;

    let mut coins: Vec<CoinFunding> = Vec::new();
    let mut total_rate = 0.0f64;
    let mut long_count = 0usize;
    let mut short_count = 0usize;
    let mut neutral_count = 0usize; // used in log output
    let mut perp_count = 0usize;

    for t in tickers {
        let tag = t.get("tag").and_then(|v| v.as_str()).unwrap_or("");
        if tag != "perpetual" { continue; }

        let pair = t.get("pair").and_then(|v| v.as_str()).unwrap_or("");
        let sym = pair.split(':').next().unwrap_or("").to_string();
        if sym.is_empty() { continue; }

        let rate = t.get("fundingRate").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let oi = t.get("openInterest").and_then(|v| v.as_f64()).unwrap_or(0.0);

        let bias = CrowdBias::from_rate(rate);
        total_rate += rate;
        perp_count += 1;

        match bias {
            CrowdBias::ExtremeLong | CrowdBias::LongCrowded | CrowdBias::SlightLong => long_count += 1,
            CrowdBias::ExtremeShort | CrowdBias::ShortCrowded | CrowdBias::SlightShort => short_count += 1,
            CrowdBias::Neutral => neutral_count += 1,
        }

        coins.push(CoinFunding { symbol: sym, rate, open_interest: oi, crowd_bias: bias });
    }

    let avg_rate = if perp_count > 0 { total_rate / perp_count as f64 } else { 0.0 };
    let sentiment = if short_count as f64 > long_count as f64 * 1.5 {
        "BEARISH".to_string()
    } else if long_count as f64 > short_count as f64 * 1.5 {
        "BULLISH".to_string()
    } else {
        "MIXED".to_string()
    };

    tracing::info!(
        "[FUNDING] Fetched {} perps | avg={:+.8} | longs={} shorts={} neutral={} | crowd={}",
        perp_count, avg_rate, long_count, short_count, neutral_count, sentiment,
    );

    Some(FundingSnapshot {
        coins,
        market_avg_rate: avg_rate,
        long_count,
        short_count,
        crowd_sentiment: sentiment,
    })
}

/// Spawn a background task that refreshes funding rates every 60s.
pub fn spawn_funding_fetcher(shared: SharedFunding) {
    tokio::spawn(async move {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(15))
            .build()
            .unwrap_or_default();

        loop {
            if let Some(snap) = fetch_funding_rates(&client).await {
                let mut w = shared.write().await;
                *w = snap;
            }
            tokio::time::sleep(std::time::Duration::from_secs(60)).await;
        }
    });
}

/// Build a funding context block for Nemo's prompts.
pub fn build_funding_block(snap: &FundingSnapshot, symbols: &[String]) -> String {
    if snap.coins.is_empty() {
        return String::new();
    }

    // Compact funding: FUND:BEAR|avg+0.0035|L45/S76|BTC+0.0035L
    let crowd_short = match snap.crowd_sentiment.as_str() {
        "BEARISH" => "BEAR",
        "BULLISH" => "BULL",
        _ => "MIX",
    };
    let mut text = format!(
        "FUND:{}|avg{:+.4}|L{}/S{}",
        crowd_short, snap.market_avg_rate, snap.long_count, snap.short_count,
    );

    for sym in symbols {
        let mut matches: Vec<&CoinFunding> = snap.coins.iter()
            .filter(|c| c.symbol.eq_ignore_ascii_case(sym) || c.symbol.eq_ignore_ascii_case(&format!("X{sym}")))
            .collect();
        matches.sort_by(|a, b| b.open_interest.partial_cmp(&a.open_interest).unwrap_or(std::cmp::Ordering::Equal));

        if let Some(cf) = matches.first() {
            if cf.crowd_bias != CrowdBias::Neutral {
                let bias = match cf.crowd_bias {
                    CrowdBias::ExtremeLong | CrowdBias::LongCrowded | CrowdBias::SlightLong => "L",
                    CrowdBias::ExtremeShort | CrowdBias::ShortCrowded | CrowdBias::SlightShort => "S",
                    CrowdBias::Neutral => "",
                };
                text.push_str(&format!("|{}{:+.4}{}", sym, cf.rate, bias));
            }
        }
    }

    let contrarian = if snap.crowd_sentiment == "BEARISH" {
        "Contrarian: crowd bearish → look for long squeezes"
    } else if snap.crowd_sentiment == "BULLISH" {
        "Contrarian: crowd bullish → be cautious on longs"
    } else {
        "Contrarian: mixed crowd → no strong contrarian signal"
    };
    text.push_str(contrarian);
    text.push('\n');

    text
}

// ── Sector Classification ──────────────────────────────────────────

pub(crate) fn sector_for<'a>(symbol: &str, meme_list: &[String]) -> &'a str {
    let upper = symbol.to_uppercase();
    if meme_list.iter().any(|m| m.eq_ignore_ascii_case(&upper)) {
        return "Memes";
    }
    match upper.as_str() {
        "BTC" | "ETH" => "Majors",
        "SOL" | "AVAX" | "NEAR" | "ATOM" | "ADA" | "DOT" => "L1",
        "LINK" | "UNI" | "AAVE" | "CRV" | "MKR" => "DeFi",
        "XRP" | "XLM" => "Payments",
        "MATIC" | "ARB" | "OP" => "L2",
        _ => "Other",
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_coin(momentum: f64, trend: i32, rsi: f64, vol: f64, imb: f64) -> serde_json::Value {
        serde_json::json!({
            "momentum_score": momentum,
            "trend_score": trend,
            "rsi": rsi,
            "vol_ratio": vol,
            "book_imbalance": imb,
            "buy_ratio": 0.55,
            "quant_corr_btc": 0.6,
            "quant_corr_eth": 0.5,
            "quant_regime": 0.0,
            "quant_vol": 0.01,
        })
    }

    #[test]
    fn breadth_counts_green_red() {
        let mut map = HashMap::new();
        map.insert("BTC".into(), make_coin(0.01, 3, 55.0, 1.2, 0.1));
        map.insert("ETH".into(), make_coin(-0.005, -1, 48.0, 0.8, -0.1));
        map.insert("SOL".into(), make_coin(0.02, 4, 62.0, 1.5, 0.2));

        let breadth = MarketBrain::compute_breadth(&map);
        assert_eq!(breadth.green_count, 2);
        assert_eq!(breadth.red_count, 1);
        assert_eq!(breadth.total_count, 3);
    }

    #[test]
    fn crash_regime_detected() {
        let btc = BtcGravity {
            trend_score: -4,
            momentum: -0.02,
            rsi: 25.0,
            is_crashing: true,
            dominance_hint: "risk_off".into(),
        };
        let breadth = Breadth {
            green_count: 1,
            red_count: 9,
            total_count: 10,
            avg_rsi: 35.0,
            avg_momentum: -0.015,
            total_vol_ratio: 0.5,
            net_book_pressure: -0.3,
        };
        assert_eq!(MarketBrain::classify_regime(&btc, &breadth), MarketRegime::Crash);
    }

    #[test]
    fn bullish_regime_detected() {
        let btc = BtcGravity {
            trend_score: 3,
            momentum: 0.01,
            rsi: 58.0,
            is_crashing: false,
            dominance_hint: "risk_on".into(),
        };
        let breadth = Breadth {
            green_count: 8,
            red_count: 2,
            total_count: 10,
            avg_rsi: 58.0,
            avg_momentum: 0.008,
            total_vol_ratio: 1.3,
            net_book_pressure: 0.15,
        };
        assert_eq!(MarketBrain::classify_regime(&btc, &breadth), MarketRegime::Bullish);
    }

    #[test]
    fn market_block_not_empty_after_update() {
        let mut brain = MarketBrain::new();
        let mut map = HashMap::new();
        map.insert("BTC".into(), make_coin(0.01, 3, 55.0, 1.2, 0.1));
        map.insert("ETH".into(), make_coin(0.005, 2, 52.0, 1.0, 0.05));

        let toml_str = r#"
[meta]
version = 1
[scoring]
bias = 0.0
[scoring.weights]
trend = 0.25
momentum = 0.18
volatility = 0.08
volume = 0.15
mean_revert = 0.08
orderflow = 0.05
quant = 0.03
universe = 0.05
whale = 0.05
cross_radar = 0.04
ml_signal = 0.04
[scoring.norm]
trend_score = [-5.0, 5.0]
macd_hist = [-1.0, 1.0]
rsi = [20.0, 80.0]
atr_pct_mult = 0.05
vol_ratio = [0.5, 2.0]
zscore = [-2.5, 2.5]
book_imbalance = [-1.0, 1.0]
buy_ratio = [0.3, 0.7]
quant_momo = [-0.03, 0.03]
quant_regime = [-0.01, 0.01]
[mlp.norm]
trend_score = [-5.0, 5.0]
macd_hist = [-1.0, 1.0]
rsi = [0.0, 100.0]
atr_pct = [0.0, 10.0]
vol_ratio = [0.0, 5.0]
zscore = [-3.0, 3.0]
market_state = [0.0, 1.0]
book_imbalance = [-1.0, 1.0]
buy_ratio = [0.0, 1.0]
spread_pct = [0.0, 2.0]
buy_slip_pct_100k = [0.0, 5.0]
quant_momo = [-0.05, 0.05]
quant_regime = [-0.02, 0.02]
quant_vol = [0.0, 0.05]
quant_liquidity = [0.0, 1.0]
rsi_p20 = [0.0, 50.0]
rsi_p80 = [50.0, 100.0]
trend_strength = [-5.0, 5.0]
markov_score = [-1.0, 1.0]
markov_prob = [0.0, 1.0]
[mlp.crash_flag]
quant_regime_le = -0.005
quant_vol_ge = 0.02
[signals.crash]
quant_regime_le = -0.005
quant_vol_ge = 0.02
confidence = 0.90
[signals.hunt]
imb_ge = 0.35
buy_ratio_ge = 0.50
momo_gt = 0.0
confidence = 0.85
[signals.buy]
imb_ge = 0.15
buy_ratio_ge = 0.50
momo_ge = 0.0
conf_base = 0.55
conf_mult = 0.45
conf_cap = 0.80
[signals.sell]
imb_le = -0.20
momo_lt = -0.001
buy_ratio_lt = 0.45
conf_base = 0.50
conf_mult = 0.50
conf_cap = 0.85
[signals.hold]
confidence = 0.60
[filter.global]
min_liquidity = 0.10
max_spread_pct = 1.5
min_vol_ratio = 0.20
[filter.lane1]
min_vol_ratio = 0.30
max_rsi = 72.0
min_trend_score = 1
[filter.lane2]
max_bb_width = 3.0
max_quant_vol = 0.03
max_ema50_dist = 2.0
max_atr_pct = 4.0
min_book_imb = 0.10
[filter.lane3]
min_rsi = 35.0
max_rsi = 70.0
max_quant_risk = 0.02
min_trend_score = 1
min_momo = 0.001
[filter.markov]
min_bull_prob = 0.45
min_score = 0.10
enabled = true
[memes]
symbols = ["DOGE","SHIB"]
[strategy]
type = "softmax"
H = 21
K = 12
w_max = 0.15
delta = 0.01
tau = 1.0
alpha = 1.0
beta = 0.5
gamma = 0.2
"#;
        let strategy: crate::strategy::StrategyToml = toml::from_str(toml_str).unwrap();
        brain.update(&map, &strategy);
        let block = brain.build_market_block();
        assert!(block.contains("FP_1d:["), "FP_1d missing from market block");
        assert!(block.contains("FP_30m:["), "fingerprint missing from market block");
        assert!(block.contains("GREED:"), "GREED missing from market block");
        assert!(block.contains("MKT:"), "MKT: line missing from market block");
        assert!(block.contains("BTC:"), "BTC: missing from market block");
    }
}
