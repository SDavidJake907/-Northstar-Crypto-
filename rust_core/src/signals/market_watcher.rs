//! Market Watcher — higher-level cross-coin intelligence for Qwen.
//!
//! Computes 6 signals that individual coin analysis misses:
//! 1. Multi-timeframe alignment (trend+momo+hurst consensus)
//! 2. Sector rotation detection (momentum flow between sectors)
//! 3. Volume anomaly detection (spikes + market-wide surges)
//! 4. Correlation shift detection (alts decoupling from BTC)
//! 5. Whale trade detection (extreme volume ratio + flow direction)
//! 6. Momentum divergence (RSI/price divergence per coin)
//!
//! Produces a compact WATCH: text block injected into AI prompts.
//! Follows same pattern as MarketBrain: called per tick, throttles internally.

use std::collections::{HashMap, VecDeque};

// ── Constants ──────────────────────────────────────────────────────

const ROLLING_WINDOW: usize = 30; // ~30 readings for divergence/history

// ── Alert Types ────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct WatchAlert {
    pub signal: WatchSignal,
    pub severity: AlertSeverity,
    pub text: String,
    pub ts: f64,
}

#[derive(Clone, Debug)]
pub enum WatchSignal {
    MultiTimeframeAlignment,
    SectorRotation,
    VolumeAnomaly,
    CorrelationShift,
    WhaleTrade,
    MomentumDivergence,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
}

impl std::fmt::Display for WatchSignal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MultiTimeframeAlignment => write!(f, "MTF"),
            Self::SectorRotation => write!(f, "Rotation"),
            Self::VolumeAnomaly => write!(f, "Volume"),
            Self::CorrelationShift => write!(f, "Correlation"),
            Self::WhaleTrade => write!(f, "Whale"),
            Self::MomentumDivergence => write!(f, "Divergence"),
        }
    }
}

impl std::fmt::Display for AlertSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Info => write!(f, "Info"),
            Self::Warning => write!(f, "Warning"),
            Self::Critical => write!(f, "Critical"),
        }
    }
}

// ── Rolling Per-Coin State ─────────────────────────────────────────

#[derive(Clone, Debug, Default)]
struct CoinRollingState {
    rsi_history: VecDeque<f64>,
    price_history: VecDeque<f64>,
}

#[derive(Clone, Debug)]
struct SectorSnapshot {
    name: String,
    avg_momentum: f64,
}

// ── MarketWatcher ──────────────────────────────────────────────────

pub struct MarketWatcher {
    coin_state: HashMap<String, CoinRollingState>,
    sector_snapshots: VecDeque<Vec<SectorSnapshot>>,
    active_alerts: Vec<WatchAlert>,
    last_compute_ts: f64,
    compute_interval_sec: f64,
}

impl MarketWatcher {
    pub fn new() -> Self {
        Self {
            coin_state: HashMap::new(),
            sector_snapshots: VecDeque::new(),
            active_alerts: Vec::new(),
            last_compute_ts: 0.0,
            compute_interval_sec: parse_env_f64("WATCHER_INTERVAL_SEC", 30.0),
        }
    }

    /// Called every tick from trading_loop. Accumulates rolling state always,
    /// recomputes signals only every `compute_interval_sec`.
    pub fn update(
        &mut self,
        features_map: &HashMap<String, serde_json::Value>,
        strategy: &crate::strategy::StrategyToml,
    ) {
        let now = crate::ai_bridge::now_ts();

        // Always accumulate rolling state (cheap)
        self.accumulate_rolling(features_map);

        // Throttle full signal computation
        if now - self.last_compute_ts < self.compute_interval_sec {
            return;
        }

        // Hot-reload interval
        self.compute_interval_sec = parse_env_f64("WATCHER_INTERVAL_SEC", 30.0);
        self.last_compute_ts = now;

        let mut alerts = Vec::new();

        // 1. Multi-timeframe alignment
        if let Some(a) = self.check_mtf_alignment(features_map, now) {
            alerts.push(a);
        }

        // 2. Sector rotation
        self.accumulate_sector_snapshot(features_map, &strategy.memes.symbols, now);
        if let Some(a) = self.check_sector_rotation(now) {
            alerts.push(a);
        }

        // 3. Volume anomaly
        if let Some(a) = Self::check_volume_anomaly(features_map, now) {
            alerts.push(a);
        }

        // 4. Correlation shift
        if let Some(a) = Self::check_correlation_shift(features_map, now) {
            alerts.push(a);
        }

        // 5. Whale trade detection
        alerts.extend(Self::check_whale_signals(features_map, now));

        // 6. Momentum divergence
        alerts.extend(self.check_momentum_divergence(features_map, now));

        // Sort by severity (Critical first), keep top 3
        alerts.sort_by(|a, b| b.severity.cmp(&a.severity));
        alerts.truncate(3);

        if !alerts.is_empty() {
            tracing::info!(
                "[WATCHER] {} alerts: {}",
                alerts.len(),
                alerts.iter().map(|a| a.text.as_str()).collect::<Vec<_>>().join(" | ")
            );
        }

        self.active_alerts = alerts;
    }

    /// Compact WATCH: text block for AI prompts. Empty if no alerts.
    #[allow(dead_code)]
    pub fn build_watch_block(&self) -> String {
        if self.active_alerts.is_empty() {
            return String::new();
        }
        let texts: Vec<&str> = self.active_alerts.iter().map(|a| a.text.as_str()).collect();
        format!("WATCH:{}", texts.join("|"))
    }

    /// Active alerts for telemetry publishing.
    pub fn alerts(&self) -> &[WatchAlert] {
        &self.active_alerts
    }

    // ── Rolling Accumulation ───────────────────────────────────────

    fn accumulate_rolling(&mut self, features_map: &HashMap<String, serde_json::Value>) {
        for (sym, feats) in features_map {
            if sym.starts_with("__") {
                continue;
            }
            let cs = self.coin_state.entry(sym.clone()).or_default();

            let rsi = feats.get("rsi").and_then(|v| v.as_f64()).unwrap_or(50.0);
            let price = feats.get("price").and_then(|v| v.as_f64()).unwrap_or(0.0);

            push_rolling(&mut cs.rsi_history, rsi);
            push_rolling(&mut cs.price_history, price);
        }
    }

    fn accumulate_sector_snapshot(
        &mut self,
        features_map: &HashMap<String, serde_json::Value>,
        meme_list: &[String],
        _now: f64,
    ) {
        let mut buckets: HashMap<&str, Vec<f64>> = HashMap::new();
        for (sym, feats) in features_map {
            if sym.starts_with("__") {
                continue;
            }
            let sector = crate::market_brain::sector_for(sym, meme_list);
            let momo = feats
                .get("momentum_score")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            buckets.entry(sector).or_default().push(momo);
        }

        let snapshot: Vec<SectorSnapshot> = buckets
            .into_iter()
            .map(|(name, momos)| {
                let avg = momos.iter().sum::<f64>() / momos.len().max(1) as f64;
                SectorSnapshot {
                    name: name.to_string(),
                    avg_momentum: avg,
                }
            })
            .collect();

        self.sector_snapshots.push_back(snapshot);
        if self.sector_snapshots.len() > 30 {
            self.sector_snapshots.pop_front();
        }
    }

    // ── Signal 1: Multi-Timeframe Alignment ────────────────────────

    fn check_mtf_alignment(
        &self,
        features_map: &HashMap<String, serde_json::Value>,
        now: f64,
    ) -> Option<WatchAlert> {
        let mut bull_aligned = 0usize;
        let mut bear_aligned = 0usize;
        let mut divergent = 0usize;
        let mut total = 0usize;

        for (sym, feats) in features_map {
            if sym.starts_with("__") {
                continue;
            }
            total += 1;

            let trend = feats.get("trend_score").and_then(|v| v.as_i64()).unwrap_or(0);
            let momo = feats
                .get("momentum_score")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let qmomo = feats
                .get("quant_momo")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let hurst = feats
                .get("hurst_exp")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.5);

            let long_bull = trend >= 3;
            let short_bull = momo > 0.15 && qmomo > 0.005;
            let long_bear = trend <= -3;
            let short_bear = momo < -0.15 && qmomo < -0.005;

            if long_bull && short_bull && hurst > 0.52 {
                bull_aligned += 1;
            } else if long_bear && short_bear && hurst > 0.52 {
                bear_aligned += 1;
            } else if (long_bull && short_bear) || (long_bear && short_bull) {
                divergent += 1;
            }
        }

        if total == 0 {
            return None;
        }
        let bull_pct = bull_aligned as f64 / total as f64;
        let bear_pct = bear_aligned as f64 / total as f64;
        let div_pct = divergent as f64 / total as f64;

        if bull_pct > 0.60 {
            Some(WatchAlert {
                signal: WatchSignal::MultiTimeframeAlignment,
                severity: AlertSeverity::Warning,
                text: format!("MTF:{:.0}%bull-aligned", bull_pct * 100.0),
                ts: now,
            })
        } else if bear_pct > 0.60 {
            Some(WatchAlert {
                signal: WatchSignal::MultiTimeframeAlignment,
                severity: AlertSeverity::Critical,
                text: format!("MTF:{:.0}%bear-aligned—CAUTION", bear_pct * 100.0),
                ts: now,
            })
        } else if div_pct > 0.40 {
            Some(WatchAlert {
                signal: WatchSignal::MultiTimeframeAlignment,
                severity: AlertSeverity::Info,
                text: format!("MTF:{:.0}%divergent(pullback?)", div_pct * 100.0),
                ts: now,
            })
        } else {
            None
        }
    }

    // ── Signal 2: Sector Rotation ──────────────────────────────────

    fn check_sector_rotation(&self, now: f64) -> Option<WatchAlert> {
        if self.sector_snapshots.len() < 10 {
            return None; // need ~5 min of data
        }

        let current = self.sector_snapshots.back()?;
        let oldest = self.sector_snapshots.front()?;

        let mut best_gain: (&str, f64) = ("", 0.0);
        let mut worst_drop: (&str, f64) = ("", 0.0);

        for cur in current {
            if let Some(old) = oldest.iter().find(|s| s.name == cur.name) {
                let delta = cur.avg_momentum - old.avg_momentum;
                if delta > best_gain.1 {
                    best_gain = (&cur.name, delta);
                }
                if delta < worst_drop.1 {
                    worst_drop = (&cur.name, delta);
                }
            }
        }

        if best_gain.1 > 0.05 && worst_drop.1 < -0.03 {
            Some(WatchAlert {
                signal: WatchSignal::SectorRotation,
                severity: AlertSeverity::Warning,
                text: format!(
                    "ROTATION:{}(+{:.2})←{}({:.2})",
                    best_gain.0, best_gain.1, worst_drop.0, worst_drop.1
                ),
                ts: now,
            })
        } else {
            None
        }
    }

    // ── Signal 3: Volume Anomaly ───────────────────────────────────

    fn check_volume_anomaly(
        features_map: &HashMap<String, serde_json::Value>,
        now: f64,
    ) -> Option<WatchAlert> {
        let mut spikes: Vec<(&str, f64)> = Vec::new();
        let mut avg_vol_ratio = 0.0;
        let mut count = 0usize;

        for (sym, feats) in features_map {
            if sym.starts_with("__") {
                continue;
            }
            let vr = feats
                .get("vol_ratio")
                .and_then(|v| v.as_f64())
                .unwrap_or(1.0);
            avg_vol_ratio += vr;
            count += 1;
            if vr >= 3.0 {
                spikes.push((sym.as_str(), vr));
            }
        }

        if count == 0 {
            return None;
        }
        avg_vol_ratio /= count as f64;

        if !spikes.is_empty() {
            spikes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let top = &spikes[..spikes.len().min(3)];
            let names: Vec<String> = top
                .iter()
                .map(|(s, v)| format!("{}({:.1}x)", s, v))
                .collect();
            Some(WatchAlert {
                signal: WatchSignal::VolumeAnomaly,
                severity: if avg_vol_ratio > 2.0 {
                    AlertSeverity::Critical
                } else {
                    AlertSeverity::Warning
                },
                text: format!("VOL-SPIKE:{} avg{:.1}x", names.join(","), avg_vol_ratio),
                ts: now,
            })
        } else if avg_vol_ratio > 1.8 {
            Some(WatchAlert {
                signal: WatchSignal::VolumeAnomaly,
                severity: AlertSeverity::Info,
                text: format!("VOL-ELEVATED:avg{:.1}x", avg_vol_ratio),
                ts: now,
            })
        } else {
            None
        }
    }

    // ── Signal 4: Correlation Shift ────────────────────────────────

    fn check_correlation_shift(
        features_map: &HashMap<String, serde_json::Value>,
        now: f64,
    ) -> Option<WatchAlert> {
        let mut corrs: Vec<f64> = Vec::new();
        for (sym, feats) in features_map {
            if sym.starts_with("__") || sym == "BTC" {
                continue;
            }
            let corr = feats
                .get("quant_corr_btc")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            corrs.push(corr);
        }
        if corrs.is_empty() {
            return None;
        }

        let avg = corrs.iter().sum::<f64>() / corrs.len() as f64;
        let low_corr_count = corrs.iter().filter(|&&c| c < 0.3).count();
        let pct_decoupled = low_corr_count as f64 / corrs.len() as f64;

        if avg < 0.2 || pct_decoupled > 0.50 {
            Some(WatchAlert {
                signal: WatchSignal::CorrelationShift,
                severity: AlertSeverity::Critical,
                text: format!(
                    "CORR-BREAK:avg{:.2},{:.0}%decoupled",
                    avg,
                    pct_decoupled * 100.0
                ),
                ts: now,
            })
        } else if avg < 0.4 {
            Some(WatchAlert {
                signal: WatchSignal::CorrelationShift,
                severity: AlertSeverity::Info,
                text: format!("CORR-LOOSE:avg{:.2}", avg),
                ts: now,
            })
        } else {
            None
        }
    }

    // ── Signal 5: Whale Trade Detection ────────────────────────────

    fn check_whale_signals(
        features_map: &HashMap<String, serde_json::Value>,
        now: f64,
    ) -> Vec<WatchAlert> {
        let mut alerts = Vec::new();
        for (sym, feats) in features_map {
            if sym.starts_with("__") {
                continue;
            }
            let vr = feats
                .get("vol_ratio")
                .and_then(|v| v.as_f64())
                .unwrap_or(1.0);
            if vr < 5.0 {
                continue;
            }

            let flow_imb = feats
                .get("flow_imbalance")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let dir = if flow_imb > 0.15 {
                "BUY"
            } else if flow_imb < -0.15 {
                "SELL"
            } else {
                "MIXED"
            };

            alerts.push(WatchAlert {
                signal: WatchSignal::WhaleTrade,
                severity: AlertSeverity::Warning,
                text: format!("WHALE:{}({:.0}x,{})", sym, vr, dir),
                ts: now,
            });
        }
        alerts
    }

    // ── Signal 6: Momentum Divergence ──────────────────────────────

    fn check_momentum_divergence(
        &self,
        features_map: &HashMap<String, serde_json::Value>,
        now: f64,
    ) -> Vec<WatchAlert> {
        let mut alerts = Vec::new();
        for (sym, cs) in &self.coin_state {
            if cs.price_history.len() < 10 || cs.rsi_history.len() < 10 {
                continue;
            }
            let feats = match features_map.get(sym) {
                Some(f) => f,
                None => continue,
            };

            let cur_price = feats.get("price").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let cur_rsi = feats.get("rsi").and_then(|v| v.as_f64()).unwrap_or(50.0);

            let lookback = cs.price_history.len().min(15);
            let old_price = cs.price_history[cs.price_history.len() - lookback];
            let old_rsi = cs.rsi_history[cs.rsi_history.len() - lookback];

            let price_pct = if old_price > 0.0 {
                (cur_price - old_price) / old_price
            } else {
                0.0
            };
            let rsi_change = cur_rsi - old_rsi;

            // Bearish divergence: price up, RSI down
            if price_pct > 0.01 && rsi_change < -5.0 && cur_rsi > 55.0 {
                alerts.push(WatchAlert {
                    signal: WatchSignal::MomentumDivergence,
                    severity: AlertSeverity::Warning,
                    text: format!(
                        "DIV-BEAR:{}(+{:.1}%,RSI{:.0}->{:.0})",
                        sym,
                        price_pct * 100.0,
                        old_rsi,
                        cur_rsi
                    ),
                    ts: now,
                });
            }
            // Bullish divergence: price down, RSI up
            else if price_pct < -0.01 && rsi_change > 5.0 && cur_rsi < 45.0 {
                alerts.push(WatchAlert {
                    signal: WatchSignal::MomentumDivergence,
                    severity: AlertSeverity::Warning,
                    text: format!(
                        "DIV-BULL:{}({:.1}%,RSI{:.0}->{:.0})",
                        sym,
                        price_pct * 100.0,
                        old_rsi,
                        cur_rsi
                    ),
                    ts: now,
                });
            }
        }
        alerts.truncate(2); // max 2 divergence alerts
        alerts
    }
}

// ── Helpers ────────────────────────────────────────────────────────

fn push_rolling(dq: &mut VecDeque<f64>, val: f64) {
    dq.push_back(val);
    if dq.len() > ROLLING_WINDOW {
        dq.pop_front();
    }
}

fn parse_env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}
