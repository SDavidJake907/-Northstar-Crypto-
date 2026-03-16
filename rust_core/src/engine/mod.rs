//! Main trading loop orchestrator — the heart of Hybrid-Kraken.
//!
//! Ported from python_orch/tinyllm_trader.py
//!
//! Flow per tick:
//!   1. Hot-reload config, reset daily state, check circuit breaker
//!   2. Read features from internal state (computed by ws/book/features engine)
//!   3. For each symbol:
//!      a. Infer regime, select profile
//!      b. Call AI bridge (NPU Qwen 1.5B pre-scan + GPU Qwen 14B decisions)
//!      c. Apply gate checks (rules_eval)
//!      d. Size position (portfolio_optimizer)
//!      e. Execute order (kraken_api)
//!      f. Check exits (TP/SL/trailing/regime/AI)
//!   4. Journal trades, write heartbeat, sleep

mod exits;
mod tick;
mod entries;
mod tools;
mod portfolio;
mod orders;
mod helpers;
pub mod regime_sm;
pub mod lane;
pub mod fee_filter;
pub mod decision_log;
pub use helpers::*;
pub(crate) use orders::{save_positions, load_positions};
pub mod journal;
pub mod trade_flow;
pub mod rules_eval;
pub mod portfolio_optimizer;
pub mod portfolio_allocator;

use crate::ai_bridge::{AiBridge, AiDecision};
use crate::config::{self, CircuitBreakerState, ProfileParams, TradingConfig};
use crate::kraken_api::KrakenApi;
use crate::strategy_helpers::compute_softmax_weights;
use self::journal::Journal;
use self::portfolio_optimizer::ProfitOptimizer;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;
use std::time::Instant;

// ── Nemo Working Memory ──────────────────────────────────────────
// Tracks last N decisions per coin so Nemo can see her own history.

const NEMO_MEMORY_MAX_PER_COIN: usize = 24;

#[derive(Clone, Debug)]
pub(crate) struct NemoMemoryEntry {
    pub(crate) ts: f64,
    pub(crate) action: String,      // BUY, SELL, HOLD
    pub(crate) confidence: f64,
    pub(crate) reason: String,
    pub(crate) price_at_decision: f64,
    pub(crate) _lane: String,
    pub(crate) source: String,       // "math", "nemo_confirm", "nemo_veto", "nemo_exit"
}

// format_nemo_memory moved to helpers.rs

// ── Position State ───────────────────────────────────────────────

// ── Bayesian Trade Quality Learning (Beta-Bernoulli per setup bucket) ──

#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub(crate) struct BetaStats {
    pub(crate) a: f64,
    pub(crate) b: f64,
}

impl BetaStats {
    pub(crate) fn new(a: f64, b: f64) -> Self { Self { a, b } }
    pub(crate) fn mean(&self) -> f64 { self.a / (self.a + self.b) }
    pub(crate) fn samples(&self) -> f64 { self.a + self.b }
}

#[derive(Clone, Debug)]
pub struct OpenPosition {
    pub symbol: String,
    pub entry_price: f64,
    pub qty: f64,
    pub remaining_qty: f64,
    pub tp_price: f64,
    pub sl_price: f64,
    pub highest_price: f64,
    pub entry_time: f64,
    pub entry_reasons: Vec<String>,
    pub entry_profile: String,
    pub entry_context: String,
    pub entry_score: f64,
    // AI meta at entry
    pub regime_label: String,
    pub quant_bias: String,
    #[allow(dead_code)]
    pub npu_action: String,
    #[allow(dead_code)]
    pub npu_conf: f64,
    /// Kraken stop-loss order txid — must cancel before selling
    pub sl_order_txid: Option<String>,
    /// Bayesian quality bucket key (e.g. "reg=sideways|lane=L2|tier=large|mtf=2|imb=strong")
    pub quality_key: String,
    /// Quality score at entry (conf * p_win * cost_adj)
    pub quality_score: f64,
    /// Time contract from entry decision: don't exit before this (unless hard risk).
    pub min_hold_sec: u32,
    /// Time contract from entry decision: thesis stale after this.
    pub max_hold_sec: u32,
    /// Time contract from entry decision: re-eval interval.
    pub reeval_sec: u32,
    /// ATR at entry time — used for deterministic ATR-based exit stack.
    pub entry_atr: f64,
    /// Lane at entry (L1/L2/L3/L4) — drives ATR trailing multiplier.
    pub entry_lane: String,
    /// Feature snapshot at entry time — 44 f64 values for GBDT training.
    pub feature_snapshot: Option<Vec<f64>>,
    /// Multi-timeframe trend alignment at entry:
    /// 0=none, 1=1d only, 2=1d+7d aligned, 3=all three aligned
    pub trend_alignment: u8,
    /// 7-day trend % at entry time (from mtf_trends.json)
    pub trend_7d_pct: f64,
    /// 30-day trend % at entry time (from mtf_trends.json)
    pub trend_30d_pct: f64,
}

#[derive(Clone, Debug)]
pub(crate) struct PendingEntryOrder {
    pub(crate) txid: String,
    pub(crate) symbol: String,
    pub(crate) _pair: String,
    pub(crate) requested_qty: f64,
    pub(crate) reserved_usd: f64,
    pub(crate) tp_price: f64,
    pub(crate) sl_price: f64,
    pub(crate) entry_profile: String,
    pub(crate) entry_context: String,
    pub(crate) entry_score: f64,
    pub(crate) regime_label: String,
    pub(crate) quant_bias: String,
    pub(crate) npu_action: String,
    pub(crate) npu_conf: f64,
    pub(crate) signal_names: Vec<String>,
    pub(crate) created_ts: f64,
    pub(crate) quality_key: String,
    pub(crate) quality_score: f64,
    pub(crate) min_hold_sec: u32,
    pub(crate) max_hold_sec: u32,
    pub(crate) reeval_sec: u32,
    pub(crate) entry_atr: f64,
    pub(crate) entry_lane: String,
    pub(crate) feature_snapshot: Option<Vec<f64>>,
    pub(crate) trend_alignment: u8,
    pub(crate) trend_7d_pct: f64,
    pub(crate) trend_30d_pct: f64,
}

impl OpenPosition {
    /// Raw P&L % (without fees).
    pub fn pnl_pct_raw(&self, current_price: f64) -> f64 {
        if self.entry_price > 0.0 {
            (current_price - self.entry_price) / self.entry_price
        } else {
            0.0
        }
    }

    /// Net P&L % after round-trip fees (buy + sell).
    pub fn pnl_pct(&self, current_price: f64, fee_pct: f64) -> f64 {
        self.pnl_pct_raw(current_price) - (2.0 * fee_pct / 100.0)
    }

    /// Net P&L USD after round-trip fees.
    pub fn pnl_usd(&self, current_price: f64, fee_pct: f64) -> f64 {
        let gross = self.remaining_qty * (current_price - self.entry_price);
        let fee_cost = self.remaining_qty * self.entry_price * (fee_pct / 100.0)
            + self.remaining_qty * current_price * (fee_pct / 100.0);
        gross - fee_cost
    }

    pub fn hold_seconds(&self) -> f64 {
        now_ts() - self.entry_time
    }
}

// ── Trading Loop State ───────────────────────────────────────────

pub struct TradingLoop {
    pub config: TradingConfig,
    pub strategy: std::sync::Arc<crate::strategy::StrategyToml>,
    pub positions: HashMap<String, OpenPosition>,
    pub(crate) strategy_weights: HashMap<String, f64>,
    pub(crate) prev_strategy_weights: HashMap<String, f64>,
    pub circuit: CircuitBreakerState,
    pub journal: Journal,
    pub optimizer: ProfitOptimizer,
    pub last_entry_ts: f64,
    pub tick_count: u64,
    pub(crate) last_heartbeat: f64,
    pub(crate) last_watchdog_check: f64,
    pub(crate) last_config_reload: Instant,
    #[allow(dead_code)]
    pub(crate) prev_momentum: HashMap<String, f64>,
    // Balance tracking
    pub available_usd: f64,
    pub equity: f64,
    pub(crate) holdings: HashMap<String, f64>,
    pub(crate) last_balance_check: f64,
    pub(crate) pending_entries: HashMap<String, PendingEntryOrder>,
    pub(crate) last_pending_check: f64,
    pub(crate) meters: RuntimeMeters,
    pub(crate) integrity_snapshot: serde_json::Value,
    pub(crate) rest_throttle_hits: u64,
    pub(crate) rest_backoff_hits: u64,
    pub(crate) manual_kill_active: bool,
    pub(crate) watchdog_ok: bool,
    pub(crate) watchdog_fail_count: usize,
    pub(crate) watchdog_block_entries: bool,
    pub(crate) watchdog_last_fail_paths: Vec<String>,
    pub(crate) hardware_block_entries: bool,
    pub(crate) hw_guard_level: HardwareGuardLevel,
    pub(crate) hw_status: HardwareStatus,
    pub(crate) hw_last_sample_ts: f64,
    pub(crate) ai_latency_p95_ms: f64,
    pub(crate) effective_ai_parallel: usize,
    pub(crate) cvar_weights: HashMap<String, f64>,
    pub(crate) cvar_weights_ts: f64,
    pub(crate) cvar_cash_frac: f64,
    pub(crate) portfolio_value: f64,
    pub(crate) gpu_math_logged: bool,
    // Telemetry event bus
    pub(crate) bus: tokio::sync::broadcast::Sender<crate::event_bus::Event>,
    pub(crate) telemetry_level: crate::event_bus::TelemetryLevel,
    pub(crate) prev_coin_state: HashMap<String, (String, String)>,
    pub(crate) last_nemo_exit_check: HashMap<String, f64>,

    // Auto-live on BTC green
    pub(crate) btc_24h_open: f64,
    pub(crate) btc_green_since: Option<Instant>,
    pub(crate) auto_live_active: bool,
    pub(crate) last_btc_ticker_ts: f64,

    // Cooldowns (single-brain mode)
    pub(crate) last_ai_call_ts: HashMap<String, f64>,
    pub(crate) last_trade_ts: HashMap<String, f64>,
    pub(crate) last_entry_price: HashMap<String, f64>,

    // Nemo working memory — last N decisions per coin
    pub(crate) nemo_memory: HashMap<String, std::collections::VecDeque<NemoMemoryEntry>>,

    // Nemo self-optimizer (optional)
    #[cfg(feature = "nemo_optimizer")]
    pub(crate) nemo_optimizer: crate::nemo_optimizer::NemoOptimizer,

    // Nemo conscious brain — outcome memory, self-reflection, episodes, identity
    pub(crate) nemo_memory_brain: crate::nemo_memory::NemoMemory,

    // AuroraKin907 memory packet — adaptive learning layer (5-layer performance feedback)
    pub(crate) memory_packet: crate::memory_packet::CachedMemoryPacket,
    pub(crate) memory_enabled: bool,

    // Captain's intel — Atlas NPU sentiment & news (shared with cloud_intel.rs task)
    pub cloud_intel: Option<crate::cloud_intel::SharedCloudIntel>,

    // Swap queue — exit watchdog writes coin swap suggestions, vision scanner reads
    pub swap_queue: Option<crate::vision_scanner::SharedSwapQueue>,

    // HTTP client for public API calls (get_top_movers, etc.)
    pub(crate) http_client: reqwest::Client,

    // News & Sentiment — Fear/Greed, CoinGecko global, trending (free APIs, every 5min)
    pub news_sentiment: crate::news_sentiment::NewsSentiment,

    // Market brain — BTC gravity, breadth, sector stats, regime classification
    pub(crate) market_brain: crate::market_brain::MarketBrain,

    // Regime State Machine — per-coin, deterministic with dwell + N-of-M confirmation
    pub(crate) regime_machines: HashMap<String, regime_sm::RegimeStateMachine>,

    // HF gate: rolling orderbook imbalance history per coin (last N polls)
    pub(crate) book_imb_hist: HashMap<String, std::collections::VecDeque<f64>>,

    // HF Bayesian trade quality learning — Beta(a,b) per setup bucket
    pub(crate) quality_stats: HashMap<String, BetaStats>,

    // HF adaptive confidence floor — tightens on losses, relaxes on wins
    pub(crate) entry_conf_floor_dyn: f64,
    pub(crate) hf_streak_wins: u32,
    pub(crate) hf_streak_losses: u32,
    /// 3-loss cooldown: block entries until this timestamp (LOSS_COOLDOWN_ENABLED)
    pub(crate) loss_cooldown_until: f64,
    pub(crate) hf_outcome_ema: f64,
    pub(crate) hf_equity_peak: f64,

    // NPU Bridge — Qwen 2.5-1.5B on Intel NPU via npu_engine.dll
    pub npu_bridge: Option<std::sync::Arc<crate::npu_bridge::NpuBridge>>,

    // Phi-3 NPU scan cache — non-blocking background scan, result cached per coin
    pub(crate) npu_scan_cache: std::collections::HashMap<String, String>,
    pub(crate) npu_scan_inflight: HashSet<String>,
    pub(crate) npu_scan_tx: tokio::sync::mpsc::Sender<(String, String, String, String, String)>,
    pub(crate) npu_scan_rx: Option<tokio::sync::mpsc::Receiver<(String, String, String, String, String)>>,

    // GPU Math Sidecar — nvmath-python batch math on port 8084
    pub gpu_math: crate::addons::gpu_math::SharedGpuMath,

    // GBDT — gradient boosted decision trees, learns from trade outcomes
    pub gbdt: crate::signals::gbdt::SharedGbdt,

    // Market watcher — cross-coin intelligence (MTF, rotation, volume, correlation, whales, divergence)
    pub(crate) market_watcher: crate::market_watcher::MarketWatcher,

    // Funding rate intelligence — live from Kraken Futures
    pub(crate) shared_funding: crate::market_brain::SharedFunding,

    // Portfolio Allocator V2 — Qwen-driven portfolio-level allocation + rebalance
    pub(crate) portfolio_allocator: crate::portfolio_allocator::PortfolioAllocator,

}

const BALANCE_CHECK_INTERVAL: f64 = 30.0; // Query Kraken balance every 30s
const FORCE_DISABLE_BREAKERS: bool = false; // Breakers now controlled by DISABLE_ALL_BREAKERS in .env
const WATCHDOG_CHECK_INTERVAL_SEC: f64 = 10.0;
const PENDING_CHECK_INTERVAL_SEC: f64 = 5.0;
// entry_order_timeout_sec moved to helpers.rs
const METERS_WINDOW_SEC: f64 = 60.0;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PendingOrderSnapshot {
    pub(crate) status: String,
    pub(crate) vol_exec: f64,
    pub(crate) cost: f64,
    pub(crate) price: Option<f64>,
    pub(crate) avg_price: Option<f64>,
}

#[derive(Clone, Debug)]
pub(crate) struct RuntimeMeters {
    pub(crate) window_start_ts: f64,
    pub(crate) entry_rejects: HashMap<String, u64>,
    pub(crate) entry_reject_total: u64,
    pub(crate) ai_calls: u64,
    pub(crate) ai_action_buy: u64,
    pub(crate) ai_action_sell: u64,
    pub(crate) ai_action_hold: u64,
    pub(crate) stale_gate_trips: u64,
    pub(crate) confidence_gate_trips: u64,
    pub(crate) model_health_gate_trips: u64,
    pub(crate) data_quality_gate_trips: u64,
    pub(crate) watchdog_gate_trips: u64,
    pub(crate) hardware_guard_trips: u64,
    pub(crate) ai_throttle_hits: u64,
    pub(crate) nemo_exit_checks: u64,
    pub(crate) nemo_exit_holds: u64,
    pub(crate) nemo_exit_sells: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum HardwareGuardLevel {
    Normal,
    SoftDegrade,
    HardBlock,
    Emergency,
}

impl HardwareGuardLevel {
    pub(crate) fn as_str(&self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::SoftDegrade => "soft_degrade",
            Self::HardBlock => "hard_block",
            Self::Emergency => "emergency",
        }
    }

    pub(crate) fn severity(&self) -> u8 {
        match self {
            Self::Normal => 0,
            Self::SoftDegrade => 1,
            Self::HardBlock => 2,
            Self::Emergency => 3,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct HardwareStatus {
    pub(crate) sampled_ts: f64,
    pub(crate) gpu_mem_used_mb: Option<f64>,
    pub(crate) gpu_mem_total_mb: Option<f64>,
    pub(crate) gpu_temp_c: Option<f64>,
    pub(crate) ram_free_mb: Option<f64>,
    pub(crate) ram_total_mb: Option<f64>,
    pub(crate) sample_error: Option<String>,
}

impl RuntimeMeters {
    pub(crate) fn new(now: f64) -> Self {
        Self {
            window_start_ts: now,
            entry_rejects: HashMap::new(),
            entry_reject_total: 0,
            ai_calls: 0,
            ai_action_buy: 0,
            ai_action_sell: 0,
            ai_action_hold: 0,
            stale_gate_trips: 0,
            confidence_gate_trips: 0,
            model_health_gate_trips: 0,
            data_quality_gate_trips: 0,
            watchdog_gate_trips: 0,
            hardware_guard_trips: 0,
            ai_throttle_hits: 0,
            nemo_exit_checks: 0,
            nemo_exit_holds: 0,
            nemo_exit_sells: 0,
        }
    }

    pub(crate) fn roll_window(&mut self, now: f64) {
        if (now - self.window_start_ts) < METERS_WINDOW_SEC {
            return;
        }
        self.window_start_ts = now;
        self.entry_rejects.clear();
        self.entry_reject_total = 0;
        self.ai_calls = 0;
        self.ai_action_buy = 0;
        self.ai_action_sell = 0;
        self.ai_action_hold = 0;
        self.stale_gate_trips = 0;
        self.confidence_gate_trips = 0;
        self.model_health_gate_trips = 0;
        self.data_quality_gate_trips = 0;
        self.watchdog_gate_trips = 0;
        self.hardware_guard_trips = 0;
        self.ai_throttle_hits = 0;
        self.nemo_exit_checks = 0;
        self.nemo_exit_holds = 0;
        self.nemo_exit_sells = 0;
    }

    pub(crate) fn bump_reject(&mut self, now: f64, reason: &str) {
        self.roll_window(now);
        self.entry_reject_total += 1;
        *self.entry_rejects.entry(reason.to_string()).or_insert(0) += 1;
    }

    pub(crate) fn bump_nemo_exit(&mut self, now: f64, action: &str) {
        self.roll_window(now);
        self.nemo_exit_checks += 1;
        match action {
            "SELL" => self.nemo_exit_sells += 1,
            _ => self.nemo_exit_holds += 1,
        }
    }

    pub(crate) fn bump_ai_action(&mut self, now: f64, action: &str) {
        self.roll_window(now);
        self.ai_calls += 1;
        match action {
            "BUY" => self.ai_action_buy += 1,
            "SELL" => self.ai_action_sell += 1,
            _ => self.ai_action_hold += 1,
        }
    }

    pub(crate) fn top_rejects(&self, n: usize) -> Vec<(String, u64)> {
        let mut rows: Vec<(String, u64)> = self
            .entry_rejects
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        rows.sort_by(|a, b| b.1.cmp(&a.1));
        rows.truncate(n);
        rows
    }
}

/// Format coin data as compact text for Qwen.
// classify_lane, should_finalize_pending_entry, parse_query_order_snapshot moved to helpers.rs

impl TradingLoop {
    pub fn new(
        config: TradingConfig,
        strategy: std::sync::Arc<crate::strategy::StrategyToml>,
        bus: tokio::sync::broadcast::Sender<crate::event_bus::Event>,
        shared_funding: crate::market_brain::SharedFunding,
        npu_arc: Option<std::sync::Arc<crate::npu_bridge::NpuBridge>>,
    ) -> Self {
        let now = now_ts();
        let journal = Journal::new(&config.journal_dir);
        let optimizer = ProfitOptimizer::new(
            config.base_usd,
            100.0, // max_position_usd
            500.0, // max_portfolio_usd
        );
        let positions = load_positions(&config.positions_file);
        let circuit = CircuitBreakerState::new(0.0); // Will be set on first balance check
        let init_conf_floor = config.env.get_f64("HF_CONF_BASE", 0.60);
        let npu_scan_queue_cap = config.env.get_u64("NPU_SCAN_QUEUE_CAP", 256).max(1) as usize;
        let (npu_scan_tx, npu_scan_rx) =
            tokio::sync::mpsc::channel::<(String, String, String, String, String)>(npu_scan_queue_cap);

        Self {
            config,
            strategy,
            positions,
            circuit,
            journal,
            optimizer,
            last_entry_ts: 0.0,
            tick_count: 0,
            last_heartbeat: 0.0,
            last_watchdog_check: 0.0,
            last_config_reload: Instant::now(),
            prev_momentum: HashMap::new(),
            available_usd: 0.0,
            equity: 0.0,
            holdings: HashMap::new(),
            last_balance_check: 0.0,
            pending_entries: HashMap::new(),
            last_pending_check: 0.0,
            meters: RuntimeMeters::new(now),
            integrity_snapshot: collect_integrity_snapshot(),
            rest_throttle_hits: 0,
            rest_backoff_hits: 0,
            manual_kill_active: false,
            watchdog_ok: true,
            watchdog_fail_count: 0,
            watchdog_block_entries: false,
            watchdog_last_fail_paths: Vec::new(),
            hardware_block_entries: false,
            hw_guard_level: HardwareGuardLevel::Normal,
            hw_status: HardwareStatus::default(),
            hw_last_sample_ts: 0.0,
            ai_latency_p95_ms: 0.0,
            effective_ai_parallel: 4,
            cvar_weights: HashMap::new(),
            cvar_weights_ts: 0.0,
            cvar_cash_frac: 0.5,
            portfolio_value: 0.0,
            gpu_math_logged: false,
            strategy_weights: HashMap::new(),
            prev_strategy_weights: HashMap::new(),
            bus,
            telemetry_level: crate::event_bus::TelemetryLevel::from_env(),
            prev_coin_state: HashMap::new(),
            last_nemo_exit_check: HashMap::new(),
            btc_24h_open: 0.0,
            btc_green_since: None,
            auto_live_active: false,
            last_btc_ticker_ts: 0.0,
            last_ai_call_ts: HashMap::new(),
            last_trade_ts: HashMap::new(),
            last_entry_price: HashMap::new(),
            nemo_memory: HashMap::new(),
            #[cfg(feature = "nemo_optimizer")]
            nemo_optimizer: crate::nemo_optimizer::NemoOptimizer::new(),
            nemo_memory_brain: crate::nemo_memory::NemoMemory::new(),
            memory_packet: crate::memory_packet::CachedMemoryPacket::new(),
            memory_enabled: false,
            npu_bridge: npu_arc.clone(),
            npu_scan_cache: std::collections::HashMap::new(),
            npu_scan_inflight: HashSet::new(),
            npu_scan_tx,
            npu_scan_rx: Some(npu_scan_rx),
            gpu_math: crate::addons::gpu_math::new_shared(),
            gbdt: std::sync::Arc::new(tokio::sync::RwLock::new(
                crate::signals::gbdt::GbdtEngine::init("data"),
            )),
            cloud_intel: None,
            swap_queue: None,
            http_client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(15))
                .build()
                .unwrap_or_default(),
            news_sentiment: crate::news_sentiment::NewsSentiment::new(),
            market_brain: crate::market_brain::MarketBrain::new(),
            regime_machines: HashMap::new(),
            book_imb_hist: HashMap::new(),
            quality_stats: load_quality_stats(),
            entry_conf_floor_dyn: init_conf_floor,
            hf_streak_wins: 0,
            hf_streak_losses: 0,
            loss_cooldown_until: 0.0,
            hf_outcome_ema: 0.0,
            hf_equity_peak: 0.0,
            market_watcher: crate::market_watcher::MarketWatcher::new(),
            shared_funding,
            portfolio_allocator: crate::portfolio_allocator::PortfolioAllocator::new(),
        }
    }

    /// Record a decision in Nemo's working memory.
    fn record_nemo_memory(&mut self, sym: &str, action: &str, confidence: f64, reason: &str, price: f64, lane: &str, source: &str) {
        if !self.memory_enabled {
            return;
        }
        let entry = NemoMemoryEntry {
            ts: now_ts(),
            action: action.to_string(),
            confidence,
            reason: reason.to_string(),
            price_at_decision: price,
            _lane: lane.to_string(),
            source: source.to_string(),
        };
        let mem = self.nemo_memory.entry(sym.to_string()).or_insert_with(std::collections::VecDeque::new);
        mem.push_back(entry);
        while mem.len() > NEMO_MEMORY_MAX_PER_COIN {
            mem.pop_front();
        }
    }

    fn prune_symbol_state(&mut self, features_map: &HashMap<String, serde_json::Value>) {
        let mut keep: HashSet<String> = features_map.keys().cloned().collect();
        keep.extend(self.positions.keys().cloned());
        keep.extend(self.pending_entries.keys().cloned());
        keep.extend(
            self.holdings
                .iter()
                .filter(|(_, qty)| **qty > 0.0)
                .map(|(sym, _)| sym.clone()),
        );

        self.prev_momentum.retain(|sym, _| keep.contains(sym));
        self.prev_coin_state.retain(|sym, _| keep.contains(sym));
        self.last_nemo_exit_check.retain(|sym, _| keep.contains(sym));
        self.last_ai_call_ts.retain(|sym, _| keep.contains(sym));
        self.last_trade_ts.retain(|sym, _| keep.contains(sym));
        self.last_entry_price.retain(|sym, _| keep.contains(sym));
        self.regime_machines.retain(|sym, _| keep.contains(sym));
        self.book_imb_hist.retain(|sym, _| keep.contains(sym));
        self.npu_scan_cache.retain(|sym, _| keep.contains(sym));
        self.npu_scan_inflight.retain(|sym| keep.contains(sym));
    }

    /// Reload CVaR portfolio weights from data/cvar_weights.json.
    fn reload_cvar_weights(&mut self) {
        let path = Path::new("data/cvar_weights.json");
        if !path.exists() {
            return;
        }
        match fs::read_to_string(path) {
            Ok(text) => {
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(&text) {
                    let file_ts = val.get("ts").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    // Staleness warning (sidecar may be down)
                    let now = now_ts();
                    let age = now - file_ts;
                    if age > 300.0 && self.tick_count % 30 == 0 {
                        tracing::warn!(
                            "[CVaR] weights stale: {:.0}s old (sidecar may be down)",
                            age
                        );
                    }
                    if file_ts <= self.cvar_weights_ts {
                        return; // no update
                    }
                    if let Some(wmap) = val.get("weights").and_then(|v| v.as_object()) {
                        let mut new_weights = HashMap::new();
                        for (sym, wv) in wmap {
                            if let Some(w) = wv.as_f64() {
                                new_weights.insert(sym.clone(), w);
                            }
                        }
                        let n = new_weights.len();
                        self.cvar_weights = new_weights;
                        self.cvar_weights_ts = file_ts;
                        self.cvar_cash_frac =
                            val.get("cash").and_then(|v| v.as_f64()).unwrap_or(0.5);
                        if self.tick_count % 30 == 0 {
                            tracing::info!(
                                "[CVaR] loaded {} weights, cash={:.1}%",
                                n,
                                self.cvar_cash_frac * 100.0
                            );
                        }
                    }
                }
            }
            Err(e) => {
                if self.tick_count % 60 == 0 {
                    tracing::warn!("[CVaR] failed to read cvar_weights.json: {}", e);
                }
            }
        }
    }

    /// Check BTC 24h green/red and auto-toggle paper/live mode.
    async fn check_btc_auto_live(&mut self, features_map: &HashMap<String, serde_json::Value>) {
        if !self.config.auto_live_on_btc_green {
            return;
        }
        let now = now_ts();
        // Poll Kraken public Ticker every 60s for BTC 24h open
        if now - self.last_btc_ticker_ts >= 60.0 {
            self.last_btc_ticker_ts = now;
            if let Ok(resp) = reqwest::Client::new()
                .get("https://api.kraken.com/0/public/Ticker?pair=XBTUSD")
                .timeout(std::time::Duration::from_secs(5))
                .send()
                .await
            {
                if let Ok(body) = resp.json::<serde_json::Value>().await {
                    // Kraken returns: {"result":{"XXBTZUSD":{"o":"97000.0",...}}}
                    if let Some(open_str) = body
                        .get("result")
                        .and_then(|r| r.get("XXBTZUSD"))
                        .and_then(|t| t.get("o"))
                        .and_then(|o| o.as_str())
                    {
                        if let Ok(open) = open_str.parse::<f64>() {
                            let was_zero = self.btc_24h_open <= 0.0;
                            self.btc_24h_open = open;
                            if was_zero {
                                tracing::info!("[BTC-TICKER] 24h_open=${:.0} — tracking started", open);
                            }
                        }
                    }
                }
            }
        }

        if self.btc_24h_open <= 0.0 {
            return; // No ticker data yet
        }

        // Get current BTC price from features
        let btc_price = features_map
            .get("BTC")
            .and_then(|f| f.get("price"))
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        if btc_price <= 0.0 {
            return;
        }

        let btc_green = btc_price > self.btc_24h_open;
        let pct = ((btc_price - self.btc_24h_open) / self.btc_24h_open) * 100.0;

        // Periodic status log (every ~25 min)
        if self.tick_count % 30 == 0 {
            let mode = if self.config.paper_trading { "PAPER" } else { "LIVE" };
            let color = if btc_green { "GREEN" } else { "RED" };
            tracing::info!(
                "[BTC-TICKER] ${:.0} vs 24h_open ${:.0} ({:+.2}%) = {} | mode={}",
                btc_price, self.btc_24h_open, pct, color, mode
            );
        }

        if btc_green {
            // Start or continue the green timer
            if self.btc_green_since.is_none() {
                self.btc_green_since = Some(Instant::now());
                tracing::info!(
                    "[BTC-TICKER] BTC GREEN ${:.0} > 24h_open ${:.0} ({:+.2}%) — cooldown started",
                    btc_price, self.btc_24h_open, pct
                );
            }
            // Check if green long enough to engage live
            if let Some(since) = self.btc_green_since {
                let green_min = since.elapsed().as_secs_f64() / 60.0;
                if green_min >= self.config.auto_live_cooldown_min
                    && self.config.paper_trading
                    && !self.auto_live_active
                {
                    self.config.paper_trading = false;
                    self.auto_live_active = true;
                    tracing::warn!(
                        "[AUTO-LIVE] BTC green {:.0}min ${:.0} > 24h_open ${:.0} ({:+.2}%) — LIVE MODE ENGAGED",
                        green_min, btc_price, self.btc_24h_open, pct
                    );
                }
            }
        } else {
            // BTC turned red
            if self.btc_green_since.is_some() {
                self.btc_green_since = None;
                tracing::info!(
                    "[BTC-TICKER] BTC RED ${:.0} <= 24h_open ${:.0} ({:+.2}%) — cooldown reset",
                    btc_price, self.btc_24h_open, pct
                );
            }
            if self.auto_live_active {
                self.config.paper_trading = true;
                self.auto_live_active = false;
                tracing::warn!(
                    "[AUTO-PAPER] BTC turned red ${:.0} <= 24h_open ${:.0} ({:+.2}%) — back to PAPER",
                    btc_price, self.btc_24h_open, pct
                );
            }
        }
    }

    fn state_checksum(&self) -> String {
        let mut keys: Vec<String> = self
            .positions
            .iter()
            .map(|(sym, p)| format!("{sym}:{:.8}:{:.8}", p.remaining_qty, p.entry_price))
            .collect();
        keys.extend(
            self.pending_entries
                .iter()
                .map(|(txid, p)| format!("pending:{txid}:{}:{:.8}", p.symbol, p.requested_qty)),
        );
        keys.sort();
        let mut hasher = Sha256::new();
        for row in keys {
            hasher.update(row.as_bytes());
            hasher.update(b"\n");
        }
        format!("{:x}", hasher.finalize())
    }

    /// Reduce tool rounds based on hardware guard level (GPU temp/mem pressure).
    pub(crate) fn effective_tool_rounds(&self, base: u8) -> u8 {
        match self.hw_guard_level {
            HardwareGuardLevel::Normal => base,
            HardwareGuardLevel::SoftDegrade => base.saturating_sub(1).max(1),
            HardwareGuardLevel::HardBlock | HardwareGuardLevel::Emergency => 1,
        }
    }

    fn effective_ai_parallel_for(&self, base_parallel: usize) -> usize {
        match self.hw_guard_level {
            HardwareGuardLevel::Normal => base_parallel,
            HardwareGuardLevel::SoftDegrade => base_parallel.saturating_sub(1).max(1),
            HardwareGuardLevel::HardBlock | HardwareGuardLevel::Emergency => 1,
        }
    }

    fn update_hardware_guard(&mut self, now: f64) {
        if !self.config.env.get_bool("HW_GUARD_ENABLED", false) {
            self.hardware_block_entries = false;
            self.hw_guard_level = HardwareGuardLevel::Normal;
            return;
        }

        let sample_every_sec = self.config.env.get_f64("HW_SAMPLE_EVERY_SEC", 15.0).max(2.0);
        if (now - self.hw_last_sample_ts) >= sample_every_sec || self.hw_last_sample_ts <= 0.0 {
            self.hw_status = collect_hardware_status(now);
            self.hw_last_sample_ts = now;
        }

        let mut level = HardwareGuardLevel::Normal;
        let mut maybe_raise = |next: HardwareGuardLevel| {
            if next.severity() > level.severity() {
                level = next;
            }
        };

        let gpu_mem_degrade_mb = self.config.env.get_f64("GPU_MEM_DEGRADE_MB", 13_500.0);
        let gpu_mem_block_mb = self.config.env.get_f64("GPU_MEM_BLOCK_MB", 14_500.0);
        let gpu_mem_emergency_mb = self.config.env.get_f64("GPU_MEM_EMERGENCY_MB", 15_200.0);
        let gpu_temp_degrade_c = self.config.env.get_f64("GPU_TEMP_DEGRADE_C", 83.0);
        let gpu_temp_block_c = self.config.env.get_f64("GPU_TEMP_BLOCK_C", 88.0);
        let ram_free_degrade_mb = self.config.env.get_f64("RAM_FREE_DEGRADE_MB", 4_000.0);
        let ram_free_block_mb = self.config.env.get_f64("RAM_FREE_BLOCK_MB", 2_500.0);
        let ai_p95_degrade_ms = self.config.env.get_f64("AI_P95_DEGRADE_MS", 2_000.0);
        let ai_p95_block_ms = self.config.env.get_f64("AI_P95_BLOCK_MS", 4_000.0);

        if let Some(used) = self.hw_status.gpu_mem_used_mb {
            if used >= gpu_mem_emergency_mb {
                maybe_raise(HardwareGuardLevel::Emergency);
            } else if used >= gpu_mem_block_mb {
                maybe_raise(HardwareGuardLevel::HardBlock);
            } else if used >= gpu_mem_degrade_mb {
                maybe_raise(HardwareGuardLevel::SoftDegrade);
            }
        }

        if let Some(temp) = self.hw_status.gpu_temp_c {
            if temp >= (gpu_temp_block_c + 3.0) {
                maybe_raise(HardwareGuardLevel::Emergency);
            } else if temp >= gpu_temp_block_c {
                maybe_raise(HardwareGuardLevel::HardBlock);
            } else if temp >= gpu_temp_degrade_c {
                maybe_raise(HardwareGuardLevel::SoftDegrade);
            }
        }

        if let Some(free_mb) = self.hw_status.ram_free_mb {
            if free_mb <= ram_free_block_mb {
                maybe_raise(HardwareGuardLevel::HardBlock);
            } else if free_mb <= ram_free_degrade_mb {
                maybe_raise(HardwareGuardLevel::SoftDegrade);
            }
        }

        if self.ai_latency_p95_ms >= ai_p95_block_ms {
            maybe_raise(HardwareGuardLevel::HardBlock);
        } else if self.ai_latency_p95_ms >= ai_p95_degrade_ms {
            maybe_raise(HardwareGuardLevel::SoftDegrade);
        }

        if level != self.hw_guard_level {
            tracing::warn!(
                "[HW-GUARD] level {} -> {} gpu_mem={:?}MB temp={:?}C ram_free={:?}MB ai_p95_ms={:.0}",
                self.hw_guard_level.as_str(),
                level.as_str(),
                self.hw_status.gpu_mem_used_mb.map(|v| v.round()),
                self.hw_status.gpu_temp_c.map(|v| v.round()),
                self.hw_status.ram_free_mb.map(|v| v.round()),
                self.ai_latency_p95_ms
            );
        }

        self.hw_guard_level = level;
        self.hardware_block_entries = matches!(
            self.hw_guard_level,
            HardwareGuardLevel::HardBlock | HardwareGuardLevel::Emergency
        );
    }

    pub fn build_heartbeat_payload(&self, now: f64) -> serde_json::Value {
        let top_rejects: Vec<serde_json::Value> = self
            .meters
            .top_rejects(5)
            .into_iter()
            .map(|(reason, count)| json!({"reason": reason, "count": count}))
            .collect();
        json!({
            "ts": now,
            "tick": self.tick_count,
            "window_sec": METERS_WINDOW_SEC,
            "window_start_ts": self.meters.window_start_ts,
            "balances": {
                "available_usd": self.available_usd,
                "portfolio_value": self.portfolio_value,
                "holdings_count": self.holdings.len(),
                "holdings": self.holdings.iter()
                    .filter(|(_, qty)| **qty > 0.001)
                    .collect::<HashMap<_, _>>()
            },
            "positions": {
                "open": self.positions.len(),
                "pending": self.pending_entries.len()
            },
            "circuit": {
                "daily_pnl_usd": self.circuit.daily_pnl,
                "unrealized_pnl_usd": self.circuit.unrealized_pnl,
                "start_balance_usd": self.circuit.start_balance_usd,
                "force_defensive": self.circuit.force_defensive,
                "stop_trading": self.circuit.stop_trading
            },
            "meters_60s": {
                "entry_reject_total": self.meters.entry_reject_total,
                "top_reject_reasons": top_rejects,
                "ai_calls": self.meters.ai_calls,
                "ai_action_buy": self.meters.ai_action_buy,
                "ai_action_sell": self.meters.ai_action_sell,
                "ai_action_hold": self.meters.ai_action_hold,
                "stale_gate_trips": self.meters.stale_gate_trips,
                "confidence_gate_trips": self.meters.confidence_gate_trips,
                "model_health_gate_trips": self.meters.model_health_gate_trips,
                "data_quality_gate_trips": self.meters.data_quality_gate_trips,
                "watchdog_gate_trips": self.meters.watchdog_gate_trips,
                "hardware_guard_trips": self.meters.hardware_guard_trips,
                "ai_throttle_hits": self.meters.ai_throttle_hits,
                "ai_exit_checks": self.meters.nemo_exit_checks,
                "ai_exit_holds": self.meters.nemo_exit_holds,
                "ai_exit_sells": self.meters.nemo_exit_sells
            },
            "rest": {
                "throttle_hits_total": self.rest_throttle_hits,
                "backoff_hits_total": self.rest_backoff_hits
            },
            "watchdog": {
                "ok": self.watchdog_ok,
                "fail_count": self.watchdog_fail_count,
                "block_entries": self.watchdog_block_entries,
                "kill_on_fail": self.config.env.get_bool("WATCHDOG_KILL_ON_FAIL", false),
                "failing_paths": self.watchdog_last_fail_paths
            },
            "hardware_guard": {
                "enabled": self.config.env.get_bool("HW_GUARD_ENABLED", false),
                "level": self.hw_guard_level.as_str(),
                "block_entries": self.hardware_block_entries,
                "effective_ai_parallel": self.effective_ai_parallel,
                "ai_latency_p95_ms": self.ai_latency_p95_ms,
                "sample": {
                    "ts": self.hw_status.sampled_ts,
                    "gpu_mem_used_mb": self.hw_status.gpu_mem_used_mb,
                    "gpu_mem_total_mb": self.hw_status.gpu_mem_total_mb,
                    "gpu_temp_c": self.hw_status.gpu_temp_c,
                    "ram_free_mb": self.hw_status.ram_free_mb,
                    "ram_total_mb": self.hw_status.ram_total_mb,
                    "sample_error": self.hw_status.sample_error
                }
            },
            "manual_kill_switch": {
                "active": self.manual_kill_active
            },
            "cvar_optimizer": {
                "num_weights": self.cvar_weights.len(),
                "weights_ts": self.cvar_weights_ts,
                "weights_age_sec": if self.cvar_weights_ts > 0.0 { now - self.cvar_weights_ts } else { -1.0 }
            },
            "state_checksum_sha256": self.state_checksum(),
            "integrity": self.integrity_snapshot
        })
    }

    fn current_unrealized_pnl(&self, features_map: &HashMap<String, serde_json::Value>) -> f64 {
        self.positions
            .values()
            .map(|pos| {
                let price = features_map
                    .get(&pos.symbol)
                    .and_then(|f| f.get("price"))
                    .and_then(|v| v.as_f64())
                    .unwrap_or(pos.entry_price);
                pos.pnl_usd(price, self.config.fee_per_side_pct)
            })
            .sum()
    }

    /// Called when a private WS order fill arrives.
    /// Updates the position's entry price to the actual fill price when it differs.
    pub fn on_order_fill(&mut self, symbol: &str, fill_price: f64, fill_qty: f64, status: &str) {
        if fill_price <= 0.0 || fill_qty <= 0.0 {
            return;
        }
        if let Some(pos) = self.positions.get_mut(symbol) {
            let prev = pos.entry_price;
            // Only update on entry fills (status=filled or partially_filled, price differs)
            if (fill_price - prev).abs() / prev.max(1e-9) > 0.0001 {
                pos.entry_price = fill_price;
                tracing::info!(
                    "[ORDER-FILL] {} entry_price updated {:.4} → {:.4} (fill qty={:.6} status={})",
                    symbol, prev, fill_price, fill_qty, status
                );
            }
        }
    }
}

// Quality stats, utilities, watchdog, hardware probes, file I/O moved to helpers.rs

#[cfg(test)]
mod tests {
    use super::{load_positions, parse_query_order_snapshot, should_finalize_pending_entry};
    use serde_json::json;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_path(file: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        std::env::temp_dir().join(format!("hk_test_{file}_{nonce}.json"))
    }

    #[test]
    fn load_positions_invalid_json_returns_empty() {
        let path = temp_path("positions_bad");
        fs::write(&path, "{invalid_json").expect("write bad json");
        let loaded = load_positions(path.to_str().expect("path str"));
        assert!(loaded.is_empty());
        let _ = fs::remove_file(path);
    }

    #[test]
    fn pending_entry_finalize_only_on_terminal_status() {
        assert!(should_finalize_pending_entry("closed", 0.25));
        assert!(should_finalize_pending_entry("canceled", 0.01));
        assert!(should_finalize_pending_entry("expired", 0.10));
        assert!(!should_finalize_pending_entry("open", 0.10));
        assert!(!should_finalize_pending_entry("closed", 0.0));
    }

    #[test]
    fn parse_query_order_snapshot_open_unfilled() {
        let txid = "TX123";
        let payload = json!({
            "result": {
                txid: {
                    "status": "open",
                    "vol_exec": "0.00000000",
                    "cost": "0.00000000"
                }
            }
        });
        let s = parse_query_order_snapshot(&payload, txid).expect("snapshot");
        assert_eq!(s.status, "open");
        assert_eq!(s.vol_exec, 0.0);
        assert_eq!(s.cost, 0.0);
    }

    #[test]
    fn parse_query_order_snapshot_closed_partial_fill() {
        let txid = "TX456";
        let payload = json!({
            "result": {
                txid: {
                    "status": "closed",
                    "vol_exec": "0.12500000",
                    "cost": "12.50000000"
                }
            }
        });
        let s = parse_query_order_snapshot(&payload, txid).expect("snapshot");
        assert_eq!(s.status, "closed");
        assert!((s.vol_exec - 0.125).abs() < 1e-12);
        assert!((s.cost - 12.5).abs() < 1e-12);
        assert!(should_finalize_pending_entry(&s.status, s.vol_exec));
    }

    #[test]
    fn parse_query_order_snapshot_cancelled_unfilled() {
        let txid = "TX789";
        let payload = json!({
            "result": {
                txid: {
                    "status": "canceled",
                    "vol_exec": "0.00000000",
                    "cost": "0.00000000"
                }
            }
        });
        let s = parse_query_order_snapshot(&payload, txid).expect("snapshot");
        assert_eq!(s.status, "canceled");
        assert_eq!(s.vol_exec, 0.0);
        assert!(!should_finalize_pending_entry(&s.status, s.vol_exec));
    }

    #[test]
    fn parse_query_order_snapshot_missing_txid_returns_none() {
        let payload = json!({ "result": {} });
        assert!(parse_query_order_snapshot(&payload, "MISSING").is_none());
    }
}
