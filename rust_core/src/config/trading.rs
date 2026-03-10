//! Trading configuration — hot-reload, circuit breaker, regime inference, profile selection.
//!
//! Ported from tinyllm_trader.py config management.
//! All thresholds hot-reloadable from .env without restart.

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

// ── Cached Env Snapshot (avoids syscalls in hot path) ────────────

#[derive(Clone, Debug, Default)]
pub struct CachedEnv {
    pub map: HashMap<String, String>,
}

impl CachedEnv {
    /// Snapshot all current environment variables into an in-memory HashMap.
    pub fn snapshot() -> Self {
        Self {
            map: std::env::vars().collect(),
        }
    }

    #[inline]
    pub fn get_bool(&self, key: &str, default: bool) -> bool {
        self.map
            .get(key)
            .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON"))
            .unwrap_or(default)
    }

    #[inline]
    pub fn get_f64(&self, key: &str, default: f64) -> f64 {
        self.map
            .get(key)
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    }

    #[inline]
    pub fn get_u64(&self, key: &str, default: u64) -> u64 {
        self.map
            .get(key)
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    }

    #[inline]
    pub fn get_str(&self, key: &str, default: &str) -> String {
        self.map
            .get(key)
            .cloned()
            .unwrap_or_else(|| default.to_string())
    }

    /// Generic parse: works for usize, u8, i32, f32, etc.
    #[inline]
    pub fn get_parsed<T: std::str::FromStr>(&self, key: &str, default: T) -> T {
        self.map
            .get(key)
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    }
}

// â"€â"€ Trading Config â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

#[derive(Clone, Debug)]
pub struct TradingConfig {
    // Core thresholds (hot-reloadable)
    pub entry_threshold: f64,
    pub exit_threshold: f64,
    pub take_profit_pct: f64,
    pub stop_loss_pct: f64,
    pub base_usd: f64,
    pub poll_seconds: f64,
    pub market_state_min: f64,

    // Profile: aggressive
    pub aggr_max_positions: usize,
    pub aggr_cooldown_sec: u64,
    pub aggr_max_spread_pct: f64,
    pub aggr_max_slip_pct: f64,
    pub aggr_min_liq: f64,
    pub aggr_min_vol_ratio: f64,
    pub aggr_circuit_breaker_pct: f64,

    // Profile: defensive
    pub def_max_positions: usize,
    pub def_cooldown_sec: u64,
    pub def_max_spread_pct: f64,
    pub def_max_slip_pct: f64,
    pub def_min_liq: f64,
    pub def_min_vol_ratio: f64,
    pub def_circuit_breaker_pct: f64,

    // Breakeven & trailing stop
    pub breakeven_activate_pct: f64,
    pub trailing_stop_activate_pct: f64,
    pub trailing_stop_pct: f64,
    pub min_hold_minutes: f64,

    // AI flags
    pub max_decision_age_sec: f64, // skip trade if AI inference took longer than this
    pub use_balance_for_risk: bool,
    pub max_daily_loss_usd: f64,
    pub crash_qregime: f64,
    pub crash_qvol: f64,
    pub adopt_kraken_holdings: bool,
    pub enable_score_drop: bool,

    // Nemo exit monitor
    pub use_nemo_exit: bool,
    pub nemo_exit_interval_sec: f64,
    pub nemo_exit_min_hold_sec: f64,
    pub nemo_exit_min_conf: f64,

    // Cooldowns & confidence cap (single-brain mode)
    pub ai_cooldown_sec: f64,
    pub trade_cooldown_sec: f64,
    pub max_confidence_for_sizing: f64,

    // Kraken fee per side as pct (0.26 = 0.26%). Round-trip = 2x this.
    pub fee_per_side_pct: f64,

    // Paper trading
    pub paper_trading: bool,

    // Auto-live: switch from paper to live when BTC is green
    pub auto_live_on_btc_green: bool,
    pub auto_live_cooldown_min: f64,

    // Paths
    pub snapshot_path: String,
    pub snapshot_shm_path: String,
    pub positions_file: String,
    pub heartbeat_file: String,
    pub journal_dir: String,

    // Cached env snapshot — refreshed every hot_reload(), zero syscalls in hot path
    pub env: CachedEnv,
    pub last_env_mtime: Option<SystemTime>,
}

impl Default for TradingConfig {
    fn default() -> Self {
        Self {
            entry_threshold: 0.50,
            exit_threshold: 0.02,
            take_profit_pct: 2.0,
            stop_loss_pct: 1.0,
            base_usd: 20.0,
            poll_seconds: 2.0,
            market_state_min: 0.35,

            aggr_max_positions: 3,
            aggr_cooldown_sec: 300,
            aggr_max_spread_pct: 0.5,
            aggr_max_slip_pct: 2.0,
            aggr_min_liq: 50_000.0,
            aggr_min_vol_ratio: 1.0,
            aggr_circuit_breaker_pct: 5.0,

            def_max_positions: 1,
            def_cooldown_sec: 600,
            def_max_spread_pct: 0.3,
            def_max_slip_pct: 1.0,
            def_min_liq: 100_000.0,
            def_min_vol_ratio: 1.5,
            def_circuit_breaker_pct: 3.0,

            breakeven_activate_pct: 0.5,
            trailing_stop_activate_pct: 1.5,
            trailing_stop_pct: 0.75,
            min_hold_minutes: 5.0,

            fee_per_side_pct: 0.26, // Kraken taker fee per side (0.26%)

            max_decision_age_sec: 5.0,
            use_balance_for_risk: true,
            max_daily_loss_usd: 0.0,
            crash_qregime: -0.005,
            crash_qvol: 0.02,
            adopt_kraken_holdings: false,
            enable_score_drop: false,

            use_nemo_exit: true,
            nemo_exit_interval_sec: 30.0,
            nemo_exit_min_hold_sec: 60.0,
            nemo_exit_min_conf: 0.6,

            ai_cooldown_sec: 5.0,
            trade_cooldown_sec: 60.0,
            max_confidence_for_sizing: 0.85,

            paper_trading: false,
            auto_live_on_btc_green: false,
            auto_live_cooldown_min: 15.0,

            snapshot_path: "data/features_snapshot.json".into(),
            snapshot_shm_path: "data/features_snapshot.shm".into(),
            positions_file: "data/positions.json".into(),
            heartbeat_file: "data/heartbeat_trader.json".into(),
            journal_dir: "npu_journal".into(),

            env: CachedEnv::default(),
            last_env_mtime: None,
        }
    }
}

impl TradingConfig {
    /// Load config from environment variables, using defaults for missing values.
    pub fn from_env() -> Self {
        let mut c = Self::default();
        let e = |key: &str| std::env::var(key).ok();
        let ef =
            |key: &str, def: f64| -> f64 { e(key).and_then(|v| v.parse().ok()).unwrap_or(def) };
        let eu =
            |key: &str, def: u64| -> u64 { e(key).and_then(|v| v.parse().ok()).unwrap_or(def) };
        let ez =
            |key: &str, def: usize| -> usize { e(key).and_then(|v| v.parse().ok()).unwrap_or(def) };
        let eb = |key: &str, def: bool| -> bool {
            e(key)
                .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON"))
                .unwrap_or(def)
        };

        c.entry_threshold = ef("ENTRY_THRESHOLD", c.entry_threshold);
        c.exit_threshold = ef("EXIT_THRESHOLD", c.exit_threshold);
        c.take_profit_pct = ef("TAKE_PROFIT_PCT", c.take_profit_pct);
        c.stop_loss_pct = ef("STOP_LOSS_PCT", c.stop_loss_pct);
        c.base_usd = ef("BASE_USD", c.base_usd);
        c.poll_seconds = ef("POLL_SECONDS", c.poll_seconds);
        c.market_state_min = ef("MARKET_STATE_MIN", c.market_state_min);

        c.aggr_max_positions = ez("AGGR_MAX_POSITIONS", c.aggr_max_positions);
        c.aggr_cooldown_sec = eu("AGGR_COOLDOWN_SEC", c.aggr_cooldown_sec);
        c.aggr_max_spread_pct = ef("AGGR_MAX_SPREAD_PCT", c.aggr_max_spread_pct);
        c.aggr_max_slip_pct = ef("AGGR_MAX_SLIP_PCT", c.aggr_max_slip_pct);
        c.aggr_min_liq = ef("AGGR_MIN_LIQ", c.aggr_min_liq);
        c.aggr_min_vol_ratio = ef("AGGR_MIN_VOL_RATIO", c.aggr_min_vol_ratio);
        c.aggr_circuit_breaker_pct = e("AGGR_MAX_DAILY_LOSS_PCT")
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| ef("AGGR_CIRCUIT_BREAKER_PCT", c.aggr_circuit_breaker_pct));

        c.def_max_positions = ez("DEF_MAX_POSITIONS", c.def_max_positions);
        c.def_cooldown_sec = eu("DEF_COOLDOWN_SEC", c.def_cooldown_sec);
        c.def_max_spread_pct = ef("DEF_MAX_SPREAD_PCT", c.def_max_spread_pct);
        c.def_max_slip_pct = ef("DEF_MAX_SLIP_PCT", c.def_max_slip_pct);
        c.def_min_liq = ef("DEF_MIN_LIQ", c.def_min_liq);
        c.def_min_vol_ratio = ef("DEF_MIN_VOL_RATIO", c.def_min_vol_ratio);
        c.def_circuit_breaker_pct = e("DEF_MAX_DAILY_LOSS_PCT")
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| ef("DEF_CIRCUIT_BREAKER_PCT", c.def_circuit_breaker_pct));

        c.breakeven_activate_pct = ef("BREAKEVEN_ACTIVATE_PCT", c.breakeven_activate_pct);
        c.trailing_stop_activate_pct =
            ef("TRAILING_STOP_ACTIVATE_PCT", c.trailing_stop_activate_pct);
        c.trailing_stop_pct = ef("TRAILING_STOP_PCT", c.trailing_stop_pct);
        c.min_hold_minutes = ef("MIN_HOLD_MINUTES", c.min_hold_minutes);

        c.fee_per_side_pct = ef("FEE_PER_SIDE_PCT", c.fee_per_side_pct);

        c.max_decision_age_sec = ef("MAX_DECISION_AGE_SEC", c.max_decision_age_sec);
        c.use_balance_for_risk = eb("USE_BALANCE_FOR_RISK", c.use_balance_for_risk);
        c.max_daily_loss_usd = ef("MAX_DAILY_LOSS_USD", c.max_daily_loss_usd);
        c.crash_qregime = ef("REGIME_CRASH_QREGIME", c.crash_qregime);
        c.crash_qvol = ef("REGIME_CRASH_QVOL", c.crash_qvol);
        c.adopt_kraken_holdings = eb("ADOPT_KRAKEN_HOLDINGS", c.adopt_kraken_holdings);
        c.enable_score_drop = eb("ENABLE_SCORE_DROP", c.enable_score_drop);

        c.use_nemo_exit = eb("USE_AI_EXIT", c.use_nemo_exit);
        c.nemo_exit_interval_sec = ef("AI_EXIT_INTERVAL_SEC", c.nemo_exit_interval_sec);
        c.nemo_exit_min_hold_sec = ef("AI_EXIT_MIN_HOLD_SEC", c.nemo_exit_min_hold_sec);
        c.nemo_exit_min_conf = ef("AI_EXIT_MIN_CONF", c.nemo_exit_min_conf);

        c.ai_cooldown_sec = ef("AI_COOLDOWN_SEC", c.ai_cooldown_sec);
        c.trade_cooldown_sec = ef("TRADE_COOLDOWN_SEC", c.trade_cooldown_sec);
        c.max_confidence_for_sizing = ef("MAX_CONFIDENCE_FOR_SIZING", c.max_confidence_for_sizing);

        c.paper_trading = eb("PAPER_TRADING", c.paper_trading);
        c.auto_live_on_btc_green = eb("AUTO_LIVE_ON_BTC_GREEN", c.auto_live_on_btc_green);
        c.auto_live_cooldown_min = ef("AUTO_LIVE_COOLDOWN_MIN", c.auto_live_cooldown_min);

        if let Some(v) = e("SNAPSHOT_PATH") {
            c.snapshot_path = v;
        }
        if let Some(v) = e("SNAPSHOT_SHM_PATH") {
            c.snapshot_shm_path = v;
        }
        if let Some(v) = e("POSITIONS_FILE") {
            c.positions_file = v;
        }
        if let Some(v) = e("HEARTBEAT_FILE") {
            c.heartbeat_file = v;
        }
        if let Some(v) = e("JOURNAL_DIR") {
            c.journal_dir = v;
        }

        // Snapshot env for zero-syscall reads in hot path
        c.env = CachedEnv::snapshot();
        c.last_env_mtime = env_file_mtime();

        c
    }

    /// Hot-reload changed values from .env without restart.
    /// Only re-parses .env if the file's mtime has changed (avoids unnecessary I/O).
    pub fn hot_reload(&mut self) {
        // Only re-read .env if file actually changed
        let current_mtime = env_file_mtime();
        let file_changed = match (current_mtime, self.last_env_mtime) {
            (Some(cur), Some(prev)) => cur != prev,
            (Some(_), None) => true,
            _ => false,
        };

        if file_changed {
            let _ = dotenvy::dotenv_override();
            self.last_env_mtime = current_mtime;
        } else {
            // No .env change — just refresh the cache from process env
            // (in case something set env vars programmatically)
        }

        // Refresh cached env snapshot (cheap: ~100µs for ~50 vars)
        self.env = CachedEnv::snapshot();

        let ef = |key: &str, def: f64| -> f64 {
            self.env.get_f64(key, def)
        };
        let ez = |key: &str, def: usize| -> usize {
            self.env.map.get(key)
                .and_then(|v| v.parse().ok())
                .unwrap_or(def)
        };
        let eu = |key: &str, def: u64| -> u64 {
            self.env.get_u64(key, def)
        };

        self.entry_threshold = ef("ENTRY_THRESHOLD", self.entry_threshold);
        self.exit_threshold = ef("EXIT_THRESHOLD", self.exit_threshold);
        self.take_profit_pct = ef("TAKE_PROFIT_PCT", self.take_profit_pct);
        self.stop_loss_pct = ef("STOP_LOSS_PCT", self.stop_loss_pct);
        self.base_usd = ef("BASE_USD", self.base_usd);
        self.poll_seconds = ef("POLL_SECONDS", self.poll_seconds);
        self.market_state_min = ef("MARKET_STATE_MIN", self.market_state_min);
        self.aggr_max_positions = ez("AGGR_MAX_POSITIONS", self.aggr_max_positions);
        self.def_max_positions = ez("DEF_MAX_POSITIONS", self.def_max_positions);
        self.aggr_cooldown_sec = eu("AGGR_COOLDOWN_SEC", self.aggr_cooldown_sec);
        self.def_cooldown_sec = eu("DEF_COOLDOWN_SEC", self.def_cooldown_sec);
        self.breakeven_activate_pct = ef("BREAKEVEN_ACTIVATE_PCT", self.breakeven_activate_pct);
        self.trailing_stop_activate_pct = ef(
            "TRAILING_STOP_ACTIVATE_PCT",
            self.trailing_stop_activate_pct,
        );
        self.trailing_stop_pct = ef("TRAILING_STOP_PCT", self.trailing_stop_pct);
        self.min_hold_minutes = ef("MIN_HOLD_MINUTES", self.min_hold_minutes);
        self.max_decision_age_sec = ef("MAX_DECISION_AGE_SEC", self.max_decision_age_sec);
        self.aggr_min_vol_ratio = ef("AGGR_MIN_VOL_RATIO", self.aggr_min_vol_ratio);
        self.def_min_vol_ratio = ef("DEF_MIN_VOL_RATIO", self.def_min_vol_ratio);
        self.aggr_max_spread_pct = ef("AGGR_MAX_SPREAD_PCT", self.aggr_max_spread_pct);
        self.aggr_max_slip_pct = ef("AGGR_MAX_SLIP_PCT", self.aggr_max_slip_pct);
        self.aggr_min_liq = ef("AGGR_MIN_LIQ", self.aggr_min_liq);
        self.def_max_spread_pct = ef("DEF_MAX_SPREAD_PCT", self.def_max_spread_pct);
        self.def_max_slip_pct = ef("DEF_MAX_SLIP_PCT", self.def_max_slip_pct);
        self.def_min_liq = ef("DEF_MIN_LIQ", self.def_min_liq);
        self.aggr_circuit_breaker_pct = std::env::var("AGGR_MAX_DAILY_LOSS_PCT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| ef("AGGR_CIRCUIT_BREAKER_PCT", self.aggr_circuit_breaker_pct));
        self.def_circuit_breaker_pct = std::env::var("DEF_MAX_DAILY_LOSS_PCT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| ef("DEF_CIRCUIT_BREAKER_PCT", self.def_circuit_breaker_pct));
        self.use_balance_for_risk = self.env.get_bool("USE_BALANCE_FOR_RISK", self.use_balance_for_risk);
        self.max_daily_loss_usd = ef("MAX_DAILY_LOSS_USD", self.max_daily_loss_usd);
        self.crash_qregime = ef("REGIME_CRASH_QREGIME", self.crash_qregime);
        self.crash_qvol = ef("REGIME_CRASH_QVOL", self.crash_qvol);
        self.adopt_kraken_holdings = self.env.get_bool("ADOPT_KRAKEN_HOLDINGS", self.adopt_kraken_holdings);
        self.enable_score_drop = self.env.get_bool("ENABLE_SCORE_DROP", self.enable_score_drop);
        self.use_nemo_exit = self.env.get_bool("USE_AI_EXIT", self.use_nemo_exit);
        self.nemo_exit_interval_sec = ef("AI_EXIT_INTERVAL_SEC", self.nemo_exit_interval_sec);
        self.nemo_exit_min_hold_sec = ef("AI_EXIT_MIN_HOLD_SEC", self.nemo_exit_min_hold_sec);
        self.nemo_exit_min_conf = ef("AI_EXIT_MIN_CONF", self.nemo_exit_min_conf);
        self.ai_cooldown_sec = ef("AI_COOLDOWN_SEC", self.ai_cooldown_sec);
        self.trade_cooldown_sec = ef("TRADE_COOLDOWN_SEC", self.trade_cooldown_sec);
        self.max_confidence_for_sizing = ef("MAX_CONFIDENCE_FOR_SIZING", self.max_confidence_for_sizing);

        // Auto-live (hot-reloadable)
        self.paper_trading = self.env.get_bool("PAPER_TRADING", self.paper_trading);
        self.auto_live_on_btc_green = self.env.get_bool("AUTO_LIVE_ON_BTC_GREEN", self.auto_live_on_btc_green);
        self.auto_live_cooldown_min = ef("AUTO_LIVE_COOLDOWN_MIN", self.auto_live_cooldown_min);
    }
}

// â"€â"€ Coin Tiers â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

/// Per-coin optimal hold time (minutes) from 1000-trade backtest.
/// Used by exit logic — coins held less than 50% of optimal won't trigger
/// trailing/breakeven stops (unless hard risk fires).
pub fn coin_optimal_hold_min(symbol: &str) -> f64 {
    match symbol.to_uppercase().as_str() {
        "XRP"  => 196.0,
        "ARB"  => 167.0,
        "ADA"  => 137.0,
        "LINK" => 114.0,
        "CRV"  => 111.0,
        "ONDO" => 101.0,
        "ATOM" => 100.0,
        "SAND" =>  95.0,
        "BTC"  =>  93.0,
        "NEAR" =>  90.0,
        "AVAX" =>  72.0,
        "ETH"  =>  68.0,
        "LTC"  =>  67.0,
        "DOT"  =>  67.0,
        "GRT"  =>  54.0,
        "SUI"  =>  54.0,
        "INJ"  =>  54.0,
        "AXS"  =>  49.0,
        "SOL"  =>  45.0,
        "TIA"  =>  42.0,
        "PEPE" =>   9.0,
        // Meme coins default to short holds
        s if coin_tier(s) == "meme" => 15.0,
        // Unknown coins — moderate default
        _ => 60.0,
    }
}

/// Get the coin tier for a symbol.
pub fn coin_tier(symbol: &str) -> &'static str {
    match symbol.to_uppercase().as_str() {
        "BTC" | "ETH" => "large",
        "SOL" | "AVAX" | "ADA" | "DOT" | "LINK" | "MATIC" | "ATOM" | "UNI" => "mid",
        "DOGE" | "SHIB" | "PEPE" | "FLOKI" | "BONK" | "WIF" | "MEME"
        | "HOUSE" | "GHIBLI" | "NEIRO" | "TURBO" | "BRETT" | "MOG"
        | "PEPECOIN" | "CHEEMS" | "SAMO" | "FUN"
        | "FARTCOIN" | "PUMP" | "XDG" | "PENGU" | "SPX" | "VIRTUAL"
        | "GRIFFAIN" | "POPCAT" | "MOODENG" | "PONKE" | "SNEK" | "MEW"
        | "GOAT" | "ZEREBRO" | "PNUT" | "GIGA" | "TOSHI" | "FWOG"
        | "MUMU" | "CHILLGUY" | "TRUMP" | "MELANIA" | "MYRO" | "SLERF"
        | "BOME" | "WEN" | "DOG" | "LADYS" | "ANDY" | "HIGHER"
        | "DEGEN" | "LUCE" | "SUNDOG" | "AI16Z"
        | "BANANAS31" | "BILLY" | "CAT" | "ESPORTS" | "LOFI"
        | "PACT" | "SPACE" | "WARD" | "COPM" => "meme",
        "ESP" | "WLFI" | "LCX" | "ACX" | "JUNO" | "CQT" => "defi",
        "XION" | "UP" | "HNT" | "AIOZ" | "DAG" => "infra",
        "OMG" | "ALT" | "TBTC" | "BIT" => "l2",
        "CYBER" | "GUN" | "PLAY" | "SLAY" | "XNY" | "REN" | "WELL" | "REP" => "small",
        _ => "mid",
    }
}

/// Behavioral meme detector for unknown/new coins.
/// Returns true if the coin BEHAVES like a meme even if not in the static list.
/// Used by lane classifier to assign L4 to new coins that pump like memes.
pub fn is_behavioral_meme(
    price: f64,
    atr_pct: f64,
    vol_ratio: f64,
    corr_btc: f64,
    symbol: &str,
) -> bool {
    // Already classified as meme — skip behavioral check
    if coin_tier(symbol) == "meme" { return false; }

    // Behavioral signals that indicate meme-like dynamics:
    // 1. Ultra-low price (< $0.01) — penny/micro cap
    let ultra_low_price = price > 0.0 && price < 0.01;

    // 2. Extreme volatility relative to price (ATR > 8% of price)
    let extreme_vol = atr_pct > 0.08;

    // 3. Volume spike with no structural basis (vol > 5x but no EMA history signal)
    let vol_spike = vol_ratio >= 5.0;

    // 4. Fully decoupled from BTC (correlation < 0.1 absolute)
    let btc_decoupled = corr_btc.abs() < 0.10;

    // Classify as behavioral meme if 3+ signals fire
    let score = ultra_low_price as u8
        + extreme_vol as u8
        + vol_spike as u8
        + btc_decoupled as u8;

    score >= 3
}

/// Kraken price decimal precision per symbol.
/// Sending too many decimals causes EOrder:Invalid price.
pub fn price_decimals(symbol: &str) -> usize {
    match symbol {
        "BTC" => 1,
        "ETH" | "LTC" | "BCH" | "ETC" | "AAVE" | "MKR" | "PAXG" => 2,
        "SOL" | "AVAX" | "LINK" | "ATOM" | "UNI" | "NEAR" => 2,
        "DOT" | "INJ" | "APT" | "SUI" | "OP" | "ARB" | "TIA" | "SEI" => 4,
        "FIL" => 3,
        "ADA" | "ALGO" | "XLM" | "XRP" | "MATIC" | "FTM" | "LDO" | "CRV" => 4,
        "PEPE" | "BONK" | "CHEEMS" => 9,
        "SHIB" | "FLOKI" | "HOUSE" | "GHIBLI" | "NEIRO" | "TURBO" | "MOG" | "SAMO" => 8,
        "DOGE" | "WIF" | "BRETT" | "PEPECOIN" | "SLAY" | "XNY" | "WELL" | "CQT" => 5,
        // New/smaller coins — safe default
        _ => 4,
    }
}

/// Format a price with the correct number of decimals for Kraken.
pub fn format_price(symbol: &str, price: f64) -> String {
    let dec = price_decimals(symbol);
    format!("{:.prec$}", price, prec = dec)
}

// â"€â"€ Regime Inference â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

/// Infer market regime from features.
///
/// Returns one of: "crash", "volatile", "bullish", "trending", "sideways"
pub fn infer_regime(
    feats: &serde_json::Value,
    quantum_state: &str,
    config: &TradingConfig,
) -> &'static str {
    let disable_cvar_gates = env_bool("DISABLE_CVAR_GATES", true);
    let f = |key: &str, def: f64| -> f64 { feats.get(key).and_then(|v| v.as_f64()).unwrap_or(def) };

    let qreg = f("quant_regime", 0.0);
    let qvol = f("quant_vol", 0.0);
    let _rsi = f("rsi", 50.0);
    let mstate = f("market_state", 0.5);
    let trend = f("trend_score", 0.0) as i32;
    let cvar_regime = feats
        .get("cvar_regime")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");

    // Crash detection (highest priority)
    if (!disable_cvar_gates && cvar_regime == "crash")
        || (qreg < config.crash_qregime && qvol.abs() > config.crash_qvol)
    {
        return "crash";
    }
    if quantum_state == "volatile" || qvol.abs() > 0.015 {
        return "volatile";
    }

    // Bullish
    if (qreg > 0.003 || mstate > 0.65) && trend >= 2 {
        return "bullish";
    }

    // Trending
    if trend >= 1 || mstate > 0.55 {
        return "trending";
    }

    "sideways"
}

fn env_bool(key: &str, default: bool) -> bool {
    match std::env::var(key) {
        Ok(v) => matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON"),
        Err(_) => default,
    }
}

/// Get .env file mtime for change detection.
fn env_file_mtime() -> Option<SystemTime> {
    // Check cwd first, then common project locations
    let candidates = [".env", "data/../.env"];
    for p in &candidates {
        if let Ok(meta) = std::fs::metadata(p) {
            if let Ok(mt) = meta.modified() {
                return Some(mt);
            }
        }
    }
    None
}

// â"€â"€ Profile Selection â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

/// Trading profile with resolved parameters.
#[derive(Clone, Debug)]
pub struct ProfileParams {
    pub name: &'static str,
    pub max_positions: usize,
    pub cooldown_sec: u64,
    pub max_spread_pct: f64,
    pub max_slip_pct: f64,
    pub min_liq: f64,
    pub min_vol_ratio: f64,
}

/// Select aggressive or defensive profile based on conditions.
///
/// Returns (profile_params, reason).
pub fn select_profile(
    config: &TradingConfig,
    regime: &str,
    market_state: f64,
    rsi: f64,
    circuit_state: &CircuitBreakerState,
) -> (ProfileParams, &'static str) {
    // Circuit breaker can force defensive
    if circuit_state.force_defensive {
        return (
            ProfileParams {
                name: "defensive",
                max_positions: config.def_max_positions,
                cooldown_sec: config.def_cooldown_sec,
                max_spread_pct: config.def_max_spread_pct,
                max_slip_pct: config.def_max_slip_pct,
                min_liq: config.def_min_liq,
                min_vol_ratio: config.def_min_vol_ratio,
            },
            "circuit_breaker",
        );
    }

    // Crash/volatile â†' always defensive
    if regime == "crash" || regime == "volatile" {
        return (
            ProfileParams {
                name: "defensive",
                max_positions: config.def_max_positions,
                cooldown_sec: config.def_cooldown_sec,
                max_spread_pct: config.def_max_spread_pct,
                max_slip_pct: config.def_max_slip_pct,
                min_liq: config.def_min_liq,
                min_vol_ratio: config.def_min_vol_ratio,
            },
            "regime_weak",
        );
    }

    // Strong market â†' aggressive
    if market_state >= 0.6 && rsi < 70.0 && (regime == "bullish" || regime == "trending") {
        return (
            ProfileParams {
                name: "aggressive",
                max_positions: config.aggr_max_positions,
                cooldown_sec: config.aggr_cooldown_sec,
                max_spread_pct: config.aggr_max_spread_pct,
                max_slip_pct: config.aggr_max_slip_pct,
                min_liq: config.aggr_min_liq,
                min_vol_ratio: config.aggr_min_vol_ratio,
            },
            "strong_market",
        );
    }

    // Default â†' defensive
    (
        ProfileParams {
            name: "defensive",
            max_positions: config.def_max_positions,
            cooldown_sec: config.def_cooldown_sec,
            max_spread_pct: config.def_max_spread_pct,
            max_slip_pct: config.def_max_slip_pct,
            min_liq: config.def_min_liq,
            min_vol_ratio: config.def_min_vol_ratio,
        },
        "default",
    )
}

// â"€â"€ Circuit Breaker â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

/// Circuit breaker state — tracks daily loss limits.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CircuitLevel {
    Green,  // Normal trading
    Yellow, // Reduced sizes (defensive)
    Red,    // Exit-only, no new entries
    Black,  // Full stop
}

impl CircuitLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Green => "GREEN",
            Self::Yellow => "YELLOW",
            Self::Red => "RED",
            Self::Black => "BLACK",
        }
    }
}

#[derive(Clone, Debug)]
pub struct CircuitBreakerState {
    pub daily_pnl: f64,
    pub unrealized_pnl: f64,
    pub start_balance_usd: f64,
    pub force_defensive: bool,
    pub stop_trading: bool,
    pub exit_only: bool,
    pub level: CircuitLevel,
    pub last_reset_day: u16,
}

impl CircuitBreakerState {
    pub fn new(start_balance: f64) -> Self {
        Self {
            daily_pnl: 0.0,
            unrealized_pnl: 0.0,
            start_balance_usd: start_balance,
            force_defensive: false,
            stop_trading: false,
            exit_only: false,
            level: CircuitLevel::Green,
            last_reset_day: current_day_of_year(),
        }
    }

    /// Check circuit breaker — 4 levels: GREEN → YELLOW → RED → BLACK
    pub fn check(&mut self, config: &TradingConfig) {
        if config.max_daily_loss_usd <= 0.0
            && config.aggr_circuit_breaker_pct <= 0.0
            && config.def_circuit_breaker_pct <= 0.0
        {
            if self.stop_trading || self.force_defensive || self.exit_only {
                println!("[CIRCUIT] Thresholds disabled — clearing to GREEN");
                self.stop_trading = false;
                self.force_defensive = false;
                self.exit_only = false;
                self.level = CircuitLevel::Green;
            }
            return;
        }

        let total_pnl = self.daily_pnl + self.unrealized_pnl;

        // 4-level circuit breaker (USD-based)
        if config.max_daily_loss_usd > 0.0 {
            let max_loss = config.max_daily_loss_usd;
            let loss = (-total_pnl).max(0.0);
            let ratio = loss / max_loss;
            let prev = self.level.clone();

            if ratio >= 0.75 {
                self.level = CircuitLevel::Black;
                self.stop_trading = true;
                self.force_defensive = true;
                self.exit_only = true;
            } else if ratio >= 0.50 {
                self.level = CircuitLevel::Red;
                self.stop_trading = false;
                self.force_defensive = true;
                self.exit_only = true;
            } else if ratio >= 0.25 {
                self.level = CircuitLevel::Yellow;
                self.stop_trading = false;
                self.force_defensive = true;
                self.exit_only = false;
            } else {
                self.level = CircuitLevel::Green;
                self.stop_trading = false;
                self.force_defensive = false;
                self.exit_only = false;
            }

            if self.level != prev {
                println!(
                    "[CIRCUIT] {} -> {} (loss=${:.2}/{:.2} = {:.0}%)",
                    prev.as_str(), self.level.as_str(), loss, max_loss, ratio * 100.0
                );
            }
            return;
        }

        // Fallback: percentage-based (legacy)
        if !config.use_balance_for_risk || self.start_balance_usd <= 0.0 || total_pnl >= 0.0 {
            return;
        }
        let loss_pct = (-total_pnl / self.start_balance_usd) * 100.0;
        if config.def_circuit_breaker_pct > 0.0 && loss_pct >= config.def_circuit_breaker_pct {
            if !self.stop_trading {
                println!("[CIRCUIT] BLACK — loss {loss_pct:.1}% >= {:.1}%", config.def_circuit_breaker_pct);
            }
            self.stop_trading = true;
            self.force_defensive = true;
            self.exit_only = true;
            self.level = CircuitLevel::Black;
        } else if config.aggr_circuit_breaker_pct > 0.0 && loss_pct >= config.aggr_circuit_breaker_pct {
            if !self.force_defensive {
                println!("[CIRCUIT] YELLOW — loss {loss_pct:.1}% >= {:.1}%", config.aggr_circuit_breaker_pct);
            }
            self.force_defensive = true;
            self.level = CircuitLevel::Yellow;
        }
    }

    /// Add realized P&L from a closed trade.
    pub fn add_pnl(&mut self, pnl_usd: f64) {
        self.daily_pnl += pnl_usd;
    }

    /// Update unrealized P&L from currently open positions.
    pub fn set_unrealized_pnl(&mut self, pnl_usd: f64) {
        self.unrealized_pnl = pnl_usd;
    }

    /// Reset at day boundary.
    pub fn maybe_reset_daily(&mut self) {
        let today = current_day_of_year();
        if today != self.last_reset_day {
            println!("[CIRCUIT] New day - resetting to GREEN");
            self.daily_pnl = 0.0;
            self.unrealized_pnl = 0.0;
            self.force_defensive = false;
            self.stop_trading = false;
            self.exit_only = false;
            self.level = CircuitLevel::Green;
            self.last_reset_day = today;
        }
    }
}

fn current_day_of_year() -> u16 {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    // Rough day-of-year (good enough for daily reset)
    ((now / 86400) % 366) as u16
}

pub fn slip_regime_mult(regime: &str) -> f64 {
    match regime {
        "bullish" => 1.3,
        "trending" => 1.2,
        "sideways" => 1.0,
        "volatile" => 0.8,
        "crash" => 0.5,
        _ => 1.0,
    }
}

// â"€â"€ Pair Formatting â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

/// Format a symbol into a Kraken trading pair (e.g., "BTC" â†' "BTCUSD").
pub fn pair_for(symbol: &str) -> String {
    let quote = std::env::var("QUOTE_CURRENCY").unwrap_or_else(|_| "USD".into());
    format!("{}{}", symbol.to_uppercase(), quote)
}

#[cfg(test)]
mod tests {
    use super::{CircuitBreakerState, TradingConfig};

    #[test]
    fn circuit_uses_realized_plus_unrealized() {
        let mut cfg = TradingConfig::default();
        cfg.aggr_circuit_breaker_pct = 2.0;
        cfg.def_circuit_breaker_pct = 5.0;

        let mut c = CircuitBreakerState::new(1000.0);
        c.add_pnl(-10.0);
        c.set_unrealized_pnl(-15.0); // total -25 => 2.5%
        c.check(&cfg);
        assert!(c.force_defensive);
        assert!(!c.stop_trading);
    }

    #[test]
    fn circuit_stops_on_total_loss_limit() {
        let mut cfg = TradingConfig::default();
        cfg.aggr_circuit_breaker_pct = 2.0;
        cfg.def_circuit_breaker_pct = 3.0;

        let mut c = CircuitBreakerState::new(1000.0);
        c.add_pnl(-20.0);
        c.set_unrealized_pnl(-15.0); // total -35 => 3.5%
        c.check(&cfg);
        assert!(c.force_defensive);
        assert!(c.stop_trading);
    }

    #[test]
    fn circuit_reset_clears_unrealized() {
        let mut c = CircuitBreakerState::new(1000.0);
        c.add_pnl(-50.0);
        c.set_unrealized_pnl(-10.0);
        c.force_defensive = true;
        c.stop_trading = true;
        c.last_reset_day = c.last_reset_day.wrapping_sub(1);
        c.maybe_reset_daily();
        assert_eq!(c.daily_pnl, 0.0);
        assert_eq!(c.unrealized_pnl, 0.0);
        assert!(!c.force_defensive);
        assert!(!c.stop_trading);
    }
}
