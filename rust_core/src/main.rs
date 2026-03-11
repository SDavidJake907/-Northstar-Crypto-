#![recursion_limit = "256"]

use anyhow::Result;
use serde_json::json;
use std::collections::{HashMap, VecDeque};
use std::env;
use std::path::PathBuf;
use std::time::Instant;

// ── Infrastructure ──────────────────────────────────────────────────
mod infra;
use infra::{book, ws, kraken_api, event_bus};

// ── Config ─────────────────────────────────────────────────────────
mod config;

// ── Signals ────────────────────────────────────────────────────────
mod signals;
use signals::{features, strategy, strategy_helpers, market_brain, market_watcher};

// ── LLM layer (AI client, prompts, parsers) ──────────────────────
mod llm;
use llm as ai_bridge;
use llm::nemo_exit;
#[allow(unused_imports)]
pub use llm::nvidia_tools;

// ── Engine (trading loop, orders, risk) ──────────────────────────
mod engine;
use engine as trading_loop;
use engine::{journal, trade_flow, portfolio_allocator};

// ── Addons (cloud intel, scanners, memory, NPU, backtester) ─────
mod addons;
use addons::{
    cloud_intel, news_sentiment, vision_scanner, mover_scanner,
    memory_packet, nemo_memory, backtester, ohlcv_fetcher,
};
// Re-export at crate root so submodules can use crate::npu_bridge::*
#[allow(unused_imports)]
pub use addons::npu_bridge;

// ── Optional modules — compile-gated via Cargo features ────────────
#[cfg(feature = "nemo_chat")]
mod nemo_chat;
#[cfg(feature = "nemo_optimizer")]
mod nemo_optimizer;

use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use trade_flow::{Trade, TradeFlowState, TradeSide};

const BOOK_HISTORY_LEN: usize = 10;

fn now_ts() -> f64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

fn parse_f64(v: &serde_json::Value) -> f64 {
    if let Some(n) = v.as_f64() {
        return n;
    }
    if let Some(s) = v.as_str() {
        return s.parse::<f64>().unwrap_or(0.0);
    }
    0.0
}

fn extract_symbol(item: &serde_json::Value) -> Option<String> {
    let pair = item.get("symbol").and_then(|s| s.as_str())?;
    let sym = pair.split('/').next().unwrap_or("").trim().to_string();
    if sym.is_empty() {
        None
    } else {
        Some(sym)
    }
}

fn parse_book_levels(item: &serde_json::Value, side: &str, only_pos: bool) -> Vec<(f64, f64)> {
    let mut out = Vec::new();
    if let Some(arr) = item.get(side).and_then(|x| x.as_array()) {
        for lvl in arr {
            let p = parse_f64(lvl.get("price").unwrap_or(&json!(0)));
            let q = parse_f64(lvl.get("qty").unwrap_or(&json!(0)));
            if only_pos {
                if p > 0.0 && q > 0.0 {
                    out.push((p, q));
                }
            } else {
                out.push((p, q));
            }
        }
    }
    out
}

fn book_metrics(book: &book::OrderBook) -> features::OrderbookMetrics {
    if book.bids.is_empty() || book.asks.is_empty() {
        return features::OrderbookMetrics::default();
    }
    let best_bid = book.bids[0].price;
    let best_ask = book.asks[0].price;
    if best_ask < best_bid {
        return features::OrderbookMetrics::default();
    }
    let mid = (best_bid + best_ask) / 2.0;
    let spread_pct = if mid > 0.0 {
        (best_ask - best_bid) / mid * 100.0
    } else {
        0.0
    };
    let bid_vol: f64 = book.bids.iter().map(|b| b.volume).sum();
    let ask_vol: f64 = book.asks.iter().map(|a| a.volume).sum();
    let denom = (bid_vol + ask_vol).max(1e-9);
    let imbalance = (bid_vol - ask_vol) / denom;
    features::OrderbookMetrics {
        spread_pct,
        imbalance,
    }
}

// ── Book Reversal Tracker ────────────────────────────────────────────

#[derive(Clone, Debug, Default)]
struct BookReversalState {
    buf: [f64; BOOK_HISTORY_LEN],
    idx: usize,
    len: usize,
    last: features::BookReversalMetrics,
}

impl BookReversalState {
    fn last_n_avg(&self, n: usize) -> f64 {
        if self.len < n || n == 0 {
            return 0.0;
        }
        let mut sum = 0.0;
        for k in 1..=n {
            let idx = (self.idx + BOOK_HISTORY_LEN - k) % BOOK_HISTORY_LEN;
            sum += self.buf[idx];
        }
        sum / n as f64
    }

    fn update(&mut self, imbalance: f64) -> features::BookReversalMetrics {
        let mut prev_trend = 0;
        if self.len >= 3 {
            let prev_avg = self.last_n_avg(3);
            if prev_avg > 0.1 {
                prev_trend = 2;
            } else if prev_avg < -0.1 {
                prev_trend = -2;
            }
        }

        self.buf[self.idx] = imbalance;
        self.idx = (self.idx + 1) % BOOK_HISTORY_LEN;
        if self.len < BOOK_HISTORY_LEN {
            self.len += 1;
        }

        let mut curr_trend = 0;
        let mut avg_imb = 0.0;
        let mut strength = 0.0;
        let mut reversal = false;
        let mut reversal_dir = 0;

        if self.len >= 3 {
            avg_imb = self.last_n_avg(3);
            if avg_imb > 0.15 {
                curr_trend = 2;
            } else if avg_imb < -0.15 {
                curr_trend = -2;
            } else if avg_imb > 0.05 {
                curr_trend = 1;
            } else if avg_imb < -0.05 {
                curr_trend = -1;
            }

            strength = (avg_imb.abs() / 0.5).min(1.0);

            if prev_trend == 2 && curr_trend <= -1 {
                reversal = true;
                reversal_dir = -1;
            } else if prev_trend == -2 && curr_trend >= 1 {
                reversal = true;
                reversal_dir = 1;
            }
        }

        self.last = features::BookReversalMetrics {
            trend: curr_trend,
            reversal,
            reversal_dir,
            strength,
            avg_imb,
        };
        self.last.clone()
    }
}

// ── Per-Symbol State ─────────────────────────────────────────────────

struct SymbolState {
    candles: VecDeque<features::Candle>,
    ref_closes: VecDeque<f64>,
    trade_flow: TradeFlowState,
    book_reversal: BookReversalState,
    ohlc_last_interval: String,
    feature_bufs: features::FeatureBuffers,
    // Denoised L2 inputs (optional, controlled via DENOISE_ALPHA)
    smoothed_book_imb: f64,
    smoothed_spread_pct: f64,
    smoothed_buy_ratio: f64,
    smoothed_sell_ratio: f64,
    smoothed_inited: bool,
}

impl SymbolState {
    fn new(trade_window_sec: f64) -> Self {
        Self {
            candles: VecDeque::new(),
            ref_closes: VecDeque::new(),
            trade_flow: TradeFlowState::new(trade_window_sec, 200),
            book_reversal: BookReversalState::default(),
            ohlc_last_interval: String::new(),
            feature_bufs: features::FeatureBuffers::new(),
            smoothed_book_imb: 0.0,
            smoothed_spread_pct: 0.0,
            smoothed_buy_ratio: 0.5,
            smoothed_sell_ratio: 0.5,
            smoothed_inited: false,
        }
    }
}

// ── Feature Serialization ────────────────────────────────────────────

fn feats_to_value(f: &features::Features) -> serde_json::Value {
    json!({
        // Price + moving averages (ema9 removed — unused downstream)
        "price": f.price,
        "ema9": f.ema9,
        "ema21": f.ema21,
        "ema50": f.ema50,
        "ema55": f.ema55,
        "ema200": f.ema200,

        // Technical indicators
        "rsi": f.rsi,
        "macd_hist": f.macd_hist,
        "atr": f.atr,
        "bb_upper": f.bb_upper,
        "bb_lower": f.bb_lower,
        "zscore": f.zscore,
        "vol_ratio": f.vol_ratio,

        // Quant metrics
        "quant_vol": f.quant_vol,
        "quant_regime": f.quant_regime,

        // Trend + momentum
        "trend_score": f.trend_score,
        "momentum_score": f.momentum_score,
        "trend_strength": f.trend_strength,

        // Orderbook
        "spread_pct": f.spread_pct,
        "book_imbalance": f.book_imbalance,
        "buy_ratio": f.buy_ratio,
        "book_trend": match f.book_trend {
            2 => "BULLISH",
            1 => "MILD_BULLISH",
            0 => "NEUTRAL",
            -1 => "MILD_BEARISH",
            -2 => "BEARISH",
            _ => "NEUTRAL",
        },
        "book_reversal": f.book_reversal,
        "book_strength": f.book_strength,
        "book_avg_imb": f.book_avg_imb,

        "trend_direction": match f.trend_strength {
            5 => "STRONG_BULL",
            4 => "BULL",
            3 => "MILD_BULL",
            2 => "WEAK_BULL",
            1 => "TICK_UP",
            -1 => "TICK_DOWN",
            -2 => "WEAK_BEAR",
            -3 => "MILD_BEAR",
            -4 => "BEAR",
            -5 => "STRONG_BEAR",
            _ => "NEUTRAL",
        },
        // Quantum math indicators
        "hurst_exp": f.hurst_exp,
        "shannon_entropy": f.shannon_entropy,
        "autocorr_lag1": f.autocorr_lag1,
        "adx": f.adx,
        "market_state": f.market_state,
    })
}

// ── Lock File Guard ──────────────────────────────────────────────────

struct LockFileGuard {
    path: PathBuf,
}

impl LockFileGuard {
    fn new(path: PathBuf) -> Self {
        Self { path }
    }
}

impl Drop for LockFileGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

// ── Main ─────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    // ── .env Loading ────────────────────────────────────────────
    let mut dotenv_loaded = false;
    let mut dotenv_note: Option<String> = None;

    // ── CPU capability + BLAS backend snapshot ──
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        let sse2 = std::is_x86_feature_detected!("sse2");
        let avx = std::is_x86_feature_detected!("avx");
        let avx2 = std::is_x86_feature_detected!("avx2");
        let fma = std::is_x86_feature_detected!("fma");
        let avx_vnni = std::is_x86_feature_detected!("avx512vnni");
        tracing::info!(
            "[CPU] SIMD: sse2={} avx={} avx2={} fma={} vnni={}",
            sse2, avx, avx2, fma, avx_vnni
        );
    }
    #[cfg(feature = "openblas-system")]
    tracing::info!("[BLAS] Backend=openblas-system");
    #[cfg(feature = "openblas-static")]
    tracing::info!("[BLAS] Backend=openblas-static");
    #[cfg(feature = "intel-mkl-system")]
    tracing::info!("[BLAS] Backend=intel-mkl-system");

    if let Ok(env_path) = std::env::var("ENV_PATH") {
        let p = std::path::PathBuf::from(env_path);
        if p.exists() {
            if dotenvy::from_path(&p).is_ok() {
                dotenv_note = Some(format!("Loaded .env from ENV_PATH: {}", p.display()));
                dotenv_loaded = true;
            } else {
                dotenv_note = Some(format!("Failed to load .env from ENV_PATH: {}", p.display()));
            }
        }
    }

    if !dotenv_loaded {
        match dotenvy::dotenv() {
            Ok(p) => {
                dotenv_note = Some(format!("Loaded .env from dotenvy auto-search: {}", p.display()));
                dotenv_loaded = true;
            }
            Err(e) => {
                dotenv_note = Some(format!("Failed to load .env from cwd/parents: {e}"));
            }
        }
    }
    let cwd = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
    let cwd_display = cwd.display().to_string();
    if !dotenv_loaded {
        if let Ok(cwd) = std::env::current_dir() {
            let mut cur = Some(cwd.as_path());
            for _ in 0..4 {
                if let Some(dir) = cur {
                    let candidate = dir.join(".env");
                    if candidate.exists() {
                        if dotenvy::from_path(&candidate).is_ok() {
                            dotenv_note = Some(format!(
                                "Loaded .env from {} (cwd missing .env)",
                                candidate.display()
                            ));
                            dotenv_loaded = true;
                            break;
                        }
                    }
                    cur = dir.parent();
                }
            }
        }
    }

    // ── Tracing ─────────────────────────────────────────────────
    let json_logs = std::env::var("LOG_JSON")
        .map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(false);
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    if json_logs {
        tracing_subscriber::fmt()
            .with_env_filter(filter)
            .json()
            .with_current_span(false)
            .with_span_list(false)
            .init();
    } else {
        tracing_subscriber::fmt().with_env_filter(filter).init();
    }
    if let Some(note) = dotenv_note {
        tracing::warn!("[MAIN] {note}");
    } else if !dotenv_loaded {
        tracing::warn!("[MAIN] .env not found in cwd or parents");
    }
    tracing::info!(
        "[BUILD] {} v{}",
        env!("CARGO_PKG_NAME"),
        env!("CARGO_PKG_VERSION")
    );
    tracing::info!("[MAIN] cwd={}", cwd_display);

    // ── Detect backtest mode early (before instance guard) ─────
    let args: Vec<String> = std::env::args().collect();
    let is_backtest = args.iter().any(|a| a == "--backtest");

    // ── Instance Guard (port bind) — skip for backtest mode ─────
    let health_port: u16 = env::var("HEALTH_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(9091);
    let health_listener = if is_backtest {
        tracing::info!("[BACKTEST] Skipping instance guard (backtest mode)");
        None
    } else {
        match std::net::TcpListener::bind(format!("127.0.0.1:{health_port}")) {
            Ok(l) => {
                l.set_nonblocking(true).ok();
                tracing::info!("[MAIN] Instance guard acquired (port {health_port})");
                Some(l)
            }
            Err(e) => {
                tracing::error!(
                    "[MAIN] Health port {health_port} unavailable ({e}) — another instance may be running. Exiting."
                );
                anyhow::bail!("Instance already running on port {health_port}");
            }
        }
    };

    // ── Lock File — skip for backtest mode ─────────────────────
    let lock_path = cwd.join("data").join("hybrid_kraken.lock");
    let _lock_guard = if is_backtest {
        tracing::info!("[BACKTEST] Skipping lock file (backtest mode)");
        None
    } else {
        if lock_path.exists() {
            // Check if the PID in the lockfile is still alive
            let mut stale = false;
            if let Ok(content) = std::fs::read_to_string(&lock_path) {
                if let Some(pid_str) = content.lines()
                    .find(|l| l.starts_with("pid="))
                    .map(|l| &l[4..])
                {
                    if let Ok(pid) = pid_str.trim().parse::<u32>() {
                        // On Windows, try to open the process to check if it's alive
                        let check = std::process::Command::new("tasklist")
                            .args(["/FI", &format!("PID eq {pid}"), "/NH"])
                            .output();
                        match check {
                            Ok(out) => {
                                let stdout = String::from_utf8_lossy(&out.stdout);
                                if !stdout.contains(&pid.to_string()) {
                                    stale = true;
                                    tracing::warn!(
                                        "[MAIN] Stale lock file (pid={pid} not running) — removing"
                                    );
                                }
                            }
                            Err(_) => {
                                stale = true; // can't check — assume stale
                            }
                        }
                    }
                }
            }
            if stale {
                let _ = std::fs::remove_file(&lock_path);
            } else {
                eprintln!(
                    "[MAIN] FATAL: lock file already exists: {}\n\
                     Another instance is running. Kill it first or remove the lock file if stale.",
                    lock_path.display()
                );
                std::process::exit(1);
            }
        }
        if let Some(parent) = lock_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(
            &lock_path,
            format!("pid={}\nts={}", std::process::id(), now_ts()),
        )?;

        let default_panic = std::panic::take_hook();
        let lock_for_panic = lock_path.clone();
        std::panic::set_hook(Box::new(move |info| {
            let _ = std::fs::remove_file(&lock_for_panic);
            // Write panic to log file so we can diagnose crashes
            let msg = format!("[PANIC] {:?}\n", info);
            let _ = std::fs::OpenOptions::new()
                .create(true).append(true)
                .open("logs/panic.log")
                .map(|mut f| { use std::io::Write; let _ = f.write_all(msg.as_bytes()); });
            tracing::error!("[PANIC] {}", info);
            default_panic(info);
        }));
        // Ctrl+C handled via tokio::signal::ctrl_c() in the main loop
        // LockFileGuard Drop + panic hook handle cleanup
        Some(LockFileGuard::new(lock_path.clone()))
    };

    if !cwd.join(".env").exists() {
        tracing::warn!(
            "[MAIN] cwd has no .env — run from project root"
        );
    }

    // ── Backtest CLI Mode ──────────────────────────────────────
    if let Some(bt_idx) = args.iter().position(|a| a == "--backtest") {
        let bt_mode = args.get(bt_idx + 1).map(|s| s.as_str()).unwrap_or("help");
        let strategy = args.iter()
            .position(|a| a == "--strategy")
            .and_then(|i| args.get(i + 1))
            .map(|s| s.as_str())
            .unwrap_or("momentum");
        let coins_arg = args.iter()
            .position(|a| a == "--coins")
            .and_then(|i| args.get(i + 1))
            .map(|s| s.as_str())
            .unwrap_or("BTC,ETH,SOL");
        let days: u32 = args.iter()
            .position(|a| a == "--days")
            .and_then(|i| args.get(i + 1))
            .and_then(|s| s.parse().ok())
            .unwrap_or(7);
        let interval: u32 = args.iter()
            .position(|a| a == "--interval")
            .and_then(|i| args.get(i + 1))
            .and_then(|s| s.parse().ok())
            .unwrap_or(5);

        let coins: Vec<String> = coins_arg
            .split(',')
            .map(|s| s.trim().to_uppercase())
            .filter(|s| !s.is_empty())
            .collect();
        let quiet = args.iter().any(|a| a == "--quiet" || a == "-q");
        let has_coin_flag = args.iter().any(|a| a == "--coins");

        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?;

        return rt.block_on(async move {
            match bt_mode {
                "journal" => {
                    let coin_filter = if has_coin_flag { Some(coins.as_slice()) } else { None };
                    tracing::info!("[BACKTEST] Mode: journal replay, strategy={strategy}");
                    backtester::run_journal_replay(strategy, coin_filter, quiet).await?;
                }
                "ohlcv" => {
                    tracing::info!(
                        "[BACKTEST] Mode: OHLCV, coins={:?}, days={days}, interval={interval}m, strategy={strategy}",
                        coins
                    );
                    backtester::run_ohlcv_backtest(&coins, days, strategy, interval).await?;
                }
                "compare" => {
                    tracing::info!(
                        "[BACKTEST] Mode: compare all strategies, coins={:?}, days={days}",
                        coins
                    );
                    backtester::run_compare(&coins, days, interval).await?;
                }
                "download" => {
                    tracing::info!(
                        "[BACKTEST] Mode: download OHLCV only, coins={:?}, days={days}",
                        coins
                    );
                    let intervals = vec![1, 5, 15, 60];
                    ohlcv_fetcher::download_all(&coins, &intervals, days).await?;
                }
                "validate" => {
                    // Validate the most recent backtest result with NVIDIA cloud
                    let results_dir = std::path::Path::new("data/backtest_results");
                    if results_dir.exists() {
                        let mut entries: Vec<_> = std::fs::read_dir(results_dir)?
                            .filter_map(|e| e.ok())
                            .filter(|e| e.path().extension().map(|x| x == "json").unwrap_or(false))
                            .collect();
                        entries.sort_by_key(|e| std::cmp::Reverse(e.metadata().ok().and_then(|m| m.modified().ok())));
                        if let Some(latest) = entries.first() {
                            let json = std::fs::read_to_string(latest.path())?;
                            tracing::info!("[BACKTEST] Validating: {}", latest.path().display());
                            let review = cloud_intel::validate_strategy(&json).await?;
                            println!("\n{review}");
                        } else {
                            eprintln!("No backtest results found in data/backtest_results/");
                        }
                    } else {
                        eprintln!("Run a backtest first: --backtest ohlcv --coins BTC,ETH --days 7");
                    }
                }
                _ => {
                    println!("Usage: hybrid_kraken_core --backtest <mode> [options]");
                    println!();
                    println!("Modes:");
                    println!("  journal   Replay trade journal through a strategy");
                    println!("  ohlcv     Backtest on historical OHLCV candle data");
                    println!("  compare   Compare all strategies side by side");
                    println!("  download  Download OHLCV data only (no backtest)");
                    println!("  validate  Send latest backtest to NVIDIA cloud for review");
                    println!();
                    println!("Options:");
                    println!("  --strategy <name>   momentum|mean_reversion|breakout|scalp (default: momentum)");
                    println!("  --coins <list>      Comma-separated coins (default: BTC,ETH,SOL)");
                    println!("  --days <n>          Days of history (default: 7)");
                    println!("  --interval <min>    Candle interval in minutes (default: 5)");
                }
            }
            Ok(())
        });
    }

    // ── Config ──────────────────────────────────────────────────
    let trading_config = config::TradingConfig::from_env();
    let addons_config = config::AddonsConfig::from_env();
    let paper_trading = trading_config.paper_trading;
    let poll_seconds = trading_config.poll_seconds;

    let symbols = env::var("SYMBOLS")
        .or_else(|_| env::var("WS_SYMBOLS"))
        .unwrap_or_else(|_| "BTC,ETH".to_string());
    let depth = env::var("WS_DEPTH")
        .ok()
        .and_then(|v| v.parse::<i64>().ok())
        .unwrap_or(10);
    let ohlc_interval = env::var("WS_OHLC_INTERVAL")
        .ok()
        .and_then(|v| v.parse::<i64>().ok())
        .unwrap_or(1);
    let debug_every = env::var("WS_DEBUG_EVERY_SEC")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(5);
    let trade_window_sec = env::var("WS_TRADE_WINDOW_SEC")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(10.0);
    let book_depth = env::var("BOOK_DEPTH")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(depth as usize);

    let symbols: Vec<String> = symbols
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    tracing::info!(
        "Hybrid Kraken core booting... symbols={:?} depth={} ohlc={}",
        symbols,
        depth,
        ohlc_interval
    );

    let client = ws::WsClient::new(debug_every);
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;

    rt.block_on(async move {
        let ws_queue_cap: usize = env::var("WS_QUEUE_CAP")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .map(|v| v.clamp(128, 65_536))
            .unwrap_or(4096);

        // ── Shared State: dynamic coin list + reconnect signal ──
        let active_coins: Arc<RwLock<Vec<String>>> =
            Arc::new(RwLock::new(symbols.clone()));
        let ws_reconnect: Arc<AtomicBool> =
            Arc::new(AtomicBool::new(false));

        // ── WS Client ──────────────────────────────────────────
        let (ws_tx, mut ws_rx) = mpsc::channel(ws_queue_cap);
        let ws_tx_clone = ws_tx.clone();
        let ac_ws = active_coins.clone();
        let wr_ws = ws_reconnect.clone();
        let mut client = client;
        tokio::spawn(async move {
            let _ = client
                .run_forever(ac_ws, wr_ws, depth, ohlc_interval, ws_tx)
                .await;
        });

        // ── AI Bridge ──────────────────────────────────────────
        let mut ai_bridge = ai_bridge::AiBridge::new(&trading_config.env);
        if let Err(e) = ai_bridge.spawn().await {
            tracing::error!("[MAIN] AI bridge failed to start: {e}");
        }

        // ── Kraken REST API (always init — needed for auto-live toggle)
        let api = match kraken_api::from_env() {
            Ok(a) => {
                if paper_trading {
                    tracing::info!("[MAIN] Paper trading mode (API ready for auto-live)");
                } else {
                    tracing::info!("[MAIN] Kraken API ready — LIVE TRADING");
                }
                Some(a)
            }
            Err(e) => {
                if paper_trading {
                    tracing::warn!("[MAIN] Paper trading mode (API init failed: {e})");
                    None
                } else {
                    tracing::error!("[MAIN] Kraken API failed: {e}");
                    return Err(anyhow::anyhow!(
                        "Live trading requested but Kraken API init failed"
                    ));
                }
            }
        };

        // ── Private WebSocket — order fill notifications ────────────
        if let Some(ref api) = api {
            match api.get_ws_token().await {
                Ok(token) => {
                    let ws_tx_priv = ws_tx_clone.clone();
                    tokio::spawn(async move {
                        ws::PrivateWsClient::run_forever(token, ws_tx_priv).await;
                    });
                    tracing::info!("[MAIN] Private WS started — order fill notifications active");
                }
                Err(e) => {
                    tracing::warn!("[MAIN] Private WS token failed: {e} — order fills via REST only");
                }
            }
        }

        // ── SPY Kill Switch DISABLED (single-brain mode) ──
        // No background Yahoo Finance polling — cleaner, no watchdog noise
        tracing::info!("[MAIN] SPY feed disabled — no market kill switch");

        // ── Strategy ───────────────────────────────────────────
        let mut strategy = strategy::Strategy::load("data/strategy.toml", "data/nemo_prompt.txt")?;

        // ── Telemetry Event Bus ────────────────────────────────
        let telemetry_bus = event_bus::make_bus();
        let ws_port: u16 = env::var("TELEMETRY_WS_PORT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8765);
        let bus_clone = telemetry_bus.clone();
        tokio::spawn(async move {
            event_bus::run_ws_server(format!("127.0.0.1:{ws_port}"), bus_clone).await;
        });

        // ── Dashboard + Health Endpoint ──────────────────────────
        if let Some(std_listener) = health_listener {
            if let Ok(listener) = tokio::net::TcpListener::from_std(std_listener) {
                tokio::spawn(async move {
                    use axum::http::{header, Method};
                    use axum::response::{Html, IntoResponse};

                    // CORS layer — localhost dashboard only, not open to any origin
                    let cors = tower_http::cors::CorsLayer::new()
                        .allow_origin([
                            "http://127.0.0.1".parse::<axum::http::HeaderValue>().unwrap(),
                            "http://localhost".parse::<axum::http::HeaderValue>().unwrap(),
                        ])
                        .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
                        .allow_headers([header::CONTENT_TYPE]);

                    let app = axum::Router::new()
                        .route(
                            "/health",
                            axum::routing::get(|| async {
                                let hb = "data/heartbeat_trader.json";
                                match std::fs::read_to_string(hb) {
                                    Ok(content) => {
                                        let mut v: serde_json::Value =
                                            serde_json::from_str(&content)
                                                .unwrap_or(json!({}));
                                        if let Some(obj) = v.as_object_mut() {
                                            let ts = obj
                                                .get("ts")
                                                .and_then(|v| v.as_f64())
                                                .unwrap_or(0.0);
                                            let now = std::time::SystemTime::now()
                                                .duration_since(std::time::UNIX_EPOCH)
                                                .map(|d| d.as_secs_f64())
                                                .unwrap_or(0.0);
                                            let age = now - ts;
                                            obj.insert(
                                                "status".into(),
                                                if age < 30.0 {
                                                    json!("ok")
                                                } else {
                                                    json!("stale")
                                                },
                                            );
                                            obj.insert(
                                                "heartbeat_age_sec".into(),
                                                json!(age as u64),
                                            );
                                        }
                                        axum::Json(v).into_response()
                                    }
                                    Err(_) => {
                                        axum::Json(json!({"status": "no_heartbeat"}))
                                            .into_response()
                                    }
                                }
                            }),
                        )
                        .route(
                            "/",
                            axum::routing::get(|| async {
                                match std::fs::read_to_string("dashboard.html") {
                                    Ok(html) => Html(html).into_response(),
                                    Err(_) => Html(
                                        "<h2>dashboard.html not found</h2><p>Place dashboard.html in the project root.</p>"
                                            .to_string(),
                                    )
                                    .into_response(),
                                }
                            }),
                        )
                        .layer(cors);

                    tracing::info!(
                        "[DASHBOARD] http://127.0.0.1:{} (health + dashboard)",
                        health_port
                    );
                    let _ = axum::serve(listener, app).await;
                });
            }
        }

        // ── Nemo Chat Server (optional) ─────────────────────────
        #[cfg(feature = "nemo_chat")]
        let chat_state = nemo_chat::new_shared_state();
        #[cfg(feature = "nemo_chat")]
        nemo_chat::spawn_chat_server(chat_state.clone());

        // ── Funding Rate Intelligence ─────────────────────────
        let shared_funding = market_brain::new_shared_funding();
        market_brain::spawn_funding_fetcher(shared_funding.clone());

        // ── Vision Scanner (Gemma 3 12B — screen capture + coin rotation) ─
        let vision_interval: u64 = addons_config.rotation_interval_sec;
        let max_tracked: usize = env::var("MAX_TRACKED_COINS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(10);
        let anchor_coins: Vec<String> = env::var("ANCHOR_COINS")
            .unwrap_or_else(|_| "BTC,ETH".into())
            .split(',')
            .map(|s| s.trim().to_uppercase())
            .filter(|s| !s.is_empty())
            .collect();
        // ── Shared Swap Queue (Atlas + exit watchdog → vision scanner → WebSocket) ──
        let swap_queue: vision_scanner::SharedSwapQueue =
            std::sync::Arc::new(tokio::sync::RwLock::new(Vec::new()));

        // ── Cloud Intel (headline sentiment via Atlas ONNX server) ──
        let cloud_intel_enabled = addons_config.cloud_intel_enabled;
        let cloud_intel_state = cloud_intel::new_shared_intel();
        if cloud_intel_enabled {
            let ci_state = cloud_intel_state.clone();
            let ci_coins = active_coins.clone();
            let ci_swap_queue = swap_queue.clone();
            tokio::spawn(async move {
                cloud_intel::run_cloud_intel(ci_state, ci_coins, ci_swap_queue).await;
            });
            tracing::info!("[MAIN] Cloud intel enabled (AI_3 ONNX on port 8083 — discovers coins → notifies Qwen)");
        } else {
            tracing::info!("[MAIN] Cloud intel disabled (set CLOUD_INTEL_ENABLED=1 to enable)");
        }

        // ── Qwen Coin Rotation Scanner (reads swap queue from exit watchdog → tool-calls) ──
        // Works independently of cloud intel — exit watchdog feeds swap suggestions,
        // cloud intel is optional bonus data if enabled.
        let vision_enabled = addons_config.rotation_enabled;
        let anchor_coins_mover = anchor_coins.clone(); // clone before vision scanner moves it
        if vision_enabled {
            let ac_vision = active_coins.clone();
            let wr_vision = ws_reconnect.clone();
            let ci_scanner = cloud_intel_state.clone();
            let sq_vision = swap_queue.clone();
            tokio::spawn(async move {
                vision_scanner::run_vision_scanner(
                    ac_vision,
                    wr_vision,
                    ci_scanner,
                    vision_interval,
                    max_tracked,
                    anchor_coins,
                    Some(sq_vision),
                )
                .await;
            });
            tracing::info!(
                "[MAIN] Qwen rotation scanner enabled (interval={}s, max_coins={}, cloud_intel={})",
                vision_interval,
                max_tracked,
                if cloud_intel_enabled { "yes" } else { "no (swap queue only)" }
            );
        } else {
            tracing::info!("[MAIN] Rotation scanner disabled (set VISION_SCANNER_ENABLED=1)");
        }

        // ── Mover Scanner (full Kraken universe → swap suggestions) ──
        let mover_scanner_enabled = addons_config.movers_enabled;
        if mover_scanner_enabled {
            let ac_mover = active_coins.clone();
            let sq_mover = swap_queue.clone();
            let anchors_mover = anchor_coins_mover.clone();
            let wr_mover = ws_reconnect.clone();
            tokio::spawn(async move {
                mover_scanner::run_mover_scanner(ac_mover, sq_mover, anchors_mover, wr_mover).await;
            });
            tracing::info!("[MAIN] Mover scanner enabled — scanning full Kraken universe");
        } else {
            tracing::info!("[MAIN] Mover scanner disabled (set MOVER_SCANNER_ENABLED=1)");
        }

        // ── Trading Loop ───────────────────────────────────────
        let snapshot_path = trading_config.snapshot_path.clone();
        let mut trading = trading_loop::TradingLoop::new(
            trading_config,
            strategy.get_arc(),
            telemetry_bus,
            shared_funding.clone(),
            None,
        );
        trading.memory_enabled = addons_config.memory_enabled;
        if cloud_intel_enabled {
            trading.cloud_intel = Some(cloud_intel_state.clone());
        }
        trading.swap_queue = Some(swap_queue.clone());

        // ── GPU Math (native CUDA via cudarc) ──
        {
            let gm = trading.gpu_math.read().await;
            if gm.is_healthy() {
                tracing::info!("[MAIN] GPU math engine initialized — batch math offloaded to CUDA");
            } else {
                tracing::info!("[MAIN] GPU math unavailable — using Rust CPU fallback");
            }
        }

        // ── GBDT (Gradient Boosted Decision Trees) ──
        {
            let gbdt = trading.gbdt.read().await;
            tracing::info!(
                "[MAIN] GBDT engine: {} samples, blend {:.0}%, model {}",
                gbdt.sample_count(),
                gbdt.blend_weight() * 100.0,
                if gbdt.has_model() { "loaded" } else { "none (cold start)" }
            );
            if gbdt.has_model() {
                let top = gbdt.top_importances(5);
                let top_str: Vec<String> = top
                    .iter()
                    .map(|(n, v)| format!("{n}:{v:.2}"))
                    .collect();
                tracing::info!("[GBDT] top importances: {}", top_str.join(", "));
            }
        }

        // ── Config Snapshot (one-time) ──
        {
            let env = &trading.config.env;
            let symbols = env.get_str("SYMBOLS", "");
            let entry_host = env.get_str("OLLAMA_HOST", "http://127.0.0.1:8081");
            let entry_model = env.get_str("OLLAMA_MODEL", "unknown");
            let exit_host = env.get_str("EXIT_AI_HOST", "http://127.0.0.1:8082");
            let exit_model = env.get_str("AI_EXIT_MODEL", entry_model.as_str());
            let npu_enabled = env.get_bool("NPU_ENABLED", false);
            let npu_device = env.get_str("NPU_DEVICE", "NPU");
            let npu_model = env.get_str("NPU_MODEL_PATH", "");
            let mode = env.get_str("MODE", "paper");
            let max_pos = env.get_parsed::<usize>("MAX_POSITIONS", 0);
            let max_cands = env.get_parsed::<usize>("AI_MAX_CANDIDATES_PER_TICK", 0);
            let poll = env.get_parsed::<u64>("POLL_SECONDS", 10);
            let ws_depth = env.get_parsed::<u64>("WS_DEPTH", 10);
            let book_depth = env.get_parsed::<u64>("BOOK_DEPTH", 10);
            let think_mode = env.get_bool("THINK_MODE", false);
            let quant_primary = env.get_bool("QUANT_PRIMARY", false);
            let log_json = env.get_bool("LOG_JSON", false);
            tracing::info!(
                "[CONFIG] mode={} symbols=[{}]",
                mode, symbols
            );
            tracing::info!(
                "[CONFIG] entry_model={} host={} think_mode={}",
                entry_model, entry_host, think_mode
            );
            tracing::info!(
                "[CONFIG] exit_model={} host={}",
                exit_model, exit_host
            );
            tracing::info!(
                "[CONFIG] npu_enabled={} device={} model_path={}",
                npu_enabled, npu_device, npu_model
            );
            tracing::info!(
                "[CONFIG] max_positions={} max_candidates_per_tick={} poll_seconds={} ws_depth={} book_depth={}",
                max_pos, max_cands, poll, ws_depth, book_depth
            );
            tracing::info!(
                "[CONFIG] quant_primary={} log_json={}",
                quant_primary, log_json
            );
        }

        let mut last_trade_tick: f64 = 0.0;

        // ── Event-Driven Exit Debounce ──────────────────────────
        let mut exit_debounce: HashMap<String, tokio::time::Instant> = HashMap::new();

        // ── Per-Symbol State ───────────────────────────────────
        let mut store = book::BookStore::new();
        let mut sym_states: HashMap<String, SymbolState> = HashMap::new();
        let mut tick: u64 = 0;
        let mut ref_closes_map: HashMap<String, VecDeque<f64>> = HashMap::new();

        tracing::info!("[MAIN] Entering main WS receive loop");

        // ── Main Loop: WS → Features → Trading ────────────────
        let mut shutdown = std::pin::pin!(tokio::signal::ctrl_c());
        loop {
            let ev = tokio::select! {
                biased;
                result = &mut shutdown => {
                    if result.is_ok() {
                        tracing::info!("[MAIN] Ctrl+C — graceful shutdown initiated");
                    }
                    break;
                }
                ev = ws_rx.recv() => {
                    match ev {
                        Some(e) => e,
                        None => break,
                    }
                }
            };
            match ev {
                ws::WsEvent::BookSnapshot(v) => {
                    if let Some(data) = v.get("data").and_then(|d| d.as_array()) {
                        for item in data {
                            if let Some(sym) = extract_symbol(item) {
                                let bids = parse_book_levels(item, "bids", true);
                                let asks = parse_book_levels(item, "asks", true);
                                store.apply_snapshot(&sym, &bids, &asks);
                            }
                        }
                    }
                }
                ws::WsEvent::BookUpdate(v) => {
                    if let Some(data) = v.get("data").and_then(|d| d.as_array()) {
                        for item in data {
                            if let Some(sym) = extract_symbol(item) {
                                let bids = parse_book_levels(item, "bids", false);
                                let asks = parse_book_levels(item, "asks", false);
                                store.apply_update(&sym, &bids, &asks);
                            }
                        }
                    }
                }
                ws::WsEvent::Ohlc(v) => {
                    if let Some(data) = v.get("data").and_then(|d| d.as_array()) {
                        for item in data {
                            if let Some(sym) = extract_symbol(item) {
                                let close = parse_f64(item.get("close").unwrap_or(&json!(0)));
                                let high = parse_f64(item.get("high").unwrap_or(&json!(0)));
                                let low = parse_f64(item.get("low").unwrap_or(&json!(0)));
                                let volume = parse_f64(item.get("volume").unwrap_or(&json!(0)));
                                let interval_begin = item
                                    .get("interval_begin")
                                    .or_else(|| item.get("timestamp"))
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("")
                                    .to_string();

                                let state = sym_states
                                    .entry(sym.clone())
                                    .or_insert_with(|| SymbolState::new(trade_window_sec));
                                let candle = features::Candle { close, high, low, volume };

                                if !interval_begin.is_empty()
                                    && interval_begin == state.ohlc_last_interval
                                {
                                    // Same interval — update in-place
                                    if let Some(last) = state.candles.back_mut() {
                                        *last = candle;
                                    } else {
                                        state.candles.push_back(candle);
                                    }
                                    if let Some(last) = state.ref_closes.back_mut() {
                                        *last = close;
                                    } else {
                                        state.ref_closes.push_back(close);
                                    }
                                } else {
                                    // New interval
                                    state.ohlc_last_interval = interval_begin;
                                    state.candles.push_back(candle);
                                    if state.candles.len() > 500 {
                                        state.candles.pop_front();
                                    }
                                    state.ref_closes.push_back(close);
                                    if state.ref_closes.len() > 500 {
                                        state.ref_closes.pop_front();
                                    }
                                }

                                // Keep ref_closes_map in sync for cross-correlation
                                ref_closes_map.insert(sym, state.ref_closes.clone());
                            }
                        }
                    }
                }
                ws::WsEvent::Trade(v) => {
                    if let Some(data) = v.get("data").and_then(|d| d.as_array()) {
                        for item in data {
                            if let Some(sym) = extract_symbol(item) {
                                let side = item
                                    .get("side")
                                    .and_then(|s| s.as_str())
                                    .unwrap_or("buy");
                                let ts = item
                                    .get("timestamp")
                                    .and_then(|t| t.as_f64())
                                    .unwrap_or_else(now_ts);
                                let volume = parse_f64(
                                    item.get("qty").unwrap_or(&json!(0)),
                                );
                                if volume <= 0.0 {
                                    continue;
                                }

                                // ── Event-Driven Exit: check SL/TP on every trade ──
                                let trade_price = parse_f64(
                                    item.get("price").unwrap_or(&json!(0)),
                                );
                                if trade_price > 0.0 && trading.positions.contains_key(&sym) {
                                    let now_inst = tokio::time::Instant::now();
                                    let debounce_ok = exit_debounce
                                        .get(&sym)
                                        .map(|last| now_inst.duration_since(*last).as_millis() > 750)
                                        .unwrap_or(true);
                                    if debounce_ok {
                                        exit_debounce.insert(sym.clone(), now_inst);
                                        if let Some(reason) = trading.check_instant_exit(&sym, trade_price) {
                                            tracing::warn!(
                                                "[INSTANT-EXIT] {} @ {:.6} — {}",
                                                sym, trade_price, reason
                                            );
                                            trading
                                                .execute_exit(&sym, trade_price, &reason, api.as_ref(), Some(&ai_bridge))
                                                .await;
                                        }
                                    }
                                }

                                let state = sym_states
                                    .entry(sym)
                                    .or_insert_with(|| SymbolState::new(trade_window_sec));
                                state.trade_flow.update(Trade {
                                    ts,
                                    side: side
                                        .parse::<TradeSide>()
                                        .unwrap_or(TradeSide::Sell),
                                    volume,
                                });
                            }
                        }
                    }
                }
                ws::WsEvent::Ticker(v) => {
                    if let Some(data) = v.get("data").and_then(|d| d.as_array()) {
                        for item in data {
                            if let Some(sym) = extract_symbol(item) {
                                let f = |key| parse_f64(item.get(key).unwrap_or(&json!(0)));
                                let last_price = f("last");
                                if last_price <= 0.0 { continue; }
                                let ticker = cloud_intel::MarketTicker {
                                    symbol: sym,
                                    last_price,
                                    change_pct: f("change_pct"),
                                    volume_24h:  f("volume"),
                                    high_24h:    f("high"),
                                    low_24h:     f("low"),
                                    vwap:        f("vwap"),
                                    bid:         f("bid"),
                                    ask:         f("ask"),
                                    updated_at: now_ts(),
                                };
                                cloud_intel::update_market_ticker(&cloud_intel_state, ticker).await;
                            }
                        }
                    }
                }
                ws::WsEvent::OrderFill(exec) => {
                    let sym_raw = exec.get("symbol").and_then(|s| s.as_str()).unwrap_or("?");
                    // Strip /USD suffix → engine symbol (e.g. "BTC/USD" → "BTC")
                    let sym = sym_raw.split('/').next().unwrap_or(sym_raw).to_uppercase();
                    let qty    = exec.get("last_qty").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    let price  = exec.get("last_price").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    let status = exec.get("order_status").and_then(|v| v.as_str()).unwrap_or("?");
                    let order_id = exec.get("order_id").and_then(|v| v.as_str()).unwrap_or("?");
                    tracing::info!(
                        "[ORDER-FILL] {} filled qty={:.6} @ {:.4} status={} id={}",
                        sym, qty, price, status, order_id
                    );
                    // Notify engine to reconcile fill price with open position
                    trading.on_order_fill(&sym, price, qty, status);
                }
            }

            tick += 1;

            // ── Periodic Feature Computation + Trading Tick ─────
            let now_t = now_ts();
            if (now_t - last_trade_tick) >= poll_seconds {
                let mut features_map: HashMap<String, serde_json::Value> = HashMap::new();
                let mut closes_map: HashMap<String, Vec<f64>> = HashMap::new();
                for (sym, state) in sym_states.iter() {
                    if !state.candles.is_empty() {
                        let closes: Vec<f64> = state.candles.iter().map(|c| c.close).collect();
                        closes_map.insert(sym.clone(), closes);
                    }
                }

                // ── GPU batch quant features (Hurst, entropy, autocorr) ──
                let log_gpu_timing = trading.config.env.get_bool("LOG_GPU_TIMING", false);
                let gpu_t0 = Instant::now();
                let gpu_health = {
                    let mut gm = trading.gpu_math.write().await;
                    gm.check_health().await
                };
                let gpu_quant: Option<HashMap<String, (f64, f64, f64)>> = if gpu_health && !closes_map.is_empty() {
                    let gm = trading.gpu_math.read().await;
                    gm.batch_features(&closes_map).await
                } else {
                    None
                };
                let gpu_corr_btc: Option<HashMap<String, f64>> = if gpu_health && closes_map.len() >= 2 {
                    let gm = trading.gpu_math.read().await;
                    gm.batch_correlation(&closes_map).await.map(|(symbols, corr)| {
                        let mut out = HashMap::new();
                        let btc_idx = symbols.iter().position(|s| s == "BTC" || s == "XBT");
                        if let Some(bi) = btc_idx {
                            for (i, sym) in symbols.iter().enumerate() {
                                if let Some(row) = corr.get(i) {
                                    let c = row.get(bi).copied().unwrap_or(0.0);
                                    out.insert(sym.clone(), c);
                                }
                            }
                        }
                        out
                    })
                } else {
                    None
                };
                let gpu_fingerprint = if gpu_health && closes_map.len() >= 2 {
                    let gm = trading.gpu_math.read().await;
                    gm.batch_fingerprint(&closes_map, "BTC", "ETH").await
                } else {
                    None
                };

                // ── Portfolio weights: GPU covariance → mean-variance optimizer ──
                if gpu_health && closes_map.len() >= 4 {
                    let gm = trading.gpu_math.read().await;
                    if let Some((symbols, cov_result)) = gm.batch_covariance(&closes_map).await {
                        let n = symbols.len();
                        if let Some(weights) = trading.optimizer.compute_weights(
                            &symbols,
                            &cov_result.cov,
                            &cov_result.mean_returns,
                            1e-4,        // lambda: ridge regularization
                            0.0,         // min weight
                            3.0 / n as f64, // max weight: no single coin > 3x average
                        ) {
                            tracing::debug!("[PORTFOLIO] {} coin weights computed", weights.len());
                            trading.optimizer.portfolio_weights = Some(weights);
                        }
                    }
                }
                let gpu_ms = gpu_t0.elapsed().as_millis() as u64;

                // ── Publish GPU quant results to telemetry bus ────────
                if trading.telemetry_level >= crate::event_bus::TelemetryLevel::Snapshot
                    && (gpu_quant.is_some() || gpu_fingerprint.is_some())
                {
                    let coins: Vec<crate::event_bus::GpuCoinQuant> = gpu_quant
                        .as_ref()
                        .map(|qmap| {
                            qmap.iter()
                                .map(|(sym, (h, e, a))| crate::event_bus::GpuCoinQuant {
                                    symbol: sym.clone(),
                                    hurst: *h,
                                    entropy: *e,
                                    autocorr: *a,
                                    corr_btc: gpu_corr_btc
                                        .as_ref()
                                        .and_then(|m| m.get(sym.as_str()).copied()),
                                })
                                .collect()
                        })
                        .unwrap_or_default();
                    let fingerprint = gpu_fingerprint.as_ref().map(|fp| crate::event_bus::GpuFingerprint {
                        r_mkt:    fp.metrics.get(0).copied().unwrap_or(0.0),
                        r_btc:    fp.metrics.get(1).copied().unwrap_or(0.0),
                        r_eth:    fp.metrics.get(2).copied().unwrap_or(0.0),
                        breadth:  fp.metrics.get(3).copied().unwrap_or(0.0),
                        median:   fp.metrics.get(4).copied().unwrap_or(0.0),
                        iqr:      fp.metrics.get(5).copied().unwrap_or(0.0),
                        rv_mkt:   fp.metrics.get(6).copied().unwrap_or(0.0),
                        corr_avg: fp.metrics.get(7).copied().unwrap_or(0.0),
                    });
                    crate::event_bus::publish(&trading.bus, crate::event_bus::Event::GpuQuant {
                        ts: now_t,
                        coins,
                        fingerprint,
                        compute_ms: gpu_ms,
                    });
                }


                let mut cpu_feat_ms_total: u64 = 0;
                let mut cpu_feat_count: usize = 0;
                let mut gpu_override_count: usize = 0;
                for (sym, state) in &mut sym_states {
                    if state.candles.is_empty() {
                        continue;
                    }
                    let candles_slice = state.candles.make_contiguous();

                    // Orderbook metrics
                    let mut orderbook = store
                        .get_top(sym, book_depth)
                        .map(|b| book_metrics(&b));

                    // Trade flow
                    let mut trade_flow = Some(state.trade_flow.get_flow());

                    // Optional denoise (EMA smoothing) to reduce market noise
                    let alpha = trading.config.env.get_f64("DENOISE_ALPHA", 0.0).clamp(0.0, 1.0);
                    if alpha > 0.0 {
                        if !state.smoothed_inited {
                            if let Some(ref ob) = orderbook {
                                state.smoothed_book_imb = ob.imbalance;
                                state.smoothed_spread_pct = ob.spread_pct;
                            }
                            if let Some(ref tf) = trade_flow {
                                state.smoothed_buy_ratio = tf.buy_ratio;
                                state.smoothed_sell_ratio = tf.sell_ratio;
                            }
                            state.smoothed_inited = true;
                        }

                        if let Some(ref ob) = orderbook {
                            state.smoothed_book_imb = alpha * ob.imbalance + (1.0 - alpha) * state.smoothed_book_imb;
                            state.smoothed_spread_pct = alpha * ob.spread_pct + (1.0 - alpha) * state.smoothed_spread_pct;
                            orderbook = Some(features::OrderbookMetrics {
                                spread_pct: state.smoothed_spread_pct,
                                imbalance: state.smoothed_book_imb,
                            });
                        }

                        if let Some(ref tf) = trade_flow {
                            state.smoothed_buy_ratio = alpha * tf.buy_ratio + (1.0 - alpha) * state.smoothed_buy_ratio;
                            state.smoothed_sell_ratio = alpha * tf.sell_ratio + (1.0 - alpha) * state.smoothed_sell_ratio;
                            let total = (state.smoothed_buy_ratio + state.smoothed_sell_ratio).max(1e-9);
                            trade_flow = Some(features::TradeFlow {
                                buy_ratio: (state.smoothed_buy_ratio / total).clamp(0.0, 1.0),
                                sell_ratio: (state.smoothed_sell_ratio / total).clamp(0.0, 1.0),
                            });
                        }
                    }

                    // Book reversal
                    let book_rev = if let Some(ob) = orderbook.as_ref() {
                        Some(state.book_reversal.update(ob.imbalance))
                    } else {
                        None
                    };

                    // Compute lightweight features (reuses pre-allocated buffers)
                    let cpu_t0 = Instant::now();
                    let mut feats = features::compute_features_buffered(
                        candles_slice,
                        orderbook.as_ref(),
                        trade_flow.as_ref(),
                        book_rev.as_ref(),
                        Some(&ref_closes_map),
                        &mut state.feature_bufs,
                    );
                    cpu_feat_ms_total += cpu_t0.elapsed().as_millis() as u64;
                    cpu_feat_count += 1;

                    // Override with GPU-computed quant features if available
                    if let Some(ref gq) = gpu_quant {
                        if let Some(&(h, e, a)) = gq.get(sym.as_str()) {
                            feats.hurst_exp = h;
                            feats.shannon_entropy = e;
                            feats.autocorr_lag1 = a;
                            gpu_override_count += 1;
                        }
                    }
                    if let Some(ref corr_btc) = gpu_corr_btc {
                        if let Some(c) = corr_btc.get(sym.as_str()) {
                            feats.quant_corr_btc = *c;
                        }
                    }

                    let scores = features::compute_action_scores(&feats);

                    // GBDT prediction (gradient boosted decision trees)
                    let fvec = signals::gbdt::features_to_vec(&feats);
                    let (gbdt_score, gbdt_blend) = {
                        let gbdt = trading.gbdt.read().await;
                        let p = gbdt.predict(&fvec);
                        let bw = gbdt.blend_weight();
                        let blended = if bw > 0.0 {
                            bw * p + (1.0 - bw) * scores.confidence
                        } else {
                            scores.confidence
                        };
                        (p, blended)
                    };

                    let quantum_state = if feats.quant_vol.abs() > 0.015 || feats.shannon_entropy > 0.85 {
                        "volatile"
                    } else if feats.hurst_exp < 0.45 && feats.shannon_entropy > 0.75 {
                        "sideways"
                    } else if feats.trend_score >= 2 || feats.momentum_score > 0.2 {
                        "trending"
                    } else {
                        "sideways"
                    };
                    let mut fval = feats_to_value(&feats);
                    // Merge action scores into the feature JSON
                    if let serde_json::Value::Object(ref mut map) = fval {
                        map.insert("s_buy".into(), json!(scores.s_buy));
                        map.insert("s_sell".into(), json!(scores.s_sell));
                        map.insert("s_hold".into(), json!(scores.s_hold));
                        map.insert("math_action".into(), json!(scores.action));
                        map.insert("math_confidence".into(), json!(scores.confidence));
                        map.insert("crash".into(), json!(scores.crash));
                        map.insert("hunt_signal".into(), json!(scores.hunt));
                        map.insert("buy_pressure".into(), json!(scores.buy_pressure));
                        map.insert("sell_pressure".into(), json!(scores.sell_pressure));
                        map.insert("trend_buy".into(), json!(scores.trend_buy));
                        map.insert("momentum_buy".into(), json!(scores.momentum_buy));
                        map.insert("mean_rev_buy".into(), json!(scores.mean_rev_buy));
                        map.insert("mean_rev_sell".into(), json!(scores.mean_rev_sell));
                        map.insert("is_l2".into(), json!(scores.is_l2));
                        // Pump detector signals
                        map.insert("pump_early".into(), json!(scores.pump_early));
                        map.insert("pump_confirmed".into(), json!(scores.pump_confirmed));
                        map.insert("pump_score".into(), json!(scores.pump_score));
                        map.insert("dynamic_threshold".into(), json!(scores.dynamic_threshold));
                        if !scores.reasons.is_empty() {
                            map.insert("missing_indicators".into(), json!(scores.reasons));
                        }
                        map.insert("quantum_state".into(), json!(quantum_state));
                        if let Some(ref corr_btc) = gpu_corr_btc {
                            if let Some(c) = corr_btc.get(sym.as_str()) {
                                map.insert("quant_corr_btc".into(), json!(c));
                            }
                        }
                        // GBDT scores
                        map.insert("gbdt_score".into(), json!(gbdt_score));
                        map.insert("gbdt_blend".into(), json!(gbdt_blend));
                        map.insert("feature_snapshot".into(), json!(fvec));
                    }
                    features_map.insert(sym.clone(), fval);
                }

                if !features_map.is_empty() {
                    if let Some(fp) = gpu_fingerprint {
                        features_map.insert("__gpu_fingerprint".to_string(), json!(fp.metrics));
                    }
                    if log_gpu_timing {
                        tracing::info!(
                            "[TIMING] gpu_batch={}ms cpu_features_total={}ms coins={} gpu_override={}",
                            gpu_ms,
                            cpu_feat_ms_total,
                            cpu_feat_count,
                            gpu_override_count
                        );
                    }
                    // Hot-reload strategy
                    strategy.hot_reload();
                    trading.strategy = strategy.get_arc();

                    let tick_start = std::time::Instant::now();
                    trading
                        .tick(&features_map, api.as_ref(), &ai_bridge)
                        .await;
                    let tick_ms = tick_start.elapsed().as_millis();
                    if tick_ms > 500 {
                        tracing::info!("[TICK] completed in {tick_ms}ms ({} coins)", features_map.len());
                    }
                    last_trade_tick = now_t;

                    // ── Auto-track any held coins missing from WS feed ──────────────
                    // If we hold a coin on Kraken that isn't in active_coins, we can't
                    // get price data for it, can't adopt it, and can't set stops on it.
                    // Fix: add it to active_coins immediately and reconnect WS.
                    {
                        let held: Vec<String> = trading.holdings.keys()
                            .filter(|s| *s != "USD" && !s.is_empty())
                            .cloned()
                            .collect();
                        if !held.is_empty() {
                            let mut ac = active_coins.write().await;
                            let mut added_syms: Vec<String> = Vec::new();
                            for sym in &held {
                                if !ac.contains(sym) {
                                    tracing::warn!(
                                        "[AUTO-TRACK] {} held on Kraken but not in WS feed — adding now",
                                        sym
                                    );
                                    ac.push(sym.clone());
                                    added_syms.push(sym.clone());
                                }
                            }
                            drop(ac);
                            if !added_syms.is_empty() {
                                ws_reconnect.store(true, std::sync::atomic::Ordering::SeqCst);
                            }
                        }
                    }

                    // ── Write features snapshot to disk (every tick) ──
                    if tick % 3 == 0 {
                        let snapshot = json!({
                            "ts": now_t,
                            "features": &features_map,
                        });
                        if let Ok(json_str) = serde_json::to_string(&snapshot) {
                            let _ = std::fs::write(&snapshot_path, json_str);
                        }
                    }

                    // Update Nemo chat state (optional)
                    #[cfg(feature = "nemo_chat")]
                    {
                        let mut cs = chat_state.write().await;
                        cs.features = features_map.clone();
                        cs.available_usd = trading.available_usd;
                        cs.equity = trading.equity;
                        cs.total_value = trading.available_usd;
                        cs.positions.clear();
                        for p in trading.positions.values() {
                            let cur_price = features_map
                                .get(&p.symbol)
                                .and_then(|v| v.get("price"))
                                .and_then(|v| v.as_f64())
                                .unwrap_or(p.entry_price);
                            let value = p.remaining_qty * cur_price;
                            cs.total_value += value;
                            cs.positions.push(json!({
                                "symbol": p.symbol,
                                "qty": p.remaining_qty,
                                "entry": p.entry_price,
                                "now": cur_price,
                                "value": value,
                                "pnl_pct": p.pnl_pct_raw(cur_price) * 100.0,
                            }));
                        }
                    }

                    if tick % 100 == 0 {
                        tracing::info!(
                            "[MAIN] tick={} symbols={} features_ready={}",
                            tick,
                            sym_states.len(),
                            features_map.len()
                        );
                    }
                }
            }
        }

        // ── Graceful Shutdown ────────────────────────────────────
        tracing::info!("[MAIN] Shutting down — saving state...");
        trading_loop::save_positions(
            &trading.config.positions_file,
            &trading.positions,
        );
        tracing::info!(
            "[MAIN] {} positions saved. Clean shutdown.",
            trading.positions.len()
        );
        Ok::<(), anyhow::Error>(())
    })?;

    let _ = std::fs::remove_file(&lock_path);
    tracing::info!("[MAIN] Clean exit — lock removed");
    Ok(())
}
