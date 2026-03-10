// ── Infrastructure ──────────────────────────────────────────────────
pub mod infra;
// Backward-compatible re-exports (consumers use crate::book, crate::error, etc.)
pub use infra::{book, error, ws, kraken_api, event_bus};

// ── Config ─────────────────────────────────────────────────────────
pub mod config;

// ── Signals ────────────────────────────────────────────────────────
pub mod signals;
pub use signals::{features, strategy, strategy_helpers, market_brain, market_watcher};

// ── LLM layer (AI client, prompts, parsers) ──────────────────────
pub mod llm;
/// Backward-compatible alias — `crate::ai_bridge::X` still works everywhere.
pub use llm as ai_bridge;
// Backward-compatible re-exports for modules moved into llm/
pub use llm::{nemo_exit, nvidia_tools};

// ── Engine (trading loop, orders, risk) ──────────────────────────
pub mod engine;
/// Backward-compatible alias — `crate::trading_loop::X` still works everywhere.
pub use engine as trading_loop;
// Backward-compatible re-exports for modules moved into engine/
pub use engine::{journal, trade_flow, rules_eval, portfolio_optimizer, portfolio_allocator};

// ── Addons (cloud intel, scanners, memory, NPU, backtester) ─────
pub mod addons;
// Backward-compatible re-exports for modules moved into addons/
pub use addons::{
    cloud_intel, news_sentiment, vision_scanner, mover_scanner,
    memory_packet, nemo_memory, npu_bridge, backtester, ohlcv_fetcher,
};

// ── Optional modules — compile-gated via Cargo features ────────────
#[cfg(feature = "nemo_chat")]
pub mod nemo_chat;
#[cfg(feature = "nemo_optimizer")]
pub mod nemo_optimizer;
