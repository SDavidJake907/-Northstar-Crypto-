//! Tiered telemetry event bus with local WebSocket broadcast.
//!
//! Level 1 (Live):     decision.signal, risk.reject  — every cycle
//! Level 2 (Snapshot): features.snapshot              — on state change only
//! Level 3 (Debug):    echo.debug                     — opt-in flag

use futures_util::{SinkExt, StreamExt};
use serde::Serialize;
use tokio::net::TcpListener;
use tokio::sync::broadcast;
use tokio_tungstenite::{accept_async, tungstenite::Message};

// ── Event Types ─────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum Event {
    /// L1: Every AI decision cycle — action + confidence + reason
    #[serde(rename = "decision.signal")]
    DecisionSignal {
        ts: f64,
        symbol: String,
        action: String,
        confidence: f64,
        lane: String,
        bucket: String,
        reason: String,
    },

    /// L1: When gate checks reject a trade
    #[serde(rename = "risk.reject")]
    RiskReject {
        ts: f64,
        symbol: String,
        reason: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        spread_pct: Option<f64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        slip_pct: Option<f64>,
    },

    /// L2: Market watcher cross-coin intelligence alerts
    #[serde(rename = "watcher.alert")]
    WatcherAlert {
        ts: f64,
        signal: String,
        severity: String,
        text: String,
    },

    /// L2: Only on bucket/lane/regime change — no spam
    #[serde(rename = "features.snapshot")]
    FeaturesSnapshot {
        ts: f64,
        symbol: String,
        bucket: String,
        lane: String,
        score: f64,
        trend_score: i32,
        #[serde(skip_serializing_if = "Option::is_none")]
        caps: Option<ConfidenceCaps>,
    },

    /// L2: GPU batch quant results — per-coin Hurst/entropy/autocorr + market fingerprint
    #[serde(rename = "gpu.quant")]
    GpuQuant {
        ts: f64,
        coins: Vec<GpuCoinQuant>,
        #[serde(skip_serializing_if = "Option::is_none")]
        fingerprint: Option<GpuFingerprint>,
        compute_ms: u64,
    },
}

#[derive(Debug, Clone, Serialize)]
pub struct GpuCoinQuant {
    pub symbol: String,
    pub hurst: f64,
    pub entropy: f64,
    pub autocorr: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub corr_btc: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct GpuFingerprint {
    pub r_mkt: f64,
    pub r_btc: f64,
    pub r_eth: f64,
    pub breadth: f64,
    pub median: f64,
    pub iqr: f64,
    pub rv_mkt: f64,
    pub corr_avg: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ConfidenceCaps {
    pub bucket_penalty: f64,
    pub whip_cap: f64,
    pub echo_cap: f64,
}

// ── Telemetry Level ─────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TelemetryLevel {
    Live,      // L1: decision.signal + risk.reject
    Snapshot,  // L2: features.snapshot on change
    Debug,     // L3: echo.debug (opt-in)
}

impl TelemetryLevel {
    pub fn from_env() -> Self {
        match std::env::var("TELEMETRY_LEVEL")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str()
        {
            "debug" | "3" => Self::Debug,
            "snapshot" | "2" => Self::Snapshot,
            _ => Self::Live, // default: L1 only
        }
    }
}

// ── Bus Creation ────────────────────────────────────────────────

pub fn make_bus() -> broadcast::Sender<Event> {
    let (tx, _rx) = broadcast::channel::<Event>(4096);
    tx
}

/// Publish an event to the bus. Silently drops if no receivers.
pub fn publish(tx: &broadcast::Sender<Event>, ev: Event) {
    let _ = tx.send(ev);
}

// ── WebSocket Server ────────────────────────────────────────────

pub async fn run_ws_server(addr: String, tx: broadcast::Sender<Event>) {
    let listener = match TcpListener::bind(&addr).await {
        Ok(l) => {
            tracing::info!("[TELEMETRY] WS server on ws://{addr}");
            l
        }
        Err(e) => {
            tracing::error!("[TELEMETRY] Failed to bind {addr}: {e}");
            return;
        }
    };

    loop {
        let (stream, peer) = match listener.accept().await {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!("[TELEMETRY] accept error: {e}");
                continue;
            }
        };

        let mut rx = tx.subscribe();
        tracing::info!("[TELEMETRY] client connected: {peer}");

        tokio::spawn(async move {
            let ws = match accept_async(stream).await {
                Ok(ws) => ws,
                Err(e) => {
                    tracing::warn!("[TELEMETRY] WS handshake error from {peer}: {e}");
                    return;
                }
            };

            let (mut write, mut read) = ws.split();

            // Read task: handle close frames and pings
            let read_handle = tokio::spawn(async move {
                while let Some(msg) = read.next().await {
                    match msg {
                        Ok(m) if m.is_close() => break,
                        Err(_) => break,
                        _ => {} // ignore other client messages
                    }
                }
            });

            // Write task: forward broadcast events as JSON
            loop {
                match rx.recv().await {
                    Ok(ev) => {
                        let json = match serde_json::to_string(&ev) {
                            Ok(s) => s,
                            Err(_) => continue,
                        };
                        if write.send(Message::Text(json.into())).await.is_err() {
                            break; // client disconnected
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        // Client fell behind — notify and continue
                        let lag_msg = format!(
                            r#"{{"type":"telemetry.lag","dropped":{n}}}"#
                        );
                        let _ = write.send(Message::Text(lag_msg.into())).await;
                    }
                    Err(_) => break, // channel closed
                }
            }

            read_handle.abort();
            tracing::info!("[TELEMETRY] client disconnected: {peer}");
        });
    }
}
