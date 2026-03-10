use anyhow::Result;
use futures_util::{SinkExt, StreamExt};
use serde_json::json;
use serde_json::Value;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::{mpsc, RwLock};
use tokio::time::{timeout, Duration, Instant};
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::Message;

const KRAKEN_WS_V2_PUBLIC: &str = "wss://ws.kraken.com/v2";
const KRAKEN_WS_V2_PRIVATE: &str = "wss://ws-auth.kraken.com/v2";

fn pair_for_symbol(sym: &str) -> Option<&'static str> {
    match sym {
        "BTC" | "XBT" => Some("BTC/USD"),
        "ETH" => Some("ETH/USD"),
        "SOL" => Some("SOL/USD"),
        "AVAX" => Some("AVAX/USD"),
        "LINK" => Some("LINK/USD"),
        "DOT" => Some("DOT/USD"),
        "MATIC" => Some("MATIC/USD"),
        "ATOM" => Some("ATOM/USD"),
        "XRP" => Some("XRP/USD"),
        "ADA" => Some("ADA/USD"),
        "LTC" => Some("LTC/USD"),
        "BCH" => Some("BCH/USD"),
        "ETC" => Some("ETC/USD"),
        "NEAR" => Some("NEAR/USD"),
        "FTM" => Some("FTM/USD"),
        "ALGO" => Some("ALGO/USD"),
        "XLM" => Some("XLM/USD"),
        "APT" => Some("APT/USD"),
        "HBAR" => Some("HBAR/USD"),
        "FIL" => Some("FIL/USD"),
        "TRX" => Some("TRX/USD"),
        "AAVE" => Some("AAVE/USD"),
        "UNI" => Some("UNI/USD"),
        "MKR" => Some("MKR/USD"),
        "CRV" => Some("CRV/USD"),
        "LDO" => Some("LDO/USD"),
        "INJ" => Some("INJ/USD"),
        "RENDER" => Some("RENDER/USD"),
        "FET" => Some("FET/USD"),
        "GRT" => Some("GRT/USD"),
        "MANA" => Some("MANA/USD"),
        "SAND" => Some("SAND/USD"),
        "AXS" => Some("AXS/USD"),
        "ENJ" => Some("ENJ/USD"),
        "CHZ" => Some("CHZ/USD"),
        "IMX" => Some("IMX/USD"),
        "BLUR" => Some("BLUR/USD"),
        "DOGE" => Some("DOGE/USD"),
        "SHIB" => Some("SHIB/USD"),
        "PEPE" => Some("PEPE/USD"),
        "FLOKI" => Some("FLOKI/USD"),
        "BONK" => Some("BONK/USD"),
        "WIF" => Some("WIF/USD"),
        "ARB" => Some("ARB/USD"),
        "OP" => Some("OP/USD"),
        "SUI" => Some("SUI/USD"),
        "SEI" => Some("SEI/USD"),
        "TIA" => Some("TIA/USD"),
        "GALA" => Some("GALA/USD"),
        "PAXG" => Some("PAXG/USD"),
        "ZEUS" => Some("ZEUS/USD"),
        "SUP" => Some("SUP/USD"),
        "CQT" => Some("CQT/USD"),
        "KEY" => Some("KEY/USD"),
        "SKR" => Some("SKR/USD"),
        "UNFI" => Some("UNFI/USD"),
        "NPC" => Some("NPC/USD"),
        "SNEK" => Some("SNEK/USD"),
        "APR" => Some("APR/USD"),
        "VULT" => Some("VULT/USD"),
        "RIVER" => Some("RIVER/USD"),
        _ => None,
    }
}

fn normalize_symbols(symbols: Vec<String>) -> Vec<String> {
    let mut out = Vec::new();
    let mut fallback: Vec<String> = Vec::new();
    for s in symbols {
        let sym = s.trim().to_uppercase();
        if sym.is_empty() {
            continue;
        }
        if sym.contains('/') {
            out.push(sym);
            continue;
        }
        if let Some(pair) = pair_for_symbol(&sym) {
            out.push(pair.to_string());
        } else {
            // Fallback to SYMBOL/USD for unmapped assets
            fallback.push(sym.clone());
            out.push(format!("{sym}/USD"));
        }
    }
    if !fallback.is_empty() {
        tracing::info!(
            "[WS] using default pair mapping for {} symbols: {}",
            fallback.len(),
            fallback.join(",")
        );
    }
    // Deduplicate (e.g. XBT and BTC both map to BTC/USD)
    let mut seen = std::collections::HashSet::new();
    out.retain(|p| seen.insert(p.clone()));
    out
}

fn validate_channel_payload(channel: &str, v: &Value) -> bool {
    let has_data = v.get("data").is_some();
    match channel {
        "trade" | "ohlc" | "ticker" => has_data,
        "book" => {
            let msg_type = v.get("type").and_then(|t| t.as_str()).unwrap_or("");
            has_data && matches!(msg_type, "snapshot" | "update")
        }
        _ => true,
    }
}

fn jitter_ms(max_ms: u64) -> u64 {
    if max_ms == 0 {
        return 0;
    }
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.subsec_nanos() as u64)
        .unwrap_or(0);
    nanos % (max_ms + 1)
}

fn next_backoff(cur: Duration) -> Duration {
    std::cmp::min(Duration::from_secs(30), cur.mul_f32(1.8))
}

#[derive(Debug, Clone)]
pub enum WsEvent {
    Trade(Value),
    BookSnapshot(Value),
    BookUpdate(Value),
    Ohlc(Value),
    Ticker(Value),
    /// Private: order fully or partially filled on the exchange.
    OrderFill(Value),
}

struct WsStats {
    messages: u64,
    trade: u64,
    book: u64,
    ohlc: u64,
    ticker: u64,
    dropped_send: u64,
    send_errors: u64,
    last_message: Instant,
    last_print: Instant,
}

impl WsStats {
    fn new() -> Self {
        let now = Instant::now();
        Self {
            messages: 0,
            trade: 0,
            book: 0,
            ohlc: 0,
            ticker: 0,
            dropped_send: 0,
            send_errors: 0,
            last_message: now,
            last_print: now,
        }
    }
}

fn try_emit(sender: &mpsc::Sender<WsEvent>, ev: WsEvent, stats: &mut WsStats) {
    match sender.try_send(ev) {
        Ok(()) => {}
        Err(tokio::sync::mpsc::error::TrySendError::Full(_)) => {
            stats.dropped_send = stats.dropped_send.saturating_add(1);
        }
        Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => {
            stats.send_errors = stats.send_errors.saturating_add(1);
        }
    }
}

pub struct WsClient {
    debug_every_sec: u64,
}

impl WsClient {
    pub fn new(debug_every_sec: u64) -> Self {
        Self {
            debug_every_sec,
        }
    }

    pub async fn run_forever(
        &mut self,
        active_coins: Arc<RwLock<Vec<String>>>,
        ws_reconnect: Arc<AtomicBool>,
        depth: i64,
        ohlc_interval: i64,
        sender: mpsc::Sender<WsEvent>,
    ) -> Result<()> {
        let initial_backoff = Duration::from_secs(1);
        let mut backoff = initial_backoff;
        loop {
            let symbols = active_coins.read().await.clone();
            if symbols.is_empty() {
                tracing::warn!("[WS] no symbols — sleeping 5s");
                tokio::time::sleep(Duration::from_secs(5)).await;
                continue;
            }
            // Clear reconnect flag before connecting
            ws_reconnect.store(false, Ordering::SeqCst);

            let start = Instant::now();
            if let Err(e) = self
                .run_once(&symbols, depth, ohlc_interval, sender.clone(), &ws_reconnect)
                .await
            {
                tracing::error!("[WS] error: {e}");
            }
            // Reset backoff if connection was stable for at least 60s
            if start.elapsed() >= Duration::from_secs(60) {
                backoff = initial_backoff;
            }
            let sleep_dur = backoff + Duration::from_millis(jitter_ms(700));
            tokio::time::sleep(sleep_dur).await;
            backoff = next_backoff(backoff);
        }
    }

    async fn run_once(
        &mut self,
        symbols: &[String],
        depth: i64,
        ohlc_interval: i64,
        sender: mpsc::Sender<WsEvent>,
        ws_reconnect: &Arc<AtomicBool>,
    ) -> Result<()> {
        let pairs = normalize_symbols(symbols.to_vec());
        if pairs.is_empty() {
            anyhow::bail!("No valid symbols to subscribe");
        }

        let resnapshot_sec: u64 = std::env::var("BOOK_RESNAPSHOT_SEC")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(300);

        let (ws_stream, _) = connect_async(KRAKEN_WS_V2_PUBLIC).await?;
        let (mut write, mut read) = ws_stream.split();

        let trade_sub = json!({"method":"subscribe","params":{"channel":"trade","symbol":pairs}});
        let book_sub =
            json!({"method":"subscribe","params":{"channel":"book","symbol":pairs,"depth":depth}});
        let ohlc_sub = json!({"method":"subscribe","params":{"channel":"ohlc","symbol":pairs,"interval":ohlc_interval}});
        let ticker_sub = json!({"method":"subscribe","params":{"channel":"ticker","symbol":pairs}});

        write.send(Message::Text(trade_sub.to_string())).await?;
        write.send(Message::Text(book_sub.to_string())).await?;
        write.send(Message::Text(ohlc_sub.to_string())).await?;
        write.send(Message::Text(ticker_sub.to_string())).await?;

        let mut stats = WsStats::new();
        let session_start = Instant::now();

        loop {
            // Vision scanner coin rotation — reconnect with new symbols
            if ws_reconnect.load(Ordering::SeqCst) {
                tracing::info!(
                    "[WS] Reconnect signal from vision scanner — resubscribing with updated coins"
                );
                return Ok(());
            }

            // Periodic reconnect for fresh book snapshots
            if resnapshot_sec > 0 && session_start.elapsed().as_secs() >= resnapshot_sec {
                tracing::info!(
                    "[WS] session age {}s >= {}s — reconnecting for fresh book snapshots (dropped={} in session)",
                    session_start.elapsed().as_secs(),
                    resnapshot_sec,
                    stats.dropped_send
                );
                return Ok(());
            }
            let msg = timeout(Duration::from_secs(45), read.next()).await;
            let msg = match msg {
                Ok(Some(Ok(m))) => m,
                Ok(Some(Err(e))) => return Err(e.into()),
                Ok(None) => return Ok(()),
                Err(_) => continue,
            };

            stats.messages += 1;
            stats.last_message = Instant::now();

            if let Message::Text(txt) = msg {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&txt) {
                    if let Some(channel) = v.get("channel").and_then(|c| c.as_str()) {
                        if let Some(msg_type) = v.get("type").and_then(|t| t.as_str()) {
                            if matches!(msg_type, "subscribed" | "unsubscribed" | "error") {
                                continue;
                            }
                        }
                        match channel {
                            "trade" => {
                                if !validate_channel_payload(channel, &v) {
                                    tracing::error!("[WS] invalid trade payload shape");
                                    continue;
                                }
                                stats.trade += 1;
                                try_emit(&sender, WsEvent::Trade(v), &mut stats);
                            }
                            "book" => {
                                if !validate_channel_payload(channel, &v) {
                                    tracing::error!("[WS] invalid book payload shape");
                                    continue;
                                }
                                stats.book += 1;
                                let msg_type = v.get("type").and_then(|t| t.as_str()).unwrap_or("");
                                if msg_type == "snapshot" {
                                    try_emit(&sender, WsEvent::BookSnapshot(v), &mut stats);
                                } else {
                                    try_emit(&sender, WsEvent::BookUpdate(v), &mut stats);
                                }
                            }
                            "ohlc" => {
                                if !validate_channel_payload(channel, &v) {
                                    tracing::error!("[WS] invalid ohlc payload shape");
                                    continue;
                                }
                                stats.ohlc += 1;
                                try_emit(&sender, WsEvent::Ohlc(v), &mut stats);
                            }
                            "ticker" => {
                                if !validate_channel_payload(channel, &v) {
                                    tracing::error!("[WS] invalid ticker payload shape");
                                    continue;
                                }
                                stats.ticker += 1;
                                try_emit(&sender, WsEvent::Ticker(v), &mut stats);
                            }
                            _ => {}
                        }
                    }
                }
            }

            if self.debug_every_sec > 0
                && stats.last_print.elapsed().as_secs() >= self.debug_every_sec
            {
                stats.last_print = Instant::now();
                tracing::info!(
                    "[WS] msgs={} trade={} book={} ohlc={} ticker={} dropped={} send_err={} last={}s",
                    stats.messages,
                    stats.trade,
                    stats.book,
                    stats.ohlc,
                    stats.ticker,
                    stats.dropped_send,
                    stats.send_errors,
                    stats.last_message.elapsed().as_secs()
                );
            }
        }
    }
}

// ── Private WebSocket — order fills ─────────────────────────────────

/// Connects to Kraken's private WebSocket and emits `WsEvent::OrderFill`
/// whenever an order is fully or partially executed.
pub struct PrivateWsClient;

impl PrivateWsClient {
    /// Run forever, reconnecting on error. Emits `WsEvent::OrderFill` for
    /// each execution event (exec_type == "trade") received from Kraken.
    pub async fn run_forever(
        token: String,
        sender: mpsc::Sender<WsEvent>,
    ) {
        let mut backoff = Duration::from_secs(2);
        loop {
            if let Err(e) = Self::run_once(&token, &sender).await {
                tracing::warn!("[PRIVATE-WS] disconnected: {e} — retry in {}s", backoff.as_secs());
            }
            tokio::time::sleep(backoff).await;
            backoff = next_backoff(backoff);
        }
    }

    async fn run_once(
        token: &str,
        sender: &mpsc::Sender<WsEvent>,
    ) -> anyhow::Result<()> {
        let (ws_stream, _) = connect_async(KRAKEN_WS_V2_PRIVATE).await?;
        let (mut write, mut read) = ws_stream.split();

        let sub = json!({
            "method": "subscribe",
            "params": {
                "channel": "executions",
                "token": token,
                "snap_orders": false
            }
        });
        write.send(Message::Text(sub.to_string())).await?;
        tracing::info!("[PRIVATE-WS] subscribed to executions channel");

        loop {
            let msg = timeout(Duration::from_secs(60), read.next()).await;
            let msg = match msg {
                Ok(Some(Ok(m))) => m,
                Ok(Some(Err(e))) => return Err(e.into()),
                Ok(None) => return Ok(()),
                Err(_) => {
                    // heartbeat ping
                    write.send(Message::Ping(vec![])).await.ok();
                    continue;
                }
            };

            if let Message::Text(txt) = msg {
                if let Ok(v) = serde_json::from_str::<Value>(&txt) {
                    let channel = v.get("channel").and_then(|c| c.as_str()).unwrap_or("");
                    let msg_type = v.get("type").and_then(|t| t.as_str()).unwrap_or("");

                    if channel == "executions" && msg_type == "update" {
                        if let Some(data) = v.get("data").and_then(|d| d.as_array()) {
                            for exec in data {
                                let exec_type = exec.get("exec_type")
                                    .and_then(|e| e.as_str())
                                    .unwrap_or("");
                                // "trade" = actual fill; "pending_new" / "new" = order placed
                                if exec_type == "trade" {
                                    let sym = exec.get("symbol")
                                        .and_then(|s| s.as_str())
                                        .unwrap_or("?");
                                    let qty = exec.get("last_qty")
                                        .and_then(|q| q.as_f64())
                                        .unwrap_or(0.0);
                                    let price = exec.get("last_price")
                                        .and_then(|p| p.as_f64())
                                        .unwrap_or(0.0);
                                    let status = exec.get("order_status")
                                        .and_then(|s| s.as_str())
                                        .unwrap_or("?");
                                    tracing::info!(
                                        "[PRIVATE-WS] FILL {} qty={:.6} @ {:.4} status={}",
                                        sym, qty, price, status
                                    );
                                    if sender.try_send(WsEvent::OrderFill(exec.clone())).is_err() {
                                        tracing::warn!("[PRIVATE-WS] OrderFill dropped — channel full");
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{next_backoff, normalize_symbols, validate_channel_payload};
    use serde_json::json;
    use tokio::time::Duration;

    #[test]
    fn normalize_symbols_drops_invalid_and_maps_pairs() {
        let input = vec![
            "btc".to_string(),
            "ETH/USD".to_string(),
            "  ".to_string(),
            "NOPE".to_string(),
        ];
        let out = normalize_symbols(input);
        assert!(out.contains(&"BTC/USD".to_string()));
        assert!(out.contains(&"ETH/USD".to_string()));
        assert!(out.contains(&"NOPE/USD".to_string()));
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn validate_channel_payload_requires_data() {
        let bad = json!({"channel":"trade","type":"update"});
        let good = json!({"channel":"trade","type":"update","data":[{}]});
        assert!(!validate_channel_payload("trade", &bad));
        assert!(validate_channel_payload("trade", &good));
    }

    #[test]
    fn backoff_caps_at_thirty_seconds() {
        let mut d = Duration::from_secs(1);
        for _ in 0..20 {
            d = next_backoff(d);
        }
        assert_eq!(d, Duration::from_secs(30));
    }
}
