//! Kraken REST API client with HMAC-SHA512 authentication.
//!
//! Ported from python_orch/execution/kraken_api.py
//! Handles: nonce management, request signing, order placement,
//! balance queries, and all private/public endpoints.

use super::error::{HybridKrakenError, Result};
use base64::{engine::general_purpose::STANDARD as B64, Engine};
use hmac::{Hmac, Mac};
use sha2::{Digest, Sha256, Sha512};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::Mutex;

type HmacSha512 = Hmac<Sha512>;

const API_URL: &str = "https://api.kraken.com";
const MAX_RETRIES: u32 = 3;

struct RestThrottleState {
    next_global_at: Instant,
    next_endpoint_at: HashMap<String, Instant>,
}

struct RestThrottle {
    state: Mutex<RestThrottleState>,
    global_min: Duration,
    endpoint_min: HashMap<&'static str, Duration>,
}

impl RestThrottle {
    fn from_env() -> Self {
        let ms = |key: &str, default_ms: u64| -> Duration {
            let val = std::env::var(key)
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(default_ms);
            Duration::from_millis(val.max(1))
        };
        let endpoint_min = HashMap::from([
            ("AddOrder", ms("REST_ADD_ORDER_MIN_MS", 900)),
            ("CancelOrder", ms("REST_CANCEL_ORDER_MIN_MS", 350)),
            ("QueryOrders", ms("REST_QUERY_ORDERS_MIN_MS", 300)),
            ("Balance", ms("REST_BALANCE_MIN_MS", 1000)),
            ("OpenOrders", ms("REST_OPEN_ORDERS_MIN_MS", 900)),
        ]);
        let now = Instant::now();
        Self {
            state: Mutex::new(RestThrottleState {
                next_global_at: now,
                next_endpoint_at: HashMap::new(),
            }),
            global_min: ms("REST_GLOBAL_MIN_MS", 180),
            endpoint_min,
        }
    }

    async fn wait_turn(&self, endpoint: &str) -> bool {
        let now = Instant::now();
        let wait_for = {
            let mut st = self.state.lock().await;
            let endpoint_at = st.next_endpoint_at.get(endpoint).copied().unwrap_or(now);
            let wait_global = st.next_global_at.saturating_duration_since(now);
            let wait_ep = endpoint_at.saturating_duration_since(now);
            let wait_for = wait_global.max(wait_ep);
            if wait_for.is_zero() {
                let next_now = Instant::now();
                st.next_global_at = next_now + self.global_min;
                let endpoint_min = self
                    .endpoint_min
                    .get(endpoint)
                    .copied()
                    .unwrap_or(self.global_min);
                st.next_endpoint_at
                    .insert(endpoint.to_string(), next_now + endpoint_min);
            }
            wait_for
        };
        if wait_for.is_zero() {
            return false;
        }
        tokio::time::sleep(wait_for).await;
        true
    }
}

/// Kraken REST API client (async, thread-safe).
pub struct KrakenApi {
    api_key: String,
    api_secret: Vec<u8>, // decoded base64
    client: reqwest::Client,
    last_nonce: AtomicU64,
    throttle: RestThrottle,
    rest_throttle_hits: AtomicU64,
    rest_backoff_hits: AtomicU64,
}

impl KrakenApi {
    /// Create a new KrakenApi client.
    ///
    /// `api_key`: Kraken API key string
    /// `api_secret`: Kraken API secret (base64-encoded string)
    pub fn new(api_key: &str, api_secret: &str) -> Result<Self> {
        let decoded_secret = B64.decode(api_secret)?;
        let now_us = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_micros() as u64)
            .unwrap_or(0);

        Ok(Self {
            api_key: api_key.to_string(),
            api_secret: decoded_secret,
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()?,
            last_nonce: AtomicU64::new(now_us),
            throttle: RestThrottle::from_env(),
            rest_throttle_hits: AtomicU64::new(0),
            rest_backoff_hits: AtomicU64::new(0),
        })
    }

    /// Generate a monotonically increasing nonce (thread-safe, microsecond precision).
    fn next_nonce(&self) -> u64 {
        let now_us = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_micros() as u64)
            .unwrap_or(0);

        loop {
            let prev = self.last_nonce.load(Ordering::Acquire);
            let next = if now_us > prev { now_us } else { prev + 1 };
            if self
                .last_nonce
                .compare_exchange(prev, next, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return next;
            }
        }
    }

    /// Sign a private API request (HMAC-SHA512).
    ///
    /// Kraken signing:
    ///   postdata = urlencode(data)
    ///   message  = urlpath.encode() + SHA256(nonce + postdata)
    ///   signature = HMAC-SHA512(base64_decode(secret), message)
    fn sign(&self, urlpath: &str, data: &[(String, String)]) -> Result<String> {
        // URL-encode the POST data
        let postdata = url::form_urlencoded::Serializer::new(String::new())
            .extend_pairs(data)
            .finish();

        // Find nonce in data
        let nonce_str = data
            .iter()
            .find(|(k, _)| k == "nonce")
            .map(|(_, v)| v.as_str())
            .unwrap_or("");

        // SHA256(nonce + postdata)
        let mut sha256 = Sha256::new();
        sha256.update(nonce_str.as_bytes());
        sha256.update(postdata.as_bytes());
        let sha256_digest = sha256.finalize();

        // message = urlpath + sha256_digest
        let mut message = urlpath.as_bytes().to_vec();
        message.extend_from_slice(&sha256_digest);

        // HMAC-SHA512(secret, message)
        let mut mac = HmacSha512::new_from_slice(&self.api_secret)
            .map_err(|e| HybridKrakenError::HmacInit(e.to_string()))?;
        mac.update(&message);
        let result = mac.finalize();

        Ok(B64.encode(result.into_bytes()))
    }

    /// Execute a signed private POST request with retries.
    async fn private_request(
        &self,
        method: &str,
        extra_data: Option<Vec<(String, String)>>,
    ) -> Result<serde_json::Value> {
        let urlpath = format!("/0/private/{method}");
        let url = format!("{API_URL}{urlpath}");

        for attempt in 0..=MAX_RETRIES {
            if self.throttle.wait_turn(method).await {
                self.rest_throttle_hits.fetch_add(1, Ordering::Relaxed);
            }
            let nonce = self.next_nonce();
            let mut data: Vec<(String, String)> = vec![("nonce".into(), nonce.to_string())];
            if let Some(ref extra) = extra_data {
                data.extend(extra.iter().cloned());
            }

            let signature = self.sign(&urlpath, &data)?;

            let resp = self
                .client
                .post(&url)
                .header("API-Key", &self.api_key)
                .header("API-Sign", &signature)
                .form(&data)
                .send()
                .await;

            match resp {
                Ok(r) => {
                    let status = r.status();
                    if status.as_u16() == 429 {
                        if attempt >= MAX_RETRIES {
                            return Err(HybridKrakenError::KrakenHttp(
                                "429 Too Many Requests".into(),
                            ));
                        }
                        self.rest_backoff_hits.fetch_add(1, Ordering::Relaxed);
                        let delay_ms = backoff_ms(attempt);
                        tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                        continue;
                    }
                    let body: serde_json::Value = r.json().await?;

                    // Kraken returns errors in JSON even on HTTP 200
                    if let Some(errors) = body.get("error").and_then(|e| e.as_array()) {
                        if !errors.is_empty() {
                            let err_str: Vec<String> = errors
                                .iter()
                                .filter_map(|e| e.as_str().map(String::from))
                                .collect();
                            if err_str
                                .iter()
                                .any(|e| e.contains("EAPI:Rate limit exceeded"))
                            {
                                self.rest_backoff_hits.fetch_add(1, Ordering::Relaxed);
                                let delay_ms = backoff_ms(attempt);
                                tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                                continue;
                            }
                            if attempt >= MAX_RETRIES {
                                return Err(HybridKrakenError::KrakenApi(err_str.join(", ")));
                            }
                        } else {
                            return Ok(body);
                        }
                    } else if !status.is_success() {
                        if attempt >= MAX_RETRIES {
                            return Err(HybridKrakenError::KrakenHttp(status.to_string()));
                        }
                    } else {
                        return Ok(body);
                    }
                }
                Err(e) => {
                    if attempt >= MAX_RETRIES {
                        return Err(HybridKrakenError::Http(e));
                    }
                }
            }

            tokio::time::sleep(Duration::from_millis(backoff_ms(attempt))).await;
        }

        Err(HybridKrakenError::Other(
            "Kraken private request exhausted retries".into(),
        ))
    }

    // ── Private endpoints ─────────────────────────────────────────

    /// Get account balance (per-asset amounts).
    pub async fn balance(&self) -> Result<serde_json::Value> {
        self.private_request("Balance", None).await
    }

    /// Get trade balance (equity, available margin, etc.).
    /// Returns result.tb = trade balance (available for trading).
    pub async fn trade_balance(&self) -> Result<serde_json::Value> {
        let params = vec![("asset".to_string(), "ZUSD".to_string())];
        self.private_request("TradeBalance", Some(params)).await
    }

    /// Place an order.
    ///
    /// `pair`: e.g. "XBTUSD"
    /// `side`: "buy" or "sell"
    /// `ordertype`: "market", "limit", etc.
    /// `volume`: order volume as string
    /// `price`: limit price (None for market orders)
    pub async fn add_order(
        &self,
        pair: &str,
        side: &str,
        ordertype: &str,
        volume: &str,
        price: Option<&str>,
    ) -> Result<serde_json::Value> {
        let mut data = vec![
            ("pair".into(), pair.into()),
            ("type".into(), side.into()),
            ("ordertype".into(), ordertype.into()),
            ("volume".into(), volume.into()),
        ];
        if let Some(p) = price {
            data.push(("price".into(), p.into()));
        }
        self.private_request("AddOrder", Some(data)).await
    }

    /// Cancel an order by transaction ID.
    pub async fn cancel_order(&self, txid: &str) -> Result<serde_json::Value> {
        self.private_request("CancelOrder", Some(vec![("txid".into(), txid.into())]))
            .await
    }

    /// Cancel all open orders.
    pub async fn cancel_all_orders(&self) -> Result<serde_json::Value> {
        self.private_request("CancelAll", None).await
    }

    /// Query specific orders by transaction IDs (comma-separated).
    pub async fn query_orders(&self, txids: &str) -> Result<serde_json::Value> {
        self.private_request("QueryOrders", Some(vec![("txid".into(), txids.into())]))
            .await
    }

    /// Get a short-lived WebSocket authentication token.
    /// Required for subscribing to private channels (executions, balances).
    pub async fn get_ws_token(&self) -> Result<String> {
        let resp = self.private_request("GetWebSocketsToken", None).await?;
        resp.get("result")
            .and_then(|r| r.get("token"))
            .and_then(|t| t.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| HybridKrakenError::Other("GetWebSocketsToken: no token in response".into()))
    }

    pub fn throttle_stats(&self) -> (u64, u64) {
        (
            self.rest_throttle_hits.load(Ordering::Relaxed),
            self.rest_backoff_hits.load(Ordering::Relaxed),
        )
    }
}

/// Create a KrakenApi from environment variables.
///
/// Reads `KRAKEN_API_KEY` and `KRAKEN_API_SECRET` from the environment.
pub fn from_env() -> Result<KrakenApi> {
    let key = std::env::var("KRAKEN_API_KEY")
        .map_err(|_| HybridKrakenError::MissingEnv("KRAKEN_API_KEY"))?;
    let secret = std::env::var("KRAKEN_API_SECRET")
        .map_err(|_| HybridKrakenError::MissingEnv("KRAKEN_API_SECRET"))?;
    KrakenApi::new(&key, &secret)
}

// ── Public Market Data (no auth) ─────────────────────────────

/// Top mover info from Kraken public ticker.
#[derive(Clone, Debug)]
pub struct MoverInfo {
    pub symbol: String,
    pub price: f64,
    pub change_24h_pct: f64,
    pub volume_usd: f64,
    pub edge_score: f64,
    pub break_even_exit: f64,
    /// 7-day price change % (0.0 if not yet fetched)
    pub trend_7d_pct: f64,
    /// 30-day price change % (0.0 if not yet fetched)
    pub trend_30d_pct: f64,
}

/// Fetch 7-day and 30-day trend % for a single coin using daily OHLC candles.
/// Returns (trend_7d_pct, trend_30d_pct). Returns (0.0, 0.0) on any error.
pub async fn fetch_daily_trend(client: &reqwest::Client, symbol: &str) -> (f64, f64) {
    let pair = format!("{}USD", symbol);
    let url = format!(
        "https://api.kraken.com/0/public/OHLC?pair={}&interval=1440",
        pair
    );
    let resp = match client.get(&url)
        .timeout(std::time::Duration::from_secs(8))
        .send().await
    {
        Ok(r) => r,
        Err(_) => return (0.0, 0.0),
    };
    let json: serde_json::Value = match resp.json().await {
        Ok(j) => j,
        Err(_) => return (0.0, 0.0),
    };
    let result = match json.get("result") {
        Some(r) => r,
        None => return (0.0, 0.0),
    };
    // Result is a map: first key is the pair name, second is "last"
    let candles = result.as_object()
        .and_then(|obj| obj.values().next())
        .and_then(|v| v.as_array());
    let candles = match candles {
        Some(c) if c.len() >= 31 => c,
        _ => return (0.0, 0.0),
    };
    // Each candle: [time, open, high, low, close, vwap, volume, count]
    let close_now = candles.last()
        .and_then(|c| c.get(4))
        .and_then(|v| v.as_str())
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.0);
    if close_now <= 0.0 { return (0.0, 0.0); }
    let len = candles.len();
    let close_7d = candles.get(len.saturating_sub(8))
        .and_then(|c| c.get(4))
        .and_then(|v| v.as_str())
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.0);
    let close_30d = candles.get(len.saturating_sub(31))
        .and_then(|c| c.get(4))
        .and_then(|v| v.as_str())
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.0);
    let trend_7d  = if close_7d  > 0.0 { (close_now - close_7d)  / close_7d  * 100.0 } else { 0.0 };
    let trend_30d = if close_30d > 0.0 { (close_now - close_30d) / close_30d * 100.0 } else { 0.0 };
    (trend_7d, trend_30d)
}

// ── EV Math Helpers ──────────────────────────────────────────

/// Round-trip fee as fraction (0.26% each side = 0.52% total).
pub const FEE_ROUNDTRIP: f64 = 0.0052;
/// Exit price needed to achieve a target net return.
/// e.g., target_return=0.05 for +5%, fb/fs=0.0026 for Kraken maker.
pub fn target_exit_price(pb: f64, fb: f64, fs: f64, target_return: f64) -> f64 {
    if pb <= 0.0 || fs >= 1.0 { return f64::NAN; }
    pb * (1.0 + fb) * (1.0 + target_return) / (1.0 - fs)
}

/// Edge score for ranking movers (price-only, no sentiment).
/// Prefers meaningful move + liquidity, penalizes fee drag.
pub fn edge_score(change_24h_pct: f64, volume_usd: f64, fee_roundtrip: f64) -> f64 {
    let chg = (change_24h_pct / 100.0).abs();
    let vol_term = (1.0 + volume_usd.max(0.0)).ln();
    let risk = chg;
    vol_term * chg / (fee_roundtrip + 0.001 + risk)
}

/// Fallback pool: 40 Kraken-supported coins (used when AssetPairs API fails).
const TOP_MOVER_CANDIDATES: &[&str] = &[
    "BTC","ETH","SOL","AVAX","LINK","ADA","DOT","PEPE","ARB","XRP",
    "GRT","CRV","LTC","SAND","ONDO","ATOM","NEAR","SUI","INJ","AXS",
    "DOGE","SHIB","MATIC","OP","FIL","AAVE","UNI","MKR","SNX","COMP",
    "APE","MANA","LDO","FTM","ALGO","XLM","TRX","HBAR","ICP","TIA",
];

/// Symbols to exclude from universe scan (stablecoins, fiat-pegged, wrapped).
const EXCLUDE_SYMBOLS: &[&str] = &[
    "USDT", "USDC", "DAI", "BUSD", "TUSD", "PYUSD", "USDP", "GUSD", "FRAX",
    "LUSD", "SUSD", "MIM", "EURT", "EUROC", "UST", "USDD", "USTC",
];

/// Cached pair names from Kraken AssetPairs (refreshed every 6 hours).
static UNIVERSE_CACHE: std::sync::Mutex<Option<(Instant, Vec<String>)>> =
    std::sync::Mutex::new(None);
const UNIVERSE_CACHE_TTL: Duration = Duration::from_secs(6 * 3600);

/// Fetch top movers from Kraken public Ticker API (no auth needed).
/// Batch-queries all 40 candidates, returns top 10 sorted by abs(change_24h_pct) desc.
pub async fn get_top_movers(client: &reqwest::Client) -> Vec<MoverInfo> {
    // Build comma-separated pair list for batch query
    let pairs: String = TOP_MOVER_CANDIDATES
        .iter()
        .map(|s| format!("{s}USD"))
        .collect::<Vec<_>>()
        .join(",");
    let url = format!("{API_URL}/0/public/Ticker?pair={pairs}");

    let resp = match client
        .get(&url)
        .timeout(Duration::from_secs(15))
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!("[TOP_MOVERS] HTTP error: {e}");
            return Vec::new();
        }
    };

    let json: serde_json::Value = match resp.json().await {
        Ok(j) => j,
        Err(e) => {
            tracing::warn!("[TOP_MOVERS] Parse error: {e}");
            return Vec::new();
        }
    };

    let result = match json.get("result").and_then(|r| r.as_object()) {
        Some(r) => r,
        None => {
            if let Some(errs) = json.get("error").and_then(|e| e.as_array()) {
                tracing::warn!("[TOP_MOVERS] API error: {:?}", errs);
            }
            return Vec::new();
        }
    };

    let mut movers = Vec::with_capacity(40);

    for (pair_key, data) in result {
        // Extract symbol from pair key (strip USD/ZUSD suffix, handle XBT→BTC)
        let symbol = pair_key_to_symbol(pair_key);
        if symbol.is_empty() {
            continue;
        }

        let price: f64 = data
            .get("c")
            .and_then(|c| c.get(0))
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.0);
        let open: f64 = data
            .get("o")
            .and_then(|o| o.as_str())
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.0);
        let vol_24h: f64 = data
            .get("v")
            .and_then(|v| v.get(1))
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.0);

        if price <= 0.0 || open <= 0.0 {
            continue;
        }

        let change_pct = ((price - open) / open) * 100.0;
        let volume_usd = vol_24h * price;

        let score = edge_score(change_pct, volume_usd, FEE_ROUNDTRIP);
        let be_exit = target_exit_price(price, 0.0026, 0.0026, 0.0); // break-even price

        movers.push(MoverInfo {
            symbol,
            price,
            change_24h_pct: change_pct,
            volume_usd,
            edge_score: score,
            break_even_exit: be_exit,
            trend_7d_pct: 0.0,
            trend_30d_pct: 0.0,
        });
    }

    // Sort by edge_score descending (not abs change — avoids chasing late moves)
    movers.sort_by(|a, b| {
        b.edge_score
            .partial_cmp(&a.edge_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    movers.truncate(10);
    movers
}

// ── Full Universe Scanner ────────────────────────────────────

/// Fetch ALL tradeable USD pair names from Kraken AssetPairs API.
/// Results are cached for 6 hours since the pair list rarely changes.
async fn fetch_all_usd_pair_names(client: &reqwest::Client) -> Vec<String> {
    // Check cache first
    if let Ok(guard) = UNIVERSE_CACHE.lock() {
        if let Some((when, ref names)) = *guard {
            if when.elapsed() < UNIVERSE_CACHE_TTL {
                return names.clone();
            }
        }
    }

    let url = format!("{API_URL}/0/public/AssetPairs");
    let resp = match client.get(&url).timeout(Duration::from_secs(20)).send().await {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!("[UNIVERSE] AssetPairs HTTP error: {e}");
            return Vec::new();
        }
    };

    let json: serde_json::Value = match resp.json().await {
        Ok(j) => j,
        Err(e) => {
            tracing::warn!("[UNIVERSE] AssetPairs parse error: {e}");
            return Vec::new();
        }
    };

    let result = match json.get("result").and_then(|r| r.as_object()) {
        Some(r) => r,
        None => {
            if let Some(errs) = json.get("error").and_then(|e| e.as_array()) {
                tracing::warn!("[UNIVERSE] AssetPairs API error: {:?}", errs);
            }
            return Vec::new();
        }
    };

    let mut pair_names = Vec::with_capacity(600);
    let mut seen_symbols = std::collections::HashSet::new();

    for (pair_name, pair_data) in result {
        // Only USD-quoted pairs
        let quote = pair_data.get("quote").and_then(|q| q.as_str()).unwrap_or("");
        if quote != "ZUSD" && quote != "USD" {
            continue;
        }
        // Must be tradeable
        let status = pair_data.get("status").and_then(|s| s.as_str()).unwrap_or("online");
        if status != "online" {
            continue;
        }
        // Convert to symbol and check exclusions
        let sym = pair_key_to_symbol(pair_name);
        if sym.is_empty() || EXCLUDE_SYMBOLS.contains(&sym.as_str()) {
            continue;
        }
        // Deduplicate (some coins have both legacy + new pair names)
        if seen_symbols.insert(sym) {
            pair_names.push(pair_name.clone());
        }
    }

    pair_names.sort();
    tracing::info!("[UNIVERSE] Discovered {} tradeable USD pairs on Kraken", pair_names.len());

    // Update cache
    if let Ok(mut guard) = UNIVERSE_CACHE.lock() {
        *guard = Some((Instant::now(), pair_names.clone()));
    }

    pair_names
}

/// Fetch top movers from the FULL Kraken universe (500+ coins).
/// Discovers all USD pairs via AssetPairs, batch-queries Ticker, returns top N by edge score.
/// Falls back to hardcoded 40-coin pool if AssetPairs fails.
pub async fn get_top_movers_universe(client: &reqwest::Client, top_n: usize) -> Vec<MoverInfo> {
    let pair_names = fetch_all_usd_pair_names(client).await;
    if pair_names.is_empty() {
        tracing::warn!("[UNIVERSE] No pairs discovered, falling back to hardcoded pool");
        return get_top_movers(client).await;
    }

    let mut all_movers = Vec::with_capacity(pair_names.len());

    // Batch Ticker queries in chunks of 150 to stay under URL length limits
    for (batch_idx, chunk) in pair_names.chunks(150).enumerate() {
        let pairs_str: String = chunk.join(",");
        let url = format!("{API_URL}/0/public/Ticker?pair={pairs_str}");

        let resp = match client.get(&url).timeout(Duration::from_secs(20)).send().await {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!("[UNIVERSE] Ticker batch {} error: {e}", batch_idx);
                continue;
            }
        };

        let json: serde_json::Value = match resp.json().await {
            Ok(j) => j,
            Err(e) => {
                tracing::warn!("[UNIVERSE] Ticker batch {} parse error: {e}", batch_idx);
                continue;
            }
        };

        let result = match json.get("result").and_then(|r| r.as_object()) {
            Some(r) => r,
            None => continue,
        };

        for (pair_key, data) in result {
            let symbol = pair_key_to_symbol(pair_key);
            if symbol.is_empty() { continue; }

            let price: f64 = data.get("c").and_then(|c| c.get(0))
                .and_then(|v| v.as_str()).and_then(|s| s.parse().ok()).unwrap_or(0.0);
            let open: f64 = data.get("o").and_then(|o| o.as_str())
                .and_then(|s| s.parse().ok()).unwrap_or(0.0);
            let vol_24h: f64 = data.get("v").and_then(|v| v.get(1))
                .and_then(|v| v.as_str()).and_then(|s| s.parse().ok()).unwrap_or(0.0);

            if price <= 0.0 || open <= 0.0 { continue; }

            let change_pct = ((price - open) / open) * 100.0;
            let volume_usd = vol_24h * price;

            // Skip ultra-low-volume coins (< $50K daily)
            if volume_usd < 50_000.0 { continue; }

            let score = edge_score(change_pct, volume_usd, FEE_ROUNDTRIP);
            let be_exit = target_exit_price(price, 0.0026, 0.0026, 0.0);

            all_movers.push(MoverInfo {
                symbol,
                price,
                change_24h_pct: change_pct,
                volume_usd,
                edge_score: score,
                break_even_exit: be_exit,
                trend_7d_pct: 0.0,
                trend_30d_pct: 0.0,
            });
        }

        // Courtesy delay between batches (Kraken public API rate limits)
        if batch_idx < pair_names.chunks(150).count() - 1 {
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
    }

    tracing::info!(
        "[UNIVERSE] Scanned {} coins with vol > $50K, returning top {}",
        all_movers.len(), top_n
    );

    all_movers.sort_by(|a, b| {
        b.edge_score.partial_cmp(&a.edge_score).unwrap_or(std::cmp::Ordering::Equal)
    });
    all_movers.truncate(top_n);
    all_movers
}

/// Convert Kraken pair key to our symbol (e.g. "XXBTZUSD" → "BTC", "SOLUSD" → "SOL").
fn pair_key_to_symbol(pair_key: &str) -> String {
    let upper = pair_key.to_uppercase();
    // Try stripping ZUSD first (Kraken legacy pairs like XXBTZUSD, XETHZUSD)
    let base = if let Some(b) = upper.strip_suffix("ZUSD") {
        b
    } else if let Some(b) = upper.strip_suffix("USD") {
        b
    } else {
        return String::new();
    };
    // Strip leading X for legacy pairs (XXBT, XETH, XXRP, etc.)
    let base = if base.len() > 3 && base.starts_with('X') {
        &base[1..]
    } else {
        base
    };
    // XBT → BTC
    if base == "XBT" {
        "BTC".to_string()
    } else {
        base.to_string()
    }
}

fn backoff_ms(attempt: u32) -> u64 {
    let base = (100u64 * (1u64 << attempt)).min(5000);
    base + (rand_jitter_ms(300))
}

fn rand_jitter_ms(max_ms: u64) -> u64 {
    if max_ms == 0 {
        return 0;
    }
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| (d.as_nanos() as u64) % max_ms)
        .unwrap_or(0)
}
