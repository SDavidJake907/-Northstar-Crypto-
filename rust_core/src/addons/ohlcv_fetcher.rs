//! OHLCV Data Fetcher — downloads historical candles from Kraken public API.
//!
//! Stores CSV files in `data/ohlcv_store/{COIN}_{interval}m.csv`.
//! Used by the backtester for strategy evaluation.

use serde::Deserialize;
use std::path::{Path, PathBuf};

/// One OHLCV candle from Kraken.
#[derive(Debug, Clone)]
pub struct Candle {
    pub timestamp: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub vwap: f64,
    pub volume: f64,
    pub count: i64,
}

/// Kraken /0/public/OHLC response.
#[derive(Deserialize)]
#[allow(dead_code)]
struct KrakenOhlcResponse {
    error: Vec<String>,
    result: Option<serde_json::Value>,
}

/// Fetch OHLCV candles for a single coin from Kraken public API.
/// `interval` is in minutes (1, 5, 15, 60, 240, 1440).
/// `since` is a Unix timestamp to fetch data after.
/// Returns (candles, last_timestamp).
async fn fetch_ohlc_page(
    client: &reqwest::Client,
    coin: &str,
    interval: u32,
    since: Option<i64>,
) -> anyhow::Result<(Vec<Candle>, i64)> {
    let pair = format!("{coin}USD");
    let mut url = format!(
        "https://api.kraken.com/0/public/OHLC?pair={pair}&interval={interval}"
    );
    if let Some(s) = since {
        url.push_str(&format!("&since={s}"));
    }

    let resp: serde_json::Value = client
        .get(&url)
        .timeout(std::time::Duration::from_secs(30))
        .send()
        .await?
        .json()
        .await?;

    // Check errors
    if let Some(errors) = resp.get("error").and_then(|e| e.as_array()) {
        if !errors.is_empty() {
            let err_str: Vec<String> = errors
                .iter()
                .filter_map(|e| e.as_str().map(String::from))
                .collect();
            if !err_str.is_empty() {
                anyhow::bail!("Kraken OHLC error: {}", err_str.join(", "));
            }
        }
    }

    let result = resp
        .get("result")
        .ok_or_else(|| anyhow::anyhow!("No result in OHLC response"))?;

    // Find the candle array (key is the pair name, varies by coin)
    let mut candles = Vec::new();
    let mut last_ts: i64 = 0;

    for (key, val) in result.as_object().unwrap_or(&serde_json::Map::new()) {
        if key == "last" {
            last_ts = val.as_i64().unwrap_or(0);
            continue;
        }
        // This is the candle array
        if let Some(arr) = val.as_array() {
            for row in arr {
                if let Some(r) = row.as_array() {
                    if r.len() >= 8 {
                        let candle = Candle {
                            timestamp: r[0].as_i64().unwrap_or(0),
                            open: parse_num(&r[1]),
                            high: parse_num(&r[2]),
                            low: parse_num(&r[3]),
                            close: parse_num(&r[4]),
                            vwap: parse_num(&r[5]),
                            volume: parse_num(&r[6]),
                            count: r[7].as_i64().unwrap_or(0),
                        };
                        candles.push(candle);
                    }
                }
            }
        }
    }

    Ok((candles, last_ts))
}

fn parse_num(v: &serde_json::Value) -> f64 {
    match v {
        serde_json::Value::String(s) => s.parse().unwrap_or(0.0),
        serde_json::Value::Number(n) => n.as_f64().unwrap_or(0.0),
        _ => 0.0,
    }
}

/// Download all available OHLCV data for a coin at a given interval.
/// Paginates until no more new data. Returns total candle count.
pub async fn download_coin_ohlcv(
    client: &reqwest::Client,
    coin: &str,
    interval: u32,
    days_back: u32,
    store_dir: &Path,
) -> anyhow::Result<usize> {
    std::fs::create_dir_all(store_dir)?;

    let csv_path = store_dir.join(format!("{}_{interval}m.csv", coin));

    // Calculate starting timestamp
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_secs() as i64;
    let since_start = now - (days_back as i64 * 86400);

    let mut all_candles: Vec<Candle> = Vec::new();
    let mut since = Some(since_start);
    let mut pages = 0;

    loop {
        let (page_candles, last) = fetch_ohlc_page(client, coin, interval, since).await?;
        let count = page_candles.len();
        pages += 1;

        if count == 0 {
            break;
        }

        all_candles.extend(page_candles);

        // Kraken returns max 720 candles per request
        // If we got fewer, we've reached the end
        if count < 720 {
            break;
        }

        // Use last timestamp for pagination
        since = Some(last);

        // Rate limit: Kraken allows ~1 req/sec for public endpoints
        tokio::time::sleep(tokio::time::Duration::from_millis(1200)).await;

        // Safety: max 50 pages (~36,000 candles)
        if pages >= 50 {
            break;
        }
    }

    // Deduplicate by timestamp
    all_candles.sort_by_key(|c| c.timestamp);
    all_candles.dedup_by_key(|c| c.timestamp);

    // Write CSV
    let mut csv_content = String::from("timestamp,open,high,low,close,vwap,volume,count\n");
    for c in &all_candles {
        csv_content.push_str(&format!(
            "{},{},{},{},{},{},{},{}\n",
            c.timestamp, c.open, c.high, c.low, c.close, c.vwap, c.volume, c.count
        ));
    }
    std::fs::write(&csv_path, &csv_content)?;

    tracing::info!(
        "[OHLCV] {coin} {interval}m: {} candles ({pages} pages) → {}",
        all_candles.len(),
        csv_path.display()
    );

    Ok(all_candles.len())
}

/// Load candles from a CSV file previously downloaded.
pub fn load_candles_from_csv(path: &Path) -> anyhow::Result<Vec<Candle>> {
    let content = std::fs::read_to_string(path)?;
    let mut candles = Vec::new();

    for (i, line) in content.lines().enumerate() {
        if i == 0 {
            continue; // skip header
        }
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 8 {
            candles.push(Candle {
                timestamp: parts[0].parse().unwrap_or(0),
                open: parts[1].parse().unwrap_or(0.0),
                high: parts[2].parse().unwrap_or(0.0),
                low: parts[3].parse().unwrap_or(0.0),
                close: parts[4].parse().unwrap_or(0.0),
                vwap: parts[5].parse().unwrap_or(0.0),
                volume: parts[6].parse().unwrap_or(0.0),
                count: parts[7].parse().unwrap_or(0),
            });
        }
    }

    Ok(candles)
}

/// Download OHLCV for multiple coins at multiple intervals.
/// Returns total candles fetched.
pub async fn download_all(
    coins: &[String],
    intervals: &[u32],
    days_back: u32,
) -> anyhow::Result<usize> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()?;

    let store_dir = PathBuf::from("data/ohlcv_store");
    let mut total = 0;

    for coin in coins {
        for &interval in intervals {
            match download_coin_ohlcv(&client, coin, interval, days_back, &store_dir).await {
                Ok(n) => total += n,
                Err(e) => {
                    tracing::warn!("[OHLCV] Failed {coin} {interval}m: {e}");
                }
            }
            // Rate limit between requests
            tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
        }
    }

    tracing::info!("[OHLCV] Total: {total} candles for {} coins", coins.len());
    Ok(total)
}
