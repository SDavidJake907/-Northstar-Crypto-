use axum::{routing::get, Json, Router};
use chrono::{Duration, Utc};
use reqwest::Client;
use serde::Serialize;
use serde_json::Value;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::time::interval;

#[derive(Clone, Debug, Serialize, Default)]
struct SentimentSnapshot {
    value: i64,
    classification: String,
    ts: i64,
    source: String,
}

#[derive(Clone, Debug, Serialize, Default)]
struct CacheState {
    updated_at: i64,
    prices: HashMap<String, f64>,
    sentiment: Option<SentimentSnapshot>,
    errors: Vec<String>,
}

type SharedState = Arc<RwLock<CacheState>>;

#[tokio::main]
async fn main() {
    let _ = dotenvy::dotenv();

    let bind_addr = std::env::var("OPENBB_CACHE_BIND")
        .unwrap_or_else(|_| "127.0.0.1:6920".to_string());
    let openbb_base = std::env::var("OPENBB_BASE_URL")
        .unwrap_or_else(|_| "http://127.0.0.1:6900".to_string());
    let openbb_provider = std::env::var("OPENBB_PROVIDER")
        .unwrap_or_else(|_| "yfinance".to_string());
    let symbols = std::env::var("OPENBB_SYMBOLS")
        .unwrap_or_else(|_| "BTCUSD,ETHUSD".to_string());
    let symbols: Vec<String> = symbols
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    let state: SharedState = Arc::new(RwLock::new(CacheState::default()));
    let client = Client::builder()
        .timeout(std::time::Duration::from_secs(20))
        .build()
        .expect("reqwest client");

    let state_bg = state.clone();
    tokio::spawn(async move {
        let mut tick = interval(std::time::Duration::from_secs(900));
        loop {
            tick.tick().await;
            let mut snapshot = CacheState::default();
            snapshot.updated_at = Utc::now().timestamp();

            // Prices
            match fetch_prices(&client, &openbb_base, &openbb_provider, &symbols).await {
                Ok(prices) => snapshot.prices = prices,
                Err(e) => snapshot.errors.push(format!("prices: {e}")),
            }

            // Sentiment (Fear & Greed Index)
            match fetch_sentiment(&client).await {
                Ok(sent) => snapshot.sentiment = Some(sent),
                Err(e) => snapshot.errors.push(format!("sentiment: {e}")),
            }

            let mut guard = state_bg.write().await;
            *guard = snapshot;
        }
    });

    let app = Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/data", get({
            let state = state.clone();
            move || async move {
                let guard = state.read().await;
                Json(guard.clone())
            }
        }));

    let addr: SocketAddr = bind_addr.parse().expect("bind addr");
    println!("[openbb-cache] listening on http://{}", addr);

    axum::serve(tokio::net::TcpListener::bind(addr).await.unwrap(), app)
        .await
        .unwrap();
}

async fn fetch_prices(
    client: &Client,
    base_url: &str,
    provider: &str,
    symbols: &[String],
) -> Result<HashMap<String, f64>, String> {
    let mut out = HashMap::new();
    let base = base_url.trim_end_matches('/');
    let end_date = Utc::now().date_naive();
    let start_date = end_date - Duration::days(2);

    for sym in symbols {
        let url = format!("{base}/api/v1/crypto/price/historical");
        let resp = client
            .get(&url)
            .query(&[
                ("symbol", sym.as_str()),
                ("provider", provider),
                ("interval", "1m"),
                ("start_date", &start_date.to_string()),
                ("end_date", &end_date.to_string()),
            ])
            .send()
            .await
            .map_err(|e| format!("{sym}: request failed: {e}"))?;

        let text = resp.text().await.map_err(|e| format!("{sym}: read failed: {e}"))?;
        let v: Value = serde_json::from_str(&text)
            .map_err(|e| format!("{sym}: json parse failed: {e}"))?;

        let results = v.get("results").and_then(|r| r.as_array()).ok_or_else(|| {
            format!("{sym}: missing results array")
        })?;

        let mut last_price = None;
        for row in results.iter().rev() {
            if let Some(p) = row.get("close").and_then(|x| x.as_f64()) {
                last_price = Some(p);
                break;
            }
            if let Some(p) = row.get("price").and_then(|x| x.as_f64()) {
                last_price = Some(p);
                break;
            }
        }

        let price = last_price.ok_or_else(|| format!("{sym}: no price found"))?;
        out.insert(sym.clone(), price);
    }

    Ok(out)
}

async fn fetch_sentiment(client: &Client) -> Result<SentimentSnapshot, String> {
    let url = "https://api.alternative.me/fng/?limit=1&format=json";
    let text = client
        .get(url)
        .send()
        .await
        .map_err(|e| format!("request failed: {e}"))?
        .text()
        .await
        .map_err(|e| format!("read failed: {e}"))?;

    let v: Value = serde_json::from_str(&text).map_err(|e| format!("json parse failed: {e}"))?;
    let first = v
        .get("data")
        .and_then(|d| d.as_array())
        .and_then(|a| a.first())
        .ok_or_else(|| "missing data".to_string())?;

    let value = first
        .get("value")
        .and_then(|x| x.as_str())
        .and_then(|s| s.parse::<i64>().ok())
        .unwrap_or(0);
    let classification = first
        .get("value_classification")
        .and_then(|x| x.as_str())
        .unwrap_or("unknown")
        .to_string();
    let ts = first
        .get("timestamp")
        .and_then(|x| x.as_str())
        .and_then(|s| s.parse::<i64>().ok())
        .unwrap_or_else(|| Utc::now().timestamp());

    Ok(SentimentSnapshot {
        value,
        classification,
        ts,
        source: "alternative.me/fng".to_string(),
    })
}