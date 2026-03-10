//! News & Sentiment module — free APIs, no keys needed.
//! Polls every 5 minutes and injects a sentiment block into Qwen's prompt.
//!
//! Data sources:
//!   1. Fear & Greed Index (alternative.me) — overall crypto market mood (0-100)
//!   2. CoinGecko Global — total market cap, BTC dominance, 24h change
//!   3. CoinGecko Trending — top trending coins (what the market is watching)
//!
//! Flow: maybe_update(now) → build_news_block() → injected into entry/exit prompts

use serde::Deserialize;

// ── API Response Types ──────────────────────────────────────

#[derive(Deserialize, Debug)]
struct FngResponse {
    data: Option<Vec<FngEntry>>,
}

#[derive(Deserialize, Debug, Clone)]
struct FngEntry {
    value: String,
    value_classification: String,
}

#[derive(Deserialize, Debug)]
struct CgGlobalResponse {
    data: Option<CgGlobalData>,
}

#[derive(Deserialize, Debug, Clone)]
struct CgGlobalData {
    total_market_cap: Option<std::collections::HashMap<String, f64>>,
    market_cap_percentage: Option<std::collections::HashMap<String, f64>>,
    market_cap_change_percentage_24h_usd: Option<f64>,
}

#[derive(Deserialize, Debug)]
struct CgTrendingResponse {
    coins: Option<Vec<CgTrendingCoin>>,
}

#[derive(Deserialize, Debug, Clone)]
struct CgTrendingCoin {
    item: Option<CgTrendingItem>,
}

#[derive(Deserialize, Debug, Clone)]
#[allow(dead_code)] // fields needed for JSON deserialization
struct CgTrendingItem {
    symbol: Option<String>,
    market_cap_rank: Option<u32>,
    data: Option<CgTrendingData>,
}

#[derive(Deserialize, Debug, Clone)]
#[allow(dead_code)] // fields needed for JSON deserialization
struct CgTrendingData {
    price: Option<f64>,
    price_change_percentage_24h: Option<std::collections::HashMap<String, f64>>,
}

// ── Cached sentiment data ───────────────────────────────────

#[derive(Debug, Clone)]
pub struct SentimentSnapshot {
    // Fear & Greed
    pub fng_value: u32,               // 0-100
    pub fng_label: String,            // "Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"

    // CoinGecko Global
    pub total_market_cap_usd: f64,    // in USD
    pub btc_dominance: f64,           // percentage
    pub market_cap_change_24h: f64,   // percentage

    // Trending coins
    pub trending: Vec<TrendingCoin>,
}

#[derive(Debug, Clone)]
pub struct TrendingCoin {
    pub symbol: String,
    pub change_24h: f64,
}

impl Default for SentimentSnapshot {
    fn default() -> Self {
        Self {
            fng_value: 50,
            fng_label: "unknown".into(),
            total_market_cap_usd: 0.0,
            btc_dominance: 0.0,
            market_cap_change_24h: 0.0,
            trending: Vec::new(),
        }
    }
}

// ── NewsSentiment state ─────────────────────────────────────

pub struct NewsSentiment {
    client: reqwest::Client,
    snapshot: SentimentSnapshot,
    last_fetch_ts: f64,
    fetch_interval_sec: f64,
}

impl NewsSentiment {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(10))
                .user_agent("kraken-hybrad9/1.0")
                .build()
                .unwrap_or_default(),
            snapshot: SentimentSnapshot::default(),
            last_fetch_ts: 0.0,
            fetch_interval_sec: 300.0, // 5 minutes
        }
    }

    /// Call once per tick — timer-gated internally
    pub async fn maybe_update(&mut self, now: f64) {
        if now - self.last_fetch_ts < self.fetch_interval_sec {
            return;
        }
        self.last_fetch_ts = now;

        // Fetch all three sources in parallel
        let (fng_result, global_result, trending_result) = tokio::join!(
            self.fetch_fear_greed(),
            self.fetch_global(),
            self.fetch_trending(),
        );

        if let Ok((value, label)) = fng_result {
            self.snapshot.fng_value = value;
            self.snapshot.fng_label = label;
            tracing::info!("[NEWS-SENTIMENT] Fear/Greed: {} ({})", self.snapshot.fng_value, self.snapshot.fng_label);
        } else if let Err(e) = fng_result {
            tracing::warn!("[NEWS-SENTIMENT] Fear/Greed fetch failed: {e}");
        }

        if let Ok((mcap, btc_dom, change_24h)) = global_result {
            self.snapshot.total_market_cap_usd = mcap;
            self.snapshot.btc_dominance = btc_dom;
            self.snapshot.market_cap_change_24h = change_24h;
            tracing::info!(
                "[NEWS-SENTIMENT] Market: ${:.0}B cap | BTC dom={:.1}% | 24h={:+.2}%",
                mcap / 1e9, btc_dom, change_24h
            );
        } else if let Err(e) = global_result {
            tracing::warn!("[NEWS-SENTIMENT] Global market fetch failed: {e}");
        }

        if let Ok(coins) = trending_result {
            let symbols: Vec<String> = coins.iter().take(5).map(|c| c.symbol.clone()).collect();
            self.snapshot.trending = coins;
            tracing::info!("[NEWS-SENTIMENT] Trending: {}", symbols.join(", "));
        } else if let Err(e) = trending_result {
            tracing::warn!("[NEWS-SENTIMENT] Trending fetch failed: {e}");
        }
    }

    /// Fear & Greed Index — alternative.me (free, no key)
    async fn fetch_fear_greed(&self) -> Result<(u32, String), String> {
        let url = "https://api.alternative.me/fng/?limit=1";
        let resp = self.client.get(url).send().await
            .map_err(|e| format!("request failed: {e}"))?;
        if !resp.status().is_success() {
            return Err(format!("HTTP {}", resp.status()));
        }
        let body: FngResponse = resp.json().await
            .map_err(|e| format!("parse failed: {e}"))?;
        let entry = body.data
            .and_then(|d| d.into_iter().next())
            .ok_or("empty response")?;
        let value = entry.value.parse::<u32>().unwrap_or(50);
        Ok((value, entry.value_classification))
    }

    /// CoinGecko Global market data (free, no key)
    async fn fetch_global(&self) -> Result<(f64, f64, f64), String> {
        let url = "https://api.coingecko.com/api/v3/global";
        let resp = self.client.get(url).send().await
            .map_err(|e| format!("request failed: {e}"))?;
        if !resp.status().is_success() {
            return Err(format!("HTTP {}", resp.status()));
        }
        let body: CgGlobalResponse = resp.json().await
            .map_err(|e| format!("parse failed: {e}"))?;
        let data = body.data.ok_or("no data field")?;
        let mcap = data.total_market_cap
            .as_ref()
            .and_then(|m| m.get("usd"))
            .copied()
            .unwrap_or(0.0);
        let btc_dom = data.market_cap_percentage
            .as_ref()
            .and_then(|m| m.get("btc"))
            .copied()
            .unwrap_or(0.0);
        let change = data.market_cap_change_percentage_24h_usd.unwrap_or(0.0);
        Ok((mcap, btc_dom, change))
    }

    /// CoinGecko Trending coins (free, no key)
    async fn fetch_trending(&self) -> Result<Vec<TrendingCoin>, String> {
        let url = "https://api.coingecko.com/api/v3/search/trending";
        let resp = self.client.get(url).send().await
            .map_err(|e| format!("request failed: {e}"))?;
        if !resp.status().is_success() {
            return Err(format!("HTTP {}", resp.status()));
        }
        let body: CgTrendingResponse = resp.json().await
            .map_err(|e| format!("parse failed: {e}"))?;
        let coins = body.coins.unwrap_or_default();
        let mut out = Vec::new();
        for c in coins.into_iter().take(10) {
            let item = match c.item {
                Some(it) => it,
                None => continue,
            };
            let symbol = item.symbol.unwrap_or_default().to_uppercase();
            let change_24h = item.data.as_ref()
                .and_then(|d| d.price_change_percentage_24h.as_ref())
                .and_then(|m| m.get("usd"))
                .copied()
                .unwrap_or(0.0);
            out.push(TrendingCoin {
                symbol,
                change_24h,
            });
        }
        Ok(out)
    }

    // ── Prompt blocks for Qwen ──────────────────────────────

    /// Full sentiment block for entry decisions
    pub fn build_news_block(&self) -> String {
        if self.snapshot.fng_label == "unknown" && self.snapshot.total_market_cap_usd == 0.0 {
            return String::new(); // no data yet
        }

        let mut out = String::from("=== NEWS & SENTIMENT ===\n");

        // Fear & Greed with trading guidance
        let fng_guidance = match self.snapshot.fng_value {
            0..=10  => "EXTREME FEAR — market panicking, avoid impulsive entries, watch for capitulation bounces",
            11..=25 => "FEAR — sentiment weak, be cautious, only high-conviction setups",
            26..=45 => "MILD FEAR — slightly bearish mood, normal caution",
            46..=55 => "NEUTRAL — no strong sentiment bias",
            56..=75 => "GREED — bullish sentiment, momentum trades favored but watch for overextension",
            76..=90 => "HIGH GREED — euphoria building, tighten stops, watch for reversal signals",
            _       => "EXTREME GREED — market euphoric, high reversal risk, defensive positioning",
        };
        out.push_str(&format!(
            "Fear/Greed: {} ({}) — {}\n",
            self.snapshot.fng_value, self.snapshot.fng_label, fng_guidance
        ));

        // Global market
        if self.snapshot.total_market_cap_usd > 0.0 {
            out.push_str(&format!(
                "Market: ${:.0}B total cap | BTC dominance: {:.1}% | 24h: {:+.2}%\n",
                self.snapshot.total_market_cap_usd / 1e9,
                self.snapshot.btc_dominance,
                self.snapshot.market_cap_change_24h,
            ));
        }

        // Trending coins (shows what the market is watching)
        if !self.snapshot.trending.is_empty() {
            let trending_str: Vec<String> = self.snapshot.trending.iter()
                .take(7)
                .map(|c| format!("{}({:+.1}%)", c.symbol, c.change_24h))
                .collect();
            out.push_str(&format!("Trending: {}\n", trending_str.join(" ")));
        }

        out
    }

    /// One-liner for exit prompts (compact)
    #[allow(dead_code)]
    pub fn build_news_short(&self) -> String {
        if self.snapshot.fng_label == "unknown" {
            return String::new();
        }
        format!(
            "Sentiment: FnG={} ({}) | Mkt {:+.1}%/24h",
            self.snapshot.fng_value,
            self.snapshot.fng_label,
            self.snapshot.market_cap_change_24h,
        )
    }

}
