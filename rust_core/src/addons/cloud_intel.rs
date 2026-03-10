//! Cloud Intelligence — Atlas (DistilRoBERTa ONNX) Sentiment Scanner.
//!
//! Atlas is the sentiment analyst of the 3-AI trading desk:
//!   - Fetches crypto headlines from CryptoPanic RSS + 3 RSS sources + CryptoCompare
//!   - Analyzes sentiment per coin via DistilRoBERTa ONNX model (atlas_server.py on port 8083)
//!   - Feeds sentiment + news events to both AI models
//!
//! Architecture: Rust fetches headlines → HTTP to Atlas server → shared state.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::config::CachedEnv;
use crate::vision_scanner::{CoinSwapSuggestion, SharedSwapQueue};

// ── Shared State ──────────────────────────────────────────────

/// Per-coin sentiment score from Atlas (NPU) analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoinSentiment {
    pub symbol: String,
    pub score: f64,        // -1.0 (very bearish) to +1.0 (very bullish)
    pub summary: String,   // brief reason / headline
    pub updated_at: f64,   // Unix timestamp
}

/// Breaking news / event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewsEvent {
    pub headline: String,
    pub impact: String,    // "high", "medium", "low"
    pub affected_coins: Vec<String>,
    pub sentiment: f64,    // -1.0 to +1.0
    pub detected_at: f64,
}

/// Live ticker data from Kraken WS v2 ticker channel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketTicker {
    pub symbol: String,
    pub last_price: f64,
    pub change_pct: f64,
    pub volume_24h: f64,
    pub high_24h: f64,
    pub low_24h: f64,
    pub vwap: f64,
    pub bid: f64,
    pub ask: f64,
    pub updated_at: f64,
}

/// Shared cloud intel state, read by trading_loop.
pub struct CloudIntelState {
    pub sentiments: HashMap<String, CoinSentiment>,
    pub events: Vec<NewsEvent>,
    pub market_data: HashMap<String, MarketTicker>,
    pub last_sentiment_scan: f64,
    pub last_news_scan: f64,
}

impl Default for CloudIntelState {
    fn default() -> Self {
        Self {
            sentiments: HashMap::new(),
            events: Vec::new(),
            market_data: HashMap::new(),
            last_sentiment_scan: 0.0,
            last_news_scan: 0.0,
        }
    }
}

pub type SharedCloudIntel = Arc<RwLock<CloudIntelState>>;

pub fn new_shared_intel() -> SharedCloudIntel {
    Arc::new(RwLock::new(CloudIntelState::default()))
}

// ── Coin Detection (ported from Python) ──────────────────────

/// Map of coin names to symbols for headline detection.
const COIN_NAMES: &[(&str, &str)] = &[
    ("bitcoin", "BTC"), ("ethereum", "ETH"), ("solana", "SOL"), ("ripple", "XRP"),
    ("cardano", "ADA"), ("avalanche", "AVAX"), ("chainlink", "LINK"), ("polkadot", "DOT"),
    ("dogecoin", "DOGE"), ("pepe", "PEPE"), ("cosmos", "ATOM"), ("near protocol", "NEAR"),
    ("near", "NEAR"), ("uniswap", "UNI"), ("aave", "AAVE"), ("injective", "INJ"),
    ("sui", "SUI"), ("litecoin", "LTC"), ("the graph", "GRT"), ("ondo", "ONDO"),
    ("celestia", "TIA"), ("render", "RENDER"), ("aptos", "APT"),
];

fn detect_coins(text: &str, coin_set: &[String]) -> Vec<String> {
    let lower = text.to_lowercase();
    let mut found = std::collections::HashSet::new();

    // Match by name
    for &(name, sym) in COIN_NAMES {
        if coin_set.iter().any(|c| c == sym) && lower.contains(name) {
            found.insert(sym.to_string());
        }
    }

    // Match by symbol (case-insensitive for longer symbols, exact for short ones)
    for sym in coin_set {
        if found.contains(sym.as_str()) { continue; }
        if sym.len() <= 3 {
            // Short symbols: require word boundary
            if text.contains(sym.as_str()) {
                found.insert(sym.clone());
            }
        } else if lower.contains(&sym.to_lowercase()) {
            found.insert(sym.clone());
        }
    }

    found.into_iter().collect()
}

/// Detect ALL coins mentioned in text — including coins NOT in the watchlist.
/// Used by Atlas to discover opportunities on untracked coins.
fn detect_all_coins(text: &str) -> Vec<String> {
    let lower = text.to_lowercase();
    let mut found = std::collections::HashSet::new();

    for &(name, sym) in COIN_NAMES {
        if lower.contains(name) {
            found.insert(sym.to_string());
        }
    }

    // Also check symbol mentions for longer symbols
    for &(_name, sym) in COIN_NAMES {
        if found.contains(sym) { continue; }
        if sym.len() > 3 && lower.contains(&sym.to_lowercase()) {
            found.insert(sym.to_string());
        }
    }

    found.into_iter().collect()
}

// ── Impact Classification ────────────────────────────────────

const HIGH_KEYWORDS: &[&str] = &[
    // Negative high-impact
    "sec ", "cftc", "lawsuit", "settlement", "charges", "fine", "ban",
    "rejection", "court", "probe", "hack", "exploit",
    "drain", "breach", "stolen", "attack", "vulnerability", "rug",
    "delist", "suspends", "halts trading", "halt", "outage", "bankruptcy",
    "insolvent", "liquidation", "default", "frozen withdrawals",
    // Positive high-impact (GAINER SIGNALS)
    "approval", "approved", "etf", "all-time high", "ath", "record high",
    "breakout", "surge", "soar", "rally", "moon", "pump",
    "institutional", "blackrock", "fidelity", "grayscale",
    "billion", "adoption", "mass adoption",
];

const MEDIUM_KEYWORDS: &[&str] = &[
    "partnership", "integration", "mainnet", "testnet", "upgrade", "hard fork",
    "proposal", "governance", "vote", "funding", "raises", "investment",
    "launch", "airdrop", "staking", "burn", "tokenomics",
    // Gainer signals
    "bullish", "outperform", "accumulation", "whale", "whale buy",
    "listing", "listed on", "coinbase", "binance",
    "tvl", "volume spike", "breakout", "golden cross",
];

fn classify_impact(title: &str) -> Option<&'static str> {
    let lower = title.to_lowercase();
    for kw in HIGH_KEYWORDS {
        if lower.contains(kw) { return Some("high"); }
    }
    for kw in MEDIUM_KEYWORDS {
        if lower.contains(kw) { return Some("medium"); }
    }
    None
}

/// Simple keyword-based sentiment as fallback (no NPU needed).
fn keyword_sentiment(headlines: &[String]) -> f64 {
    let pos_words = ["surge", "rally", "bullish", "gain", "rise", "soar", "pump",
                     "record", "ath", "adoption", "partnership", "launch", "upgrade",
                     "approval", "buy"];
    let neg_words = ["crash", "drop", "bearish", "fall", "dump", "hack", "lawsuit",
                     "sec", "exploit", "breach", "ban", "outage", "liquidation",
                     "bankruptcy", "sell"];

    let (mut pos, mut neg) = (0u32, 0u32);
    for h in headlines {
        let lower = h.to_lowercase();
        for w in &pos_words { if lower.contains(w) { pos += 1; } }
        for w in &neg_words { if lower.contains(w) { neg += 1; } }
    }
    let total = pos + neg;
    if total == 0 { return 0.0; }
    ((pos as f64) - (neg as f64)) / (total as f64)
}

// ── RSS Headline Fetching ────────────────────────────────────

const RSS_FEEDS: &[(&str, &str)] = &[
    ("CryptoPanic", "https://cryptopanic.com/news/rss/"),
    ("CoinTelegraph", "https://cointelegraph.com/rss"),
    ("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
    ("Decrypt", "https://decrypt.co/feed"),
];
const CRYPTOCOMPARE_URL: &str = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN";

#[derive(Debug)]
#[allow(dead_code)]
struct Headline {
    title: String,
    _source: String,
}

/// Fetch RSS feed and extract headlines (simple XML parsing, no crate needed).
async fn fetch_rss(client: &reqwest::Client, name: &str, url: &str) -> Vec<Headline> {
    let mut headlines = Vec::new();
    let resp = match client.get(url).send().await {
        Ok(r) if r.status().is_success() => r,
        Ok(r) => {
            tracing::debug!("[RSS] {name} HTTP {}", r.status());
            return headlines;
        }
        Err(e) => {
            tracing::debug!("[RSS] {name} failed: {e}");
            return headlines;
        }
    };

    let text = match resp.text().await {
        Ok(t) => t,
        Err(_) => return headlines,
    };

    // Simple XML parsing: extract <title> from each <item>
    for item_block in text.split("<item>").skip(1).take(30) {
        if let (Some(ts), Some(te)) = (item_block.find("<title>"), item_block.find("</title>")) {
            let title_start = ts + 7; // "<title>".len()
            if title_start < te {
                let mut title = item_block[title_start..te].to_string();
                // Strip CDATA
                if title.starts_with("<![CDATA[") {
                    title = title.trim_start_matches("<![CDATA[")
                        .trim_end_matches("]]>")
                        .to_string();
                }
                let title = title.trim().to_string();
                if !title.is_empty() {
                    headlines.push(Headline { title, _source: name.to_string() });
                }
            }
        }
    }

    tracing::debug!("[RSS] {name}: {} headlines", headlines.len());
    headlines
}

/// Fetch headlines from CryptoCompare JSON API.
async fn fetch_cryptocompare(client: &reqwest::Client) -> Vec<Headline> {
    let mut headlines = Vec::new();
    let resp = match client.get(CRYPTOCOMPARE_URL).send().await {
        Ok(r) if r.status().is_success() => r,
        _ => return headlines,
    };

    if let Ok(json) = resp.json::<serde_json::Value>().await {
        if let Some(data) = json.get("Data").and_then(|d| d.as_array()) {
            for item in data.iter().take(30) {
                if let Some(title) = item.get("title").and_then(|t| t.as_str()) {
                    if !title.is_empty() {
                        headlines.push(Headline {
                            title: title.to_string(),
                            _source: "CryptoCompare".to_string(),
                        });
                    }
                }
            }
        }
    }

    tracing::debug!("[CRYPTOCOMPARE] {} headlines", headlines.len());
    headlines
}

/// Fetch all headlines from all free sources.
async fn fetch_all_headlines(client: &reqwest::Client) -> Vec<Headline> {
    let mut tasks = Vec::new();
    for &(name, url) in RSS_FEEDS {
        let c = client.clone();
        let n = name.to_string();
        let u = url.to_string();
        tasks.push(tokio::spawn(async move { fetch_rss(&c, &n, &u).await }));
    }
    let cc_client = client.clone();
    tasks.push(tokio::spawn(async move { fetch_cryptocompare(&cc_client).await }));

    let mut all = Vec::new();
    for task in tasks {
        if let Ok(mut hl) = task.await {
            all.append(&mut hl);
        }
    }

    // Deduplicate by title
    let mut seen = std::collections::HashSet::new();
    all.retain(|h| {
        let key = h.title.to_lowercase().chars().take(80).collect::<String>();
        seen.insert(key)
    });

    tracing::info!("[CLOUD-INTEL] Fetched {} unique headlines from {} sources",
        all.len(), RSS_FEEDS.len() + 1);
    all
}

// ── Atlas HTTP Sentiment Analysis ────────────────────────────

/// Response from Atlas server /batch endpoint.
#[derive(Debug, Deserialize)]
struct AtlasResult {
    label: String,       // "positive", "negative", "neutral"
    score: f64,          // confidence 0.0-1.0
    #[allow(dead_code)]
    scores: Option<AtlasScores>,
}

#[derive(Debug, Deserialize)]
struct AtlasScores {
    positive: f64,
    negative: f64,
    #[allow(dead_code)]
    neutral: f64,
}

/// Analyze sentiment for all coins via Atlas HTTP server (DistilRoBERTa ONNX).
/// Sends headlines in batch, aggregates per-coin scores.
async fn atlas_batch_sentiment(
    client: &reqwest::Client,
    atlas_url: &str,
    coin_headlines: &HashMap<String, Vec<String>>,
) -> HashMap<String, f64> {
    if coin_headlines.is_empty() { return HashMap::new(); }

    // Collect unique headlines and map them back to coins
    let mut all_headlines: Vec<String> = Vec::new();
    let mut headline_to_coins: HashMap<String, Vec<String>> = HashMap::new();

    for (sym, headlines) in coin_headlines {
        for hl in headlines.iter().take(5) {
            let key = hl.clone();
            headline_to_coins.entry(key.clone())
                .or_default()
                .push(sym.clone());
            if !all_headlines.contains(&key) {
                all_headlines.push(key);
            }
        }
    }

    if all_headlines.is_empty() { return HashMap::new(); }

    // Call Atlas batch endpoint
    let body = serde_json::json!({ "texts": all_headlines });
    let resp = match client.post(&format!("{}/batch", atlas_url))
        .json(&body)
        .send()
        .await
    {
        Ok(r) if r.status().is_success() => r,
        Ok(r) => {
            tracing::warn!("[CLOUD-INTEL] AI_3 HTTP {} — falling back to keywords", r.status());
            return HashMap::new();
        }
        Err(e) => {
            tracing::warn!("[CLOUD-INTEL] AI_3 unreachable: {e} — falling back to keywords");
            return HashMap::new();
        }
    };

    let atlas_results: Vec<AtlasResult> = match resp.json().await {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!("[CLOUD-INTEL] AI_3 JSON parse error: {e}");
            return HashMap::new();
        }
    };

    // Aggregate per-coin: average (positive - negative) scores
    let mut coin_scores: HashMap<String, Vec<f64>> = HashMap::new();

    for (i, result) in atlas_results.iter().enumerate() {
        if i >= all_headlines.len() { break; }
        let hl = &all_headlines[i];

        // Convert classification to -1.0 to +1.0 score
        let score = if let Some(ref scores) = result.scores {
            scores.positive - scores.negative  // ranges -1.0 to +1.0
        } else {
            match result.label.as_str() {
                "positive" => result.score,
                "negative" => -result.score,
                _ => 0.0,
            }
        };

        if let Some(coins) = headline_to_coins.get(hl) {
            for sym in coins {
                coin_scores.entry(sym.clone()).or_default().push(score);
            }
        }
    }

    // Average per coin
    let mut results = HashMap::new();
    for (sym, scores) in &coin_scores {
        if scores.is_empty() { continue; }
        let avg = scores.iter().sum::<f64>() / scores.len() as f64;
        results.insert(sym.clone(), avg.clamp(-1.0, 1.0));
    }

    tracing::info!("[CLOUD-INTEL] AI_3 scored {}/{} coins ({} headlines)",
        results.len(), coin_headlines.len(), all_headlines.len());

    results
}

// ── Atlas Scan (fetches headlines + runs NPU sentiment) ──────

async fn run_atlas_scan(
    client: &reqwest::Client,
    atlas_url: &str,
    coins: &[String],
    intel: &SharedCloudIntel,
    swap_queue: Option<&SharedSwapQueue>,
) {
    let headlines = fetch_all_headlines(client).await;
    if headlines.is_empty() { return; }

    let now = now_ts();

    // Phase 1: Detect coins from headlines
    let mut coin_headlines: HashMap<String, Vec<String>> = HashMap::new();
    let mut event_candidates: Vec<(String, Vec<String>)> = Vec::new(); // (title, affected_coins)

    for hl in &headlines {
        let affected = detect_coins(&hl.title, coins);
        if affected.is_empty() { continue; }

        for sym in &affected {
            coin_headlines.entry(sym.clone())
                .or_default()
                .push(hl.title.clone());
        }
        event_candidates.push((hl.title.clone(), affected));
    }

    // Phase 2: Atlas HTTP batch sentiment analysis
    let headlines_for_atlas: HashMap<String, Vec<String>> = coin_headlines.iter()
        .map(|(sym, hl)| (sym.clone(), hl.iter().take(5).cloned().collect()))
        .collect();
    let npu_scores = atlas_batch_sentiment(client, atlas_url, &headlines_for_atlas).await;

    // Phase 3: Build results
    let mut sentiments: HashMap<String, CoinSentiment> = HashMap::new();
    let mut events: Vec<NewsEvent> = Vec::new();

    for (sym, hl_list) in &coin_headlines {
        let mut score = npu_scores.get(sym).copied()
            .unwrap_or_else(|| keyword_sentiment(hl_list));

        // Asymmetric shrinkage: conservative on negative, aggressive on positive
        // Positive signals = opportunities — let them through stronger
        let n = hl_list.len();
        if score > 0.0 {
            // Bullish: light shrinkage — Atlas found a potential gainer
            if n == 1 { score *= 0.70; }
            else if n == 2 { score *= 0.85; }
        } else {
            // Bearish: heavy shrinkage — be cautious about false negatives
            if n == 1 { score *= 0.50; }
            else if n == 2 { score *= 0.75; }
        }

        score = score.clamp(-1.0, 1.0);

        // Best headline as summary
        let summary = hl_list.first()
            .map(|s| s.chars().take(80).collect::<String>())
            .unwrap_or_default();

        sentiments.insert(sym.clone(), CoinSentiment {
            symbol: sym.clone(),
            score: (score * 1000.0).round() / 1000.0,
            summary,
            updated_at: now,
        });
    }

    // Build events from impact keywords
    let mut seen_events = std::collections::HashSet::new();
    for (title, affected) in &event_candidates {
        let impact = match classify_impact(title) {
            Some(i) => i,
            None => continue,
        };
        let key: String = title.chars().take(60).collect();
        if !seen_events.insert(key) { continue; }

        let coin_score = affected.first()
            .and_then(|s| sentiments.get(s))
            .map(|s| s.score)
            .unwrap_or(0.0);

        events.push(NewsEvent {
            headline: title.chars().take(120).collect(),
            impact: impact.to_string(),
            affected_coins: affected.clone(),
            sentiment: coin_score,
            detected_at: now,
        });
    }

    // Update shared state
    let mut state = intel.write().await;
    for (sym, sent) in &sentiments {
        state.sentiments.insert(sym.clone(), sent.clone());
    }
    state.last_sentiment_scan = now;
    state.events.retain(|e| now - e.detected_at < 86400.0);
    for ev in &events {
        state.events.push(ev.clone());
    }
    state.last_news_scan = now;
    drop(state);

    // Log
    let top = sentiments.values()
        .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal));
    tracing::info!(
        "[CLOUD-INTEL] AI_3 scan: {} headlines → {} coins scored, {} events | device=ONNX | top: {} ({:+.2})",
        headlines.len(),
        sentiments.len(),
        events.len(),
        top.map(|s| s.symbol.as_str()).unwrap_or("?"),
        top.map(|s| s.score).unwrap_or(0.0),
    );

    for e in &events {
        tracing::info!(
            "[CLOUD-INTEL]   [{:>6}] {:+.2} | {} | {}",
            e.impact, e.sentiment,
            e.affected_coins.join(","),
            e.headline.chars().take(80).collect::<String>()
        );
    }

    // ── Atlas Discovery: find opportunities on UNTRACKED coins (zero extra HTTP) ──
    if let Some(sq) = swap_queue {
        let tracked_upper: Vec<String> = coins.iter().map(|c| c.to_uppercase()).collect();
        let mut best: Option<(String, f64, String)> = None; // (symbol, score, headline)

        for hl in &headlines {
            let impact = classify_impact(&hl.title);
            if impact != Some("high") { continue; } // only high-impact
            for sym in detect_all_coins(&hl.title) {
                if tracked_upper.contains(&sym) { continue; }
                let score = keyword_sentiment(&[hl.title.clone()]);
                if score >= 0.40 {
                    if best.as_ref().map_or(true, |b| score > b.1) {
                        best = Some((sym, score, hl.title.chars().take(80).collect()));
                    }
                }
            }
        }

        if let Some((add_sym, score, headline)) = best {
            let anchors = ["BTC", "ETH"];
            let weakest = sentiments.values()
                .filter(|s| !anchors.contains(&s.symbol.as_str()))
                .min_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal))
                .map(|s| s.symbol.clone());
            if let Some(remove_sym) = weakest {
                tracing::info!("[AI3-DISCOVERY] {} ({:+.2}) → swap for {}", add_sym, score, remove_sym);
                let mut q = sq.write().await;
                q.push(CoinSwapSuggestion {
                    add: vec![add_sym.clone()],
                    remove: vec![remove_sym],
                    reason: format!("AI_3: {} {:+.2} | {}", add_sym, score, headline),
                    timestamp: now,
                });
            }
        }
    }

    // Persist to disk
    let _ = std::fs::create_dir_all("data/cloud_intel");
    if !sentiments.is_empty() {
        if let Ok(json) = serde_json::to_string_pretty(&sentiments) {
            let path = format!(
                "data/cloud_intel/sentiment_{}.json",
                chrono::Utc::now().format("%Y%m%d_%H%M%S")
            );
            let _ = std::fs::write(&path, &json);
        }
    }
}

// ── Strategy Validation (on-demand, still uses NVIDIA if available) ──

pub async fn validate_strategy(
    backtest_json: &str,
) -> anyhow::Result<String> {
    let env = CachedEnv::snapshot();
    let api_key = env.get_str("NVIDIA_API_KEY", "");
    if api_key.is_empty() {
        anyhow::bail!("NVIDIA_API_KEY not set — strategy validation requires cloud model");
    }
    let model = env.get_str("NVIDIA_MODEL", "nvidia/llama-3.3-nemotron-super-49b-v1.5");

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()?;

    let system_prompt = r#"You are a quantitative trading strategy reviewer. Analyze the backtest results provided and give your professional assessment.

Evaluate:
1. Win rate sustainability (is it statistically significant?)
2. Profit factor adequacy (>1.5 is good, >2.0 is excellent)
3. Max drawdown risk (>20% is concerning)
4. Fee impact (are profits real after fees?)
5. Strategy robustness (does it work across different coins?)

Output a structured review:
- GRADE: A/B/C/D/F
- DEPLOY: YES/NO/CAUTIOUS
- STRENGTHS: bullet points
- WEAKNESSES: bullet points
- RECOMMENDATIONS: specific improvements"#;

    let user_prompt = format!(
        "Review this crypto trading strategy backtest:\n\n{backtest_json}"
    );

    let nvidia_url = env.get_str("NVIDIA_API_URL", "https://integrate.api.nvidia.com/v1");
    let response: String = crate::ai_bridge::infer_cloud(
        &client, &api_key, &model,
        system_prompt, &user_prompt,
        0.3_f32, 2048_i32, &nvidia_url,
    ).await.map_err(|e: String| anyhow::anyhow!(e))?;

    let path = format!(
        "data/backtest_results/validation_{}.txt",
        chrono::Utc::now().format("%Y%m%d_%H%M%S")
    );
    std::fs::create_dir_all("data/backtest_results")?;
    std::fs::write(&path, &response)?;
    tracing::info!("[CLOUD-INTEL] Strategy validation saved to {path}");

    Ok(response)
}

// ── Public API for Trading Loop ───────────────────────────────

/// Update market ticker from Kraken WS v2 ticker channel.
pub async fn update_market_ticker(intel: &SharedCloudIntel, ticker: MarketTicker) {
    let mut state = intel.write().await;
    state.market_data.insert(ticker.symbol.clone(), ticker);
}

/// Build CLOUD_SENT + NEWS_ALERT prompt lines for a given coin.
pub fn build_cloud_prompt(state: &CloudIntelState, sym: &str, event_sentiment_floor: f64) -> String {
    let now = now_ts();
    let mut out = String::new();

    if let Some(sent) = state.sentiments.get(sym) {
        let age_min = (now - sent.updated_at) / 60.0;
        if age_min < 120.0 {
            let price_tag = state.market_data.get(sym)
                .filter(|t| (now - t.updated_at) < 300.0)
                .map(|t| format!("|{}{:+.1}%", format_price_compact(t.last_price), t.change_pct))
                .unwrap_or_default();
            out.push_str(&format!(
                "\nCLOUD_SENT:{:+.2}|{}|{:.0}m_ago{}",
                sent.score, sent.summary, age_min, price_tag
            ));
        }
    }

    for ev in &state.events {
        if !ev.affected_coins.iter().any(|c| c.eq_ignore_ascii_case(sym)) {
            continue;
        }
        let age_min = (now - ev.detected_at) / 60.0;
        if age_min >= 120.0 { continue; }

        if ev.impact == "high" && ev.sentiment >= 0.30 {
            // POSITIVE high-impact = OPPORTUNITY for AI_1 to act on
            out.push_str(&format!(
                "\nOPPORTUNITY:{}|{:+.2}|{:.0}m_ago",
                ev.headline, ev.sentiment, age_min
            ));
        } else if ev.impact == "high" && ev.sentiment < event_sentiment_floor {
            // NEGATIVE high-impact = WARNING
            out.push_str(&format!(
                "\nNEWS_ALERT:{}|{:+.2}|{:.0}m_ago",
                ev.headline, ev.sentiment, age_min
            ));
        } else if ev.impact == "medium" && ev.sentiment >= 0.40 {
            // MEDIUM positive = worth mentioning
            out.push_str(&format!(
                "\nSCOUT_TIP:{}|{:+.2}|{:.0}m_ago",
                ev.headline, ev.sentiment, age_min
            ));
        }
    }

    out
}

// ── Main Scanner Loop ─────────────────────────────────────────

pub async fn run_cloud_intel(
    intel: SharedCloudIntel,
    active_coins: Arc<RwLock<Vec<String>>>,
    swap_queue: SharedSwapQueue,
) {
    let env = CachedEnv::snapshot();
    let scan_interval: u64 = env.get_parsed("CLOUD_SENTIMENT_INTERVAL_SEC", 300);
    let atlas_url = env.get_str("AI3_HOST", "http://127.0.0.1:8083");

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(15))
        .user_agent("kraken-hybrad9/2.0 AI3-ONNX")
        .build()
        .unwrap_or_default();

    let coins = active_coins.read().await.clone();
    let interval_min = scan_interval as f64 / 60.0;
    tracing::info!(
        "[CLOUD-INTEL] Starting AI_3 Scout (ONNX HTTP, url={}, interval={:.0}min, coins={})",
        atlas_url, interval_min, coins.len()
    );
    tracing::info!(
        "[CLOUD-INTEL] Free sources: CryptoPanic RSS + CoinTelegraph + CoinDesk + Decrypt + CryptoCompare"
    );
    tracing::info!(
        "[CLOUD-INTEL] AI_3 can discover untracked coins and push swap suggestions to AI_1"
    );

    // Health check Atlas
    match client.get(&format!("{}/health", atlas_url)).send().await {
        Ok(r) if r.status().is_success() => {
            tracing::info!("[CLOUD-INTEL] AI_3 server healthy at {}", atlas_url);
        }
        _ => {
            tracing::warn!("[CLOUD-INTEL] AI_3 server not reachable at {} — will use keyword fallback", atlas_url);
        }
    }

    // ── Auto-cleanup: delete cloud_intel files older than 48h ──────
    let keep_hours: u64 = env.get_parsed("CLOUD_INTEL_KEEP_HOURS", 48);
    let cutoff = std::time::SystemTime::now()
        .checked_sub(std::time::Duration::from_secs(keep_hours * 3600))
        .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
    let mut deleted = 0usize;
    if let Ok(entries) = std::fs::read_dir("data/cloud_intel") {
        for entry in entries.flatten() {
            if let Ok(meta) = entry.metadata() {
                if let Ok(modified) = meta.modified() {
                    if modified < cutoff {
                        let _ = std::fs::remove_file(entry.path());
                        deleted += 1;
                    }
                }
            }
        }
    }
    if deleted > 0 {
        tracing::info!("[CLOUD-INTEL] Pruned {} old files (>{keep_hours}h)", deleted);
    }

    // Initial scan
    run_atlas_scan(&client, &atlas_url, &coins, &intel, Some(&swap_queue)).await;

    // Main loop
    let mut ticker = tokio::time::interval(
        tokio::time::Duration::from_secs(scan_interval)
    );
    ticker.tick().await;

    loop {
        ticker.tick().await;
        // Read current dynamic coin list (changes when AI_1 rotates coins)
        let current_coins = active_coins.read().await.clone();
        run_atlas_scan(&client, &atlas_url, &current_coins, &intel, Some(&swap_queue)).await;
    }
}

/// Format price compactly for prompt injection.
pub fn format_price_compact(price: f64) -> String {
    match price {
        p if p >= 100_000.0 => format!("${:.0}K", p / 1_000.0),
        p if p >= 1_000.0   => format!("${:.1}K", p / 1_000.0),
        p if p >= 1.0       => format!("${:.2}", p),
        p if p >= 0.001     => format!("${:.4}", p),
        p                   => format!("${:.6}", p),
    }
}

fn now_ts() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}
