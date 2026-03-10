//! Nemo Live Chat — HTTP endpoint that lets you talk to Nemo with live data.
//!
//! Runs on localhost:9090/chat. Accepts POST with {"message":"your text"}.
//! Injects live features from all 37 coins so Nemo can see real-time data.

use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::RwLock;
use tokio::time::Duration;

/// Shared state updated by the main trading loop every tick.
pub struct LiveState {
    pub features: HashMap<String, serde_json::Value>,
    pub positions: Vec<serde_json::Value>,
    pub available_usd: f64,
    pub equity: f64,
    pub total_value: f64,
}

impl Default for LiveState {
    fn default() -> Self {
        Self {
            features: HashMap::new(),
            positions: Vec::new(),
            available_usd: 0.0,
            equity: 0.0,
            total_value: 0.0,
        }
    }
}

pub type SharedLiveState = Arc<RwLock<LiveState>>;

pub fn new_shared_state() -> SharedLiveState {
    Arc::new(RwLock::new(LiveState::default()))
}

/// Build a compact summary from live features.
/// Shows holdings with full detail, then top 10 coins by book signal strength.
fn build_live_context(state: &LiveState) -> String {
    let mut lines = Vec::new();
    lines.push(format!(
        "PORTFOLIO: ${:.2} total | ${:.2} cash | ${:.2} equity | {} positions",
        state.total_value,
        state.available_usd,
        state.equity,
        state.positions.len()
    ));

    // Holdings with full detail
    let held_syms: Vec<String> = state.positions.iter()
        .filter_map(|p| p["symbol"].as_str().map(String::from))
        .collect();

    if !state.positions.is_empty() {
        lines.push(String::new());
        lines.push("=== YOUR HOLDINGS (ranked by value) ===".to_string());
        let mut sorted_pos = state.positions.clone();
        sorted_pos.sort_by(|a, b| {
            b["value"].as_f64().unwrap_or(0.0)
                .partial_cmp(&a["value"].as_f64().unwrap_or(0.0))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for p in &sorted_pos {
            let sym = p["symbol"].as_str().unwrap_or("?");
            let qty = p["qty"].as_f64().unwrap_or(0.0);
            let entry = p["entry"].as_f64().unwrap_or(0.0);
            let now = p["now"].as_f64().unwrap_or(0.0);
            let val = p["value"].as_f64().unwrap_or(0.0);
            let pnl = p["pnl_pct"].as_f64().unwrap_or(0.0);
            // Get book data for this holding
            let (imb, buy_r, spread) = state.features.get(sym)
                .map(|f| (
                    f.get("book_imbalance").and_then(|v| v.as_f64()).unwrap_or(0.0),
                    f.get("buy_ratio").and_then(|v| v.as_f64()).unwrap_or(0.0),
                    f.get("spread_pct").and_then(|v| v.as_f64()).unwrap_or(0.0),
                ))
                .unwrap_or((0.0, 0.0, 0.0));
            lines.push(format!(
                "  {sym:6} ${val:.2} | qty={qty} entry=${entry} now=${now} PnL={pnl:+.1}% | Book: imb={imb:+.3} buy={buy_r:.2} spread={spread:.3}%"
            ));
        }
    }

    // Top 10 non-held coins by absolute book imbalance
    lines.push(String::new());
    lines.push("=== TOP 14 WATCHLIST (strongest book signals) ===".to_string());

    let mut coin_scores: Vec<(String, f64, &serde_json::Value)> = Vec::new();
    for (sym, f) in &state.features {
        if sym.starts_with("__") || held_syms.contains(sym) {
            continue;
        }
        let price = f.get("price").and_then(|v| v.as_f64()).unwrap_or(0.0);
        if price <= 0.0 {
            continue;
        }
        let imb = f.get("book_imbalance").and_then(|v| v.as_f64()).unwrap_or(0.0);
        coin_scores.push((sym.clone(), imb.abs(), f));
    }
    coin_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    coin_scores.truncate(14);

    for (sym, _, f) in &coin_scores {
        let price = f.get("price").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let rsi = f.get("rsi").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let trend = f.get("trend_score").and_then(|v| v.as_f64()).unwrap_or(0.0) as i32;
        let momo = f.get("momentum_score").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let imb = f.get("book_imbalance").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let buy_r = f.get("buy_ratio").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let spread = f.get("spread_pct").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let _slip = f.get("slippage_buy").and_then(|v| v.as_f64()).unwrap_or(0.0);
        // DEXScreener data
        let dex_vol = f.get("dex_vol_24h").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let dex_liq = f.get("dex_liquidity").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let dex_chg = f.get("dex_chg_24h").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let dex_buys = f.get("dex_buys").and_then(|v| v.as_u64()).unwrap_or(0);
        let dex_sells = f.get("dex_sells").and_then(|v| v.as_u64()).unwrap_or(0);
        let dex_fdv = f.get("dex_fdv").and_then(|v| v.as_f64()).unwrap_or(0.0);

        let mut line = format!(
            "  {sym:10} ${price} | RSI={rsi:.1} Trend={trend:+} Momo={momo:+.4} | Book: imb={imb:+.3} buy={buy_r:.2} spread={spread:.3}%"
        );
        if dex_vol > 0.0 {
            line.push_str(&format!(
                " | DEX: vol=${:.0}K liq=${:.0}K chg={:+.1}% b/s={}/{} fdv=${:.0}K",
                dex_vol / 1000.0,
                dex_liq / 1000.0,
                dex_chg,
                dex_buys,
                dex_sells,
                dex_fdv / 1000.0,
            ));
        }
        lines.push(line);
    }

    lines.join("\n")
}

/// Spawn the Nemo chat HTTP server on localhost:9090.
pub fn spawn_chat_server(state: SharedLiveState) {
    let ollama_url = std::env::var("OLLAMA_HOST")
        .unwrap_or_else(|_| "http://127.0.0.1:11434".to_string());
    let model = std::env::var("OLLAMA_MODEL")
        .unwrap_or_else(|_| "qwen2.5-14b-instruct".to_string());
    let port: u16 = std::env::var("AI_CHAT_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(9090);

    tokio::spawn(async move {
        let listener = match TcpListener::bind(format!("127.0.0.1:{port}")).await {
            Ok(l) => {
                tracing::info!("[AI-CHAT] Live chat server on http://127.0.0.1:{port}/chat");
                l
            }
            Err(e) => {
                tracing::error!("[AI-CHAT] Failed to bind port {port}: {e}");
                return;
            }
        };

        let client = loop {
            match reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(60))
                .build()
            {
                Ok(c) => break c,
                Err(e) => {
                    tracing::error!("[AI-CHAT] failed to build HTTP client: {e}; retrying in 5s");
                    tokio::time::sleep(Duration::from_secs(5)).await;
                }
            }
        };

        loop {
            let (mut socket, addr) = match listener.accept().await {
                Ok(s) => s,
                Err(_) => continue,
            };

            let state = state.clone();
            let client = client.clone();
            let ollama_url = ollama_url.clone();
            let model = model.clone();

            tokio::spawn(async move {
                const MAX_BODY: usize = 65_536;
                let mut buf = vec![0u8; MAX_BODY];
                let n = match tokio::time::timeout(
                    Duration::from_secs(10),
                    socket.read(&mut buf),
                ).await {
                    Ok(Ok(n)) if n > 0 => n,
                    _ => return,
                };
                let request = String::from_utf8_lossy(&buf[..n]);

                // Reject cross-origin requests: Host must be 127.0.0.1 or localhost
                let host_ok = request.lines().any(|l| {
                    let l = l.to_ascii_lowercase();
                    l.starts_with("host:") && (l.contains("127.0.0.1") || l.contains("localhost"))
                });
                if !host_ok {
                    let resp = "HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n";
                    let _ = socket.write_all(resp.as_bytes()).await;
                    return;
                }

                // Parse HTTP: find body after \r\n\r\n
                let body = request
                    .find("\r\n\r\n")
                    .map(|i| &request[i + 4..])
                    .unwrap_or("");

                // Check it's POST /chat
                if !request.starts_with("POST /chat") {
                    // Serve a simple info page for GET /
                    let info = "AI Live Chat - POST /chat with {\"message\":\"your text\"}";
                    let resp = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: {}\r\n\r\n{}",
                        info.len(),
                        info
                    );
                    let _ = socket.write_all(resp.as_bytes()).await;
                    return;
                }

                // Parse JSON message
                let user_msg = match serde_json::from_str::<serde_json::Value>(body) {
                    Ok(v) => v["message"]
                        .as_str()
                        .unwrap_or("Hello AI!")
                        .to_string(),
                    Err(_) => {
                        let err = r#"{"error":"send JSON: {\"message\":\"your text\"}"}"#;
                        let resp = format!(
                            "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\n\
                             Content-Length: {}\r\n\r\n{}",
                            err.len(),
                            err
                        );
                        let _ = socket.write_all(resp.as_bytes()).await;
                        return;
                    }
                };

                tracing::info!("[AI-CHAT] from {addr}: {}", &user_msg[..user_msg.len().min(80)]);

                // Build context from live state
                let live_context = {
                    let st = state.read().await;
                    build_live_context(&st)
                };

                // Think mode support (Qwen3)
                let think_mode = std::env::var("THINK_MODE")
                    .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                    .unwrap_or(false);

                // Live data goes in system prompt so it doesn't interfere with /no_think control token
                let sys_content = format!(
                    "You are Nemo, the autonomous trading intelligence of the Hybrad9 desk. \
                    Your name is Nemo — always refer to yourself as Nemo. \
                    You run on an RTX 5060 Ti GPU with Qwen3 4B via llama-server. \
                    Your brain stack: native CUDA GPU math (Hurst, entropy, correlation), \
                    GBDT machine learning (learns from every trade), 44-feature math scoring engine, \
                    and you (Nemo) for contextual reasoning. \
                    You have a Market Brain (BTC gravity, breadth, sectors, regime, greed index, multi-scale fingerprint) \
                    and a Conscious Brain (trade memory, self-reflection, episodic memory). \
                    You analyze coins via Kraken WebSocket v2 and trade via Kraken REST API with limit orders. \
                    CRITICAL OUTPUT RULE: Output ONLY your direct conversational reply. \
                    Never output reasoning, planning, internal monologue, or phrases like 'I need to', 'The user is asking', 'Let me', or 'I should'. \
                    Start your response immediately. Be concise, precise, and confident.\n\n\
                    === LIVE ENGINE DATA ===\n{live_context}"
                );

                // User message is just the query + Qwen3 no-think control token at end
                let chat_user_msg = if think_mode {
                    format!("{user_msg}\n/think")
                } else {
                    format!("{user_msg}\n/no_think")
                };
                // Qwen3 always runs thinking mode internally; give enough tokens for
                // both the hidden think block and the actual reply.
                let chat_max_tokens = if think_mode { 3000 } else { 2500 };

                let req_url = format!("{ollama_url}/v1/chat/completions");
                let req_body = json!({
                    "model": model,
                    "messages": [
                        { "role": "system", "content": sys_content },
                        { "role": "user", "content": chat_user_msg }
                    ],
                    "temperature": 0.7,
                    "max_tokens": chat_max_tokens,
                    "stream": false,
                    "enable_thinking": think_mode
                });

                let result = client
                    .post(&req_url)
                    .json(&req_body)
                    .send()
                    .await;

                let reply = match result {
                    Ok(r) => match r.json::<serde_json::Value>().await {
                        Ok(v) => {
                            let msg = &v["choices"][0]["message"];
                            let raw = {
                                let content = msg["content"].as_str().unwrap_or("");
                                if content.is_empty() {
                                    msg["reasoning_content"].as_str()
                                        .unwrap_or("(no response)").to_string()
                                } else {
                                    content.to_string()
                                }
                            };
                            // Strip any <think>...</think> blocks the server may have leaked
                            let raw = if let (Some(ts), Some(te)) = (raw.find("<think>"), raw.find("</think>")) {
                                if te > ts {
                                    let after = raw[te + 8..].trim().to_string();
                                    if after.is_empty() { raw } else { after }
                                } else { raw }
                            } else { raw };
                            raw.trim().to_string()
                        },
                        Err(e) => format!("LLM parse error: {e}"),
                    },
                    Err(e) => format!("LLM connection error: {e}"),
                };

                tracing::info!(
                    "[AI-CHAT] reply ({} chars): {}",
                    reply.len(),
                    &reply[..reply.len().min(100)]
                );

                let response_json = json!({
                    "reply": reply,
                    "coins_live": state.read().await.features.len(),
                    "portfolio_value": state.read().await.total_value,
                });
                let response_body = response_json.to_string();

                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\
                     Content-Length: {}\r\n\r\n{}",
                    response_body.len(),
                    response_body
                );
                let _ = socket.write_all(resp.as_bytes()).await;
            });
        }
    });
}
