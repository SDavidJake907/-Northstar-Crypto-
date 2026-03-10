//! Coin Rotation Scanner — AI_1 rotates coins via tool calls.
//!
//! Atlas (NPU:8083) scans CryptoPanic headlines → updates SharedCloudIntel.
//! AI_1 (GPU:8081) reads that data and uses tool-calling to rotate the watchlist.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::ai_bridge::{InferToolResult, LlmToolCall};
use crate::cloud_intel::SharedCloudIntel;
use crate::config::CachedEnv;

// ── Shared Swap Queue (exit watchdog → vision scanner) ──────

/// Coin swap suggestion from the exit watchdog (AI_2).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CoinSwapSuggestion {
    pub add: Vec<String>,
    pub remove: Vec<String>,
    pub reason: String,
    pub timestamp: f64,
}

/// Shared swap queue: exit model writes, vision scanner reads + clears.
pub type SharedSwapQueue = Arc<RwLock<Vec<CoinSwapSuggestion>>>;

// ── Tool Definitions ────────────────────────────────────────

pub fn scanner_tools() -> Vec<serde_json::Value> {
    vec![
        json!({
            "type": "function",
            "function": {
                "name": "rotate_coins",
                "description": "Add or remove coins from the active watchlist. Only coins from COIN_POOL may be added. Anchor coins (BTC,ETH) cannot be removed.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "add": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Coin symbols to ADD to watchlist (e.g. [\"DOGE\",\"SHIB\"])"
                        },
                        "remove": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Coin symbols to REMOVE from watchlist (e.g. [\"NEAR\",\"ATOM\"])"
                        }
                    }
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "get_current_coins",
                "description": "Get the current list of tracked coins.",
                "parameters": { "type": "object", "properties": {} }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "get_ticker_summary",
                "description": "Get current price, 24h change, and volume for a coin from Kraken.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Coin symbol, e.g. \"SOL\", \"DOGE\", \"AVAX\""
                        }
                    },
                    "required": ["symbol"]
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "get_market_overview",
                "description": "Get BTC and ETH current status (price, trend, volume) as a market overview.",
                "parameters": { "type": "object", "properties": {} }
            }
        }),
    ]
}

// ── AI_1 System Prompt (cloud intel driven) ─────────────────

fn rotation_system_prompt(
    current_coins: &[String],
    anchors: &[String],
    max_tracked: usize,
    coin_pool: &[String],
) -> String {
    let pool_str = if coin_pool.is_empty() {
        "No pool defined — use any liquid Kraken coin".to_string()
    } else {
        coin_pool.join(", ")
    };
    format!(
        r#"You are AI_1, the coin rotation AI for the CODE ZERO trading desk.
You are part of a 3-AI team: YOU rotate coins, AI_2 makes entries, AI_3 watches exits.
AI_4 (Intel NPU) scans headlines every 5min and gives sentiment scores.

Your job: Rotate the watchlist to focus on the MOST PROFITABLE coins RIGHT NOW.

Current tracked ({current}/{max}): {coins}
Anchor coins (NEVER remove): {anchors}
Max slots: {max}

APPROVED COIN POOL (ONLY add from this list):
{pool}

ADD a coin when ALL of these are true:
- Sentiment score >= +0.40 (positive news momentum)
- 24h vol >= 2x normal OR 24h gain >= +10% with rising volume
- NOT overbought (RSI < 72) — don't chase the top of a pump
- Use get_ticker_summary to confirm price and volume are real before adding

REMOVE a coin when ANY of these are true:
- Sentiment score <= -0.25 (negative news flow building)
- 24h vol < 0.3x normal for 2+ cycles (dead, no opportunity)
- 24h loss > -8% with declining volume (distribution, sellers in control)
- Never remove anchor coins

PERFORMANCE RULES (use Nemo memory if available):
- PRIORITIZE coins with win_rate >= 50% across >= 3 trades
- DEPRIORITIZE coins with win_rate < 25% across >= 3 trades
- PRIORITIZE coins with profit_factor >= 2.0
- New coins with < 3 trades: use vol and sentiment only

HOT MOVER RULE:
- If a coin is up >= +15% in 24h with vol >= 5x: add it immediately, displace lowest-performing tracked coin
- Volume spike alone is NOT enough — price must also be moving UP

TOOL WORKFLOW:
Step 1 — call get_market_overview to read BTC/ETH direction
Step 2 — call get_ticker_summary on any candidate before adding it
Step 3 — call rotate_coins to execute the swap (max 2 swaps per cycle)
Step 4 — reply with 1 sentence: what you changed and why (use % and # not coin names if possible)

If data is mixed or uncertain: do NOT rotate. Stability beats churn."#,
        coins = current_coins.join(", "),
        anchors = anchors.join(", "),
        max = max_tracked,
        current = current_coins.len(),
        pool = pool_str,
    )
}

// ── Direct LLM Inference (text + tools) ─────────────────────

/// Call AI_1 on llama-server with text messages and tool definitions.
async fn infer_with_tools(
    client: &reqwest::Client,
    base_url: &str,
    model: &str,
    messages: &[serde_json::Value],
    tools: &[serde_json::Value],
    temperature: f32,
    max_tokens: i32,
) -> anyhow::Result<InferToolResult> {
    let url = format!("{base_url}/v1/chat/completions");
    let mut body = json!({
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": 0.9,
        "stream": false,
    });

    if !tools.is_empty() {
        body.as_object_mut()
            .unwrap()
            .insert("tools".into(), serde_json::Value::Array(tools.to_vec()));
    }

    let resp = client
        .post(&url)
        .json(&body)
        .timeout(std::time::Duration::from_secs(120))
        .send()
        .await?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        anyhow::bail!("LLM returned {status}: {}", text.chars().take(300).collect::<String>());
    }

    let json: serde_json::Value = resp.json().await?;
    let msg = json
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("message"));

    // Check for tool_calls
    if let Some(tool_calls) = msg.and_then(|m| m.get("tool_calls")).and_then(|tc| tc.as_array()) {
        if !tool_calls.is_empty() {
            let mut calls = Vec::new();
            for tc in tool_calls {
                let id = tc
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("call_0")
                    .to_string();
                let func = tc.get("function").unwrap_or(&serde_json::Value::Null);
                let name = func
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string();
                let args_raw = func.get("arguments");
                let arguments = match args_raw {
                    Some(serde_json::Value::String(s)) => {
                        serde_json::from_str(s).unwrap_or(json!({}))
                    }
                    Some(v) if v.is_object() => v.clone(),
                    _ => json!({}),
                };
                calls.push(LlmToolCall {
                    id,
                    function_name: name,
                    arguments,
                });
            }
            return Ok(InferToolResult::ToolCalls(calls));
        }
    }

    // Final text response
    let content = msg
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .unwrap_or("")
        .to_string();

    Ok(InferToolResult::FinalResponse(content))
}

// ── Kraken Public Ticker API ────────────────────────────────

pub async fn fetch_ticker(client: &reqwest::Client, symbol: &str) -> String {
    let pair = format!("{symbol}USD");
    let url = format!("https://api.kraken.com/0/public/Ticker?pair={pair}");

    let resp = match client.get(&url).send().await {
        Ok(r) => r,
        Err(e) => return format!("{symbol}: HTTP error: {e}"),
    };
    let json: serde_json::Value = match resp.json().await {
        Ok(j) => j,
        Err(e) => return format!("{symbol}: parse error: {e}"),
    };

    if let Some(result) = json.get("result").and_then(|r| r.as_object()) {
        if let Some((_pair_key, data)) = result.iter().next() {
            let last = data
                .get("c")
                .and_then(|c| c.get(0))
                .and_then(|v| v.as_str())
                .unwrap_or("?");
            let vol = data
                .get("v")
                .and_then(|v| v.get(1))
                .and_then(|v| v.as_str())
                .unwrap_or("?");
            let high = data
                .get("h")
                .and_then(|h| h.get(1))
                .and_then(|v| v.as_str())
                .unwrap_or("?");
            let low = data
                .get("l")
                .and_then(|l| l.get(1))
                .and_then(|v| v.as_str())
                .unwrap_or("?");
            let open = data
                .get("o")
                .and_then(|o| o.as_str())
                .unwrap_or("0");

            let last_f: f64 = last.parse().unwrap_or(0.0);
            let open_f: f64 = open.parse().unwrap_or(0.0);
            let change_pct = if open_f > 0.0 {
                ((last_f - open_f) / open_f) * 100.0
            } else {
                0.0
            };

            return format!(
                "{symbol}: ${last} ({change_pct:+.2}%) vol={vol} hi={high} lo={low}"
            );
        }
    }

    if let Some(errs) = json.get("error").and_then(|e| e.as_array()) {
        return format!("{symbol}: API error: {:?}", errs);
    }

    format!("{symbol}: no data")
}

// ── .env SYMBOLS Persistence ────────────────────────────────

/// Find the .env file by walking up from cwd (mirrors main.rs logic).
fn find_env_path() -> Option<std::path::PathBuf> {
    let mut cur = std::env::current_dir().ok()?;
    for _ in 0..4 {
        let candidate = cur.join(".env");
        if candidate.exists() {
            return Some(candidate);
        }
        if !cur.pop() {
            break;
        }
    }
    None
}

/// Overwrite the SYMBOLS= line in .env with the new coin list.
fn persist_symbols_to_env(coins: &[String]) {
    let path = match find_env_path() {
        Some(p) => p,
        None => {
            tracing::warn!("[ROTATE] .env not found — swap not persisted");
            return;
        }
    };

    let content = match std::fs::read_to_string(&path) {
        Ok(c) => c,
        Err(e) => {
            tracing::warn!("[ROTATE] Could not read .env: {e}");
            return;
        }
    };

    let new_symbols = coins.join(",");
    let has_trailing_newline = content.ends_with('\n');

    let new_content: String = content
        .lines()
        .map(|line| {
            if line.starts_with("SYMBOLS=") {
                format!("SYMBOLS={new_symbols}")
            } else {
                line.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("\n");

    let new_content = if has_trailing_newline {
        format!("{new_content}\n")
    } else {
        new_content
    };

    // Atomic write: write to .env.tmp then rename
    let tmp = path.with_extension("env.tmp");
    if let Err(e) = std::fs::write(&tmp, &new_content) {
        tracing::warn!("[ROTATE] Could not write .env.tmp: {e}");
        return;
    }
    if let Err(e) = std::fs::rename(&tmp, &path) {
        tracing::warn!("[ROTATE] Could not rename .env.tmp → .env: {e}");
        return;
    }

    tracing::info!("[ROTATE] .env SYMBOLS updated → {new_symbols}");
}

// ── Tool Execution ──────────────────────────────────────────

pub async fn execute_scanner_tool(
    tool_name: &str,
    args: &serde_json::Value,
    active_coins: &Arc<RwLock<Vec<String>>>,
    ws_reconnect: &Arc<AtomicBool>,
    anchors: &[String],
    max_tracked: usize,
    http_client: &reqwest::Client,
) -> String {
    match tool_name {
        "rotate_coins" => {
            let add: Vec<String> = args
                .get("add")
                .and_then(|a| a.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.trim().to_uppercase()))
                        .filter(|s| !s.is_empty())
                        .collect()
                })
                .unwrap_or_default();
            let remove: Vec<String> = args
                .get("remove")
                .and_then(|a| a.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.trim().to_uppercase()))
                        .filter(|s| !s.is_empty())
                        .collect()
                })
                .unwrap_or_default();

            let anchors_upper: Vec<String> =
                anchors.iter().map(|a| a.to_uppercase()).collect();

            // Read COIN_POOL for validation (empty = allow any)
            let coin_pool: std::collections::HashSet<String> = std::env::var("COIN_POOL")
                .unwrap_or_default()
                .split(',')
                .map(|s| s.trim().to_uppercase())
                .filter(|s| !s.is_empty())
                .collect();

            let mut coins = active_coins.write().await;
            let before = coins.clone();

            // Remove (skip anchors)
            let mut blocked = Vec::new();
            for sym in &remove {
                if anchors_upper.contains(sym) {
                    blocked.push(sym.clone());
                } else {
                    coins.retain(|c| c != sym);
                }
            }

            // Add — use max_tracked as cap so list can grow up to the configured limit
            let cap = max_tracked.max(before.len());
            let mut added = Vec::new();
            let mut not_in_pool = Vec::new();
            for sym in &add {
                if !coin_pool.is_empty() && !coin_pool.contains(sym) {
                    not_in_pool.push(sym.clone());
                    continue;
                }
                if !coins.contains(sym) && coins.len() < cap {
                    coins.push(sym.clone());
                    added.push(sym.clone());
                }
            }

            let after = coins.clone();
            let removed_count = before.len().saturating_sub(after.len()) + added.len();
            drop(coins);

            if before != after {
                ws_reconnect.store(true, Ordering::SeqCst);
                tracing::info!(
                    "[ROTATE] {} → {} (added: {:?}, removed: {:?})",
                    before.join(","),
                    after.join(","),
                    added,
                    remove.iter().filter(|r| !blocked.contains(r)).collect::<Vec<_>>()
                );
                persist_symbols_to_env(&after);
            }
            let mut msg = format!(
                "Rotation applied ({} removed, {} added, {}/{} total). Now tracking: [{}]",
                removed_count, added.len(), after.len(), cap, after.join(", ")
            );
            if !blocked.is_empty() {
                msg.push_str(&format!(
                    " (blocked anchor removal: {})",
                    blocked.join(", ")
                ));
            }
            if !not_in_pool.is_empty() {
                msg.push_str(&format!(
                    " (rejected — not in COIN_POOL: {})",
                    not_in_pool.join(", ")
                ));
            }
            msg
        }

        "get_current_coins" => {
            let coins = active_coins.read().await;
            format!("Tracking {} coins: {}", coins.len(), coins.join(", "))
        }

        "get_ticker_summary" => {
            let symbol = args
                .get("symbol")
                .and_then(|s| s.as_str())
                .unwrap_or("BTC")
                .to_uppercase();
            fetch_ticker(http_client, &symbol).await
        }

        "get_market_overview" => {
            let btc = fetch_ticker(http_client, "BTC").await;
            let eth = fetch_ticker(http_client, "ETH").await;
            format!("Market Overview:\n  {btc}\n  {eth}")
        }

        _ => format!("Unknown tool: {tool_name}"),
    }
}

// ── Swap Suggestion Formatter ────────────────────────────────

fn format_swap_suggestions(suggestions: &[CoinSwapSuggestion], current_coins: &[String]) -> String {
    let mut msg = String::from(
        "The exit watchdog (AI_2 on CPU) suggests these coin swaps based on market scanning:\n\n"
    );
    for (i, s) in suggestions.iter().enumerate() {
        msg.push_str(&format!(
            "Suggestion {}: add={:?} remove={:?}\n  Reason: {}\n\n",
            i + 1, s.add, s.remove, s.reason,
        ));
    }
    msg.push_str(&format!("Current tracked: [{}]\n\n", current_coins.join(", ")));
    msg.push_str(
        "Verify these suggestions:\n\
         1. Use get_ticker_summary on suggested coins to check volume and price action\n\
         2. If the suggestion looks good, call rotate_coins to execute it\n\
         3. If volume is too low or the move looks fake, skip it\n\
         4. Give a 1-2 sentence summary of what you did"
    );
    msg
}

// ── Main Scanner Loop ───────────────────────────────────────

pub async fn run_vision_scanner(
    active_coins: Arc<RwLock<Vec<String>>>,
    ws_reconnect: Arc<AtomicBool>,
    _cloud_intel: SharedCloudIntel,
    interval_sec: u64,
    max_tracked: usize,
    anchor_coins: Vec<String>,
    swap_queue: Option<SharedSwapQueue>,
) {
    let http_client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .unwrap_or_default();
    let tools = scanner_tools();
    let env = CachedEnv::snapshot();
    let max_rounds: usize = env.get_parsed("VISION_MAX_ROUNDS", 3);
    let llm_url = env.get_str("OLLAMA_HOST", "http://127.0.0.1:8081");
    let llm_model = env.get_str("OLLAMA_MODEL", "qwen2.5-14b");

    // Let engine warm up + wait for first cloud intel scan
    let warmup: u64 = env.get_parsed("WARMUP_SECONDS", 60);
    tracing::info!(
        "[SCANNER] AI_1 rotation scanner starting in {}s (interval={}s, max_coins={})",
        warmup + 60, interval_sec, max_tracked
    );
    tokio::time::sleep(tokio::time::Duration::from_secs(warmup + 60)).await;

    let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(interval_sec));
    interval.tick().await; // consume first tick

    loop {
        interval.tick().await;

        // ── Process exit watchdog swap suggestions first ──
        if let Some(ref sq) = swap_queue {
            let suggestions = {
                let q = sq.read().await;
                if q.is_empty() { Vec::new() } else { q.clone() }
            };
            if !suggestions.is_empty() {
                tracing::info!(
                    "[SCANNER] {} swap suggestion(s) from exit watchdog — confirming with AI_1",
                    suggestions.len()
                );

                let current_coins = active_coins.read().await.clone();
                let swap_prompt = format_swap_suggestions(&suggestions, &current_coins);
                let coin_pool_vec: Vec<String> = std::env::var("COIN_POOL")
                    .unwrap_or_default()
                    .split(',')
                    .map(|s| s.trim().to_uppercase())
                    .filter(|s| !s.is_empty())
                    .collect();
                let sys = rotation_system_prompt(&current_coins, &anchor_coins, max_tracked, &coin_pool_vec);
                let mut swap_msgs = vec![
                    json!({ "role": "system", "content": sys }),
                    json!({ "role": "user", "content": swap_prompt }),
                ];

                let env = CachedEnv::snapshot();
                let swap_temperature: f32 = env.get_parsed("VISION_TEMPERATURE", 0.4);
                let swap_max_tokens: i32 = env.get_parsed("VISION_MAX_TOKENS", 512);

                // Multi-turn tool-calling loop for swap confirmation
                let mut swap_round = 0usize;
                loop {
                    if swap_round >= max_rounds { break; }
                    swap_round += 1;

                    let result = infer_with_tools(
                        &http_client, &llm_url, &llm_model,
                        &swap_msgs, &tools, swap_temperature, swap_max_tokens,
                    ).await;

                    match result {
                        Ok(InferToolResult::ToolCalls(calls)) => {
                            tracing::info!("[SCANNER-SWAP] Round {swap_round}: {} tool call(s)", calls.len());
                            let tc_json: Vec<serde_json::Value> = calls.iter().map(|c| {
                                json!({
                                    "id": c.id,
                                    "type": "function",
                                    "function": { "name": c.function_name, "arguments": c.arguments.to_string() }
                                })
                            }).collect();
                            swap_msgs.push(json!({ "role": "assistant", "tool_calls": tc_json }));

                            for call in &calls {
                                let result_text = execute_scanner_tool(
                                    &call.function_name, &call.arguments,
                                    &active_coins, &ws_reconnect, &anchor_coins,
                                    max_tracked, &http_client,
                                ).await;
                                tracing::info!(
                                    "[SCANNER-SWAP] tool::{} → {}",
                                    call.function_name,
                                    result_text.chars().take(120).collect::<String>()
                                );
                                swap_msgs.push(json!({
                                    "role": "tool",
                                    "tool_call_id": call.id,
                                    "content": result_text
                                }));
                            }
                        }
                        Ok(InferToolResult::FinalResponse(text)) => {
                            if !text.is_empty() {
                                tracing::info!(
                                    "[SCANNER-SWAP] AI_1 says: {}",
                                    text.chars().take(300).collect::<String>()
                                );
                            }
                            break;
                        }
                        Err(e) => {
                            tracing::error!("[SCANNER-SWAP] Inference error: {e}");
                            break;
                        }
                    }
                }

                // Clear processed suggestions
                sq.write().await.clear();
            }
        }

        // Cloud intel (Atlas) is a market-level mood scanner — not a coin analyst.
        // Rotation decisions come from the exit watchdog's swap queue only.
        // Atlas market mood is available via SharedCloudIntel if needed elsewhere.

        let coins_now = active_coins.read().await;
        tracing::debug!(
            "[SCANNER] ── Cycle done. Tracking: {} | Next in {interval_sec}s ──",
            coins_now.join(",")
        );
    }
}
