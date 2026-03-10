//! Backtester — replay journal or OHLCV data through AI strategies.
//!
//! Two modes:
//!   A) Journal Replay: feed past trades to Qwen3-4B, compare AI decision vs actual outcome
//!   B) OHLCV Backtest: walk through candles, let AI make BUY/HOLD/SELL decisions, simulate P&L
//!
//! Runs via CLI: `hybrid_kraken_core.exe --backtest journal|ohlcv|compare`

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::journal::TradeRecord;
use crate::ohlcv_fetcher::{self, Candle};

// ── Configuration ──────────────────────────────────────────────

pub struct BacktestConfig {
    pub ai_host: String,
    pub ai_model: String,
    pub temperature: f32,
    pub max_tokens: i32,
    pub fee_round_trip: f64, // 0.0052
    pub base_usd: f64,      // position size
    pub take_profit_pct: f64,
    pub stop_loss_pct: f64,
    pub trailing_stop_activate_pct: f64,
    pub trailing_stop_pct: f64,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            ai_host: std::env::var("BACKTEST_AI_HOST")
                .unwrap_or_else(|_| "http://127.0.0.1:8082".into()),
            ai_model: std::env::var("BACKTEST_AI_MODEL")
                .unwrap_or_else(|_| "qwen3-4b".into()),
            temperature: std::env::var("BACKTEST_AI_TEMPERATURE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.3),
            max_tokens: std::env::var("BACKTEST_AI_MAX_TOKENS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(512),
            fee_round_trip: std::env::var("FEE_ROUND_TRIP")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.0052),
            base_usd: std::env::var("BASE_USD")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(20.0),
            take_profit_pct: std::env::var("TAKE_PROFIT_PCT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(5.0),
            stop_loss_pct: std::env::var("STOP_LOSS_PCT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(3.5),
            trailing_stop_activate_pct: std::env::var("TRAILING_STOP_ACTIVATE_PCT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(2.5),
            trailing_stop_pct: std::env::var("TRAILING_STOP_PCT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.75),
        }
    }
}

// ── Results ────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResult {
    pub strategy_name: String,
    pub mode: String, // "journal" or "ohlcv"
    pub coin: String,
    pub total_trades: usize,
    pub wins: usize,
    pub losses: usize,
    pub win_rate: f64,
    pub total_pnl_pct: f64,
    pub total_pnl_usd: f64,
    pub max_drawdown_pct: f64,
    pub avg_hold_minutes: f64,
    pub profit_factor: f64,
    pub fee_total_usd: f64,
    pub fee_adjusted_pnl_usd: f64,
    pub trades: Vec<BacktestTrade>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestTrade {
    pub symbol: String,
    pub entry_price: f64,
    pub exit_price: f64,
    pub pnl_pct: f64,
    pub pnl_usd: f64,
    pub hold_minutes: f64,
    pub ai_action: String,
    pub ai_confidence: f64,
    pub ai_reasoning: String,
    pub result: String,
}

// ── AI Decision Parsing ────────────────────────────────────────

#[derive(Debug, Clone)]
struct AiDecision {
    action: String,     // BUY, SELL, HOLD
    confidence: f64,    // 0.0 - 1.0
    reasoning: String,
    _stop_loss_pct: f64,
    _take_profit_pct: f64,
}

fn parse_ai_decision(text: &str) -> AiDecision {
    let text_clean = text.trim();

    // Try JSON parse first
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(text_clean) {
        return AiDecision {
            action: v.get("action")
                .and_then(|a| a.as_str())
                .unwrap_or("HOLD")
                .to_uppercase(),
            confidence: v.get("confidence")
                .and_then(|c| c.as_f64())
                .unwrap_or(0.5),
            reasoning: v.get("reasoning")
                .and_then(|r| r.as_str())
                .unwrap_or("")
                .to_string(),
            _stop_loss_pct: v.get("stop_loss_pct")
                .and_then(|s| s.as_f64())
                .unwrap_or(3.5),
            _take_profit_pct: v.get("take_profit_pct")
                .and_then(|t| t.as_f64())
                .unwrap_or(5.0),
        };
    }

    // Try to find JSON block in text (between { and })
    if let Some(start) = text_clean.find('{') {
        if let Some(end) = text_clean.rfind('}') {
            let json_str = &text_clean[start..=end];
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(json_str) {
                return AiDecision {
                    action: v.get("action")
                        .and_then(|a| a.as_str())
                        .unwrap_or("HOLD")
                        .to_uppercase(),
                    confidence: v.get("confidence")
                        .and_then(|c| c.as_f64())
                        .unwrap_or(0.5),
                    reasoning: v.get("reasoning")
                        .and_then(|r| r.as_str())
                        .unwrap_or("")
                        .to_string(),
                    _stop_loss_pct: v.get("stop_loss_pct")
                        .and_then(|s| s.as_f64())
                        .unwrap_or(3.5),
                    _take_profit_pct: v.get("take_profit_pct")
                        .and_then(|t| t.as_f64())
                        .unwrap_or(5.0),
                };
            }
        }
    }

    // Fallback: parse first line for action keyword
    let first_line = text_clean.lines().next().unwrap_or("").to_uppercase();
    let action = if first_line.contains("BUY") {
        "BUY"
    } else if first_line.contains("SELL") {
        "SELL"
    } else {
        "HOLD"
    };

    // Try to extract confidence from text
    let confidence = extract_confidence(text_clean);

    AiDecision {
        action: action.to_string(),
        confidence,
        reasoning: text_clean.chars().take(200).collect(),
        _stop_loss_pct: 3.5,
        _take_profit_pct: 5.0,
    }
}

fn extract_confidence(text: &str) -> f64 {
    // Look for patterns like "confidence: 72%", "0.72", "72%"
    for line in text.lines() {
        let lower = line.to_lowercase();
        if lower.contains("confidence") || lower.contains("conf") {
            // Try to find a number
            for word in line.split_whitespace() {
                let clean = word.trim_matches(|c: char| !c.is_numeric() && c != '.');
                if let Ok(n) = clean.parse::<f64>() {
                    if n > 1.0 && n <= 100.0 {
                        return n / 100.0;
                    } else if n > 0.0 && n <= 1.0 {
                        return n;
                    }
                }
            }
        }
    }
    0.5 // default
}

// ── LLM Inference (calls Qwen3-4B on port 8082) ───────────────

async fn infer_backtest(
    client: &reqwest::Client,
    config: &BacktestConfig,
    system_prompt: &str,
    user_prompt: &str,
) -> anyhow::Result<String> {
    let url = format!("{}/v1/chat/completions", config.ai_host);

    // Append /no_think to suppress Qwen3 thinking mode (saves tokens + time)
    let user_with_nothinker = format!("{user_prompt}\n/no_think");

    let body = serde_json::json!({
        "model": config.ai_model,
        "messages": [
            { "role": "system", "content": system_prompt },
            { "role": "user", "content": user_with_nothinker }
        ],
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "stream": false
    });

    let resp = client
        .post(&url)
        .json(&body)
        .timeout(std::time::Duration::from_secs(120))
        .send()
        .await?;

    let json: serde_json::Value = resp.json().await?;

    let content = json
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .unwrap_or("")
        .to_string();

    // Strip think tags if present (Qwen3 think mode)
    let cleaned = strip_think_tags(&content);
    Ok(cleaned)
}

fn strip_think_tags(text: &str) -> String {
    if let Some(end) = text.find("</think>") {
        text[end + 8..].trim().to_string()
    } else {
        text.trim().to_string()
    }
}

// ── Strategy Prompts ──────────────────────────────────────────

fn strategy_system_prompt(strategy: &str) -> String {
    let base = r#"You are a crypto trading strategy evaluator. Analyze the market data provided and make a trading decision.

You MUST respond with ONLY a JSON object in this exact format:
{"action":"BUY","confidence":0.72,"reasoning":"brief reason","stop_loss_pct":3.5,"take_profit_pct":5.0}

Rules:
- action: "BUY", "SELL", or "HOLD" (uppercase)
- confidence: 0.0 to 1.0 (your conviction level)
- reasoning: 1-2 sentences explaining your decision
- stop_loss_pct: suggested stop loss percentage
- take_profit_pct: suggested take profit percentage
- Consider fees: ~0.52% round-trip (must overcome this to profit)
- Only output JSON, no other text"#;

    let strategy_rules = match strategy {
        "momentum" => r#"
Strategy: MOMENTUM / TREND-FOLLOWING
- BUY when: price > EMA21 > EMA55, RSI 40-65 (not overbought), positive momentum, volume above average
- SELL when: price < EMA21, RSI > 75 (overbought), momentum fading
- HOLD otherwise
- Prefer strong directional moves, avoid choppy sideways"#,

        "mean_reversion" => r#"
Strategy: MEAN REVERSION
- BUY when: RSI < 30 (oversold), price near lower Bollinger Band, volume spike suggesting reversal
- SELL when: RSI > 70 (overbought), price near upper Bollinger Band
- HOLD otherwise
- Look for extreme readings that are likely to revert to mean
- Be careful of strong trends (mean reversion fails in trends)"#,

        "breakout" => r#"
Strategy: BREAKOUT
- BUY when: price breaks above resistance with volume > 2x average, RSI rising from 40-60 range
- SELL when: breakout fails (price returns below breakout level), or target reached
- HOLD otherwise
- Volume confirmation is critical — no volume = false breakout
- Set tight stops below breakout level"#,

        "scalp" => r#"
Strategy: SCALP (Quick In/Out)
- BUY when: strong order book imbalance (>0.2 buy), tight spread (<0.1%), RSI 45-55, micro-momentum positive
- SELL when: imbalance flips negative, spread widens, or small profit target hit (0.5-1%)
- HOLD otherwise
- Very selective — only trade when conditions are perfect
- Tight stops (1%), quick profits (0.5-1.5%)"#,

        _ => r#"
Strategy: BALANCED
- BUY when: multiple indicators align (trend + momentum + volume + book)
- SELL when: majority of indicators turn negative
- HOLD when: mixed signals
- Weight RSI, trend, momentum, and volume equally"#,
    };

    format!("{base}\n{strategy_rules}")
}

// ── Mode A: Journal Replay ────────────────────────────────────

pub async fn run_journal_replay(
    strategy: &str,
    coin_filter: Option<&[String]>,
    quiet: bool,
) -> anyhow::Result<BacktestResult> {
    let config = BacktestConfig::default();
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(60))
        .build()?;

    // Load journal
    let journal_path = PathBuf::from("npu_journal/trades.jsonl");
    if !journal_path.exists() {
        anyhow::bail!("Journal not found at {}", journal_path.display());
    }

    let content = std::fs::read_to_string(&journal_path)?;
    let mut trades: Vec<TradeRecord> = Vec::new();
    for line in content.lines() {
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<TradeRecord>(line) {
            Ok(t) => trades.push(t),
            Err(_) => continue,
        }
    }

    // Filter by coin if specified
    if let Some(coins) = coin_filter {
        let coins_upper: Vec<String> = coins.iter().map(|c| c.to_uppercase()).collect();
        trades.retain(|t| coins_upper.iter().any(|c| t.symbol.to_uppercase().contains(c)));
    }

    let coin_label = coin_filter
        .map(|c| c.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(","))
        .unwrap_or_else(|| "ALL".to_string());

    tracing::info!(
        "[BACKTEST] Journal replay: {} trades, strategy={}, coins={}",
        trades.len(),
        strategy,
        coin_label
    );

    let system_prompt = strategy_system_prompt(strategy);
    let mut bt_trades = Vec::new();
    let mut wins = 0usize;
    let mut losses = 0usize;
    let mut total_pnl_usd = 0.0f64;
    let mut total_pnl_pct = 0.0f64;
    let mut total_fees = 0.0f64;
    let mut peak_equity = 0.0f64;
    let mut equity = 0.0f64;
    let mut max_drawdown = 0.0f64;
    let mut total_hold = 0.0f64;

    let total = trades.len();
    for (i, trade) in trades.iter().enumerate() {
        // Build user prompt with trade context
        let user_prompt = format!(
            r#"Coin: {symbol}
Entry context: {context}
Entry reasons: {reasons}
Entry price: ${entry_price:.6}
RSI info: (extracted from context)
Regime: {regime}

Based on this setup, would you have entered this trade?
Remember: fees are ~0.52% round-trip, so the move must exceed that to profit."#,
            symbol = trade.symbol,
            context = trade.entry_context,
            reasons = trade.entry_reasons.join(", "),
            entry_price = trade.entry_price,
            regime = extract_regime(&trade.entry_context),
        );

        // Call Qwen3-4B
        let response = match infer_backtest(&client, &config, &system_prompt, &user_prompt).await {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!("[BACKTEST] Trade {}/{}: AI error: {}", i + 1, total, e);
                continue;
            }
        };

        let decision = parse_ai_decision(&response);

        // Compare AI decision vs actual outcome
        let fee = config.base_usd * config.fee_round_trip;
        total_fees += fee;

        let bt = BacktestTrade {
            symbol: trade.symbol.clone(),
            entry_price: trade.entry_price,
            exit_price: trade.exit_price,
            pnl_pct: trade.pnl_percent,
            pnl_usd: trade.pnl_usd,
            hold_minutes: trade.hold_minutes,
            ai_action: decision.action.clone(),
            ai_confidence: decision.confidence,
            ai_reasoning: decision.reasoning.clone(),
            result: trade.result.clone(),
        };

        // Track stats only for trades the AI would have taken
        if decision.action == "BUY" {
            if trade.result == "WIN" {
                wins += 1;
            } else {
                losses += 1;
            }
            total_pnl_usd += trade.pnl_usd - fee;
            total_pnl_pct += trade.pnl_percent;
            equity += trade.pnl_usd - fee;
            if equity > peak_equity {
                peak_equity = equity;
            }
            let dd = if peak_equity > 0.0 {
                (peak_equity - equity) / peak_equity * 100.0
            } else {
                0.0
            };
            if dd > max_drawdown {
                max_drawdown = dd;
            }
            total_hold += trade.hold_minutes;
        }

        bt_trades.push(bt);

        // Progress log every 10 trades (unless quiet)
        if !quiet && (i + 1) % 10 == 0 {
            let wr = if (wins + losses) > 0 { wins as f64 / (wins + losses) as f64 * 100.0 } else { 0.0 };
            tracing::info!(
                "[BACKTEST] Progress: {}/{} | AI took {}/{} | WR={:.0}% | PnL=${:.2}",
                i + 1, total, wins + losses, i + 1, wr, total_pnl_usd
            );
        }

        // Small delay to not overwhelm the LLM
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    }

    let ai_trades = wins + losses;
    let win_rate = if ai_trades > 0 {
        wins as f64 / ai_trades as f64
    } else {
        0.0
    };
    let avg_hold = if ai_trades > 0 {
        total_hold / ai_trades as f64
    } else {
        0.0
    };
    let gross_wins: f64 = bt_trades
        .iter()
        .filter(|t| t.ai_action == "BUY" && t.pnl_usd > 0.0)
        .map(|t| t.pnl_usd)
        .sum();
    let gross_losses: f64 = bt_trades
        .iter()
        .filter(|t| t.ai_action == "BUY" && t.pnl_usd < 0.0)
        .map(|t| t.pnl_usd.abs())
        .sum();
    let profit_factor = if gross_losses > 0.0 {
        gross_wins / gross_losses
    } else if gross_wins > 0.0 {
        f64::INFINITY
    } else {
        0.0
    };

    let result = BacktestResult {
        strategy_name: strategy.to_string(),
        mode: "journal".to_string(),
        coin: coin_label,
        total_trades: trades.len(),
        wins,
        losses,
        win_rate,
        total_pnl_pct,
        total_pnl_usd,
        max_drawdown_pct: max_drawdown,
        avg_hold_minutes: avg_hold,
        profit_factor,
        fee_total_usd: total_fees,
        fee_adjusted_pnl_usd: total_pnl_usd,
        trades: bt_trades,
    };

    // Save result
    save_result(&result, strategy)?;

    // Print summary
    print_summary(&result);

    Ok(result)
}

// ── Mode B: OHLCV Backtest ────────────────────────────────────

pub async fn run_ohlcv_backtest(
    coins: &[String],
    days: u32,
    strategy: &str,
    interval: u32,
) -> anyhow::Result<Vec<BacktestResult>> {
    let config = BacktestConfig::default();
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(60))
        .build()?;

    // Download OHLCV data first
    tracing::info!(
        "[BACKTEST] Downloading {days} days of {interval}m candles for {} coins...",
        coins.len()
    );
    ohlcv_fetcher::download_all(coins, &[interval], days).await?;

    let system_prompt = strategy_system_prompt(strategy);
    let mut all_results = Vec::new();

    for coin in coins {
        let csv_path = PathBuf::from(format!("data/ohlcv_store/{coin}_{interval}m.csv"));
        if !csv_path.exists() {
            tracing::warn!("[BACKTEST] No data for {coin} — skipping");
            continue;
        }

        let candles = ohlcv_fetcher::load_candles_from_csv(&csv_path)?;
        if candles.len() < 60 {
            tracing::warn!(
                "[BACKTEST] {coin}: only {} candles, need 60+ — skipping",
                candles.len()
            );
            continue;
        }

        tracing::info!(
            "[BACKTEST] {coin}: {} candles, strategy={}",
            candles.len(),
            strategy
        );

        let result = backtest_coin(&client, &config, &system_prompt, coin, &candles, strategy).await?;
        print_summary(&result);
        all_results.push(result);
    }

    // Save combined results
    let combined_path = format!(
        "data/backtest_results/{strategy}_combined_{}.json",
        chrono::Utc::now().format("%Y%m%d_%H%M%S")
    );
    std::fs::create_dir_all("data/backtest_results")?;
    let json = serde_json::to_string_pretty(&all_results)?;
    std::fs::write(&combined_path, &json)?;
    tracing::info!("[BACKTEST] Combined results saved to {combined_path}");

    Ok(all_results)
}

async fn backtest_coin(
    client: &reqwest::Client,
    config: &BacktestConfig,
    system_prompt: &str,
    coin: &str,
    candles: &[Candle],
    strategy: &str,
) -> anyhow::Result<BacktestResult> {
    let mut bt_trades = Vec::new();
    let mut wins = 0usize;
    let mut losses = 0usize;
    let mut total_pnl_usd = 0.0f64;
    let mut total_pnl_pct = 0.0f64;
    let mut total_fees = 0.0f64;
    let mut peak_equity = 0.0f64;
    let mut equity = 0.0f64;
    let mut max_drawdown = 0.0f64;
    let mut total_hold = 0.0f64;

    // Virtual position tracking
    let mut in_position = false;
    let mut entry_price = 0.0f64;
    let mut entry_idx = 0usize;
    let mut highest_since_entry = 0.0f64;

    // We evaluate every N candles (not every single one — too slow)
    let eval_interval = 15.max(candles.len() / 200); // ~200 evaluations max
    let lookback = 55; // need at least 55 candles for EMA55

    let mut i = lookback;
    while i < candles.len() {
        let window = &candles[i.saturating_sub(lookback)..=i];
        let current = &candles[i];

        if in_position {
            // Check exit conditions
            let pnl_pct = (current.close - entry_price) / entry_price * 100.0;

            // Update highest
            if current.high > highest_since_entry {
                highest_since_entry = current.high;
            }

            // Stop loss
            if pnl_pct <= -config.stop_loss_pct {
                let fee = config.base_usd * config.fee_round_trip;
                let pnl_usd = config.base_usd * pnl_pct / 100.0 - fee;
                total_fees += fee;
                total_pnl_usd += pnl_usd;
                total_pnl_pct += pnl_pct / 100.0;
                losses += 1;
                let hold_min = (current.timestamp - candles[entry_idx].timestamp) as f64 / 60.0;
                total_hold += hold_min;

                equity += pnl_usd;
                if equity > peak_equity { peak_equity = equity; }
                let dd = if peak_equity > 0.0 { (peak_equity - equity) / peak_equity * 100.0 } else { 0.0 };
                if dd > max_drawdown { max_drawdown = dd; }

                bt_trades.push(BacktestTrade {
                    symbol: coin.to_string(),
                    entry_price,
                    exit_price: current.close,
                    pnl_pct: pnl_pct / 100.0,
                    pnl_usd,
                    hold_minutes: hold_min,
                    ai_action: "STOP_LOSS".into(),
                    ai_confidence: 0.0,
                    ai_reasoning: format!("Stop loss hit at {pnl_pct:.2}%"),
                    result: "LOSS".into(),
                });

                in_position = false;
                i += eval_interval;
                continue;
            }

            // Take profit
            if pnl_pct >= config.take_profit_pct {
                let fee = config.base_usd * config.fee_round_trip;
                let pnl_usd = config.base_usd * pnl_pct / 100.0 - fee;
                total_fees += fee;
                total_pnl_usd += pnl_usd;
                total_pnl_pct += pnl_pct / 100.0;
                wins += 1;
                let hold_min = (current.timestamp - candles[entry_idx].timestamp) as f64 / 60.0;
                total_hold += hold_min;

                equity += pnl_usd;
                if equity > peak_equity { peak_equity = equity; }

                bt_trades.push(BacktestTrade {
                    symbol: coin.to_string(),
                    entry_price,
                    exit_price: current.close,
                    pnl_pct: pnl_pct / 100.0,
                    pnl_usd,
                    hold_minutes: hold_min,
                    ai_action: "TAKE_PROFIT".into(),
                    ai_confidence: 0.0,
                    ai_reasoning: format!("Take profit hit at {pnl_pct:.2}%"),
                    result: "WIN".into(),
                });

                in_position = false;
                i += eval_interval;
                continue;
            }

            // Trailing stop
            let trail_pct = (highest_since_entry - entry_price) / entry_price * 100.0;
            if trail_pct >= config.trailing_stop_activate_pct {
                let trail_drop = (highest_since_entry - current.close) / highest_since_entry * 100.0;
                if trail_drop >= config.trailing_stop_pct {
                    let fee = config.base_usd * config.fee_round_trip;
                    let pnl_usd = config.base_usd * pnl_pct / 100.0 - fee;
                    total_fees += fee;
                    total_pnl_usd += pnl_usd;
                    total_pnl_pct += pnl_pct / 100.0;
                    let hold_min = (current.timestamp - candles[entry_idx].timestamp) as f64 / 60.0;
                    total_hold += hold_min;

                    if pnl_pct > 0.0 { wins += 1; } else { losses += 1; }

                    equity += pnl_usd;
                    if equity > peak_equity { peak_equity = equity; }
                    let dd = if peak_equity > 0.0 { (peak_equity - equity) / peak_equity * 100.0 } else { 0.0 };
                    if dd > max_drawdown { max_drawdown = dd; }

                    bt_trades.push(BacktestTrade {
                        symbol: coin.to_string(),
                        entry_price,
                        exit_price: current.close,
                        pnl_pct: pnl_pct / 100.0,
                        pnl_usd,
                        hold_minutes: hold_min,
                        ai_action: "TRAILING_STOP".into(),
                        ai_confidence: 0.0,
                        ai_reasoning: format!("Trailing stop: peak={highest_since_entry:.4} drop={trail_drop:.2}%"),
                        result: if pnl_pct > 0.0 { "WIN" } else { "LOSS" }.into(),
                    });

                    in_position = false;
                    i += eval_interval;
                    continue;
                }
            }

            // Continue holding — advance by 1 candle for exit checks
            i += 1;
            continue;
        }

        // Not in position — ask AI whether to enter
        let features = compute_simple_features(window);
        let user_prompt = format!(
            r#"Coin: {coin}
Price: ${close:.6}
24h range: ${low:.6} - ${high:.6}
RSI(14): {rsi:.1}
EMA21: ${ema21:.6}
EMA55: ${ema55:.6}
Price vs EMA21: {pct_ema21:+.2}%
Momentum (5-bar): {momentum:+.4}%
Volume ratio (vs 20-bar avg): {vol_ratio:.2}x
Candle body: {body_pct:+.3}%
Consecutive direction: {consec}

Should you enter a trade? Remember fees are ~0.52% round-trip."#,
            close = current.close,
            low = current.low,
            high = current.high,
            rsi = features.rsi,
            ema21 = features.ema21,
            ema55 = features.ema55,
            pct_ema21 = features.pct_vs_ema21,
            momentum = features.momentum_pct,
            vol_ratio = features.volume_ratio,
            body_pct = features.body_pct,
            consec = features.consecutive,
        );

        let response = match infer_backtest(client, config, system_prompt, &user_prompt).await {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!("[BACKTEST] {coin} candle {i}: AI error: {e}");
                i += eval_interval;
                continue;
            }
        };

        let decision = parse_ai_decision(&response);

        if decision.action == "BUY" && decision.confidence >= 0.60 {
            in_position = true;
            entry_price = current.close;
            entry_idx = i;
            highest_since_entry = current.high;
        }

        i += eval_interval;

        // Small delay
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
    }

    // Close any open position at last candle
    if in_position && !candles.is_empty() {
        let last = candles.last().unwrap();
        let pnl_pct = (last.close - entry_price) / entry_price * 100.0;
        let fee = config.base_usd * config.fee_round_trip;
        let pnl_usd = config.base_usd * pnl_pct / 100.0 - fee;
        total_fees += fee;
        total_pnl_usd += pnl_usd;
        total_pnl_pct += pnl_pct / 100.0;
        let hold_min = (last.timestamp - candles[entry_idx].timestamp) as f64 / 60.0;
        total_hold += hold_min;
        if pnl_pct > 0.0 { wins += 1; } else { losses += 1; }

        bt_trades.push(BacktestTrade {
            symbol: coin.to_string(),
            entry_price,
            exit_price: last.close,
            pnl_pct: pnl_pct / 100.0,
            pnl_usd,
            hold_minutes: hold_min,
            ai_action: "FORCED_CLOSE".into(),
            ai_confidence: 0.0,
            ai_reasoning: "End of data — closing position".into(),
            result: if pnl_pct > 0.0 { "WIN" } else { "LOSS" }.into(),
        });
    }

    let ai_trades = wins + losses;
    let win_rate = if ai_trades > 0 { wins as f64 / ai_trades as f64 } else { 0.0 };
    let avg_hold = if ai_trades > 0 { total_hold / ai_trades as f64 } else { 0.0 };
    let gross_wins: f64 = bt_trades.iter().filter(|t| t.pnl_usd > 0.0).map(|t| t.pnl_usd).sum();
    let gross_losses: f64 = bt_trades.iter().filter(|t| t.pnl_usd < 0.0).map(|t| t.pnl_usd.abs()).sum();
    let profit_factor = if gross_losses > 0.0 { gross_wins / gross_losses } else if gross_wins > 0.0 { f64::INFINITY } else { 0.0 };

    Ok(BacktestResult {
        strategy_name: strategy.to_string(),
        mode: "ohlcv".to_string(),
        coin: coin.to_string(),
        total_trades: ai_trades,
        wins,
        losses,
        win_rate,
        total_pnl_pct,
        total_pnl_usd,
        max_drawdown_pct: max_drawdown,
        avg_hold_minutes: avg_hold,
        profit_factor,
        fee_total_usd: total_fees,
        fee_adjusted_pnl_usd: total_pnl_usd,
        trades: bt_trades,
    })
}

// ── Compare Mode ──────────────────────────────────────────────

pub async fn run_compare(
    coins: &[String],
    days: u32,
    interval: u32,
) -> anyhow::Result<()> {
    let strategies = ["momentum", "mean_reversion", "breakout", "scalp"];

    println!("\n============================================================");
    println!("  STRATEGY COMPARISON — {} coins, {} days", coins.len(), days);
    println!("============================================================\n");

    for strat in &strategies {
        tracing::info!("[BACKTEST] === Running strategy: {strat} ===");
        let results = run_ohlcv_backtest(coins, days, strat, interval).await?;

        // Aggregate across coins
        let total_trades: usize = results.iter().map(|r| r.total_trades).sum();
        let total_wins: usize = results.iter().map(|r| r.wins).sum();
        let total_pnl: f64 = results.iter().map(|r| r.fee_adjusted_pnl_usd).sum();
        let wr = if total_trades > 0 { total_wins as f64 / total_trades as f64 * 100.0 } else { 0.0 };

        println!(
            "  {strat:15} | trades={total_trades:4} | wins={total_wins:4} | WR={wr:5.1}% | PnL=${total_pnl:+.2}"
        );
    }

    println!();
    Ok(())
}

// ── Simple Feature Computation ────────────────────────────────

struct SimpleFeatures {
    rsi: f64,
    ema21: f64,
    ema55: f64,
    pct_vs_ema21: f64,
    momentum_pct: f64,
    volume_ratio: f64,
    body_pct: f64,
    consecutive: i32,
}

fn compute_simple_features(candles: &[Candle]) -> SimpleFeatures {
    let n = candles.len();
    if n < 2 {
        return SimpleFeatures {
            rsi: 50.0, ema21: 0.0, ema55: 0.0, pct_vs_ema21: 0.0,
            momentum_pct: 0.0, volume_ratio: 1.0, body_pct: 0.0, consecutive: 0,
        };
    }

    let last = &candles[n - 1];

    // RSI(14)
    let rsi_period = 14.min(n - 1);
    let mut gains = 0.0f64;
    let mut loss_sum = 0.0f64;
    for i in (n - rsi_period)..n {
        let change = candles[i].close - candles[i - 1].close;
        if change > 0.0 { gains += change; } else { loss_sum += change.abs(); }
    }
    let avg_gain = gains / rsi_period as f64;
    let avg_loss = loss_sum / rsi_period as f64;
    let rsi = if avg_loss == 0.0 { 100.0 } else {
        let rs = avg_gain / avg_loss;
        100.0 - 100.0 / (1.0 + rs)
    };

    // EMA21
    let ema21 = compute_ema(candles, 21.min(n));
    let ema55 = compute_ema(candles, 55.min(n));

    let pct_vs_ema21 = if ema21 > 0.0 { (last.close - ema21) / ema21 * 100.0 } else { 0.0 };

    // Momentum (5-bar)
    let mom_lookback = 5.min(n - 1);
    let momentum_pct = (last.close - candles[n - 1 - mom_lookback].close)
        / candles[n - 1 - mom_lookback].close * 100.0;

    // Volume ratio (vs 20-bar average)
    let vol_window = 20.min(n);
    let avg_vol: f64 = candles[n - vol_window..n].iter().map(|c| c.volume).sum::<f64>() / vol_window as f64;
    let volume_ratio = if avg_vol > 0.0 { last.volume / avg_vol } else { 1.0 };

    // Candle body
    let body_pct = if last.open > 0.0 { (last.close - last.open) / last.open * 100.0 } else { 0.0 };

    // Consecutive same-direction candles
    let mut consec = 0i32;
    let dir = if last.close >= last.open { 1 } else { -1 };
    for i in (0..n).rev() {
        let c_dir = if candles[i].close >= candles[i].open { 1 } else { -1 };
        if c_dir == dir { consec += dir; } else { break; }
    }

    SimpleFeatures {
        rsi, ema21, ema55, pct_vs_ema21,
        momentum_pct, volume_ratio, body_pct, consecutive: consec,
    }
}

fn compute_ema(candles: &[Candle], period: usize) -> f64 {
    if candles.is_empty() || period == 0 { return 0.0; }
    let mult = 2.0 / (period as f64 + 1.0);
    let mut ema = candles[0].close;
    for c in candles.iter().skip(1) {
        ema = (c.close - ema) * mult + ema;
    }
    ema
}

// ── Helpers ───────────────────────────────────────────────────

fn extract_regime(context: &str) -> &str {
    if context.contains("trending") { "trending" }
    else if context.contains("volatile") { "volatile" }
    else if context.contains("crash") { "crash" }
    else if context.contains("sideways") { "sideways" }
    else { "unknown" }
}

fn save_result(result: &BacktestResult, strategy: &str) -> anyhow::Result<()> {
    std::fs::create_dir_all("data/backtest_results")?;
    let filename = format!(
        "data/backtest_results/{}_{}_{}_{}.json",
        result.mode,
        strategy,
        result.coin,
        chrono::Utc::now().format("%Y%m%d_%H%M%S")
    );
    let json = serde_json::to_string_pretty(result)?;
    std::fs::write(&filename, &json)?;
    tracing::info!("[BACKTEST] Results saved to {filename}");
    Ok(())
}

fn print_summary(result: &BacktestResult) {
    println!("\n╔══════════════════════════════════════════════════╗");
    println!("║  BACKTEST: {} — {} ({})", result.strategy_name, result.coin, result.mode);
    println!("╠══════════════════════════════════════════════════╣");
    println!("║  Trades:     {:>6}                              ║", result.total_trades);
    println!("║  Wins:       {:>6}                              ║", result.wins);
    println!("║  Losses:     {:>6}                              ║", result.losses);
    println!("║  Win Rate:   {:>5.1}%                              ║", result.win_rate * 100.0);
    println!("║  PnL (USD):  {:>+8.2}                           ║", result.fee_adjusted_pnl_usd);
    println!("║  PnL (%):    {:>+8.4}                           ║", result.total_pnl_pct);
    println!("║  Max DD:     {:>5.1}%                              ║", result.max_drawdown_pct);
    println!("║  Avg Hold:   {:>6.1} min                          ║", result.avg_hold_minutes);
    println!("║  Profit F:   {:>6.2}                              ║", result.profit_factor);
    println!("║  Fees:       ${:>6.2}                              ║", result.fee_total_usd);
    println!("╚══════════════════════════════════════════════════╝");
}
