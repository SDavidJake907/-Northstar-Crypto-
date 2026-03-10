//! Exit logic — exit checks, tool-calling exit decisions, ai exit scoring, trade execution.

use super::*;
use crate::ai_bridge::{AiBridge};
use crate::config;
use crate::kraken_api::KrakenApi;
use crate::journal::TradeRecord;
use serde_json::json;
use std::collections::HashMap;

impl super::TradingLoop {
    // ── Exit Tool-Calling Support ────────────────────────────────

    /// Build OpenAI-compatible tool definitions for exit decision tool calling.
    pub(crate) fn build_exit_tool_schemas() -> Vec<serde_json::Value> {
        vec![
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": "get_coin_features",
                    "description": "Get full technical indicators for a coin: trend, momentum, RSI, volume, book imbalance, spread, EMAs, Bollinger, Hurst, entropy, ADX, quant scores, and action scores.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": { "type": "string", "description": "Coin symbol, e.g. 'BTC'" }
                        },
                        "required": ["symbol"]
                    }
                }
            }),
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": "get_trade_history",
                    "description": "Get last N closed trades for a coin. Shows PnL%, hold time, exit reason, WIN/LOSS result, and overall coin stats.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": { "type": "string", "description": "Coin symbol, e.g. 'SOL'" },
                            "limit": { "type": "integer", "description": "Max trades to return (1-10, default 5)" }
                        },
                        "required": ["symbol"]
                    }
                }
            }),
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": "get_market_context",
                    "description": "Get current market regime, BTC gravity (trend/rsi/crash), breadth (green/red, avg RSI), and sector momentum breakdown.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            }),
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": "get_ai_memory",
                    "description": "Get your track record for a coin: win/loss count, avg PnL, behavioral comments, recent decisions, and notable episodes (big wins/losses/panic sells).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": { "type": "string", "description": "Coin symbol, e.g. 'ETH'" }
                        },
                        "required": ["symbol"]
                    }
                }
            }),
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": "get_correlated_coins",
                    "description": "Get performance of other coins in the same sector: their momentum, trend, RSI, and volume right now.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": { "type": "string", "description": "Coin to find sector peers for" }
                        },
                        "required": ["symbol"]
                    }
                }
            }),
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": "get_engine_status",
                    "description": "Get YOUR engine status: portfolio value, cash, positions count, 60s meters (AI calls, buys, sells, rejects, exit checks), watchdog health, circuit breaker state, top reject reasons, and hardware status. Use this to understand your own performance and health.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            }),
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": "get_top_movers",
                    "description": "Search the FULL Kraken universe (500+ coins) for top movers ranked by edge score (volume * change / fees). Returns top 20. Use to discover hot coins beyond the current watchlist.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            }),
        ]
    }

    /// Execute a single tool call and return the result as text.
    pub(crate) async fn execute_exit_tool(
        &self,
        tool_name: &str,
        args: &serde_json::Value,
        features_map: &HashMap<String, serde_json::Value>,
    ) -> String {
        match tool_name {
            "get_coin_features" => self.tool_get_coin_features(args, features_map),
            "get_trade_history" => self.tool_get_trade_history(args),
            "get_market_context" => self.tool_get_market_context(),
            "get_ai_memory" => self.tool_get_nemo_memory(args),
            "get_correlated_coins" => self.tool_get_correlated_coins(args, features_map),
            "get_engine_status" => self.tool_get_engine_status(),
            "get_top_movers" => self.tool_get_top_movers().await,
            _ => format!("Unknown tool: {}", tool_name),
        }
    }

    /// Multi-turn tool-calling exit decision.
    /// Qwen gets a minimal position signal, can call tools for more data, then decides.
    /// Falls back to single-shot get_exit_decision() on any error.
    pub(crate) async fn get_exit_decision_with_tools(
        &self,
        symbol: &str,
        initial_prompt: &str,
        features_map: &HashMap<String, serde_json::Value>,
        ai: &AiBridge,
    ) -> crate::ai_bridge::AiExitDecision {
        let tier = config::coin_tier(symbol);
        let system_prompt = if tier == "meme" {
            crate::ai_bridge::load_nemo_meme_exit_prompt()
        } else {
            crate::ai_bridge::load_nemo_exit_prompt()
        };
        let tools = Self::build_exit_tool_schemas();
        let start = std::time::Instant::now();

        let mut messages: Vec<serde_json::Value> = vec![
            serde_json::json!({
                "role": "system",
                "content": format!(
                    "{}\n\nYou have tools to request additional data. \
                     Call tools if you need more context, or respond directly \
                     with ACTION|CONFIDENCE|REASON if the signal is clear enough.\n/no_think",
                    system_prompt
                )
            }),
            serde_json::json!({
                "role": "user",
                "content": initial_prompt
            }),
        ];

        let max_rounds: u8 = self.config.env.get_parsed("AI_EXIT_TOOL_MAX_ROUNDS", 3u8);
        let tool_max_tokens: i32 = self.config.env.get_parsed("AI_EXIT_TOOL_MAX_TOKENS", 512i32);

        for round in 0..max_rounds {
            let result = ai
                .infer_with_tools_exit(
                    &messages,
                    &tools,
                    self.config.env.get_parsed("AI_EXIT_TEMPERATURE", 0.5f32),
                    tool_max_tokens,
                    Some(false), // /no_think — structured exit ladder, skip hidden reasoning
                )
                .await;

            match result {
                Ok(crate::ai_bridge::InferToolResult::ToolCalls(calls)) => {
                    let tool_names: Vec<&str> =
                        calls.iter().map(|c| c.function_name.as_str()).collect();
                    tracing::info!(
                        "[AI-TOOL] {} round={} tools={:?}",
                        symbol,
                        round + 1,
                        tool_names
                    );

                    // Append assistant's tool-call message
                    let tool_call_json: Vec<serde_json::Value> = calls
                        .iter()
                        .map(|c| {
                            serde_json::json!({
                                "id": c.id,
                                "type": "function",
                                "function": {
                                    "name": c.function_name,
                                    "arguments": serde_json::to_string(&c.arguments).unwrap_or_default()
                                }
                            })
                        })
                        .collect();

                    messages.push(serde_json::json!({
                        "role": "assistant",
                        "tool_calls": tool_call_json
                    }));

                    // Execute each tool and append results
                    for call in &calls {
                        let result_text = self.execute_exit_tool(
                            &call.function_name,
                            &call.arguments,
                            features_map,
                        ).await;
                        tracing::debug!(
                            "[AI-TOOL] {} tool={} result_len={}",
                            symbol,
                            call.function_name,
                            result_text.len()
                        );
                        messages.push(serde_json::json!({
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": result_text
                        }));
                    }
                }
                Ok(crate::ai_bridge::InferToolResult::FinalResponse(text)) => {
                    let ms = start.elapsed().as_millis() as u64;
                    let mut decision = crate::ai_bridge::parse_exit_decision(&text);
                    decision.latency_ms = ms;
                    decision.source = format!("ai_exit_tools_r{}", round + 1);
                    tracing::info!(
                        "[AI-TOOL] {} -> {} conf={:.2} rounds={} reason={} ({ms}ms)",
                        symbol,
                        decision.action,
                        decision.confidence,
                        round + 1,
                        decision.reason
                    );
                    return decision;
                }
                Err(e) => {
                    let ms = start.elapsed().as_millis() as u64;
                    tracing::warn!(
                        "[AI-TOOL] {} tool-calling failed round={}: {} — falling back ({ms}ms)",
                        symbol,
                        round + 1,
                        e
                    );
                    return ai.get_exit_decision(initial_prompt, symbol).await;
                }
            }
        }

        // Exhausted all rounds — force a final answer with no tools
        tracing::info!(
            "[AI-TOOL] {} exhausted {} rounds, forcing final answer",
            symbol,
            max_rounds
        );
        messages.push(serde_json::json!({
            "role": "user",
            "content": "You have used all tool calls. Give your final decision now: ACTION|CONFIDENCE|REASON"
        }));

        let result = ai
            .infer_with_tools_exit(
                &messages,
                &[],
                self.config.env.get_parsed("AI_EXIT_TEMPERATURE", 0.5f32),
                128,
                Some(true),
            )
            .await;

        let ms = start.elapsed().as_millis() as u64;
        match result {
            Ok(crate::ai_bridge::InferToolResult::FinalResponse(text)) => {
                let mut decision = crate::ai_bridge::parse_exit_decision(&text);
                decision.latency_ms = ms;
                decision.source = "ai_exit_tools_forced".into();
                decision
            }
            _ => {
                tracing::error!("[AI-TOOL] {} exit LLM returned no usable response — defaulting HOLD, flagging error", symbol);
                crate::nemo_optimizer::flag_error("exit_ai", &format!("{symbol} exit LLM returned no usable response after all rounds"));
                crate::ai_bridge::AiExitDecision {
                    latency_ms: ms,
                    ..crate::ai_bridge::AiExitDecision::default()
                }
            }
        }
    }

    // ── AI Exit Monitor ───────────────────────────────────────────

    /// Deterministic exit monitor: 4-bucket scoring (trend/momentum/orderflow/whale)
    /// with confirmation counting and sigmoid confidence. Pure math — no LLM needed.
    /// Optionally calls LLM as second opinion when AI_EXIT_LLM=1.
    pub(crate) async fn nemo_exit_check(
        &mut self,
        features_map: &HashMap<String, serde_json::Value>,
        ai: &AiBridge,
        api: Option<&KrakenApi>,
    ) {
        if !self.config.use_nemo_exit {
            return;
        }

        let now = now_ts();
        let base_interval = self.config.nemo_exit_interval_sec;
        let global_min_hold = self.config.nemo_exit_min_hold_sec;
        let fee = self.config.fee_per_side_pct;
        let use_llm = self.config.env.get_bool("AI_EXIT_LLM", false);

        // Collect symbols to check (can't borrow self mutably in loop)
        let to_check: Vec<(String, f64, f64, f64)> = self.positions.iter()
            .filter_map(|(sym, pos)| {
                let hold_sec = pos.hold_seconds();
                // Time contract: use per-position min_hold from entry decision,
                // fall back to global config if entry didn't set one.
                let min_hold = if pos.min_hold_sec > 0 {
                    pos.min_hold_sec as f64
                } else {
                    global_min_hold
                };
                if hold_sec < min_hold {
                    return None;
                }
                // Re-eval interval: use per-position reeval_sec from entry decision.
                // This respects the entry's time intent — trend trades re-eval less often.
                let feats_opt = features_map.get(sym.as_str());
                let price_now = feats_opt
                    .and_then(|f| f.get("price").and_then(|v| v.as_f64()))
                    .unwrap_or(0.0);
                let pnl_now = if price_now > 0.0 { pos.pnl_pct(price_now, fee) } else { 0.0 };
                let interval = if pos.reeval_sec > 0 {
                    // Use entry's time contract for re-eval cadence
                    // But if in profit, check more often (half the reeval interval)
                    if pnl_now > 0.02 {
                        (pos.reeval_sec as f64 * 0.5).max(60.0)
                    } else {
                        pos.reeval_sec as f64
                    }
                } else {
                    // Legacy fallback: old hardcoded intervals
                    let regime = feats_opt
                        .and_then(|f| f.get("regime").and_then(|v| v.as_str()))
                        .unwrap_or("unknown");
                    if pnl_now > 0.0 {
                        120.0
                    } else if regime == "bullish" {
                        base_interval
                    } else {
                        300.0
                    }
                };
                let last_check = self.last_nemo_exit_check.get(sym).copied().unwrap_or(0.0);
                if now - last_check < interval {
                    return None;
                }
                let feats = features_map.get(sym.as_str())?;
                let price = feats.get("price").and_then(|v| v.as_f64()).unwrap_or(0.0);
                if price <= 0.0 {
                    return None;
                }
                let pnl_pct = pos.pnl_pct(price, fee);
                let hold_min = hold_sec / 60.0;
                Some((sym.clone(), price, pnl_pct, hold_min))
            })
            .collect();

        for (sym, price, pnl_pct, hold_min) in &to_check {
            let feats = match features_map.get(sym.as_str()) {
                Some(v) => v,
                None => continue,
            };
            // pnl_gate removed — Nemo can always cut losers

            // Build feature vector for deterministic scoring
            let input = crate::nemo_exit::NemoExitInput {
                pnl_pct: *pnl_pct * 100.0,
                hold_minutes: *hold_min,
                min_hold_minutes: config::coin_optimal_hold_min(sym) * 0.5,
                min_conf: self.config.nemo_exit_min_conf,
                trend_score: feats.get("trend_score").and_then(|v| v.as_f64()).unwrap_or(0.0),
                momentum_score: feats.get("momentum_score").and_then(|v| v.as_f64()).unwrap_or(0.0),
                book_imbalance: feats.get("book_imbalance").and_then(|v| v.as_f64()).unwrap_or(0.0),
                buy_ratio: feats.get("buy_ratio").and_then(|v| v.as_f64()).unwrap_or(0.5),
                whale_score: feats.get("whale_score").and_then(|v| v.as_f64()).unwrap_or(0.0),
                macd_hist: feats.get("macd_hist").and_then(|v| v.as_f64()).unwrap_or(0.0),
                zscore: feats.get("zscore").and_then(|v| v.as_f64()).unwrap_or(0.0),
            };

            let result = crate::nemo_exit::nemo_exit_score(input);

            // Log every evaluation with validity masks
            tracing::info!(
                "[AI-EXIT] {} {} conf={:.2} confirms={}/{} valid={}/3 score={:.2} \
                 T={:.2}{} M={:.2}{} O={:.2}{} W={:.2}{} zpen={:.1} | {}",
                sym, result.action.as_str(), result.confidence,
                result.confirmations, result.required,
                result.valid_votes, result.score,
                result.bucket_trend.sev, if result.bucket_trend.valid { "" } else { "!" },
                result.bucket_momentum.sev, if result.bucket_momentum.valid { "" } else { "!" },
                result.bucket_orderflow.sev, if result.bucket_orderflow.valid { "" } else { "!" },
                result.bucket_whale.sev, if result.bucket_whale.valid { "" } else { "!" },
                result.z_penalty,
                result.reason,
            );

            // Update cooldown regardless of outcome
            self.last_nemo_exit_check.insert(sym.clone(), now_ts());

            // LLM advisory mode: LLM runs and logs opinion, but does NOT execute exits
            let llm_advisory = self.config.env.get_bool("AI_EXIT_LLM_ADVISORY", false);

            // Always run LLM on GPU for every position — she watches everything
            if use_llm {
                let pos = match self.positions.get(sym.as_str()) {
                    Some(p) => p,
                    None => continue,
                };
                let _validity = crate::nemo_exit::validity_summary(&result);
                // MATRIX_EXIT_V2: lean pipe-delimited position signal
                let net_pnl = pnl_pct * 100.0; // already fee-adjusted
                let base_prompt = format!(
                    "POS:{}|{:.2}>{:.2}|NET:{:+.1}%|{:.0}m|R{:.0}|T{:+.0}|M{:+.2}|V{:.1}|MATH:{}|{:.2}|{}/{}",
                    sym, pos.entry_price, price, net_pnl, hold_min,
                    feats.get("rsi").and_then(|v| v.as_f64()).unwrap_or(50.0),
                    feats.get("trend_score").and_then(|v| v.as_f64()).unwrap_or(0.0),
                    feats.get("momentum_score").and_then(|v| v.as_f64()).unwrap_or(0.0),
                    feats.get("vol_ratio").and_then(|v| v.as_f64()).unwrap_or(1.0),
                    result.action.as_str(), result.confidence,
                    result.confirmations, result.required,
                );

                let mut context = String::new();
                let market_short = self.market_brain.build_market_short();
                if !market_short.is_empty() {
                    context.push_str(&market_short);
                    context.push('\n');
                }
                let watch_block = self.market_watcher.build_watch_block();
                if !watch_block.is_empty() {
                    context.push_str(&watch_block);
                    context.push('\n');
                }
                let news_short = self.news_sentiment.build_news_short();
                if !news_short.is_empty() {
                    context.push_str(&news_short);
                    context.push('\n');
                }
                if self.memory_enabled {
                    if let Some(pkt) = self.memory_packet.get() {
                        context.push_str(&pkt.format_for_exit(sym));
                        context.push('\n');
                    }
                }

                // ── NPU Signal Bridge (Intel AI Boost → Nemo context) ──────
                if let Ok(raw) = std::fs::read_to_string("data/npu_signals.json") {
                    if let Ok(npu) = serde_json::from_str::<serde_json::Value>(&raw) {
                        if let Some(sig) = npu["positions"][sym].as_object() {
                            let score    = sig.get("exit_score").and_then(|v| v.as_f64()).unwrap_or(0.0);
                            let decision = sig.get("decision").and_then(|v| v.as_str()).unwrap_or("HOLD");
                            let momentum = sig.get("momentum").and_then(|v| v.as_f64()).unwrap_or(0.0);
                            let drawdown = sig.get("drawdown_pct").and_then(|v| v.as_f64()).unwrap_or(0.0);
                            let win_prob = sig.get("win_prob").and_then(|v| v.as_f64()).unwrap_or(0.5);
                            let urgency  = sig.get("urgency").and_then(|v| v.as_f64()).unwrap_or(0.0);
                            let npu_line = format!(
                                "NPU:{} score={:.0} momentum={:+.0} drawdown={:.1}% winprob={:.0}% urgency={:.2}",
                                decision, score, momentum, drawdown, win_prob * 100.0, urgency
                            );
                            context.push_str(&npu_line);
                            context.push('\n');
                            if score >= 70.0 {
                                tracing::info!(
                                    "[NPU-SIGNAL] {} → {} score={:.0} urgency={:.2} — injected into Nemo",
                                    sym, decision, score, urgency
                                );
                            }
                        }
                    }
                }

                let prompt = if context.is_empty() {
                    base_prompt
                } else {
                    format!("{}\n{}", context.trim_end(), base_prompt)
                };
                // Exit brain gets position data only — no context blocks
                let exit_tools_enabled = self.config.env.get_bool("AI_EXIT_TOOLS", false);
                // Only use tool-calling for ambiguous exits: NET between -1.5% and +2.0%
                // AND math/signals are not decisive. Clear trend breaks and clear profits
                // don't need tools — just answer directly for speed.
                let net_pnl_pct = pnl_pct * 100.0;
                let math_decisive = result.confirmations >= result.required && result.confidence > 0.75;
                let use_exit_tools = exit_tools_enabled
                    && net_pnl_pct > -1.5 && net_pnl_pct < 2.0
                    && !math_decisive;
                let llm = if use_exit_tools {
                    self.get_exit_decision_with_tools(sym, &prompt, features_map, ai).await
                } else {
                    ai.get_exit_decision(&prompt, sym).await
                };
                self.meters.bump_nemo_exit(now_ts(), &llm.action);
                self.record_nemo_memory(sym, &llm.action, llm.confidence, &llm.reason, *price, "", "ai_exit");

                // Log AI_2 decision for 49B optimizer
                crate::engine::decision_log::log_ai2(&crate::engine::decision_log::Ai2Decision {
                    ts:          now,
                    symbol:      sym.clone(),
                    action:      llm.action.clone(),
                    confidence:  llm.confidence,
                    reason:      llm.reason.clone(),
                    pnl_pct:     pnl_pct * 100.0,
                    hold_min:    *hold_min,
                    regime:      feats.get("quantum_state").and_then(|v| v.as_str()).unwrap_or("unknown").to_string(),
                    latency_ms:  llm.latency_ms,
                });

                // Push any SCAN suggestion to the swap queue for vision scanner
                // Enforce swap-only: adds must not exceed removes (WS list can't grow)
                if let Some(scan) = &llm.scan {
                    // Filter out meme coins from removal suggestions — memes are protected
                    let filtered_remove: Vec<String> = scan.remove.iter()
                        .filter(|s| config::coin_tier(s) != "meme")
                        .cloned()
                        .collect();
                    if scan.add.len() > filtered_remove.len() {
                        tracing::warn!(
                            "[SCAN] Rejected: {} adds > {} removes (after meme protection) — must swap, not grow",
                            scan.add.len(), filtered_remove.len()
                        );
                    } else if let Some(ref sq) = self.swap_queue {
                        let suggestion = crate::vision_scanner::CoinSwapSuggestion {
                            add: scan.add.clone(),
                            remove: filtered_remove,
                            reason: scan.reason.clone(),
                            timestamp: now_ts(),
                        };
                        tracing::info!(
                            "[SCAN] Queued swap: add={:?} remove={:?} reason={}",
                            suggestion.add, suggestion.remove, suggestion.reason
                        );
                        sq.write().await.push(suggestion);
                    }
                }

                if llm_advisory {
                    // Advisory mode: log LLM opinion but do NOT execute exits from LLM path
                    tracing::info!(
                        "[AI-ADVISORY] {} math={} llm={} llm_conf={:.2} (advisory-only) | {}",
                        sym, result.action.as_str(), llm.action, llm.confidence, llm.reason
                    );
                    // Math-only exit: deterministic stack still has authority
                    if result.action == crate::nemo_exit::NemoAction::Sell {
                        let reason = format!("ai_math(llm_advisory={}/{:.2} {})",
                            llm.action, llm.confidence, result.reason);
                        self.execute_exit(sym, *price, &reason, api, Some(ai)).await;
                    }
                } else {
                    tracing::info!(
                        "[AI-GPU] {} math={} llm={} llm_conf={:.2} | {}",
                        sym, result.action.as_str(), llm.action, llm.confidence, llm.reason
                    );

                    // SELL requires BOTH math and LLM to agree
                    if result.action == crate::nemo_exit::NemoAction::Sell && llm.action == "SELL" {
                        let reason = format!("ai_dual(math+llm conf={:.2}/{:.2} {})",
                            result.confidence, llm.confidence, result.reason);
                        self.execute_exit(sym, *price, &reason, api, Some(ai)).await;
                        continue;
                    }

                    // LLM can also trigger SELL independently if very confident
                    if llm.action == "SELL" && llm.confidence >= 0.85 {
                        let reason = format!("ai_llm(conf={:.2} {})", llm.confidence, llm.reason);
                        tracing::info!("[AI-GPU] {} LLM override SELL (conf={:.2})", sym, llm.confidence);
                        self.execute_exit(sym, *price, &reason, api, Some(ai)).await;
                        continue;
                    }
                }
            } else {
                // No LLM — math-only mode
                self.meters.bump_nemo_exit(now_ts(), result.action.as_str());
                if result.action == crate::nemo_exit::NemoAction::Sell {
                    let reason = format!("ai_math({})", result.reason);
                    self.execute_exit(sym, *price, &reason, api, Some(ai)).await;
                }
            }
        }
    }

    // ── Exit Logic ───────────────────────────────────────────────

    /// ATR trailing multiplier by lane — read from env with lane-specific defaults.
    fn atr_trail_mult_for_lane(&self, lane: &str) -> f64 {
        match lane {
            "L1" => self.config.env.get_f64("ATR_TRAIL_L1", 2.0),
            "L2" => self.config.env.get_f64("ATR_TRAIL_L2", 1.3),
            "L3" => self.config.env.get_f64("ATR_TRAIL_L3", 1.5),
            "L4" => self.config.env.get_f64("ATR_TRAIL_L4", 1.0),
            _ => self.config.env.get_f64("ATR_TRAIL_L3", 1.5),
        }
    }

    /// Lightweight exit check for real-time WS trade prices.
    /// Called on every incoming trade event — only checks hard-risk thresholds
    /// (SL, TP, trailing stop, crash). No features or AI needed.
    pub fn check_instant_exit(&mut self, symbol: &str, price: f64) -> Option<String> {
        // Pre-read ATR trail config to avoid borrow conflict with positions.get_mut()
        let atr_trail_l1 = self.config.env.get_f64("ATR_TRAIL_L1", 2.0);
        let atr_trail_l2 = self.config.env.get_f64("ATR_TRAIL_L2", 1.3);
        let atr_trail_l3 = self.config.env.get_f64("ATR_TRAIL_L3", 1.5);
        let atr_trail_l4 = self.config.env.get_f64("ATR_TRAIL_L4", 1.0);
        let trailing_activate = self.config.trailing_stop_activate_pct;
        let trailing_static = self.config.trailing_stop_pct;
        let btc_crashing = self.market_brain.is_btc_crashing();

        let pos = self.positions.get_mut(symbol)?;
        let fee = self.config.fee_per_side_pct;
        let pnl_pct = pos.pnl_pct(price, fee);

        // Update highest price for trailing stop tracking
        pos.highest_price = pos.highest_price.max(price);

        // Stop loss
        if price <= pos.sl_price {
            return Some(format!("instant_stop_loss({:.2}%)", pnl_pct * 100.0));
        }
        // Take profit
        if price >= pos.tp_price {
            return Some(format!("instant_take_profit({:.2}%)", pnl_pct * 100.0));
        }
        // ATR-based trailing stop (lane-adaptive) — falls back to static if entry_atr==0
        if pnl_pct >= trailing_activate / 100.0 {
            let trail_price = if pos.entry_atr > 0.0 {
                let mult = match pos.entry_lane.as_str() {
                    "L1" => atr_trail_l1,
                    "L2" => atr_trail_l2,
                    "L3" => atr_trail_l3,
                    "L4" => atr_trail_l4,
                    _ => atr_trail_l3,
                };
                let trail_dist = pos.entry_atr * mult;
                pos.highest_price - trail_dist
            } else {
                pos.highest_price * (1.0 - trailing_static / 100.0)
            };
            if price <= trail_price {
                return Some(format!("instant_atr_trailing_stop({:.2}%)", pnl_pct * 100.0));
            }
        }
        // Regime crash — if market brain detects crash, exit immediately
        if btc_crashing && pnl_pct < 0.0 {
            return Some("instant_regime_crash".into());
        }
        None
    }

    /// Check if a position should be exited. Returns exit reason if yes.
    pub(crate) fn check_exit(
        &self,
        symbol: &str,
        feats: &serde_json::Value,
        price: f64,
        regime: &str,
    ) -> Option<String> {
        let pos = self.positions.get(symbol)?;
        let fee = self.config.fee_per_side_pct;
        let pnl_pct = pos.pnl_pct(price, fee);
        let hold_sec = pos.hold_seconds();
        let hold_minutes = hold_sec / 60.0;
        // Time contract: tiered hold based on multi-timeframe trend alignment.
        // alignment=3 (all 3 TFs green) → 12h hold to let winners run.
        // alignment=2 (1d+7d) → 3h hold.
        // alignment=1 or 0 → 30 min hold (short-term scalp mode).
        // Per-position explicit min_hold_sec from AI entry always takes priority.
        let min_hold_min = if pos.min_hold_sec > 0 {
            pos.min_hold_sec as f64 / 60.0
        } else {
            let hold_short  = self.config.env.get_f64("HOLD_MIN_SHORT",  1800.0) / 60.0;
            let hold_medium = self.config.env.get_f64("HOLD_MIN_MEDIUM", 10800.0) / 60.0;
            let hold_long   = self.config.env.get_f64("HOLD_MIN_LONG",   43200.0) / 60.0;
            match pos.trend_alignment {
                3 => hold_long,
                2 => hold_medium,
                _ => hold_short,
            }
        };
        let before_min_hold = hold_minutes < min_hold_min;

        // Stop loss
        if price <= pos.sl_price {
            return Some(format!("stop_loss({:.2}%)", pnl_pct * 100.0));
        }

        // Take profit
        if price >= pos.tp_price {
            return Some(format!("take_profit({:.2}%)", pnl_pct * 100.0));
        }

        // Stale thesis: max_hold_sec exceeded — entry thesis is likely wrong or stale.
        // Exit unless currently in strong profit (let winners run overrides staleness).
        if pos.max_hold_sec > 0 && hold_sec > pos.max_hold_sec as f64 {
            if pnl_pct < 0.02 {
                return Some(format!(
                    "stale_thesis(held {:.0}m > max {}m, pnl={:.2}%)",
                    hold_minutes, pos.max_hold_sec / 60, pnl_pct * 100.0,
                ));
            }
        }

        // Avoid fee churn from micro exits before minimum hold window.
        // Keep hard-risk exits (SL/regime crash) always-on.
        if !before_min_hold {
            // ── Profit ladder trailing stop ──────────────────────────────
            // As the position reaches higher profit peaks, the trail tightens
            // to lock in more of the gain. Fires before ATR trail.
            let peak_pct = (pos.highest_price - pos.entry_price) / pos.entry_price * 100.0;
            let l4_pct  = self.config.env.get_f64("HOLD_L4_PCT",  10.0);
            let l3_pct  = self.config.env.get_f64("HOLD_L3_PCT",   7.0);
            let l2_pct  = self.config.env.get_f64("HOLD_L2_PCT",   4.0);
            let l1_pct  = self.config.env.get_f64("HOLD_L1_PCT",   2.0);
            let l4_stop = self.config.env.get_f64("HOLD_L4_STOP",  0.88);
            let l3_stop = self.config.env.get_f64("HOLD_L3_STOP",  0.80);
            let l2_stop = self.config.env.get_f64("HOLD_L2_STOP",  0.70);
            let l1_stop = self.config.env.get_f64("HOLD_L1_STOP",  0.60);
            let ladder_frac = if peak_pct >= l4_pct      { l4_stop }
                              else if peak_pct >= l3_pct { l3_stop }
                              else if peak_pct >= l2_pct { l2_stop }
                              else if peak_pct >= l1_pct { l1_stop }
                              else                        { 0.0 };
            if ladder_frac > 0.0 {
                let ladder_floor = pos.highest_price * ladder_frac;
                if price <= ladder_floor {
                    return Some(format!(
                        "ladder_trail(peak={:.2}%,lock={:.0}%,pnl={:.2}%)",
                        peak_pct, ladder_frac * 100.0, pnl_pct * 100.0
                    ));
                }
            }

            // ATR-based trailing stop (lane-adaptive) — falls back to static if entry_atr==0
            if pnl_pct >= self.config.trailing_stop_activate_pct / 100.0 {
                let trail_price = if pos.entry_atr > 0.0 {
                    let mult = self.atr_trail_mult_for_lane(&pos.entry_lane);
                    let trail_dist = pos.entry_atr * mult;
                    pos.highest_price - trail_dist
                } else {
                    pos.highest_price * (1.0 - self.config.trailing_stop_pct / 100.0)
                };
                if price <= trail_price {
                    return Some(format!("atr_trailing_stop({:.2}%)", pnl_pct * 100.0));
                }
            }

            // Breakeven stop: once position reached breakeven_activate_pct above entry,
            // exit if price drops back below entry.
            let peak_gain_pct = (pos.highest_price - pos.entry_price) / pos.entry_price;
            if peak_gain_pct >= self.config.breakeven_activate_pct / 100.0
                && price <= pos.entry_price
            {
                return Some(format!("breakeven_stop({:.2}%)", pnl_pct * 100.0));
            }

            // Score drop below exit threshold (gated — 0% win rate historically)
            if self.config.enable_score_drop {
                let score = feats
                    .get("weighted_score")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0);
                if score < -self.config.exit_threshold && pnl_pct < -0.005 {
                    return Some(format!("score_drop({score:.3})"));
                }
            }

            // Book reversal (strong bearish)
            let book_reversal = feats
                .get("book_reversal")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let book_dir = feats
                .get("book_reversal_dir")
                .and_then(|v| v.as_i64())
                .unwrap_or(0);
            // -1 = bull-to-bear reversal, +1 = bear-to-bull
            if book_reversal && book_dir == -1 && pnl_pct < 0.01 {
                return Some("book_reversal_bearish".into());
            }
        }

        // Regime crash → immediate exit
        if regime == "crash" {
            return Some("regime_crash".into());
        }

        None
    }

    // ── Exit Execution ───────────────────────────────────────────

    pub(crate) async fn execute_exit(
        &mut self,
        symbol: &str,
        price: f64,
        reason: &str,
        api: Option<&KrakenApi>,
        ai: Option<&crate::ai_bridge::AiBridge>,
    ) {
        let pos = match self.positions.get(symbol) {
            Some(p) => p.clone(),
            None => return,
        };

        let fee = self.config.fee_per_side_pct;
        let pnl_pct = pos.pnl_pct(price, fee);
        let pnl_usd = pos.pnl_usd(price, fee);
        let hold_sec = pos.hold_seconds();
        let pair = config::pair_for(symbol);

        // Cancel any stop-loss order FIRST — it locks the coins on Kraken
        if let Some(ref sl_txid) = pos.sl_order_txid {
            if let Some(api) = api {
                match api.cancel_order(sl_txid).await {
                    Ok(_) => tracing::info!("[SL-CANCEL] {} cancelled sl_txid={}", symbol, sl_txid),
                    Err(e) => tracing::warn!("[SL-CANCEL] {} failed (may already be gone): {e}", symbol),
                }
            }
        }

        // Dust detection: if position value < Kraken minimum ($5), write off as dust
        let position_usd = pos.remaining_qty * price;
        let is_dust = position_usd < 5.0 && !self.config.paper_trading;

        // Place market sell order — position stays in state until confirmed
        let _sell_limit = config::format_price(symbol, price);
        if is_dust {
            tracing::warn!(
                "[DUST-EXIT] {} val=${:.2} < $5 minimum — writing off as dust (qty={:.8})",
                symbol, position_usd, pos.remaining_qty
            );
            // Skip API sell — fall through to cleanup below
        } else if self.config.paper_trading {
            tracing::info!(
                "[PAPER-EXIT] {} qty={:.6} limit={:.4} pnl={:+.2}% ${:+.2} reason={}",
                symbol,
                pos.remaining_qty,
                price,
                pnl_pct * 100.0,
                pnl_usd,
                reason
            );
        } else if let Some(api) = api {
            let vol_str = format!("{:.8}", pos.remaining_qty);
            match api.add_order(&pair, "sell", "market", &vol_str, None).await {
                Ok(_) => tracing::info!("[EXIT-SOLD] {} qty={:.8} @ {:.6}", symbol, pos.remaining_qty, price),
                Err(e) => {
                    let e_str = e.to_string();
                    if e_str.contains("Insufficient funds") || e_str.contains("EOrder:Insufficient") {
                        // State mismatch — Kraken has less than bot thinks (partial fill / dust).
                        // Writing off position to prevent repeated failed orders and account lockout.
                        tracing::warn!(
                            "[DUST-WRITEOFF] {} insufficient funds on sell (val=${:.2}) — writing off dust position",
                            symbol, position_usd
                        );
                        // Fall through to cleanup below
                    } else {
                        tracing::error!("[EXIT-FAIL] {} sell failed, position KEPT in state: {e}", symbol);
                        crate::nemo_optimizer::flag_error("exit_order", &format!("{symbol} sell order failed: {e}"));
                        return; // Don't remove position — sell didn't go through
                    }
                }
            }
        }

        // Sell confirmed (or paper) — NOW remove position from state
        self.positions.remove(symbol);
        self.last_nemo_exit_check.remove(symbol);

        // Track P&L and return funds to available balance
        self.circuit.add_pnl(pnl_usd);
        self.optimizer.close_position(symbol, price);
        let returned_usd = pos.remaining_qty * price;
        self.available_usd += returned_usd;

        // Points (simple: +2 for win, -1 for loss)
        let points = if pnl_pct > 0.001 { 2 } else { -1 };
        let result = TradeRecord::classify_result(pnl_pct);

        // Journal trade
        self.journal.record_trade(TradeRecord {
            symbol: symbol.to_string(),
            entry_time: pos.entry_time,
            exit_time: now_ts(),
            entry_price: pos.entry_price,
            exit_price: price,
            quantity: pos.remaining_qty,
            pnl_percent: pnl_pct,
            pnl_usd,
            result: result.to_string(),
            hold_minutes: hold_sec / 60.0,
            entry_reasons: pos.entry_reasons.clone(),
            exit_reason: reason.to_string(),
            entry_context: pos.entry_context.clone(),
            points,
            points_reason: format!("{result} {reason}"),
            feature_snapshot: pos.feature_snapshot.clone(),
            trend_alignment: pos.trend_alignment,
            trend_7d_pct: pos.trend_7d_pct,
            trend_30d_pct: pos.trend_30d_pct,
        });

        // Record GBDT training sample (feature snapshot → trade outcome)
        if let Some(ref snapshot) = pos.feature_snapshot {
            let mut gbdt = self.gbdt.write().await;
            gbdt.record_sample(snapshot.clone(), pnl_pct);
        }

        // Record outcome for AI's conscious memory (track record + episodes)
        // Skip dust sweeps — they pollute memory with fake losses
        if reason != "dust_sweep" {
            let outcome = crate::nemo_memory::NemoMemory::build_outcome(
                symbol, pnl_pct, hold_sec / 60.0, reason,
                &pos.entry_reasons, &pos.regime_label,
            );
            self.nemo_memory_brain.record_outcome(outcome.clone());
            self.memory_packet.invalidate();

            // Embed trade outcome for NVIDIA semantic memory
            if let Some(ai) = ai {
                self.nemo_memory_brain
                    .embed_and_store_outcome(&outcome, ai.client())
                    .await;
            }
        }

        // ── HF: Update adaptive confidence floor on every trade close ──
        self.update_adaptive_conf_floor(pnl_pct);

        // ── HF: Update Bayesian quality stats for setup bucket ──
        if !pos.quality_key.is_empty() {
            let prior_a = self.config.env.get_f64("HF_QL_PRIOR_A", 2.0);
            let prior_b = self.config.env.get_f64("HF_QL_PRIOR_B", 2.0);
            let s = self.quality_stats.entry(pos.quality_key.clone())
                .or_insert(BetaStats::new(prior_a, prior_b));
            if pnl_pct > 0.001 {
                s.a += 1.0;
            } else if pnl_pct < -0.001 {
                s.b += 1.0;
            }
            tracing::info!(
                "[HF-QL] {} bucket=\"{}\" a={:.0} b={:.0} p_win={:.2}%",
                symbol, pos.quality_key, s.a, s.b, s.mean() * 100.0
            );
            save_quality_stats(&self.quality_stats);
        }

        save_positions(&self.config.positions_file, &self.positions);

        tracing::info!(
            "[EXIT] {} price={:.4} pnl={:+.2}% ${:+.2} hold={:.0}m reason={} regime={} entry_score={:.2}",
            symbol,
            price,
            pnl_pct * 100.0,
            pnl_usd,
            hold_sec / 60.0,
            reason,
            pos.regime_label,
            pos.entry_score,
        );
        log_event(
            "info",
            "exit_executed",
            json!({
                "symbol": symbol,
                "price": price,
                "pnl_pct": pnl_pct,
                "pnl_usd": pnl_usd,
                "hold_sec": hold_sec,
                "reason": reason
            }),
        );

        // ── AI Trade Reviewer (background) ──
        // Post-mortem analysis of every closed trade
        {
            let review_data = format!(
                "CLOSED TRADE: {} entry=${:.4} exit=${:.4} pnl={:+.2}% hold={:.0}min \
                 entry_reason={} exit_reason={} regime={}",
                symbol, pos.entry_price, price, pnl_pct * 100.0, hold_sec / 60.0,
                pos.entry_reasons.join(","),
                reason,
                pos.regime_label,
            );
            let sym_str = symbol.to_string();
            let ollama_url = self.config.env.get_str("OLLAMA_HOST", "http://127.0.0.1:11434");
            let model = self.config.env.get_str("OLLAMA_MODEL", "qwen2.5-14b-instruct");
            tokio::spawn(async move {
                let client = reqwest::Client::builder()
                    .timeout(std::time::Duration::from_secs(30))
                    .build().unwrap_or_else(|_| reqwest::Client::new());
                let system = crate::ai_bridge::load_nemo_review_prompt();
                let req_url = format!("{ollama_url}/v1/chat/completions");
                let body = serde_json::json!({
                    "model": model, "stream": false,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": review_data}
                    ],
                    "temperature": 0.5, "max_tokens": 512
                });
                let start = tokio::time::Instant::now();
                match client.post(&req_url).json(&body).send().await {
                    Ok(resp) => {
                        let ms = start.elapsed().as_millis();
                        if let Ok(json) = resp.json::<serde_json::Value>().await {
                            let raw = json["choices"][0]["message"]["content"]
                                .as_str().unwrap_or("").trim();
                            if let Some(line) = raw.lines().find(|l| l.starts_with("REVIEW|")) {
                                tracing::info!("[AI-REVIEW] {} {} ({ms}ms)", sym_str, line);
                            } else {
                                tracing::info!("[AI-REVIEW] {} raw: {} ({ms}ms)", sym_str,
                                    raw.lines().next().unwrap_or(""));
                            }
                        }
                    }
                    Err(e) => tracing::warn!("[AI-REVIEW] {} failed: {e}", sym_str),
                }
            });
        }
    }
}
