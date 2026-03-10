//! Entry logic — entry confirmation, tool-calling, position sizing, order placement.

use super::*;

// ── MTF Trend Helpers — reads data/mtf_trends.json written by mover_scanner ──

fn read_mtf_json(symbol: &str) -> Option<serde_json::Value> {
    let raw = std::fs::read_to_string("data/mtf_trends.json").ok()?;
    let map: serde_json::Value = serde_json::from_str(&raw).ok()?;
    map.get(symbol).or_else(|| map.get(&symbol.to_uppercase())).cloned()
}

pub fn read_mtf_trend_7d(symbol: &str) -> f64 {
    read_mtf_json(symbol)
        .and_then(|v| v.get("trend_7d").and_then(|x| x.as_f64()))
        .unwrap_or(0.0)
}

pub fn read_mtf_trend_30d(symbol: &str) -> f64 {
    read_mtf_json(symbol)
        .and_then(|v| v.get("trend_30d").and_then(|x| x.as_f64()))
        .unwrap_or(0.0)
}

pub fn read_mtf_alignment(symbol: &str) -> u8 {
    let entry = match read_mtf_json(symbol) {
        Some(v) => v,
        None => return 0,
    };
    let _trend_1d_min  = std::env::var("TREND_1D_MIN").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(1.0);
    let trend_7d_min  = std::env::var("TREND_7D_MIN").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(4.0);
    let trend_30d_min = std::env::var("TREND_30D_MIN").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(5.0);
    let t7d  = entry.get("trend_7d").and_then(|x| x.as_f64()).unwrap_or(0.0);
    let t30d = entry.get("trend_30d").and_then(|x| x.as_f64()).unwrap_or(0.0);
    // 1d alignment assumed if coin is in the entry pipeline (quant already checked momentum)
    let ok_1d  = true;
    let ok_7d  = t7d  >= trend_7d_min;
    let ok_30d = t30d >= trend_30d_min;
    match (ok_1d, ok_7d, ok_30d) {
        (true, true, true)  => 3,
        (true, true, false) => 2,
        (true, false, _)    => 1,
        _                   => 0,
    }
}

impl super::TradingLoop {
    // ── Entry Tool-Calling Support ────────────────────────────────

    /// Build OpenAI-compatible tool definitions for entry decisions.
    /// Same 5 tools as exit + get_open_positions + update_stop_loss.
    pub(crate) fn build_entry_tool_schemas() -> Vec<serde_json::Value> {
        let mut tools = Self::build_exit_tool_schemas();
        tools.push(serde_json::json!({
            "type": "function",
            "function": {
                "name": "get_open_positions",
                "description": "Get all currently open positions: symbol, entry price, current PnL%, hold time, and USD value. Use this to check exposure before confirming a new entry.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }));
        tools.push(serde_json::json!({
            "type": "function",
            "function": {
                "name": "update_stop_loss",
                "description": "Adjust STOP_LOSS_PCT in .env. Use when volatility suggests current stop is too tight or wide. Range: 0.8–4.0.",
                "parameters": { "type": "object", "properties": {
                    "stop_loss_pct": { "type": "number", "description": "New SL % (0.8–4.0)" },
                    "reason": { "type": "string" }
                }, "required": ["stop_loss_pct", "reason"] }
            }
        }));
        tools.push(serde_json::json!({
            "type": "function",
            "function": {
                "name": "update_take_profit",
                "description": "Adjust TAKE_PROFIT_PCT in .env. Use in high-momentum setups to let winners run further. Range: 1.0–6.0.",
                "parameters": { "type": "object", "properties": {
                    "take_profit_pct": { "type": "number", "description": "New TP % (1.0–6.0)" },
                    "reason": { "type": "string" }
                }, "required": ["take_profit_pct", "reason"] }
            }
        }));
        tools.push(serde_json::json!({
            "type": "function",
            "function": {
                "name": "get_volatility_regime",
                "description": "Get current volatility regime for a coin: EXPANDING/CONTRACTING/NORMAL, ATR%, suggested stop multiplier.",
                "parameters": { "type": "object", "properties": {
                    "symbol": { "type": "string" }
                }, "required": ["symbol"] }
            }
        }));
        tools.push(serde_json::json!({
            "type": "function",
            "function": {
                "name": "get_position_health",
                "description": "Assess health of an open position: PnL, drawdown from peak, hold time, stop distance. Use before opening new position.",
                "parameters": { "type": "object", "properties": {
                    "symbol": { "type": "string" }
                }, "required": ["symbol"] }
            }
        }));
        tools.push(serde_json::json!({
            "type": "function",
            "function": {
                "name": "get_sentiment_score",
                "description": "Get Atlas NPU sentiment score and Fear/Greed for a coin. Positive = bullish news momentum.",
                "parameters": { "type": "object", "properties": {
                    "symbol": { "type": "string" }
                }, "required": ["symbol"] }
            }
        }));
        tools.push(serde_json::json!({
            "type": "function",
            "function": {
                "name": "get_orderbook_depth",
                "description": "Get orderbook imbalance, buy pressure, spread, and book strength for a coin. High imbalance = imminent move.",
                "parameters": { "type": "object", "properties": {
                    "symbol": { "type": "string" }
                }, "required": ["symbol"] }
            }
        }));
        tools
    }

    /// Execute a tool call for entry decisions (superset of exit tools + positions).
    pub(crate) async fn execute_entry_tool(
        &self,
        tool_name: &str,
        args: &serde_json::Value,
        features_map: &HashMap<String, serde_json::Value>,
    ) -> String {
        let now = crate::engine::helpers::now_ts();
        match tool_name {
            "get_open_positions"    => self.tool_get_open_positions(features_map),
            "update_stop_loss"      => self.tool_update_stop_loss(args),
            "update_take_profit"    => self.tool_update_take_profit(args),
            "get_volatility_regime" => self.tool_get_volatility_regime(args, features_map),
            "get_position_health"   => self.tool_get_position_health(args, features_map, now),
            "get_sentiment_score"   => self.tool_get_sentiment_score(args, features_map),
            "get_orderbook_depth"   => self.tool_get_orderbook_depth(args, features_map),
            _ => self.execute_exit_tool(tool_name, args, features_map).await,
        }
    }

    /// Tool: update_stop_loss — 9B can adjust STOP_LOSS_PCT based on market conditions.
    pub(crate) fn tool_update_stop_loss(&self, args: &serde_json::Value) -> String {
        let pct = match args.get("stop_loss_pct").and_then(|v| v.as_f64()) {
            Some(v) => v,
            None => return "ERROR: stop_loss_pct missing".to_string(),
        };
        let reason = args.get("reason")
            .and_then(|v| v.as_str())
            .unwrap_or("no reason given");

        // Safety bounds — 9B cannot set outside 0.8%–4.0%
        if !(0.8..=4.0).contains(&pct) {
            return format!("REJECTED: {:.2}% is outside allowed range 0.8–4.0%", pct);
        }

        // Read current value for comparison
        let current = self.config.env.get_f64("STOP_LOSS_PCT", 1.5);

        // Write to .env via same mechanism as 49B optimizer
        match crate::nemo_optimizer::apply_single_param_change("STOP_LOSS_PCT", pct, reason) {
            Ok(_) => {
                tracing::info!(
                    "[TOOL-SL] 9B updated STOP_LOSS_PCT {:.2}% -> {:.2}% | reason: {}",
                    current, pct, reason
                );
                crate::nemo_optimizer::flag_error(
                    "tool_update_stop_loss",
                    &format!("9B set SL={:.2}% (was {:.2}%): {}", pct, current, reason)
                );
                format!("OK: STOP_LOSS_PCT updated {:.2}% -> {:.2}%. Reason: {}", current, pct, reason)
            }
            Err(e) => format!("ERROR: could not write .env: {}", e),
        }
    }

    /// Tool: update_take_profit — 9B adjusts TAKE_PROFIT_PCT based on momentum strength.
    pub(crate) fn tool_update_take_profit(&self, args: &serde_json::Value) -> String {
        let pct = match args.get("take_profit_pct").and_then(|v| v.as_f64()) {
            Some(v) => v,
            None => return "ERROR: take_profit_pct missing".to_string(),
        };
        let reason = args.get("reason").and_then(|v| v.as_str()).unwrap_or("no reason");
        if !(1.0..=6.0).contains(&pct) {
            return format!("REJECTED: {:.2}% outside allowed 1.0–6.0%", pct);
        }
        let current = self.config.env.get_f64("TAKE_PROFIT_PCT", 2.0);
        match crate::nemo_optimizer::apply_single_param_change("TAKE_PROFIT_PCT", pct, reason) {
            Ok(_) => {
                tracing::info!("[TOOL-TP] 9B updated TAKE_PROFIT_PCT {:.2}% -> {:.2}% | {}", current, pct, reason);
                format!("OK: TAKE_PROFIT_PCT updated {:.2}% -> {:.2}%. Reason: {}", current, pct, reason)
            }
            Err(e) => format!("ERROR: {}", e),
        }
    }

    /// Tool: get_open_positions — shows current portfolio exposure.
    pub(crate) fn tool_get_open_positions(&self, features_map: &HashMap<String, serde_json::Value>) -> String {
        if self.positions.is_empty() {
            return "No open positions. Portfolio is 100% cash.".to_string();
        }
        let fee = self.config.fee_per_side_pct;
        let mut out = format!("OPEN POSITIONS ({}):\n", self.positions.len());
        for (sym, pos) in &self.positions {
            let price = features_map.get(sym.as_str())
                .and_then(|f| f.get("price").and_then(|v| v.as_f64()))
                .unwrap_or(pos.entry_price);
            let pnl = pos.pnl_pct(price, fee) * 100.0;
            let hold_min = pos.hold_seconds() / 60.0;
            let usd_val = pos.remaining_qty * price;
            out.push_str(&format!(
                "  {} entry=${:.4} now=${:.4} PnL={:+.2}% hold={:.0}min ${:.2}\n",
                sym, pos.entry_price, price, pnl, hold_min, usd_val,
            ));
        }
        out.push_str(&format!("PENDING ENTRIES: {}\n", self.pending_entries.len()));
        out.push_str(&format!("MAX POSITIONS: {}", self.config.def_max_positions));
        out
    }

    /// Multi-turn tool-calling entry confirmation for a single candidate.
    /// Qwen sees candidate data, can call tools for more context, then confirms BUY or HOLD.
    /// Falls back to single-shot batch confirmation on error.
    pub(crate) async fn get_entry_confirmation_with_tools(
        &self,
        symbol: &str,
        candidate_prompt: &str,
        features_map: &HashMap<String, serde_json::Value>,
        ai: &AiBridge,
    ) -> crate::ai_bridge::AiExitDecision {
        let tier = config::coin_tier(symbol);
        let system_prompt = if tier == "meme" {
            crate::ai_bridge::load_nemo_meme_entry_prompt()
        } else {
            crate::ai_bridge::load_nemo_entry_prompt()
        };
        let tools = Self::build_entry_tool_schemas();
        let start = std::time::Instant::now();

        let mut messages: Vec<serde_json::Value> = vec![
            serde_json::json!({
                "role": "system",
                "content": format!(
                    "{}\n\nYou have tools to request additional data before deciding. \
                     Call tools if you need more context (trade history, sector peers, \
                     current positions), or respond directly with \
                     SYMBOL|BUY or HOLD|CONFIDENCE|REASON if the signal is clear.\n/no_think",
                    system_prompt
                )
            }),
            serde_json::json!({
                "role": "user",
                "content": candidate_prompt
            }),
        ];

        let base_rounds: u8 = self.config.env.get_parsed("AI_ENTRY_TOOL_MAX_ROUNDS", 3u8);
        // Schedule + hardware guard cap tool rounds
        let sched = crate::config::schedule::current_overlay(&self.config.env);
        let max_rounds = self.effective_tool_rounds(base_rounds.min(sched.max_tool_rounds));
        let tool_max_tokens: i32 = self.config.env.get_parsed("AI_ENTRY_TOOL_MAX_TOKENS", 512i32);
        let temperature: f32 = self.config.env.get_parsed("MODEL_TEMPERATURE", 0.5f32);
        let tool_think_override = if self.config.env.map.contains_key("AI_ENTRY_TOOL_THINK") {
            Some(self.config.env.get_bool("AI_ENTRY_TOOL_THINK", false))
        } else {
            None
        };
        for round in 0..max_rounds {
            let result = ai
                .infer_with_tools_entry(
                    &messages,
                    &tools,
                    temperature,
                    tool_max_tokens,
                    tool_think_override, // override THINK_MODE only if AI_ENTRY_TOOL_THINK is set
                )
                .await;

            match result {
                Ok(crate::ai_bridge::InferToolResult::ToolCalls(calls)) => {
                    let tool_names: Vec<&str> =
                        calls.iter().map(|c| c.function_name.as_str()).collect();
                    tracing::info!(
                        "[AI-ENTRY-TOOL] {} round={} tools={:?}",
                        symbol, round + 1, tool_names
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
                        let result_text = self.execute_entry_tool(
                            &call.function_name,
                            &call.arguments,
                            features_map,
                        ).await;
                        tracing::debug!(
                            "[AI-ENTRY-TOOL] {} tool={} result_len={}",
                            symbol, call.function_name, result_text.len()
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
                    // Parse entry response: SYMBOL|BUY/HOLD|CONFIDENCE|REASON
                    let mut decision = Self::parse_entry_tool_response(&text, symbol);
                    decision.latency_ms = ms;
                    decision.source = format!("ai_entry_tools_r{}", round + 1);
                    tracing::info!(
                        "[AI-ENTRY-TOOL] {} -> {} conf={:.2} rounds={} reason={} ({ms}ms)",
                        symbol, decision.action, decision.confidence, round + 1, decision.reason
                    );
                    return decision;
                }
                Err(e) => {
                    let ms = start.elapsed().as_millis() as u64;
                    tracing::warn!(
                        "[AI-ENTRY-TOOL] {} tool-calling failed round={}: {} — falling back ({ms}ms)",
                        symbol, round + 1, e
                    );
                    // Fallback: single-shot entry confirmation
                    return ai.get_entry_confirmation(candidate_prompt, symbol).await;
                }
            }
        }

        // Exhausted all rounds — force a final answer
        tracing::info!(
            "[AI-ENTRY-TOOL] {} exhausted {} rounds, forcing final answer",
            symbol, max_rounds
        );
        messages.push(serde_json::json!({
            "role": "user",
            "content": format!("You have used all tool calls. Give your final decision now: {}|BUY or HOLD|CONFIDENCE|REASON", symbol)
        }));

        let result = ai
            .infer_with_tools_entry(&messages, &[], 0.5, 128, Some(true))
            .await;

        let ms = start.elapsed().as_millis() as u64;
        match result {
            Ok(crate::ai_bridge::InferToolResult::FinalResponse(text)) => {
                let mut decision = Self::parse_entry_tool_response(&text, symbol);
                decision.latency_ms = ms;
                decision.source = "ai_entry_tools_forced".into();
                decision
            }
            _ => {
                tracing::error!("[AI-ENTRY-TOOL] {} forced answer also failed — defaulting HOLD, flagging error", symbol);
                crate::nemo_optimizer::flag_error("entry_ai", &format!("{symbol} LLM returned no usable response after all rounds"));
                crate::ai_bridge::AiExitDecision {
                    latency_ms: ms,
                    ..crate::ai_bridge::AiExitDecision::default()
                }
            }
        }
    }

    /// Parse entry tool-calling response.
    /// Handles both: "SYMBOL|BUY|0.75|reason" and "BUY|0.75|reason"
    pub(crate) fn parse_entry_tool_response(text: &str, _expected_symbol: &str) -> crate::ai_bridge::AiExitDecision {
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || !line.contains('|') { continue; }
            let parts: Vec<&str> = line.splitn(4, '|').collect();
            if parts.len() < 2 { continue; }

            // Check if first part is action or symbol
            let first = parts[0].trim().to_uppercase();
            let is_action = first == "BUY" || first == "HOLD" || first == "SELL";

            let (action, confidence, reason) = if is_action {
                // BUY|CONF|REASON
                let act = if first == "BUY" { "BUY" } else { "HOLD" }.to_string();
                let conf = parts.get(1)
                    .and_then(|s| s.trim().parse::<f64>().ok())
                    .map(|c| c.clamp(0.0, 1.0))
                    .unwrap_or(0.5);
                let rsn = parts.get(2)
                    .map(|s| s.trim().to_string())
                    .unwrap_or_else(|| "entry_tools".into());
                (act, conf, rsn)
            } else {
                // SYMBOL|ACTION|CONF|REASON
                let act = parts.get(1)
                    .map(|s| s.trim().to_uppercase())
                    .unwrap_or_default();
                let act = if act == "BUY" { "BUY" } else { "HOLD" }.to_string();
                let conf = parts.get(2)
                    .and_then(|s| s.trim().parse::<f64>().ok())
                    .map(|c| c.clamp(0.0, 1.0))
                    .unwrap_or(0.5);
                let rsn = parts.get(3)
                    .map(|s| s.trim().to_string())
                    .unwrap_or_else(|| "entry_tools".into());
                (act, conf, rsn)
            };

            return crate::ai_bridge::AiExitDecision {
                action,
                confidence,
                reason,
                source: "ai_entry_tools".into(),
                latency_ms: 0,
                scan: None,
            };
        }

        // Fallback: try parse_exit_decision which handles various formats
        crate::ai_bridge::parse_exit_decision(text)
    }

    // ── Entry Execution ──────────────────────────────────────────

    pub(crate) async fn execute_entry(
        &mut self,
        symbol: &str,
        price: f64,
        _feats: &serde_json::Value,
        ai: &AiDecision,
        profile: &ProfileParams,
        regime: &str,
        tier: &str,
        signal_names: &[String],
        api: Option<&KrakenApi>,
        quality_key: &str,
        quality_score: f64,
    ) {
        tracing::info!(
            "[EXEC-ATTEMPT] {} price={:.6} conf={:.2} regime={} tier={}",
            symbol, price, ai.confidence, regime, tier
        );
        // Size the position — Pure math: USD = base * min(K, K_cap) * (1 - E)
        let is_meme = tier == "meme";
        let base = if is_meme {
            self.config.env.get_f64("MEME_BASE_USD", self.config.base_usd * 0.5).max(5.0)
        } else {
            self.config.base_usd.max(5.0)
        };
        // Kelly-scaled sizing: usd = base * K
        // Floor = min_order / base so Kelly never produces sub-minimum orders
        let kelly = self.journal.kelly_fraction(symbol);
        let min_order = self.config.env.get_f64("MIN_ORDER_USD", 5.0);
        let kelly_floor = (min_order / base).max(0.25);
        let kelly_scaled = base * kelly.max(kelly_floor);

        // ── Progressive Loss Throttle: scale size by daily P&L ──
        let loss_throttle = if self.config.max_daily_loss_usd > 0.0 {
            let total_loss = -(self.circuit.daily_pnl + self.circuit.unrealized_pnl).max(0.0);
            let f = 1.0 - (total_loss / self.config.max_daily_loss_usd).clamp(0.0, 1.0);
            if f < 1.0 {
                tracing::info!("[THROTTLE] loss=${:.2}/{:.2} → factor={:.2}",
                    total_loss, self.config.max_daily_loss_usd, f);
            }
            f
        } else {
            1.0
        };

        // ── Portfolio weight scaling: apply mean-variance optimal weight if available ──
        let weight_scale = if let Some(ref weights) = self.optimizer.portfolio_weights {
            if let Some(&w) = weights.get(symbol) {
                let n = weights.len().max(1) as f64;
                // Normalize: average weight = 1/n, scale relative to average (clamp 0.5–2.0×)
                (w * n).clamp(0.5, 2.0)
            } else { 1.0 }
        } else { 1.0 };

        let usd_raw = if self.equity > 0.0 && self.equity < 100.0 {
            (kelly_scaled * weight_scale).min(if is_meme { 10.0 } else { 20.0 })
        } else {
            kelly_scaled * weight_scale
        };
        let usd = usd_raw * loss_throttle;


        tracing::info!(
            "[SIZE] {}|K{:.2}|LT{:.2}|${:.0}>${:.2}",
            symbol, kelly, loss_throttle, base, usd,
        );
        let qty = usd / price;

        if qty <= 0.0 || usd < min_order {
            tracing::warn!(
                "[SIZE-REJECT] {} usd=${:.2} < min=${:.0} (K={:.2} base=${:.0}) — order too small",
                symbol, usd, min_order, kelly, base
            );
            return;
        }

        // Detect L2 lane for tighter scalp parameters
        let is_l2 = _feats.get("is_l2").and_then(|v| v.as_bool()).unwrap_or(false)
            || _feats.get("lane_tag").and_then(|v| v.as_str()).unwrap_or("") == "L2";

        // TP/SL: Dynamic ATR-scaled targets
        // Memes get wider range from .env (they swing harder)
        let rt_fee = 2.0 * self.config.fee_per_side_pct / 100.0;
        let atr = _feats.get("atr").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let atr_pct = if price > 0.0 { (atr / price) * 100.0 } else { 0.5 };
        let cfg_tp = self.config.take_profit_pct;
        let cfg_sl = self.config.stop_loss_pct;
        let meme_tp_cfg = self.config.env.get_f64("MEME_TAKE_PROFIT_PCT", 5.0);
        let meme_sl_cfg = self.config.env.get_f64("MEME_STOP_LOSS_PCT", 2.0);
        tracing::info!(
            "[TP-SL-INPUT] {} atr={:.6} atr%={:.2} is_meme={} is_l2={} cfg_tp={:.2} cfg_sl={:.2} meme_tp={:.2} meme_sl={:.2} limit_px={}",
            symbol, atr, atr_pct, is_meme, is_l2, cfg_tp, cfg_sl, meme_tp_cfg, meme_sl_cfg,
            ai.limit_price.map(|v| format!("{v:.6}")).unwrap_or_else(|| "none".to_string())
        );
        // ATR hard stop multipliers (env-configurable per lane)
        let hard_stop_default = self.config.env.get_f64("ATR_HARD_STOP_MULT", 1.75);
        let (tp_pct, sl_pct) = if is_meme {
            // Meme coins: wider TP/SL — they pump and dump harder
            let tp = (atr_pct * 2.5).clamp(2.0, meme_tp_cfg);
            let sl = (atr_pct * 2.0).clamp(2.0, meme_sl_cfg);
            (tp, sl)
        } else if is_l2 {
            // L2 mean-revert: ATR-based, clamped to user-configured TP/SL as floor
            let sl_mult = self.config.env.get_f64("ATR_HARD_STOP_L2", 1.5);
            let tp = (atr_pct * 1.5).clamp(cfg_tp, 4.0);
            let sl = (atr_pct * sl_mult).clamp(cfg_sl, 3.0);
            (tp, sl)
        } else {
            // L1 trend / L3 moderate: 2.0× / 1.5× ATR hard stop
            let lane_tag = _feats.get("lane_tag").and_then(|v| v.as_str()).unwrap_or("L3");
            let sl_mult = if lane_tag == "L1" {
                self.config.env.get_f64("ATR_HARD_STOP_L1", 2.0)
            } else {
                self.config.env.get_f64("ATR_HARD_STOP_L3", hard_stop_default)
            };
            let tp = (atr_pct * 2.0).clamp(2.0, cfg_tp);
            let sl = (atr_pct * sl_mult).clamp(2.5, cfg_sl.max(3.5));
            (tp, sl)
        };
        // ── FEE MARGIN CHECK: SL must cover 2x round-trip fees ──
        let fee_margin_mult = self.config.env.get_f64("FEE_MARGIN_MULTIPLIER", 2.0);
        if sl_pct < rt_fee * 100.0 * fee_margin_mult {
            tracing::info!(
                "[FEE-MARGIN] {} REJECT: sl={:.2}% < {:.0}x fees={:.2}%",
                symbol, sl_pct, fee_margin_mult, rt_fee * 100.0
            );
            return;
        }

        let tp_price_raw = price * (1.0 + tp_pct / 100.0);
        let tp_price = price * (1.0 + tp_pct / 100.0 + rt_fee);
        let sl_price = price * (1.0 - sl_pct / 100.0 - rt_fee);
        let risk = (price - sl_price).max(1e-9);
        let reward = (tp_price - price).max(0.0);
        let rr = reward / risk;
        let net_profit_pct = ((tp_price_raw - price) / price) * 100.0 - (rt_fee * 100.0);
        tracing::info!(
            "[EXEC-MATH] {} entry={:.6} tp={:.6} sl={:.6} tp%={:.2} sl%={:.2} fee%={:.2} rr={:.2} net%={:.2}",
            symbol, price, tp_price, sl_price, tp_pct, sl_pct, rt_fee * 100.0, rr, net_profit_pct
        );
        let pair = config::pair_for(symbol);

        // Limit price offset — regime-aware, per-coin override, meme default, fallback to global
        // Bullish/Euphoric: tighter offset (fill fast, ride momentum)
        // Bearish/Crash: wider offset (demand a better price)
        // Sideways: use .env default
        let coin_key = format!("LIMIT_OFFSET_PCT_{}", symbol);
        let base_offset = if is_meme {
            self.config.env.get_f64("MEME_LIMIT_OFFSET_PCT", 1.5)
        } else {
            self.config.env.get_f64("LIMIT_OFFSET_PCT", 0.05)
        };
        let regime_multiplier = match regime {
            "euphoric" => 0.3,   // tight — fill fast, ride momentum
            "bullish"  => 0.5,   // slightly tighter
            "sideways" => 0.4,   // tight — mean reversion needs fills
            "bearish"  => 1.5,   // wider — demand a better price
            "crash"    => 2.0,   // widest — only enter at deep discount
            _          => 0.5,
        };
        let limit_offset_pct = self.config.env.get_f64(&coin_key, base_offset) * regime_multiplier;
        let limit_offset = 1.0 - (limit_offset_pct / 100.0);
        let limit_px = ai.limit_price.unwrap_or(price * limit_offset);
        let _limit_str = config::format_price(symbol, limit_px);

        // Place LIMIT order (less fees than market)
        if self.config.paper_trading {
            tracing::info!(
                "[PAPER-ENTER] {} qty={:.6} usd=${:.2} limit={:.4}",
                symbol,
                qty,
                usd,
                limit_px
            );
            // Paper mode assumes immediate fill.
            let pos = OpenPosition {
                symbol: symbol.to_string(),
                entry_price: price,
                qty,
                remaining_qty: qty,
                tp_price,
                sl_price,
                highest_price: price,
                entry_time: now_ts(),
                entry_reasons: signal_names.to_vec(),
                entry_profile: profile.name.to_string(),
                entry_context: format!(
                    "profile={} score={:.2} regime={} tier={}",
                    profile.name, ai.confidence, regime, tier
                ),
                entry_score: ai.confidence,
                regime_label: ai.regime_label.clone(),
                quant_bias: ai.quant_bias.clone(),
                npu_action: ai.action.clone(),
                npu_conf: ai.confidence,
                sl_order_txid: None,
                quality_key: quality_key.to_string(),
                quality_score,
                min_hold_sec: ai.min_hold_sec,
                max_hold_sec: ai.max_hold_sec,
                reeval_sec: ai.reeval_sec,
                entry_atr: atr,
                entry_lane: ai.lane.clone(),
                feature_snapshot: _feats.get("feature_snapshot").and_then(|v| {
                    v.as_array().map(|arr| arr.iter().filter_map(|x| x.as_f64()).collect())
                }),
                trend_alignment: crate::engine::entries::read_mtf_alignment(symbol),
                trend_7d_pct: crate::engine::entries::read_mtf_trend_7d(symbol),
                trend_30d_pct: crate::engine::entries::read_mtf_trend_30d(symbol),
            };
            tracing::info!(
                "[TIME-CONTRACT] {} min={}s max={}s reeval={}s lane={}",
                symbol, ai.min_hold_sec, ai.max_hold_sec, ai.reeval_sec, ai.lane,
            );
            self.optimizer
                .open_position(symbol, price, qty, usd, ai.confidence, tier);
            self.available_usd = (self.available_usd - usd).max(0.0);
            self.positions.insert(symbol.to_string(), pos);
            self.last_entry_ts = now_ts();
            save_positions(&self.config.positions_file, &self.positions);
        } else if let Some(api) = api {
            let vol_str = format!("{qty:.8}");
            // Always use limit orders — better fees (maker rate)
            let lp = config::format_price(symbol, limit_px);
            let (order_type, order_price) = ("limit", Some(lp));
            tracing::info!(
                "[ORDER-DEBUG] {} pair={} qty={} limit={:.6} usd=${:.2}",
                symbol, pair, vol_str, limit_px, usd
            );
            match api.add_order(&pair, "buy", order_type, &vol_str, order_price.as_deref()).await {
                Ok(resp) => {
                    let txid = resp
                        .get("result")
                        .and_then(|r| r.get("txid"))
                        .and_then(|t| t.as_array())
                        .and_then(|arr| arr.first())
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());
                    let Some(txid) = txid else {
                        tracing::error!(
                            "[ENTER-FAIL] {} missing txid in AddOrder response",
                            symbol
                        );
                        return;
                    };
                    tracing::info!(
                        "[ENTER-{}] {} qty={:.6} usd=${:.2} price={:.4} tp={:.2}% sl={:.2}% txid={}",
                        if is_l2 { "LIMIT-L2" } else { "MARKET" },
                        symbol,
                        qty,
                        usd,
                        price,
                        tp_pct,
                        sl_pct,
                        txid
                    );
                    self.available_usd = (self.available_usd - usd).max(0.0);
                    self.last_entry_ts = now_ts();
                    let pending = PendingEntryOrder {
                        txid: txid.clone(),
                        symbol: symbol.to_string(),
                        _pair: pair.clone(),
                        requested_qty: qty,
                        reserved_usd: usd,
                        tp_price,
                        sl_price,
                        entry_profile: profile.name.to_string(),
                        entry_context: format!(
                            "profile={} score={:.2} regime={} tier={}",
                            profile.name, ai.confidence, regime, tier
                        ),
                        entry_score: ai.confidence,
                        regime_label: ai.regime_label.clone(),
                        quant_bias: ai.quant_bias.clone(),
                        npu_action: ai.action.clone(),
                        npu_conf: ai.confidence,
                        signal_names: signal_names.to_vec(),
                        created_ts: now_ts(),
                        quality_key: quality_key.to_string(),
                        quality_score,
                        min_hold_sec: ai.min_hold_sec,
                        max_hold_sec: ai.max_hold_sec,
                        reeval_sec: ai.reeval_sec,
                        entry_atr: atr,
                        entry_lane: ai.lane.clone(),
                        feature_snapshot: _feats.get("feature_snapshot").and_then(|v| {
                            v.as_array().map(|arr| arr.iter().filter_map(|x| x.as_f64()).collect())
                        }),
                        trend_alignment: crate::engine::entries::read_mtf_alignment(symbol),
                        trend_7d_pct: crate::engine::entries::read_mtf_trend_7d(symbol),
                        trend_30d_pct: crate::engine::entries::read_mtf_trend_30d(symbol),
                    };
                    self.pending_entries.insert(txid.clone(), pending);
                    log_event(
                        "info",
                        "entry_submitted",
                        json!({
                            "symbol": symbol,
                            "pair": pair,
                            "txid": txid,
                            "qty": qty,
                            "limit_price": limit_px,
                            "reserved_usd": usd
                        }),
                    );
                }
                Err(e) => {
                    tracing::error!("[ENTER-FAIL] {} error: {e}", symbol);
                    crate::nemo_optimizer::flag_error("order_entry", &format!("{symbol} order failed: {e}"));
                    return;
                }
            }
        } else {
            tracing::warn!(
                "[EXEC-BLOCKED] {} no API client (LIVE mode expected but api=None)",
                symbol
            );
            return;
        }

        // Journal thought
        self.journal.log_thought(
            symbol,
            "BUY",
            ai.confidence * 100.0,
            &format!(
                "regime={} quant={} signals={} usd=${:.2}",
                ai.regime_label,
                ai.quant_bias,
                signal_names.join("+"),
                usd
            ),
        );

        tracing::info!(
            "[ENTER] {} score={:.2} price={:.4} usd=${:.2} tp={:.4} sl={:.4} regime={} reasons={}",
            symbol,
            ai.confidence,
            price,
            usd,
            tp_price,
            sl_price,
            regime,
            signal_names.join(",")
        );
    }

    // ── HF Adaptive Confidence Floor ────────────────────────────

    pub(crate) fn update_adaptive_conf_floor(&mut self, pnl_pct: f64) {
        let base = self.config.env.get_f64("HF_CONF_BASE", 0.60);
        let minv = self.config.env.get_f64("HF_CONF_MIN", 0.55);
        let maxv = self.config.env.get_f64("HF_CONF_MAX", 0.75);

        let alpha = self.config.env.get_f64("HF_OUTCOME_EMA_ALPHA", 0.10);
        let loss_ema_pen = self.config.env.get_f64("HF_LOSS_EMA_PENALTY", 0.12);

        let lose_step = self.config.env.get_f64("HF_LOSE_STREAK_STEP", 0.01);
        let lose_cap = self.config.env.get_f64("HF_LOSE_STREAK_MAX_BONUS", 0.06);

        let win_step = self.config.env.get_f64("HF_WIN_STREAK_STEP", 0.005);
        let win_cap = self.config.env.get_f64("HF_WIN_STREAK_MAX_BONUS", 0.03);

        let dd_step = self.config.env.get_f64("HF_DRAWDOWN_STEP", 0.10);
        let dd_cap = self.config.env.get_f64("HF_DRAWDOWN_MAX", 0.08);

        // Outcome: +1 win, -1 loss, 0 breakeven
        let outcome = if pnl_pct > 0.001 { 1.0 } else if pnl_pct < -0.001 { -1.0 } else { 0.0 };

        if outcome > 0.0 {
            self.hf_streak_wins += 1;
            self.hf_streak_losses = 0;
        } else if outcome < 0.0 {
            self.hf_streak_losses += 1;
            self.hf_streak_wins = 0;
            // 3-loss cooldown: block entries for LOSS_COOLDOWN_SEC after N consecutive losses
            let cooldown_enabled = self.config.env.get_bool("LOSS_COOLDOWN_ENABLED", false);
            let cooldown_consecutive = self.config.env.get_f64("LOSS_COOLDOWN_CONSECUTIVE", 3.0) as u32;
            let cooldown_sec = self.config.env.get_f64("LOSS_COOLDOWN_SEC", 3600.0);
            if cooldown_enabled && self.hf_streak_losses >= cooldown_consecutive {
                let now = now_ts();
                self.loss_cooldown_until = now + cooldown_sec;
                tracing::warn!(
                    "[LOSS-COOLDOWN] {} consecutive losses → blocking entries for {:.0}s (until ts={:.0})",
                    self.hf_streak_losses, cooldown_sec, self.loss_cooldown_until
                );
            }
        }

        // EMA of outcomes (range ~[-1, +1])
        self.hf_outcome_ema = (1.0 - alpha) * self.hf_outcome_ema + alpha * outcome;

        // Drawdown from equity peak
        let equity = self.portfolio_value.max(0.0);
        if self.hf_equity_peak <= 0.0 || equity > self.hf_equity_peak {
            self.hf_equity_peak = equity;
        }
        let dd = if self.hf_equity_peak > 0.0 {
            ((self.hf_equity_peak - equity) / self.hf_equity_peak).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // Compute penalties/bonuses
        let loss_penalty = (-self.hf_outcome_ema).max(0.0) * loss_ema_pen;
        let lose_streak_penalty = (self.hf_streak_losses as f64 * lose_step).min(lose_cap);
        let win_streak_bonus = (self.hf_streak_wins as f64 * win_step).min(win_cap);
        let drawdown_penalty = (dd * dd_step).min(dd_cap);

        let new_floor = (base + loss_penalty + lose_streak_penalty + drawdown_penalty - win_streak_bonus)
            .clamp(minv, maxv);

        // Log only on material change
        if (new_floor - self.entry_conf_floor_dyn).abs() >= 0.01 {
            tracing::info!(
                "[HF-CONF] floor {:.2}→{:.2} | ema={:+.2} W{} L{} dd={:.1}%",
                self.entry_conf_floor_dyn, new_floor, self.hf_outcome_ema,
                self.hf_streak_wins, self.hf_streak_losses, dd * 100.0
            );
        }

        self.entry_conf_floor_dyn = new_floor;
    }

    pub(crate) fn record_entry_reject(&mut self, now: f64, sym: &str, reason: &str, data_quality: bool) {
        let code = reason.split_whitespace().next().unwrap_or("unknown");
        self.meters.bump_reject(now, code);
        if data_quality {
            self.meters.data_quality_gate_trips += 1;
        }
        tracing::info!("[ENTRY-REJECT] {sym}: {reason}");
        // L1: Emit risk.reject for post-AI gate rejections
        crate::event_bus::publish(&self.bus, crate::event_bus::Event::RiskReject {
            ts: now,
            symbol: sym.to_string(),
            reason: reason.to_string(),
            spread_pct: None,
            slip_pct: None,
        });
    }
}


