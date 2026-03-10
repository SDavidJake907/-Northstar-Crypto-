//! Portfolio allocation — AI-driven portfolio-level rebalancing and context builders.

use super::*;
use serde_json::json;

use crate::config;
use crate::journal::TradeRecord;
use crate::kraken_api::KrakenApi;
use crate::ai_bridge::AiBridge;

impl super::TradingLoop {
    /// Build portfolio context string for AI prompt.
    pub(crate) fn build_portfolio_context_with_prices(
        &self,
        features_map: &HashMap<String, serde_json::Value>,
    ) -> String {
        let _fee_pct = self.config.env.get_f64("FEE_PCT", 0.26);

        // Calculate total portfolio value
        let mut total_value = self.available_usd;
        let mut pos_lines: Vec<String> = Vec::new();

        for p in self.positions.values() {
            let cur_price = features_map
                .get(&p.symbol)
                .and_then(|v| v.get("price"))
                .and_then(|v| v.as_f64())
                .unwrap_or(p.entry_price);
            let val = p.remaining_qty * cur_price;
            total_value += val;
            let pnl_pct = p.pnl_pct_raw(cur_price) * 100.0;
            pos_lines.push(format!(
                "{} qty={:.4} entry={:.4} now={:.4} val=${:.2} pnl={:+.2}%",
                p.symbol, p.remaining_qty, p.entry_price, cur_price, val, pnl_pct
            ));
        }

        let mut lines = Vec::new();
        lines.push(format!(
            "PORTFOLIO: total=${:.2} free=${:.2} positions={}",
            total_value,
            self.available_usd,
            self.positions.len()
        ));
        for pl in &pos_lines {
            lines.push(pl.clone());
        }
        lines.join("\n")
    }

    pub(crate) fn build_portfolio_snapshot(
        &self,
        features_map: &HashMap<String, serde_json::Value>,
    ) -> serde_json::Value {
        let mut total_value = self.available_usd;
        let mut positions: Vec<serde_json::Value> = Vec::new();

        for p in self.positions.values() {
            let cur_price = features_map
                .get(&p.symbol)
                .and_then(|v| v.get("price"))
                .and_then(|v| v.as_f64())
                .unwrap_or(p.entry_price);
            let value = p.remaining_qty * cur_price;
            total_value += value;
            let pnl_pct = p.pnl_pct_raw(cur_price) * 100.0;
            positions.push(json!({
                "symbol": p.symbol,
                "qty": p.remaining_qty,
                "entry": p.entry_price,
                "now": cur_price,
                "value": value,
                "pnl_pct": pnl_pct,
            }));
        }

        json!({
            "total_value": total_value,
            "free_usd": self.available_usd,
            "positions": positions,
        })
    }

    // ── Portfolio Allocator V2 ──────────────────────────────────────
    // AI-driven portfolio-level allocation + deterministic rebalance.
    // Flow: build_allocator_snapshot() → AI_1 (timer-gated) → enforce_constraints()
    //       → compute deltas → sell-first, buy-second → Kraken REST

    /// Build the PortfolioSnapshot that AI_1 sees for allocation decisions.
    pub(crate) fn build_allocator_snapshot(
        &self,
        features_map: &HashMap<String, serde_json::Value>,
    ) -> crate::portfolio_allocator::PortfolioSnapshot {
        use crate::portfolio_allocator::*;

        let regime_label = self.market_brain.regime().to_string();
        let alloc_regime = match regime_label.as_str() {
            "CRASH" | "BEARISH" => "CONTRACTION",
            "SIDEWAYS" => "ACCUMULATION",
            "BULLISH" => "EXPANSION",
            "EUPHORIC" => "MANIA",
            _ => "ACCUMULATION",
        };

        // Extract market brain data via state() accessor
        let mb_state = self.market_brain.state();
        let green_count = match mb_state {
            Some(s) => s.breadth.green_count,
            None => 0,
        };
        let confidence = crate::market_brain::regime_confidence_base(self.market_brain.regime());

        let regime = RegimeInfo {
            label: alloc_regime.to_string(),
            confidence,
            signals: RegimeSignals {
                fear_greed: 0,
                btc_dominance_pct: 0.0,
                total_mcap_change_24h_pct: 0.0,
                funding_avg: 0.0,
                green_count,
                coin_count: features_map.len(),
            },
        };

        // Use equity from Kraken (accurate) or fallback to computed portfolio_value
        let total_value = if self.equity > 1.0 { self.equity } else { self.portfolio_value.max(1.0) };
        let mut candidates = Vec::new();
        for (sym, feats) in features_map {
            let price = feats.get("price").and_then(|v| v.as_f64()).unwrap_or(0.0);
            if price <= 0.0 { continue; }

            let scores = compute_scores(feats);
            let sector = coin_sector(sym).to_string();

            // Current weight in portfolio
            let held_qty = self.holdings.get(sym).copied().unwrap_or(0.0);
            let current_pct = if total_value > 0.0 {
                held_qty * price / total_value * 100.0
            } else {
                0.0
            };

            // PnL from tracked position (if any)
            let pnl_pct = self.positions.get(sym)
                .map(|p| p.pnl_pct_raw(price) * 100.0)
                .unwrap_or(0.0);

            candidates.push(CandidateInfo {
                symbol: sym.clone(),
                sector,
                price,
                scores,
                current_pct,
                pnl_pct,
                rsi: feats.get("rsi").and_then(|v| v.as_f64()).unwrap_or(50.0),
                hurst: feats.get("hurst_exp").and_then(|v| v.as_f64()).unwrap_or(0.5),
                entropy: feats.get("shannon_entropy").and_then(|v| v.as_f64()).unwrap_or(0.5),
                kelly: self.journal.kelly_fraction(sym),
                book_imbalance: feats.get("book_imbalance").and_then(|v| v.as_f64()).unwrap_or(0.0),
                momentum: feats.get("momentum_score").and_then(|v| v.as_f64()).unwrap_or(0.0),
            });
        }

        let constraints = build_constraints();

        PortfolioSnapshot {
            asof_utc: {
                use std::time::{SystemTime, UNIX_EPOCH};
                let secs = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0);
                format!("{secs}")
            },
            regime,
            constraints,
            candidates,
        }
    }

    /// Portfolio Allocator V2 — run allocation + rebalance.
    /// Replaces per-coin entry logic when MODE=portfolio_allocator.
    /// Exits (TP/SL/trailing/Nemo) still run normally above this call.
    ///
    /// Safety layers (in order):
    ///   1. Confidence filter — skip cycle if AI confidence < threshold
    ///   2. Portfolio drift guard — skip if total shift > max_shift%
    ///   3. Rate limiter — clamp each delta to max_step% per cycle
    ///   4. Stablecoin safety floor — ensure min cash reserve before buys
    ///   5. Turnover cap — limit total portfolio churn
    ///   6. Flight recorder — log full state every cycle
    pub(crate) async fn run_portfolio_allocator(
        &mut self,
        features_map: &HashMap<String, serde_json::Value>,
        api: Option<&KrakenApi>,
        _ai: &AiBridge,
    ) {
        let now = now_ts();
        let t0 = std::time::Instant::now();

        // Build snapshot for allocator
        let snapshot = self.build_allocator_snapshot(features_map);

        // Warmup guard: need enough candidates with data
        if snapshot.candidates.len() < 5 {
            tracing::debug!(
                "[ALLOCATOR] Warmup: only {} candidates, need 5+ — skipping",
                snapshot.candidates.len()
            );
            return;
        }

        // Run allocator (timer-gated internally — every ALLOC_INTERVAL_SEC)
        let alloc = match self.portfolio_allocator.maybe_run(now, &snapshot).await {
            Some(a) => a.clone(),
            None => return,
        };
        let qwen_ms = t0.elapsed().as_millis();

        // ── SAFETY 1: Confidence Filter ──
        // Skip cycle if AI isn't confident enough about this allocation.
        let min_confidence = self.config.env.get_f64("ALLOCATOR_MIN_CONFIDENCE", 0.60);
        if alloc.confidence < min_confidence {
            tracing::warn!(
                "[ALLOCATOR] Confidence too low ({:.2} < {:.2}) — skipping cycle",
                alloc.confidence, min_confidence
            );
            return;
        }

        // Compute current weights from Kraken holdings
        let total_value = if self.equity > 1.0 { self.equity } else { self.portfolio_value.max(1.0) };
        let cash_pct = self.available_usd / total_value * 100.0;
        let mut current_pct: HashMap<String, f64> = HashMap::new();
        for (sym, qty) in &self.holdings {
            if sym == "USD" { continue; }
            let price = features_map.get(sym)
                .and_then(|f| f.get("price").and_then(|v| v.as_f64()))
                .unwrap_or(0.0);
            if price > 0.0 {
                current_pct.insert(sym.clone(), qty * price / total_value * 100.0);
            }
        }

        // Rebalance guardrails from .env (all hot-reloadable)
        let band = self.config.env.get_f64("REBALANCE_BAND_PCT", 1.5);
        let step_cap = self.config.env.get_f64("ALLOCATOR_MAX_STEP_PCT", 3.0);
        let min_order_usd = self.config.env.get_f64("MIN_ORDER_USD", 50.0);
        let max_turnover = self.config.env.get_f64("ALLOCATOR_MAX_TURNOVER_PCT", 15.0);
        let max_shift = self.config.env.get_f64("ALLOCATOR_MAX_SHIFT_PCT", 25.0);
        let min_stable_pct = self.config.env.get_f64("ALLOCATOR_MIN_STABLE_PCT", 5.0);

        // ── SAFETY 2: Portfolio Drift Guard ──
        // Compute total raw shift before capping — if too large, skip entirely.
        // First cycle is exempt: no previous allocation to compare against, and the
        // rate limiter + turnover cap already protect against wild swings.
        let raw_total_shift: f64 = alloc.allocations.iter().map(|entry| {
            let cur = current_pct.get(&entry.symbol).copied().unwrap_or(0.0);
            (entry.pct - cur).abs()
        }).sum();
        let is_first_cycle = self.portfolio_allocator.alloc_count() <= 1;
        if raw_total_shift > max_shift && !is_first_cycle {
            tracing::error!(
                "[ALLOCATOR] Portfolio shift too large ({:.1}% > {:.1}%) — SKIPPING cycle (possible bad AI output)",
                raw_total_shift, max_shift
            );
            return;
        }
        if is_first_cycle && raw_total_shift > max_shift {
            tracing::info!(
                "[ALLOCATOR] First cycle — allowing large shift ({:.1}%), rate limiter will smooth it",
                raw_total_shift
            );
        }

        struct RebalAction {
            symbol: String,
            delta_pct: f64,
            delta_usd: f64,
        }

        let mut sells: Vec<RebalAction> = Vec::new();
        let mut buys: Vec<RebalAction> = Vec::new();
        let mut total_turnover: f64 = 0.0;

        // Build rebalance actions from allocator output
        for entry in &alloc.allocations {
            let cur = current_pct.get(&entry.symbol).copied().unwrap_or(0.0);
            let delta = entry.pct - cur;

            // Band filter — ignore small deltas
            if delta.abs() < band { continue; }

            // ── SAFETY 3: Rate Limiter — smooth transitions, max step_cap% per cycle ──
            let capped_delta = delta.clamp(-step_cap, step_cap);
            let delta_usd = capped_delta / 100.0 * total_value;

            // Minimum order size
            if delta_usd.abs() < min_order_usd { continue; }

            // ── SAFETY 5: Turnover cap — limit total portfolio churn per cycle ──
            if total_turnover + capped_delta.abs() > max_turnover { continue; }
            total_turnover += capped_delta.abs();

            let action = RebalAction {
                symbol: entry.symbol.clone(),
                delta_pct: capped_delta,
                delta_usd,
            };

            if delta_usd < 0.0 {
                sells.push(action);
            } else {
                buys.push(action);
            }
        }

        // Check for held coins the allocator didn't include — reduce toward 0
        for (sym, cur_pct) in &current_pct {
            if *cur_pct > band && !alloc.allocations.iter().any(|a| a.symbol == *sym) {
                let delta = (-*cur_pct).max(-step_cap);
                let delta_usd = delta / 100.0 * total_value;
                if delta_usd.abs() >= min_order_usd && total_turnover + delta.abs() <= max_turnover {
                    total_turnover += delta.abs();
                    sells.push(RebalAction {
                        symbol: sym.clone(),
                        delta_pct: delta,
                        delta_usd,
                    });
                }
            }
        }

        // ── SAFETY 6: Minimum Hold Time Guard ──
        // Don't sell positions that haven't had enough time to develop.
        // Positions sold too early = guaranteed fee loss.
        let min_hold_min = self.config.env.get_f64("ALLOCATOR_MIN_HOLD_MIN", 60.0);
        sells.retain(|action| {
            if let Some(pos) = self.positions.get(&action.symbol) {
                let hold_min = (now - pos.entry_time) / 60.0;
                if hold_min < min_hold_min {
                    tracing::info!(
                        "[ALLOCATOR] HOLD-GUARD: {} held {:.0}min < {:.0}min minimum — skipping sell",
                        action.symbol, hold_min, min_hold_min
                    );
                    return false;
                }
            }
            true
        });

        // ── SAFETY 7: Fee-Breakeven Guard ──
        // Don't sell positions in the fee zone (PnL between -0.52% and +0.52%).
        // Selling here = guaranteed net loss after round-trip fees.
        // Exception: stop-loss positions (PnL < -2%) should still be sold.
        let fee_rt = 0.0052; // 0.52% round-trip
        sells.retain(|action| {
            if let Some(pos) = self.positions.get(&action.symbol) {
                let price = features_map.get(&action.symbol)
                    .and_then(|f| f.get("price").and_then(|v| v.as_f64()))
                    .unwrap_or(0.0);
                if price > 0.0 {
                    let pnl_pct = (price - pos.entry_price) / pos.entry_price;
                    // In the fee zone: hasn't moved enough to cover fees
                    if pnl_pct.abs() < fee_rt && pnl_pct > -0.02 {
                        tracing::info!(
                            "[ALLOCATOR] FEE-GUARD: {} pnl={:+.2}% inside fee zone (±0.52%) — skipping sell",
                            action.symbol, pnl_pct * 100.0
                        );
                        return false;
                    }
                }
            }
            true
        });

        // ── SAFETY 8: Flight Recorder — log full state every cycle ──
        let target_str: String = alloc.allocations.iter()
            .map(|a| format!("{}:{:.0}%", a.symbol, a.pct))
            .collect::<Vec<_>>().join(" ");
        let current_str: String = current_pct.iter()
            .filter(|(_, v)| **v > 0.5)
            .map(|(k, v)| format!("{k}:{v:.0}%"))
            .collect::<Vec<_>>().join(" ");
        tracing::info!(
            "[ALLOCATOR-STATE] portfolio=${:.2} cash={:.1}% regime={} conf={:.2} risk={:.0} \
             qwen={qwen_ms}ms shift={:.1}% turnover={:.1}% | target=[{target_str}] current=[{current_str}]",
            total_value, cash_pct, snapshot.regime.label, alloc.confidence,
            alloc.risk_level, raw_total_shift, total_turnover
        );

        if sells.is_empty() && buys.is_empty() {
            if self.tick_count % 30 == 0 {
                tracing::info!("[ALLOCATOR] No rebalance needed (all within {band:.1}% band)");
            }
            return;
        }

        tracing::info!(
            "[ALLOCATOR] Rebalancing: {} sells, {} buys, turnover={:.1}%",
            sells.len(), buys.len(), total_turnover
        );

        // ── Execute sells first (free up cash for buys) ──
        for s in &sells {
            let price = features_map.get(&s.symbol)
                .and_then(|f| f.get("price").and_then(|v| v.as_f64()))
                .unwrap_or(0.0);
            if price <= 0.0 { continue; }
            let held = self.holdings.get(&s.symbol).copied().unwrap_or(0.0);
            let qty_to_sell = (s.delta_usd.abs() / price).min(held);
            if qty_to_sell <= 0.0 { continue; }

            tracing::info!(
                "[ALLOCATOR-SELL] {} delta={:+.1}% ${:.2} qty={:.6}",
                s.symbol, s.delta_pct, s.delta_usd.abs(), qty_to_sell
            );

            let mut sell_confirmed = false;
            if self.config.paper_trading {
                tracing::info!("[PAPER-ALLOC-SELL] {} qty={:.6}", s.symbol, qty_to_sell);
                self.available_usd += qty_to_sell * price;
                sell_confirmed = true;
            } else if let Some(api) = api {
                let pair = config::pair_for(&s.symbol);
                let vol_str = format!("{qty_to_sell:.8}");
                match api.add_order(&pair, "sell", "market", &vol_str, None).await {
                    Ok(_) => {
                        tracing::info!("[ALLOCATOR-SOLD] {} qty={:.8} @ {:.4}", s.symbol, qty_to_sell, price);
                        self.available_usd += qty_to_sell * price;
                        sell_confirmed = true;
                    }
                    Err(e) => tracing::error!("[ALLOCATOR-SELL-FAIL] {} position kept: {e}", s.symbol),
                }
            }

            // Close tracked position ONLY if sell went through
            let remaining = held - qty_to_sell;
            if sell_confirmed && remaining < 0.01 {
                if let Some(pos) = self.positions.remove(&s.symbol) {
                    let fee = self.config.fee_per_side_pct;
                    let pnl_pct = pos.pnl_pct(price, fee);
                    let pnl_usd = pos.pnl_usd(price, fee);
                    self.circuit.add_pnl(pnl_usd);
                    self.journal.record_trade(TradeRecord {
                        symbol: s.symbol.clone(),
                        entry_time: pos.entry_time,
                        exit_time: now_ts(),
                        entry_price: pos.entry_price,
                        exit_price: price,
                        quantity: pos.remaining_qty,
                        pnl_percent: pnl_pct,
                        pnl_usd,
                        result: TradeRecord::classify_result(pnl_pct).to_string(),
                        hold_minutes: pos.hold_seconds() / 60.0,
                        entry_reasons: pos.entry_reasons.clone(),
                        exit_reason: "allocator_rebalance".to_string(),
                        entry_context: pos.entry_context.clone(),
                        points: if pnl_pct > 0.001 { 2 } else { -1 },
                        points_reason: format!("allocator_rebalance pnl={:+.2}%", pnl_pct * 100.0),
                        feature_snapshot: pos.feature_snapshot.clone(),
                        trend_alignment: pos.trend_alignment,
                        trend_7d_pct: pos.trend_7d_pct,
                        trend_30d_pct: pos.trend_30d_pct,
                    });
                    // Record GBDT training sample
                    if let Some(ref snapshot) = pos.feature_snapshot {
                        let mut gbdt = self.gbdt.write().await;
                        gbdt.record_sample(snapshot.clone(), pnl_pct);
                    }
                    tracing::info!(
                        "[ALLOCATOR-EXIT] {} pnl={:+.2}% ${:+.2}",
                        s.symbol, pnl_pct * 100.0, pnl_usd
                    );
                }
            }
        }

        // ── Cancel stale open orders before placing new ones ──
        if let Some(api) = api {
            match api.cancel_all_orders().await {
                Ok(resp) => {
                    let count = resp.get("result")
                        .and_then(|r| r.get("count"))
                        .and_then(|c| c.as_i64())
                        .unwrap_or(0);
                    if count > 0 {
                        tracing::info!("[ALLOCATOR] Canceled {count} stale open orders before rebalance");
                    }
                }
                Err(e) => tracing::warn!("[ALLOCATOR] CancelAll failed: {e}"),
            }
        }

        // ── Then execute buys ──
        for b in &buys {
            let price = features_map.get(&b.symbol)
                .and_then(|f| f.get("price").and_then(|v| v.as_f64()))
                .unwrap_or(0.0);
            if price <= 0.0 { continue; }

            // ── SAFETY 4: Stablecoin Safety Floor ──
            // Ensure we keep min_stable_pct% as cash reserve after this buy
            let cash_after_pct = (self.available_usd - b.delta_usd) / total_value * 100.0;
            let buy_usd = if cash_after_pct < min_stable_pct {
                // Reduce buy to maintain floor
                let max_spend = self.available_usd - (min_stable_pct / 100.0 * total_value);
                if max_spend < min_order_usd {
                    tracing::info!(
                        "[ALLOCATOR] Skipping {} buy — would breach {:.0}% cash floor",
                        b.symbol, min_stable_pct
                    );
                    continue;
                }
                max_spend
            } else {
                b.delta_usd.min(self.available_usd * 0.95)
            };
            if buy_usd < min_order_usd { continue; }
            let qty = buy_usd / price;

            tracing::info!(
                "[ALLOCATOR-BUY] {} delta={:+.1}% ${:.2} qty={:.6}",
                b.symbol, b.delta_pct, buy_usd, qty
            );

            let pair = config::pair_for(&b.symbol);

            if self.config.paper_trading {
                tracing::info!("[PAPER-ALLOC-BUY] {} qty={:.6} ${:.2}", b.symbol, qty, buy_usd);
                let atr = features_map.get(&b.symbol)
                    .and_then(|f| f.get("atr").and_then(|v| v.as_f64()))
                    .unwrap_or(0.0);
                let atr_pct = if price > 0.0 { atr / price * 100.0 } else { 1.0 };
                let tp_pct = (atr_pct * 2.0).clamp(2.0, self.config.take_profit_pct);
                let sl_pct = (atr_pct * 1.5).clamp(2.5, self.config.stop_loss_pct.max(3.5));
                let rt_fee = 2.0 * self.config.fee_per_side_pct / 100.0;
                let pos = OpenPosition {
                    symbol: b.symbol.clone(),
                    entry_price: price,
                    qty,
                    remaining_qty: qty,
                    tp_price: price * (1.0 + tp_pct / 100.0 + rt_fee),
                    sl_price: price * (1.0 - sl_pct / 100.0 - rt_fee),
                    highest_price: price,
                    entry_time: now_ts(),
                    entry_reasons: vec!["allocator_rebalance".into()],
                    entry_profile: "allocator".into(),
                    entry_context: format!("allocator delta={:+.1}%", b.delta_pct),
                    entry_score: alloc.confidence,
                    regime_label: self.market_brain.regime().to_string(),
                    quant_bias: "neutral".into(),
                    npu_action: "BUY".into(),
                    npu_conf: alloc.confidence,
                    sl_order_txid: None,
                    quality_key: String::new(),
                    quality_score: 0.0,
                    min_hold_sec: 900,   // allocator rebalances: moderate defaults
                    max_hold_sec: 21600,
                    reeval_sec: 600,
                    entry_atr: atr,
                    entry_lane: "L3".to_string(),
                    feature_snapshot: None,
                    trend_alignment: 0,
                    trend_7d_pct: 0.0,
                    trend_30d_pct: 0.0,
                };
                self.positions.insert(b.symbol.clone(), pos);
                self.available_usd = (self.available_usd - buy_usd).max(0.0);
            } else if let Some(api) = api {
                let offset_pct = self.config.env.get_f64("LIMIT_OFFSET_PCT", 0.05);
                let limit_px = price * (1.0 - offset_pct / 100.0);
                let lp = config::format_price(&b.symbol, limit_px);
                let vol_str = format!("{qty:.8}");
                match api.add_order(&pair, "buy", "limit", &vol_str, Some(&lp)).await {
                    Ok(resp) => {
                        let txid = resp.get("result")
                            .and_then(|r| r.get("txid"))
                            .and_then(|t| t.as_array())
                            .and_then(|arr| arr.first())
                            .and_then(|v| v.as_str())
                            .unwrap_or("unknown");
                        tracing::info!(
                            "[ALLOCATOR-ORDER] {} BUY qty={:.6} limit={} txid={}",
                            b.symbol, qty, lp, txid
                        );
                        self.available_usd = (self.available_usd - buy_usd).max(0.0);
                    }
                    Err(e) => tracing::error!("[ALLOCATOR-BUY-FAIL] {} error: {e}", b.symbol),
                }
            }
        }

        save_positions(&self.config.positions_file, &self.positions);
    }
}
