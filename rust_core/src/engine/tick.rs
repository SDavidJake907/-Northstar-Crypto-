//! Main tick method — one complete trading loop cycle.

use super::*;
use super::regime_sm;
use super::lane;
use super::fee_filter;

impl super::TradingLoop {
    /// Run one complete tick of the trading loop.
    ///
    /// `features_map`: {symbol → feature JSON} from the Rust feature engine
    /// `api`: Kraken API client (None for paper trading)
    /// `ai`: AI bridge to Python workers
    pub async fn tick(
        &mut self,
        features_map: &HashMap<String, serde_json::Value>,
        api: Option<&KrakenApi>,
        ai: &AiBridge,
    ) {
        self.tick_count += 1;
        let now = now_ts();

        // Drain Phi-3 background scan results into cache (non-blocking)
        if let Some(ref mut rx) = self.npu_scan_rx {
            while let Ok((sym, verdict, lane, action, reason)) = rx.try_recv() {
                let advisory = format!("PHI3:{} lane={} action={} reason={}", verdict, lane, action, reason);
                // Log AI_3 decision for 49B optimizer
                crate::engine::decision_log::log_ai3(&crate::engine::decision_log::Ai3Decision {
                    ts:         now,
                    symbol:     sym.clone(),
                    verdict:    verdict.clone(),
                    lane:       lane.clone(),
                    action:     action.clone(),
                    reason:     reason.clone(),
                    latency_ms: 0, // latency logged in debug, not passed through channel
                });
                self.npu_scan_cache.insert(sym, advisory);
            }
        }
        let log_flow = self.config.env.get_bool("LOG_TICK_FLOW", false);
        let log_every = self.config.env.get_u64("LOG_TICK_FLOW_EVERY_N", 10).max(1);
        let tick = self.tick_count;
        let memory_enabled = self.memory_enabled;
        if !self.gpu_math_logged {
            let gm = self.gpu_math.read().await;
            if gm.is_healthy() {
                tracing::info!("[GPU-MATH] active — CUDA batch math enabled");
            } else {
                tracing::info!("[GPU-MATH] inactive — CPU fallback in use");
            }
            self.gpu_math_logged = true;
        }
        let flow_log = |stage: &str| {
            if log_flow && (tick % log_every == 0) {
                tracing::info!("[FLOW] tick={} stage={}", tick, stage);
            }
        };
        let disable_breakers = FORCE_DISABLE_BREAKERS || self.config.env.get_bool("DISABLE_ALL_BREAKERS", true);
        self.meters.roll_window(now);
        flow_log("start");
        if let Some(api_client) = api {
            let (throttle_hits, backoff_hits) = api_client.throttle_stats();
            self.rest_throttle_hits = throttle_hits;
            self.rest_backoff_hits = backoff_hits;
        }

        // ── 1. Housekeeping ──────────────────────────────────────
        flow_log("housekeeping");
        if !disable_breakers {
            self.circuit.maybe_reset_daily();
        }

        // Hot-reload config every 10s
        if self.last_config_reload.elapsed().as_secs() >= 10 {
            self.config.hot_reload();
            self.last_config_reload = Instant::now();
        }

        // Auto-live: toggle paper/live based on BTC 24h green/red
        self.check_btc_auto_live(features_map).await;

        // Update market brain (BTC gravity, breadth, sectors, regime)
        self.market_brain.update(features_map, &self.strategy);

        // Recompute memory packet (adaptive learning layer — 60s cache)
        if memory_enabled {
            self.memory_packet.maybe_recompute(
                now,
                &self.nemo_memory_brain,
                &self.journal,
                &self.circuit,
                self.market_brain.regime().as_str(),
                self.config.fee_per_side_pct,
            );
        }

        // Update news sentiment (Fear/Greed, CoinGecko global, trending — free APIs, timer-gated)
        let now_secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();
        self.news_sentiment.maybe_update(now_secs).await;

        // Update market watcher (MTF, sector rotation, volume, correlation, whales, divergence)
        self.market_watcher.update(features_map, &self.strategy);
        for alert in self.market_watcher.alerts() {
            crate::event_bus::publish(&self.bus, crate::event_bus::Event::WatcherAlert {
                ts: alert.ts,
                signal: format!("{}", alert.signal),
                severity: format!("{}", alert.severity),
                text: alert.text.clone(),
            });
        }

        // Reload CVaR weights every 5 ticks (~10s)
        if self.tick_count % 5 == 0 {
            self.reload_cvar_weights();
        }
        self.manual_kill_active = if disable_breakers {
            false
        } else {
            self.config.env.get_bool("MANUAL_KILL_SWITCH", false)
        };
        if self.manual_kill_active {
            if self.tick_count.is_multiple_of(2) {
                tracing::error!(
                    "[MANUAL-KILL] enabled: forcing HOLD mode and closing open positions"
                );
            }
            let syms: Vec<String> = self.positions.keys().cloned().collect();
            for sym in &syms {
                let price = features_map
                    .get(sym)
                    .and_then(|f| f.get("price"))
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0);
                if price > 0.0 {
                    self.execute_exit(sym, price, "manual_kill_switch", api, None)
                        .await;
                }
            }
            return;
        }

        // ── File-based Kill Switch: touch data/KILL_SWITCH to block entries ──
        let file_kill_active = std::path::Path::new("data/KILL_SWITCH").exists();
        if file_kill_active {
            if self.tick_count % 6 == 0 {
                tracing::warn!("[FILE-KILL] data/KILL_SWITCH detected — blocking new entries (exits still run)");
            }
        }

        self.prev_strategy_weights = self.strategy_weights.clone();
        self.strategy_weights = compute_softmax_weights(&self.strategy.strategy, features_map);

        flow_log("hardware_guard");
        if disable_breakers {
            self.hardware_block_entries = false;
        } else {
            self.update_hardware_guard(now);
            if self.hardware_block_entries {
                self.meters.hardware_guard_trips += 1;
                self.meters.data_quality_gate_trips += 1;
            }
        }

        // Heartbeat
        flow_log("heartbeat");
        if (now - self.last_heartbeat) >= 5.0 {
            let payload = self.build_heartbeat_payload(now);
            write_heartbeat(&self.config.heartbeat_file, &payload);
            self.last_heartbeat = now;
        }

        flow_log("watchdog");
        let watchdog_interval =
            self.config.env.get_f64("WATCHDOG_CHECK_INTERVAL_SEC", WATCHDOG_CHECK_INTERVAL_SEC).max(2.0);
        if (now - self.last_watchdog_check) >= watchdog_interval {
            let report = run_watchdog(now, &self.config.heartbeat_file);
            self.watchdog_ok = report.ok;
            self.watchdog_fail_count = report.fail_count;
            self.watchdog_last_fail_paths = report.fail_paths;
            if disable_breakers {
                self.watchdog_block_entries = false;
            } else {
                self.watchdog_block_entries =
                    !self.watchdog_ok && self.config.env.get_bool("WATCHDOG_BLOCK_ENTRIES", true);
                if self.watchdog_block_entries {
                    self.meters.watchdog_gate_trips += 1;
                    self.meters.data_quality_gate_trips += 1;
                }
                if !self.watchdog_ok && self.config.env.get_bool("WATCHDOG_KILL_ON_FAIL", false) {
                    tracing::error!("[WATCHDOG-KILL] watchdog failure: closing open positions");
                    crate::nemo_optimizer::flag_error("watchdog", "Watchdog kill triggered: data quality failure, all positions closed");
                    let syms: Vec<String> = self.positions.keys().cloned().collect();
                    for sym in &syms {
                        let price = features_map
                            .get(sym)
                            .and_then(|f| f.get("price"))
                            .and_then(|v| v.as_f64())
                            .unwrap_or(0.0);
                        if price > 0.0 {
                            self.execute_exit(sym, price, "watchdog_kill_switch", api, None)
                                .await;
                        }
                    }
                    self.last_watchdog_check = now;
                    return;
                }
            }
            self.last_watchdog_check = now;
        }


        // ── Balance check (every 30s) ────────────────────────────
        flow_log("balance_refresh");
        if (now - self.last_balance_check) >= BALANCE_CHECK_INTERVAL {
            if let Some(api) = api {
                // 1) Get per-asset holdings + USD cash from Balance endpoint
                match api.balance().await {
                    Ok(resp) => {
                        if let Some(result) = resp.get("result").and_then(|r| r.as_object()) {
                            // Parse USD cash (Kraken uses "ZUSD" or "USD")
                            self.available_usd = result
                                .get("ZUSD")
                                .or_else(|| result.get("USD"))
                                .and_then(|v| v.as_str())
                                .and_then(|s| s.parse::<f64>().ok())
                                .unwrap_or(0.0);

                            self.holdings.clear();
                            for (k, v) in result {
                                if let Some(s) = v.as_str() {
                                    if let Ok(amt) = s.parse::<f64>() {
                                        if amt > 0.0 {
                                            self.holdings.insert(kraken_to_symbol(k), amt);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => {
                        if self.tick_count.is_multiple_of(10) {
                            tracing::error!("[BALANCE] Holdings query failed: {e}");
                            crate::nemo_optimizer::flag_error("balance", &format!("Holdings query failed: {e}"));
                        }
                    }
                }

                // 2) Get trade balance for equity tracking (portfolio value)
                self.equity = 0.0;
                match api.trade_balance().await {
                    Ok(resp) => {
                        if let Some(result) = resp.get("result").and_then(|r| r.as_object()) {
                            let equity = result
                                .get("e")
                                .and_then(|v| v.as_str())
                                .and_then(|s| s.parse::<f64>().ok())
                                .unwrap_or(0.0);
                            self.equity = equity;

                            if self.circuit.start_balance_usd <= 0.0 {
                                self.circuit.start_balance_usd = equity;
                                tracing::info!(
                                    "[BALANCE] Initial cash: ${:.2} | equity: ${:.2}",
                                    self.available_usd, equity
                                );
                            }

                            if self.tick_count.is_multiple_of(4) {
                                tracing::info!(
                                    "[BALANCE] ${:.2} USD cash | ${:.2} equity | {} assets held",
                                    self.available_usd,
                                    equity,
                                    self.holdings.len()
                                );
                            }
                        }
                    }
                    Err(e) => {
                        if self.tick_count.is_multiple_of(10) {
                            tracing::error!("[BALANCE] TradeBalance query failed: {e}");
                        }
                    }
                }
            }
            self.last_balance_check = now;
        }

        // ── Portfolio value: USD + holdings × live price ─────────
        {
            flow_log("portfolio_value");
            let mut total = self.available_usd;
            for (sym, qty) in &self.holdings {
                if sym == "USD" {
                    continue;
                }
                // Look up live price from features map
                let price = features_map
                    .get(sym)
                    .and_then(|f| f.get("price"))
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0);
                if price > 0.0 {
                    total += qty * price;
                }
            }
            self.portfolio_value = total;
        }

        // ── Auto-adopt: reconcile Kraken holdings → tracked positions ──
        // If we hold a coin on Kraken but don't have a position for it, adopt it
        // so the engine can manage exits. Runs every 10 ticks (~100s).
        // Gated behind ADOPT_KRAKEN_HOLDINGS (default off — auto-adopted positions had 5.4% win rate).
        flow_log("auto_adopt");
        if self.config.adopt_kraken_holdings && self.tick_count % 10 == 1 {
            for (sym, qty) in &self.holdings {
                if sym == "USD" || *qty <= 0.0 {
                    continue;
                }
                if self.positions.contains_key(sym.as_str()) {
                    continue;
                }
                // Only adopt coins we have live price data for (i.e. in our WS feed)
                let price = features_map
                    .get(sym.as_str())
                    .and_then(|f| f.get("price"))
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0);
                if price <= 0.0 {
                    continue;
                }
                // Skip dust — not worth managing (would immediately dust-exit)
                let usd_value = *qty * price;
                // Use a lower threshold for adoption than for new orders —
                // we want to protect existing holdings even if slightly under MIN_ORDER_USD.
                let min_adopt = self.config.env.get_f64("MIN_ORDER_USD", 5.0) * 0.80;
                if usd_value < min_adopt {
                    continue;
                }
                let rt_fee = 0.005; // ~0.5% round-trip fee buffer
                let tp = price * (1.0 + self.config.take_profit_pct / 100.0 + rt_fee);
                let sl = price * (1.0 - self.config.stop_loss_pct / 100.0 - rt_fee);
                let pos = OpenPosition {
                    symbol: sym.clone(),
                    entry_price: price,
                    qty: *qty,
                    remaining_qty: *qty,
                    tp_price: tp,
                    sl_price: sl,
                    highest_price: price,
                    entry_time: now,
                    entry_reasons: vec!["auto_adopted_from_kraken".to_string()],
                    entry_profile: "defensive".to_string(),
                    entry_context: "pre-existing Kraken holding adopted for management".to_string(),
                    entry_score: 0.5,
                    regime_label: "UNKNOWN".to_string(),
                    quant_bias: "NEUTRAL".to_string(),
                    npu_action: String::new(),
                    npu_conf: 0.0,
                    sl_order_txid: None,
                    quality_key: String::new(),
                    quality_score: 0.0,
                    min_hold_sec: 300,   // adopted positions: conservative defaults
                    max_hold_sec: 7200,
                    reeval_sec: 300,
                    entry_atr: 0.0,
                    entry_lane: "L3".to_string(),
                    feature_snapshot: None,
                    trend_alignment: 0,
                    trend_7d_pct: 0.0,
                    trend_30d_pct: 0.0,
                };
                tracing::info!(
                    "[AUTO-ADOPT] {} — {:.6} @ ${:.6} (TP=${:.6} SL=${:.6})",
                    sym,
                    qty,
                    price,
                    tp,
                    sl
                );
                self.positions.insert(sym.clone(), pos);
                save_positions(&self.config.positions_file, &self.positions);
            }
        }

        // ── Reverse sync: remove phantom positions not on Kraken ──
        // Always runs (every 10 ticks) regardless of adopt_kraken_holdings.
        // If we track a position but Kraken says we don't hold it, it's a ghost.
        if self.tick_count % 10 == 1 && !self.holdings.is_empty() {
            let stale: Vec<String> = self
                .positions
                .keys()
                .filter(|sym| {
                    let kraken_qty = self.holdings.get(sym.as_str()).copied().unwrap_or(0.0);
                    kraken_qty <= 0.0
                })
                .cloned()
                .collect();
            if !stale.is_empty() {
                for sym in &stale {
                    tracing::warn!(
                        "[SYNC] {} — phantom position removed (not held on Kraken)",
                        sym
                    );
                    self.positions.remove(sym);
                }
                save_positions(&self.config.positions_file, &self.positions);
            }
        }

        flow_log("reconcile_pending");
        if !self.config.paper_trading
            && (now - self.last_pending_check) >= PENDING_CHECK_INTERVAL_SEC
        {
            if let Some(api) = api {
                self.reconcile_pending_entries(api).await;
            }
            self.last_pending_check = now;
        }

        // Circuit breaker (realized + unrealized P&L)
        flow_log("circuit_breaker");
        let unrealized_pnl = self.current_unrealized_pnl(features_map);
        if disable_breakers {
            self.circuit.force_defensive = false;
            self.circuit.stop_trading = false;
        } else {
            self.circuit.set_unrealized_pnl(unrealized_pnl);
            self.circuit.check(&self.config);
            if self.circuit.stop_trading {
                if self.tick_count.is_multiple_of(60) {
                    tracing::info!("[LOOP] Trading stopped by circuit breaker");
                }
                return;
            }
        }

        // Build memory context for AI (includes portfolio)
        flow_log("build_ai_context");
        let portfolio_ctx = self.build_portfolio_context_with_prices(features_map);
        let journal_ctx = self.journal.build_context();
        let _memory_context = format!("{portfolio_ctx}\n{journal_ctx}");
        // Schedule overlay: auto-adjust parameters by AKST time block
        let schedule = crate::config::schedule::current_overlay(&self.config.env);
        if self.tick_count.is_multiple_of(60) && self.config.env.get_bool("SCHEDULE_ENABLED", false) {
            tracing::info!(
                "[SCHEDULE] block={} mode={} margin={:.2} rounds={} concurrent={} atr_tight={:.1}",
                schedule.block_name, schedule.mode, schedule.margin_gate,
                schedule.max_tool_rounds, schedule.max_concurrent, schedule.atr_tighten,
            );
        }

        let ai_parallel: usize = {
            let base: usize = self.config.env.get_parsed("AI_PARALLEL_COINS", 4).clamp(1, 32);
            // Schedule caps concurrency
            base.min(schedule.max_concurrent)
        };
        self.effective_ai_parallel = self.effective_ai_parallel_for(ai_parallel);
        if self.effective_ai_parallel < ai_parallel {
            self.meters.ai_throttle_hits += 1;
        }
        let _ai_batch_size: usize = self.config.env.get_parsed("AI_BATCH_SIZE", 1).clamp(1, 12);

        // ════════════════════════════════════════════════════════════
        //  EXIT CHECKS (all coins with open positions)
        // ════════════════════════════════════════════════════════════
        flow_log("exit_check_all");
        {
            let held_syms: Vec<String> = self.positions.keys().cloned().collect();
            for sym in &held_syms {
                let feats = match features_map.get(sym.as_str()) {
                    Some(v) => v,
                    None => continue,
                };
                let price = feats.get("price").and_then(|v| v.as_f64()).unwrap_or(0.0);
                if price <= 0.0 { continue; }

                // Dust sweep: proactively remove positions below Kraken's $5 minimum
                if !self.config.paper_trading {
                    if let Some(pos) = self.positions.get(sym.as_str()) {
                        let pos_val = pos.remaining_qty * price;
                        if pos_val < 5.0 {
                            tracing::warn!(
                                "[DUST-SWEEP] {} val=${:.2} < $5 — cleaning dust position",
                                sym, pos_val
                            );
                            self.execute_exit(sym, price, "dust_sweep", api, Some(ai)).await;
                            continue;
                        }
                    }
                }

                if let Some(pos) = self.positions.get_mut(sym.as_str()) {
                    pos.highest_price = pos.highest_price.max(price);
                    self.optimizer.update_position(sym, price);
                }

                let quantum_state = feats.get("quantum_state")
                    .and_then(|v| v.as_str()).unwrap_or("sideways");
                let regime = config::infer_regime(feats, quantum_state, &self.config);

                let exit_reason = self.check_exit(sym, feats, price, regime);
                if let Some(reason) = exit_reason {
                    self.execute_exit(sym, price, &reason, api, Some(ai)).await;
                }
            }
        }

        // ── AI EXIT MONITOR ──
        if self.config.use_nemo_exit && !self.positions.is_empty() {
            flow_log("ai_exit_check");
            self.nemo_exit_check(features_map, ai, api).await;
        }

        // ── AI SELF-OPTIMIZER (optional) ──
        #[cfg(feature = "nemo_optimizer")]
        self.nemo_optimizer.maybe_run(now, &self.journal, ai, self.portfolio_value).await;

        // ── AI SELF-REFLECTION (every 4-6h) ──
        if memory_enabled {
            self.nemo_memory_brain.maybe_reflect(now, ai).await;
        }


        // ── MODE SWITCH: per_coin (classic) vs portfolio_allocator (V2) ──
        // per_coin = existing per-coin BUY/HOLD/SELL decisions via AI_1
        // portfolio_allocator = portfolio-level allocation via AI_1 + rebalance
        // Exits (TP/SL/trailing/AI) always run above this point regardless of mode.
        let mode = self.config.env.get_str("MODE", "per_coin");
        if mode == "portfolio_allocator" {
            flow_log("portfolio_allocator");
            self.run_portfolio_allocator(features_map, api, ai).await;
            return;
        }

        // ════════════════════════════════════════════════════════════
        //  SINGLE ENGINE: all coins → gate checks → AI_1 → entries
        // ════════════════════════════════════════════════════════════
        flow_log("engine_ai");

        // Build portfolio snapshot once per tick for AI prompts
        let _portfolio_snapshot = self.build_portfolio_snapshot(features_map);

        // Loss cooldown gate — block all entries after N consecutive losses
        if self.loss_cooldown_until > 0.0 && now < self.loss_cooldown_until {
            let remaining = self.loss_cooldown_until - now;
            if self.tick_count.is_multiple_of(10) {
                tracing::warn!(
                    "[LOSS-COOLDOWN] entries blocked — {:.0}s remaining (streak={})",
                    remaining, self.hf_streak_losses
                );
            }
            return;
        }

        // BTC crash gate — skip all entries if BTC is crashing
        if self.market_brain.is_btc_crashing() {
            tracing::warn!(
                "[MARKET-BRAIN] BTC crash detected — skipping all entries (regime={})",
                self.market_brain.regime(),
            );
            return;
        }

        // Collect ALL candidates — gates are advisory, AI_1 decides
        #[derive(Clone)]
        struct CoinCandidate {
            sym: String,
            price: f64,
            regime_str: String,
            tier: &'static str,
            profile: config::ProfileParams,
            gate_warnings: Vec<String>,
        }
        let mut candidates: Vec<CoinCandidate> = Vec::with_capacity(features_map.len());
        let mut gate_warned = 0u32;
        let mut gate_rejected = 0u32;
        let mut gate_reject_counts: HashMap<String, usize> = HashMap::new();
        let mut hard_reject_counts: HashMap<String, usize> = HashMap::new();
        let mut advisory_counts: HashMap<String, usize> = HashMap::new();

        let bump = |map: &mut HashMap<String, usize>, key: &str| {
            *map.entry(key.to_string()).or_insert(0) += 1;
        };
        let warn_key = |s: &str| -> String {
            s.split(|c| c == '=' || c == '(').next().unwrap_or(s).to_string()
        };

        // Snapshot sentiment scores once before the loop (avoid repeated lock acquisitions)
        let sentiment_snapshot: std::collections::HashMap<String, f64> =
            if let Some(intel) = &self.cloud_intel {
                let state = intel.read().await;
                state.sentiments.iter().map(|(k, v)| (k.clone(), v.score)).collect()
            } else {
                std::collections::HashMap::new()
            };

        for (sym, feats) in features_map.iter() {
            let price = feats.get("price").and_then(|v| v.as_f64()).unwrap_or(0.0);
            if price <= 0.0 { continue; }

            // ── HF: Update rolling orderbook imbalance history ──
            let hf_imb_win = self.config.env.get_f64("HF_IMB_WIN", 5.0) as usize;
            let book_imb_now = feats.get("book_imbalance").and_then(|v| v.as_f64()).unwrap_or(0.0);
            {
                let q = self.book_imb_hist.entry(sym.clone()).or_insert_with(std::collections::VecDeque::new);
                q.push_back(book_imb_now);
                while q.len() > hf_imb_win { q.pop_front(); }
            }

            // ── PIPELINE GATE A: Regime State Machine ────────────────
            // Read MTF early — needed for regime SM + lane assignment
            let mtf_7d = {
                static MTF_CACHE: std::sync::OnceLock<
                    std::sync::Mutex<(f64, serde_json::Value)>
                > = std::sync::OnceLock::new();
                let cache = MTF_CACHE.get_or_init(|| {
                    std::sync::Mutex::new((0.0, serde_json::Value::Null))
                });
                let mut guard = cache.lock().unwrap_or_else(|e| e.into_inner());
                if now - guard.0 > 15.0 {
                    guard.1 = std::fs::read_to_string("data/mtf_trends.json")
                        .ok()
                        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
                        .unwrap_or(serde_json::Value::Null);
                    guard.0 = now;
                }
                guard.1.get(sym)
                    .or_else(|| guard.1.get(&sym.to_uppercase()))
                    .and_then(|v| v.get("trend_7d").and_then(|x| x.as_f64()))
                    .unwrap_or(-99.0)
            };

            let atr_pct      = feats.get("atr").and_then(|v| v.as_f64()).unwrap_or(0.0) / price.max(0.0001);
            let raw_spread   = feats.get("spread_pct").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let spread_pct   = if raw_spread <= 0.0 { atr_pct * 0.10 } else { raw_spread };
            let atr_norm     = atr_pct.max(0.010);
            let vol_ratio_pre= feats.get("vol_ratio").and_then(|v| v.as_f64()).unwrap_or(1.0);
            let momentum_pre = feats.get("momentum_buy").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let zscore_pre   = feats.get("zscore").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let imbalance_pre= feats.get("book_imbalance").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let corr_btc     = feats.get("quant_corr_btc").and_then(|v| v.as_f64()).unwrap_or(1.0);
            let rsi          = feats.get("rsi").and_then(|v| v.as_f64()).unwrap_or(50.0);

            // Estimate candle count from OHLC data availability
            let candle_count = self.tick_count.min(200) as u32;

            let regime = {
                let sm = self.regime_machines
                    .entry(sym.to_string())
                    .or_default();
                let inputs = regime_sm::inputs_from_feats(feats, mtf_7d, candle_count);
                sm.update(&inputs)
            };

            // regime_str for backward compat with select_profile / gate scoring
            let regime_str  = regime.as_str().to_string();
            let _quantum_state = feats.get("quantum_state")
                .and_then(|v| v.as_str()).unwrap_or("sideways");
            let market_state= feats.get("market_state").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let is_l2       = feats.get("is_l2").and_then(|v| v.as_bool()).unwrap_or(false);
            let mut gate_warnings: Vec<String> = Vec::new();

            // Advisory: market state (non-blocking)
            if market_state < self.config.market_state_min && !is_l2 {
                gate_warnings.push(format!("low_mkt={:.2}", market_state));
            }
            let (profile_now, _) =
                config::select_profile(&self.config, &regime_str, market_state, rsi, &self.circuit);

            // ── PIPELINE GATE B: assign_lane() ───────────────────────
            let tier_static  = config::coin_tier(sym);
            let is_behavioral_meme = tier_static != "large"
                && config::is_behavioral_meme(price, atr_pct, vol_ratio_pre, corr_btc, sym);

            let lane_result = lane::assign_lane(&lane::LaneInputs {
                symbol:             sym,
                regime,
                tier:               tier_static,
                is_behavioral_meme,
                mtf_7d,
                vol_ratio:          vol_ratio_pre,
                momentum:           momentum_pre,
                zscore:             zscore_pre,
                imbalance:          imbalance_pre,
                spread_pct,
                atr_norm,
            });

            // Derive tier string for downstream compat (profile gates, meme caps)
            let tier = match &lane_result {
                lane::LaneResult::Assigned(lane::Lane::L4) => "meme",
                _ => tier_static,
            };

            // Lane block → advisory only (BEARISH and zero-price still hard-block above)
            // AI sees the warning and makes the final call
            let assigned_lane = if lane_result.is_blocked() {
                gate_warned += 1;
                gate_warnings.push(format!("lane_advisory={}", lane_result.reason()));
                bump(&mut advisory_counts, "lane_advisory");
                // Default to L3 so AI has a valid lane context
                lane::Lane::L3
            } else {
                lane_result.lane().unwrap()
            };

            // ── PIPELINE GATE C: fee_gate() — advisory only ───────────
            let trend_score_pre = feats.get("trend_score").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let fee_result = fee_filter::fee_gate(&fee_filter::FeeInputs {
                symbol:      sym,
                lane:        assigned_lane,
                atr_norm,
                spread_pct,
                trend_score: trend_score_pre,
                zscore:      zscore_pre,
                momentum:    momentum_pre,
                vol_ratio:   vol_ratio_pre,
            });

            if !fee_result.passed {
                // Advisory — AI sees the fee context, not a hard block
                gate_warnings.push(format!("fee_advisory={} net={:.2}%", fee_result.reason, fee_result.net_edge_pct));
                bump(&mut advisory_counts, "fee_advisory");
            }

            // Advisory: spread/liquidity/slippage (was hard gate)
            if tier != "large" {
                let gate_feats = rules_eval::GateFeatures::from_json(feats);
                let gate_config = rules_eval::GateConfig {
                    min_liquidity: profile_now.min_liq,
                    min_vol_ratio: profile_now.min_vol_ratio,
                    max_spread_pct: profile_now.max_spread_pct,
                    max_slip_pct: profile_now.max_slip_pct,
                    min_fill_100k: self.config.env.get_f64("MIN_FILL_100K", 1_000.0).max(0.0),
                    require_liq_metrics: self.config.env.get_bool("REQUIRE_LIQ_METRICS", false),
                    max_atr_ratio: self.config.env.get_f64("MAX_ATR_RATIO", 2.5),
                    use_scoring: self.config.env.get_bool("GATE_USE_SCORING", false),
                    profit_target_multiplier: self.config.env.get_f64("GATE_PROFIT_MULTIPLIER", 1.0),
                };
                let slip_mult = config::slip_regime_mult(&regime_str)
                    * if is_l2 { schedule.l2_aggression } else { 1.0 };
                let (gate_ok, gate_reasons) =
                    rules_eval::passes_gate_checks(&gate_feats, &gate_config, &tier, slip_mult);
                if !gate_ok {
                    for r in &gate_reasons {
                        bump(&mut gate_reject_counts, r);
                        gate_warnings.push(r.clone());
                    }
                    crate::event_bus::publish(&self.bus, crate::event_bus::Event::RiskReject {
                        ts: now,
                        symbol: sym.to_string(),
                        reason: gate_reasons.join(", "),
                        spread_pct: Some(gate_feats.spread_pct),
                        slip_pct: gate_feats.buy_slip_pct_100k,
                    });
                }
            }

            // Advisory: falling knife (was hard gate)
            let direction = feats.get("trend_direction")
                .and_then(|v| v.as_str()).unwrap_or("NEUTRAL");
            if matches!(direction, "FALLING" | "DOWN" | "STRONG_BEAR" | "BEAR") && rsi < 35.0 {
                gate_warnings.push(format!("falling_knife_rsi={:.0}", rsi));
            }

            // Sentiment gate: advisory only — AI sees the warning and decides
            if self.config.env.get_bool("SENTIMENT_GATE_ENABLED", true) {
                let score = sentiment_snapshot.get(sym).copied().unwrap_or(0.0);
                let min_score = self.config.env.get_f64("SENTIMENT_GATE_MIN", -0.25);
                if score < min_score {
                    gate_warnings.push(format!("sentiment={:.2}", score));
                }
            }

            if !gate_warnings.is_empty() {
                gate_warned += 1;
                for w in &gate_warnings {
                    let key = warn_key(w);
                    bump(&mut advisory_counts, &key);
                }
            }
            candidates.push(CoinCandidate {
                sym: sym.clone(),
                price,
                regime_str,
                tier,
                profile: profile_now,
                gate_warnings,
            });
        }

        // ── GCD-STYLE ELIMINATION (stepwise reduction) ──────────────
        // Each step removes one class of weak candidates, like gcd(a,b) = gcd(b, a mod b).
        let max_candidates: usize = self.config.env.get_f64("MAX_CANDIDATES_PER_TICK", 5.0) as usize;
        let pre_rank_count = candidates.len();

        // ALL GATES ARE ADVISORY — data is passed to AI_1, she decides
        // (hard filters were blocking all candidates before AI could evaluate)
        let crash_cut = 0usize;
        let lane_cut = 0usize;
        let trend_cut = 0usize;
        let risk_cut = 0usize;
        let vol_cut = 0usize;

        // Step 5: Sort survivors by composite strength + liquidity, truncate
        // Cap at min(MAX_CANDIDATES_PER_TICK, schedule.max_concurrent)
        let effective_max = max_candidates.min(schedule.max_concurrent);
        if candidates.len() > effective_max {
            candidates.sort_by(|a, b| {
                let score = |c: &CoinCandidate| -> f64 {
                    let feats = match features_map.get(&c.sym) {
                        Some(v) => v,
                        None => return 0.0,
                    };
                    let s_buy = feats.get("s_buy").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    let s_sell = feats.get("s_sell").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    let momentum = feats.get("momentum_score").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    let spread = feats.get("spread_pct").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    let vol_ratio = feats.get("vol_ratio").and_then(|v| v.as_f64()).unwrap_or(1.0);
                    let liquidity = feats.get("liquidity").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    let vol_boost = if vol_ratio >= 3.0 { (vol_ratio - 2.0) * 0.8 } else { 0.0 };
                    (s_buy - s_sell) + momentum * 0.5 - spread * 0.5 + vol_boost + liquidity * 0.1
                };
                score(b).partial_cmp(&score(a)).unwrap_or(std::cmp::Ordering::Equal)
            });
            candidates.truncate(effective_max);
        }

        if self.tick_count.is_multiple_of(10) && (crash_cut + lane_cut + trend_cut + risk_cut + vol_cut) > 0 {
            tracing::info!(
                "[GCD] Eliminated: crash={} lane={} trend={} risk={} vol_l2={} | {} survivors",
                crash_cut, lane_cut, trend_cut, risk_cut, vol_cut, candidates.len()
            );
        }

        if self.tick_count.is_multiple_of(5) {
            let syms: Vec<&str> = candidates.iter().map(|c| c.sym.as_str()).collect();
            tracing::info!(
                "[ENGINE] {} coins → {} passed gates, {} gate-warned, {} hard-rejected → top {} candidates: {:?}",
                features_map.len(),
                pre_rank_count,
                gate_warned,
                gate_rejected,
                candidates.len(),
                syms,
            );
            let format_map = |map: &HashMap<String, usize>| -> String {
                let mut items: Vec<(&String, &usize)> = map.iter().filter(|(_, v)| **v > 0).collect();
                items.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
                items.into_iter()
                    .map(|(k, v)| format!("{k}={v}"))
                    .collect::<Vec<String>>()
                    .join(", ")
            };
            let top_n = |map: &HashMap<String, usize>, n: usize| -> String {
                let mut items: Vec<(&String, &usize)> = map.iter().filter(|(_, v)| **v > 0).collect();
                items.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
                items.into_iter()
                    .take(n)
                    .map(|(k, v)| format!("{k}={v}"))
                    .collect::<Vec<String>>()
                    .join(", ")
            };
            let hard = format_map(&hard_reject_counts);
            let rejects = format_map(&gate_reject_counts);
            let warns = format_map(&advisory_counts);
            if !hard.is_empty() || !rejects.is_empty() || !warns.is_empty() {
                tracing::info!(
                    "[GATE-DETAIL] hard_reject=[{}] gate_reject=[{}] warnings=[{}]",
                    hard, rejects, warns
                );
                let hard_top = top_n(&hard_reject_counts, 3);
                let rej_top = top_n(&gate_reject_counts, 3);
                let warn_top = top_n(&advisory_counts, 3);
                tracing::info!(
                    "[GATE-TOP] hard=[{}] reject=[{}] warn=[{}]",
                    hard_top, rej_top, warn_top
                );
            }
        }

        // -- DECISION ENGINE (Math-only — deterministic, no LLM for entries) --
        {
            let mut all_decisions: std::collections::HashMap<String, crate::ai_bridge::AiDecision> =
                std::collections::HashMap::new();

            for cand in &candidates {
                let sym = cand.sym.as_str();
                let feats = match features_map.get(sym) {
                    Some(v) => v,
                    None => continue,
                };

                // Math engine decision (already computed by Rust features engine)
                let math_action = feats.get("math_action")
                    .and_then(|v| v.as_str()).unwrap_or("HOLD");
                let math_confidence = feats.get("math_confidence")
                    .and_then(|v| v.as_f64()).unwrap_or(0.0);

                // Deterministic lane classification
                let lane = classify_lane(feats, cand.price);

                // Build reason codes from key signals
                let trend = feats.get("trend_score").and_then(|v| v.as_f64()).unwrap_or(0.0) as i32;
                let book_imb = feats.get("book_imbalance").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let rsi = feats.get("rsi").and_then(|v| v.as_f64()).unwrap_or(50.0);
                let momentum = feats.get("momentum_score").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let vol = feats.get("vol_ratio").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let buy_ratio = feats.get("buy_ratio").and_then(|v| v.as_f64()).unwrap_or(0.5);

                let commentary = format!(
                    "trend{:+} imb{:+.2} rsi{:.0} mom{:.3} vol{:.1}x br{:.2} {}",
                    trend, book_imb, rsi, momentum, vol, buy_ratio, cand.regime_str
                );

                let ai_decision = crate::ai_bridge::AiDecision {
                    action: math_action.to_string(),
                    confidence: math_confidence,
                    trend_score: trend,
                    lane: lane.clone(),
                    reason: commentary.clone(),
                    regime_label: cand.regime_str.clone(),
                    limit_price: None, // use .env LIMIT_OFFSET_PCT at order placement
                    source: "math_engine".into(),
                    decision_ts: now_ts(),
                    ..crate::ai_bridge::AiDecision::default()
                };

                // Record math decision in AI memory
                self.record_nemo_memory(sym, math_action, math_confidence, &commentary, cand.price, &lane, "math");

                all_decisions.insert(sym.to_string(), ai_decision);
            }

            // Log math decisions periodically
            if self.tick_count.is_multiple_of(5) {
                let mut buy = 0u32;
                let mut sell = 0u32;
                let mut hold = 0u32;
                for d in all_decisions.values() {
                    match d.action.as_str() {
                        "BUY" => buy += 1,
                        "SELL" => sell += 1,
                        _ => hold += 1,
                    }
                }
                tracing::info!(
                    "[DECISION-SUMMARY] math BUY={} SELL={} HOLD={} (candidates={})",
                    buy, sell, hold, all_decisions.len()
                );
                for (sym, d) in &all_decisions {
                    tracing::info!(
                        "[MATH-DECIDE] {} {}|{:.2}|{} | {}",
                        sym, d.action, d.confidence, d.lane, d.reason
                    );
                }
            }

            // ══════════════════════════════════════════════════════
            //  PHASE A: Collect BUY candidates that pass all filters
            //  Flow: for each coin → cooldown → confidence → action filter
            //        → dynamic threshold → regime gate → position/cash checks
            //        → NPU pre-scan (PASS/REJECT) → build entry prompt
            //        → push to entry_candidates vec for AI_1 batch
            // ══════════════════════════════════════════════════════
            struct EntryCandidate {
                sym: String,
                price: f64,
                regime_str: String,
                tier: &'static str,
                profile: config::ProfileParams,
                ai_decision: crate::ai_bridge::AiDecision,
                feats_entry: serde_json::Value,
                entry_prompt: String,
                quality_key: String,
                quality_score: f64,
                is_nemo_override: bool,
            }
            let mut entry_candidates: Vec<EntryCandidate> = Vec::new();
            let mut entry_reject_counts: HashMap<&'static str, usize> = HashMap::new();
            let mut bump_entry = |key: &'static str| {
                *entry_reject_counts.entry(key).or_insert(0) += 1;
            };

            for cand in &candidates {
                let sym = cand.sym.as_str();
                let price = cand.price;
                let feats = match features_map.get(sym) {
                    Some(v) => v,
                    None => continue,
                };

                // ── AI cooldown: skip if we just asked about this coin ──
                if let Some(&last_ai) = self.last_ai_call_ts.get(sym) {
                    if (now - last_ai) < self.config.ai_cooldown_sec {
                        bump_entry("ai_cooldown");
                        continue;
                    }
                }
                self.last_ai_call_ts.insert(sym.to_string(), now);

                // ── Trade cooldown: skip if we just traded this coin ──
                if let Some(&last_trade) = self.last_trade_ts.get(sym) {
                    if (now - last_trade) < self.config.trade_cooldown_sec {
                        bump_entry("trade_cooldown");
                        continue;
                    }
                }

                // ── Price cooldown: require X% drop before re-entry ──
                let price_cool_pct = self.config.env.get_f64("PRICE_COOLDOWN_PCT", 0.3);
                if price_cool_pct > 0.0 {
                    if let Some(&last_px) = self.last_entry_price.get(sym) {
                        if last_px > 0.0 && price >= last_px * (1.0 - price_cool_pct / 100.0) {
                            self.record_entry_reject(now, sym,
                                &format!("price_cooldown px={:.2}>=last={:.2}*{:.1}%",
                                    price, last_px, price_cool_pct), false);
                            bump_entry("price_cooldown");
                            continue;
                        }
                    }
                }

                let mut ai_decision = match all_decisions.remove(sym) {
                    Some(d) => d,
                    None => continue,
                };
                self.meters.bump_ai_action(now, &ai_decision.action);
                ai_decision.regime_label = cand.regime_str.clone();

                // ── L1: Emit decision.signal ──
                crate::event_bus::publish(&self.bus, crate::event_bus::Event::DecisionSignal {
                    ts: now,
                    symbol: sym.to_string(),
                    action: ai_decision.action.clone(),
                    confidence: ai_decision.confidence,
                    lane: ai_decision.lane.clone(),
                    bucket: cand.regime_str.clone(),
                    reason: ai_decision.reason.clone(),
                });

                // ── L2: Emit features.snapshot only on bucket change ──
                if self.telemetry_level >= crate::event_bus::TelemetryLevel::Snapshot {
                    let new_bucket = cand.regime_str.clone();
                    let changed = self.prev_coin_state.get(sym)
                        .map(|prev| prev.0 != new_bucket)
                        .unwrap_or(true);
                    if changed {
                        let score = feats.get("weighted_score").and_then(|v| v.as_f64()).unwrap_or(0.0);
                        let trend = feats.get("trend_score").and_then(|v| v.as_f64()).unwrap_or(0.0) as i32;
                        crate::event_bus::publish(&self.bus, crate::event_bus::Event::FeaturesSnapshot {
                            ts: now,
                            symbol: sym.to_string(),
                            bucket: cand.regime_str.clone(),
                            lane: "L1".into(),
                            score,
                            trend_score: trend,
                            caps: None,
                        });
                        self.prev_coin_state.insert(sym.to_string(), (new_bucket, "L1".into()));
                    }
                }
                // limit_price resolved at order placement from .env LIMIT_OFFSET_PCT

                // ── ADX TREND FILTER: no trend = halve confidence ──
                let gate_adx = feats.get("adx").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let adx_thresh = self.config.env.get_f64("ADX_NO_TREND_THRESH", 20.0);
                if gate_adx < adx_thresh && gate_adx > 0.0 {
                    let old_conf = ai_decision.confidence;
                    ai_decision.confidence *= 0.5;
                    tracing::debug!(
                        "[ADX-FILTER] {} ADX={:.1}<{:.0} → conf {:.2}→{:.2}",
                        sym, gate_adx, adx_thresh, old_conf, ai_decision.confidence
                    );
                }

                // Math confidence is advisory — AI_1 makes final decision.
                // Only hard-reject if math confidence is extremely low (< 0.15)
                if ai_decision.confidence < 0.15 {
                    self.record_entry_reject(
                        now, sym,
                        &format!("conf_very_low conf={:.2}<0.15", ai_decision.confidence),
                        false,
                    );
                    continue;
                }

                // ── ENTROPY GATE: noisy + low volume = skip, noisy + high volume = let AI_1 see it ──
                let gate_entropy = feats.get("shannon_entropy").and_then(|v| v.as_f64()).unwrap_or(0.5);
                let gate_vol = feats.get("vol_ratio").and_then(|v| v.as_f64()).unwrap_or(1.0);
                if gate_entropy >= 0.85 && gate_vol < 3.0 {
                    tracing::info!(
                        "[NOISE-ADVISORY] {} E={:.2}>=0.85 vol={:.1} — passing to AI_1",
                        sym, gate_entropy, gate_vol
                    );
                }

                // ── FEE-AWARE GATE: REMOVED (was blocking all trades on 1-min ATR)
                // Fee protection is now handled by FEE_MARGIN_MULTIPLIER in execute_entry()
                // which checks SL distance > 2x round-trip fees at order time.

                // ── HEDGE-FUND ENTRY FILTER (HF_) — all .env tunable ──────────
                // Order: spread → chop → sideways → imbalance persistence → MTF trend → L2
                // These gates run BEFORE AI override — AI decides yes/no only after
                // the strategy passes the fund-grade risk/structure locks.

                let hf_is_l2 = feats.get("is_l2").and_then(|v| v.as_bool()).unwrap_or(false);
                let hf_hurst = feats.get("hurst_exp").and_then(|v| v.as_f64()).unwrap_or(0.5);
                let hf_entropy = feats.get("shannon_entropy").and_then(|v| v.as_f64()).unwrap_or(0.5);
                let hf_atr = feats.get("atr").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let hf_atr_pct = if price > 0.0 { (hf_atr / price) * 100.0 } else { 0.0 };
                let hf_spread = feats.get("spread_pct").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let hf_trend = feats.get("trend_score").and_then(|v| v.as_f64()).unwrap_or(0.0) as i32;
                let hf_vol_ratio = feats.get("vol_ratio").and_then(|v| v.as_f64()).unwrap_or(1.0);

                // ── HF: Spread cap ──
                let hf_spread_cap = if cand.tier == "meme" {
                    self.config.env.get_f64("HF_SPREAD_CAP_MEME", 0.50)
                } else {
                    self.config.env.get_f64("HF_SPREAD_CAP", 0.20)
                };
                if hf_spread > hf_spread_cap {
                    tracing::info!(
                        "[HF-SPREAD-ADVISORY] {} spread={:.3}%>cap={:.2}% — passing to AI_1",
                        sym, hf_spread, hf_spread_cap
                    );
                }

                // ── HF: Chop / chaos filter ──
                let hf_hurst_min = self.config.env.get_f64("HF_HURST_MIN", 0.58);
                let hf_entropy_max = self.config.env.get_f64("HF_ENTROPY_MAX", 0.55);
                let hf_atr_max = if cand.tier == "meme" {
                    self.config.env.get_f64("HF_ATR_PCT_MAX_MEME", 5.00)
                } else {
                    self.config.env.get_f64("HF_ATR_PCT_MAX", 2.50)
                };
                let chop_ok = hf_hurst >= hf_hurst_min
                    && hf_entropy <= hf_entropy_max
                    && hf_atr_pct <= hf_atr_max;
                if !chop_ok {
                    // Advisory only — AI_1 sees raw Hurst/entropy/ATR in features and decides
                    tracing::debug!(
                        "[HF-CHOP-ADVISORY] {} H={:.2}<{:.2} E={:.2}>{:.2} ATR={:.2}%>{:.1}% — passing to AI_1",
                        sym, hf_hurst, hf_hurst_min, hf_entropy, hf_entropy_max, hf_atr_pct, hf_atr_max
                    );
                }

                // ── HF: Sideways regime — advisory only ──
                // L2 lane exists specifically for sideways trading, so don't hard-reject
                if cand.regime_str == "sideways" {
                    let sw_hurst = self.config.env.get_f64("HF_SIDEWAYS_HURST_MIN", 0.62);
                    let sw_trend = self.config.env.get_f64("HF_SIDEWAYS_TREND_MIN", 2.0) as i32;
                    if !(hf_hurst >= sw_hurst && hf_trend >= sw_trend) {
                        tracing::debug!(
                            "[HF-SIDEWAYS-ADVISORY] {} H={:.2}<{:.2} T={}<{} — passing to AI_1",
                            sym, hf_hurst, sw_hurst, hf_trend, sw_trend
                        );
                    }
                }

                // ── HF LOCK A: Orderbook Imbalance Persistence ──
                // Require sustained buy pressure, not a one-tick spoof.
                // L2 mean-reversion exempt (buys dips with negative imbalance).
                if !hf_is_l2 {
                    let hf_imb_min_count = self.config.env.get_f64("HF_IMB_MIN_COUNT", 5.0) as usize;
                    let (min_now, min_avg) = if cand.tier == "meme" {
                        (self.config.env.get_f64("HF_IMB_MIN_NOW_MEME", 0.18),
                         self.config.env.get_f64("HF_IMB_MIN_AVG_MEME", 0.12))
                    } else {
                        (self.config.env.get_f64("HF_IMB_MIN_NOW", 0.12),
                         self.config.env.get_f64("HF_IMB_MIN_AVG", 0.08))
                    };
                    let imb_ok = if let Some(q) = self.book_imb_hist.get(sym) {
                        if q.len() < hf_imb_min_count {
                            false
                        } else {
                            let now_imb = *q.back().unwrap_or(&0.0);
                            let avg_imb = q.iter().copied().sum::<f64>() / q.len() as f64;
                            now_imb >= min_now && avg_imb >= min_avg
                        }
                    } else {
                        false
                    };
                    // Bonus path: volume spike + decent imbalance = allow
                    let hf_vol_bonus = self.config.env.get_f64("HF_VOLRATIO_BONUS_MIN", 2.50);
                    let hf_imb_bonus = self.config.env.get_f64("HF_IMB_BONUS_MIN", 0.12);
                    let cur_imb = feats.get("book_imbalance").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    let imb_ok = imb_ok || (hf_vol_ratio >= hf_vol_bonus && cur_imb >= hf_imb_bonus);
                    if !imb_ok {
                        tracing::debug!("[HF-IMB-ADVISORY] {} imb_not_persistent tier={} — passing to AI_1", sym, cand.tier);
                    }
                }

                // ── HF LOCK B: Multi-Timeframe Trend Votes (EMA21/55/200) ──
                let hf_ema21 = feats.get("ema21").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let hf_ema55 = feats.get("ema55").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let hf_ema200 = feats.get("ema200").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let hf_mtf_min = self.config.env.get_f64("HF_MTF_VOTES_MIN", 2.0) as i32;
                if !hf_is_l2 {
                    let mut votes = 0i32;
                    if hf_ema21 > 0.0 && price > hf_ema21 { votes += 1; }
                    if hf_ema55 > 0.0 && price > hf_ema55 { votes += 1; }
                    if hf_ema200 > 0.0 && price > hf_ema200 { votes += 1; }
                    if votes < hf_mtf_min {
                        tracing::debug!("[HF-MTF-ADVISORY] {} votes={}/{} — passing to AI_1", sym, votes, hf_mtf_min);
                    }
                } else {
                    // L2 mean-reversion: don't chase above EMA21
                    let hf_l2_chase = self.config.env.get_f64("HF_L2_CHASE_MULT", 1.005);
                    if hf_ema21 > 0.0 && price > hf_ema21 * hf_l2_chase {
                        tracing::debug!("[HF-L2-ADVISORY] {} chase price>{:.4}*{:.3} — passing to AI_1", sym, hf_ema21, hf_l2_chase);
                    }
                    // L2: require reversion setup (zscore or BB lower touch)
                    let hf_l2_z = self.config.env.get_f64("HF_L2_Z_MIN", -0.90);
                    let hf_l2_bb = self.config.env.get_f64("HF_L2_BB_MULT", 1.01);
                    let hf_z = feats.get("zscore").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    let hf_bb_lower = feats.get("bb_lower").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    if !(hf_z <= hf_l2_z || (hf_bb_lower > 0.0 && price <= hf_bb_lower * hf_l2_bb)) {
                        tracing::debug!("[HF-L2-ADVISORY] {} setup_missing z={:.2}>{:.2} bb={:.4} — passing to AI_1", sym, hf_z, hf_l2_z, hf_bb_lower);
                    }
                }

                // ── AI Override: Math is advisor, AI has final say ──
                // SELL always blocked. HOLD with strong features → ask AI for override.
                // Read pump detector signals computed in features.rs
                let pump_early     = feats.get("pump_early").and_then(|v| v.as_bool()).unwrap_or(false);
                let pump_confirmed = feats.get("pump_confirmed").and_then(|v| v.as_bool()).unwrap_or(false);
                let pump_score     = feats.get("pump_score").and_then(|v| v.as_f64()).unwrap_or(0.0);

                let pump_vol = feats.get("vol_ratio").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let pump_imb = feats.get("book_imbalance").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let pump_mom = feats.get("momentum_score").and_then(|v| v.as_f64()).unwrap_or(0.0);
                if pump_confirmed {
                    tracing::info!("[PUMP-CONFIRMED] {} vol={:.1}x imb={:.2} mom={:.2} — fast-tracking to AI",
                        sym, pump_vol, pump_imb, pump_mom);
                } else if pump_early {
                    tracing::info!("[PUMP-EARLY] {} vol={:.1}x imb={:.2} — early pump signal detected",
                        sym, pump_vol, pump_imb);
                }

                let is_nemo_override = ai_decision.action == "HOLD" && {
                    let o_trend = feats.get("trend_score").and_then(|v| v.as_f64()).unwrap_or(0.0) as i32;
                    let o_buy_ratio = feats.get("buy_ratio").and_then(|v| v.as_f64()).unwrap_or(0.5);
                    let o_momentum = feats.get("momentum_score").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    let o_book_imb = feats.get("book_imbalance").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    let o_vol_ratio = feats.get("vol_ratio").and_then(|v| v.as_f64()).unwrap_or(1.0);
                    // PUMP SIGNALS: always override to AI if pump detected
                    pump_early || pump_confirmed
                    // Strong features: trend positive + at least one confirming signal
                    || (o_trend >= 2 && (o_momentum > 0.3 || o_buy_ratio > 0.55 || o_book_imb > 0.15))
                    || o_vol_ratio >= 2.0
                    || o_book_imb >= 0.15
                };

                // Math action is ADVISORY — AI_1 sees it and makes final call.
                // Hard-block only on SELL with very high confidence (>0.85) and no redeeming features.
                let math_advisory = self.config.env.get_bool("MATH_ENTRY_ADVISORY", true);
                if !math_advisory {
                    // Legacy mode: math SELL hard-blocks
                    if ai_decision.action == "SELL" {
                        continue;
                    }
                    if ai_decision.action == "HOLD" && !is_nemo_override {
                        continue;
                    }
                } else {
                    // Advisory mode: only hard-block extreme SELLs (conf>0.85, no redeeming signals)
                    if ai_decision.action == "SELL" && ai_decision.confidence > 0.85 && !is_nemo_override {
                        self.record_entry_reject(now, sym,
                            &format!("math_hard_sell conf={:.2}>0.85", ai_decision.confidence), false);
                        continue;
                    }
                    if ai_decision.action == "SELL" || (ai_decision.action == "HOLD" && !is_nemo_override) {
                        tracing::info!(
                            "[MATH-ADVISORY] {} math={}|{:.2} — passing to AI_1 for context evaluation",
                            sym, ai_decision.action, ai_decision.confidence
                        );
                    }
                }
                if is_nemo_override {
                    tracing::info!(
                        "[AI-OVERRIDE] {} math=HOLD but strong features → sending to AI for final say",
                        sym
                    );
                }

                // Dynamic threshold (only for math BUY — override candidates skip this,
                // AI's confidence is the real gate for those)
                if !is_nemo_override {
                    let dynamic_base = feats.get("dynamic_threshold")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(self.config.entry_threshold);
                    let tier_adj: f64 = match cand.tier {
                        "large" => -0.03,
                        "meme"  => 0.05,
                        _       => 0.0,
                    };
                    let spread_pct = feats.get("spread_pct")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.0);
                    let coin_threshold = (dynamic_base + tier_adj + spread_pct * 0.05).clamp(0.30, 0.80);
                    if ai_decision.confidence < coin_threshold {
                        tracing::info!(
                            "[THRESHOLD-ADVISORY] {} conf={:.2}<{:.2} (dynamic={:.2} tier={} spread={:.2}%) — passing to AI_1",
                            sym, ai_decision.confidence, coin_threshold, dynamic_base, cand.tier, spread_pct
                        );
                    }
                }

                // ── HF: ADAPTIVE CONFIDENCE FLOOR — advisory only, AI_1 makes final call ──
                // Dynamic floor tightens on losses, relaxes on wins.
                // Logged as advisory — does NOT block. AI_1 sees all data and decides.
                let hf_conf_floor = self.entry_conf_floor_dyn;
                if ai_decision.confidence < hf_conf_floor {
                    tracing::info!(
                        "[HF-CONF-ADVISORY] {} conf={:.2}<floor={:.2}{} — passing to AI_1",
                        sym, ai_decision.confidence, hf_conf_floor,
                        if is_nemo_override { " (override)" } else { "" },
                    );
                }

                // ── REGIME CONTEXT — advisory only, AI_1 decides everything ──
                let greed = self.market_brain.state()
                    .map(|s| s.greed_index)
                    .unwrap_or(50.0);
                let _regime = self.market_brain.regime();
                // All regime/greed info is in the data line — AI_1 sees it and decides autonomously
                if self.tick_count.is_multiple_of(30) {
                    tracing::info!(
                        "[REGIME-INFO] {} regime={} greed={:.0} tier={} — AI decides",
                        sym, cand.regime_str, greed, cand.tier
                    );
                }

                // Skip if already holding
                if self.positions.contains_key(sym) {
                    continue;
                }

                // Meme position cap — separate from global max_positions
                if cand.tier == "meme" {
                    let meme_max = self.config.env.get_f64("MEME_MAX_POSITIONS", 2.0) as usize;
                    let meme_count = self.positions.keys()
                        .filter(|s| config::coin_tier(s) == "meme")
                        .count();
                    if meme_count >= meme_max {
                        self.record_entry_reject(now, sym,
                            &format!("meme_cap_reached {}/{}", meme_count, meme_max), false);
                        continue;
                    }
                }

                // ── Portfolio balance check (Algorithm 7) ──
                if !self.positions.is_empty() && self.portfolio_value > 0.0 {
                    let num_pos = self.positions.len().max(1) as f64;
                    let target_alloc = 1.0 / (num_pos + 1.0);
                    let current_sym_value = self.positions.get(sym)
                        .map(|p| p.qty * price)
                        .unwrap_or(0.0);
                    let current_alloc = current_sym_value / self.portfolio_value;
                    let score7d = feats.get("weighted_score").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    if current_alloc > target_alloc * 1.5 && score7d < 0.0 {
                        self.record_entry_reject(
                            now, sym,
                            &format!("overweight_bearish alloc={:.1}%>{:.1}% score={:.3}",
                                current_alloc * 100.0, target_alloc * 150.0, score7d),
                            false,
                        );
                        continue;
                    }
                }

                // Global entry gates
                if file_kill_active {
                    self.record_entry_reject(now, sym, "file_kill_switch", true);
                    continue;
                }
                if self.circuit.exit_only {
                    self.record_entry_reject(now, sym,
                        &format!("circuit_{}", self.circuit.level.as_str()), true);
                    continue;
                }
                if self.watchdog_block_entries {
                    self.record_entry_reject(now, sym,
                        &format!("watchdog_unhealthy failures={}", self.watchdog_fail_count), true);
                    continue;
                }
                if self.hardware_block_entries {
                    self.record_entry_reject(now, sym,
                        &format!("hardware_guard level={}", self.hw_guard_level.as_str()), true);
                    continue;
                }
                if self.pending_entries.values().any(|p| p.symbol == *sym) {
                    self.record_entry_reject(now, sym, "pending_entry_exists", false);
                    continue;
                }

                // Cooldown + max positions
                let cooldown = cand.profile.cooldown_sec;
                if cooldown > 0 && (now - self.last_entry_ts) < cooldown as f64 {
                    continue;
                }
                if (self.positions.len() + self.pending_entries.len()) >= cand.profile.max_positions {
                    continue;
                }

                // Cash guard
                let cash_reserve_floor = self.config.env.get_f64("CASH_RESERVE_FLOOR", 2.0);
                let cash_reserve_pct = self.config.env.get_f64("CASH_RESERVE_PCT", 0.10);
                let cash_reserve = (self.portfolio_value * cash_reserve_pct).max(cash_reserve_floor);
                let min_entry_usd = self.config.base_usd.max(5.0) + cash_reserve;
                if self.available_usd < min_entry_usd {
                    break;
                }

                // Micro-price EMA confirmation for L2 (AI suggestion #3)
                // L2 BUY needs price near or below EMA20 — confirms mean-reversion entry
                let e_is_l2_pre = feats.get("is_l2").and_then(|v| v.as_bool()).unwrap_or(false);
                if e_is_l2_pre && ai_decision.action == "BUY" {
                    let ema21 = feats.get("ema21").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    if ema21 > 0.0 && price > ema21 * 1.01 {
                        // Price is >0.5% above EMA21 — not a good mean-reversion entry
                        self.record_entry_reject(now, sym, "l2_above_ema21", false);
                        continue;
                    }
                }

                // ── NPU Pre-Scan: fast lane check BEFORE GPU AI_1 sees this coin ──
                // AI_3 on Intel NPU (~200ms) classifies lane from raw indicators.
                // Only coins that PASS (valid lane + BUY action) reach AI_1 on GPU.
                let npu_verify_enabled = self.config.env.get_bool("NPU_VERIFY_ENABLED", false);
                // ── Phi-3 NPU scan: ADVISORY, non-blocking — fires in background, uses cached result ──
                // phi3_server.py takes ~5s on NPU. We spawn the task and use last cached verdict
                // so the tick loop never waits for Phi-3. Nemotron sees the opinion when available.
                let npu_advisory: Option<String> = if npu_verify_enabled {
                    // Check cache from previous scan (stored in npu_scan_cache on TradingLoop)
                    let cached = self.npu_scan_cache.get(sym).cloned();
                    // Spawn a fresh scan in the background — result will be in cache next tick
                    let feats_clone = feats.clone();
                    let sym_owned = sym.to_string();
                    let cache_tx = self.npu_scan_tx.clone();
                    tokio::spawn(async move {
                        let scan = crate::ai_bridge::npu_scan_coin(&feats_clone).await;
                        tracing::debug!(
                            "[NPU-SCAN] {} verdict={} lane={} action={} | {} ({}ms)",
                            sym_owned, scan.verdict, scan.lane, scan.recomputed_action, scan.explanation, scan.latency_ms
                        );
                        let _ = cache_tx.send((sym_owned, scan.verdict, scan.lane, scan.recomputed_action, scan.explanation));
                    });
                    cached
                } else {
                    None
                };

                // Build entry prompt for batch
                let e_trend = feats.get("trend_score").and_then(|v| v.as_f64()).unwrap_or(0.0) as i32;
                let e_lane = classify_lane(feats, price);

                let mut feats_entry = feats.clone();
                if let Some(obj) = feats_entry.as_object_mut() {
                    let cvar_w = self.cvar_weights.get(sym).copied().unwrap_or(0.0);
                    obj.insert("cvar_weight".to_string(), json!(cvar_w));
                    obj.insert("lane_tag".to_string(), json!(e_lane.as_str()));
                }
                let e_is_l2 = feats.get("is_l2").and_then(|v| v.as_bool()).unwrap_or(false);
                let e_mr_buy = feats.get("mean_rev_buy").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let e_mr_sell = feats.get("mean_rev_sell").and_then(|v| v.as_f64()).unwrap_or(0.0);

                let e_kelly = self.journal.kelly_fraction(sym);

                // MATRIX_ENTRY_V2: lean pipe-delimited signal per coin
                let e_rsi = feats.get("rsi").and_then(|v| v.as_f64()).unwrap_or(50.0);
                let e_mom = feats.get("momentum_score").and_then(|v| v.as_f64()).unwrap_or(0.0);

                // Read MTF trend alignment from mover scanner cache
                let (e_trend_7d, e_trend_30d, e_mtf_align) = {
                    let mtf = std::fs::read_to_string("data/mtf_trends.json")
                        .ok()
                        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
                        .and_then(|v| v.get(sym).or_else(|| v.get(&sym.to_uppercase())).cloned());
                    let t7d  = mtf.as_ref().and_then(|v| v.get("trend_7d").and_then(|x| x.as_f64())).unwrap_or(0.0);
                    let t30d = mtf.as_ref().and_then(|v| v.get("trend_30d").and_then(|x| x.as_f64())).unwrap_or(0.0);
                    let align = crate::engine::entries::read_mtf_alignment(sym);
                    (t7d, t30d, align)
                };
                // ── Hard MTF Gate: block entries in confirmed weekly downtrend ──
                // Only fires when we have real data (non-zero). Hot movers exempt.
                let mtf_gate_min = self.config.env.get_f64("MTF_HARD_GATE_7D", -5.0);
                let e_vol_ratio = feats.get("vol_ratio").and_then(|v| v.as_f64()).unwrap_or(1.0);
                let is_hot_mover = e_vol_ratio >= 3.0 || pump_confirmed;
                if e_trend_7d != 0.0 && e_trend_7d < mtf_gate_min && !is_hot_mover {
                    tracing::info!("[MTF-GATE] {} BLOCKED — 7d={:.1}% < {:.1}% threshold | vol={:.1}x",
                        sym, e_trend_7d, mtf_gate_min, e_vol_ratio);
                    self.record_entry_reject(now, sym, "mtf_bearish_7d", false);
                    continue;
                }

                // MTF alignment label for AI context
                let mtf_label = match e_mtf_align {
                    3 => format!("|MTF:ALL({:+.0}%7d,{:+.0}%30d)", e_trend_7d, e_trend_30d),
                    2 => format!("|MTF:1d+7d({:+.0}%7d)", e_trend_7d),
                    1 => "|MTF:1d_only".to_string(),
                    _ => String::new(),
                };
                let mut entry_prompt = format!(
                    "{}|${:.0}|{}|{}|R{:.0}|K{:.2}|M{:+.2}|T{:+}{}|FEE:0.52%RT",
                    sym, price, cand.regime_str, e_lane,
                    e_rsi, e_kelly, e_mom, e_trend, mtf_label,
                );
                if !cand.gate_warnings.is_empty() {
                    let adv = format_advisories(&cand.gate_warnings);
                    entry_prompt.push_str(&format!("|{}", adv));
                }
                if is_nemo_override {
                    entry_prompt.push_str("|OVR");
                }
                // Pump detector labels — AI sees these and knows to act fast
                if pump_confirmed {
                    entry_prompt.push_str(&format!("|PUMP_CONFIRMED(score={:.1})", pump_score));
                } else if pump_early {
                    entry_prompt.push_str(&format!("|PUMP_EARLY(score={:.1})", pump_score));
                }
                if let Some(ref adv) = npu_advisory {
                    entry_prompt.push_str(&format!("|{}", adv));
                }
                if e_is_l2 {
                    entry_prompt.push_str(&format!("|MR{:.2}/{:.2}", e_mr_buy, e_mr_sell));
                }

                // Inject AI working memory (last N decisions for this coin)
                if memory_enabled {
                    if let Some(mem) = self.nemo_memory.get(sym) {
                        if !mem.is_empty() {
                            entry_prompt.push('\n');
                            entry_prompt.push_str(&format_nemo_memory(mem, price));
                        }
                    }
                }

                // Inject news sentiment (Fear/Greed, market cap, trending — free APIs)
                {
                    let news_block = self.news_sentiment.build_news_block();
                    if !news_block.is_empty() {
                        entry_prompt.push('\n');
                        entry_prompt.push_str(&news_block);
                    }
                }

                // Inject Captain's sentiment + price context (AI3 NPU)
                if let Some(ref ci) = self.cloud_intel {
                    if let Ok(state) = ci.try_read() {
                        let cloud_lines = crate::cloud_intel::build_cloud_prompt(&state, sym, -0.3);
                        entry_prompt.push_str(&cloud_lines);
                    }
                }

                // ── HF: Compute Bayesian quality score for candidate ranking ──
                let ql_ema21 = feats.get("ema21").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let ql_ema55 = feats.get("ema55").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let ql_ema200 = feats.get("ema200").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let ql_votes = (ql_ema21 > 0.0 && price > ql_ema21) as i32
                    + (ql_ema55 > 0.0 && price > ql_ema55) as i32
                    + (ql_ema200 > 0.0 && price > ql_ema200) as i32;
                let ql_imb_avg = self.book_imb_hist.get(sym)
                    .map(|q| if q.is_empty() { 0.0 } else { q.iter().copied().sum::<f64>() / q.len() as f64 })
                    .unwrap_or(0.0);
                let ql_imb_tag = if ql_imb_avg >= 0.12 { "strong" } else { "weak" };
                let ql_key = format!(
                    "reg={}|lane={}|tier={}|mtf={}|imb={}",
                    cand.regime_str, e_lane, cand.tier, ql_votes, ql_imb_tag
                );
                let ql_prior_a = self.config.env.get_f64("HF_QL_PRIOR_A", 2.0);
                let ql_prior_b = self.config.env.get_f64("HF_QL_PRIOR_B", 2.0);
                let ql_min_samples = self.config.env.get_f64("HF_QL_MIN_SAMPLES", 12.0);
                let ql_min_pwin = self.config.env.get_f64("HF_QL_MIN_PWIN", 0.40);
                let ql_max_pwin = self.config.env.get_f64("HF_QL_MAX_PWIN", 0.70);
                let ql_cost_k = self.config.env.get_f64("HF_QL_COST_K", 35.0);
                let ql_stats = self.quality_stats.get(&ql_key)
                    .copied()
                    .unwrap_or(BetaStats::new(ql_prior_a, ql_prior_b));
                let ql_raw_pwin = ql_stats.mean();
                // Blend with 0.50 if not enough samples
                let ql_blend = if ql_stats.samples() < ql_min_samples + ql_prior_a + ql_prior_b {
                    let w = (ql_stats.samples() - ql_prior_a - ql_prior_b).max(0.0) / ql_min_samples;
                    ql_raw_pwin * w + 0.50 * (1.0 - w)
                } else {
                    ql_raw_pwin
                };
                let ql_pwin = ql_blend.clamp(ql_min_pwin, ql_max_pwin);
                let ql_spread = feats.get("spread_pct").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let ql_fee_cost = 0.0052 + ql_spread / 100.0; // ~0.52% RT fees + spread
                let ql_cost_adj = (1.0 - ql_fee_cost * ql_cost_k).clamp(0.6, 1.0);
                let ql_score = ai_decision.confidence * ql_pwin * ql_cost_adj;

                entry_candidates.push(EntryCandidate {
                    sym: sym.to_string(),
                    price,
                    regime_str: cand.regime_str.clone(),
                    tier: cand.tier,
                    profile: cand.profile.clone(),
                    ai_decision,
                    feats_entry,
                    entry_prompt,
                    quality_key: ql_key,
                    quality_score: ql_score,
                    is_nemo_override,
                });
            }

            // ── HF: Rank candidates by quality score (best setups first) ──
            entry_candidates.sort_by(|a, b| b.quality_score.partial_cmp(&a.quality_score).unwrap_or(std::cmp::Ordering::Equal));
            let max_cands = self.config.env.get_f64("AI_MAX_CANDIDATES_PER_TICK", 5.0) as usize;
            if entry_candidates.len() > max_cands {
                tracing::info!(
                    "[HF-QL] truncating {} → {} candidates by quality",
                    entry_candidates.len(), max_cands
                );
                entry_candidates.truncate(max_cands);
            }
            if entry_candidates.len() > 1 {
                let scores: Vec<String> = entry_candidates.iter()
                    .map(|ec| format!("{}={:.3}", ec.sym, ec.quality_score))
                    .collect();
                tracing::info!("[HF-QL] ranked: {}", scores.join(" > "));
            }
            if self.tick_count.is_multiple_of(5) {
                let mut items: Vec<(&&str, &usize)> = entry_reject_counts.iter().filter(|(_, v)| **v > 0).collect();
                items.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
                let top: String = items
                    .into_iter()
                    .take(5)
                    .map(|(k, v)| format!("{k}={v}"))
                    .collect::<Vec<String>>()
                    .join(", ");
                if !top.is_empty() {
                    tracing::info!(
                        "[ENTRY-REJECT] top={}",
                        top
                    );
                }
                tracing::info!(
                    "[ENTRY-SUMMARY] candidates={} entry_candidates={}",
                    candidates.len(),
                    entry_candidates.len()
                );
            }

            // ══════════════════════════════════════════════════════
            //  PHASE B: AI_1 batch confirmation (final authority)
            //  Only NPU-PASS'd candidates reach here.
            //  AI_1 sees: market context + memory + per-coin prompt
            //  BUY = confirmed → execute. HOLD = vetoed → skip.
            // ══════════════════════════════════════════════════════
            if !entry_candidates.is_empty() {
                let nvidia_cfg = crate::nvidia_tools::NvidiaToolsConfig::from_env();
                // Prepend market context + funding + memory
                let market_block = self.market_brain.build_market_slim();
                let watch_block = self.market_watcher.build_watch_block();
                let memory_block = if memory_enabled {
                    self.nemo_memory_brain.build_memory_block()
                } else {
                    String::new()
                };
                let coach_block = {
                    let p = std::path::Path::new("data/nemo_coaching_notes.txt");
                    match std::fs::read_to_string(p) {
                        Ok(s) if !s.trim().is_empty() => format!("=== COACH RULES ===\n{}\n", s.trim()),
                        _ => String::new(),
                    }
                };
                let macro_block = {
                    let path = std::path::Path::new("data/macro_note.txt");
                    if path.exists() {
                        match std::fs::read_to_string(path) {
                            Ok(s) if !s.trim().is_empty() => format!("=== MACRO INTEL ===\n{}\n", s.trim()),
                            _ => String::new(),
                        }
                    } else { String::new() }
                };
                let funding_block = {
                    let snap = self.shared_funding.read().await;
                    let syms: Vec<String> = entry_candidates.iter().map(|ec| ec.sym.clone()).collect();
                    crate::market_brain::build_funding_block(&snap, &syms)
                };
                let context_prefix = if market_block.is_empty()
                    && watch_block.is_empty()
                    && memory_block.is_empty() && funding_block.is_empty()
                    && coach_block.is_empty() && macro_block.is_empty()
                {
                    String::new()
                } else {
                    format!(
                        "{}\n{}\n{}\n{}\n{}\n{}\n",
                        market_block, watch_block, funding_block, memory_block, coach_block, macro_block
                    )
                };

                let mut batch_prompts: Vec<(String, String)> = Vec::with_capacity(entry_candidates.len());
                for ec in &entry_candidates {
                    let mut full_prompt = if context_prefix.is_empty() {
                        ec.entry_prompt.clone()
                    } else {
                        format!("{}{}", context_prefix, ec.entry_prompt)
                    };

                    // NVIDIA semantic memory: find similar past trades for this candidate
                    if nvidia_cfg.embed_enabled && self.nemo_memory_brain.embedding_store.len() > 0 {
                        let desc = format!(
                            "{} regime={} lane={}",
                            ec.sym, ec.regime_str, ec.ai_decision.lane,
                        );
                        let similar = self.nemo_memory_brain
                            .build_similar_trades_text(&desc, ai.client())
                            .await;
                        if !similar.is_empty() {
                            full_prompt.push('\n');
                            full_prompt.push_str(&similar);
                        }
                    }

                    // AuroraKin907 memory packet: inject adaptive learning context
                    if memory_enabled {
                        if let Some(pkt) = self.memory_packet.get() {
                            full_prompt.push('\n');
                            full_prompt.push_str(&pkt.format_for_entry(&ec.sym));
                        }
                    }

                    batch_prompts.push((ec.sym.clone(), full_prompt));
                }

                let use_entry_tools = self.config.env.get_bool("AI_ENTRY_TOOLS", false);
                let max_tool_candidates: usize = self
                    .config
                    .env
                    .get_parsed("AI_ENTRY_TOOL_MAX_CANDIDATES", 1_usize);

                let nemo_results = if use_entry_tools && batch_prompts.len() <= max_tool_candidates {
                    // Per-candidate tool calling: AI_1 investigates each candidate individually
                    let mut results = std::collections::HashMap::new();
                    for (sym, prompt) in &batch_prompts {
                        let decision = self.get_entry_confirmation_with_tools(
                            sym, prompt, features_map, ai,
                        ).await;
                        results.insert(sym.clone(), decision);
                    }
                    results
                } else {
                    if use_entry_tools && batch_prompts.len() > max_tool_candidates {
                        tracing::info!(
                            "[AI-ENTRY-TOOL] skipping tool-calls: candidates={} > max_tool_candidates={}",
                            batch_prompts.len(),
                            max_tool_candidates
                        );
                    }
                    // Original batch mode: all candidates in one shot
                    ai.get_batch_entry_confirmation(&batch_prompts).await
                };

                // ── PHASE C: Execute entries on Kraken ──
                // QUANT_PRIMARY=1: quant score is primary, AI_1 adjusts confidence weight
                // QUANT_PRIMARY=0: AI_1 is final authority (legacy behavior)
                let quant_primary = self.config.env.get_bool("QUANT_PRIMARY", false);

                for mut ec in entry_candidates {
                    let nemo_confirm = nemo_results.get(&ec.sym)
                        .cloned()
                        .unwrap_or_default();

                    // Log AI_1 decision for 49B optimizer to read
                    crate::engine::decision_log::log_ai1(&crate::engine::decision_log::Ai1Decision {
                        ts:          now,
                        symbol:      ec.sym.clone(),
                        action:      nemo_confirm.action.clone(),
                        confidence:  nemo_confirm.confidence,
                        regime:      ec.regime_str.clone(),
                        lane:        ec.tier.to_string(),
                        tools_used:  vec![],  // populated when tool-calling is used
                        tool_rounds: 0,
                        reason:      nemo_confirm.reason.clone(),
                        source:      nemo_confirm.source.clone(),
                        latency_ms:  nemo_confirm.latency_ms,
                    });

                    let math_action_orig = ec.ai_decision.action.clone();
                    let math_conf_orig = ec.ai_decision.confidence;

                    if quant_primary {
                        // ═══ QUANT-PRIMARY MODE ═══
                        // Quant score is the primary decision. AI_1 provides confidence weight adjustment.
                        // BUY from quant is required. HOLD with is_nemo_override allowed (strong features + AI says BUY).
                        if ec.ai_decision.action != "BUY" {
                            let skip = if ec.is_nemo_override && ec.ai_decision.action == "HOLD" && nemo_confirm.action == "BUY" {
                                // Override: math=HOLD but strong signals + AI=BUY → promote to BUY
                                ec.ai_decision.action = "BUY".to_string();
                                tracing::info!(
                                    "[QUANT-OVERRIDE] {} math=HOLD but strong features + AI=BUY({:.2}) → promoting to BUY",
                                    ec.sym, nemo_confirm.confidence
                                );
                                false
                            } else {
                                true
                            };
                            if skip {
                                tracing::debug!(
                                    "[QUANT-SKIP] {} math={}({:.2}) quant_primary requires BUY — skipping",
                                    ec.sym, math_action_orig, math_conf_orig
                                );
                                continue;
                            }
                        }

                        let qwen_weight = if nemo_confirm.action == "BUY" {
                            // AI_1 agrees: weight = 1.0 + (conf - 0.5) * 0.3
                            1.0 + (nemo_confirm.confidence - 0.5) * 0.3
                        } else {
                            // AI_1 says HOLD: reduce by 30%
                            0.7
                        };
                        let quant_conf = ec.ai_decision.confidence;
                        let final_conf = (quant_conf * qwen_weight).clamp(0.0, 1.0);

                        // Must pass schedule-adjusted margin gate
                        let margin = (final_conf - (1.0 - final_conf)).abs();
                        if margin < schedule.margin_gate {
                            self.record_entry_reject(
                                now, &ec.sym,
                                &format!("quant_margin_gate margin={:.2}<{:.2} (sched={})",
                                    margin, schedule.margin_gate, schedule.block_name),
                                false,
                            );
                            continue;
                        }

                        ec.ai_decision.confidence = final_conf;
                        self.record_nemo_memory(
                            &ec.sym, "BUY", final_conf,
                            &format!("quant_primary w={:.2} ai={}/{:.2}",
                                qwen_weight, nemo_confirm.action, nemo_confirm.confidence),
                            ec.price, &ec.ai_decision.lane, "quant_primary",
                        );
                        tracing::info!(
                            "[QUANT-PRIMARY] {} quant={:.2} ai={}({:.2}) weight={:.2} → final={:.2} ({}ms)",
                            ec.sym, quant_conf, nemo_confirm.action, nemo_confirm.confidence,
                            qwen_weight, final_conf, nemo_confirm.latency_ms,
                        );
                    } else {
                        // ═══ LEGACY MODE: AI_1 is final authority ═══
                        if nemo_confirm.action != "BUY" && nemo_confirm.action != "HOLD" {
                            self.record_nemo_memory(&ec.sym, &nemo_confirm.action, nemo_confirm.confidence, &nemo_confirm.reason, ec.price, &ec.ai_decision.lane, "ai_veto");
                            self.record_entry_reject(
                                now, &ec.sym,
                                &format!("ai_veto conf={:.2} reason={}", nemo_confirm.confidence, nemo_confirm.reason),
                                false,
                            );
                            continue;
                        }
                        if nemo_confirm.action == "HOLD" {
                            tracing::info!(
                                "[AI-HOLD] {} conf={:.2} | {} ({}ms)",
                                ec.sym, nemo_confirm.confidence, nemo_confirm.reason, nemo_confirm.latency_ms
                            );
                            continue;
                        }
                        let was_nemo_override = ec.ai_decision.action == "HOLD";

                        if was_nemo_override {
                            ec.ai_decision.confidence = nemo_confirm.confidence;
                            self.record_nemo_memory(&ec.sym, "BUY", nemo_confirm.confidence, &nemo_confirm.reason, ec.price, &ec.ai_decision.lane, "ai_override_buy");
                            tracing::info!(
                                "[AI-OVERRIDE-BUY] {} math=HOLD→AI=BUY conf={:.2} | {} ({}ms)",
                                ec.sym, nemo_confirm.confidence, nemo_confirm.reason, nemo_confirm.latency_ms
                            );
                        } else {
                            self.record_nemo_memory(&ec.sym, "BUY", nemo_confirm.confidence, &nemo_confirm.reason, ec.price, &ec.ai_decision.lane, "ai_confirm");
                            tracing::info!(
                                "[AI-CONFIRM] {} BUY conf={:.2} | {} ({}ms)",
                                ec.sym, nemo_confirm.confidence, nemo_confirm.reason, nemo_confirm.latency_ms
                            );
                        }
                    }

                    // ── Confidence cap before sizing ──
                    ec.ai_decision.confidence = ec.ai_decision.confidence.min(self.config.max_confidence_for_sizing);

                    // Record trade timestamp + price for cooldowns
                    self.last_trade_ts.insert(ec.sym.clone(), now);
                    self.last_entry_price.insert(ec.sym.clone(), ec.price);

                    let mut signal_names = vec![
                        format!("math:{}:{:.2}", math_action_orig, math_conf_orig),
                        format!("ai:{}:{:.2}", nemo_confirm.action, nemo_confirm.confidence),
                        format!("regime:{}", ec.regime_str),
                    ];
                    if quant_primary {
                        signal_names.push("quant_primary".to_string());
                    }

                    self.execute_entry(
                        &ec.sym,
                        ec.price,
                        &ec.feats_entry,
                        &ec.ai_decision,
                        &ec.profile,
                        &ec.regime_str,
                        ec.tier,
                        &signal_names,
                        api,
                        &ec.quality_key,
                        ec.quality_score,
                    )
                    .await;
                }
            } else {
                tracing::info!("[ENTRY-SKIP] No entry candidates after gates");
            }
        }

    }
}
