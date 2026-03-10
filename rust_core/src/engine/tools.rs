//! AI tool implementations — methods called by the LLM tool-calling loop.

use super::*;

impl super::TradingLoop {
    /// Tool: get_top_movers — search full Kraken universe (500+ coins) for top movers.
    pub(crate) async fn tool_get_top_movers(&self) -> String {
        const HIGH_EDGE: &[&str] = &["GRT", "CRV", "ARB", "LTC", "SAND", "ONDO", "ATOM", "NEAR"];
        const HIGH_EDGE_PF: &[(&str, f64)] = &[
            ("GRT", 18.7), ("CRV", 5.6), ("ARB", 5.2), ("LTC", 2.7),
            ("SAND", 1.7), ("ONDO", 1.4), ("ATOM", 1.3), ("NEAR", 1.3),
        ];

        let movers = crate::kraken_api::get_top_movers_universe(&self.http_client, 20).await;
        if movers.is_empty() {
            return "TOP MOVERS: (unavailable — API error)".to_string();
        }

        // Show which coins are already tracked
        let tracked: Vec<String> = self.config.env.get_str("SYMBOLS", "")
            .split(',')
            .map(|s| s.trim().to_uppercase())
            .filter(|s| !s.is_empty())
            .collect();

        let mut out = format!("TOP MOVERS — FULL KRAKEN UNIVERSE ({} results, vol>$50K):\n", movers.len());
        for (i, m) in movers.iter().enumerate() {
            let vol_str = if m.volume_usd >= 1_000_000.0 {
                format!("${:.1}M", m.volume_usd / 1_000_000.0)
            } else {
                format!("${:.0}K", m.volume_usd / 1_000.0)
            };
            let edge_tag = if HIGH_EDGE.contains(&m.symbol.as_str()) {
                let pf = HIGH_EDGE_PF.iter()
                    .find(|(s, _)| *s == m.symbol)
                    .map(|(_, p)| *p)
                    .unwrap_or(0.0);
                format!("  *** HIGH EDGE PF={:.1} ***", pf)
            } else {
                String::new()
            };
            let tracked_tag = if tracked.contains(&m.symbol) { " [TRACKED]" } else { "" };
            out.push_str(&format!(
                " {:2}. {:6} {:+.1}%  ${:.4}  vol={}  edge={:.2}  BE=${:.4}{}{}\n",
                i + 1, m.symbol, m.change_24h_pct, m.price, vol_str,
                m.edge_score, m.break_even_exit, edge_tag, tracked_tag
            ));
        }
        out.push_str(&format!(
            "\nCurrently tracking {} coins. SWAP ONLY: to add a coin, you MUST remove one first.",
            tracked.len()
        ));
        out
    }

    pub(crate) fn tool_get_coin_features(
        &self,
        args: &serde_json::Value,
        features_map: &HashMap<String, serde_json::Value>,
    ) -> String {
        let symbol = args
            .get("symbol")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_uppercase();
        match features_map.get(&symbol) {
            Some(feats) => {
                let f = |key: &str| -> String {
                    feats
                        .get(key)
                        .map(|v| format!("{}", v))
                        .unwrap_or_else(|| "null".to_string())
                };
                format!(
                    "FEATURES {}:\n\
                     price={} rsi={} trend_score={} momentum_score={}\n\
                     vol_ratio={} book_imbalance={} buy_ratio={} spread_pct={}\n\
                     zscore={} atr={} macd_hist={}\n\
                     ema9={} ema21={} ema50={}\n\
                     bb_upper={} bb_lower={}\n\
                     hurst={} entropy={} adx={} autocorr={}\n\
                     quant_vol={} quant_momo={} quant_regime={} quant_meanrev={}\n\
                     quant_liquidity={} quant_risk={} quant_corr_btc={} quant_corr_eth={}\n\
                     s_buy={} s_sell={} s_hold={}\n\
                     flow_imbalance={} book_trend={} book_strength={}",
                    symbol,
                    f("price"), f("rsi"), f("trend_score"), f("momentum_score"),
                    f("vol_ratio"), f("book_imbalance"), f("buy_ratio"), f("spread_pct"),
                    f("zscore"), f("atr"), f("macd_hist"),
                    f("ema9"), f("ema21"), f("ema50"),
                    f("bb_upper"), f("bb_lower"),
                    f("hurst_exp"), f("shannon_entropy"), f("adx"), f("autocorr_lag1"),
                    f("quant_vol"), f("quant_momo"), f("quant_regime"), f("quant_meanrev"),
                    f("quant_liquidity"), f("quant_risk"), f("quant_corr_btc"), f("quant_corr_eth"),
                    f("s_buy"), f("s_sell"), f("s_hold"),
                    f("flow_imbalance"), f("book_trend"), f("book_strength"),
                )
            }
            None => format!("No features available for {}", symbol),
        }
    }

    pub(crate) fn tool_get_trade_history(&self, args: &serde_json::Value) -> String {
        let symbol = args
            .get("symbol")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_uppercase();
        let limit = args
            .get("limit")
            .and_then(|v| v.as_u64())
            .unwrap_or(5)
            .min(10) as usize;

        let trades = self.journal.query_trades(Some(&symbol), None, limit);
        if trades.is_empty() {
            return format!("No trade history for {}", symbol);
        }
        let mut out = format!("TRADE HISTORY {} (last {}):\n", symbol, trades.len());
        for t in &trades {
            out.push_str(&format!(
                "  {} {:+.2}% ${:+.2} hold={:.0}min exit={}\n",
                t.result,
                t.pnl_percent * 100.0,
                t.pnl_usd,
                t.hold_minutes,
                t.exit_reason,
            ));
        }
        out.push_str(&format!("STATS: {}", self.journal.coin_stats(&symbol)));
        out
    }

    pub(crate) fn tool_get_market_context(&self) -> String {
        let market_block = self.market_brain.build_market_block();
        if market_block.is_empty() {
            return "Market context not yet available (warming up)".to_string();
        }
        let mut out = market_block;
        if let Some(state) = self.market_brain.state() {
            out.push_str("\nSECTORS:\n");
            for s in &state.sectors {
                out.push_str(&format!(
                    "  {} ({}): mom={:+.3} trend={:+.1} vol={:.1}x green={:.0}%\n",
                    s.name, s.coin_count, s.avg_momentum, s.avg_trend,
                    s.avg_vol_ratio, s.green_pct * 100.0,
                ));
            }
        }
        out
    }

    pub(crate) fn tool_get_nemo_memory(&self, args: &serde_json::Value) -> String {
        let symbol = args
            .get("symbol")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_uppercase();

        let mut out = String::new();

        // Per-coin track record from nemo_memory_brain
        if let Some(tr) = self.nemo_memory_brain.track_record() {
            for cr in &tr.per_coin {
                if cr.symbol.eq_ignore_ascii_case(&symbol) {
                    out.push_str(&format!(
                        "TRACK RECORD {}: {} trades, {}W/{}L, avg PnL {:+.2}%",
                        cr.symbol, cr.trade_count, cr.win_count, cr.loss_count,
                        cr.avg_pnl_pct * 100.0,
                    ));
                    if !cr.comment.is_empty() {
                        out.push_str(&format!(" -- {}", cr.comment));
                    }
                    out.push('\n');
                    break;
                }
            }
        }

        // Recent working memory decisions
        if let Some(entries) = self.nemo_memory.get(symbol.as_str()) {
            if !entries.is_empty() {
                out.push_str("RECENT DECISIONS:\n");
                let now = now_ts();
                for e in entries.iter().rev().take(8) {
                    let age_min = ((now - e.ts) / 60.0).round() as i64;
                    out.push_str(&format!(
                        "  {}m ago: {} conf={:.2} src={} | {}\n",
                        age_min, e.action, e.confidence, e.source, e.reason,
                    ));
                }
            }
        }

        // Episodes (big wins/losses/panic sells)
        let episodes = self.nemo_memory_brain.recent_episodes_for(&symbol, 3);
        if !episodes.is_empty() {
            out.push_str("EPISODES:\n");
            for ep in &episodes {
                out.push_str(&format!("  {}\n", ep));
            }
        }

        if out.is_empty() {
            format!("No memory for {}", symbol)
        } else {
            out
        }
    }

    pub(crate) fn tool_get_correlated_coins(
        &self,
        args: &serde_json::Value,
        features_map: &HashMap<String, serde_json::Value>,
    ) -> String {
        let symbol = args
            .get("symbol")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_uppercase();

        let my_tier = crate::config::coin_tier(&symbol);

        let mut peers: Vec<(String, f64, i64, f64, f64)> = Vec::new();
        for (sym, feats) in features_map {
            if sym.eq_ignore_ascii_case(&symbol) {
                continue;
            }
            let peer_tier = crate::config::coin_tier(sym);
            if peer_tier == my_tier {
                let momentum = feats
                    .get("momentum_score")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0);
                let trend = feats
                    .get("trend_score")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(0);
                let rsi = feats.get("rsi").and_then(|v| v.as_f64()).unwrap_or(50.0);
                let vol = feats
                    .get("vol_ratio")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(1.0);
                peers.push((sym.clone(), momentum, trend, rsi, vol));
            }
        }

        if peers.is_empty() {
            return format!("{} is tier '{}' — no other coins in this tier", symbol, my_tier);
        }

        peers.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut out = format!(
            "TIER '{}' peers for {} ({} coins):\n",
            my_tier,
            symbol,
            peers.len()
        );
        for (sym, mom, trend, rsi, vol) in &peers {
            out.push_str(&format!(
                "  {}: mom={:+.3} trend={:+} rsi={:.0} vol={:.1}x\n",
                sym, mom, trend, rsi, vol,
            ));
        }

        let avg_mom: f64 = peers.iter().map(|p| p.1).sum::<f64>() / peers.len() as f64;
        let green = peers.iter().filter(|p| p.1 > 0.0).count();
        out.push_str(&format!(
            "  TIER AVG: mom={:+.3}, {}/{} green",
            avg_mom, green, peers.len()
        ));
        out
    }

    /// Tool: get_engine_status — Nemo can see her own engine health and performance.
    pub(crate) fn tool_get_engine_status(&self) -> String {
        let mut out = String::with_capacity(1024);

        // Portfolio
        out.push_str(&format!(
            "ENGINE STATUS (tick {}):\n\
             Portfolio: ${:.2} equity | ${:.2} cash\n\
             Positions: {} open | {} pending\n",
            self.tick_count,
            self.equity,
            self.available_usd,
            self.positions.len(),
            self.pending_entries.len(),
        ));

        // Circuit breaker
        out.push_str(&format!(
            "Circuit: daily_pnl=${:+.2} unrealized=${:+.2} {}{}\n",
            self.circuit.daily_pnl,
            self.circuit.unrealized_pnl,
            if self.circuit.stop_trading { "STOPPED " } else { "" },
            if self.circuit.force_defensive { "DEFENSIVE" } else { "NORMAL" },
        ));

        // Watchdog
        out.push_str(&format!(
            "Watchdog: {} (fails={}){}",
            if self.watchdog_ok { "OK" } else { "FAILING" },
            self.watchdog_fail_count,
            if self.watchdog_block_entries { " BLOCKING" } else { "" },
        ));
        if !self.watchdog_last_fail_paths.is_empty() {
            out.push_str(&format!(" paths={:?}", self.watchdog_last_fail_paths));
        }
        out.push('\n');

        // 60s meters
        let m = &self.meters;
        out.push_str(&format!(
            "60s Meters: {} AI calls | {} buy | {} hold | {} sell | {} rejects | {} exit-checks\n",
            m.ai_calls, m.ai_action_buy, m.ai_action_hold, m.ai_action_sell,
            m.entry_reject_total, m.nemo_exit_checks,
        ));

        // Top reject reasons
        if !m.entry_rejects.is_empty() {
            let mut rejects: Vec<_> = m.entry_rejects.iter().collect();
            rejects.sort_by(|a, b| b.1.cmp(a.1));
            out.push_str("Top rejects: ");
            for (i, (reason, count)) in rejects.iter().take(5).enumerate() {
                if i > 0 { out.push_str(", "); }
                out.push_str(&format!("{}({})", reason, count));
            }
            out.push('\n');
        }

        // Hardware
        out.push_str(&format!(
            "Hardware: {} | AI latency p95={:.0}ms | parallel={}\n",
            if self.hardware_block_entries { "BLOCKED" } else { "OK" },
            self.ai_latency_p95_ms,
            self.effective_ai_parallel,
        ));

        // Holdings from Kraken
        if !self.holdings.is_empty() {
            out.push_str("Kraken holdings: ");
            for (sym, qty) in &self.holdings {
                out.push_str(&format!("{}={:.4} ", sym, qty));
            }
            out.push('\n');
        }

        out
    }

    // ── New tools recommended by 49B optimizer ───────────────────────────

    /// Tool: get_volatility_regime — current ATR vs historical baseline.
    /// Returns: EXPANDING / CONTRACTING / NORMAL + suggested stop multiplier.
    pub(crate) fn tool_get_volatility_regime(
        &self,
        args: &serde_json::Value,
        features_map: &HashMap<String, serde_json::Value>,
    ) -> String {
        let sym = args.get("symbol").and_then(|v| v.as_str()).unwrap_or("");
        let feats = match features_map.get(sym) {
            Some(f) => f,
            None => return format!("{}: no feature data", sym),
        };
        let price  = feats.get("price").and_then(|v| v.as_f64()).unwrap_or(1.0).max(0.0001);
        let atr    = feats.get("atr").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let atr_norm = atr / price;
        // vol_ratio as proxy for historical vol comparison
        let vol_ratio = feats.get("vol_ratio").and_then(|v| v.as_f64()).unwrap_or(1.0);
        let hurst    = feats.get("hurst_exp").and_then(|v| v.as_f64()).unwrap_or(0.5);

        let regime = if atr_norm > 0.025 || vol_ratio > 3.0 {
            "EXPANDING — high vol, breakout/volatile conditions"
        } else if atr_norm < 0.008 && vol_ratio < 0.5 {
            "CONTRACTING — low vol, consolidation, await breakout"
        } else {
            "NORMAL — standard conditions"
        };

        let stop_mult = if atr_norm > 0.025 { 2.5 }
                        else if atr_norm > 0.015 { 2.0 }
                        else { 1.5 };

        format!(
            "VOLATILITY REGIME for {}:\n\
             Regime: {}\n\
             ATR: {:.4}% of price\n\
             Vol ratio: {:.2}x avg\n\
             Hurst exponent: {:.3} (>0.5=trending, <0.5=reverting)\n\
             Suggested stop multiplier: {:.1}x ATR\n\
             Suggested SL: {:.2}% (ATR x {})\n",
            sym, regime,
            atr_norm * 100.0,
            vol_ratio,
            hurst,
            stop_mult,
            atr_norm * stop_mult * 100.0,
            stop_mult,
        )
    }

    /// Tool: get_position_health — assess an open position's risk state.
    pub(crate) fn tool_get_position_health(
        &self,
        args: &serde_json::Value,
        features_map: &HashMap<String, serde_json::Value>,
        now: f64,
    ) -> String {
        let sym = args.get("symbol").and_then(|v| v.as_str()).unwrap_or("");
        let pos = match self.positions.get(sym) {
            Some(p) => p,
            None => return format!("{}: no open position", sym),
        };
        let feats  = features_map.get(sym);
        let price  = feats.and_then(|f| f.get("price")).and_then(|v| v.as_f64()).unwrap_or(pos.entry_price);
        let fee    = self.config.fee_per_side_pct;
        let pnl    = pos.pnl_pct(price, fee) * 100.0;
        let peak   = ((pos.highest_price - pos.entry_price) / pos.entry_price) * 100.0;
        let dd     = pnl - peak; // drawdown from peak
        let hold_min = pos.hold_seconds() / 60.0;
        let sl_dist  = ((price - pos.sl_price) / price) * 100.0;
        let tp_dist  = ((pos.tp_price - price) / price) * 100.0;

        let health = if pnl > 1.0 && sl_dist > 1.5 { "HEALTHY — in profit, stop protected" }
                     else if pnl < -1.5 { "AT RISK — approaching stop, consider exit" }
                     else if dd < -2.0 { "DETERIORATING — drawn down from peak" }
                     else { "NEUTRAL — within normal range" };

        format!(
            "POSITION HEALTH for {}:\n\
             Status: {}\n\
             Entry: ${:.4} | Current: ${:.4}\n\
             PnL: {:+.2}% | Peak: {:+.2}% | DD from peak: {:+.2}%\n\
             Hold time: {:.0} min\n\
             Stop distance: {:.2}% below | TP distance: {:.2}% above\n\
             Lane: {} | Profile: {}\n",
            sym, health,
            pos.entry_price, price,
            pnl, peak, dd,
            hold_min,
            sl_dist, tp_dist,
            pos.entry_lane, pos.entry_profile,
        )
    }

    /// Tool: get_sentiment_score — Atlas NPU + cloud intel sentiment for a coin.
    pub(crate) fn tool_get_sentiment_score(
        &self,
        args: &serde_json::Value,
        features_map: &HashMap<String, serde_json::Value>,
    ) -> String {
        let sym = args.get("symbol").and_then(|v| v.as_str()).unwrap_or("");
        let feats = features_map.get(sym);
        // Cloud intel sentiment score injected into features by atlas pipeline
        let score = feats
            .and_then(|f| f.get("sentiment_score"))
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let label = if score >  0.30 { "BULLISH" }
                    else if score < -0.25 { "BEARISH" }
                    else { "NEUTRAL" };
        // Use build_news_block for fear/greed context
        let news_ctx = self.news_sentiment.build_news_block();
        let fng_line = news_ctx.lines().next().unwrap_or("Fear/Greed: unknown").to_string();
        format!(
            "SENTIMENT for {}:\n\
             Atlas NPU score: {:.3} ({})\n\
             Market: {}\n\
             Interpretation: {}\n",
            sym, score, label,
            fng_line,
            if score > 0.3 { "Positive news momentum — supports entry" }
            else if score < -0.25 { "Negative news — avoid entry, exit bias" }
            else { "No strong news signal" },
        )
    }

    /// Tool: get_orderbook_depth — bid/ask imbalance and liquidity for entry sizing.
    pub(crate) fn tool_get_orderbook_depth(
        &self,
        args: &serde_json::Value,
        features_map: &HashMap<String, serde_json::Value>,
    ) -> String {
        let sym = args.get("symbol").and_then(|v| v.as_str()).unwrap_or("");
        let feats = match features_map.get(sym) {
            Some(f) => f,
            None => return format!("{}: no feature data", sym),
        };
        let imb     = feats.get("book_imbalance").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let strength= feats.get("book_strength").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let trend   = feats.get("book_trend").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let buy_r   = feats.get("buy_ratio").and_then(|v| v.as_f64()).unwrap_or(0.5);
        let spread  = feats.get("spread_pct").and_then(|v| v.as_f64()).unwrap_or(0.0);

        let bias = if imb > 0.25 { "BUY PRESSURE — bids dominating" }
                   else if imb < -0.25 { "SELL PRESSURE — asks dominating" }
                   else { "BALANCED — no clear pressure" };

        format!(
            "ORDERBOOK for {}:\n\
             Imbalance: {:+.3} ({})\n\
             Book strength: {:.3}\n\
             Trend: {:+.3}\n\
             Buy ratio: {:.1}%\n\
             Spread: {:.4}%\n\
             Signal: {}\n",
            sym,
            imb, bias,
            strength,
            trend,
            buy_r * 100.0,
            spread * 100.0,
            if imb > 0.3 && strength > 0.5 { "Strong buy wall — entry favorable" }
            else if imb < -0.3 { "Sell pressure — wait or avoid" }
            else { "Neutral orderbook" },
        )
    }
}
