//! Background scanner: discovers breakout coins from the full Kraken universe
//! and suggests swaps into the vision scanner pipeline for Qwen 14B confirmation.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::SystemTime;
use tokio::sync::RwLock;

use crate::kraken_api;
use crate::vision_scanner::{CoinSwapSuggestion, SharedSwapQueue};

/// Run the mover scanner loop.  Scans the full Kraken universe every
/// `MOVER_SCAN_INTERVAL_SEC` seconds, identifies hot coins not in the
/// current watchlist, and either fast-tracks or medium-tracks them into
/// the active set without AI involvement.
pub async fn run_mover_scanner(
    active_coins: Arc<RwLock<Vec<String>>>,
    swap_queue: SharedSwapQueue,
    anchor_coins: Vec<String>,
    ws_reconnect: Arc<AtomicBool>,
) {
    let interval_sec: u64 = std::env::var("MOVER_SCAN_INTERVAL_SEC")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(300);

    let min_change_pct: f64 = std::env::var("MIN_MOVER_CHANGE_PCT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10.0);

    let min_volume_usd: f64 = std::env::var("MIN_MOVER_VOLUME_USD")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(100_000.0);

    let max_suggestions: usize = std::env::var("MAX_MOVER_SUGGESTIONS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2);

    tracing::info!(
        "[MOVER-SCAN] Started (interval={}s, min_change={}%, min_vol=${}, max_suggestions={})",
        interval_sec,
        min_change_pct,
        min_volume_usd,
        max_suggestions,
    );

    let fast_track_cooldown_sec: u64 = std::env::var("MOVER_FAST_TRACK_COOLDOWN_SEC")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(360);

    let client = reqwest::Client::new();
    let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(interval_sec));
    let mut last_fast_track_ts: u64 = 0;

    loop {
        interval.tick().await;

        // ── 1. Fetch full universe ──────────────────────────────
        let universe = kraken_api::get_top_movers_universe(&client, 200).await;
        if universe.is_empty() {
            tracing::warn!("[MOVER-SCAN] Universe scan returned 0 coins — skipping cycle");
            continue;
        }

        let current_watchlist: Vec<String> = active_coins.read().await.clone();

        // ── 1b. Enrich top candidates with 7d/30d trend data ────
        // Fetch daily OHLC for top 20 candidates by 24h change.
        // Weighted edge_score = 0.40*trend_7d + 0.30*trend_1d + 0.30*trend_30d
        let trend_7d_min:  f64 = std::env::var("TREND_7D_MIN").ok().and_then(|v| v.parse().ok()).unwrap_or(4.0);
        let trend_30d_min: f64 = std::env::var("TREND_30D_MIN").ok().and_then(|v| v.parse().ok()).unwrap_or(5.0);

        let mut enriched_universe = universe.clone();
        let top_candidates: Vec<String> = {
            let mut sorted = universe.clone();
            sorted.sort_by(|a, b| b.change_24h_pct.partial_cmp(&a.change_24h_pct).unwrap_or(std::cmp::Ordering::Equal));
            sorted.iter()
                .filter(|m| !current_watchlist.iter().any(|w| w.eq_ignore_ascii_case(&m.symbol)))
                .take(20)
                .map(|m| m.symbol.clone())
                .collect()
        };

        for sym in &top_candidates {
            let (t7d, t30d) = crate::kraken_api::fetch_daily_trend(&client, sym).await;
            if let Some(m) = enriched_universe.iter_mut().find(|m| m.symbol.eq_ignore_ascii_case(sym)) {
                m.trend_7d_pct  = t7d;
                m.trend_30d_pct = t30d;
                // Weighted multi-timeframe score
                let weighted = 0.40 * t7d + 0.30 * m.change_24h_pct + 0.30 * t30d;
                if weighted > m.edge_score {
                    m.edge_score = weighted;
                }
            }
            // Courtesy delay — Kraken public rate limit
            tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
        }

        // Also fetch MTF for current watchlist coins — they are never in top_candidates
        // so without this, all active SYMBOLS have 0.0 trend data and the MTF gate is blind.
        let watchlist_fetch: Vec<String> = current_watchlist.iter()
            .filter(|w| !top_candidates.iter().any(|c| c.eq_ignore_ascii_case(w)))
            .take(40)
            .cloned()
            .collect();

        let mut watchlist_trends: std::collections::HashMap<String, (f64, f64)> = std::collections::HashMap::new();
        for sym in &watchlist_fetch {
            let (t7d, t30d) = crate::kraken_api::fetch_daily_trend(&client, sym).await;
            if t7d != 0.0 || t30d != 0.0 {
                watchlist_trends.insert(sym.clone(), (t7d, t30d));
            }
            tokio::time::sleep(tokio::time::Duration::from_millis(300)).await;
        }

        // Write MTF trend cache — rotation candidates + current watchlist
        let mut mtf_map = serde_json::Map::new();
        // Rotation candidates
        for m in enriched_universe.iter().filter(|m| m.trend_7d_pct != 0.0 || m.trend_30d_pct != 0.0) {
            mtf_map.insert(m.symbol.clone(), serde_json::json!({
                "trend_7d":      m.trend_7d_pct,
                "trend_30d":     m.trend_30d_pct,
                "trend_7d_ok":   m.trend_7d_pct  >= trend_7d_min,
                "trend_30d_ok":  m.trend_30d_pct >= trend_30d_min,
            }));
        }
        // Active watchlist coins
        for (sym, (t7d, t30d)) in &watchlist_trends {
            mtf_map.insert(sym.clone(), serde_json::json!({
                "trend_7d":      t7d,
                "trend_30d":     t30d,
                "trend_7d_ok":   t7d  >= &trend_7d_min,
                "trend_30d_ok":  t30d >= &trend_30d_min,
            }));
        }
        let mtf_json: serde_json::Value = mtf_map.into();
        let _ = std::fs::write("data/mtf_trends.json", serde_json::to_string_pretty(&mtf_json).unwrap_or_default());
        tracing::info!("[MTF] Wrote trends for {} coins ({} watchlist + {} rotation)",
            mtf_json.as_object().map(|m| m.len()).unwrap_or(0),
            watchlist_trends.len(),
            enriched_universe.iter().filter(|m| m.trend_7d_pct != 0.0 || m.trend_30d_pct != 0.0).count());

        let universe = enriched_universe;

        // ── 2. Identify hot movers NOT in watchlist ─────────────
        let hot_movers: Vec<&kraken_api::MoverInfo> = universe
            .iter()
            .filter(|m| {
                !current_watchlist.iter().any(|w| w.eq_ignore_ascii_case(&m.symbol))
                    && m.change_24h_pct >= min_change_pct
                    && m.volume_usd >= min_volume_usd
            })
            .take(max_suggestions)
            .collect();
        // (universe is already sorted by edge_score desc from get_top_movers_universe)

        let hot_count = universe
            .iter()
            .filter(|m| {
                !current_watchlist.iter().any(|w| w.eq_ignore_ascii_case(&m.symbol))
                    && m.change_24h_pct >= min_change_pct
                    && m.volume_usd >= min_volume_usd
            })
            .count();

        tracing::info!(
            "[MOVER-SCAN] Scanned {} coins, {} hot movers found (not tracked)",
            universe.len(),
            hot_count,
        );

        if hot_movers.is_empty() {
            continue;
        }

        // ── FAST-TRACK: directly promote breakout coins into active watchlist ──
        // Bypasses the AI approval queue for coins that clearly pass all criteria.
        // Cooldown prevents WS reconnect spam when multiple hot movers appear at once.
        let fast_track_enabled = std::env::var("MOVER_FAST_TRACK_ENABLED")
            .ok().map(|v| v.trim() == "1").unwrap_or(false);
        if fast_track_enabled {
            let now_sec = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            let cooldown_elapsed = now_sec.saturating_sub(last_fast_track_ts);

            if cooldown_elapsed >= fast_track_cooldown_sec {
                let ft_min_score  = std::env::var("MOVER_FAST_TRACK_MIN_SCORE")
                    .ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(12.0);
                let ft_min_vol    = std::env::var("MOVER_FAST_TRACK_MIN_VOL_USD")
                    .ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(500_000.0);
                let ft_min_change = std::env::var("MOVER_FAST_TRACK_MIN_CHANGE_PCT")
                    .ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(15.0);
                let max_active    = std::env::var("MAX_TRACKED_COINS")
                    .ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(75);

                let fast_candidates: Vec<&kraken_api::MoverInfo> = universe.iter()
                    .filter(|m| {
                        !current_watchlist.iter().any(|w| w.eq_ignore_ascii_case(&m.symbol))
                            && m.edge_score     >= ft_min_score
                            && m.volume_usd     >= ft_min_vol
                            && m.change_24h_pct >= ft_min_change
                    })
                    .collect();

                if !fast_candidates.is_empty() {
                    let mut coins = active_coins.write().await;
                    let mut promoted = false;
                    for m in fast_candidates {
                        if coins.len() >= max_active { break; }
                        if !coins.iter().any(|c| c.eq_ignore_ascii_case(&m.symbol)) {
                            tracing::info!(
                                "[MOVER-FAST-TRACK] Auto-promoted {} — score={:.2} vol=${:.1}M chg={:+.1}% (cooldown={}s)",
                                m.symbol, m.edge_score, m.volume_usd / 1_000_000.0,
                                m.change_24h_pct, fast_track_cooldown_sec
                            );
                            coins.push(m.symbol.clone());
                            promoted = true;
                        }
                    }
                    if promoted {
                        last_fast_track_ts = now_sec;
                    }
                }
            } else {
                tracing::debug!(
                    "[MOVER-FAST-TRACK] Cooldown active — {}s remaining",
                    fast_track_cooldown_sec.saturating_sub(cooldown_elapsed)
                );
            }
        }

        // Log top hot movers
        for m in &hot_movers {
            let vol_label = if m.volume_usd >= 1_000_000.0 {
                format!("${:.1}M", m.volume_usd / 1_000_000.0)
            } else {
                format!("${:.0}K", m.volume_usd / 1_000.0)
            };
            tracing::info!(
                "[MOVER-SCAN] Top: {} {:+.1}% vol={} score={:.2}",
                m.symbol,
                m.change_24h_pct,
                vol_label,
                m.edge_score,
            );
        }

        // ── 3. Identify weakest tracked coins for removal ───────
        // Exclude anchor coins (BTC, ETH) and meme coins — they can never be removed
        let mut weakest: Vec<(&str, f64)> = current_watchlist
            .iter()
            .filter(|sym| !anchor_coins.iter().any(|a| a.eq_ignore_ascii_case(sym)))
            .filter(|sym| crate::config::coin_tier(sym) != "meme")
            .map(|sym| {
                let score = universe
                    .iter()
                    .find(|m| m.symbol.eq_ignore_ascii_case(sym))
                    .map(|m| m.edge_score)
                    .unwrap_or(-1.0); // not in top 200 → very cold
                (sym.as_str(), score)
            })
            .collect();
        weakest.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // ── 3b. MEDIUM-TRACK: Rust-only swap for score 8–12 ─────
        // No AI confirmation — Rust decides based on pure score improvement.
        // Fills the gap between fast-track (≥12) and AI queue (anything else).
        let medium_track_enabled = std::env::var("MOVER_MEDIUM_TRACK_ENABLED")
            .ok().map(|v| v.trim() == "1").unwrap_or(true); // ON by default
        let mt_min_score: f64 = std::env::var("MOVER_MEDIUM_TRACK_MIN_SCORE")
            .ok().and_then(|v| v.parse().ok()).unwrap_or(8.0);
        let mt_cooldown_sec: u64 = std::env::var("MOVER_MEDIUM_TRACK_COOLDOWN_SEC")
            .ok().and_then(|v| v.parse().ok()).unwrap_or(600);

        if medium_track_enabled {
            let now_sec = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .map(|d| d.as_secs()).unwrap_or(0);
            let mt_elapsed = now_sec.saturating_sub(last_fast_track_ts); // reuse cooldown tracker

            if mt_elapsed >= mt_cooldown_sec {
                let max_active = std::env::var("MAX_TRACKED_COINS")
                    .ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(75);

                // Find medium-track candidates: score 8–12, not already in watchlist
                let mt_candidates: Vec<&crate::kraken_api::MoverInfo> = universe.iter()
                    .filter(|m| {
                        !current_watchlist.iter().any(|w| w.eq_ignore_ascii_case(&m.symbol))
                            && m.edge_score >= mt_min_score
                            && m.edge_score < fast_track_cooldown_sec as f64 // below fast-track threshold
                            && m.volume_usd >= min_volume_usd
                            && m.change_24h_pct >= min_change_pct
                    })
                    .take(2)
                    .collect();

                if !mt_candidates.is_empty() {
                    let mut coins = active_coins.write().await;
                    let current_len = coins.len();
                    for (i, m) in mt_candidates.iter().enumerate() {
                        if coins.len() >= max_active {
                            // Remove weakest to make room
                            if let Some((remove_sym, remove_score)) = weakest.get(i) {
                                if m.edge_score > remove_score * 1.5 {
                                    coins.retain(|c| !c.eq_ignore_ascii_case(remove_sym));
                                    tracing::info!(
                                        "[MOVER-MEDIUM-TRACK] Swap: +{} (score={:.1}) −{} (score={:.1})",
                                        m.symbol, m.edge_score, remove_sym, remove_score
                                    );
                                }
                            }
                        }
                        if coins.len() < max_active && !coins.iter().any(|c| c.eq_ignore_ascii_case(&m.symbol)) {
                            tracing::info!(
                                "[MOVER-MEDIUM-TRACK] Auto-added {} score={:.1} vol=${:.0}K chg={:+.1}%",
                                m.symbol, m.edge_score, m.volume_usd / 1000.0, m.change_24h_pct
                            );
                            coins.push(m.symbol.clone());
                        }
                    }
                    if coins.len() != current_len {
                        drop(coins);
                        ws_reconnect.store(true, std::sync::atomic::Ordering::SeqCst);
                    }
                }
            }
        }

        // ── 4. Build swap suggestions (for AI vision scanner if enabled) ────
        let now_ts = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);

        // Read existing queue to avoid duplicates
        let existing_adds: Vec<String> = {
            let q = swap_queue.read().await;
            q.iter().flat_map(|s| s.add.clone()).collect()
        };

        let mut suggestions = Vec::new();
        let mut used_removals = 0usize;

        for hot in &hot_movers {
            // Skip if already queued
            if existing_adds.iter().any(|a| a.eq_ignore_ascii_case(&hot.symbol)) {
                tracing::debug!("[MOVER-SCAN] {} already in swap queue — skipping", hot.symbol);
                continue;
            }

            // Find next weakest removal candidate
            if used_removals >= weakest.len() {
                break; // no more coins to remove
            }
            let (remove_sym, remove_score) = weakest[used_removals];

            // Only suggest if meaningful improvement (1.5× threshold)
            if hot.edge_score <= remove_score * 1.5 {
                tracing::debug!(
                    "[MOVER-SCAN] {} score={:.2} not enough over {} score={:.2} — skipping",
                    hot.symbol,
                    hot.edge_score,
                    remove_sym,
                    remove_score,
                );
                used_removals += 1;
                continue;
            }

            tracing::info!(
                "[MOVER-SCAN] Suggesting swap: add={} remove={} ({} score={:.2} >> {} score={:.2})",
                hot.symbol,
                remove_sym,
                hot.symbol,
                hot.edge_score,
                remove_sym,
                remove_score,
            );

            suggestions.push(CoinSwapSuggestion {
                add: vec![hot.symbol.clone()],
                remove: vec![remove_sym.to_string()],
                reason: format!(
                    "Mover scanner: {} {:+.1}% edge={:.2} replaces {} edge={:.2}",
                    hot.symbol, hot.change_24h_pct, hot.edge_score, remove_sym, remove_score,
                ),
                timestamp: now_ts,
            });

            used_removals += 1;
        }

        // ── 5. Push to swap queue ───────────────────────────────
        if !suggestions.is_empty() {
            let count = suggestions.len();
            let mut q = swap_queue.write().await;
            q.extend(suggestions);
            tracing::info!("[MOVER-SCAN] Pushed {} swap suggestion(s) to queue", count);
        }
    }
}
