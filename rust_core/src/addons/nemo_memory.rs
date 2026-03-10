//! Nemo Memory — conscious brain with 4 memory layers.
//!
//! Layer 1: Outcome Memory — every closed trade scored + stored, per-coin track record
//! Layer 2: Self-Reflection — LLM reviews own behavior every 4-6 hours
//! Layer 3: Episodic Memory — big events stored permanently (last 20)
//! Layer 4: Personality Continuity — evolving identity.txt
//!
//! All layers injected into Nemo's prompt via `build_memory_block()`.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, OpenOptions};
use std::io::{BufRead, Write as IoWrite};
use std::path::{Path, PathBuf};

use crate::config::CachedEnv;

const OUTCOME_WINDOW_DAYS: f64 = 7.0;
const MAX_OUTCOMES: usize = 500;
const MAX_EPISODES: usize = 20;
const MAX_REFLECTIONS: usize = 10;
const MAX_IDENTITY_LINES: usize = 10;
const DEFAULT_REFLECTION_INTERVAL: f64 = 14400.0; // 4 hours

// ── Data Structs ───────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OutcomeRecord {
    pub ts: f64,
    pub symbol: String,
    pub side: String,
    pub pnl_pct: f64,
    pub hold_minutes: f64,
    pub exit_reason: String,
    pub entry_reasons: Vec<String>,
    pub tag: String,
    pub regime_label: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CoinTrackRecord {
    pub symbol: String,
    pub trade_count: u32,
    pub win_count: u32,
    pub loss_count: u32,
    pub avg_pnl_pct: f64,
    pub comment: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GlobalTrackRecord {
    pub window_days: f64,
    pub per_coin: Vec<CoinTrackRecord>,
    pub total_trades: u32,
    pub win_rate: f64,
    pub avg_pnl_pct: f64,
    pub computed_at: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SelfReflectionNote {
    pub ts: f64,
    pub text: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Episode {
    pub ts: f64,
    pub symbol: String,
    pub event_type: String,
    pub summary: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct IdentityNotes {
    pub lines: Vec<String>,
}

// ── NemoMemory ─────────────────────────────────────────────────────

pub struct NemoMemory {
    base_dir: PathBuf,
    outcomes: Vec<OutcomeRecord>,
    track_record: Option<GlobalTrackRecord>,
    reflections: Vec<SelfReflectionNote>,
    episodes: Vec<Episode>,
    identity: IdentityNotes,
    last_reflection_ts: f64,
    reflection_interval_sec: f64,
    pub embedding_store: crate::nvidia_tools::EmbeddingStore,
}

impl NemoMemory {
    /// Load all persistent state from data/nemo_memory/ directory.
    pub fn new() -> Self {
        let base_dir = PathBuf::from("data/nemo_memory");
        let _ = fs::create_dir_all(&base_dir);

        let mut outcomes = load_jsonl::<OutcomeRecord>(&base_dir.join("outcomes.jsonl"));
        // Auto-clean: purge dust sweeps from outcomes on startup
        let before_out = outcomes.len();
        outcomes.retain(|o| !o.exit_reason.contains("dust_sweep"));
        let purged_out = before_out - outcomes.len();
        if purged_out > 0 {
            save_jsonl(&outcomes, &base_dir.join("outcomes.jsonl"));
            tracing::info!("[AI-MEMORY] Auto-clean: purged {} dust outcomes", purged_out);
        }
        // Prune outcomes to 7-day window or MAX_OUTCOMES (whichever is smaller)
        let cutoff = crate::ai_bridge::now_ts() - (OUTCOME_WINDOW_DAYS * 86400.0);
        outcomes.retain(|o| o.ts >= cutoff);
        if outcomes.len() > MAX_OUTCOMES {
            outcomes.drain(..outcomes.len() - MAX_OUTCOMES);
        }
        let reflections = load_jsonl::<SelfReflectionNote>(&base_dir.join("reflections.jsonl"));
        let mut episodes = load_jsonl::<Episode>(&base_dir.join("episodes.jsonl"));
        // Auto-clean: purge panic_sell/dust episodes on startup
        let before_ep = episodes.len();
        episodes.retain(|e| {
            let s = e.summary.to_lowercase();
            !s.contains("panic_sell") && !s.contains("dust_sweep") && !s.contains("dust sweep")
        });
        let purged_ep = before_ep - episodes.len();
        if purged_ep > 0 {
            save_jsonl(&episodes, &base_dir.join("episodes.jsonl"));
            tracing::info!("[AI-MEMORY] Auto-clean: purged {} dust/panic episodes", purged_ep);
        }
        let track_record = load_json::<GlobalTrackRecord>(&base_dir.join("track_record.json"));
        let identity = load_identity(&base_dir.join("identity.txt"));

        // Trim episodes to max
        while episodes.len() > MAX_EPISODES {
            episodes.remove(0);
        }

        let reflection_interval_sec: f64 = std::env::var("AI_REFLECTION_INTERVAL_SEC")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_REFLECTION_INTERVAL);

        let embedding_store =
            crate::nvidia_tools::EmbeddingStore::load(&base_dir.join("embeddings.jsonl"));

        tracing::info!(
            "[AI-MEMORY] Loaded {} outcomes, {} reflections, {} episodes, {} identity lines, {} embeddings",
            outcomes.len(),
            reflections.len(),
            episodes.len(),
            identity.lines.len(),
            embedding_store.len(),
        );

        Self {
            base_dir,
            outcomes,
            track_record,
            reflections,
            episodes,
            identity,
            last_reflection_ts: 0.0,
            reflection_interval_sec,
            embedding_store,
        }
    }

    // ── Layer 1: Outcome Recording ─────────────────────────────────

    /// Build an OutcomeRecord from execute_exit() data.
    pub fn build_outcome(
        symbol: &str,
        pnl_pct: f64,
        hold_minutes: f64,
        exit_reason: &str,
        entry_reasons: &[String],
        regime_label: &str,
    ) -> OutcomeRecord {
        let tag = auto_tag(pnl_pct, hold_minutes, exit_reason);
        OutcomeRecord {
            ts: crate::ai_bridge::now_ts(),
            symbol: symbol.to_string(),
            side: "long".to_string(),
            pnl_pct,
            hold_minutes,
            exit_reason: exit_reason.to_string(),
            entry_reasons: entry_reasons.to_vec(),
            tag,
            regime_label: regime_label.to_string(),
        }
    }

    /// Record a closed trade outcome. Appends to JSONL, recomputes track record,
    /// checks for episode-worthy events.
    pub fn record_outcome(&mut self, outcome: OutcomeRecord) {
        tracing::info!(
            "[AI-MEMORY] Recording outcome: {} pnl={:+.2}% tag={} exit={}",
            outcome.symbol,
            outcome.pnl_pct * 100.0,
            outcome.tag,
            outcome.exit_reason,
        );

        // Check episode-worthiness BEFORE appending
        if let Some(episode) = check_episode(&outcome) {
            self.add_episode(episode);
        }

        // Append to JSONL (crash-safe)
        append_jsonl(&self.base_dir.join("outcomes.jsonl"), &outcome);
        self.outcomes.push(outcome);

        // Prune old outcomes to prevent unbounded memory growth
        let now = crate::ai_bridge::now_ts();
        let cutoff = now - (OUTCOME_WINDOW_DAYS * 86400.0);
        self.outcomes.retain(|o| o.ts >= cutoff);
        if self.outcomes.len() > MAX_OUTCOMES {
            self.outcomes.drain(..self.outcomes.len() - MAX_OUTCOMES);
        }

        // Recompute track record
        self.recompute_track_record(now);
    }

    /// Embed a closed trade outcome and store in the semantic memory.
    /// Graceful: logs warning and continues if NVIDIA API call fails.
    pub async fn embed_and_store_outcome(
        &mut self,
        outcome: &OutcomeRecord,
        client: &reqwest::Client,
    ) {
        let cfg = crate::nvidia_tools::NvidiaToolsConfig::from_env();
        if !cfg.embed_enabled || cfg.api_key.is_empty() {
            return;
        }

        // Build human-readable text for embedding
        let text = format!(
            "{} {} regime={} pnl={:+.2}% hold={:.0}min exit={} tag={} reasons={}",
            outcome.symbol,
            outcome.side,
            outcome.regime_label,
            outcome.pnl_pct * 100.0,
            outcome.hold_minutes,
            outcome.exit_reason,
            outcome.tag,
            outcome.entry_reasons.join(","),
        );

        let result: Result<Vec<Vec<f32>>, String> =
            crate::nvidia_tools::embed_texts(client, &cfg, &[text.as_str()]).await;
        match result {
            Ok(vecs) => {
                if let Some(embedding) = vecs.into_iter().next() {
                    let record = crate::nvidia_tools::EmbeddingRecord {
                        ts: outcome.ts,
                        symbol: outcome.symbol.clone(),
                        text,
                        embedding,
                    };
                    self.embedding_store.add(record);
                    tracing::info!(
                        "[NVIDIA-EMBED] Stored embedding for {} (total={})",
                        outcome.symbol,
                        self.embedding_store.len(),
                    );
                }
            }
            Err(e) => {
                tracing::warn!("[NVIDIA-EMBED] Failed to embed outcome: {e}");
            }
        }
    }

    /// Find similar past trades for a given setup description.
    /// Returns a formatted text block for prompt injection.
    pub async fn build_similar_trades_text(
        &self,
        setup_description: &str,
        client: &reqwest::Client,
    ) -> String {
        let cfg = crate::nvidia_tools::NvidiaToolsConfig::from_env();
        if !cfg.embed_enabled || cfg.api_key.is_empty() || self.embedding_store.len() == 0 {
            return String::new();
        }

        let result: Result<Vec<Vec<f32>>, String> =
            crate::nvidia_tools::embed_texts(client, &cfg, &[setup_description]).await;
        match result {
            Ok(vecs) => {
                if let Some(query_vec) = vecs.first() {
                    let similar = self.embedding_store.find_similar(query_vec, cfg.similar_trades_k);
                    crate::nvidia_tools::build_similar_trades_block(&similar)
                } else {
                    String::new()
                }
            }
            Err(e) => {
                tracing::warn!("[NVIDIA-EMBED] Failed to embed query: {e}");
                String::new()
            }
        }
    }

    /// Aggregate per-coin stats from outcomes within the 7-day window.
    fn recompute_track_record(&mut self, now: f64) {
        let cutoff = now - (OUTCOME_WINDOW_DAYS * 86400.0);
        let recent: Vec<&OutcomeRecord> = self.outcomes.iter().filter(|o| o.ts >= cutoff).collect();

        let mut per_coin: HashMap<String, Vec<&OutcomeRecord>> = HashMap::new();
        for o in &recent {
            per_coin.entry(o.symbol.clone()).or_default().push(o);
        }

        let mut coin_records: Vec<CoinTrackRecord> = per_coin
            .iter()
            .map(|(sym, trades)| {
                let wins = trades.iter().filter(|t| t.pnl_pct > 0.001).count() as u32;
                let losses = trades.iter().filter(|t| t.pnl_pct < -0.001).count() as u32;
                let avg_pnl = if trades.is_empty() {
                    0.0
                } else {
                    trades.iter().map(|t| t.pnl_pct).sum::<f64>() / trades.len() as f64
                };
                let comment = generate_coin_comment(trades);
                CoinTrackRecord {
                    symbol: sym.clone(),
                    trade_count: trades.len() as u32,
                    win_count: wins,
                    loss_count: losses,
                    avg_pnl_pct: avg_pnl,
                    comment,
                }
            })
            .collect();

        // Sort by trade count descending (most active first)
        coin_records.sort_by(|a, b| b.trade_count.cmp(&a.trade_count));

        let total = recent.len() as u32;
        let wins = recent.iter().filter(|t| t.pnl_pct > 0.001).count();
        let win_rate = if total > 0 {
            wins as f64 / total as f64
        } else {
            0.0
        };
        let avg_pnl = if total > 0 {
            recent.iter().map(|t| t.pnl_pct).sum::<f64>() / total as f64
        } else {
            0.0
        };

        let tr = GlobalTrackRecord {
            window_days: OUTCOME_WINDOW_DAYS,
            per_coin: coin_records,
            total_trades: total,
            win_rate,
            avg_pnl_pct: avg_pnl,
            computed_at: now,
        };

        save_json(&self.base_dir.join("track_record.json"), &tr);
        self.track_record = Some(tr);
    }

    // ── Layer 2: Self-Reflection ───────────────────────────────────

    /// Called every tick. Checks timer, runs reflection if due.
    pub async fn maybe_reflect(&mut self, now: f64, ai: &crate::ai_bridge::AiBridge) {
        // Re-read interval (hot-reloadable)
        let env = CachedEnv::snapshot();
        self.reflection_interval_sec = env.get_f64("AI_REFLECTION_INTERVAL_SEC", DEFAULT_REFLECTION_INTERVAL);

        if (now - self.last_reflection_ts) < self.reflection_interval_sec {
            return;
        }
        if self.outcomes.len() < 5 {
            return; // Not enough data yet
        }
        // Minimum uptime guard: don't reflect in first 10 minutes of a fresh start.
        // Prevents back-to-back 49B NIM calls (optimizer + reflection) at startup crashing the bot.
        if self.last_reflection_ts == 0.0 {
            self.last_reflection_ts = now; // defer to next interval
            return;
        }
        self.last_reflection_ts = now;
        self.run_self_reflection(ai).await;
    }

    async fn run_self_reflection(&mut self, ai: &crate::ai_bridge::AiBridge) {
        let system_prompt = load_reflection_prompt();
        let user_prompt = self.build_reflection_user_prompt();

        tracing::info!(
            "[AI-REFLECT] Starting self-reflection ({} outcomes, {} chars prompt)",
            self.outcomes.len(),
            user_prompt.len(),
        );
        let start = std::time::Instant::now();

        match ai.infer_for_optimizer(&system_prompt, &user_prompt).await {
            Ok(text) => {
                let ms = start.elapsed().as_millis();
                let trimmed = text.trim().to_string();
                tracing::info!(
                    "[AI-REFLECT] Reflection complete ({ms}ms): {}",
                    if trimmed.len() > 200 {
                        format!("{}...", &trimmed[..200])
                    } else {
                        trimmed.clone()
                    },
                );

                // Parse tool calls from response
                self.process_reflection_tools(&trimmed);

                // Extract reflection text (bullet points — everything that's not a tool call)
                let reflection_text = extract_reflection_text(&trimmed);
                if !reflection_text.is_empty() {
                    let note = SelfReflectionNote {
                        ts: crate::ai_bridge::now_ts(),
                        text: reflection_text,
                    };
                    append_jsonl(&self.base_dir.join("reflections.jsonl"), &note);
                    self.reflections.push(note);
                    while self.reflections.len() > MAX_REFLECTIONS {
                        self.reflections.remove(0);
                    }
                }
            }
            Err(e) => {
                tracing::warn!("[AI-REFLECT] Reflection failed: {e}");
            }
        }
    }

    /// Parse and execute tool calls from reflection LLM response.
    fn process_reflection_tools(&mut self, response: &str) {
        // Find all JSON objects in the response
        for json_str in extract_all_json(response) {
            let parsed: serde_json::Value = match serde_json::from_str(&json_str) {
                Ok(v) => v,
                Err(_) => continue,
            };

            let tool = match parsed.get("tool").and_then(|v| v.as_str()) {
                Some(t) => t,
                None => continue,
            };

            match tool {
                "write_identity" => {
                    if let Some(lines) = parsed.get("args")
                        .and_then(|a| a.get("lines"))
                        .and_then(|v| v.as_array())
                    {
                        let new_lines: Vec<String> = lines.iter()
                            .filter_map(|l| l.as_str().map(|s| s.to_string()))
                            .collect();
                        if !new_lines.is_empty() {
                            tracing::info!(
                                "[AI-REFLECT] Writing {} identity lines",
                                new_lines.len(),
                            );
                            self.update_identity(new_lines);
                        }
                    }
                }
                "write_episode" => {
                    if let Some(args) = parsed.get("args") {
                        let symbol = args.get("symbol")
                            .and_then(|v| v.as_str()).unwrap_or("UNKNOWN").to_string();
                        let event_type = args.get("event_type")
                            .and_then(|v| v.as_str()).unwrap_or("reflection_note").to_string();
                        let summary = args.get("summary")
                            .and_then(|v| v.as_str()).unwrap_or("").to_string();
                        if !summary.is_empty() {
                            tracing::info!(
                                "[AI-REFLECT] Flagging episode: {} {} — {}",
                                event_type, symbol, summary,
                            );
                            self.add_episode(Episode {
                                ts: crate::ai_bridge::now_ts(),
                                symbol,
                                event_type,
                                summary,
                            });
                        }
                    }
                }
                "write_coin_note" => {
                    if let Some(args) = parsed.get("args") {
                        let symbol = args.get("symbol")
                            .and_then(|v| v.as_str()).unwrap_or("").to_string();
                        let note = args.get("note")
                            .and_then(|v| v.as_str()).unwrap_or("").to_string();
                        if !symbol.is_empty() && !note.is_empty() {
                            tracing::info!(
                                "[AI-REFLECT] Coin note for {}: {}",
                                symbol, note,
                            );
                            self.add_episode(Episode {
                                ts: crate::ai_bridge::now_ts(),
                                symbol,
                                event_type: "coin_note".to_string(),
                                summary: note,
                            });
                        }
                    }
                }
                _ => {
                    tracing::debug!("[AI-REFLECT] Unknown tool '{}', ignoring", tool);
                }
            }
        }
    }

    fn build_reflection_user_prompt(&self) -> String {
        let mut prompt = String::with_capacity(4096);
        let now = crate::ai_bridge::now_ts();
        let cutoff_24h = now - 86400.0;

        // Recent outcomes (last 24h)
        let recent: Vec<&OutcomeRecord> = self.outcomes.iter().filter(|o| o.ts >= cutoff_24h).collect();
        prompt.push_str(&format!("YOUR RECENT TRADES (last 24h, {} total):\n", recent.len()));
        for o in recent.iter().take(30) {
            let result = if o.pnl_pct > 0.001 { "WIN" } else { "LOSS" };
            prompt.push_str(&format!(
                "  {} {} {:+.2}% hold={:.0}min exit={} tag={} regime={}\n",
                o.symbol,
                result,
                o.pnl_pct * 100.0,
                o.hold_minutes,
                o.exit_reason,
                o.tag,
                o.regime_label,
            ));
        }

        // 7-day track record
        if let Some(ref tr) = self.track_record {
            prompt.push_str(&format!(
                "\n7-DAY STATS: {} trades, WR={:.0}%, avg PnL={:+.3}%\n",
                tr.total_trades,
                tr.win_rate * 100.0,
                tr.avg_pnl_pct * 100.0,
            ));
            prompt.push_str("Per-coin breakdown:\n");
            for cr in &tr.per_coin {
                if cr.trade_count >= 2 {
                    prompt.push_str(&format!(
                        "  {}: {}T {}W/{}L avg={:+.2}%",
                        cr.symbol, cr.trade_count, cr.win_count, cr.loss_count,
                        cr.avg_pnl_pct * 100.0,
                    ));
                    if !cr.comment.is_empty() {
                        prompt.push_str(&format!(" -- {}", cr.comment));
                    }
                    prompt.push('\n');
                }
            }
        }

        // Recent episodes
        if !self.episodes.is_empty() {
            prompt.push_str("\nNOTABLE EPISODES:\n");
            for ep in self.episodes.iter().rev().take(10) {
                prompt.push_str(&format!("  - {}\n", ep.summary));
            }
        }

        prompt.push_str(
            "\nAnalyze your trading behavior. \
             What mistakes are you repeating? What's working well? \
             Be specific — reference actual coins and trade data. \
             Max 5 bullet points, each starting with \"- \".",
        );

        prompt
    }

    // ── Layer 3: Episodic Memory ───────────────────────────────────

    fn add_episode(&mut self, episode: Episode) {
        tracing::info!(
            "[AI-MEMORY] New episode: {} {} — {}",
            episode.event_type,
            episode.symbol,
            episode.summary,
        );
        append_jsonl(&self.base_dir.join("episodes.jsonl"), &episode);
        self.episodes.push(episode);
        while self.episodes.len() > MAX_EPISODES {
            self.episodes.remove(0);
        }
    }

    // ── Layer 4: Identity ──────────────────────────────────────────

    pub fn update_identity(&mut self, new_lines: Vec<String>) {
        self.identity.lines = new_lines;
        while self.identity.lines.len() > MAX_IDENTITY_LINES {
            self.identity.lines.remove(0);
        }
        save_identity(&self.base_dir.join("identity.txt"), &self.identity);
    }

    // ── Prompt Text Builders ───────────────────────────────────────

    pub fn build_track_record_text(&self) -> String {
        let tr = match &self.track_record {
            Some(tr) if tr.total_trades > 0 => tr,
            _ => return String::new(),
        };

        let mut text = format!("TRACK RECORD (last {} days):\n", tr.window_days as u32);
        for cr in &tr.per_coin {
            if cr.trade_count == 0 {
                continue;
            }
            text.push_str(&format!(
                "- {}: {} trades, {}W/{}L, avg PnL {:+.2}%",
                cr.symbol, cr.trade_count, cr.win_count, cr.loss_count,
                cr.avg_pnl_pct * 100.0,
            ));
            if !cr.comment.is_empty() {
                text.push_str(&format!(" -- {}", cr.comment));
            }
            text.push('\n');
        }
        text
    }

    pub fn build_reflection_text(&self) -> String {
        if self.reflections.is_empty() {
            return String::new();
        }
        let last = self.reflections.last().unwrap();
        format!("YOUR OWN NOTES FROM LAST REVIEW:\n{}\n", last.text)
    }

    pub fn build_episodes_text(&self) -> String {
        if self.episodes.is_empty() {
            return String::new();
        }
        let mut text = "RECENT EPISODES:\n".to_string();
        for ep in self.episodes.iter().rev().take(MAX_EPISODES) {
            text.push_str(&format!("- {}\n", ep.summary));
        }
        text
    }

    pub fn build_identity_text(&self) -> String {
        if self.identity.lines.is_empty() {
            return String::new();
        }
        let mut text = "IDENTITY:\n".to_string();
        for line in &self.identity.lines {
            text.push_str(&format!("- {}\n", line));
        }
        text
    }

    /// Read-only access to the global track record (for tool calls).
    pub fn track_record(&self) -> Option<&GlobalTrackRecord> {
        self.track_record.as_ref()
    }

    /// Read-only access to outcome records (for memory packet computation).
    pub fn outcomes(&self) -> &[OutcomeRecord] {
        &self.outcomes
    }

    /// Get recent episodes mentioning a specific symbol (for tool calls).
    pub fn recent_episodes_for(&self, symbol: &str, limit: usize) -> Vec<String> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();
        let upper = symbol.to_uppercase();
        self.episodes
            .iter()
            .rev()
            .filter(|ep| ep.summary.to_uppercase().contains(&upper) || ep.symbol.eq_ignore_ascii_case(symbol))
            .take(limit)
            .map(|ep| {
                let age_hours = (now - ep.ts) / 3600.0;
                format!("{:.0}h ago: {}", age_hours, ep.summary)
            })
            .collect()
    }

    /// Combines all 4 layers into one block for prompt injection.
    pub fn build_memory_block(&self) -> String {
        let tr = self.build_track_record_text();
        let refl = self.build_reflection_text();
        let ep = self.build_episodes_text();
        let id = self.build_identity_text();

        if tr.is_empty() && refl.is_empty() && ep.is_empty() && id.is_empty() {
            return String::new();
        }

        let mut block = String::with_capacity(2048);
        block.push_str("=== YOUR MEMORY ===\n");
        if !tr.is_empty() {
            block.push_str(&tr);
            block.push('\n');
        }
        if !refl.is_empty() {
            block.push_str(&refl);
            block.push('\n');
        }
        if !ep.is_empty() {
            block.push_str(&ep);
            block.push('\n');
        }
        if !id.is_empty() {
            block.push_str(&id);
        }
        block
    }
}

// ── Auto-Tagging ───────────────────────────────────────────────────

fn auto_tag(pnl_pct: f64, hold_minutes: f64, exit_reason: &str) -> String {
    if hold_minutes < 5.0 && pnl_pct < -0.003 {
        "panic_sell".to_string()
    } else if exit_reason.contains("whale") {
        "whale_exit".to_string()
    } else if exit_reason.contains("regime_crash") {
        "crash_exit".to_string()
    } else if exit_reason.contains("stop_loss") {
        "stop_loss".to_string()
    } else if exit_reason.contains("take_profit") {
        "take_profit".to_string()
    } else if exit_reason.contains("trailing") {
        "trailing_stop".to_string()
    } else if pnl_pct > 0.02 {
        "big_win".to_string()
    } else if pnl_pct < -0.015 {
        "big_loss".to_string()
    } else {
        "normal".to_string()
    }
}

// ── Episode Detection ──────────────────────────────────────────────

fn check_episode(outcome: &OutcomeRecord) -> Option<Episode> {
    let event_type = if outcome.pnl_pct > 0.02 {
        "big_win"
    } else if outcome.pnl_pct < -0.015 {
        "big_loss"
    } else if outcome.hold_minutes < 5.0 && outcome.pnl_pct < -0.003 {
        "panic_sell"
    } else if outcome.exit_reason.contains("regime_crash") {
        "crash_exit"
    } else if outcome.tag.contains("whale") {
        "whale_event"
    } else if outcome.exit_reason.contains("nemo_dual") {
        "nemo_override"
    } else {
        return None;
    };

    Some(Episode {
        ts: outcome.ts,
        symbol: outcome.symbol.clone(),
        event_type: event_type.to_string(),
        summary: format!(
            "{} {} pnl={:+.2}% hold={:.0}min exit={}",
            event_type,
            outcome.symbol,
            outcome.pnl_pct * 100.0,
            outcome.hold_minutes,
            outcome.exit_reason,
        ),
    })
}

// ── Behavioral Comment Generator ───────────────────────────────────

fn generate_coin_comment(trades: &[&OutcomeRecord]) -> String {
    if trades.is_empty() {
        return String::new();
    }

    let n = trades.len();
    let wins = trades.iter().filter(|t| t.pnl_pct > 0.001).count();
    let win_rate = wins as f64 / n as f64;
    let panic_count = trades.iter().filter(|t| t.tag == "panic_sell").count();
    let stop_count = trades.iter().filter(|t| t.tag == "stop_loss").count();
    let avg_hold: f64 = trades.iter().map(|t| t.hold_minutes).sum::<f64>() / n as f64;

    // Priority: most actionable comment first
    if n >= 4 && win_rate < 0.15 {
        return "STOP TRADING THIS COIN".to_string();
    }
    if panic_count > 1 {
        return format!("you panic-sell too often ({} of {} trades)", panic_count, n);
    }
    if n >= 3 && stop_count as f64 / n as f64 > 0.6 {
        return "most trades hit stop-loss -- entry timing issue".to_string();
    }
    if avg_hold < 10.0 && n >= 3 {
        return format!("very short holds (avg {:.0}min) -- consider waiting longer", avg_hold);
    }
    if win_rate > 0.7 && n >= 3 {
        return "your reads are accurate here".to_string();
    }

    String::new()
}

// ── JSON + Text Extraction Helpers ─────────────────────────────────

/// Extract all JSON objects from a text response (finds `{...}` blocks).
fn extract_all_json(text: &str) -> Vec<String> {
    let mut results = Vec::new();
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '{' {
            let mut depth = 0;
            let start = i;
            while i < chars.len() {
                match chars[i] {
                    '{' => depth += 1,
                    '}' => {
                        depth -= 1;
                        if depth == 0 {
                            let json_str: String = chars[start..=i].iter().collect();
                            // Only include if it looks like a tool call
                            if json_str.contains("\"tool\"") {
                                results.push(json_str);
                            }
                            break;
                        }
                    }
                    _ => {}
                }
                i += 1;
            }
        }
        i += 1;
    }
    results
}

/// Extract reflection text (bullet points) while removing JSON tool calls.
fn extract_reflection_text(text: &str) -> String {
    let mut lines = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim();
        // Skip empty lines and JSON tool calls
        if trimmed.is_empty() { continue; }
        if trimmed.starts_with('{') && trimmed.contains("\"tool\"") { continue; }
        lines.push(trimmed.to_string());
    }
    lines.join("\n")
}

// ── Reflection Prompt Loading ──────────────────────────────────────

fn load_reflection_prompt() -> String {
    crate::ai_bridge::load_prompt_from_paths(
        "AI_REFLECTION_PROMPT_PATH",
        "data/nemo_reflection_prompt.txt",
    )
    .unwrap_or_else(|| {
        "You are AI, a crypto trading AI reviewing your own performance. \
         Analyze the trade data and identify behavioral patterns. \
         Focus on actionable insights about YOUR behavior, not market conditions. \
         Reply with up to 5 bullet points, each starting with \"- \"."
            .to_string()
    })
}

// ── File I/O Helpers ───────────────────────────────────────────────

fn append_jsonl<T: Serialize>(path: &Path, item: &T) {
    match OpenOptions::new().create(true).append(true).open(path) {
        Ok(mut f) => {
            if let Ok(json) = serde_json::to_string(item) {
                let _ = writeln!(f, "{}", json);
            }
        }
        Err(e) => {
            tracing::warn!("[AI-MEMORY] Failed to write {}: {e}", path.display());
        }
    }
}

fn save_jsonl<T: Serialize>(items: &[T], path: &Path) {
    match fs::File::create(path) {
        Ok(mut f) => {
            for item in items {
                if let Ok(json) = serde_json::to_string(item) {
                    let _ = writeln!(f, "{}", json);
                }
            }
        }
        Err(e) => {
            tracing::warn!("[AI-MEMORY] Failed to save {}: {e}", path.display());
        }
    }
}

fn load_jsonl<T: for<'de> Deserialize<'de>>(path: &Path) -> Vec<T> {
    let file = match fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return Vec::new(),
    };
    let reader = std::io::BufReader::new(file);
    let mut items = Vec::new();
    for line in reader.lines() {
        if let Ok(line) = line {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            if let Ok(item) = serde_json::from_str::<T>(trimmed) {
                items.push(item);
            }
        }
    }
    items
}

fn save_json<T: Serialize>(path: &Path, item: &T) {
    match serde_json::to_string_pretty(item) {
        Ok(json) => {
            if let Err(e) = fs::write(path, json) {
                tracing::warn!("[AI-MEMORY] Failed to write {}: {e}", path.display());
            }
        }
        Err(e) => {
            tracing::warn!("[AI-MEMORY] Failed to serialize: {e}");
        }
    }
}

fn load_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Option<T> {
    let content = fs::read_to_string(path).ok()?;
    serde_json::from_str(&content).ok()
}

fn load_identity(path: &Path) -> IdentityNotes {
    match fs::read_to_string(path) {
        Ok(content) => {
            let lines: Vec<String> = content
                .lines()
                .map(|l| l.trim().to_string())
                .filter(|l| !l.is_empty())
                .collect();
            IdentityNotes { lines }
        }
        Err(_) => IdentityNotes::default(),
    }
}

fn save_identity(path: &Path, identity: &IdentityNotes) {
    let content = identity.lines.join("\n");
    if let Err(e) = fs::write(path, content) {
        tracing::warn!("[AI-MEMORY] Failed to write identity: {e}");
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_tag_panic_sell() {
        assert_eq!(auto_tag(-0.005, 3.0, "stop_loss(-0.5%)"), "panic_sell");
    }

    #[test]
    fn auto_tag_stop_loss() {
        assert_eq!(auto_tag(-0.008, 15.0, "stop_loss(-0.8%)"), "stop_loss");
    }

    #[test]
    fn auto_tag_take_profit() {
        assert_eq!(auto_tag(0.015, 30.0, "take_profit(1.5%)"), "take_profit");
    }

    #[test]
    fn auto_tag_big_win() {
        assert_eq!(auto_tag(0.025, 45.0, "trailing_stop(2.5%)"), "trailing_stop");
    }

    #[test]
    fn episode_detected_for_big_loss() {
        let outcome = OutcomeRecord {
            ts: 1000.0,
            symbol: "SOL".into(),
            side: "long".into(),
            pnl_pct: -0.02,
            hold_minutes: 12.0,
            exit_reason: "stop_loss(-2.0%)".into(),
            entry_reasons: vec!["trend_ok".into()],
            tag: "big_loss".into(),
            regime_label: "bearish".into(),
        };
        let ep = check_episode(&outcome);
        assert!(ep.is_some());
        assert_eq!(ep.unwrap().event_type, "big_loss");
    }

    #[test]
    fn no_episode_for_normal_trade() {
        let outcome = OutcomeRecord {
            ts: 1000.0,
            symbol: "BTC".into(),
            side: "long".into(),
            pnl_pct: 0.005,
            hold_minutes: 30.0,
            exit_reason: "take_profit(0.5%)".into(),
            entry_reasons: vec!["trend_ok".into()],
            tag: "normal".into(),
            regime_label: "bullish".into(),
        };
        assert!(check_episode(&outcome).is_none());
    }

    #[test]
    fn coin_comment_stop_trading() {
        let trades: Vec<OutcomeRecord> = (0..4)
            .map(|i| OutcomeRecord {
                ts: i as f64 * 100.0,
                symbol: "AVAX".into(),
                side: "long".into(),
                pnl_pct: -0.008,
                hold_minutes: 20.0,
                exit_reason: "stop_loss".into(),
                entry_reasons: vec![],
                tag: "stop_loss".into(),
                regime_label: "sideways".into(),
            })
            .collect();
        let refs: Vec<&OutcomeRecord> = trades.iter().collect();
        let comment = generate_coin_comment(&refs);
        assert_eq!(comment, "STOP TRADING THIS COIN");
    }

    #[test]
    fn memory_block_empty_when_no_data() {
        let mem = NemoMemory {
            base_dir: PathBuf::from("/tmp/test"),
            outcomes: vec![],
            track_record: None,
            reflections: vec![],
            episodes: vec![],
            identity: IdentityNotes::default(),
            last_reflection_ts: 0.0,
            reflection_interval_sec: DEFAULT_REFLECTION_INTERVAL,
            embedding_store: crate::nvidia_tools::EmbeddingStore::default(),
        };
        assert!(mem.build_memory_block().is_empty());
    }
}
