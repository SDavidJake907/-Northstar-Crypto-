//! Portfolio Allocator V2 — AI-driven portfolio-level allocation.
//!
//! Flow: build_snapshot() → call AI_1 → enforce_constraints() → save + return
//! Runs every ALLOC_INTERVAL_SEC (default 10min). AI_1 sees regime + scores + constraints,
//! outputs target allocations (symbol → pct). Native code clamps and renormalizes.
//!
//! This runs ALONGSIDE the existing per-coin system. The allocations are saved
//! to data/snapshots/allocations/ and returned for the trading loop to use.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

fn parse_env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(default)
}

fn now_utc_string() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // Simple ISO-like timestamp without chrono
    let s = secs % 60;
    let m = (secs / 60) % 60;
    let h = (secs / 3600) % 24;
    let days = secs / 86400;
    // Approximate date from epoch days (good enough for filenames)
    let y = 1970 + days / 365;
    let rem = days % 365;
    let mo = rem / 30 + 1;
    let d = rem % 30 + 1;
    format!("{:04}-{:02}-{:02}T{:02}-{:02}-{:02}", y, mo, d, h, m, s)
}

// ── Types ───────────────────────────────────────────────────

#[derive(Debug, Serialize, Clone)]
pub struct PortfolioSnapshot {
    pub asof_utc: String,
    pub regime: RegimeInfo,
    pub constraints: Constraints,
    pub candidates: Vec<CandidateInfo>,
}

#[derive(Debug, Serialize, Clone)]
pub struct RegimeInfo {
    pub label: String,
    pub confidence: f64,
    pub signals: RegimeSignals,
}

#[derive(Debug, Serialize, Clone)]
pub struct RegimeSignals {
    pub fear_greed: u32,
    pub btc_dominance_pct: f64,
    pub total_mcap_change_24h_pct: f64,
    pub funding_avg: f64,
    pub green_count: usize,
    pub coin_count: usize,
}

#[derive(Debug, Serialize, Clone)]
pub struct Constraints {
    pub max_asset_pct: HashMap<String, f64>,
    pub max_sector_pct: HashMap<String, f64>,
    pub max_total_risk_pct: f64,
    pub target_num_positions: usize,
}

#[derive(Debug, Serialize, Clone)]
pub struct CandidateInfo {
    pub symbol: String,
    pub sector: String,
    pub price: f64,
    pub scores: CandidateScores,
    pub current_pct: f64,
    pub pnl_pct: f64,
    // Matrix signal fields
    pub rsi: f64,
    pub hurst: f64,
    pub entropy: f64,
    pub kelly: f64,
    pub book_imbalance: f64,
    pub momentum: f64,
}

#[derive(Debug, Serialize, Clone)]
pub struct CandidateScores {
    pub fundamental: f64,
    pub trend: f64,
    pub liquidity: f64,
    pub volatility: f64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct QwenAllocation {
    pub risk_level: f64,
    #[serde(default = "default_confidence")]
    pub confidence: f64,
    pub allocations: Vec<AllocationEntry>,
    #[serde(default)]
    pub hedges: Vec<serde_json::Value>,
    #[serde(default)]
    pub warnings: Vec<String>,
}

fn default_confidence() -> f64 { 0.70 }

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AllocationEntry {
    pub symbol: String,
    pub pct: f64,
    pub reason: String,
}

// ── Allocator State ─────────────────────────────────────────

pub struct PortfolioAllocator {
    client: reqwest::Client,
    last_run_ts: f64,
    pub last_allocation: Option<QwenAllocation>,
    pub last_snapshot_path: Option<String>,
    run_count: u32,
}

impl PortfolioAllocator {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(90)) // Think mode needs 30-60s
                .build()
                .unwrap_or_default(),
            last_run_ts: 0.0,
            last_allocation: None,
            last_snapshot_path: None,
            run_count: 0,
        }
    }

    pub fn alloc_count(&self) -> u32 {
        self.run_count
    }

    /// Run allocator if enough time has passed. Returns latest allocation.
    pub async fn maybe_run(
        &mut self,
        now: f64,
        snapshot: &PortfolioSnapshot,
    ) -> Option<&QwenAllocation> {
        let interval = parse_env_f64("ALLOC_INTERVAL_SEC", 600.0);
        if now - self.last_run_ts < interval {
            return None; // Not time yet — only run on fresh intervals
        }

        let enabled = std::env::var("ALLOC_ENABLED")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        if !enabled {
            return None;
        }

        self.last_run_ts = now;
        self.run_count += 1;

        match self.run_allocation(snapshot).await {
            Ok(alloc) => {
                // Save snapshot
                if let Ok(path) = save_allocation_snapshot(snapshot, &alloc) {
                    self.last_snapshot_path = Some(path);
                }
                tracing::info!(
                    "[ALLOCATOR] Regime={} risk={:.0} positions={} | {}",
                    snapshot.regime.label,
                    alloc.risk_level,
                    alloc.allocations.len(),
                    alloc.allocations.iter()
                        .map(|a| format!("{}:{:.0}%", a.symbol, a.pct))
                        .collect::<Vec<_>>()
                        .join(" "),
                );
                self.last_allocation = Some(alloc);
                self.last_allocation.as_ref()
            }
            Err(e) => {
                tracing::warn!("[ALLOCATOR] Failed: {e} — using fallback");
                let fallback = fallback_allocator(snapshot);
                self.last_allocation = Some(fallback);
                self.last_allocation.as_ref()
            }
        }
    }

    /// Call AI_1 and parse allocation response
    async fn run_allocation(&self, snapshot: &PortfolioSnapshot) -> Result<QwenAllocation, String> {
        let host = std::env::var("OLLAMA_HOST")
            .unwrap_or_else(|_| "http://127.0.0.1:8081".into());
        let url = format!("{host}/v1/chat/completions");

        let system = load_allocator_prompt();
        let user_msg = build_compact_prompt(snapshot);

        tracing::info!(
            "[ALLOCATOR] Sending to AI_1: system={}chars user={}chars candidates={}",
            system.len(), user_msg.len(), snapshot.candidates.len()
        );
        tracing::info!(
            "[ALLOCATOR] Compact prompt preview: {}",
            &user_msg[..user_msg.len().min(400)]
        );

        // Allocator always uses /no_think — needs fast structured JSON output
        // Think mode is for exits/optimizer where deep reasoning helps
        let user_msg_final = format!("{}\n/no_think", user_msg);
        let max_tokens = 1024;

        let body = serde_json::json!({
            "model": std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "qwen3-14b".into()),
            "messages": [
                { "role": "system", "content": system },
                { "role": "user", "content": user_msg_final }
            ],
            "temperature": 0.3,
            "max_tokens": max_tokens,
            "stream": false
        });

        let t0 = std::time::Instant::now();
        let resp = self.client.post(&url).json(&body).send().await
            .map_err(|e| format!("HTTP error: {e}"))?;
        let text = resp.text().await
            .map_err(|e| format!("read error: {e}"))?;
        let ms = t0.elapsed().as_millis();

        // Extract content from OpenAI response
        let content = if let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) {
            let msg = &json["choices"][0]["message"];
            let c = msg["content"].as_str().unwrap_or("").trim();
            if c.is_empty() {
                // Fallback: some models put output in reasoning_content
                // Skip in think mode — reasoning_content is internal monologue, not JSON
                msg["reasoning_content"].as_str().unwrap_or("").trim().to_string()
            } else {
                c.to_string()
            }
        } else {
            text.clone()
        };

        // Strip <think>...</think> blocks (DeepSeek R1 reasoning chains)
        let content = {
            let mut s = content;
            while let Some(start) = s.find("<think>") {
                if let Some(end) = s.find("</think>") {
                    s = format!("{}{}", &s[..start], &s[end + 8..]);
                } else {
                    s = s[..start].to_string();
                    break;
                }
            }
            s
        };
        let clean = content.trim();
        tracing::info!("[ALLOCATOR] AI_1 responded in {ms}ms ({} chars)", clean.len());
        tracing::info!("[ALLOCATOR] Raw AI_1 output: {}", &clean[..clean.len().min(800)]);

        // Parse JSON — brace-counting to find matched { }
        let json_start = match clean.find('{') {
            Some(s) => s,
            None => return Err("No JSON in response".into()),
        };
        // Count braces to find the matching closing }
        let mut depth = 0i32;
        let mut json_end = json_start;
        for (i, ch) in clean[json_start..].char_indices() {
            match ch {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        json_end = json_start + i;
                        break;
                    }
                }
                _ => {}
            }
        }
        let raw: QwenAllocation = serde_json::from_str(&clean[json_start..=json_end])
            .map_err(|e| format!("JSON parse failed: {e}"))?;

        // Enforce constraints
        enforce_constraints(snapshot, raw)
    }
}

// ── Compact Matrix Prompt Builder ────────────────────────────

fn build_compact_prompt(snapshot: &PortfolioSnapshot) -> String {
    let mut out = String::with_capacity(512);

    // Header: regime + portfolio state
    let cash_pct: f64 = 100.0 - snapshot.candidates.iter().map(|c| c.current_pct).sum::<f64>();
    let header = format!(
        "REGIME:{} GREEN:{}/{} CONF:{:.2} CASH:{:.0}%\n",
        snapshot.regime.label,
        snapshot.regime.signals.green_count,
        snapshot.regime.signals.coin_count,
        snapshot.regime.confidence,
        cash_pct.max(0.0),
    );
    out.push_str(&header);

    // Column header
    out.push_str("SYM|SEC|H|E|K|RSI|IMB|MOM|CUR%|PNL%|$\n");
    let header_len = out.len();

    // One line per coin — pure signal
    for c in &snapshot.candidates {
        let regime_char = if c.entropy >= 0.85 {
            'N'
        } else if c.hurst > 0.55 && c.momentum > 0.05 {
            'T'
        } else if c.hurst < 0.45 && c.momentum < -0.05 {
            'M'
        } else {
            'S'
        };

        let sec = match c.sector.as_str() {
            "MONETARY" => "MO",
            "L1" => "L1",
            "INFRA" => "IN",
            "MEME" => "ME",
            _ => "OT",
        };

        out.push_str(&format!(
            "{}|{}{}|{:.2}|{:.2}|{:.2}|{:.0}|{:+.2}|{:+.2}|{:.0}|{:+.1}|{:.0}\n",
            c.symbol, sec, regime_char,
            c.hurst, c.entropy, c.kelly,
            c.rsi, c.book_imbalance, c.momentum,
            c.current_pct, c.pnl_pct, c.price,
        ));
    }
    let matrix_len = out.len() - header_len;

    // Constraints
    out.push_str(&format!(
        "MAX: BTC<{:.0}% ETH<{:.0}% MEME<{:.0}% TARGET:{} pos\n",
        snapshot.constraints.max_asset_pct.get("BTC").copied().unwrap_or(50.0),
        snapshot.constraints.max_asset_pct.get("ETH").copied().unwrap_or(40.0),
        snapshot.constraints.max_sector_pct.get("MEME").copied().unwrap_or(10.0),
        snapshot.constraints.target_num_positions,
    ));
    out.push_str("FEES=0.52%RT. K=0 means no edge—skip. Small shifts(<5%) lose to fees. Prefer HOLD.\n");
    out.push_str("RESPOND JSON: {\"risk_level\":N,\"confidence\":N,\"allocations\":[{\"symbol\":\"X\",\"pct\":N,\"reason\":\"X\"}]}\n");
    let tail_len = out.len() - header_len - matrix_len;

    // Diagnostic: log component sizes
    tracing::info!(
        "[PROMPT-LEN] header={} matrix={} tail={} total={}",
        header_len, matrix_len, tail_len, out.len()
    );
    // Head/tail preview
    let head: String = out.chars().take(300).collect();
    let tail: String = out.chars().rev().take(300).collect::<String>().chars().rev().collect();
    tracing::info!("[PROMPT-HEAD] {}", head);
    tracing::info!("[PROMPT-TAIL] {}", tail);

    out
}

// ── Prompt Loader ───────────────────────────────────────────

fn load_allocator_prompt() -> String {
    crate::ai_bridge::load_prompt_from_paths(
        "PORTFOLIO_ALLOCATOR_PROMPT_PATH",
        "data/portfolio_allocator_prompt.txt",
    )
    .unwrap_or_else(|| {
        "You are a crypto portfolio manager. Output JSON with allocations summing to 100.".to_string()
    })
}

// ── Constraint Enforcement ──────────────────────────────────

fn enforce_constraints(snapshot: &PortfolioSnapshot, mut alloc: QwenAllocation) -> Result<QwenAllocation, String> {
    let allowed: std::collections::HashSet<&str> = snapshot.candidates.iter()
        .map(|c| c.symbol.as_str())
        .collect();

    // Only allow candidates
    alloc.allocations.retain(|a| allowed.contains(a.symbol.as_str()));

    // Clamp per-asset
    for a in alloc.allocations.iter_mut() {
        let cap = snapshot.constraints.max_asset_pct.get(&a.symbol)
            .or_else(|| {
                let sector = snapshot.candidates.iter()
                    .find(|c| c.symbol == a.symbol)
                    .map(|c| c.sector.as_str())
                    .unwrap_or("OTHER");
                match sector {
                    "MONETARY" => Some(&50.0),
                    "L1" => Some(&25.0),
                    "MEME" => Some(&10.0),
                    _ => Some(&20.0),
                }
            })
            .copied()
            .unwrap_or(20.0);
        a.pct = a.pct.clamp(0.0, cap);
    }

    // Clamp per-sector
    let sector_of = |sym: &str| -> String {
        snapshot.candidates.iter()
            .find(|c| c.symbol == sym)
            .map(|c| c.sector.clone())
            .unwrap_or_else(|| "OTHER".into())
    };

    let mut sector_sum: HashMap<String, f64> = HashMap::new();
    for a in &alloc.allocations {
        *sector_sum.entry(sector_of(&a.symbol)).or_insert(0.0) += a.pct;
    }
    for (sec, sum) in &sector_sum {
        if let Some(&cap) = snapshot.constraints.max_sector_pct.get(sec) {
            if *sum > cap {
                let scale = cap / sum;
                for a in alloc.allocations.iter_mut() {
                    if sector_of(&a.symbol) == *sec {
                        a.pct *= scale;
                    }
                }
            }
        }
    }

    // Remove zero allocations
    alloc.allocations.retain(|a| a.pct > 0.5);

    // Minimum diversification — at least 3 positions required
    if alloc.allocations.len() < 3 {
        return Err(format!(
            "Only {} positions after constraints (need 3+) — using fallback",
            alloc.allocations.len()
        ));
    }

    // Renormalize: if total > 100 scale down, if total < 100 remainder is cash (unallocated)
    let total: f64 = alloc.allocations.iter().map(|a| a.pct).sum();
    if total <= 0.0 {
        return Err("No valid allocations after constraints".into());
    }
    if total > 100.5 {
        // Scale down proportionally, but respect caps
        let scale = 100.0 / total;
        for a in alloc.allocations.iter_mut() {
            a.pct = (a.pct * scale * 100.0).round() / 100.0;
        }
    }
    // If total < 100, the remainder stays as cash — do NOT inflate allocations

    // Risk level clamped
    alloc.risk_level = alloc.risk_level.clamp(0.0, 100.0);

    Ok(alloc)
}

// ── Fallback Allocator ──────────────────────────────────────

fn composite(s: &CandidateScores) -> f64 {
    0.35 * s.fundamental + 0.30 * s.trend + 0.20 * s.liquidity + 0.15 * (100.0 - s.volatility)
}

fn fallback_allocator(snapshot: &PortfolioSnapshot) -> QwenAllocation {
    let mut scored: Vec<(&CandidateInfo, f64)> = snapshot.candidates.iter()
        .filter(|c| c.sector != "MEME") // No memes in defensive fallback
        .map(|c| (c, composite(&c.scores)))
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Ensure BTC is always included (move to front if present)
    if let Some(btc_idx) = scored.iter().position(|(c, _)| c.symbol == "BTC") {
        if btc_idx > 0 {
            let btc = scored.remove(btc_idx);
            scored.insert(0, btc);
        }
    }

    let n = snapshot.constraints.target_num_positions.min(scored.len()).max(3);
    let n = n.min(scored.len());
    if n < 3 {
        // Not enough candidates — return BTC-heavy defensive allocation
        return QwenAllocation {
            risk_level: 30.0,
            confidence: 0.65,
            allocations: vec![
                AllocationEntry { symbol: "BTC".into(), pct: 50.0, reason: "fallback_defensive".into() },
                AllocationEntry { symbol: "ETH".into(), pct: 30.0, reason: "fallback_defensive".into() },
            ],
            hedges: vec![],
            warnings: vec!["Fallback: too few candidates, defensive BTC/ETH".into()],
        };
    }
    let selected = &scored[..n];

    // BTC gets minimum 25% floor, rest score-weighted
    let btc_floor = 25.0;
    let remaining = 75.0;
    let non_btc: Vec<_> = selected.iter().filter(|(c, _)| c.symbol != "BTC").collect();
    let non_btc_sum: f64 = non_btc.iter().map(|x| x.1.max(1.0)).sum();

    let mut allocs = Vec::new();
    for (c, sc) in selected {
        let pct = if c.symbol == "BTC" {
            let score_pct = remaining * sc.max(1.0) / non_btc_sum.max(1.0);
            btc_floor + score_pct.min(25.0) // BTC: 25-50%
        } else {
            remaining * sc.max(1.0) / non_btc_sum.max(1.0)
        };
        allocs.push(AllocationEntry {
            symbol: c.symbol.clone(),
            pct: (pct * 100.0).round() / 100.0,
            reason: "fallback_score_weighted".into(),
        });
    }

    QwenAllocation {
        risk_level: 50.0,
        confidence: 0.65,
        allocations: allocs,
        hedges: vec![],
        warnings: vec!["Used deterministic fallback allocator".into()],
    }
}

// ── Snapshot Persistence ────────────────────────────────────

fn save_allocation_snapshot(snapshot: &PortfolioSnapshot, alloc: &QwenAllocation) -> Result<String, String> {
    let ts = now_utc_string();
    let dir = "data/snapshots/allocations";
    std::fs::create_dir_all(dir).map_err(|e| format!("mkdir failed: {e}"))?;

    let path = format!("{}/{}.json", dir, ts);
    let output = serde_json::json!({
        "timestamp": snapshot.asof_utc,
        "regime": snapshot.regime.label,
        "regime_confidence": snapshot.regime.confidence,
        "risk_level": alloc.risk_level,
        "allocations": alloc.allocations,
        "warnings": alloc.warnings,
    });

    std::fs::write(&path, serde_json::to_string_pretty(&output).unwrap_or_default())
        .map_err(|e| format!("write failed: {e}"))?;

    Ok(path)
}

// ── Snapshot Builder ────────────────────────────────────────
// Called from trading_loop to build the input for the allocator.

pub fn coin_sector(symbol: &str) -> &'static str {
    match symbol {
        "BTC" => "MONETARY",
        "ETH" | "SOL" | "AVAX" | "NEAR" | "ADA" | "DOT" | "ATOM" => "L1",
        "LINK" => "INFRA",
        "XRP" => "L1",
        "DOGE" | "SHIB" | "PEPE" | "FLOKI" | "BONK" => "MEME",
        _ => "OTHER",
    }
}

/// Convert raw features (from features_map) into normalized 0-100 scores.
pub fn compute_scores(feats: &serde_json::Value) -> CandidateScores {
    // Fundamental: weighted_score typically -1.0 to +1.0, map to 0-100
    let ws = feats.get("weighted_score").and_then(|v| v.as_f64()).unwrap_or(0.0);
    let fundamental = ((ws + 1.0) * 50.0).clamp(0.0, 100.0);

    // Trend: trend_score -3 to +3, map to 0-100
    let ts = feats.get("trend_score").and_then(|v| v.as_f64()).unwrap_or(0.0);
    let trend = ((ts + 3.0) / 6.0 * 100.0).clamp(0.0, 100.0);

    // Liquidity: vol_ratio typically 0.1-10+, map log scale to 0-100
    let vr = feats.get("vol_ratio").and_then(|v| v.as_f64()).unwrap_or(1.0);
    let liquidity = (vr.ln().max(-2.0) / 3.0 * 100.0 + 50.0).clamp(0.0, 100.0);

    // Volatility: ATR as % of price, higher = more volatile = worse
    let atr = feats.get("atr").and_then(|v| v.as_f64()).unwrap_or(0.0);
    let price = feats.get("price").and_then(|v| v.as_f64()).unwrap_or(1.0);
    let atr_pct = if price > 0.0 { atr / price * 100.0 } else { 5.0 };
    let volatility = (atr_pct * 20.0).clamp(0.0, 100.0); // 5% ATR = 100

    CandidateScores { fundamental, trend, liquidity, volatility }
}

/// Build default constraints from .env
pub fn build_constraints() -> Constraints {
    use parse_env_f64;

    let mut max_asset = HashMap::new();
    max_asset.insert("BTC".into(), parse_env_f64("ALLOC_MAX_BTC_PCT", 50.0));
    max_asset.insert("ETH".into(), parse_env_f64("ALLOC_MAX_ETH_PCT", 40.0));

    let mut max_sector = HashMap::new();
    max_sector.insert("MONETARY".into(), 70.0);
    max_sector.insert("L1".into(), parse_env_f64("ALLOC_MAX_SECTOR_L1_PCT", 50.0));
    max_sector.insert("DEFI".into(), parse_env_f64("ALLOC_MAX_SECTOR_DEFI_PCT", 35.0));
    max_sector.insert("AI".into(), parse_env_f64("ALLOC_MAX_SECTOR_AI_PCT", 35.0));
    max_sector.insert("MEME".into(), parse_env_f64("ALLOC_MAX_MEME_PCT", 10.0));
    max_sector.insert("INFRA".into(), 35.0);

    Constraints {
        max_asset_pct: max_asset,
        max_sector_pct: max_sector,
        max_total_risk_pct: 100.0,
        target_num_positions: parse_env_f64("ALLOC_TARGET_POSITIONS", 6.0) as usize,
    }
}
