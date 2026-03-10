//! LLM client — AiBridge struct, HTTP inference, NVIDIA cloud, NPU scanning.

use std::time::Instant;
use super::{AiBridgeError, BridgeResult, AiDecision, AiExitDecision, LlmToolCall, InferToolResult};
use super::parse::{parse_openai_content, parse_exit_decision, parse_entry_confirmation, parse_decision, strip_think_tags, extract_all_decisions};
use super::prompts::{load_nemo_prompt, load_nemo_exit_prompt, load_nemo_scan_prompt, load_nemo_entry_prompt};

// ── LLM HTTP Inference (OpenAI-compatible) ──────────────────

// ── AI Bridge ───────────────────────────────────────────────

/// AI Bridge — per-coin inference via local Ollama HTTP API.
///
/// Entry model: qwen2.5:1.5b — per-coin scoring decisions.
/// Exit model: Qwen 2.5-14B — hold/sell decisions for open positions.
pub struct AiBridge {
    ollama_url: String,
    model: String,
    temperature: f32,
    max_tokens: i32,
    client: reqwest::Client,
    // Second engine: Qwen on CPU for exit monitoring + market scanning
    exit_url: String,
    exit_url_from_env: bool,
    exit_model: String,
    exit_temperature: f32,
    exit_max_tokens: i32,
    // Cached env values (read once at construction)
    pub(crate) backend: String,
    pub(crate) batch_size: usize,
    pub(crate) nvidia_key: String,
    pub(crate) nvidia_url: String,
    pub(crate) nvidia_model: String,
    pub(crate) nvidia_optimizer_model: String, // separate model for optimizer (may differ from entry model)
    pub(crate) think_mode: bool,
    optimizer_temperature: f32,
    optimizer_max_tokens: i32,
    nemo_warmup: bool,
}

impl AiBridge {
    pub fn new(env: &crate::config::CachedEnv) -> Self {
        let ollama_url = {
            let raw = env.get_str("OLLAMA_HOST", "http://localhost:11434");
            let allow_insecure = env.get_bool("OLLAMA_ALLOW_INSECURE", false);
            match url::Url::parse(&raw) {
                Ok(u) => {
                    let scheme = u.scheme();
                    let host = u.host_str().unwrap_or("");
                    let is_loopback = host.eq_ignore_ascii_case("localhost")
                        || host == "127.0.0.1"
                        || host == "::1";
                    if scheme == "http" && !is_loopback && !allow_insecure {
                        eprintln!(
                            "[AI-BRIDGE] WARNING: refusing insecure remote OLLAMA_HOST={raw}; \
                             set OLLAMA_ALLOW_INSECURE=1 to override"
                        );
                        "http://localhost:11434".to_string()
                    } else {
                        raw
                    }
                }
                Err(_) => {
                    eprintln!("[AI-BRIDGE] WARNING: invalid OLLAMA_HOST={raw}; using localhost");
                    "http://localhost:11434".to_string()
                }
            }
        };
        let model = env.get_str("OLLAMA_MODEL", "qwen2.5-14b-instruct");
        let exit_model = env.get_str("AI_EXIT_MODEL", &model);
        let exit_url_from_env = env.map.contains_key("EXIT_AI_HOST");
        let exit_url = env.get_str("EXIT_AI_HOST", "http://127.0.0.1:8082");
        let temperature: f32 = env.get_parsed("MODEL_TEMPERATURE", 0.6_f32);
        let exit_temperature: f32 = env.get_parsed("AI_EXIT_TEMPERATURE", temperature);
        let exit_max_tokens: i32 = env.get_parsed("NEMO_EXIT_MAX_TOKENS", 1024_i32);
        if exit_url != ollama_url || exit_model != model {
            println!("[AI-BRIDGE] Exit: {exit_model} @ {exit_url} (Temp={exit_temperature} max_tok={exit_max_tokens})");
            println!("[AI-BRIDGE] Entry: {model} @ {ollama_url} (Temp={temperature})");
        } else {
            println!("[AI-BRIDGE] WARNING: Exit model shares entry host; set EXIT_AI_HOST to use a separate exit server");
        }
        let max_tokens: i32 = env.get_parsed("MODEL_MAX_TOKENS", 384_i32);

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(90)) // Think mode needs 30-60s per request
            .build()
            .unwrap_or_else(|e| {
                eprintln!("[AI-BRIDGE] WARNING: failed to create HTTP client: {e}; using default client");
                reqwest::Client::new()
            });

        // Cached env values
        let backend = env.get_str("AI_BACKEND", "ollama");
        let batch_size: usize = env.get_parsed("AI_BATCH_SIZE", 1_usize);
        let nvidia_key = env.get_str("NVIDIA_API_KEY", "");
        let nvidia_url = env.get_str("NVIDIA_API_URL", "https://integrate.api.nvidia.com/v1");
        let nvidia_model = env.get_str("NVIDIA_MODEL", "");
        // Optimizer may use a larger/slower model (e.g. 49B Super) while entry uses a fast model (9B)
        let nvidia_optimizer_model = {
            let m = env.get_str("NVIDIA_OPTIMIZER_MODEL", "");
            if m.is_empty() { nvidia_model.clone() } else { m }
        };
        let think_mode = env.get_bool("THINK_MODE", false);
        let optimizer_temperature: f32 = env.get_parsed("NEMO_OPTIMIZER_TEMPERATURE", 0.3_f32);
        let optimizer_max_tokens: i32 = env.get_parsed("NEMO_OPTIMIZER_MAX_TOKENS", 2048_i32);
        let nemo_warmup = env.get_bool("AI_WARMUP", true);

        // Single-brain mode: Ollama only, no TRT
        println!("[AI-BRIDGE] Ollama per-coin: {model} (max_tok={max_tokens})");
        println!("[AI-BRIDGE]   Host: {ollama_url}  Temp={temperature}");

        Self {
            ollama_url,
            model,
            temperature,
            max_tokens,
            client,
            exit_url,
            exit_url_from_env,
            exit_model,
            exit_temperature,
            exit_max_tokens,
            backend,
            batch_size,
            nvidia_key,
            nvidia_url,
            nvidia_model,
            nvidia_optimizer_model,
            think_mode,
            optimizer_temperature,
            optimizer_max_tokens,
            nemo_warmup,
        }
    }

    /// Human-readable label for optimizer routing/model.
    pub fn optimizer_model_label(&self) -> String {
        // Optimizer uses NIM cloud when optimizer model + API key are set (regardless of AI_BACKEND)
        if !self.nvidia_optimizer_model.is_empty() && !self.nvidia_key.is_empty() {
            format!("nvidia_cloud:{}", self.nvidia_optimizer_model)
        } else if self.backend == "nvidia_cloud" {
            format!("nvidia_cloud:{}(default)", self.model)
        } else {
            format!("local:{}", self.model)
        }
    }

    /// Shared HTTP client — used by nvidia_tools for API calls.
    pub fn client(&self) -> &reqwest::Client { &self.client }
    #[allow(dead_code)]
    pub fn ollama_url(&self) -> &str { &self.ollama_url }
    #[allow(dead_code)]
    pub fn model_name(&self) -> &str { &self.model }
    #[allow(dead_code)]
    pub fn temperature_val(&self) -> f32 { self.temperature }

    /// Verify LLM backend is available and warm up the model.
    pub async fn spawn(&mut self) -> BridgeResult<()> {
        let backend = &self.backend;

        if backend == "nvidia_cloud" {
            // NVIDIA NIM cloud API health check — verify API key works
            if self.nvidia_key.is_empty() {
                return Err(AiBridgeError::Http(
                    "NVIDIA_API_KEY not set — get one from build.nvidia.com".into()
                ));
            }
            let cloud_model = if self.nvidia_model.is_empty() { &self.model } else { &self.nvidia_model };
            println!("[AI-BRIDGE] NVIDIA cloud mode — model: {cloud_model}");
            let api_key = &self.nvidia_key;
            println!("[AI-BRIDGE]   API key: {}...{}", &api_key[..8.min(api_key.len())], &api_key[api_key.len().saturating_sub(4)..]);
        } else if backend == "openai" {
            // OpenAI-compatible health check — try /health (llama-server), fallback to / (Ollama)
            let health_url = format!("{}/health", self.ollama_url);
            let root_url = self.ollama_url.clone();
            let resp = self.client.get(&health_url).send().await;
            let ok = match resp {
                Ok(r) if r.status().is_success() => true,
                _ => {
                    // Fallback: Ollama serves OpenAI at /v1/ but health is at /
                    let fallback = self.client.get(&root_url).send().await.map_err(|e| {
                        AiBridgeError::Http(format!("LLM server not reachable at {}: {e}", self.ollama_url))
                    })?;
                    fallback.status().is_success()
                }
            };
            if !ok {
                return Err(AiBridgeError::Http(format!(
                    "LLM server not reachable at {}", self.ollama_url
                )));
            }
            println!("[AI-BRIDGE] LLM connected at {} (OpenAI mode)", self.ollama_url);
        } else {
            // Ollama health check
            let url = format!("{}/api/tags", self.ollama_url);
            let resp = self.client.get(&url).send().await.map_err(|e| {
                AiBridgeError::Http(format!("Ollama not reachable at {}: {e}", self.ollama_url))
            })?;

            if !resp.status().is_success() {
                return Err(AiBridgeError::Http(format!(
                    "Ollama returned status {}",
                    resp.status()
                )));
            }

            let json: serde_json::Value = resp
                .json()
                .await
                .map_err(|e| AiBridgeError::Http(format!("invalid response: {e}")))?;

            if let Some(models) = json.get("models").and_then(|m| m.as_array()) {
                let names: Vec<&str> = models
                    .iter()
                    .filter_map(|m| m.get("name").and_then(|n| n.as_str()))
                    .collect();
                println!(
                    "[AI-BRIDGE] Ollama connected — {} models: {}",
                    names.len(),
                    names.join(", ")
                );

                if !names.iter().any(|n| n.starts_with(self.model.as_str())) {
                    eprintln!(
                        "[AI-BRIDGE] WARNING: model '{}' not found in Ollama",
                        self.model
                    );
                }
            }
        }

        if self.batch_size > 1 {
            println!("[AI-BRIDGE] Ready (BATCH mode, size={})", self.batch_size);
        } else {
            println!("[AI-BRIDGE] Ready (per-coin mode)");
        }
        if self.nemo_warmup {
            self.warm_up().await;
        }
        Ok(())
    }

    /// Warm-load the model so it stays active after service restart.
    async fn warm_up(&self) {
        let system_prompt = load_nemo_prompt();
        let user_prompt = r#"{"warmup":true}"#;
        let warm_tokens = self.max_tokens.min(16);
        let start = Instant::now();
        let result = infer_ollama(
            &self.client,
            &self.ollama_url,
            &self.model,
            &system_prompt,
            user_prompt,
            self.temperature,
            warm_tokens,
            None,
            Some(false), // warm-up: no think needed
            &self.backend,
            &self.nvidia_key,
            &self.nvidia_url,
            &self.nvidia_model,
            self.think_mode,
        )
        .await;
        let ms = start.elapsed().as_millis() as u64;
        match result {
            Ok(_) => println!("[AI-BRIDGE] Warm-up OK ({ms}ms)"),
            Err(e) => println!("[AI-BRIDGE] Warm-up failed ({ms}ms): {e}"),
        }
    }

    // ── Local inference (Ollama only) ────────────────────

    /// Run local inference via Ollama HTTP. Single path, no fallback needed.
    /// `think`: Some(true) = force /think, Some(false) = force /no_think, None = use env
    #[allow(dead_code)]
    async fn infer_local(
        &self,
        system_prompt: &str,
        user_prompt: &str,
        temperature: f32,
        max_tokens: i32,
        think: Option<bool>,
    ) -> BridgeResult<String> {
        infer_ollama(
            &self.client,
            &self.ollama_url,
            &self.model,
            system_prompt,
            user_prompt,
            temperature,
            max_tokens,
            Some(4096),
            think,
            &self.backend,
            &self.nvidia_key,
            &self.nvidia_url,
            &self.nvidia_model,
            self.think_mode,
        )
        .await
    }

    // ── Public API ──────────────────────────────────────────

    /// Get per-coin decision from Qwen via llama-server (OpenAI API).
    ///
    /// Returns validated decision; on failure returns HOLD (safe -- no trades).
    #[allow(dead_code)]
    pub async fn get_decision(&self, user_prompt: &str, tick_count: u64) -> AiDecision {
        if tick_count % 10 == 0 {
            let chars = user_prompt.len();
            let approx_tokens = chars / 4;
            tracing::info!("[QWEN-AI] prompt_size chars={chars} ~{approx_tokens}tok");
        }

        let system_prompt = load_nemo_prompt();
        let start = Instant::now();

        // Local inference — entry decisions use THINK_MODE env (default=think for tool-calling)
        let local_result = self.infer_local(
            &system_prompt,
            user_prompt,
            self.temperature,
            self.max_tokens,
            None, // respect THINK_MODE env
        ).await;
        let ms = start.elapsed().as_millis() as u64;

        match local_result {
            Ok(text) => {
                if let Some(mut d) = parse_decision(&text) {
                    d.latency_ms = ms;
                    d.decision_ts = now_ts();
                    // Log the raw response and parsed source for debugging
                    let raw_preview: String = text.chars().take(120).collect();
                    tracing::info!(
                        "[LOCAL-AI] {ms}ms {} → {}|{:.2}|{}|{} src={} raw=\"{}\"",
                        d.action, d.action, d.confidence, d.lane,
                        d.hold_hours.unwrap_or(0), d.source, raw_preview
                    );
                    d.source = "local_ollama".into();
                    d
                } else {
                    tracing::warn!("[LOCAL-AI] {ms}ms parse_fail -- raw: {}", &text[..text.floor_char_boundary(200)]);
                    AiDecision { latency_ms: ms, decision_ts: now_ts(), ..AiDecision::default() }
                }
            }
            Err(e) => {
                tracing::error!("[LOCAL-AI] inference failed ({ms}ms): {e}");
                AiDecision { latency_ms: ms, decision_ts: now_ts(), ..AiDecision::default() }
            }
        }
    }

    /// Get batch decisions — multiple coins in one Ollama call.
    ///
    /// Sends all coin data as a JSON array so Nemo sees them as "pillars"
    /// side by side. Returns a map of symbol → decision.
    #[allow(dead_code)]
    pub async fn get_batch_decisions(
        &self,
        coins: &[(String, serde_json::Value)],
        tick_count: u64,
    ) -> std::collections::HashMap<String, AiDecision> {
        use std::collections::HashMap;

        let batch_size = coins.len();
        if batch_size == 0 {
            return HashMap::new();
        }

        // Build numbered batch prompt — each coin on its own line with symbol label
        let mut coin_lines = String::new();
        for (i, (sym, data)) in coins.iter().enumerate() {
            let compact = serde_json::to_string(data).unwrap_or_default();
            coin_lines.push_str(&format!("#{} {}: {}\n", i + 1, sym, compact));
        }
        let user_prompt = format!(
            "Analyze {} coins. For EACH coin reply with one JSON (include \"symbol\"):\n\
             {coin_lines}\n\
             Reply with {batch_size} JSON objects, one per line. Only JSON, no text.",
            batch_size
        );

        if tick_count % 5 == 0 {
            let chars = user_prompt.len();
            let approx_tokens = chars / 4;
            tracing::info!(
                "[QWEN-AI] batch={batch_size} prompt_size chars={chars} ~{approx_tokens}tok"
            );
        }

        let system_prompt = load_nemo_prompt();
        let start = Instant::now();

        // Scale output tokens: ~96 per coin (each JSON line ~80-90 tokens), capped
        let batch_max_tokens = ((batch_size as i32) * 128).min(2048).max(self.max_tokens);

        let result = self.infer_local(
            &system_prompt,
            &user_prompt,
            self.temperature,
            batch_max_tokens,
            None, // respect THINK_MODE env
        )
        .await;
        let ms = start.elapsed().as_millis() as u64;

        let mut decisions: HashMap<String, AiDecision> = HashMap::new();

        match result {
            Ok(text) => {
                // Extract all JSON objects from response
                let parsed = extract_all_decisions(&text);
                let found = parsed.len();

                for (sym, mut d) in parsed {
                    d.latency_ms = ms;
                    d.decision_ts = now_ts();
                    d.source = "qwen_batch".into();
                    decisions.insert(sym, d);
                }

                tracing::info!(
                    "[QWEN-AI] batch={batch_size} found={found} latency={ms}ms"
                );

                if found == 0 {
                    let preview: String = text.chars().take(600).collect();
                    tracing::warn!(
                        "[QWEN-AI] batch found=0! Raw response:\n{preview}"
                    );
                } else if found < batch_size {
                    tracing::warn!(
                        "[QWEN-AI] batch incomplete: expected={batch_size} got={found}"
                    );
                }
            }
            Err(e) => {
                tracing::error!("[QWEN-AI] batch inference failed ({ms}ms): {e}");
            }
        }

        // Fill missing coins with default HOLD
        for (sym, _) in coins {
            decisions.entry(sym.clone()).or_insert_with(|| AiDecision {
                latency_ms: ms,
                decision_ts: now_ts(),
                ..AiDecision::default()
            });
        }

        decisions
    }

    /// Market scan: Nemo analyzes ALL coins and gives market direction + top picks.
    ///
    /// Returns raw text for parsing by the trading loop.
    #[allow(dead_code)]
    pub async fn market_scan(&self, all_coins_prompt: &str) -> (String, u64) {
        let system_prompt = load_nemo_scan_prompt();
        let start = Instant::now();

        let result = infer_ollama(
            &self.client,
            &self.ollama_url,
            &self.model,
            &system_prompt,
            all_coins_prompt,
            self.temperature,
            256, // needs more tokens to analyze 10 coins
            None,
            None, // respect THINK_MODE env
            &self.backend,
            &self.nvidia_key,
            &self.nvidia_url,
            &self.nvidia_model,
            self.think_mode,
        )
        .await;
        let ms = start.elapsed().as_millis() as u64;

        match result {
            Ok(text) => {
                tracing::info!("[AI-SCAN] Market scan complete ({ms}ms)");
                (text, ms)
            }
            Err(e) => {
                tracing::warn!("[AI-SCAN] Market scan failed ({ms}ms): {e}");
                ("MARKET|UNKNOWN|scan_failed".to_string(), ms)
            }
        }
    }

    /// Get entry confirmation from Nemo for a BUY candidate.
    ///
    /// Uses the entry-specific constitution (nemo_entry_prompt.txt).
    /// Returns HOLD by default on failure (safe — no trade).
    #[allow(dead_code)]
    pub async fn get_entry_confirmation(&self, candidate_prompt: &str, symbol: &str) -> AiExitDecision {
        let system_prompt = load_nemo_entry_prompt();
        let start = Instant::now();

        let result = infer_ollama(
            &self.client,
            &self.ollama_url,
            &self.model, // use entry model (may be same as exit)
            &system_prompt,
            candidate_prompt,
            self.temperature,
            self.max_tokens.max(400), // min 400 — Nemotron needs room for think + answer
            None,
            None, // respect THINK_MODE env
            &self.backend,
            &self.nvidia_key,
            &self.nvidia_url,
            &self.nvidia_model,
            self.think_mode,
        )
        .await;
        let ms = start.elapsed().as_millis() as u64;

        match result {
            Ok(text) => {
                let mut d = parse_entry_confirmation(&text);
                d.latency_ms = ms;
                tracing::info!(
                    "[AI-ENTRY] {} -> {} conf={:.2} reason={} ({ms}ms)",
                    symbol, d.action, d.confidence, d.reason
                );
                d
            }
            Err(e) => {
                tracing::warn!("[AI-ENTRY] {} inference failed ({ms}ms): {e}", symbol);
                AiExitDecision { latency_ms: ms, ..AiExitDecision::default() }
            }
        }
    }

    /// Batch entry confirmation: Nemo sees ALL candidates at once and can compare.
    ///
    /// Sends all candidate data in one prompt so the model can rank them
    /// relative to each other ("SOL is better than ADA right now").
    /// Returns per-symbol BUY/HOLD decisions.
    pub async fn get_batch_entry_confirmation(
        &self,
        candidates: &[(String, String)],  // (symbol, formatted_prompt)
    ) -> std::collections::HashMap<String, AiExitDecision> {
        use std::collections::HashMap;
        let mut results: HashMap<String, AiExitDecision> = HashMap::new();

        if candidates.is_empty() {
            return results;
        }

        let system_prompt = load_nemo_entry_prompt();
        let n = candidates.len();

        // MATRIX_ENTRY_V2: compact batch prompt (action-oriented, no hold bias)
        let mut batch_prompt = format!(
            "{} candidates. Evaluate each independently. BUY when 2+ signals align. Fees=0.52% RT.\n",
            n
        );
        // Cap batch prompt size to avoid context overflows.
        let max_chars: usize = std::env::var("AI_BATCH_MAX_CHARS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(6000);
        let mut added = 0usize;
        let mut truncated = 0usize;
        for (_i, (_sym, prompt)) in candidates.iter().enumerate() {
            let remaining = max_chars.saturating_sub(batch_prompt.len() + 1);
            if remaining < 80 {
                break;
            }
            let p = if prompt.len() > remaining {
                truncated += 1;
                prompt.chars().take(remaining).collect::<String>()
            } else {
                prompt.clone()
            };
            batch_prompt.push_str(&p);
            batch_prompt.push('\n');
            added += 1;
        }
        if added < n {
            tracing::warn!(
                "[AI-BATCH-ENTRY] prompt capped: added {added}/{n} candidates (max_chars={max_chars})"
            );
        }
        if truncated > 0 {
            tracing::warn!(
                "[AI-BATCH-ENTRY] prompt truncated: {truncated} candidate lines trimmed (max_chars={max_chars})"
            );
        }
        batch_prompt.push_str(&format!(
            "Reply {} lines: SYMBOL|BUY/HOLD|CONFIDENCE|REASON",
            added.max(1)
        ));
        tracing::debug!(
            "[AI-BATCH-ENTRY] prompt_chars={} system_chars={} total_chars={}",
            batch_prompt.len(), system_prompt.len(), batch_prompt.len() + system_prompt.len()
        );

        // Scale tokens: ~300 per coin (think block ~150 + answer line ~150), cap at 2048
        let max_tokens = ((n as i32) * 300).min(2048).max(512);

        let start = Instant::now();
        let result = infer_ollama(
            &self.client,
            &self.ollama_url,
            &self.model,
            &system_prompt,
            &batch_prompt,
            self.temperature,
            max_tokens,
            None,
            None, // respect THINK_MODE env
            &self.backend,
            &self.nvidia_key,
            &self.nvidia_url,
            &self.nvidia_model,
            self.think_mode,
        )
        .await;
        let ms = start.elapsed().as_millis() as u64;

        match result {
            Ok(text) => {
                // Parse each line: SYMBOL|ACTION|CONFIDENCE|REASON
                for line in text.lines() {
                    let line = line.trim();
                    if line.is_empty() || !line.contains('|') { continue; }
                    let parts: Vec<&str> = line.splitn(4, '|').collect();
                    if parts.len() < 3 { continue; }

                    let sym = parts[0].trim().to_uppercase();
                    // Clean symbol: remove leading #, digits, spaces, then strip /USD or USD suffix
                    let sym_clean = sym.trim_start_matches(|c: char| c == '#' || c.is_ascii_digit() || c == ' ' || c == '.')
                        .trim()
                        .trim_end_matches("/USD")
                        .trim_end_matches("USD")
                        .trim()
                        .to_string();
                    if sym_clean.is_empty() { continue; }

                    // Detect no-symbol format: BUY|0.76|reason (Nemo skipped symbol for single candidate)
                    let is_action_word = matches!(sym_clean.as_str(), "BUY" | "SELL" | "HOLD");
                    let (final_sym, action, confidence, reason) = if is_action_word && n == 1 {
                        // Re-parse: parts[0]=ACTION, parts[1]=CONFIDENCE, parts[2]=REASON
                        let act = if sym_clean == "BUY" { "BUY" } else { "HOLD" }.to_string();
                        let conf = parts.get(1)
                            .and_then(|s| s.trim().parse::<f64>().ok())
                            .map(|c| c.clamp(0.0, 1.0))
                            .unwrap_or(0.5);
                        let rsn = parts.get(2)
                            .map(|s| s.trim().to_string())
                            .unwrap_or_else(|| "batch_confirm".into());
                        tracing::info!("[AI-BATCH-ENTRY] no-symbol format detected, mapping to '{}'", candidates[0].0);
                        (candidates[0].0.clone(), act, conf, rsn)
                    } else {
                        // Normal: SYMBOL|ACTION|CONFIDENCE|REASON
                        let act = if parts[1].trim().eq_ignore_ascii_case("BUY") {
                            "BUY"
                        } else {
                            "HOLD"
                        }.to_string();
                        let conf = parts.get(2)
                            .and_then(|s| s.trim().parse::<f64>().ok())
                            .map(|c| c.clamp(0.0, 1.0))
                            .unwrap_or(0.5);
                        let rsn = parts.get(3)
                            .map(|s| s.trim().to_string())
                            .unwrap_or_else(|| "batch_confirm".into());
                        (sym_clean, act, conf, rsn)
                    };

                    results.insert(final_sym.clone(), AiExitDecision {
                        action,
                        confidence,
                        reason,
                        source: "ai_batch_entry".into(),
                        latency_ms: ms,
                        scan: None,
                    });
                }

                let confirmed = results.values().filter(|d| d.action == "BUY").count();
                tracing::info!(
                    "[AI-BATCH-ENTRY] {n} candidates → {} confirmed, {} vetoed ({ms}ms)",
                    confirmed, n - confirmed
                );

                // Single-candidate fallback: if Nemo returned a result under wrong symbol,
                // remap it to the actual candidate symbol
                if n == 1 && !results.is_empty() {
                    let expected_sym = &candidates[0].0;
                    if !results.contains_key(expected_sym) {
                        // Nemo used a different symbol name — take whatever she returned
                        if let Some(wrong_key) = results.keys().next().cloned() {
                            if let Some(decision) = results.remove(&wrong_key) {
                                tracing::info!("[AI-BATCH-ENTRY] symbol remap: '{}' → '{}'", wrong_key, expected_sym);
                                results.insert(expected_sym.clone(), decision);
                            }
                        }
                    }
                }

                // JSON fallback: if pipe parser found nothing, try parsing as JSON
                if results.is_empty() {
                    for line in text.lines() {
                        let line = line.trim();
                        if !line.starts_with('{') { continue; }
                        if let Ok(obj) = serde_json::from_str::<serde_json::Value>(line) {
                            let act = obj.get("action")
                                .and_then(|v| v.as_str())
                                .unwrap_or("HOLD")
                                .to_uppercase();
                            let conf = obj.get("confidence")
                                .and_then(|v| v.as_f64())
                                .map(|c| c.clamp(0.0, 1.0))
                                .unwrap_or(0.5);
                            let rsn = obj.get("reason")
                                .and_then(|v| v.as_str())
                                .unwrap_or("json_confirm")
                                .to_string();
                            // Map to candidate symbol (JSON response typically lacks symbol field)
                            let sym = obj.get("symbol")
                                .and_then(|v| v.as_str())
                                .map(|s| s.to_uppercase())
                                .unwrap_or_else(|| if n == 1 { candidates[0].0.clone() } else { String::new() });
                            if sym.is_empty() { continue; }
                            let action = if act == "BUY" { "BUY" } else { "HOLD" }.to_string();
                            tracing::info!("[AI-BATCH-ENTRY] JSON fallback: {sym} {action} conf={conf:.2}");
                            results.insert(sym, AiExitDecision {
                                action,
                                confidence: conf,
                                reason: rsn,
                                source: "ai_batch_entry".into(),
                                latency_ms: ms,
                                scan: None,
                            });
                        }
                    }
                    // Recount after JSON fallback
                    if results.is_empty() {
                        let preview: String = text.chars().take(300).collect();
                        tracing::warn!("[AI-BATCH-ENTRY] No parseable lines! Raw: {preview}");
                    } else {
                        let confirmed = results.values().filter(|d| d.action == "BUY").count();
                        tracing::info!(
                            "[AI-BATCH-ENTRY] JSON fallback: {n} candidates → {} confirmed, {} vetoed ({ms}ms)",
                            confirmed, n - confirmed
                        );
                    }
                }
            }
            Err(e) => {
                tracing::error!("[AI-BATCH-ENTRY] inference failed ({ms}ms): {e}");
                crate::nemo_optimizer::flag_error("entry_ai", &format!("batch inference failed: {e}"));
            }
        }

        // Fill missing with default HOLD (safe — no trade)
        for (sym, _) in candidates {
            results.entry(sym.clone()).or_insert_with(|| AiExitDecision {
                latency_ms: ms,
                ..AiExitDecision::default()
            });
        }

        results
    }

    /// Generic inference for the Nemo self-optimizer.
    /// Uses main model with low temperature for deterministic param tuning.
    pub async fn infer_for_optimizer(
        &self,
        system_prompt: &str,
        user_prompt: &str,
    ) -> Result<String, String> {
        // Optimizer always uses NIM cloud when NVIDIA_OPTIMIZER_MODEL + API key are set,
        // regardless of AI_BACKEND (which controls entry/exit routing to local llama-server).
        let optimizer_backend = if !self.nvidia_optimizer_model.is_empty() && !self.nvidia_key.is_empty() {
            "nvidia_cloud"
        } else {
            &self.backend
        };
        infer_ollama(
            &self.client,
            &self.ollama_url,
            &self.model,
            system_prompt,
            user_prompt,
            self.optimizer_temperature,
            self.optimizer_max_tokens,
            Some(4096),
            Some(true), // optimizer: use /think for deep analysis
            optimizer_backend,
            &self.nvidia_key,
            &self.nvidia_url,
            &self.nvidia_optimizer_model,
            self.think_mode,
        )
        .await
        .map_err(|e| format!("{e}"))
    }

    /// Get exit decision from Nemo for a single open position.
    ///
    /// Uses the exit-specific constitution (nemo_exit_prompt.txt).
    /// Returns HOLD by default on failure (safe — keeps position open).
    pub async fn get_exit_decision(&self, position_prompt: &str, symbol: &str) -> AiExitDecision {
        let system_prompt = load_nemo_exit_prompt();
        let start = Instant::now();

        // Exit decisions: routed to Qwen (CPU) via EXIT_AI_HOST
        // /no_think for speed — prompt has structured decision ladder
        let result = infer_ollama(
            &self.client,
            &self.exit_url,
            &self.exit_model,
            &system_prompt,
            position_prompt,
            self.exit_temperature,
            self.exit_max_tokens,
            None,
            None, // respect THINK_MODE — let her reason
            &self.backend,
            &self.nvidia_key,
            &self.nvidia_url,
            &self.nvidia_model,
            self.think_mode,
        )
        .await;
        let ms = start.elapsed().as_millis() as u64;

        match result {
            Ok(text) => {
                let mut d = parse_exit_decision(&text);
                d.latency_ms = ms;
                tracing::info!(
                    "[AI-EXIT] {} -> {} conf={:.2} reason={} ({ms}ms)",
                    symbol, d.action, d.confidence, d.reason
                );
                d
            }
            Err(e) => {
                tracing::warn!("[AI-EXIT] {} inference failed ({ms}ms): {e}", symbol);
                if !self.exit_url_from_env {
                    let retry = infer_ollama(
                        &self.client,
                        &self.ollama_url,
                        &self.model,
                        &system_prompt,
                        position_prompt,
                        self.exit_temperature,
                        self.exit_max_tokens,
                        None,
                        None,
                        &self.backend,
                        &self.nvidia_key,
                        &self.nvidia_url,
                        &self.nvidia_model,
                        self.think_mode,
                    )
                    .await;
                    let total_ms = start.elapsed().as_millis() as u64;
                    match retry {
                        Ok(text) => {
                            let mut d = parse_exit_decision(&text);
                            d.latency_ms = total_ms;
                            tracing::info!(
                                "[AI-EXIT] {} -> {} conf={:.2} reason={} ({total_ms}ms, fallback)",
                                symbol, d.action, d.confidence, d.reason
                            );
                            return d;
                        }
                        Err(e2) => {
                            tracing::warn!(
                                "[AI-EXIT] {} fallback failed ({total_ms}ms): {e2}",
                                symbol
                            );
                            return AiExitDecision { latency_ms: total_ms, ..AiExitDecision::default() };
                        }
                    }
                }
                AiExitDecision { latency_ms: ms, ..AiExitDecision::default() }
            }
        }
    }

    /// Send a multi-turn message list with tool definitions to the ENTRY model.
    /// Returns either tool calls to execute, or the final text response.
    pub async fn infer_with_tools_entry(
        &self,
        messages: &[serde_json::Value],
        tools: &[serde_json::Value],
        temperature: f32,
        max_tokens: i32,
        think_override: Option<bool>,
    ) -> BridgeResult<InferToolResult> {
        self.infer_with_tools_at(
            &self.ollama_url,
            &self.model,
            messages,
            tools,
            temperature,
            max_tokens,
            think_override,
        )
        .await
    }

    /// Send a multi-turn message list with tool definitions to the EXIT model.
    /// Returns either tool calls to execute, or the final text response.
    pub async fn infer_with_tools_exit(
        &self,
        messages: &[serde_json::Value],
        tools: &[serde_json::Value],
        temperature: f32,
        max_tokens: i32,
        think_override: Option<bool>,
    ) -> BridgeResult<InferToolResult> {
        self.infer_with_tools_at(
            &self.exit_url,
            &self.exit_model,
            messages,
            tools,
            temperature,
            max_tokens,
            think_override,
        )
        .await
    }

    /// Internal: send tool-calling request to a specific model endpoint.
    ///
    /// Uses the OpenAI-compatible `/v1/chat/completions` endpoint with `tools` field.
    /// Only works with AI_BACKEND=openai (llama-server).
    async fn infer_with_tools_at(
        &self,
        base_url: &str,
        model: &str,
        messages: &[serde_json::Value],
        tools: &[serde_json::Value],
        temperature: f32,
        max_tokens: i32,
        think_override: Option<bool>,
    ) -> BridgeResult<InferToolResult> {
        if self.backend != "openai" {
            return Err(AiBridgeError::Http(
                "tool calling requires AI_BACKEND=openai (llama-server)".into(),
            ));
        }

        let think_mode = match think_override {
            Some(v) => v,
            None => self.think_mode,
        };
        // Think mode: modest bump, not 3072 — structured prompts don't need deep reasoning
        let final_max_tokens = if think_mode {
            (max_tokens + 256).min(1024)
        } else {
            max_tokens
        };

        let url = format!("{}/v1/chat/completions", base_url);
        let mut body = serde_json::json!({
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": final_max_tokens,
            "top_p": 0.9,
            "stream": false,
        });

        // Only include tools if non-empty — avoids confusing the model
        if !tools.is_empty() {
            if let Some(obj) = body.as_object_mut() {
                obj.insert(
                    "tools".to_string(),
                    serde_json::Value::Array(tools.to_vec()),
                );
            }
        }

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| AiBridgeError::Http(format!("tool request failed: {e}")))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(AiBridgeError::Http(format!(
                "LLM server returned {status}: {text}"
            )));
        }

        let json: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| AiBridgeError::Http(format!("invalid JSON response: {e}")))?;

        let msg = json
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("message"));

        // Check for tool_calls in the response
        if let Some(tool_calls) = msg.and_then(|m| m.get("tool_calls")).and_then(|tc| tc.as_array())
        {
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
                    // Arguments can be a string (OpenAI format) or already-parsed object
                    let arguments = match args_raw {
                        Some(serde_json::Value::String(s)) => {
                            serde_json::from_str(s).unwrap_or(serde_json::json!({}))
                        }
                        Some(v) if v.is_object() => v.clone(),
                        _ => serde_json::json!({}),
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

        // No tool calls: extract content as final response
        let content = msg
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .unwrap_or("")
            .to_string();

        Ok(InferToolResult::FinalResponse(strip_think_tags(&content)))
    }
}

// ── NVIDIA NIM Cloud API ────────────────────────────────────

/// Call NVIDIA NIM cloud API (OpenAI-compatible) for text completion.
pub(crate) async fn infer_cloud(
    client: &reqwest::Client,
    api_key: &str,
    model: &str,
    system_prompt: &str,
    user_prompt: &str,
    temperature: f32,
    max_tokens: i32,
    base_url: &str,
) -> Result<String, String> {
    let url = format!("{base_url}/chat/completions");
    let body = serde_json::json!({
        "model": model,
        "messages": [
            { "role": "system", "content": system_prompt },
            { "role": "user", "content": user_prompt }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": false
    });
    let resp = client.post(&url)
        .bearer_auth(api_key)
        .json(&body)
        .timeout(std::time::Duration::from_secs(60))
        .send().await
        .map_err(|e| format!("HTTP error: {e}"))?;
    if !resp.status().is_success() {
        let t = resp.text().await.unwrap_or_default();
        return Err(format!("API error: {}", t.chars().take(300).collect::<String>()));
    }
    let json: serde_json::Value = resp.json().await.map_err(|e| format!("JSON: {e}"))?;
    let content = json
        .get("choices").and_then(|c| c.get(0))
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .unwrap_or("")
        .to_string();
    Ok(super::parse::strip_think_tags(&content))
}

// ── LLM HTTP Inference (OpenAI-compatible) ──────────────────

/// Call LLM inference endpoint (OpenAI-compatible: llama-server, vLLM, NVIDIA NIM).
/// `think_override`: Some(true) = force /think, Some(false) = force /no_think, None = use THINK_MODE env
pub(crate) async fn infer_ollama(
    client: &reqwest::Client,
    base_url: &str,
    model: &str,
    system_prompt: &str,
    user_prompt: &str,
    temperature: f32,
    max_tokens: i32,
    _num_ctx: Option<i32>,
    think_override: Option<bool>,
    backend: &str,
    nvidia_key: &str,
    nvidia_url: &str,
    nvidia_model: &str,
    env_think_mode: bool,
) -> BridgeResult<String> {
    // ── Think Mode (Qwen3): /think for deep reasoning, /no_think for fast responses ──
    // NOTE: these are Qwen-specific control tokens — NOT sent to nvidia_cloud (Nemotron)
    let think_mode = match think_override {
        Some(v) => v,
        None => env_think_mode,
    };

    // ── NVIDIA NIM cloud backend (Nemotron) ──
    if backend == "nvidia_cloud" {
        if nvidia_key.is_empty() {
            return Err(AiBridgeError::Http(
                "NVIDIA_API_KEY not set — required for nvidia_cloud backend".into()
            ));
        }
        let cloud_model = if nvidia_model.is_empty() { model } else { nvidia_model };
        let url = format!("{nvidia_url}/chat/completions");

        // Nemotron does not use Qwen /think tokens — send user_prompt as-is.
        // Budget: give enough room for think block + answer; caller controls max_tokens.
        let body = serde_json::json!({
            "model": cloud_model,
            "messages": [
                { "role": "system", "content": system_prompt },
                { "role": "user", "content": user_prompt }
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 0.9,
            "stream": false
        });

        let resp = client
            .post(&url)
            .header("Authorization", format!("Bearer {nvidia_key}"))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| AiBridgeError::Http(format!("NVIDIA cloud request failed: {e}")))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(AiBridgeError::Http(format!(
                "NVIDIA cloud returned {status}: {text}"
            )));
        }

        let json: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| AiBridgeError::Http(format!("NVIDIA cloud invalid JSON: {e}")))?;

        // Strip <think>...</think> blocks — Nemotron reasoning is in content field
        let content = parse_openai_content(&json, false);
        return Ok(strip_think_tags(&content));
    }

    // ── Local Qwen-specific think tokens ──
    let final_prompt = if think_mode {
        format!("{}\n/think", user_prompt)
    } else {
        format!("{}\n/no_think", user_prompt)
    };
    let final_max_tokens = if think_mode {
        (max_tokens + 256).min(1024)
    } else {
        max_tokens
    };

    // ── Default: OpenAI-compatible (llama-server, vLLM, etc.) ──
    let url = format!("{base_url}/v1/chat/completions");
    let body = serde_json::json!({
        "model": model,
        "messages": [
            { "role": "system", "content": system_prompt },
            { "role": "user", "content": &final_prompt }
        ],
        "temperature": temperature,
        "max_tokens": final_max_tokens,
        "top_p": 0.9,
        "stream": false,
        "enable_thinking": think_mode
    });

    let resp = client
        .post(&url)
        .json(&body)
        .send()
        .await
        .map_err(|e| AiBridgeError::Http(format!("request failed: {e}")))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(AiBridgeError::Http(format!(
            "LLM server returned {status}: {text}"
        )));
    }

    let json: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| AiBridgeError::Http(format!("invalid JSON response: {e}")))?;

    let content = parse_openai_content(&json, think_mode);

    if think_mode {
        if let Some(reasoning) = json
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("reasoning_content"))
            .and_then(|c| c.as_str())
            .filter(|s| !s.is_empty())
        {
            let preview: String = reasoning.chars().take(200).collect();
            tracing::debug!("[THINK] reasoning preview: {}...", preview);
        }
    }

    Ok(strip_think_tags(&content))
}

// ── NPU Pre-Scanner ────────────────────────────────────────

/// Calls the Phi-3 lane scanner service (phi3_server.py on port 8084).
/// Returns PASS/REJECT verdict with lane + reason.
pub(crate) async fn npu_scan_coin(
    feats: &serde_json::Value,
) -> super::LaneVerifyResult {
    let host = std::env::var("PHI3_HOST")
        .unwrap_or_else(|_| "http://127.0.0.1:8084".to_string());

    let sym     = feats.get("symbol").and_then(|v| v.as_str()).unwrap_or("?");
    let price   = feats.get("price").and_then(|v| v.as_f64()).unwrap_or(0.0);
    let rsi     = feats.get("rsi").and_then(|v| v.as_f64()).unwrap_or(50.0);
    let trend   = feats.get("trend_score").and_then(|v| v.as_f64()).unwrap_or(0.0) as i32;
    let imb     = feats.get("book_imbalance").and_then(|v| v.as_f64()).unwrap_or(0.0);
    let mom     = feats.get("momentum_score").and_then(|v| v.as_f64()).unwrap_or(0.0);
    let vol     = feats.get("vol_ratio").and_then(|v| v.as_f64()).unwrap_or(1.0);
    let spread  = feats.get("spread_pct").and_then(|v| v.as_f64()).unwrap_or(0.0);
    let regime  = feats.get("regime").and_then(|v| v.as_str()).unwrap_or("unknown");
    let tier    = feats.get("tier").and_then(|v| v.as_str()).unwrap_or("mid");

    let body = serde_json::json!({
        "sym": sym, "price": price, "rsi": rsi, "trend": trend,
        "imb": imb, "mom": mom, "vol": vol, "spread": spread,
        "regime": regime, "tier": tier,
    });

    let client = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build() {
        Ok(c) => c,
        Err(_) => return super::LaneVerifyResult {
            verdict: "PASS".into(), lane: "UNKNOWN".into(),
            recomputed_action: "BUY".into(),
            explanation: "phi3_client_err".into(), latency_ms: 0,
        },
    };

    let t0 = std::time::Instant::now();
    let resp = client.post(format!("{host}/scan"))
        .header("Host", "127.0.0.1")
        .json(&body)
        .send()
        .await;

    match resp {
        Ok(r) => match r.json::<serde_json::Value>().await {
            Ok(v) => {
                let latency_ms = t0.elapsed().as_millis() as u64;
                super::LaneVerifyResult {
                    verdict:           v["verdict"].as_str().unwrap_or("PASS").to_string(),
                    lane:              v["lane"].as_str().unwrap_or("L2").to_string(),
                    recomputed_action: v["action"].as_str().unwrap_or("BUY").to_string(),
                    explanation:       v["reason"].as_str().unwrap_or("").to_string(),
                    latency_ms,
                }
            }
            Err(_) => super::LaneVerifyResult {
                verdict: "PASS".into(), lane: "UNKNOWN".into(),
                recomputed_action: "BUY".into(),
                explanation: "phi3_parse_err".into(), latency_ms: 0,
            },
        },
        Err(_) => super::LaneVerifyResult {
            verdict: "PASS".into(), lane: "UNKNOWN".into(),
            recomputed_action: "BUY".into(),
            explanation: "phi3_unavailable".into(), latency_ms: 0,
        },
    }
}

// ── Utility ─────────────────────────────────────────────────

pub fn now_ts() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}
