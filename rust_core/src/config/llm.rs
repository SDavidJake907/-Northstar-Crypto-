//! LLM configuration — entry/exit model endpoints, generation parameters, backend selection.

/// LLM endpoint and generation configuration.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct LlmConfig {
    // Backend selection
    pub backend: String, // "openai" (local llama-server) or "nvidia_cloud"

    // Entry model (GPU — Qwen 14B)
    pub entry_host: String,
    pub entry_model: String,
    pub entry_temperature: f32,

    // Exit model (GPU — same server as entry)
    pub exit_host: String,
    pub exit_model: String,
    pub exit_temperature: f32,

    // Generation limits
    pub max_tokens: i32,
    pub batch_size: usize,
    pub max_decision_age_sec: f64,

    // Think mode (enable reasoning tokens)
    pub think_mode: bool,

    // NVIDIA cloud (backup)
    pub nvidia_api_key: String,
    pub nvidia_url: String,
    pub nvidia_model: String,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            backend: "openai".into(),
            entry_host: "http://127.0.0.1:8081".into(),
            entry_model: "qwen2.5-14b".into(),
            entry_temperature: 0.6,
            exit_host: "http://127.0.0.1:8081".into(),
            exit_model: "qwen2.5:14b-instruct".into(),
            exit_temperature: 0.3,
            max_tokens: 2048,
            batch_size: 1,
            max_decision_age_sec: 30.0,
            think_mode: true,
            nvidia_api_key: String::new(),
            nvidia_url: "https://integrate.api.nvidia.com/v1".into(),
            nvidia_model: "nvidia/llama-3.3-nemotron-super-49b-v1.5".into(),
        }
    }
}

#[allow(dead_code)]
impl LlmConfig {
    pub fn from_env() -> Self {
        let mut c = Self::default();
        let e = |key: &str| std::env::var(key).ok();
        let legacy = |new_key: &str, old_key: &str| -> Option<String> {
            e(new_key).or_else(|| {
                let v = e(old_key);
                if v.is_some() {
                    tracing::warn!("[CONFIG] Using legacy key {old_key} → migrate to {new_key}");
                }
                v
            })
        };

        if let Some(v) = legacy("LLM_BACKEND", "AI_BACKEND") {
            c.backend = v;
        }
        if let Some(v) = legacy("LLM_ENTRY_HOST", "OLLAMA_HOST") {
            c.entry_host = v;
        }
        if let Some(v) = legacy("LLM_ENTRY_MODEL", "OLLAMA_MODEL") {
            c.entry_model = v;
        }
        if let Some(v) = legacy("LLM_EXIT_HOST", "EXIT_AI_HOST") {
            c.exit_host = v;
        }
        if let Some(v) = legacy("LLM_EXIT_MODEL", "AI_EXIT_MODEL") {
            c.exit_model = v.clone();
            // If no explicit exit model, use entry model
        }
        if c.exit_model.is_empty() {
            c.exit_model = c.entry_model.clone();
        }

        if let Some(v) = legacy("LLM_ENTRY_TEMP", "MODEL_TEMPERATURE") {
            if let Ok(f) = v.parse() { c.entry_temperature = f; }
        }
        if let Some(v) = legacy("LLM_EXIT_TEMP", "AI_EXIT_TEMPERATURE") {
            if let Ok(f) = v.parse() { c.exit_temperature = f; }
        }
        if let Some(v) = legacy("LLM_MAX_TOKENS", "MODEL_MAX_TOKENS") {
            if let Ok(i) = v.parse() { c.max_tokens = i; }
        }
        if let Some(v) = legacy("LLM_BATCH_SIZE", "AI_BATCH_SIZE") {
            if let Ok(i) = v.parse() { c.batch_size = i; }
        }
        if let Some(v) = legacy("LLM_MAX_DECISION_AGE_SEC", "MAX_DECISION_AGE_SEC") {
            if let Ok(f) = v.parse() { c.max_decision_age_sec = f; }
        }

        // Think mode
        if let Some(v) = e("THINK_MODE") {
            c.think_mode = matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "on");
        }

        // NVIDIA cloud
        if let Some(v) = legacy("LLM_NVIDIA_API_KEY", "NVIDIA_API_KEY") {
            c.nvidia_api_key = v;
        }
        if let Some(v) = legacy("LLM_NVIDIA_URL", "NVIDIA_API_URL") {
            c.nvidia_url = v;
        }
        if let Some(v) = legacy("LLM_NVIDIA_MODEL", "NVIDIA_MODEL") {
            c.nvidia_model = v;
        }

        c
    }

    /// Hot-reload mutable fields (temperatures, tokens, batch size).
    /// Host/model changes are logged but require reconnect.
    pub fn hot_reload(&mut self) {
        let e = |key: &str| std::env::var(key).ok();
        let legacy = |new_key: &str, old_key: &str| -> Option<String> {
            e(new_key).or_else(|| e(old_key))
        };

        if let Some(v) = legacy("LLM_ENTRY_TEMP", "MODEL_TEMPERATURE") {
            if let Ok(f) = v.parse() { self.entry_temperature = f; }
        }
        if let Some(v) = legacy("LLM_EXIT_TEMP", "AI_EXIT_TEMPERATURE") {
            if let Ok(f) = v.parse() { self.exit_temperature = f; }
        }
        if let Some(v) = legacy("LLM_MAX_TOKENS", "MODEL_MAX_TOKENS") {
            if let Ok(i) = v.parse() { self.max_tokens = i; }
        }
        if let Some(v) = legacy("LLM_BATCH_SIZE", "AI_BATCH_SIZE") {
            if let Ok(i) = v.parse() { self.batch_size = i; }
        }
        if let Some(v) = legacy("LLM_MAX_DECISION_AGE_SEC", "MAX_DECISION_AGE_SEC") {
            if let Ok(f) = v.parse() { self.max_decision_age_sec = f; }
        }
        if let Some(v) = e("THINK_MODE") {
            self.think_mode = matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "on");
        }
    }
}
