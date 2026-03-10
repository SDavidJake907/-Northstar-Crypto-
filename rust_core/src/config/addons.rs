//! Addon configuration — feature toggles and cadence intervals for optional subsystems.

/// Configuration for optional addon subsystems.
#[derive(Clone, Debug)]
pub struct AddonsConfig {
    // Cloud intel
    pub cloud_intel_enabled: bool,
    pub cloud_intel_interval_sec: u64,

    // Atlas sentiment
    pub atlas_sentiment_enabled: bool,
    pub atlas_host: String,

    // Vision scanner (coin rotation)
    pub rotation_enabled: bool,
    pub rotation_interval_sec: u64,
    pub rotation_max_rounds: usize,

    // Mover scanner
    pub movers_enabled: bool,
    pub movers_interval_sec: u64,
    pub max_mover_suggestions: usize,

    // Nemo memory / conscious brain
    #[allow(dead_code)]
    pub memory_enabled: bool,
    pub reflection_interval_sec: f64,

    // NPU bridge
    pub npu_enabled: bool,
    pub npu_dll_path: String,
    pub npu_model_path: String,
    pub npu_device: String,

    // NVIDIA NIM tools (embeddings, reranking)
    pub nvidia_embed_enabled: bool,
    pub nvidia_embed_model: String,
    pub nvidia_rerank_enabled: bool,
    pub nvidia_rerank_model: String,
    pub nvidia_second_brain: bool,
    pub nvidia_second_brain_min_conf: f64,
    pub nvidia_similar_trades_k: usize,
}

impl Default for AddonsConfig {
    fn default() -> Self {
        Self {
            cloud_intel_enabled: false,
            cloud_intel_interval_sec: 900,
            atlas_sentiment_enabled: false,
            atlas_host: "http://127.0.0.1:8083".into(),
            rotation_enabled: false,
            rotation_interval_sec: 900,
            rotation_max_rounds: 6,
            movers_enabled: false,
            movers_interval_sec: 300,
            max_mover_suggestions: 3,
            memory_enabled: false,
            reflection_interval_sec: 14400.0,
            npu_enabled: false,
            npu_dll_path: String::new(),
            npu_model_path: String::new(),
            npu_device: "NPU".into(),
            nvidia_embed_enabled: false,
            nvidia_embed_model: "nvidia/nv-embedqa-e5-v5".into(),
            nvidia_rerank_enabled: false,
            nvidia_rerank_model: "nvidia/llama-3.2-nv-rerankqa-1b-v2".into(),
            nvidia_second_brain: false,
            nvidia_second_brain_min_conf: 0.80,
            nvidia_similar_trades_k: 5,
        }
    }
}

impl AddonsConfig {
    pub fn from_env() -> Self {
        let mut c = Self::default();
        let e = |key: &str| std::env::var(key).ok();
        let eb = |key: &str, def: bool| -> bool {
            e(key)
                .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "on"))
                .unwrap_or(def)
        };
        let legacy = |new_key: &str, old_key: &str| -> Option<String> {
            e(new_key).or_else(|| {
                let v = e(old_key);
                if v.is_some() {
                    tracing::warn!("[CONFIG] Using legacy key {old_key} → migrate to {new_key}");
                }
                v
            })
        };
        let legacy_bool = |new_key: &str, old_key: &str, def: bool| -> bool {
            legacy(new_key, old_key)
                .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "on"))
                .unwrap_or(def)
        };
        let legacy_u64 = |new_key: &str, old_key: &str, def: u64| -> u64 {
            legacy(new_key, old_key)
                .and_then(|v| v.parse().ok())
                .unwrap_or(def)
        };

        c.cloud_intel_enabled = legacy_bool("ADDON_CLOUD_INTEL", "CLOUD_INTEL_ENABLED", c.cloud_intel_enabled);
        c.cloud_intel_interval_sec = legacy_u64("CADENCE_CLOUD_INTEL_SEC", "CLOUD_SENTIMENT_INTERVAL_SEC", c.cloud_intel_interval_sec);

        c.atlas_sentiment_enabled = eb("ADDON_AI3_SENTIMENT", c.atlas_sentiment_enabled);
        if let Some(v) = e("AI3_HOST") { c.atlas_host = v; }

        c.rotation_enabled = legacy_bool("ADDON_ROTATION", "VISION_SCANNER_ENABLED", c.rotation_enabled);
        c.rotation_interval_sec = legacy_u64("CADENCE_ROTATION_SEC", "VISION_SCAN_INTERVAL_SEC", c.rotation_interval_sec);
        if let Some(v) = e("VISION_MAX_ROUNDS").or_else(|| e("ROTATION_MAX_ROUNDS")) {
            if let Ok(i) = v.parse() { c.rotation_max_rounds = i; }
        }

        c.movers_enabled = legacy_bool("ADDON_MOVERS", "MOVER_SCANNER_ENABLED", c.movers_enabled);
        c.movers_interval_sec = legacy_u64("CADENCE_MOVERS_SEC", "MOVER_SCAN_INTERVAL_SEC", c.movers_interval_sec);
        if let Some(v) = e("MAX_MOVER_SUGGESTIONS") {
            if let Ok(i) = v.parse() { c.max_mover_suggestions = i; }
        }

        c.memory_enabled = legacy_bool("ADDON_MEMORY", "NEMO_MEMORY_ENABLED", c.memory_enabled);
        // Nemo memory
        if let Some(v) = e("AI_REFLECTION_INTERVAL_SEC") {
            if let Ok(f) = v.parse() { c.reflection_interval_sec = f; }
        }

        // NPU
        c.npu_enabled = eb("NPU_ENABLED", c.npu_enabled);
        if let Some(v) = e("NPU_DLL_PATH") { c.npu_dll_path = v; }
        if let Some(v) = e("NPU_MODEL_PATH") { c.npu_model_path = v; }
        if let Some(v) = e("NPU_DEVICE") { c.npu_device = v; }

        // NVIDIA NIM tools
        c.nvidia_embed_enabled = eb("NVIDIA_EMBED_ENABLED", c.nvidia_embed_enabled);
        if let Some(v) = e("NVIDIA_EMBED_MODEL") { c.nvidia_embed_model = v; }
        c.nvidia_rerank_enabled = eb("NVIDIA_RERANK_ENABLED", c.nvidia_rerank_enabled);
        if let Some(v) = e("NVIDIA_RERANK_MODEL") { c.nvidia_rerank_model = v; }
        c.nvidia_second_brain = eb("NVIDIA_SECOND_BRAIN", c.nvidia_second_brain);
        if let Some(v) = e("NVIDIA_SECOND_BRAIN_MIN_CONF") {
            if let Ok(f) = v.parse() { c.nvidia_second_brain_min_conf = f; }
        }
        if let Some(v) = e("NVIDIA_SIMILAR_TRADES_K") {
            if let Ok(i) = v.parse() { c.nvidia_similar_trades_k = i; }
        }

        c
    }

    pub fn hot_reload(&mut self) {
        let e = |key: &str| std::env::var(key).ok();
        let legacy_bool = |new_key: &str, old_key: &str, def: bool| -> bool {
            e(new_key).or_else(|| e(old_key))
                .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "on"))
                .unwrap_or(def)
        };
        let legacy_u64 = |new_key: &str, old_key: &str, def: u64| -> u64 {
            e(new_key).or_else(|| e(old_key))
                .and_then(|v| v.parse().ok())
                .unwrap_or(def)
        };

        self.cloud_intel_enabled = legacy_bool("ADDON_CLOUD_INTEL", "CLOUD_INTEL_ENABLED", self.cloud_intel_enabled);
        self.cloud_intel_interval_sec = legacy_u64("CADENCE_CLOUD_INTEL_SEC", "CLOUD_SENTIMENT_INTERVAL_SEC", self.cloud_intel_interval_sec);
        self.rotation_enabled = legacy_bool("ADDON_ROTATION", "VISION_SCANNER_ENABLED", self.rotation_enabled);
        self.movers_enabled = legacy_bool("ADDON_MOVERS", "MOVER_SCANNER_ENABLED", self.movers_enabled);
        self.movers_interval_sec = legacy_u64("CADENCE_MOVERS_SEC", "MOVER_SCAN_INTERVAL_SEC", self.movers_interval_sec);
        self.memory_enabled = legacy_bool("ADDON_MEMORY", "NEMO_MEMORY_ENABLED", self.memory_enabled);
    }
}
