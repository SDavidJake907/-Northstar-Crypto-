//! NVIDIA NIM tools — semantic embeddings and cloud inference.

use std::path::Path;

// ── Config ──────────────────────────────────────────────────

pub struct NvidiaToolsConfig {
    pub embed_enabled: bool,
    pub api_key: String,
    pub embed_model: String,
    pub similar_trades_k: usize,
}

impl NvidiaToolsConfig {
    pub fn from_env() -> Self {
        let api_key = std::env::var("NVIDIA_API_KEY").unwrap_or_default();
        let embed_enabled = std::env::var("NVIDIA_EMBED_ENABLED")
            .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE"))
            .unwrap_or(false);
        let embed_model = std::env::var("NVIDIA_EMBED_MODEL")
            .unwrap_or_else(|_| "nvidia/nv-embedqa-e5-v5".into());
        let similar_trades_k = std::env::var("NVIDIA_SIMILAR_TRADES_K")
            .ok().and_then(|v| v.parse().ok()).unwrap_or(5);
        Self { embed_enabled, api_key, embed_model, similar_trades_k }
    }
}

// ── Embedding Record & Store ──────────────────────────────────

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct EmbeddingRecord {
    pub ts: f64,
    pub symbol: String,
    pub text: String,
    pub embedding: Vec<f32>,
}

#[derive(Default)]
pub struct EmbeddingStore {
    records: Vec<EmbeddingRecord>,
    path: Option<std::path::PathBuf>,
}

impl EmbeddingStore {
    pub fn load(path: &Path) -> Self {
        let mut records = Vec::new();
        if let Ok(content) = std::fs::read_to_string(path) {
            for line in content.lines() {
                if let Ok(r) = serde_json::from_str::<EmbeddingRecord>(line) {
                    records.push(r);
                }
            }
        }
        Self { records, path: Some(path.to_path_buf()) }
    }

    pub fn add(&mut self, record: EmbeddingRecord) {
        if let Some(ref p) = self.path {
            if let Ok(line) = serde_json::to_string(&record) {
                let _ = std::fs::OpenOptions::new()
                    .create(true).append(true)
                    .open(p)
                    .and_then(|mut f| { use std::io::Write; writeln!(f, "{line}") });
            }
        }
        self.records.push(record);
    }

    pub fn len(&self) -> usize { self.records.len() }
    pub fn is_empty(&self) -> bool { self.records.is_empty() }

    pub fn find_similar(&self, query: &[f32], k: usize) -> Vec<&EmbeddingRecord> {
        let mut scored: Vec<(f64, &EmbeddingRecord)> = self.records.iter()
            .map(|r| (cosine_similarity(query, &r.embedding), r))
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().take(k).map(|(_, r)| r).collect()
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() { return 0.0; }
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| *x as f64 * *y as f64).sum();
    let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na * nb) }
}

// ── NVIDIA NIM Embedding API ──────────────────────────────────

pub async fn embed_texts(
    client: &reqwest::Client,
    cfg: &NvidiaToolsConfig,
    texts: &[&str],
) -> Result<Vec<Vec<f32>>, String> {
    if !cfg.embed_enabled || cfg.api_key.is_empty() {
        return Err("nvidia embedding disabled or no api key".into());
    }
    let url = "https://integrate.api.nvidia.com/v1/embeddings";
    let body = serde_json::json!({
        "input": texts,
        "model": cfg.embed_model,
        "input_type": "query",
        "encoding_format": "float",
        "truncate": "NONE"
    });
    let resp = client.post(url)
        .bearer_auth(&cfg.api_key)
        .json(&body)
        .timeout(std::time::Duration::from_secs(30))
        .send().await
        .map_err(|e| format!("HTTP error: {e}"))?;
    if !resp.status().is_success() {
        let t = resp.text().await.unwrap_or_default();
        return Err(format!("API error: {}", t.chars().take(200).collect::<String>()));
    }
    let json: serde_json::Value = resp.json().await.map_err(|e| format!("JSON: {e}"))?;
    let data = json.get("data").and_then(|d| d.as_array()).ok_or("no data")?;
    let mut result = Vec::new();
    for item in data {
        let emb = item.get("embedding")
            .and_then(|e| e.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect::<Vec<_>>())
            .unwrap_or_default();
        result.push(emb);
    }
    Ok(result)
}

// ── Prompt Builder ────────────────────────────────────────────

pub fn build_similar_trades_block(records: &[&EmbeddingRecord]) -> String {
    if records.is_empty() { return String::new(); }
    let mut out = String::from("=== SIMILAR PAST TRADES ===\n");
    for r in records {
        out.push_str(&format!("[{}] {}\n", r.symbol, r.text));
    }
    out
}
