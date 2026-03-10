//! LLM layer — AI client, prompt builders, response parsers, entry/exit decisions.

pub mod client;
pub mod parse;
pub mod prompts;
pub mod nemo_exit;
pub mod nvidia_tools;

use serde::{Deserialize, Serialize};
use thiserror::Error;

// ── Error Types ────────────────────────────────────────────────────

pub type BridgeResult<T> = std::result::Result<T, AiBridgeError>;

#[derive(Debug, Error)]
pub enum AiBridgeError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("http error: {0}")]
    Http(String),
}

// ── AI Decision (used by execute_entry with synthetic decisions) ──

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AiDecision {
    pub action: String,
    pub confidence: f64,
    pub trend_score: i32,
    pub score: f64,
    pub lane: String,
    pub reason: String,
    pub regime_label: String,
    pub quant_bias: String,
    pub limit_price: Option<f64>,
    pub hold_hours: Option<u32>,
    pub decision_ts: f64,
    pub source: String,
    pub latency_ms: u64,
    /// Decision contrast: second-best action the model considered.
    #[serde(default)]
    pub alt_action: String,
    /// Confidence the model assigned to the alternative action.
    #[serde(default)]
    pub alt_confidence: f64,
    /// Gap between best and second-best confidence (decision clarity).
    #[serde(default)]
    pub margin: f64,
    /// Time contract: minimum hold before exit (unless hard risk).
    #[serde(default)]
    pub min_hold_sec: u32,
    /// Time contract: thesis is stale after this long.
    #[serde(default)]
    pub max_hold_sec: u32,
    /// Time contract: re-evaluate interval if nothing meaningful happens.
    #[serde(default)]
    pub reeval_sec: u32,
}

/// Minimum margin required to act (BUY/SELL). Below this → forced HOLD.
/// 0.52% RT fees mean thin-edge trades are fee donations.
pub const MARGIN_GATE: f64 = 0.15;

impl Default for AiDecision {
    fn default() -> Self {
        Self {
            action: "HOLD".into(),
            confidence: 0.0,
            trend_score: 0,
            score: 0.0,
            lane: "rejected".into(),
            reason: "no_ai_response".into(),
            regime_label: "UNKNOWN".into(),
            quant_bias: "NEUTRAL".into(),
            limit_price: None,
            hold_hours: None,
            decision_ts: 0.0,
            source: "none".into(),
            latency_ms: 0,
            alt_action: "HOLD".into(),
            alt_confidence: 0.0,
            margin: 0.0,
            min_hold_sec: 0,
            max_hold_sec: 0,
            reeval_sec: 0,
        }
    }
}

/// Engine-side time contract defaults by lane/trade type.
/// These are applied when the model doesn't provide time fields,
/// and as clamps when it does (model can't exceed these bounds).
pub struct TimeContract {
    pub min_hold_sec: u32,
    pub max_hold_sec: u32,
    pub reeval_sec: u32,
}

impl TimeContract {
    /// Derive sensible time contract from lane + entry context.
    /// TREND: patient holds. MEAN_REVERT: quick in/out. HOLD: zero.
    pub fn from_lane(lane: &str, action: &str) -> Self {
        if action == "HOLD" {
            return Self { min_hold_sec: 0, max_hold_sec: 0, reeval_sec: 0 };
        }
        match lane {
            // L1 = trending above all EMAs — give it room
            "L1" => Self { min_hold_sec: 1800, max_hold_sec: 43200, reeval_sec: 600 },
            // L3 = moderate trend — moderate patience
            "L3" => Self { min_hold_sec: 900, max_hold_sec: 21600, reeval_sec: 300 },
            // L2 = sideways/compression/mean-revert — quick
            "L2" => Self { min_hold_sec: 120, max_hold_sec: 3600, reeval_sec: 120 },
            // L4 = meme pump/scalp — fast in, fast out
            "L4" => Self { min_hold_sec: 60, max_hold_sec: 1800, reeval_sec: 60 },
            // rejected or unknown — conservative
            _ => Self { min_hold_sec: 300, max_hold_sec: 7200, reeval_sec: 300 },
        }
    }

    /// Absolute engine limits (model can't go outside these).
    pub const ABS_MIN_HOLD: u32 = 60;       // 1 minute minimum
    pub const ABS_MAX_HOLD: u32 = 86400;     // 24 hours maximum
    pub const ABS_MIN_REEVAL: u32 = 30;      // 30s minimum reeval
    pub const ABS_MAX_REEVAL: u32 = 900;     // 15m maximum reeval

    /// Clamp model-proposed values to engine bounds.
    pub fn clamp_model(model_min: u32, model_max: u32, model_reeval: u32, lane: &str, action: &str) -> Self {
        let defaults = Self::from_lane(lane, action);
        if action == "HOLD" {
            return defaults;
        }
        // If model proposed 0, use defaults. Otherwise clamp to absolute bounds.
        let min_h = if model_min == 0 { defaults.min_hold_sec }
            else { model_min.clamp(Self::ABS_MIN_HOLD, defaults.max_hold_sec) };
        let max_h = if model_max == 0 { defaults.max_hold_sec }
            else { model_max.clamp(min_h, Self::ABS_MAX_HOLD) };
        let reeval = if model_reeval == 0 { defaults.reeval_sec }
            else { model_reeval.clamp(Self::ABS_MIN_REEVAL, Self::ABS_MAX_REEVAL) };
        Self { min_hold_sec: min_h, max_hold_sec: max_h, reeval_sec: reeval }
    }
}

// ── AI Exit Decision (used by nemo_exit_check) ──

/// Parsed SCAN: suggestion from exit model response (optional second line).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScanSuggestion {
    pub add: Vec<String>,
    pub remove: Vec<String>,
    pub reason: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AiExitDecision {
    pub action: String,      // "HOLD" or "SELL"
    pub confidence: f64,
    pub reason: String,
    pub source: String,
    pub latency_ms: u64,
    /// Optional coin swap suggestion parsed from SCAN: line
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scan: Option<ScanSuggestion>,
}

impl Default for AiExitDecision {
    fn default() -> Self {
        Self {
            action: "HOLD".into(),
            confidence: 0.0,
            reason: "no_exit_response".into(),
            source: "none".into(),
            latency_ms: 0,
            scan: None,
        }
    }
}

// ── Tool-Calling Types (used by exit decision tool loop) ──

#[derive(Debug, Clone)]
pub struct LlmToolCall {
    pub id: String,
    pub function_name: String,
    pub arguments: serde_json::Value,
}

#[derive(Debug)]
pub enum InferToolResult {
    ToolCalls(Vec<LlmToolCall>),
    FinalResponse(String),
}

// ── NPU Scan Result ──

/// Result from NPU coin scan (pre-filter before GPU Qwen)
#[derive(Debug, Clone)]
pub struct LaneVerifyResult {
    pub verdict: String,       // "PASS" or "REJECT"
    pub lane: String,          // "L1", "L2", "L3", "NONE"
    pub recomputed_action: String,
    pub explanation: String,
    pub latency_ms: u64,
}

impl Default for LaneVerifyResult {
    fn default() -> Self {
        Self {
            verdict: "REJECT".into(),
            lane: "NONE".into(),
            recomputed_action: "HOLD".into(),
            explanation: "scanner_unavailable".into(),
            latency_ms: 0,
        }
    }
}

pub use client::AiBridge;
pub(crate) use client::{infer_cloud, npu_scan_coin, now_ts};
pub(crate) use parse::parse_exit_decision;
pub(crate) use prompts::{
    load_nemo_exit_prompt, load_nemo_entry_prompt,
    load_nemo_meme_entry_prompt, load_nemo_meme_exit_prompt,
    load_nemo_review_prompt, load_prompt_from_paths,
};

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::parse::*;
    use super::*;

    // ── Pipe format tests (PRIMARY) ──

    #[test]
    fn pipe_basic_buy() {
        let text = "BUY|0.72|L1|3|trend+3 momentum rising";
        let d = parse_decision(text).unwrap();
        assert_eq!(d.action, "BUY");
        assert!((d.confidence - 0.72).abs() < 1e-6);
        assert_eq!(d.lane, "L1");
        assert_eq!(d.hold_hours, Some(3));
        assert_eq!(d.source, "qwen_pipe");
    }

    #[test]
    fn pipe_hold_no_hours() {
        let text = "HOLD|0.60|L3|0|mixed signals low conviction";
        let d = parse_decision(text).unwrap();
        assert_eq!(d.action, "HOLD");
        assert!(d.hold_hours.is_none());
    }

    #[test]
    fn pipe_sell() {
        let text = "SELL|0.85|L1|0|crash regime momentum negative";
        let d = parse_decision(text).unwrap();
        assert_eq!(d.action, "SELL");
        assert!((d.confidence - 0.85).abs() < 1e-6);
    }

    #[test]
    fn pipe_with_extra_whitespace() {
        let text = "  BUY | 0.78 | L2 | 2 | ema spread positive bb compression  ";
        let d = parse_decision(text).unwrap();
        assert_eq!(d.action, "BUY");
        assert!((d.confidence - 0.78).abs() < 1e-6);
        assert_eq!(d.lane, "L2");
        assert_eq!(d.hold_hours, Some(2));
    }

    #[test]
    fn pipe_multiline_takes_first() {
        let text = "Here is my analysis:\nBUY|0.70|L1|2|trend positive\nExtra text here";
        let d = parse_decision(text).unwrap();
        assert_eq!(d.action, "BUY");
    }

    #[test]
    fn pipe_minimal_two_fields() {
        let text = "HOLD|0.55";
        let d = parse_decision(text).unwrap();
        assert_eq!(d.action, "HOLD");
        assert!((d.confidence - 0.55).abs() < 1e-6);
    }

    #[test]
    fn pipe_clamps_confidence() {
        let text = "BUY|1.50|L1|3|overconfident";
        let d = parse_decision(text).unwrap();
        assert_eq!(d.confidence, 1.0);
    }

    #[test]
    fn pipe_invalid_action_becomes_hold() {
        let text = "MAYBE|0.60|L1|0|unsure";
        let d = parse_decision(text).unwrap();
        assert_eq!(d.action, "HOLD");
    }

    #[test]
    fn pipe_reason_with_pipes() {
        // Reason field may contain pipes — splitn(5,..) keeps them
        let text = "BUY|0.72|L1|3|trend+3 | momentum > 0 | regime trending";
        let d = parse_decision(text).unwrap();
        assert!(d.reason.contains("momentum"));
    }

    #[test]
    fn pipe_missing_fields_format() {
        let text = "HOLD|0.50|rejected|0|MISSING_FIELDS";
        let d = parse_decision(text).unwrap();
        assert_eq!(d.action, "HOLD");
        assert_eq!(d.lane, "rejected");
    }

    // ── JSON fallback tests ──

    #[test]
    fn json_fallback_basic() {
        let text = r#"{"action":"BUY","confidence":0.72,"trend_score":3,"score":0.62,"lane":"L1","reason":"ok"}"#;
        let d = parse_decision(text).unwrap();
        assert_eq!(d.action, "BUY");
        assert_eq!(d.source, "qwen_json_fallback");
    }

    #[test]
    fn json_fallback_clamps() {
        let text = r#"{"action":"SELL","confidence":2.0,"trend_score":9,"score":-2.0,"lane":"L9","reason":"x"}"#;
        let d = parse_decision(text).unwrap();
        assert_eq!(d.action, "SELL");
        assert_eq!(d.confidence, 1.0);
        assert_eq!(d.score, -1.0);
        assert_eq!(d.lane, "rejected");
    }

    #[test]
    fn json_fallback_extracts_from_messy() {
        let text = r#"analysis: {"action":"HOLD","confidence":0.5,"trend_score":0,"score":0.0,"lane":"rejected","reason":"x"} done"#;
        let d = parse_decision(text).unwrap();
        assert_eq!(d.action, "HOLD");
    }

    #[test]
    fn json_fallback_trend_score_string() {
        let raw = r#"{"action":"BUY","confidence":0.77,"trend_score":"+5","score":0.77,"lane":"L1","reason":"ok"}"#;
        let d = parse_decision(raw).unwrap();
        assert_eq!(d.action, "BUY");
        assert_eq!(d.trend_score, 5);
    }

    #[test]
    fn json_fallback_truncated_mid_reason() {
        let raw = r#"{"action":"BUY","confidence":0.77,"trend_score":5,"score":0.77,"lane":"L1","hold_hours":3,"reason":"agree math — strong trend (+5) and rising momentum (1.000), hold for m"#;
        let d = parse_decision(raw).unwrap();
        assert_eq!(d.action, "BUY");
    }

    #[test]
    fn json_fallback_trailing_backtick() {
        let raw = "{\n  \"action\": \"BUY\",\n  \"confidence\": 0.78,\n  \"trend_score\": 5,\n  \"score\": 0.78,\n  \"lane\": \"L1\",\n  \"hold_hours\": 4,\n  \"reason\": \"agree math\"\n}\n`";
        let d = parse_decision(raw).unwrap();
        assert_eq!(d.action, "BUY");
    }

    #[test]
    fn json_fallback_null_action() {
        let raw = r#"{"confidence":0.5,"trend_score":0,"score":0.0}"#;
        let d = parse_decision(raw).unwrap();
        assert_eq!(d.action, "HOLD");
    }

    // ── General tests ──

    #[test]
    fn default_decision_is_hold() {
        let d = AiDecision::default();
        assert_eq!(d.action, "HOLD");
        assert_eq!(d.confidence, 0.0);
        assert_eq!(d.source, "none");
    }

    #[test]
    fn bare_word_fallback() {
        let d = parse_decision("BUY").unwrap();
        assert_eq!(d.action, "BUY");
        assert_eq!(d.source, "qwen_bare");
    }

    #[test]
    fn sanitize_strips_markdown_fences() {
        let raw = "```json\n{\"action\":\"HOLD\",\"confidence\":0.7}\n```";
        let clean = sanitize_model_json(raw);
        assert!(clean.starts_with('{'));
        assert!(clean.ends_with('}'));
    }

    #[test]
    fn sanitize_strips_trailing_commas() {
        let raw = r#"{"action":"BUY","confidence":0.9,"reason":"ok",}"#;
        let d = parse_decision(raw).unwrap();
        assert_eq!(d.action, "BUY");
    }

    #[test]
    fn sanitize_preserves_comments_inside_strings() {
        let raw = r#"{"action":"HOLD","confidence":0.5,"reason":"price // stabilizing"}"#;
        let d = parse_decision(raw).unwrap();
        assert!(d.reason.contains("//"));
    }
}
