//! Response parsing — JSON extraction, pipe format parsing, model output sanitization.

use serde::Deserialize;
use super::{AiDecision, AiExitDecision, ScanSuggestion};

/// Parse `choices[0].message.content` from an OpenAI-compatible JSON response.
pub(crate) fn parse_openai_content(json: &serde_json::Value, skip_reasoning_fallback: bool) -> String {
    let msg = json
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("message"));
    msg.and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .or_else(|| {
            if skip_reasoning_fallback {
                None
            } else {
                msg.and_then(|m| m.get("reasoning_content"))
                    .and_then(|c| c.as_str())
                    .map(|s| s.trim())
                    .filter(|s| !s.is_empty())
            }
        })
        .unwrap_or("")
        .to_string()
}

pub(crate) fn parse_exit_decision(text: &str) -> AiExitDecision {
    // First, extract any SCAN: suggestion (optional second line)
    let scan = parse_scan_line(text);

    // Try pipe format first: ACTION|CONFIDENCE|REASON_CODES
    let pipe_line = text.lines()
        .map(|l| l.trim())
        .find(|l| l.contains('|') && !l.starts_with("SCAN:") && !l.is_empty());
    if let Some(line) = pipe_line {
        let parts: Vec<&str> = line.splitn(3, '|').collect();
        if parts.len() >= 2 {
            let action = if parts[0].trim().eq_ignore_ascii_case("SELL") {
                "SELL"
            } else {
                "HOLD"
            }.to_string();
            let confidence = parts.get(1)
                .and_then(|s| s.trim().parse::<f64>().ok())
                .map(|c| clamp(c, 0.0, 1.0))
                .unwrap_or(0.5);
            let reason = parts.get(2)
                .map(|s| s.trim().to_string())
                .unwrap_or_else(|| "pipe_exit".to_string());
            return AiExitDecision {
                action,
                confidence,
                reason,
                source: "ai_exit_pipe".into(),
                latency_ms: 0,
                scan,
            };
        }
    }

    // JSON fallback
    let sanitized = sanitize_model_json(text);
    let fixed = fix_truncated_json(&sanitized);
    let v = serde_json::from_str::<serde_json::Value>(&fixed)
        .ok()
        .or_else(|| extract_decision_json(&fixed));
    if let Some(v) = v {
        let action = v.get("action")
            .and_then(|a| a.as_str())
            .map(|a| if a.eq_ignore_ascii_case("SELL") { "SELL" } else { "HOLD" })
            .unwrap_or("HOLD")
            .to_string();
        let confidence = v.get("confidence")
            .and_then(|c| c.as_f64())
            .map(|c| clamp(c, 0.0, 1.0))
            .unwrap_or(0.5);
        let reason = v.get("reason")
            .and_then(|r| r.as_str())
            .unwrap_or("no_reason")
            .to_string();
        return AiExitDecision {
            action,
            confidence,
            reason,
            source: "ai_exit_json".into(),
            latency_ms: 0,
            scan,
        };
    }

    // Bare word fallback
    let upper = text.trim().to_uppercase();
    if upper.starts_with("SELL") {
        AiExitDecision {
            action: "SELL".into(),
            confidence: 0.65,
            reason: "bare_word_sell".into(),
            source: "ai_exit_bare".into(),
            latency_ms: 0,
            scan,
        }
    } else {
        AiExitDecision { scan, ..AiExitDecision::default() }
    }
}

#[allow(dead_code)]
pub(crate) fn parse_entry_confirmation(text: &str) -> AiExitDecision {
    // Parse pipe format: ACTION|CONFIDENCE|REASON
    let pipe_line = text.lines()
        .map(|l| l.trim())
        .find(|l| l.contains('|') && !l.is_empty());
    if let Some(line) = pipe_line {
        let parts: Vec<&str> = line.splitn(3, '|').collect();
        if parts.len() >= 2 {
            let action = if parts[0].trim().eq_ignore_ascii_case("BUY") {
                "BUY"
            } else {
                "HOLD"
            }.to_string();
            let confidence = parts.get(1)
                .and_then(|s| s.trim().parse::<f64>().ok())
                .map(|c| clamp(c, 0.0, 1.0))
                .unwrap_or(0.5);
            let reason = parts.get(2)
                .map(|s| s.trim().to_string())
                .unwrap_or_else(|| "entry_confirm".to_string());
            return AiExitDecision {
                action,
                confidence,
                reason,
                source: "ai_entry_pipe".into(),
                latency_ms: 0,
                scan: None,
            };
        }
    }
    // Default: HOLD (safe — don't enter)
    AiExitDecision::default()
}

/// Parse optional SCAN: line from exit model response.
/// Format: `SCAN:{"add":["GRT"],"remove":["TRX"],"reason":"..."}`
pub(crate) fn parse_scan_line(text: &str) -> Option<ScanSuggestion> {
    for line in text.lines() {
        let trimmed = line.trim();
        if let Some(json_str) = trimmed.strip_prefix("SCAN:") {
            match serde_json::from_str::<ScanSuggestion>(json_str) {
                Ok(mut scan) => {
                    // Normalize symbols to uppercase
                    scan.add = scan.add.iter().map(|s| s.trim().to_uppercase()).collect();
                    scan.remove = scan.remove.iter().map(|s| s.trim().to_uppercase()).collect();
                    // Never remove BTC or ETH
                    scan.remove.retain(|s| s != "BTC" && s != "ETH");
                    tracing::info!(
                        "[SCAN] Parsed: add={:?} remove={:?} reason={}",
                        scan.add, scan.remove, scan.reason
                    );
                    return Some(scan);
                }
                Err(e) => {
                    tracing::warn!("[SCAN] Failed to parse SCAN line: {e} — raw: {json_str}");
                }
            }
        }
    }
    None
}

#[allow(dead_code)]
pub(crate) fn parse_decision(text: &str) -> Option<AiDecision> {
    // ── Primary: pipe format  ACTION|CONF|LANE|HOLD_H|REASON ──
    if let Some(d) = parse_pipe_format(text) {
        return Some(d);
    }

    // ── Fallback: JSON (in case model returns JSON anyway) ──
    let sanitized = sanitize_model_json(text);
    let fixed = fix_truncated_json(&sanitized);
    let v = serde_json::from_str::<serde_json::Value>(&fixed)
        .ok()
        .or_else(|| extract_decision_json(&fixed));
    if let Some(v) = v {
        if let Ok(raw) = serde_json::from_value::<AiDecisionRaw>(v) {
            let action_str = raw.action.as_deref().unwrap_or("HOLD");
            let action = parse_action(action_str);
            let confidence = if raw.confidence < 0.01 && action != "HOLD" {
                // Model returned BUY/SELL with near-zero confidence — treat as low-confidence
                0.30
            } else {
                clamp(raw.confidence, 0.0, 1.0)
            };
            let trend_score = raw.trend_score.round() as i32;
            let score = clamp(raw.score, -1.0, 1.0);
            let lane = normalize_lane(raw.lane.as_deref().unwrap_or("rejected"));
            let reason = raw.reason.unwrap_or_else(|| "no_reason".to_string());
            let quant_bias = match action.as_str() {
                "BUY" => "LONG",
                "SELL" => "SHORT",
                _ => "NEUTRAL",
            }
            .to_string();
            let hold_hours = if raw.hold_hours > 0.0 {
                Some((raw.hold_hours.round() as u32).clamp(1, 4))
            } else {
                None
            };

            // Decision contrast: extract alt fields
            let alt_action = raw.alt_action.map(|a| parse_action(&a)).unwrap_or_else(|| "HOLD".into());
            let alt_confidence = clamp(raw.alt_confidence, 0.0, 1.0);
            let margin_raw = raw.margin;
            // If model reported margin, use it; otherwise compute from confidence gap
            let margin = if margin_raw > 0.001 {
                clamp(margin_raw, 0.0, 1.0)
            } else if alt_confidence > 0.001 {
                (confidence - alt_confidence).max(0.0)
            } else {
                // Model didn't provide contrast — treat as unknown, allow action
                1.0
            };

            // MARGIN ADVISORY: log thin-edge warning but let AI decide
            let (action, reason, quant_bias) = if action != "HOLD" && margin < super::MARGIN_GATE && margin < 0.99 {
                tracing::info!(
                    "[MARGIN-ADVISORY] {} thin edge: margin={:.3} < {:.2} (alt={}@{:.2}) — allowing AI decision",
                    action, margin, super::MARGIN_GATE, alt_action, alt_confidence,
                );
                (action, format!("thin_edge:{:.3} {}", margin, reason), quant_bias)
            } else {
                (action, reason, quant_bias)
            };

            // Time contract: model proposes, engine clamps
            let tc = super::TimeContract::clamp_model(
                raw.min_hold_sec as u32,
                raw.max_hold_sec as u32,
                raw.reeval_sec as u32,
                &lane,
                &action,
            );

            return Some(AiDecision {
                action, confidence, trend_score, score, lane, reason,
                regime_label: "UNKNOWN".into(), quant_bias, limit_price: None,
                hold_hours, decision_ts: 0.0, source: "qwen_json_fallback".into(),
                latency_ms: 0,
                alt_action, alt_confidence, margin,
                min_hold_sec: tc.min_hold_sec,
                max_hold_sec: tc.max_hold_sec,
                reeval_sec: tc.reeval_sec,
            });
        }
    }

    // ── Last resort: bare action word ──
    parse_bare_action(text)
}

/// Parse pipe-delimited format: ACTION|CONF|LANE|HOLD_H|REASON
/// Example: BUY|0.72|L1|3|trend+3 momentum rising
#[allow(dead_code)]
fn parse_pipe_format(text: &str) -> Option<AiDecision> {
    // Take the first non-empty line that contains a pipe
    let line = text.lines()
        .map(|l| l.trim())
        .find(|l| l.contains('|') && !l.is_empty())?;

    let parts: Vec<&str> = line.splitn(5, '|').collect();
    if parts.len() < 2 {
        return None;
    }

    let action = parse_action(parts[0].trim());
    let confidence = parts.get(1)
        .and_then(|s| s.trim().parse::<f64>().ok())
        .map(|c| clamp(c, 0.0, 1.0))
        .unwrap_or(if action == "HOLD" { 0.60 } else { 0.65 });
    let lane = parts.get(2)
        .map(|s| normalize_lane(s.trim()))
        .unwrap_or_else(|| "rejected".to_string());
    let hold_hours_raw = parts.get(3)
        .and_then(|s| s.trim().parse::<f64>().ok())
        .unwrap_or(0.0);
    let hold_hours = if hold_hours_raw > 0.0 {
        Some((hold_hours_raw.round() as u32).clamp(1, 4))
    } else {
        None
    };
    let reason = parts.get(4)
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "pipe_format".to_string());

    let quant_bias = match action.as_str() {
        "BUY" => "LONG",
        "SELL" => "SHORT",
        _ => "NEUTRAL",
    }.to_string();

    // Time contract from lane defaults (pipe format doesn't carry time fields)
    let tc = super::TimeContract::from_lane(&lane, &action);

    Some(AiDecision {
        action,
        confidence,
        trend_score: 0, // not needed in pipe format — engine has it
        score: confidence, // mirror confidence as score
        lane,
        reason,
        regime_label: "UNKNOWN".into(),
        quant_bias,
        limit_price: None,
        hold_hours,
        decision_ts: 0.0,
        source: "qwen_pipe".into(),
        latency_ms: 0,
        alt_action: "HOLD".into(),
        alt_confidence: 0.0,
        margin: 1.0, // pipe format lacks contrast data — don't gate
        min_hold_sec: tc.min_hold_sec,
        max_hold_sec: tc.max_hold_sec,
        reeval_sec: tc.reeval_sec,
    })
}

#[allow(dead_code)]
fn parse_bare_action(text: &str) -> Option<AiDecision> {
    let trimmed = text.trim().to_uppercase();
    let action = match trimmed.as_str() {
        "BUY" => "BUY",
        "SELL" => "SELL",
        "HOLD" => "HOLD",
        _ => return None,
    };
    Some(AiDecision {
        action: action.to_string(),
        confidence: 0.70,
        trend_score: 0,
        score: 0.0,
        lane: "rejected".to_string(),
        reason: "bare_word_fallback".to_string(),
        source: "qwen_bare".into(),
        ..AiDecision::default()
    })
}

#[allow(dead_code)]
fn parse_action(action: &str) -> String {
    match action.to_uppercase().as_str() {
        "BUY" => "BUY".to_string(),
        "SELL" => "SELL".to_string(),
        _ => "HOLD".to_string(),
    }
}

#[allow(dead_code)]
fn normalize_lane(lane: &str) -> String {
    match lane.to_uppercase().as_str() {
        "L1" | "L1_MOMENTUM" => "L1".to_string(),
        "L2" | "L2_COMPRESSION" => "L2".to_string(),
        "L3" | "L3_TREND" => "L3".to_string(),
        "L4" | "L4_MEME" | "L4_PUMP" => "L4".to_string(),
        _ => "rejected".to_string(),
    }
}

/// Extract first JSON object from messy model output.
fn extract_decision_json(text: &str) -> Option<serde_json::Value> {
    for (i, _) in text.match_indices('{') {
        let slice = &text[i..];
        for (j, _) in slice.match_indices('}').rev() {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&slice[..=j]) {
                if v.is_object() {
                    return Some(v);
                }
            }
        }
    }
    None
}

/// Extract all JSON objects from batch response text, each with a "symbol" field.
#[allow(dead_code)]
pub(crate) fn extract_all_decisions(text: &str) -> Vec<(String, AiDecision)> {
    let mut results = Vec::new();
    let trimmed = text.trim();

    // Try JSON array first: [{...}, {...}, ...]
    if trimmed.starts_with('[') {
        if let Ok(arr) = serde_json::from_str::<Vec<serde_json::Value>>(trimmed) {
            for v in &arr {
                if let Ok(json_str) = serde_json::to_string(v) {
                    if let Some(d) = try_parse_with_symbol(&json_str) {
                        results.push(d);
                    }
                }
            }
            if !results.is_empty() {
                return results;
            }
        }
    }

    // Try nested dict: {"SYM": {"action":...}, "SYM2": {"action":...}}
    if trimmed.starts_with('{') {
        if let Ok(map) = serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(trimmed) {
            // Check if values are objects with "action" field (nested dict format)
            let is_nested = map.values().any(|v| v.get("action").is_some());
            if is_nested {
                for (sym, v) in &map {
                    // Inject symbol into the inner object and try to parse
                    if let serde_json::Value::Object(mut inner) = v.clone() {
                        inner.insert("symbol".to_string(), serde_json::Value::String(sym.clone()));
                        if let Ok(json_str) = serde_json::to_string(&inner) {
                            if let Some(d) = try_parse_with_symbol(&json_str) {
                                results.push(d);
                            }
                        }
                    }
                }
                if !results.is_empty() {
                    return results;
                }
            }
        }
    }

    // Try line-by-line (one JSON per line)
    for line in text.lines() {
        let line = line.trim().trim_start_matches(',').trim();
        if line.starts_with('{') {
            // Strip trailing comma if present
            let clean = line.trim_end_matches(',');
            if let Some(d) = try_parse_with_symbol(clean) {
                results.push(d);
            }
        }
    }

    // If line-by-line found nothing, scan for all JSON objects
    if results.is_empty() {
        let mut search_from = 0;
        while search_from < text.len() {
            if let Some(start) = text[search_from..].find('{') {
                let abs_start = search_from + start;
                // Find matching closing brace
                let mut depth = 0;
                let mut end = abs_start;
                for (i, ch) in text[abs_start..].char_indices() {
                    match ch {
                        '{' => depth += 1,
                        '}' => {
                            depth -= 1;
                            if depth == 0 {
                                end = abs_start + i;
                                break;
                            }
                        }
                        _ => {}
                    }
                }
                if depth == 0 && end > abs_start {
                    let candidate = &text[abs_start..=end];
                    if let Some(d) = try_parse_with_symbol(candidate) {
                        results.push(d);
                    }
                    search_from = end + 1;
                } else {
                    search_from = abs_start + 1;
                }
            } else {
                break;
            }
        }
    }

    results
}

/// Try to parse a JSON string as a decision with a "symbol" field.
#[allow(dead_code)]
fn try_parse_with_symbol(text: &str) -> Option<(String, AiDecision)> {
    let v: serde_json::Value = serde_json::from_str(text).ok()?;
    let symbol = v.get("symbol").and_then(|s| s.as_str())?.to_uppercase();
    let decision = parse_decision(text)?;
    Some((symbol, decision))
}

pub(crate) fn clamp(v: f64, lo: f64, hi: f64) -> f64 {
    if v < lo { lo } else if v > hi { hi } else { v }
}

#[allow(dead_code)]
pub(crate) fn env_bool(key: &str, default: bool) -> bool {
    match std::env::var(key) {
        Ok(v) => matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON"),
        Err(_) => default,
    }
}

/// Pre-process raw model output: strip markdown fences, JS comments, trailing commas.
/// Then fix math expressions in score fields so the result is valid JSON.
pub(crate) fn sanitize_model_json(text: &str) -> String {
    // Step 0: Strip <think>...</think> blocks (DeepSeek R1 reasoning chains)
    let no_think = strip_think_tags(text);

    // Step 1: Strip markdown code fences (```json ... ``` or ``` ... ```)
    let stripped = strip_markdown_fences(&no_think);

    // Step 2: Strip single-line JS/C++ comments ( // ... ) outside of strings
    let no_comments = strip_json_comments(&stripped);

    // Step 3: Remove trailing commas before } or ] (e.g. {"a":1,} → {"a":1})
    let no_trailing = strip_trailing_commas(&no_comments);

    // Step 4: Fix math expressions in score/trend_score fields
    fix_score_math(&no_trailing)
}

/// Strip <think>...</think> blocks from DeepSeek R1 reasoning models.
pub(crate) fn strip_think_tags(text: &str) -> String {
    let mut result = text.to_string();
    while let Some(start) = result.find("<think>") {
        if let Some(end) = result.find("</think>") {
            result = format!("{}{}", &result[..start], &result[end + 8..]);
        } else {
            // Unclosed <think> — strip everything from <think> onward, take what's before
            result = result[..start].to_string();
            break;
        }
    }
    result.trim().to_string()
}

/// Strip ```json / ``` fences — extract only the content between them.
fn strip_markdown_fences(text: &str) -> String {
    let trimmed = text.trim();
    // Match opening fence: ```json or ```JSON or just ```
    if let Some(fence_start) = trimmed.find("```") {
        // Find end of opening fence line
        let after_fence = &trimmed[fence_start + 3..];
        let content_start = after_fence.find('\n').map(|i| fence_start + 3 + i + 1).unwrap_or(fence_start + 3);
        // Find closing fence
        let content = &trimmed[content_start..];
        if let Some(close) = content.find("```") {
            return content[..close].trim().to_string();
        }
        // No closing fence — take everything after the opening
        return content.trim().to_string();
    }
    trimmed.to_string()
}

/// Strip single-line comments (// ...) that are NOT inside JSON string values.
fn strip_json_comments(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut in_string = false;
    let mut escape_next = false;
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        if escape_next {
            result.push(chars[i]);
            escape_next = false;
            i += 1;
            continue;
        }

        if chars[i] == '\\' && in_string {
            result.push(chars[i]);
            escape_next = true;
            i += 1;
            continue;
        }

        if chars[i] == '"' {
            in_string = !in_string;
            result.push(chars[i]);
            i += 1;
            continue;
        }

        if !in_string && chars[i] == '/' && i + 1 < chars.len() && chars[i + 1] == '/' {
            // Skip to end of line
            while i < chars.len() && chars[i] != '\n' {
                i += 1;
            }
            continue;
        }

        result.push(chars[i]);
        i += 1;
    }
    result
}

/// Remove trailing commas before } or ] (common LLM mistake).
fn strip_trailing_commas(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut in_string = false;
    let mut escape_next = false;
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        if escape_next {
            result.push(chars[i]);
            escape_next = false;
            i += 1;
            continue;
        }
        if chars[i] == '\\' && in_string {
            result.push(chars[i]);
            escape_next = true;
            i += 1;
            continue;
        }
        if chars[i] == '"' {
            in_string = !in_string;
            result.push(chars[i]);
            i += 1;
            continue;
        }

        if !in_string && chars[i] == ',' {
            // Look ahead past whitespace for } or ]
            let mut j = i + 1;
            while j < chars.len() && (chars[j] == ' ' || chars[j] == '\n' || chars[j] == '\r' || chars[j] == '\t') {
                j += 1;
            }
            if j < chars.len() && (chars[j] == '}' || chars[j] == ']') {
                // Skip the trailing comma
                i += 1;
                continue;
            }
        }

        result.push(chars[i]);
        i += 1;
    }
    result
}

/// Fix math expressions in "score" and "trend_score" fields.
/// E.g. `"score":0.30* -5 + 0.22*0` → `"score":0.0`
fn fix_score_math(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let mut ci = 0;

    while ci < chars.len() {
        let (byte_pos, _) = chars[ci];
        let remaining = &text[byte_pos..];

        let is_score_field = remaining.starts_with("\"score\":")
            || remaining.starts_with("\"trend_score\":");

        if is_score_field {
            if let Some(colon_off) = remaining.find(':') {
                let key_and_colon = &remaining[..=colon_off];
                result.push_str(key_and_colon);
                let target_byte = byte_pos + colon_off + 1;
                while ci < chars.len() && chars[ci].0 < target_byte {
                    ci += 1;
                }
                while ci < chars.len() && chars[ci].1 == ' ' {
                    result.push(' ');
                    ci += 1;
                }
                if ci < chars.len() && (chars[ci].1 == '-' || chars[ci].1 == '+' || chars[ci].1.is_ascii_digit()) {
                    let value_start_byte = chars[ci].0;
                    let mut cj = ci;
                    while cj < chars.len() && chars[cj].1 != ',' && chars[cj].1 != '}' && chars[cj].1 != ']' {
                        cj += 1;
                    }
                    let value_end_byte = if cj < chars.len() { chars[cj].0 } else { text.len() };
                    let val_str = text[value_start_byte..value_end_byte].trim();
                    if val_str.parse::<f64>().is_ok() {
                        result.push_str(val_str);
                    } else {
                        result.push_str("0.0");
                    }
                    ci = cj;
                } else {
                    continue;
                }
            } else {
                result.push(chars[ci].1);
                ci += 1;
            }
        } else {
            result.push(chars[ci].1);
            ci += 1;
        }
    }
    result
}

fn fix_truncated_json(text: &str) -> String {
    let trimmed = text.trim();
    if !trimmed.starts_with('{') {
        return trimmed.to_string();
    }
    if trimmed.ends_with('}') {
        return trimmed.to_string();
    }
    // Model output was truncated mid-JSON. Try to salvage by:
    // 1. Find the last complete key-value pair (ends with , or just before a key)
    // 2. Close any open string, then close the object

    let mut s = trimmed.to_string();

    // If we're inside an open string value, find last unescaped quote
    let mut in_string = false;
    let mut last_complete_comma = 0; // byte offset of last comma outside a string
    let mut escape_next = false;
    for (i, ch) in s.char_indices() {
        if escape_next {
            escape_next = false;
            continue;
        }
        if ch == '\\' && in_string {
            escape_next = true;
            continue;
        }
        if ch == '"' {
            in_string = !in_string;
        }
        if !in_string && ch == ',' {
            last_complete_comma = i;
        }
    }

    if in_string && last_complete_comma > 0 {
        // Truncated inside a string value — cut back to last complete field
        s.truncate(last_complete_comma);
    } else if in_string {
        // Only one field and it's truncated — close the string
        s.push('"');
    }

    // Remove any trailing comma before closing
    let trimmed_end = s.trim_end();
    if trimmed_end.ends_with(',') {
        s = trimmed_end.trim_end_matches(',').to_string();
    }

    if !s.ends_with('}') {
        s.push('}');
    }
    s
}

// ── // -- AI Decision Parsing (safe-by-construction guardrail) --

/// Deserialize f64 leniently: handles null, strings, booleans, and missing values.
#[allow(dead_code)]
fn deserialize_f64_lenient<'de, D>(deserializer: D) -> Result<f64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de;
    struct F64Visitor;
    impl<'de> de::Visitor<'de> for F64Visitor {
        type Value = f64;
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str("a number or null")
        }
        fn visit_f64<E: de::Error>(self, v: f64) -> Result<f64, E> { Ok(v) }
        fn visit_i64<E: de::Error>(self, v: i64) -> Result<f64, E> { Ok(v as f64) }
        fn visit_u64<E: de::Error>(self, v: u64) -> Result<f64, E> { Ok(v as f64) }
        fn visit_unit<E: de::Error>(self) -> Result<f64, E> { Ok(0.0) }
        fn visit_none<E: de::Error>(self) -> Result<f64, E> { Ok(0.0) }
        fn visit_some<D2: serde::Deserializer<'de>>(self, d: D2) -> Result<f64, D2::Error> {
            d.deserialize_any(Self)
        }
        fn visit_str<E: de::Error>(self, v: &str) -> Result<f64, E> {
            Ok(v.parse::<f64>().unwrap_or(0.0))
        }
        fn visit_bool<E: de::Error>(self, v: bool) -> Result<f64, E> {
            Ok(if v { 1.0 } else { 0.0 })
        }
    }
    deserializer.deserialize_any(F64Visitor)
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct AiDecisionRaw {
    #[serde(default)]
    action: Option<String>,
    #[serde(default, deserialize_with = "deserialize_f64_lenient")]
    confidence: f64,
    #[serde(default, deserialize_with = "deserialize_f64_lenient")]
    trend_score: f64,
    #[serde(default, deserialize_with = "deserialize_f64_lenient")]
    score: f64,
    #[serde(default)]
    lane: Option<String>,
    #[serde(default)]
    reason: Option<String>,
    #[serde(default, deserialize_with = "deserialize_f64_lenient")]
    hold_hours: f64,
    // Decision contrast fields
    #[serde(default)]
    alt_action: Option<String>,
    #[serde(default, deserialize_with = "deserialize_f64_lenient")]
    alt_confidence: f64,
    #[serde(default, deserialize_with = "deserialize_f64_lenient")]
    margin: f64,
    #[serde(default)]
    veto_flag: Option<String>,
    #[serde(default)]
    flags: Option<Vec<String>>,
    // Time contract fields
    #[serde(default, deserialize_with = "deserialize_f64_lenient")]
    min_hold_sec: f64,
    #[serde(default, deserialize_with = "deserialize_f64_lenient")]
    max_hold_sec: f64,
    #[serde(default, deserialize_with = "deserialize_f64_lenient")]
    reeval_sec: f64,
}
