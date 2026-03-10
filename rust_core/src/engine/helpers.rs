//! Pure helper functions — no TradingLoop state, no `self`.
//! Extracted from trading_loop.rs for readability.

use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use super::{HardwareStatus, NemoMemoryEntry, PendingOrderSnapshot};

// ── Env parsing ─────────────────────────────────────────────────

pub fn parse_env_bool(key: &str, default: bool) -> bool {
    std::env::var(key)
        .map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(default)
}

pub fn parse_env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(default)
}

pub fn parse_env_paths(key: &str) -> Vec<PathBuf> {
    std::env::var(key)
        .ok()
        .map(|s| {
            s.split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(PathBuf::from)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default()
}

// ── Time ────────────────────────────────────────────────────────

pub fn now_ts() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

pub fn entry_order_timeout_sec() -> f64 {
    parse_env_f64("ENTRY_ORDER_TIMEOUT_SEC", 600.0)
}

// ── File I/O ────────────────────────────────────────────────────

pub fn atomic_write(path: &str, content: &str) -> std::io::Result<()> {
    let path_obj = Path::new(path);
    let parent = path_obj.parent().unwrap_or_else(|| Path::new("."));
    let file_name = path_obj
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("state.json");
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let tmp_path = parent.join(format!(".{file_name}.tmp-{}-{nonce}", std::process::id()));

    {
        let mut f = fs::File::create(&tmp_path)?;
        f.write_all(content.as_bytes())?;
        f.sync_all()?;
    }

    match fs::rename(&tmp_path, path_obj) {
        Ok(()) => Ok(()),
        Err(e) => {
            if path_obj.exists() {
                let _ = fs::remove_file(path_obj);
                fs::rename(&tmp_path, path_obj)
            } else {
                let _ = fs::remove_file(&tmp_path);
                Err(e)
            }
        }
    }
}

pub fn append_jsonl(path: &str, value: &serde_json::Value) -> std::io::Result<()> {
    let p = Path::new(path);
    if let Some(parent) = p.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let mut f = fs::OpenOptions::new().create(true).append(true).open(p)?;
    let line = serde_json::to_string(value).unwrap_or_else(|_| "{}".to_string());
    f.write_all(line.as_bytes())?;
    f.write_all(b"\n")?;
    f.flush()?;
    Ok(())
}

pub fn file_age_sec(path: &Path, now: f64) -> Option<f64> {
    let meta = fs::metadata(path).ok()?;
    let modified = meta.modified().ok()?;
    let mtime = modified
        .duration_since(UNIX_EPOCH)
        .ok()
        .map(|d| d.as_secs_f64())?;
    Some((now - mtime).max(0.0))
}

pub fn resolve_existing_path(raw: &str, cwd: &str) -> Option<PathBuf> {
    let p = PathBuf::from(raw);
    if p.is_file() {
        return Some(p);
    }
    let joined = Path::new(cwd).join(raw);
    if joined.is_file() {
        return Some(joined);
    }
    if let Some(parent) = Path::new(cwd).parent() {
        let parent_joined = parent.join(raw);
        if parent_joined.is_file() {
            return Some(parent_joined);
        }
    }
    None
}

pub fn sha256_file_hex(path: &Path) -> Option<String> {
    let bytes = fs::read(path).ok()?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    Some(format!("{:x}", hasher.finalize()))
}

// ── Logging ─────────────────────────────────────────────────────

pub fn log_event(level: &str, event: &str, fields: serde_json::Value) {
    if parse_env_bool("LOG_JSON", false) {
        let rec = json!({
            "ts": now_ts(),
            "level": level,
            "event": event,
            "fields": fields
        });
        tracing::info!("{rec}");
    }
}

// ── Heartbeat & Watchdog ────────────────────────────────────────

pub fn write_heartbeat(path: &str, payload: &serde_json::Value) {
    match serde_json::to_string(payload) {
        Ok(body) => {
            let _ = atomic_write(path, &body);
        }
        Err(e) => {
            tracing::error!("[HEARTBEAT] serialize failed: {e}");
        }
    }
}

pub struct WatchdogReport {
    pub ok: bool,
    pub fail_count: usize,
    pub fail_paths: Vec<String>,
}

pub fn default_watchdog_inputs(trader_heartbeat: &str) -> Vec<PathBuf> {
    let mut watched = vec![PathBuf::from(trader_heartbeat)];

    for root in ["data"] {
        if let Ok(entries) = fs::read_dir(root) {
            for entry in entries.flatten() {
                let path = entry.path();
                if !path.is_file() {
                    continue;
                }
                let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
                    continue;
                };
                let lower = name.to_ascii_lowercase();
                if lower.starts_with("heartbeat")
                    && lower.ends_with(".json")
                    && lower != "watchdog_status.json"
                {
                    watched.push(path);
                }
            }
        }
    }

    let mut seen = HashSet::new();
    watched.retain(|p| seen.insert(p.to_string_lossy().to_string()));
    watched
}

pub fn run_watchdog(now: f64, trader_heartbeat: &str) -> WatchdogReport {
    if !parse_env_bool("WATCHDOG_ENABLED", false) {
        return WatchdogReport {
            ok: true,
            fail_count: 0,
            fail_paths: Vec::new(),
        };
    }

    let stale_sec = parse_env_f64("WATCHDOG_STALE_SEC", 20.0).max(1.0);
    let mut watched = parse_env_paths("WATCHDOG_INPUTS");
    if watched.is_empty() {
        watched = default_watchdog_inputs(trader_heartbeat);
    }

    let mut checks = Vec::new();
    let mut all_ok = true;
    let mut fail_paths = Vec::new();
    for path in watched {
        let exists = path.exists();
        let age = file_age_sec(&path, now);
        let ok = exists && age.map(|a| a <= stale_sec).unwrap_or(false);
        if !ok {
            all_ok = false;
            fail_paths.push(path.to_string_lossy().to_string());
        }
        checks.push(json!({
            "path": path.to_string_lossy(),
            "exists": exists,
            "age_sec": age,
            "ok": ok,
        }));
    }

    let watchdog_file =
        std::env::var("WATCHDOG_FILE").unwrap_or_else(|_| "data/watchdog_status.json".into());
    let payload = json!({
        "ts": now,
        "stale_after_sec": stale_sec,
        "ok": all_ok,
        "checked_count": checks.len(),
        "fail_count": fail_paths.len(),
        "failing_paths": fail_paths.clone(),
        "checks": checks,
    });
    if let Ok(s) = serde_json::to_string_pretty(&payload) {
        let _ = atomic_write(&watchdog_file, &s);
    }
    if parse_env_bool("WATCHDOG_AUDIT_ENABLED", true) {
        let audit_file = std::env::var("WATCHDOG_AUDIT_FILE")
            .unwrap_or_else(|_| "data/watchdog_events.jsonl".into());
        let _ = append_jsonl(
            &audit_file,
            &json!({
                "ts": now,
                "ok": all_ok,
                "checked_count": payload.get("checked_count").and_then(|v| v.as_u64()).unwrap_or(0),
                "fail_count": fail_paths.len(),
                "failing_paths": fail_paths,
            }),
        );
    }
    if !all_ok {
        tracing::error!("[WATCHDOG] stale/missing heartbeat detected (status: {watchdog_file})");
    }
    WatchdogReport {
        ok: all_ok,
        fail_count: fail_paths.len(),
        fail_paths,
    }
}

// ── Hardware probes ─────────────────────────────────────────────

pub(crate) fn collect_hardware_status(now: f64) -> HardwareStatus {
    let mut out = HardwareStatus {
        sampled_ts: now,
        ..HardwareStatus::default()
    };
    let hw = std::thread::scope(|s| {
        let gpu_handle = s.spawn(query_nvidia_smi);
        let ram_handle = s.spawn(query_system_memory_mb);
        (
            gpu_handle.join().ok().flatten(),
            ram_handle.join().ok().flatten(),
        )
    });
    if let Some((used, total, temp)) = hw.0 {
        out.gpu_mem_used_mb = Some(used);
        out.gpu_mem_total_mb = Some(total);
        out.gpu_temp_c = Some(temp);
    }
    if let Some((free_mb, total_mb)) = hw.1 {
        out.ram_free_mb = Some(free_mb);
        out.ram_total_mb = Some(total_mb);
    }
    if out.gpu_mem_used_mb.is_none() && out.ram_free_mb.is_none() {
        out.sample_error = Some("no_hardware_probe_data".to_string());
    }
    out
}

fn query_nvidia_smi() -> Option<(f64, f64, f64)> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let line = text.lines().next()?.trim();
    let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
    if parts.len() < 3 {
        return None;
    }
    let used = parts[0].parse::<f64>().ok()?;
    let total = parts[1].parse::<f64>().ok()?;
    let temp = parts[2].parse::<f64>().ok()?;
    Some((used, total, temp))
}

fn query_system_memory_mb() -> Option<(f64, f64)> {
    let output = Command::new("powershell")
        .args([
            "-NoProfile",
            "-Command",
            "$os=Get-CimInstance Win32_OperatingSystem; \"$($os.FreePhysicalMemory),$($os.TotalVisibleMemorySize)\"",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let line = text.lines().next()?.trim();
    let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
    if parts.len() < 2 {
        return None;
    }
    let free_kb = parts[0].parse::<f64>().ok()?;
    let total_kb = parts[1].parse::<f64>().ok()?;
    Some((free_kb / 1024.0, total_kb / 1024.0))
}

// ── Integrity / diagnostics ─────────────────────────────────────

pub fn collect_integrity_snapshot() -> serde_json::Value {
    let cwd = std::env::current_dir()
        .ok()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| ".".to_string());
    let lock_path = Path::new("Cargo.lock");
    let rustc_version = Command::new("rustc")
        .arg("--version")
        .output()
        .ok()
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());
    let startup_ts = now_ts();

    let mut dll_rows: Vec<serde_json::Value> = Vec::new();
    for var in ["GPU_CPP_DLL", "BOOK_ENGINE_DLL", "NPU_CPP_DLL"] {
        let path_str = match std::env::var(var) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let resolved = resolve_existing_path(&path_str, &cwd);
        let hash = resolved
            .as_ref()
            .and_then(|p| sha256_file_hex(p))
            .unwrap_or_else(|| "missing".to_string());
        let resolved_str = resolved.map(|p| p.display().to_string());
        dll_rows.push(json!({
            "env": var,
            "path": path_str,
            "resolved": resolved_str,
            "sha256": hash
        }));
    }

    json!({
        "startup_ts": startup_ts,
        "cwd": cwd,
        "rustc": rustc_version,
        "cargo_lock_sha256": sha256_file_hex(lock_path),
        "dlls": dll_rows
    })
}

// ── Nemo memory formatting ──────────────────────────────────────

pub(crate) fn format_nemo_memory(entries: &std::collections::VecDeque<NemoMemoryEntry>, current_price: f64) -> String {
    if entries.is_empty() {
        return String::new();
    }
    let now = now_ts();
    let mut lines = Vec::new();
    for e in entries.iter().rev().take(6) {
        let age_min = ((now - e.ts) / 60.0).round() as i64;
        let pnl = if e.price_at_decision > 0.0 {
            current_price / e.price_at_decision * 100.0 - 100.0
        } else { 0.0 };
        lines.push(format!(
            "{}m|{}|{:.2}|{:+.1}%",
            age_min, e.action, e.confidence, pnl
        ));
    }
    format!("HIST:{}", lines.join(","))
}

// ── Lane classification ─────────────────────────────────────────

/// Deterministic lane classification — replaces LLM lane assignment.
/// L1 = momentum breakout, L2 = compression/sideways, L3 = trend following, L4 = meme pump.
pub fn classify_lane(feats: &serde_json::Value, price: f64) -> String {
    let ema9 = feats.get("ema9").and_then(|v| v.as_f64()).unwrap_or(0.0);
    let ema21 = feats.get("ema21").and_then(|v| v.as_f64()).unwrap_or(0.0);
    let ema50 = feats.get("ema50").and_then(|v| v.as_f64()).unwrap_or(0.0);
    let rsi = feats.get("rsi").and_then(|v| v.as_f64()).unwrap_or(50.0);
    let momentum = feats.get("momentum_score").and_then(|v| v.as_f64()).unwrap_or(0.0);
    let vol = feats.get("vol_ratio").and_then(|v| v.as_f64()).unwrap_or(0.0);
    let trend = feats.get("trend_score").and_then(|v| v.as_f64()).unwrap_or(0.0) as i32;
    let book_imb = feats.get("book_imbalance").and_then(|v| v.as_f64()).unwrap_or(0.0);
    let buy_ratio = feats.get("buy_ratio").and_then(|v| v.as_f64()).unwrap_or(0.5);
    let bb_upper = feats.get("bb_upper").and_then(|v| v.as_f64()).unwrap_or(0.0);
    let bb_lower = feats.get("bb_lower").and_then(|v| v.as_f64()).unwrap_or(0.0);
    let bb_mid = (bb_upper + bb_lower) / 2.0;
    let bb_width = if bb_mid > 0.0 { (bb_upper - bb_lower) / bb_mid } else { 0.0 };
    let ema_spread = if price > 0.0 { (ema21 - ema50) / price } else { 0.0 };
    let zscore = feats.get("zscore").and_then(|v| v.as_f64()).unwrap_or(0.0);
    let macd = feats.get("macd_hist").and_then(|v| v.as_f64()).unwrap_or(0.0);
    let atr = feats.get("atr").and_then(|v| v.as_f64()).unwrap_or(0.0);
    let feats_l2 = feats.get("is_l2").and_then(|v| v.as_bool()).unwrap_or(false);

    // L4 MEME PUMP: Volume spike + book pressure + momentum — no EMA/RSI required
    // Triggers on: vol >= 2x OR book_imb >= 0.20 + buy_ratio > 0.55 + momentum > 0
    // This lane exists for meme coins that don't have enough history for EMA50
    if ema50 <= 0.0 && price > 0.0 {
        // No EMA data yet — use meme-native signals
        let vol_hot = vol >= 2.0;
        let book_hot = book_imb >= 0.15 && buy_ratio > 0.55;
        let mom_hot = momentum > 0.0;
        if vol_hot || (book_hot && mom_hot) {
            return "L4".to_string();
        }
        // Even without strong signals, memes with any activity get L4
        if book_imb > 0.0 || momentum > 0.0 || vol > 0.5 {
            return "L4".to_string();
        }
    }

    // BEHAVIORAL MEME DETECTOR: catches unknown/new meme coins by behavior
    // A coin ACTS like a meme if 3+ of these fire:
    //   1. Ultra-low price (< $0.01) — micro cap penny dynamics
    //   2. Extreme ATR (> 8% of price) — wild volatility
    //   3. Massive vol spike (>= 5x) — momentum without structure
    //   4. BTC decoupled (corr < 0.10) — moves on its own, not macro
    {
        let atr_val = feats.get("atr").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let atr_pct = if price > 0.0 { atr_val / price } else { 0.0 };
        let corr_btc = feats.get("quant_corr_btc").and_then(|v| v.as_f64()).unwrap_or(0.5);
        let ultra_low = price > 0.0 && price < 0.01;
        let extreme_atr = atr_pct > 0.08;
        let vol_spike = vol >= 5.0;
        let btc_decoupled = corr_btc.abs() < 0.10;
        let behavioral_score = ultra_low as u8 + extreme_atr as u8 + vol_spike as u8 + btc_decoupled as u8;
        if behavioral_score >= 3 {
            return "L4".to_string();
        }
    }

    // L1 MOMENTUM: Price>EMA21, EMA9>EMA21, Momentum>0, Vol>=1.1, Trend>=1
    // RSI>80 is advisory only — strong momentum can push well past 80
    if price > ema21 && ema9 > ema21 && momentum > 0.0 && vol >= 1.1 && trend >= 1 {
        return "L1".to_string();
    }

    // L2 COMPRESSION / SIDEWAYS (mean-reversion)
    let eps = price * 0.015;
    let emas_braided = ema9 > 0.0 && ema21 > 0.0 && ema50 > 0.0
        && (ema9 - ema21).abs() < eps && (ema21 - ema50).abs() < eps;
    let macd_flat = price > 0.0 && (macd / price).abs() < 0.0015;
    let rsi_neutral = rsi >= 40.0 && rsi <= 60.0;
    let atr_low = price > 0.0 && atr < price * 0.02;
    let z_contained = zscore.abs() < 1.0;
    let flow_neutral = book_imb.abs() < 0.20;

    let l2_hits = [emas_braided, macd_flat, rsi_neutral, atr_low, z_contained, flow_neutral]
        .iter().filter(|&&x| x).count();

    if l2_hits >= 5 || (feats_l2 && bb_width < 0.06 && bb_width > 0.0) {
        return "L2".to_string();
    }

    // Legacy L2 fallback
    if bb_width < 0.06 && bb_width > 0.0 && vol < 0.8 && ema_spread.abs() < 0.03 && book_imb >= -0.15 {
        return "L2".to_string();
    }

    // L3 TREND: Price>EMA50, EMA21>EMA50, 40<RSI<70, Trend>=0
    if price > ema50 && ema21 > ema50 && rsi > 40.0 && rsi < 70.0 && trend >= 0 {
        return "L3".to_string();
    }

    "rejected".to_string()
}

// ── Order snapshot parsing ──────────────────────────────────────

pub fn should_finalize_pending_entry(status: &str, vol_exec: f64) -> bool {
    vol_exec > 0.0 && matches!(status, "closed" | "canceled" | "cancelled" | "expired")
}

pub(crate) fn parse_query_order_snapshot(
    payload: &serde_json::Value,
    txid: &str,
) -> Option<PendingOrderSnapshot> {
    let order_obj = payload
        .get("result")
        .and_then(|r| r.get(txid))
        .and_then(|v| v.as_object())?;

    let status = order_obj
        .get("status")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_ascii_lowercase();
    let vol_exec = order_obj
        .get("vol_exec")
        .and_then(|v| v.as_str())
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.0);
    let cost = order_obj
        .get("cost")
        .and_then(|v| v.as_str())
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.0);
    // Kraken provides price/avg_price for accurate fill pricing
    let price = order_obj
        .get("price")
        .and_then(|v| v.as_str())
        .and_then(|s| s.parse::<f64>().ok())
        .filter(|&p| p > 0.0);
    let avg_price = order_obj
        .get("avg_price")
        .and_then(|v| v.as_str())
        .and_then(|s| s.parse::<f64>().ok())
        .filter(|&p| p > 0.0);

    Some(PendingOrderSnapshot {
        status,
        vol_exec,
        cost,
        price,
        avg_price,
    })
}

// ── Kraken symbol mapping ───────────────────────────────────────

pub fn kraken_to_symbol(kraken_name: &str) -> String {
    match kraken_name {
        "ZUSD" | "USD" => "USD".into(),
        "XXBT" | "XBT" => "BTC".into(),
        "XETH" => "ETH".into(),
        "XLTC" => "LTC".into(),
        "XXRP" => "XRP".into(),
        "XXLM" => "XLM".into(),
        other => other.to_string(),
    }
}

// ── Quality stats persistence ───────────────────────────────────

use super::BetaStats;

pub(crate) fn load_quality_stats() -> std::collections::HashMap<String, BetaStats> {
    let path = "data/quality_stats.json";
    match std::fs::read_to_string(path) {
        Ok(data) => serde_json::from_str(&data).unwrap_or_default(),
        Err(_) => std::collections::HashMap::new(),
    }
}

pub(crate) fn save_quality_stats(stats: &std::collections::HashMap<String, BetaStats>) {
    let path = "data/quality_stats.json";
    if let Ok(json_str) = serde_json::to_string_pretty(stats) {
        if let Err(e) = std::fs::write(path, &json_str) {
            tracing::error!("[QL] Failed to save quality_stats: {e}");
        }
    }
}

// ── Advisory Classification ─────────────────────────────────

/// Severity level for gate advisories.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum AdvLevel { Info, Warn, Crit }

impl AdvLevel {
    pub fn tag(self) -> &'static str {
        match self { Self::Info => "I", Self::Warn => "W", Self::Crit => "C" }
    }
}

/// Classify a raw gate warning string into (level, code).
/// Returns the severity and a cleaned-up code string.
pub fn classify_advisory(raw: &str) -> (AdvLevel, &str) {
    if raw.starts_with("illiquid") { return (AdvLevel::Crit, raw); }
    if raw.starts_with("falling_knife") { return (AdvLevel::Crit, raw); }
    if raw.starts_with("slip_high") { return (AdvLevel::Warn, raw); }
    if raw.starts_with("spread_high") { return (AdvLevel::Warn, raw); }
    if raw.starts_with("atr_spike") { return (AdvLevel::Warn, raw); }
    if raw.starts_with("liq_low") { return (AdvLevel::Warn, raw); }
    if raw.starts_with("vol_low") { return (AdvLevel::Warn, raw); }
    if raw.starts_with("low_mkt") { return (AdvLevel::Warn, raw); }
    if raw.contains("missing") { return (AdvLevel::Info, raw); }
    (AdvLevel::Warn, raw)
}

/// Format gate warnings as a structured advisory string.
/// Output: `ADV:C:illiquid(450<500),W:spread_high(0.18>0.15),C:falling_knife_rsi=34`
pub fn format_advisories(warnings: &[String]) -> String {
    if warnings.is_empty() { return String::new(); }
    let parts: Vec<String> = warnings.iter().map(|w| {
        let (level, code) = classify_advisory(w);
        format!("{}:{}", level.tag(), code)
    }).collect();
    format!("ADV:{}", parts.join(","))
}
