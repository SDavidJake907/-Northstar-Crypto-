//! Order management — pending entry reconciliation, position persistence.

use super::*;
use crate::kraken_api::KrakenApi;
use serde_json::json;
use std::collections::HashMap;
use std::fs;

impl super::TradingLoop {
    pub(crate) async fn reconcile_pending_entries(&mut self, api: &KrakenApi) {
        if self.pending_entries.is_empty() {
            return;
        }
        let now = now_ts();
        let txids: Vec<String> = self.pending_entries.keys().cloned().collect();

        // Batch query all pending txids in one REST call (reduces rate-limit pressure)
        let txids_csv = txids.join(",");
        let status_payload = match api.query_orders(&txids_csv).await {
            Ok(v) => v,
            Err(e) => {
                tracing::error!("[ENTRY-PENDING] batch QueryOrders failed: {e}");
                crate::nemo_optimizer::flag_error("kraken_api", &format!("QueryOrders failed: {e}"));
                return;
            }
        };

        for txid in txids {
            let Some(pending) = self.pending_entries.get(&txid).cloned() else {
                continue;
            };
            let age = now - pending.created_ts;

            let Some(order) = parse_query_order_snapshot(&status_payload, &txid) else {
                continue;
            };
            let status = order.status;
            let vol_exec = order.vol_exec;
            let cost = order.cost;

            if status == "open" && age >= entry_order_timeout_sec() {
                let _ = api.cancel_order(&txid).await;
                let limit_px = if pending.requested_qty > 0.0 {
                    pending.reserved_usd / pending.requested_qty
                } else { 0.0 };
                self.journal.record_cancellation(crate::journal::CancelRecord {
                    symbol: pending.symbol.clone(),
                    limit_price: limit_px,
                    reserved_usd: pending.reserved_usd,
                    age_sec: age,
                    reason: "timeout".into(),
                    timestamp: now,
                });
                log_event(
                    "warn",
                    "entry_timeout_cancel",
                    json!({"symbol": pending.symbol, "txid": txid, "age_sec": age}),
                );
                continue;
            }

            if matches!(
                status.as_str(),
                "canceled" | "cancelled" | "expired" | "closed"
            ) {
                self.pending_entries.remove(&txid);
            }

            if should_finalize_pending_entry(status.as_str(), vol_exec) {
                // Prefer exchange-provided avg_price, then price, then cost/vol_exec
                let mut fill_price = order.avg_price
                    .or(order.price)
                    .unwrap_or(0.0);

                // Fallback: derive from cost/volume (only when both present)
                if fill_price <= 0.0 && vol_exec > 0.0 && cost > 0.0 {
                    fill_price = cost / vol_exec;
                }

                // SAFETY: never use cost as price. If no valid price, keep pending.
                if fill_price <= 0.0 {
                    tracing::warn!(
                        "[ENTRY-PENDING] {} txid={} vol_exec={:.8} but no fill price (cost={:.4}); waiting",
                        pending.symbol, txid, vol_exec, cost
                    );
                    continue;
                }

                self.finalize_filled_entry(&pending, vol_exec, fill_price, Some(api))
                    .await;
                continue;
            }

            if matches!(status.as_str(), "canceled" | "cancelled" | "expired") {
                self.available_usd += pending.reserved_usd;
                let limit_px = if pending.requested_qty > 0.0 {
                    pending.reserved_usd / pending.requested_qty
                } else { 0.0 };
                self.journal.record_cancellation(crate::journal::CancelRecord {
                    symbol: pending.symbol.clone(),
                    limit_price: limit_px,
                    reserved_usd: pending.reserved_usd,
                    age_sec: age,
                    reason: "cancelled_unfilled".into(),
                    timestamp: now,
                });
                log_event(
                    "info",
                    "entry_cancelled_unfilled",
                    json!({
                        "symbol": pending.symbol,
                        "txid": pending.txid,
                        "reserved_usd": pending.reserved_usd
                    }),
                );
            }
        }
    }

    pub(crate) async fn finalize_filled_entry(
        &mut self,
        pending: &PendingEntryOrder,
        filled_qty: f64,
        filled_price: f64,
        _api: Option<&KrakenApi>,
    ) {
        if filled_qty <= 0.0 || filled_price <= 0.0 {
            return;
        }
        let usd_filled = filled_qty * filled_price;
        if pending.reserved_usd > usd_filled {
            self.available_usd += pending.reserved_usd - usd_filled;
        }

        let pos = OpenPosition {
            symbol: pending.symbol.clone(),
            entry_price: filled_price,
            qty: filled_qty,
            remaining_qty: filled_qty,
            tp_price: pending.tp_price,
            sl_price: pending.sl_price,
            highest_price: filled_price,
            entry_time: pending.created_ts,
            entry_reasons: pending.signal_names.clone(),
            entry_profile: pending.entry_profile.clone(),
            entry_context: pending.entry_context.clone(),
            entry_score: pending.entry_score,
            regime_label: pending.regime_label.clone(),
            quant_bias: pending.quant_bias.clone(),
            npu_action: pending.npu_action.clone(),
            npu_conf: pending.npu_conf,
            sl_order_txid: None,
            quality_key: pending.quality_key.clone(),
            quality_score: pending.quality_score,
            min_hold_sec: pending.min_hold_sec,
            max_hold_sec: pending.max_hold_sec,
            reeval_sec: pending.reeval_sec,
            entry_atr: pending.entry_atr,
            entry_lane: pending.entry_lane.clone(),
            feature_snapshot: pending.feature_snapshot.clone(),
            trend_alignment: pending.trend_alignment,
            trend_7d_pct: pending.trend_7d_pct,
            trend_30d_pct: pending.trend_30d_pct,
        };
        self.optimizer.open_position(
            &pending.symbol,
            filled_price,
            filled_qty,
            usd_filled,
            pending.entry_score,
            config::coin_tier(&pending.symbol),
        );
        self.positions.insert(pending.symbol.clone(), pos);
        save_positions(&self.config.positions_file, &self.positions);

        // No Kraken-side stop-loss orders — bot checks SL internally every tick.
        // Kraken SL orders lock coins and block market sells. Internal check_exit() is faster.

        log_event(
            "info",
            "entry_filled",
            json!({
                "symbol": pending.symbol,
                "txid": pending.txid,
                "qty": filled_qty,
                "price": filled_price,
                "usd": usd_filled,
                "requested_qty": pending.requested_qty
            }),
        );
    }
}

// ── Position persistence ─────────────────────────────────────────

pub(crate) fn save_positions(path: &str, positions: &HashMap<String, OpenPosition>) {
    let data: Vec<serde_json::Value> = positions
        .values()
        .map(|p| {
            json!({
                "symbol": p.symbol,
                "entry": p.entry_price,
                "qty": p.qty,
                "remaining_qty": p.remaining_qty,
                "tp": p.tp_price,
                "sl": p.sl_price,
                "highest_price": p.highest_price,
                "entry_time": p.entry_time,
                "entry_reasons": p.entry_reasons,
                "entry_profile": p.entry_profile,
                "entry_context": p.entry_context,
                "entry_score": p.entry_score,
                "regime_label": p.regime_label,
                "quant_bias": p.quant_bias,
                "sl_order_txid": p.sl_order_txid,
                "quality_key": p.quality_key,
                "quality_score": p.quality_score,
                "min_hold_sec": p.min_hold_sec,
                "max_hold_sec": p.max_hold_sec,
                "reeval_sec": p.reeval_sec,
                "entry_atr": p.entry_atr,
                "entry_lane": p.entry_lane,
                "feature_snapshot": p.feature_snapshot,
            })
        })
        .collect();

    if let Ok(json_str) = serde_json::to_string_pretty(&data) {
        if let Err(e) = atomic_write(path, &json_str) {
            tracing::error!("[POSITIONS] Failed to persist positions atomically: {e}");
            crate::nemo_optimizer::flag_error("positions_file", &format!("atomic write failed: {e}"));
        }
    }
}

pub(crate) fn load_positions(path: &str) -> HashMap<String, OpenPosition> {
    let mut positions = HashMap::new();

    let data = match fs::read_to_string(path) {
        Ok(d) => d,
        Err(e) => {
            if e.kind() != std::io::ErrorKind::NotFound {
                tracing::error!("[POSITIONS] Failed to read {}: {e}", path);
            }
            return positions;
        }
    };

    let arr: Vec<serde_json::Value> = match serde_json::from_str(&data) {
        Ok(a) => a,
        Err(e) => {
            tracing::error!("[POSITIONS] Failed to parse {}: {e}", path);
            return positions;
        }
    };

    for v in arr {
        let symbol = v
            .get("symbol")
            .and_then(|s| s.as_str())
            .unwrap_or("")
            .to_string();
        if symbol.is_empty() {
            continue;
        }

        let f = |key: &str, def: f64| -> f64 { v.get(key).and_then(|x| x.as_f64()).unwrap_or(def) };

        let pos = OpenPosition {
            symbol: symbol.clone(),
            entry_price: f("entry", 0.0),
            qty: f("qty", 0.0),
            remaining_qty: f("remaining_qty", f("qty", 0.0)),
            tp_price: f("tp", 0.0),
            sl_price: f("sl", 0.0),
            highest_price: f("highest_price", f("entry", 0.0)),
            entry_time: f("entry_time", 0.0),
            entry_reasons: v
                .get("entry_reasons")
                .and_then(|a| a.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|s| s.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default(),
            entry_profile: v
                .get("entry_profile")
                .and_then(|s| s.as_str())
                .unwrap_or("defensive")
                .to_string(),
            entry_context: v
                .get("entry_context")
                .and_then(|s| s.as_str())
                .unwrap_or("")
                .to_string(),
            entry_score: f("entry_score", 0.0),
            regime_label: v
                .get("regime_label")
                .and_then(|s| s.as_str())
                .unwrap_or("UNKNOWN")
                .to_string(),
            quant_bias: v
                .get("quant_bias")
                .and_then(|s| s.as_str())
                .unwrap_or("NEUTRAL")
                .to_string(),
            npu_action: "HOLD".into(),
            npu_conf: 0.0,
            sl_order_txid: v.get("sl_order_txid").and_then(|s| s.as_str()).map(String::from),
            quality_key: v.get("quality_key").and_then(|s| s.as_str()).unwrap_or("").to_string(),
            quality_score: f("quality_score", 0.0),
            min_hold_sec: f("min_hold_sec", 300.0) as u32,
            max_hold_sec: f("max_hold_sec", 7200.0) as u32,
            reeval_sec: f("reeval_sec", 300.0) as u32,
            entry_atr: f("entry_atr", 0.0),
            entry_lane: v.get("entry_lane").and_then(|s| s.as_str()).unwrap_or("L3").to_string(),
            feature_snapshot: v.get("feature_snapshot").and_then(|a| {
                a.as_array().map(|arr| arr.iter().filter_map(|x| x.as_f64()).collect())
            }),
            trend_alignment: v.get("trend_alignment").and_then(|x| x.as_u64()).unwrap_or(0) as u8,
            trend_7d_pct: f("trend_7d_pct", 0.0),
            trend_30d_pct: f("trend_30d_pct", 0.0),
        };

        // Validate: skip corrupt positions that would cause bad math
        if pos.entry_price <= 0.0 || pos.remaining_qty <= 0.0 {
            tracing::warn!(
                "[POSITIONS] skipping {} — invalid entry_price={:.6} qty={:.8}",
                symbol, pos.entry_price, pos.remaining_qty
            );
            continue;
        }

        positions.insert(symbol, pos);
    }

    if !positions.is_empty() {
        tracing::info!("[LOOP] Loaded {} positions from disk", positions.len());
    }

    positions
}
