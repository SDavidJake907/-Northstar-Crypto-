# Optimization Plan: Performance & Profitability Tuning

This plan outlines low-risk technical adjustments to address the latency and "over-filtering" issues identified in the A-Z review. These changes focus on the Python bridge and TOML configuration to minimize risk to the live Rust core.

## Proposed Changes

### 1. Reduce IPC Latency (NPU Bridge)
The current 8s polling interval in the NPU bridge combined with file I/O creates a significant lag (~100s cumulative delay observed).
- [ ] **[MODIFY] [npu_bridge.py](file:///c:/Users/kitti/Projects/kraken-hybrad9/npu_bridge.py)**: Change `POLL_SEC` from `8` to `2`.
- [ ] **[MODIFY] [check_status.py](file:///c:/Users/kitti/Projects/kraken-hybrad9/check_status.py)**: Update stale warning threshold from `60s` to `15s` to reflect higher throughput expectations.

### 2. Relax AI Veto Thresholds
The AI is currently "too scared" by moderate RSI levels (blocking trades at RSI > 72), which often misses early momentum in trending markets.
- [ ] **[MODIFY] [npu_bridge.py](file:///c:/Users/kitti/Projects/kraken-hybrad9/npu_bridge.py)**: In `score_coin`, change the `reversal_risk` RSI trigger from `72` to `82` for L1/L4 lanes (momentum/memes).
- [ ] **[MODIFY] [npu_bridge.py](file:///c:/Users/kitti/Projects/kraken-hybrad9/npu_bridge.py)**: Adjust the `score -= 15` overbought penalty to trigger at `RSI > 80` instead of `75`.

### 3. Strategy Parameter Tuning
The current Kelly/Softmax delta is too wide, preventing small but profitable rebalances.
- [ ] **[MODIFY] [data/strategy.toml](file:///c:/Users/kitti/Projects/kraken-hybrad9/data/strategy.toml)**: Set `delta = 0.005` (from `0.01`).
- [ ] **[MODIFY] [data/strategy.toml](file:///c:/Users/kitti/Projects/kraken-hybrad9/data/strategy.toml)**: Set `tau = 0.7` (from `1.0`) to increase concentration in high-conviction signals.

## Verification Plan

### Automated Verification
- Run `python npu_bridge.py --test` (if available) or simply observe the `npu_bridge.log` to ensure the poll interval is indeed ~2s.
- Verify `data/npu_signals.json` is being updated with the new timestamps.

### Manual Verification
- **Monitor Latency:** Run `python check_status.py` and ensure the "Feature snapshot" age stays under 10s.
- **Review AI Recommendations:** Check `npu_signals.json` for coins with RSI between 72-80 to ensure they are now being classified as `WATCH` or `BUY` rather than `AVOID`.
- **Verify Order Triggers:** Monitor `core.log` for rebalance orders triggered by the smaller `delta` threshold.
