# Implementation Plan: Profitability Tuning (Small Account Optimization)

This plan adjusts the strategy parameters to overcome the "Fee Trap" and noise-related stop-losses identified in the [PROFITABILITY_ANALYSIS.md](file:///c:/Users/kitti/Projects/kraken-hybrad9/docs/PROFITABILITY_ANALYSIS.md).

## User Review Required

> [!IMPORTANT]
> **Portfolio Consolidation**: We are reducing the number of simultaneous trades from 12 to 3. This ensures each position has enough USD value to clear Kraken's minimums and that the 'edge' isn't diluted by fees.
> **Infrastructure Check**: The bot currently reports that the LLM servers are unreachable. These changes will improve the 'math' side, but the AI supervisors must be restarted for full effectiveness.

## Proposed Changes

### Strategy Configuration
#### [MODIFY] [strategy.toml](file:///c:/Users/kitti/Projects/kraken-hybrad9/data/strategy.toml)
*   Update `K = 3` (from 12) to focus the $40 balance into fewer, larger positions.

### Environment Parameters
#### [MODIFY] [.env](file:///c:/Users/kitti/Projects/kraken-hybrad9/.env)
*   `ENTRY_THRESHOLD = 0.65` (from 0.5): Filter for higher-probability signals.
*   `STOP_LOSS_PCT = 3.0` (from 1.5): Provide breathing room for crypto volatility.
*   `MEME_STOP_LOSS_PCT = 3.0` (from 2.0): Align meme risk with wider SL.
*   `NEMO_OPTIMIZER_REVERT_DRAWDOWN_PCT = 15.0` (from 5.0): Prevent premature reverts of AI learning.

---

## Verification Plan

### Automated Verification
*   **Config Reload**: The Rust engine logs `[STRATEGY] loaded strategy.toml v1` every 10s if changed. We will monitor `bot.log` to confirm the reload.
*   **State Check**: Verify `heartbeat_trader.json` reflects the new active configuration if applicable.

### Manual Verification
*   **Port Check**: User should verify if `ollama` or `llama-server` is running on ports 8081/8082 to fix the 'blindness' issue.
*   **Order Execution**: Monitor the next trade to ensure it uses the larger size (~$13) instead of $3.33.
