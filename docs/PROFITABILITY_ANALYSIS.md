# Profitability Analysis & Tuning Report

## Executive Summary: The "Mathematical Headwind"
The system is architecturally sound ("Production Grade"), but currently failing due to a mismatch between **Account Size**, **Friction (Fees)**, and **Portfolio Density**. 

Currently, the bot is operating in a "blind" state because the LLM servers are unreachable, but even with AI active, the following mathematical realities are preventing profit.

---

## 1. The "Fee Trap" & Small Account Paradox
*   **Balance**: ~$40.00 - $48.00 USD.
*   **Fees**: 0.26% Taker (0.52% round-trip).
*   **Portfolio Density (K=12)**: The `strategy.toml` specifies 12 positions. This splits $40 into **$3.33 per position**.
*   **Impact**: 
    1.  **Kraken Minimums**: Most Kraken pairs require a minimum of $5.00 - $10.00 per order. $3.33 positions will simply fail or result in "dust" that cannot be easily sold.
    2.  **Fee Dominance**: On a $3.00 trade, the 0.52% fee is negligible in USD, but the **Spread** on small notionals is often higher. When wins and losses are both ~2%, a 0.52% fee eats **25% of every winning trade**.

> [!IMPORTANT]
> To survive on a $40 account, the bot must trade **fewer coins with larger sizes** to overcome the fee/spread threshold.

---

## 2. Optimizer "Revert Loop"
The `nemo_optimizer.jsonl` shows the AI correctly identifying that the `ENTRY_THRESHOLD` needs to be higher (0.60 - 0.65) to improve quality. However:
*   **Issue**: `NEMO_OPTIMIZER_REVERT_DRAWDOWN_PCT` is set to **5.0%**.
*   **The Trap**: On a $40 account, a single $2.00 fluctuation (noise) triggers a full "Revert", undoing the AI's learning. The system is too "scared" to let a better strategy play out.

---

## 3. Infrastructure Failure (Blindness)
The logs (`bot.log`) reveal:
`[MAIN] AI bridge failed to start: http error: LLM server not reachable at http://127.0.0.1:8081`
*   **Result**: The bot is skipping its best entries (`ENTRY-SKIP`) and likely holding losers too long because the AI Exit Supervisor cannot provide "SELL" signals. It is currently a "Math-only" bot with broken exit logic.

---

## 4. Specific Tuning Recommendations

### **A. Consolidate the Portfolio**
*   **File**: [strategy.toml](file:///c:/Users/kitti/Projects/kraken-hybrad9/data/strategy.toml)
*   **Current**: `K = 12`
*   **Change**: `K = 3`
*   **Reason**: With $40, you want ~$13 per position. This clears Kraken minimums and ensures the "edge" is not diluted across 12 different sources of noise.

### **B. Raise the Quality Floor**
*   **File**: [.env](file:///c:/Users/kitti/Projects/kraken-hybrad9/.env)
*   **Current**: `ENTRY_THRESHOLD = 0.5`
*   **Change**: `ENTRY_THRESHOLD = 0.65`
*   **Reason**: Your "wins" almost all occur when the score is > 0.61. Everything below 0.6 is high-noise/high-stop-loss territory.

### **C. Widen the "Breathing Room"**
*   **File**: [.env](file:///c:/Users/kitti/Projects/kraken-hybrad9/.env)
*   **Current**: `STOP_LOSS_PCT = 1.5`
*   **Change**: `STOP_LOSS_PCT = 3.0`
*   **Reason**: Crypto volatility (especially memes) frequently "wicks" 1.5% before continuing the trend. You are being "stopped out" of winning trades by noise.

### **D. Loosen the Optimizer's "Fear"**
*   **File**: [.env](file:///c:/Users/kitti/Projects/kraken-hybrad9/.env)
*   **Current**: `NEMO_OPTIMIZER_REVERT_DRAWDOWN_PCT = 5.0`
*   **Change**: `NEMO_OPTIMIZER_REVERT_DRAWDOWN_PCT = 15.0`
*   **Reason**: Allow the AI more room to breathe. 5% is too tight for a small account.

---

## 5. Summary Recommendation
1.  **Fix Infrastructure**: Ensure `llama-server` or `ollama` is actually running on ports 8081 and 8082.
2.  **Focus Fire**: Reduce the number of simultaneous trades.
3.  **Harden Entries**: Only take the "Cream of the Crop" signals.

This system is a **Ferrari being driven in a parking lot**. It needs more fuel (Balance) or a smaller track (fewer coins) to show its true performance.
