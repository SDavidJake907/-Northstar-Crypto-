# NorthStar - Architecture Summary

Institutional-grade modular architecture. Rated 9.3/10 by Google Gemini.

## 1. Data Engine
- Kraken WebSocket v2 real-time data (74 coins)
- Exchange-agnostic modular design
- Private WebSocket for order fills

## 2. GPU-Accelerated Scoring (Custom CUDA Kernels)
- Hurst Exponent, Shannon Entropy, Covariance Matrix
- EMA stack, RSI, ATR, MACD, Bollinger Bands, Z-Score
- GBDT model trained on past trade outcomes
- All 74 coins processed in parallel under 10ms

## 3. Regime State Machine
- BULLISH / SIDEWAYS / BEARISH / VOLATILE / UNKNOWN
- Per-coin with dwell time and N-of-M confirmation
- 7-day and 30-day MTF trend verification

## 4. Lane Assignment and Fee Gate
- L1 trend / L2 mean-revert / L3 moderate / L4 meme
- Behavioral meme detector
- Fee gate blocks negative-expectancy trades before AI

## 5. 4-Layer AI Brain
- AI_1: Nemotron 9B, 14 tools, entry confirmation (GPU)
- AI_2: OpenReasoning 1.5B, exit advisory (GPU)
- AI_3: Phi-3 mini, 200ms pre-scan (Intel NPU)
- 49B: Nemotron Super, hourly optimizer (NVIDIA NIM)

## 6. Risk Management
- Kelly criterion position sizing
- ATR-based stops per lane
- Circuit breaker and 3-loss cooldown
- Auto-adopt holdings with stop-loss

## 7. Decision Trace
- Every gate, AI decision, sizing calc logged to JSONL
- 49B reads all 3 model decision logs hourly
- Fully auditable and transparent
