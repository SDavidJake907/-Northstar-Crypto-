# Hybrid Kraken â€” Full Port Migration Map

**Architecture:** Rust/C++ replaces Python wherever possible.
Python remains only for AI model inference (llama-cpp-python + OpenVINO NPU).

## Phase 1: Rust Core (Fast Path) â€” COMPLETE

**Replaces:** `gpu_worker.py`, `data/kraken_ws_v2.py`, `features/`, `utils/shared_cache.py`

| Module | File | Status |
|--------|------|--------|
| WebSocket v2 ingest (book/trade/ohlc) | `ws.rs` | Done |
| OrderBook state (BTreeMap) | `book.rs` | Done |
| Feature engine (EMA, RSI, MACD, ATR, BB, Z-score, momentum) | `features.rs` | Done |
| Markov regime detector (per-symbol, Laplace smoothing) | `main.rs` | Done |
| Book reversal tracker (3-candle avg, trend flips) | `main.rs` | Done |
| RSI adaptive percentiles (rolling 200-sample) | `main.rs` | Done |
| Market state scoring (8-component weighted) | `main.rs` | Done |
| Trade flow aggregation (time-window + maxlen) | `main.rs` | Done |
| Snapshot writer (atomic JSON + SHM mmap) | `snapshot.rs` | Done |
| C++ BookEngine DLL FFI (CVaR, slippage, flow) | `main.rs` | Done |

## Phase 2: Scoring â€” COMPLETE

**Replaces:** `scoring/score.py`, `scoring/trade_mlp.py`, `scoring/markov.py`, `scoring/quantum_scoring.py`

| Module | Python Source | Port Notes |
|--------|-------------|------------|
| Score aggregation | `scoring/score.py` | `scoring.rs` â€” 7D weighted scoring | Done |
| MLP classifier | `scoring/trade_mlp.py` | `trade_mlp.rs` â€” 24D vector + rule-based BUY/HOLD/SELL | Done |
| Markov scoring | `scoring/markov.py` | `markov.rs` â€” extracted from main.rs, JSON serialization | Done |
| Quantum scoring | `scoring/quantum_scoring.py` | `quantum.rs` â€” superposition + simulated annealing | Done |
| NPU scoring | `scoring/npu_score.py` | stays Python (OpenVINO) | N/A |

## Phase 3: Execution + Risk â€” COMPLETE

**Replaces:** `execution/kraken_api.py`, `utils/rules_eval.py`, `utils/portfolio_risk.py`, `utils/portfolio_optimizer.py`

| Module | Python Source | Rust File | Status |
|--------|-------------|-----------|--------|
| Kraken REST API | `execution/kraken_api.py` | `kraken_api.rs` â€” HMAC-SHA512 auth, atomic nonce, retries | Done |
| Rule evaluation | `utils/rules_eval.py` | `rules_eval.rs` â€” 11+ entry signals, gate checks, regime overrides | Done |
| Portfolio risk | `utils/portfolio_risk.py` | `portfolio_risk.rs` â€” CVaR calc, position sizing, rebalance triggers | Done |
| Portfolio optimizer | `utils/portfolio_optimizer.py` | `portfolio_optimizer.rs` â€” confidence/regime multipliers, pyramiding | Done |

## Phase 4: Full Orchestration â€” COMPLETE

**Replaces:** `tinyllm_trader.py` (partially), `memory/journal.py`, `watchdog.py`

| Module | Python Source | Rust File | Status |
|--------|-------------|-----------|--------|
| Config + circuit breaker | `tinyllm_trader.py` | `config.rs` â€” hot-reload .env, regime inference, profile selection | Done |
| Trade journaling | `memory/journal.py` | `journal.rs` â€” crash-safe JSONL, pattern learning, AI memory context | Done |
| AI subprocess bridge | `tinyllm_trader.py` | `ai_bridge.rs` â€” JSON-over-stdio IPC to Python AI workers | Done |
| Main trading loop | `tinyllm_trader.py` | `trading_loop.rs` â€” full tick loop, entry/exit logic, position mgmt | Done |

## Stays in Python (Forever)

| Module | File | Reason |
|--------|------|--------|
| GPU inference (DeepSeek + Qwen2.5) | `python_orch/ai_worker.py` | llama-cpp-python bindings |
| NPU inference (Qwen3) | `utils/npu_qwen.py` | OpenVINO Python API |
| NPU scoring model | `scoring/npu_score.py` | OpenVINO Python API |

## Key Principles

1. Swap module-by-module with feature parity
2. Snapshot JSON schema is the IPC contract between Rust and Python
3. AI inference stays Python â€” called from Rust via subprocess or shared memory
4. Each phase is independently testable before moving to the next
5. C++ BookEngine DLL is optional â€” pure-Rust fallback always works

