# LLM Integration Guide for Kraken‑Hybrid9 (single‑model setup)

## 1. Configuration
Create a config file at `rust_core/config/llm.toml` (you may need to create the `config` directory). Example:
```toml
model_name = "qwen2.5:1.5b"
temperature = 0.5
max_tokens = 1024
margin_gate = 0.15
batch_size = 10
use_gpu = false
```
Load this config in `AiBridge::new` and use the values for all requests.

## 2. Prompt Management
Place prompts in a `prompts/` folder:
- `entry_qwen2.5_1.5b.txt`
- `exit_qwen2.5_1.5b.txt`
Load the appropriate file based on `model_name` from the config. Validate the prompt length against the model’s token limit during startup.

## 3. Code Clean‑up
- Remove unused imports (`LaneVerifyResult`, `load_npu_scan_prompt`) from `src/llm/client.rs`.
- (Optional) Split `src/llm/mod.rs` into smaller modules (`types.rs`, `constants.rs`, `decision.rs`).
- Cache the base Ollama URL inside `AiBridge` instead of rebuilding it on each call.

## 4. Logging & Observability
- Use `tracing` to log each LLM request/response (model, temperature, request ID).
- Log errors with full HTTP status and response body.
- Add a health‑check method `AiBridge::ping()` that hits `/api/version` before the trading loop starts.

## 5. Runtime Flexibility
Expose environment variables to override config values:
- `LLM_MODEL`
- `LLM_TEMPERATURE`
- `LLM_MAX_TOKENS`
- `LLM_USE_GPU`
- `LLM_BATCH_SIZE`
These allow tuning without recompiling.

## 6. Batch Error Isolation
In `get_batch_decisions` and `get_batch_entry_confirmation`, return a `HashMap<String, Result<…, BridgeError>>` so a failure for one symbol does not abort the whole batch.

## 7. GPU / CPU Path
Add a runtime flag (`use_gpu`) and log which path is taken. If `use_gpu` is true but the driver is unavailable, fall back to CPU with a warning.

## 8. Testing
- Add mock‑based integration tests for `AiBridge` (e.g., using `wiremock`).
- Verify prompt loading, request construction, and error handling for different model configs.

These changes are **non‑breaking** and can be introduced incrementally. They improve maintainability, observability, and make it trivial to tune the single LLM model in production.
