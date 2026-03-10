# Sync Guide for Nemotron and Qwen

## Overview
This document outlines practical steps to keep the Nemotron and Qwen LLM components in sync within the **Kraken‑Hybrad9** system without moving or modifying existing code.

## 1. Configuration Alignment
- **Model Versions**: Ensure both `nemotron_schedule.md` and any Qwen configuration files reference the same model version identifiers (e.g., `v1.0`).
- **Environment Variables**:
  - `NEMOTRON_MODEL` – path or identifier for the Nemotron model.
  - `QWEN_MODEL` – path or identifier for the Qwen model.
  - Keep these variables in a shared `.env` file so both services read the same values.

## 2. Shared Data Store
- Store intermediate data (e.g., prompts, embeddings) in a common directory such as `data/shared/`.
- Use a consistent naming scheme: `<timestamp>_prompt.json` and `<timestamp>_embedding.npy`.
- Both Nemotron and Qwen should read/write to this location to avoid duplication.

## 3. Scheduling & Orchestration
- Use a single scheduler (e.g., Windows Task Scheduler or a cron‑like service) to trigger both models sequentially or in parallel.
- Example schedule entry (pseudo‑code):
  ```
  # Run Nemotron first, then Qwen
  python scripts/run_nemotron.py && python scripts/run_qwen.py
  ```
- Ensure the scheduler logs to `logs/sync.log` for troubleshooting.

## 4. Logging Consistency
- Configure both models to use the same logging format and destination (`logs/` directory).
- Include a `model` field in each log entry to differentiate sources.

## 5. Health Checks
- Implement lightweight health‑check scripts that verify both services are reachable and responding.
- Example (Python):
  ```python
  import requests
  def check(url):
      try:
          r = requests.get(url, timeout=5)
          return r.status_code == 200
      except Exception:
          return False
  print('Nemotron:', check('http://localhost:8000/health'))
  print('Qwen:', check('http://localhost:8001/health'))
  ```
- Schedule this script to run every few minutes and alert on failures.

## 6. Version Control
- Keep a `sync_requirements.txt` file listing the exact versions of both models and any shared libraries.
- Update this file whenever a model version changes and commit to the repository.

## 7. Documentation
- Add a short section in `README.md` linking to this sync guide.
- Ensure team members know to consult this file before making model‑related changes.

---
*This guide is saved as a reference for keeping Nemotron and Qwen synchronized without altering existing code or file locations.*
