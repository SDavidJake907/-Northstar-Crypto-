# Data Assessment and Cleanup

## Goal Description
Assess the amount of historical data stored in the `data` directory, decide whether Nemotron should process old data, and implement an automated cleanup mechanism to remove stale or temporary files, improving storage usage and system performance.

## Proposed Changes
---
### Data Assessment
- Create a Python script `scripts/assess_data.py` that:
  - Recursively scans `data/` and computes total size.
  - Lists files older than a configurable threshold (e.g., 30 days).
  - Outputs a summary report.

---
### Cleanup Script
- Create `scripts/cleanup_data.py` that:
  - Uses the same age threshold to delete old files.
  - Supports a `--dry-run` flag to preview deletions.
  - Logs actions to `logs/cleanup.log`.
  - Can be scheduled via Windows Task Scheduler.

---
### Nemotron Historical Data Decision
- Add a configuration option in `nemotron_schedule.md` (or a config file) `process_historical: bool`.
- Update the Nemotron pipeline (if applicable) to respect this flag and skip processing files older than the threshold.

---
### Retention Policy
- Define default retention: keep files for 60 days, configurable via environment variable `DATA_RETENTION_DAYS`.
- Document policy in `docs/data_retention_policy.md`.

---
### Verification Plan
#### Automated Tests
- Add unit tests in `tests/test_cleanup.py` to verify:
  - `assess_data.py` correctly calculates size.
  - `cleanup_data.py` deletes only files older than the threshold and respects `--dry-run`.
- Run tests with `pytest tests/test_cleanup.py`.

#### Manual Verification
- Run `python scripts/assess_data.py` and confirm output matches manual `du` size.
- Execute `python scripts/cleanup_data.py --dry-run` and review logged files.
- Schedule the script via Task Scheduler and verify it runs without errors.
- Check Nemotron logs to ensure historical data processing respects the new flag.

## Verification Plan Summary
The plan includes automated unit tests and manual steps to confirm correct size reporting, safe cleanup behavior, and proper integration with Nemotron's processing pipeline.
