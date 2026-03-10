#!/usr/bin/env python
"""
Reset persisted state for Nemo + portfolio data.

Deletes state files that accumulate cached market context so you can restart
`scripts/nemo_ws_service.py` fresh. Safe to run while the service is stopped.
"""

from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
TARGETS = [
    "data/nemo_state.json",
    "data/watchdog_events.jsonl",
    "data/watchdog_status.json",
    "data/heartbeat_trader.json",
    "data/heartbeat_cvar.json",
    "data/cvar_weights.json",
    "data/positions.json",
]


def main():
    deleted = []
    for rel in TARGETS:
        path = ROOT / rel
        if path.exists():
            try:
                path.unlink()
                deleted.append(rel)
            except OSError:
                print(f"failed to delete {rel}")
    if deleted:
        print("deleted:", ", ".join(deleted))
    else:
        print("nothing to delete; files already clean.")


if __name__ == "__main__":
    main()
