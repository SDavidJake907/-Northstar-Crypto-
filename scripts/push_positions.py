#!/usr/bin/env python
"""
Push positions snapshot to Nemo HTTP service.

Reads `data/positions.json` and POSTs the payload to
`http://{host}:{port}/positions`. By default the call is fire-and-forget,
but you can run it on a timer to keep Nemo in sync with the Rust bot.

Usage:
  python scripts/push_positions.py         # send once
  python scripts/push_positions.py --loop --interval 30
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    print("Install `requests` first: pip install requests", file=sys.stderr)
    sys.exit(1)


ROOT = Path(__file__).resolve().parents[1]
POSITIONS_PATH = ROOT / "data" / "positions.json"


def read_positions() -> dict:
    if not POSITIONS_PATH.exists():
        raise FileNotFoundError(f"{POSITIONS_PATH} missing")
    with open(POSITIONS_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def push(host: str, port: int) -> None:
    url = f"http://{host}:{port}/positions"
    payload = read_positions()
    resp = requests.post(url, json=payload, timeout=5)
    resp.raise_for_status()
    print(f"[push] {len(payload)} entries -> {url} ({resp.status_code})")


def main():
    parser = argparse.ArgumentParser(description="Push positions to Nemo HTTP service.")
    parser.add_argument("--host", default=os.environ.get("NEMO_HTTP_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("NEMO_HTTP_PORT", "8079")))
    parser.add_argument("--loop", action="store_true", help="keep pushing every --interval seconds")
    parser.add_argument("--interval", type=int, default=30, help="seconds between pushes (loop mode)")
    args = parser.parse_args()

    while True:
        try:
            push(args.host, args.port)
        except Exception as exc:
            print(f"[push] error: {exc}", file=sys.stderr)
        if not args.loop:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
