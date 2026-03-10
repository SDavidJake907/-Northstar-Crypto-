#!/usr/bin/env python
"""
Portfolio float & hold monitor
-----------------------------
Reads `data/positions.json` plus the persisted `data/nemo_state.json`
to compute how long each position has been open and the current float
(unrealized PnL).

Run this alongside the WS service to keep an eye on exposure, rotation,
and how much USD is sitting in each symbol.
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
POSITIONS_PATH = ROOT / "data" / "positions.json"
STATE_PATH = ROOT / "data" / "nemo_state.json"


def load_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def newest_price(symbol: str, state):
    if not state:
        return None
    trades = state.get("trades", {}).get(symbol, [])
    for item in reversed(trades):
        if isinstance(item, dict):
            price = item.get("price") or item.get("P")
        elif isinstance(item, list) and item:
            price = item[0]
        else:
            continue
        try:
            return float(price)
        except (TypeError, ValueError):
            continue

    book = state.get("book", {}).get(symbol)
    if book:
        data = book.get("data", [])
        if isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, dict):
                mid = first.get("mid")
                if mid:
                    return float(mid)
            elif isinstance(first, list):
                try:
                    return float(first[2])
                except (IndexError, ValueError):
                    pass
    ohlc = state.get("ohlc", {}).get(symbol, [])
    if ohlc:
        last = ohlc[-1]
        if isinstance(last, dict):
            close = last.get("close")
        elif isinstance(last, list) and len(last) > 4:
            close = last[4]
        else:
            close = None
        if close:
            try:
                return float(close)
            except (TypeError, ValueError):
                pass
    return None


def format_delta(seconds: float) -> str:
    delta = timedelta(seconds=int(seconds))
    days = delta.days
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m {seconds}s"


def summarize_positions(state, positions):
    now = time.time()
    header = f"{'Symbol':<6} {'Hold':<10} {'Entry':>9} {'Price':>8} {'Float':>10}"
    rows = []
    total_float = 0.0
    for pos in positions:
        symbol = pos.get("symbol", "UNKNOWN")
        entry = pos.get("entry", 0.0)
        qty = pos.get("qty", 0.0)
        entry_time = pos.get("entry_time", now)
        hold_secs = max(0.0, now - entry_time)
        hold = format_delta(hold_secs)
        price = newest_price(symbol, state) or entry
        pnl = (price - entry) * qty
        float_str = f"${pnl:,.2f}"
        rows.append((symbol, hold, f"{entry:.5g}", f"{price:.5g}", float_str))
        total_float += pnl
    rows.sort(key=lambda r: r[0])
    return header, rows, total_float


def main():
    parser = argparse.ArgumentParser(description="Monitor portfolio float + hold time.")
    parser.add_argument("--state", default=os.environ.get("NEMO_STATE_PATH", str(STATE_PATH)))
    args = parser.parse_args()

    positions = load_json(POSITIONS_PATH)
    if not positions:
        print(f"{POSITIONS_PATH} missing or empty", file=sys.stderr)
        sys.exit(1)
    state = load_json(Path(args.state))

    header, rows, total = summarize_positions(state, positions)
    print(header)
    print("-" * len(header))
    for sym, hold, entry, price, float_str in rows:
        print(f"{sym:<6} {hold:<10} {entry:>9} {price:>8} {float_str:>10}")
    print("-" * len(header))
    print(f"{'Total float':<6} {total:>34,.2f}")


if __name__ == "__main__":
    main()
