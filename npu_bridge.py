"""
NPU Signal Bridge — kraken-hybrad9
===================================
Two-layer signal engine feeding Nemo:

  LAYER 1 — EXIT WATCHER (open positions only)
    Reads positions.json, tracks live prices, runs NPU dynamic exit engine
    (momentum, Monte Carlo win prob, drawdown, pattern recognition).

  LAYER 2 — ENTRY SCANNER (all active coins)
    Reads features_snapshot.json (51 coins, updated every tick by Rust bot),
    scores every coin for momentum strength, reversal risk, and entry quality.
    Nemo sees a ranked hot-list and avoid-list before deciding entries.

Output: data/npu_signals.json  (read by Rust bot every tick)

Usage: python npu_bridge.py
"""

import json
import os
import sys
import time
import requests
from pathlib import Path
from datetime import datetime

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_DIR      = Path(__file__).resolve().parent
POSITIONS_FILE   = PROJECT_DIR / "data" / "positions.json"
SNAPSHOT_FILE    = PROJECT_DIR / "data" / "features_snapshot.json"
SIGNALS_FILE     = PROJECT_DIR / "data" / "npu_signals.json"
NPU_KRAKEN_DIR   = Path(r"C:\Users\kitti\Desktop\NPU-Kraken")

# ── Config ───────────────────────────────────────────────────────────────────
POLL_SEC      = 8        # How often to re-analyze (seconds)
MIN_HISTORY   = 10       # Minimum price ticks before analysis is meaningful
KRAKEN_REST   = "https://api.kraken.com/0/public/Ticker"

# ── Import NPU-Kraken dynamic exit engine ────────────────────────────────────
sys.path.insert(0, str(NPU_KRAKEN_DIR))
try:
    from npu.dynamic_exit import NPUDynamicExit
    print("[NPU-BRIDGE] NPU dynamic exit engine loaded OK")
except Exception as e:
    print(f"[NPU-BRIDGE] FATAL: could not load NPU engine: {e}")
    sys.exit(1)

# ── Kraken pair map ──────────────────────────────────────────────────────────
PAIR_MAP = {
    "BTC": "XXBTZUSD", "ETH": "XETHZUSD", "SOL": "SOLUSD",
    "XRP": "XXRPZUSD", "ADA": "ADAUSD",   "DOT": "DOTUSD",
    "DOGE": "XDOGEUSD","LTC": "XLTCZUSD", "NEAR": "NEARUSD",
    "APT": "APTUSD",   "JTO": "JTOUSD",   "FLOKI": "FLOKIUSD",
    "FLR": "FLRUSD",   "ARB": "ARBUSD",   "ATOM": "ATOMUSD",
    "AVAX": "AVAXUSD", "PEPE": "PEPEUSD", "INJ": "INJUSD",
    "HBAR": "HBARUSD", "SUI": "SUIUSD",   "UNI": "UNIUSD",
    "AAVE": "AAVEUSD", "JUP": "JUPUSD",   "ONDO": "ONDOUSD",
    "FIL": "FILUSD",   "COMP": "COMPUSD", "CRV": "CRVUSD",
    "BERA": "BERAUSD", "WIF": "WIFUSD",   "ENA": "ENAUSD",
    "BCH": "BCHUSD",   "VET": "VETUSD",   "KAS": "KASUSD",
    "SEI": "SEIUSD",   "RUNE": "RUNEUSD", "TAO": "TAOUSD",
    "PENDLE": "PENDLEUSD", "TIA": "TIAUSD",
}


def kraken_pair(symbol: str) -> str:
    return PAIR_MAP.get(symbol.upper(), f"{symbol.upper()}USD")


def get_live_price(symbol: str) -> float:
    """Fetch last trade price from Kraken public REST."""
    try:
        pair = kraken_pair(symbol)
        r = requests.get(KRAKEN_REST, params={"pair": pair}, timeout=5)
        data = r.json()
        if data.get("error"):
            return 0.0
        result = data.get("result", {})
        for v in result.values():
            return float(v["c"][0])  # last trade price
    except Exception:
        pass
    return 0.0


def load_positions() -> list:
    """Load open positions from positions.json."""
    try:
        with open(POSITIONS_FILE) as f:
            return json.load(f)
    except Exception:
        return []


def load_snapshot() -> dict:
    """Load features_snapshot.json written by the Rust bot every tick."""
    try:
        with open(SNAPSHOT_FILE) as f:
            data = json.load(f)
            return data.get("features", {})
    except Exception:
        return {}


def score_coin(sym: str, f: dict) -> dict:
    """
    NPU entry scan score for one coin.
    Uses features already computed by the Rust GPU math stack.
    Returns a compact signal dict.
    """
    momentum   = float(f.get("momentum_score", 0.0))
    trend      = float(f.get("trend_score", 0.0))
    rsi        = float(f.get("rsi", 50.0))
    vol_ratio  = float(f.get("vol_ratio", 1.0))
    book_rev   = bool(f.get("book_reversal", False))
    book_imb   = float(f.get("book_imbalance", 0.0))
    spread     = float(f.get("spread_pct", 0.0))
    zscore     = float(f.get("zscore", 0.0))
    math_act   = str(f.get("math_action", "HOLD"))
    math_conf  = float(f.get("math_confidence", 0.0))
    crash      = bool(f.get("crash", False))
    gbdt       = float(f.get("gbdt_score", 0.0))
    price      = float(f.get("price", 0.0))

    # ── NPU Entry Score (0-100) ───────────────────────────────────────────
    # Strong momentum + bullish trend + healthy book = high score
    score = 50.0
    score += min(momentum * 15, 25)          # momentum: up to +25
    score += min(trend * 3, 15)              # trend: up to +15
    score += min((vol_ratio - 1.0) * 5, 10) # vol surge: up to +10
    score += book_imb * 10                   # book pressure
    score -= abs(zscore) * 2                 # mean-reversion penalty
    score -= spread * 20                     # wide spread penalty
    if rsi > 75: score -= 15                 # overbought
    if rsi < 30: score -= 10                 # oversold/crash
    if book_rev:  score -= 20               # book reversal = bearish
    if crash:     score -= 30               # crash flag
    if math_act == "SELL": score -= 10
    if math_act == "BUY":  score += 10
    if gbdt > 0.6:         score += 5
    score = max(0.0, min(100.0, score))

    # ── Reversal risk ────────────────────────────────────────────────────
    reversal_risk = "LOW"
    if book_rev or rsi > 72 or momentum < -0.2:
        reversal_risk = "HIGH"
    elif momentum < -0.05 or trend < 0:
        reversal_risk = "MEDIUM"

    # ── Entry recommendation ─────────────────────────────────────────────
    if score >= 70 and reversal_risk == "LOW":
        recommendation = "STRONG_BUY"
    elif score >= 55:
        recommendation = "BUY"
    elif score <= 25 or reversal_risk == "HIGH":
        recommendation = "AVOID"
    else:
        recommendation = "WATCH"

    return {
        "symbol":        sym,
        "npu_score":     round(score, 1),
        "momentum":      round(momentum, 2),
        "trend":         round(trend, 1),
        "rsi":           round(rsi, 1),
        "vol_ratio":     round(vol_ratio, 2),
        "reversal_risk": reversal_risk,
        "recommendation": recommendation,
        "math_action":   math_act,
        "math_conf":     round(math_conf, 2),
        "price":         price,
        "ts":            time.time(),
    }


def write_signals(signals: dict):
    """Atomically write NPU signals to data/npu_signals.json."""
    tmp = str(SIGNALS_FILE) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(signals, f, indent=2)
    os.replace(tmp, str(SIGNALS_FILE))


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    print(f"[NPU-BRIDGE] Starting — writing signals to {SIGNALS_FILE}")
    print(f"[NPU-BRIDGE] Poll interval: {POLL_SEC}s")

    engine = NPUDynamicExit()
    tracked: set = set()
    tick_count = 0

    while True:
        try:
            positions = load_positions()
            signals = {
                "ts": time.time(),
                "generated": datetime.utcnow().isoformat() + "Z",
                "positions": {}
            }

            active_symbols = {p["symbol"] for p in positions}

            # Remove stale tracked positions
            for sym in list(tracked):
                if sym not in active_symbols:
                    try:
                        engine.close_position(sym)
                    except Exception:
                        pass
                    tracked.discard(sym)

            for pos in positions:
                sym = pos["symbol"]
                entry = pos.get("entry") or pos.get("entry_price", 0.0)
                qty   = pos.get("remaining_qty") or pos.get("qty", 0.0)

                if entry <= 0 or qty <= 0:
                    continue

                # Register new positions
                if sym not in tracked:
                    try:
                        engine.register_position(sym, entry, qty)
                        tracked.add(sym)
                        print(f"[NPU-BRIDGE] Registered {sym} @ {entry:.6f}")
                    except Exception as e:
                        print(f"[NPU-BRIDGE] Register error {sym}: {e}")
                        continue

                # Get live price
                price = get_live_price(sym)
                if price <= 0:
                    print(f"[NPU-BRIDGE] No price for {sym} — skipping")
                    continue

                # Update price history
                try:
                    engine.update_price(sym, price)
                except Exception as e:
                    print(f"[NPU-BRIDGE] update_price error {sym}: {e}")
                    continue

                # Run exit analysis
                try:
                    analysis = engine.analyze_exit(sym, price)
                    if analysis is None:
                        continue

                    exit_score    = getattr(analysis, "exit_score", 0)
                    decision      = getattr(analysis, "decision", None)
                    decision_str  = decision.value if decision else "HOLD"
                    momentum      = getattr(analysis, "momentum_score", 0.0)
                    win_prob      = getattr(analysis, "win_probability", 0.5)
                    drawdown      = getattr(analysis, "drawdown_from_peak", 0.0)
                    current_pnl   = getattr(analysis, "current_pnl_pct", 0.0)
                    peak_pnl      = getattr(analysis, "peak_pnl_pct", 0.0)
                    reasons       = getattr(analysis, "reasons", [])
                    urgency       = getattr(analysis, "urgency", 0.0)

                    action = analysis.get_exit_action() if hasattr(analysis, "get_exit_action") else decision_str

                    sig = {
                        "symbol":       sym,
                        "price":        price,
                        "exit_score":   round(exit_score, 1),
                        "decision":     decision_str,
                        "action":       action,
                        "momentum":     round(float(momentum), 1),
                        "win_prob":     round(float(win_prob), 2),
                        "drawdown_pct": round(float(drawdown), 2),
                        "current_pnl":  round(float(current_pnl), 2),
                        "peak_pnl":     round(float(peak_pnl), 2),
                        "urgency":      round(float(urgency), 2),
                        "reasons":      reasons[:3],  # top 3 reasons
                        "ts":           time.time(),
                    }
                    signals["positions"][sym] = sig

                    # Log actionable signals
                    if exit_score >= 50:
                        print(f"[NPU-BRIDGE] ⚠  {sym} exit_score={exit_score:.0f} "
                              f"decision={decision_str} momentum={momentum:.0f} "
                              f"pnl={current_pnl:+.2f}% urgency={urgency:.2f}")

                except Exception as e:
                    print(f"[NPU-BRIDGE] analyze_exit error {sym}: {e}")

            # ── LAYER 2: Entry scanner — all active coins ─────────────
            snapshot = load_snapshot()
            scan_results = {}
            for sym, feats in snapshot.items():
                try:
                    scan_results[sym] = score_coin(sym, feats)
                except Exception:
                    pass

            # Rank and summarise
            ranked = sorted(scan_results.values(), key=lambda x: x["npu_score"], reverse=True)
            hot    = [r["symbol"] for r in ranked if r["recommendation"] in ("STRONG_BUY", "BUY")][:5]
            avoid  = [r["symbol"] for r in ranked if r["recommendation"] == "AVOID"][:5]

            signals["scan"] = {
                "total_coins": len(scan_results),
                "hot":   hot,
                "avoid": avoid,
                "scores": {r["symbol"]: r for r in ranked},
                "ts": time.time(),
            }

            write_signals(signals)
            tick_count += 1

            if tick_count % 5 == 0:
                pos_syms = list(signals["positions"].keys())
                print(f"[NPU-BRIDGE] tick={tick_count} positions={pos_syms} "
                      f"hot={hot[:3]} avoid={avoid[:3]}")

        except Exception as e:
            print(f"[NPU-BRIDGE] loop error: {e}")

        time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()
