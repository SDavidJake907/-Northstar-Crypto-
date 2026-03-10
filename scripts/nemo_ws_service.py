#!/usr/bin/env python
"""
Nemo WS Service
---------------
Standalone service that connects to Kraken WS v2, maintains live market context,
and persists state to data/nemo_state.json so Nemo can recover context even if
the Rust core restarts.

Optional local HTTP endpoints:
  GET  /state      -> current in-memory state (JSON)
  POST /positions  -> update positions snapshot (JSON body)

Dependencies:
  pip install websockets
"""

import asyncio
import json
import os
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

try:
    import websockets
except Exception:
    websockets = None


ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"
STATE_PATH = ROOT / "data" / "nemo_state.json"


def load_env(path: Path) -> dict:
    env = {}
    if not path.exists():
        return env
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def now_ts() -> float:
    return time.time()


class StateStore:
    def __init__(self, max_trades: int):
        self.lock = threading.Lock()
        self.state = {
            "ts": now_ts(),
            "symbols": [],
            "channels": [],
            "stats": {"messages": 0, "trade": 0, "book": 0, "ohlc": 0},
            "trades": {},
            "book": {},
            "ohlc": {},
            "positions": {},
        }
        self.max_trades = max_trades

    def update_meta(self, symbols, channels):
        with self.lock:
            self.state["symbols"] = symbols
            self.state["channels"] = channels

    def add_trade(self, symbol: str, item: dict):
        with self.lock:
            self.state["stats"]["trade"] += 1
            self.state["trades"].setdefault(symbol, [])
            self.state["trades"][symbol].append(item)
            if len(self.state["trades"][symbol]) > self.max_trades:
                self.state["trades"][symbol] = self.state["trades"][symbol][-self.max_trades :]

    def set_book(self, symbol: str, payload):
        with self.lock:
            self.state["stats"]["book"] += 1
            self.state["book"][symbol] = payload

    def set_ohlc(self, symbol: str, payload):
        with self.lock:
            self.state["stats"]["ohlc"] += 1
            self.state["ohlc"][symbol] = payload

    def bump_messages(self):
        with self.lock:
            self.state["stats"]["messages"] += 1
            self.state["ts"] = now_ts()

    def set_positions(self, payload):
        with self.lock:
            self.state["positions"] = payload

    def snapshot(self):
        with self.lock:
            return json.loads(json.dumps(self.state))


class ApiHandler(BaseHTTPRequestHandler):
    store = None  # set later

    def _send(self, code, body):
        data = json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path != "/state":
            return self._send(404, {"error": "not_found"})
        return self._send(200, ApiHandler.store.snapshot())

    def do_POST(self):
        if self.path != "/positions":
            return self._send(404, {"error": "not_found"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8", errors="ignore")
            payload = json.loads(raw) if raw else {}
        except Exception:
            return self._send(400, {"error": "invalid_json"})
        ApiHandler.store.set_positions(payload)
        return self._send(200, {"ok": True})

    def log_message(self, format, *args):
        return


async def ws_loop(store: StateStore, env: dict, stop_event: asyncio.Event):
    if websockets is None:
        raise RuntimeError("websockets package not installed. Run: pip install websockets")

    ws_url = env.get("NEMO_WS_URL", "wss://ws.kraken.com/v2")
    channels = [c.strip() for c in env.get("NEMO_WS_CHANNELS", "trade").split(",") if c.strip()]
    raw_symbols = [s.strip() for s in env.get("SYMBOLS", "").split(",") if s.strip()]
    if not raw_symbols:
        raise RuntimeError("SYMBOLS is empty. Set SYMBOLS in .env")
    # Kraken WS v2 requires "BTC/USD" format, .env has bare "BTC"
    symbols = [s if "/" in s else f"{s}/USD" for s in raw_symbols]

    depth = int(env.get("WS_DEPTH", "5"))
    ohlc_interval = int(env.get("WS_OHLC_INTERVAL", "1"))
    store.update_meta(symbols, channels)
    print(f"[nemo-ws] connecting to {ws_url} — {len(symbols)} symbols, channels={channels}", flush=True)

    async with websockets.connect(ws_url, ping_interval=20, ping_timeout=20) as ws:
        for ch in channels:
            params = {"channel": ch, "symbol": symbols}
            if ch == "book":
                params["depth"] = depth
            if ch == "ohlc":
                params["interval"] = ohlc_interval
            sub = {"method": "subscribe", "params": params}
            await ws.send(json.dumps(sub))

        while not stop_event.is_set():
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=30)
            except asyncio.TimeoutError:
                continue

            store.bump_messages()
            try:
                v = json.loads(msg)
            except Exception:
                continue

            channel = v.get("channel")
            msg_type = v.get("type")
            if msg_type in ("subscribed", "unsubscribed", "error"):
                continue
            data = v.get("data") or []

            if channel == "trade":
                for item in data:
                    sym = item.get("symbol") or item.get("pair") or "UNKNOWN"
                    store.add_trade(sym, item)
            elif channel == "book":
                for item in data:
                    sym = item.get("symbol") or item.get("pair") or "UNKNOWN"
                    store.set_book(sym, item)
            elif channel == "ohlc":
                for item in data:
                    sym = item.get("symbol") or item.get("pair") or "UNKNOWN"
                    store.set_ohlc(sym, item)


async def persist_loop(store: StateStore, interval_sec: int, stop_event: asyncio.Event):
    while not stop_event.is_set():
        snapshot = store.snapshot()
        atomic_write(STATE_PATH, json.dumps(snapshot))
        await asyncio.sleep(interval_sec)


def run_http_server(store: StateStore, host: str, port: int):
    ApiHandler.store = store
    httpd = ThreadingHTTPServer((host, port), ApiHandler)
    httpd.serve_forever()


async def main():
    env = load_env(ENV_PATH)
    max_trades = int(env.get("NEMO_MAX_TRADES", "200"))
    persist_sec = int(env.get("NEMO_STATE_WRITE_SEC", "5"))
    http_host = env.get("NEMO_HTTP_HOST", "127.0.0.1")
    http_port = int(env.get("NEMO_HTTP_PORT", "8079"))

    store = StateStore(max_trades=max_trades)
    stop_event = asyncio.Event()

    def _stop(*_):
        stop_event.set()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    t = threading.Thread(target=run_http_server, args=(store, http_host, http_port), daemon=True)
    t.start()

    await asyncio.gather(
        ws_loop(store, env, stop_event),
        persist_loop(store, persist_sec, stop_event),
    )


if __name__ == "__main__":
    asyncio.run(main())
