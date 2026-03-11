"""
Phi-3 Lane Scanner — Intel NPU via OpenVINO GenAI (port 8084).
Fast pre-filter for entry candidates before the 9B sees them.

Given coin features, outputs: PASS or REJECT + lane + reason.
Runs in ~300-800ms on NPU, saving 11s 9B calls on bad setups.

Endpoints:
  POST /scan   {"sym":"LINK","price":8.78,"rsi":50,"trend":-1,...} -> {"verdict","lane","action","reason","latency_ms"}
  GET  /health -> {"status":"ok","device":"NPU"}
"""

import json
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

import openvino_genai as ov_genai

MODEL_DIR = Path(r"C:\Users\kitti\Projects\kraken-hybrad9\openvino_models\phi-3-mini-int4")
PORT = 8084
MAX_BODY = 65_536
ALLOWED_HOSTS = {"127.0.0.1", "localhost"}

print(f"[PHI3] Loading Phi-3-mini-int4 from {MODEL_DIR} ...")
t0 = time.time()

# Try NPU first, fall back to CPU (GPU is maxed at 13GB)
pipe = None
device_used = "CPU"
for dev in ["NPU", "CPU"]:
    try:
        pipe = ov_genai.LLMPipeline(str(MODEL_DIR), dev)
        device_used = dev
        print(f"[PHI3] Pipeline loaded on {dev} in {time.time()-t0:.1f}s")
        break
    except Exception as e:
        print(f"[PHI3] {dev} failed: {e}")

if pipe is None:
    raise RuntimeError("[PHI3] Could not load Phi-3 on any device")

# Warm up — first inference is slow (JIT compilation)
print("[PHI3] Warming up...")
t1 = time.time()
_ = pipe.generate("PASS or REJECT: BTC trend=+3 imb=+0.30",
                  max_new_tokens=10, do_sample=False)
print(f"[PHI3] Warmup done in {time.time()-t1:.1f}s — ready on {device_used}")


SYSTEM_PROMPT = (
    "You are GATE — the first filter in a live crypto trading pipeline. "
    "Your job: scan coin metrics and decide PASS or REJECT before the entry AI sees them. "
    "You run on NPU for speed. You must be fast and decisive. "
    "\n\n"
    "PASS means: this setup has potential — send to SCOUT for full analysis. "
    "REJECT means: this setup has a clear disqualifying signal — block it now. "
    "\n\n"
    "PASS when ANY of: "
    "vol >= 2.0x (hot mover — never block these) | "
    "trend >= +3 AND mom > 0.1 (strong bullish momentum) | "
    "imbalance > +0.30 AND vol >= 1.5x (real buy pressure) | "
    "regime is bullish or trending AND trend >= +2. "
    "\n\n"
    "REJECT when ALL of these are true: "
    "trend <= -2 AND mom < -0.05 AND vol < 1.2x (weak bearish no volume). "
    "Or: spread > 8% (cost too high). "
    "Or: RSI > 88 (extreme overbought with no vol backing). "
    "\n\n"
    "LANE assignment: "
    "L1 = momentum play (vol >= 2x OR trend >= +3). "
    "L2 = compression or standard setup (everything else). "
    "\n\n"
    "CRITICAL: When vol >= 3x — ALWAYS PASS. Hot movers are Simeon's core edge. Never block them. "
    "\n\n"
    "Output exactly three colon-separated fields and nothing else: "
    "PASS or REJECT : L1 or L2 : reason (max 10 words). "
    "\n\n"
    "Examples: "
    "PASS:L1:vol=3.2x hot mover momentum confirmed | "
    "PASS:L2:trend +3 imbalance positive standard entry | "
    "REJECT:L2:trend -3 momentum negative low volume | "
    "PASS:L1:vol=4x always pass hot mover"
)


def build_prompt(data: dict) -> str:
    sym     = data.get("sym", "?")
    price   = data.get("price", 0.0)
    rsi     = data.get("rsi", 50.0)
    trend   = data.get("trend", 0)
    imb     = data.get("imb", 0.0)
    mom     = data.get("mom", 0.0)
    vol     = data.get("vol", 1.0)
    spread  = data.get("spread", 0.0)
    regime  = data.get("regime", "unknown")
    tier    = data.get("tier", "mid")

    coin_line = (
        f"{sym} ${price:.4f} | RSI={rsi:.0f} trend={trend:+} "
        f"imb={imb:+.2f} mom={mom:+.3f} vol={vol:.1f}x "
        f"spread={spread:.2f}% regime={regime} tier={tier}"
    )

    # Phi-3 chat template
    return (
        f"<|system|>\n{SYSTEM_PROMPT}<|end|>\n"
        f"<|user|>\n{coin_line}<|end|>\n"
        f"<|assistant|>\n"
    )


def parse_response(raw: str) -> tuple[str, str, str, str]:
    """Parse 'PASS:L2:reason' → (verdict, lane, action, reason)"""
    raw = raw.strip().split("\n")[0].strip()
    parts = [p.strip() for p in raw.split(":")]
    verdict = "PASS" if parts and parts[0].upper() == "PASS" else "REJECT"
    lane = "L2"
    if len(parts) > 1 and parts[1].upper() in ("L1", "L2"):
        lane = parts[1].upper()
    reason = ":".join(parts[2:]).strip() if len(parts) > 2 else raw
    action = "BUY" if verdict == "PASS" else "HOLD"
    return verdict, lane, action, reason[:120]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # suppress default access log

    def _check_host(self) -> bool:
        host = self.headers.get("Host", "").split(":")[0].strip()
        return host in ALLOWED_HOSTS

    def _read_body(self) -> bytes | None:
        try:
            length = int(self.headers.get("Content-Length", 0))
            if length > MAX_BODY:
                return None
            return self.rfile.read(length)
        except Exception:
            return None

    def _send_json(self, code: int, obj: dict):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if not self._check_host():
            self._send_json(403, {"error": "forbidden"})
            return
        if self.path == "/health":
            self._send_json(200, {"status": "ok", "device": device_used})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if not self._check_host():
            self._send_json(403, {"error": "forbidden"})
            return
        if self.path != "/scan":
            self._send_json(404, {"error": "not found"})
            return

        raw = self._read_body()
        if raw is None:
            self._send_json(400, {"error": "bad request"})
            return

        try:
            data = json.loads(raw)
        except Exception:
            self._send_json(400, {"error": "invalid json"})
            return

        prompt = build_prompt(data)
        t_start = time.perf_counter()
        try:
            response = pipe.generate(
                prompt,
                max_new_tokens=40,
                do_sample=False,
                temperature=1.0,
            )
            raw_text = str(response).strip()
        except Exception as e:
            self._send_json(500, {"error": str(e)})
            return

        latency_ms = int((time.perf_counter() - t_start) * 1000)
        verdict, lane, action, reason = parse_response(raw_text)

        print(f"[PHI3] {data.get('sym','?')} -> {verdict}:{lane} | {reason} ({latency_ms}ms)")

        self._send_json(200, {
            "verdict":    verdict,
            "lane":       lane,
            "action":     action,
            "reason":     reason,
            "latency_ms": latency_ms,
            "raw":        raw_text,
        })


print(f"[PHI3] Listening on http://127.0.0.1:{PORT}/scan")
server = HTTPServer(("127.0.0.1", PORT), Handler)
server.serve_forever()
