"""
Atlas Sentiment Server — DistilRoBERTa ONNX on Intel NPU (port 8083).
Replaces FinBERT. Classifies financial headlines as positive/negative/neutral.
~81MB model, runs on Intel AI Boost NPU via OpenVINO. Falls back to CPU.

Endpoints:
  POST /predict  {"text": "Bitcoin surges past 100K"} → {"label","score","scores"}
  POST /batch    {"texts": ["...", "..."]}             → [{"label","score","scores"}, ...]
  GET  /health   → {"status": "ok", "device": "NPU"}
"""

import json
import time
import numpy as np
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler

from openvino import Core
from tokenizers import Tokenizer

MODEL_DIR = Path(__file__).parent / "models" / "sentiment"
MODEL_PATH = str(MODEL_DIR / "model.onnx")
PORT = 8083

# ── Load model + tokenizer at startup ──────────────────────────
print(f"[ATLAS] Loading DistilRoBERTa ONNX from {MODEL_DIR} ...")

tokenizer = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
tokenizer.enable_padding(pad_id=1, pad_token="<pad>")
tokenizer.enable_truncation(max_length=512)

# Try NPU first, fall back to CPU
core = Core()
available = core.available_devices
print(f"[ATLAS] Available devices: {available}")

device = "CPU"  # default
model = core.read_model(MODEL_PATH)

# NPU requires static shapes — fix batch=1, seq_len=128
NPU_SEQ_LEN = 128

for try_device in ["NPU", "CPU"]:
    if try_device not in available:
        continue
    try:
        if try_device == "NPU":
            # Reshape to static dims for NPU
            model.reshape({
                "input_ids": [1, NPU_SEQ_LEN],
                "attention_mask": [1, NPU_SEQ_LEN],
            })
            print(f"[ATLAS] Reshaped model to static [1, {NPU_SEQ_LEN}] for NPU")
        print(f"[ATLAS] Compiling model on {try_device}...")
        compiled = core.compile_model(model, try_device)
        device = try_device
        print(f"[ATLAS] Model compiled on {device}")
        break
    except Exception as e:
        print(f"[ATLAS] {try_device} failed: {e}")
        if try_device == "NPU":
            print("[ATLAS] Falling back to CPU...")
            # Re-read model with dynamic shapes for CPU
            model = core.read_model(MODEL_PATH)

LABELS = {0: "negative", 1: "neutral", 2: "positive"}
infer_request = compiled.create_infer_request()

print(f"[ATLAS] Ready on {device}. Labels: {LABELS}")


def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def predict_batch(texts):
    encoded = tokenizer.encode_batch(texts)

    if device == "NPU":
        # NPU: static shape [1, 128] — process one at a time, pad/truncate
        all_logits = []
        for enc in encoded:
            ids = enc.ids[:NPU_SEQ_LEN]
            mask = enc.attention_mask[:NPU_SEQ_LEN]
            # Pad to fixed length
            pad_len = NPU_SEQ_LEN - len(ids)
            if pad_len > 0:
                ids = ids + [1] * pad_len      # pad_id=1
                mask = mask + [0] * pad_len
            input_ids = np.array([ids], dtype=np.int64)
            attention_mask = np.array([mask], dtype=np.int64)
            output = compiled({"input_ids": input_ids, "attention_mask": attention_mask})
            all_logits.append(list(output.values())[0][0])
        logits = np.array(all_logits)
    else:
        # CPU: dynamic shapes — batch all at once
        input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
        output = compiled({"input_ids": input_ids, "attention_mask": attention_mask})
        logits = list(output.values())[0]

    probs = softmax(logits)
    results = []
    for i in range(len(texts)):
        idx = int(np.argmax(probs[i]))
        results.append({
            "label": LABELS[idx],
            "score": round(float(probs[i][idx]), 4),
            "scores": {
                "negative": round(float(probs[i][0]), 4),
                "neutral": round(float(probs[i][1]), 4),
                "positive": round(float(probs[i][2]), 4),
            },
        })
    return results


MAX_BODY_BYTES = 65_536   # 64 KB hard limit
ALLOWED_HOSTS  = {"127.0.0.1", "localhost", f"127.0.0.1:{PORT}", f"localhost:{PORT}"}


class Handler(BaseHTTPRequestHandler):
    def _check_host(self) -> bool:
        host = self.headers.get("Host", "")
        if host not in ALLOWED_HOSTS:
            self._json_response({"error": "forbidden"}, 403)
            return False
        return True

    def _read_body(self):
        raw_len = self.headers.get("Content-Length", "")
        if not raw_len.isdigit():
            self._json_response({"error": "Content-Length required"}, 411)
            return None
        length = int(raw_len)
        if length > MAX_BODY_BYTES:
            self._json_response({"error": "request too large"}, 413)
            return None
        try:
            return json.loads(self.rfile.read(length))
        except Exception:
            self._json_response({"error": "invalid JSON"}, 400)
            return None

    def do_GET(self):
        if not self._check_host():
            return
        if self.path == "/health":
            self._json_response({
                "status": "ok",
                "model": "distilroberta-financial-sentiment",
                "device": device,
            })
        else:
            self._json_response({"error": "not found"}, 404)

    def do_POST(self):
        if not self._check_host():
            return
        body = self._read_body()
        if body is None:
            return

        if self.path == "/predict":
            text = body.get("text", "")
            if not isinstance(text, str) or not text:
                self._json_response({"error": "missing 'text'"}, 400)
                return
            result = predict_batch([text[:4096]])[0]
            self._json_response(result)

        elif self.path == "/batch":
            texts = body.get("texts", [])
            if not isinstance(texts, list) or not texts:
                self._json_response({"error": "missing 'texts'"}, 400)
                return
            texts = [str(t)[:4096] for t in texts[:200]]  # cap at 200 items
            results = predict_batch(texts)
            self._json_response(results)

        else:
            self._json_response({"error": "not found"}, 404)

    def _json_response(self, data, code=200):
        out = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, fmt, *args):
        pass  # silence per-request logs


if __name__ == "__main__":
    server = HTTPServer(("127.0.0.1", PORT), Handler)
    print(f"[ATLAS] Serving on http://127.0.0.1:{PORT} (localhost only)")
    print(f"[ATLAS]   Device: {device}")
    print(f"[ATLAS]   POST /predict  {{\"text\": \"...\"}}")
    print(f"[ATLAS]   POST /batch    {{\"texts\": [...]}}")
    print(f"[ATLAS]   GET  /health")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[ATLAS] Shutting down.")
