"""Resident Laya inference server: weights load once, POST /predict per decision.

Usage (torch lives in /tmp/nanojev-venv; laya repo used read-only):
    PYTHONPATH=$HOME/Github/laya /tmp/nanojev-venv/bin/python \\
        scripts/serve_laya.py --port 8931 [--model-dir ...] [--device cuda]

Endpoints:
    GET  /health   -> {"ready": true, "model": ...}
    POST /predict  body {state: dict, criteria: {name: desc}}
                   -> {choice, probabilities, confidence, input_tokens, model,
                       infer_ms}
"""

from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

AGENT = None
MODEL_NAME = ""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, obj: dict) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send(200, {"ready": AGENT is not None, "model": MODEL_NAME})
        else:
            self._send(404, {"error": "unknown endpoint"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/predict":
            self._send(404, {"error": "unknown endpoint"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 2_000_000:
                raise ValueError("request must be 1..2000000 bytes")
            req = json.loads(self.rfile.read(length))
            criteria = req["criteria"]
            if not 2 <= len(criteria) <= 255:
                raise ValueError(f"need 2-255 candidates, got {len(criteria)}")
            t0 = time.monotonic()
            resp = AGENT.system_one(req["state"], {"tactic": {
                "type": "choice",
                "instructions": "Given the real-time strategy game state, "
                                "pick the best tactic.",
                "criteria": criteria,
            }})
            infer_ms = (time.monotonic() - t0) * 1000.0
            ans = resp["answers"]["tactic"]
            self._send(200, {
                "choice": ans["choice"],
                "probabilities": ans.get("probabilities", {}),
                "confidence": ans.get("confidence", 0.0),
                "input_tokens": (resp.get("usage") or {}).get("input_tokens", 0),
                "model": resp.get("model", "laya-rl-agent"),
                "infer_ms": round(infer_ms, 1),
            })
        except (ValueError, KeyError, TypeError) as e:
            self._send(400, {"error": str(e)})
        except Exception:  # noqa: BLE001
            self._send(500, {"error": "laya inference failed; see server log"})
            raise

    def log_message(self, fmt, *args):  # noqa: N802
        print(fmt % args, flush=True)


def main() -> int:
    global AGENT, MODEL_NAME
    from laya_predict import _resolve_model_dir  # noqa: E402

    p = argparse.ArgumentParser(description="Resident Laya server.")
    p.add_argument("--port", type=int, default=8931)
    p.add_argument("--model-dir", default=None)
    p.add_argument("--device", default=None)
    a = p.parse_args()

    import laya  # noqa: E402

    model_dir = _resolve_model_dir(a.model_dir)
    print(f"[serve_laya] loading {model_dir} ...", flush=True)
    t0 = time.monotonic()
    AGENT = laya.load(model_dir, device=a.device)
    MODEL_NAME = "laya-rl-agent"
    print(f"[serve_laya] ready in {(time.monotonic()-t0):.1f}s", flush=True)
    print(json.dumps({"url": f"http://127.0.0.1:{a.port}", "ready": True}),
          flush=True)
    server = HTTPServer(("127.0.0.1", a.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
