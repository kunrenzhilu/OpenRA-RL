"""Resident NanoJev inference server: 2.3GB weights load once.

Same payload contract as predict_toy_decisions / serve_decisions:
    POST /predict  body {states: [{id, state, questions: {tactic: ...}}]}
                   -> DecisionPredictor result dict (+infer_s)

Usage (torch lives in /tmp/nanojev-venv):
    /tmp/nanojev-venv/bin/python scripts/serve_nanojev.py \\
        --checkpoint-dir /tmp/NanoJev-unified --port 8932
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ENGINE = None


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, obj: dict) -> None:
        body = json.dumps(obj, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send(200, {"ready": ENGINE is not None})
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
            from predict_toy_decisions import validate_request
            payload = json.loads(self.rfile.read(length))
            states = validate_request(payload)
            questions = [q for s in states for q in s["questions"].values()]
            paths = sum(1 if q["type"] == "boolean" else len(q["criteria"])
                        for q in questions)
            if len(states) > 32 or len(questions) > 96 or paths > 256:
                raise ValueError("limit: 32 states, 96 questions, 256 paths")
            t0 = time.perf_counter()
            result = ENGINE.predict(payload)
            result.setdefault("execution", {})["server_evaluation_seconds"] = (
                time.perf_counter() - t0)
            self._send(200, result)
        except (ValueError, TypeError, KeyError) as e:
            self._send(400, {"error": str(e)})
        except Exception:  # noqa: BLE001
            self._send(500, {"error": "nanojev inference failed; see server log"})
            raise

    def log_message(self, fmt, *args):  # noqa: N802
        print(fmt % args, flush=True)


def main() -> int:
    global ENGINE
    p = argparse.ArgumentParser(description="Resident NanoJev server.")
    p.add_argument("--checkpoint-dir", default="/tmp/NanoJev-unified")
    p.add_argument("--port", type=int, default=8932)
    p.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    p.add_argument("--predict-script-dir",
                   default=str(Path.home() / "Github" / "NanoJev" / "scripts"))
    a = p.parse_args()

    sys.path.insert(0, a.predict_script_dir)
    from predict_toy_decisions import DecisionPredictor  # noqa: E402

    print(f"[serve_nanojev] loading {a.checkpoint_dir} ...", flush=True)
    t0 = time.monotonic()
    ENGINE = DecisionPredictor(a.checkpoint_dir, precision=a.precision)
    print(f"[serve_nanojev] ready in {(time.monotonic()-t0):.1f}s", flush=True)
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
