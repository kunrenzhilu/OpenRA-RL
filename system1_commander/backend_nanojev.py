"""NanoJev local backend: subprocess fan-out to the NanoJev checkpoint.

Runs one ``Choice`` question per decision ("tactic": pick a candidate) in a
subprocess under ``/tmp/nanojev-venv`` (torch 2.14.0 / transformers 5.17.0 /
CUDA), via ``scripts/predict_toy_decisions.py`` ``DecisionPredictor`` CLI.
The demo interpreter (``.venv-taskC``) never imports torch.

Payload: adaptor state JSON dumped to ``str`` + candidate name/description
criteria (Choice 2-255). Over-long inputs ERROR out of the checkpoint
(``max_length`` exceeded -> RuntimeError, never truncated). Per-request
limits of ``serve_decisions.py`` (32 states / 96 questions / 256 paths) are
nowhere near hit: 1 state / 1 question / <=9 paths per call.

Trade-off: weights reload per predict() (~2.3GB). Slow but stateless and
robust; fine for the taskC smoke. A persistent worker can come in taskD.

Transport selection (2026-09-21 play-to-end):
  - resident server (preferred for full games): NANOJEV_URL=http://127.0.0.1:8932
    served by scripts/serve_nanojev.py (DecisionPredictor loads once; plain
    urllib POST, stdlib only, no new deps in the driver venv).
  - subprocess fallback: one nanojev-venv call per decision (~8s).
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

from system1_commander.backend_base import Prediction, System1Backend
from system1_commander.candidates import Candidate

DEFAULT_CHECKPOINT_DIR = "/tmp/NanoJev-unified"
DEFAULT_PYTHON_BIN = "/tmp/nanojev-venv/bin/python"
DEFAULT_PREDICT_SCRIPT = (
    Path.home() / "Github" / "NanoJev" / "scripts" / "predict_toy_decisions.py"
)
SUBPROCESS_TIMEOUT_S = 600.0
HTTP_TIMEOUT_S = 300.0


def _resolve(name: str, explicit: str | None, default: str) -> str:
    return explicit or os.environ.get(name, default)


class NanoJevBackend(System1Backend):
    name = "nanojev"

    def __init__(self, checkpoint_dir: str | None = None,
                 python_bin: str | None = None,
                 predict_script: str | None = None,
                 timeout_s: float = SUBPROCESS_TIMEOUT_S,
                 max_length: int | None = None,
                 server_url: str | None = None):
        self.server_url = server_url or os.environ.get("NANOJEV_URL", "")
        self.checkpoint_dir = _resolve("NANOJEV_CHECKPOINT_DIR", checkpoint_dir,
                                       DEFAULT_CHECKPOINT_DIR)
        self.python_bin = _resolve("NANOJEV_PYTHON", python_bin, DEFAULT_PYTHON_BIN)
        self.predict_script = _resolve("NANOJEV_PREDICT_SCRIPT", predict_script,
                                       str(DEFAULT_PREDICT_SCRIPT))
        self.timeout_s = timeout_s
        self.max_length = max_length
        if self.server_url:
            return  # resident server owns weights; nothing local to check
        problems = [
            f"{label} missing: {path}"
            for label, path in (
                ("checkpoint-dir", self.checkpoint_dir),
                ("python", self.python_bin),
                ("predict script", self.predict_script),
            )
            if not Path(path).exists()
        ]
        if problems:
            raise RuntimeError("NanoJevBackend unavailable: " + "; ".join(problems))

    def _run_checkpoint(self, payload: dict) -> dict:
        if self.server_url:
            return self._run_http(payload)
        with tempfile.TemporaryDirectory(prefix="nanojev-req-") as tmp:
            req = Path(tmp) / "request.json"
            req.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            cmd = [self.python_bin, self.predict_script,
                   "--checkpoint-dir", self.checkpoint_dir,
                   "--input", str(req)]
            if self.max_length is not None:
                cmd += ["--max-length", str(self.max_length)]
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True,
                                      timeout=self.timeout_s)
            except subprocess.TimeoutExpired as e:
                raise RuntimeError(
                    f"NanoJev inference timed out after {self.timeout_s}s") from e
            if proc.returncode != 0:
                msg = proc.stderr.strip() or proc.stdout.strip() or "unknown error"
                try:
                    err = json.loads(msg.splitlines()[-1])
                    msg = err.get("message", msg)
                except (ValueError, AttributeError):
                    pass
                raise RuntimeError(f"NanoJev inference failed: {msg}")
            try:
                return json.loads(proc.stdout)
            except ValueError as e:
                raise RuntimeError(
                    f"NanoJev inference returned unparsable stdout: "
                    f"{proc.stdout[:300]!r}") from e

    def _run_http(self, payload: dict) -> dict:
        req = urllib.request.Request(
            self.server_url.rstrip("/") + "/predict",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S) as r:
                return json.loads(r.read().decode())
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"NanoJev server call failed: {e}") from e

    def predict(self, state: dict, candidates: list[Candidate]) -> Prediction:
        t0 = time.monotonic()
        names = [c.name for c in candidates]
        if not 2 <= len(names) <= 255:
            raise RuntimeError(
                f"NanoJev Choice needs 2-255 candidates, got {len(names)}")
        criteria = {}
        for c in candidates:
            desc = (c.description or "").strip()
            if not c.name.strip() or not desc:
                raise RuntimeError(
                    "NanoJev Choice candidate names/descriptions must be non-empty")
            criteria[c.name] = desc
        state_str = json.dumps(state, ensure_ascii=False, separators=(",", ":"))
        if not state_str:
            raise RuntimeError("NanoJev state serializes empty")
        payload = {"states": [{
            "id": "s0",
            "state": state_str,
            "questions": {"tactic": {
                "type": "choice",
                "instructions": (
                    "Given the real-time strategy game state, "
                    "pick the best tactic."),
                "criteria": criteria,
            }},
        }]}
        result = self._run_checkpoint(payload)
        latency_ms = (time.monotonic() - t0) * 1000.0
        try:
            ans = result["states"][0]["answers"]["tactic"]
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(
                f"NanoJev result missing states[0].answers.tactic: "
                f"{str(result)[:300]!r}") from e
        choice = ans.get("choice", "")
        probs = dict(ans.get("probabilities") or {})
        confidence = float(probs.get(choice, 0.0)) if choice else 0.0
        name_set = set(names)
        detail: dict = {
            "checkpoint_dir": self.checkpoint_dir,
            "transport": "http" if self.server_url else "subprocess",
            "execution": result.get("execution", {}),
        }
        if choice not in name_set:
            confidence = 0.0
            detail["invalid_choice"] = choice
            choice = sorted(name_set)[0]
        if not probs and choice:
            probs = {choice: max(confidence, 0.01)}
        return Prediction(choice=choice, probs=probs, confidence=confidence,
                          latency_ms=latency_ms, cost_usd=0.0,
                          backend=self.name, detail=detail)
