"""Laya local classifier backend (real, 2026-09-21).

One Choice question per decision ("tactic": pick a candidate) against the
local english checkpoint (convaiinnovations/laya, ModernBERT-large 421M).

Two transports (chosen at construction):
  - resident server (preferred): LAYA_URL=http://127.0.0.1:8931 served by
    scripts/serve_laya.py (weights load once, GPU resident, ~35ms/decision).
  - subprocess fallback: one /tmp/nanojev-venv python call per decision via
    scripts/laya_predict.py (weights reload each time, seconds per call).

Driver venv never imports torch: both paths stay in the nanojev venv
(CUDA) or plain HTTP. ~/Github/laya is used read-only via PYTHONPATH.
Weights are solidified at `.data/laya-weights-1c5edc1` (HF rev 1c5edc1);
the legacy /tmp/laya-hf cache no longer exists (WSL reboot wipes /tmp).
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

DEFAULT_PYTHON_BIN = "/tmp/nanojev-venv/bin/python"
DEFAULT_PREDICT_SCRIPT = str(
    Path(__file__).resolve().parent.parent / "scripts" / "laya_predict.py")
DEFAULT_LAYA_REPO = str(Path.home() / "Github" / "laya")
SUBPROCESS_TIMEOUT_S = 600.0
HTTP_TIMEOUT_S = 120.0


def _default_model_dir() -> str | None:
    """Solidified Laya weights for the subprocess path (server path owns
    its weights via serve_laya argv). Worktree `.data/` first, then the
    main checkout's `.data/`."""
    cands = [
        Path(__file__).resolve().parent.parent / ".data" / "laya-weights-1c5edc1",
        Path.home() / "Github" / "openra-commander" / ".data" / "laya-weights-1c5edc1",
    ]
    for p in cands:
        if (p / "model.safetensors").exists() or (p / "rl_agent_config.json").exists():
            return str(p)
    return None


def _resolve(name: str, explicit: str | None, default: str) -> str:
    return explicit or os.environ.get(name, default)


class LayaBackend(System1Backend):
    name = "laya"

    def __init__(self, server_url: str | None = None,
                 python_bin: str | None = None,
                 predict_script: str | None = None,
                 timeout_s: float = SUBPROCESS_TIMEOUT_S):
        self.server_url = server_url or os.environ.get("LAYA_URL", "")
        self.python_bin = _resolve("LAYA_PYTHON", python_bin, DEFAULT_PYTHON_BIN)
        self.predict_script = _resolve("LAYA_PREDICT_SCRIPT", predict_script,
                                       DEFAULT_PREDICT_SCRIPT)
        self.timeout_s = timeout_s
        if not self.server_url:
            problems = [
                f"{label} missing: {path}"
                for label, path in (
                    ("python", self.python_bin),
                    ("predict script", self.predict_script),
                    ("laya repo", DEFAULT_LAYA_REPO),
                )
                if not Path(path).exists()
            ]
            if problems:
                raise RuntimeError("LayaBackend unavailable: " + "; ".join(problems))

    def _predict_http(self, state: dict, criteria: dict) -> dict:
        req = urllib.request.Request(
            self.server_url.rstrip("/") + "/predict",
            data=json.dumps({"state": state, "criteria": criteria}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S) as r:
            return json.loads(r.read().decode())

    def _predict_subprocess(self, state: dict, criteria: dict) -> dict:
        with tempfile.TemporaryDirectory(prefix="laya-req-") as tmp:
            req_path = Path(tmp) / "request.json"
            req_path.write_text(json.dumps({"state": state, "criteria": criteria},
                                           ensure_ascii=False), encoding="utf-8")
            env = dict(os.environ)
            env["PYTHONPATH"] = DEFAULT_LAYA_REPO + (
                ":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
            cmd = [self.python_bin, self.predict_script, "--input", str(req_path)]
            model_dir = env.get("LAYA_MODEL_DIR") or _default_model_dir()
            if model_dir and Path(model_dir).exists():
                cmd += ["--model-dir", model_dir]
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True,
                                      timeout=self.timeout_s, env=env)
            except subprocess.TimeoutExpired as e:
                raise RuntimeError(
                    f"Laya inference timed out after {self.timeout_s}s") from e
            if proc.returncode != 0:
                msg = (proc.stderr.strip() or proc.stdout.strip() or "unknown error")
                raise RuntimeError(f"Laya inference failed: {msg[-500:]}")
            try:
                return json.loads(proc.stdout)
            except ValueError as e:
                raise RuntimeError(
                    "Laya inference returned unparsable stdout: "
                    f"{proc.stdout[:300]!r}") from e

    def predict(self, state: dict, candidates: list[Candidate]) -> Prediction:
        t0 = time.monotonic()
        names = [c.name for c in candidates]
        if not 2 <= len(names) <= 255:
            raise RuntimeError(
                f"Laya Choice needs 2-255 candidates, got {len(names)}")
        criteria = {}
        for c in candidates:
            desc = (c.description or "").strip()
            if not c.name.strip() or not desc:
                raise RuntimeError(
                    "Laya Choice candidate names/descriptions must be non-empty")
            criteria[c.name] = desc
        transport = "http" if self.server_url else "subprocess"
        if self.server_url:
            result = self._predict_http(state, criteria)
        else:
            result = self._predict_subprocess(state, criteria)
        latency_ms = (time.monotonic() - t0) * 1000.0
        choice = result.get("choice", "")
        probs = {k: float(v) for k, v in (result.get("probabilities") or {}).items()}
        confidence = float(result.get("confidence", 0.0) or 0.0)
        name_set = set(names)
        detail: dict = {
            "transport": transport,
            "model": result.get("model", "laya-rl-agent"),
            "input_tokens": result.get("input_tokens", 0),
        }
        for k in ("load_ms", "infer_ms"):
            if k in result:
                detail[k] = result[k]
        if choice not in name_set:
            confidence = 0.0
            detail["invalid_choice"] = choice
            choice = sorted(name_set)[0]
        if not probs and choice:
            probs = {choice: max(confidence, 0.01)}
        return Prediction(choice=choice, probs=probs, confidence=confidence,
                          latency_ms=latency_ms, cost_usd=0.0,
                          backend=self.name, detail=detail)
