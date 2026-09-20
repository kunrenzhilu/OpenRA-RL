"""Single-shot Laya inference helper (runs under /tmp/nanojev-venv).

Reads a request JSON file {state: dict, criteria: {name: desc}},
loads the local Laya english checkpoint once, runs one Choice question
("tactic"), and prints the result JSON to stdout.

Needs: torch + transformers + safetensors + huggingface_hub in this
interpreter, laya package importable (PYTHONPATH=~/Github/laya, read-only),
weights pre-downloaded to a local dir (LAYA_MODEL_DIR or --model-dir).

Stdout is ONLY the result JSON (no progress prints) so the parent can
json.loads the whole stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

DEFAULT_MODEL_DIR = os.environ.get("LAYA_MODEL_DIR", "/tmp/laya-weights")


def _resolve_model_dir(explicit: str | None) -> str:
    if explicit:
        return explicit
    env = os.environ.get("LAYA_MODEL_DIR")
    if env:
        return env
    # Fall back to the HF snapshot cache layout (snapshot_download target).
    cache_root = os.environ.get("LAYA_HF_CACHE", "/tmp/laya-hf")
    models_root = os.path.join(cache_root, "models--convaiinnovations--laya", "snapshots")
    if os.path.isdir(models_root):
        snaps = sorted(os.listdir(models_root))
        if snaps:
            return os.path.join(models_root, snaps[0])
    return DEFAULT_MODEL_DIR


def main() -> int:
    p = argparse.ArgumentParser(description="Single Laya tactic decision.")
    p.add_argument("--input", required=True, help="Request JSON path.")
    p.add_argument("--model-dir", default=None)
    p.add_argument("--device", default=None, help="e.g. cuda, cpu. Default: auto.")
    a = p.parse_args()

    import laya  # noqa: E402

    model_dir = _resolve_model_dir(a.model_dir)
    with open(a.input, encoding="utf-8") as f:
        req = json.load(f)
    state = req["state"]
    criteria = req["criteria"]
    if not 2 <= len(criteria) <= 255:
        raise SystemExit(f"need 2-255 candidates, got {len(criteria)}")

    t0 = time.monotonic()
    agent = laya.load(model_dir, device=a.device)
    load_ms = (time.monotonic() - t0) * 1000.0
    t1 = time.monotonic()
    resp = agent.system_one(state, {"tactic": {
        "type": "choice",
        "instructions": "Given the real-time strategy game state, pick the best tactic.",
        "criteria": criteria,
    }})
    infer_ms = (time.monotonic() - t1) * 1000.0
    ans = resp["answers"]["tactic"]
    out = {
        "choice": ans["choice"],
        "probabilities": ans.get("probabilities", {}),
        "confidence": ans.get("confidence", 0.0),
        "input_tokens": (resp.get("usage") or {}).get("input_tokens", 0),
        "model": resp.get("model", "laya-rl-agent"),
        "load_ms": round(load_ms, 1),
        "infer_ms": round(infer_ms, 1),
    }
    sys.stdout.write(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
