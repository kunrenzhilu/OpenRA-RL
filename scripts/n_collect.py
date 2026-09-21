"""N-pre state collector (NanoJev-max): short nanojev game + state dump.

Wraps demo_loop.run with monkey-patched state builders that tee every
built state dict to <log-dir>/states.jsonl (one row per decision, aligned
to orders.jsonl by tick). No changes to demo_loop/state/candidates.

Usage:
    NANOJEV_URL=http://127.0.0.1:8932 PYTHONPATH=<worktree> <venv>/bin/python \\
        scripts/n_collect.py --log-dir .runs/nanojmax-npre-collect-XXX \\
        --max-decisions 50 --url http://localhost:8001 --mix eco,combat
"""

from __future__ import annotations

import argparse
import json
import sys

import system1_commander as s1


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="N-pre state collector.")
    p.add_argument("--log-dir", required=True)
    p.add_argument("--max-decisions", type=int, default=50)
    p.add_argument("--url", default="http://localhost:8001")
    p.add_argument("--mix", default="eco,combat")
    p.add_argument("--ticks-per-decision", type=int, default=50)
    p.add_argument("--map", default="pitfight.oramap")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    import os

    os.makedirs(args.log_dir, exist_ok=True)
    states_path = os.path.join(args.log_dir, "states.jsonl")
    f_states = open(states_path, "w")
    seq = [0]

    _orig_eco = s1.build_eco_state
    _orig_combat = s1.build_combat_state

    def _tee(kind, state):
        rec = {"seq": seq[0], "kind": kind, "tick": state.get("tick"),
               "state": state}
        f_states.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f_states.flush()
        seq[0] += 1
        return state

    def eco(snap):
        return _tee("eco", _orig_eco(snap))

    def combat(snap, *a, **k):
        return _tee("combat", _orig_combat(snap, *a, **k))

    s1.build_eco_state = eco
    s1.build_combat_state = combat

    from system1_commander.demo_loop import main as demo_main

    argv = ["--backend", "nanojev",
           "--mix", args.mix,
           "--max-decisions", str(args.max_decisions),
           "--ticks-per-decision", str(args.ticks_per_decision),
           "--log-dir", args.log_dir,
           "--url", args.url,
           "--map", args.map]
    print(f"[n_collect] states -> {states_path}", flush=True)
    try:
        return demo_main(argv)
    finally:
        f_states.close()
        print(f"[n_collect] dumped {seq[0]} states", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
