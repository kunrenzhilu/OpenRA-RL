"""N-pre discrimination probe (NanoJev-max, FROZEN for judge recount).

Replays every true-vote state (orders.jsonl rows with a nanojev
prediction and no fix_bypass, aligned to states.jsonl by tick) through
the resident NanoJev service and records FULL probs to replay.jsonl.

Frozen verdicts (thresholds frozen, no grey zone):
  (a) entropy of the replay top1 distribution < 0.5 bits -> CONSTANT
      (deny) -> exit; >= 0.5 bits -> VARIES (affirm) -> continue.
  (b) split replays by replay-conf median; high-half audit-effective
      rate minus low-half rate > 5pp -> CORRELATED (affirm) ->
      continue; <= 5pp -> ZERO-CORRELATION (deny) -> exit.
      Effectiveness = audit verdict == "effective" for that tick's
      orders row (N5' label calibre, no re-judgement).
Either deny -> early-exit (OR, no third state).

NOTE (sample deviation, supervisor-approved): the plan froze n=179
(A1 states), but A1 orders.jsonl stores no state text, so the replay
set is the N-pre collection game instead (n = its true-vote rows).
Thresholds unchanged.

Usage:
    NANOJEV_URL=http://127.0.0.1:8932 PYTHONPATH=<worktree> <venv>/bin/python \\
        scripts/n_discrimination.py --log-dir .runs/nanojmax-npre-collect-XXX
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from collections import Counter
from pathlib import Path

WORKTREE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKTREE))

ENTROPY_THRESHOLD_BITS = 0.5
GAP_THRESHOLD_PP = 5.0


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="N-pre discrimination probe.")
    p.add_argument("--log-dir", required=True)
    return p.parse_args(argv)


def _entropy(counter: Counter, n: int) -> float:
    h = 0.0
    for c in counter.values():
        p = c / n
        h -= p * math.log2(p)
    return h


def main(argv=None) -> int:
    args = _parse_args(argv)
    from system1_commander.audit import audit_file
    from system1_commander.backend_nanojev import NanoJevBackend
    from system1_commander.candidates import (
        list_combat_candidates,
        list_eco_candidates,
    )

    logdir = args.log_dir
    states, orders = {}, []
    with open(os.path.join(logdir, "states.jsonl")) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                states[r["tick"]] = r
    with open(os.path.join(logdir, "orders.jsonl")) as f:
        for line in f:
            line = line.strip()
            if line:
                orders.append(json.loads(line))
    decisions = [o for o in orders if o.get("kind", "decision") == "decision"]
    audit = audit_file(os.path.join(logdir, "orders.jsonl"))
    verdict_by_tick = {r["tick"]: r["verdict"] for r in audit["rows"]}

    # True-vote rows only: nanojev prediction, no fix bypass, state present.
    targets = [o for o in decisions
               if o.get("prediction", {}).get("backend") == "nanojev"
               and not o.get("fix_bypass")
               and o.get("tick") in states]
    print(f"[n_disc] decisions={len(decisions)} true-vote targets={len(targets)}")
    if not targets:
        print("[n_disc] NO TARGETS -> cannot judge")
        return 3

    pools = {"eco": list_eco_candidates(), "combat": list_combat_candidates()}
    backend = NanoJevBackend()
    replay_path = os.path.join(logdir, "replay.jsonl")
    rows = []
    with open(replay_path, "w") as f:
        for o in targets:
            st = states[o["tick"]]
            pred = backend.predict(st["state"], pools[st["kind"]])
            rec = {"tick": o["tick"], "i": o.get("i"), "kind": st["kind"],
                   "choice": pred.choice, "probs": pred.probs,
                   "confidence": pred.confidence,
                   "latency_ms": pred.latency_ms,
                   "orig_choice": o["prediction"]["choice"],
                   "orig_conf": o["prediction"]["confidence"],
                   "gate": o["gate"],
                   "audit": verdict_by_tick.get(o["tick"], "?")}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            rows.append(rec)
            print(f"[n_disc] tick={o['tick']} {st['kind']:>6} "
                  f"replay={pred.choice}@{pred.confidence:.3f} "
                  f"orig={o['prediction']['choice']}@{o['prediction']['confidence']:.3f} "
                  f"{pred.latency_ms:.0f}ms audit={rec['audit']}", flush=True)

    n = len(rows)
    top1 = Counter(r["choice"] for r in rows)
    h = _entropy(top1, n)
    verdict_a = "AFFIRM(varies)" if h >= ENTROPY_THRESHOLD_BITS else "DENY(constant)"
    print(f"\n(a) top1 dist: {dict(top1)} entropy={h:.3f} bits "
          f"(thr {ENTROPY_THRESHOLD_BITS}) -> {verdict_a}")

    med = statistics.median(r["confidence"] for r in rows)
    hi = [r for r in rows if r["confidence"] >= med]
    lo = [r for r in rows if r["confidence"] < med]
    # Degenerate median (all-equal conf): fall back to index split so both
    # halves are non-empty; the gap is then ~0 by construction (honest deny).
    if not hi or not lo:
        half = n // 2
        lo, hi = rows[:half], rows[half:]
        print(f"[n_disc] degenerate median {med:.3f}: index-split "
              f"lo={len(lo)} hi={len(hi)}")
    eff = lambda rs: sum(1 for r in rs if r["audit"] == "effective") / len(rs) * 100
    ehi, elo = eff(hi), eff(lo)
    gap = ehi - elo
    verdict_b = ("AFFIRM(correlated)" if gap > GAP_THRESHOLD_PP
                 else "DENY(zero-correlation)")
    print(f"(b) conf median={med:.3f} hi_n={len(hi)} eff={ehi:.1f}% | "
          f"lo_n={len(lo)} eff={elo:.1f}% gap={gap:+.1f}pp "
          f"(thr >{GAP_THRESHOLD_PP}pp) -> {verdict_b}")

    cont = verdict_a.startswith("AFFIRM") and verdict_b.startswith("AFFIRM")
    print(f"\nN-pre discrimination: {'CONTINUE' if cont else 'EARLY-EXIT'} "
          f"(n={n}, replay -> {replay_path})")
    return 0 if cont else 4


if __name__ == "__main__":
    raise SystemExit(main())
