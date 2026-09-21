"""Jev-max J1 ablation recount (judge tool, zero cost, stdlib only).

Recomputes flip-rate + mean KL(rich||poor) from an orders.jsonl WITHOUT
trusting demo_loop's math: top-1 and KL are reimplemented here (same
definitions as plan §1). Compares against bench.json's ablation block.

Usage:
    PYTHONPATH=. python scripts/j_ablation.py .runs/jevmax-j1/orders.jsonl [--bench .runs/jevmax-j1/bench.json]
"""

from __future__ import annotations

import argparse
import json
import math
import sys


def top1_of(probs: dict | None) -> str | None:
    if not probs:
        return None
    return max(sorted(probs), key=lambda k: float(probs[k] or 0.0))


def kl_div_bits(p: dict | None, q: dict | None, eps: float = 1e-9) -> float:
    p, q = dict(p or {}), dict(q or {})
    keys = set(p) | set(q)
    if not keys:
        return 0.0
    ps = sum(float(p.get(k, 0.0) or 0.0) for k in keys) or 1.0
    qs = sum(float(q.get(k, 0.0) or 0.0) for k in keys) or 1.0
    n = len(keys)
    kl = 0.0
    for k in keys:
        pk = ((float(p.get(k, 0.0) or 0.0) / ps) + eps) / (1.0 + eps * n)
        qk = ((float(q.get(k, 0.0) or 0.0) / qs) + eps) / (1.0 + eps * n)
        kl += pk * math.log2(pk / qk)
    return kl


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Recount J1 ablation from orders.jsonl.")
    ap.add_argument("orders", help="path to orders.jsonl")
    ap.add_argument("--bench", default=None, help="path to bench.json (optional check)")
    args = ap.parse_args(argv)

    rows = []
    with open(args.orders) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    samples = []
    for r in rows:
        ab = r.get("ablation")
        if not ab:
            continue
        # Voted ticks: poor vote = entry.prediction. Bypass-row ticks: the
        # vote was discarded, poor vote lives inside the ablation record.
        poor_probs = ab.get("poor_probs")
        if poor_probs is None:
            poor_probs = (r.get("prediction") or {}).get("probs") or {}
        rich_probs = ab.get("rich_probs") or {}
        t_poor, t_rich = top1_of(poor_probs), top1_of(rich_probs)
        samples.append({
            "i": r.get("i"),
            "tick": r.get("tick"),
            "kind": r.get("state_kind"),
            "poor_top1": t_poor,
            "rich_top1": t_rich,
            "flipped": bool(t_poor != t_rich),
            "kl_bits": round(kl_div_bits(rich_probs, poor_probs), 4),
            "discarded_by": ab.get("discarded_by"),
            "logged_flip": ab.get("flipped"),
            "logged_kl": ab.get("kl_bits"),
        })

    n = len(samples)
    flips = sum(1 for s in samples if s["flipped"])
    mean_kl = sum(s["kl_bits"] for s in samples) / n if n else 0.0
    voted = [s for s in samples if not s["discarded_by"]]
    vflips = sum(1 for s in voted if s["flipped"])
    vkl = sum(s["kl_bits"] for s in voted) / len(voted) if voted else 0.0
    out = {"n": n, "n_voted": len(voted), "n_discarded": n - len(voted),
           "flips": flips,
           "flip_rate": round(flips / n, 4) if n else 0.0,
           "mean_kl_bits": round(mean_kl, 4),
           "voted_flip_rate": round(vflips / len(voted), 4) if voted else 0.0,
           "voted_mean_kl_bits": round(vkl, 4)}
    print(f"{'i':>4} {'tick':>6} {'kind':>7} {'poor':>14} {'rich':>14} "
          f"{'flip':>5} {'kl':>8} {'discarded_by':>12}")
    for s in samples:
        print(f"{s['i']:>4} {s['tick']:>6} {str(s['kind']):>7} "
              f"{str(s['poor_top1']):>14} {str(s['rich_top1']):>14} "
              f"{str(s['flipped']):>5} {s['kl_bits']:>8.4f} "
              f"{str(s['discarded_by']):>12}")
    print(json.dumps(out, indent=2))

    mism = [s for s in samples
            if s["flipped"] != s["logged_flip"]
            or abs((s["kl_bits"] or 0) - (s["logged_kl"] or 0)) > 1e-3]
    if mism:
        print(f"WARN: {len(mism)} samples disagree with logged values: "
              f"{[s['i'] for s in mism]}")
    if args.bench:
        with open(args.bench) as f:
            bench_ab = json.load(f).get("ablation", {})
        keys = ["n", "n_voted", "n_discarded", "flips", "flip_rate",
                "mean_kl_bits", "voted_flip_rate", "voted_mean_kl_bits"]
        ok = all(abs((bench_ab.get(k) or 0) - (out.get(k) or 0)) < 1e-3
                 for k in keys)
        print("bench.json ablation block:", json.dumps(bench_ab),
              "-> MATCH" if ok else "-> MISMATCH")
        return 0 if ok and not mism else 2
    return 0 if not mism else 2


if __name__ == "__main__":
    raise SystemExit(main())
