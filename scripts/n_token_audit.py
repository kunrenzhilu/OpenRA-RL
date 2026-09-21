"""N-pre token audit (NanoJev-max): three-book reconciliation with the
checkpoint tokenizer.

Reproduces `prepare_examples` (predict_toy_decisions.py:85-126) encoding
*exactly* (segment-wise `tokenizer.encode(..., add_special_tokens=False)`
+ eos) but WITHOUT loading weights (tokenizer only, CPU).

Three books per state (single `tactic` Choice, same instructions/criteria
as backend_nanojev.py:157-168):
  1. state_toks_qwen  = tokens of the "State:\\n<state_str>\\n" segment
  2. path_max         = max candidate leaf tokens (prefix + Candidate +
                        Decision + eos) -- THIS is what max_length gates
  3. req_total        = sum of all candidate leaf tokens (one forward pass)

Also prints py_toks (estimate_tokens, the demo_loop `toks=` ruler) to show
the two rulers side by side.

Output: per-state table + summary + T1 recommendation. Judge-rerunnable.

Usage (needs transformers; use the nanojev venv, CPU only):
    /tmp/nanojev-venv/bin/python scripts/n_token_audit.py \\
        --states .runs/nanojmax-npre-collect-XXX/states.jsonl \\
        --checkpoint-dir /home/tomkun/Github/openra-commander/.data/nanojev-unified-047b927
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

WORKTREE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKTREE))

INSTRUCTIONS = ("Given the real-time strategy game state, "
                "pick the best tactic.")


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="N-pre token audit.")
    p.add_argument("--states", required=True, help="states.jsonl from n_collect")
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--out", default="", help="optional JSON report path")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    from transformers import AutoTokenizer

    from system1_commander.candidates import (
        list_combat_candidates,
        list_eco_candidates,
    )
    from system1_commander.state import estimate_tokens

    tok = AutoTokenizer.from_pretrained(
        str(Path(args.checkpoint_dir) / "tokenizer"),
        local_files_only=True, trust_remote_code=False)
    assert isinstance(tok.eos_token_id, int) and tok.eos_token_id >= 0
    pools = {"eco": list_eco_candidates(), "combat": list_combat_candidates()}

    rows = []
    with open(args.states) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    print(f"{'seq':>4} {'kind':>6} {'tick':>6} {'py_toks':>7} "
          f"{'state_qw':>8} {'ncand':>5} {'path_min':>8} {'path_max':>8} "
          f"{'req_total':>9}")
    worst = None
    for rec in rows:
        kind = rec["kind"]
        state = rec["state"]
        cands = pools[kind]
        criteria = {c.name: c.description for c in cands}
        state_str = json.dumps(state, ensure_ascii=False, separators=(",", ":"))
        py_toks = estimate_tokens(state_str)
        seg0 = f"State:\n{state_str}\n"
        seg1 = ("Question type: choice\nQuestion:\n" f"{INSTRUCTIONS}\n")
        state_qw = len(tok.encode(seg0, add_special_tokens=False))
        prefix = (tok.encode(seg0, add_special_tokens=False)
                  + tok.encode(seg1, add_special_tokens=False))
        leaves = []
        for key in criteria:
            t = f"{key}: {criteria[key]}"
            leaf = (prefix
                    + tok.encode(f"Candidate:\n{t}\nDecision:",
                                 add_special_tokens=False)
                    + [tok.eos_token_id])
            leaves.append(len(leaf))
        path_min, path_max = min(leaves), max(leaves)
        req_total = sum(leaves)
        print(f"{rec['seq']:>4} {kind:>6} {rec['tick']:>6} {py_toks:>7} "
              f"{state_qw:>8} {len(cands):>5} {path_min:>8} {path_max:>8} "
              f"{req_total:>9}")
        if worst is None or path_max > worst[0]:
            worst = (path_max, rec["seq"], kind, rec["tick"])

    pmax, wseq, wkind, wtick = worst
    print(f"\nworst path: {pmax} toks (seq={wseq} {wkind} tick={wtick}); "
          f"max_length=512 headroom = {512 - pmax}")
    print("max_length gates the SINGLE path (max leaf), not req_total.")
    t1 = {"worst_path_max": pmax, "worst_seq": wseq, "n_states": len(rows),
          "headroom_512": 512 - pmax,
          "rule": ("T1: rich-state path_max MUST stay <= 512 (default tier). "
                   "Only escalate --max-length tier-by-tier if flips/entropy "
                   "rise but look truncated.")}
    print("T1:", json.dumps(t1, ensure_ascii=False))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(t1, f, indent=2, ensure_ascii=False)
    return 0 if pmax <= 512 else 2


if __name__ == "__main__":
    raise SystemExit(main())
