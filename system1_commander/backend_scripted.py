"""Rule fallback backend (P0, zero model cost).

Combat: visible enemy -> attack_nearest, else guard_choke.
Also deploys the MCV first if the state flags an undeployed one.
"""

from __future__ import annotations

import time

from system1_commander.backend_base import Prediction, System1Backend
from system1_commander.candidates import Candidate


def _power_of(state: dict) -> int:
    try:
        return int(state.get("power", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _building_busy(state: dict) -> bool:
    return any(
        isinstance(p, dict) and p.get("queue_type") == "Building"
        for p in (state.get("production") or [])
    )


def a2_preferred(state: dict, names: list[str]) -> list[str]:
    """NanoJev plan A2 pre-perception priority head (rule-derived only).

    Operates on the built state *dict* (same keys the builders read), so
    `gate.predict_with_ban` and `ScriptedBackend` agree on one ranking.
    Returns the rule-derived head in priority order, filtered to `names`;
    may be empty (caller falls back to probs tie-break, then ballot order).
    Eco: ready non-empty -> place_ready; Building queue busy -> wait (no new
    build spam); else build order by precondition: no power -> build_powr,
    power but no barr -> build_barr, else train_harv / wait.
    Combat: no head (any pick is an empty packet when troopless; the ban
    loop walks the ballot and lands on all-banned -> wait).
    """
    name_set = set(names)
    head: list[str] = []
    if (state.get("kind") or "") == "eco" or "production" in state:
        ready = state.get("ready_to_place") or []
        if ready and "place_ready" in name_set:
            head.append("place_ready")
        if _building_busy(state) and "wait" in name_set:
            head.append("wait")
        btypes = {b.get("type") for b in (state.get("buildings") or [])
                  if isinstance(b, dict)}
        if _power_of(state) <= 0 or "powr" not in btypes:
            if "build_powr" in name_set:
                head.append("build_powr")
        elif "barr" not in btypes:
            if "build_barr" in name_set:
                head.append("build_barr")
        else:
            for n in ("train_harv", "wait"):
                if n in name_set:
                    head.append(n)
    return head


def a2_fallback_rank(state: dict, names: list[str]) -> list[str]:
    """Full preference order: A2 head first, then remaining ballot order."""
    head = a2_preferred(state, names)
    return head + [n for n in names if n not in head]


class ScriptedBackend(System1Backend):
    name = "scripted"

    def predict(self, state: dict, candidates: list[Candidate]) -> Prediction:
        t0 = time.monotonic()
        names = {c.name for c in candidates}
        if state.get("mcv_undeployed") is not None and "deploy_mcv" in names:
            choice, conf = "deploy_mcv", 1.0
        elif (state.get("kind") or "") == "eco" or "production" in state:
            # A2 前置感知 (NanoJev plan §A2): alphabetical-determinism guard.
            # `sorted(names)[0]` always landed on build_barr ('b' < 'd' < ...);
            # rank by build preconditions instead. Combat states keep the
            # legacy branch below untouched.
            rank = a2_fallback_rank(state, [c.name for c in candidates])
            choice = rank[0]
            conf = 1.0 if choice in ("place_ready", "wait", "build_powr") else 0.5
        elif state.get("enemies"):
            choice = "attack_nearest" if "attack_nearest" in names else sorted(names)[0]
            conf = 1.0 if choice == "attack_nearest" else 0.5
        else:
            choice = "guard_choke" if "guard_choke" in names else sorted(names)[0]
            conf = 1.0 if choice == "guard_choke" else 0.5
        return Prediction(
            choice=choice,
            probs={choice: 1.0},
            confidence=conf,
            latency_ms=(time.monotonic() - t0) * 1000.0,
            cost_usd=0.0,
            backend=self.name,
            detail={"rule": "eco:A2-preconditions place/wait/powr/barr/harv; combat:enemy_visible->attack_nearest else guard_choke"},
        )
