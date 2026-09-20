"""Rule fallback backend (P0, zero model cost).

Combat: visible enemy -> attack_nearest, else guard_choke.
Also deploys the MCV first if the state flags an undeployed one.
"""

from __future__ import annotations

import time

from system1_commander.backend_base import Prediction, System1Backend
from system1_commander.candidates import Candidate


class ScriptedBackend(System1Backend):
    name = "scripted"

    def predict(self, state: dict, candidates: list[Candidate]) -> Prediction:
        t0 = time.monotonic()
        names = {c.name for c in candidates}
        if state.get("mcv_undeployed") is not None and "deploy_mcv" in names:
            choice, conf = "deploy_mcv", 1.0
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
            detail={"rule": "enemy_visible->attack_nearest else guard_choke"},
        )
