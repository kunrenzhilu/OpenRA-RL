"""System1 commander: fast discrete-decision adaptor (brain-side router).

Slow LLM plans across minutes; System1 picks one legal candidate every N
ticks; the scripted bot / engine executes. See
workspace/openra/plan/openra-taskC-system1-commander-plan-20260920.md.
"""

from system1_commander.backend_base import Prediction, System1Backend
from system1_commander.candidates import Candidate, list_combat_candidates, list_eco_candidates
from system1_commander.gate import GateConfig, GateDecision, apply_gate
from system1_commander.state import (
    build_combat_state,
    build_eco_state,
    build_snapshot,
    estimate_tokens,
    state_to_json,
)

__all__ = [
    "Prediction",
    "System1Backend",
    "Candidate",
    "list_combat_candidates",
    "list_eco_candidates",
    "GateConfig",
    "GateDecision",
    "apply_gate",
    "build_combat_state",
    "build_eco_state",
    "build_snapshot",
    "estimate_tokens",
    "state_to_json",
]
