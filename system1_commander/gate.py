"""Confidence gate: high -> execute, medium -> downgrade, low -> scripted fallback."""

from __future__ import annotations

from dataclasses import dataclass

from system1_commander.backend_base import Prediction, System1Backend
from system1_commander.candidates import Candidate

DOWNGRADE_ORDER = ("guard_choke", "hold_position", "wait")


@dataclass
class GateConfig:
    high: float = 0.7
    low: float = 0.4
    destructive_min: float = 0.9


@dataclass
class GateDecision:
    mode: str  # "execute" | "downgrade" | "fallback"
    choice_name: str
    reason: str

    def to_dict(self) -> dict:
        return {"mode": self.mode, "choice": self.choice_name, "reason": self.reason}


def apply_gate(
    pred: Prediction,
    candidates: list[Candidate],
    scripted: System1Backend,
    state: dict,
    config: GateConfig | None = None,
) -> GateDecision:
    cfg = config or GateConfig()
    by_name = {c.name: c for c in candidates}
    if pred.choice not in by_name:
        fb = scripted.predict(state, candidates)
        return GateDecision("fallback", fb.choice,
                            f"unknown choice '{pred.choice}' -> scripted {fb.choice}")
    cand = by_name[pred.choice]
    if cand.destructive and pred.confidence < cfg.destructive_min:
        fb = scripted.predict(state, candidates)
        return GateDecision("fallback", fb.choice,
                            f"destructive '{pred.choice}' conf {pred.confidence:.2f} < {cfg.destructive_min}")
    if pred.confidence >= cfg.high:
        return GateDecision("execute", pred.choice, f"conf {pred.confidence:.2f} >= {cfg.high}")
    if pred.confidence >= cfg.low:
        for name in DOWNGRADE_ORDER:
            if name in by_name:
                return GateDecision("downgrade", name,
                                    f"conf {pred.confidence:.2f} in [{cfg.low},{cfg.high}) -> {name}")
    fb = scripted.predict(state, candidates)
    return GateDecision("fallback", fb.choice, f"conf {pred.confidence:.2f} < {cfg.low} -> scripted")
