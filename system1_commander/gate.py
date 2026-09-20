"""Confidence gate: high -> execute, medium -> downgrade, low -> scripted fallback."""

from __future__ import annotations

from dataclasses import dataclass

from system1_commander.backend_base import Prediction, System1Backend
from system1_commander.candidates import Candidate

DOWNGRADE_ORDER = ("guard_choke", "hold_position", "wait")


@dataclass
class GateConfig:
    # Recalibrated 2026-09-21 from P1 evidence (system1-p1-jev-eco):
    # Jev correct decisions (build_powr) came at conf 0.37-0.54 with
    # top1-top2 margin 0.16-0.42, so the old high=0.70 killed every eco
    # decision (all downgraded to wait, nothing ever built).
    # New rule: execute if conf >= high (0.40) OR margin >= margin_min.
    high: float = 0.40
    low: float = 0.25
    destructive_min: float = 0.9
    margin_min: float = 0.15


@dataclass
class GateDecision:
    mode: str  # "execute" | "downgrade" | "fallback"
    choice_name: str
    reason: str

    def to_dict(self) -> dict:
        return {"mode": self.mode, "choice": self.choice_name, "reason": self.reason}


def top1_margin(probs: dict) -> float:
    """Top1-top2 probability gap; 1.0 when there is a single candidate."""
    vals = sorted((float(v) for v in (probs or {}).values()), reverse=True)
    if len(vals) < 2:
        return 1.0
    return vals[0] - vals[1]


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
    if pred.confidence >= cfg.high or top1_margin(pred.probs) >= cfg.margin_min:
        return GateDecision(
            "execute", pred.choice,
            f"conf {pred.confidence:.2f}>= {cfg.high} or "
            f"margin {top1_margin(pred.probs):.2f}>= {cfg.margin_min}")
    if pred.confidence >= cfg.low:
        for name in DOWNGRADE_ORDER:
            if name in by_name:
                return GateDecision("downgrade", name,
                                    f"conf {pred.confidence:.2f} in [{cfg.low},{cfg.high}) "
                                    f"margin {top1_margin(pred.probs):.2f} -> {name}")
    fb = scripted.predict(state, candidates)
    return GateDecision("fallback", fb.choice, f"conf {pred.confidence:.2f} < {cfg.low} -> scripted")
