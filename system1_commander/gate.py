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


def predict_with_ban(
    state: dict,
    candidates: list[Candidate],
    banned: set[str] | None,
    scripted: System1Backend,
    probs: dict | None = None,
) -> Prediction:
    """Fallback with per-game failure memory (NanoJev plan §A1).

    1. Scripted first pick; use it when not banned (banned empty ==
       legacy behaviour, bit-identical for Jev/Laya callers).
    2. Else walk the A2 pre-perception priority head (`a2_preferred`).
    3. Else NanoJev's own `probs` only as tie-break (excl. banned).
    4. Else ballot order excl. banned; all banned -> synthetic `wait`.
    """
    from system1_commander.backend_scripted import a2_preferred

    names = [c.name for c in candidates]
    banned = set(banned or ())
    first = scripted.predict(state, candidates)
    if first.choice not in banned:
        return first
    for n in a2_preferred(state, names):
        if n not in banned:
            fb = scripted.predict(state, candidates)
            return Prediction(
                choice=n, probs={n: 1.0}, confidence=0.5,
                latency_ms=fb.latency_ms, cost_usd=0.0,
                backend=fb.backend,
                detail={"rule": "a2-preferred after ban",
                        "banned": sorted(banned), "scripted_first": first.choice},
            )
    probs = dict(probs or {})
    ranked = sorted(
        (n for n in names if n not in banned),
        key=lambda n: (-float(probs.get(n, 0.0) or 0.0), names.index(n)),
    )
    if ranked:
        n = ranked[0]
        return Prediction(
            choice=n, probs={n: max(float(probs.get(n, 0.0) or 0.0), 0.01)},
            confidence=0.0, latency_ms=first.latency_ms, cost_usd=0.0,
            backend=first.backend,
            detail={"rule": "probs-tiebreak after ban",
                    "banned": sorted(banned), "scripted_first": first.choice},
        )
    return Prediction(
        choice="wait", probs={"wait": 1.0}, confidence=0.0,
        latency_ms=first.latency_ms, cost_usd=0.0, backend=first.backend,
        detail={"rule": "all-banned->wait", "banned": sorted(banned)},
    )


def apply_gate(
    pred: Prediction,
    candidates: list[Candidate],
    scripted: System1Backend,
    state: dict,
    config: GateConfig | None = None,
    banned: set[str] | None = None,
) -> GateDecision:
    cfg = config or GateConfig()
    by_name = {c.name: c for c in candidates}
    if pred.choice not in by_name:
        fb = predict_with_ban(state, candidates, banned, scripted, pred.probs)
        return GateDecision("fallback", fb.choice,
                            f"unknown choice '{pred.choice}' -> scripted {fb.choice}")
    cand = by_name[pred.choice]
    if cand.destructive and pred.confidence < cfg.destructive_min:
        fb = predict_with_ban(state, candidates, banned, scripted, pred.probs)
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
    fb = predict_with_ban(state, candidates, banned, scripted, pred.probs)
    return GateDecision("fallback", fb.choice, f"conf {pred.confidence:.2f} < {cfg.low} -> scripted")
