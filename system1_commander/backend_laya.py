"""Laya classifier backend (STUB - control comparison only, not in taskC).

TODO (only if weights + GPU appear):
  1. Needs Laya weights (3 checkpoints ~1.16B) + Router(preload=True) resident
     to avoid the 7-10s cold load; T4 single query ~33ms.
  2. Run ONLY choices with <=12 candidates (Laya crashes above ~20 options);
     record act_probability alongside the choice.
  3. If weights are absent, demo_loop must downgrade to scripted comparison
     and must NOT block the main chain.

Do NOT pip install heavy deps into shared venvs for this stub.
"""

from __future__ import annotations

try:
    import laya  # noqa: F401
    _LAYA_AVAILABLE = True
except ImportError:
    _LAYA_AVAILABLE = False

from system1_commander.backend_base import Prediction, System1Backend
from system1_commander.candidates import Candidate


class LayaBackend(System1Backend):
    name = "laya"

    def __init__(self, *args, **kwargs):
        if not _LAYA_AVAILABLE:
            raise RuntimeError("Laya stub: laya weights/package not present (by design).")
        raise NotImplementedError(
            "Laya backend not implemented in taskC (see module docstring TODO).")

    def predict(self, state: dict, candidates: list[Candidate]) -> Prediction:
        raise NotImplementedError(
            "Laya backend not implemented in taskC (see module docstring TODO).")
