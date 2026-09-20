"""Backend contract: predict(state, candidates) -> Prediction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from system1_commander.candidates import Candidate


@dataclass
class Prediction:
    choice: str
    probs: dict = field(default_factory=dict)
    confidence: float = 0.0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    backend: str = ""
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "choice": self.choice,
            "probs": self.probs,
            "confidence": self.confidence,
            "latency_ms": self.latency_ms,
            "cost_usd": self.cost_usd,
            "backend": self.backend,
            "detail": self.detail,
        }


class System1Backend(ABC):
    name: str = "base"

    @abstractmethod
    def predict(self, state: dict, candidates: list[Candidate]) -> Prediction:
        """Pick one candidate name for the given JSON-able state."""
