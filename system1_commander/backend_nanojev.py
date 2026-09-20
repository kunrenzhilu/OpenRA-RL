"""NanoJev local backend (STUB - not implemented in taskC).

TODO(taskD):
  1. Needs a CUDA GPU (non-CUDA interpreters are rejected outright).
  2. Needs a DecisionPredictor checkpoint: config.json + best.safetensors +
     tokenizer (see unified_game_pipeline.py / BalancedQuestionSampler).
  3. Wire batch_states packing, freeze the state.py serialization version and
     record it in continuation_policy_id (checkpoint + controller + seed +
     code hash), long states must ERROR not truncate (no prefix sharing).
  4. Implement predict() returning Prediction like backend_jev.py.

Do NOT pip install torch/transformers into shared venvs for this stub.
"""

from __future__ import annotations

try:
    import torch  # noqa: F401
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from system1_commander.backend_base import Prediction, System1Backend
from system1_commander.candidates import Candidate


class NanoJevBackend(System1Backend):
    name = "nanojev"

    def __init__(self, *args, **kwargs):
        if not _TORCH_AVAILABLE:
            raise RuntimeError("NanoJev stub: torch not importable here (by design).")
        raise NotImplementedError(
            "NanoJev backend not implemented in taskC (see module docstring TODO).")

    def predict(self, state: dict, candidates: list[Candidate]) -> Prediction:
        raise NotImplementedError(
            "NanoJev backend not implemented in taskC (see module docstring TODO).")
