"""Jev cloud backend: one fan-out system_one call per decision.

Fan-out per call: 1 Choice (pick candidate) + 2 Noul (risk flags) + 1 Score
(danger 0-4). Input billed at $0.042/Mtok, output free.
429/529 (and connection) errors use exponential backoff, then re-raise.
"""

from __future__ import annotations

import os
import time

from system1_commander.backend_base import Prediction, System1Backend
from system1_commander.candidates import (
    COMBAT_MACRO_NAMES,
    MACRO_NAMES,
    Candidate,
)

JEV_MODEL = "jev-1.13.0"
INPUT_PRICE_PER_MTOK = 0.042
MAX_ATTEMPTS = 4
BACKOFF_BASE_S = 1.0


def _retryable(exc: BaseException) -> bool:
    status = getattr(exc, "status_code", None)
    if status in (429, 529):
        return True
    try:
        from typesafe_sdk import (
            TypeSafeAPIConnectionError,
            TypeSafeAPIError,
            TypeSafeAPITimeoutError,
            TypeSafeInternalServerError,
            TypeSafeRateLimitError,
        )
        return isinstance(exc, (
            TypeSafeRateLimitError,
            TypeSafeInternalServerError,
            TypeSafeAPIConnectionError,
            TypeSafeAPITimeoutError,
            TypeSafeAPIError,  # includes other 5xx; non-retryable 4xx surface after retries
        ))
    except ImportError:
        return False


class JevBackend(System1Backend):
    name = "jev"

    def __init__(self, model: str = JEV_MODEL, api_key: str | None = None,
                 timeout_s: float = 60.0):
        try:
            from typesafe_sdk import TypeSafeClient
        except ImportError as e:
            raise RuntimeError(
                "typesafe_sdk not installed in this interpreter; "
                "run Jev backend with .venv-taskC (it has typesafe-sdk 0.7.0). "
                "Do NOT pip install heavy deps into other venvs."
            ) from e
        key = api_key or os.environ.get("TYPESAFE_API_KEY", "")
        if not key:
            raise RuntimeError("TYPESAFE_API_KEY is not set in the environment.")
        self._client = TypeSafeClient(api_key=key, model=model, timeout=timeout_s)
        self.model = model

    def _questions(self, candidates: list[Candidate]) -> dict:
        from typesafe_sdk import Choice, Noul, Score
        names = {c.name for c in candidates}
        # Jev-max J2: phase-1 macro ballot gets the phase-goal manual
        # (criteria = macro descriptions; the iron backup below is context
        # text — legality is enforced by the harness ballot prefilter, and
        # the finished-building-first rule by the place-first guard).
        # Backend-agnostic: plain Candidates.
        if names and names <= set(MACRO_NAMES):
            tactic_instructions = (
                "Pick the base-wide phase goal for this tick. "
                "Phase goals: open_powr = secure power first (default "
                "opening); rush_barr = early Barracks into rifle infantry "
                "pressure; fast_weap = rush War Factory to unlock "
                "harvesters and tanks; econ_harv = grow ore income with "
                "another harvester; armor_push = mass light tanks for a "
                "decisive push. "
                "fast_weap is the phase goal that unlocks vehicle "
                "production; econ_harv / armor_push grow income and armor "
                "once the economy supports them. "
                "Iron backup (context; the harness places finished "
                "buildings first): a non-empty \"ready_to_place\" or "
                "\"queue_blocked_by_unplaced\" blocks the queue until the "
                "finished building is placed, and open_powr starts by "
                "placing it. "
                "Pick one option from this ballot."
            )
        elif names and names <= set(COMBAT_MACRO_NAMES):
            tactic_instructions = (
                "Pick the combat posture for this tick. "
                "Postures: defend_hold = screen the base, do not "
                "overextend; probe_attack = hit the nearest visible enemy "
                "while keeping the force intact; all_in_commit = drive the "
                "whole force at the enemy centroid. "
                "Context: the posture maps onto the attack/defense verbs "
                "on the next step. "
                "Pick one option from this ballot."
            )
        else:
            tactic_instructions = (
                "Given the real-time strategy game state, pick the best tactic. "
                "Iron rules (context; the harness ballot carries the "
                "admissible options): (1) a finished building in "
                "\"ready_to_place\" (or \"queue_blocked_by_unplaced\") "
                "blocks all further production until placed; "
                "\"place_ready\" (or a directional place_*) is the option "
                "that places it. "
                "(2) With 0 \"harvesters\" there is no income; "
                "\"train_harv\" (when listed in \"can_make\") grows income, "
                "while a bare Construction Yard with empty "
                "available_production calls for \"build_powr\" to open "
                "the queue. "
                "(3) \"can_make\" lists the currently producible "
                "build/train options; pick from the ballot."
            )
        return {
            "tactic": Choice(
                instructions=tactic_instructions,
                criteria={c.name: c.description for c in candidates},
            ),
            "risk_under_attack": Noul(
                instructions="Is our base currently under attack?",
                criteria={"true": "Enemies are attacking our base/buildings.",
                          "false": "No immediate threat to our base."},
            ),
            "risk_overextend": Noul(
                instructions="Would attacking now overextend our weak force?",
                criteria={"true": "Attacking now risks losing weak units for no gain.",
                          "false": "Attacking now is safe or favorable."},
            ),
            # Jev-max J3: fan-out extras in the SAME call (near-zero added
            # latency per the official fan-out pattern). goal routes code;
            # phase/threat are speculative (use-if-relevant per pattern).
            "goal": Choice(
                instructions=(
                    "What should our base focus on in this phase of the game?"
                ),
                criteria={
                    "expand_eco": "Grow the economy: power, harvesters, ore income.",
                    "tech_up": "Climb technology: War Factory and advanced structures.",
                    "mass_army": "Build an attack force: infantry and tanks.",
                    "defend": "Defend the base: hold position, do not overextend.",
                },
            ),
            "threat_recall": Noul(
                instructions="Should our forces fall back to defend the base?",
                criteria={"true": "Yes: recall forces, the base needs defense.",
                          "false": "No: forces can stay out / keep pressure."},
            ),
            "phase": Score(
                instructions="Which phase of the game is this?",
                criteria=["early game opening", "mid game buildup",
                          "late game endgame"],
            ),
            "danger": Score(
                instructions="Rate the overall danger to our forces and base.",
                criteria=["no danger", "minor threat", "moderate threat",
                          "severe threat", "about to be overrun"],
            ),
        }

    def predict(self, state: dict, candidates: list[Candidate]) -> Prediction:
        t0 = time.monotonic()
        questions = self._questions(candidates)
        last_exc: BaseException | None = None
        resp = None
        for attempt in range(MAX_ATTEMPTS):
            try:
                resp = self._client.system_one(state=state, questions=questions)
                break
            except Exception as e:  # noqa: BLE001
                last_exc = e
                if not _retryable(e) or attempt == MAX_ATTEMPTS - 1:
                    raise
                time.sleep(BACKOFF_BASE_S * (2 ** attempt))
        assert resp is not None  # loop either breaks with resp or raises
        latency_ms = (time.monotonic() - t0) * 1000.0

        choices = getattr(resp, "choices", None)
        if not choices:
            answers = getattr(resp, "answers", {}) or {}
            choices = answers.get("choices", {})
        ans = choices.get("tactic") if isinstance(choices, dict) else None
        choice = getattr(ans, "choice", "") if ans is not None else ""
        probs = dict(getattr(ans, "probabilities", {}) or {})
        confidence = float(getattr(ans, "confidence", 0.0) or 0.0)
        usage = getattr(resp, "usage", None)
        in_tok = int(getattr(usage, "input_tokens", 0) or 0)
        out_tok = int(getattr(usage, "output_tokens", 0) or 0)
        cost = in_tok * INPUT_PRICE_PER_MTOK / 1e6

        names = {c.name for c in candidates}
        detail = {
            "model": getattr(resp, "model", self.model),
            "input_tokens": in_tok,
            "output_tokens": out_tok,
        }
        try:
            # Jev-max J3: every Choice answer (tactic + goal) lands in
            # detail["choices"] so demo_loop can route on goal; tactic also
            # stays top-level for the legacy gate path.
            if isinstance(choices, dict):
                detail["choices"] = {
                    k: {
                        "choice": getattr(v, "choice", None),
                        "probs": dict(getattr(v, "probabilities", {}) or {}),
                        "confidence": float(getattr(v, "confidence", 0.0) or 0.0),
                    }
                    for k, v in choices.items()
                }
            nouls = getattr(resp, "nouls", None) or (getattr(resp, "answers", {}) or {}).get("nouls", {})
            scores = getattr(resp, "scores", None) or (getattr(resp, "answers", {}) or {}).get("scores", {})
            if isinstance(nouls, dict):
                detail["nouls"] = {k: getattr(v, "noul", None) for k, v in nouls.items()}
            if isinstance(scores, dict):
                detail["scores"] = {k: getattr(v, "score", None) for k, v in scores.items()}
        except Exception:  # noqa: BLE001
            pass
        if choice not in names:
            confidence = 0.0
            detail["invalid_choice"] = choice
            choice = sorted(names)[0]
        if not probs and choice:
            probs = {choice: max(confidence, 0.01)}
        return Prediction(choice=choice, probs=probs, confidence=confidence,
                          latency_ms=latency_ms, cost_usd=cost,
                          backend=self.name, detail=detail)
