"""N0 unit tests (NanoJev-max): downgrade path respects `banned`.

No server, no model. Covers: skip-banned-head, all-banned ->
predict_with_ban, legacy bit-identity with banned=None/empty, and the
no-table-candidate fall-through.
"""

from system1_commander.backend_base import Prediction
from system1_commander.backend_scripted import ScriptedBackend
from system1_commander.candidates import (
    list_combat_candidates,
    list_eco_candidates,
)
from system1_commander.gate import GateConfig, apply_gate


def _flat_pred(choice: str, names: list[str], conf: float = 0.30) -> Prediction:
    # Flat probs -> top1-top2 margin ~0, so conf alone picks the band.
    p = 1.0 / len(names)
    return Prediction(choice=choice, probs={n: p for n in names},
                      confidence=conf, latency_ms=1.0, cost_usd=0.0,
                      backend="nanojev")


def _combat_state() -> dict:
    return {"kind": "combat",
            "enemies": [{"id": 9, "type": "e1"}]}


def test_downgrade_skips_banned_head():
    cands = list_combat_candidates()
    pred = _flat_pred("attack_nearest", [c.name for c in cands])
    g = apply_gate(pred, cands, ScriptedBackend(), _combat_state(),
                   GateConfig(), banned={"guard_choke"})
    assert (g.mode, g.choice_name) == ("downgrade", "hold_position")


def test_downgrade_all_banned_falls_to_predict_with_ban():
    cands = list_combat_candidates()
    pred = _flat_pred("attack_nearest", [c.name for c in cands])
    g = apply_gate(pred, cands, ScriptedBackend(), _combat_state(),
                   GateConfig(),
                   banned={"guard_choke", "hold_position", "wait"})
    # Table fully banned -> predict_with_ban; scripted first pick
    # (attack_nearest, unbanned) wins.
    assert (g.mode, g.choice_name) == ("fallback", "attack_nearest")
    assert "all banned" in g.reason


def test_downgrade_no_ban_legacy_combat_and_eco():
    scripted = ScriptedBackend()
    cands = list_combat_candidates()
    pred = _flat_pred("attack_nearest", [c.name for c in cands])
    g = apply_gate(pred, cands, scripted, _combat_state(),
                   GateConfig(), banned=None)
    assert (g.mode, g.choice_name) == ("downgrade", "guard_choke")
    eco = list_eco_candidates()
    epred = _flat_pred("build_powr", [c.name for c in eco])
    g2 = apply_gate(epred, eco, scripted,
                    {"kind": "eco", "production": []},
                    GateConfig(), banned=None)
    assert (g2.mode, g2.choice_name) == ("downgrade", "wait")


def test_downgrade_empty_ban_set_is_legacy():
    cands = list_combat_candidates()
    pred = _flat_pred("attack_nearest", [c.name for c in cands])
    g = apply_gate(pred, cands, ScriptedBackend(), _combat_state(),
                   GateConfig(), banned=set())
    assert (g.mode, g.choice_name) == ("downgrade", "guard_choke")


def test_downgrade_ignores_irrelevant_ban():
    cands = list_combat_candidates()
    pred = _flat_pred("attack_nearest", [c.name for c in cands])
    g = apply_gate(pred, cands, ScriptedBackend(), _combat_state(),
                   GateConfig(), banned={"attack_nearest"})
    assert (g.mode, g.choice_name) == ("downgrade", "guard_choke")


def test_downgrade_no_table_candidate_falls_through():
    cands = [c for c in list_combat_candidates()
             if c.name in ("attack_nearest", "focus_weakest")]
    pred = _flat_pred("attack_nearest", [c.name for c in cands])
    g = apply_gate(pred, cands, ScriptedBackend(), _combat_state(),
                   GateConfig(), banned={"guard_choke"})
    # Legacy fall-through (unchanged by N0): scripted fallback.
    assert (g.mode, g.choice_name) == ("fallback", "attack_nearest")
