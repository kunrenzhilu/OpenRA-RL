"""A-track unit tests (no server, no model): Laya prefilter/skip schema +
NanoJev ban-memory dedup + executor no-retry.

Covers Laya plan §A1/A3 (ballot, exec_flags, skip schema) and NanoJev
plan §A1/A3 (failed_set dedup, all-banned->wait, single batch on FAILED).
"""

import asyncio

from system1_commander.backend_base import Prediction
from system1_commander.backend_scripted import ScriptedBackend, a2_preferred
from system1_commander.candidates import (
    exec_flags,
    list_combat_candidates,
    list_eco_candidates,
    list_executable_candidates,
    make_skip_entry,
)
from system1_commander.executor import Executor
from system1_commander.gate import GateConfig, apply_gate, predict_with_ban
from system1_commander.state import Snapshot, build_eco_state


def _eco_snap(**kw) -> Snapshot:
    base = dict(
        own_units=[],
        own_buildings=[],
        enemies=[],
        production=[],
        available_production=[],
        ready_to_place=[],
        cash=5000,
        power_balance=0,
    )
    base.update(kw)
    return Snapshot(**base)


def test_prefilter_eco_place_ready_needs_queue():
    cands = list_eco_candidates()
    names_empty = [c.name for c in list_executable_candidates(cands, _eco_snap())]
    assert "place_ready" not in names_empty
    assert "wait" in names_empty  # wait never filtered
    snap_ready = _eco_snap(
        ready_to_place=["powr"],
        production=[{"queue_type": "Building", "item": "powr", "progress": 1.0}],
    )
    names_ready = [c.name for c in list_executable_candidates(cands, snap_ready)]
    assert "place_ready" in names_ready


def test_prefilter_combat_troopless_ballot_empty_and_skip_schema():
    cands = list_combat_candidates()
    snap = _eco_snap(enemies=[{"id": 9, "type": "e1"}])  # visible foe, zero own combat
    assert list_executable_candidates(cands, snap) == []
    flags = exec_flags(snap)
    assert set(flags) == {"queue_nonempty", "own_combat", "nearest_enemy",
                          "weakest_enemy", "weak_own"}
    entry = make_skip_entry(i=3, tick=100, state_kind="combat",
                            state_tokens=49, flags=flags)
    assert entry["kind"] == "skip"
    assert entry["ballot"] == []
    assert entry["gate"]["mode"] == "skip-empty-ballot"
    assert "prediction" not in entry and "confidence" not in entry
    assert set(entry["exec_flags"]) == set(flags)


def test_prefilter_combat_with_troops_keeps_attack():
    snap = _eco_snap(
        own_units=[{"id": 1, "type": "e1", "cell_x": 40, "cell_y": 40,
                    "hp_percent": 1.0}],
    )
    foe = {"id": 9, "type": "e1", "cell_x": 44, "cell_y": 44, "hp_percent": 0.5}
    snap.enemies = [foe]
    snap.nearest_enemy = foe
    snap.weakest_enemy = foe
    names = [c.name for c in
             list_executable_candidates(list_combat_candidates(), snap)]
    assert "attack_nearest" in names and "guard_choke" in names


def _lowconf_pred(names: list[str]) -> Prediction:
    n = len(names)
    return Prediction(choice=names[0],
                      probs={nm: 1.0 / n for nm in names},
                      confidence=0.19, latency_ms=1.0, cost_usd=0.0,
                      backend="nanojev", detail={})


def test_banned_dedup_same_state_never_repeats_twice():
    """Judge criterion: `choice + fail` groups occur <= 2 times per game."""
    scripted = ScriptedBackend()
    cands = list_eco_candidates()
    names = [c.name for c in cands]
    state = build_eco_state(_eco_snap())  # power 0, empty prod -> A2: build_powr
    assert a2_preferred(state, names)[0] == "build_powr"
    failed: set[str] = set()
    seen: list[str] = []
    for _ in range(5):
        gate = apply_gate(_lowconf_pred(names), cands, scripted, state,
                          GateConfig(), banned=failed)
        assert gate.mode == "fallback"
        assert gate.choice_name not in failed  # never repeats a banned pick
        seen.append(gate.choice_name)
        failed.add(gate.choice_name)  # caller records batch_ok=false
    assert len(set(seen)) == 5  # 5 decisions, 5 distinct choices


def test_all_banned_lands_on_wait():
    scripted = ScriptedBackend()
    cands = list_combat_candidates()
    names = [c.name for c in cands]
    state = {"kind": "combat", "own": [], "enemies": []}
    fb = predict_with_ban(state, cands, set(names), scripted, probs={})
    assert fb.choice == "wait"
    assert fb.detail.get("rule") == "all-banned->wait"


def test_executor_no_same_action_retry():
    calls: list[str] = []

    class FakeClient:
        async def call_tool(self, name, **kwargs):
            calls.append(name)
            return "FAILED (No valid commands generated)"

    ex = Executor(FakeClient())
    rec = asyncio.run(ex.execute("build_barr", [{"tool": "x"}], tick=0,
                                 guard_actions=None))
    assert rec.batch_ok is False
    assert rec.retried is False
    assert calls == ["batch"]  # one shot; no identical second batch
    calls.clear()
    rec2 = asyncio.run(ex.execute("build_barr", [{"tool": "x"}], tick=0,
                                  guard_actions=[{"tool": "guard"}]))
    assert calls == ["batch", "batch"]  # fail + guard re-dispatch only
    assert rec2.fell_back_to_guard is True
