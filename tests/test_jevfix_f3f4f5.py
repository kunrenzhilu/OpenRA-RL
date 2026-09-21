"""Jevfix F3/F4/F5 unit tests (pure functions only, zero cost).

F3: directional place variants + enemy bearing helper.
F4: same-function fallback (predict_with_ban) + new CLI flags.
F5: combat macros + split_harass/screen_mcv + prefilter + ballot keys.
"""

from system1_commander.backend_jev import JevBackend
from system1_commander.candidates import (
    COMBAT_MACRO_DEFAULT_ATOMIC,
    COMBAT_MACRO_NAMES,
    PLACE_DIRS,
    RULE1_PLACE_NAMES,
    list_combat_candidates,
    list_combat_macros,
    list_eco_candidates,
    list_executable_candidates,
    resolve_macro_execution,
)
from system1_commander.demo_loop import (
    _parse_args,
    attach_enemy_bearing,
    bearing8,
)
from system1_commander.gate import predict_with_ban
from system1_commander.backend_scripted import ScriptedBackend
from system1_commander.state import Snapshot


def _snap(**kw) -> Snapshot:
    base = dict(base_cell=(45, 85), available_production=[],
                ready_to_place=[], production=[])
    base.update(kw)
    return Snapshot(**base)


def _combat_snap(n_own=2):
    own = [{"id": i, "type": "e1", "cell_x": 40 + i, "cell_y": 80 + i}
           for i in range(1, n_own + 1)]
    foe = {"id": 99, "type": "e1", "cell_x": 60, "cell_y": 60}
    return _snap(own_units=own, enemies=[foe], nearest_enemy=foe,
                 weakest_enemy=foe, enemy_centroid=(60, 60),
                 weak_own_ids=[])


# ── F3 ─────────────────────────────────────────────────────────────

def test_place_dirs_repo_convention():
    assert [d for d, _, _ in PLACE_DIRS] == ["N", "E", "S", "W"]
    assert dict((d, (dx, dy)) for d, dx, dy in PLACE_DIRS) == {
        "N": (0, -3), "E": (3, 0), "S": (0, 3), "W": (-3, 0)}
    assert set(RULE1_PLACE_NAMES) == {"place_ready", "place_N", "place_E",
                                      "place_S", "place_W"}


def test_place_variants_admit_only_when_blocked():
    names_all = {c.name for c in list_eco_candidates()}
    assert set(RULE1_PLACE_NAMES) <= names_all
    open_snap = _snap()
    assert not ({c.name for c in list_executable_candidates(
        list_eco_candidates(), open_snap)} & set(RULE1_PLACE_NAMES))
    blocked = _snap(ready_to_place=["powr"])
    admitted = {c.name for c in list_executable_candidates(
        list_eco_candidates(), blocked)}
    assert set(RULE1_PLACE_NAMES) <= admitted


def test_place_variant_offsets():
    by = {c.name: c for c in list_eco_candidates()}
    s = _snap(ready_to_place=["barr"])
    assert by["place_N"].actions(s) == [
        {"tool": "place_building", "building_type": "barr",
         "cell_x": 45, "cell_y": 82}]
    assert by["place_E"].actions(s)[0]["cell_x"] == 48
    assert by["place_S"].actions(s)[0]["cell_y"] == 88
    assert by["place_W"].actions(s)[0]["cell_x"] == 42
    assert by["place_N"].actions(_snap()) == []


def test_bearing8_and_attach():
    assert bearing8(5, 0) == "E" and bearing8(-5, 0) == "W"
    assert bearing8(0, 5) == "S" and bearing8(0, -5) == "N"
    assert bearing8(0, 0) == "here"
    assert bearing8(3, 4) == "SE" and bearing8(-3, -4) == "NW"
    st = attach_enemy_bearing({"kind": "eco"}, _combat_snap())
    assert st["enemy_dir"] == "NE"  # (60-45, 60-85) = (+15,-25)
    assert st["enemy_dist"] == "close"  # chebyshev 25
    assert st["kind"] == "eco"  # non-destructive copy
    st2 = attach_enemy_bearing({"kind": "eco"}, _snap())
    assert (st2["enemy_dir"], st2["enemy_dist"]) == ("none", "none")


# ── F4 ─────────────────────────────────────────────────────────────

def test_second_choice_same_function_ban():
    cands = list_combat_candidates()
    state = {"kind": "combat", "enemies": [{"id": 1}]}
    fb = predict_with_ban(state, cands, {"attack_nearest"},
                          ScriptedBackend(), {"guard_choke": 0.9})
    assert fb.choice == "guard_choke"
    # Unbanned top pick passes through (legacy behaviour).
    fb2 = predict_with_ban(state, cands, set(), ScriptedBackend(), None)
    assert fb2.choice == "attack_nearest"


def test_new_cli_flags():
    a = _parse_args(["--double-sample", "--second-choice", "--combat-macros"])
    assert a.double_sample and a.second_choice and a.combat_macros
    assert a.double_sample_every == 10
    d = _parse_args([])
    assert not d.double_sample and not d.second_choice \
        and not d.combat_macros


# ── F5 ─────────────────────────────────────────────────────────────

def test_combat_macro_table_and_map():
    ms = list_combat_macros()
    assert [m.name for m in ms] == list(COMBAT_MACRO_NAMES)
    assert set(COMBAT_MACRO_DEFAULT_ATOMIC) == set(COMBAT_MACRO_NAMES)
    assert COMBAT_MACRO_DEFAULT_ATOMIC["defend_hold"] == "guard_choke"
    assert COMBAT_MACRO_DEFAULT_ATOMIC["probe_attack"] == "attack_nearest"
    assert COMBAT_MACRO_DEFAULT_ATOMIC["all_in_commit"] == \
        "all_combat_attack_move"
    b = JevBackend.__new__(JevBackend)
    q = b._questions(ms)["tactic"].instructions
    assert "posture" in q


def test_new_verbs_builders():
    by = {c.name: c for c in list_combat_candidates()}
    s = _combat_snap(n_own=4)
    sp = by["split_harass"].actions(s)
    assert len(sp) == 2
    assert sp[0]["tool"] == "attack_target"
    assert sp[0]["target_actor_id"] == 99
    assert sp[1] == {"tool": "stop_units", "unit_ids": "3,4"}
    sc = by["screen_mcv"].actions(s)
    assert sc == [{"tool": "move_units", "unit_ids": "all_combat",
                   "target_x": 45, "target_y": 85}]
    assert by["split_harass"].actions(_snap()) == []
    assert by["screen_mcv"].actions(_snap()) == []


def test_new_verbs_prefilter():
    names = {c.name for c in list_executable_candidates(
        list_combat_candidates(), _combat_snap())}
    assert {"split_harass", "screen_mcv"} <= names
    names_bare = {c.name for c in list_executable_candidates(
        list_combat_candidates(), _snap())}
    assert "split_harass" not in names_bare
    assert "screen_mcv" not in names_bare


def test_combat_macro_execution_mapped():
    by = {c.name: c for c in list_combat_candidates()}
    s = _combat_snap()
    atomic, acts, tag = resolve_macro_execution(
        "probe_attack", s, by, COMBAT_MACRO_DEFAULT_ATOMIC)
    assert (atomic, tag) == ("attack_nearest", "mapped")
    assert acts[0]["tool"] == "attack_target"
    # No own force: atomic illegal -> named wait, never a re-vote.
    atomic, acts, tag = resolve_macro_execution(
        "all_in_commit", _snap(), by, COMBAT_MACRO_DEFAULT_ATOMIC)
    assert (atomic, acts, tag) == ("wait", [], "macro-illegal-wait")
