"""Jevfix F6/F1/F2 unit tests (pure functions only, zero cost).

F6: macro ballot never prefiltered; demand-proxy mapping.
F1: resolve_macro_execution priority + retirement table + weak threshold.
F2: iron-rule reword (no MUST/prefer/Only) + train_* admission + ballot keys.
"""

import re

from system1_commander.backend_jev import JevBackend
from system1_commander.candidates import (
    COMBAT_MACRO_NAMES,
    DEMAND_PROXY_MACROS,
    IRON_RETIREMENT,
    MACRO_DEFAULT_ATOMIC,
    MACRO_NAMES,
    MACRO_WEAK_CONF,
    demand_proxy_action,
    exec_flags,
    iron_retired,
    list_eco_candidates,
    list_executable_candidates,
    list_executable_macros,
    list_macro_candidates,
    resolve_macro_execution,
)
from system1_commander.state import Snapshot


def _snap(**kw) -> Snapshot:
    base = dict(base_cell=(45, 85), available_production=[],
                ready_to_place=[], production=[])
    base.update(kw)
    return Snapshot(**base)


def _by_name():
    return {c.name: c for c in list_eco_candidates()}


# ── F6 ─────────────────────────────────────────────────────────────

def test_macro_ballot_never_prefiltered():
    ms = list_macro_candidates()
    assert [m.name for m in ms] == list(MACRO_NAMES)
    assert len(list_executable_macros(ms, _snap())) == 5  # conyard: all stay
    full = _snap(available_production=["harv", "1tnk"])
    assert len(list_executable_macros(ms, full)) == 5


def test_demand_proxy_maps_to_build_weap():
    by = _by_name()
    assert set(DEMAND_PROXY_MACROS) == {"econ_harv", "armor_push"}
    p = demand_proxy_action("econ_harv", _snap(), by)
    assert p is not None and p[0] == "build_weap"
    assert p[1] == [{"tool": "build_structure", "building_type": "weap"}]
    p2 = demand_proxy_action("armor_push", _snap(), by)
    assert p2 is not None and p2[0] == "build_weap"
    # Producible -> no proxy (normal mapping applies).
    assert demand_proxy_action(
        "econ_harv", _snap(available_production=["harv"]), by) is None
    assert demand_proxy_action(
        "armor_push", _snap(available_production=["1tnk"]), by) is None
    # Structural macros never proxy.
    assert demand_proxy_action("fast_weap", _snap(), by) is None


# ── F1 ─────────────────────────────────────────────────────────────

def test_default_map_covers_all_macros():
    assert set(MACRO_DEFAULT_ATOMIC) == set(MACRO_NAMES)
    assert MACRO_DEFAULT_ATOMIC["fast_weap"] == "build_weap"
    assert MACRO_DEFAULT_ATOMIC["econ_harv"] == "train_harv"
    assert MACRO_DEFAULT_ATOMIC["armor_push"] == "train_1tnk"


def test_resolve_place_first_wins():
    by = _by_name()
    s = _snap(ready_to_place=["powr"],
              production=[{"queue_type": "Building", "item": "powr",
                           "progress": 1.0}])
    atomic, acts, tag = resolve_macro_execution("fast_weap", s, by)
    assert (atomic, tag) == ("place_ready", "place-first")
    assert acts[0]["tool"] == "place_building"


def test_resolve_demand_proxy_before_default():
    by = _by_name()
    atomic, acts, tag = resolve_macro_execution("econ_harv", _snap(), by)
    assert (atomic, tag) == ("build_weap", "demand-proxy")
    assert acts == [{"tool": "build_structure", "building_type": "weap"}]
    atomic, acts, tag = resolve_macro_execution(
        "econ_harv", _snap(available_production=["harv"]), by)
    assert (atomic, tag) == ("train_harv", "mapped")


def test_resolve_queue_blocked_is_macro_wait():
    by = _by_name()
    s = _snap(production=[{"queue_type": "Building", "item": "powr",
                           "progress": 0.5}])
    atomic, acts, tag = resolve_macro_execution("fast_weap", s, by)
    assert (atomic, acts, tag) == ("wait", [], "macro-wait")


def test_resolve_illegal_never_revotes():
    by = _by_name()
    # Unknown macro and missing atomic both fall through to named waits.
    assert resolve_macro_execution("nope", _snap(), by)[2] == \
        "macro-illegal-wait"
    thin = {k: v for k, v in by.items() if k != "build_weap"}
    assert resolve_macro_execution("fast_weap", _snap(), thin)[2] == \
        "macro-illegal-wait"
    # train_1tnk filtered from the ballot when unproducible: armor_push
    # proxies instead of going illegal.
    assert resolve_macro_execution("armor_push", _snap(), by)[2] == \
        "demand-proxy"


def test_iron_retirement_table_frozen():
    assert set(IRON_RETIREMENT) == {"rule1_place_first",
                                    "rule2_queue_guard",
                                    "rule3_can_make_veto"}
    assert iron_retired(two_phase_on=True, kind="eco", phase2=True) == [
        "rule2_queue_guard", "rule3_can_make_veto"]
    assert iron_retired(two_phase_on=True, kind="eco", phase2=False) == []
    assert iron_retired(two_phase_on=True, kind="combat", phase2=True) == []
    assert iron_retired(two_phase_on=False, kind="eco", phase2=True) == []
    assert MACRO_WEAK_CONF == 0.40


# ── F2 ─────────────────────────────────────────────────────────────

def _tactic(cands):
    b = JevBackend.__new__(JevBackend)  # no key needed for _questions
    return b._questions(cands)["tactic"].instructions


def test_iron_reword_no_prescription_keeps_anchors():
    mq = _tactic(list_macro_candidates())
    aq = _tactic(list_eco_candidates())
    assert "phase goal" in mq and "Iron backup" in mq
    assert "(2)" in aq and "train_harv" in aq
    for q in (mq, aq):
        assert re.findall(r"(?i)\bmust\b|\bprefer\b|\bonly\b", q) == []


def test_train_admission_in_harness():
    full = _snap(available_production=["e1", "harv", "1tnk", "powr"])
    names = {c.name for c in list_executable_candidates(
        list_eco_candidates(), full)}
    assert {"train_e1", "train_harv", "train_1tnk"} <= names
    bare = _snap(available_production=[])
    names_bare = {c.name for c in list_executable_candidates(
        list_eco_candidates(), bare)}
    assert not {n for n in names_bare if n.startswith("train_")}
    assert "build_powr" in names_bare and "wait" in names_bare
    partial = _snap(available_production=["e1"])
    names_p = {c.name for c in list_executable_candidates(
        list_eco_candidates(), partial)}
    assert "train_e1" in names_p and "train_harv" not in names_p


def test_exec_flags_frozen_five_keys():
    # A-track schema freeze: recount keys stay exactly five; train_*
    # admission is recounted from the per-row ballot, not from flags.
    f = exec_flags(_snap(available_production=["harv", "powr"]))
    assert set(f) == {"queue_nonempty", "own_combat", "nearest_enemy",
                      "weakest_enemy", "weak_own"}


def test_combat_macro_names_disjoint():
    assert not (set(COMBAT_MACRO_NAMES) & set(MACRO_NAMES))
