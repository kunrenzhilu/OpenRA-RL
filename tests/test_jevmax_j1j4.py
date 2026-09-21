"""Jev-max J1-J4 unit tests (pure functions only, zero cost).

Covers: rich-field token caps + oldest-trim, state_to_json budget param,
top1/KL helpers, macro table + prefilter + builders, Jev macro-question
branch + fan-out questions, goal routing, summarizer + history cap.
"""

import json

from system1_commander.backend_jev import JevBackend
from system1_commander.candidates import (
    MACRO_NAMES,
    list_eco_candidates,
    list_executable_macros,
    list_macro_candidates,
)
from system1_commander.demo_loop import (
    MEMORY_K,
    fanout_of,
    jev_goal_route,
    kl_div_bits,
    summarize_trajectory,
    top1_of,
    truthy_noul,
)
from system1_commander.state import (
    HISTORY_TOKEN_BUDGET,
    RICH_TOKEN_CAPS,
    Snapshot,
    attach_history,
    attach_rich_fields,
    build_eco_state,
    estimate_tokens,
    observe_trackers,
    reset_trackers,
    state_to_json,
)


def _fat_snap() -> Snapshot:
    units = [{"id": i, "type": "harv" if i % 3 == 0 else "e1",
              "cell_x": 40 + i, "cell_y": 5 + i} for i in range(1, 13)]
    foes = [{"id": 100 + i, "type": f"e{i % 4 + 1}",
             "cell_x": 60 + i, "cell_y": 30 + i} for i in range(8)]
    blds = [{"actor_id": i, "type": t, "hp_percent": 1.0, "producing_item": ""}
            for i, t in enumerate(["fact", "powr", "powr", "barr"], start=1)]
    return Snapshot(
        tick=5000, cash=3000, ore=1200, power_balance=30,
        harvester_count=4, kills=6, losses=2, kills_cost=3000,
        deaths_cost=800, army_value=2500, order_count=40,
        own_units=units, own_buildings=blds, enemies=foes,
        production=[{"queue_type": "Building", "item": "weap",
                     "progress": 0.5}],
        available_production=["powr", "barr", "weap", "harv", "1tnk", "e1"],
        base_cell=(45, 5), minimap="Map (23x23):\n" + ("#" * 23 + "\n") * 23,
    )


# ── J1 ─────────────────────────────────────────────────────────────

def test_rich_fields_within_caps():
    reset_trackers()
    snap = _fat_snap()
    observe_trackers(snap)
    rich = attach_rich_fields(build_eco_state(snap), snap)
    assert set(MACRO_NAMES)  # table import sanity (used below too)
    for key, cap in RICH_TOKEN_CAPS.items():
        assert key in rich, key
        toks = estimate_tokens(json.dumps(rich[key], separators=(",", ":")))
        assert toks <= cap, (key, toks, cap)


def test_field_set_frozen_five():
    reset_trackers()
    snap = _fat_snap()
    observe_trackers(snap)
    poor = build_eco_state(snap)
    rich = attach_rich_fields(dict(poor), snap)
    assert set(rich) - set(poor) == set(RICH_TOKEN_CAPS)


def test_enemy_memory_trims_oldest_first():
    reset_trackers()
    base = _fat_snap()
    # Observe 30 distinct enemy types across ticks; cap forces trimming.
    for t in range(30):
        s = Snapshot(tick=100 + t, base_cell=(45, 5),
                     enemies=[{"id": t, "type": f"foe{t:02d}",
                               "cell_x": 60, "cell_y": 40}])
        observe_trackers(s)
    rich = attach_rich_fields(build_eco_state(base), base)
    seen = rich["enemy_history"]["last_seen"]
    toks = estimate_tokens(json.dumps(rich["enemy_history"],
                                      separators=(",", ":")))
    assert toks <= RICH_TOKEN_CAPS["enemy_history"]
    # Newest-first output; the oldest types must be gone if trimmed.
    if len(seen) < 30:
        types = {r["t"] for r in seen}
        assert "foe29" in types and "foe00" not in types


def test_state_to_json_budget_param():
    reset_trackers()
    snap = _fat_snap()
    observe_trackers(snap)
    rich = attach_rich_fields(build_eco_state(snap), snap)
    _, default_toks = state_to_json(rich)
    _, big_toks = state_to_json(rich, 4000)
    assert default_toks == big_toks  # fits default; budget is a no-op
    small_text, small_toks = state_to_json(rich, 50)
    assert small_toks < default_toks
    assert json.loads(small_text)["kind"] == "eco"  # core survives


def test_top1_and_kl():
    assert top1_of({"a": 0.2, "b": 0.7}) == "b"
    assert top1_of({}) is None
    assert top1_of({"b": 0.5, "a": 0.5}) == "a"  # tie: sorted first
    assert kl_div_bits({"a": 1.0}, {"a": 1.0}) == 0.0
    assert kl_div_bits({}, {}) == 0.0
    kl = kl_div_bits({"a": 0.5, "b": 0.5}, {"a": 0.9, "b": 0.1})
    assert 0.5 < kl < 1.0


# ── J2 ─────────────────────────────────────────────────────────────

def test_macro_table_full_and_qualified():
    ms = list_macro_candidates()
    assert [m.name for m in ms] == list(MACRO_NAMES)
    assert "fast_weap" in MACRO_NAMES  # disqualifier guard (§2)
    assert all(m.kind == "eco" for m in ms)


def test_macro_prefilter_can_make():
    # Jevfix F6 (rev4): macro ballot 永不预过滤 —— 5 宏恒在票上，
    # can_make 只影响执行映射（demand-proxy），不影响上票。
    ms = list_macro_candidates()
    conyard = Snapshot(available_production=[])
    assert [m.name for m in list_executable_macros(ms, conyard)] == [
        "open_powr", "rush_barr", "fast_weap", "econ_harv", "armor_push"]
    full = Snapshot(available_production=["harv", "1tnk"])
    assert [m.name for m in list_executable_macros(ms, full)] == [
        "open_powr", "rush_barr", "fast_weap", "econ_harv", "armor_push"]


def test_macro_builders_first_legal_step():
    ms = {m.name: m for m in list_macro_candidates()}
    conyard = Snapshot(base_cell=(45, 5), available_production=[])
    assert ms["open_powr"].actions(conyard) == [
        {"tool": "build_structure", "building_type": "powr"}]
    assert ms["econ_harv"].actions(conyard) == []
    assert ms["armor_push"].actions(conyard) == []
    placed = Snapshot(base_cell=(45, 5), ready_to_place=["powr"],
                      production=[{"queue_type": "Building", "item": "powr",
                                   "progress": 1.0}])
    assert ms["open_powr"].actions(placed)[0]["tool"] == "place_building"


def test_jev_macro_instructions_branch():
    b = JevBackend.__new__(JevBackend)  # no key needed for _questions
    mq = b._questions(list_macro_candidates())["tactic"].instructions
    assert "phase goal" in mq and "Iron backup" in mq
    aq = b._questions(list_eco_candidates())["tactic"].instructions
    assert "(2)" in aq and "train_harv" in aq  # atomic iron rules intact


# ── J3 ─────────────────────────────────────────────────────────────

def test_jev_fanout_questions_present():
    b = JevBackend.__new__(JevBackend)
    q = b._questions(list_eco_candidates())
    assert set(q) >= {"tactic", "goal", "phase", "threat_recall",
                      "risk_under_attack", "risk_overextend", "danger"}


def test_fanout_of_and_truthy():
    d = {"choices": {"goal": {"choice": "defend", "confidence": 0.8}},
         "nouls": {"threat_recall": True}, "scores": {"phase": 2, "danger": 1}}
    fo = fanout_of(d)
    assert (fo["goal"], fo["phase"], fo["threat_recall"]) == ("defend", 2, True)
    assert all(v is None for v in fanout_of({"rule": "x"}).values())
    assert truthy_noul(True) and not truthy_noul("false")
    assert not truthy_noul(None)


def test_goal_routing_matrix():
    r, n = jev_goal_route(gate_mode="downgrade", gate_choice="wait",
                          ballot_names=["wait"], fanout={"goal": "expand_eco"})
    assert r == "__fallback__" and "downgrade" in n
    r, _ = jev_goal_route(gate_mode="execute", gate_choice="attack_nearest",
                          ballot_names=["attack_nearest", "guard_choke"],
                          fanout={"goal": "defend", "threat_recall": True})
    assert r == "guard_choke"
    r, _ = jev_goal_route(gate_mode="execute", gate_choice="build_powr",
                          ballot_names=["build_powr"],
                          fanout={"goal": "defend", "threat_recall": True})
    assert r is None  # non-aggressive tactic passes through


# ── J4 ─────────────────────────────────────────────────────────────

def test_memory_k_and_summarizer_window():
    assert MEMORY_K == 10
    rows = [{"kind": "decision", "i": i, "tick": 100 + i * 25,
             "choice": "build_powr", "gate": {"mode": "execute"},
             "actions": [{"tool": "x"}], "batch_ok": True,
             "batch_note": json.dumps(
                 {"economy": {"cash": 5000 - i * 10}, "own_buildings": 1,
                  "own_units": 1, "visible_enemies": 0, "production": []})}
            for i in range(15)]
    rows.append({"kind": "alarm", "i": 99})  # ignored
    s = summarize_trajectory(rows)
    assert s["n"] == 10 and s["segs"][0]["i"] == 5
    assert len(s["cash_curve"]) == 5
    assert summarize_trajectory([])["n"] == 0
    # Pure: same input -> same output.
    assert summarize_trajectory(rows) == summarize_trajectory(rows)


def test_history_budget_and_oldest_trim():
    rows = [{"kind": "decision", "i": i, "tick": i,
             "choice": f"choice_{i % 3}_with_a_long_name",
             "gate": {"mode": "execute"}, "actions": [], "batch_ok": True,
             "batch_note": "empty batch (advance only)"} for i in range(50)]
    st = attach_history({"kind": "eco"}, summarize_trajectory(rows, k=50))
    toks = estimate_tokens(json.dumps(st["history"], separators=(",", ":")))
    assert toks <= HISTORY_TOKEN_BUDGET == 300
    assert attach_history({"a": 1}, None) == {"a": 1}
