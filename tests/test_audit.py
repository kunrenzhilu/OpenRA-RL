"""Truth-audit tests (offline replay, no server, zero cost).

Fixture = first 5 lines of the real Jev game
(.runs/system1-jev-20260921/orders.jsonl): deploy row + decisions 0..3,
all advance-only empties. Guards the rule: batch_ok must never again be
counted as "orders placed".
"""

import os

from system1_commander.audit import audit_file, parse_batch_state

_FIXTURE = os.path.join(
    os.path.dirname(__file__), "fixtures", "system1-jev-first5.jsonl"
)


def test_fixture_truth_counts():
    out = audit_file(_FIXTURE)
    s = out["summary"]
    assert s["decisions"] == 4
    assert s["map_visible_orders"] == 0
    assert s["effective"] == 0
    assert s["noop_ok_count"] == 0
    assert s["noop_ok"] == 0
    assert s["empty_ok"] == 4
    assert s["failed"] == 0
    assert s["cash_spent"] == 0
    assert s["queue_peak_full"] == 0
    assert [r["verdict"] for r in out["rows"]] == ["empty_ok"] * 4


def test_baseline_parses_deploy_state():
    import json

    with open(_FIXTURE) as f:
        deploy = json.loads(f.readline())
    st = parse_batch_state(deploy["batch_note"])
    assert st is not None and not st["truncated"]
    assert st["cash"] == 5000
    assert st["own_buildings"] == 0
    assert st["own_units"] == 1
    assert st["production"] == []


def test_empty_note_has_no_state():
    assert parse_batch_state("empty batch (advance only)") is None
