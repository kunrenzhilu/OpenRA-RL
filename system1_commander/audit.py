"""Offline truth audit for System1 orders.jsonl (no server, zero cost).

Old scoring counted ``batch_ok=true`` as "order placed". That lied: a batch
can be accepted while changing nothing on the map (e.g. queueing a 20th
``powr`` behind a head item that never gets placed). This module replays
``orders.jsonl`` offline and diffs the *observed map state* carried inside
each ``batch_note`` JSON blob::

    economy.cash, own_buildings, own_units, production[], visible_enemies

against the previous observation, yielding the ground-truth five-tuple::

    placed_delta, units_delta, queue_delta, cash_delta, enemy_delta

Verdict rules (one line each):
- ``effective``: ``placed_delta != 0`` or ``units_delta != 0`` (something
  visibly appeared / disappeared on the map).
- ``side_change``: map counts unchanged but queue/cash/enemies moved
  (sidebar-only change, e.g. +1 queued ``powr@0%``).
- ``noop_ok``: ``batch_ok`` with a fully-observed all-zero five-tuple
  (accepted but no-op).
- ``noop_ok_capped``: like ``noop_ok`` but the note was cut at 500 chars so
  the queue tail is unobservable (prefix saturated) -- not counted as clean.
- ``empty_ok``: ``batch_ok`` with no actions and no observable state
  (advance-only tick, e.g. ``wait`` / infeasible candidate -> ``[]``).
- ``failed``: ``batch_ok == false`` (server rejected: ``FAILED`` /
  ``No valid commands``).

Notes
-----
* ``Executor.to_dict`` truncates ``batch_note`` to 500 chars, so long
  production queues are cut mid-list. ``parse_batch_state`` therefore falls
  back to regex salvage for ``cash``/counts plus the visible production
  prefix, and flags the row ``truncated``.
* Observations are sparse (only non-empty batches + deploy carry state), so
  each row diffs against the *previous observed* state; the row records how
  many decisions/ticks the gap spans (``gap_decisions``/``gap_ticks``) so a
  delta is never silently attributed to the wrong order (e.g. the MCV
  deploy taking effect during the gap before the first ``build_powr``).
"""

from __future__ import annotations

import json
import re
from typing import Any

FAILURE_MARKERS = ("FAILED", "no valid commands", "No valid commands", "error")

_PROD_ITEM_RE = re.compile(r'"([A-Za-z0-9_]+@\d+%)"')
_CASH_RE = re.compile(r'"cash"\s*:\s*(-?\d+)')
_BUILDINGS_RE = re.compile(r'"own_buildings"\s*:\s*(-?\d+)')
_UNITS_RE = re.compile(r'"own_units"\s*:\s*(-?\d+)')
_ENEMIES_RE = re.compile(r'"visible_enemies"\s*:\s*(-?\d+)')


def _salvage(note: str) -> dict[str, Any] | None:
    """Regex-salvage scalars + visible production prefix from a cut note."""
    m_cash = _CASH_RE.search(note)
    if not m_cash:
        return None
    def _int(rx: re.Pattern[str]) -> int | None:
        m = rx.search(note)
        return int(m.group(1)) if m else None
    prod = _PROD_ITEM_RE.findall(note)
    # Production array cut <=> note no longer parses as JSON but held items.
    truncated = True
    return {
        "cash": int(m_cash.group(1)),
        "own_buildings": _int(_BUILDINGS_RE),
        "own_units": _int(_UNITS_RE),
        "visible_enemies": _int(_ENEMIES_RE),
        "production": prod,
        "truncated": truncated,
    }


def parse_batch_state(note: Any) -> dict[str, Any] | None:
    """Extract observed map state from a ``batch_note``.

    Returns ``None`` when the note carries no state (``empty batch ...`` /
    ``FAILED`` replies). Otherwise ``{cash, own_buildings, own_units,
    visible_enemies, production, truncated}``; any field may be ``None``
    when only salvageable partially.
    """
    if not isinstance(note, str) or not note.lstrip().startswith("{"):
        clean = note if isinstance(note, str) else ""
        if clean.startswith("retry:") or clean.startswith("guard fallback:"):
            clean = clean.split(":", 1)[1].strip()
        if clean.lstrip().startswith("{"):
            return parse_batch_state(clean)
        return None
    try:
        d = json.loads(note)
    except (json.JSONDecodeError, ValueError):
        return _salvage(note)
    if not isinstance(d, dict) or "economy" not in d:
        return _salvage(note)
    eco = d.get("economy") or {}
    prod = d.get("production") or []
    return {
        "cash": eco.get("cash"),
        "own_buildings": d.get("own_buildings"),
        "own_units": d.get("own_units"),
        "visible_enemies": d.get("visible_enemies"),
        "production": [str(x) for x in prod] if isinstance(prod, list) else [],
        "truncated": False,
    }


def _looks_failed(note: Any) -> bool:
    text = note if isinstance(note, str) else json.dumps(note, default=str)
    return any(m in text for m in FAILURE_MARKERS)


def _spend_kind(choice: str) -> str:
    if choice.startswith("train_"):
        return "train"
    if choice.startswith("build_"):
        return "build"
    return "other"


def audit_decisions(
    decisions: list[dict],
    baseline: dict | None = None,
) -> dict[str, Any]:
    """Audit decision rows; ``baseline`` is an optional deploy-row dict.

    Returns ``{"rows": [...], "summary": {...}}``. Pure function, no I/O.
    """
    prev_state: dict[str, Any] | None = None
    prev_tick = 0
    prev_idx = -1
    last_order_choice = ""
    if baseline:
        prev_state = parse_batch_state(baseline.get("batch_note", ""))
        prev_tick = int(baseline.get("tick", 0) or 0)

    rows: list[dict] = []
    cash_spent = 0
    spend_by_kind = {"build": 0, "train": 0, "other": 0}
    queue_peak_full = 0
    queue_peak_visible = 0

    for n, rec in enumerate(decisions):
        choice = str(rec.get("choice", ""))
        actions = rec.get("actions") or []
        batch_ok = bool(rec.get("batch_ok", False))
        note = rec.get("batch_note", "")
        tick = int(rec.get("tick", 0) or 0)
        state = parse_batch_state(note) if batch_ok else None
        failed_note = (not batch_ok) or _looks_failed(note)

        row: dict[str, Any] = {
            "i": rec.get("i", n),
            "tick": tick,
            "choice": choice,
            "batch_ok": batch_ok,
            "n_actions": len(actions),
            "observed": state is not None,
            "truncated": bool(state and state.get("truncated")),
            "verdict": "",
            "delta": None,
            "gap_decisions": 0,
            "gap_ticks": 0,
        }

        if failed_note:
            row["verdict"] = "failed"
        elif state is None:
            row["verdict"] = "empty_ok" if (batch_ok and not actions) else "failed"
        elif prev_state is None or prev_state.get("cash") is None:
            # First observation (no baseline): nothing to diff against.
            row["verdict"] = "side_change" if actions else "empty_ok"
            prev_state, prev_tick, prev_idx = state, tick, n
        else:
            delta = {
                "placed_delta": (state.get("own_buildings") or 0)
                - (prev_state.get("own_buildings") or 0),
                "units_delta": (state.get("own_units") or 0)
                - (prev_state.get("own_units") or 0),
                "queue_delta": len(state.get("production") or [])
                - len(prev_state.get("production") or []),
                "cash_delta": (state.get("cash") or 0) - (prev_state.get("cash") or 0),
                "enemy_delta": (state.get("visible_enemies") or 0)
                - (prev_state.get("visible_enemies") or 0),
            }
            row["delta"] = delta
            row["gap_decisions"] = n - prev_idx
            row["gap_ticks"] = tick - prev_tick
            if delta["cash_delta"] < 0:
                spend = -delta["cash_delta"]
                cash_spent += spend
                kind = _spend_kind(last_order_choice or choice)
                spend_by_kind[kind] = spend_by_kind.get(kind, 0) + spend
            if delta["placed_delta"] != 0 or delta["units_delta"] != 0:
                row["verdict"] = "effective"
            elif all(v == 0 for v in delta.values()):
                row["verdict"] = (
                    "noop_ok_capped" if row["truncated"] else "noop_ok"
                )
            else:
                row["verdict"] = "side_change"
            prev_state, prev_tick, prev_idx = state, tick, n

        if state is not None:
            q = len(state.get("production") or [])
            queue_peak_visible = max(queue_peak_visible, q)
            if not row["truncated"]:
                queue_peak_full = max(queue_peak_full, q)
        if actions:
            last_order_choice = choice
        rows.append(row)

    def _count(v: str) -> int:
        return sum(1 for r in rows if r["verdict"] == v)

    summary = {
        "decisions": len(decisions),
        "effective": _count("effective"),
        "noop_ok": _count("noop_ok"),
        "noop_ok_capped": _count("noop_ok_capped"),
        "side_change": _count("side_change"),
        "empty_ok": _count("empty_ok"),
        "failed": _count("failed"),
        "map_visible_orders": _count("effective"),
        "noop_ok_count": _count("noop_ok"),
        "cash_spent": cash_spent,
        "cash_spent_build": spend_by_kind.get("build", 0),
        "cash_spent_train": spend_by_kind.get("train", 0),
        "cash_spent_other": spend_by_kind.get("other", 0),
        "queue_peak_full": queue_peak_full,
        "queue_peak_visible": queue_peak_visible,
    }
    return {"rows": rows, "summary": summary}


def audit_file(path: str) -> dict[str, Any]:
    """Offline replay of one ``orders.jsonl`` file (deploy row = baseline)."""
    baseline, decisions = None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("kind") == "deploy" and baseline is None:
                baseline = rec
            elif rec.get("kind", "decision") == "decision":
                decisions.append(rec)
    out = audit_decisions(decisions, baseline)
    out["summary"]["orders_file"] = path
    return out
