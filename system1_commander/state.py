"""Observation -> compact English JSON state (<=800 tokens).

All arithmetic is done here, never by the model (Jev arithmetic is
unreliable): power balance, K/D cost ratio, funds buckets, HP buckets,
Chebyshev distance buckets. Output is standard JSON (json.dumps).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

COMBAT_UNIT_TYPES = {"e1", "e2", "e3", "e4", "1tnk", "2tnk", "3tnk", "arty", "jeep", "apc"}
STATE_TOKEN_BUDGET = 800  # per plan sec2


def hp_bucket(hp: float) -> str:
    if hp >= 0.66:
        return "healthy"
    if hp >= 0.33:
        return "weak"
    return "critical"


def dist_bucket(d_cells: float) -> str:
    if d_cells <= 8:
        return "in_range"
    if d_cells <= 25:
        return "close"
    return "far"


def funds_bucket(funds: int) -> str:
    if funds < 500:
        return "broke"
    if funds < 1500:
        return "tight"
    if funds < 4000:
        return "ok"
    return "rich"


def _uid(d: dict | None) -> int | None:
    """Actor id across server shapes: live summaries use `id`, proto models use `actor_id`."""
    if not d:
        return None
    for k in ("actor_id", "id"):
        try:
            if d.get(k) is not None:
                return int(d[k])
        except (TypeError, ValueError):
            continue
    return None


def _hp(d: dict) -> float:
    """HP fraction; live fix5 summaries omit it -> default 1.0 (healthy)."""
    try:
        return float(d.get("hp_percent", 1.0) or 0.0)
    except (TypeError, ValueError):
        return 1.0


def chebyshev(ax: int, ay: int, bx: int, by: int) -> int:
    return max(abs(ax - bx), abs(ay - by))


@dataclass
class Snapshot:
    """Merged live snapshot from get_game_state + get_units + get_buildings."""

    tick: int = 0
    map_name: str = ""
    cash: int = 0
    ore: int = 0
    power_balance: int = 0
    harvester_count: int = 0
    kd_ratio: float = 0.0
    army_value: int = 0
    kills: int = 0
    losses: int = 0
    kills_cost: int = 0
    deaths_cost: int = 0
    order_count: int = 0
    own_units: list = field(default_factory=list)  # raw dicts
    own_buildings: list = field(default_factory=list)
    enemies: list = field(default_factory=list)
    enemy_buildings: list = field(default_factory=list)
    production: list = field(default_factory=list)
    available_production: list = field(default_factory=list)
    # derived
    base_cell: tuple = (0, 0)
    enemy_centroid: tuple = (0, 0)
    mcv_id: int | None = None
    has_fact: bool = False
    weak_own_ids: list = field(default_factory=list)
    nearest_enemy: dict | None = None
    weakest_enemy: dict | None = None
    explored_percent: float = 0.0
    reward_vector: dict = field(default_factory=dict)
    ready_to_place: list = field(default_factory=list)


def _num(d: dict, key: str, default: int = 0) -> int:
    try:
        return int(d.get(key, default) or 0)
    except (TypeError, ValueError):
        return default


def _as_list(v):
    """Server sometimes returns a scalar (e.g. production: 0/1) instead of a
    list; coerce defensively so one odd field never kills a full game."""
    if isinstance(v, list):
        return v
    if isinstance(v, tuple):
        return list(v)
    return []


def build_snapshot(game_state: dict, units: list | None = None, buildings: list | None = None) -> Snapshot:
    """Merge tool outputs into a Snapshot. Missing keys degrade gracefully."""
    gs = game_state or {}
    snap = Snapshot()
    snap.tick = _num(gs, "tick")
    mp = gs.get("map", {}) or {}
    snap.map_name = str(gs.get("map_name", "") or mp.get("map_name", ""))
    eco = gs.get("economy", {}) or {}
    snap.cash = _num(eco, "cash")
    snap.ore = _num(eco, "ore")
    snap.power_balance = _num(eco, "power_provided") - _num(eco, "power_drained")
    snap.harvester_count = _num(eco, "harvester_count")
    mil = gs.get("military", {}) or {}
    snap.kills = _num(mil, "units_killed")
    snap.losses = _num(mil, "units_lost")
    snap.kills_cost = _num(mil, "kills_cost")
    snap.deaths_cost = _num(mil, "deaths_cost")
    snap.army_value = _num(mil, "army_value")
    snap.order_count = _num(mil, "order_count")
    snap.kd_ratio = round(snap.kills_cost / max(snap.deaths_cost, 1), 2)

    if units is not None:
        snap.own_units = _as_list(units)
    else:
        snap.own_units = _as_list(gs.get("units_summary", []) or gs.get("units", []) or [])
    if buildings is not None:
        snap.own_buildings = _as_list(buildings)
    else:
        snap.own_buildings = _as_list(gs.get("buildings_summary", []) or gs.get("buildings", []) or [])
    snap.enemies = _as_list(gs.get("enemy_summary", []) or gs.get("visible_enemies", []) or [])
    snap.enemy_buildings = _as_list(
        gs.get("enemy_buildings_summary", []) or gs.get("visible_enemy_buildings", []) or [])
    snap.production = _as_list(
        gs.get("production", []) or gs.get("production_queues", []) or
        gs.get("production_items", []) or [])
    snap.available_production = _as_list(gs.get("available_production", []) or [])
    try:
        snap.explored_percent = float(gs.get("explored_percent", 0.0) or 0.0)
    except (TypeError, ValueError):
        snap.explored_percent = 0.0
    snap.reward_vector = dict(gs.get("reward_vector", {}) or {})

    facts = [b for b in snap.own_units + snap.own_buildings if b.get("type") == "fact"]
    snap.has_fact = len([b for b in snap.own_buildings if b.get("type") == "fact"]) > 0
    mcv = next((u for u in snap.own_units if u.get("type") == "mcv"), None)
    snap.mcv_id = _uid(mcv) if mcv is not None else None
    if facts:
        f0 = facts[0]
        snap.base_cell = (int(f0.get("cell_x", 0) or 0), int(f0.get("cell_y", 0) or 0))
    elif mcv is not None:
        snap.base_cell = (int(mcv.get("cell_x", 0) or 0), int(mcv.get("cell_y", 0) or 0))

    if snap.enemies:
        cx = sum(int(e.get("cell_x", 0) or 0) for e in snap.enemies) // len(snap.enemies)
        cy = sum(int(e.get("cell_y", 0) or 0) for e in snap.enemies) // len(snap.enemies)
        snap.enemy_centroid = (cx, cy)
        snap.nearest_enemy = min(
            snap.enemies,
            key=lambda e: chebyshev(e.get("cell_x", 0) or 0, e.get("cell_y", 0) or 0,
                                    snap.base_cell[0], snap.base_cell[1]),
        )
        snap.weakest_enemy = min(snap.enemies, key=_hp)

    snap.weak_own_ids = [
        _uid(u) for u in snap.own_units
        if u.get("type") in COMBAT_UNIT_TYPES and _hp(u) < 0.33
    ]
    snap.weak_own_ids = [i for i in snap.weak_own_ids if i is not None]
    snap.ready_to_place = [
        p.get("item", "") for p in snap.production
        if p.get("queue_type") == "Building" and float(p.get("progress", 0.0) or 0.0) >= 0.99
    ]
    return snap


def _unit_entry(u: dict, ref_cell: tuple) -> dict:
    cx, cy = int(u.get("cell_x", 0) or 0), int(u.get("cell_y", 0) or 0)
    return {
        "id": _uid(u),
        "type": str(u.get("type", "?")),
        "cell": [cx, cy],
        "hp": hp_bucket(_hp(u)),
        "dist": dist_bucket(chebyshev(cx, cy, ref_cell[0], ref_cell[1])),
    }


def build_combat_state(snap: Snapshot, max_own: int = 12, max_enemy: int = 12) -> dict:
    """Local fight state: own combat units + visible enemies + flags."""
    combat = [u for u in snap.own_units if u.get("type") in COMBAT_UNIT_TYPES][:max_own]
    foes = snap.enemies[:max_enemy]
    under_attack = any(
        chebyshev(int(e.get("cell_x", 0) or 0), int(e.get("cell_y", 0) or 0),
                  snap.base_cell[0], snap.base_cell[1]) <= 12
        for e in foes
    ) if foes else False
    return {
        "kind": "combat",
        "tick": snap.tick,
        "map": snap.map_name,
        "funds": funds_bucket(snap.cash + snap.ore),
        "power": snap.power_balance,
        "kd": snap.kd_ratio,
        "base": list(snap.base_cell),
        "own": [_unit_entry(u, snap.base_cell) for u in combat],
        "enemies": [_unit_entry(e, snap.base_cell) for e in foes],
        "enemy_buildings_seen": len(snap.enemy_buildings),
        "under_attack": under_attack,
        "weak_ids": snap.weak_own_ids[:6],
        "mcv_undeployed": snap.mcv_id if not snap.has_fact else None,
    }


def build_eco_state(snap: Snapshot) -> dict:
    """Global economy state: cash/ore/power + buildings + production."""
    return {
        "kind": "eco",
        "tick": snap.tick,
        "map": snap.map_name,
        "cash": snap.cash,
        "ore": snap.ore,
        "funds": funds_bucket(snap.cash + snap.ore),
        "power": snap.power_balance,
        "harvesters": snap.harvester_count,
        "buildings": [
            {"id": int(b.get("actor_id", 0) or 0), "type": str(b.get("type", "?")),
             "hp": hp_bucket(float(b.get("hp_percent", 1.0) or 0.0)),
             "producing": b.get("producing_item", "") or ""}
            for b in snap.own_buildings
        ],
        "production": [
            {"queue": p.get("queue_type", ""), "item": p.get("item", ""),
             "progress": round(float(p.get("progress", 0.0) or 0.0), 2)}
            for p in snap.production
        ],
        "can_make": snap.available_production[:20],
        "ready_to_place": snap.ready_to_place,
        "mcv_undeployed": snap.mcv_id if not snap.has_fact else None,
    }


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def state_to_json(state: dict) -> tuple[str, int]:
    """Serialize with budget enforcement (trim lists if over budget)."""
    text = json.dumps(state, separators=(",", ":"))
    toks = estimate_tokens(text)
    if toks <= STATE_TOKEN_BUDGET:
        return text, toks
    trimmed = dict(state)
    for key in ("enemies", "own", "can_make", "production"):
        if isinstance(trimmed.get(key), list) and len(trimmed[key]) > 4:
            trimmed[key] = trimmed[key][:4]
    text = json.dumps(trimmed, separators=(",", ":"))
    return text, estimate_tokens(text)
