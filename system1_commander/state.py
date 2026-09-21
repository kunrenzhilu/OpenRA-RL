"""Observation -> compact English JSON state (<=800 tokens).

All arithmetic is done here, never by the model (Jev arithmetic is
unreliable): power balance, K/D cost ratio, funds buckets, HP buckets,
Chebyshev distance buckets. Output is standard JSON (json.dumps).
"""

from __future__ import annotations

import json
import re
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
    # FIX-#4 hooks: derived bools so the brain does literal matching
    # instead of parsing the production list itself.
    queue_blocked_by_unplaced: bool = False
    building_in_progress: bool = False
    # Jev-max J1: raw ASCII minimap from get_game_state ("minimap" key,
    # 23x23 server-side). Carried verbatim; capped at attach time.
    minimap: str = ""


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


# Live `production_items` entries carry no queue_type, so unit-vs-structure is
# inferred from the item name. Mislabels are safe: trained units never sit at
# 100% awaiting placement, and place_building re-validates readiness
# server-side (a spurious place just fails the batch).
_UNIT_QUEUE_ITEMS = COMBAT_UNIT_TYPES | {"e6", "medic", "dog", "harv", "mcv"}

_PROD_ITEM_RE = re.compile(r"^(?P<item>.+?)@(?P<pct>\d+(?:\.\d+)?)%")


def _parse_production_items(items) -> list[dict]:
    """Parse live `production_items` strings ("powr@14%(~155 ticks)") into
    {queue_type, item, progress} dicts (progress as 0.0-1.0 fraction)."""
    out = []
    for s in _as_list(items):
        if isinstance(s, dict):
            out.append(s)
            continue
        m = _PROD_ITEM_RE.match(str(s))
        if not m:
            continue
        name = m.group("item")
        try:
            prog = float(m.group("pct")) / 100.0
        except (TypeError, ValueError):
            continue
        out.append({
            "queue_type": "Unit" if name in _UNIT_QUEUE_ITEMS else "Building",
            "item": name,
            "progress": prog,
        })
    return out


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
    _raw_prod = gs.get("production", [])
    if isinstance(_raw_prod, list) and any(isinstance(p, dict) for p in _raw_prod):
        snap.production = [p for p in _raw_prod if isinstance(p, dict)]
    else:
        # Live server shape: no `production` key; `production_queues` is an
        # int count (truthy when busy — never use it as the list) and
        # `production_items` are "item@NN%(~N ticks)" strings (server:
        # openra_environment.py get_game_state summary).
        snap.production = _parse_production_items(gs.get("production_items", []))
    snap.available_production = _as_list(gs.get("available_production", []) or [])
    try:
        snap.explored_percent = float(gs.get("explored_percent", 0.0) or 0.0)
    except (TypeError, ValueError):
        snap.explored_percent = 0.0
    snap.reward_vector = dict(gs.get("reward_vector", {}) or {})
    _mm = gs.get("minimap", "")
    snap.minimap = _mm if isinstance(_mm, str) else ""

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
    # FIX-#4 derived hooks (same sources as FIX-#1/#2 in demo_loop.py).
    snap.queue_blocked_by_unplaced = bool(snap.ready_to_place)
    def _prog(p: dict) -> float:
        try:
            return float(p.get("progress", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0
    snap.building_in_progress = any(
        p.get("queue_type") == "Building" and _prog(p) < 0.99
        for p in snap.production
    )
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
        "queue_blocked_by_unplaced": snap.queue_blocked_by_unplaced,
        "building_in_progress": snap.building_in_progress,
        "mcv_undeployed": snap.mcv_id if not snap.has_fact else None,
    }


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def state_to_json(state: dict, budget: int = STATE_TOKEN_BUDGET) -> tuple[str, int]:
    """Serialize with budget enforcement (trim lists if over budget).

    Jev-max J1: budget is a parameter (plan §1 state ladder 800 -> 2k -> 4k);
    default keeps the legacy 800. Rich keys trim first (minimap head-cut,
    enemy_history to 2 rows) so the poor-state core survives.
    """
    text = json.dumps(state, separators=(",", ":"))
    toks = estimate_tokens(text)
    if toks <= budget:
        return text, toks
    trimmed = dict(state)
    for key in ("enemies", "own", "can_make", "production"):
        if isinstance(trimmed.get(key), list) and len(trimmed[key]) > 4:
            trimmed[key] = trimmed[key][:4]
    if isinstance(trimmed.get("minimap"), str):
        trimmed["minimap"] = trimmed["minimap"][:400]
    _eh = trimmed.get("enemy_history")
    if isinstance(_eh, dict) and isinstance(_eh.get("last_seen"), list):
        _eh = dict(_eh)
        _eh["last_seen"] = _eh["last_seen"][:2]
        trimmed["enemy_history"] = _eh
    text = json.dumps(trimmed, separators=(",", ":"))
    return text, estimate_tokens(text)


# ── Jev-max J1: rich-state fields (backend-agnostic; FROZEN set of five) ──
# plan openra-taskC-jevmax-plan-20260921 §1. All arithmetic stays Python-side.
# Field set is FROZEN (minerals / enemy_history / traj / phase / minimap);
# do not add/remove. Over-cap lists trim OLDEST first.
RICH_TOKEN_CAPS = {
    "minerals": 120,
    "enemy_history": 150,
    "traj": 200,
    "phase": 60,
    "minimap": 200,
}


def _cap_text(text: str, cap_toks: int) -> str:
    return text[: max(0, cap_toks * 4)]


def _fits(obj: Any, cap_toks: int) -> bool:
    return estimate_tokens(json.dumps(obj, separators=(",", ":"))) <= cap_toks


def _trim_lists_oldest(obj: dict, list_keys: list[str], cap_toks: int) -> dict:
    """Drop oldest (front) list items until the dict fits the token cap.

    Convention: trimmed lists are stored oldest-first. At most one item per
    key per pass, round-robin, so no single key is gutted first.
    """
    obj = dict(obj)
    for k in list_keys:
        if isinstance(obj.get(k), list):
            obj[k] = list(obj[k])
    while not _fits(obj, cap_toks):
        dropped = False
        for k in list_keys:
            if isinstance(obj.get(k), list) and len(obj[k]) > 1:
                obj[k] = obj[k][1:]
                dropped = True
        if not dropped:
            break
    return obj


def _dir8(dx: int, dy: int) -> str:
    """Coarse compass bucket of an offset (screen coords, y grows south)."""
    if dx == 0 and dy == 0:
        return "here"
    ax, ay = abs(dx), abs(dy)
    if ax >= 2 * ay:
        return "E" if dx > 0 else "W"
    if ay >= 2 * ax:
        return "S" if dy > 0 else "N"
    if dx > 0:
        return "SE" if dy > 0 else "NE"
    return "SW" if dy > 0 else "NW"


class EnemyMemory:
    """Last-seen record per enemy type (out of sight != never existed)."""

    def __init__(self) -> None:
        self._seen: dict[str, dict] = {}

    def reset(self) -> None:
        self._seen.clear()

    def observe(self, snap: Snapshot) -> None:
        by_type: dict[str, list] = {}
        for e in snap.enemies:
            by_type.setdefault(str(e.get("type", "?")), []).append(e)
        for t, es in by_type.items():
            # Representative cell: closest to our base (the sharp end).
            rep = min(es, key=lambda e: chebyshev(
                int(e.get("cell_x", 0) or 0), int(e.get("cell_y", 0) or 0),
                snap.base_cell[0], snap.base_cell[1]))
            self._seen[t] = {
                "cell": (int(rep.get("cell_x", 0) or 0),
                         int(rep.get("cell_y", 0) or 0)),
                "tick": snap.tick,
                "count": len(es),
            }

    def summarize(self, base_cell: tuple, now_tick: int) -> dict:
        rows = []
        for t, rec in self._seen.items():
            cx, cy = rec["cell"]
            dx, dy = cx - base_cell[0], cy - base_cell[1]
            rows.append({
                "t": t,
                "dir": _dir8(dx, dy),
                "dist": dist_bucket(chebyshev(cx, cy, base_cell[0], base_cell[1])),
                "ago": now_tick - int(rec["tick"]),
                "n": int(rec["count"]),
                "_tick": int(rec["tick"]),
            })
        # Oldest-first for trim-oldest, then newest-first for the model.
        rows.sort(key=lambda r: r["_tick"])
        out: dict = {"last_seen": rows, "n_types": len(rows)}
        out = _trim_lists_oldest(out, ["last_seen"],
                                 RICH_TOKEN_CAPS["enemy_history"])
        for r in out["last_seen"]:
            r.pop("_tick", None)
        out["last_seen"] = out["last_seen"][::-1]
        out["n_types"] = len(out["last_seen"])
        return out


class TrajectoryTracker:
    """Cumulative own-side trajectory: built-ever, cash curve, K/D, power."""

    def __init__(self) -> None:
        self.built: dict[str, int] = {}
        self.cash_pts: list[int] = []
        self.kills = 0
        self.losses = 0
        self.power_pts: list[int] = []
        self.powr_n = 0

    def reset(self) -> None:
        self.__init__()

    def observe(self, snap: Snapshot) -> None:
        counts: dict[str, int] = {}
        for b in snap.own_buildings:
            t = str(b.get("type", "?"))
            counts[t] = counts.get(t, 0) + 1
        for t, c in counts.items():
            self.built[t] = max(self.built.get(t, 0), c)
        self.cash_pts.append(int(snap.cash) + int(snap.ore))
        self.kills = int(snap.kills)
        self.losses = int(snap.losses)
        self.power_pts.append(int(snap.power_balance))
        self.powr_n = counts.get("powr", 0)

    def cash_curve(self, n: int = 5) -> list[int]:
        pts = self.cash_pts
        if not pts:
            return []
        if len(pts) <= n:
            return list(pts)
        idx = [round(i * (len(pts) - 1) / (n - 1)) for i in range(n)]
        return [pts[j] for j in idx]

    def power_trend(self) -> str:
        pts = self.power_pts[-4:]
        if len(pts) < 2:
            return "flat"
        d = pts[-1] - pts[0]
        if d > 10:
            return "up"
        if d < -10:
            return "down"
        return "flat"

    def summarize_traj(self) -> dict:
        out: dict = {
            "built": dict(sorted(self.built.items())),
            "cash_curve": self.cash_curve(5),
            "kills": self.kills,
            "losses": self.losses,
        }
        # `built` has no age order; cash_curve is newest-valuable. Trim the
        # curve from the front (oldest samples) if absurdly over cap.
        return _trim_lists_oldest(out, ["cash_curve"],
                                  RICH_TOKEN_CAPS["traj"])

    def summarize_phase(self, tick: int) -> dict:
        if tick < 3000:
            stage = "early"
        elif tick < 9000:
            stage = "mid"
        else:
            stage = "late"
        return {
            "stage": stage,
            "power_trend": self.power_trend(),
            "powr": self.powr_n,
            "tick": tick,
        }


def build_minerals(snap: Snapshot) -> dict:
    """Ore picture. The server exposes no ore-field channel, so harvester
    positions proxy the fields (harvesters sit on ore); offsets are
    base-relative cells. Honest about the proxy in `src`."""
    bx, by = snap.base_cell
    harvs = [u for u in snap.own_units if u.get("type") == "harv"]
    rel = [[int(u.get("cell_x", 0) or 0) - bx,
            int(u.get("cell_y", 0) or 0) - by] for u in harvs[:4]]
    nearest = None
    if harvs:
        h0 = min(harvs, key=lambda u: chebyshev(
            int(u.get("cell_x", 0) or 0), int(u.get("cell_y", 0) or 0),
            bx, by))
        hx, hy = int(h0.get("cell_x", 0) or 0), int(h0.get("cell_y", 0) or 0)
        nearest = {"dx": hx - bx, "dy": hy - by,
                   "dist": dist_bucket(chebyshev(hx, hy, bx, by))}
    return {
        "ore": int(snap.ore),
        "harv_n": len(harvs),
        "harv_rel": rel,
        "nearest": nearest,
        "src": "harv-pos-proxy",
    }


# Module-singleton trackers: one game per process; demo_loop resets per run.
_ENEMY_MEM = EnemyMemory()
_TRAJ = TrajectoryTracker()


def observe_trackers(snap: Snapshot) -> None:
    """Feed one fresh snapshot into the J1 trackers (call per decision)."""
    _ENEMY_MEM.observe(snap)
    _TRAJ.observe(snap)


def reset_trackers() -> None:
    _ENEMY_MEM.reset()
    _TRAJ.reset()


def attach_rich_fields(state: dict, snap: Snapshot) -> dict:
    """Return a copy of `state` plus the frozen five J1 rich fields.

    Backend-agnostic: pure function of (poor state, snapshot, trackers).
    Each field respects its RICH_TOKEN_CAPS entry (oldest trimmed).
    """
    out = dict(state)
    out["minerals"] = _trim_lists_oldest(
        build_minerals(snap), ["harv_rel"], RICH_TOKEN_CAPS["minerals"])
    out["enemy_history"] = _ENEMY_MEM.summarize(snap.base_cell, snap.tick)
    out["traj"] = _TRAJ.summarize_traj()
    out["phase"] = _TRAJ.summarize_phase(snap.tick)
    out["minimap"] = _cap_text(snap.minimap or "",
                               RICH_TOKEN_CAPS["minimap"])
    return out
