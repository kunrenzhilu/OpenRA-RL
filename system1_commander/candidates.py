"""Legal candidate sets. Code enumerates options; the model only picks one.

Batch action dicts use MCP tool format: {"tool": <name>, params...}.
`*_id` params are ints; `unit_ids` is a string selector ("all_combat",
"all_idle", "id1,id2", group name). An empty action list means
"advance time only" (used by `wait`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from system1_commander.state import COMBAT_UNIT_TYPES, Snapshot, _uid


@dataclass
class Candidate:
    name: str
    kind: str  # "combat" | "eco"
    description: str  # one English sentence for the Choice criteria
    build: Callable[[Snapshot], list[dict]] = field(repr=False)
    destructive: bool = False

    def actions(self, snap: Snapshot) -> list[dict]:
        return self.build(snap)


def _choke(snap: Snapshot) -> list[int]:
    if snap.enemies:
        return [(snap.base_cell[0] + snap.enemy_centroid[0]) // 2,
                (snap.base_cell[1] + snap.enemy_centroid[1]) // 2]
    return list(snap.base_cell)


def _own_combat(snap: Snapshot) -> bool:
    return any(u.get("type") in COMBAT_UNIT_TYPES for u in snap.own_units)


def _combat_or_empty(names_actions: list[dict]) -> list[dict]:
    return names_actions


# ── combat builders ──────────────────────────────────────────────

def _attack_nearest(snap: Snapshot) -> list[dict]:
    if snap.nearest_enemy is None or not _own_combat(snap):
        return []
    return [{"tool": "attack_target", "unit_ids": "all_combat",
             "target_actor_id": _uid(snap.nearest_enemy)}]


def _focus_weakest(snap: Snapshot) -> list[dict]:
    if snap.weakest_enemy is None or not _own_combat(snap):
        return []
    return [{"tool": "attack_target", "unit_ids": "all_combat",
             "target_actor_id": _uid(snap.weakest_enemy)}]


def _pull_back_weak(snap: Snapshot) -> list[dict]:
    if not snap.weak_own_ids:
        return []
    ids = ",".join(str(i) for i in snap.weak_own_ids[:6])
    return [{"tool": "move_units", "unit_ids": ids,
             "target_x": snap.base_cell[0], "target_y": snap.base_cell[1]}]


def _guard_choke(snap: Snapshot) -> list[dict]:
    if not _own_combat(snap):
        return []
    choke = _choke(snap)
    return [
        {"tool": "move_units", "unit_ids": "all_combat",
         "target_x": choke[0], "target_y": choke[1]},
        {"tool": "set_stance", "unit_ids": "all_combat", "stance": "defend"},
    ]


def _all_combat_attack_move(snap: Snapshot) -> list[dict]:
    if not _own_combat(snap):
        return []
    tx, ty = snap.enemy_centroid if snap.enemies else (46, 46)
    return [{"tool": "attack_move", "unit_ids": "all_combat",
             "target_x": int(tx), "target_y": int(ty)}]


def _hold_position(snap: Snapshot) -> list[dict]:
    if not _own_combat(snap):
        return []
    return _combat_or_empty([{"tool": "stop_units", "unit_ids": "all_combat"}])


def list_combat_candidates() -> list[Candidate]:
    return [
        Candidate("attack_nearest", "combat",
                  "Order all combat units to attack the enemy closest to our base.",
                  _attack_nearest),
        Candidate("focus_weakest", "combat",
                  "Focus all combat units fire onto the most damaged visible enemy.",
                  _focus_weakest),
        Candidate("pull_back_weak", "combat",
                  "Pull critically damaged friendlies back to the base to survive.",
                  _pull_back_weak),
        Candidate("guard_choke", "combat",
                  "Hold a defensive position between our base and the enemy in defend stance.",
                  _guard_choke),
        Candidate("all_combat_attack_move", "combat",
                  "Advance all combat units toward the enemy centroid, engaging on contact.",
                  _all_combat_attack_move),
        Candidate("hold_position", "combat",
                  "Stop all combat units and hold the current position.",
                  _hold_position),
    ]


# ── eco builders ─────────────────────────────────────────────────

def _deploy_mcv(snap: Snapshot) -> list[dict]:
    if snap.mcv_id is None or snap.has_fact:
        return []
    return [{"tool": "deploy_unit", "unit_id": int(snap.mcv_id)}]


def _build_structure(btype: str, desc: str) -> Candidate:
    return Candidate(
        f"build_{btype}", "eco", desc,
        lambda snap, b=btype: [{"tool": "build_structure", "building_type": b}],
    )


def _train_unit(utype: str, desc: str) -> Candidate:
    return Candidate(
        f"train_{utype}", "eco", desc,
        lambda snap, u=utype: [{"tool": "build_unit", "unit_type": u, "count": 1}],
    )


def _place_ready(snap: Snapshot) -> list[dict]:
    if not snap.ready_to_place:
        return []
    btype = snap.ready_to_place[0]
    bx, by = snap.base_cell
    return [{"tool": "place_building", "building_type": btype,
             "cell_x": bx + 3, "cell_y": by}]


def list_eco_candidates() -> list[Candidate]:
    return [
        Candidate("deploy_mcv", "eco",
                  "Deploy the MCV into a Construction Yard to unlock the base.",
                  _deploy_mcv),
        _build_structure("powr", "Start a Power Plant to raise the power balance."),
        _build_structure("barr", "Start infantry production structure (faction default)."),
        _build_structure("weap", "Start a War Factory to unlock vehicles."),
        _train_unit("e1", "Train one rifle infantry for defense and scouting."),
        _train_unit("harv", "Train one harvester to raise ore income."),
        _train_unit("1tnk", "Train one light tank for the attack force."),
        Candidate("place_ready", "eco",
                  "Place a finished building from the queue next to the base.",
                  _place_ready),
        Candidate("wait", "eco",
                  "Issue no order and let production and construction progress.",
                  lambda snap: []),
    ]
