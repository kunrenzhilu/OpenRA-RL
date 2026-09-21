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


def list_combat_candidates() -> list[Candidate]:    return [
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


# ── A-track (Laya plan §A1): executable prefilter + skip schema ─────────
# Predicates are same-source as the builders' `[]` conditions above; no new
# semantics are invented here. `wait` / `deploy_mcv` are never filtered.

def exec_flags(snap: Snapshot) -> dict:
    """Predicate-input snapshot for judge recount (Laya plan A1/A3).

    `queue_nonempty` is the `place_ready` predicate input (ready-to-place
    queue non-empty), so the judge can recount the ballot with the
    `candidates.py` predicates without trusting the executor's report.
    """
    return {
        "queue_nonempty": bool(snap.ready_to_place),
        "own_combat": _own_combat(snap),
        "nearest_enemy": snap.nearest_enemy is not None,
        "weakest_enemy": snap.weakest_enemy is not None,
        "weak_own": bool(snap.weak_own_ids),
    }


def list_executable_candidates(candidates: list[Candidate],
                               snap: Snapshot) -> list[Candidate]:
    """Drop candidates whose builder would return `[]` for this snapshot."""
    flags = exec_flags(snap)
    out: list[Candidate] = []
    for c in candidates:
        if c.kind == "eco" and c.name == "place_ready":
            if not flags["queue_nonempty"]:
                continue
        elif c.kind == "combat":
            if c.name == "attack_nearest":
                if not (flags["own_combat"] and flags["nearest_enemy"]):
                    continue
            elif c.name == "focus_weakest":
                if not (flags["own_combat"] and flags["weakest_enemy"]):
                    continue
            elif c.name in ("guard_choke", "all_combat_attack_move",
                            "hold_position"):
                if not flags["own_combat"]:
                    continue
            elif c.name == "pull_back_weak":
                if not flags["weak_own"]:
                    continue
        out.append(c)
    return out


def make_skip_entry(*, i: int, tick: int, state_kind: str, state_tokens: int,
                    flags: dict, advance_ticks: int = 0,
                    advance_interrupted: bool = False) -> dict:
    """Schema-frozen skip entry (Laya plan A1): empty ballot after prefilter.

    No `prediction`/`confidence`/`probs` keys by design; `gate.mode` is the
    fixed `skip-empty-ballot` value outside the execute/downgrade/fallback
    domain so bench aggregation must list it separately.
    """
    return {
        "kind": "skip",
        "i": i,
        "tick": tick,
        "state_kind": state_kind,
        "state_tokens": state_tokens,
        "ballot": [],
        "exec_flags": dict(flags),
        "gate": {"mode": "skip-empty-ballot", "choice": "none",
                 "reason": "ballot empty after prefilter"},
        "advance_ticks": advance_ticks,
        "advance_interrupted": advance_interrupted,
    }


# ── Jev-max J2: macro layer (options framework; plan §2, FULL table) ─────
# Five macros, copied verbatim from the plan (a macro set WITHOUT fast_weap
# is disqualified: it cuts the War Factory chain = fix3 harv absence again).
# Each macro maps onto the existing executor.batch + advance chain as a
# multi-step script; the builder below returns the FIRST currently-legal
# step (place-first, then build/train, else [] = wait). Descriptions are
# the phase-1 "phase-goal manual" (soft-replaces iron rule 2; the iron
# backup itself stays in backend_jev). Backend-agnostic: plain Candidates.
MACRO_NAMES = ("open_powr", "rush_barr", "fast_weap", "econ_harv", "armor_push")


def _building_in_queue(snap: Snapshot) -> bool:
    return any(p.get("queue_type") == "Building" for p in (snap.production or []))


def _has_placed(snap: Snapshot, btype: str) -> bool:
    return any(b.get("type") == btype for b in snap.own_buildings)


def _macro_open_powr(snap: Snapshot) -> list[dict]:
    """open_powr: powr opening (baseline) -> build_powr + place on arrival."""
    if snap.ready_to_place:
        return _place_ready(snap)
    if _building_in_queue(snap):
        return []
    return [{"tool": "build_structure", "building_type": "powr"}]


def _macro_rush_barr(snap: Snapshot) -> list[dict]:
    """rush_barr: forward Barracks -> build_barr + place + train_e1 x N."""
    if snap.ready_to_place:
        return _place_ready(snap)
    if _building_in_queue(snap):
        if _has_placed(snap, "barr"):
            return [{"tool": "build_unit", "unit_type": "e1", "count": 1}]
        return []
    if _has_placed(snap, "barr"):
        return [{"tool": "build_unit", "unit_type": "e1", "count": 1}]
    return [{"tool": "build_structure", "building_type": "barr"}]


def _macro_fast_weap(snap: Snapshot) -> list[dict]:
    """fast_weap: tech rush -> build_weap + place (SOLE harv/tank prereq)."""
    if snap.ready_to_place:
        return _place_ready(snap)
    if _building_in_queue(snap):
        return []
    return [{"tool": "build_structure", "building_type": "weap"}]


def _macro_econ_harv(snap: Snapshot) -> list[dict]:
    """econ_harv: grow income -> train_harv (prefiltered by can_make)."""
    if "harv" not in {str(x) for x in (snap.available_production or [])}:
        return []
    return [{"tool": "build_unit", "unit_type": "harv", "count": 1}]


def _macro_armor_push(snap: Snapshot) -> list[dict]:
    """armor_push: tank army -> train_1tnk x N (prefiltered by can_make)."""
    if "1tnk" not in {str(x) for x in (snap.available_production or [])}:
        return []
    return [{"tool": "build_unit", "unit_type": "1tnk", "count": 1}]


def list_macro_candidates() -> list[Candidate]:
    return [
        Candidate("open_powr", "eco",
                  "Opening: secure power first - start a Power Plant, "
                  "or place the finished building to unblock the queue.",
                  _macro_open_powr),
        Candidate("rush_barr", "eco",
                  "Early pressure: raise Barracks, place it, then train "
                  "rifle infantry to contest the map.",
                  _macro_rush_barr),
        Candidate("fast_weap", "eco",
                  "Tech rush: raise a War Factory to unlock harvesters "
                  "and tanks (the only path to vehicle income).",
                  _macro_fast_weap),
        Candidate("econ_harv", "eco",
                  "Economy: train a harvester to grow ore income "
                  "(only when harvesters are producible).",
                  _macro_econ_harv),
        Candidate("armor_push", "eco",
                  "Armor: mass light tanks for a decisive push "
                  "(only when tanks are producible).",
                  _macro_armor_push),
    ]


def list_executable_macros(macros: list[Candidate],
                           snap: Snapshot) -> list[Candidate]:
    """Drop macros whose precondition fails this tick.

    Same loop+continue shape as list_executable_candidates: econ_harv
    needs harv in can_make, armor_push needs 1tnk; the three structural
    macros always stay (their builders degrade to place/wait steps).
    """
    can = {str(x) for x in (snap.available_production or [])}
    out: list[Candidate] = []
    for c in macros:
        if c.name == "econ_harv":
            if "harv" not in can:
                continue
        elif c.name == "armor_push":
            if "1tnk" not in can:
                continue
        out.append(c)
    return out
