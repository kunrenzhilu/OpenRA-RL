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


def _split_harass(snap: Snapshot) -> list[dict]:
    """Jevfix F5: half the force hits the nearest enemy, half holds."""
    if snap.nearest_enemy is None or not _own_combat(snap):
        return []
    ids = [str(_uid(u)) for u in snap.own_units
           if u.get("type") in COMBAT_UNIT_TYPES and _uid(u) is not None]
    if not ids:
        return []
    k = max(1, len(ids) // 2)
    atk, hold = ids[:k], ids[k:]
    out = [{"tool": "attack_target", "unit_ids": ",".join(atk),
            "target_actor_id": _uid(snap.nearest_enemy)}]
    if hold:
        out.append({"tool": "stop_units", "unit_ids": ",".join(hold)})
    return out


def _screen_mcv(snap: Snapshot) -> list[dict]:
    """Jevfix F5: whole force falls back to the base circle."""
    if not _own_combat(snap):
        return []
    return [{"tool": "move_units", "unit_ids": "all_combat",
             "target_x": snap.base_cell[0], "target_y": snap.base_cell[1]}]


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
        Candidate("split_harass", "combat",
                  "Send half the force to hit the nearest enemy while the rest holds.",
                  _split_harass),
        Candidate("screen_mcv", "combat",
                  "Pull the whole force back to screen the base.",
                  _screen_mcv),
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


# ── Jevfix F3: directional place variants ────────────────────────────
# `place_ready` splits into 4 voted directions so placement intent is
# expressible on the ballot. Offsets follow the repo screen-coords
# convention (y grows south, cf. state._dir8): N=(0,-3), E=(3,0),
# S=(0,3), W=(-3,0). Admission = ready_to_place non-empty (same guard as
# place_ready, enforced in list_executable_candidates). The legacy
# place_ready (base+(3,0) = E) stays for the harness guards (FIX-#1,
# place-first); the ballot carries all five when unblocked.
PLACE_DIRS = (("N", 0, -3), ("E", 3, 0), ("S", 0, 3), ("W", -3, 0))


def _place_dir(dx: int, dy: int):
    def _build(snap: Snapshot) -> list[dict]:
        if not snap.ready_to_place:
            return []
        btype = snap.ready_to_place[0]
        bx, by = snap.base_cell
        return [{"tool": "place_building", "building_type": btype,
                 "cell_x": bx + dx, "cell_y": by + dy}]
    return _build


def list_place_variants() -> list[Candidate]:
    return [
        Candidate(f"place_{d}", "eco",
                  f"Place the finished building {d} of the base "
                  f"(offset {dx},{dy} from the Construction Yard).",
                  _place_dir(dx, dy))
        for d, dx, dy in PLACE_DIRS
    ]


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
        *list_place_variants(),
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
    Key set is FROZEN (atrack test pins it); Jevfix F2 recounts train_*
    admission from the per-row `ballot` + the state's `can_make` instead.
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
    """Drop candidates whose builder would return `[]` for this snapshot.

    Jevfix F2: `train_*` admission moved out of the model instructions
    into this harness prefilter (a unit is votable only when listed in
    `available_production`; empty `can_make` admits no train_* at all).
    """
    flags = exec_flags(snap)
    can = {str(x) for x in (snap.available_production or [])}
    out: list[Candidate] = []
    for c in candidates:
        if c.kind == "eco" and c.name == "place_ready":
            if not flags["queue_nonempty"]:
                continue
        elif c.kind == "eco" and c.name.startswith("place_"):
            # Jevfix F3: directional place variants share the place guard.
            if not flags["queue_nonempty"]:
                continue
        elif c.kind == "eco" and c.name.startswith("train_"):
            if c.name[len("train_"):] not in can:
                continue
        elif c.kind == "combat":
            if c.name == "attack_nearest":
                if not (flags["own_combat"] and flags["nearest_enemy"]):
                    continue
            elif c.name == "focus_weakest":
                if not (flags["own_combat"] and flags["weakest_enemy"]):
                    continue
            elif c.name in ("guard_choke", "all_combat_attack_move",
                            "hold_position", "screen_mcv"):
                if not flags["own_combat"]:
                    continue
            elif c.name in ("pull_back_weak",):
                if not flags["weak_own"]:
                    continue
            elif c.name == "split_harass":
                # Jevfix F5: split needs a target like attack_nearest.
                if not (flags["own_combat"] and flags["nearest_enemy"]):
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


# ── Jevfix F5: combat macro layer ───────────────────────────────────
# defend_hold / probe_attack / all_in_commit vote the combat posture in
# phase-1; phase-2 maps onto the existing atomic verbs via
# COMBAT_MACRO_DEFAULT_ATOMIC (same resolve_macro_execution machinery as
# the eco macros; no place-first guard for combat).
COMBAT_MACRO_NAMES = ("defend_hold", "probe_attack", "all_in_commit")

COMBAT_MACRO_DEFAULT_ATOMIC = {
    "defend_hold": "guard_choke",
    "probe_attack": "attack_nearest",
    "all_in_commit": "all_combat_attack_move",
}

RULE1_PLACE_NAMES = ("place_ready", "place_N", "place_E", "place_S", "place_W")


def list_combat_macros() -> list[Candidate]:
    return [
        Candidate("defend_hold", "combat",
                  "Posture: hold a defensive screen between our base and "
                  "the enemy, do not overextend.",
                  _guard_choke),
        Candidate("probe_attack", "combat",
                  "Posture: probe with an attack on the nearest visible "
                  "enemy while keeping the force intact.",
                  _attack_nearest),
        Candidate("all_in_commit", "combat",
                  "Posture: commit the whole force toward the enemy "
                  "centroid, engaging on contact.",
                  _all_combat_attack_move),
    ]


def list_executable_macros(macros: list[Candidate],
                           snap: Snapshot) -> list[Candidate]:
    """Jevfix F6: NO prefilter — the macro ballot always carries all 5.

    Demand and executability are separated: an ine executable macro stays
    votable and maps to a proxy action at execution time (see
    `demand_proxy_action`); keeping it off the ballot mistook "cannot"
    for "does not want". The `snap` arg stays for call-site compatibility.
    """
    return list(macros)


# ── Jevfix F1/F6: macro -> atomic execution ──────────────────────────
# Default mapping (used only when executable; F6 demand-proxy wins when
# the macro's own unit is not producible). Stored next to the macro
# definitions per the plan. open_powr's place half lives in the
# place-first guard inside `resolve_macro_execution`.
MACRO_DEFAULT_ATOMIC = {
    "fast_weap": "build_weap",
    "open_powr": "build_powr",
    "rush_barr": "build_barr",
    "econ_harv": "train_harv",
    "armor_push": "train_1tnk",
}

# Macros whose demand maps to a War Factory proxy when their own unit is
# not producible (F6: vote econ_harv with harv not in can_make -> build
# the weap that unlocks it; armor_push likewise).
DEMAND_PROXY_MACROS = {
    "econ_harv": ("harv", "build_weap"),
    "armor_push": ("1tnk", "build_weap"),
}

# Jevfix F1 (JudgeB B2): iron-rule retirement table, frozen to three rows.
# Only [two-phase active ∩ kind==eco ∩ phase-2] retires rule 2/3 (the
# FIX-#2 queue guard and the can_make veto give way to the macro mapping);
# single-phase eco, combat, and post-p95-fallback never retire. Rule 1
# (place-first) never retires — it is the place-first guard below.
IRON_RETIREMENT = {
    "rule1_place_first": "never retires (place-first guard always runs)",
    "rule2_queue_guard": "retires only in two-phase eco phase-2",
    "rule3_can_make_veto": "retires only in two-phase eco phase-2",
}

MACRO_WEAK_CONF = 0.40


def iron_retired(*, two_phase_on: bool, kind: str, phase2: bool) -> list[str]:
    """Which iron rules retire for this decision (F1 B2 table)."""
    if two_phase_on and kind == "eco" and phase2:
        return ["rule2_queue_guard", "rule3_can_make_veto"]
    return []


def demand_proxy_action(macro_name: str, snap: Snapshot,
                        by_name: dict) -> tuple[str, list[dict]] | None:
    """F6 demand-proxy: (proxy_atomic, actions) or None.

    Returns None when the macro needs no proxy (its unit is producible,
    or it is not a demand macro). The caller tags the landing
    `demand-proxy` and attributes it to the voted macro.
    """
    spec = DEMAND_PROXY_MACROS.get(macro_name)
    if spec is None:
        return None
    unit, proxy = spec
    can = {str(x) for x in (snap.available_production or [])}
    if unit in can:
        return None
    cand = by_name.get(proxy)
    if cand is None:
        return None
    return proxy, cand.actions(snap)


def resolve_macro_execution(macro_name: str, snap: Snapshot,
                            by_name: dict,
                            atomic_map: dict | None = None,
                            ) -> tuple[str, list[dict], str]:
    """F1 macro-mapped execution. Returns (atomic, actions, tag).

    Priority (frozen): place-first guard → demand-proxy conditional
    mapping → B2 retirement (recorded by the caller) → legality check →
    `macro-wait` / `macro-illegal-wait`. NEVER re-votes: illegal or empty
    mappings fall through to `wait`, never back to the model.
    Tags: "place-first" | "demand-proxy" | "mapped" | "macro-wait" |
    "macro-illegal-wait".
    """
    amap = atomic_map if atomic_map is not None else MACRO_DEFAULT_ATOMIC
    eco_macro = macro_name in MACRO_DEFAULT_ATOMIC
    # 1. place-first guard (rule 1 never retires; same order as the macro
    # builders: a finished building unblocks the queue first).
    if eco_macro and snap.ready_to_place and "place_ready" in by_name:
        acts = by_name["place_ready"].actions(snap)
        if acts:
            return "place_ready", acts, "place-first"
    # 2. F6 demand-proxy conditional mapping.
    proxy = demand_proxy_action(macro_name, snap, by_name)
    if proxy is not None:
        pname, pacts = proxy
        if pacts:
            return pname, pacts, "demand-proxy"
        return "wait", [], "macro-wait"
    # 3-4. default mapping + legality (same check as the ballot prefilter).
    atomic = amap.get(macro_name)
    if atomic is None or atomic not in by_name:
        return "wait", [], "macro-illegal-wait"
    if not list_executable_candidates([by_name[atomic]], snap):
        return "wait", [], "macro-illegal-wait"
    # Queue-blocked build_* would spam the Building queue (FIX-#2
    # semantics, now attributed to the macro instead of the atomics).
    if atomic.startswith("build_") and _building_in_queue(snap):
        return "wait", [], "macro-wait"
    acts = by_name[atomic].actions(snap)
    if not acts:
        return "wait", [], "macro-wait"
    return atomic, acts, "mapped"
