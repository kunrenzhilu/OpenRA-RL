"""Decision loop: every N ticks -> state -> backend -> gate -> batch -> advance.

Usage:
    PYTHONPATH=<worktree> <venv>/bin/python -m system1_commander.demo_loop \\
        --backend scripted --state combat --ticks-per-decision 25 \\
        --max-decisions 4 --log-dir .runs/system1-p0

Outputs in log-dir: orders.jsonl (one record per decision), bench.json
(summary), replay.txt (replay path + hash). Replays (*.orarep) and *.jsonl
are never committed.

Bench carries ground-truth fields from system1_commander.audit
(map_visible_orders / noop_ok_count / cash_spent ...): never use
``batch_ok`` counts as "orders placed" again.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone


# NanoJev plan §A3: same-kind consecutive FAILED (incl. empty packets)
# reaching K forces `wait` until the state materially changes.
STREAK_ALARM_K = 5


# ── Jev-max J1: ablation helpers (pure; judge recounts from orders.jsonl) ──

def top1_of(probs: dict | None) -> str | None:
    """Top-1 key with deterministic tie-break (sorted names, first max)."""
    if not probs:
        return None
    return max(sorted(probs), key=lambda k: float(probs[k] or 0.0))


def kl_div_bits(p: dict | None, q: dict | None, eps: float = 1e-9) -> float:
    """KL(p||q) in bits over the union of keys (eps-smoothed).

    Plan §1: p = rich-branch probs, q = poor-branch probs. Both sides are
    renormalized first so unnormalized backend probs still compare.
    """
    import math

    p, q = dict(p or {}), dict(q or {})
    keys = set(p) | set(q)
    if not keys:
        return 0.0
    ps = sum(float(p.get(k, 0.0) or 0.0) for k in keys) or 1.0
    qs = sum(float(q.get(k, 0.0) or 0.0) for k in keys) or 1.0
    n = len(keys)
    kl = 0.0
    for k in keys:
        pk = ((float(p.get(k, 0.0) or 0.0) / ps) + eps) / (1.0 + eps * n)
        qk = ((float(q.get(k, 0.0) or 0.0) / qs) + eps) / (1.0 + eps * n)
        kl += pk * math.log2(pk / qk)
    return kl


# ── Jev-max J3: fan-out extract + goal routing (pure) ──

def truthy_noul(v) -> bool:
    """Normalize a Noul answer (bool or 'true'/'false' str) to bool."""
    if isinstance(v, bool):
        return v
    return str(v or "").strip().lower() in ("true", "1", "yes", "y")


def fanout_of(pred_detail: dict | None) -> dict:
    """Extract {goal, goal_conf, phase, threat_recall, danger} from a
    predict detail dict. Backend-agnostic: unknown backends yield Nones."""
    d = pred_detail or {}
    ch = d.get("choices") or {}
    goal = ch.get("goal") or {}
    nl = d.get("nouls") or {}
    sc = d.get("scores") or {}
    return {
        "goal": goal.get("choice"),
        "goal_conf": goal.get("confidence"),
        "phase": sc.get("phase"),
        "threat_recall": nl.get("threat_recall"),
        "danger": sc.get("danger"),
    }


_FALLBACK_SENTINEL = "__fallback__"
_AGGRESSIVE_TACTICS = ("attack_nearest", "all_combat_attack_move")
_DEFENSIVE_FALLBACKS = ("guard_choke", "hold_position")


def jev_goal_route(*, gate_mode: str, gate_choice: str,
                   ballot_names: list[str],
                   fanout: dict | None) -> tuple[str | None, str]:
    """Pure J3 routing decision. Returns (override|None, note).

    - downgrade -> ALWAYS re-resolve via fallback (downgrade results are
      discarded: A-track breakpoint 2 unfixed, plan §3 footnote).
      Signals _FALLBACK_SENTINEL (caller runs predict_with_ban).
    - goal=defend + threat_recall + aggressive tactic -> defensive choice
      when on the ballot, else the sentinel.
    - otherwise None (keep the gate's verdict).
    """
    fo = fanout or {}
    if gate_mode == "downgrade":
        return _FALLBACK_SENTINEL, "downgrade-discarded->fallback"
    if (fo.get("goal") == "defend"
            and truthy_noul(fo.get("threat_recall"))
            and gate_choice in _AGGRESSIVE_TACTICS):
        for n in _DEFENSIVE_FALLBACKS:
            if n in ballot_names:
                return n, "goal=defend+threat->defensive"
        return _FALLBACK_SENTINEL, "goal=defend+threat->fallback"
    return None, ""


# ── Jev-max J4: trajectory summarizer (pure; K=10 segments) ──
MEMORY_K = 10


def summarize_trajectory(rows: list[dict] | None, k: int = MEMORY_K) -> dict:
    """Compress the last K decision rows into a memory dict.

    Pure function of orders.jsonl rows: verdicts via audit_decisions,
    cash curve via parse_batch_state (observed cash only). The judge
    replays it offline with zero cost. Non-decision rows (deploy/alarm/
    skip) are ignored. `segs` is oldest-first so attach_history can trim
    the oldest first under its token budget.
    """
    from system1_commander.audit import audit_decisions, parse_batch_state

    dec = [r for r in (rows or []) if r.get("kind", "decision") == "decision"]
    dec = dec[-max(1, int(k)):]
    verdicts = ["" for _ in dec]
    try:
        verdicts = [str(r.get("verdict", ""))
                    for r in audit_decisions(dec)["rows"]]
    except Exception:  # noqa: BLE001 - memory must never break the loop
        pass
    segs, cash = [], []
    for r, v in zip(dec, verdicts):
        try:
            st = parse_batch_state(r.get("batch_note", ""))
        except Exception:  # noqa: BLE001
            st = None
        if isinstance(st, dict) and st.get("cash") is not None:
            try:
                cash.append(int(st["cash"]))
            except (TypeError, ValueError):
                pass
        segs.append({"i": r.get("i"), "tick": r.get("tick"),
                     "choice": str(r.get("choice", "")),
                     "gate": str((r.get("gate") or {}).get("mode", "")),
                     "verdict": v})
    if len(cash) <= 5:
        curve = list(cash)
    else:
        curve = [cash[round(i * (len(cash) - 1) / 4)] for i in range(5)]
    counts: dict[str, int] = {}
    for s in segs:
        counts[s["choice"]] = counts.get(s["choice"], 0) + 1
    vcounts: dict[str, int] = {}
    for v in verdicts:
        vcounts[v or "?"] = vcounts.get(v or "?", 0) + 1
    return {
        "k": int(k),
        "n": len(segs),
        "tick_span": [segs[0]["tick"], segs[-1]["tick"]] if segs else [],
        "choice_counts": counts,
        "verdict_counts": vcounts,
        "cash_curve": curve,
        "segs": segs,
    }


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="System1 commander demo loop (P0).")
    p.add_argument("--backend", choices=["scripted", "jev", "nanojev", "laya"], default="scripted")
    p.add_argument("--state", choices=["combat", "eco"], default="combat")
    p.add_argument("--mix", default="",
                   help="Comma list to alternate states per decision, e.g. 'eco,combat'. "
                        "Overrides --state when set.")
    p.add_argument("--play-to-end", action="store_true",
                   help="Loop until done/result; surrender at --max-ticks.")
    p.add_argument("--max-ticks", type=int, default=15000,
                   help="Tick budget for --play-to-end (~10 game-minutes); overruns surrender.")
    p.add_argument("--ticks-per-decision", type=int, default=25)
    p.add_argument("--max-decisions", type=int, default=4)
    p.add_argument("--log-dir", default=".runs/system1-p0")
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--map", default="pitfight.oramap",
                   help="Map for the match. Explicit (not server default) so reruns "
                        "stay comparable after container rebuilds.")
    # Jev-max J1: ablation + state budget + cost cap (all backend-agnostic).
    p.add_argument("--ablation", action="store_true",
                   help="J1: on sampled ticks ask both poor and rich state. "
                        "The poor branch always executes; the rich branch is "
                        "record-only (poor_hash/rich_hash + dual full probs).")
    p.add_argument("--ablation-every", type=int, default=10,
                   help="J1: sample 1 tick per N decisions (default 10).")
    p.add_argument("--ablation-seed", type=int, default=7,
                   help="J1: fixed seed; sample phase = seed %% every "
                        "(recorded in bench).")
    p.add_argument("--state-budget", type=int, default=800,
                   help="J1: token budget for state_to_json (800 -> 2k -> 4k "
                        "ladder per plan §1).")
    p.add_argument("--cost-cap", type=float, default=0.30,
                   help="Plan §7: per-game USD cap. Once hit, remaining "
                        "decisions turn to wait (stop asking, save the game). "
                        "0 disables.")
    # Jev-max J2: two-phase macro -> atomic (eco decisions only).
    p.add_argument("--two-phase", action="store_true",
                   help="J2: phase-1 votes a macro, phase-2 picks the atomic "
                        "action with macro=<round-1> prefixed to the state. "
                        "Auto-reverts to single-phase if two-phase p95 > 4s.")
    # Jev-max J3: goal routing (execute/fallback only, downgrade discarded).
    p.add_argument("--goal-route", action="store_true",
                   help="J3: route on the fan-out goal Choice (downgrade "
                        "results discarded -> fallback; goal=defend + threat "
                        "overrides aggressive tactics). No-op for backends "
                        "without goal answers.")
    # Jev-max J4: external memory on the poor state (baseline capability).
    p.add_argument("--memory", action="store_true",
                   help="J4: attach the K=10 trajectory summary as the "
                        "history state field (300 toks, oldest trimmed). "
                        "With --ablation, memory is held constant and the "
                        "rich branch adds only the five J1 fields.")
    return p.parse_args(argv)


def _make_backend(name: str):
    if name == "scripted":
        from system1_commander.backend_scripted import ScriptedBackend
        return ScriptedBackend()
    if name == "jev":
        from system1_commander.backend_jev import JevBackend
        return JevBackend()
    if name == "nanojev":
        from system1_commander.backend_nanojev import NanoJevBackend
        return NanoJevBackend()
    if name == "laya":
        from system1_commander.backend_laya import LayaBackend
        return LayaBackend()
    raise ValueError(f"unknown backend: {name}")


async def run(args) -> dict:
    from system1_commander import (
        apply_gate,
        build_combat_state,
        build_eco_state,
        build_snapshot,
        estimate_tokens,
        exec_flags,
        list_combat_candidates,
        list_eco_candidates,
        list_executable_candidates,
        make_skip_entry,
        state_to_json,
    )
    from system1_commander.executor import Executor
    from system1_commander.gate import GateConfig, GateDecision, predict_with_ban
    from system1_commander.backend_base import Prediction
    from system1_commander.state import Snapshot
    from system1_commander.state import (  # Jev-max J1 (module import: __init__ untouched)
        attach_history,  # J4
        attach_rich_fields,
        observe_trackers,
        reset_trackers,
    )
    from system1_commander.candidates import (  # Jev-max J2 (same reason)
        list_executable_macros,
        list_macro_candidates,
    )

    os.makedirs(args.log_dir, exist_ok=True)
    orders_path = os.path.join(args.log_dir, "orders.jsonl")
    bench_path = os.path.join(args.log_dir, "bench.json")
    replay_path_txt = os.path.join(args.log_dir, "replay.txt")

    from openra_env.mcp_ws_client import OpenRAMCPClient

    backend = _make_backend(args.backend)
    kinds = [k.strip() for k in args.mix.split(",") if k.strip()] or [args.state]
    for k in kinds:
        if k not in ("combat", "eco"):
            raise ValueError(f"--mix/--state kind must be combat|eco, got {k!r}")
    pools = {"combat": list_combat_candidates(), "eco": list_eco_candidates()}
    builders = {"combat": build_combat_state, "eco": build_eco_state}
    gate_cfg = GateConfig()
    if args.play_to_end:
        decision_cap = args.max_ticks // max(1, args.ticks_per_decision) + 20
    else:
        decision_cap = args.max_decisions

    t_start = time.monotonic()
    latencies, costs, state_toks = [], [], []
    fix_flags = []  # parallel to latencies/costs: True when row bypassed predict (#1/#2)
    decisions_log = []
    gate_modes, kinds_used = [], []
    # FIX-#1 breaker: consecutive bypass-place failures; >=3 disables #1 for the game.
    fix1_disabled, fix1_consec_fails = False, 0
    # A-track (NanoJev plan §A1/A3): per-game failure memory + losing
    # streaks. banned (failed_set) never clears within a game; streak
    # resets on material state change or success. Jev #1/#2/breaker above untouched.
    failed_set: set[str] = set()
    streak: dict[str, int] = {}
    last_sig: dict[str, tuple] = {}
    game_done, game_result, surrendered = False, "", False
    error_note = ""
    n_buildings_0, mil0 = 0, (0, 0, 0, 0, 0)
    last_snap, snap_end, replay_info = None, None, ""
    deploy_rec = None
    # Jev-max J1: ablation phase + §7 cost cap (stop-asking-save-game).
    _abl_every = max(1, int(args.ablation_every or 10))
    abl_offset = int(args.ablation_seed or 0) % _abl_every
    cap_hit, cap_hit_i = False, None
    # Jev-max J2: two-phase serial p95 must stay <= 4s, else revert (§2).
    two_phase_on = bool(args.two_phase)
    two_phase_lat: list[float] = []
    two_phase_disabled_at = None

    try:
        async with OpenRAMCPClient(base_url=args.url, message_timeout_s=300.0) as client:
            ex = Executor(client)
            print(f"[demo] reset @ {args.url} map={args.map} ...", flush=True)
            await client.reset(map_name=args.map)
            # Unpause: a fresh match starts paused (world created, tick ~3).
            # try-agent uses start+end planning to launch; we run no planning,
            # so open/close it immediately to start the match clock.
            try:
                await ex.tool("start_planning_phase")
            except Exception as e:  # noqa: BLE001
                print(f"[demo] start_planning_phase note: {e}", flush=True)
            try:
                await ex.tool("end_planning_phase", strategy="system1 demo: no central plan")
            except Exception as e:  # noqa: BLE001
                print(f"[demo] end_planning_phase note: {e}", flush=True)
            # Cold container: dotnet JIT + map load takes ~1-2 min. Poll until
            # the match exists (units spawned); advancing too early fails.
            ready_snap = None
            for _ in range(60):
                await asyncio.sleep(5)
                try:
                    gs, units, buildings = await ex.fetch_raw()
                    ready_snap = build_snapshot(gs, units, buildings)
                except Exception as e:  # noqa: BLE001
                    print(f"[demo] waiting for game... ({e})", flush=True)
                    continue
                if ready_snap.own_units or ready_snap.tick > 10:
                    break
                print(f"[demo] waiting for game... tick={ready_snap.tick} "
                      f"units={len(ready_snap.own_units)}", flush=True)
            else:
                print("[demo] WARNING: game never looked ready; continuing anyway",
                      flush=True)
            # Warmup: freshly reset games spawn empty (tick 0); advance so the
            # MCV / starting units exist before the first snapshot.
            await ex.advance(25)
            gs, units, buildings = await ex.fetch_raw()
            snap = build_snapshot(gs, units, buildings)
            n_buildings_0 = len(snap.own_buildings)
            mil0 = (snap.kills, snap.losses, snap.kills_cost, snap.deaths_cost, snap.order_count)
            print(f"[demo] start: tick={snap.tick} map={snap.map_name or '?'} "
                  f"units={len(snap.own_units)} buildings={n_buildings_0} "
                  f"mcv={snap.mcv_id} fact={snap.has_fact}", flush=True)

            deploy_rec = await ex.ensure_deployed(snap)
            if deploy_rec is not None:
                print(f"[demo] deploy_mcv: ok={deploy_rec.batch_ok} note={deploy_rec.batch_note[:120]}",
                      flush=True)
                gs, units, buildings = await ex.fetch_raw()
                snap = build_snapshot(gs, units, buildings)
                print(f"[demo] post-deploy: buildings={len(snap.own_buildings)}", flush=True)

            with open(orders_path, "w") as f_orders:
                if deploy_rec is not None:
                    f_orders.write(json.dumps({"kind": "deploy", **deploy_rec.to_dict()}) + "\n")
                reset_trackers()  # Jev-max J1: fresh memory per game.

                for i in range(decision_cap):
                    gs, units, buildings = await ex.fetch_raw()
                    snap = build_snapshot(gs, units, buildings)
                    last_snap = snap
                    if isinstance(gs, dict) and gs.get("done"):
                        game_done, game_result = True, str(gs.get("result", ""))
                        print(f"[demo] game over: {game_result}", flush=True)
                        break
                    if args.play_to_end and snap.tick >= args.max_ticks:
                        print(f"[demo] tick budget {args.max_ticks} hit at tick={snap.tick}; "
                              f"surrendering", flush=True)
                        try:
                            await ex.tool("surrender")
                        except Exception as e:  # noqa: BLE001
                            print(f"[demo] surrender tool failed: {e}", flush=True)
                        surrendered = True
                        # Drain to game end so the engine flushes the replay:
                        # surrender alone only queues the order; ticks must advance
                        # for done=True. Cap 8x25=200 ticks, then give up honestly.
                        for _ in range(8):
                            try:
                                await ex.advance(25)
                            except Exception as e:  # noqa: BLE001
                                print(f"[demo] drain advance failed: {e}", flush=True)
                                break
                            gs, units, buildings = await ex.fetch_raw()
                            if isinstance(gs, dict) and gs.get("done"):
                                game_done, game_result = True, str(gs.get("result", ""))
                                print(f"[demo] game over after surrender: {game_result}",
                                      flush=True)
                                break
                        else:
                            print("[demo] drain budget exhausted, game not done; "
                                  "continuing without done flag", flush=True)
                        gs, units, buildings = await ex.fetch_raw()
                        snap_end_probe = build_snapshot(gs, units, buildings)
                        game_result = str(gs.get("result", "") or game_result)
                        game_done = bool(isinstance(gs, dict) and gs.get("done"))
                        snap = snap_end_probe
                        break
                    kind = kinds[i % len(kinds)]
                    candidates = pools[kind]
                    by_name = {c.name: c for c in candidates}
                    observe_trackers(snap)  # Jev-max J1: feed enemy/traj memory.
                    state = builders[kind](snap)
                    if args.memory:  # Jev-max J4: history on the poor state.
                        _mem_rows = [e for e in decisions_log
                                     if e.get("kind") == "decision"]
                        state = attach_history(
                            state, summarize_trajectory(_mem_rows))
                    state_json, toks = state_to_json(state, args.state_budget)
                    state_toks.append(toks)
                    guard_actions = by_name["guard_choke"].actions(snap) if "guard_choke" in by_name else None
                    # FIX-#1: ready_to_place non-empty hard-cut to place_ready
                    # (eco only, builder double-checked, skips predict).
                    fix_tag = None
                    # A-track row state (defaults for fix-bypass rows).
                    ballot_names: list[str] | None = None
                    is_skip = False
                    abl_entry = None  # Jev-max J1: filled on sampled ticks.
                    mchoice, macro_entry = None, None  # Jev-max J2.
                    fanout_rec, route_note = None, None  # Jev-max J3.
                    discard_reason = None  # Jev-max J1: why a vote was discarded.
                    # Plan §7 cost cap: stop asking, save the game. Once hit,
                    # every remaining decision is wait (advance only); the
                    # ablation sampler below also stops (pred is synthetic).
                    cap_stop = cap_hit
                    if (not cap_hit and (args.cost_cap or 0) > 0
                            and sum(costs) >= args.cost_cap):
                        cap_hit, cap_hit_i, cap_stop = True, i, True
                        print(f"[demo] COST CAP ${args.cost_cap:.2f} hit at d{i}; "
                              f"remaining decisions -> wait", flush=True)
                    if cap_stop:
                        pred = Prediction(
                            choice="wait", probs={"wait": 1.0},
                            confidence=1.0, latency_ms=0.0, cost_usd=0.0,
                            backend="cap-stop",
                            detail={"skipped_predict": True},
                        )
                        gate = GateDecision(
                            "downgrade", "wait",
                            f"COST-CAP ${args.cost_cap:.2f}: stop asking, save game",
                        )
                        actions = []
                    elif kind == "eco" and not fix1_disabled and snap.ready_to_place:
                        _place_actions = by_name["place_ready"].actions(snap)
                        if _place_actions:
                            fix_tag = "#1"
                            pred = Prediction(
                                choice="place_ready", probs={"place_ready": 1.0},
                                confidence=1.0, latency_ms=0.0, cost_usd=0.0,
                                backend="fix-bypass",
                                detail={"skipped_predict": True},
                            )
                            gate = GateDecision(
                                "execute", "place_ready",
                                f"FIX-#1 hard-cut: ready_to_place={snap.ready_to_place}",
                            )
                            actions = _place_actions
                    # FIX-#2: Building queue non-empty guard (copies
                    # examples/scripted_bot.py:287-292 semantics: any Building
                    # queue item counts as in progress, progress unchecked).
                    # Ready rows already handled by #1; here only the
                    # 0%<=progress<99% in-progress case remains -> wait,
                    # skipping predict (plan verdict (a)).
                    if fix_tag is None and not cap_stop and kind == "eco" and not snap.ready_to_place:
                        _building_in_queue = any(
                            p.get("queue_type") == "Building"
                            for p in (snap.production or [])
                        )
                        if _building_in_queue:
                            fix_tag = "#2"
                            pred = Prediction(
                                choice="wait", probs={"wait": 1.0},
                                confidence=1.0, latency_ms=0.0, cost_usd=0.0,
                                backend="fix-bypass",
                                detail={"skipped_predict": True},
                            )
                            gate = GateDecision(
                                "downgrade", "wait",
                                "FIX-#2 guard: building in queue, skip build_* spam",
                            )
                            actions = by_name["wait"].actions(snap)
                    if fix_tag is None and not cap_stop:
                        # A-track wiring (additive; Jev #1/#2/breaker above untouched).
                        flags = exec_flags(snap)
                        use_cands = candidates
                        if args.backend == "laya":
                            # Laya plan §A1: prefilter; empty ballot -> skip entry.
                            use_cands = list_executable_candidates(candidates, snap)
                            ballot_names = [c.name for c in use_cands]
                        if args.backend == "nanojev":
                            # NanoJev plan §A3: streak resets on material
                            # state change (banned never clears: same
                            # preconditions => same failure).
                            sig = (snap.cash, len(snap.own_buildings),
                                   tuple(snap.ready_to_place))
                            if (last_sig.get(kind) is not None
                                    and last_sig.get(kind) != sig):
                                streak[kind] = 0
                            last_sig[kind] = sig
                        laya_empty = args.backend == "laya" and len(use_cands) == 0
                        laya_wait_only = (args.backend == "laya"
                                          and len(use_cands) == 1
                                          and use_cands[0].name == "wait")
                        nano_force_wait = (args.backend == "nanojev"
                                           and streak.get(kind, 0) >= STREAK_ALARM_K)
                        pred = None
                        is_skip = laya_empty
                        if is_skip:
                            gate = None
                            actions = []
                        elif laya_wait_only:
                            # Laya plan §A1: ballot is [wait]; skip Laya call.
                            gate = GateDecision(
                                "downgrade", "wait",
                                "prefilter wait-only, skipped Laya call")
                            actions = []
                        elif nano_force_wait:
                            # NanoJev plan §A3: save RTT, wait for state change.
                            discard_reason = "streak"  # Jev-max J1 ablation note.
                            gate = GateDecision(
                                "fallback", "wait",
                                f"losing streak>={STREAK_ALARM_K}, "
                                "wait until state changes")
                            actions = []
                        else:
                            # Jev-max J2: phase-1 macro vote (eco only, bypass
                            # rows never reach here). Advisory: no gate, the
                            # vote only prefixes phase-2 state + the log.
                            state2 = state
                            if two_phase_on and kind == "eco":
                                _macros = list_executable_macros(
                                    list_macro_candidates(), snap)
                                if _macros:
                                    _mnames = {m.name for m in _macros}
                                    pred1 = await asyncio.to_thread(
                                        backend.predict, state, _macros)
                                    costs.append(pred1.cost_usd)
                                    mchoice = (pred1.choice
                                               if pred1.choice in _mnames
                                               else _macros[0].name)
                                    state2 = {"macro": mchoice, **state}
                                    _, _toks2 = state_to_json(
                                        state2, args.state_budget)
                                    macro_entry = {
                                        "choice": mchoice,
                                        "probs": pred1.probs,
                                        "confidence": pred1.confidence,
                                        "latency_ms": round(
                                            pred1.latency_ms, 1),
                                        "cost_usd": pred1.cost_usd,
                                        "ballot": sorted(_mnames),
                                        "phase2_tokens": _toks2,
                                    }
                            pred = await asyncio.to_thread(backend.predict, state2, use_cands)
                            gate = apply_gate(
                                pred, use_cands, _make_backend("scripted"),
                                state, gate_cfg,
                                banned=(failed_set
                                        if args.backend == "nanojev" else None))
                            actions = (by_name[gate.choice_name].actions(snap)
                                       if gate.choice_name in by_name else [])
                            if macro_entry is not None:
                                # Combined wall-clock latency for the decision;
                                # phase-1 keeps its own numbers in the entry.
                                pred.latency_ms = (pred.latency_ms or 0.0) + (
                                    macro_entry["latency_ms"] or 0.0)
                                two_phase_lat.append(pred.latency_ms)
                                if len(two_phase_lat) >= 3:
                                    _p95 = sorted(two_phase_lat)[
                                        max(0, int(len(two_phase_lat) * 0.95) - 1)]
                                    if _p95 > 4000.0:
                                        two_phase_on = False
                                        two_phase_disabled_at = i
                                        print(
                                            "[demo] TWO-PHASE p95 "
                                            f"{_p95:.0f}ms > 4000ms at d{i}; "
                                            "reverting to single-phase",
                                            flush=True)
                            # Jev-max J3: fan-out record + goal routing. The
                            # fanout record is kept whenever the backend
                            # answered (Jev); routing needs --goal-route.
                            fanout_rec = fanout_of(pred.detail)
                            if not any(v is not None
                                       for v in fanout_rec.values()):
                                fanout_rec = None
                            if args.goal_route and fanout_rec is not None:
                                _ov, _note = jev_goal_route(
                                    gate_mode=gate.mode,
                                    gate_choice=gate.choice_name,
                                    ballot_names=[c.name for c in use_cands],
                                    fanout=fanout_rec)
                                if _ov == _FALLBACK_SENTINEL:
                                    _sb = _make_backend("scripted")
                                    _fb = predict_with_ban(
                                        state, use_cands,
                                        (failed_set if args.backend == "nanojev"
                                         else None), _sb, pred.probs)
                                    gate = GateDecision(
                                        "fallback", _fb.choice,
                                        gate.reason + f" [jevmax-route: {_note}]")
                                    actions = (by_name[gate.choice_name].actions(snap)
                                               if gate.choice_name in by_name else [])
                                    route_note = _note
                                elif _ov is not None:
                                    gate = GateDecision(
                                        "fallback", _ov,
                                        gate.reason + f" [jevmax-route: {_note}]")
                                    actions = (by_name[_ov].actions(snap)
                                               if _ov in by_name else [])
                                    route_note = _note
                            # Jev-max J1 ablation: same-tick poor/rich dual ask.
                            # ALWAYS executes the poor branch (pred/gate/actions
                            # above untouched); the rich branch is record-only.
                            if (args.ablation and pred is not None
                                    and ((i + abl_offset) % _abl_every == 0)):
                                rich_state = attach_rich_fields(dict(state), snap)
                                rich_json, rich_toks = state_to_json(
                                    rich_state, args.state_budget)
                                pred_r = await asyncio.to_thread(
                                    backend.predict, rich_state, use_cands)
                                costs.append(pred_r.cost_usd)
                                t_poor = top1_of(pred.probs)
                                t_rich = top1_of(pred_r.probs)
                                abl_entry = {
                                    "poor_hash": hashlib.sha1(
                                        state_json.encode()).hexdigest()[:12],
                                    "rich_hash": hashlib.sha1(
                                        rich_json.encode()).hexdigest()[:12],
                                    "rich_choice": pred_r.choice,
                                    "rich_probs": pred_r.probs,
                                    "rich_conf": pred_r.confidence,
                                    "rich_tokens": rich_toks,
                                    "rich_latency_ms": round(pred_r.latency_ms, 1),
                                    "rich_cost_usd": pred_r.cost_usd,
                                    "poor_top1": t_poor,
                                    "rich_top1": t_rich,
                                    "flipped": bool(t_poor != t_rich),
                                    "kl_bits": round(kl_div_bits(
                                        pred_r.probs, pred.probs), 4),
                                }
                    # Jev-max J1 ablation, bypass-row arm (plan §1 literal:
                    # EVERY 10th decision tick dual-asks, even when a hard
                    # rule discards the vote; the executed action above is
                    # untouched, so the trajectory never forks). Cap/skip
                    # rows never sample (cap stops asking, skip has no ballot).
                    if (args.ablation and abl_entry is None and not is_skip
                            and not cap_stop
                            and ((i + abl_offset) % _abl_every == 0)):
                        _ab_ballot = candidates
                        if args.backend == "laya":
                            _ab_ballot = list_executable_candidates(
                                candidates, snap)
                        if len(_ab_ballot) >= 2:
                            pred_p = await asyncio.to_thread(
                                backend.predict, state, _ab_ballot)
                            rich_state = attach_rich_fields(
                                dict(state), snap)
                            rich_json, rich_toks = state_to_json(
                                rich_state, args.state_budget)
                            pred_r = await asyncio.to_thread(
                                backend.predict, rich_state, _ab_ballot)
                            costs.append(pred_p.cost_usd)
                            costs.append(pred_r.cost_usd)
                            _tp = top1_of(pred_p.probs)
                            _tr = top1_of(pred_r.probs)
                            abl_entry = {
                                "poor_hash": hashlib.sha1(
                                    state_json.encode()).hexdigest()[:12],
                                "rich_hash": hashlib.sha1(
                                    rich_json.encode()).hexdigest()[:12],
                                "poor_choice": pred_p.choice,
                                "poor_probs": pred_p.probs,
                                "poor_conf": pred_p.confidence,
                                "rich_choice": pred_r.choice,
                                "rich_probs": pred_r.probs,
                                "rich_conf": pred_r.confidence,
                                "rich_tokens": rich_toks,
                                "rich_latency_ms": round(
                                    pred_r.latency_ms, 1),
                                "rich_cost_usd": pred_r.cost_usd,
                                "poor_top1": _tp,
                                "rich_top1": _tr,
                                "flipped": bool(_tp != _tr),
                                "kl_bits": round(kl_div_bits(
                                    pred_r.probs, pred_p.probs), 4),
                                "discarded_by": (
                                    fix_tag if fix_tag is not None
                                    else (discard_reason or "unknown")),
                            }
                    if is_skip:
                        moved, interrupted = await ex.advance(args.ticks_per_decision)
                        entry = make_skip_entry(
                            i=i, tick=snap.tick, state_kind=kind,
                            state_tokens=toks, flags=flags,
                            advance_ticks=moved,
                            advance_interrupted=interrupted)
                        latencies.append(0.0)
                        costs.append(0.0)
                        fix_flags.append(True)  # no inference, like fix-bypass
                        gate_modes.append("skip-empty-ballot")
                        kinds_used.append(kind)
                    else:
                        latencies.append(pred.latency_ms if pred is not None else 0.0)
                        costs.append(pred.cost_usd if pred is not None else 0.0)
                        fix_flags.append(fix_tag is not None or pred is None or cap_stop)
                        gate_modes.append(gate.mode)
                        kinds_used.append(kind)
                        rec = await ex.execute(gate.choice_name, actions, snap.tick, guard_actions)
                        moved, interrupted = await ex.advance(args.ticks_per_decision)
                        rec.advance_ticks = moved
                        rec.advance_interrupted = interrupted
                        entry = {
                            "kind": "decision", "i": i, "tick": snap.tick,
                            "state_kind": kind,
                            "state_tokens": toks,
                            "gate": gate.to_dict(),
                            **rec.to_dict(),
                        }
                        if pred is not None:
                            entry["prediction"] = pred.to_dict()
                        if ballot_names is not None:
                            entry["ballot"] = ballot_names
                            entry["exec_flags"] = flags
                        if fix_tag is not None:
                            entry["fix_bypass"] = fix_tag
                        if cap_stop:
                            entry["cap_stop"] = True
                        if abl_entry is not None:
                            entry["ablation"] = abl_entry
                        if mchoice is not None:
                            entry["macro"] = mchoice
                            entry["phase1"] = macro_entry
                        if fanout_rec is not None:
                            entry["jev_fanout"] = fanout_rec
                        if route_note is not None:
                            entry["goal_route"] = route_note
                        # Laya plan §A2: honest books; batch_ok / gate.mode frozen.
                        if not rec.actions and rec.choice != "wait":
                            entry["noop_alarm"] = True
                            entry["gate"]["alarm"] = "empty-execute"
                            entry["batch_note"] = f"noop ALARM: {rec.choice} empty"
                        # NanoJev plan §A1/A3: per-game failure memory + streak.
                        if args.backend == "nanojev":
                            failed = (not rec.batch_ok) or (
                                not rec.actions and rec.choice != "wait")
                            if failed:
                                failed_set.add(rec.choice)
                                streak[kind] = streak.get(kind, 0) + 1
                                if streak[kind] == STREAK_ALARM_K:
                                    f_orders.write(json.dumps({
                                        "kind": "alarm", "i": i,
                                        "tick": snap.tick, "state_kind": kind,
                                        "streak": STREAK_ALARM_K,
                                        "banned": sorted(failed_set),
                                    }) + "\n")
                            else:
                                streak[kind] = 0
                    # FIX-#1 breaker: a bypass place that changes nothing on
                    # the map (online signal) counts a consecutive failure;
                    # success resets. Only #1-issued places count.
                    if fix_tag == "#1":
                        from system1_commander.audit import parse_batch_state as _parse_bs
                        _post = _parse_bs(rec.batch_note)
                        if _post is not None and _post.get("own_buildings") is not None:
                            _placed_ok = _post["own_buildings"] > len(snap.own_buildings)
                        else:
                            _placed_ok = bool(rec.batch_ok)
                        if _placed_ok:
                            fix1_consec_fails = 0
                        else:
                            fix1_consec_fails += 1
                            if fix1_consec_fails >= 3:
                                fix1_disabled = True
                                entry["fix1_breaker"] = "tripped"
                    f_orders.write(json.dumps(entry) + "\n")
                    f_orders.flush()
                    decisions_log.append(entry)
                    if is_skip:
                        print(f"[demo] d{i}: tick={snap.tick} [{kind}] toks={toks} "
                              f"skip-empty-ballot adv={moved}{'!' if interrupted else ''}",
                              flush=True)
                    else:
                        _pstr = (f"{pred.choice}@{pred.confidence:.2f}"
                                 if pred is not None else "none")
                        _plat = pred.latency_ms if pred is not None else 0.0
                        _pcost = pred.cost_usd if pred is not None else 0.0
                        print(f"[demo] d{i}: tick={snap.tick} [{kind}] toks={toks} "
                              f"pred={_pstr} gate={gate.mode}:{gate.choice_name} "
                              f"batch_ok={rec.batch_ok} adv={moved}{'!' if interrupted else ''} "
                              f"{_plat:.0f}ms ${_pcost:.6f}"
                              f"{' fix=' + fix_tag if fix_tag else ''}"
                              f"{' macro=' + str(mchoice) if mchoice else ''}", flush=True)
                    if game_done:
                        break

            gs, units, buildings = await ex.fetch_raw()
            snap_end = build_snapshot(gs, units, buildings)
            try:
                r = await ex.tool("get_replay_path")
                replay_info = r if isinstance(r, str) else json.dumps(r)
            except Exception as e:  # noqa: BLE001
                replay_info = f"get_replay_path failed: {e}"
    except Exception as e:  # noqa: BLE001
        # Crash mid-game (e.g. server restarted): keep partial orders + bench.
        error_note = f"{type(e).__name__}: {e}"[:300]
        print(f"[demo] CRASHED, writing partial bench: {error_note}", flush=True)

    if snap_end is None:
        snap_end = last_snap if last_snap is not None else Snapshot()

    replay_hash = ""
    for token in replay_info.replace('"', " ").replace("'", " ").split():
        if token.endswith(".orarep"):
            try:
                with open(token, "rb") as f:
                    replay_hash = hashlib.sha256(f.read()).hexdigest()
                replay_info = token
            except OSError:
                pass
            break
    with open(replay_path_txt, "w") as f:
        f.write(f"replay={replay_info}\nsha256={replay_hash}\n")

    wall_s = time.monotonic() - t_start
    # FIX-#1/#2 bookkeeping: latency percentiles must EXCLUDE fix-bypass
    # rows (0.0 would drag p50 down); cost sums are unaffected (0.0).
    _lat_real = [v for v, f in zip(latencies, fix_flags) if not f] or [0.0]
    _fix1_n = sum(1 for e in decisions_log if e.get("fix_bypass") == "#1")
    _fix2_n = sum(1 for e in decisions_log if e.get("fix_bypass") == "#2")
    # A-track honest books: FAILED totals are reported, never hidden.
    # (skip entries carry no batch_ok and are excluded by the default.)
    _failed_by_choice: dict[str, int] = {}
    for _e in decisions_log:
        if _e.get("kind") == "decision" and not _e.get("batch_ok", True):
            _c = str(_e.get("choice", "?"))
            _failed_by_choice[_c] = _failed_by_choice.get(_c, 0) + 1
    # Jev-max J1: ablation tally (judge recounts from orders.jsonl entries).
    _abl_rows = [e["ablation"] for e in decisions_log if e.get("ablation")]
    _abl_flips = sum(1 for a in _abl_rows if a.get("flipped"))
    _abl_kl = [float(a.get("kl_bits", 0.0) or 0.0) for a in _abl_rows]
    _abl_voted = [a for a in _abl_rows if not a.get("discarded_by")]
    _abl_vflips = sum(1 for a in _abl_voted if a.get("flipped"))
    _abl_vkl = [float(a.get("kl_bits", 0.0) or 0.0) for a in _abl_voted]
    # Jev-max J3: fan-out tallies (Noul/Score truly wired into stats).
    _fo_rows = [e.get("jev_fanout") or {} for e in decisions_log
                if e.get("jev_fanout")]
    _goal_counts: dict[str, int] = {}
    for _f in _fo_rows:
        _g = str(_f.get("goal"))
        _goal_counts[_g] = _goal_counts.get(_g, 0) + 1
    bench = {
        "backend": args.backend,
        "map": snap_end.map_name,
        "state_kind": f"mix:{','.join(kinds)}" if args.mix else args.state,
        "ticks_per_decision": args.ticks_per_decision,
        "play_to_end": args.play_to_end, "max_ticks": args.max_ticks,
        "surrendered": surrendered,
        "crashed": bool(error_note), "error": error_note,
        "decisions": len(decisions_log),
        "decisions_by_kind": {k: kinds_used.count(k) for k in set(kinds_used)},
        "gate_rule": (f"execute if conf>= {gate_cfg.high} or top1-top2 margin>= {gate_cfg.margin_min}; "
                      f"downgrade band [{gate_cfg.low},{gate_cfg.high}); else scripted fallback "
                      f"(destructive needs conf>= {gate_cfg.destructive_min})"),
        "gate_modes": {m: gate_modes.count(m) for m in set(gate_modes)},
        "failed_by_choice": _failed_by_choice,
        "ticks_total": snap_end.tick,
        "wall_s": round(wall_s, 1),
        "buildings_before": n_buildings_0, "buildings_after": len(snap_end.own_buildings),
        "units_after": len(snap_end.own_units),
        "enemies_visible_end": len(snap_end.enemies),
        "kills_delta": snap_end.kills - mil0[0], "losses_delta": snap_end.losses - mil0[1],
        "kills_cost_delta": snap_end.kills_cost - mil0[2],
        "deaths_cost_delta": snap_end.deaths_cost - mil0[3],
        "orders_delta": snap_end.order_count - mil0[4],
        "latency_ms_p50": round(statistics.median(_lat_real), 1),
        "latency_ms_p95": round(sorted(_lat_real)[max(0, int(len(_lat_real) * 0.95) - 1)], 1),
        "latency_note": "p50/p95 exclude fix_bypass rows",
        "fix_bypass_counts": {"#1": _fix1_n, "#2": _fix2_n},
        "state_budget": args.state_budget,
        "ablation": {
            "enabled": bool(args.ablation),
            "seed": int(args.ablation_seed or 0),
            "every": _abl_every,
            "offset": abl_offset,
            "n": len(_abl_rows),
            "n_voted": len(_abl_voted),
            "n_discarded": len(_abl_rows) - len(_abl_voted),
            "flips": _abl_flips,
            "flip_rate": round(_abl_flips / len(_abl_rows), 4) if _abl_rows else 0.0,
            "mean_kl_bits": round(sum(_abl_kl) / len(_abl_kl), 4) if _abl_kl else 0.0,
            "voted_flip_rate": round(_abl_vflips / len(_abl_voted), 4) if _abl_voted else 0.0,
            "voted_mean_kl_bits": round(sum(_abl_vkl) / len(_abl_vkl), 4) if _abl_vkl else 0.0,
        },
        "cost_cap": {"cap_usd": args.cost_cap, "hit": cap_hit,
                     "hit_at_i": cap_hit_i},
        "two_phase": {
            "enabled": bool(args.two_phase),
            "n": len(two_phase_lat),
            "disabled_at_i": two_phase_disabled_at,
            "p95_ms": round(sorted(two_phase_lat)[
                max(0, int(len(two_phase_lat) * 0.95) - 1)], 1) if two_phase_lat else 0.0,
        },
        "fanout": {
            "n": len(_fo_rows),
            "goal_counts": _goal_counts,
            "threat_true": sum(1 for _f in _fo_rows
                               if truthy_noul(_f.get("threat_recall"))),
            "routed": sum(1 for e in decisions_log if e.get("goal_route")),
        },
        "memory": {"enabled": bool(args.memory), "k": MEMORY_K},
        "harvesters_end": snap_end.harvester_count,
        "cash_end": snap_end.cash, "ore_end": snap_end.ore,
        "state_tokens_p50": statistics.median(state_toks) if state_toks else 0,
        "cost_usd_total": round(sum(costs), 6),
        "game_done": game_done, "game_result": game_result,
        "replay": replay_info, "replay_sha256": replay_hash,
        "orders_log": os.path.abspath(orders_path),
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    # Ground truth (audit.py): map-observed orders, not batch_ok counts.
    # Deploy record (if any) seeds the baseline so its take-effect delta is
    # not misattributed to a later decision. Pure offline diff, no server.
    try:
        from system1_commander.audit import audit_decisions
        _truth = audit_decisions(
            decisions_log,
            baseline=deploy_rec.to_dict() if deploy_rec is not None else None,
        )["summary"]
    except Exception:  # noqa: BLE001
        _truth = {}
    bench.update({
        "map_visible_orders": _truth.get("map_visible_orders", 0),
        "noop_ok_count": _truth.get("noop_ok", 0),
        "cash_spent": _truth.get("cash_spent", 0),
        "empty_ok_count": _truth.get("empty_ok", 0),
        "failed_real_count": _truth.get("failed", 0),
        "queue_peak_full": _truth.get("queue_peak_full", 0),
        "queue_peak_visible": _truth.get("queue_peak_visible", 0),
    })
    with open(bench_path, "w") as f:
        json.dump(bench, f, indent=2)
    print(f"[demo] bench -> {os.path.abspath(bench_path)}", flush=True)
    print(json.dumps(bench, indent=2))
    return bench


def main(argv=None) -> int:
    args = _parse_args(argv)
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("[demo] interrupted", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
