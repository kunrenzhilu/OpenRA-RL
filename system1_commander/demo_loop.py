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
        list_combat_candidates,
        list_eco_candidates,
        state_to_json,
    )
    from system1_commander.executor import Executor
    from system1_commander.gate import GateConfig
    from system1_commander.state import Snapshot

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
    decisions_log = []
    gate_modes, kinds_used = [], []
    game_done, game_result, surrendered = False, "", False
    error_note = ""
    n_buildings_0, mil0 = 0, (0, 0, 0, 0, 0)
    last_snap, snap_end, replay_info = None, None, ""
    deploy_rec = None

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
                    state = builders[kind](snap)
                    state_json, toks = state_to_json(state)
                    state_toks.append(toks)
                    pred = await asyncio.to_thread(backend.predict, state, candidates)
                    latencies.append(pred.latency_ms)
                    costs.append(pred.cost_usd)
                    gate = apply_gate(pred, candidates, _make_backend("scripted"), state, gate_cfg)
                    gate_modes.append(gate.mode)
                    kinds_used.append(kind)
                    actions = by_name[gate.choice_name].actions(snap)
                    guard_actions = by_name["guard_choke"].actions(snap) if "guard_choke" in by_name else None
                    rec = await ex.execute(gate.choice_name, actions, snap.tick, guard_actions)
                    moved, interrupted = await ex.advance(args.ticks_per_decision)
                    rec.advance_ticks = moved
                    rec.advance_interrupted = interrupted
                    entry = {
                        "kind": "decision", "i": i, "tick": snap.tick,
                        "state_kind": kind,
                        "state_tokens": toks,
                        "prediction": pred.to_dict(), "gate": gate.to_dict(),
                        **rec.to_dict(),
                    }
                    f_orders.write(json.dumps(entry) + "\n")
                    f_orders.flush()
                    decisions_log.append(entry)
                    print(f"[demo] d{i}: tick={snap.tick} [{kind}] toks={toks} "
                          f"pred={pred.choice}@{pred.confidence:.2f} gate={gate.mode}:{gate.choice_name} "
                          f"batch_ok={rec.batch_ok} adv={moved}{'!' if interrupted else ''} "
                          f"{pred.latency_ms:.0f}ms ${pred.cost_usd:.6f}", flush=True)
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
        "ticks_total": snap_end.tick,
        "wall_s": round(wall_s, 1),
        "buildings_before": n_buildings_0, "buildings_after": len(snap_end.own_buildings),
        "units_after": len(snap_end.own_units),
        "enemies_visible_end": len(snap_end.enemies),
        "kills_delta": snap_end.kills - mil0[0], "losses_delta": snap_end.losses - mil0[1],
        "kills_cost_delta": snap_end.kills_cost - mil0[2],
        "deaths_cost_delta": snap_end.deaths_cost - mil0[3],
        "orders_delta": snap_end.order_count - mil0[4],
        "latency_ms_p50": round(statistics.median(latencies), 1) if latencies else 0.0,
        "latency_ms_p95": round(sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)], 1) if latencies else 0.0,
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
