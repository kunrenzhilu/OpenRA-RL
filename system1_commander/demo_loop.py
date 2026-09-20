"""Decision loop: every N ticks -> state -> backend -> gate -> batch -> advance.

Usage:
    PYTHONPATH=<worktree> <venv>/bin/python -m system1_commander.demo_loop \\
        --backend scripted --state combat --ticks-per-decision 25 \\
        --max-decisions 4 --log-dir .runs/system1-p0

Outputs in log-dir: orders.jsonl (one record per decision), bench.json
(summary), replay.txt (replay path + hash). Replays (*.orarep) and *.jsonl
are never committed.
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
    p.add_argument("--backend", choices=["scripted", "jev"], default="scripted")
    p.add_argument("--state", choices=["combat", "eco"], default="combat")
    p.add_argument("--ticks-per-decision", type=int, default=25)
    p.add_argument("--max-decisions", type=int, default=4)
    p.add_argument("--log-dir", default=".runs/system1-p0")
    p.add_argument("--url", default="http://localhost:8000")
    return p.parse_args(argv)


def _make_backend(name: str):
    if name == "scripted":
        from system1_commander.backend_scripted import ScriptedBackend
        return ScriptedBackend()
    if name == "jev":
        from system1_commander.backend_jev import JevBackend
        return JevBackend()
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

    os.makedirs(args.log_dir, exist_ok=True)
    orders_path = os.path.join(args.log_dir, "orders.jsonl")
    bench_path = os.path.join(args.log_dir, "bench.json")
    replay_path_txt = os.path.join(args.log_dir, "replay.txt")

    from openra_env.mcp_ws_client import OpenRAMCPClient

    backend = _make_backend(args.backend)
    candidates = list_combat_candidates() if args.state == "combat" else list_eco_candidates()
    by_name = {c.name: c for c in candidates}
    build_state = build_combat_state if args.state == "combat" else build_eco_state

    t_start = time.monotonic()
    latencies, costs, state_toks = [], [], []
    decisions_log = []
    game_done, game_result = False, ""

    async with OpenRAMCPClient(base_url=args.url, message_timeout_s=300.0) as client:
        ex = Executor(client)
        print(f"[demo] reset @ {args.url} ...", flush=True)
        await client.reset()
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

            for i in range(args.max_decisions):
                gs, units, buildings = await ex.fetch_raw()
                snap = build_snapshot(gs, units, buildings)
                if isinstance(gs, dict) and gs.get("done"):
                    game_done, game_result = True, str(gs.get("result", ""))
                    print(f"[demo] game over: {game_result}", flush=True)
                    break
                state = build_state(snap)
                state_json, toks = state_to_json(state)
                state_toks.append(toks)
                pred = await asyncio.to_thread(backend.predict, state, candidates)
                latencies.append(pred.latency_ms)
                costs.append(pred.cost_usd)
                gate = apply_gate(pred, candidates, _make_backend("scripted"), state, GateConfig())
                actions = by_name[gate.choice_name].actions(snap)
                guard_actions = by_name["guard_choke"].actions(snap) if "guard_choke" in by_name else None
                rec = await ex.execute(gate.choice_name, actions, snap.tick, guard_actions)
                moved, interrupted = await ex.advance(args.ticks_per_decision)
                rec.advance_ticks = moved
                rec.advance_interrupted = interrupted
                entry = {
                    "kind": "decision", "i": i, "tick": snap.tick,
                    "state_tokens": toks,
                    "prediction": pred.to_dict(), "gate": gate.to_dict(),
                    **rec.to_dict(),
                }
                f_orders.write(json.dumps(entry) + "\n")
                f_orders.flush()
                decisions_log.append(entry)
                print(f"[demo] d{i}: tick={snap.tick} toks={toks} "
                      f"pred={pred.choice}@{pred.confidence:.2f} gate={gate.mode}:{gate.choice_name} "
                      f"batch_ok={rec.batch_ok} adv={moved}{'!' if interrupted else ''} "
                      f"{pred.latency_ms:.0f}ms ${pred.cost_usd:.6f}", flush=True)
                if game_done:
                    break

        gs, units, buildings = await ex.fetch_raw()
        snap_end = build_snapshot(gs, units, buildings)
        replay_info = ""
        try:
            r = await ex.tool("get_replay_path")
            replay_info = r if isinstance(r, str) else json.dumps(r)
        except Exception as e:  # noqa: BLE001
            replay_info = f"get_replay_path failed: {e}"

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
        "backend": args.backend, "state_kind": args.state,
        "ticks_per_decision": args.ticks_per_decision,
        "decisions": len(decisions_log),
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
