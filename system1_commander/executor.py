"""Choice -> batch([...]) -> advance(ticks) against OpenRAMCPClient.

`*_id` params go out as int; `unit_ids` stays a string selector.
`FAILED` / `no valid commands` batch replies are normal responses: retry
once, then re-dispatch as guard.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

FAILURE_MARKERS = ("FAILED", "no valid commands", "No valid commands", "error")

ADVANCE_CHUNK = 50  # server clamps advance to [1,50]


def _is_failure(reply: Any) -> bool:
    text = reply if isinstance(reply, str) else json.dumps(reply, default=str)
    return any(m in text for m in FAILURE_MARKERS)


def normalize_actions(actions: list[dict]) -> list[dict]:
    norm = []
    for a in actions:
        b = dict(a)
        for k, v in list(b.items()):
            if k.endswith("_id") and k != "unit_ids" and isinstance(v, str) and v.isdigit():
                b[k] = int(v)
        norm.append(b)
    return norm


@dataclass
class ExecRecord:
    tick: int = 0
    choice: str = ""
    actions: list = field(default_factory=list)
    batch_ok: bool = False
    batch_note: str = ""
    retried: bool = False
    fell_back_to_guard: bool = False
    advance_ticks: int = 0
    advance_interrupted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "tick": self.tick,
            "choice": self.choice,
            "actions": self.actions,
            "batch_ok": self.batch_ok,
            "batch_note": self.batch_note[:500],
            "retried": self.retried,
            "fell_back_to_guard": self.fell_back_to_guard,
            "advance_ticks": self.advance_ticks,
            "advance_interrupted": self.advance_interrupted,
        }


class Executor:
    def __init__(self, client):
        self.client = client

    async def tool(self, name: str, **kwargs) -> Any:
        return await self.client.call_tool(name, **kwargs)

    async def fetch_raw(self) -> tuple[dict, list, list]:
        # NOTE: deployed fix5 server only exposes get_game_state summaries
        # (no get_units/get_buildings tools); unit lists come from
        # units_summary/buildings_summary/enemy_summary keys.
        gs = await self.tool("get_game_state")
        if isinstance(gs, str):
            gs = json.loads(gs)
        return gs, None, None

    async def advance(self, ticks: int) -> tuple[int, bool]:
        moved, interrupted = 0, False
        while moved < ticks:
            chunk = max(1, min(ADVANCE_CHUNK, ticks - moved))
            try:
                r = await self.tool("advance", ticks=chunk)
                d = json.loads(r) if isinstance(r, str) else (r or {})
            except Exception:  # noqa: BLE001
                break
            moved += int(d.get("actual_ticks_advanced", chunk) or chunk)
            if d.get("interrupted"):
                interrupted = True
                break
            if d.get("done") or d.get("game_over"):
                break
        return moved, interrupted

    async def batch(self, actions: list[dict]) -> tuple[bool, str]:
        if not actions:
            return True, "empty batch (advance only)"
        try:
            r = await self.tool("batch", actions=normalize_actions(actions))
        except Exception as e:  # noqa: BLE001
            return False, f"batch exception: {e}"
        text = r if isinstance(r, str) else json.dumps(r, default=str)
        if _is_failure(r):
            return False, text[:500]
        return True, text[:500]

    async def execute(self, choice_name: str, actions: list[dict], tick: int,
                      guard_actions: list[dict] | None = None) -> ExecRecord:
        rec = ExecRecord(tick=tick, choice=choice_name, actions=actions)
        ok, note = await self.batch(actions)
        rec.batch_ok, rec.batch_note = ok, note
        # NanoJev plan A3: same-action retry deleted (a second identical
        # batch after FAILED can never succeed and wastes 1 RTT; it caused
        # half the 130 identical deaths). Any retry must come from the
        # caller with a swapped action via a new execute() call. `retried`
        # stays in the schema (always False) for bench compatibility.
        # Guard re-dispatch below is a *different* action, so it stays.
        if not ok and guard_actions:
            ok3, note3 = await self.batch(guard_actions)
            rec.batch_ok = ok3
            rec.batch_note = f"guard fallback: {note3}"
            rec.fell_back_to_guard = True
        return rec

    async def ensure_deployed(self, snap) -> ExecRecord | None:
        """Deploy MCV if needed. Returns the deploy record or None."""
        if snap.mcv_id is None or snap.has_fact:
            return None
        rec = ExecRecord(tick=snap.tick, choice="deploy_mcv",
                         actions=[{"tool": "deploy_unit", "unit_id": int(snap.mcv_id)}])
        ok, note = await self.batch(rec.actions)
        rec.batch_ok, rec.batch_note = ok, note
        await self.advance(25)
        return rec
