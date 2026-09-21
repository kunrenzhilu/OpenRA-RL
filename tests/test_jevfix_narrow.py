"""Jevfix narrowing 备选局单测（纯函数，零成本）。

覆盖任务书三项：缩票集成分、5 tick 到期重投、到期/落地边界。
只测新增独立路径（narrow_* / build_narrowed_ballot），不碰 F1/F2 实现。
"""

from system1_commander.candidates import (
    NARROW_HORIZON,
    build_narrowed_ballot,
    list_eco_candidates,
    narrow_next,
    narrow_target_for_macro,
)
from system1_commander.state import Snapshot


def _snap(**kw) -> Snapshot:
    base = dict(base_cell=(45, 85), available_production=[],
                ready_to_place=[], production=[])
    base.update(kw)
    return Snapshot(**base)


def _by_name():
    return {c.name: c for c in list_eco_candidates()}


# ── 缩票集成分 ────────────────────────────────────────────────────

def test_horizon_is_five():
    assert NARROW_HORIZON == 5


def test_narrow_ballot_core_pair():
    by = _by_name()
    ballot = build_narrowed_ballot("build_weap", _snap(), by)
    assert [c.name for c in ballot] == ["build_weap", "wait"]


def test_narrow_ballot_adds_place_when_blocked():
    by = _by_name()
    s = _snap(ready_to_place=["powr"])
    ballot = build_narrowed_ballot("build_weap", s, by)
    assert [c.name for c in ballot] == ["build_weap", "place_ready", "wait"]


def test_narrow_target_structural_macro():
    assert narrow_target_for_macro("fast_weap", _snap()) == "build_weap"
    assert narrow_target_for_macro("open_powr", _snap()) == "build_powr"


def test_narrow_demand_proxy_enters_ballot():
    # 不可造宏 → build_weap 进缩票集（与 F6 demand-proxy 同条件）。
    assert narrow_target_for_macro("econ_harv", _snap()) == "build_weap"
    assert narrow_target_for_macro("armor_push", _snap()) == "build_weap"
    by = _by_name()
    ballot = build_narrowed_ballot(
        narrow_target_for_macro("econ_harv", _snap()), _snap(), by)
    assert ballot[0].name == "build_weap"
    # 可造时走默认映射。
    assert narrow_target_for_macro(
        "econ_harv", _snap(available_production=["harv"])) == "train_harv"
    assert narrow_target_for_macro(
        "armor_push", _snap(available_production=["1tnk"])) == "train_1tnk"


# ── 5 tick 到期重投 ───────────────────────────────────────────────

def test_five_ticks_expire():
    rem = NARROW_HORIZON
    for k in range(5):
        rem, rel = narrow_next(remaining_before=rem, landed=False)
        if k < 4:
            assert rel is None and rem == NARROW_HORIZON - k - 1
        else:
            assert (rem, rel) == (0, "expired")


# ── 到期/落地边界 ─────────────────────────────────────────────────

def test_landed_first_vote_releases_at_once():
    # 首票即落地 → 立即重投宏，剩余 budget 丢弃。
    assert narrow_next(remaining_before=5, landed=True) == (0, "landed")


def test_landed_on_last_tick_wins_over_expiry():
    # 第 5 票落地 → 记 landed（不记 expired），优先级冻结在此。
    assert narrow_next(remaining_before=1, landed=True) == (0, "landed")
    assert narrow_next(remaining_before=1, landed=False) == (0, "expired")


def test_non_target_vote_never_releases():
    # wait/place 票（landed=False）不释放承诺。
    rem, rel = narrow_next(remaining_before=3, landed=False)
    assert (rem, rel) == (2, None)
