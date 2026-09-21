# NanoJev-max N-pre 双探针报告（early-exit）

- 日期：2026-09-21
- plan：`openra-taskC-nanojevmax-plan-20260921.md` rev3（NanoJev-max）
- 分支：`tom/taskC-max-impl`（worktree `tom-taskC-max`，基线 `d76e434`）
- 权重：`~/Github/openra-commander/.data/nanojev-unified-047b927`（sha `047b927`，`WEIGHTS.md` 门禁通过）
- 服务：`http://127.0.0.1:8932`（常驻，未重启；`/health ready=true`；全程 http，无 subprocess 回退行）
- 游戏服：`http://localhost:8001`
- python 偏离声明：指令指定的 `.venv-taskC` 缺 `openenv`，无法 import `demo_loop`；
  改用本 repo 现成 `~/Github/openra-commander/.venv/bin/python`（openenv+pytest 齐），
  未安装任何新依赖。tokenizer 计数用 `/tmp/nanojev-venv/bin/python`（仅 encode，不加载权重）。
  待 supervisor 确认。

## 1. N-pre 采集局（允许成本）

A1 `orders.jsonl` 不存 state 明文，无法重放 179 states。按 supervisor 指令打短收集局：

- 局：`.runs/nanojmax-npre-collect-20260921`（nanojev / mix eco,combat / 50 决策 / 50 ticks）
- bench：decisions=50（真投票 45 + bypass 5：#1×2、#2×3），wall=125.8s（stdout 时钟；bench `wall_s=123.7`，口径差 2.1s，不影响任何冻结判定），
  gate=fallback/downgrade（真 execute=0，与 A1 同构），map_visible_orders=2，
  empty_ok=45，failed_real=1，cash_spent=800，game_done=false（短局主动收尾）
- 产物：`orders.jsonl` + `states.jsonl`（50 states，`scripts/n_collect.py` tee 落盘）
  + `bench.json` + `replay.txt`（齐套；`.runs/` 不 commit，本地备查）
- 冒烟局：`.runs/nanojmax-npre-smoke-20260921`（2 决策，23s，仅验证收集链路）

## 2. (a) token 对账探针（`scripts/n_token_audit.py`，Qwen tokenizer 实测）

逐段 encode 精确复现 `prepare_examples`（`predict_toy_decisions.py:85-126`）。
max_length 卡的是**单候选 path**（max leaf），不是整请求。

| 书 | 口径 | 实测（50 states） |
|---|---|---|
| state 段 | `State:\n…\n` 段 Qwen tokens | eco ~109–171，combat ~74–76（Python 尺 `toks=` 约为其 0.7×，两把尺不对齐） |
| 单候选 path | max leaf（含 prefix+Candidate+Decision+eos） | **worst 213**（seq=16 eco tick=1066；eco path_max 全量 151–213，其中 seq≥14 后稳态 208–213；combat path_max 全量 118–120，112 为 path_min） |
| 整请求 | 单 tactic 问全候选 leaves 之和 | eco ~1888，combat ~706（仅参考，不进门控） |

### T1 冻结

- 当前穷状态 path_max=213，512 档余量 **299 tokens**。
- **T1 规则（冻结）**：富状态 path_max 必须 ≤512（默认档）；只在"翻转率/熵涨但疑似被截断"时
  逐档加 `--max-length`（上限 40960 由 backbone 卡死）。N1' 短局若开打，先用 512 档。
- 证据：`.runs/nanojmax-npre-collect-20260921/T1.json` + 上表（judge 可重跑脚本复算）。

## 3. (b) discrimination 探针（`scripts/n_discrimination.py`，冻结脚本，n=45）

重放 45 真投票行（replay conf 与原 conf 到小数点后 3 位完全一致，服务确定性完美）。
全量 probs 已落盘 `replay.jsonl`。

- **(a) 分布恒定检验**：top1 分布 `{build_weap:1, all_combat_attack_move:25, build_powr:19}`，
  熵 **1.118 bits** ≥ 0.5 → **AFFIRM（有变化）**。
  注记（不改判）：变化几乎全来自 kind 间差异；同 kind 内仍恒定
  （combat 25/25 同一候选，eco 19/20 同一候选）。冻结口径是整体熵，不量子集，故仍判 AFFIRM。
- **(b) conf-有效性检验**：conf 中位数 0.249；高半区（n=23）audit 有效率 **0.0%**，
  低半区（n=22）**9.1%**（2 effective 全在低半区：tick180 conf0.201、tick526 conf0.189），
  gap **−9.1pp** ≤ 5pp → **DENY（零相关，且方向反转）**。
  注记（不改判）：effective 事件仅 2/45，功效弱；且 audit verdict 归因于 gate 执行动作
  （本局全 fallback/scripted，无真 execute），不是 nanojev 投票本身——此局限 plan 已冻结"不重判"。
- **判定：任一否定 → EARLY-EXIT**（OR 关系，无第三态；实现者按冻结阈值判，judge 重算复核）。

## 4. 出口：转 §8-early（训练轨立项数据包）

按 plan §8-early，本探针后停止一切 prompt 实验（N1'–N4' 不做，不再打任何对局）。
可用数据仅三样（本地 `.runs/nanojmax-npre-collect-20260921/`，不 commit）：

1. token 对账表（§2 + `T1.json`，Qwen 尺）
2. discrimination 分布（§3：top1 分布 + 熵 1.118 + conf 中位 0.249 + gap −9.1pp）
3. 50 states 重放集（`states.jsonl`）+ 全分布标签（`replay.jsonl`，45 行全量 probs）
   —— plan 原文 179 states 因 A1 无 state 明文，实际为 50 states（45 真行），阈值未动。

另：§8-early 要求另立 micro 项 M-log 补 logging（`probs_full` + `history` 字段）——
本分支未做（`demo_loop.py`/`state.py` 属 Jev 轨禁碰文件），owner 待 supervisor 指派。

## 5. TBD 冻结值清单

| 编号 | 状态 |
|---|---|
| T1 富状态安全预算 | ✅ 冻结（§2：512 默认档，余量 299） |
| T2 两阶段 p95 上限 | ⏸️ 未测（early-exit，不打探针；owner=后续 full 路径） |
| T3 1问vs4问边际延迟 | ⏸️ 未测（同上；N3' backend 改造未做） |
| T4 N5' 分桶/显著性输入 | ⏸️ 不可达（N5' 需累计真行 ≥400；当前 45） |
| T5 advance/fetch 开销 | ⏸️ 未测（同上） |

## 6. 护栏与 cap

- 探针 cap 30min：未触发（收集局 126s + 重放 8s + token 审计 3s）。
- 单局 45min cap：N/A（未打全量局）。
- subprocess 回退：0 行（全程 http）。
- 超时行（>8000ms）：0 行（重放最大 1268ms）。

## 7. 本分支 commits（Nano 轨）

1. `45a4ad1` N0：gate downgrade 查 banned + `tests/test_gate_n0.py`（6 单测）
2. `79d4205` N-pre：`scripts/n_collect.py` 收集探针
3. `8730129` N-pre：`scripts/n_token_audit.py` + `scripts/n_discrimination.py`（冻结）
4. （本报告）N-pre 双探针报告

回归：pytest 15/15（N0 6 + atrack 6 + audit 3，主仓 .venv）。
N0 修只活本分支，不合 A 轨（plan N0 双条件未满足）。
