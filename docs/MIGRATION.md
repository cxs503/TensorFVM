# TensorFVM MIGRATION 骨架（历史重写迁移 runbook）

> 状态：**骨架（skeleton）**。本文件是 FVM-SLIM-GUARD（feature_ledger，D6）
> 三件套之一：pre-commit 拦大 blob + CI 体积门 + 本迁移骨架。
> 详细决策与实测数字见 `docs/REPO-SLIM-PLAN.md`（m23832 复核，HEAD c656f8f9）。
> 本骨架只回答三件事：**何时触发、怎么执行、怎么回滚**。

## 1. 目的与触发条件

目的：把可达 pack 从当前 ~1.33 GiB 压到推荐档（R2c 保图 ≈ 382.6 MiB /
R3 最大收益 ≈ 63.9 MiB），并靠防腐三件套阻止再生（25–35 MiB/日）。

触发条件（满足其一即提请 owner 裁决执行）：

| # | 条件 | 当前值（2026-11-17 实测） |
|---|---|---|
| T1 | CI pack 门 WARN（基线 +50 MiB） | 基线 = 本单 `reanchor` 记录值 |
| T2 | CI pack 门 FAIL（基线 +150 MiB 或绝对上限 2 GiB） | 1.33 GiB |
| T3 | 单 blob > 512 KiB 的提交被 pre-commit 拦截 ≥ 3 次/周（说明再生未止） | 0（守卫刚上线） |
| T4 | owner 主动裁决（D1–D7 任一翻转） | 见 §3 |

## 2. 当前基线（实测）

| 指标 | 数值 | 取法 |
|---|---|---|
| 可达 pack（size-pack） | 1.33 GiB | `git count-objects -vH`（本单 reanchor 记录于 `scripts/guard/pack_baseline.json`） |
| 对象库垃圾 | ~0 MiB | 同上（D6-tmp_pack 已消解） |
| 推荐档 R2c（保 png） | 382.6 MiB（-72%） | `docs/REPO-SLIM-PLAN.md` §9 复核 |
| 最大收益档 R3（删全部二进制） | 63.9 MiB（-95%） | 同上 |
| 再生速率 | 25–35 MiB/日 | feature_ledger FVM-SLIM-GUARD.why |

## 3. 决策点 D1–D7（状态表，交 owner）

| # | 决策点 | 状态 |
|---|---|---|
| D1 | 删除范围（json/csv 是否同删） | 待决（建议 B 档保留清单） |
| D2 | 协调面（共享对象库影响） | 已部分消解 |
| D3 | 重写执行窗口 | 待决 |
| D4 | 远端分支清理（215 个） | 待决 |
| D5 | 验证提交保留策略 | 待决 |
| D6 | 对象库垃圾 gc | **已消解**（平台侧已回收，garbage≈0） |
| D7 | 共享 .git commit-graph 过期（fsck rc=16） | 待决（`git commit-graph write` 修复） |

## 4. 执行 runbook（骨架，细化在 m23832 runbook 上）

1. **冻结**：owner 宣布冻结窗口；期间禁 push（防腐三件套保持运行）。
2. **回滚镜像**：`git clone --mirror` → `git fsck --full` → `tar + sha256` 留档。
3. **scratch 克隆上重写**：`git filter-repo`（规则单一事实源 =
   `scripts/audit/measure_rewrite_pack.py`，与实测 pack 同一套 glob）。
4. **不变量校验**：src/tests 树去 pyc 后逐字节不变；`pytest -q` 全绿。
5. **体积对比**：重写前后 `count-objects -vH` 对账（目标见 §2）。
6. **force-push**：命令只打印，**永不自动推送**；owner 在冻结窗口手工执行。
7. **reanchor**：`python3 scripts/guard/repo_guard.py reanchor` 重记基线，
   防腐三件套继续运行。

## 5. 回滚

- 重写前镜像克隆 + tar+sha256 留档（§4.2）；
- 回滚 = 从镜像克隆恢复 + 重新 force-push 镜像内容；
- 不变量校验失败 → 丢弃 scratch 克隆，主仓零影响（重写只在 scratch 上进行）。

## 6. 状态

| 项 | 状态 |
|---|---|
| 防腐三件套（pre-commit / CI 门 / 本骨架） | 本单（m23951）交付 |
| 基线 reanchor | 本单执行（`scripts/guard/pack_baseline.json`） |
| 历史重写执行 | **未执行**（等 owner 裁决 D1–D5/D7） |
| D7 commit-graph 修复 | 待决（写共享 .git，需 owner 放行） |
