# FVM-SLIM-GUARD 验证欠账闭环（round 2）：防腐三件套独立复核 + 测试覆盖补齐

mission m24451 · 2026-11-17 · 分支 agent/m24451 · 锚点 feature_ledger `FVM-SLIM-GUARD`
HEAD `bb0397af`（origin/main，含 #1083 m24405 / #1084 m24430）

## 1. 案例介绍（背景）

TensorFVM 可达 pack 1.33 GiB，docs/ 大 blob 再生 25–35 MiB/日
（ledger `FVM-SLIM-GUARD.why`）。防腐三件套（D6）已由前序单交付并合并：

| 件 | 文件 | 交付单 |
|---|---|---|
| ① pre-commit 拦大 blob | `.githooks/pre-commit` + `scripts/guard/repo_guard.py staged` | #976/#989 (m23951/m24008) |
| ② CI 体积门 | `.github/workflows/repo-guard.yml` + `repo_guard.py ci`（blob/pack/live 三门） | 同上 |
| ③ MIGRATION 骨架 | `docs/MIGRATION.md` + `docs/MIGRATION-m23216.md` | 同上 |

验收（ledger）：`{"gate":"512KiB 拦截+CI 阈值门"}`。

**本单定位（验证欠账）**：m24405（#1083，11-17 14:17 合并）已做过一轮对抗核验
并修了 3 个缺陷（sparse 安全注入试验、HEAD-first gitattributes 测试、live 门
submodule 容错）。本单在其之上做**第二轮独立复核**，专攻前两轮未覆盖的三处：

1. **验收边界未钉死**：「512KiB 拦截」的精确边界（恰好 512 KiB 放行 /
   512 KiB+1 B 拦截）此前只有 600 KiB 探针，边界两侧无回归测试；
2. **`reanchor` / `report` 两个子命令零测试**：MIGRATION runbook §4.7 依赖
   `reanchor` 重记基线，但无任何测试保证其行为；
3. **CI blob 门的 base 解析路径未测**：此前测试只覆盖 `HEAD~1` 回退路径，
   真实 CI 走的 `origin/main` 路径无回归测试。

## 2. 怎么做的

- **基线对齐**：本 worktree 初始 HEAD 为 api-sync 快照（28588f34，树=686b8fb1，
  早于 #1083）。为避免 PR 反向回退 #1083 的修复，先把分支移到 origin/main
  （bb0397af）再开工（`git checkout --detach` + `git branch -f`，未 push）。
- **测试补齐**（`tests/test_repo_guard.py`，+5 项，全部在 tmp 一次性 git 仓上
  驱动真实子进程，stdlib+git，无 torch）：
  - `test_staged_boundary_exactly_limit_passes`：恰好 524288 B → rc=0；
  - `test_staged_boundary_one_byte_over_blocks`：524289 B → rc=1 且点名 offender；
  - `test_ci_blob_gate_uses_origin_main_base`：构造 origin/main=seed、
    HEAD~1=大文件、HEAD=小文件的场景——两候选 base 给出相反裁决，
    rc=1 证明走的是 origin/main 而非 HEAD~1；
  - `test_cmd_reanchor`：repack 后 `reanchor` 重记 pack+live 基线，
    两 JSON 均含 `pack_bytes`/`live_bytes`>0 与 `measured_at`；
  - `test_cmd_report`：`report` 输出含全部 10 个文档化键，
    `blob_limit_bytes==524288`。
- **真实仓实跑**（`docs/FVM-SLIM-GUARD-VERIFY-M24451/collect_evidence.py`，
  程序化采集，零手抄）：`staged`/`ci`/`live`/`report` 四子命令 +
  `injection_test.py` e2e 探针 + `pytest tests/test_repo_guard.py`，
  全部原始 rc/JSON 落 `evidence.json`。
- **图**：`figures/fig_guard_state.png`（matplotlib，数字全部读自
  evidence.json）。

## 3. 计算结果（程序化注入，零手抄；来源 evidence.json @ 2026-11-17T16:48Z）

| 量 | 值 | 判定 |
|---|---|---|
| staged 门（真实仓，空暂存区） | rc=0 | PASS |
| CI blob 门（base=origin/main，limit 512 KiB） | rc=0 | PASS |
| pack 门 | 1 438 814 044 B（1.34 GiB），基线 1 428 076 625 B，漂移 **+10.2 MiB** | OK（WARN 线 +50 MiB） |
| live 门 | 197 562 552 B（188.4 MiB），基线 197 122 081 B，漂移 **+430 KiB** | OK（WARN 线 +50 MiB） |
| 对象库垃圾 | 117 B | — |
| e2e 注入探针（真实钩子接线） | 4/4：600 KiB 拦截 / 100 B 放行 / 白名单 npz 放行 / 逃生门放行 | PASS |
| 守卫测试套件 | **22 passed**（1.6 s；前序 17 + 本单 5） | PASS |
| 钩子接线 | `core.hooksPath=.githooks`，`.githooks/pre-commit` 已跟踪 | PASS |

再生速率复核：基线 11-16 20:08Z → 本单 11-17 16:48Z（≈20.7 h），
pack +10.2 MiB ≈ **23.5 MiB/日**——落在 ledger 预估带（25–35 MiB/日）下沿，
按此速率约 2.1 天触 pack WARN、6.4 天触 FAIL，与 MIGRATION.md §1 触发条件
T1/T2 的裁决窗口一致。

## 4. 结果图

`figures/fig_guard_state.png`：A=pack 漂移门（基线/WARN/FAIL/绝对上限 vs
实测）；B=live 体积门（基线/WARN/RED vs 实测）；C=e2e 注入探针 4 用例。
**无场量云图**：本单为仓库基础设施任务，无物理场量；比较曲线以
「门阈值 vs 实测体积」条形图替代（与 m23951 同口径）。

## 5. 注意点

- 本单 worktree 为 sparse-checkout（cone 不含 `.githooks/`、`.github/`、
  `.gitattributes`）：接线类断言一律从 HEAD 读（`git show`），与 #1083 的
  策略一致；本单新增 5 项测试全部在 tmp 仓上跑，不依赖 worktree 物化状态。
- `reanchor` 会**改写** `scripts/guard/*.json` 基线——本单只在 tmp 仓上测，
  真实仓基线未动（重写落地前不应重锚）。
- CI 门在浅克隆且无 origin/main 时 blob 门降级跳过（WARNING 留痕），
  pack/live 门不受影响；本单实跑走的是完整 `origin/main` 路径。
- 存量 >512 KiB 已跟踪文件不追溯（CI blob 门只查 diff 范围），存量清理归
  MIGRATION 重写（D1 裁决，owner 待决）。
- 本单未执行任何历史重写/force-push（MIGRATION §4.6：命令只打印，永不自动推送）。

## 6. 误差判定

本单无物理量，<3% 判据不适用（原因：仓库基础设施任务，无解析/文献基准）。
替代判据（逐项，全部 PASS）：

| 判据 | 期望 | 实测 | 判定 |
|---|---|---|---|
| 恰好 512 KiB 暂存 | 放行（rc=0） | rc=0 | PASS |
| 512 KiB+1 B 暂存 | 拦截（rc=1） | rc=1，点名 offender | PASS |
| CI blob 门 base=origin/main | 用 origin/main 裁决 | rc=1（反证 HEAD~1 会 rc=0） | PASS |
| reanchor 重记双基线 | 两 JSON 合法且 >0 | 合法，>0 | PASS |
| report 普查键 | 10 键齐全 | 10/10 | PASS |
| 真实仓 ci 三门 | rc=0 | rc=0 | PASS |
| e2e 钩子拦截 | 600 KiB commit 被拒 | rc=1，HEAD 不变 | PASS |

**结论**：验收「512KiB 拦截 + CI 阈值门」在真实仓上成立且边界行为已钉死；
三件套保持运行，pack 漂移 +10.2 MiB（<WARN 线），仓库仍在缓慢长胖，
MIGRATION 触发条件 T1 尚未触发，D1–D5/D7 仍待 owner 裁决。
