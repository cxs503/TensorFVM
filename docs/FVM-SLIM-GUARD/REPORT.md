# FVM-SLIM-GUARD：TensorFVM 仓库防腐三件套（D6）

mission m23951 · 2026-11-16 · 分支 agent/m23951 · 锚点 feature_ledger `FVM-SLIM-GUARD`

## 1. 案例介绍（背景）

TensorFVM 可达 pack 已达 **1.33 GiB**（`git count-objects -vH` 实测），
docs/ 下持续再生大 blob（.pt/.npz/.png），速率 25–35 MiB/日
（feature_ledger `FVM-SLIM-GUARD.why`：「再生 25-35MiB/日,6周长回1GiB」）。
瘦身方案（R2c/R3，见 `docs/REPO-SLIM-PLAN.md`）已复核成立但执行待 owner
裁决 D1–D5/D7；**在重写落地前，仓库会持续长胖**。本单交付防腐三件套：

| 件 | 文件 | 作用 |
|---|---|---|
| ① pre-commit 拦大 blob | `.githooks/pre-commit` + `scripts/guard/repo_guard.py staged` | 提交时拦截 >512 KiB 的暂存 blob |
| ② CI 体积门 | `.github/workflows/repo-guard.yml` + `repo_guard.py ci` | blob 门（diff 范围）+ pack 漂移门（基线+预算/绝对上限） |
| ③ MIGRATION 骨架 | `docs/MIGRATION.md` | 历史重写迁移 runbook 骨架（触发条件/执行/回滚） |

验收（ledger）：`{"gate":"512KiB 拦截+CI 阈值门"}`。

## 2. 怎么做的

- **单一事实源** `scripts/guard/repo_guard.py`（纯 stdlib，无 torch）：
  - `staged`：解析 `git diff --cached --raw` 的新 blob sha，`git cat-file -s`
    取大小，>512 KiB（524288 B）即 rc=1 并列出 offender；
    逃生门 `FVM_GUARD_ALLOW_LARGE=1`（须在 commit message 里写明理由）。
  - `ci`：(a) `base..HEAD` 变更文件中 >512 KiB 者 rc=1（base 解析
    `origin/main` → `HEAD~1`）；(b) `count-objects -vH` 的 size-pack 对
    `scripts/guard/pack_baseline.json` 基线：+50 MiB WARN、+150 MiB FAIL，
    另有 2 GiB 绝对上限兜底。
  - `reanchor`：历史重写后重记基线（MIGRATION §4.7）。
  - `report`：JSON 普查（供报告程序化注入）。
- **钩子接线**：`core.hooksPath=.githooks` 已在仓库配置中，`.githooks/pre-commit`
  为薄壳（cd 到 toplevel 后 exec 守卫脚本），chmod +x。
- **CI**：`repo-guard.yml`（push/PR/workflow_dispatch，`ci-loong64`，
  `fetch-depth: 0`，只跑 `repo_guard.py ci`，不装 torch）。
- **测试** `tests/test_repo_guard.py`：10 项，全部在 tmp 一次性 git 仓上
  用 `FVM_GUARD_REPO_ROOT` 环境覆盖驱动真实子进程。

## 3. 计算结果（程序化注入，零手抄）

| 量 | 值 | 来源 |
|---|---|---|
| pack 基线（reanchor） | 1 428 076 625 B = 1.33 GiB | `scripts/guard/pack_baseline.json` |
| WARN 阈值 | 1 480 505 425 B（基线+50 MiB） | `repo_guard.py ci` 输出 |
| FAIL 阈值 | 1 585 363 025 B（基线+150 MiB） | 同上 |
| 绝对上限 | 2 GiB | 同上 |
| 对象库垃圾 | 117 B | 同上 |
| 守卫测试 | **10 passed**（0.70 s） | `pytest tests/test_repo_guard.py -q` |
| e2e 探针（600 KiB 暂存） | commit 被拒 rc=1，HEAD 不变 | 本 worktree 实测 |
| e2e 小文件（100 B） | staged/ci 均 rc=0 | 同上 |
| 逃生门 | `FVM_GUARD_ALLOW_LARGE=1` → rc=0 | 单测 + 语义 |

## 4. 结果图

`figures/fig_pack_gate.png`：pack 体积门全景（基线/WARN/FAIL/绝对上限 vs
R2c/R3 瘦身目标）。**无场量云图**：本单为仓库基础设施任务，无物理场量；
比较曲线以「门阈值 vs 瘦身目标」条形图替代（matplotlib 生成）。

## 5. 注意点

- 钩子依赖 `core.hooksPath=.githooks`（已配置）；新克隆者若全局配置不同，
  需 `git config core.hooksPath .githooks`。
- 本 worktree 为 sparse-checkout + blob:none partial clone：探针文件须落在
  sparse cone 内（根目录路径会被拒）；`git cat-file -s` 对 promisor 对象
  只取大小、不拉内容，故 partial clone 下可用。
- 存量 >512 KiB 的已跟踪文件**不追溯**（CI blob 门只查 diff 范围），
  存量清理归 MIGRATION 重写（D1 裁决）。
- 基线漂移预算（+50/+150 MiB）按 25–35 MiB/日再生速率设定：无守卫时
  约 1.5–2 天触 WARN、4.5–6 天触 FAIL，足以逼出迁移裁决。
- 若 CI 环境为浅克隆且无 origin/main，blob 门降级跳过（有 WARNING 留痕），
  pack 门不受影响。

## 6. 误差判定

本单无物理量，<3% 判据不适用（原因：基础设施任务，无解析/文献基准）。
替代判据（逐项）：

| 判据 | 期望 | 实测 | 判定 |
|---|---|---|---|
| 512 KiB 拦截阈值 | 524288 B | 524288 B（`DEFAULT_BLOB_LIMIT`） | 一致 |
| 600 KiB 暂存 blob | 拦截 | rc=1，offender 列出 | 通过 |
| 100 B 暂存 blob | 放行 | rc=0 | 通过 |
| 600 KiB 已提交变更（CI 门） | 拦截 | rc=1 | 通过 |
| pack 漂移 0 B | OK | OK（drift 0 B） | 通过 |
| 阈值逻辑（OK/WARN/FAIL/兜底/无基线） | 5 分支 | 5/5 断言通过 | 通过 |
