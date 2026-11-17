# FVM-SLIM-GUARD 验收核验与防腐三件套加固报告

> 任务 m24405（锚点 `FVM-SLIM-GUARD`，D6 三件套）· 分支 `agent/m24405`
> 结论：**三件套在仓内已完整落地且经独立对抗核验有效**；本次核验发现并修复 3 个真实缺陷
> （2 个稀疏检出健壮性 + 1 个潜在 CI 崩溃向量），全部测试绿。
> 机器证据：`docs/FVM-SLIM-GUARD-VERIFY/evidence.json`、`injection-test.json`。

## 1. 案例介绍（背景与对象）

TensorFVM 仓库可达 pack 已达 **1.34 GiB**，且历史观测再生速率 **25–35 MiB/日**
（feature_ledger FVM-SLIM-GUARD.why：6 周即可长回 1 GiB）。防腐三件套（D6）是
历史瘦身方案（`docs/REPO-SLIM-PLAN.md` / `docs/MIGRATION.md`）的**守门配套**：

| # | 组件 | 接线 | 引擎 |
|---|---|---|---|
| 1 | pre-commit 拦大 blob（512 KiB） | `.githooks/pre-commit` + `core.hooksPath=.githooks`（主仓已设置） | `repo_guard.py staged` |
| 2 | CI 体积阈值门 | `.github/workflows/repo-guard.yml` | `repo_guard.py ci`（blob 门 + pack 漂移门 + live 体积门） |
| 3 | MIGRATION 骨架 | `docs/MIGRATION.md`、`docs/MIGRATION-m23216.md` | 触发条件 T1–T4 / 基线 / 决策点 D1–D7 / runbook / 回滚 |

白名单机制：`.gitattributes` 的 `guard-allow-large` 属性（可复现生产工件族
npz/png/svg/pdf/msh/gz/csv）；逃生舱 `FVM_GUARD_ALLOW_LARGE=1`（须在 commit message 说明理由）。

## 2. 怎么做的（核验方法）

本单接手时账面状态为 `in_progress` 但仓内实现已存在，故本单工作 = **独立对抗核验 + 缺陷修复**：

1. 通读 `repo_guard.py`（450 行）全部模式（staged/ci/live/report/reanchor）与接线文件；
2. 复核阈值语义与验收逐条对表（512 KiB 拦截 / CI 阈值门 / MIGRATION 骨架五节齐全）；
3. 在本工作树实跑三道验证：pytest 守卫套件、ruff lint、端到端注入试验
   （临时一次性 git 仓接线**真实**钩子后注入 blob 跑 `git commit`，不触碰真实历史）；
4. 对发现的每个失败给出最小修复 + 回归测试，重跑至全绿。

## 3. 核验结果（程序化取证，零手抄）

### 3.1 修复前基线（复现的失败）

| 缺陷 | 位置 | 症状 | 证据 |
|---|---|---|---|
| A | `scripts/guard/injection_test.py::_setup_guard` | 稀疏工作树下 `FileNotFoundError: .gitattributes`（`shutil.copy2` 直拷未检出的接线文件，违背其自身 docstring 承诺的 HEAD 读取策略） | 注入试验崩溃 traceback |
| B | `tests/test_repo_guard.py::test_gitattributes_whitelist_present` | 断言工作树存在 `.gitattributes` → 稀疏检出下假失败（**17 passed, 1 failed**） | pytest 输出 |
| C | `scripts/guard/repo_guard.py::live_artifact_bytes` | `git ls-tree -r -l` 对子模块条目 size 打印 `-`，`int("-")` 抛 ValueError → 一旦加入子模块整个 CI 体积门崩溃（潜伏缺陷） | 代码审查 + monkeypatch 回归测试 |

### 3.2 修复后（全部绿）

| 验证 | 结果 |
|---|---|
| `pytest tests/test_repo_guard.py tests/test_slim_runbook_globs.py` | **19 passed, 2 skipped**（修复前 17+1 failed） |
| `ruff check`（3 个改动文件） | All checks passed |
| 注入试验（端到端，4 用例） | **all_pass = true**（见 3.3） |
| `repo_guard.py ci`（真实仓实跑） | rc=0：pack 1.3 GiB（基线漂移 +10.2 MiB < +50 MiB WARN）OK；live 188.3 MiB（漂移 +0.3 MiB）OK |
| `repo_guard.py staged` | rc=0（512 KiB 门就绪） |

### 3.3 注入试验明细（真实钩子接线，临时仓）

| 用例 | 注入 | 大小 | 钩子行为 | 期望 | 判定 |
|---|---|---|---|---|---|
| large-unlisted | `big.bin` | 600 KiB | 拦截 | 拦截 | PASS |
| small | `small.txt` | 0.1 KiB | 放行 | 放行 | PASS |
| large-whitelisted-npz | `docs/verification-x/f/fields.npz` | 600 KiB | 放行（`guard-allow-large`） | 放行 | PASS |
| large-unlisted-docs | `docs/verification-x/f/fields.bin` | 600 KiB | 拦截 | 拦截 | PASS |

## 4. 结果图说明

本单为**仓库工具/门禁核验任务**，无物理场数据，故无云图/曲线图；定量证据以
§3 表格 + `evidence.json` / `injection-test.json`（机器可读）承载。

## 5. 注意点与局限

- **稀疏检出陷阱**：本仓工作树为 sparse-checkout（`.githooks/`、`.github/`、`.gitattributes`
  不落地）。任何接线类测试/脚本必须从 HEAD（`git show`）读取，不能假设工作树文件存在——
  本次缺陷 A/B 均源于此。CI 全量检出不受影响，但本地稀疏工作树的开发者会踩雷。
- 子模块 size 容错已修（缺陷 C）；当前仓无子模块，属预防性加固。
- pack/live 基线在**任何历史重写后必须 `repo_guard.py reanchor` 重锚**（MIGRATION 骨架已写明）。
- `resolve_base()` 优先 `origin/main`、退 `HEAD~1`：浅克隆且无 origin 时 blob 门降级为
  警告跳过（设计如此，已在输出中显式声明）。

## 6. 误差判定

本单验收为**二值门禁语义**（拦截/放行、门开/门关），无物理基准量可比误差：
- 「512 KiB 拦截」：注入试验 4/4 用例按预期触发/放行 → **满足（非 <3% 误差类指标）**；
- 「CI 阈值门」：三道门（blob/pack/live）真实仓实跑 rc=0，阈值语义经纯函数单测覆盖 → **满足**；
- 无基准须声明原因：门禁类验收不存在实验/文献基准，以上述二值判定替代。
