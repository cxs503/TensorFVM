# PR #175 修红报告 — TensorFVM（mission m4493）

## 1 案例介绍（背景与对象）

- 仓库：TensorFVM（PyTorch 有限体积不可压求解器库，`src` + `tests` + `scripts`）。
- 任务对象：PR #175（分支 `agent/m4249`，内容 = lint 门拆分：`pyproject.toml` 声明
  `[tool.ruff.lint].select = ["E9","F63","F7","F82"]`、`scripts/lint_ratchet.py` 变更文件棘轮、
  `.github/workflows/ci.yml` 把 lint 门拆成阻塞 + informational、`tests/test_lint_ratchet.py` 契约测试、
  `docs/ci-lint-gate/*`）。
- CI 症状：`test (3.11)` = failure（`lint (advisory)` 不在本单症状内）。
- 参数/环境：本地 Python 3.11.6、pytest 9.2.0、ruff 0.15.0、torch 2.7.1；
  工作树 = `origin/main`（ac7c6db6）+ PR #175 head（6dfd8692）的合并态，
  即 GitHub `pull_request` 事件检出的同一棵树。

## 2 怎么做的（方法/流程）

1. 取 PR head：`git fetch origin agent/m4249`，用 `git merge FETCH_HEAD` 把 PR 内容并入工作分支，
   复现 CI 的「PR + 当前 main」合并态（不用 `reset --hard`，工作树安全策略禁止）。
2. 因为无 `gh` CLI、GitHub API 匿名访问 403、无可用 token，**无法读取 CI 日志**，
   故改为本地全量复现：`pytest -q --timeout=1800`（与 CI 的 `test` job 同命令、同 Python 3.11）。
3. 全量跑到 ~30% 出现唯一 `F`。二分定位（按 `--collect-only` 顺序切块重跑）命中
   `tests/test_multi_grid_gci.py::test_reference_agreement_accepts_a_multi_grid_study`。
4. 读栈：`reference_agreement()`（`src/tensorfvm/verification/richardson.py:242`）读取
   `study.fallback_reason`，而多网格路径的 `MultiGridStudy`（`src/tensorfvm/verification/multi_grid.py:63`）
   没有该字段 → `AttributeError`。单网格 `GCIStudy` 有该字段，多网格漏了。
5. 最小修复（只动 1 个源文件，9 行含注释）：
   - `MultiGridStudy` 增补 `fallback_reason: str = "none"`（带默认值，位置构造向后兼容）；
   - `multi_grid_order()` 在构造时填入真实归因：拟合成功 `"none"`、误差非单调 `"non-monotone"`、
     拟合失败 `"unfittable"`——与单网格路径的 `non-monotone` / `implausible-order` 语义对齐，
     使 GCI 覆盖判定始终带着「该带宽是否来自降级规则」的限定语。

## 3 计算结果（程序化实测）

| 检查 | 修复前 | 修复后 |
|---|---|---|
| `pytest tests/test_multi_grid_gci.py::test_reference_agreement_accepts_a_multi_grid_study` | 1 failed（AttributeError: fallback_reason） | 1 passed |
| `pytest tests/test_multi_grid_gci.py tests/test_gci_fallback_band.py tests/test_richardson_gci.py tests/test_gci_band_coverage.py tests/test_lint_ratchet.py tests/test_multi_element_flow.py` | — | **88 passed** |
| `pytest tests/test_report_traceability.py tests/test_evidence_checkout_policy.py tests/test_lint_ratchet.py`（PR 自带文档/契约面） | — | 21 passed |
| `pytest tests/test_multi_element_* tests/test_multi_grid_gci.py tests/test_multi_solid_overset.py tests/test_mutation_*` | 1 failed / 74 passed | 0 failed |

修复前后均为同一条命令、同一解释器（Python 3.11.6），除 `multi_grid.py` 外无文件改动。

## 4 结果图

无图。理由：本单交付物是**后端/CI 缺陷修复**（一个缺失的 dataclass 字段 + 归因赋值），
没有场量、网格收敛曲线或力学量可绘制；关键量是「测试通过/失败」与「异常类型」，
已在上表逐项程序化列出（零手抄）。按报告规范写明理由，不用装饰性图充数。

## 5 注意点（坑/局限/适用范围）

- **环境差异坑（本轮真实踩到，值得后来者复用）**：本地全量跑得到唯一 F，但它与 PR 的 lint 改动无关，
  是 main 上既有的多网格 GCI 缺陷；CI 只在 `test (3.11)` 暴露，因为该 job 会装 gmsh、跑全量测试。
- `git reset --hard` 被工作树安全策略拦截；复现 PR 合并态请用 `git merge FETCH_HEAD`。
- 远程复现类任务在无 `gh`/无 token 时读不到 check-run 日志：先用
  `pytest --collect-only -q` 取得顺序表，再按块二分，比盲跑全量省一个数量级时间；
  后台跑套件必须 `setsid`（否则 shell 120s 超时会把进程组整组收割，日志会在中途"假死"）。
- 局限：本地无 `gmsh`（CI 装了），`tests/test_naca_overset.py`、`tests/test_gmsh_external.py` 等
  gmsh 依赖用例在本地是 skip、在 CI 会真跑；本单未得到它们的 CI 结论。
  另机器被多个并发任务压满 CPU，全量套件在本单窗口内未能跑完，故以「定位到唯一已知失败 + 二分块全绿」为证。
- 适用范围：`MultiGridStudy.fallback_reason` 只新增字段，未改变任何 GCI 数值、带宽与判据逻辑，
  对既有调用方向后兼容。

## 6 误差判定

关键量是布尔/计数型（测试退出码、失败用例数、异常类型），基准 = 同版本 pytest 对同一棵树的独立复算：

| 关键量 | 基准 | 实测 | 相对误差 | <3%? |
|---|---|---|---|---|
| 目标用例失败数 | 0（期望） | 0 | 0% | 是 |
| 相关 6 文件套件通过数 | 88 | 88 | 0% | 是 |
| 二分块通过数 | 74（修复后 75 = 74+1） | 75 | 0% | 是 |

honest-FAIL 声明：**未**直接观察到 GitHub 上 PR #175 的 `test (3.11)` 变绿——
本环境无 GitHub 凭据，check-run 不可读；本单证据是「本地按 CI 同一命令复现红 → 定位 → 修复 → 同命令转绿」。
全量 1459 个用例在本机并发压力下未能一次跑完，故不宣称「全量零失败」。