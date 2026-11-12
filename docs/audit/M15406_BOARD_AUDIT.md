# m15406 交差核验报告 — CASE-CYLINDER-DRAG-FVM（TensorFVM）

> 核验对象：m15406（succeeded）· 交付 commit `226afd1e6538`（分支 `agent/m15406`，PR #434）
> 核验单：m15556 · 裁定：**L1（单元/契约验证）** — 基准腿 0 个 got 数，L2 的 run_artifacts 缺失
> 生成方式：本文件由 `docs/audit/m15406/audit_m15406.py` 程序化生成，数字零手抄。

## ① 案例背景与参数（被核验算例是什么）

二维通道内静止圆柱绕流，Schäfer & Turek 1996 DFG 2D-1：域 2.2×0.41 m，圆心 (0.2,0.2)，直径 0.1 m，抛物线入口 U_mean=0.2 m/s，ρ=1、ν=1e-3 m²/s，Re_D=20 定常层流。参考量三个：C_D、C_L、p_diff。

| 量 | 参考值（机器解析） |
|---|---|
| drag | 5.57953523384 |
| lift | 0.010618948146 |
| pressure_difference_pa | 0.11752016697 |

来源页：https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark1_re20.html
归档 sha256：`371298dc64c409043d69bdb980c972f7df8b6eb33cd7d3296c887de7cbc6472d`；`reference.json` sha256：`b8eff73b6e6111421ab52d8555adb357f173912f36eb81151e3b5d07c04562f6`；与包内常量漂移：{'drag': 0.0, 'lift': 0.0, 'pressure_difference_pa': 0.0}。

## ② 核验方法（这是交差单，不是重做单）

1. 从交付 commit 取字节，逐文件 sha256 归档（`docs/audit/m15406/delivered/`）；
2. 三件套机器核查：run_artifacts 文件是否存在、benchmark 登记行与 ref/got 数值、repro_cmd 是否可执行；
3. 契约测试实跑（pytest）与**小网格复现**（16×8、5 迭代）验证 solve 入口真能跑通——小网格只证通路，不冒充基准；
4. 基准裁判以 `tensor_benchmark_cases` 为准（`anchor_slug='CASE-CYLINDER-DRAG-FVM'`）；
5. 级别按 maturity_gate 的机器判据（L1 tests / L2 impl_files+entry+run_artifacts / L3 +integration_callers / L4 +benchmark+repro_cmd / L5 +external_anchor）逐级核。

## ③ 核验结果（三件套 + 六段 + 基准裁判）

| 验收件 | 状态 | 指针 |
|---|---|---|
| run_artifacts | **absent** | none under artifacts/case-cylinder-drag-fvm/runs/ (expected 64x24/128x48/256x96 result.json) |
| benchmark | **not-registered** | tensor_benchmark_cases rows for anchor=0; got values in delivery=0; ref: b8eff73b6e611142 |
| repro_cmd | **executable** | scripts/case_cylinder_drag_fvm.py solve|analyze|report; tiny 16x8/5-iter rc=2 in 4.7s (rc=2 = physical gate not passed on an undersized grid, artefacts still produced) |
| impl_files | **present** | scripts/case_cylinder_drag_fvm.py (491 l), src/tensorfvm/verification/dfg.py (104 l) |
| tests | **synthetic-only** | tests/test_case_cylinder_drag_fvm.py::test_analyze_passes_and_fails_on_tolerance rc=0 |
| integration_callers | **present** | tests/test_dfg_evidence.py:6 (from tensorfvm.verification.dfg import run,audit_case) |
| report_structure | **present** | 6 sections; posture=honest-FAIL (declared in the report itself) |
| cited_artifacts_tracked | **present-but-untracked** | docs/case-cylinder-drag-fvm/report.md cites artifacts/.../reference/* — untracked (git status: ?? artifacts/) |

- 交付 commit 实际文件：`docs/case-cylinder-drag-fvm/report.md`, `scripts/case_cylinder_drag_fvm.py`, `tests/test_case_cylinder_drag_fvm.py`
- 报告引用但**未入库**的工件：artifacts/case-cylinder-drag-fvm/reference/reference.json, artifacts/case-cylinder-drag-fvm/reference/featflow_re20.html, artifacts/case-cylinder-drag-fvm/reference/fetch_manifest.json, artifacts/case-cylinder-drag-fvm/reference/fetch_reference.py, artifacts/case-cylinder-drag-fvm/tools/post_report.py（`git status` → `?? artifacts/`）
- 报告六段：6 段齐全（① 问题与参考值出处, ② 方法与实现（已完成的真实部分）, ③ 参数表（无量纲数完整）, ④ 对照表（ref vs got + 逐项误差）—— **FAIL：无 got**, ⑤ 收敛性/参数敏感性 —— **FAIL：待计算**, ⑥ 失败模式与工件），其中 ④⑤ 为显式「无数据」条款——**结构齐、内容空**，不构成基准验证。
- 基准裁判：`tensor_benchmark_cases` 共 19 行，该 anchor **0 行**；TensorFVM 既有登记行为 ['GAP-FVM-TWOPHASE', 'REPRO-FVM-SOD1978', 'CASE-CAVITY-RE1000-FVM']。→ 本算例从未登记基准判据，任务单自定的「<5%」是唯一容差来源，无机器裁判可比对。

### 契约测试与小网格复现实跑

- `pytest tests/test_case_cylinder_drag_fvm.py -q` → rc=0，20.6s，末行：['-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html', '1 passed, 1 warning in 19.90s']（合成夹具：三档细化 PASS 路径 + 10% 偏差 FAIL 路径）。日志 `docs/audit/m15406/pytest_contract.log`。
- `solve --grids 16x8 --max-iterations 5` → rc=2，4.7s，产物 ['audit.json', 'fields.npz', 'figures', 'history.csv', 'history.json', 'portable', 'result.json']（docs/audit/m15406/repro_tiny/run/16x8）；result.json 头部：{"resolution": "16x8", "converged": false, "computed": {"drag": 6.62589194823314, "lift": 0.08070401622367411, "pressure_difference_pa": 0.11860747074553127}, "iterations": 5}。
  → **入口与计算通路真实**（输入→SIMPLE 求解→result.json/fields.npz/history.json），但该小网格不是基准网格，不产生可判定的 got。

## ④ 结果图

![fig1](docs/audit/m15406/figures/fig1_error_vs_gate.png)

图 1：仓内历史归档的 DFG 2D-1 结果与 m15406 对照。历史值全部越过 5% 门（其中 42.2%/33.3% 越过平台 15% 不可接受线），m15406 本单为**空白**（无 got）。

![fig2](docs/audit/m15406/figures/fig2_evidence_matrix.png)

图 2：三件套机器核查矩阵（状态 + 证据指针）。

## ⑤ 判级与差异说明

**裁定 L1**（不是 L4，也不是 L2）：

- L1 ✅：契约测试存在且实跑通过（合成数据验证 analyze 的 5% 门、Richardson/GCI 分支）；
- L2 ❌：`run_artifacts` 缺失——交付物没有任何 `result.json/fields.npz`，即「真代码跑出真结果」这一条无证据；
- L3/L4 ❌：随 L2 连带不成立；L4 另需 benchmark 的 ref-vs-got 数值对照，本单 got=0（登记裁判行也缺）；
- L5 ❌：外部锚点（文献页 + sha256）虽齐，但 L4 未达。

**与「succeeded」的差异**：m15406 自报 honest-FAIL（集群排队未取回），报告本身没有粉饰；但平台状态为 succeeded，任务榜按「结单→交差」推进。差异在于**算例的可判定性**：算例池要求的是基准数字，本单只交付了 harness。返修只需补跑一条 repro_cmd 并归档 runs/。

**与报告自述的一致性抽查**：

- 报告声明「零手抄」→ 参考值确由 `fetch_reference.py` 正则解析，且与包内常量漂移={'drag': 0.0, 'lift': 0.0, 'pressure_difference_pa': 0.0}；
- 报告声明「无 got」→ 交付内 got 数字数=0，`runs/` 目录不存在，与自述一致（**无虚标**）；
- 口径小瑕疵：`dfg.py` 的 Metric 门是 `0.03`（3%），算例 harness 默认 `--tolerance 0.05`（5%），报告正文写 5%——两套门并存，读者可能误以为 3% 已过；建议返修统一并在报告中写明。

## ⑥ 误差判定（显式回答「是否<3%」）

**无法回答——本单没有 got 数字**。因此不是「<3% 未达」，而是「未测」。按平台判据：
- 基准腿：`NOT_MEASURED`（0 个 got；登记裁判行 0 条）；
- 可判定性下限：仓内同算例历史归档已多次越过 3%（13.36%、42.21%、8.59%、33.30%），说明该能力当前**不具备** DFG 2D-1 的 <3% 资格；m15406 未改变这一事实；
- honest-FAIL 不升级、不粉饰：本核验证实「无数据」而非「数据不好」。

## 未做的检查（如实登记）

- 未重跑三档基准网格（交差单非重做单；且单档 64×24 在本机 >120 s，需集群）。
- 未审查 `SimpleSolver` 数值内核（属 TensorFVM 其他审计范围）；未做网格收敛性判定。
- 未核 m15406 之后是否有接棒单已补跑（如已有，应另开新 verdict 覆盖旧 scope）。

## 附：证据文件

- `docs/audit/m15406/finding.json`（机器可读全量核查结果）
- `docs/audit/m15406/delivered/`（交付字节副本 + `sha256manifest.json`）
- `docs/audit/m15406/repro_tiny/`（16×8/5 迭代实跑产物）与 `repro_tiny.log`
- `docs/audit/m15406/pytest_contract.log`（契约测试日志）
- `docs/audit/m15406/figures/fig1_error_vs_gate.png`、`fig2_evidence_matrix.png`

