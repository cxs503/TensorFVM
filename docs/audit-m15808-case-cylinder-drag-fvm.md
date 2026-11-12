# 交差核验：m15615 CASE-CYLINDER-DRAG-FVM（TensorFVM）

- 被核任务：`tensor_dev_missions.id=15615`，status=succeeded，PR #448 merged，claim_token 无，decision=`[auto-approve]`，
  `server_quality_audit={gate_valid:false, key_present:false, gate_missing_warning:true}`（即通过自动批准而非门禁裁定）。
- 核验任务：m15808（board-audit，交差核验单，不重做算例）。
- 核验工作树：`/root/tp-agent/work/m15808`（分支 agent/m15808），被核工作树 `/root/tp-agent/work/m15615`（只读取工件）。
- 本文件所有数字由 `artifacts/case-cylinder-drag-fvm/analysis/summary.json` 与各 run 的 `result.json` **程序化注入**（零手抄）。

## 1. 基准裁判（唯一裁判）状态：**案号缺失（ABSENT）**

```sql
SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='CASE-CYLINDER-DRAG-FVM';  -- 0 rows
SELECT anchor_slug FROM tensor_benchmark_cases;  -- 22 rows，无任何 cylinder anchor
```

`tensor_benchmark_cases` 中不存在本锚点（也不存在任何 cylinder 锚点）→ 平台裁判无案号可裁。
本核验改用**任务自身预声明判据**（两级，均在开工前声明，非事后回填）：

- 任务榜 goal 原文：容差判据 `<5%`（`tensor_dev_missions.id=15615` goal 字段）。
- 运行件内嵌判据：`runs/*/result.json` 的 `metrics[*].limit = 0.03`（3%）。

## 2. 三件套核验

| 项 | 判定 | 证据 |
|---|---|---|
| 实现（真代码，非 MOCK/STUB） | REAL | `src/tensorfvm/verification/dfg.py:18-44`（body-fitted `SolverConfig` + 主求解器 `SimpleSolver` + 多边形壁面牵引 + 最小二乘壁面梯度/无滑移修正 + 周向角插值 p_diff）；`scripts/case_cylinder_drag_fvm.py`（solve/analyze/report） |
| 入口可执行 | 通过 | `PYTHONPATH=src python -m tensorfvm.verification.dfg --output ... --grids 32x12 --max-iterations 200` 本机重跑 70 s，`drag=6.670193228309547` 与归档 `result.json` **逐位相等**（迭代残差 8.16669e-06 亦同）；`--help` 正常 |
| 运行工件真实存在 | 通过（但原分支未提交） | m15615 工作树 `artifacts/case-cylinder-drag-fvm/runs/{32x12,64x24}/{fields.npz,result.json,history.json,history.csv}`；`git status` 为 `?? artifacts/` → **原 PR 未含工件**，本次核验已把它们提交进 agent/m15808 分支 |
| 基准 ref vs got 数值 | **未达标** | 见下表；C_D/C_L/p_diff 三项全超 5% 与 3% |
| repro_cmd | 部分失效→已修 | `solve` 路径实测可复现（逐位相等）；`analyze`/`report` 路径**从来跑不通**（见 §5 F4），本次已修并端到端跑通 |
| 报告六段 | 齐全 | `docs/case-cylinder-drag-fvm/report.md` ①问题+ref 出处 ②方法与实现 ③参数表 ④对照表 ⑤收敛/敏感性 ⑥失败模式与工件 均在；且如实写"部分通过/honest-FAIL"，未粉饰 |

### ref vs got（裁判口径）

参考值（DFG 2D-1 / Turek & Schäfer 1996）：C_D=5.57953523384，C_L=0.010618948146，p_diff=0.11752016697 Pa。

| grid | converged | iters | C_D | err % | C_L | err % | p_diff Pa | err % | viscous share of C_D |
|---|---|---|---|---|---|---|---|---|---|
| 32x12 | no | 200 | 6.670193 | +19.547 | +0.051804 | +387.85 | 0.129441 | +10.143 | 34.0% |
| 64x24 | yes | 976 | 6.324792 | +13.357 | +0.015102 | +42.21 | 0.130991 | +11.463 | 33.5% |

- 最小误差量 C_D：64×24 为 **+13.357%**，超任务 5% 容差 2.67 倍、超 3% 判据 4.45 倍。
- C_L：+42.215%（超平台 15% 不可接受线）；p_diff：+11.463%（超 5%，未超 15%）。
- 网格加密趋势：C_D 误差 19.547%→13.357% 单调下降（方向正确），但仅两档、且第二档才收敛，**无观测阶/GCI**（脚本自身也输出 `convergence: not available (needs three grids)`）。

## 3. 判级：**L3**（功能实质，非文档厚度）

- **L2 达标**：独立功能实现——真实输入→求解→输出（贴体 SIMPLE + 壁面牵引），工件真实且**逐位可复现**。
- **L3 达标**：系统集成——`dfg.py` 直接驱动主求解器 `src/tensorfvm/solver.py::SimpleSolver`（`SolverConfig(mesh_type='body-fitted')`）与 `tensorfvm/cylinder.py::export_result`，单一 CLI 产出完整证据包（fields.npz/result.json/history/report.pdf）。
- **L4 不达标**：基准验证要求"完整算例+误差分析+可复现"并且**判据通过**；本例三项量全部超预声明容差（13.36%/42.22%/11.46%），且缺第三档网格→无网格无关性证据。
- **L5 不达标**：无任何通过基准的外部锚定，更无科研增量。

## 4. 被核方"汇报 vs 实测"一致性

m15615 headline 自述"交付至 L2"，报告正文自述"部分通过/超 3% 门限/honest-FAIL"，与本核验实测**一致，无虚标**。
差异仅在本核验给出更高一级（L3 而非 L2），因为集成面（主求解器调用方）真实存在且在仓内可 grep 到。

## 5. Findings（严重度降序）

- **F1（阻断 L4，可复现）**：三项基准量全部超预声明容差：C_D +13.357%、C_L +42.215%、p_diff +11.463%（64×24）。C_L 已超平台 15% 不可接受线。
- **F2（缺陷，已修，高）**：文档化 repro_cmd 的 `analyze`/`report` 分支在真实工件上**从未成功运行**：
  (a) 默认参考文件 `artifacts/case-cylinder-drag-fvm/reference/reference.json` 不在仓内 → `FileNotFoundError`；
  (b) `probe_pressure()`/`figures()` 假设 `cell_centers_m` 是 (n,2) 平铺表，而包导出格式是结构化 (ny,nx,2) + 平铺压力 → `ValueError: operands could not be broadcast (12,2)(384,)` 与 `griddata: Input points array must have 2 dimensions`。
  修复：新增 `reference.json`（值与 `dfg.py:14 REFERENCE` 逐位一致 + `tests/test_case_cylinder_reference_binding.py` 防漂移）、`probe_pressure`/`figures` 归一化 reshape、`source_sha256=null` 时报告行降级为 "unbound"。修后 `analyze` 端到端跑通并**逐项复现报告数字**（13.3570/42.215/11.4630）。
- **F3（证据耐久性，中，已修）**：运行工件与参考文件原不在 PR 内（`?? artifacts/`），新 checkout 无法核验；已提交进本分支（1.1 MB，含 7 张图）。
- **F4（物理诊断，中，假设待证）**：64×24 阻力分解 pressure 4.2077 + viscous 2.1171（黏性占 **33.5%**）。DFG 2D-1 在细网格上黏性份额应远低于此（量级 ~10%），提示**多边形壁面梯度/牵引的黏性力被高估**，很可能就是 13% C_D 超差的主因。本核验无外网（curl 被安全策略拦截）故未取原页分解表，标记为待证假设 H1：若 128×48 黏性份额不下降，则该算例的壁面牵引实现存在系统性偏差。
- **F5（指标判别力，中）**：p_diff 对采样方式极度敏感——最小二乘壁压外推给 +11.46%，而同一工件上最朴素的逆距离单元探针给 **−30.93%**（横跨 42 个百分点）。故"p_diff 3%"在 32×12/64×24 这一网格族上不是有判别力的指标，必须固定并要求记录采样算子（`result.json:reference.pressure_sampling` 已记录，属加分项）。
- **F6（测试有效性，低/如实）**：`tests/test_case_cylinder_drag_fvm.py` 用**合成数据**只验"比较/判罚/收敛拟合"管线（1 passed），不验物理；其 docstring 已诚实声明。`tests/test_dfg_evidence.py` 3 passed（牵引一致性/无滑移/质量重构/metric 诚实性），属真实不变量校验。
- **F7（流程，低）**：`tensor_benchmark_cases` 无 cylinder 锚点 → 算例池的"基准裁判"整体未落地（22/22 表中 0 个 cylinder 案号）；且 m15615 经 auto-approve 通过（`server_quality_audit.gate_valid=false`），"succeeded" 只表示交付完成，不表示精度达标。

- **F8（平台门禁弱点，可复用知识，中）**：`maturity_gate.judge()` 仅校验 `benchmark` 键"含数字"（`has_numbers()`），**不校验数字是否在容差内**——本次实测：把本 verdict 声明为 L4 时 `judge(4, evidence)` 返回 `(4, [])` 通过，而同一份 evidence 的 `benchmark.verdict` 明写 FAIL（C_D +13.357%）。即"机器门禁无法区分'有数字'与'达标'"。建议（属 TensorPlatform 侧共性机制，本单不改）：benchmark 键增加 `within_tolerance: bool` 并由门校验 `benchmark.verdict`/最长误差 vs `tolerance`。本次自报 level=3，未利用该弱点。

## 6. 未做/局限（不得读作"通过"）

- 未重跑 64×24（约 530 s）与 128×48/256×96（m15615 自述集群排队未取回）：**网格无关性/观测阶 = not_run**，L4 的"收敛证据"因此不可补。
- 未取回集群收敛序列 → 无法排除"仅网格不足"这一解释，也无法排除 F4 的系统性壁面偏差；两者需 128×48 才能分辨。
- 无外网：参考页未重新抓取，`source_sha256` 留空（已在 reference.json `provenance` 写明）；参考值与代码常量逐位一致，但独立性受此限制。
- 未跨架构复核（集群 loong64 与本机同为 loong64 按 bootstrap 记载；本机重跑已逐位一致，故未另投集群）。

## 7. 结论（落表）

`tensor_maturity_verdicts` **id=374**：repo=TensorFVM, scope=`m15615 CASE-CYLINDER-DRAG-FVM`, mission_id=15808, **level=3**, verdict_by=board-audit；门禁自检 `judge(3, evidence)` → `(3, [])`（confirmed），`judge(4, …)` 亦过（见 F8，故本单不取 L4）。



**L3**：真实现 + 真集成 + 真工件 + 逐位可复现，但基准判据（<5%，运行件内嵌 3%）**未通过**，L4 拒绝授予；m15615 的 succeeded 语义为"交付完成"而非"精度达标"，报告本人已 honest-FAIL，无虚标。
返修路径（若要冲 L4）：①补 128×48（+256×96）网格并对 C_D 做观测阶/GCI；②查 F4 黏性份额是否随加密下降（壁面梯度/牵引实现）；③把本锚点登记进 `tensor_benchmark_cases`（tolerance 建议 5%，并绑定固定壁压采样算子）。
