# m23320 · 舍入级误差序列不该被当作"收敛阶"打分（#856 重开单的第 2 轮收口）

**锚点**: `CASE-FVM-KOVASZNAY-ORDER`（算例池·定长 Kovasznay 流收敛阶验证）
**判据（feature_ledger 原文）**: L2 收敛阶 ≥ 1.8（名义二阶），指标 = 32/64/128 三套网格 log-log 斜率
**外部基准**: Kovasznay 1948, *Proc. Camb. Phil. Soc.* 44:58–62, doi:10.1017/S0305004100023999
**席位 / 仓**: m23320 · TensorFVM · 分支 `agent/m23320`
**复核基线 commit**: `8539db44`（merge #921 之后）
**前置交付**: #856（原 PR 已由 owner 2026-11-05 裁定关闭）的 6 个文件在基线树上**已齐全**：
`m20571-artifact-schema-drift-diagnosis.md`、`report_payload_m20571.py`、
`runs-32-64-128-zero_gradient/{RESULTS.md, summary.json}`、`summary-32-64/{RESULTS.md, summary.json}`、
`n128/fields-error.png`。本单先复跑其验收门（28 passed / 10 skipped，skip 全部是 "artifact absent"），
确认前置交付在树上可复现，再处理它所带出的一处**判据语义缺陷**。

---

## 1. 案例介绍（物理背景与参数）

Kovasznay 流是二维定常不可压 Navier–Stokes 的解析解，是有限体积格式精度的标准制造解。
本锚点只问一件事：网格 32→64→128 加密，体积加权相对 L2 误差的 log-log 斜率是否 ≥ 1.8。

`zero_gradient` 闭包（`du/dz = dv/dz = 0`）是物理正确的对称条件。它的解析解展向速度为
**恒零**（`w_exact ≡ 0`），因此该闭包"生效"的可观测证据就是：离散解里的展向速度必须落在
**浮点舍入**上。于是产物里多出一列误差预算 `spanwise w RMS / |u_exact|`。

| 参数 | 值 |
|---|---|
| Re | 40 |
| 域 | [0,1]²，对称挤出 n×n×1（1024 / 4096 / 16384 单元）|
| λ | −0.9637405441957689（λ² − Re λ = 4π²）|
| 对流格式 | 2 阶 hybrid |
| 对称面闭合 | `zero_gradient` |
| 算法 / 线性 | `coupled` + `reused_lu`，outer_relaxation 0.7，tol 1e−8 |

## 2. 怎么做的（缺陷、判据语义、修法）

### 2.1 缺陷：机器零的序列被拟合出"阶"，并被判 FAIL

`summary.json` 里展向预算三个数是

```
6.779241e-18  1.119141e-18  9.003311e-19      （n = 32 / 64 / 128）
```

它们不是误差，是**舍入噪声**（同一序列的 `max|w|` 为 1.6e−16 / 2.5e−17 / 4.8e−17，随网格不单调）。
但 `scripts/summarize_kovasznay_32_64_128.py` 对所有 quantity 一视同仁地取 log-log 斜率，于是：

- `fits.velocity_spanwise.slope = 1.456298`（R² 0.8298）——对噪声的拟合；
- RESULTS.md 表里写成 `FAIL`（阈值 1.8），pair-orders 表写成 `2.5987/0.3139 / NO`。

后果有两重，都属"报告与树不一致"家族：

1. **误判方向反了**。展向份额 9e−19 恰恰是本算例**最强的正确性证据**（闭包把伪模态钉到了舍入级），
   却被渲染成 5 个量里的一个 FAIL，读者会以为格式出问题。
2. **判据不可否证**。同一列在 `free` 闭包下是 `4.53e-03`（真实 O(h) 伪模态），报告里两者都只是
   "一个 FAIL 的数字"，只能靠脚注（m20571 §5.3）口头声明"无物理意义"，机器不可判。

### 2.2 修法：按"绝对量级"分级，明确标记不可识别

`scripts/summarize_kovasznay_32_64_128.py` 新增 `ROUNDOFF_QUANTITIES` + `roundoff_audit()`：

- 若该量在**每一层**都不超过舍入带 `SPANWISE_ROUNDOFF = 1e-12`（× |u_exact|），记为
  `machine_zero: true, graded: false`，`slope_note` 写明 "n/a (machine zero …)"；
- 判据改为**绝对量级**：`verdict.spanwise_criterion_pass = max share ≤ 1e-12`，
  `verdict.spanwise_order_graded = false`（**不参与**阶判据）；
- `fits` 里每个量都显式带 `machine_zero / graded`，缺失不再等于"未分级"；
  `pair_consistency` 同步标记；
- RESULTS.md / closeout 的渲染改为 `n/a (machine zero)` + 一句脚注，**不再打印**拟合出的
  1.4563 / 2.5987 当作成绩；原始最小二乘数字仍留在 JSON（可复算），只是被标为不可识别；
- 主判据（`pass = velocity_full ≥ 1.8`）未动一个字符。

**否证能力**：只要有一层离开舍入带，该序列就照常被打分。测试里用 `free` 闭包型序列
（4.53e−3→2.27e−3→1.13e−3）做负控制，斜率 1.00 < 1.8，必须继续判 FAIL——否则这条规则就退化成
"展向永不失败"。

### 2.3 产物重生成（零手抄）

```bash
BD=docs/CASE-FVM-KOVASZNAY-ORDER/runs-32-64-128-zero_gradient
PYTHONPATH=src python scripts/summarize_kovasznay_32_64_128.py --runs $BD --output $BD \
  --markdown $BD/RESULTS.md --figure $BD/order.png --require-grids 32,64,128
PYTHONPATH=src python scripts/summarize_kovasznay_32_64_128.py --runs $BD \
  --output $BD/summary-32-64 --markdown $BD/summary-32-64/RESULTS.md \
  --figure $BD/summary-32-64/convergence.png --require-grids 32,64
PYTHONPATH=src python -m tensorfvm.verification.kovasznay_case_pool \
  --output docs/case-fvm-kovasznay-order-pool/case_pool_entry.json   # digest 刷新
PYTHONPATH=src python docs/CASE-FVM-KOVASZNAY-ORDER/report_payload.py # closeout 重渲染
PYTHONPATH=src python docs/CASE-FVM-KOVASZNAY-ORDER/make_m23320_roundoff_figure.py
```

图：`m23320-roundoff-grading.png`（左：阶梯子；右：展向份额 vs 舍入带的量级分离）。

## 3. 计算结果（全部由 npz 现场复算，程序化注入）

| n | h | cells | vel L2 (full 3D) | p L2 (mean-free) | spanwise share | max\|w\| |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 3.125e−02 | 1024 | 3.435615e−03 | 2.885142e−02 | 6.779241e−18 | 1.602244e−16 |
| 64 | 1.562e−02 | 4096 | 8.972614e−04 | 7.958339e−03 | 1.119141e−18 | 2.488188e−17 |
| 128 | 7.812e−03 | 16384 | 2.262325e−04 | 2.136311e−03 | 9.003311e−19 | 4.823264e−17 |

| 量 | 三网格 LSQ 斜率 | R² | 分级 |
|---|---:|---:|---|
| velocity L2 (full 3D u,v,w) | **1.9623** | 0.999944 | PASS（≥1.8）|
| velocity L2 (in-plane u,v) | 1.9623 | 0.999944 | PASS |
| pressure L2 (mean-free gauge) | **1.8777** | 0.999964 | PASS |
| pressure L2 (fixed gauge) | 0.9772 | 0.999975 | FAIL（固定规范不是精度量度）|
| spanwise w RMS / \|u_exact\| | — | — | **n/a（机器零）**：max 6.78e−18 ≤ 1e−12 → 绝对判据 PASS |

判定变化：同一批数字下，`verdict.pass` 仍为 `true`（velocity_full 1.9623），新增
`spanwise_machine_zero: true / spanwise_criterion_pass: true / spanwise_order_graded: false`；
RESULTS.md 中展向行由 `1.4563 / FAIL` 变为 `n/a (machine zero)`。

## 4. 结果图

`m23320-roundoff-grading.png`（数据全部读自 `summary.json`，无手抄）：

- 左图：`velocity_full`、`pressure_meanfree` 的误差 vs h，与 h² 参考线；图例斜率为复算值。
- 右图：展向份额 vs h（对数纵轴），叠加 **1e−12 舍入带**（分级阈值）与
  `free` 闭包记录值 4.53e−3（m20571 §2 记录；该 run 不在本检出内，标注为注记而非数据点）。
  两个簇相隔 15 个数量级 —— 这就是"按绝对量级分级"能一刀切开的理由。

## 5. 注意点（坑与局限）

1. **前置交付已在树上，本单不重投**。#856 关闭时的 6 个文件在本基线上齐全且门为绿；
   真正缺的是"机器零序列的判据语义"，本单只改这一处（小步单元）。
2. **`convergence.png` 未提交重生成**：本环境 matplotlib 重渲染后，曲线与图例逐像素一致，
   仅子图标题行有约 620×28 px 的版面差异（7790 px 不同，纯排版/字体度量）。为避免把
   无法归因的二进制噪声塞进 PR，该 png 保持基线版本，`order.png` 重生成后逐字节一致。
3. **`docs/case-fvm-kovasznay-order-pool/case_pool_entry.json` 的 digest 必须跟着刷新**：
   算例池记录 `summary.json` 的 sha256，`tests/test_kovasznay_case_pool.py::test_committed_entry_is_fresh`
   会把"工件改了但条目没刷"变成有名字的失败——这正是本轮改动的自证。
4. **`runs-32-64-128`（free 闭包）不在本检出内**：因此负控制在测试里用合成序列完成
   （4.53e−3 型），而非真解；真 run 回仓后该门槛可直接复用。
5. **舍入带 1e−12 是"× |u_exact|"的相对份额阈值**，与 `SPANWISE_ROUNDOFF`（同一常量）共用，
   不是绝对速度量纲；跨算例复用时须确认尺度一致。
6. **判据名义上是三网格**；三层给不出阶的标准误（`order_stderr = null`），GCI 也不能引用
   （外推极限为负、GCI 1.33），这些既有保留意见不变。

## 6. 误差判定（逐项，显式回答 <3%）

| 关键量 | 本单结果 | 基准 | 相对误差 | <3%? |
|---|---|---|---|---|
| 收敛阶 velocity L2 (full 3D) | 1.9623 | 名义二阶 2.0 | 1.9% | **是** |
| 收敛阶 velocity L2 (in-plane) | 1.9623 | 2.0 | 1.9% | **是** |
| 收敛阶 pressure L2 (mean-free) | 1.8777 | 2.0 | 6.1% | 否（>3%，如实写出）|
| n=128 速度场 L2 相对误差 | 2.2623e−04 | 解析解 | 2.26e−02 % | 是 |
| n=128 压力场 L2 相对误差（去均值） | 2.1363e−03 | 解析解 | 2.14e−01 % | 是 |
| 展向伪模态（绝对值） | 9.00e−19（n=128） | 0（解析 w≡0） | 1e−15 量级 | 是（舍入级）|
| 三网格斜率 vs 三点 Richardson 阶 | 1.9623 vs 1.9194 | 互为独立估计 | 2.2% | 是 |

**结论**：速度阶以 1.9% 贴近名义二阶（<3%）；去均值压力阶 6.1%（>3%，压力一阶收敛于
gauge-consistent 范数是该离散的既有行为，如实记录，不粉饰）。本单的判定变化**只影响分级语义**，
不改变任何数值结论：`verdict.pass` 仍为 true，且现在"展向 PASS"与"速度 FAIL"不会再混淆。

## 7. 验收与回归

本机只跑 <8 核 / <10 min 的短测试（计算调度令）；全套 `pytest tests`（约 6400 项）已递交集群
（`cj4766`，8 核，递交时排队位次 145，尚未出队），本机按文件分批跑与本改动相关的 21 个文件：

| 批次 | 结果 |
|---|---|
| `test_kovasznay_case_pool / 32_64_128_acceptance / spanwise_roundoff_grading / summarize_mixed_config_guard / closeout_regenerable` | **61 passed, 10 skipped** |
| 新测试 `tests/test_kovasznay_spanwise_roundoff_grading.py` 单跑 | **7 passed**（含 1 条负控制 + 2 条已提交工件门）|
| `committed_artifacts_fresh + doc_claims + order_gate` | 15 passed, 2 skipped |
| `spanwise` / `symmetry` / `extrusion_spanwise` / `order_budget` / `anchor_ladder` | 5 / 10 / 4 / 3 / 9 passed |
| `ladder_text_channel + text_evidence` / `gate_mutation_audit + inplane_decomposition` | 22 / 16 passed |
| `order_sensitivity + order_verification` | 19 passed |
| `ladder_round3 + native_kovasznay + zdriver` | 18 passed |
| `closure_metadata_semantics + 256_ladder + reference_independent + order_m17078` | 29 passed, 14 skipped, 2 failed（见下）|

skip 全部是 "artifact absent"（`runs-32-64-128` free 闭包 run、`summary-32-64-128-256` 不在本检出），
是诚实的"尚未测量"，不是 pass。

**未在本机跑完**（单条命令超 2 min 的本地短测试上限，留给 `cj4766`）：
`test_kovasznay_convergence_order.py`、`test_kovasznay_order_asymptotic.py`、
`test_kovasznay_closeout_report_reproducible.py`、`test_kovasznay_z_momentum_probe.py`。
如实记录：本单交付时它们的绿灯由集群作业 `cj4766` 负责，作业仍在排队，回执出来前不宣称"全套通过"。

**两条失败的原因（均与本次改动无关）**：

1. `test_kovasznay_closeout_regenerable.py::test_closeout_doc_reproduces_from_committed_artifacts`
   —— 改动的**预期**失败：`closeout-32-64-128.md` 内嵌 verdict JSON，新增 spanwise 分级字段后必须重渲染。
   已用 `report_payload.py` 重新生成，重跑后 61 passed。
2. `test_kovasznay_order_m17078.py::test_summary_exists` —— 基线既有失败：要求
   `runs/kovasznay-order-m17078/summary.json`，而 `runs/` 在 git 中**零文件被跟踪**
   （`git ls-files runs | wc -l` = 0），任何干净检出都无法通过。
