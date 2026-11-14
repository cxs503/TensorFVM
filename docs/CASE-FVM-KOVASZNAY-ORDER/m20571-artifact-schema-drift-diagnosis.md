# CASE-FVM-KOVASZNAY-ORDER · 验收门在 HEAD 上是红的：工件被交付过滤器掉包后的 schema 漂移

**锚点**: `CASE-FVM-KOVASZNAY-ORDER`（算例池·定长 Kovasznay 流收敛阶验证）
**判据（feature_ledger 原文）**: L2 收敛阶 ≥ 1.8（名义二阶），指标 = 32/64/128 三套网格上的 log-log 斜率
**外部基准**: Kovasznay 1948, *Proc. Camb. Phil. Soc.* 44:58–62, doi:10.1017/S0305004100023999
**席位 / 仓**: m20571 · TensorFVM · 分支 `agent/m20571`
**复核基线 commit**: `cc132a8e`（merge #667 之后）

---

## 1. 案例介绍（物理背景与参数）

Kovasznay 流是二维定常不可压 Navier–Stokes 方程的解析解，满足
`div u = 0`、`(u·∇)u + ∇p − μΔu = 0`，无体力项，是有限体积格式精度的标准制造解。
本锚点的判据只关心一件事：把网格 32→64→128 加密，体积加权相对 L2 误差的 log-log 斜率
是否 ≥ 1.8（名义二阶）。

| 参数 | 值 |
|---|---|
| Re | 40 |
| ρ / μ = 1/Re | 1 / 0.025 |
| 域 | [0,1]²，对称挤出 n×n×1 |
| λ | −0.9637405441957689（λ² − Re λ = 4π²）|
| 对流格式 | 2 阶 hybrid |
| 对称面闭合 | `zero_gradient`（`du/dz = dv/dz = 0`）|
| 算法 / 线性 | coupled + reused_lu，outer_relaxation 0.7，tol 1e−8 |

## 2. 本单发现：判据的数字是对的，但**门在 HEAD 上是红的**

在 `cc132a8e` 上直接跑该锚点的验收门：

```bash
PYTHONPATH=src python -m pytest tests/test_kovasznay_32_64_128_acceptance.py \
    tests/test_summarize_kovasznay_mixed_config_guard.py -q
```

```
FAILED tests/test_kovasznay_32_64_128_acceptance.py::test_pair_orders_and_three_level_order_reproduce[anchor_zg]
1 failed, 23 passed, 9 skipped
E   KeyError: 'pair_consistency'
```

前单 `RESULT_32_64_128.md` §2.3 写的是「验收门转绿：22 passed, 8 skipped」，
`ORDER_32_64_128_STATUS.md` 与 `RESULT_32_64_128.md` 都以「判据达成」收尾。
该陈述在**提交后的树**上不可复现——判据的数值结论成立（见 §3），但承载它的
证据文件与读它的门已经不在同一代，门是红的。这正是 owner 反复强调的
「报告 PASS ≠ 树 PASS」的一类：数字对、证据链断。

### 2.1 根因（可复算，非推测）

1. m20158 把 `scripts/summarize_kovasznay_32_64_128.py` 升级为写下
   `pair_consistency` / `three_level_estimates` 两个佐证块，验收门同时改成读它们
   （`test_pair_orders_and_three_level_order_reproduce`，见 `tests/…acceptance.py:210-241`）。
2. m20239 用升级后的脚本重新生成了 `summary.json`（npz 里 n128 已落盘，数字见 §3）。
3. **但那份重新生成的工件没有进仓**：m20239 的落盘 commit `b21ccaf2` 的提交信息是
   `m20239: matured mission deliverables（10 个文件被门禁滤除）`，该 commit 只含 2 个
   markdown 文件（`RESULT_32_64_128.md`、`m20239-n128-harness-diagnosis.md`），
   重新生成的 `summary.json` / `RESULTS.md` / `order.png` 属于被滤除的 10 个文件。
4. 结果是：树里留下的是 m20158 **升级前**的 `summary.json`
   （顶层键 `title/levels/observed_orders/fits/verdict/provenance`，无 `pair_consistency`），
   而门已经读新键 → `KeyError`。数值层没有任何问题，证据层掉了一代。

仓库内同病的第二处：`runs-32-64-128-zero_gradient/summary-32-64/summary.json`
（32/64 两网格的中间产物）也是升级前 schema；它是 `ORDER_32_64_128_STATUS.md` §2
与 `m17913-symmetry-closure-runner.md` §3.2 引用的证据，本单一并回填。

## 3. 计算结果（全部由 `summarize_kovasznay_32_64_128.py` 从 npz 现场复算）

重算命令（本单实跑，退出码 0）：

```bash
BD=docs/CASE-FVM-KOVASZNAY-ORDER/runs-32-64-128-zero_gradient
PYTHONPATH=src python scripts/summarize_kovasznay_32_64_128.py --runs $BD --output $BD \
  --markdown $BD/RESULTS.md --figure $BD/order.png --require-grids 32,64,128
PYTHONPATH=src python scripts/summarize_kovasznay_32_64_128.py --runs $BD \
  --output $BD/summary-32-64 --markdown $BD/summary-32-64/RESULTS.md \
  --figure $BD/summary-32-64/convergence.png --require-grids 32,64
```

| n | h | cells | vel L2 (full 3D) | max\|w\| | p L2 (固定规范) | p L2 (去均值) |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 3.125e−02 | 1024 | 3.435615e−03 | 1.60e−16 | 7.843493e−02 | 2.885142e−02 |
| 64 | 1.562e−02 | 4096 | 8.972614e−04 | 2.49e−17 | 3.960905e−02 | 7.958339e−03 |
| 128 | 7.812e−03 | 16384 | 2.262325e−04 | 4.82e−17 | 2.023767e−02 | 2.136311e−03 |

| 量 | 32→64 | 64→128 | 三网格 LSQ 斜率 | R² | 判据 ≥1.8 |
|---|---:|---:|---:|---:|:--|
| 速度 L2（3D 全分量） | 1.93697 | 1.98772 | **1.9623** | 0.999944 | **PASS** |
| 速度 L2（面内 u,v） | 1.93697 | 1.98772 | **1.9623** | 0.999944 | **PASS** |
| 压力 L2（去均值规范） | 1.85810 | 1.89735 | **1.8777** | 0.999964 | **PASS** |
| 压力 L2（固定规范） | 0.98567 | 0.96879 | 0.9772 | 0.999975 | FAIL（见 §5.2）|
| 伪展向 w RMS 份额 | 2.5987 | 0.3139 | 1.4563 | 0.829783 | FAIL（见 §5.3）|

三网格拟合 `f(h)=L+C h^p` 再识别的阶：速度 1.9194、去均值压力 1.8434（3 层无法给出
阶的标准误，summary 里显式写 `null`，不假装有不确定度）。

图（均由 npz 生成，零手抄）：

| 图 | 内容 | 路径 |
|---|---|---|
| `order.png` | 三个网格的 log-log 误差与 1/2 阶参考线 | `runs-32-64-128-zero_gradient/order.png` |
| `fields-error.png` | n=128 的 u_x（数值/解析）、`|u_x−u_x^exact|`、`p−p^exact` 云图 | `runs-32-64-128-zero_gradient/n128/fields-error.png` |

`fields-error.png` 由本单新增的 `scripts/plot_kovasznay_field_error.py` 从
`n128-fields.npz` 生成（该脚本按 (z,y,x) 索引格重新排序单元中心，格点数与单元数不符
直接 `SystemExit`，不做静默乱序绘制）：n=128 时 `max|Δu_x| = 7.277e−04`，
即约 `3.6e−4` 倍的 `max|u_x|`，与 L2 速度误差 2.262e−04 同一量级。

## 4. 本单改了什么（最小、可验证）

| 变更 | 文件 | 作用 |
|---|---|---|
| 用当前 summarizer 回填三网格工件 | `runs-32-64-128-zero_gradient/{summary.json,RESULTS.md,order.png}` | 门重新读得到 `pair_consistency` / `three_level_estimates` |
| 回填两网格中间工件 | `runs-32-64-128-zero_gradient/summary-32-64/{summary.json,RESULTS.md,convergence.png}` | 消除同一处 schema 漂移的第二实例 |
| 新增 schema 新鲜度门 + 负控制 | `tests/test_kovasznay_32_64_128_acceptance.py`（`test_committed_summary_carries_current_summarizer_schema`、`test_schema_guard_rejects_pre_upgrade_summary`）| 工件与 summarizer 是否同一代，变成**有名字的**失败，而不是数值测试里的 `KeyError` |
| 新增场量/误差图脚本 | `scripts/plot_kovasznay_field_error.py` | 报告图可复算、可重生成 |

验收门前后对比（同一条命令）：

| 状态 | 结果 |
|---|---|
| HEAD `cc132a8e` | **1 failed, 23 passed, 9 skipped**（`KeyError: 'pair_consistency'`）|
| 本单修复后（回填工件 + 新增 4 个 schema 测试）| **28 passed, 9 skipped** |
| 负控制：把 `summary.json` 换回 HEAD 版本 | **3 failed, 19 passed, 9 skipped**（其中 `test_committed_summary_carries_current_summarizer_schema[runs-32-64-128-zero_gradient]` 指名报出缺的键）|

9 个 skip 全是 `anchor`（`free` 牵引自由闭合）那套 32/64/128 的 n64/n128 从未跑过，
即诚实的 **not-measured**，不是 pass；`free` 闭合留 O(h) 伪展向速度，会把全 3D 阶压到 ~1.68，
本就不适用该判据口径（`validate_levels` 对混合闭合硬报错）。

## 5. 注意点与局限

1. **报告 PASS 与树 PASS 是两件事**。本次失败不是数值失败，而是「重新生成的证据没进仓」。
   任何以「XX passed」收尾的结论都应附 commit 号，并在该 commit 的树上可复现；
   `（N 个文件被门禁滤除）` 的提交信息应当被视为**证据完整性告警**而非琐事。
2. **固定规范压力的阶（0.977）不是精度量度**：动量方程只定压力梯度，固定规范下压力带
   随网格变化的常数偏差；有物理意义的是去均值规范（1.8777）。
3. **伪展向 w 份额的斜率（1.4563, R²=0.83）无物理意义**：绝对值在 1e−18 ~ 1e−19，
   纯浮点舍入；列出来只是不让误差预算丢一个 FAIL 量。
4. **三层 GCI 不可识别**：`f(h)=L+C h^p` 拟合得到的外推极限 `L` 为负（−1.49e−05）、
   GCI 达 1.33（133%）。三层恰好三点，无剩余自由度，Richardson/GCI 在这里只能当
   「三点是否单调」的诊断用，**不能**当作精度不确定度引用；要 GCI 必须再加一层（n=256）。
5. **历史文档里不可复算的面内列**：`docs/case-kovasznay-order-order-verification/RESULTS_INPLANE.md`
   的 n=32 面内速度（3.72450e−03）与 `runs-32-64-128/n32/n32-fields.npz` 重算值
   （3.431614e−03）不一致，前单已记录，本单不引用该表。
6. **遗留隐患（未在本单修）**：`scripts/make_kovasznay_order_figure.py` 把 16/32/64 的误差
   硬编码在源码里，并且用 `velocity_l2_inplane = velocity_l2_full * 0.97`、
   `7.98671e-03 / 3.63` 这样**编造**的系数造「面内列」和斜率。该脚本在仓内无任何
   引用（`grep -rn make_kovasznay_order_figure` 为空），是孤儿；但它产出的
   `convergence_inplane.png` 一旦被引用就是一张伪证据图。建议后续单予以删除或改为读 npz。

## 6. 误差判定（逐项，显式回答 <3%）

判据本身是**收敛阶**，不是单点误差百分比；两件事分开回答，避免混淆：

| 项 | 数值 | 基准 | 判定 |
|---|---|---|---|
| 三网格速度 L2 阶 | 1.9623（R²=0.999944） | ≥ 1.8 | **PASS**（余量 9.0%）|
| 32→64 / 64→128 单段阶 | 1.9370 / 1.9877（spread 0.0508 ≤ band 0.20）| 两段均 ≥ 1.8 | **PASS**（一致）|
| 三网格去均值压力阶 | 1.8777 | ≥ 1.8 | **PASS** |
| 三网格拟合再识别阶 | 1.9194 | ≥ 1.8 | PASS（佐证，不替代判据）|
| n=128 速度 L2 误差 | 2.262e−04 | 解析解 | **远小于 3%（0.0226%）** |
| n=128 去均值压力 L2 误差 | 2.136e−03 | 解析解 | **0.214%，< 3%** |
| n=128 场量最大偏差 `max|Δu_x|` | 7.277e−04 | `max|u_x| ≈ 2` | 3.6e−4 相对，< 3% |
| n=32 压力（去均值）L2 | 2.885e−02 | 解析解 | 2.9%，< 3%（最粗网格）|
| 固定规范压力阶 | 0.9772 | ≥ 1.8 | **FAIL — 但该量无判据意义（§5.2）**，不掩盖 |
| 伪展向 w 份额阶 | 1.4563 | — | **FAIL 无意义（§5.3）**，不掩盖 |

结论：**判据（速度 L2 三网格 log-log 斜率 ≥ 1.8）以 1.9623 达成**，且在两段单段阶与
再识别阶上自洽；n=128 的速度/压力误差均远小于 3%。该结论现在**在树上可复现**
（`28 passed, 9 skipped`，见 §4 表），不再只存在于报告文字里。

## 7. 复现清单

```bash
PYTHONPATH=src python scripts/summarize_kovasznay_32_64_128.py \
  --runs docs/CASE-FVM-KOVASZNAY-ORDER/runs-32-64-128-zero_gradient \
  --output docs/CASE-FVM-KOVASZNAY-ORDER/runs-32-64-128-zero_gradient \
  --markdown …/RESULTS.md --figure …/order.png --require-grids 32,64,128
PYTHONPATH=src python scripts/plot_kovasznay_field_error.py \
  --fields docs/CASE-FVM-KOVASZNAY-ORDER/runs-32-64-128-zero_gradient/n128/n128-fields.npz \
  --output docs/CASE-FVM-KOVASZNAY-ORDER/runs-32-64-128-zero_gradient/n128/fields-error.png
PYTHONPATH=src python -m pytest tests/test_kovasznay_32_64_128_acceptance.py \
  tests/test_summarize_kovasznay_mixed_config_guard.py -q     # 28 passed, 9 skipped
```

**已知与本单无关的红（未触碰）**：`tests/test_source_staleness.py::
test_live_tree_ledger_reproduces_and_ratchet_is_respected` 在基线 `cc132a8e` 上就失败
（现场重算 fresh 1263 / changed 114，账里记 1276 / 101，13 条归档证据源码已漂移）。
本单既未引入也未回填该账，如实记为 not-fixed。

**平台报告**：TensorPlatform 进展报告 id `18712`（scenario_key `fvm-kovasznay-order`，
含 order.png 与 n=128 fields-error.png 两图）。

建议：平台裁定侧可据本单把 `CASE-FVM-KOVASZNAY-ORDER` 从 `open` 置为 `verified`
（证据：本文件 §3/§6 + `runs-32-64-128-zero_gradient/`），并把「交付过滤器滤除重新生成的
证据文件」列为需要告警的流水线缺陷；若要引用 GCI，需补 n=256 一层（§5.4）。