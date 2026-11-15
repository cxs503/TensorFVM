## Kovasznay order estimator: negative control

Verdict: PASS -- reference slope 1.9623, control slope 1.2456, margin 0.7167

| series | convection order | closure | velocity slope | p mean-free slope | reaches >= 1.8 |
|---|:--:|:--:|---:|---:|:--|
| second order (convection_order=2) | 2 | zero_gradient | 1.9623 | 1.8777 | yes |
| first order (convection_order=1) | 1 | zero_gradient | 1.2456 | 1.0138 | no |

| series | n | h | vel L2 (full) | p L2 (mean-free) | converged |
|---|---:|---:|---:|---:|:--|
| second order (convection_order=2) | 32 | 0.031250 | 3.435615e-03 | 2.885142e-02 | yes |
| second order (convection_order=2) | 64 | 0.015625 | 8.972614e-04 | 7.958339e-03 | yes |
| second order (convection_order=2) | 128 | 0.007812 | 2.262325e-04 | 2.136311e-03 | yes |
| first order (convection_order=1) | 32 | 0.031250 | 4.997776e-03 | 2.046736e-02 | yes |
| first order (convection_order=1) | 64 | 0.015625 | 1.939693e-03 | 8.948565e-03 | yes |
| first order (convection_order=1) | 128 | 0.007812 | 8.888359e-04 | 5.019535e-03 | yes |

Gates: {"control_premise_ok": true, "all_levels_converged": true, "reference_passes_threshold": true, "control_rejected_by_threshold": true, "control_below_ceiling": true, "margin_reached": true}
# Kovasznay 收敛阶判据的负向对照（敏感性验证）
**锚点**: CASE-FVM-KOVASZNAY-ORDER · **仓**: TensorFVM · **判据来源**: `tensor_benchmark_cases` id=137 / feature ledger
## ① 案例介绍
Kovasznay (1948) 精确解（Re=40，单位域，无源项定常 NS）是验收序列的基准。验收判据只问「32/64/128 三套网格上全三维速度相对 L2 误差的 log-log 斜率是否 ≥ 1.8」。本文件回答的是判据本身的可信度问题：**这个斜率估计器会不会对任何格式都报二阶？** 为此在同一算例、同一网格族、同一展向闭合下，只改动对流格式阶，做一条预期被拒的序列（负向对照）。
| 序列 | 对流格式阶 | 展向闭合 | Re | 网格 |
|---|:--:|:--:|---:|---|
| 参考（验收证据） | 2 | zero_gradient | 40.0 | 32/64/128 · n×n×1 |
| 对照（本文件新增） | 1 | zero_gradient | 40.0 | 32/64/128 · n×n×1 |
## ② 怎么做的
1. `scripts/verify_kovasznay_order.py --convection-order 1 --symmetry-closure zero_gradient` 逐网格求解，写出 `cases.json` 与 `n<N>-fields.npz`（场量 + 解析场 + 体积权重），运行在集群队列（不占控制面）。
2. `scripts/kovasznay_order_sensitivity.py` 从两套序列的 npz **重算**各层 L2，再各自拟合斜率；并强制两条序列的四元配置（Re / 闭合 / 对流阶 / 算法）**只在对流阶上不同**，否则拒绝给出判别裕度（否则对照不成立）。
3. 闸门（脚本退出码与 pytest 共用同一判据）：参考序列须过 threshold；对照序列必须被 threshold **拒**（斜率 ≤ 1.5）；两者斜率差须 ≥ 0.5。
4. 任一条件不成立即如实记 FAIL——那意味着验收数字不构成「二阶」的证据。
## ③ 计算结果
| 序列 | 速度 L2 斜率（全 3D） | 压力 L2 斜率（去均值） | 是否越过判据 |
|---|---:|---:|:--|
| 参考（二阶） | 1.9623 | 1.8777 | 是 |
| 对照（一阶 upwind） | 1.2456 | 1.0138 | 否 |
| 判别裕度 | 0.7167 | — | 需求 ≥ 0.5 |
逐层 L2（体积加权，误差由 npz 重算）：
| 序列 | n | h | 速度 L2（全 3D） | 压力 L2（去均值） | 收敛 |
逐段观测阶（两点估计）：
| 序列 | 32→64 | 64→128 |
|---|---:|---:|
| second order (convection_order=2) | 1.9370 | 1.9877 |
| first order (convection_order=1) | 1.3655 | 1.1258 |
## ④ 结果图
见 `docs/CASE-FVM-KOVASZNAY-ORDER/order-sensitivity.png`：左图两条序列的误差-网格log-log 曲线与 h¹/h² 参考线；右图为拟合斜率与判据线（含判别裕度）。
## ⑤ 注意点
- 对照序列**故意失败**：它是负向对照，不是验收失败；两条序列的配置差异被脚本锁死为「仅对流阶」，因此斜率差只能归因于格式阶。
- 一阶 upwind 在此算例上仍有扩散主导的收敛行为，斜率不必然恰为 1；判据取「被 threshold 拒绝且裕度 ≥ 阈值差」而非「等于 1」。
- 参考序列斜率受三网格最小二乘限制，不含网格族独立性检验；结论的适用范围仍是 Re=40 + `zero_gradient` 展向闭合 + 二阶对流。
- 上游 `free`（无牵引）闭合序列只有 n=32，未参与任何拟合（诚实 not-measured）。
## ⑥ 误差判定
- 参考序列速度斜率 1.9623 ≥ 1.8：**通过**（判据非误差百分比，故此处不适用 <3% 门槛；最细网格速度 L2 见上表，量级 2.26e-04）。
- 对照序列速度斜率 1.2456 < 1.8：**被拒**，与一阶格式的名义阶一致，说明估计器可分辨格式阶。
- 判别裕度 0.7167 vs 需求 0.5：**满足**。
- 闸门总判定：**PASS**（`control_premise_ok=True`, `all_levels_converged=True`, `reference_passes_threshold=True`, `control_rejected_by_threshold=True`, `control_below_ceiling=True`, `margin_reached=True`）。
## 复现
```bash
PYTHONPATH=src python scripts/verify_kovasznay_order.py --output docs/CASE-FVM-KOVASZNAY-ORDER/runs-32-64-128-firstorder/n<N> --n <N> --convection-order 1 --symmetry-closure zero_gradient
PYTHONPATH=src python scripts/kovasznay_order_sensitivity.py --reference docs/CASE-FVM-KOVASZNAY-ORDER/runs-32-64-128-zero_gradient --control docs/CASE-FVM-KOVASZNAY-ORDER/runs-32-64-128-firstorder --output docs/CASE-FVM-KOVASZNAY-ORDER/order-sensitivity.json --markdown docs/CASE-FVM-KOVASZNAY-ORDER/runs-32-64-128-firstorder/RESULTS.md --figure docs/CASE-FVM-KOVASZNAY-ORDER/order-sensitivity.png
PYTHONPATH=src python -m pytest tests/test_kovasznay_order_sensitivity.py -q
```
生成器: `scripts/render_kovasznay_negative_control_doc.py`；数据源: `order-sensitivity.json`（ndjson: {"reference_slope": 1.9623452238776438, "control_slope": 1.2456486182245672, "margin": 0.7166966056530766}）
