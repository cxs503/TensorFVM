# CASE-FVM-KOVASZNAY-ORDER — 32/64/128 收敛阶（对称闭包修复版，honest partial 32/64）

**日期**: 2026-11-13 · **席**: m17536 (tp-merge-2 generalist) · **仓**: TensorFVM
**锚点**: `CASE-FVM-KOVASZNAY-ORDER`（L2 收敛阶 ≥ 1.8 on 32/64/128）
**基准**: Kovasznay (1948) Proc. Cambridge Philos. Soc. 44:58–62, doi:10.1017/S0305004100023999

## 1. 为什么另起一套（本单的增量）

上一单（m17163）已把该锚的证据链做成"从原始场量重算"的机器门，但 32/64/128 只完成 n=32，
且用的是**历史 free 闭包**：对称挤出的两张 z 面把切向分量当自由未知量（traction-free），
留下一个 O(h) 伪展向速度，把**全 3D 速度范数**的收敛阶压到 1.68（16/32/64），
只能靠"面内 (u,v) 预算"绕过判据辩论。

本单把闭包修正为物理正确的对称条件（du/dz = dv/dz = 0，
复用求解器既有 `zero_gradient_faces` 机制，见 `tensorfvm.verification.kovasznay_symmetry`），
于是伪展向模态被钉到舍入级，**判据可以直接落在全 3D 速度范数上**，不再需要分解辩护。

改动（最小、可验证）：
- `scripts/verify_kovasznay_order.py`：新增 `--symmetry-closure {free,zero_gradient}`（默认 free 向后兼容），
  闭包写进 `cases.json`/`summary.json`；
- `scripts/summarize_kovasznay_32_64_128.py`：level 携带 `symmetry_closure`，混闭包直接报错；
  恒零预算（展向）不再取 log(0) 编造斜率，记 `n/a (identically zero)`；verdict 增 `spanwise_machine_zero`；
- `tests/test_kovasznay_32_64_128_acceptance.py`：新增 `anchor_zg` 证据集（本目录）、
  闭包一致性断言、machine-zero 断言、**全 3D 速度判据**门。

## 2. 计算结果（32/64 两套网格，本机 nice 19 单线程）

Re=40，κ=2π，λ=−0.963741，对称挤出 n×n×1，二阶对流，`coupled`+`reused_lu`，Picard 松弛 0.7，tol 1e−8。

| n | h | cells | vel L2 (全 3D) | vel L2 (面内) | w RMS 份额 | p L2 (去均值) | outer | 收敛 |
|---:|---:|---:|---:|---:|---:|---:|---:|:--|
| 32 | 0.03125 | 1024 | 3.435615e−03 | 3.435615e−03 | 6.78e−18 | 2.885142e−02 | 22 | yes |
| 64 | 0.01562 | 4096 | 8.972614e−04 | 8.972614e−04 | 1.12e−18 | 7.958339e−03 | 22 | yes |

- max|w| 三级（含 32/64）= **1.602e−16**（机器零）；对照 free 闭包 n=32 的 max|w| = 4.53e−03。
- 32→64 观测阶：全 3D **1.9370**（R²=1.000000）、面内 1.9370、压力(去均值) 1.8581、压力(固定规范) 0.9857。
- 图：`partial-32-64/order_32_64.png`（log-log 收敛曲线 + 场量 + 中心线剖面）。
- 数字全部由 `partial-32-64/RESULTS-32-64.md`/`summary.json` 从 `n<N>-fields.npz` 重算生成，零手抄。

## 3. 判据判定（honest partial）

| 判据 | 目标 | 现状 |
|---|---|---|
| 全 3D 速度 L2 阶（32/64/128 三网格最小二乘斜率） | ≥ 1.8 | **未闭合**：n=128 未回（见 §4）；32→64 单段 1.9370 已达标 |
| 面内 (u,v) L2 阶 | ≥ 1.8 | 32→64 段 1.9370 |
| 压力(去均值) L2 阶 | ≥ 1.8 | 32→64 段 1.8581 |
| 伪展向模态 | ~0 | 1.6e−16（PASS） |

**结论**：闭包修复后判据的物理载体（全 3D 速度）已回到二阶区，但锚点要求的**三网格**斜率尚未测量，
故 `feature_ledger` 该锚保持 **open**，本单不声明达标。另外 `tests/..._acceptance.py::anchor_zg`
三网格门仍 skip（honest "not measured yet"，不是 pass）。

## 4. 注意点（坑与局限）

1. **算力**：控制面 load15 从 119 升到 163（nproc=128），按容量纪律重活全部递交集群；
   但集群队列积压 ~405 单（cj1571=n64 zg / cj1572=n128 zg / cj1564=n64 free / cj1565=n128 free 均 queued），
   因此 32/64 改在本机 nice 19 单线程跑（n=32 实测 40 s、n=64 约 6 min，计入 3 核/128，未扰动他人）。
2. **n=128 仍在跑**：本机 PID 869627（n=128 zg，约 2 min/outer 迭代）；跑完后只需
   `python scripts/summarize_kovasznay_32_64_128.py --runs docs/CASE-FVM-KOVASZNAY-ORDER/runs-32-64-128-zero_gradient --output <同目录> --require-grids 32,64,128`
   即可生成三网格 `summary.json`（门会自动变 green 或 honest FAIL），或用集群 cj1572 的 fetch 工件。
3. **跨文档数字不一致（本单新发现，未修）**：`docs/case-kovasznay-order-order-verification/RESULTS_INPLANE.md`
   的 n=32 面内 3.72450e−03 / 压力(去均值) 7.98671e−03 **不能**从本仓已提交的 n=32 原始场
   （`runs-32-64-128/n32/n32-fields.npz`：面内 3.431614e−03、压力去均值 2.895337e−02）复现——
   该文档的 n=32 面内/压力应该来自另一套 run 或另一种口径。凡不能从 npz 复算的历史数字，
   本单一律不用作判据。
4. **压力固定规范阶只有 0.986**：动量方程只定压力梯度，固定规范下压力含一个随网格变化的常数偏差，
   固定规范阶不是格式精度的量度；有意义的是去均值规范（1.8581）。这与上单口径一致。
