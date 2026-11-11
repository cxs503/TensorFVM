# Lax 激波管算例（TensorFVM）——精确解对照与网格/参数敏感性

> 算例：Lax (1954) 一维 Euler 激波管；求解器 `tensorfvm.compressible`（MUSCL-Hancock + HLLC + mc 限制器，CFL=0.4）。
> 生成：`2026-11-11T20:58:57Z`；耗时 265.5 s；torch 2.7.1，线程数 8。
> 本文件与全部数字由 `scripts/lax_shock_tube_benchmark.py` 程序化生成，源数据 `artifacts/lax_report.json`（零手抄）。

## ① 问题与参考出处

**物理背景。** 一维理想气体激波管（Riemann 问题）：隔膜两侧压差驱动产生左行稀疏波 + 接触间断 + 右行激波。与 Sod 相比，Lax 算例的右波是**强激波**（p*/p_R = 4.319，ρ*_R/ρ_R = 2.608），接触两侧星区密度相差 3.785 倍，左行稀疏波扇跨 [0.079, 0.238]；强激波、强稀疏波与接触间断同时在场，对通量波速估计与接触锐化更敏感。

**星区状态。** u*/c*_L = 0.483（亚声速），u*/c_R = 1.209。

**参考出处。** 初值数据取自 Lax, P. D. (1954), Comm. Pure Appl. Math. 7:159-193；同一数据集被 Toro, E. F. (2009), Riemann Solvers and Numerical Methods for Fluid Dynamics, 3rd ed., Wiley, ch. 4 (test problem 2 + Sec. 4.2 exact-solution construction) 复述为标准一维测试算例。Ref values are the exact solution of the stated initial-value problem, recomputed at run time by two independent solvers (in-repo Toro iteration + from-scratch bisection in tensorfvm.compressible.lax); no digit is transcribed from the printed table.

**自相似精确解（本节所有数值由程序算出，见 `artifacts/lax_report.json:exact`）。** 左波=rarefaction，右波=shock；星区压力 p* = 2.466098、星区速度 u* = 1.528723、ρ*_L = 0.344568、ρ*_R = 1.304085；波速：稀疏波头 -2.633565、稀疏波尾 -1.636697、接触 1.528723、激波 2.479321；t = 0.16 时位于 x = 0.0786 / 0.2381 / 0.7446 / 0.8967。

**参考值的独立复核（两套互不共享代码的精确解）。**

| 量 | 仓内 Toro 迭代 | 独立二分重写 | 相对差 |
|---|---|---|---|
| p* | 2.466098 | 2.466098 | 9.004e-16 |
| u* | 1.528723 | 1.528723 | 2.905e-16 |
| ρ*_L | 0.344568 | 0.344568 | 8.055e-16 |
| ρ*_R | 1.304085 | 1.304085 | 3.405e-16 |

整条自相似剖面上（4001 个 ξ 采样点）两套解的最大绝对偏差 5.773e-15（判据 ≤ 1.000e-09 相对）。右激波上的 Rankine-Hugoniot 残差（S(U*-U)=F*-F 三分量）：质量 1.374e-16、动量 0.000000、能量 0.000000。

## ② 方法与实现

- 求解器：仓内 `tensorfvm.compressible`（`EulerSolver`：有限体积，二阶 MUSCL-Hancock 预测-校正，HLLC 通量，CFL 自适应步长，outflow/periodic 边界），**未新增求解器代码**，只新增算例 + 计量层 `tensorfvm.compressible.lax`。
- 计量：以自相似坐标 ξ=(x-x0)/t 取精确解，逐网格比较 ρ/u/p；另取 4 个星区平台值（p*, u*, ρ*_L, ρ*_R，窗口离间断 15% 子区间）与 3 个波面位置（稀疏波扇中点、接触、激波；**同一估计器**同时作用于数值场与 20001 点细网格上的精确场，沿场量跨中值做亚网格线性插值定位，估计器偏差在比较中抵消）。
- 参考值：两套独立精确解（仓内 Toro 压力函数迭代 + 本模块从零重写的二分求解 + Rankine-Hugoniot 自洽检验），互证到机器精度后再作为 ref。
- 复现：`python scripts/lax_shock_tube_benchmark.py --out docs/lax-shock-tube`。

## ③ 参数表（全部无量纲）

| 参数 | 值 |
|---|---|
| γ（两侧理想气体） | 1.4 |
| 左态 ρ_L / u_L / p_L | 0.445 / 0.698 / 3.528 |
| 右态 ρ_R / u_R / p_R | 0.5 / 0.0 / 0.571 |
| 计算域 / 隔膜位置 x0 | [0, 1] / 0.5 |
| 终止时间 t_end | 0.16 |
| c_L / c_R | 3.3316 / 1.2644 |
| 星区马赫数 u*/c*_L | 0.4829 |
| 网格 | nx ∈ [200, 400, 800, 1600]（均匀，ny=1） |
| 通量/重构/限制器/CFL | HLLC / MUSCL-Hancock / mc / 0.4 |
| 边界 | outflow/outflow/periodic/periodic |
| 验收判据 | 平台值/波面位置/L1 相对误差 ≤ 2% |

## ④ 对照表（ref vs got，nx = 1600）

| 量 | ref（精确解） | got（数值） | 相对误差 | <2% |
|---|---|---|---|---|
| 平台 p_star | 2.466098 | 2.466098 | 0.000% | OK |
| 平台 u_star | 1.528723 | 1.528723 | 0.000% | OK |
| 平台 rho_star_l | 0.344568 | 0.344568 | 0.000% | OK |
| 平台 rho_star_r | 1.304085 | 1.304085 | 0.000% | OK |
| 波面 x_fan（稀疏波扇中点） | 0.1543 | 0.1544 | 0.040% | OK |
| 波面 x_contact（接触间断） | 0.7446 | 0.7450 | 0.053% | OK |
| 波面 x_shock（激波） | 0.8967 | 0.8968 | 0.016% | OK |
| 全场相对 L1 误差 rho | — | — | 0.358% | OK |
| 全场相对 L∞ 误差 rho | — | — | 40.629% | 参照项 |
| 全场相对 L1 误差 u | — | — | 0.056% | OK |
| 全场相对 L∞ 误差 u | — | — | 28.824% | 参照项 |
| 全场相对 L1 误差 p | — | — | 0.035% | OK |
| 全场相对 L∞ 误差 p | — | — | 14.559% | 参照项 |

**判定：全部通过**（nx=1600；质量漂移 4.699e-16，正性修补次数 0）。

## ⑤ 收敛性与参数敏感性

| nx | L1ρ | L1u | L1p | ρ*_R 平台误差 | 步数 | 秒 |
|---|---|---|---|---|---|---|
| 200 | 1.943% | 0.676% | 0.360% | 0.040% | 378 | 6.35 |
| 400 | 1.015% | 0.228% | 0.144% | 0.004% | 754 | 14.61 |
| 800 | 0.631% | 0.166% | 0.089% | 0.001% | 1507 | 43.14 |
| 1600 | 0.358% | 0.056% | 0.035% | 0.000% | 3012 | 125.46 |

- 观测收敛阶（rho，逐对网格）：0.94, 0.69, 0.82
- 观测收敛阶（u，逐对网格）：1.57, 0.46, 1.57
- 观测收敛阶（p，逐对网格）：1.33, 0.70, 1.35

参数敏感性（nx = 400，与同网格基线对照）：

| 变体 | L1ρ | L1u | L1p | p* 平台误差 | ρ*_R 平台误差 |
|---|---|---|---|---|---|
| 基线 mc/order2/cfl0.4 | 1.015% | 0.228% | 0.144% | 0.001% | 0.004% |
| limiter=minmod | 1.689% | 0.372% | 0.230% | 0.001% | 0.045% |
| limiter=vanleer | 1.167% | 0.259% | 0.160% | 0.001% | 0.006% |
| order=1 | 4.274% | 1.387% | 0.863% | 0.034% | 1.556% |
| cfl=0.2 | 1.059% | 0.238% | 0.150% | 0.000% | 0.004% |
| cfl=0.8 | 0.942% | 0.227% | 0.145% | 0.001% | 0.005% |

## ⑥ 失败模式与工件

**已知失败模式 / 局限。**
  1. 接触间断与激波处仍是 O(dx) 抹平：全场 L1 误差随网格一阶下降（见 ⑤ 观测阶），因此 ≤2% 的判据只能在足够细的网格上满足；平台值与波面位置对网格不敏感，是本算例的主判据。
  2. 精确解在 t→0 与 x=x0 处是奇点，网格越细时间步越小，成本线性增长。
  3. L∞ 误差在间断处保持 O(1)（本网格族 ρ 约 40%、p 约 15%）：这是迎风格式在接触/激波上的固有抹平，不应作为收敛判据；判据取平台值、波面位置与 L1。
  4. 本算例为单材料理想气体（pinf=0），不覆盖 stiffened-gas/多材料路径（那由 `two_gas_interface` / `shock_bubble` 覆盖）。

**工件。**
- `artifacts/lax_report.json`：全部数值（ref、独立复核、RH 残差、扫描、敏感性、判定）。
- `figures/lax_profiles_vs_exact.png`：Numerical vs exact rho/u/p profiles (nx = 1600, MUSCL-Hancock + MC limiter, HLLC).
- `figures/lax_convergence.png`：Relative L1 error vs nx (log-log) and observed order; dashed line = 2 % acceptance level.
- `figures/lax_sensitivity.png`：Sensitivity of the post-shock density plateau error to limiter / order / CFL at nx = 400; dashed line = 2 % criterion.
- 复现命令：`python scripts/lax_shock_tube_benchmark.py --out docs/lax-shock-tube`（单进程约 265.5 s）。

**生态摩擦（如实记）。**
  1. `python -m tensorfvm.compressible` 的 `--legs` 只认 sod/interface/rm/bubble/vortex；Lax 之前仓里根本没有该算例（新算例只能走新入口，未改动既有 CLI 语义）。
  2. 默认 torch 线程数=128 时 200 网格 1-D 算例 23 ms/步，远慢于单线程 I/O 受限的实际成本，小算例应显式限制线程数。
  3. 知识库检索端点（KE engine :8021）离线，`开工三查` 只能退化为 API/仓内文档检索。
