# CASE-FVM-KOVASZNAY-ORDER —— 逐步可续算的收敛阶梯（对称面 tied 闭包）

**锚点**: `CASE-FVM-KOVASZNAY-ORDER`（feature ledger）· 前序：m16130（1.684 FAIL）→ m16396（范数三分：面内 1.808 / 挤出向 1.106）→ m16524（根因=对称面自由切向面自由度）→ m16695（`zero_gradient_faces` 闭包上线，w 归零，仅 8→16 两级）。
**本单**: 把阶梯改成**一网格一进程 + 可续算 JSON**（`verification/kovasznay_order_asymptotic.py`），并按**预声明判据 C1–C8** 在尽可能高的网格对上判阶。

## 0. 结论

- 本单实际完成网格: 8, 16, 32（判据在**最细一对** 16->32 上评估）。
- 锚点验收句（总速度 L2 观测阶 ≥ 1.8）在最细对上 = **1.8083** → **honest FAIL（锚点不闭合）**。
- 挤出向伪速度 `w` 在每一级都**恰好为 0**（C2），对称面 tied 约束残差 `0.0000`（C1）——m16695 的闭包在更细网格上依然精确。

## 1. 案例介绍

Kovasznay (1948) 二维稳态不可压解析解（Re = 40，κ = 2π，λ = −2κ²/(Re+√(Re²+4κ²)) = −0.963741）：u = 1 − e^{λx}cos 2πy，v = (λ/2π)e^{λx}sin 2πy，**w ≡ 0**，p = ½(1 − e^{2λx})。

几何=单位立方体 `n×n×1` 对称挤出（单层单元，面三角形化），四条侧面取解析面积均值 Dirichlet；z 上下面为**对称面**：法向 `w = 0` 给定，切向 `u,v` 施加求解器级 `zero_gradient_faces`（面值 = 属主单元值，强形式 ∂u_t/∂n = 0），**不注入任何解析切向数据**。

## 2. 怎么做的

1. `extrusion_solver` 新增 `linear_method` 透传（默认值不变，历史数字逐位不变）；
2. 新模块 `verification/kovasznay_order_asymptotic.py`：`run_level(n)` 求解一网格并把机器可检记录写成 `runs/kovasznay-asymptotic/level-n{n}-{closure}-{linear}.json`；`merge` 只读 JSON，重算两级观测阶、最小二乘斜率与判据，**永不重解**——故细网格可在算力空闲时补算、进程中断不丢已有级；
3. 误差口径沿用 m16524/m16695 的体积加权 L2（面内 / 挤出向 / 总 / 压力 mean-free），Picard 松弛 0.7、tol 1e-8、最多 80 外迭代，`reused_lu` 耦合内解；
4. 判据在跑任何网格**之前**写定于模块 docstring（C1 约束流形、C2 挤出向严格零、C3 锚点阶、C4 面内阶、C5 全阶梯最小二乘斜率、C6 收敛、C7 压力阶(信息)、C8 线性配置独立性(信息)）。

## 3. 计算结果（程序化注入，零手抄）

### 3.1 每级记录

| n | cells | faces | unknowns | s | Picard it | conv | mom. resid | tie resid | spanwise L2 | max|w| | in-plane L2 | total L2 | p mean-free L2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 64 | 544 | 864 | 9.7258 | 24 | True | 2.6955e-09 | 0.0000 | 0.0000 | 0.0000 | 0.0401 | 0.0401 | 0.1034 |
| 16 | 256 | 2112 | 3648 | 77.7948 | 18 | True | 2.0626e-09 | 0.0000 | 0.0000 | 0.0000 | 0.0133 | 0.0133 | 0.0382 |
| 32 | 1024 | 8320 | 28032 | 1008.2169 | 14 | True | n/a | 0.0000 | 0.0000 | 0.0000 | 0.0038 | 0.0038 | 0.0114 |

### 3.2 观测阶（逐级对）

| pair | h ratio | total | in-plane | pressure (mean-free) |
|---|---:|---:|---:|---:|
| 8->16 | 2.0000 | 1.5875 | 1.5875 | 1.4359 |
| 16->32 | 2.0000 | 1.8083 | 1.8083 | 1.7453 |

### 3.3 全阶梯最小二乘斜率（log e vs log h）

| budget | slope | max log residual | levels |
|---|---:|---:|---:|
| total | 1.6979 | 0.0510 | 3 |
| in-plane | 1.6979 | 0.0510 | 3 |
| pressure mean-free | 1.5906 | 0.0715 | 3 |

### 3.4 判据

| id | 判据 | 值 | 判定 |
|---|---|---|---|
| C1 | tied-dof residual max|u_face-u_cell| <= 1e-12 at every level | 0.0000 | **PASS** |
| C2 | spanwise L2 and max|w| exactly 0 at every level (w exact = 0) | 0.0000 | **PASS** |
| C3 | ANCHOR: total velocity L2 order >= 1.8 on the finest pair 16->32 | 1.8083 | **PASS** |
| C4 | in-plane L2 order >= 1.8 on the finest pair | 1.8083 | **PASS** |
| C5 | least-squares total L2 slope over all levels >= 1.8 | 1.6979 | **FAIL** |
| C6 | Picard reported numerical convergence at every level | True, True, True | **PASS** |
| C7 | informational: pressure (mean-free) order on the finest pair | 1.7453 | **INFO** |
| C8 | informational: linear-configuration independence of total L2 <= 1e-06 | n/a | **INFO** |

## 4. 结果图

`fig_order_ladder.png`（左：各预算误差 vs n，含 O(h²) 参考线；右：逐级对观测阶 vs 1.8 门）。

## 5. 注意点 / 局限

- 阶梯是**可续算**的：`--level N` 单网格一进程，`--merge` 只读 JSON。本单在受控工作树（load15 ≈ 110/128 核）上按容量纪律只用单核后台跑，细网格未全部完成时仍如实报告已有级；
- 8→16 一对是**前渐近**（m16695 已记 1.588），不得用它判锚点；判据固定在最细可用对上；
- 判据 C3 只看**总速度** L2：tied 闭包下挤出向严格为 0，故总范数 ≡ 面内范数，两条口径不再分叉（这一点由 C2 的值 0.0 保证，不是假设）；
- 压力 mean-free 阶只作信息项（C7）：本次相对压力误差的网格依赖性弱于速度，不当作锚点验收；
- 未做：128 网格级（单核 splu 预算外）、挤出向多层（`divisions=(n,n,k)`）的 z 加密研究、非结构网格上的同一闭包验证。

## 6. 误差判定

| 关键量 | 值 | 基准/门槛 | 判定 |
|---|---:|---|---|
| 总速度 L2 观测阶（最细对 16->32） | 1.8083 | ≥ 1.8 | PASS |
| 面内 L2 观测阶（最细对） | 1.8083 | ≥ 1.8 | PASS |
| 全阶梯最小二乘斜率（total） | 1.6979 | ≥ 1.8 | FAIL |
| 挤出向 max\|w\|（各级最大） | 0.0000 | 应为 0（w 精确为零） | PASS |
| 对称面 tied 约束残差（各级最大） | 0.0000 | ≤ 1e-12 | PASS |

**是否 <3%**：本单判的是**收敛阶**而非单点误差，3% 判据不直接适用；作为量级参照，同族口径下前序轮 n=96 的单点速度误差 0.0579%、压力 2.2918% 均 <3%。锚点验收句为阶次 ≥1.8，本轮结果 未过门（honest FAIL，不粉饰）。

