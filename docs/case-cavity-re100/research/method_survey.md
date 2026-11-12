# CASE-CAVITY-RE100 — ②方法谱系与③选型沉淀（第 2 轮）

## 1. 问题设定与基准出处

方腔驱动流（lid-driven cavity）：`[0,L]^2`，L=1，三壁无滑移，顶盖 `u=U=1`，
`Re = UL/ν = 100` → `ν = 0.01`（ρ=1）。稳态。判据来源：

> U. Ghia, K. N. Ghia, C. T. Shin, *High-Re solutions for incompressible flow using the
> Navier–Stokes equations and a multigrid method*, **J. Comput. Phys. 48 (1982) 387–411**,
> Table I（x=0.5 竖中线上的 u）、Table II（y=0.5 横中线上的 v）；
> 主涡中心 (0.617, 0.734)L。

本仓转录唯一存放处 `src/tensorfvm/lid_cavity_reference.py`（自检
`self_check()`＋`tests/test_ghia_reference_integrity.py`／`test_ghia_reference_consistency.py`
钉住极值与单调性）。本单**不重抄任何数字**，一律 `table(100.0, which)` 取值。

## 2. 方法谱系（怎么做这个算例才算对）

| 文献 | 方法 | 网格/精度 | 对本单的意义 |
|---|---|---|---|
| Ghia et al. 1982 | 涡量-流函数 + 强隐式耦合(CSI) + 多重网格，中心差分 | 129×129 / 257×257 | 判据本身；其表格 4–5 位有效数字，**含 ~1% 量级的制表不确定度** |
| Schreiber & Roache 1983 | 高阶时间推进 + 压力梯度 | ~129² | 与 Ghia 同量级，说明 129² 是「1% 级」分辨率 |
| Botella & Peyret 1998 | Chebyshev–Tau 谱方法（稳态 Newton 驱动） | 谱精度 | 公认比 1982 表更准的**二阶基准**；用于判断残差是否来自参考而非求解器 |
| Erturk et al. 2005 / Bruneau & Saad 2006 | 高精度 FDM / 多重网格 FVM（Re 可达 1e4 量级） | 细网格 | 说明 Re=100 是「低 Re、易达标」段，难点在**边界与提取**，不在稳定性 |
| Hou et al. 1995（LBM） | LBM 基准 | — | 与本仓 `cross_comparison_fvm_lbm` 同族，非本单判据 |

方法学要点（选型理由，逐条对应实现）：
1. **动壁边界**：镜像 ghost `u_ghost = 2U − u_adjacent` 保证壁面无穿透＋切向速度精确
   （与 pylbm 的动量型动壁在连续极限等价）。
2. **闭合方腔的压力问题**：全 Neumann，奇异性靠零模钉死＋`sum(div u*)=0`；本仓用 DCT-II
   对角化直接求逆，`max|div u|` 可达 1e-13 量级（本单作为独立完整性判据 C6）。
3. **提取**：u 在 x=L/2 的面（交错网格天然落在面上），v 在 y=L/2；与 Ghia 表坐标
   **不对齐**（表点为 0.9766/0.9688/… 非均匀），必须有插值 → 用三次 Hermite
   （`hermite_interpolate`，误差 ~1e-3 量级，见 `docs/verification-lid-cavity-ladder`
   的提取审计），并把插值误差与求解误差分开报告。
4. **判据**：算例池只写「容差 <5%」，本单把它拆成 7 条机检（逐点最大绝对偏差 ×2、
   极值相对误差 ×2、主涡中心位移、散度完整性、稳态残差），**任何一条挂即整体 FAIL**，
   且每条给出裕度，避免「平均掩蔽局部」。
5. **收敛性**：不靠「误差随网格变小」的空话，而是(a) 逐分辨率对照表；(b) 把各分辨率
   解块平均到共同 32² 网格做**自收敛**差范数，用 `p = log(e_coarse/e_fine)/log 2` 给
   观测阶；(c) 极值的 Richardson 外推作为连续极限估计。

## 3. 选型结论（迁移方案）

- **方法**：不换方法。复用 `tensorfvm.lid_cavity.LidCavitySolver`（交错 MAC 有限体积、
  van Leer 守恒通量、显式投影、DCT-II Neumann Poisson）＋ Ghia 转录层。
- **改动面**：新增 `src/tensorfvm/case_cavity_re100.py`（算例一等公民：运行→逐点打分→
  7 条判据→工件）、`scripts/case_cavity_re100.py`（CLI/合并/作图）、
  `tests/test_case_cavity_re100.py`（机检门，从工件**重算**误差，不信任存档结论）。
- **分辨率阶梯**：32/64/128（本机），256（集群 cj1128，若能取回则入档）。
- **与 Ghia 表的偏差方向说明**：本仓 n=128 的 `u_min ≈ −0.2138`，比 Ghia 的 −0.21090
  深 1.4%；谱方法文献把 Re=100 的 `u_min` 定在 −0.214 附近，故这 1.4% 里主要成分是
  **1982 表的制表误差**，不是求解误差——这一点用自收敛+Richardson 外推在本单内独立佐证，
  不引用无法核验的二手数字。
