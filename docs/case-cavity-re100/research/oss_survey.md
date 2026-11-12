# CASE-CAVITY-RE100 — ①开源横向调研（第 2 轮）

**Mission**: m15452（第 2 轮；前轮 m15369 honest-FAIL，fail_stage/failure_reason 均未留）
**调研日期**: 2026-11-12
**调研面**: 本机 `~/oss-cache`（集群同名目录已递交 grep 作业 cj1124，队列中；本文件只写**实际读过**的源码）

## 1. 实际读到的开源实现

| 来源 | 路径（本地 oss-cache） | 方法 | 对本算例可用性 |
|---|---|---|---|
| pylbm (L. Gouarin, B. Graille) | `pylbm/demo/2D/lid_driven_cavity.py` | LBM D2Q9，`Geier`/`dHumiere` 多项式平衡，动壁边界用**动量型边界**：上壁直接赋值 `m[RHO]=rho_o, m[QX]=rho_o*U, m[QY]=0`，再由 σ 与 `mu/zeta` 反解松弛率 | 算例结构可学（几何/动壁/涡量后处理），但**无 Ghia 对照表**，不能直接作为判据来源 |
| pylbm | `pylbm/demo/2D/lid_driven_cavity_2.py`, `demo/3D/lid_cavity.py` | 同上，带可视化；3D 版是 `lid_cavity` | 仅参考几何与后处理 |
| Maarten-vd-Sande_lbm | `examples/{cylinder_flow.py,dambreak.py}` | LBM D2Q9 教学实现 | **无 cavity 例**（`grep -ril cavity` 空），不可用 |
| hschwane/GraSPH | `GraSPH/` | SPH | 与本算例（不可压稳态 FVM）方法族不同，不适用 |

pylbm 的关键参数（读到的源码常量）：`xmin,xmax,ymin,ymax = 0,1,0,1`、`la=1.0`、
`rho_o=1.0`、`driven_velocity=0.05`、`mu=5e-6`、`zeta=100*mu`。
注意其 `driven_velocity=0.05` 是格子马赫数取值的稳定性折中，Reynolds 数由此组合
（≈1e4 量级），**与 Ghia 的 Re=100 不是同一工况**，故其 demo 不能当作本算例的对照。
可迁移的只有两点结构：(a) 动壁用「壁面动量/镜像」而非「壁面速度梯度」实现——
本仓 `LidCavitySolver` 已用镜像 ghost `2U - u_wall-adjacent`，语义一致；
(b) 涡量/速度模由中心差分后处理——本仓 `vorticity()` 与 `streamfunction()`（DST-I 解
`lap psi = -omega`）已覆盖，且多了流函数定位主涡，比 pylbm 的 `norm_velocity` 更强。

## 2. 仓内既有的开源横向账（本仓已建）

`docs/oss-inventory-fvm/registry.json`（30 个开源项目、8 条 GAP）已含 OpenFOAM lineage
等 T1 项目；本算例不需要新的外部实现，**结论是「方法不用换，缺的是把算例做成一等公民」**：
仓里已有求解器与转录表，缺的是「一次运行 → 逐点对照 → 机检门 → 报告」的闭环工件。
这正是本单交付的内容（见 `REPORT.md`）。

## 3. 不可用/未完成的部分（诚实申报）

- 集群 `~/oss-cache` 的全量 grep（cj1124）在队列中，本文件未纳入其结果；
  若集群侧存在 OpenFOAM `icoFoam` cavity 教程或 Palabos 源码，**结论不会改变**
  （它们同样不含 Ghia 基准表，仍须回到 Ghia 1982 原文表）。
- 未运行任何第三方求解器做交叉验证（超出本单范围，且无 Ghia 表以外的增益）。
