# NativeVelocitySpace3D 精确面法向消元审计（m2891）

模块 `tensorfvm.native_velocity_space3d` 被 `native_hybrid_viscous3d` → `native_hybrid_stokes3d` 生产路径导入，但此前没有任何直接测试：`normal_faces` 的旋转基分支只通过求解器级用例间接触及。本审计把该模块两条自述主张变成可机器裁决的判据：

1. “P 的列正交，把约化增量映射到无约束单元速度与容许面速度”；
2. “不使用法向罚函数（no normal penalty）”。

审计网格两块：结构化通道块 8 个六面体 / 72 个面（内部面 24、边界面 48，法向轴对齐），斜四面体对 2 个单元 / 7 个面（边界面法向三分量均非零）；随机种子 20261103。

## 1 代数不变量（重建独立约束矩阵 C 与稠密参考解）

| 配置 | 约化自由度 | ‖PᵀP−I‖∞ | ‖C P‖∞ | 零空间维数差 | restrict∘expand | ‖C u‖∞（投影后） | u_elim vs u_KKT | 判定 |
|---|---|---|---|---|---|---|---|---|
| channel_cartesian_traction_free | 240 | 0.00e+00 | 0.00e+00 | 0 | 0.00e+00 | 0.00e+00 | 0.00e+00 | PASS |
| channel_cartesian_one_component | 239 | 0.00e+00 | 0.00e+00 | 0 | 0.00e+00 | 0.00e+00 | 1.55e-15 | PASS |
| channel_normal_axis_aligned | 239 | 0.00e+00 | 0.00e+00 | 0 | 0.00e+00 | 0.00e+00 | 1.47e-15 | PASS |
| tetra_cartesian_traction_free | 27 | 0.00e+00 | 0.00e+00 | 0 | 0.00e+00 | 0.00e+00 | 0.00e+00 | PASS |
| tetra_cartesian_one_component | 26 | 0.00e+00 | 0.00e+00 | 0 | 0.00e+00 | 0.00e+00 | 3.59e-16 | PASS |
| tetra_normal_slanted | 26 | 2.22e-16 | 1.52e-18 | 0 | 2.78e-17 | 0.00e+00 | 3.98e-16 | PASS |
| tetra_normal_plus_cartesian | 24 | 2.22e-16 | 1.52e-18 | 0 | 2.78e-17 | 0.00e+00 | 2.61e-16 | PASS |
| tetra_normal_two_faces | 25 | 2.22e-16 | 9.42e-18 | 0 | 2.78e-17 | 0.00e+00 | 5.15e-16 | PASS |

零空间维数差 `nullity(C) − rank(P) = 0` 是“精确消元、无罚项”的离散证据：约化空间的维数正好等于硬约束的零空间维数，既不多（罚项会多出伪自由度）也不少。
`‖C u_KKT‖∞` 随 KKT 硬约束求解为机器精度，`u_elim` 与 `u_KKT` 相对差见表第 8 列。

## 2 罚函数参照：消元必须是 ε→0 的极限，且误差一阶

| 配置 | 局部收敛阶（逐级） | ε=0.1 | ε=0.01 | ε=0.001 | ε=0.0001 | ε=1e-05 | ε=1e-06 | ‖Cu‖ε=0.1 | ‖Cu‖ε=0.01 | ‖Cu‖ε=0.001 | ‖Cu‖ε=0.0001 | ‖Cu‖ε=1e-05 | ‖Cu‖ε=1e-06 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| channel_cartesian_one_component | 0.09 / 0.45 / 0.87 / 0.98 / 1.00 | 1.56e-01 | 1.27e-01 | 4.50e-02 | 6.02e-03 | 6.24e-04 | 6.26e-05 | 1.22e-03 | 9.94e-04 | 3.51e-04 | 4.71e-05 | 4.87e-06 | 4.89e-07 |
| channel_normal_axis_aligned | 0.09 / 0.47 / 0.88 / 0.99 / 1.00 | 2.09e-01 | 1.69e-01 | 5.77e-02 | 7.61e-03 | 7.86e-04 | 7.88e-05 | 1.61e-03 | 1.30e-03 | 4.43e-04 | 5.85e-05 | 6.04e-06 | 6.06e-07 |
| tetra_cartesian_one_component | 0.44 / 0.87 / 0.98 / 1.00 / 1.00 | 3.09e-01 | 1.13e-01 | 1.54e-02 | 1.60e-03 | 1.60e-04 | 1.60e-05 | 1.49e-02 | 5.47e-03 | 7.44e-04 | 7.72e-05 | 7.75e-06 | 7.75e-07 |
| tetra_normal_slanted | 0.39 / 0.84 / 0.98 / 1.00 / 1.00 | 4.86e-01 | 1.99e-01 | 2.89e-02 | 3.03e-03 | 3.04e-04 | 3.04e-05 | 2.54e-02 | 1.04e-02 | 1.51e-03 | 1.58e-04 | 1.59e-05 | 1.59e-06 |
| tetra_normal_plus_cartesian | 0.47 / 0.88 / 0.99 / 1.00 / 1.00 | 5.12e-01 | 1.75e-01 | 2.31e-02 | 2.38e-03 | 2.39e-04 | 2.39e-05 | 2.72e-02 | 9.30e-03 | 1.23e-03 | 1.27e-04 | 1.27e-05 | 1.27e-06 |
| tetra_normal_two_faces | 0.40 / 0.84 / 0.98 / 1.00 / 1.00 | 6.87e-01 | 2.76e-01 | 3.97e-02 | 4.16e-03 | 4.18e-04 | 4.18e-05 | 2.79e-02 | 1.12e-02 | 1.61e-03 | 1.68e-04 | 1.69e-05 | 1.69e-06 |

斜率≈1 说明“罚刚度 1/ε”解与消元解的差就是一阶 ε；消元解本身不依赖任何罚系数，即模块自述的 no normal penalty 成立。

## 3 旋转基的切平面精确性

| 配置（rotated=True） | max|n·u_face| |
|---|---|
| channel_normal_axis_aligned | 0.00e+00 |
| tetra_normal_slanted | 0.00e+00 |
| tetra_normal_plus_cartesian | 0.00e+00 |
| tetra_normal_two_faces | 2.78e-17 |

轴对齐与斜法向（三分量均非零）两种旋转基都在机器精度上把面自由度压在切平面内，未被约束的两个切向自由度经 `restrict` 往返无损。

## 4 下游算子：自由滑移消元不产生零能模态

| 算子 | 约化自由度 | 对称性误差 | λ_min | λ_max | 零模态数 |
|---|---|---|---|---|---|
| free_slip_box | 27 | 0.00e+00 | 0.027861 | 12.684658 | 0 |
| no_slip_box | 3 | 0.00e+00 | 12.000000 | 12.000000 | 0 |

全部边界面为法向约束（自由滑移）时约化算子仍严格正定，不存在因消元引入的伪零能模态；全 Dirichlet（无滑移）单胞只剩 3 个自由面自由度，说明两个分支都按合同维数收缩。

## 5 负控：过定输入必须继续被拒绝

| 负控用例 | 结果 | 抛出的信息 |
|---|---|---|
| normal_on_interior_face | 拒绝 | normal velocity constraints require boundary faces |
| normal_with_one_component | 拒绝 | normal velocity constraints require three components |
| normal_over_cartesian | 拒绝 | normal and Cartesian component constraints cannot overlap on a face |

## 6 判定

全部代数不变量、负控与下游算子判据：**PASS**。审计未发现模块缺陷（消元在整个测试范围内是精确的）；本单关闭的是“已合并功能零直接测试”的验证欠账，并把此前只靠求解器用例间接覆盖的旋转基分支纳入回归网。

判据阈值：正交性/约束残差/往返/幂等 ≤1e-13，KKT 对照 ≤1e-11（相对），罚函数斜率 >0.9，自由滑移算子 λ_min>0 且零模态数=0。

复算：`python3 scripts/audit_native_velocity_space3d.py`（本机 <2 s，无迭代求解）；回归：`python3 -m pytest tests/test_native_velocity_space3d.py`。
