# TensorFVM 当前状态：2026-10-09

审查基线 `939363506892e0ffef482a27e38c815dd5dc5f25`。本轮由独立子 agent 只读分析源码、重算发布原场，并实际执行 CPU/CUDA 三维单步。二维不可压层流已有可信基础；三维仍为实验内核，尚不能承担已验证的 SUBOFF 上浮破冰流体后端。

## P0：压力投影不相容

`src/tensorfvm/solver3d.py` 的第 171 行中心导数构成第 225 行散度 D 和第 380–385 行速度修正梯度 G；第 268 行压力 A 则是最近邻七点拉普拉斯。中心差分 DG 产生隔一单元的模板，因而 A 与 -DG 不一致。第 401 行 continuity 定义为 max|div u| D/U。

| 实际检查 | 结果 |
|---|---:|
| CPU 48×32×12，dt=0.005，PCG 迭代 | 150 |
| PCG 残差 / 目标 | 1.2178787e-5 / 1.6627987e-5，通过 |
| 修正后 continuity | **0.89199797，未满足不可压投影要求** |
| 内部脉冲 A 与 -DG 相对 L2 差 | **80.44547%** |
| 脉冲中心 A / -DG | 57.742222 / 14.435556 |
| CUDA 32×24×4，PCG 迭代 | 103 |
| CUDA 残差 / 目标 | 4.15051e-6 / 6.40005e-6，通过 |
| CUDA continuity | **0.580335** |

`benchmark_cylinder3d.py:87` 仅检查末步有限值、CFL<1 与 PCG 成功，未要求连续性或全部历史步的 PCG 成功。其 smoke 通过不能证明不可压物理解合格。

复现核心检查（OMP/MKL 各设为 1，PYTHONPATH=src）：

```python
import torch
from tensorfvm.solver3d import Cylinder3DSolver, Cylinder3DConfig
s = Cylinder3DSolver(Cylinder3DConfig(
    nx=48, ny=32, nz=12, max_steps=1,
    pressure_iterations=250, time_step=.005))
print(s.step())
p = torch.zeros_like(s.pressure)
p[4, 12, 30] = 1
s._apply_pressure_boundaries(p)
g = torch.stack([
    s._derivative(p, 2, s.mesh.dx),
    s._derivative(p, 1, s.mesh.dy),
    s._derivative(p, 0, s.mesh.dz)], -1)
a, dg = s._pressure_operator(p), -s._divergence(g)
q = s._pressure_unknown
print(torch.linalg.vector_norm((a-dg)[q]) / torch.linalg.vector_norm(a[q]))
```

应采用守恒面通量及相容压力修正（MAC 或 Rhie–Chow），同时核对固壁零法向通量和压力边界；直接换成中心 DG 仍有棋盘格风险。之后添加算子恒等、逐单元通量散度、固定壁守恒及每步 PCG/连续性门。性能优化排在正确性之后。

## Benchmark 的真实范围

静态 SI 通道：`scripts/validate_suite_channel.py` 的三网格原场、哈希与牵引审计通过；另以 NumPy 从 `si-fields.json` 重算解析误差：

| 网格 | 速度 L2 | 压力梯度误差 |
|---|---:|---:|
| 36×12 | 0.634358% | 1.369853% |
| 72×24 | 0.161971% | 0.344019% |
| 144×48 | 0.042848% | 0.078365% |

此资格仅覆盖静态层流通道。

- NACA0012 Re=1000、4°：历史单网格 Cl/Cd 误差 0.5976%/0.8160%，523 次迭代。`benchmark_airfoil.py:32` 有残差和升阻力门，但发布材料缺完整流场、面牵引和源哈希。本次临时复跑约两分钟未结束，限时终止，**没有宣称当前源码重新通过**。
- SA 平板：历史 Cf 误差 6.6506%、y+=0.81186，真实门槛为 20%（`benchmark_flat_plate.py:22,66`）。缺多网格及完整发布原场，不能当成 3% 验证。
- Re3900 SA-URANS：发布结果失败，Cl RMS=1.070505e-5、St=0.1，未解析周期升力；见 `docs/cylinder-urans/report.rst`。
- DFG：存在三网格 runner（`benchmark_dfg.py:110`），总通过取最细网格（:144），本次未找到发布完整 raw 证据，不认证。
- 30P30N：参考阻力归一化不清，明确仅诊断（`benchmark_30p30n.py:212`）；返回码 0 只表示方程收敛（:429）。
- 跨仓 NumPy 审计证明原始账本一致，不代表执行了 FVM 流体方程或新增移动边界能力。

## 能力与缺口

二维包含 MAC SIMPLE、贴体同位 SIMPLE/Rhie–Chow、非正交修正/最小二乘梯度、O/C-grid 与 Gmsh v2 网格；后端注册和 Mesh2D 契约已有实现。贴体对流一阶迎风（`body_fitted.py:522`）、瞬态隐式 Euler（:896），有实验 SA，尚无验证的转捩/可压模型。

贴体矩阵以 owner/neighbor index_add_ 作用、对角预条件 CG/BiCGSTAB（:641,647）；缺 AMG/多重网格。三维仅静止笛卡尔阶梯圆柱、显式 Euler、中心对流和 Smagorinsky。其 ν_eff Δu（`solver3d.py:375`）不是完整可变黏度守恒应力散度，表面力缺黏性壁力（:349）。三维贴体仅有几何/拓扑，求解器明确拒绝（:125）。

三维真实 z-slab 场、halo 和全局规约已实现（`runtime.py:143`）；二维未域分解。当前一张 RTX3090、PyTorch 2.14.1+cu130 单卡可运行；多 GPU NCCL、工程规模性能和强弱扩展未验证。

静态通道可输出 SI 壁面牵引（`benchmark_suite_channel.py:76`），但尚无通用实际移动边界、ALE/cut-cell、VOF、6DOF/变形网格、版本化求解器 checkpoint/rollback。现阶段适合作独立参考与审计，不能称为双向 FSI 的已验证替代后端。

## 推进顺序

1. **P0**：修复三维相容投影与固壁通量，加入每步验收，保留当前失败证据。
2. **P1**：二维翼型/DFG 发布可追溯 raw 流场、面牵引、残差和源码哈希，至少三网格；先验证静态 traction→FEM 的力/矩/功和单位，再开发移动边界及完整重启。
3. **P2**：守恒二阶对流/时间、贴体三维、可扩展预条件和 GPU 性能；自由液面及破冰在守恒基础稳定后推进。
