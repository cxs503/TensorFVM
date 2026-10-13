# 混合标量单元约束及守恒反力

2026-10-13。此模块用于后续 omega 壁邻单元约束，**不代表完整壁面函数已接入，也不代表 SUBOFF 物理精度通过**。

## 离散与收支

`NativeHybridScalar3D(..., cell_constraints=(indices,values))` 将指定单元从自由未知量中消去，共享面仍按原方程求解。几何扩散算子不被改写；MMP 修正与雅可比使用实际自由节点。默认没有单元约束，保留已有算法。

对于原单元方程

```
R = storage + div(advective_flux) + diffusion_reaction + destruction - production
```

在约束单元上定义附加源反力 `lambda=R`，非约束单元 lambda=0。局部和全局账本检验原收支减去 lambda；工作账本同时扣除 phi*lambda，并把它计入工作尺度。输出始终保留 `cell_balance_without_constraint`、`global_balance_without_constraint`、原单元方程残差、`cell_constraint_reaction`、`cell_constraint_work`、约束索引和指定值，不能把约束前的大残差隐藏为零。

`cell_constraint_error` 单独检查指定值。即使一个错误的全常量场具有零扩散、零工作缺陷，也不能在指定值不满足时通过。当前稳态自由方程残差与指定值误差是两项不同条件，调用者必须一起检查。指定值只是边界/模型数据，不能由参考内部场拟合。

SST 的 `step` 与 `steady_residual` 新增可选 `omega_constraints`，要求约束 omega 严格为正，并输出同样的反力账本。当前 `NativeHybridRans3D` 的 SUBOFF 运行器尚未生成或传入壁面函数约束，因此在运行的低 Re SUBOFF 物理方程没有变化。

## 已执行验证

| 验证 | 实际检查 |
|---|---|
| 非零内部约束、外边界扩散 | 线性与非线性 MMP 均验证原始非零收支由约束反力平衡，工作使用同一反力 |
| 储存、耗散和对流 | 常量场由 1 更新到指定 2，dt=0.25、sink=1；独立解析约束反力为每单元 6V，工作为 12V |
| 错误指定值 | 零自由方程/工作残差的错误常量场仍被约束误差拒绝 |
| 全部未知量指定 | 处理空自由系统，避免错误调用零维稀疏分解 |
| 独立求解交叉核对 | 将原矩阵与约束组成拉格朗日乘子系统，速度无关的标量消元解和反力分别与乘子解核对；包含非恒定边界、对流、储存、源与耗散 |
| SST 接口 | 指定 omega 精确满足、原反力显式保存；当前指定值偏离被检测；零/负 omega 被拒绝 |
| 回归 | 无约束标量、MMP、精度、SST、SIMPLE/SIMPLEC 与 RANS 续算测试通过 |

三组日志分别为 27 项、22 项、26 项通过，测试文件存在重叠，不能相加。最后一次添加的输出索引/指定值字段是诊断元数据，不改变已测试的数值路径。源码和测试快照及 SHA256 随报告保存。

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python -m pytest -q \
  tests/test_native_scalar_cell_constraints.py tests/test_native_hybrid_scalar3d.py \
  tests/test_native_scalar_mmp3d.py tests/test_native_scalar_mmp_newton3d.py \
  tests/test_native_scalar_precision3d.py tests/test_native_hybrid_sst3d.py \
  tests/test_native_hybrid_rans3d.py
```

## 尚未完成的壁面耦合

还需要同一壁面采样位置上的 Spalding 动量牵引、Foundation 12 对应的 omega 目标与近壁生成项、一单元多壁面的处理及完整动量/标量工作收支。已有单元约束不能代替这些环节。参见 [Spalding 导数和接入差异](../verification-wall-law-reference/linearization/README.md)。先完成当前原生 SST 稳态核验，再进入匹配 y+ 的高 Re SUBOFF 与网格收敛、Cp/Cd 实验对照。
