# Native SIMPLEC：实现与验证

后续进展：[原生混合 Stokes / SIMPLE / SIMPLEC 接入](../verification-native-hybrid-flow/README.md)。旧配点求解器的失败状态仍保留；以下为历史记录。

日期：2026-10-12。**算法选项通过小案例验证；SUBOFF 物理精度未通过。**

## 1. 修改范围

`NativeSteadySettings.pressure_velocity_algorithm` 支持 `simple` 和 `simplec`。SIMPLEC 供 `NativeSimpleSolver3D`、`NativeSimpleSSTRansSolver3D` 使用，默认仍保留 SIMPLE。SUBOFF 运行器用 `--algorithm simplec` 启用；该选项默认压力松弛为 1，用户可显式覆盖。

对于方程 `aP*u' + Σ(aN*uN') = -G*p'`，SIMPLE 采用 `1/aP_relaxed`，SIMPLEC 采用 `1/(aP_relaxed + ΣaN)`。求和只包含同一速度分量的非对角邻居项；不能把不同分量的速度修正当作同一个数。墙面张量耦合保留在动量求解中，压力增量响应仍是分量对角近似。

分母下限为 `0.1*aP_relaxed`，并在每条历史中输出 `simplec_limited_dofs`，不隐藏限制。该下限与 [OpenFOAM Foundation 12 官方压力修正实现](https://github.com/OpenFOAM/OpenFOAM-12/blob/master/applications/modules/incompressibleFluid/correctPressure.C)采用的保护原则一致。

速度更新和压力方程使用同一响应。压力矩阵来自实际面通量对压力增量的导数，继续包含偏斜插值、变量系数和壁面阻力影响。项目原有稳态 Rhie–Chow 方程保持不变，以便比较 SIMPLE/SIMPLEC 的同一离散稳态解。**这不是 OpenFOAM 完整 rAtU/HbyA/面插值实现的逐行移植，两者仍不能视作相同离散。**

## 2. 小案例验证

运行：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
python -m pytest tests/test_native_simple3d.py -q
```

结果：**18 passed**，原始输出见 [tests.txt](tests.txt)。新增/扩展检查包括：

- 偏斜三维网格上的仿射速度/压力解析解，含对流和无对流。
- 一次完整压力修正后，存储面通量和独立重建面通量同时守恒，含压力出口和压力参考点两种条件。
- SST + Spalding 通道的当前非线性方程收敛，含/不含面偏斜修正。
- 非仿射 Stokes 问题中，SIMPLE 和 SIMPLEC 收敛到一致的速度、压力和通量，绝对差容限 2e−8。

这些验证不证明复杂 SUBOFF 网格上的稳健性，也未建立普遍加速倍数。

## 3. 原 SUBOFF 网格首步诊断：未通过

同一旧 9,068 单元网格，Re=13,218,056.8096，SIMPLEC + SST + Spalding，动量松弛 0.3、压力松弛 1；一轮/一步。可复现命令：

```bash
PYTHONPATH=src python scripts/run_native_suboff_rans.py \
  --mesh docs/verification-suboff-single-domain-mesh/coarse-tail-transition/suboff-hybrid.msh \
  --output results/native-suboff-simplec-new-diagnostic \
  --reynolds 13218056.809580836 --algorithm simplec \
  --outer-iterations 1 --inner-iterations 1 --maximum-speed 5
```

| 诊断量 | 结果 |
|---|---:|
| 首步真实线性相对残差 | 5.29e−13 |
| 存储面通量散度最大值 | 3.89e−9 |
| 当前系数下重建通量散度最大值 | 7.09e3 |
| 最大速度/U∞ | 66.05 |
| SIMPLEC 分母限制自由度数 | 0 |

速度超过显式保护阈值 5，计算中止并保留未裁剪的异常状态用于诊断。线性解精确与存储通量守恒仍然不足以说明非线性解正确。首步改变了对流系数与稳态通量响应，当前重建一致性尚未满足；不能据此归因于单一因素。

配置、完整错误信息和动量历史见 `evidence/`。现有[黏性耗散性缺陷](../verification-native-small-cases/README.md)与[压力条件数诊断](../verification-native-hybrid-foundations/README.md)仍未由本次算法选项修复。继续保持数值收敛、空间收敛和物理通过均为 false。

## 4. 与网格工作的关系

第一层变薄和尾部加密先在独立 OpenFOAM SIMPLEC/SST 基线上验证，见 [OpenFOAM 基线](../verification-openfoam-suboff/README.md)。新 snappy 网格尚未用于上述原生求解器首步诊断，不能把两者的结果混作同网格比较。
