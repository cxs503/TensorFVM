# 原生混合后端的完整 SUBOFF 外流域启动验证

## 1. 本阶段结论

首次在本阶段修复后的原生单元/共享面速度、压力后端上，完成真实 SUBOFF 外流域的无体力 Re_L=10 定常层流计算。
它包含无滑移艇体、贴体六面体层、端部棱柱、外部四面体和远场；不是薄层标量制造解，也未调用 OpenFOAM。
四步全耦合 Newton 后，当前原 Navier–Stokes 方程、连续性、完整状态更新和力/能量收支通过既定门槛。

**此结果只证明实际几何上的数值求解链路已运行并收敛，不是实验工况 benchmark 达标。**
当前只有一档粗网格、没有独立同工况 Cp/Cd 参考对照、没有湍流模型或壁面函数。
所报阻力以 L² 为参考面积，不可与其他归一化定义或高 Re 实验阻力直接比较。

## 2. 几何、网格与方程

裸艇母线使用仓库已有的 TensorLBM 来源 SUBOFF 几何。源代码及生成证据见 [mesh](mesh/)。
长度 L=1，外流域 `[-1,3] × [-1,1] × [-1,1]`，U∞=1，ρ=1，ν=0.1，Re_L=10。

| 项目 | 当前设置 |
|---|---|
| 总单元/面 | 1331 / 3556 |
| 近壁六面体 | 160 |
| 艏艉封口棱柱 | 32 |
| 外部四面体 | 1139 |
| 周向分段 | 8 |
| 近壁层 | 2 层，首层 0.002L，增长率 1.2，总厚度 0.0044L |
| 外部特征尺寸 | 0.8L |
| 拼接 | 共形共享三角面，无 overset 插值 |

这是启动网格：多面体艇体体积为 0.00745717L³，对同母线轴对称积分 0.00845891L³ 偏差约 11.84%。
节点在母线上不代表面内几何准确；此网格明显不能用于 <3% 物理精度认证。

艇体速度为零；入口及侧面远场给均匀速度 `(1,0,0)`；出口给零总牵引，回流速度显式给 `(1,0,0)`。
体力为零。初值来自同一原生离散的 Stokes 解，没有填入解析内部场或参考压力。
动量采用二阶共享面上风重构和完整黏性应力，压力通过实际面散度的负转置耦合；原始单元/面未知量均保留。

## 3. 数值收敛和守恒

| 指标 | Re=10 Newton 最终值 |
|---|---:|
| 当前动量相对残差 | 1.377e-15 |
| 质量散度最大值（保存面通量复算） | 1.473e-14 |
| 单元积分动量缺陷最大值 | 1.350e-15 |
| 界面牵引不平衡最大值 | 1.375e-15 |
| 艇体总力与远场动量通量闭合误差 | 1.998e-15 |
| 能量收支误差 | 1.887e-15 |
| 最终速度/压力相对更新 | 2.103e-15 / 1.367e-14 |
| 轴向艇体力，ρU∞²L² 单位 | 0.396433299 |
| Cd,L² = 2 Fx/(ρU∞²L²) | 0.792866598 |
| 最大单元速度 / U∞ | 1.047342 |
| 总耗时（该机器 CPU） | 130.03 秒 |

原方程与变化门槛为 1e-8；全部所列积分力与能量收支也要求 <1e-8。
[原场独立审计](re10-newton/artifact-audit.json)从保存的面速度重新计算通量、散度、单元积分动量和整体力闭合，
逐一核对源文件、网格和原场哈希。[迭代历史](re10-newton/history.json)包括每轮实际壁面力，
[汇总](re10-newton/summary.json)明确保留 `physical_suboff_benchmark_passed=false`。

第一次启动沿用了初始化的默认 1e-9 散度门槛，Stokes 散度为 2.719e-9，因此在进入 NS 前被拒绝。
该失败场及实际未平衡 NS 收支完整保存于 [re10-initialization-rejected](re10-initialization-rejected/summary.json)。
随后将初始化门槛显式设为既定的 1e-8；后续四步 NS 将最终散度降到约 1e-14，没有放宽最终验收门槛。

## 4. 实际网格与流场

![实际外流域及贴体网格、速度与压力](re10-newton/mesh-and-flow.png)

图为 z/L=0.001 的真实单元剖切；流场按实际单元值着色，没有解析插值替代计算结果。
如此粗的外部网格仍有强烈单元分辨率痕迹，必须继续加密和核查几何/远场敏感性。

![两种壁面压力代理量与当前方程收敛](re10-newton/wall-proxies-convergence.png)

目前保存两种诊断量：相邻单元压力 `2p_owner/(ρU∞²)` 与负法向总牵引 `−2(t·n)/(ρU∞² A)`。
前者不是经验证的壁面压力重构，后者含黏性法向应力及数值稳定项；均不能直接当作已经验证的表面 Cp。
此网格两种代理量的面积加权绝对 RMS 差约为 0.4904，不能选择其中更好看的曲线报告达标。
下一阶段需通过独立曲面牵引/压力检验和网格加密确定壁面压力的提取方法。

## 5. 复现

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src
python scripts/prepare_suboff_hybrid_mesh.py --axial 12 --azimuthal 8 --layers 2 --first-height .002 --growth 1.2 --far-size .8 --output results/suboff-startup-mesh-new
python scripts/run_native_hybrid_suboff.py --mesh results/suboff-startup-mesh-new/suboff-hybrid.msh --reynolds 10 --output results/suboff-startup-re10-new
python scripts/report_native_hybrid_suboff.py results/suboff-startup-re10-new
```

所有输出目录必须为空。计算会冻结源码与输入网格；失败不得覆盖原证据或作为通过。
环境为 CPU/SciPy，GPU 未测试。
SIMPLEC 的同网格 Re=10 独立求解采用对角基准、动量/压力松弛 1。
其求解器相对残差门槛已通过，Cd 与 Newton 相对差为 1.296e-9，但单元积分动量缺陷 1.0524e-8、界面牵引缺陷 1.2907e-8，略高于既定 1e-8 门槛。
因此[整体验收明确失败](re10-simplec-ledger-rejected/summary.json)，没有将小误差四舍五入为通过；[独立对照](algorithm-comparison.json)与全部原场保存。
下一步需继续收紧迭代收敛，保留原积分收支门槛。
Re=100 Newton 启动仍在进行，目前出现明显回溯和停滞，未计为成功。
