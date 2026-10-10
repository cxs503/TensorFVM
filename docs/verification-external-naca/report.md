# 圆柱与 NACA0012 贴体有限体积 benchmark

## 问题、算法与验收

圆柱采用 DFG 2D-1，Re=20，通道2.2×0.41 m、直径0.1 m、平均入口0.2 m/s、抛物线入口。参考来源：[FeatFlow](https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark1_re20.html)。翼型采用 NACA0012、Re=1000、攻角4°、单位弦长、20×16计算域、翼型前缘(6,8)。参考来源：[Di Ilio 2020](https://arxiv.org/html/2006.10487)，图10/11 present study。

原生贴体 SIMPLE、Rhie–Chow、一次迎风对流、非正交扩散，CPU使用SciPy SuperLU稀疏直接解，翼型最细网格用缓存LU预条件GMRES求解当前矩阵（缓存只作预条件器，不滞后离散方程），并启用既有Anderson混合加速；记录混合后真实方程残差，另查<10⁻⁷物理通量缺陷，CUDA使用Torch Krylov；两者只加速相同线性方程，并检查真实线性残差。三网格主报告另启用CPU全网格耦合Picard残差修正（仅圆柱，翼型采用原SIMPLE），保留真实周期连通、非正交扩散及物理Rhie–Chow，独立检查耦合矩阵响应与真实线性残差；最终记录所有修正后的实际方程残差与<10⁻⁷通量固定点缺陷。名为 c-grid 的翼型网格实际为周期径向拓扑。圆柱 Cd、Cl、前后压力差及翼型 Cd、Cl 分别严格小于3%，同时满足实际稳态残差。保存完整原始场、面通量、分项牵引、历史与源文件哈希。

## 参考值修正

旧翼型0.205/0.120图上估读不能支撑严格3%验收。本轮从矢量路径读出4°处坐标：Cl=27.41/190.95×1.4，Cd=23.87/190.95，各±0.0001图形读取区间。使用两端点最坏相对误差；此区间不包含作者数值方法误差。旧报告保留为历史记录，其通过结论撤回。4°没有匹配的公开Cp表，因此Cp曲线仅作计算结果展示。

## 软件使用

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python -m tensorfvm.verification.external_flow --output results/external --case cylinder --grids 64x24 128x48 256x96 --solver coupled
# 翼型: --case naca --grids 64x26 128x52 256x104
# --device cuda 为相同方程的 GPU 计算，单次耗时不是重复性能测量
```

## 计算结果

|案例|网格|设备|迭代/收敛|Cd|Cl|最大物理误差|全部通过|
|---|---|---|---|---|---|---|---|
|naca|64x26|cpu|608/True|0.1209792|0.20622549|3.2991%|False|
|naca|128x52|cpu|915/True|0.14878141|0.24299638|20.9758%|False|
|naca|256x104|cpu|546/True|0.1448016|0.2329089|15.9538%|False|

## 验证范围

独立审计重新积分压力及黏性牵引，检查壁面无滑移和质量通量，重算严格参考验收；未独立重跑全部离散方程。网格增加并不自动证明渐近收敛。未执行匹配 LBM 计算时不作性能或精度优劣排名。本轮为二维稳态层流；高Re尾涡、湍流及三维流动需要额外验证。

![naca 64x26 pressure](64x26/figures/pressure.png)

![naca 64x26 velocity](64x26/figures/velocity.png)

![naca 64x26 surface-pressure](64x26/figures/surface-pressure.png)

![naca 128x52 pressure](128x52/figures/pressure.png)

![naca 128x52 velocity](128x52/figures/velocity.png)

![naca 128x52 surface-pressure](128x52/figures/surface-pressure.png)

![naca 256x104 pressure](256x104/figures/pressure.png)

![naca 256x104 velocity](256x104/figures/velocity.png)

![naca 256x104 surface-pressure](256x104/figures/surface-pressure.png)

## 网格误差与残差补充

相邻网格变化和参考误差阶见 `refinement-audit.json`；这些阶基于当前参考，并非无参考的 Richardson 外推。三网格均有实际结果才可讨论趋势，单次耗时只用于复现预算。审计重新检查圆柱前后压力的周期角度插值、所有实际线性求解残差，以及真实压力/黏性力积分。

![网格误差](refinement.png)

![64x26 residuals](64x26/figures/residuals.png)

![128x52 residuals](128x52/figures/residuals.png)

![256x104 residuals](256x104/figures/residuals.png)

## 独立方程复核

`equations-audit.json` 从保存场独立用 NumPy 重建最小二乘梯度、完整层流动量、质量守恒和无伪时间的物理 Rhie–Chow 通量，另外重建壁面无滑移梯度与压力。只复用原几何生成及给定边界数据；审计没有调用原求解器的梯度/动量/通量计算。固定点通量相对缺陷要求 <10⁻⁶，独立方程状态见JSON，不能以内部残差取代此审查。

## 局部场与实际网格

可编辑论文源见 `report.tex`，完整PDF含局部速度分量、尾流和网格。

![64x26/figures/pressure](64x26/figures/pressure.png)

![64x26/figures/velocity](64x26/figures/velocity.png)

![64x26/figures/surface-pressure](64x26/figures/surface-pressure.png)

![64x26/figures/residuals](64x26/figures/residuals.png)

![64x26/figures/u-component](64x26/figures/u-component.png)

![64x26/figures/v-component](64x26/figures/v-component.png)

![64x26/figures/wake-detail](64x26/figures/wake-detail.png)

![64x26/figures/mesh-detail](64x26/figures/mesh-detail.png)

![128x52/figures/pressure](128x52/figures/pressure.png)

![128x52/figures/velocity](128x52/figures/velocity.png)

![128x52/figures/surface-pressure](128x52/figures/surface-pressure.png)

![128x52/figures/residuals](128x52/figures/residuals.png)

![128x52/figures/u-component](128x52/figures/u-component.png)

![128x52/figures/v-component](128x52/figures/v-component.png)

![128x52/figures/wake-detail](128x52/figures/wake-detail.png)

![128x52/figures/mesh-detail](128x52/figures/mesh-detail.png)

![256x104/figures/pressure](256x104/figures/pressure.png)

![256x104/figures/velocity](256x104/figures/velocity.png)

![256x104/figures/surface-pressure](256x104/figures/surface-pressure.png)

![256x104/figures/residuals](256x104/figures/residuals.png)

![256x104/figures/u-component](256x104/figures/u-component.png)

![256x104/figures/v-component](256x104/figures/v-component.png)

![256x104/figures/wake-detail](256x104/figures/wake-detail.png)

![256x104/figures/mesh-detail](256x104/figures/mesh-detail.png)

## 耗时口径

`elapsed_s` 从求解器构造之后开始，包含实际迭代及线性组装/求解，不包括网格准备、文件导出和作图。各工况单次运行不作统计性能资格或TensorLBM加速比。
