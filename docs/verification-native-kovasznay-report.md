# Kovasznay 流解析基准验证报告

## 摘要

本报告对 TensorFVM 原生混合有限体积后端（`NativeHybridNavierStokes3D`）在 Kovasznay 稳态解析解上的精度进行系统验证。Kovasznay 流是无体力、非零对流的精确 Navier–Stokes 解，Re=40，单位域。验证采用对称挤出网格（n×n×1）加密序列，速度与压力误差以体积加权相对 L2 范数评估。最细 96×96×1 网格（9,216 单元）速度误差 0.0579%、固定规范压力误差 2.2918%，均通过 <3% 门槛。

## 1. 物理背景与数学模型

Kovasznay 流是二维稳态不可压 Navier–Stokes 方程的解析解（Kovasznay, 1948）：

```
λ = −2κ²/(Re + √(Re² + 4κ²)),  κ = 2π
u = 1 − exp(λx)cos(κy)
v = (λ/κ)exp(λx)sin(κy)
p = ½(1 − exp(2λx))
```

参数：Re=40，ρ=1，μ=1/40，单位域 [0,1]²。该解满足 div(u)=0 且 (u·∇)u + ∇p − μΔu = 0，无体力项。

## 2. 数值方法

- 求解器：`NativeHybridNavierStokes3D`（混合有限体积，面通量守恒）
- 网格：六面体对称挤出 n×n×1，z 方向对称面 w=0
- 边界：全部外边界给定解析速度面积均值
- 压力参考：首单元固定为解析值
- 线性求解：`coupled_linear_method='reused_lu'`
- 非线性：Picard 外层松弛 0.7，tolerance=1e−8，max 120 步

## 3. 计算结果

| 网格 | 单元数 | 速度误差 (%) | 压力误差 (%) | 通过 <3% |
|---|---:|---:|---:|---|
| 4×4×1 | 16 | 8.7723 | 43.7074 | 否 |
| 8×8×1 | 64 | 5.0348 | 23.6912 | 否 |
| 16×16×1 | 256 | 1.7469 | 9.8465 | 否 |
| 32×32×1 | 1,024 | 0.4928 | 5.2380 | 否 |
| 64×64×1 | 4,096 | 0.1287 | 3.1610 | 否 |
| 96×96×1 | 9,216 | 0.0579 | 2.2918 | **是** |

观察阶（64→96）：速度 1.97，压力 0.79。

## 4. 误差判定

- 最细网格速度误差 0.0579% < 3%：**通过**
- 最细网格压力误差 2.2918% < 3%：**通过**
- 64×64 网格压力误差 3.1610% > 3%：未通过（保留原始数据）

## 5. 局限性与适用范围

- 本验证为层流解析解，不包含湍流模型（SST/SA）验证
- 挤出方向未做系统加密，完整三维立方体序列尚未通过精度门槛
- 三维偏斜混合网格上的压力误差仍较高（见 abc-diagnostics），根因待查
- 结果不能外推到 SUBOFF 等复杂几何或高 Re 湍流

## 6. 复现

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
python scripts/verify_native_kovasznay.py --output results/kovasznay-replay --fine-n 96
```

证据目录：`docs/verification-native-kovasznay/run/`（cases.json, summary.json, 场数据）

## 参考文献

- Kovasznay, L.I.G. (1948). "Laminar flow behind a two-dimensional grid". Proc. Cambridge Philos. Soc. 44: 58–62.
- DeepXDE Kovasznay 示例：https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/Kovasznay.flow.html
