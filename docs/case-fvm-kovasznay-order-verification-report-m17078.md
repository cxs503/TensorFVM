# CASE-FVM-KOVASZNAY-ORDER 收敛阶验证报告（第2轮·收口）

**任务**: m17078 · TensorFVM
**锚点**: CASE-FVM-KOVASZNAY-ORDER
**日期**: 2026-11-13
**状态**: ✅ PASS — 速度 L2 收敛阶 1.988 ≥ 1.8（名义二阶）

---

## 一、案例介绍

Kovasznay 流是二维稳态不可压 Navier–Stokes 方程的经典解析解（Kovasznay, 1948, Proc. Camb. Phil. Soc. 44:58, doi:10.1017/S0305004100023999）。
该解满足 div(u)=0 且 (u·∇)u + ∇p − μΔu = 0（无体力项），是验证有限体积求解器精度和收敛阶的理想制造解基准。

**参数表**:

| 参数 | 值 |
|---|---|
| Re | 40 |
| ρ | 1 |
| μ = 1/Re | 0.025 |
| 域 | [0,1]²（对称挤出 n×n×1）|
| λ | −0.9637405441957689 |
| 压力参考 | 首单元固定为解析值 |
| 非线性 | Picard 外层松弛 0.7，tol=1e−8 |
| 线性 | reused_lu（首次 LU，后续 GMRES 预条件）|
| 对流格式 | 2阶 hybrid |
| 对称面闭合 | zero_gradient（强对称条件）|

**验收判据**（feature ledger 原文）:
> L2 收敛阶 >= 1.8 (名义二阶) | 指标: log-log 斜率 on 32/64/128 三套网格

---

## 二、方法

### 2.1 解析解

Kovasznay 解的形式为：
```
u = 1 − exp(λx)·cos(2πy)
v = (λ/2π)·exp(λx)·sin(2πy)
w = 0
p = ½(1 − exp(2λx))
```
其中 λ = −2(2π)²/(Re + √(Re² + 4(2π)²)) = −0.9637405441957689，满足 λ² − Re·λ = 4π²。

### 2.2 求解器配置

- **求解器**: `NativeHybridNavierStokes3D`（混合有限体积，共位网格，Rhie-Chow 插值）
- **算法**: coupled（耦合求解速度-压力系统）
- **对流离散**: 2阶 hybrid 格式（中心差分 + 一阶迎风混合）
- **线性求解**: reused_lu（首次完整 LU 分解，后续 GMRES 以 LU 为预条件）
- **对称面处理**: `zero_gradient_faces` — 对称面上切向分量与所属单元值绑定（强对称条件 ∂u_t/∂n = 0）

### 2.3 误差度量

体积加权相对 L2 误差：
```
L2(u) = √[ Σ V_c |u_c − u_exact,c|² / Σ V_c |u_exact,c|² ]
```
压力误差采用均值归零（mean-free）以消除压力规范自由度。

### 2.4 工具链

- `src/tensorfvm/verification/native_hybrid_ns.py`: `kovasznay_solver()`
- `scripts/run_kovasznay_convergence_study.py`: 完整收敛研究驱动脚本
- 每个网格层独立求解，结果写入 JSON（含 sha256 溯源）

---

## 三、计算结果

### 3.1 逐层误差表

| n | 单元数 | 速度 L2 (全) | 速度 L2 (in-plane) | 压力 L2 (mean-free) | max|w| | 外层迭代 | 收敛 |
|---|---|---|---|---|---|---|---|
| 32 | 1,024 | 3.436e-03 | 3.436e-03 | 1.140e-02 | 1.60e-16 | 22 | ✅ |
| 64 | 4,096 | 8.973e-04 | 8.973e-04 | 3.145e-03 | 2.49e-17 | 21 | ✅ |
| 128 | 16,384 | 2.262e-04 | 2.262e-04 | 8.444e-04 | 7.11e-17 | 21 | ✅ |

### 3.2 收敛阶

**两两观测阶**:

| 网格对 | 速度 L2 (全) | 速度 L2 (in-plane) | 压力 L2 |
|---|---|---|---|
| 32→64 | 1.937 | 1.937 | 1.857 |
| 64→128 | **1.988** | **1.988** | 1.897 |

**最小二乘拟合斜率**（全三组）:

| 量 | 拟合斜率 | 最大对数残差 |
|---|---|---|
| 速度 L2 (全) | 1.962 | 1.17e-02 |
| 速度 L2 (in-plane) | 1.962 | 1.17e-02 |
| 压力 L2 (mean-free) | 1.877 | 9.22e-03 |

### 3.3 收敛阶图

![convergence_loglog](convergence_loglog.png)

*图: Kovasznay Re=40 收敛阶 log-log 图。速度 in-plane L2 误差（蓝色方块）与压力 mean-free L2 误差（绿色三角）均呈现接近二阶收敛。参考线为一阶（黑色虚线）和二阶（黑色点划线）。*

---

## 四、验收判定

| 判据 | 阈值 | 测量值 | 状态 |
|---|---|---|---|
| 最细网格对速度 L2 阶 (64→128) | ≥ 1.8 | **1.988** | ✅ PASS |
| 最小二乘速度 L2 斜率 | ≥ 1.8 | **1.962** | ✅ PASS |
| 压力 L2 阶 (64→128) | ≥ 1.8 | 1.897 | ✅ PASS |
| 所有层数值收敛 | 是 | 是 | ✅ PASS |
| spanwise w 伪影 | ≈ 0 | max|w| < 1e-15 | ✅ PASS |

**最终 verdict: ✅ PASS**

---

## 五、注意点

1. **对称面闭合**: 使用 `zero_gradient` 闭合（强对称条件）消除了早期版本中因自由切向自由度导致的 O(h) spanwise 伪影。max|w| 在所有层均 < 1e-15（机器精度）。
2. **spanwise 阶计算**: 由于 spanwise 误差在机器精度水平，其观测阶（2.60 → -0.22）无物理意义，仅为数值噪声。
3. **计算时间**: n=32 约 41s，n=64 约 236s，n=128 约 1128s（本机 load ≈ 60-90 环境下）。
4. **压力收敛**: 压力 L2 阶（1.877）略低于速度（1.962），但仍满足 ≥ 1.8 阈值。
5. **历史对比**: 前一轮（m16904）使用 `free` 闭合时速度全 L2 阶仅 1.684（FAIL），本轮 `zero_gradient` 闭合将阶提升至 1.988（PASS）。

---

## 六、误差判定

| 关键量 | 基准 | 测量值 (n=128) | 相对误差 | < 3%? |
|---|---|---|---|---|
| 速度 L2 (in-plane) | 解析解 | 2.262e-04 | 0.023% | ✅ |
| 压力 L2 (mean-free) | 解析解 | 8.444e-04 | 0.084% | ✅ |
| 收敛阶 (64→128) | ≥ 1.8 | 1.988 | — | ✅ |
| 收敛阶 (LSQ) | ≥ 1.8 | 1.962 | — | ✅ |
| max|w| | 0 | 7.11e-17 | ✅ |

**结论**: 所有关键量均满足验收标准。TensorFVM 的混合有限体积不可压 NS 求解器在 Kovasznay 解析解上展现出名义二阶收敛。

---

## 交付物

| 文件 | 说明 |
|---|---|
| `runs/kovasznay-order-m17078/summary.json` | 完整结果汇总（含 sha256 溯源）|
| `runs/kovasznay-order-m17078/level_n32.json` | n=32 层结果 |
| `runs/kovasznay-order-m17078/level_n64.json` | n=64 层结果 |
| `runs/kovasznay-order-m17078/level_n128.json` | n=128 层结果 |
| `runs/kovasznay-order-m17078/convergence_loglog.png` | 收敛阶 log-log 图 |
| `scripts/run_kovasznay_convergence_study.py` | 收敛研究驱动脚本 |
| `docs/case-fvm-kovasznay-order-verification-report-m17078.md` | 本报告 |

---

## sha256 溯源

| 工件 | sha256 |
|---|---|
| summary.json | 7952d03344e0e156570046f32bbf705335e891d450b0678db2f0a15f135c7783 |
| level_n32.json | d582e3a028ef3456a1ddc0fe0c19398873b390a03b8cfad2f61a2b9ecef9f155 |
| level_n64.json | 871b52073c9fb908243ddee97b3109257bca38515c4f910976e7accbd03c6943 |
| level_n128.json | e21aa0ad64c199e0d0553c4659ef664a7988e18960a65366e73e442138f64dd7 |

---

## 参考文献

Kovasznay, L. I. G. (1948). "Laminar flow behind a two-dimensional grid". *Proc. Cambridge Philos. Soc.* 44:58-62. doi:10.1017/S0305004100023999
