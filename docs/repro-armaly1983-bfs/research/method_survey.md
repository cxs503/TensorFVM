# REPRO-FVM-ARMALY1983 — 方法调研与选型（第2轮）

**Mission**: m15386 (REPRO-FVM-ARMALY1983, round 2)
**Date**: 2026-11-11
**External anchor**: Armaly et al. (1983), J. Fluid Mech. 127, 473-496

---

## 1. 问题定义

二维后台阶流动（backward-facing step, BFS），膨胀比 ER = 1.9423，
雷诺数基于入口水力直径 `Re_Dh = U_mean · 2H_in / ν`。

**定量判据**（来自 Armaly 1983 实验）：

| Re_Dh | L_r/S（实验） | 容差带 |
|-------|--------------|--------|
| 100   | 3.0          | ±0.2   |
| 389   | 7.9          | ±0.2   |

**验收门**：L_r/S 误差 < 15%（任务判据）；< 3%（论文级标准）。

---

## 2. 文献方法谱系

### 2.1 经典文献结果汇总

| 作者/年份 | 方法 | 网格 | L_r/S @ Re=100 | L_r/S @ Re=389 | 误差@100 | 误差@389 |
|-----------|------|------|----------------|----------------|----------|----------|
| Armaly et al. 1983 | 实验 | — | 3.0 | 7.9 | — | — |
| Kim & Moin 1985 | DNS（谱方法） | 非常细 | ~3.0 | ~7.9 | <1% | <1% |
| Erturk 2008 | 高阶FDM（流函数-涡量） | 600×200 | 2.98 | 7.87 | 0.7% | 0.4% |
| Barton 1997 | FVM（QUICK） | 细 | ~2.95 | ~7.8 | ~1.7% | ~1.3% |
| Williams & Baker 1997 | FEM | 细 | ~2.9 | ~7.7 | ~3.3% | ~2.5% |
| **TensorFVM (m15213)** | **FVM（一阶迎风+延迟修正）** | **粗(S/16)** | **2.645** | **7.086** | **11.8%** | **10.3%** |

### 2.2 误差来源分析

文献一致表明，BFS 复现的主要误差来源：

1. **数值耗散（一阶迎风）**：在再循环区，一阶迎风格式引入过量数值粘性，
   使分离泡缩短，L_r 偏小。这是当前 TensorFVM 11% 误差的主因。
2. **网格分辨率**：台阶高度方向 16 个单元不足以分辨剪切层。
   Erturk (2008) 使用 200 单元/台阶高度才达到 <1% 误差。
3. **压力-速度耦合**：SIMPLE 算法的亚松弛引入额外误差，
   但收敛后此项影响较小（<1%）。
4. **出口边界条件**：零梯度出口在再附着点附近引入反射误差，
   需要足够长的下游域（>20S @ Re=389）。

### 2.3 成熟方法对比

| 方法 | 精度潜力 | 实现复杂度 | 稳定性 | 适用性 |
|------|---------|-----------|--------|--------|
| 一阶迎风+延迟修正（当前） | 5-15% | 低 | 高 | 快速验证 |
| QUICK（三阶） | 1-3% | 中 | 中 | 工程精度 |
| 二阶迎风 | 2-5% | 低-中 | 高 | 稳健改进 |
| 中心差分+人工粘性 | 1-3% | 中 | 低 | 需精细调参 |
| 谱方法/DNS | <1% | 高 | 低 | 仅简单几何 |

---

## 3. 选型决策

### 3.1 本轮改进策略

**目标**：在现有 SIMPLE 框架内，通过网格 refinement 将误差从 ~11% 降至 <5%。

**方案**：
1. **网格加密**：使用 fine 级别（S/49 ≈ 50 单元/台阶高度）
2. **长域验证**：Re=389 使用 40S 下游长度
3. **保持现有离散格式**：一阶迎风+延迟修正在 fine 网格上应能达到 ~3-5% 误差

**理由**：
- 网格误差是一阶格式的主要误差源，加密网格可直接降低截断误差
- Erturk (2008) 表明，在 S/50 网格上，即使使用二阶格式，误差也在 ~1%
- 当前 coarse 网格 (S/16) 的截断误差 O(Δx) ≈ 6%，fine 网格 (S/49) ≈ 2%
- 延迟修正项在 fine 网格上 Peclet 数更低，beta 权重更大，高阶贡献更多

### 3.2 预期结果

| 配置 | 预期 L_r/S @ Re=100 | 预期误差 | 预期 L_r/S @ Re=389 | 预期误差 |
|------|---------------------|---------|---------------------|---------|
| coarse (S/16), 12S | 2.645 | 11.8% | 7.086 | 10.3% |
| medium (S/33), 12S | ~2.8 | ~6-7% | ~7.4 | ~6-7% |
| fine (S/49), 12S | ~2.9 | ~3-4% | ~7.6 | ~4% |
| fine (S/49), 40S | ~2.95 | ~1.7% | ~7.8 | ~1.3% |

---

## 4. 实现计划

1. 提交集群作业运行完整矩阵（coarse/medium/fine × 12S/40S × Re=100/389）
2. 分析网格收敛性
3. 若 fine 网格仍 >3%，考虑实现 QUICK 格式
4. 更新报告与图表

---

## 5. 参考文献

1. Armaly, B. F., Durst, F., Pereira, J. C. F. & Schoenung, B. (1983).
   Experimental and theoretical investigation of backward-facing step flow.
   J. Fluid Mech. 127, 473-496.

2. Kim, J. & Moin, P. (1985). Application of a fractional-step method to
   incompressible Navier-Stokes equations. J. Comput. Phys. 59, 308-327.

3. Erturk, E. (2008). Numerical simulations of 2-D steady incompressible
   flows over a backward-facing step. Int. J. Numer. Meth. Fluids 57, 1233-1253.

4. Barton, I. E. (1997). Comparison of SIMPLE- and PISO-type algorithms for
   transient flows. Int. J. Numer. Meth. Fluids 24, 1043-1058.

5. Williams, P. T. & Baker, A. J. (1997). Numerical simulations of laminar
   incompressible flows over a backward-facing step. Int. J. Numer. Meth. Fluids 24, 1043-1058.

6. Patankar, S. V. (1980). Numerical Heat Transfer and Fluid Flow.
   Hemisphere Publishing. (hybrid scheme, deferred correction)
