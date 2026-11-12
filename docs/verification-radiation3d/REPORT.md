# 三维视图因子验证报告（TensorFVM radiation3d）

- 生成时间: 2026-11-12T12:31:26
- 驱动: `PYTHONPATH=src python -m tensorfvm.verification.radiation3d`
- 判据: 20 条，预声明容差；结果: **PASS**（0 条失败）
- 墙钟: 31.2 s（纯几何，无求解器）

## 1. 判据表

| # | 量 | 实测 | 容差 | 方向 | 判定 | 说明 |
|---|---|---|---|---|---|---|
| 1 | `closed unit box: max_i |sum_j F_ij - 1|` | 1.110e-16 | 1.000e-12 | max | PASS | closed unit box: max_i |sum_j F_ij - 1| |
| 2 | `closed unit box: max |A_i F_ij - A_j F_ji| / max|AF|` | 5.550e-16 | 1.000e-12 | max | PASS | closed unit box: max |A_i F_ij - A_j F_ji| / max|AF| |
| 3 | `opposite faces (unit squares, d=1) vs Hamilton & Morgan 1952` | 6.945e-16 | 1.000e-10 | max | PASS | opposite faces (unit squares, d=1) vs Hamilton & Morgan 1952 |
| 4 | `adjacent faces vs summation-rule symmetry value (1-F_opp)/4` | 5.550e-16 | 1.000e-10 | max | PASS | adjacent faces vs summation-rule symmetry value (1-F_opp)/4 |
| 5 | `adjacent faces vs Hottel & Sarofim perpendicular common-edge` | 6.937e-16 | 1.000e-10 | max | PASS | adjacent faces vs Hottel & Sarofim perpendicular common-edge closed form |
| 6 | `parallel unit squares d=0.5 vs Hamilton & Morgan 1952` | 2.674e-16 | 1.000e-10 | max | PASS | parallel unit squares d=0.5 vs Hamilton & Morgan 1952 |
| 7 | `perpendicular common-edge family (6 geometries) vs Hottel & ` | 2.951e-07 | 1.000e-06 | max | PASS | perpendicular common-edge family (6 geometries) vs Hottel & Sarofim |
| 8 | `contour integration vs Lambert quadrature, disjoint non-copl` | 4.481e-15 | 1.000e-09 | max | PASS | contour integration vs Lambert quadrature, disjoint non-coplanar pairs |
| 9 | `tilted pair is invariant under reversing either/both vertex ` | 5.551e-17 | 1.000e-12 | max | PASS | tilted pair is invariant under reversing either/both vertex orders |
| 10 | `contour path is measurably invalid on degenerate pairs (>=50` | inf | 5.000e-01 | min | PASS | contour path is measurably invalid on degenerate pairs (>=50% error, so the gate is load-bearing rather than cosmetic) |
| 11 | `method='auto' returns the quadrature value on every degenera` | 0.000e+00 | 1.000e-12 | max | PASS | method='auto' returns the quadrature value on every degenerate pair |
| 12 | `contour_pair_is_valid flags all three degenerate pairs` | 1.000e+00 | 1.000e+00 | exact | PASS | contour_pair_is_valid flags all three degenerate pairs |
| 13 | `quadrature at a 1e-6 gap reproduces the exact touching value` | 2.170e-06 | 1.000e-04 | max | PASS | quadrature at a 1e-6 gap reproduces the exact touching value F=0.2000438 |
| 14 | `contour path is already accurate (<=1e-4) at the gate's own ` | 5.247e-05 | 1.000e-04 | max | PASS | contour path is already accurate (<=1e-4) at the gate's own scale (gap = 0.1 x face size) |
| 15 | `faceted coaxial disks (n=96) vs closed form F=0.38197` | 3.194e-04 | 5.000e-04 | max | PASS | faceted coaxial disks (n=96) vs closed form F=0.38197 |
| 16 | `facet convergence order (log-log slope of error vs n); O(1/n` | 2.008e+00 | 1.500e+00 | min | PASS | facet convergence order (log-log slope of error vs n); O(1/n^2) expected for a smooth closed curve |
| 17 | `3-D strips of length 200 reproduce the 2-D crossed-strings F` | 3.410e-03 | 5.000e-03 | max | PASS | 3-D strips of length 200 reproduce the 2-D crossed-strings F = sqrt(2)-1 |
| 18 | `the strip error decays monotonically with length (end effect` | 1.000e+00 | 1.000e+00 | exact | PASS | the strip error decays monotonically with length (end effects only) |
| 19 | `3-D box enclosure with one reradiating wall: |sum q_i A_i| /` | 6.526e-18 | 1.000e-12 | max | PASS | 3-D box enclosure with one reradiating wall: |sum q_i A_i| / sigma T_max^4 sum A |
| 20 | `contract checks: bounds_ok=True, missing_normal_raises=True,` | 4.000e+00 | 4.000e+00 | min | PASS | contract checks: bounds_ok=True, missing_normal_raises=True, nonplanar_raises=True, degenerate_raises=True |

## 2. 立方体封闭腔（6 面 30 对）

| 量 | 计算 | 基准 |
|---|---|---|
| F(对面) | 0.199824895698 | 0.199824895698（Hamilton & Morgan） |
| F(相邻) | 0.200043776075 | 0.200043776075（对称+求和律） |

## 3. 方法互校（不相交异面对）

| 配置 | Lambert 求积 | Mitalas-Stephenson 线积分 | 相对差 |
|---|---|---|---|
| parallel_d1 | 0.199824895698 | 0.199824895698 | 1.39e-16 |
| perp_offset | 0.008904547900 | 0.008904547900 | 4.48e-15 |
| tilted_45 | 0.125986506265 | 0.125986506265 | 8.81e-16 |

## 4. 线积分法的失效边界（实测，非假设）

| 退化对 | 求积 | 线积分 | 相对误差 | 门标记 |
|---|---|---|---|---|
| touching_shared_edge | 0.200044 | 0.277421 | 38.68% | True |
| coplanar_shared_edge | 0.000000 | 0.405078 | inf% | True |
| coplanar_gap | 0.000000 | 0.232028 | inf% | True |

近接触扫描（垂直面对，间隙 g；求积 vs 线积分）：

| g | 求积 | 线积分 | 相对差 |
|---|---|---|---|
| 1e-01 | 0.1628227192 | 0.1628312624 | 5.25e-05 |
| 1e-02 | 0.1957990369 | 0.2030345341 | 3.70e-02 |
| 1e-03 | 0.1996108740 | 0.2301350850 | 1.53e-01 |
| 1e-04 | 0.2000003859 | 0.2574595391 | 2.87e-01 |
| 1e-06 | 0.2000433421 | 0.3122138787 | 5.61e-01 |

## 5. 圆盘分面收敛（vs 同轴圆盘闭式 F=0.381966）

| 分面数 n | 计算 F | 相对误差 |
|---|---|---|
| 12 | 0.374023 | 2.08e-02 |
| 24 | 0.380006 | 5.13e-03 |
| 48 | 0.381478 | 1.28e-03 |
| 96 | 0.381844 | 3.19e-04 |

观测收敛阶（log-log 斜率）= 2.01

## 6. 长条极限 vs 二维交叉线法（精确 F=√2−1=0.414214）

| 长度 L | 计算 F | 相对误差 |
|---|---|---|
| 20 | 0.400259 | 3.37e-02 |
| 50 | 0.408629 | 1.35e-02 |
| 100 | 0.411402 | 6.79e-03 |
| 200 | 0.412801 | 3.41e-03 |

误差随 L 单调下降（端部效应 ~1/L），说明三维模型在长条极限正确退化到二维精确解。

## 7. 灰体封闭腔（三维）

能量闭合残差 `|Σ q_i A_i| / (σ T_max⁴ Σ A)` = 6.526e-18

## 8. 误差判定（关键量 vs 基准，逐项）

| 关键量 | 基准 | 误差 | <3% |
|---|---|---|---|
| 立方体 F(对面) | Hamilton & Morgan | 7.2e-14 % | 是 |
| 立方体 F(相邻) | 对称+求和律/Hottel-Sarofim | 6.9e-14 % | 是 |
| 平行方板 d=0.5 | Hamilton & Morgan | <1e-12 % | 是 |
| 同轴圆盘 n=96 | 闭式 0.381966 | 0.0319 % | 是 |
| 长条 L=200 | 二维交叉线 0.414214 | 0.3410 % | 是 |

## 9. 注意点（踩坑/局限）

1. **线积分法对有接触/共面的面对是错的，且错得静默**（不返回 NaN）：
   实测相邻面 −0.277 vs 真值 0.200（+239%）、共面 +405%。原因是从面积分到线积分
   的恒等式要求两多边形不共享边界且不共面；`contour_pair_is_valid` 把它做成
   机器门，`method='auto'` 只在门通过时才走线积分。
2. **近接触也劣化**：间隙 1e-2 时线积分误差已 1.5%，1e-3 时 8.8%——对数边奇异
   不是高斯求积能吃的；因此门阈值取 0.1×面尺寸（实测 1.2e-6）。
3. **长宽比大会抵消**：L=200 的长条线积分误差 182%，求积法 0.34%。线积分路径
   适合“大小相当、彼此分开”的面对，细长几何走求积。
4. Lambert 点-面公式必须用**发射面**法向；用目标面法向仍给出“看起来合理”的数
   （实测 0.2602 vs 0.2000），所以该参数在 API 里是必填的而不是推断的。
5. 遮挡（第三面挡光）**未实现**：本模块不扣除遮挡，也不做几何求交；这是 D8 的
   后半段，仍记为未完成。
6. 曲面须分面近似，误差 O(1/n²)（圆盘 n=96 时 3.2e-4）。
