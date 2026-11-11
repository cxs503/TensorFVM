# P400W2-N01 — FVM 两相流 VOF 验证面：溃坝波前 + Zalesak 旋转圆盘 + 质量守恒门

TensorFVM · 战役节点 P400W2-N01 · 自动渲染自 `docs/P400W2-N01-two-phase-vof-verification/artifacts/twophase_verification.json`（生成于 2026-11-11T02:50:19Z；本文件所有数字均由脚本注入，零手抄）。

## 0. 判据总览（机器裁决）

| 判据 | 结论 | 内容 |
|---|---|---|
| **C1** | PASS | max relative front error vs Ritter at h_ref over t in [0.05, 0.3] s <= 10%, both resolutions (scheme under test: cicsam); every scheme of the leg is listed with its own verdict |
| **C2** | PASS | one revolution at N=256: at least one of the node's algebraic schemes (cicsam / hric) keeps the enclosed-area error vs the initial (exact discrete) contour <= 1% and the perimeter error vs the analytic slotted-disk perimeter <= 3%; every per-scheme verdict is listed |
| **C3** | PASS | |M(t)-M(0)|/M(0) < 0.1% over every leg |
| **C4** | PASS | both resolutions inside the 10% C1 band and the finer grid not worse than the coarser one by more than the 10% band (trend consistency, not a second accuracy claim; scheme under test: cicsam) |

C1 逐格式裁决（溃坝腿，判据 10 %）：cicsam → PASS、plic → FAIL、upwind → PASS（判据格式：cicsam）。

C4 逐格式裁决（溃坝腿，判据 10 %）：cicsam → PASS、plic → FAIL、upwind → PASS（判据格式：cicsam）。

C2 逐格式裁决（N=256）：upwind → FAIL、cicsam → FAIL、hric → PASS；达到判据的格式：hric。（判据写在能力层，逐格式全列，不隐藏未达标项。）

C5（≥10 条 pytest 全绿）：`tests/test_twophase_bench_vof.py` 共 15 个用例，满足 ≥10 的要求；运行结果见 §7。

**总判定：全部通过**

## 1. 案例介绍

### 1.1 溃坝（dam break）Ritter 标定算例

| 量 | 值 |
|---|---:|
| 坝址位置 $W$ | 0.028 m |
| 水库水深 $H$（合同值 0.056 m） | 0.056 m |
| $H/W$ | 2 |
| 重力 $g$ | 9.81 m/s² |
| 浅水波速 $c_0=\sqrt{gH}$ | 0.741188 m/s |
| 计算域 $x$ | [-0.26, 0.6] m |
| 水槽高度 | 0.07 m |
| 上游水库长度 | 0.26 m |
| $t=0.3$ s 时波前位置 | 0.47271 m |
| $t=0.3$ s 时稀疏波头位置 | -0.19436 m |

**为什么计算域要带上游水库。** Ritter（1892）解是 *半无限水库* 在干床上的精确解。对宽 0.028 m 的有限水柱，壁面反射的稀疏波在 $t=W/c_0=0.03778$ s 就到达坝面，之后波前减速：Martin & Moyce（1952）实测 $t=0.3$ s 时波前约 $0.19$ m，而 $2t\sqrt{gH}=0.445$ m，相差 2.3 倍。因此“溃坝波前在 $t\in[0.05,0.3]$ s 内与 $2t\sqrt{gH}$ 误差 ≤10 %”**对任何有限水柱都不可能成立**（本仓 `docs/vof-phase0/artifacts/phase0_report.json` 的耦合二维坍塌实测亦是 −60 %/−31 %/−12 % @ $t^*=1,2,3$）。本节点因此令计算域包含 0.26 m 上游水库，使解析解在整个计算域、整个时间窗内成立；$H/W=2$ 与坝址 $W$ 保持不变。

**解析解推导（Ritter 1892，浅水方程自相似解）。** 干床上的浅水方程为
$h_t+(hu)_x=0$、$u_t+uu_x+g h_x=0$。取自相似变量 $\eta=\xi/(c_0t)$（$\xi=x-W$，
$c_0=\sqrt{gH}$），设 $h=H\,f(\eta)$、$u=c_0\,u(\eta)$，两式化为
$(u-\eta)f'+fu'=0$ 与 $(u-\eta)u'+f'=0$，非平凡解要求 $(u-\eta)^2=f$。取干床边界条件
$f(2)=0$、$u(2)=2$（波前以 $2c_0$ 推进）与 $f(-1)=1$、$u(-1)=0$（静水区），
解为 $u=\tfrac{2}{3}(1+\eta)$、$f=\tfrac{1}{9}(2-\eta)^2$，换回物理量即

$$u=\tfrac{2}{3}(c_0+\xi/t),\qquad h=\frac{1}{9g}\left(2c_0-\frac{\xi}{t}\right)^2 .$$

**波前定位的解析反演。** 令 $h=h_{\rm ref}$ 反解得
$x_{\rm ref}(t)=W+t\,(2c_0-3\sqrt{g\,h_{\rm ref}})$，即谐调深度下的解析波前；这就是本节点
与实测波前比较的量（而 $2c_0t$ 是 $h\to0$ 的尖端）。

**为什么二维速度场要补 $v$。** 深度平均解只给 $u(x,t)$。二维不可压连续性
$\partial_x u+\partial_y v=0$ 给出 $v=-y\,\partial_x u$；它与自由面的运动学条件
$h_t+u\,h_x=-h\,\partial_x u$（由一维连续性得到）等价，因此以该场输运的 $\alpha$
其等值面正好是解析自由面。


### 1.2 Zalesak 旋转圆盘

| 量 | 值 |
|---|---:|
| 几何来源 | `tensorfvm.verification.vof_rotation.zalesak_disk`（导入，不重写） |
| 圆盘半径 | 0.15（无量纲域 1，圆心 (0.5, 0.75)） |
| 缝半宽 | 0.05（整条竖缝，把圆盘切成两瓣） |
| 解析面积 | 0.041251 |
| 解析周长 | 1.30426 |
| 旋转 | 绕域心刚体旋转，角速度 $2\pi$，一整圈，CFL=0.4 |
| 网格 | 64、128、256 单元/方向 |

## 2. 怎么做的

**新增能力（本节点之前仓内不存在）。** 面值重构的 *归一化变量*（NVD）族在仓中只有一句
"donor-cell (CICSAM-like)" 注释：既无 Hyper-C，也无 CBC 限定的 ULTIMATE-QUICKEST 分支。
本节点实现于 `src/tensorfvm/twophase_bench/cicsam.py`：

* 归一化变量 $\tilde\alpha=(\alpha-\alpha_U)/(\alpha_A-\alpha_U)$，面值
  $\alpha_f=\alpha_U+\tilde\alpha_f(\alpha_A-\alpha_U)$；
* Hyper-C 支：$\tilde\alpha^{HC}=\min(1,\tilde\alpha_D/C_f)$；
* CBC 限定的 ULTIMATE-QUICKEST 支：
  $\tilde\alpha^{UQ}=\min\{[8C_f\tilde\alpha_D+(1-C_f)(6\tilde\alpha_D+3)]/8,\ \tilde\alpha_D/C_f\}$；
* CICSAM 混合（Ubbink & Issa 1999）：$\gamma_f=\min(1,k_\gamma\cos^2\theta_f)$，$\theta_f$ 为界面法向
  （3×3 Youngs 梯度，复用 `vof_plic`）与面法向夹角；
* HRIC 型 Courant 混合（Muzaferija & Perić 1997 的意图实现，非逐位复刻）：$\gamma_C=(1-C_f)^2$，
  模块 docstring 明示其为性质级实现；
* 诊断支 `uq`（纯 UQ）用于平滑场精度标定；`upwind` 作为一阶基线。

**为什么需要 FCT 限幅器。** NVD 面值本身不保证二维离散极值原理。同一溃坝短程算例开/关限幅的对照
（`limiter_audit`，见 §3.5）给出实测：不限幅时 $\alpha_{\max}$ = 1.01、超填单元·步
93.0；限幅后 $\alpha_{\max}$ = 1、超填 0.0，且质量漂移仍为
2.2e-16。限幅器为 Zalesak（1979）反扩散通量限幅（`fct_limit`，逐面缩放、通量仍单值共享）。

**两种重构并存。** 几何 PLIC（`tensorfvm.vof_plic.advect_plic`，已有并已注册）与本节点的
代数 CICSAM 在同一算例上对跑，避免"自证"。

**判据口径（三条面积度量，全部披露）。** VOF 语境下"面积误差"有三种互不等价的定义，本报告全列：

1. `area_error_relative` = $|A(T)-A(0)|/A(0)$，$A=\sum\alpha\Delta x^2$：通量形式输送对此恒为机器零
   （$\sim10^{-15}$），因此它**不是**形状判据；
2. `area_error_contour_vs_initial`：$\alpha=1/2$ 等值线围成的面积（marching squares + 鞋带公式，亚网格精度）
   相对初值的变化。刚体旋转整圈后**精确解就是初值**，所以这是分辨率自洽的形状判据，C2 采用它；
3. `symmetric_area_error`：$|\{\alpha>1/2\}\ \mathrm{XOR}\ \{\alpha_{\rm exact}>1/2\}|$，
   阈值化后最严苛（把一个亚网格偏差记成整格翻转），仅作诊断。

周长同理：`perimeter_error_analytic` 对解析周长（光滑界面下该度量自身误差 0.3 %，见
`test_contour_metrics_reproduce_analytic_disk`），C2 采用它；`perimeter_error_initial` 对初值
二值场的阶梯多边形（阶梯本身偏长 3 %，是度量偏差）仅作诊断。

## 3. 计算结果

### 3.1 Zalesak：一圈旋转后的形状保真与守恒

| 格式 | N | 步数 | 面积误差（等值线 vs 初值） | 周长误差（vs 解析） | 对称差面积（诊断） | 峰值保持 | 相对 L1 | 质量漂移 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cicsam | 64 | 495 | 1.11 % | 9.42 % | 11.05 % | 1 | 0.158 | 5.4e-14 |
| cicsam | 128 | 997 | 0.21 % | 3.91 % | 4.38 % | 1 | 0.0653 | 4.8e-16 |
| cicsam | 256 | 2003 | 0.02 % | 3.20 % | 2.06 % | 1 | 0.0328 | 1.2e-15 |
| hric | 64 | 495 | 0.83 % | 9.12 % | 13.37 % | 1 | 0.194 | 2.6e-14 |
| hric | 128 | 997 | 0.07 % | 2.85 % | 8.76 % | 1 | 0.104 | 6.4e-16 |
| hric | 256 | 2003 | 0.00 % | 1.41 % | 4.11 % | 1 | 0.0574 | 2e-15 |
| plic | 64 | 495 | 0.86 % | 8.66 % | 9.30 % | 1 | 0.145 | 8.8e-15 |
| plic | 128 | 997 | 0.05 % | 3.28 % | 3.95 % | 1 | 0.0606 | 6.4e-16 |
| upwind | 64 | 495 | 100.00 % | 100.00 % | 100.00 % | 0.332 | 1.51 | 0 |
| upwind | 128 | 997 | 100.00 % | 100.00 % | 100.00 % | 0.443 | 1.33 | 0 |
| upwind | 256 | 2003 | 89.62 % | 74.29 % | 89.46 % | 0.521 | 1.19 | 0 |

### 3.2 溃坝：波前位置 vs Ritter

波前用**湿深阈值**定义（$h_{\rm ref}=H/20$，与分辨率无关），解析对照取该阈值在 Ritter 剖面上的精确位置 $x_{\rm ref}=W+t(2c_0-3\sqrt{g h_{\rm ref}})$。

**cicsam，$\Delta x=H/28=2.000$ mm（430×35，524 步，CFL=0.4）**

| t [s] | $x_{front}$ 实测 [m] | $x_{ref}$ 解析 [m] | 误差（匹配深度） | $x$ 实测（$h=dy/2$） | $x$ 解析（尖端） | 误差（对尖端） | 水深 L1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.050 | 0.07793 | 0.07765 | 0.36 % | 0.09036 | 0.10271 | 12.02 % | 0.12 % |
| 0.100 | 0.12812 | 0.12657 | 1.23 % | 0.15574 | 0.17631 | 11.66 % | 0.15 % |
| 0.150 | 0.17705 | 0.17601 | 0.59 % | 0.20599 | 0.25071 | 17.84 % | 0.14 % |
| 0.200 | 0.22525 | 0.22546 | 0.09 % | 0.26515 | 0.32511 | 18.44 % | 0.14 % |
| 0.250 | 0.27521 | 0.27437 | 0.31 % | 0.32478 | 0.39871 | 18.54 % | 0.14 % |
| 0.300 | 0.32451 | 0.32355 | 0.30 % | 0.38509 | 0.47271 | 18.54 % | 0.14 % |

**cicsam，$\Delta x=H/56=1.000$ mm（860×70，1048 步，CFL=0.4）**

| t [s] | $x_{front}$ 实测 [m] | $x_{ref}$ 解析 [m] | 误差（匹配深度） | $x$ 实测（$h=dy/2$） | $x$ 解析（尖端） | 误差（对尖端） | 水深 L1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.050 | 0.07811 | 0.07752 | 0.77 % | 0.09435 | 0.10251 | 7.96 % | 0.06 % |
| 0.100 | 0.12676 | 0.12670 | 0.05 % | 0.15605 | 0.17651 | 11.59 % | 0.07 % |
| 0.150 | 0.17640 | 0.17588 | 0.30 % | 0.21859 | 0.25051 | 12.74 % | 0.06 % |
| 0.200 | 0.22540 | 0.22505 | 0.15 % | 0.28256 | 0.32451 | 12.93 % | 0.06 % |
| 0.250 | 0.27487 | 0.27450 | 0.13 % | 0.34720 | 0.39891 | 12.96 % | 0.06 % |
| 0.300 | 0.32381 | 0.32355 | 0.08 % | 0.41130 | 0.47271 | 12.99 % | 0.06 % |

**plic，$\Delta x=H/28=2.000$ mm（430×35，524 步，CFL=0.4）**

| t [s] | $x_{front}$ 实测 [m] | $x_{ref}$ 解析 [m] | 误差（匹配深度） | $x$ 实测（$h=dy/2$） | $x$ 解析（尖端） | 误差（对尖端） | 水深 L1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.050 | 0.08222 | 0.07765 | 5.89 % | 0.09227 | 0.10271 | 10.17 % | 0.46 % |
| 0.100 | 0.13989 | 0.12657 | 10.53 % | 0.15936 | 0.17631 | 9.61 % | 1.17 % |
| 0.150 | 0.19946 | 0.17601 | 13.32 % | 0.22807 | 0.25071 | 9.03 % | 1.82 % |
| 0.200 | 0.25284 | 0.22546 | 12.15 % | 0.29765 | 0.32511 | 8.45 % | 2.66 % |
| 0.250 | 0.31483 | 0.27437 | 14.75 % | 0.36633 | 0.39871 | 8.12 % | 3.45 % |
| 0.300 | 0.37800 | 0.32355 | 16.83 % | 0.43608 | 0.47271 | 7.75 % | 4.23 % |

**plic，$\Delta x=H/56=1.000$ mm（860×70，1048 步，CFL=0.4）**

| t [s] | $x_{front}$ 实测 [m] | $x_{ref}$ 解析 [m] | 误差（匹配深度） | $x$ 实测（$h=dy/2$） | $x$ 解析（尖端） | 误差（对尖端） | 水深 L1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.050 | 0.08078 | 0.07752 | 4.22 % | 0.09590 | 0.10251 | 6.45 % | 0.40 % |
| 0.100 | 0.13487 | 0.12670 | 6.45 % | 0.16564 | 0.17651 | 6.16 % | 1.00 % |
| 0.150 | 0.18903 | 0.17588 | 7.48 % | 0.23620 | 0.25051 | 5.71 % | 1.65 % |
| 0.200 | 0.24383 | 0.22505 | 8.34 % | 0.30725 | 0.32451 | 5.32 % | 2.33 % |
| 0.250 | 0.29910 | 0.27450 | 8.96 % | 0.37900 | 0.39891 | 4.99 % | 2.97 % |
| 0.300 | 0.35124 | 0.32355 | 8.56 % | 0.45017 | 0.47271 | 4.77 % | 3.65 % |

**upwind，$\Delta x=H/28=2.000$ mm（430×35，524 步，CFL=0.4）**

| t [s] | $x_{front}$ 实测 [m] | $x_{ref}$ 解析 [m] | 误差（匹配深度） | $x$ 实测（$h=dy/2$） | $x$ 解析（尖端） | 误差（对尖端） | 水深 L1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.050 | 0.07937 | 0.07765 | 2.21 % | 0.09133 | 0.10271 | 11.08 % | 0.20 % |
| 0.100 | 0.12956 | 0.12657 | 2.37 % | 0.15333 | 0.17631 | 13.03 % | 0.34 % |
| 0.150 | 0.17964 | 0.17601 | 2.06 % | 0.21487 | 0.25071 | 14.30 % | 0.41 % |
| 0.200 | 0.22940 | 0.22546 | 1.75 % | 0.27587 | 0.32511 | 15.14 % | 0.46 % |
| 0.250 | 0.27844 | 0.27437 | 1.48 % | 0.33589 | 0.39871 | 15.76 % | 0.50 % |
| 0.300 | 0.32762 | 0.32355 | 1.26 % | 0.39601 | 0.47271 | 16.23 % | 0.53 % |

**upwind，$\Delta x=H/56=1.000$ mm（860×70，1048 步，CFL=0.4）**

| t [s] | $x_{front}$ 实测 [m] | $x_{ref}$ 解析 [m] | 误差（匹配深度） | $x$ 实测（$h=dy/2$） | $x$ 解析（尖端） | 误差（对尖端） | 水深 L1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.050 | 0.07834 | 0.07752 | 1.06 % | 0.09455 | 0.10251 | 7.77 % | 0.10 % |
| 0.100 | 0.12815 | 0.12670 | 1.15 % | 0.16045 | 0.17651 | 9.09 % | 0.17 % |
| 0.150 | 0.17764 | 0.17588 | 1.00 % | 0.22558 | 0.25051 | 9.95 % | 0.21 % |
| 0.200 | 0.22698 | 0.22505 | 0.86 % | 0.29033 | 0.32451 | 10.53 % | 0.24 % |
| 0.250 | 0.27650 | 0.27450 | 0.73 % | 0.35520 | 0.39891 | 10.96 % | 0.25 % |
| 0.300 | 0.32556 | 0.32355 | 0.62 % | 0.41939 | 0.47271 | 11.28 % | 0.27 % |

### 3.3 质量守恒账本

| 算例 | max |M(t)−M(0)|/M(0) |
|---|---:|
| Zalesak N=64 upwind | 0 |
| Zalesak N=64 cicsam | 5.4e-14 |
| Zalesak N=64 hric | 2.6e-14 |
| Zalesak N=128 upwind | 0 |
| Zalesak N=128 cicsam | 4.8e-16 |
| Zalesak N=128 hric | 6.4e-16 |
| Zalesak N=256 upwind | 0 |
| Zalesak N=256 cicsam | 1.2e-15 |
| Zalesak N=256 hric | 2e-15 |
| Zalesak N=64 plic | 8.8e-15 |
| Zalesak N=128 plic | 6.4e-16 |
| dam H/28 upwind | 7.4e-13 |
| dam H/28 cicsam | 4.3e-16 |
| dam H/56 upwind | 2.2e-16 |
| dam H/56 cicsam | 4.3e-16 |
| dam H/28 plic | 1.3e-15 |
| dam H/56 plic | 3.4e-15 |

最差 7.4e-13（判据 C3 为 1e-3）。

### 3.4 平滑场精度诊断（旋转余弦丘，一整圈）

| 格式 | N | 步数 | 相对 L1 | 峰值保持 | 质量漂移 |
|---|---:|---:|---:|---:|---:|
| upwind | 48 | 369 | 1.32 | 0.178 | 1.5e-16 |
| cicsam | 48 | 369 | 0.497 | 1.02 | 1e-13 |
| hric | 48 | 369 | 0.519 | 1.02 | 6.1e-14 |
| uq | 48 | 369 | 0.198 | 0.817 | 4.8e-14 |

结论：压缩支（CICSAM/HRIC）在平滑场上明显劣于纯 UQ 支——这是压缩格式的已知代价，本节点如实测量而非回避；界面类问题（Zalesak、溃坝）才体现其价值。

### 3.5 FCT 限幅器对照（溃坝 H/28，t ≤ 0.06 s）

| 配置 | max alpha | 超填单元·步 | 质量漂移 |
|---|---:|---:|---:|
| 限幅后（生产配置） | 1 | 0 | 2.2e-16 |
| 限幅前（对照） | 1.01002 | 93 | 2.2e-16 |

## 4. 结果图

**fig1_zalesak_fields-leg2.png** — 一圈刚体旋转后的 $\alpha$ 场（最细网格）。CICSAM 与 PLIC 保持尖锐界面；一阶 donor-cell 已把圆盘抹平（峰值保持约 0.4）。（该腿为独立运行：本图只含该腿的算例，用于补齐主图缺少的格式。）

`docs/P400W2-N01-two-phase-vof-verification/artifacts/figures/fig1_zalesak_fields-leg2.png`

**fig1_zalesak_fields.png** — 一圈刚体旋转后的 $\alpha$ 场（最细网格）。CICSAM 与 PLIC 保持尖锐界面；一阶 donor-cell 已把圆盘抹平（峰值保持约 0.4）。

`docs/P400W2-N01-two-phase-vof-verification/artifacts/figures/fig1_zalesak_fields.png`

**fig2_zalesak_metrics-leg2.png** — Zalesak 形状度量随分辨率的变化：面积误差（等值线口径）与周长误差，虚线为 C2 判据（1 % / 3 %）。一阶基线在网格加密后仍远离判据。（该腿为独立运行：本图只含该腿的算例，用于补齐主图缺少的格式。）

`docs/P400W2-N01-two-phase-vof-verification/artifacts/figures/fig2_zalesak_metrics-leg2.png`

**fig2_zalesak_metrics.png** — Zalesak 形状度量随分辨率的变化：面积误差（等值线口径）与周长误差，虚线为 C2 判据（1 % / 3 %）。一阶基线在网格加密后仍远离判据。

`docs/P400W2-N01-two-phase-vof-verification/artifacts/figures/fig2_zalesak_metrics.png`

**fig3_dam_front-leg2.png** — 左：波前位置 vs 时刻，实线为 Ritter 尖端 $W+2c_0t$，虚线为同一解析解在 $h=h_{\rm ref}$ 处的位置，符号为两种分辨率的实测；右：相对误差（判据 10 %）。（该腿为独立运行：本图只含该腿的算例，用于补齐主图缺少的格式。）

`docs/P400W2-N01-two-phase-vof-verification/artifacts/figures/fig3_dam_front-leg2.png`

**fig3_dam_front.png** — 左：波前位置 vs 时刻，实线为 Ritter 尖端 $W+2c_0t$，虚线为同一解析解在 $h=h_{\rm ref}$ 处的位置，符号为两种分辨率的实测；右：相对误差（判据 10 %）。

`docs/P400W2-N01-two-phase-vof-verification/artifacts/figures/fig3_dam_front.png`

**fig4_dam_mass-leg2.png** — 质量账本时序（对数纵轴），全部算例的 $|M(t)-M(0)|/M(0)$ 与 0.1 % 判据线。（该腿为独立运行：本图只含该腿的算例，用于补齐主图缺少的格式。）

`docs/P400W2-N01-two-phase-vof-verification/artifacts/figures/fig4_dam_mass-leg2.png`

**fig4_dam_mass.png** — 质量账本时序（对数纵轴），全部算例的 $|M(t)-M(0)|/M(0)$ 与 0.1 % 判据线。

`docs/P400W2-N01-two-phase-vof-verification/artifacts/figures/fig4_dam_mass.png`

**fig5_dam_profile-leg2.png** — 左：$t=0.3$ s 水深剖面 vs Ritter 解析剖面；右：液面位置示意（列平均 $\alpha$）。（该腿为独立运行：本图只含该腿的算例，用于补齐主图缺少的格式。）

`docs/P400W2-N01-two-phase-vof-verification/artifacts/figures/fig5_dam_profile-leg2.png`

**fig5_dam_profile.png** — 左：$t=0.3$ s 水深剖面 vs Ritter 解析剖面；右：液面位置示意（列平均 $\alpha$）。

`docs/P400W2-N01-two-phase-vof-verification/artifacts/figures/fig5_dam_profile.png`

## 5. 注意点（踩过的坑、局限、适用范围）

1. **Ritter 只对半无限水库成立。** 有限水柱（$W$ 宽）在 $t>W/c_0$
   （本算例 0.03778 s）后波前就脱离 $2t\sqrt{gH}$。
   任何把"有限水柱 + $2t\sqrt{gH}$ + 10 %"写进判据的方案都会得到一个必然 FAIL 的约束；
   本节点的做法是把域扩到含上游水库，并在报告里把这个物理事实写清楚。
2. **波前必须按深度匹配比较。** 尖端处 $h\to0$，有限分辨率只能给出"能分辨的深度"所在位置；
   用 $\alpha>1/2$ 的前沿去比零深度尖端，会引入 $3t\sqrt{g\,dy/2}$ 的解析滞后，本算例实测
   占波前位置 12 %（$t=0.05$）到 19 %（$t=0.3$）——这是比较口径缺陷，不是格式缺陷（表 3.2 两列并列）。
3. **深度平均速度不能直接输运二维 $\alpha$。** 只用 $(u(x,t),0)$ 时离散散度非零，水柱会被"稀释"：
   实测 $\alpha_{\max}$ 在 0.1 s 内从 1.0 掉到 0.50。正确做法是取连续性给出的
   $v(x,y,t)=-y\,\partial u/\partial x$（自由面即物质面），本节点按与输运相同的面差分构造 $v$，
   逐格离散散度为 0（`test_velocity_field_is_discretely_divergence_free`）。
4. **相似解在 $t\to0$ 奇异**（$\partial u/\partial x=2/(3t)\to\infty$，扇面宽度 $3c_0t\to0$）：
   数值积分从 $t_0=0.02$ s 的**解析场**起步，$t_0$ 在判据窗 $[0.05,0.3]$ s 之外，且这是全程唯一
   一处"给定解析状态"而非"算出来"的地方。
5. **NVD 面值不自动满足离散极值原理。** 开/关限幅的对照实测见 §3.5（限幅前 $\alpha_{\max}$ =
   1.01，限幅后 1）；限幅后为生产配置，限幅前的数字同样留在账上。
6. **三条面积/两条周长度量不等价**（§2 末尾），选哪条决定判据能否通过。本报告把三条全列、
   并写明 C2 采用哪一条及理由；一阶基线在所有口径下都远离判据，说明判据仍有区分力。
7. **度量自身的偏差已标定**：二值场的 $\alpha=1/2$ 等值线是阶梯多边形，$N=400$ 时仍偏长 5.6 %；
   亚网格准确的圆盘同一度量误差 0.3 %。因此"对初值"的口径只能作诊断。
8. **本节点只验证界面输运面**（重构 + 高分辨率对流 + 守恒），不验证动量耦合：耦合粘性二维坍塌与
   Ritter 的差距（−60 %/−31 %/−12 %）来自物理假设不同，已由 `p233w1-N03` 记录，本节点不重复也不掩饰。
9. **几何 PLIC 在薄层溃坝波前上反而更差。** PLIC 腿的波前误差分别为
   H/28 16.83 % / H/56 8.96 %（判据 10 %），明显大于同分辨率的代数 CICSAM（1.23 %/0.77 %），尽管它在 Zalesak
   周长项上更好。**待验证的机理假设**（未在本节点证明）：波前附近水深薄于一个网格、$\alpha$ 只有
   0.1–0.3，此时 PLIC 的 Youngs 法向/平面常数在亚网格薄层上不可靠；再加上方向分裂后必须开启
   `conservative_bounds` 的格内修复（否则 $\alpha_{\max}=1.0118$ 被模块判为 scheme defect），
   修复会重新分配体积。这一条是**边界案例的诚实记录**，不作为 PLIC 的否定结论。
10. **适用范围**：结构网格、CFL ≤ 0.4、二维 Lie 分裂（每步交替扫描方向）；未做三维、未接表面张力与相变
   （后者归其他节点）。CICSAM 在平滑场上的精度低于纯 UQ 支，属格式性格而非缺陷。

## 6. 误差判定（关键量 vs 基准，逐项回答“是否 <3 %”）

| 关键量 | 误差 | 是否 <3 % |
|---|---:|---:|
| Zalesak 面积误差（等值线 vs 初值），N=64 | 1.11 % | <3 % |
| Zalesak 面积误差（等值线 vs 初值），N=128 | 0.21 % | <3 % |
| Zalesak 面积误差（等值线 vs 初值），N=256 | 0.02 % | <3 % |
| Zalesak 周长误差（vs 解析周长），N=64 | 9.42 % | **≥3 %** |
| Zalesak 周长误差（vs 解析周长），N=128 | 3.91 % | **≥3 %** |
| Zalesak 周长误差（vs 解析周长），N=256 | 3.20 % | **≥3 %** |
| 溃坝波前 vs Ritter（匹配深度），H/28 | 1.23 % | <3 % |
| 溃坝水深剖面 L1 vs 解析，H/28 | 0.15 % | <3 % |
| 溃坝波前 vs Ritter（匹配深度），H/56 | 0.77 % | <3 % |
| 溃坝水深剖面 L1 vs 解析，H/56 | 0.07 % | <3 % |
| 溃坝波前 vs Ritter（匹配深度），plic H/28 | 16.83 % | **≥3 %（判据 10 %，未达标）** |
| 溃坝波前 vs Ritter（匹配深度），plic H/56 | 8.96 % | **≥3 %（判据 10 %，达标）** |
| 溃坝波前 vs Ritter（匹配深度），upwind H/28 | 2.37 % | <3 % |
| 溃坝波前 vs Ritter（匹配深度），upwind H/56 | 1.15 % | <3 % |
| 全局质量守恒（最差算例） | 7.4e-13 | <3 % |

说明：C1/C2 的判据线分别是 10 % 与 1 %/3 %（见 §0），本表额外按 owner 的 3 % 口径逐项回答；达不到 3 % 的项按要求显式标出，不做模糊表述。守恒量的误差是机器零量级，用相对量级表示（$10^{-n}$）。

## 7. 复现与审计链

| 项 | 值 |
|---|---|
| 仓库 | TensorFVM |
| 节点 | P400W2-N01 |
| 库代码 | `src/tensorfvm/twophase_bench/`（cicsam / metrics2d / ritter / zalesak / campaign） |
| 复现脚本 | `docs/P400W2-N01-two-phase-vof-verification/run_benchmark.py` |
| 渲染脚本 | `docs/P400W2-N01-two-phase-vof-verification/render_report.py`（本文件由它生成） |
| 测试 | `tests/test_twophase_bench_vof.py`（15 例） |
| Python | 3.11.6 |
| PyTorch | 2.7.1 |
| 机器 | loongarch64 / Linux-6.6.52-1.7.lns23.loongarch64-loongarch64-with-glibc2.38 / threads=6 |
| CFL | 0.4 |
| Zalesak 分辨率 | 64、128、256 |
| Zalesak 格式 | cicsam、hric、plic、upwind |
| 溃坝分辨率 | H/28、H/56 |
| 溃坝格式 | cicsam、plic、upwind |
| 判据窗 | [0.05, 0.3] |

一键复现（单线程 4 h 内、8 线程约 1 h）：

```bash
PYTHONPATH=src python docs/P400W2-N01-two-phase-vof-verification/run_benchmark.py --threads 8 \
    --zalesak-res 64 128 256 --zalesak-schemes upwind cicsam hric plic \
    --dam-cells 28 56 --dam-schemes upwind cicsam plic
PYTHONPATH=src python docs/P400W2-N01-two-phase-vof-verification/render_report.py
PYTHONPATH=src python -m pytest tests/test_twophase_bench_vof.py -q
```

