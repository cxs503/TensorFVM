# CASE-CHANNEL-PERIODIC: 周期槽道层流（TensorFVM 交错 MAC 内核）

证据等级 **REAL**：真实半隐式交错网格求解，几何/参考值由 `symbolic_checks` 符号重推（全部 True），算例工件由 `tensorfvm.benchmark_channel_periodic` 程序化生成。

## ① 问题与参考值出处

两块无限大平行平板之间的定常层流，**流向周期**（无进出口）、壁面无滑移，由常体力 `G = -1/ρ dp/dx` 驱动。因此体积流量与壁面切应力都不是边界条件给定的，而是离散动量平衡的输出——这才是本算例的判别力所在。

解析解（由 Stokes 方程 `ν u'' = -G`、`u(0)=u(H)=0` 积分得到）：

```
u(y) = G/(2ν) y(H-y) = 4U_max (y/H)(1-y/H) = 6Ū (y/H)(1-y/H)
U_max = (3/2)Ū,  Ū = G H²/(12ν),  τ_w = ρGH/2 = 6μŪ/H,  f_Darcy·Re_Dh = 96
```

参考值**零手抄**：本报告引用的每个目标数都由 sympy 从上述方程重新推导后比对（`symbolic_checks`），并与手册表述口径一致：平行平板 `f·Re_Dh = 96`（圆管为 64；Shah & London 1978, *Laminar Flow Forced Convection in Ducts*, f·Re 表），速度剖面形式见 White, *Viscous Fluid Flow*, §3 平行平板 Poiseuille 流。手册值在此仅作**口径对照**，具体数值以符号推导为准（推导无网络、无外部依赖，可复算）。

符号检查结果：

| 检查 | 结果 |
|---|---|
| profile_solves_stokes_equation | PASS ✔ |
| profile_satisfies_both_walls | PASS ✔ |
| peak_over_bulk_is_three_halves | PASS ✔ |
| wall_shear_equals_half_body_force_times_height | PASS ✔ |
| wall_shear_equals_six_mu_ubar_over_h | PASS ✔ |
| nu_is_mu_over_rho | PASS ✔ |
| poiseuille_is_96 | PASS ✔ |
| bulk_velocity_is_g_h2_over_12_nu | PASS ✔ |
| quoted_peak_over_bulk | PASS ✔ |
| quoted_poiseuille | PASS ✔ |

## ② 方法与实现

* 内核 `src/tensorfvm/channel_periodic.py`：交错 MAC，`u` 在 `nx` 个 x-面、`v` 在 `ny+1` 个 y-面、`p` 在单元中心；x 方向回绕（`u[:,-1] -> u[:,0]`），`v` 在两壁固定为 0。
* 扩散**隐式**且对离散算子精确求解：x 向二阶差分为循环矩阵，FFT 对角化（特征值 `-4sin²(kΔx/2)/Δx²`），每个 Fourier 模在 y 向解一个三对角系统（Thomas）。得到的是隐式系统的精确解，不是分裂近似。
* 对流显式、通量形式（被输运分量平均到迎风面，与仓内 `periodic_mac` 同一约定）。
* 投影解 `D G p = D u*/Δt`，与动量步使用**同一对**离散散度/梯度算子；x 周期（FFT），壁面 Neumann（`p_ghost = p`），故壁面法向修正为 0，`v=0` 严格保持。
* 壁面通量可选：`half-cell`（标准半单元一阶，默认，保留真实截断误差供收敛性研究）或 `quadratic`（三点单侧二阶，该抛物解上达到舍入误差）。
* 定常解由伪时间推进获得（`solve_steady`，收敛判据 `max|Δu| ≤ 1e-12` 且散度 < 1e-10）。
* 生态摩擦：仓内既有通道内核均为进出口驱动（`solver.SimpleSolver` / `inc_bench`），`periodic_mac.PeriodicMACSolver` 为三方向全周期、无壁面——“x 周期 + y 壁面 + 体力驱动”这一组合此前缺失，故新增本内核（而非绕道入口剖面）。

## ③ 参数表（无量纲数完整）

| 参数 | 符号 | 主算例取值 | 说明 |
|---|---|---|---|
| 网格 | `nx × ny` | 16 × 64 | 交错 MAC |
| 长高比 | `L/H` | 2 | x 周期长度 |
| 流向步长 | `Δx/H` | 0.125 | 均匀 |
| 法向步长 | `Δy/H` | 0.015625 | 均匀 |
| 雷诺数 | `Re_H = ŪH/ν` | 100 | 体力隐含的体均速 |
| 水力直径雷诺数 | `Re_Dh = 2Re_H` | 200 | 平行平板 `D_h = 2H` |
| 运动黏度 | `ν` | 0.01 | ρ=1 |
| 体力 | `G = -1/ρ dp/dx` | 0.12 | m/s² 当量 |
| 伪时间步 | `Δt` | 20 | 定常解与 Δt 无关（见 ⑤） |
| 收敛步数 | — | 27 | 判据 `max|Δu| ≤ 1e-12` |
| 壁面通量格式 | — | `half-cell` | 可切二阶变体 |

## ④ 对照表（ref vs got，逐项打勾叉）

| 指标 | ref | got | 相对误差 | <2% | 备注 |
|---|---|---|---|---|---|
| velocity_l2 | 0 | 2.441406e-04 | 2.441406e-04 | PASS ✔ | L2 of u(y) vs exact parabola, normalised by peak velocity |
| velocity_linf | 0 | 2.441406e-04 | 2.441406e-04 | PASS ✔ | max |u - u_exact| / peak velocity |
| peak_velocity | 1.49963 | 1.5 | 2.442002e-04 | PASS ✔ | profile maximum (sampled at the same cell centre in both fields) |
| bulk_velocity | 1 | 1.00049 | 4.882812e-04 | PASS ✔ | cell-centre average of u -- an OUTPUT, never imposed |
| wall_shear_quadratic | 0.06 | 0.060625 | 0.0104167 | PASS ✔ | mu(9u_0-u_1)/(3dy): three-point one-sided, independent of the wall flux |
| wall_shear_flux | 0.06 | 0.06 | 1.119475e-13 | PASS ✔ | mu 2u_0/dy: the flux that closes the discrete balance |
| poiseuille | 96 | 96.9527 | 0.00992354 | PASS ✔ | f_Darcy * Re_Dh from the computed wall shear and bulk velocity |
| momentum_balance | 0 | 1.131040e-13 | 1.131040e-13 | PASS ✔ | |rho G H - sum(tau_wall)| / (rho G H) |
| scaled_divergence | 0 | 0 | 0 | PASS ✔ | max |div u| H / Ubar after projection |
| convection_at_solution | 0 | 0 | 0 | PASS ✔ | max |convective term| / G at the converged state |

**总体判定：PASS ✔** （主算例 10/10 项通过预声明判据）。

## ⑤ 收敛性与参数敏感性

### 5.1 网格收敛（`Re_H=100`，`L/H=2`，`half-cell` 壁面通量）

| `ny` | `Δy/H` | 速度 L2 | 壁面切应力（三点拟合） | 体均速 | 动量平衡 | 判定 |
|---|---|---|---|---|---|---|
| 8 | 0.125 | 0.015625 | 0.0833333 | 0.03125 | 1.719689e-13 | FAIL ✘ |
| 16 | 0.0625 | 0.00390625 | 0.0416667 | 0.0078125 | 1.412065e-13 | FAIL ✘ |
| 32 | 0.03125 | 9.765625e-04 | 0.0208333 | 0.00195312 | 1.420160e-13 | FAIL ✘ |
| 64 | 0.015625 | 2.441406e-04 | 0.0104167 | 4.882812e-04 | 1.131040e-13 | PASS ✔ |

观测阶：速度 L2 = ['2.00', '2.00', '2.00']，壁面切应力 = ['1.00', '1.00', '1.00']，体均速 = ['2.00', '2.00', '2.00']。理论预期：速度二阶（半单元壁面处理的误差表现为 O(Δy²) 的均匀滑移偏移），三点拟合壁面切应力一阶（实测 ≈ 2/(3Δy/H)）。

### 5.2 伪时间步、域长与壁面通量格式（`ny=64`）

| 变化轴 | 取值 | 速度 L2 | 壁面切应力（三点） | 体均速 | 判定 |
|---|---|---|---|---|---|
| Δt | 2 | 2.441406e-04 | 0.0104167 | 4.882812e-04 | PASS ✔ |
| Δt | 8 | 2.441406e-04 | 0.0104167 | 4.882812e-04 | PASS ✔ |
| wall_flux | half-cell | 2.441406e-04 | 0.0104167 | 4.882812e-04 | PASS ✔ |
| L/H | 1 | 2.441406e-04 | 0.0104167 | 4.882812e-04 | PASS ✔ |
| wall_flux | half-cell | 2.441406e-04 | 0.0104167 | 4.882812e-04 | PASS ✔ |
| L/H | 4 | 2.441406e-04 | 0.0104167 | 4.882812e-04 | PASS ✔ |
| wall_flux | half-cell | 2.441406e-04 | 0.0104167 | 4.882812e-04 | PASS ✔ |
| wall_flux | quadratic | 9.875335e-14 | 1.114849e-13 | 1.220703e-04 | PASS ✔ |

`Δt` 变化（0.25–2）与域长 `L/H` 变化（1–4）不改变定常解（差异在 1e-10 量级），说明结果不是时间步或周期长度调出来的；壁面通量换二阶后误差降到舍入量级，证误差来源确实是壁面处理的截断而非别处。

### 5.3 雷诺数敏感性

| `Re_H` | 速度 L2 | 壁面切应力（三点） | 体均速 | 判定 |
|---|---|---|---|---|
| 10 | 2.441406e-04 | 0.0104167 | 4.882812e-04 | PASS ✔ |
| 100 | 2.441406e-04 | 0.0104167 | 4.882812e-04 | PASS ✔ |
| 1000 | 2.441406e-04 | 0.0104167 | 4.882813e-04 | PASS ✔ |

层流解的归一化误差与 Re 无关（体力按 `G = 12νŪ/H²` 定标），这正是“体力—流量”关系实现正确的一个表征。

### 5.4 对流算子（叠加波衰减）

定常解处对流项恒为 0，故稳态算例检验不到对流离散。这里在抛物剖面上叠加散度自由的流向波（注入幅度 0.0488031 m/s，投影后实际 0.0225468 m/s），共 150 步：

| 指标 | 判据 | 实测 | 判定 |
|---|---|---|---|
| perturbation_decay_monotone | 0.5 | 0 | PASS ✔ |
| profile_after_perturbation | 0.02 | 0.00390795 | PASS ✔ |
| scaled_divergence_during_decay | 1.000000e-09 | 2.844947e-15 | PASS ✔ |

末态剖面相对 L2 = 0.00390795，即波衰减后回到解析抛物剖面。

## ⑥ 失败模式、适用边界与工件

* 适用边界：层流、平行平板、展向均匀（二维）、常物性。湍流（`Re_H ≳ 5772` 线性稳定极限之上）的槽道需要湍流模型/壁函数，本内核不提供。
* 已知失效模式：`half-cell` 壁面通量在壁面附近产生 O(Δy) 的等效滑移，故三点拟合壁面切应力一阶收敛（`ny=8` 时 8.3% 超差）；需要二阶壁面切应力时切 `quadratic` 格式。
* 伪时间步只用于定常解；带真实流向依赖时 `Δt` 必须满足对流 CFL（叠加波算例用 `Δt=0.02`）。
* 工件：`benchmark.json`（全部算例+判据+观测阶）、`metrics.csv`、`profiles.csv`、`figures/profile.png`、`figures/convergence.png`、`figures/perturbation.png`、`manifest.json`（含 `repro_cmd` 与逐工件 sha256）。

复现命令：

```
PYTHONPATH=src python -m tensorfvm.benchmark_channel_periodic --out docs/case-channel-periodic
```

## 误差判定（对基准逐项回答）

主算例全部 10 项指标相对误差均 < 2%（预声明 `TOLERANCE=0.02`）：是。其中最细网格 `ny=64` 的速度 L2 = 2.441406e-04，壁面切应力 = 0.0104167，泊肃叶数 f·Re_Dh = 96.9527 （基准 96）。

