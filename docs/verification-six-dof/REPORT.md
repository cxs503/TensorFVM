# TensorFVM `six_dof` 验证面（盒形驳船 10×5×2 m）

> 驱动：`PYTHONPATH=src python -m tensorfvm.verification.six_dof`（运行时 1.7 s）
> 工件：`summary.json`（本文件所有数字的唯一来源）、`fig_six_dof.png`
> 测试：`tests/test_six_dof.py`（14 测）。判据在评估前预声明，失败即记 FAIL，不覆盖。

## 1 案例介绍

`src/tensorfvm/six_dof.py` 是 `wave_tank`（耐波时域耦合）与操纵链共用的刚体运动积分器，
**在本单之前零测试**（`tests/` 中无任何 `six_dof` 文件）。验证对象是一个等吃水盒形驳船：

| 量 | 值 |
|---|---|
| L × B × T | 10 × 5 × 2 m |
| 水线面面积 A_wp | 50.0 m² |
| 水线面纵惯矩 I_wp = B L³/12 | 416.6667 m⁴ |
| 质量 m = ρLBT | 100000 kg |
| 纵摇惯矩 I_yy | 866667 kg·m² |
| 垂荡恢复刚度 C33 = ρgA_wp | 490500 N/m |
| 纵摇恢复刚度 C55 = ρgI_wp | 4087500 N·m/rad |
| ω_n(垂荡) / ω_n(纵摇) | 2.214723 / 2.171715 rad/s |

物理背景：该积分器用**辛欧拉（symplectic Euler）**推进
`m_eff ẍ = F_hydrostatic + F_damping + F_ext`，其中恢复力/力矩为线性水静力项。
选择这一对象的原因是它的每一个物理响应都有闭式解，可以把"实现"与"公式"逐项对账。

## 2 怎么做的

1. **闭式锚（先冻结后跑）**：垂荡/纵摇恢复力 `-ρgA_wp z`、`-ρgI_wp θ`；自然频率
   `√(k/m_eff)`；无阻尼自由响应 `z0 cos(ω_n t)`；有阻尼自由响应
   `z0 e^(-ζω_n t)[cos(ω_d t) + (ζω_n/ω_d) sin(ω_d t)]`；驱动稳态幅值
   `(F0/k)/√((1-r²)² + (2ζr)²)`。
2. **交叉法**：`rao()` 是频域闭式，但本驱动不拿公式对公式——而是让积分器在谐波外力
   `F0 cos(ωt)` 下跑到稳态，**实测幅值**再与 `(F0/k)·rao(ω)` 比（时域 vs 频域）；
   另外用 `scipy.integrate.solve_ivp(DOP853, rtol=1e-12)` 作为第二条独立积分路线。
3. **Richardson 观测阶**：无阻尼自由垂荡做 dt 阶梯，检查观测阶 p≈1（辛欧拉是一阶）。
4. **三条负控**：翻转恢复力符号必须被锚拒绝（≥50%）；非辛的显式欧拉必须出现 ≥100% 能量增长；
   忽略用户配置阻尼（历史上硬编码 ζ=0.05）必须在共振处与正确值差 ≥50%。
5. **缺陷修复**（本单查获 2 处）：
   - `rao()` 过去**忽略 `config.damping`**、硬编码 `ζ=0.05`，docstring 还写"undamped"。
     对 ζ=0.2 的配置，共振点给出 10.0 而正确值 2.5（**300% 误差**）。已改为经
     `damping_ratio()` 从 `config.damping` 反算 ζ（无配置时回落到 0.05 占位，保持向后兼容），
     并新增 `rao(..., zeta=)` 显式覆盖。
   - `history['time']` 用 `len(history)*dt`，当调用方改变 dt（流体求解器子步进）时时间轴错位。
     已改为按实际 dt 累加（`self.time`）。

## 3 计算结果

判据总表（19/19 PASS）：

| criterion | value | gate | mode | status |
|---|---|---|---|---|
| V1_restoring_coefficients_exact | 0.000e+00 | 1.000e-15 | max | PASS |
| V2_natural_frequency_exact | 0.000e+00 | 1.000e-15 | max | PASS |
| V3a_undamped_harmonic_observed_order | 1.025e+00 | 8.500e-01 | min | PASS |
| V3b_undamped_harmonic_fine_grid_error | 1.199e-02 | 3.000e-02 | max | PASS |
| V3c_undamped_harmonic_order_upper | 1.025e+00 | 1.150e+00 | max | PASS |
| V4a_damped_decay_rate | 5.925e-04 | 3.000e-03 | max | PASS |
| V4b_damped_trace_vs_closed_form | 2.076e-03 | 3.000e-02 | max | PASS |
| V5a_energy_bounded | 7.916e-03 | 5.000e-02 | max | PASS |
| V5b_energy_not_secular | 5.000e-03 | 1.000e-02 | max | PASS |
| V5c_control_explicit_euler_grows | 1.381e+02 | 1.000e+00 | min | PASS |
| V6_pitch_dof_harmonic | 1.199e-02 | 3.000e-02 | max | PASS |
| V7a_rao_time_domain_vs_formula | 8.024e-03 | 3.000e-02 | max | PASS |
| V7b_damping_ratio_from_config | 0.000e+00 | 1.000e-12 | max | PASS |
| V7c_resonance_equals_one_over_2zeta | 0.000e+00 | 1.000e-12 | max | PASS |
| V7d_rao_limits | 1.000e-04 | 1.000e-02 | max | PASS |
| V7e_control_ignoring_configured_damping | 3.000e+00 | 5.000e-01 | min | PASS |
| V8_independent_integrator_crosscheck | 6.056e-03 | 1.000e-02 | max | PASS |
| V9_history_time_nonuniform_dt | 0.000e+00 | 1.000e-12 | max | PASS |
| V10_control_flipped_restoring_rejected | 1.807e+07 | 5.000e-01 | min | PASS |

无阻尼自由垂荡（对 `z0 cos(ω_n t)` 的 dt 阶梯）：

| steps | dt [s] | relative L2 |
|---|---|---|
| 100 | 8.511e-02 | 1.078e-01 |
| 200 | 4.256e-02 | 5.045e-02 |
| 400 | 2.128e-02 | 2.438e-02 |
| 800 | 1.064e-02 | 1.199e-02 |
观测阶 p = **1.0247**（一阶，超收敛到 1.025）。

有阻尼自由响应（峰值包络对数斜率 vs 解析 ζω_n）：

| ζ | measured decay | ζω_n | rel err | trace rel L2 (1 period) | peak time shift [s] |
|---|---|---|---|---|---|
| 0.05 | 1.108e-01 | 1.107e-01 | 1.803e-04 | 1.788e-03 | -1.419e-03 |
| 0.2 | 4.432e-01 | 4.429e-01 | 5.925e-04 | 2.076e-03 | -1.419e-03 |

RAO：时域实测稳态幅值 vs `(F0/k)·rao(ω)`（子表节选全部 10 个工况）：

| ζ | ω/ω_n | time domain [m] | formula [m] | rel err |
|---|---|---|---|---|
| 0.05 | 0.5 | 2.711e-02 | 2.712e-02 | 3.572e-04 |
| 0.05 | 0.9 | 9.678e-02 | 9.697e-02 | 2.035e-03 |
| 0.05 | 1.0 | 2.039e-01 | 2.039e-01 | 1.278e-05 |
| 0.05 | 1.5 | 1.623e-02 | 1.619e-02 | 1.989e-03 |
| 0.05 | 2.0 | 6.835e-03 | 6.781e-03 | 8.024e-03 |
| 0.2 | 0.5 | 2.623e-02 | 2.627e-02 | 1.305e-03 |
| 0.2 | 0.9 | 4.998e-02 | 5.008e-02 | 2.162e-03 |
| 0.2 | 1.0 | 5.097e-02 | 5.097e-02 | 9.139e-06 |
| 0.2 | 1.5 | 1.473e-02 | 1.470e-02 | 2.062e-03 |
| 0.2 | 2.0 | 6.575e-03 | 6.566e-03 | 1.317e-03 |

能量（辛欧拉 vs 显式欧拉，50 个自然周期）：

| 量 | 值 |
|---|---|
| 辛欧拉 max|E-E0|/E0 | 7.916e-03 |
| 前 1/5 窗均值 / 后 1/5 窗均值 | 5.000e-03 / 5.000e-03 |
| 显式欧拉 max|E-E0|/E0（负控） | 1.381e+02 |

独立积分器交叉核对（DOP853, rtol=1e-12）：相对 L2 = **6.056e-03**。
时间轴记账（非均匀 dt）：误差 **0.000e+00**。

## 4 结果图

`fig_six_dof.png`（四联图）：(a) 无阻尼自由垂荡与闭式解；(b) dt 阶梯与 slope-1 参考线；
(c) 辛欧拉能量有界 vs 显式欧拉发散；(d) RAO：时域标记 vs 闭式曲线（ζ=0.05/0.2）。

## 5 注意点（踩坑与局限）

- **闭式解漏项陷阱**：有阻尼从静止释放的解必须带
  `(ζω_n/ω_d) sin(ω_d t)` 项。本单第一版漏了它，在 ζ=0.2 处表现出 **19%** 的假幅值误差，
  会被误判成"积分器缺陷"。修正后同一轨迹误差降到 2.076e-03。
- **一阶方法的相位漂移**：辛欧拉有 O(dt·t) 的相位误差，多周期逐点比较测的是相位而非阻尼。
  因此门控窗口取一个自然周期，多周期值只报告（`trace_rel_l2_full`）不判。
- **局限 1**：`six_dof` 无纵荡-纵摇、垂荡-纵摇耦合项（质量矩阵是对角的），本验证面只覆盖
  解耦 SDOF 路径；耦合项存在与否不能用这些锚判定。
- **局限 2**：阻尼是线性的、恢复力是线性水静力；无粘性阻尼/非线性恢复（大幅横摇 GZ 曲线）不在本面内。
- **局限 3**：`--quick` 模式（RAO 20 周期/300 步、DOP853 2000 步）会超出 3% 门，
  只能作烟测，不得用于判级。
- **无关的既有失败（如实记）**：`tests/test_wave_tank.py` 有 2 个测试在**本单改动前后同样失败**
  （`WaveTank' object has no attribute 'solver'`，已用 `git stash` 对照确认），属既有缺陷，未在本单修。

## 6 误差判定

- 关键量 1：无阻尼自由垂荡（最细网格相对 L2）**1.199e-02**（1.20%），**<3% ✅**
- 关键量 2：有阻尼衰减率（最差）**5.925e-04**（0.059%），**<3% ✅**
- 关键量 3：RAO 时域 vs 频域（10 工况最差）**8.024e-03**（0.80%），**<3% ✅**
- 关键量 4：独立积分器 DOP853 交叉核对 **6.056e-03**（0.61%），**<3% ✅**
- 关键量 5：能量（辛欧拉）最大相对漂移 **7.916e-03**，**<3% ✅**（且无世俗增长）
- 修复前缺陷量化：ζ=0.2 共振 RAO 由 10.0 → 2.5，**相对误差 300%**（远超 15% 不可接受线 → 必须修）

结论：19/19 判据 PASS，关键量全部 <3%；两处缺陷已修并有回归测试；
未覆盖项（耦合质量矩阵、非线性恢复、粘性阻尼）如上明示，不做粉饰。
