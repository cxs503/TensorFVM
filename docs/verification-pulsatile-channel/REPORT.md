# Pulsatile (Womersley) plane-channel flow — verification report

**Anchor**: `NEW: FVM 无振荡/脉动流非定常验证面` (capability: `PeriodicChannelSolver` under time-dependent body force)
**Generated**: 2026-11-12T20:38:41+00:00  **Runtime**: 79.7 s  **Verdict**: PASS 16/16 criteria

## 1. 案例介绍（物理背景 / 几何 / 参数表）

定常 Poiseuille 槽道早已对解析抛物线验证过；**没有**任何验证面对的是同一个生产求解器
（`channel_periodic.PeriodicChannelSolver`：x 周期、y 壁面、显式对流 + 隐式扩散 + 精确离散投影）
的**非定常**行为：时间精度、驱动与速度的相位关系、振荡驱动下的壁面切应力。

物理问题：平面槽道 `0 <= y <= H`，两壁静止，流向体积力 `G(t) = G0 cos(omega t)`

```
du/dt = G0 cos(omega t) + nu d2u/dy2 ,   u(0,t) = u(H,t) = 0
```

精确周期解（Womersley 1955 平面槽道形式）：

```
uhat(y) = G0/(i omega) [ 1 - cosh(m (y - H/2)) / cosh(m H/2) ],  m = sqrt(i omega/nu)
u(y,t)  = Re{ uhat(y) e^{i omega t} },      delta = sqrt(2 nu / omega)（Stokes 层厚度）
```

它给出三个**独立的**解析锚：`omega -> 0` 归到定常抛物线；整体流量
`uhat_b = G0/(i omega)[1 - tanh(mH/2)/(mH/2)]`；壁面梯度 `duhat/dy(0) = G0/(i omega) m tanh(mH/2)`
（即壁面切应力随 W 增大**超前**驱动最多 45 度）。参考：Womersley, J. Physiol. 127 (1955) 553；
Batchelor (1967) §4.6；Stokes 层标度 `delta = sqrt(2 nu/omega)`。

| 参数 | 值 |
|---|---|
| 槽道高度 H | 1.0 m |
| 运动黏度 nu | 0.01 m^2/s |
| 密度 rho | 1.0 kg/m^3 |
| 体积力幅值 G0 | 0.012 m/s^2 |
| Womersley 数 W = H/delta | 2 / 5 / 10（主序列），40 / 80（解析锚） |
| 主网格 ny x nx | 32 x 4（y 向 32 单元，x 向 4 单元，x 均匀） |
| 时间步 dt/T | 1/100, 1/200, 1/400 |
| 采样 | 最后一个周期内均匀 8 个相位 |

## 2. 怎么做的（方法 / 流程 / 关键设置 / 工具链）

1. **先冻结解析式再碰求解器**：解析式本身先过 5 条判据（A1–A5），其中 A1 用**数值微分**
   把闭式解代回动量方程（不依赖推导自身代数），A4 用 Stokes 层衰减律
   `|uhat(y) - uhat(H/2)| = (G0/omega) exp(-y/delta)` 反求斜率，A2 取 `omega -> 0` 极限
   与解析抛物线对比。
2. **生产求解器接上时变驱动**：`PeriodicChannelSolver.step(body_force_m_s2=None)` 新增可选
   逐位置参数（`None` 时行为与旧的常量驱动**逐位相同**，A5 与
   `test_forcing_hook_matches_the_constant_force_path` 双重把关）。逐步取时间中点
   `G0 cos(omega (t + dt/2))` 作二阶居中。
3. **初值 = 精确周期解在 t=0 的值**，因此不需要等启动瞬态衰减；积分一个周期后再采样一个周期
   的 8 个相位（采样窗口严格为一个周期、不重复端点——重复端点会让单频 DFT 幅值系统性偏低）。
4. **度量**：剖面误差用 L2 范数除以解在一个周期内的**包络** `max_t ||u_exact||`
   （相位过零点时逐相位范数趋零，会把绝对误差放大成无意义的百分比，见 §5 坑 2）；
   流量与壁面切应力用单频 DFT 取复幅值，比较幅值比与相位差。
5. 工具链：numpy 闭式 + torch 求解器 + matplotlib 作图；全部判据阈值在跑之前写在模块 docstring 里。

## 3. 计算结果（程序化注入，零手抄）

### 3.1 判据总表

| id | 判据 | 实测 | 阈值 | 判定 |
|---|---|---|---|---|
| A1 | closed-form PDE residual (numerical derivatives) | 2.6570e-09 | < 1.000e-06 | PASS |
| A2 | omega->0 limit equals the analytic parabola | 1.1245e-09 | < 1.000e-06 | PASS |
| A3 | bulk / wall-gradient formulas vs numeric | 4.9802e-09 | < 1.000e-06 | PASS |
| A4 | Stokes-layer decay slope vs -1/delta | 4.1293e-03 | < 2.000e-02 | PASS |
| A5 | steady solver bulk velocity vs G0 H^2/(12 nu) | 4.8828e-04 | < 1.000e-03 | PASS |
| B1 | solver vs closed form, W=2, ny=32, dt/T=1/200 | 1.5429e-02 | < 3.000e-02 | PASS |
| B2 | solver vs closed form, W=5, ny=32, dt/T=1/200 | 6.0871e-03 | < 3.000e-02 | PASS |
| B3 | solver vs closed form, W=10, ny=32, dt/T=1/200 | 3.9393e-03 | < 3.000e-02 | PASS |
| B4a | bulk-velocity amplitude, W=5 | 3.3207e-03 | < 3.000e-02 | PASS |
| B4b | bulk-velocity phase lag, W=5 | 1.0727e-01 | < 5.000e+00 | PASS |
| B5 | wall-shear amplitude, W=5, ny=64 (3-point fit) | 1.6520e-02 | < 3.000e-02 | PASS |
| B6 | temporal convergence order at W=5 | 8.7449e-01 | > 7.000e-01 | PASS |
| B7 | wall-normal convergence ny/2 -> ny at W=5 | 1.0000e+00 | > 5.000e-01 | PASS |
| B8a | spanwise/normal velocity stays zero at W=5 | 0.0000e+00 | < 1.000e-12 | PASS |
| B8b | cell divergence after projection at W=5 | 0.0000e+00 | < 1.000e-10 | PASS |
| C1 | oscillatory vs steady profile difference (W=5) | 1.3414e+00 | > 1.000e-01 | PASS |

### 3.2 主序列：求解器 vs 闭式解（8 个相位）

| W | delta/H | dt | 步数 | 最大 L2 误差（包络归一） | 逐相位误差 |
|---|---|---|---|---|---|
| 2.0 | 0.707 | 0.7854 | 400 | 1.5429e-02 | 1.0e-02, 1.6e-03, 1.2e-02, 1.5e-02, 1.0e-02, 1.6e-03, 1.2e-02, 1.5e-02 |
| 5.0 | 0.283 | 0.1257 | 400 | 6.0871e-03 | 4.1e-03, 5.8e-03, 4.1e-03, 1.4e-03, 4.5e-03, 6.1e-03, 4.3e-03, 1.4e-03 |
| 10.0 | 0.141 | 0.0314 | 400 | 3.9393e-03 | 3.9e-03, 3.7e-03, 3.0e-03, 3.2e-03, 3.9e-03, 3.8e-03, 3.1e-03, 3.2e-03 |

### 3.3 时间收敛（W=5, ny=32）

| T/dt | 步数 | 最大 L2 误差 |
|---|---|---|
| 100 | 192 | 1.0646e-02 |
| 200 | 400 | 6.0871e-03 |
| 400 | 800 | 3.1674e-03 |

拟合阶数 **p = 0.874**（backward Euler + 显式对流，理论一阶）

### 3.4 壁面切应力幅值（W=5, ny=64）

- 三点单边估计器：2.574568e-03 Pa vs 精确 2.532728e-03 Pa → 误差 1.6520e-02
- 格式自洽（一阶）壁面通量：误差 8.1914e-04（只报不判，用于把壁面求积误差与内部精度分开）

- 精确壁面切应力相位 -46.28 deg，流量相位 -66.696 deg → 壁面切应力**超前**流量 20.41 deg

### 3.5 定常回归（常量驱动路径未变）

- 生产求解器定常流量 0.100048828 m/s vs `G0 H^2/(12 nu)` = 0.100000000 m/s，相对误差 4.883e-04（ny=64，O(dy^2)：ny=32 时 1.95e-3、ny=64 时 4.88e-4，比值 4.0）

## 4. 结果图

`fig_pulsatile_channel.png`（三面板）：(a) 峰值相位剖面，求解器（空心圆）vs 闭式解（实线），W=2/5/10；
(b) 时间收敛阶梯（W=5，拟合斜率）；(c) 流量幅值随 W 的滚降，精确 vs 求解器。
图由 `src/tensorfvm/verification/pulsatile_channel.py::figure()` 直接由 `summary.json` 同源数据生成。

## 5. 注意点（踩过的坑 / 局限 / 适用范围）

1. **重复端点会让单频 DFT 幅值系统性偏低**：第一版采样窗口含 `t+T` 的重复样本（9 个而不是 8 个），
   流量幅值误差报成 **7.6 %（假 FAIL）**；去掉重复端点后是 **0.33 %**。判据本身的实现错误会伪装成
   求解器错误——这类地方必须用"已知正确答案"（精确解即其自身）反测。
2. **逐相位范数归一化在过零点失效**：相位过半零时 `||u_exact|| -> 0`，同一绝对误差会变成
   5.6 % 的假大误差，且逐相位误差呈 2.1 %/5.6 % 交替；改为按周期包络归一后各相位一致。
3. **收敛阶的符号**：`dt` 是**减小**的，阶数公式除以 `log(dt_coarse/dt_fine)` 而不是反过来，
   否则会得到 p = −0.87 的假 FAIL（本单差点报错）。
4. **`omega -> 0` 的舍入地板**：`1 - cosh(m z)/cosh(m H/2)` 在 `omega H^2/nu = 1e-6` 时相消，
   舍入地板约 1e-10；判据阈值定 1e-6 而不是 1e-9（1e-9 会把舍入判成物理错误）。
5. **适用范围**：层流（本单最大 Re_H ~ 100 量级）、一维（x 均匀）、常量物性、壁面静止；
   不含壁面振荡驱动（Stokes 第二问题，本求解器的壁面是齐次无滑移，**不支持**）、
   不含湍流/可压缩/自由面。
6. **时间精度是真短板**：观测阶 0.875，一阶时间格式在振荡问题上的相位误差是主要误差源
   （dt/T=1/100 时相位差 4.9 deg 量级）。若后续要算振荡流的高保真相位（如振荡附加质量、
   壁面切应力相位），需要二阶时间格式（Crank–Nicolson / BDF2），本条已记入欠账。

## 6. 误差判定（逐项 vs 基准，是否 <3%）

| 关键量 | 基准 | 实测误差 | 是否 <3% |
|---|---|---|---|

| B1 | W=2 剖面 L2（8 相位） | 闭式 Womersley 解 | 1.5429e-02 | ✅ 是 |
| B2 | W=5 剖面 L2（8 相位） | 闭式 Womersley 解 | 6.0871e-03 | ✅ 是 |
| B3 | W=10 剖面 L2（8 相位） | 闭式 Womersley 解 | 3.9393e-03 | ✅ 是 |
| B4a | 流量幅值（W=5） | 闭式 uhat_b | 3.3207e-03 | ✅ 是 |
| B5 | 壁面切应力幅值（W=5, ny=64） | 闭式 duhat/dy(0) | 1.6520e-02 | ✅ 是 |
| B4b | 流量相位（W=5） | 闭式相位 | 0.107 deg（<5 deg 门） | ✅ 是 |
| A5 | 定常流量（回归） | G0H^2/(12nu) | 4.883e-04 | ✅ 是 |

**判定**：所有对解析基准的关键量误差 **均 < 3 %**（剖面 0.39 % / 0.61 % / 1.54 %，流量幅值 0.33 %，壁面切应力 1.65 %）；相位差 0.11 deg（<5 deg 门）。时间收敛阶 0.875（一阶格式的自洽结果，非精度缺陷，但记为待改进项）。

## 7. 证据指针

- 求解器改动：`src/tensorfvm/channel_periodic.py::PeriodicChannelSolver.step(body_force_m_s2=None)`
- 验证驱动：`src/tensorfvm/verification/pulsatile_channel.py`（16 条预声明判据）
- 测试：`tests/test_pulsatile_channel.py`（12 测全绿，53 s）
- 产物：`docs/verification-pulsatile-channel/{summary.json,fig_pulsatile_channel.png,REPORT.md}`
- 复跑：`PYTHONPATH=src python -m tensorfvm.verification.pulsatile_channel`

