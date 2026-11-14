# CASE-FVM-CHANNEL-KMM180：把锚点判据搬到 FVM 本体上（验证欠账回收）

锚点 `CASE-FVM-CHANNEL-KMM180`（湍流槽道 `Re_tau=180` 平均速度对数律）。
验收：`y+>30` 上 `u+(y+)` 相对 DNS 参考 RMS 偏差 `< 5 %`。

## ① 案例介绍（补的是哪一条欠账）

该锚点此前**全部**判定读数都来自 **1-D 平衡 ODE 积分**
（`rans_validation.channel.solve_channel`）：`benchmark_channel_kmm1987`、
`kmm180_windows` / `_metric_norm` / `_refband` / `_direct_anchor` / `_loglaw` /
`_budget` / `_sublayer` / `_sa`。`docs/verification-channel-kmm180/REPORT.md`
把缺口写成待办：「未做：3-D 求解器的壁面律一致性」；`kmm180_metric_norm` 亦声明
「no claim is made about the 2-/3-D FVM discretisation」。FVM 一侧自己的槽道审计
（`docs/verification-native-channel-sst`）用的是**体积加权**范数 + 3 % 门槛，与锚点的
**非加权单元中心** RMS + 5 % 判据不是同一个度量，数字不可比。

本单把锚点判据原样搬到 FVM 解上：同一参考表、同一 `y+>30` 下界、同一 5 % 容差、
同一个度量函数 `kmm1987.log_region_deviation`；不新造指标、不改求解器/闭包/参考数据。

| 项 | 值 |
|---|---|
| 求解器 | 原生混合 FVM 充分发展槽道 + SST-2003（`verification.native_developed_channel`）|
| 驱动 | `dp*/dx = 1`，半槽道（`h=1`），壁面无滑移，中心对称，流向/展向挤出 |
| 参考 | `docs/rans-turbulence-validation-matrix/dns/0180`（Lee & Moser 2015，`Re_tau=182.088`，`Cf=0.008123`，`U_bulk+=15.691`）|
| 网格档 | `n=16/32/64`（`stretch=3`）+ `n=64`（`stretch=1.2`），`axial=n/8`，`spanwise=2` |
| 壁面单位 | 由解自身积分壁面牵引：`tau_w=-sum(T_wall,x)/A`，`u_tau=sqrt(tau_w/rho)`，`y+=y u_tau/nu`，`Cf=2 tau_w/U_b^2` |
| 容差 | 5 %（`kmm1987.TOLERANCE_LOG_REGION`，未改）|

## ② 怎么做的

1. `fvm_channel_profile(n, stretch, ...)`：跑原生 FVM SST 充分发展槽道，按 `y` 层把挤出方向归约成壁法向剖面，并**用解自身的壁面牵引**求 `u_tau`、`Cf`、`Re_tau`；
2. `grade_profile`：把该剖面交给锚点度量函数 `kmm1987.log_region_deviation`，分别评 `legacy_full`（`[30, 表顶]`）与 `log_layer`（`[30, 0.2 Re_tau]`）两个窗口；窗口内点数 < 3 记 `WITHHELD`（**不内插、不放宽**）；
3. `cross_solver_audit`：同档同 `Re_tau` 跑 1-D 平衡 ODE、同函数评级，并在共同 `y+` 上用 PCHIP 差分（不外插），把「离散差」与「闭包差」分开；
4. 作业：`run_compute` 递交（`cj3054` 首次暴露窄窗 0 点崩溃 → 修复为逐窗评级；`cj3061` 四档全部收敛）。

复现：
```
PYTHONPATH=src python -m tensorfvm.verification.kmm180_fvm_bridge \
  --cases 16:3 32:3 64:3 64:1.2 --out docs/case-fvm-channel-kmm180-fvm-bridge
```

## ③ 计算结果（`cj3061`，n330，程序化打印，无手抄）

| n | stretch | 收敛 | `Cf` | `Cf` 误差 vs DNS | `legacy_full` 判定 | `log_layer` 判定 |
|---|---|---|---|---|---|---|
| 16 | 3.0 | ✓ | 0.00758096 | −6.68 % | **PASS** | WITHHELD（窗口内 0 点）|
| 32 | 3.0 | ✓ | 0.00806066 | −0.77 % | **PASS** | WITHHELD（窗口内 <3 点）|
| 64 | 3.0 | ✓ | 0.00835113 | +2.80 % | **PASS** | **PASS** |
| 64 | 1.2 | ✓ | 0.00793452 | −2.32 % | **PASS** | **PASS** |

客观事实（不加解释）：

* `legacy_full` 窗口上，FVM SST 四档全部满足锚点 5 % 判据；
* `log_layer` 窗口（`[30, 0.2Re_tau]`，本雷诺数下仅约 6 个壁面单位宽）在 `stretch=3` 的 `n=16/32` 网格上**无法评级**（窗口内点数不足）——这解释了为什么 FVM 一侧此前从未报出锚点判据读数；
* 把网格挪到能把 3 个以上单元放进该窗口（`n=64`，两种 stretch）后，FVM 在**窄窗也 PASS**；而同一闭包的 1-D 平衡 ODE 在 `n=128/256` 上窄窗读数 5.72 %/6.21 %（`docs/verification-channel-kmm180-direct-anchor/REPORT.md`）——即 **FVM 解没有继承 1-D ODE 的窄窗失效**，方向与「FVM 更差」的预期相反。

## ④ 结果图

**未出图**：作业节点（n330）未安装 matplotlib，绘图步骤在 `summary.json` 写出前抛 `ModuleNotFoundError`，因此四档的 RMS 数值表、`u+` 剖面图与 1-D-vs-FVM 差分表**均未落盘**（`not_run`，不是 `passed`）。模块已加绘图降级保护（缺 matplotlib 时返回 `None` 并记录 `figure_missing_reason`），补跑即可出图与出表。

## ⑤ 注意点

* **归一化不同源，这是最关键的一点**：1-D 求解器按构造把 `u_tau` 钉成 1；FVM 的 `u_tau` 由自身壁面牵引得出，其 `Cf` 与 DNS 差 −6.68 %…+2.80 %。因此「FVM 窄窗 PASS」里既有离散效应，也混着壁面单位归一化的差异，**不能**直接读成「FVM 的闭包比 1-D ODE 更准」；
* `Cf` 的符号随档位翻转（−6.68 % → −0.77 % → +2.80 %），说明该量在本雷诺数下**未收敛**，任何单档 `Cf` 结论都不成立；
* 窄窗在本雷诺数下只有约 6 个壁面单位宽，`stretch=3` 的壁面加密网格把它几乎整段跳过——窗口可判性是网格决定的，不是物理决定的（与 `kmm180_metric_norm` 的结论同向）；
* 挤出方向层内速度极差 `u_spread_max` 随产物输出，用于确认剖面确为一维物理剖面；本单未出现非零极差告警；
* 本单不改求解器、闭包、参考数据与容差。

## ⑥ 误差判定（逐项，含「是否 <3 %」）

| 关键量 | 读数 | 是否 <3 % |
|---|---|---|
| FVM `Cf` vs DNS（n=16/32/64-s3/64-s1.2）| −6.68 % / −0.77 % / +2.80 % / −2.32 % | 否（仅 n=32 与 n=64-s1.2 在 3 % 内）|
| FVM `u+` 窄窗 RMS vs DNS | **未落盘（not_run）** | 未能判定 |
| FVM vs 1-D ODE 的 `Δu+` RMS / `Δu_tau` / `ΔCf` | **未落盘（not_run）** | 未能判定 |
| FVM `u+` legacy 窗 RMS vs DNS | 判定 PASS（阈值 5 %），具体 RMS 值未落盘 | 未能判定 |

**诚实结论**：本单确立了两件可核的事——(a) 锚点判据现在能作用在 FVM 解上，且窄窗在默认 FVM 网格上常常不可判（`WITHHELD`）；(b) 在能判的档位上 FVM 过锚点判据。但「FVM 是否真正继承/改善 1-D ODE 的对数律读数」这一问，因数值表未落盘而**仍是未完成**（`not_run`），不得报为已通过。
