# 自然对流方腔 Ra = 1e4 验证欠账清偿：显式时间步漏计热扩散导致的迟发型发散 (m6444)

**任务**: m6444 · selfstart 自找活（TensorFVM，verifier 席） · 日期 2026-11-08
**性质**: 验证欠账 + 缺陷修复（单核，轻量；load15 ≈ 124 未起集群）
**靶子出处**: `docs/benchmarks/incompressible-verification.md` §2.1「`Ra = 1e4` is **WITHHELD**:
the N = 64 run diverged late in the transient」。

## 1. 案例（物理背景与参数）

竖直差分加热方腔（Differentially heated square cavity），高度 `H = 1`，左壁热、右壁冷、
上下绝热、四壁无滑移；Boussinesq 近似，`Pr = 0.71`，`Ra = 1e4`（`alpha = 1`、`nu = Pr`）。
基准为 de Vahl Davis (1983)：`Nu = 2.243`、`|psi|max = 5.071`、`u_max = 16.178`、`v_max = 19.617`。
求解器：`src/tensorfvm/incompressible_benchmarks.py::StaggeredFlow`（MAC 交错网格，
van Leer 限制器，DCT-II 精确压力解，显式推进）。

| 参数 | 值 |
|---|---|
| Ra / Pr | 1e4 / 0.71 |
| 热扩散系数 alpha | 1.0（`_energy_tendency` 中硬编码 → 本轮显式参数化） |
| 运动黏性 nu | 0.71 |
| CFL | 0.5（默认值；此前的「成功」算例均手调 cfl ≤ 0.3） |
| 网格 | N = 16 / 32 / 64 方腔 |
| 收敛判据 | `max|du/dt|, |dv/dt| ≤ 1e-7`（每 100 步检查） |

## 2. 怎么做的

1. **复现**：默认 `cfl = 0.5` 运行 Ra = 1e4 N = 16 → 速度量级 `|u| ~ 20–50`、
   `change ~ 1e4` 长期不降（20 000 步不收敛），而 `cfl = 0.1` 时 2 300 步内收敛到
   `u_max = 16.23`（de Vahl Davis 16.178，0.3 %）。→ 说明不是「晚期发散」，而是
   **时间推进不稳定**，且只在低 Ra 出现。
2. **定位**：`StaggeredFlow.stable_time_step()` 只用了动量黏性
   `dt <= 0.25 h^2 / nu`（`nu = Pr = 0.71` → `0.352 h^2`），而能量方程的显式扩散用的是
   `alpha = 1`，其 2-D 五点拉普拉斯显式界为 `h^2/(4 alpha) = 0.25 h^2`。低 Ra 起始流动很慢，
   对流 CFL 界不起作用，`dt` 被钉在 **1.41× 超界的扩散上限**上 → 热模态指数增长。
   高 Ra 算例恰好因为流速快、`dt` 先被对流 CFL 压小，把这个缺陷掩盖了。
3. **修复**（最小改动，`src/tensorfvm/incompressible_benchmarks.py`）：
   * 新增 `alpha`（默认 1.0）参数并在 `_energy_tendency` 中使用（此前硬编码 1）；
   * `stable_time_step` 的扩散项改为 `max(nu, alpha)`，能量在场时再乘安全因子
     `DIFFUSION_SAFETY = 0.5`（裸界 `h^2/(4D)` 只是**边缘稳定**，实测 1.0 仍发散）；
   * 纯动量算例（`energy=False`，BFS/突扩）保持历史上界不变 → 既有工件不受影响
     （实测 Ra = 1e5 N = 16 重跑：`Nu = 5.1151`、`|psi|max = 11.330`，
     与存档 `5.115144 / 11.3298` 一致，证明**只改步长、不改定常解**）。
4. **证据生成**：用官方 runner `scripts/run_incompressible_benchmarks.py convection`
   （`--cfl 0.5`）写出 `raw_natural_convection_Ra1e+04_N{16,32}.json` + 场 `.npz`，
   并重生成 `summary.json`；下表数字全部由 JSON 程序化注入。

## 3. 计算结果

| N | Nu (hot wall) | error vs 2.243 | $|\psi|_{max}$ | error vs 5.071 | $u_{max}$ | error vs 16.178 | converged | steps |
|---|---|---|---|---|---|---|---|---|
| 16 | 2.3327 | +4.00 % | 5.171 | +1.97 % | 16.202 | +0.15 % | True | 1500 |
| 32 | 2.2670 | +1.07 % | 5.065 | -0.11 % | 16.234 | +0.35 % | True | 5500 |
| 64 | 2.2504 | +0.33 % | 5.067 | -0.09 % | 16.207 | +0.18 % | True | 20500 |

（`u_max` 取面向速度绝对最大值；参考值同为 de Vahl Davis 表。）

图：`fig_ra1e4_convergence.png`（Nu 与 |psi|max 随网格与参考线）、
`fig_ra1e4_fields.png`（N = 32 的温度场与流函数/流线）。
