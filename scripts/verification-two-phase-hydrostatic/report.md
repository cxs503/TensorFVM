# 两相瞬态投影路径（Chorin）静水/表面张力验证欠账审计

TensorFVM `src/tensorfvm/two_phase_solver.py::TwoPhaseSolver._projection_step`
（`tp.enabled=True` 的瞬态路径）在合并后只有 `xfail(strict=True)` 复现，没有
对解析基准的定量核对。本审计只读地量化四处判据，并把可修的一处（全封闭域
全 Neumann 压强 Poisson）改成 fail-fast。

## 1 案例与参数

| 项 | 值 |
|---|---|
| 计算域 | 1.0 m × 1.0 m，二维 MAC 交错网格，全流体 |
| 相物性 | ρ_l = 1000, ρ_g = 1 kg/m³；μ_l = μ_g = 1e-3 Pa·s（液滴算例 1e-2） |
| 重力 | g = 9.81 m/s²（−y），顶面大气 Dirichlet，底/侧壁无通量 |
| 静水锚点 | 界面 y = 0.5 m，`initialize_hydrostatic_pressure(0.5)` |
| 网格 / 时间步 | n = 16 / 32 / 64；dt = 1e-5 s，10–20 步 |
| 液滴算例 | n = 32，R = 0.2 m，σ = 0.07 N/m，g = 0 |

解析基准（无需外部数据）：
1. 连续静水剖面 `p = ρ_g g(H−y)`（气）与 `ρ_g g(H−h_i) + ρ_l g(h_i−y)`（液），
   界面**连续**，逐面须满足 `dp/dy = −ρ_face g`；
2. 静止分层柱在重力下 `u = v = 0` 恒成立（离散方程稳态解）；
3. 二维静态圆滴 Young–Laplace：`Δp = σ/R`（二维，非 `2σ/R`）；
4. 全封闭域压强 Poisson 为纯 Neumann，压力水平不定 → 必须 fail-fast。

## 2 方法

`scripts/audit_two_phase_hydrostatic.py`（只读，不改 `src/`）四个探针：
A 锚点连续性——取界面两侧 cell 的压差与面一致值 `−ρ_face g dy` 比对，并输出
逐 v-面静水平衡残差剖面；B 静止柱漂移——每步记录 `max|u|, max|v|`；
C 液滴压差——滴内/外均值差 vs `σ/R`；D 封闭域——是否抛 `ValueError`。
结果程序化写入 `evidence.json`、`report-table.md` 和下图，零手抄。

![audit](two_phase_projection_audit.png)

## 3 结果

| 项 | 判据 | 实测 | 判定 |
|---|---|---|---|
| A 静水锚点连续 | 界面跳变 = -76.717 Pa | -71.812 Pa（缺 4.905 Pa 气柱项） | FAIL |
| B 静止柱保持静止 | max\|v\| < 1e-9 m/s | 见下表（10 步内发散） | FAIL |
| C Young–Laplace σ/R | 误差 < 3%（σ/R = 0.350 Pa） | 0.000e+00 Pa（100.0%） | FAIL |
| D 全封闭域 fail-fast | 抛 ValueError | 抛 ValueError | PASS（本单修复） |

静止分层柱 `max|v|` 演化（dt = 1e-5 s，g = 9.81 m/s²）：

| 网格 | 第 1 步 | 第 10 步 | 增长倍数 |
|---|---|---|---|
| n=16 | 3.148e-02 | 1.190e+38 | 3.779e+39 |
| n=32 | 1.700e-01 | 9.233e+49 | 5.431e+50 |
| n=64 | 8.571e-01 | 4.574e+61 | 5.337e+61 |

封闭域：`抛 ValueError（消息：sealed domain: the pressure Poisson problem is all-Neumann, …）`。

## 4 根因（读码定位，按影响排序）

1. **投影丢弃了携带压强**：`_projection_step` 的预测步只有扩散+重力，压强 Poisson
   解的是 `∇·(1/ρ∇p) = ∇·u*/dt` 的**总压**，且从零初值求解；对均匀重力，
   `v* = −g·dt` 本身无散度，于是 `p = 0`、修正为零——自由落体是离散方程的
   合法解，静水压无法维持。
2. **锚点不连续**：`initialize_hydrostatic_pressure` 液相支缺 `ρ_g g(H−h_i)`
   （默认水/气 4.905 Pa），界面面上一侧梯度与 `−ρ_face g` 差 78 Pa/m，
   相当于 0.16 m/s² 的虚假加速度。
3. **表面张力未进预测步**：`sigma/surface_tension` 只在 `TwoPhaseProperties.csf`
   里存在，瞬态路径从未调用，静态液滴 `Δp = 0`。
4. **顶部大气 Dirichlet 加在 cell 中心**而非 y = H 的面（斜率差 ρ_g g dy/2）。
5. **预测步 stencil 少算侧壁列**：扩散/重力只更新 `v[1:-1, 1:-1]`，
   而压差修正作用于 `v[1:-1, :]`，壁列凭空获得 `−g·dt`。

修 1+2+4+5 需要重写瞬态路径的压强携带形式（incremental projection + 面一致
Dirichlet + 一致 stencil）；本单尝试过一版部分修补，在 64² 上仍不稳定，
按纪律**不落未验证的求解器改动**，只保留可独立验证的 D 项。

## 5 注意点 / 局限 / 适用范围

- 本审计是**失败面量化**，不是求解器修复；`src/` 仅增加封闭域 fail-fast 一处，
  流体结果与合并时逐位相同。
- B 项在 64² 上第 10 步 `max|v|` 已达 4.6e+61 m/s，
  说明当前瞬态路径对「静止两相」这一类问题不可用；任何基于它的算例须先声明。
- 二维 σ/R 判据不适用于三维（三维为 2σ/R）；液滴面积守恒另测（已通过）。
- 增长倍数依赖 dt 与网格，表中给的是绝对量级，不构成收敛阶结论。

## 6 误差判定（逐项，显式回答「是否<3%」）

| 关键量 | 基准 | 实测 | 相对误差 | <3%？ |
|---|---|---|---|---|
| 界面压差（锚点连续性） | -76.717 Pa | -71.812 Pa | 6.39% | 否 |
| 静止柱 max\|v\|（10 步） | 0 m/s | 4.57e+61 m/s | 无界 | 否 |
| Young–Laplace Δp | 0.350 Pa | 0.000e+00 Pa | 100.0% | 否 |

三项均**不满足 <3%**：A、C 是真实缺陷（锚点/缺源项），B 是 1+2+4+5 的合成
后果。唯一 PASS 的是 D（fail-fast 守门），其判据是二值（是否抛异常），不适用
百分比误差。

## 7 复现

```bash
PYTHONPATH=src python scripts/audit_two_phase_hydrostatic.py   --out scripts/verification-two-phase-hydrostatic
python -m pytest tests/test_two_phase_solver_verification.py -q
```
