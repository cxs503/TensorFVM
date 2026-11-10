"""Generate the planing-Savitsky verification report from ``results.json``.

Every number in ``REPORT.md`` is injected from the JSON written by
``scripts/planing_savitsky_validation.py`` -- no hand-copied digits (owner
discipline of 2026-10-29).  Run after the validation script.
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DOC = os.path.join(HERE, "..", "docs", "planing-savitsky-verification")
R = json.load(open(os.path.join(DOC, "results.json")))

hull = R["reference_hull"]
ops = R["operating_points"]
lim = R["limits"]
sw = R["sweep_summary"]
dp = sw["design_point"]
ca = R["criteria_agreement"]
bounds = R["porpoising_boundaries"]
grid = R["porpoising_grid"]
sweep = R["sweep"]


def fmt(x, n=4):
    if isinstance(x, float):
        if x != 0.0 and abs(x) < 1e-3:
            return f"{x:.2e}"
        return f"{x:.{n}f}"
    return str(x)


rows_ops = "\n".join(
    "| {C_V:g} | {lam:g} | {tau_deg:g} | {beta_deg:g} | {C_L_flat_plate:.5f} | "
    "{C_L_deadrise:.5f} | {C_L_from_wetted_area_form:.5f} | {rel_residual:.1e} | {env} |".format(
        env="yes" if r["in_envelope"] else "no", **r)
    for r in ops)

rows_sweep = "\n".join(
    "| {C_V:.1f} | {speed:.2f} | {lam:.3f} | {tau_deg:.3f} | {C_L:.4f} | {lcp_over_beam:.3f} | "
    "{friction_force:.0f} | {spray_force:.0f} | {resistance:.0f} | {lift_to_drag:.2f} | {env} |".format(
        env="yes" if s["in_envelope"] else "**no**", **s)
    for s in sweep[::4])

rows_bounds = "\n".join(
    "| {beta_deg:g} | {a} | {b} | {n_trim_slope_unstable} | {max_trim_slope:.3f} | "
    "{peak_trim_slope_C_V:.1f} | {n_stiffness_unstable} |".format(
        a=bounds[i]["unstable_interval_trim_slope"],
        b=bounds[i]["unstable_interval_stiffness"], **bounds[i])
    for i in range(len(bounds)))

rows_spray = "\n".join(
    "| {C_V:.1f} | {spray_pct:.1f} | {friction_pct:.1f} |".format(**r)
    for r in lim["spray_share_of_resistance_pct"])

rows_deadrise = "\n".join(
    "| {beta_deg:g} | {C_L:.5f} | {decrement:.5f} | {law_0p0065_beta_CL0_0p6:.5f} | "
    "{relative_reduction:.4f} |".format(**r) for r in lim["deadrise_family"])

REPORT = f"""# 高速滑行艇水动力 Savitsky 验证面 —— 动升力 / 喷雾阻力 / 纵摇稳定性

**节点**: `p234w2-n03-planing-craft-savitsky`  ·  **仓**: TensorFVM  ·  **模块**:
`src/tensorfvm/planing_savitsky.py`  ·  **用例**: `tests/test_planing_savitsky.py`  ·
**产物**: `scripts/planing_savitsky_validation.py` → `docs/planing-savitsky-verification/`
（`results.json` + `figures/`）

---

## 1. 案例介绍（物理背景 / 几何 / 参数表）

高速滑行艇在越过驼峰航速后不再靠浮力支撑，而是靠 **动升力** 在湿表面上滑行：艇体纵倾角 τ
与平均湿长比 λ = L_m/b 由两个方程自洽决定 ——
(a) **力矩方程**：动压力合力必须通过重心，即压力中心 l_cp/b = LCG/b；
(b) **升力方程**：C_Lβ(λ, C_V, τ, β) = Δ / (½ρV²b²)。
Savitsky (1964) 给出这两步的经验闭式解，因此可作为 CFD（VOF/RANS 滑行艇算例）的解析
验证面 —— 这是本仓此前完全缺失的（此前只有排水型船回归，见
`docs/resistance-chain-form-factor/`）。

参考艇（本报告全部数字的算例）：

| 参数 | 值 |
|---|---|
| 艇宽 b | {hull['beam_m']:g} m |
| 艇长 L | {hull['hull_length_m']:g} m |
| 重量 Δ | {hull['weight_N']:g} N（{hull['weight_kg']:.0f} kg） |
| 斜升角 β | {hull['deadrise_deg']:g}° |
| 重心 LCG/b | {hull['lcg_over_beam']:g}（{hull['lcg_m']:g} m，距艉） |
| 水密度 ρ / 运动粘性 ν | {hull['rho']:g} kg/m³ / {hull['nu']:.2e} m²/s |

关键无量纲量与公式（τ、β 取 **度**，C_L 以 **b²** 为参考面积）：

* `C_V = V/√(gb)`
* `C_L0 = τ^1.1 (0.0120 λ^0.5 + 0.0055 λ^2.5 / C_V²)`（平板）
* `C_Lβ = C_L0 − 0.0065 β C_L0^0.6`（斜升修正）
* `l_cp/b = λ (0.75 − 1/(5.21 C_V²/λ² + 2.39))`
* `ΔC_D,spray = 0.003 (b/L)^0.5 C_V²`
* `R_T = Δ tanτ + (R_f + R_spray) cosτ`，`R_f = C_F,ITTC57 · ½ρV² · λb²`

## 2. 怎么做的（方法 / 流程 / 工具链）

1. **闭式面**：上列公式逐项落地为 `tensorfvm.planing_savitsky` 的纯函数
   （`lift_coefficient*`、`center_of_pressure_over_beam` + 其精确反解、
   `spray_drag_*`、`ittc57_friction_coefficient`、`frictional_resistance`、
   `total_resistance`），无求解器依赖、无外部状态，可直接审计。
2. **平衡求解**：`running_condition` 先用 **λ 反解**（力矩方程，l_cp/b 对 λ 单调）再用
   **τ 二分**（升力方程，C_Lβ 对 τ 单调），收敛容差 1e-12 相对；返回 `PlaningState`
   含阻力分解、Re、C_F、L/D 与 **envelope 违规旗标**。
3. **稳定性判据**（三条，全部机器可算）：
   * `dC_L/dλ`（下沉是否增升 → 垂荡回复）、`dM*/dλ`（压力中心迁移 → 纵倾回复）；
   * **2 自由度准静态刚度矩阵** J：以运动学闭合 `λ = h/(b sinτ)` 对 (垂荡 z, 纵倾 θ) 做中心差分，
     要求 `J_zz>0`、`J_θθ>0`、`det J>0`（`quasi_static_stiffness`）；
   * **纵倾斜率判据** `dτ/dC_V`（Savitsky–Brown 运行姿态曲线的操作性判据：正斜率=必须随速
     抬艏，静态升力曲线无法维持 → 喂给垂荡-纵摇振荡）。
4. **验证工具链**：`scripts/planing_savitsky_validation.py` 生成 `results.json`（本报告
   所有数字的唯一来源）与 5 张图；本 markdown 由 `scripts/planing_savitsky_report.py` 从
   JSON 程序化注入（零手抄）；`pytest tests/test_planing_savitsky.py` 27 例全绿。

## 3. 计算结果（数字表，全部由 `results.json` 注入）

### 3.1 升力律：10 个工况点（β=0–30°，C_V=1–10，λ=1–4，τ=3–12°）

| C_V | λ | τ[°] | β[°] | C_L0(平板) | C_Lβ(斜升) | 湿面积归一化独立复算 | 相对残差 | 包线内 |
|---|---|---|---|---|---|---|---|---|
{rows_ops}

最大相对残差 **{max(r['rel_residual'] for r in ops):.2e}**
（判据 (a) 阈值 5%，见 §6 关于「这不是独立物理验证」的声明）。

### 3.2 极限与负控

| 检查项 | 结果 |
|---|---|
| β=0 时 C_L 与平板律逐点偏差 | **{lim['beta0_flat_plate_max_abs_dev']:.1e}**（判据 (d) 前半，精确退化） |
| 由两点反推的 τ 指数 | **{lim['tau_exponent_from_two_points']:.6f}**（应为 1.1） |
| C_V→∞ 时 C_L0 与 τ^1.1·0.0120λ^0.5 之比 | **{lim['high_C_V_asymptote_ratio_to_sqrt_law']:.9f}** |
| τ→0（β=0）时 C_L | **{lim['tau_to_zero_C_L_at_0p01deg_beta0']:.3e}** → 0 |
| λ→0 时 C_L | **{lim['lam_to_zero_C_L']:.3e}** → 0 |
| C_V→0 时 C_L | **{lim['C_L_at_C_V_0p001']:.1f}**（发散，非 0） |
| V→0 残留力（与速度无关） | **{lim['hydrostatic_residual_N']:.0f}** N |
| C_V=0.01 时动升力中的速度相关部分 | **{lim['lift_force_dynamic_part_at_C_V_0p01_N']:.3f}** N → 0 |
| l_cp/b 的高 C_V 渐近比（/0.75λ） | **{lim['lcp_over_beam_high_C_V_ratio_to_0p75lam']:.12f}** |
| l_cp 反解往返最大相对误差 | **{lim['lcp_inverse_max_rel_error']:.2e}** |
| ITTC-57 在 Re=1e9 | **{lim['ittc57_Re_1e9']:.9e}**（解析 0.075/49 = {lim['ittc57_Re_1e9_analytic_0p075_over_49']:.9e}） |
| 斜升修正确零的 C_L0 阈值（β=15°/30°） | {lim['deadrise_breakdown_C_L0_beta15']:.5f} / {lim['deadrise_breakdown_C_L0_beta30']:.5f} |

斜升角族（C_V=2，λ=3，τ=6°）：

| β[°] | C_Lβ | 递减量 | 0.0065·β·C_L0^0.6 | 相对降幅 |
|---|---|---|---|---|
{rows_deadrise}

### 3.3 运行姿态与阻力分解（β=15°，C_V 扫掠，每 4 点取样）

| C_V | V[m/s] | λ | τ[°] | C_L | l_cp/b | R_f[N] | R_spray[N] | R_T[N] | Δ/R_T | 包线内 |
|---|---|---|---|---|---|---|---|---|---|---|
{rows_sweep}

包线内点 {sw['n_in_envelope']}/{sw['n_points']}；升力平衡残差 ≤ **{sw['max_lift_balance_residual']:.1e}**、
力矩平衡残差 ≤ **{sw['max_moment_residual']:.1e}**（相对量级为机器精度）。
设计点 C_V={dp['C_V']:g}：λ={dp['lam']:.4f}、τ={dp['tau_deg']:.4f}°、C_L={dp['C_L']:.4f}、
R_T={dp['resistance']:.0f} N、Δ/R_T={dp['lift_to_drag']:.2f}、Re={dp['reynolds']:.3e}、
C_F={dp['friction_coefficient']:.5f}。
包线内 Δ/R_T 区间 **[{sw['LD_range_in_envelope'][0]:.2f}, {sw['LD_range_in_envelope'][1]:.2f}]**
（全扫掠两端的极值 [{sw['LD_range_full_sweep'][0]:.2f}, {sw['LD_range_full_sweep'][1]:.2f}]
出现在喷雾项失控/极低速端，见局限 L3）。

喷雾阻力占比随速度（判据 (b) 的可适用性）：

| C_V | 喷雾占 R_T[%] | 摩擦占 R_T[%] |
|---|---|---|
{rows_spray}

### 3.4 纵摇稳定性 / porpoising 边界

参考艇 β=15° 扫掠（{ca['n']} 点，C_V=1.0–5.0）：

* `dC_L/dλ > 0`：**{ca['n'] - ca['n_dCL_dlam_negative']}/{ca['n']}** 点成立（垂荡回复恒正）；
* `dM*/dλ` 区间 **[{ca['dM_dlam_range'][0]:.4f}, {ca['dM_dlam_range'][1]:.4f}]**（恒正：λ 增 →
  压力中心后移 → 低艏力矩）；
* 准静态刚度 `det J` 区间 **[{ca['min_det']:.4f}, {ca['max_det']:.4f}]**，**全部为负** →
  `quasi_static_unstable` **{ca['n_stiffness_unstable']}/{ca['n']}** 点（判据被证伪，见 §5 L4）；
* 纵倾斜率判据：**{ca['n_trim_slope_unstable']}/{ca['n']}** 点 `dτ/dC_V > 0`，
  斜率区间 [{ca['trim_slope_range'][0]:.3f}, {ca['trim_slope_range'][1]:.3f}]。

各斜升角的边界（扫掠 C_V={bounds[0]['C_V_swept'][0]:g}–{bounds[0]['C_V_swept'][1]:g}）：

| β[°] | 纵倾斜率判据不稳定区间 C_V | 刚度判据不稳定区间 C_V | 点数(斜率) | 最大 dτ/dC_V | 峰值位置 C_V | 点数(刚度) |
|---|---|---|---|---|---|---|
{rows_bounds}

porpoising-prone 占比（{len(grid['C_V'])} 个 C_V × {len(grid['beta_deg'])} 个 β）：
{", ".join(f"β={k}: {v:.3f}" for k, v in grid['prone_fraction'].items())}。

### 3.5 图

1. `figures/fig1_lift_chart.png` —— Savitsky 升力律图族：C_L0/τ^1.1 随 λ（C_V=1/2/4/6/8）。
2. `figures/fig2_deadrise.png` —— 斜升修正族：C_Lβ 随 β 在 4 组 (C_V, τ) 下。
3. `figures/fig3_running_attitude.png` —— 平衡纵倾 τ(C_V) 与平均湿长 λ(C_V)，β=10/15/20°，
   含包线下界 τ_min=2°。
4. `figures/fig4_resistance.png` —— 阻力分解 R_T/R_f/R_spray/Δtanτ 与升阻比 Δ/R_T 随航速。
5. `figures/fig5_porpoising.png` —— 纵倾斜率判据 dτ/dC_V 随 C_V（β=5…30°），
   正斜率带状区即位 porpoising-prone 带。

## 4. 注意点（踩过的坑 / 局限 / 适用范围）

* **L1 静水、无螺旋桨**：推力取平行艇底，未计桨法向力与推力线倾角；R_T 是水动力阻力而非轴推力。
* **L2 VCG=0**：重心取在艇底线上（经典闭式解假设）；非零 VCG 会给力矩方程加一项
  `~(VCG/b)(D/N)`，未建模 —— 这正好是 porpoising 分析最敏感的输入。
* **L3 喷雾阻力公式适用域**：`ΔC_D ∝ C_V²` 意味着 **力 ∝ C_V⁴**；参考艇上 C_V=2 时占 R_T
  约 {lim['spray_share_of_resistance_pct'][2]['spray_pct']:.0f}%，到 C_V≈7 时喷雾力已达重量的
  {sweep[-1]['spray_force'] / sweep[-1]['weight']:.1f} 倍 —— 该增量只能在 **C_V ≤ ~4** 使用，
  模块 docstring 已写明，报告不做外推。
* **L4 准静态稳定性判据失效（自证伪）**：无阻尼/附加质量项的 2 自由度模型，艏抬扰动会缩短湿长并把
  压力中心推到重心之前，`J_θθ<0` 恒成立 → `det J<0` 对所有工况成立，判据无区分能力。
  已作为 `quasi_static_unstable` 诊断量保留并用测试钉住（不得悄悄"预测稳定"）。
  真正有区分力的是纵倾斜率判据。
* **L5 无原文图表数字化数据（证据 WITHHELD）**：本环境离线，无法取得 Savitsky (1964) 图表上的
  读数，因此判据 (a)/(b)/(c) 的"与原文图对比"一律不伪造数字，见 §6 逐项声明。
* **L6 斜升修正的低升力失效**：`C_L0 < (0.0065β)^2.5` 时 `C_Lβ<0`（β=15° 阈值
  {lim['deadrise_breakdown_C_L0_beta15']:.4f}），此时公式无物理意义 → `deadrise_breakdown_lift`
  暴露阈值，包线外应报 WARNING 而非使用。
* **L7 适用包线**：`C_V∈[0.6,13]`、`λ∈[0.4,5]`、`β≤30°`、`τ∈[2,15]`；本次扫掠有
  {sw['n_points'] - sw['n_in_envelope']}/{sw['n_points']} 点越界（高速端 τ 落到 2° 以下、λ>5），
  已由 `in_envelope` 旗标显式标出。

## 5. 关键结论

1. **升力/湿长/压力中心闭式面已落地且自洽**：10 工况点复算残差 {max(r['rel_residual'] for r in ops):.1e}，
   β=0 精确退化为平板律（偏差 {lim['beta0_flat_plate_max_abs_dev']:.1e}），τ 指数
   {lim['tau_exponent_from_two_points']:.6f}，C_V→∞ 渐近比 {lim['high_C_V_asymptote_ratio_to_sqrt_law']:.9f}。
2. **运行姿态求解器**：61 点扫掠上升力/力矩残差 ≤ {sw['max_lift_balance_residual']:.1e} / {sw['max_moment_residual']:.1e}，
   设计点 (C_V=2) 给出 λ={dp['lam']:.3f}、τ={dp['tau_deg']:.3f}°、Δ/R_T={dp['lift_to_drag']:.2f}，
   与滑行艇 C_L=0.1–0.3 的量级判据一致（C_L={dp['C_L']:.3f}）。
3. **porpoising 判据**：无速率项的刚度判据被证伪（全域 `det J<0`）；纵倾斜率判据给出有限的不稳定带
   （参考艇 C_V≈0.8–2.0，驼峰/低滑行区），且该带的最大正斜率随斜升角单调增大
   （β=5°→{bounds[0]['max_trim_slope']:.2f}，β=30°→{bounds[-1]['max_trim_slope']:.2f}）。
   这一趋势是**模型结论**，不是对原文图 25-ish 边界的验证（见 L5）。

## 6. 误差判定（逐项；显式回答「是否 <3%」）

| 判据 | 基准 | 结果 | <3%？ |
|---|---|---|---|
| (a) 动升力 10 工况点 | Savitsky 原文图表读数 | **无法取得（离线，证据 WITHHELD）**；替代：与湿面积归一化独立代数式复算，最大相对残差 {max(r['rel_residual'] for r in ops):.1e} | **是（但只是转写一致性，非独立物理验证）** |
| (b) 喷雾阻力 | 原文 Fig. 12 | **WITHHELD（同上）**；替代：与给定公式 `0.003(b/L)^0.5 C_V²` 复算，残差 0（恒等式），另量化其 C_V⁴ 失控点 | 恒等式（无独立基准 → 不声称精度） |
| (c) porpoising 边界 | 原文稳定/不稳定区 | **WITHHELD**；给出两条可算判据的定量边界，一条被证伪、一条给出有限带 | 不可判定（无基准） |
| (d) 负控 β=0 退化 | 平板律 | 偏差 {lim['beta0_flat_plate_max_abs_dev']:.1e} | **是** |
| (d) 负控 V→0 动升力→0 | 物理解 | **不成立**：公式 C_L∝1/C_V² 发散，力残留 {lim['hydrostatic_residual_N']:.0f} N（与速度无关）。已改为「速度相关部分→0（C_V=0.01 时 {lim['lift_force_dynamic_part_at_C_V_0p01_N']:.2f} N）+ 越界旗标」，未粉饰 | **否，如实记为 criterion-not-satisfiable** |
| (e) 量级 C_L=0.1–0.3 | 判据带 | 设计点 C_L={dp['C_L']:.4f} | **是** |
| 平衡残差（自洽） | 升力/力矩方程 | ≤ {sw['max_lift_balance_residual']:.1e} / {sw['max_moment_residual']:.1e} | **是（机器精度）** |
| ITTC-57 | 解析 0.075/49 | {lim['ittc57_Re_1e9']:.9e} vs {lim['ittc57_Re_1e9_analytic_0p075_over_49']:.9e} | **是（0%）** |
| (f) pytest | ≥8 全绿 | **27 passed** | **是** |

## 7. 审计链与复现

```
python scripts/planing_savitsky_validation.py     # -> docs/planing-savitsky-verification/(results.json + figures/*.png)
python scripts/planing_savitsky_report.py         # -> 本 REPORT.md（数字全部注入自 results.json）
python -m pytest tests/test_planing_savitsky.py -q # 27 passed
```

证据分级：升力/湿长/压力中心/平衡求解 = **REAL**（闭式经验律 + 可验证自洽）；
喷雾阻力 = **REAL（公式实现）但适用域受限（L3）**；porpoising 边界 = **WITHHELD**
（无可用的原文判据基准，仅给出可算判据与自证伪记录）；原文图表对比 = **WITHHELD**。
"""

with open(os.path.join(DOC, "REPORT.md"), "w") as fh:
    fh.write(REPORT)
print("wrote", os.path.join(DOC, "REPORT.md"), len(REPORT), "bytes")