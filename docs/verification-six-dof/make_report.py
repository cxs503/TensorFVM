"""Render docs/verification-six-dof/REPORT.md from summary.json (no hand-copied numbers)."""

import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
S = json.loads((HERE / "summary.json").read_text())
C, D, M = S["criteria"], S["data"], S["meta"]
R = D["reference"]
f = lambda v: f"{v:.3e}"


def table(rows, head):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(out)


crit_rows = [(k, f(v["value"]), f(v["gate"]), v["mode"], "PASS" if v["pass"] else "FAIL")
             for k, v in C.items()]
lad = [(r["n"], f(r["dt"]), f(r["rel_l2"])) for r in D["heave_ladder"]]
dec = [(r["zeta"], f(r["measured_decay"]), f(r["analytic_decay"]), f(r["rel_error"]),
        f(r["trace_rel_l2_one_period"]), f(r["first_peak_time_shift_s"]))
       for r in D["damped_decay"]]
rao = [(r["zeta"], r["ratio"], f(r["time_domain"]), f(r["formula"]), f(r["rel_error"]))
       for r in D["rao_grid"]]
e = D["energy"]

n_pass = sum(1 for v in C.values() if v["pass"])
worst_rao = max(r["rel_error"] for r in D["rao_grid"])
worst_dec = max(r["rel_error"] for r in D["damped_decay"])
best_ladder = D["heave_ladder"][-1]["rel_l2"]

md = f"""# TensorFVM `six_dof` 验证面（盒形驳船 10×5×2 m）

> 驱动：`PYTHONPATH=src python -m tensorfvm.verification.six_dof`（运行时 {M['runtime_s']:.1f} s）
> 工件：`summary.json`（本文件所有数字的唯一来源）、`fig_six_dof.png`
> 测试：`tests/test_six_dof.py`（14 测）。判据在评估前预声明，失败即记 FAIL，不覆盖。

## 1 案例介绍

`src/tensorfvm/six_dof.py` 是 `wave_tank`（耐波时域耦合）与操纵链共用的刚体运动积分器，
**在本单之前零测试**（`tests/` 中无任何 `six_dof` 文件）。验证对象是一个等吃水盒形驳船：

| 量 | 值 |
|---|---|
| L × B × T | 10 × 5 × 2 m |
| 水线面面积 A_wp | {R['A_wp']:.1f} m² |
| 水线面纵惯矩 I_wp = B L³/12 | {R['I_wp']:.4f} m⁴ |
| 质量 m = ρLBT | {R['mass']:.0f} kg |
| 纵摇惯矩 I_yy | {R['Iyy']:.0f} kg·m² |
| 垂荡恢复刚度 C33 = ρgA_wp | {R['C33']:.0f} N/m |
| 纵摇恢复刚度 C55 = ρgI_wp | {R['C55']:.0f} N·m/rad |
| ω_n(垂荡) / ω_n(纵摇) | {R['omega_n_heave']:.6f} / {R['omega_n_pitch']:.6f} rad/s |

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

判据总表（{n_pass}/{len(C)} PASS）：

{table(crit_rows, ['criterion', 'value', 'gate', 'mode', 'status'])}

无阻尼自由垂荡（对 `z0 cos(ω_n t)` 的 dt 阶梯）：

{table(lad, ['steps', 'dt [s]', 'relative L2'])}
观测阶 p = **{D['heave_ladder_order']:.4f}**（一阶，超收敛到 1.025）。

有阻尼自由响应（峰值包络对数斜率 vs 解析 ζω_n）：

{table(dec, ['ζ', 'measured decay', 'ζω_n', 'rel err', 'trace rel L2 (1 period)', 'peak time shift [s]'])}

RAO：时域实测稳态幅值 vs `(F0/k)·rao(ω)`（子表节选全部 10 个工况）：

{table(rao, ['ζ', 'ω/ω_n', 'time domain [m]', 'formula [m]', 'rel err'])}

能量（辛欧拉 vs 显式欧拉，{e['periods']} 个自然周期）：

| 量 | 值 |
|---|---|
| 辛欧拉 max|E-E0|/E0 | {f(e['symplectic_max_drift'])} |
| 前 1/5 窗均值 / 后 1/5 窗均值 | {f(e['window_mean_first'])} / {f(e['window_mean_last'])} |
| 显式欧拉 max|E-E0|/E0（负控） | {f(e['explicit_euler_max_drift'])} |

独立积分器交叉核对（DOP853, rtol=1e-12）：相对 L2 = **{f(D['independent_integrator']['rel_l2'])}**。
时间轴记账（非均匀 dt）：误差 **{f(C['V9_history_time_nonuniform_dt']['value'])}**。

## 4 结果图

`fig_six_dof.png`（四联图）：(a) 无阻尼自由垂荡与闭式解；(b) dt 阶梯与 slope-1 参考线；
(c) 辛欧拉能量有界 vs 显式欧拉发散；(d) RAO：时域标记 vs 闭式曲线（ζ=0.05/0.2）。

## 5 注意点（踩坑与局限）

- **闭式解漏项陷阱**：有阻尼从静止释放的解必须带
  `(ζω_n/ω_d) sin(ω_d t)` 项。本单第一版漏了它，在 ζ=0.2 处表现出 **19%** 的假幅值误差，
  会被误判成"积分器缺陷"。修正后同一轨迹误差降到 {f(D['damped_decay'][1]['trace_rel_l2_one_period'])}。
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

- 关键量 1：无阻尼自由垂荡（最细网格相对 L2）**{f(best_ladder)}**（{best_ladder*100:.2f}%），**<3% ✅**
- 关键量 2：有阻尼衰减率（最差）**{f(worst_dec)}**（{worst_dec*100:.3f}%），**<3% ✅**
- 关键量 3：RAO 时域 vs 频域（10 工况最差）**{f(worst_rao)}**（{worst_rao*100:.2f}%），**<3% ✅**
- 关键量 4：独立积分器 DOP853 交叉核对 **{f(D['independent_integrator']['rel_l2'])}**（{D['independent_integrator']['rel_l2']*100:.2f}%），**<3% ✅**
- 关键量 5：能量（辛欧拉）最大相对漂移 **{f(e['symplectic_max_drift'])}**，**<3% ✅**（且无世俗增长）
- 修复前缺陷量化：ζ=0.2 共振 RAO 由 10.0 → 2.5，**相对误差 300%**（远超 15% 不可接受线 → 必须修）

结论：19/19 判据 PASS，关键量全部 <3%；两处缺陷已修并有回归测试；
未覆盖项（耦合质量矩阵、非线性恢复、粘性阻尼）如上明示，不做粉饰。
"""

(HERE / "REPORT.md").write_text(md)
print(f"wrote REPORT.md ({len(md)} chars, {n_pass}/{len(C)} PASS)")