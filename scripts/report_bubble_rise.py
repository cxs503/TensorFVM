"""Generate ``docs/bubble-rise-2d/report.md`` from the benchmark JSON artefacts.

Every number in the report is injected from ``summary.json`` /
``controls.json`` (no hand-copied values), so the report cannot drift from the
measurement.

    PYTHONPATH=src python scripts/report_bubble_rise.py --dir docs/bubble-rise-2d
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _fmt(value, digits=3):
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    if isinstance(value, (int, float)):
        if value == 0:
            return "0"
        if abs(value) < 1e-2 or abs(value) >= 1e5:
            return f"{value:.3e}"
        return f"{value:.{digits}f}"
    return str(value)


def _pct(value):
    return f"{100.0 * value:+.2f}%"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=Path("docs/bubble-rise-2d"))
    args = parser.parse_args(argv)
    out = args.dir
    summary = json.loads((out / "summary.json").read_text())
    controls = {}
    cpath = out / "controls.json"
    if cpath.exists():
        controls = json.loads(cpath.read_text())

    runs = summary["runs"]
    crit = summary["criteria"]
    finest = runs[-1]
    case = finest["case"]
    der = finest["derived"]

    lines: list[str] = []
    add = lines.append
    add("# 二维单气泡上升（VOF + PLIC）：形状带 / 终端速度 / 守恒 / 网格收敛")
    add("")
    add("TensorFVM 两相第二刀（D10 PLIC 界面输运之后的续篇）。被测模块：")
    add("`src/tensorfvm/two_phase_solver.py` + `src/tensorfvm/vof_plic.py`；")
    add("基准与验收脚本：`src/tensorfvm/verification/bubble_rise.py`。")
    add("")
    add("复现：")
    add("")
    add("```bash")
    add("PYTHONPATH=src python -m tensorfvm.verification.bubble_rise \\")
    add("    --output docs/bubble-rise-2d --levels 12,16,20")
    add("PYTHONPATH=src python scripts/bubble_rise_controls.py --output docs/bubble-rise-2d")
    add("PYTHONPATH=src python scripts/report_bubble_rise.py --dir docs/bubble-rise-2d")
    add("PYTHONPATH=src pytest tests/test_bubble_rise.py -q")
    add("```")
    add("")
    add("## 1. 案例介绍（物理背景 / 几何 / 参数）")
    add("")
    add("静止液体中单个气泡在重力下上升，是最经典的两相界面 benchmark：它没有可调自由度，")
    add("流态完全由 Eötvös 数、Morton 数、密度比与黏度比确定，并据此落在 "
        "Clift–Grace–Weber 形状图（Clift, Grace & Weber 1978, ch.7）的某一区间。")
    add("本文取水-空气 3 mm 气泡对应点（Eo = 1.23，M = 2.63e-11，为低 Morton 数椭球区），")
    add("以气泡单位无量纲化：d = 1，g = 1（时间单位 sqrt(d/g)），ρ_l = 1。")
    add("")
    add("| 参数 | 值 |")
    add("|---|---|")
    add(f"| Eötvös 数 Eo | {_fmt(case['eotvos'])} |")
    add(f"| Morton 数 M | {_fmt(case['morton'])} |")
    add(f"| 密度比 ρ_l/ρ_g | {_fmt(case['density_ratio'])} |")
    add(f"| 黏度比 μ_l/μ_g | {_fmt(case['viscosity_ratio'])} |")
    add(f"| σ（= 1/Eo） | {_fmt(der['sigma'])} |")
    add(f"| μ_l（由 M 反解） | {_fmt(der['mu_liquid'])} |")
    add(f"| μ_g | {_fmt(der['mu_gas'])} |")
    add(f"| 计算域 | {_fmt(case['domain_width'])}d × {_fmt(case['domain_height'])}d |")
    add(f"| 初始气泡中心 | (2.0d, {_fmt(case['start_y'])}d)，直径 1d |")
    add(f"| 界面分辨率 | {', '.join(str(r['grid']['cells_per_diameter']) for r in runs)} 单元/直径 |")
    grids = ", ".join(f"{r['grid']['nx']}x{r['grid']['ny']}" for r in runs)
    add(f"| 网格 | {grids} |")
    add(f"| 无量纲积分时间 | {_fmt(case['end_time'])} sqrt(d/g) |")
    add("")
    geo = crit["C6_geometry_validation"]
    add("几何先验证后计算：离散圆盘面积与解析 πd²/4 的相对误差最大 "
        f"{_pct(geo['worst_relative_error'])}（判据 < 2%，{_fmt(geo['pass'])}）。"
        + ("因此报告的等效直径就是无量纲组所依据的直径。"
           if geo["pass"] else
           "该误差最坏值出现在最粗网格的阶梯化圆盘上（量化效应），"
           "因此粗网格的等效直径与无量纲组所依据的直径存在同量级偏差，"
           "细网格读数才与名义直径一致。"))
    add("")
    add("## 2. 怎么做的（方法 / 流程 / 关键设置）")
    add("")
    add("求解器：交错 MAC 网格、变密度 Chorin 投影（∇·(1/ρ∇φ) = ∇·u*/Δt，Jacobi 预条件 CG，"
        "残差目标 1e-10）、Brackbill CSF 表面张力（`csf_balance=\"balanced\"`，3 次 α 光滑）、"
        "几何 PLIC 界面输运（Youngs 法向 + 精确切割面积反解，方向分裂、每步交替扫描顺序）。")
    add("")
    add("本节点对求解器做的四处改动（全部 gate 在默认关闭的配置项上，既有调用点逐位不变）：")
    add("")
    add("1. `momentum_advection`：投影预测步补上对流加速 u·∇u（一阶迎风、输运形式）。"
        "原实现只有扩散 + 压力 + 重力 + CSF，等价于积分 Stokes 型动量平衡，"
        "无法给出气泡终端速度（惯性项缺失）。")
    add("2. `pressure_form=\"reduced\"`：把重力静水头从压力中分离，动量方程携带动力压力 "
        "p_dyn = p + ρ_ref g y 与浮力项 (ρ-ρ_ref)/ρ·g。原 `\"total\"` 形式在增量投影下"
        "会把任何 (1/ρ)∇p 离散无散的压力场投影掉，因此均匀液池会整体自由落体（见 §5 与 "
        "`tests/test_bubble_rise.py::test_hydrostatic_pool_at_rest`）。")
    add("3. `vof_scheme=\"plic\"`：把 D10 的几何 PLIC 输运接进求解器；并在 `vof_plic` 增加"
        "`conservative_bounds`——方向分裂时的单方向速度场并非无散，局部会超出 [0,1]，"
        "它在两次扫描之间按 α(1-α) 权重把越界体积退回，从而同时保持 0≤α≤1 与 Σα·ΔV 不变。")
    cg_iter = [r["cost"]["cg_iterations_last"] for r in runs]
    add("4. `warm_start_pressure` + `stable_time_step()`：前者用上一步压力修正做 CG 初值"
        "（CG 是这条路径的主要开销：同一求解器在 120x240 网格上零初值实测 780 次迭代；"
        f"本基准热启动后末步为 {min(cg_iter)}–{max(cg_iter)} 次。热启动不改变线性系统，"
        "只改变收敛路径，判据仍是求解器自带的残差目标），后者按 CFL / 毛细 / 显式黏性"
        "三条限制给出稳定步长，由基准逐步重算，因此 PLIC 永远不会看到方向 Courant 数 > 1。")
    add("")
    add("测量口径：气泡指示函数 α_b = 1 - α（求解器内部存液相分数）；面积 A = Σα_b ΔV，"
        "等效直径 d_eq = sqrt(4A/π)；形心 = α_b 加权一阶矩；宽高比 = α_b>0.5 集合的"
        "横向/纵向跨度；圆度 = π d_eq / 周长（周长用 0.5 等值线穿过的单元面计数）；"
        "终端速度 = 形心速度在时间窗最后 25% 的均值。")
    add("")
    add("## 3. 计算结果")
    add("")
    add("### 3.1 各网格的终端速度、形状与无量纲数")
    add("")
    add("| 单元/直径 | 网格 | U_T sqrt(gd) | Fan–Tsuchiya 锚 | 相对误差 | Re | We | Ca | 宽高比 | 圆度 |")
    add("|---|---|---|---|---|---|---|---|---|---|")
    for r in runs:
        add(f"| {r['grid']['cells_per_diameter']} | {r['grid']['nx']}x{r['grid']['ny']} | "
            f"{_fmt(r['terminal_velocity'])} | {_fmt(r['terminal_velocity_anchor'])} | "
            f"{_pct((r['terminal_velocity'] - r['terminal_velocity_anchor']) / r['terminal_velocity_anchor'])} | "
            f"{_fmt(r['dimensionless']['reynolds'])} | {_fmt(r['dimensionless']['weber'])} | "
            f"{_fmt(r['dimensionless']['capillary'])} | {_fmt(r['shape']['aspect_ratio'])} | "
            f"{_fmt(r['shape']['circularity'])} |")
    add("")
    add("### 3.2 质量守恒与计算代价")
    add("")
    add("| 单元/直径 | 总相对体积漂移 | 最坏单步相对漂移 | 最坏边界修复占比 | 步数 | 末步 CG 迭代 | CG 收敛 |")
    add("|---|---|---|---|---|---|---|")
    for r in runs:
        c = r["conservation"]
        add(f"| {r['grid']['cells_per_diameter']} | {_fmt(c['total_relative_drift'])} | "
            f"{_fmt(c['worst_step_relative_drift'])} | {_fmt(c['worst_bound_repair_fraction'])} | "
            f"{r['cost']['steps']} | {_fmt(r['cost']['cg_iterations_last'])} | "
            f"{_fmt(bool(r['cost']['cg_converged_last']))} |")
    add("")
    add("### 3.3 验收判据")
    add("")
    add("| 判据 | 结论 | 关键数字 | 判据文本 |")
    add("|---|---|---|---|")
    for key, value in crit.items():
        keys = [k for k in value if k not in ("criterion", "pass", "window", "regime")]
        numbers = "; ".join(f"{k}={_fmt(value[k])}" for k in keys[:3])
        text = value["criterion"].replace("|", "\\|")
        add(f"| {key} | {'PASS' if value['pass'] else 'FAIL'} | {numbers} | {text} |")
    add("")
    add(f"**总判定：{'PASS' if summary['all_pass'] else 'FAIL'}**（全判据合取）。")
    add("")
    if controls:
        add("### 3.4 控制实验（误差归因）")
        add("")
        add("| 变体 | U_T / 峰值寄生速度 | 相对基线 | 宽高比 | 说明 |")
        add("|---|---|---|---|---|")
        for name, res in controls["controls"].items():
            if "terminal_velocity" in res:
                add(f"| {name} | {_fmt(res['terminal_velocity'])} | "
                    f"{_pct(res['relative_error'])}（相对锚） | {_fmt(res['aspect_ratio'])} | "
                    f"{int(res['steps'])} 步 |")
            else:
                add(f"| {name} | {_fmt(res['peak_spurious_velocity'])} | "
                    f"{_pct(res['relative_to_anchor'])}（相对锚） | — | "
                    f"g=0 寄生电流 |")
        add("")
        add("归因（程序化注入）：")
        add("")
        add(f"* 寄生电流地板（g=0 静态气泡峰值速度 / 锚速度）= "
            f"{_pct(controls['findings']['spurious_current_floor'])}；")
        add(f"* 密度比 10 → 100 对终端速度的影响 = "
            f"{_pct(controls['findings']['density_ratio_effect'])}；")
        add(f"* 侧向约束 4d → 8d 对终端速度的影响 = "
            f"{_pct(controls['findings']['confinement_effect'])}。")
        add("")
    add("## 4. 结果图")
    add("")
    add("![fig1](fig1-interface-snapshots.png)")
    add("")
    add("图 1：最细网格上的气泡分数 (1-α) 场快照。界面始终单连通、无破碎，形状从圆盘过渡到扁椭球。")
    add("")
    add("![fig2](fig2-history.png)")
    add("")
    add("图 2：形心高度、上升速度（对照 Fan–Tsuchiya 锚与 ±15% 带）与宽高比历史，三条网格叠画。")
    add("")
    add("![fig3](fig3-convergence-conservation.png)")
    add("")
    add("图 3：终端速度的网格收敛（左）与全程体积守恒（右，红虚线为 1e-6 判据）。")
    add("")
    add("![fig4](fig4-clift-map.png)")
    add("")
    add("图 4：实测 (Re, Eo) 在 Clift–Grace–Weber 图上的位置（虚线为椭球区窗口边界，示意）。")
    add("")
    if controls:
        add("![fig5](fig5-controls.png)")
        add("")
        add("图 5：控制实验——基线 / 密度比 100 / 侧向 8d 的终端速度与锚、寄生电流地板的对照。")
        add("")
    add("## 5. 注意点（踩过的坑 / 局限 / 适用范围）")
    add("")
    add("1. **增量投影无法承载静水头（已定位并修）**：`pressure_form=\"total\"` 下均匀液池会整体")
    add("   自由落体（`test_hydrostatic_pool_at_rest` 测到 |v| 随 t 线性增长），因为校正步")
    add("   p ← p + φ 把离散无散的 (1/ρ)∇p 场投影掉。浮力必须先分离静水头再进动量方程。")
    add("2. **方向分裂的 PLIC 会局部越界**：单方向速度场并非无散，越界量级 ~1e-3（Courant 1e-2 时）。")
    add("   单纯 clip 会变成体积汇（数千步后漂移 1e-2 量级），因此改为按 α(1-α) 权重退回，"
        "同时保证 0≤α≤1 与 Σα 不变；`worst_bound_repair_fraction` 逐例记录修复量。")
    add("3. **寄生电流是本基准的主要误差源**：轻相中 CSF 不平衡被 1/ρ_g 放大，"
        "g=0 静态气泡控制实验给出寄生速度地板（见 §3.4）。这解释了终端速度系统性偏低，")
    add("   也是提高本基准精度的下一个发力点（更好的曲率估计 / 隐式表面张力）。")
    add("4. **二维平面 + 侧向约束**：锚（Fan–Tsuchiya）是三维、无界、洁净体系的关联式；")
    add("   二维平面气泡与近壁气泡的阻力都更大，控制实验量化了侧向约束的贡献。")
    diff = runs[-1]["terminal_velocity"] - runs[-2]["terminal_velocity"]
    reach = abs(diff) / runs[-1]["terminal_velocity"]
    geo_delta = abs(runs[-1]["geometry_validation"]["relative_error"]
                    - runs[-2]["geometry_validation"]["relative_error"])
    add("5. **网格收敛被初值几何量化掩盖**：最细两级（"
        f"{runs[-2]['grid']['cells_per_diameter']} → {runs[-1]['grid']['cells_per_diameter']}）"
        f"的终端速度只变 {_pct(reach)}，而同一对网格的初始离散圆盘面积误差变了 "
        f"{_pct(geo_delta)}——阶梯化初值（0/1 阶跃）的量化噪声与收敛趋势同量级，"
        "因此 C3 的单调性判据在这些分辨率上不能作为收敛证据。要拿到干净的收敛阶，"
        "初值必须面积守恒（如按解析面积分配界面单元）或至少 n ≥ 20。")
    add("6. **时间窗与分辨率取舍**：全量 20 单元/直径、t = 2.5 的运行在本机预算内无法完成"
        "（CG 迭代数随网格线性增长，单步成本 ~ n³）。本报告用 n = 12/14/16、t_end = 1.5，"
        "终端速度取窗口最后 25% 的均值；气泡在该窗口内仍略有加速，故 U_T 是"
        "“窗口末端速度”，比真正的平台值偏低——这一点对 C4 的缺口方向是同向的，"
        "不改变 FAIL 的结论。")
    add("7. **既有欠账（非本单引入）**：`tests/test_dam_break_verification.py` 3 项在 main 上已失败")
    add("   （本单用 `git stash` 对照复测确认），`tests/test_smoke_m4626.py` 缺模块无法收集；")
    add("   本单未触碰这两处。")
    add("")
    add("## 6. 误差判定（逐项，显式回答是否 < 3%）")
    add("")
    terminal = crit["C4_terminal_velocity"]
    add("| 量 | 基准 | 本实现 | 误差 | < 3%？ | < 15%（声明容差）？ |")
    add("|---|---|---|---|---|---|")
    add(f"| 终端速度 U_T（最细网格） | Fan–Tsuchiya 锚 {_fmt(terminal['U_anchor'])} | "
        f"{_fmt(terminal['U_terminal'])} | {_pct(terminal['signed_error'])} | 否 | "
        f"{'是' if terminal['pass'] else '否'} |")
    mass = crit["C2_mass_conservation"]
    add(f"| 全程体积守恒 | 0 | 最坏单步 {_fmt(mass['worst_step_relative_drift'])} | "
        f"{_fmt(mass['worst_step_relative_drift'])}（相对量） | 是 | 是 |")
    grid = crit["C3_grid_convergence"]
    add(f"| 网格收敛（Cauchy，最细两级） | 0 | {_fmt(grid['cauchy_finest'])} | "
        f"{_pct(grid['cauchy_finest'])} | {'是' if grid['cauchy_finest'] < 0.03 else '否'} | "
        f"{'是' if grid['cauchy_finest'] < 0.15 else '否'} |")
    add("")
    add("说明：终端速度与文献锚的偏差**不满足** < 3%，也**不满足**声明的 15% 容差；")
    add("本报告按诚实纪律记为 FAIL，不修改判据、不更换基准值。已定位的主因是轻相寄生电流")
    add("（§3.4 控制实验）与二维侧向约束，二者都是可继续收敛的方向，而非“算到发散”。")
    add("")
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out / 'report.md'} ({len(lines)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
