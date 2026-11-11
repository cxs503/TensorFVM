"""CASE-CYLINDER-DRAG-FVM: run and judge the DFG 2D-1 cylinder drag benchmark.

The case reuses the solver capability that already lives in this repository
(``tensorfvm.verification.dfg``: body-fitted SIMPLE/Rhie-Chow laminar solver on
the DFG channel-with-cylinder geometry, polygon wall traction, extrapolated wall
pressure).  No new solver is introduced here; this script adds the *case-level*
harness required by the benchmark-case pool:

  solve    run one or more grids exactly as the package verification entry point
           does (``python -m tensorfvm.verification.dfg``) so the artifacts are
           bit-for-bit the package's own evidence format;
  analyze  compare against the machine-parsed reference
           (``artifacts/case-cylinder-drag-fvm/reference/reference.json``),
           judge every quantity against the case tolerance (default 5 %),
           compute grid-refinement order / Richardson estimate, alternative
           pressure-sampling sensitivity and force decomposition, and emit
           tables + figures.

Usage
  python scripts/case_cylinder_drag_fvm.py solve   --grids 64x24 128x48 256x96 --out runs/
  python scripts/case_cylinder_drag_fvm.py analyze --runs runs/ --out analysis/ \\
         --reference artifacts/case-cylinder-drag-fvm/reference/reference.json

Exit code: 0 = case passed at the finest grid, 2 = tolerance not met.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

QUANTITIES = {  # reference key -> (symbol, unit, role)
    "drag": ("C_D", "-", "primary"),
    "lift": ("C_L", "-", "secondary"),
    "pressure_difference_pa": ("p_diff", "Pa", "secondary"),
}


def solve(args) -> int:
    from tensorfvm.verification import dfg
    argv = ["--output", str(args.out), "--grids", *args.grids,
            "--max-iterations", str(args.max_iterations)]
    return dfg.main(argv)


def load_run(directory: Path) -> dict:
    result = json.loads((directory / "result.json").read_text())
    result["directory"] = str(directory)
    return result


def probe_pressure(fields, point) -> float:
    """Inverse-distance (1/r^2) interpolation of cell pressure at a probe point."""
    import numpy as np
    centers = fields["cell_centers_m"]
    pressure = fields["cell_pressure_pa"]
    d2 = ((centers - np.asarray(point)) ** 2).sum(1)
    weight = 1.0 / np.maximum(d2, 1e-30)
    return float((weight * pressure).sum() / weight.sum())


def force_split(fields) -> dict:
    """Drag split from the stored polygon-wall traction (N/m, per unit depth)."""
    import numpy as np
    pressure = fields["cylinder_pressure_force_n_m"].sum(0)
    viscous = fields["cylinder_viscous_force_n_m"].sum(0)
    scale = float(fields["force_scale_n_m"])
    return {"pressure": float(pressure[0] / scale), "viscous": float(viscous[0] / scale),
            "total": float((pressure[0] + viscous[0]) / scale),
            "lift_total": float((pressure[1] + viscous[1]) / scale)}


def richardson(runs: list[dict], key: str, ratio: float = 2.0) -> dict:
    """Observed order p and Richardson estimate from the three finest grids."""
    usable = [r for r in runs if key in r["computed"]]
    if len(usable) < 3:
        return {"available": False, "reason": "needs three grids"}
    coarse, mid, fine = usable[-3], usable[-2], usable[-1]
    e_coarse_mid = mid["computed"][key] - coarse["computed"][key]
    e_mid_fine = fine["computed"][key] - mid["computed"][key]
    if e_mid_fine == 0 or e_coarse_mid / e_mid_fine <= 0:
        return {"available": False,
                "reason": "non-monotone or exactly converged differences",
                "differences": [e_coarse_mid, e_mid_fine]}
    order = math.log(abs(e_coarse_mid / e_mid_fine)) / math.log(ratio)
    extrapolated = fine["computed"][key] + (fine["computed"][key] - mid["computed"][key]) / (
        ratio ** order - 1.0)
    gci = 1.25 * abs((fine["computed"][key] - mid["computed"][key]) / mid["computed"][key]) / (
        ratio ** order - 1.0)
    return {"available": True, "observed_order": order,
            "richardson_extrapolated": extrapolated, "gci_fine": gci,
            "differences": [e_coarse_mid, e_mid_fine]}


def figures(runs: list[dict], reference: dict, tolerance: float, out: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out.mkdir(parents=True, exist_ok=True)
    made, works = [], [r for r in runs if r["computed"]]
    if not works:
        return made
    symbol = {k: v[0] for k, v in QUANTITIES.items()}

    # 1. Drag coefficient under grid refinement against the reference line.
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    h = [r["spacing_m"] for r in works]
    cd = [r["computed"]["drag"] for r in works]
    ax.plot(h, cd, "o-", label="TensorFVM (body-fitted SIMPLE)")
    ax.axhline(reference["drag"], color="k", ls="--", label="reference $C_D$")
    for x, y, r in zip(h, cd, works):
        ax.annotate(f"{r['resolution']}\n{100 * (y - reference['drag']) / reference['drag']:+.3f}%",
                    (x, y), textcoords="offset points", xytext=(8, 6), fontsize=7)
    ax.set_xscale("log")
    ax.set(xlabel="cell spacing $h$ (m)", ylabel="$C_D$",
           title="DFG 2D-1 cylinder drag under grid refinement")
    ax.grid(alpha=.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "cd_refinement.png", dpi=160)
    plt.close(fig)
    made.append("cd_refinement.png")

    # 2. Relative error per quantity with the tolerance band.
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    labels = [r["resolution"] for r in works]
    x = np.arange(len(works))
    width = 0.8 / len(QUANTITIES)
    for i, key in enumerate(QUANTITIES):
        errors = [100 * (r["computed"][key] - reference[key]) / reference[key] for r in works]
        ax.bar(x + i * width, errors, width, label=symbol[key])
    ax.axhline(tolerance * 100, color="r", ls="--", label=f"+{tolerance:.0%} gate")
    ax.axhline(-tolerance * 100, color="r", ls="--")
    ax.axhline(0, color="k", lw=.8)
    ax.set_xticks(x + width * (len(QUANTITIES) - 1) / 2, labels)
    ax.set(ylabel="relative error vs reference (%)", title="Quantity-by-quantity error vs DFG 2D-1 reference")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(out / "error_bars.png", dpi=160)
    plt.close(fig)
    made.append("error_bars.png")

    # 3. Pressure/viscous drag split of the finest grid.
    finest = works[-1]
    fields = np.load(Path(finest["directory"]) / "fields.npz", allow_pickle=False)
    split = force_split(fields)
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ax.bar(["pressure", "viscous", "total"], [split["pressure"], split["viscous"], split["total"]],
           color=["tab:orange", "tab:blue", "tab:green"])
    ax.axhline(reference["drag"], color="k", ls="--", label="reference $C_D$")
    ax.set(ylabel="$C_D$ contribution", title=f"Drag decomposition, grid {finest['resolution']}")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(out / "drag_decomposition.png", dpi=160)
    plt.close(fig)
    made.append("drag_decomposition.png")

    # 4. Fields of the finest grid (pressure and speed).
    from scipy.interpolate import griddata
    centers = fields["cell_centers_m"]
    xi = np.linspace(0.0, 2.2, 440)
    yi = np.linspace(0.0, 0.41, 84)
    XI, YI = np.meshgrid(xi, yi)
    speed = np.sqrt((fields["cell_velocity_m_s"] ** 2).sum(1))
    for name, values, unit, cmap in (
            ("field_pressure.png", fields["cell_pressure_pa"], "p (Pa)", "coolwarm"),
            ("field_speed.png", speed, "|u| (m/s)", "viridis")):
        grid = griddata(centers, values, (XI, YI), method="linear")
        fig, ax = plt.subplots(figsize=(7.6, 2.4))
        contour = ax.contourf(XI, YI, grid, levels=40, cmap=cmap)
        circle = plt.Circle((0.2, 0.2), 0.05, color="w", ec="k", zorder=3)
        ax.add_patch(circle)
        fig.colorbar(contour, ax=ax, label=unit)
        ax.set(xlabel="x (m)", ylabel="y (m)",
               title=f"DFG 2D-1 {unit} on grid {finest['resolution']}")
        ax.set_aspect("equal")
        fig.tight_layout()
        fig.savefig(out / name, dpi=160)
        plt.close(fig)
        made.append(name)

    # 5. Wall pressure coefficient around the cylinder.
    angle = fields["cylinder_angle_rad"]
    wall = fields["cylinder_wall_pressure_pa"]
    order = np.argsort(angle)
    cp = (wall - wall.mean()) / (0.5 * 1.0 * 0.2 ** 2)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(np.degrees(angle[order]), cp[order], "o-", ms=2.5)
    ax.set(xlabel="angle around cylinder (deg, 0 = front stagnation side)",
           ylabel="$C_p$ (mean-referenced)", title="Cylinder wall pressure, finest grid")
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out / "wall_pressure.png", dpi=160)
    plt.close(fig)
    made.append("wall_pressure.png")

    # 6. Residual history.
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for r in works:
        history = json.loads((Path(r["directory"]) / "history.json").read_text())
        ax.semilogy([q["momentum"] for q in history], label=r["resolution"])
    ax.set(xlabel="SIMPLE iteration", ylabel="momentum residual", title="Convergence history")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3, which="both")
    fig.tight_layout()
    fig.savefig(out / "residuals.png", dpi=160)
    plt.close(fig)
    made.append("residuals.png")
    return made


def analyze(args) -> int:
    import numpy as np

    reference = json.loads(Path(args.reference).read_text())
    ref = reference["values"]
    runs = []
    for directory in sorted(p for p in Path(args.runs).iterdir() if (p / "result.json").exists()):
        run = load_run(directory)
        fields = np.load(directory / "fields.npz", allow_pickle=False)
        run["dp_nearest_cell"] = probe_pressure(fields, (0.15, 0.2)) - probe_pressure(
            fields, (0.25, 0.2))
        run["force_split"] = force_split(fields)
        runs.append(run)
    if not runs:
        raise SystemExit("no run directories with result.json found")
    runs.sort(key=lambda r: -r["spacing_m"])
    for run in runs:
        print("loaded", run["directory"])

    table = []
    for run in runs:
        row = {"resolution": run["resolution"], "cells": None, "spacing_m": run["spacing_m"],
               "iterations": run["iterations"], "converged": run["converged"],
               "elapsed_s": run["elapsed_s"], "computed": {}, "error_pct": {},
               "passed": {}, "dp_nearest_cell_pa": run["dp_nearest_cell"],
               "force_split": run["force_split"]}
        for key in QUANTITIES:
            value = run["computed"][key]
            error = (value - ref[key]) / ref[key]
            row["computed"][key] = value
            row["error_pct"][key] = 100 * error
            row["passed"][key] = abs(error) <= args.tolerance
        row["case_passed"] = all(row["passed"].values()) and bool(run["converged"])
        table.append(row)

    finest = table[-1]
    convergence = {key: richardson(table, key) for key in QUANTITIES}
    sensitivity = {
        "grid_refinement": {key: convergence[key] for key in QUANTITIES},
        "pressure_probe_interpolation": {
            "dfg_wall_extrapolation_pa": finest["computed"]["pressure_difference_pa"],
            "inverse_distance_cell_centre_pa": finest["dp_nearest_cell_pa"],
            "delta_pa": finest["dp_nearest_cell_pa"] - finest["computed"]["pressure_difference_pa"],
            "relative_to_reference_pct": 100 * (finest["dp_nearest_cell_pa"] - ref[
                "pressure_difference_pa"]) / ref["pressure_difference_pa"],
            "note": "the official metric uses least-squares wall extrapolation + periodic angle "
                    "interpolation; the crude probe shows how much of the residual is sampling "
                    "rather than flow error"},
        "force_decomposition_finest": finest["force_split"],
        "iteration_counts": {r["resolution"]: [r["iterations"], r["converged"]] for r in table},
    }
    summary = {
        "case": "CASE-CYLINDER-DRAG-FVM",
        "reference": reference,
        "tolerance": args.tolerance,
        "grids": table,
        "convergence": convergence,
        "sensitivity": sensitivity,
        "case_passed": finest["case_passed"],
        "verdict_scope": ("case pass = every quantity within tolerance on the finest computed "
                          "grid AND a converged SIMPLE solve; it is not a grid-independence "
                          "proof (see observed order / GCI)"),
        "provenance": {"torch": __import__("torch").__version__,
                       "numpy": np.__version__, "python": sys.version.split()[0]},
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    made = figures(runs, ref, args.tolerance, out / "figures")

    lines = ["# CASE-CYLINDER-DRAG-FVM - reference vs computed", "",
             f"Reference: {reference['citation']}", "",
             f"Source page: {reference['source_url']} (sha256 {reference['source_sha256'][:16]}...)",
             "", "| grid | C_D | err % | C_L | err % | p_diff (Pa) | err % | verdict |",
             "|---|---|---|---|---|---|---|---|"]
    for row in table:
        mark = lambda k: ("PASS" if row["passed"][k] else "FAIL")
        lines.append(
            f"| {row['resolution']} | {row['computed']['drag']:.6f} | "
            f"{row['error_pct']['drag']:+.4f} | {row['computed']['lift']:+.6f} | "
            f"{row['error_pct']['lift']:+.3f} | {row['computed']['pressure_difference_pa']:.6f} | "
            f"{row['error_pct']['pressure_difference_pa']:+.4f} | "
            f"C_D {mark('drag')} / C_L {mark('lift')} / p_diff {mark('pressure_difference_pa')} |")
    lines += ["", f"Case tolerance: {args.tolerance:.0%}; case verdict on finest grid: "
              f"**{'PASS' if finest['case_passed'] else 'FAIL'}**", "", "## Convergence", ""]
    for key, info in convergence.items():
        if info["available"]:
            lines.append(f"- {key}: observed order p = {info['observed_order']:.3f}, "
                         f"Richardson {info['richardson_extrapolated']:.8f}, "
                         f"GCI(finest) = {info['gci_fine']:.3e}")
        else:
            lines.append(f"- {key}: not available ({info['reason']})")
    lines += ["", "## Sensitivity", "",
              f"- p_diff via crude inverse-distance cell-centre probing on finest grid: "
              f"{finest['dp_nearest_cell_pa']:.6f} Pa "
              f"({sensitivity['pressure_probe_interpolation']['relative_to_reference_pct']:+.4f}% vs ref)",
              f"- drag split (pressure/viscous) on finest grid: "
              f"{finest['force_split']['pressure']:.4f} / {finest['force_split']['viscous']:.4f}",
              f"- iterations to converge: "
              f"{sensitivity['iteration_counts']}", "",
              f"Figures: {', '.join(made)}"]
    (out / "comparison.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if finest["case_passed"] else 2


def report(args) -> int:
    """Render the six-section case report, numbers injected programmatically."""
    summary = json.loads(Path(args.summary).read_text())
    reference = summary["reference"]
    ref = reference["values"]
    grids = summary["grids"]
    config = json.loads((Path(args.runs) / grids[0]["resolution"] / "result.json").read_text())["config"]
    rho, nu = config["density"], config["viscosity"]
    u_mean = 2.0 / 3.0 * config["inlet_velocity"]
    diameter = 2 * config["cylinder_radius"]
    re_d = u_mean * diameter / nu
    re_h = u_mean * config["height"] / nu
    blockage = diameter / config["height"]
    lines = []
    a = lines.append
    a("# CASE-CYLINDER-DRAG-FVM: DFG 2D-1 圆柱绕流阻力算例（TensorFVM）\n")
    a("**判据（先于计算声明）**：三个参考量（$C_D$、$C_L$、$p_{diff}$）在**最细网格**上相对误差均 <5%，"
      "且 SIMPLE 求解收敛，才算本算例通过；5% 是本算例池容差，仓库内部 3% 物理门另列。\n")
    a("## ① 问题与参考值出处\n")
    a("二维通道内静止圆柱绕流（Schäfer & Turek 1996 经典基准）：矩形域 "
      "$[0,2.2]\\times[0,0.41]$，圆心 $(0.2,0.2)$、半径 $r=0.05$，左侧抛物线入口，"
      "上下壁与圆柱壁面无滑移，右侧 do-nothing 出口，$Re=20$ 定常层流。\n")
    a(f"- 参考文献：{reference['citation']}")
    a(f"- 数值来源页：{reference['source_url']}（section “{reference.get('source_page_section', 'n/a')}”），"
      f"归档 sha256 = `{reference.get('source_sha256', 'n/a')}`（原始 HTML 存于 `artifacts/case-cylinder-drag-fvm/reference/`）。")
    a("- **零手抄**：参考值由 `artifacts/case-cylinder-drag-fvm/reference/fetch_reference.py` "
      "对归档页面做正则解析后写入 `reference.json`，并与包内常量做漂移比对（drift = 无）。\n")
    a("参考量定义：$C_D = 2F_D/(U_{mean}^2 L)$，$C_L = 2F_L/(U_{mean}^2 L)$，"
      "$p_{diff}=p(0.15,0.2)-p(0.25,0.2)$，$L=2r=0.1\\,m$，$U_{mean}=0.2\\,m/s$。\n")
    a("## ② 方法与实现\n")
    a("复用仓库既有能力 `tensorfvm.verification.dfg`（不另起炉灶）：")
    a("贴体网格 + SIMPLE 压力-速度耦合 + Rhie–Chow 面通量，层流不可压；"
      "圆柱壁面牵引由最小二乘壁面梯度（含无滑移修正）与面法向压力积分得到，"
      "压力力/黏性力分项存盘；$p_{diff}$ 用壁面压力最小二乘外推到多边形壁面中心后按周向角线性插值。\n")
    a("本单新增的是**算例级 harness**（`scripts/case_cylinder_drag_fvm.py`）：")
    a("- `solve`：按包内入口点跑指定网格序列，产物为包内原生格式（`result.json`/`fields.npz`/`history.json`）；")
    a("- `analyze`：对照机器解析出的参考值逐项判 5% 门，并给出网格收敛阶/Richardson、"
      "压力采样替代方案敏感性、阻力分项、图件；")
    a("- `report`：由 `summary.json` 程序化注入所有数字生成本报告（零手抄）。\n")
    a("一键复现（集群作业体，禁指定节点）：\n```bash")
    a("PYTHONPATH=src python scripts/case_cylinder_drag_fvm.py solve \\")
    a("    --grids 64x24 128x48 256x96 --max-iterations 3000 \\")
    a("    --out artifacts/case-cylinder-drag-fvm/runs")
    a("python scripts/case_cylinder_drag_fvm.py analyze \\")
    a("    --runs artifacts/case-cylinder-drag-fvm/runs \\")
    a("    --out artifacts/case-cylinder-drag-fvm/analysis")
    a("python scripts/case_cylinder_drag_fvm.py report \\")
    a("    --runs artifacts/case-cylinder-drag-fvm/runs \\")
    a("    --summary artifacts/case-cylinder-drag-fvm/analysis/summary.json \\")
    a("    --out docs/case-cylinder-drag-fvm/report.md")
    a("```\n")
    a("## ③ 参数表（无量纲数完整）\n")
    a("| 量 | 符号 | 值 | 说明 |")
    a("|---|---|---|---|")
    a(f"| 密度 | $\\rho$ | {rho:g} kg/m$^3$ | 基准规定 |")
    a(f"| 运动黏度 | $\\nu$ | {nu:g} m$^2$/s | 基准规定 |")
    a(f"| 入口最大速度 | $U_{{max}}$ | {config['inlet_velocity']:g} m/s | 抛物线剖面峰值 |")
    a(f"| 入口平均速度 | $U_{{mean}}$ | {u_mean:g} m/s | $=2U_{{max}}/3$ |")
    a(f"| 圆柱直径 | $D=2r$ | {diameter:g} m | 特征长度 |")
    a(f"| 通道长/高 | $L\\times H$ | {config['length']:g} × {config['height']:g} m | 域尺寸 |")
    a(f"| 阻塞比 | $D/H$ | {blockage:.4f} | 通道约束强度 |")
    a(f"| 域长径比 | $L/D$ | {config['length'] / diameter:.1f} | 尾迹长度 |")
    a(f"| 雷诺数（直径） | $Re_D=U_{{mean}}D/\\nu$ | {re_d:.4f} | 算例定义（=20） |")
    a(f"| 雷诺数（通道高） | $Re_H=U_{{mean}}H/\\nu$ | {re_h:.2f} | 通道尺度参考 |")
    a("| 网格 | $n_x\\times n_y$ | " + ", ".join(g["resolution"] for g in grids) +
      " | 三档阶梯细化，间距比约 2 |")
    a("| 收敛判据 | — | 无量纲残差 1e-6 / 最大迭代 3000 | `SolverConfig.tolerance` |\n")
    a("## ④ 对照表（ref vs got + 逐项误差打勾叉）\n")
    a("| 网格 | 单元级距 $h$ (m) | $C_D$ got | $C_D$ err | $C_L$ got | $C_L$ err | $p_{diff}$ got (Pa) | $p_{diff}$ err | <5% 判定 |")
    a("|---|---|---|---|---|---|---|---|---|")
    for g in grids:
        def cell(key, fmt="{:.6f}"):
            mark = "✅" if g["passed"][key] else "❌"
            return f"{fmt.format(g['computed'][key])} | {g['error_pct'][key]:+.4f}% {mark}"
        a(f"| {g['resolution']} | {g['spacing_m']:.5f} | {cell('drag')} | {cell('lift', '{:+.6f}')} "
          f"| {cell('pressure_difference_pa')} | "
          f"{'✅ 通过' if g['case_passed'] else '❌ 未通过'} |")
    a(f"| **reference** | — | {ref['drag']:.11f} | — | {ref['lift']:.12f} | — | {ref['pressure_difference_pa']:.11f} | — | — |\n")
    a(f"**算例判定（最细网格 5% 门）**：{'✅ PASS' if summary['case_passed'] else '❌ FAIL'}\n")
    a("## ⑤ 收敛性与参数敏感性（至少两点）\n")
    a("**(1) 网格细化**：三档网格 $h$ 减半，观测收敛阶与 Richardson 估计如下\n")
    a("| 量 | 观测阶 p | Richardson 外推 | 最细网格 GCI |")
    a("|---|---|---|---|")
    for key, info in summary["convergence"].items():
        if info["available"]:
            a(f"| {QUANTITIES[key][0]} | {info['observed_order']:.3f} | "
              f"{info['richardson_extrapolated']:.8f} | {info['gci_fine']:.3e} |")
        else:
            a(f"| {QUANTITIES[key][0]} | 不可得（{info['reason']}） | — | — |")
    sens = summary["sensitivity"]
    a("\n**(2) 压力差采样方案敏感性**：官方口径（壁面最小二乘外推 + 周向角插值）与"
      "粗糙口径（探针点附近单元压力反距离插值）在最细网格上对比：\n")
    a(f"- 官方口径 $p_{{diff}}$ = {sens['pressure_probe_interpolation']['dfg_wall_extrapolation_pa']:.6f} Pa")
    a(f"- 粗糙口径 $p_{{diff}}$ = {sens['pressure_probe_interpolation']['inverse_distance_cell_centre_pa']:.6f} Pa "
      f"（相对参考 {sens['pressure_probe_interpolation']['relative_to_reference_pct']:+.4f}%）")
    a(f"- 差值 = {sens['pressure_probe_interpolation']['delta_pa']:+.6f} Pa "
      "→ 该量对采样/重构方式敏感，报告误差必须声明口径。\n")
    a("**(3) 阻力分项**（最细网格，$C_D$ 贡献）：压力 "
      f"{sens['force_decomposition_finest']['pressure']:.4f} + 黏性 "
      f"{sens['force_decomposition_finest']['viscous']:.4f} = "
      f"{sens['force_decomposition_finest']['total']:.4f}\n")
    a("**(4) 迭代/收敛**：" + ", ".join(
        f"{k}: {v[0]} iters（{'收敛' if v[1] else '未收敛'}）" for k, v in sens["iteration_counts"].items()) + "\n")
    a("![drag refinement](../artifacts/case-cylinder-drag-fvm/analysis/figures/cd_refinement.png)\n")
    a("![error bars](../artifacts/case-cylinder-drag-fvm/analysis/figures/error_bars.png)\n")
    a("![pressure field](../artifacts/case-cylinder-drag-fvm/analysis/figures/field_pressure.png)\n")
    a("![speed field](../artifacts/case-cylinder-drag-fvm/analysis/figures/field_speed.png)\n")
    a("![drag decomposition](../artifacts/case-cylinder-drag-fvm/analysis/figures/drag_decomposition.png)\n")
    a("![wall pressure](../artifacts/case-cylinder-drag-fvm/analysis/figures/wall_pressure.png)\n")
    a("![residuals](../artifacts/case-cylinder-drag-fvm/analysis/figures/residuals.png)\n")
    a("## ⑥ 失败模式与工件\n")
    a("已知失败模式（本算例自身与本仓历史都出现过）：\n")
    a("1. **粗网格欠解析**：本仓历史记录 64×24 曾出现 $C_D$ 误差约 13%（见 "
      "`docs/benchmarks-index.md`），故本单不认单网格结果，必须阶梯细化；")
    a("2. **壁面压力/牵引重构口径**：$p_{diff}$ 对采样方式敏感（见 ⑤(2)），"
      "粗糙探针口径会把误差抬高；圆柱壁面梯度若不修正无滑移会产生伪黏性力；")
    a("3. **近零量判据病态**：$C_L$ 参考值仅 0.0106，$Re=20$ 定常解本应上下对称，"
      "相对误差对极小的非对称数值残差极其敏感——$C_L$ 的 5% 门应视为严格附加门而非主判据；")
    a("4. **收敛不足**：迭代上限截断会同时抬高 $C_D$ 与 $C_L$ 误差，"
      "故每档记录实际迭代数与最终残差，未收敛即判 FAIL。\n")
    a("工件（均在 worktree 内，真实可复算）：\n")
    a("- `artifacts/case-cylinder-drag-fvm/reference/`：归档参考页 HTML + sha256 + 解析脚本 + `reference.json`；")
    a("- `artifacts/case-cylinder-drag-fvm/runs/<grid>/`：`result.json`（含 config/历史/指标）、`fields.npz`（全场、面通量、壁面牵引）、`history.json`；")
    a("- `artifacts/case-cylinder-drag-fvm/analysis/`：`summary.json`、`comparison.md`、`figures/*.png`；")
    a("- `scripts/case_cylinder_drag_fvm.py`（solve/analyze/report 三入口）、`tests/test_case_cylinder_drag_fvm.py`（合成数据契约测试）。\n")
    a("## 生态摩擦（如实记）\n")
    a("- knowledge_base 检索引擎（127.0.0.1:8021）离线：`/tensor/knowledge/search` 返回 engine_offline，"
      "只能退化为“仓内 grep + docs/benchmarks-index.md”做法；")
    a("- 任务要求查 `oss_solutions` 同域，但平台 OpenAPI 无对应端点（无 oss/solution 路径），"
      "该查项无法执行，如实记为缺项；")
    a("- `negative_knowledge_ledger` 同域检索（q=“TensorFVM cylinder drag”）0 命中，说明历次圆柱失败未入否定性台账；")
    a("- 1996 原文 PDF 未能公开抓取（付费墙/无稳定镜像），参考值以 Turek 组官网页 “Reference results” 为准并归档 sha256，"
      "页面自身标注其值取自 Nabh Preprint 42/98；")
    a("- 本地控制面（99）单档 64×24 的 200 次迭代 >120 s，重算一律走 cluster_queue；"
      "集群排队（递交时 312 个作业在前）是本单主要墙钟成本。\n")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    solver = sub.add_parser("solve")
    solver.add_argument("--grids", nargs="+", default=["64x24", "128x48", "256x96"])
    solver.add_argument("--max-iterations", type=int, default=3000)
    solver.add_argument("--out", type=Path, required=True)
    analyzer = sub.add_parser("analyze")
    analyzer.add_argument("--runs", type=Path, required=True)
    analyzer.add_argument("--out", type=Path, required=True)
    analyzer.add_argument("--reference", type=Path,
                          default=REPO / "artifacts/case-cylinder-drag-fvm/reference/reference.json")
    analyzer.add_argument("--tolerance", type=float, default=0.05)
    renderer = sub.add_parser("report")
    renderer.add_argument("--runs", type=Path, required=True)
    renderer.add_argument("--summary", type=Path, required=True)
    renderer.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    hand = {"solve": solve, "analyze": analyze, "report": report}[args.command]
    return hand(args)


if __name__ == "__main__":
    raise SystemExit(main())