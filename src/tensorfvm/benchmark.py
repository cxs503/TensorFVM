"""Reproducible plane-Poiseuille verification with dependency-free SVG plots."""

import argparse
import csv
from html import escape
import json
import math
from pathlib import Path
import platform

import torch

from .cylinder import export_result
from .solver import SimpleSolver, SolverConfig


ERROR_LIMIT = 0.03


def compare_poiseuille(result):
    """Compare a developed section and pressure slope, not the entrance region."""
    c = result.config
    if c.mesh_type != "cartesian" or c.cylinder_radius or c.length < 6 * c.height:
        raise ValueError("benchmark requires an unobstructed Cartesian channel with L/H >= 6")
    u, _ = result.cell_center_velocity()
    section = int(torch.argmin(torch.abs(result.x - 5 * c.height)).item())
    exact = 6 * c.inlet_velocity * result.y / c.height * (1 - result.y / c.height)
    difference = u[:, section] - exact
    region = (result.x >= 3 * c.height) & (result.x <= 5 * c.height)
    x = result.x[region]
    if x.numel() < 2:
        raise ValueError("at least two pressure columns are required in 3H <= x <= 5H")
    pressure = result.p[:, region].mean(dim=0)
    centered_x = x - x.mean()
    slope = float((centered_x * (pressure - pressure.mean())).sum()
                  / centered_x.square().sum())
    exact_slope = -12 * c.viscosity * c.inlet_velocity / c.height ** 2
    errors = {
        "velocity_l2": float(torch.linalg.vector_norm(difference)
                             / torch.linalg.vector_norm(exact)),
        "velocity_linf": float(difference.abs().max() / exact.abs().max()),
        "pressure_gradient": abs(slope - exact_slope) / abs(exact_slope),
    }
    finite = all(torch.isfinite(field).all().item()
                 for field in (result.u, result.v, result.p))
    residuals = result.history[-1] if result.history else {}
    converged = result.converged and all(
        math.isfinite(residuals.get(name, math.inf))
        and residuals.get(name, math.inf) < c.tolerance
        for name in ("continuity", "momentum", "mass_imbalance")
    )
    return {
        "nx": c.nx, "ny": c.ny, "section_x": float(result.x[section]),
        "iterations": len(result.history), "converged": bool(converged),
        "final_residuals": residuals, "errors": errors,
        "pressure_gradient": slope, "reference_pressure_gradient": exact_slope,
        "passed": bool(finite and converged and all(
            math.isfinite(error) and error < ERROR_LIMIT for error in errors.values())),
    }


def write_contour(result, field, title, destination):
    """Draw actual cell values with physical aspect ratio and a labelled scale."""
    values = field.detach().cpu()
    low, high = float(values.min()), float(values.max())
    width, height = 900, 900 * result.config.height / result.config.length
    dx, dy = width / result.config.nx, height / result.config.ny

    def color(value):
        fraction = (value - low) / (high - low) if high > low else 0.5
        fraction = max(0.0, min(1.0, fraction))
        return f"rgb({round(255 * fraction)},70,{round(255 * (1 - fraction))})"

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="{height + 170:g}" '
        f'viewBox="0 0 1000 {height + 170:g}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="50" y="25" font-family="sans-serif" font-size="18">{escape(title)}</text>',
    ]
    for j in range(result.config.ny):
        for i in range(result.config.nx):
            parts.append(
                f'<rect x="{50 + i * dx:g}" y="{45 + (result.config.ny - 1 - j) * dy:g}" '
                f'width="{dx:g}" height="{dy:g}" fill="{color(float(values[j, i]))}"/>'
            )
    parts.extend([
        f'<rect x="50" y="45" width="{width:g}" height="{height:g}" fill="none" stroke="black"/>',
        f'<text x="50" y="{height + 65:g}" font-family="sans-serif">x/H = 0</text>',
        f'<text x="885" y="{height + 65:g}" font-family="sans-serif">x/H = 6</text>',
        '<text x="5" y="55" font-family="sans-serif">y/H=1</text>',
        f'<text x="5" y="{height + 45:g}" font-family="sans-serif">0</text>',
    ])
    for i in range(100):
        parts.append(f'<rect x="{50 + 9 * i}" y="{height + 90:g}" width="9" height="15" '
                     f'fill="{color(low + (high - low) * i / 99)}"/>')
    for fraction in (0, 0.25, 0.5, 0.75, 1):
        parts.append(f'<text x="{50 + 900 * fraction:g}" y="{height + 125:g}" '
                     f'font-family="sans-serif">{low + (high - low) * fraction:.4g}</text>')
    parts.append("</svg>")
    destination.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_report(directory, summary):
    rows = []
    for case in summary["cases"]:
        errors = case["errors"]
        rows.append(
            f"   * - {case['nx']} × {case['ny']}\n"
            f"     - {case['iterations']}\n"
            f"     - {case['section_x']:.5f}\n"
            f"     - {100 * errors['velocity_l2']:.4f}%\n"
            f"     - {100 * errors['velocity_linf']:.4f}%\n"
            f"     - {100 * errors['pressure_gradient']:.4f}%\n"
            f"     - {'通过' if case['passed'] else '失败'}"
        )
    finest = summary["cases"][-1]
    text = f"""Poiseuille benchmark 计算说明
=============================

验证范围与参考解
----------------

本算例对标二维平行板充分发展层流的 Poiseuille 解析解，而不是圆柱或
NACA 公开绕流数据。它验证笛卡尔有限体积离散、壁面条件及 SIMPLE 耦合；
不能据此宣称贴体网格、圆柱阻力或翼型升阻力误差小于 3%。

参考：`Plane Poiseuille flow
<https://en.wikipedia.org/wiki/Hagen%E2%80%93Poiseuille_equation#Plane_Poiseuille_flow>`_。
由稳态充分发展方程 ``mu * d²u/dy² = dp/dx``、两壁无滑移及
截面平均速度 U 得到独立解析解：

* ``u_ref(y) = 6 U (y/H) (1-y/H)``，``v_ref = 0``；
* ``(dp/dx)_ref = -12 mu U / H² = -12``。

设置与复现
----------

H=1，L=6，U=1，rho=1，Re_H=1，mu=1，无圆柱；
入口均匀速度，上下壁面无滑移，出口零表压、预测速度零法向梯度。
因此入口段并非充分发展，不能用全域速度或入口总压降与充分发展解析解比较。
初始化为求解器默认均匀速度、零压力，不植入解析解。

float64，CPU 单线程；速度/压力欠松弛 0.7/0.3，
最大迭代 {summary['max_iterations']} 次，连续性、动量及质量不平衡容差均为 1e-6。
运行环境：Python {summary['python']}，PyTorch {summary['torch']}。
在仓库根目录安装项目后运行（再次运行会覆盖同名输出）::

    python -m tensorfvm.benchmark --output results/benchmark

返回码 0 表示所有网格已收敛且每项误差严格小于 3%；2 表示未达标，
1 表示运行失败。失败计算也保留诊断数据，不隐藏或截断误差。
每档网格的 fields.csv、history.json、summary.json 保留完整场与残差；
benchmark.json 为机器可读验收数据，profiles.csv 为速度对标数据。

误差定义与结果
--------------

速度取最接近 x=5H 的单元中心截面，与同一 y 坐标的解析解比较：
``E_L2 = ||u-u_ref||₂ / ||u_ref||₂``；
``E_Linf = max|u-u_ref| / max|u_ref|``。
后者是以截面最大参考速度归一化的误差，不是近壁逐点相对误差。
压力取 3H ≤ x ≤ 5H 的截面平均值作最小二乘直线拟合，
``E_p = |slope-slope_ref| / |slope_ref|``。
比较梯度避开表压参考值和入口附加压降；各误差独立检验，不作平均。

.. list-table:: 实际计算误差（严格阈值 < 3%）
   :header-rows: 1

   * - 网格 nx × ny
     - 迭代次数
     - 截面 x/H
     - 速度 L2
     - 速度 Linf
     - 压力梯度
     - 验收
{chr(10).join(rows)}

总体结论：{'全部通过' if summary['passed'] else '未全部通过'}。
更粗的 24 × 8 网格在相同设置下压梯度误差约 3.03%，不满足阈值，
已作为回归测试中的拒绝案例；以上三档网格才是正式验收网格。
最细网格连续性残差 {finest['final_residuals'].get('continuity', math.inf):.6g}，
动量残差 {finest['final_residuals'].get('momentum', math.inf):.6g}，
质量不平衡 {finest['final_residuals'].get('mass_imbalance', math.inf):.6g}。
网格加密结果用于观察离散误差趋势，不代替无限域或绕流 benchmark 验证。

计算云图
--------

以下均来自最细网格实际数值解（单元常值着色，无解析解替换、插值平滑或
纵横比例拉伸）。x 从左向右，y 从下向上；色标为对应无量纲值。

.. image:: velocity.svg
   :alt: 数值速度模云图，颜色范围见色标

.. image:: pressure.svg
   :alt: 数值表压云图，颜色范围见色标

速度图可见入口发展及下游抛物线分布，压力图可见下游近线性降压。
"""
    (directory / "report.rst").write_text(text, encoding="utf-8")


def run_benchmark(directory, max_iterations=1000):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    cases, profiles = [], []
    for ny in (12, 24, 48):
        config = SolverConfig(
            nx=3 * ny, ny=ny, length=6, height=1, cylinder_radius=None,
            reynolds=1, tolerance=1e-6, max_iterations=max_iterations,
        )
        result = SimpleSolver(config).solve()
        cases.append(compare_poiseuille(result))
        export_result(result, directory / f"{config.nx}x{ny}")
        u, v = result.cell_center_velocity()
        section = int(torch.argmin(torch.abs(result.x - 5)).item())
        for j in range(ny):
            y = float(result.y[j])
            profiles.append((config.nx, ny, float(result.x[section]), y,
                             float(u[j, section]), 6 * y * (1 - y)))
    summary = {
        "benchmark": "plane-poiseuille", "error_limit": ERROR_LIMIT,
        "python": platform.python_version(), "torch": str(torch.__version__),
        "max_iterations": max_iterations, "cases": cases,
        "passed": all(case["passed"] for case in cases),
    }
    with (directory / "benchmark.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")
    with (directory / "profiles.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("nx", "ny", "x", "y", "u", "u_reference"))
        writer.writerows(profiles)
    grid = f"({config.nx} x {config.ny})"
    write_contour(result, torch.sqrt(u.square() + v.square()) / config.inlet_velocity,
                  f"Computed speed |V|/U {grid}", directory / "velocity.svg")
    write_contour(result, result.p / (config.density * config.inlet_velocity ** 2),
                  f"Computed gauge pressure p/(rho U^2) {grid}", directory / "pressure.svg")
    write_report(directory, summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Poiseuille benchmark：误差严格小于 3%")
    parser.add_argument("--output", type=Path, default=Path("results/benchmark"))
    parser.add_argument("--max-iterations", type=int, default=1000)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    try:
        summary = run_benchmark(args.output, args.max_iterations)
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, f"benchmark 失败：{error}\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
