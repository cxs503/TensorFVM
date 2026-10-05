"""DFG 2D-1 cylinder benchmark with reproducible fitted-grid outputs."""

import argparse
import json
from pathlib import Path

import torch

from .cylinder import export_result
from .solver import SimpleSolver, SolverConfig


REFERENCE_DRAG = 5.579535
ERROR_LIMIT = 0.03
GRIDS = ((64, 24), (128, 48), (256, 96))


def compare_dfg(result):
    """Compare the converged cylinder drag coefficient with Schäfer–Turek DFG 2D-1."""
    coefficients = result.aerodynamic_coefficients
    if coefficients is None or "drag" not in coefficients:
        raise ValueError("DFG comparison requires a body-fitted cylinder result")
    drag = float(coefficients["drag"])
    error = abs(drag - REFERENCE_DRAG) / REFERENCE_DRAG
    residuals = result.history[-1] if result.history else {}
    converged = bool(result.converged and all(
        name in residuals and torch.isfinite(torch.tensor(residuals[name])).item()
        and residuals[name] < result.config.tolerance
        for name in ("continuity", "momentum", "mass_imbalance")
    ))
    return {
        "nx": result.config.nx,
        "ny": result.config.ny,
        "iterations": len(result.history),
        "converged": converged,
        "residuals": residuals,
        "drag_coefficient": drag,
        "reference_drag_coefficient": REFERENCE_DRAG,
        "drag_relative_error": error,
        "passed": bool(converged and torch.isfinite(torch.tensor(error)).item()
                       and error < ERROR_LIMIT),
    }


def _write_report(directory, summary):
    rows = []
    for case in summary["cases"]:
        rows.append(
            f"   * - {case['nx']} × {case['ny']}\n"
            f"     - {case['iterations']}\n"
            f"     - {case['drag_coefficient']:.8g}\n"
            f"     - {100 * case['drag_relative_error']:.4f}%\n"
            f"     - {'通过' if case['passed'] else '失败'}"
        )
    report = f"""Schäfer–Turek DFG 2D-1 圆柱绕流基准
=========================================

基准工况
--------

采用公开的 DFG 2D-1 稳态圆柱基准：通道长 2.2、高 0.41，圆柱直径
0.1、圆心 (0.2, 0.2)，入口截面平均速度 0.2，密度 1，动力黏度 0.001，
因此 Re=20。入口为抛物线剖面 ``u(y)=6 U_mean (y/H)(1-y/H)``，
上下壁面和圆柱无滑移，出口定压。参考阻力系数为 ``Cd=5.579535``，
定义长度为圆柱直径、速度为截面平均入口速度。

参考来源：Schäfer & Turek (1996), “Benchmark Computations of Laminar
Flow Around a Cylinder,” Notes on Numerical Fluid Mechanics 52, 547–566；
公开基准说明见
`DFG 2D-1 <https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark1_re20.html>`_。

运行设置
--------

在仓库根目录运行::

    python -m tensorfvm.benchmark_dfg --output results/dfg

所有网格须收敛且圆柱阻力系数相对参考值误差严格小于 3% 才返回 0；
误差或收敛失败返回 2。每个网格目录包含完整字段、网格、残差、升阻力和
速度/压力 SVG；benchmark.json 与本报告保留验收指标。

结果
----

.. list-table:: 相对公开阻力系数的验收（阈值 < 3%）
   :header-rows: 1

   * - 网格 nx × ny
     - 迭代次数
     - Cd
     - 相对误差
     - 验收
{chr(10).join(rows)}

总体结论：{'全部通过' if summary['passed'] else '未通过'}。

最细网格计算流场云图（单元常值着色、按物理四边形网格绘制）：

.. image:: 256x96/velocity.svg
   :alt: DFG 圆柱绕流速度模云图

.. image:: 256x96/pressure.svg
   :alt: DFG 圆柱绕流表压云图
"""
    (directory / "report.rst").write_text(report, encoding="utf-8")


def run_benchmark(directory, max_iterations=1000):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    cases = []
    for nx, ny in GRIDS:
        config = SolverConfig(
            mesh_type="body-fitted",
            inlet_profile="parabolic",
            nx=nx,
            ny=ny,
            length=2.2,
            height=0.41,
            cylinder_x=0.2,
            cylinder_y=0.2,
            cylinder_radius=0.05,
            inlet_velocity=0.2,
            density=1,
            reynolds=20,
            tolerance=1e-6,
            max_iterations=max_iterations,
        )
        result = SimpleSolver(config).solve()
        cases.append(compare_dfg(result))
        export_result(result, directory / f"{nx}x{ny}")
    summary = {
        "benchmark": "DFG-2D-1",
        "reference": "Schaefer-Turek-1996",
        "reference_drag_coefficient": REFERENCE_DRAG,
        "error_limit": ERROR_LIMIT,
        "max_iterations": max_iterations,
        "cases": cases,
        "passed": all(case["passed"] for case in cases),
    }
    (directory / "benchmark.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    _write_report(directory, summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Schäfer–Turek DFG 2D-1 圆柱基准（阻力误差 < 3%）"
    )
    parser.add_argument("--output", type=Path, default=Path("results/dfg"))
    parser.add_argument("--max-iterations", type=int, default=1000)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    try:
        summary = run_benchmark(args.output, args.max_iterations)
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, f"DFG benchmark 失败：{error}\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
