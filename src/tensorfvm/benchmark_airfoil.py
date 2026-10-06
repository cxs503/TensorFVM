"""Quantitative steady NACA 0012 validation at Re=1000 and alpha=4 degrees.

The reference is Di Ilio et al. (2020), arXiv:2006.10487, Figure 10/11.
At four degrees the published solution is steady; higher-incidence cases become
periodic and are outside this repository's steady SIMPLE model.
"""

import argparse
import json
import math
from pathlib import Path

import torch

from .cylinder import export_result
from .solver import SimpleSolver, SolverConfig


ERROR_LIMIT = 0.03
REFERENCE = {
    "airfoil": "NACA 0012",
    "reynolds": 1000.0,
    "angle_of_attack_degrees": 4.0,
    "lift_coefficient": 0.205,
    "drag_coefficient": 0.120,
    "citation": "Di Ilio et al., Fluid flow around NACA 0012 airfoil at low-Reynolds numbers with hybrid lattice Boltzmann method (2020)",
    "url": "https://arxiv.org/abs/2006.10487",
    "source_location": "Figures 10 and 11",
}


def compare_airfoil(result):
    """Apply strict, independent 3% checks to published lift and drag."""
    c = result.config
    if (c.mesh_type != "c-grid" or c.airfoil_code != "0012"
            or not math.isclose(c.reynolds, REFERENCE["reynolds"])
            or not math.isclose(c.angle_of_attack,
                                REFERENCE["angle_of_attack_degrees"])):
        raise ValueError("airfoil benchmark requires NACA 0012, Re=1000, alpha=4 degrees")
    coefficients = result.aerodynamic_coefficients or {}
    computed = {name: float(coefficients.get(name, math.nan))
                for name in ("lift", "drag")}
    reference = {"lift": REFERENCE["lift_coefficient"],
                 "drag": REFERENCE["drag_coefficient"]}
    errors = {name: abs(computed[name] - reference[name]) / abs(reference[name])
              for name in reference}
    residuals = result.history[-1] if result.history else {}
    converged = bool(result.converged and all(
        math.isfinite(residuals.get(name, math.nan))
        and residuals[name] < c.tolerance
        for name in ("continuity", "momentum", "mass_imbalance")
    ))
    finite = all(math.isfinite(value) for value in (*computed.values(), *errors.values()))
    return {
        "nx": c.nx, "ny": c.ny, "iterations": len(result.history),
        "converged": converged, "final_residuals": residuals,
        "computed": computed, "reference": reference, "relative_errors": errors,
        "passed": bool(converged and finite
                       and all(error < ERROR_LIMIT for error in errors.values())),
    }


def _write_report(directory, summary):
    case = summary["case"]
    text = f"""NACA 0012 低雷诺数定量验证
============================

基准与适用范围
--------------

本验证采用 Di Ilio 等人的二维不可压缩层流结果（arXiv:2006.10487），
NACA 0012，Re=1000，攻角 4°。文献指出攻角低于 8° 时解为稳态，因而与
本项目的稳态 SIMPLE 模型相容。参考值取其图 10、11：Cl=0.205，Cd=0.120。
升力和阻力相对误差分别独立要求严格小于 3%，不以平均误差替代。

压力系数说明
------------

程序仍在 ``airfoil.csv`` 导出完整表面 Cp。该文献只给出了 8° 的 Cp 曲线，
而 8° 已处于非稳态起始点，且未提供机器可读表格；因此不得把图像估读数据
伪装成 4° 的定量参考，本稳态验收不对 Cp 声称 3% 误差。

结果
----

* 网格：{case['nx']} × {case['ny']}
* 迭代：{case['iterations']}，收敛：{case['converged']}
* Cl={case['computed']['lift']:.8g}，相对误差 {100*case['relative_errors']['lift']:.4f}%
* Cd={case['computed']['drag']:.8g}，相对误差 {100*case['relative_errors']['drag']:.4f}%
* 结论：{'通过' if case['passed'] else '失败'}

复现命令::

    python -m tensorfvm.benchmark_airfoil --output results/airfoil-benchmark
"""
    (directory / "report.rst").write_text(text, encoding="utf-8")


def run_benchmark(directory, max_iterations=1200, nx=64, ny=26):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config = SolverConfig(
        mesh_type="c-grid", airfoil_code="0012", reynolds=1000,
        angle_of_attack=4, nx=nx, ny=ny, length=20, height=16,
        airfoil_x=6, airfoil_y=8, airfoil_chord=1,
        velocity_relaxation=0.5, pressure_relaxation=0.3,
        pseudo_time_step=0.2, tolerance=1e-5,
        max_iterations=max_iterations,
    )
    result = SimpleSolver(config).solve()
    export_result(result, directory / f"{nx}x{ny}")
    case = compare_airfoil(result)
    summary = {
        "benchmark": "NACA0012-Re1000-alpha4",
        "error_limit": ERROR_LIMIT,
        "reference_metadata": REFERENCE,
        "cp_acceptance": {
            "available": False,
            "reason": "the matching alpha=4 reference has no tabulated Cp data",
        },
        "case": case,
        "passed": case["passed"],
    }
    (directory / "benchmark.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    _write_report(directory, summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="NACA 0012, Re=1000, alpha=4° 稳态升阻力定量验证（误差 <3%）"
    )
    parser.add_argument("--output", type=Path,
                        default=Path("results/airfoil-benchmark"))
    parser.add_argument("--max-iterations", type=int, default=1200)
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--ny", type=int, default=26)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    try:
        summary = run_benchmark(args.output, args.max_iterations, args.nx, args.ny)
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, f"翼型 benchmark 失败：{error}\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
