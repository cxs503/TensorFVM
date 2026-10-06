"""High-Reynolds-number turbulent flat-plate verification for Spalart--Allmaras.

The case is a smooth, zero-pressure-gradient plate with the turbulent boundary
layer tripped at the leading edge.  It is deliberately a steady 2-D RANS case:
a high-Re cylinder would be intrinsically unsteady and three-dimensional, so it
would not validate this solver's present assumptions.
"""

import argparse
import json
import math
from pathlib import Path

import torch

from .cylinder import export_result
from .solver import SimpleSolver, SolverConfig


# Schlichting's smooth fully turbulent flat-plate average skin-friction law.
# It applies when the boundary layer is turbulent from the leading edge.
ERROR_LIMIT = 0.20
REFERENCE_REYNOLDS = 100_000.0


def reference_skin_friction(reynolds: float) -> float:
    """Return the smooth, fully turbulent average plate skin-friction law."""
    return 0.074 / reynolds ** 0.2


def compare_flat_plate(result):
    """Compare integrated wall friction, convergence, and first-cell y-plus."""
    c = result.config
    if (c.mesh_type != "flat-plate" or c.turbulence_model != "spalart-allmaras"
            or c.reynolds < 10_000):
        raise ValueError("flat-plate benchmark requires high-Re Spalart-Allmaras flat-plate RANS")
    forces = result.surface_forces or {}
    if "plate" not in forces:
        raise ValueError("flat-plate comparison requires integrated plate force")
    drag = float(forces["plate"]["x"])
    scale = 0.5 * c.density * c.inlet_velocity ** 2 * c.length
    skin_friction = drag / scale
    reference = reference_skin_friction(c.reynolds)
    error = abs(skin_friction - reference) / reference
    wall_shear = drag / c.length
    friction_velocity = math.sqrt(max(wall_shear, 0) / c.density)
    first_center = float(result.y[0].min())
    y_plus = first_center * friction_velocity / (c.viscosity / c.density)
    residuals = result.history[-1] if result.history else {}
    required = ("continuity", "momentum", "mass_imbalance", "turbulence")
    converged = bool(result.converged and all(
        math.isfinite(residuals.get(name, math.nan))
        and residuals[name] < c.tolerance for name in required
    ))
    finite = all(math.isfinite(value) for value in (drag, skin_friction, error, y_plus))
    return {
        "nx": c.nx,
        "ny": c.ny,
        "iterations": len(result.history),
        "converged": converged,
        "final_residuals": residuals,
        "skin_friction_coefficient": skin_friction,
        "reference_skin_friction_coefficient": reference,
        "relative_error": error,
        "first_cell_y_plus": y_plus,
        "passed": bool(finite and converged and y_plus < 1
                       and error < ERROR_LIMIT),
    }


def _write_report(directory: Path, summary: dict) -> None:
    case = summary["case"]
    text = f"""Spalart--Allmaras 高雷诺数平板 RANS 验证
========================================

范围
----

本算例是零压梯度光滑平板边界层，Re_L={REFERENCE_REYNOLDS:g}，入口和
上方远场均为均匀来流，底边为从前缘开始完全湍流的无滑移平板。它验证
当前稳态二维 RANS 与 Spalart--Allmaras 一方程闭合的基础实现；不代表
高 Re 圆柱绕流。后者通常是非稳态、三维问题，需要 URANS、DES 或 LES。

对标量为平板长度上的平均摩擦阻力系数。参考关联式为
``C_f = 0.074 Re_L^(-1/5)``（Schlichting 平滑、前缘即完全湍流平板）。
该关联式本身不是 DNS/实验逐点数据；本项目采用严格小于 {100 * ERROR_LIMIT:g}%
的集成摩擦误差作为初始 RANS 回归阈值，同时要求第一单元 ``y+ < 1`` 和
连续性、动量、质量不平衡、SA 输运残差均收敛。

结果
----

* 网格：{case['nx']} × {case['ny']}
* 迭代：{case['iterations']}，收敛：{case['converged']}
* 数值平均 C_f：{case['skin_friction_coefficient']:.8g}
* 关联式 C_f：{case['reference_skin_friction_coefficient']:.8g}
* 相对误差：{100 * case['relative_error']:.4f}%
* 第一单元 y+：{case['first_cell_y_plus']:.5g}
* 结论：{'通过' if case['passed'] else '失败'}

复现::

    python -m tensorfvm.benchmark_flat_plate --output results/flat-plate

``turbulence.csv`` 导出 SA 工作变量 ``nu_tilde`` 与运动学涡黏度 ``nu_t``。
该模型没有转捩、可压缩修正、曲率修正或壁函数；定量用于翼型、圆柱前仍应
进行独立网格无关性与公开实验数据验证。
"""
    (directory / "report.rst").write_text(text, encoding="utf-8")


def run_benchmark(directory, max_iterations=1800, nx=64, ny=56):
    """Run the wall-resolved Re=1e5 flat-plate SA verification case."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config = SolverConfig(
        mesh_type="flat-plate",
        turbulence_model="spalart-allmaras",
        nx=nx,
        ny=ny,
        length=1.0,
        height=0.2,
        cylinder_radius=None,
        inlet_velocity=1.0,
        reynolds=REFERENCE_REYNOLDS,
        velocity_relaxation=0.3,
        pressure_relaxation=0.2,
        turbulence_relaxation=0.3,
        pseudo_time_step=0.02,
        sa_freestream_ratio=3.0,
        flat_plate_stretching=4.0,
        tolerance=1e-5,
        max_iterations=max_iterations,
    )
    result = SimpleSolver(config).solve()
    export_result(result, directory / f"{nx}x{ny}")
    case = compare_flat_plate(result)
    summary = {
        "benchmark": "turbulent-flat-plate-SA",
        "reference_reynolds": REFERENCE_REYNOLDS,
        "error_limit": ERROR_LIMIT,
        "reference": "Schlichting smooth fully turbulent flat plate: Cf=0.074 Re_L^(-1/5)",
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
        description="Re=1e5 Spalart-Allmaras 平板 RANS 基准（平均 Cf 误差 <20%）"
    )
    parser.add_argument("--output", type=Path, default=Path("results/flat-plate"))
    parser.add_argument("--max-iterations", type=int, default=1800)
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--ny", type=int, default=56)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    try:
        summary = run_benchmark(args.output, args.max_iterations, args.nx, args.ny)
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, f"flat-plate benchmark 失败：{error}\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
