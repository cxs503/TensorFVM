"""Diagnostic Re=3900 external-cylinder URANS case with SA turbulence closure.

This is intentionally stricter than a mere finite-field smoke test: it records
force time histories and declares the comparison unsuccessful when the 2-D
URANS calculation does not resolve a periodic lift signal.  At Re=3900 the
physical wake is three-dimensional, so a successful 2-D SA calculation is only
a screening result, not a replacement for 3-D LES/DES validation.
"""

import argparse
import json
import math
from pathlib import Path

import torch

from .cylinder import export_result
from .solver import SimpleSolver, SolverConfig


REFERENCE = {
    "reynolds": 3900.0,
    "mean_drag": 1.12,
    "strouhal": 0.20,
    "citation": (
        "Parnaudeau et al. (2008), Experimental and numerical studies of "
        "the flow over a circular cylinder at Reynolds number 3900"
    ),
}
DRAG_ERROR_LIMIT = 0.25
STROUHAL_ERROR_LIMIT = 0.20


def _force_statistics(force_history, warmup_steps, time_step, diameter, velocity):
    samples = force_history[warmup_steps:]
    if len(samples) < 32:
        raise ValueError("at least 32 post-warmup force samples are required")
    drag = torch.tensor([item["drag"] for item in samples], dtype=torch.float64)
    lift = torch.tensor([item["lift"] for item in samples], dtype=torch.float64)
    lift_fluctuation = lift - lift.mean()
    lift_rms = float(torch.sqrt(torch.mean(lift_fluctuation.square())))
    spectrum = torch.fft.rfft(lift_fluctuation).abs()
    index = int(torch.argmax(spectrum[1:]).item()) + 1
    frequency = index / (len(samples) * time_step)
    return {
        "sample_count": len(samples),
        "mean_drag": float(drag.mean()),
        "lift_rms": lift_rms,
        "strouhal": frequency * diameter / velocity,
    }


def compare_cylinder_urans(result, warmup_steps=200):
    """Check whether the URANS history resolves Re=3900 force oscillations."""
    c = result.config
    if (c.mesh_type != "body-fitted" or c.outer_boundary != "far-field"
            or c.turbulence_model != "spalart-allmaras" or c.time_step is None
            or not math.isclose(c.reynolds, REFERENCE["reynolds"])):
        raise ValueError("comparison requires the external Re=3900 SA-URANS cylinder case")
    stats = _force_statistics(result.force_history or [], warmup_steps, c.time_step,
                              2 * c.cylinder_radius, c.inlet_velocity)
    drag_error = abs(stats["mean_drag"] - REFERENCE["mean_drag"]) / REFERENCE["mean_drag"]
    st_error = abs(stats["strouhal"] - REFERENCE["strouhal"]) / REFERENCE["strouhal"]
    residuals = result.history[-1] if result.history else {}
    finite = all(math.isfinite(value) for value in (*stats.values(), drag_error, st_error))
    # A nearly zero Cl_rms means the computation relaxed to the symmetric 2-D
    # RANS state.  Such a state cannot claim to reproduce vortex shedding even
    # if its mean drag happens to look plausible.
    periodic = stats["lift_rms"] > 0.01 and 0.1 < stats["strouhal"] < 0.3
    inner_resolved = all(
        math.isfinite(residuals.get(name, math.nan)) and residuals[name] < c.tolerance
        for name in ("continuity", "momentum", "mass_imbalance", "turbulence")
    )
    return {
        **stats,
        "iterations": len(result.history),
        "final_residuals": residuals,
        "reference": {"mean_drag": REFERENCE["mean_drag"],
                      "strouhal": REFERENCE["strouhal"]},
        "drag_relative_error": drag_error,
        "strouhal_relative_error": st_error,
        "periodic_lift_resolved": periodic,
        "inner_steps_resolved": inner_resolved,
        "passed": bool(finite and periodic and inner_resolved
                       and drag_error < DRAG_ERROR_LIMIT
                       and st_error < STROUHAL_ERROR_LIMIT),
    }


def _write_report(directory: Path, summary: dict) -> None:
    case = summary["case"]
    text = f"""Re=3900 外流圆柱 SA-URANS 诊断
================================

本算例为直径 D=1 的外流圆柱，Re_D=3900；计算域为 20D × 12D，圆心在
(5D, 6D)，上下外边界为均匀来流远场而非通道壁面。使用 O 型贴体网格、
原始完全湍流 Spalart--Allmaras 闭合、隐式 Euler 时间推进和 SIMPLE 内迭代。
在初始速度中加入固定的反对称微扰，以避免精确对称初值掩盖尾迹不稳定性。

验收不是将二维 URANS 宣称为 Re=3900 的充分模型：该流动存在三维湍流尾迹，
文献通常以 LES/DES 或三维计算验证。这里要求同时解析出非零升力波动和
0.1 < St < 0.3；若未满足，程序明确报告未通过，而不是用稳态对称解冒充
涡脱落。参考时均 Cd=1.12、St=0.20，来自 {REFERENCE['citation']}。
仅在已解析周期信号、每个物理时间步内残差收敛且 Cd/St 分别严格小于
{100 * DRAG_ERROR_LIMIT:g}% / {100 * STROUHAL_ERROR_LIMIT:g}% 误差时通过。

结果
----

* 物理步数：{case['iterations']}
* 后处理样本：{case['sample_count']}
* 平均 Cd：{case['mean_drag']:.7g}（参考 {REFERENCE['mean_drag']:.4g}）
* Cl RMS：{case['lift_rms']:.7g}
* St：{case['strouhal']:.7g}（参考 {REFERENCE['strouhal']:.4g}）
* 是否解析周期升力：{case['periodic_lift_resolved']}
* 是否通过：{case['passed']}

复现::

    python -m tensorfvm.benchmark_cylinder_urans --output results/cylinder-urans

输出中的 ``forces.json`` 是逐物理时间步的 Cd/Cl 历史；它是 St 计算的原始数据。
"""
    (directory / "report.rst").write_text(text, encoding="utf-8")


def run_benchmark(directory, max_iterations=300, nx=48, ny=24, warmup_steps=100):
    """Run a bounded-cost Re=3900 URANS diagnostic; retain data on failure."""
    if warmup_steps >= max_iterations - 31:
        raise ValueError("max_iterations must leave at least 32 post-warmup samples")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config = SolverConfig(
        mesh_type="body-fitted",
        turbulence_model="spalart-allmaras",
        outer_boundary="far-field",
        body_fitted_stretching=7.0,
        nx=nx,
        ny=ny,
        length=20.0,
        height=12.0,
        cylinder_x=5.0,
        cylinder_y=6.0,
        cylinder_radius=0.5,
        inlet_velocity=1.0,
        reynolds=REFERENCE["reynolds"],
        velocity_relaxation=0.7,
        pressure_relaxation=0.3,
        turbulence_relaxation=0.5,
        sa_freestream_ratio=0.5,
        time_step=0.05,
        inner_iterations=2,
        initial_perturbation=0.05,
        tolerance=1e-2,
        max_iterations=max_iterations,
    )
    result = SimpleSolver(config).solve()
    export_result(result, directory / f"{nx}x{ny}")
    case = compare_cylinder_urans(result, warmup_steps)
    summary = {
        "benchmark": "external-cylinder-SA-URANS-Re3900",
        "reference": REFERENCE,
        "drag_error_limit": DRAG_ERROR_LIMIT,
        "strouhal_error_limit": STROUHAL_ERROR_LIMIT,
        "warmup_steps": warmup_steps,
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
        description="Re=3900 外流圆柱 SA-URANS 诊断（Cd、St 与周期升力）"
    )
    parser.add_argument("--output", type=Path, default=Path("results/cylinder-urans"))
    parser.add_argument("--max-iterations", type=int, default=300)
    parser.add_argument("--nx", type=int, default=48)
    parser.add_argument("--ny", type=int, default=24)
    parser.add_argument("--warmup-steps", type=int, default=100)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    try:
        summary = run_benchmark(args.output, args.max_iterations, args.nx, args.ny,
                                args.warmup_steps)
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, f"cylinder URANS benchmark 失败：{error}\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
