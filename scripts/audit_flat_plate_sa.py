"""Measure the Spalart--Allmaras flat-plate drag the regression gate never measures.

The regression test ``test_flat_plate_comparison_requires_sa_convergence_and_strict_limit``
feeds ``compare_flat_plate`` a synthetic result whose wall force is *defined* as
the reference value, so it certifies the comparison code, not the solver.  This
script runs the real documented case on a chosen mesh and records the average
skin friction actually reached, together with the residual history, so the
achieved value can be placed inside the literature band from
:mod:`tensorfvm.flat_plate_reference`.

Usage::

    PYTHONPATH=src python scripts/audit_flat_plate_sa.py --nx 48 --ny 40 \
        --iterations 250 --output artifacts/flat-plate-sa/48x40.json

Nothing here changes solver behaviour; the recorded value is a *pseudo-transient
sample* unless the reported ``converged`` flag is true and every residual is
below the configured tolerance.
"""

import argparse
import json
import time
from pathlib import Path

import torch

from tensorfvm import SimpleSolver, SolverConfig
from tensorfvm.benchmark_flat_plate import (
    ERROR_LIMIT,
    REFERENCE_REYNOLDS,
    reference_skin_friction,
)
from tensorfvm.flat_plate_reference import band_membership, reference_band


def build_config(nx, ny, iterations):
    return SolverConfig(
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
        max_iterations=iterations,
    )


def measure(config, output, threads=8):
    """Solve the case and persist the physical drag record."""
    torch.set_num_threads(threads)
    started = time.time()
    result = SimpleSolver(config).solve()
    elapsed = time.time() - started
    drag = float(result.surface_forces["plate"]["x"])
    scale = 0.5 * config.density * config.inlet_velocity**2 * config.length
    skin = drag / scale
    reference = reference_skin_friction(config.reynolds)
    residuals = dict(result.history[-1]) if result.history else {}
    record = {
        "case": "turbulent-flat-plate-SA",
        "reference_reynolds": REFERENCE_REYNOLDS,
        "grid": [config.nx, config.ny],
        "iterations": len(result.history),
        "iteration_budget": config.max_iterations,
        "converged": bool(result.converged),
        "residuals": residuals,
        "all_residuals_below_tolerance": all(
            isinstance(v, (int, float)) and abs(v) < config.tolerance
            for k, v in residuals.items()
            if k != "iteration"
        ),
        "plate_drag": drag,
        "skin_friction": skin,
        "reference_schlichting": reference,
        "relative_error_vs_schlichting": abs(skin - reference) / reference,
        "inside_literature_band": bool(band_membership(skin, config.reynolds)),
        "gate_error_limit": ERROR_LIMIT,
        "elapsed_seconds": round(elapsed, 1),
        "note": "pseudo-transient sample; not a qualification unless converged",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=48)
    parser.add_argument("--ny", type=int, default=40)
    parser.add_argument("--iterations", type=int, default=250)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/flat-plate-sa/run.json")
    )
    arguments = parser.parse_args(argv)
    config = build_config(arguments.nx, arguments.ny, arguments.iterations)
    record = measure(config, arguments.output, threads=arguments.threads)
    band = reference_band(config.reynolds)
    print(
        json.dumps(
            {
                "grid": record["grid"],
                "iterations": record["iterations"],
                "converged": record["converged"],
                "skin_friction": record["skin_friction"],
                "band": band,
                "gate_error_limit": ERROR_LIMIT,
                "elapsed_seconds": record["elapsed_seconds"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
