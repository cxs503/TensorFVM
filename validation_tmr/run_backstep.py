#!/usr/bin/env python3
"""TMR 2D backward-facing step: one grid level per invocation.

Case definition: NASA Turbulence Modeling Resource "2DBFS: 2D Backward Facing
Step" validation case (Driver & Seegmiller 1985 experiment, Re_H = 36000 based
on step height, expansion ratio 9/8, M = 0.128 treated as incompressible),
official nested PLOT3D grid, Spalart--Allmaras turbulence model.

Usage::

    python3 validation_tmr/run_backstep.py --grid 4levdn --max-iterations 6000
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from validation_tmr.common import (build_solver, environment, restore, run,  # noqa: E402
                                  save_state, sha256, write_json)
from validation_tmr.postprocess import backstep_metrics  # noqa: E402

GRIDS = {
    "5levdn": "backstep5_5levdn_derived.p2dfmt",
    "4levdn": "backstep5_4levdn.p2dfmt",
    "3levdn": "backstep5_3levdn.p2dfmt",
    "2levdn": "backstep5_2levdn.p2dfmt",
}
REFERENCE_DIR = REPO / "validation_tmr/reference/tmr"
STEP_HEIGHT = 1.0
REYNOLDS_STEP = 36000.0
VISCOSITY = 1.0 * STEP_HEIGHT / REYNOLDS_STEP


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", required=True, choices=sorted(GRIDS))
    parser.add_argument("--output", type=Path, default=REPO / "validation_tmr/runs/backstep")
    parser.add_argument("--max-iterations", type=int, default=6000)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--restart", action="store_true",
                        help="continue from the checkpointed field of this label")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    grid = REFERENCE_DIR / "grids" / GRIDS[args.grid]
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    label = f"backstep-{args.grid}"
    solver = build_solver(grid, VISCOSITY, wall_at_top=True)
    settings = dict(
        case="NASA TMR 2DBFS backward facing step validation",
        grid=args.grid, grid_file=str(grid.relative_to(REPO)),
        grid_sha256=sha256(grid), cells=int(solver.count),
        zones=[list(map(int, z)) for z in solver.mesh.zone_dims],
        extent=list(map(float, solver.mesh.extent)),
        step_height=STEP_HEIGHT, reynolds_step=REYNOLDS_STEP,
        inlet_velocity=1.0, density=1.0, viscosity=VISCOSITY,
        turbulence_model="spalart-allmaras", sa_freestream_ratio=3.0,
        wall_bc="no-slip bottom wall, step face and upper channel wall",
        top_bc=f"grid_top_patch={solver.config.grid_top_patch}",
        inlet_bc="uniform U=1 at x/H=-130 (TMR adjusted inflow length)",
        outlet_bc="pressure reference",
        max_iterations=args.max_iterations, tolerance=args.tolerance,
        relaxation=dict(velocity=solver.config.velocity_relaxation,
                        pressure=solver.config.pressure_relaxation,
                        turbulence=solver.config.turbulence_relaxation),
        linear_solver="LU-preconditioned (validation_tmr/linear.py)",
        references={name: dict(file=name, sha256=sha256(REFERENCE_DIR / name))
                    for name in ("Backstep_validation_cf.exp.dat",
                                 "Backstep_validation_cp.expnew.dat",
                                 "Backstep_validation_profiles.exp.dat")},
        environment=environment())
    if args.restart:
        settings["restart"] = restore(solver, output / f"{label}-fields.npz")
    write_json(output / f"{label}-settings.json", settings)
    history = run(solver, output, label, args.max_iterations, args.tolerance)
    state = save_state(solver, output, label, history)
    with np.load(output / f"{label}-fields.npz") as handle:
        fields = {name: handle[name] for name in handle.files}
    with np.load(output / f"{label}-wall.npz") as handle:
        wall = {name: handle[name] for name in handle.files}
    metrics = backstep_metrics(fields, wall, REFERENCE_DIR, VISCOSITY)
    write_json(output / f"{label}-metrics.json", dict(state=state, metrics=metrics))
    print(json.dumps({k: v for k, v in metrics.items()
                      if not isinstance(v, (list, dict))}, indent=2))


if __name__ == "__main__":
    main()