#!/usr/bin/env python3
"""TMR 2D zero-pressure-gradient flat plate: one grid level per invocation.

Case definition: NASA Turbulence Modeling Resource "2DZP: 2D Zero Pressure
Gradient Flat Plate Validation Case", Re = 5 million per unit length, official
nested PLOT3D grid, Spalart--Allmaras turbulence model.  Quantities of interest
per TMR: Cf versus Re_theta (Karman--Schoenherr theory) and u+ versus y+ at
Re_theta = 10000 (Coles law of the wall).

Usage::

    python3 validation_tmr/run_flatplate.py --grid 69x49 --max-iterations 6000
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
from validation_tmr.postprocess import flat_plate_metrics  # noqa: E402

GRIDS = {
    "35x25": "flatplate_clust2_4levelsdown_35x25.p2dfmt",
    "69x49": "flatplate_clust2_3levelsdown_69x49.p2dfmt",
    "137x97": "flatplate_clust2_2levelsdown_137x97.p2dfmt",
    "273x193": "flatplate_clust2_1leveldown_273x193.p2dfmt",
}
REFERENCE_DIR = REPO / "validation_tmr/reference/tmr"
VISCOSITY = 2e-7  # U=1, Re=5e6 per unit length


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", required=True, choices=sorted(GRIDS))
    parser.add_argument("--output", type=Path, default=REPO / "validation_tmr/runs/flatplate")
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
    label = f"flatplate-{args.grid}"
    solver = build_solver(grid, VISCOSITY, wall_at_top=False)
    settings = dict(
        case="NASA TMR 2DZP flat plate validation",
        grid=args.grid, grid_file=str(grid.relative_to(REPO)),
        grid_sha256=sha256(grid), cells=int(solver.count),
        zones=[list(map(int, z)) for z in solver.mesh.zone_dims],
        extent=list(map(float, solver.mesh.extent)),
        reynolds_per_unit_length=1.0 / VISCOSITY,
        inlet_velocity=1.0, density=1.0, viscosity=VISCOSITY,
        turbulence_model="spalart-allmaras", sa_freestream_ratio=3.0,
        wall_bc="no-slip bottom plate", top_bc=f"grid_top_patch={solver.config.grid_top_patch}",
        inlet_bc="uniform U=1", outlet_bc="pressure reference",
        max_iterations=args.max_iterations, tolerance=args.tolerance,
        relaxation=dict(velocity=solver.config.velocity_relaxation,
                        pressure=solver.config.pressure_relaxation,
                        turbulence=solver.config.turbulence_relaxation),
        linear_solver="LU-preconditioned (validation_tmr/linear.py)",
        references={name: dict(file=name, sha256=sha256(REFERENCE_DIR / name))
                    for name in ("FlatPlate_validation_cf_K-S.dat",
                                 "FlatPlate_validation_u+y+.dat")},
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
    metrics = flat_plate_metrics(fields, wall, REFERENCE_DIR, VISCOSITY)
    write_json(output / f"{label}-metrics.json", dict(state=state, metrics=metrics))
    print(json.dumps({k: v for k, v in metrics.items() if not isinstance(v, (list, dict))},
                     indent=2))


if __name__ == "__main__":
    main()