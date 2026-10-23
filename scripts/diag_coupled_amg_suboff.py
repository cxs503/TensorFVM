"""Diagnostic: monolithic (u, p, flux) solve on the frozen SUBOFF mesh.

The archived SUBOFF attempts all used ``algorithm: simple`` (segregated).  This
script runs the coupled Newton-Krylov path on the same mesh, same Reynolds
number and same wall treatment, with either a direct sparse factorization or
the block-AMG preconditioner (ILU velocity block + smoothed-aggregation AMG on
the SPD-shifted Schur pressure block).

Nothing is clipped and no reference value is fitted: the printed ledger
identity, the original-equation momentum residual and the maximum speed are the
evidence.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from tensorfvm.native_collocated3d import NativeSteadySettings
from tensorfvm.native_mesh3d import NativeMesh3D
from tensorfvm.native_projection3d import NativeProjection3D
from tensorfvm.native_rans3d import NativeRansSettings, NativeSSTRansSolver3D
from tensorfvm.native_sst3d import inflow_from_intensity


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reynolds", type=float, default=1.2e7)
    parser.add_argument("--relaxation", type=float, default=0.5)
    parser.add_argument("--linear-method", choices=["direct", "amg"], default="direct")
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--outer-iterations", type=int, default=1)
    parser.add_argument("--convection-order", type=int, default=1)
    parser.add_argument("--intensity", type=float, default=0.005)
    parser.add_argument("--length-scale", type=float, default=0.05)
    parser.add_argument("--pseudo-dt", type=float, default=1e-4)
    parser.add_argument("--wall-treatment", default="spalding")
    args = parser.parse_args(argv)

    mesh = NativeMesh3D.from_gmsh(args.mesh, triangulate_faces=True)
    fixed = (mesh.neighbor < 0) & (mesh.patch_labels != "outlet")
    outlet = mesh.patch_labels == "outlet"
    direction = np.tile([1.0, 0.0, 0.0], (len(mesh.owner), 1))
    direction[mesh.patch_labels == "wall"] = 0.0
    settings = NativeSteadySettings(
        viscosity=1 / args.reynolds,
        max_iterations=args.iterations,
        nonlinear_tolerance=1e-8,
        relaxation=args.relaxation,
        linear_method=args.linear_method,
        bounded_convection=True,
        convection_order=args.convection_order,
    )
    solver = NativeSSTRansSolver3D(
        mesh, fixed, direction, outlet, settings=settings,
        rans_settings=NativeRansSettings(
            max_outer_iterations=args.outer_iterations,
            pseudo_time_step=args.pseudo_dt,
            nonlinear_tolerance=1e-7,
            wall_treatment=args.wall_treatment,
        ),
    )
    solver.u[:, 0] = 1.0
    k0, omega0 = inflow_from_intensity(1.0, args.intensity, args.length_scale)
    solver.k = np.full(solver.N, k0)
    solver.omega = np.full(solver.N, omega0)
    solver.k_boundary = np.full(solver.F, k0)
    solver.omega_boundary = np.full(solver.F, omega0)
    solver._update_momentum_coefficients(solver.current_closure())
    projection = NativeProjection3D(mesh, outlet)
    _, projected = projection.project(
        solver.velocity_flux_matrix @ solver.u.ravel() + solver.velocity_flux_source)
    solver.initialize_state(solver.u, solver.p, projected)

    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    start = time.perf_counter()

    def callback(row):
        record = {
            "iteration": int(row["iteration"]),
            "elapsed_seconds": time.perf_counter() - start,
            "maximum_speed": float(np.linalg.norm(solver.u, axis=1).max()),
            "velocity_change": row.get("velocity_change"),
            "momentum_residual": row.get("momentum_residual"),
            "face_flux_consistency_max": row.get("face_flux_consistency_max"),
            "native_divergence_max": row.get("native_divergence_max"),
            "true_linear_relative_residual": row.get("true_linear_relative_residual"),
            "linear_iterations": row.get("linear_iterations"),
            "amg_operator_complexity": getattr(solver, "amg_operator_complexity", None),
            "amg_pressure_shift": getattr(solver, "amg_pressure_shift", None),
            "amg_levels": getattr(solver, "amg_levels", None),
        }
        records.append(record)
        print(
            "it %2d t%6.1fs speed %.4f velchg %.2e mom %.3e cons %.2e div %.1e "
            "lin %.1e li %s amg %s/%s"
            % (
                record["iteration"], record["elapsed_seconds"], record["maximum_speed"],
                record["velocity_change"] or float("nan"),
                record["momentum_residual"] or float("nan"),
                record["face_flux_consistency_max"] or float("nan"),
                record["native_divergence_max"] or float("nan"),
                record["true_linear_relative_residual"] or float("nan"),
                record["linear_iterations"], record["amg_levels"],
                record["amg_operator_complexity"],
            ),
            flush=True,
        )

    status = "completed"
    try:
        result = solver.solve(callback=callback)
        converged = bool(result["converged"])
    except Exception as error:  # keep the failure as evidence
        converged = False
        status = f"failed: {type(error).__name__}: {error}"
    forces = {}
    try:
        wall = solver.forces("wall")
        # The runner reports forces on the wall patch; no experimental value is
        # fitted or compared here.
        forces = {
            "pressure_force": wall["pressure_force"].tolist(),
            "viscous_force": wall["viscous_force"].tolist(),
            "total_force": wall["total_force"].tolist(),
            "drag_force_x": float(wall["total_force"][0]),
        }
        np.savez(args.output / "fields.npz", velocity=solver.u, pressure=solver.p,
                 flux=solver.flux, k=solver.k if solver.k is not None else np.array([]),
                 omega=solver.omega if solver.omega is not None else np.array([]))
    except Exception as error:  # forces are optional evidence
        forces = {"error": f"{type(error).__name__}: {error}"}
    summary = {
        "status": status,
        "forces": forces,
        "converged": converged,
        "config": {
            "reynolds": args.reynolds,
            "relaxation": args.relaxation,
            "linear_method": args.linear_method,
            "iterations": args.iterations,
            "outer_iterations": args.outer_iterations,
            "convection_order": args.convection_order,
            "wall_treatment": args.wall_treatment,
            "cells": int(solver.N),
        },
        "records": records,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
