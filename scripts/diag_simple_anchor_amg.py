"""Diagnostic: SIMPLE + SST on the frozen SUBOFF mesh.

Varies three things that the archived attempts never separated:

* ``--pressure-relaxation``: 1 keeps the face-flux ledger exact (the accepted
  step is a full consistent SIMPLE step).  Values below 1 mix the whole
  triplet toward the previous state.
* ``--increment-damping``: Tikhonov anchor on the pressure increment, as a
  fraction of the median pressure-operator diagonal.  It is O(dp), so the
  fixed point is untouched; it only bounds near-null pressure levels.
* ``--linear-method``: ``direct`` (splu, iterative refinement) or ``amg``
  (pyamg smoothed aggregation on the Schur pressure operator with an SPD
  shift, gated on the unshifted true residual, falling back to splu).

No convergence claim is made here: the run prints the ledger identity, the
original-equation momentum residual and the maximum speed after every step.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from tensorfvm.native_collocated3d import NativeSteadySettings
from tensorfvm.native_mesh3d import NativeMesh3D
from tensorfvm.native_projection3d import NativeProjection3D
from tensorfvm.native_rans3d import NativeRansSettings, NativeSimpleSSTRansSolver3D
from tensorfvm.native_sst3d import inflow_from_intensity


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reynolds", type=float, default=1.2e7)
    parser.add_argument("--relaxation", type=float, default=0.5)
    parser.add_argument("--pressure-relaxation", type=float, default=1.0)
    parser.add_argument("--increment-damping", type=float, default=0.0)
    parser.add_argument("--linear-method", choices=["direct", "amg"], default="direct")
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--outer-iterations", type=int, default=1)
    parser.add_argument("--convection-order", type=int, default=1)
    parser.add_argument("--intensity", type=float, default=0.005)
    parser.add_argument("--length-scale", type=float, default=0.05)
    parser.add_argument("--pseudo-dt", type=float, default=1e-4)
    parser.add_argument("--wall-treatment", default="spalding")
    parser.add_argument("--initial", choices=["consistent", "ledger", "projected", "freestream"], default="consistent",
        help="consistent: ledger pressure Poisson seed, divergence free and on the ledger manifold "
             "(the fix for the discarded projection); ledger: ledger manifold with p=0 (large first "
             "correction); projected: external divergence-free projection (off the ledger); "
             "freestream: unprojected flux")
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
        pressure_relaxation=args.pressure_relaxation,
        pressure_increment_damping=args.increment_damping,
        pressure_velocity_algorithm="simple",
    )
    solver = NativeSimpleSSTRansSolver3D(
        mesh,
        fixed,
        direction,
        outlet,
        settings=settings,
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
    qstar = solver.velocity_flux_matrix @ solver.u.ravel() + solver.velocity_flux_source
    if args.initial == "projected":
        projection = NativeProjection3D(mesh, outlet)
        _, qstar = projection.project(qstar)
    elif args.initial == "consistent":
        # Seed the numerical pressure by the ledger pressure Poisson:
        #   div(correction p) = -div(B u + qsource)
        # so that the seed flux is both divergence free and exactly on the
        # ledger manifold.  Projecting the flux alone (--initial projected)
        # leaves the state off the manifold; leaving p=0 (--initial ledger)
        # leaves an O(1) divergence for the first correction to remove.
        for _ in range(3):
            _, _, _, correction, source_step = solver._system(qstar)
            base = solver.velocity_flux_matrix @ solver.u.ravel() + source_step
            operator = (solver.viscous.divergence_matrix @ correction).tocsc()
            rhs_seed = np.asarray(-solver.viscous.divergence_matrix @ base).ravel()
            row_magnitude = np.asarray(abs(operator).sum(axis=1)).ravel()
            # Cells whose every face is fixed have no pressure response at all;
            # pin those rows (identity) instead of deleting them, which keeps
            # the matrix square and the seed unique.
            empty = row_magnitude <= 1e-300
            operator = (sparse.diags(empty.astype(float)) +
                        sparse.diags((~empty).astype(float)) @ operator).tocsc()
            rhs_seed[empty] = 0.
            matrix = operator.tocsc()
            row_scale = np.maximum(np.asarray(abs(matrix).sum(axis=1)).ravel(), 1e-30)
            factor = splu((sparse.diags(1 / row_scale) @ matrix).tocsc())
            pressure = factor.solve(rhs_seed / row_scale)
            qstar = base + correction @ pressure
            solver.p = pressure
    elif args.initial == "ledger":
        # Put the seed on the ledger manifold the SIMPLE update maintains:
        # flux = B u + diag(alpha) pressure_flux_defect p + qsource.  An
        # externally projected flux is divergence-free but off this manifold,
        # so the first sweep starts with an O(1) consistency defect (the
        # discarded-projection root cause).
        _, _, _, correction, source_step = solver._system(qstar)
        qstar = solver.velocity_flux_matrix @ solver.u.ravel() + correction @ solver.p + source_step
    solver.initialize_state(solver.u, solver.p, qstar)

    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    start = time.perf_counter()

    def callback(row):
        speed = float(np.linalg.norm(solver.u, axis=1).max())
        record = {
            "iteration": int(row["iteration"]),
            "elapsed_seconds": time.perf_counter() - start,
            "maximum_speed": speed,
            "velocity_change": row.get("velocity_change"),
            "momentum_residual": row.get("momentum_residual"),
            "face_flux_consistency_max": row.get("face_flux_consistency_max"),
            "ledger_identity_defect_max": row.get("ledger_identity_defect_max"),
            "native_divergence_max": row.get("native_divergence_max"),
            "pressure_increment_flux_response_max": row.get("pressure_increment_flux_response_max"),
            "maximum_increment": row.get("maximum_increment"),
            "pressure_amg_shift": row.get("pressure_amg_shift"),
            "pressure_amg_levels": row.get("pressure_amg_levels"),
            "pressure_amg_iterations": row.get("pressure_amg_iterations"),
            "pressure_linear_solver": row.get("pressure_linear_solver"),
            "linear_iterations": row.get("linear_iterations"),
        }
        records.append(record)
        print(
            "it %2d t%6.1fs speed %.4f velchg %.2e mom %.3e cons %.2e div %.1e "
            "dpmax %.1e amg %s/%s"
            % (
                record["iteration"],
                record["elapsed_seconds"],
                speed,
                record["velocity_change"] or float("nan"),
                record["momentum_residual"] or float("nan"),
                record["face_flux_consistency_max"] or float("nan"),
                record["native_divergence_max"] or float("nan"),
                record["maximum_increment"] or float("nan"),
                record["pressure_linear_solver"],
                record["pressure_amg_iterations"],
            ),
            flush=True,
        )

    status = "completed"
    try:
        result = solver.solve(callback=callback)
        converged = bool(result["converged"])
    except Exception as error:  # evidence: keep the failure, do not hide it
        converged = False
        status = f"failed: {type(error).__name__}: {error}"
    summary = {
        "status": status,
        "converged": converged,
        "config": {
            "reynolds": args.reynolds,
            "relaxation": args.relaxation,
            "pressure_relaxation": args.pressure_relaxation,
            "pressure_increment_damping": args.increment_damping,
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
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
