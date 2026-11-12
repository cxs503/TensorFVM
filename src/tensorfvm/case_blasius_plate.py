"""CASE-FVM-BLASIUS-PLATE: laminar flat-plate boundary layer vs the Blasius solution.

Case definition (benchmark-pool anchor ``CASE-FVM-BLASIUS-PLATE``)
-----------------------------------------------------------------
Zero-pressure-gradient smooth flat plate, incompressible, laminar from the
leading edge.  At ``Re_x = 1e5`` the similarity solution gives the local skin
friction coefficient and the shape factor; the case asks whether this
repository's numerical devices reproduce them to better than 3 %.

Reference values (literature, *not* produced by this repository)
--------------------------------------------------------------
* Blasius, H. (1908), *Grenzschichten in Flüssigkeiten mit kleiner Reibung*,
  Z. Math. Phys. **56**, 1--37.  The similarity solution is
  ``f''' + f f'' / 2 = 0``, ``f(0) = f'(0) = 0``, ``f'(inf) = 1``; the wall
  curvature is ``f''(0) = 0.332057``, hence
  ``c_f = 2 f''(0) / sqrt(Re_x) = 0.664 / sqrt(Re_x)``.
* Schlichting, H., *Boundary-Layer Theory*, 7th ed., §7, Table 7.1
  (exact Blasius solution, flat plate at zero incidence):
  ``c_f sqrt(Re_x) = 0.664``,  ``delta*/x sqrt(Re_x) = 1.7208``,
  ``theta/x sqrt(Re_x) = 0.664``,  ``H = delta*/theta = 2.5916``.
* Secondary DOI quoted by the case sheet (task/vehicle-resistance context):
  ``10.1002/zamm.19800600419``.

The four digits above are the reference side of the comparison and are entered
as literature constants, not as numbers read off any output of this code.  The
module additionally solves the Blasius ODE here (:func:`similarity_crosscheck`)
so that the documented constants can be *audited* without a library: the ODE
must return ``2 f''(0) = 0.66411``, i.e. the tabulated ``0.664`` to 0.02 %.

Numerical devices compared against the reference
------------------------------------------------
``fvm``
    The repository's 2-D incompressible finite-volume SIMPLE solver
    (:class:`tensorfvm.body_fitted.BodyFittedSolver`) on the stretched
    ``flat-plate`` body-fitted grid, laminar closure.  This is the flagship
    device: local wall shear is taken from the solver's own no-slip face force
    (the same reconstruction the drag benchmark uses), so the comparison
    measures the FVM discretisation, not a post-processing model.
``march``
    The thin-shear-layer boundary-layer march with the laminar closure
    (:func:`tensorfvm.rans_ship_resistance.march_boundary_layer`,
    ``model='laminar'``), the affordable rung already anchored in
    ``docs/RANS_verification.md``.

Both are compared with the same reference and the same 3 % gate, so the report
states which rung of the device ladder actually delivers the case.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np

__all__ = [
    "CASE_ID", "LITERATURE", "ERROR_LIMIT", "REFERENCE_RE_X",
    "blasius_reference", "similarity_crosscheck", "boundary_layer_integrals",
    "wall_shear_profile", "compare_against_reference", "run_fvm_case",
    "run_march_case", "run_march_sensitivity", "FVM_CONFIGS",
    "MARCH_SENSITIVITY",
]

CASE_ID = "CASE-FVM-BLASIUS-PLATE"

#: Gate of the case sheet: relative error below 3 % on ``c_f`` and ``H``.
ERROR_LIMIT = 0.03

#: Station of the comparison.
REFERENCE_RE_X = 1.0e5

#: Literature constants (see the module docstring for provenance).  Every value
#: is normalised exactly as printed in the reference so the report can quote the
#: table column without rescaling.
LITERATURE = {
    "cf_sqrt_re_x": 0.664,        # Blasius 1908 / Schlichting §7 Table 7.1
    "delta_star_sqrt_re_x": 1.7208,
    "theta_sqrt_re_x": 0.664,
    "shape_factor": 2.5916,       # H = delta*/theta = 1.7208 / 0.664
    "wall_curvature": 0.332057,   # f''(0)
}


# ---------------------------------------------------------------------------
# reference side
# ---------------------------------------------------------------------------
def blasius_reference(re_x: float = REFERENCE_RE_X) -> Dict[str, float]:
    """Blasius similarity solution at ``re_x`` from the literature constants."""
    re_x = float(re_x)
    if not math.isfinite(re_x) or re_x <= 0.0:
        raise ValueError("re_x must be finite and positive")
    root = math.sqrt(re_x)
    return {
        "re_x": re_x,
        "cf": LITERATURE["cf_sqrt_re_x"] / root,
        "delta_star": LITERATURE["delta_star_sqrt_re_x"] / root,
        "theta": LITERATURE["theta_sqrt_re_x"] / root,
        "H": LITERATURE["shape_factor"],
    }


def similarity_crosscheck(eta_max: float = 12.0, n: int = 1201) -> Dict[str, float]:
    """Solve the Blasius ODE in this repository and audit the table digits.

    This is an *audit* of the reference constants, not a replacement for them:
    it answers "can the documented digits be reproduced from first principles
    here" (``f''(0)``, and the integral thicknesses normalised in the same way
    as the table) so a mis-transcribed constant cannot pass unnoticed.
    """
    from .rans_ship_resistance.anchors import blasius_solution

    solution = blasius_solution(eta_max=eta_max, n=n)
    eta = np.asarray(solution["eta"], dtype=float)
    fp = np.asarray(solution["fp"], dtype=float)
    fpp0 = float(solution["fpp0"])
    # delta*/x sqrt(Re_x) = int (1 - f') d eta ; theta/x sqrt(Re_x) = int f'(1-f') d eta
    one_minus = np.clip(1.0 - fp, 0.0, None)
    delta_star = float(np.trapezoid(one_minus, eta))
    theta = float(np.trapezoid(fp * one_minus, eta))
    return {
        "fpp0": fpp0,
        "two_fpp0": 2.0 * fpp0,
        "cf_sqrt_re_x_literature": LITERATURE["cf_sqrt_re_x"],
        "cf_constant_relative_deviation":
            abs(2.0 * fpp0 - LITERATURE["cf_sqrt_re_x"]) / LITERATURE["cf_sqrt_re_x"],
        "delta_star_sqrt_re_x": delta_star,
        "theta_sqrt_re_x": theta,
        "H": delta_star / theta,
        "shape_factor_literature": LITERATURE["shape_factor"],
        "shape_factor_relative_deviation":
            abs(delta_star / theta - LITERATURE["shape_factor"])
            / LITERATURE["shape_factor"],
    }


# ---------------------------------------------------------------------------
# measured side: shared post-processing
# ---------------------------------------------------------------------------
def boundary_layer_integrals(y, u, u_e: Optional[float] = None,
                             truncate_at_delta_99: bool = False) -> Dict[str, float]:
    """Integral thicknesses of a wall-normal profile.

    ``y`` must start at the wall (``y[0] == 0`` with ``u[0] == 0``) and increase;
    the profile is treated as piecewise linear between nodes, which is what the
    trapezoidal rule on the half-cell start assumes.  By default the whole
    supplied profile is integrated, i.e. the classical definitions
    ``delta* = int (1 - u/U_e) dy`` and ``theta = int (u/U_e)(1 - u/U_e) dy`` over
    the full domain -- truncating at ``delta_99`` instead drops the slowly
    decaying tail and biases ``theta`` low by ~0.9 % even on the exact Blasius
    profile (see ``tests/test_case_blasius_plate.py``), so truncation is kept only
    as an explicit diagnostic.

    ``u_e`` defaults to the outermost node, i.e. the local edge velocity; on a
    monotone boundary-layer profile that is also ``max(u)``.  Normalising with a
    fixed freestream value instead would let the outer-flow displacement
    correction (a few tenths of a percent over the domain height) enter
    ``delta*`` directly.
    """
    y = np.asarray(y, dtype=float)
    u = np.asarray(u, dtype=float)
    if y.ndim != 1 or y.shape != u.shape or y.size < 3:
        raise ValueError("y and u must be 1-D arrays of equal length >= 3")
    if np.any(np.diff(y) <= 0):
        raise ValueError("y must be strictly increasing")
    if u_e is None:
        u_e = float(u.max())
    u_e = float(u_e)
    if not (math.isfinite(u_e) and u_e > 0.0):
        raise ValueError("u_e must be finite and positive")
    ratio = np.clip(u / u_e, 0.0, 1.0)

    # delta_99 by linear interpolation on the last crossing of 0.99
    target = 0.99
    below = np.nonzero(ratio < target)[0]
    if below.size == 0 or below[-1] + 1 >= y.size:
        raise ValueError("profile does not reach 0.99 u_e inside the domain")
    i = int(below[-1])
    w = (target - ratio[i]) / (ratio[i + 1] - ratio[i])
    delta_99 = float(y[i] + w * (y[i + 1] - y[i]))

    if truncate_at_delta_99:
        yy = np.concatenate([y[: i + 1], [delta_99]])
        rr = np.concatenate([ratio[: i + 1], [target]])
    else:
        yy, rr = y, ratio
    delta_star = float(np.trapezoid(1.0 - rr, yy))
    theta = float(np.trapezoid(rr * (1.0 - rr), yy))
    if theta <= 0.0:
        raise ValueError("non-positive momentum thickness")
    return {
        "delta_99": delta_99,
        "delta_star": delta_star,
        "theta": theta,
        "H": delta_star / theta,
        "u_e": u_e,
        "truncated_at_delta_99": bool(truncate_at_delta_99),
        "nodes_inside_delta_99": int(i + 1),
    }


def wall_shear_profile(solver) -> Dict[str, np.ndarray]:
    """Local skin-friction coefficient on every no-slip wall face.

    The face force is taken from the solver's own :meth:`_surface_force`
    reconstruction (one call per face); the per-face values are therefore
    exactly the integrand of the plate drag the drag benchmark reports, and
    their sum reproduces the whole-plate force.  ``c_f`` is normalised with
    ``rho U_inf^2 / 2``.
    """
    import torch

    mesh = solver.mesh
    mask = mesh.masks["wall"]
    index = torch.nonzero(mask).ravel()
    if index.numel() == 0:
        raise ValueError("mesh has no no-slip wall faces")
    forces = []
    for k in index.tolist():
        single = torch.zeros_like(mask)
        single[k] = True
        forces.append(float(solver._surface_force(single)[0]))
    forces = torch.tensor(forces, dtype=torch.float64)
    x = mesh.face_centers[index, 0].clone()
    length = mesh.face_lengths[index].clone()
    q = 0.5 * solver.config.density * solver.config.inlet_velocity ** 2
    cf = forces.abs() / (q * length)
    order = torch.argsort(x)
    total = float(solver._surface_force(mask)[0])
    return {
        "x": x[order].detach().cpu().numpy(),
        "cf": cf[order].detach().cpu().numpy(),
        "force_per_face": forces[order].detach().cpu().numpy(),
        "sum_of_face_forces": float(forces.sum()),
        "whole_plate_force": total,
        "force_sum_closure": abs(float(forces.sum()) - total) / max(abs(total), 1e-30),
    }


def compare_against_reference(got_cf: float, got_H: float,
                              re_x: float = REFERENCE_RE_X,
                              limit: float = ERROR_LIMIT) -> Dict[str, object]:
    """Relative errors of ``c_f`` and ``H`` against the Blasius values."""
    ref = blasius_reference(re_x)
    out = {"re_x": float(re_x), "error_limit": float(limit),
           "reference": ref, "got": {"cf": float(got_cf), "H": float(got_H)}}
    for key, value in (("cf", got_cf), ("H", got_H)):
        rel = (float(value) - ref[key]) / ref[key]
        out[key] = {"relative_error": rel, "abs_relative_error": abs(rel),
                    "passed": bool(abs(rel) < limit)}
    out["passed"] = bool(out["cf"]["passed"] and out["H"]["passed"])
    return out


# ---------------------------------------------------------------------------
# measured side: the two devices
# ---------------------------------------------------------------------------
#: Grid/solver settings for the FVM device.  ``wall-*``/``stream-*`` are the
#: one-at-a-time variations of the convergence/sensitivity study.
FVM_CONFIGS: Dict[str, Dict[str, object]] = {
    "fvm-baseline": dict(nx=64, ny=56),
    "fvm-wall-coarse": dict(nx=64, ny=40, velocity_relaxation=0.5,
                            pseudo_time_step=0.05),
    "fvm-stream-coarse": dict(nx=32, ny=56, velocity_relaxation=0.5,
                              pseudo_time_step=0.05),
    "fvm-relaxation": dict(nx=64, ny=56, velocity_relaxation=0.5,
                            pseudo_time_step=0.05),
}


def run_fvm_case(nx: int = 64, ny: int = 84, height: float = 0.15,
                 stretching: float = 4.0, reynolds: float = REFERENCE_RE_X,
                 max_iterations: int = 1200, tolerance: float = 1e-5,
                 velocity_relaxation: float = 0.3,
                 pressure_relaxation: float = 0.2,
                 pseudo_time_step: Optional[float] = 0.02,
                 trace_every: int = 25, threads: int = 2,
                 length: float = 1.0, progress: bool = True,
                 trace_path=None, save_field: bool = True) -> Dict[str, object]:
    """Solve the laminar flat plate with the 2-D FVM and measure ``c_f``/``H``.

    The plate occupies ``0 <= x <= length`` of the ``flat-plate`` body-fitted
    grid (``y = 0`` is the no-slip wall); ``U_inf = 1`` and ``Re_L = reynolds``,
    so ``Re_x = x reynolds / length``.  The comparison station is the trailing
    column, ``Re_x = reynolds (1 - dx/2)``.
    """
    import json
    import time
    from pathlib import Path

    import torch

    from .solver import SimpleSolver, SolverConfig

    torch.set_num_threads(int(threads))
    config = SolverConfig(
        nx=nx, ny=ny, length=length, height=height,
        cylinder_radius=None, inlet_velocity=1.0, reynolds=reynolds,
        mesh_type="flat-plate", turbulence_model="laminar",
        velocity_relaxation=velocity_relaxation,
        pressure_relaxation=pressure_relaxation,
        pseudo_time_step=pseudo_time_step, flat_plate_stretching=stretching,
        tolerance=tolerance, max_iterations=max_iterations,
    )
    started = time.time()
    solver = SimpleSolver(config)
    trace = []
    converged_at = None
    for it in range(1, max_iterations + 1):
        metrics = solver.step()
        if it % trace_every == 0 or it == 1 or solver.converged:
            profile = wall_shear_profile(solver)
            column = _trailing_column(solver)
            entry = {
                "iteration": it,
                "cf_trailing": float(profile["cf"][-1]),
                "H_trailing": float(column["H"]),
                "continuity": metrics["continuity"],
                "momentum": metrics["momentum"],
                "elapsed_s": time.time() - started,
            }
            trace.append(entry)
            if trace_path is not None:
                Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
                with Path(trace_path).open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(entry) + "\n")
            if progress:
                print(f"it={it:5d} cf={entry['cf_trailing']:.6e} "
                      f"H={entry['H_trailing']:.5f} mom={metrics['momentum']:.3e} "
                      f"t={entry['elapsed_s']:.0f}s", flush=True)
        if solver.converged:
            converged_at = it
            break

    profile = wall_shear_profile(solver)
    column = _trailing_column(solver)
    x_trailing = float(profile["x"][-1])
    re_x = x_trailing * reynolds / length
    nu = config.viscosity / config.density
    comparison = compare_against_reference(float(profile["cf"][-1]), column["H"], re_x)
    # whole-plate average, an independent cross-check on the same shear field
    plate_cf = abs(profile["whole_plate_force"]) / (
        0.5 * config.density * config.inlet_velocity ** 2 * length)
    reference_plate = 2.0 * LITERATURE["cf_sqrt_re_x"] / math.sqrt(re_x)
    return {
        "case": CASE_ID,
        "device": "fvm",
        "config": {
            "nx": nx, "ny": ny, "length": length, "height": height,
            "stretching": stretching, "reynolds": reynolds,
            "inlet_velocity": 1.0, "kinematic_viscosity": nu,
            "turbulence_model": "laminar",
            "velocity_relaxation": velocity_relaxation,
            "pressure_relaxation": pressure_relaxation,
            "pseudo_time_step": pseudo_time_step,
            "tolerance": tolerance, "max_iterations": max_iterations,
            "station": "trailing wall face (last column)",
        },
        "grid": {
            "cells": int(nx * ny),
            "first_cell_height": float(solver.mesh.centers[:, 1].min()),
            "nodes_inside_delta_99": column["nodes_inside_delta_99"],
            "wall_faces": int(profile["cf"].size),
            "force_sum_closure": profile["force_sum_closure"],
        },
        "run": {
            "iterations": len(solver.history),
            "converged": bool(solver.converged),
            "converged_at": converged_at,
            "elapsed_s": time.time() - started,
            "final_residuals": dict(solver.history[-1]),
        },
        "measured": {
            "x_trailing": x_trailing,
            "re_x": re_x,
            "cf": float(profile["cf"][-1]),
            "H": column["H"],
            "delta_star": column["delta_star"],
            "theta": column["theta"],
            "delta_99": column["delta_99"],
            "u_e": column["u_e"],
            "u_e_over_u_inf": column["u_e"] / config.inlet_velocity,
            "plate_mean_cf": plate_cf,
            "plate_mean_cf_reference": reference_plate,
            "plate_mean_relative_error": (plate_cf - reference_plate) / reference_plate,
        },
        "comparison": comparison,
        "trace": trace,
        "profile_trailing": _profile_export(solver),
        "cf_distribution": {"x": profile["x"].tolist(),
                            "cf": profile["cf"].tolist()},
        "field": _field_export(solver) if save_field else None,
    }


def _profile_export(solver) -> Dict[str, list]:
    """Wall-normal profile at the trailing column, with the Blasius comparison."""
    mesh = solver.mesh
    nx, ny = solver.config.nx, solver.config.ny
    index = np.arange(nx - 1, nx * ny, nx)
    y = mesh.centers[:, nx - 1, 1].detach().cpu().numpy()
    u = solver.velocity[index, 0].detach().cpu().numpy()
    order = np.argsort(y)
    y, u = y[order], u[order]
    u_e = float(max(u.max(), 1e-30))
    nu = solver.config.viscosity / solver.config.density
    x = float(mesh.centers[0, nx - 1, 0])
    eta = y * math.sqrt(u_e / (nu * x))
    from .rans_ship_resistance.anchors import blasius_solution

    reference = blasius_solution(eta_max=float(eta.max()), n=401)
    blasius = np.interp(eta, np.asarray(reference["eta"]),
                        np.asarray(reference["fp"]))
    return {"y": y.tolist(), "u": u.tolist(),
            "eta": eta.tolist(),
            "u_over_ue": (u / u_e).tolist(),
            "blasius_u_over_ue": blasius.tolist(),
            "u_e": u_e, "x": x}


def _field_export(solver) -> Dict[str, list]:
    """Cell-centred velocity magnitude and coordinates for a field plot."""
    mesh = solver.mesh
    nx, ny = solver.config.nx, solver.config.ny
    velocity = solver.velocity.detach().cpu().numpy()
    speed = np.sqrt(velocity[:, 0] ** 2 + velocity[:, 1] ** 2).reshape(ny, nx)
    return {"x": mesh.centers[0, :, 0].detach().cpu().numpy().tolist(),
            "y": mesh.centers[:, 0, 1].detach().cpu().numpy().tolist(),
            "speed": speed.tolist()}


def _trailing_column(solver) -> Dict[str, float]:
    """Boundary-layer integrals of the wall-normal profile at ``i = nx - 1``."""
    import torch

    mesh = solver.mesh
    centers = mesh.centers            # (ny, nx, 2)
    nx = solver.config.nx
    ny = solver.config.ny
    index = torch.arange(nx - 1, nx * ny, nx)   # flat cell index, column i = nx - 1
    y = centers[:, nx - 1, 1].detach().cpu().numpy()
    u = solver.velocity[index, 0].detach().cpu().numpy()
    order = np.argsort(y)
    y = np.concatenate([[0.0], y[order]])
    u = np.concatenate([[0.0], u[order]])
    return boundary_layer_integrals(y, u)


def run_march_case(stations: int = 401, length: float = 1.0,
                   reynolds: float = REFERENCE_RE_X, ny: int = 256,
                   stretching: float = 8.0) -> Dict[str, object]:
    """Laminar thin-shear-layer march on the same plate (affordable rung)."""
    from .rans_ship_resistance.boundary_layer import march_boundary_layer

    x = np.linspace(length / stations, length, stations)
    ue = np.ones_like(x)
    nu = length * 1.0 / reynolds
    result = march_boundary_layer(x, ue, nu, model="laminar", trip_x=None,
                                  ny=ny, stretching=stretching)
    cf = float(np.asarray(result.cf)[-1])
    H = float(np.asarray(result.H)[-1])
    re_x = float(x[-1]) * reynolds / length
    comparison = compare_against_reference(cf, H, re_x)
    return {
        "case": CASE_ID,
        "device": "march",
        "config": {"stations": stations, "length": length, "reynolds": reynolds,
                   "kinematic_viscosity": nu, "ny": ny, "stretching": stretching,
                   "model": "laminar", "trip_x": None},
        "measured": {
            "x_trailing": float(x[-1]), "re_x": re_x, "cf": cf, "H": H,
            "delta_star": float(np.asarray(result.delta_star)[-1]),
            "theta": float(np.asarray(result.theta)[-1]),
            "continuity_residual": float(result.continuity_residual),
        },
        "comparison": comparison,
    }

#: Wall-normal and streamwise resolution variants of the marching rung.
MARCH_SENSITIVITY = {
    "ny-128": dict(ny=128, stations=401),
    "ny-256": dict(ny=256, stations=401),
    "ny-512": dict(ny=512, stations=401),
    "stations-101": dict(ny=256, stations=101),
    "stations-1601": dict(ny=256, stations=1601),
}


def run_march_sensitivity() -> Dict[str, object]:
    """One-at-a-time resolution study of the marching rung.

    Reported as relative changes of ``c_f`` and ``H`` against the reference
    configuration, so the report can state the numerical uncertainty of the rung
    rather than a single number.
    """
    rows = []
    for tag, options in MARCH_SENSITIVITY.items():
        result = run_march_case(**options)
        rows.append({
            "tag": tag,
            **{k: options[k] for k in ("ny", "stations")},
            "cf": result["measured"]["cf"],
            "H": result["measured"]["H"],
            "cf_relative_error": result["comparison"]["cf"]["relative_error"],
            "H_relative_error": result["comparison"]["H"]["relative_error"],
            "passed": result["comparison"]["passed"],
        })
    base = rows[1]  # ny-256, 401 stations
    for row in rows:
        row["cf_change_vs_base"] = (row["cf"] - base["cf"]) / base["cf"]
        row["H_change_vs_base"] = (row["H"] - base["H"]) / base["H"]
    return {"case": CASE_ID, "device": "march", "kind": "sensitivity",
            "rows": rows, "reference_tag": base["tag"]}
