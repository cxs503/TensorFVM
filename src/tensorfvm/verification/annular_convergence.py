"""Grid-convergence (observed order / Richardson / GCI) verification for the annular case.

The annular backend in :mod:`tensorfvm.annular` already compares one grid against
the closed-form annular Poiseuille solution inside a 3% band. A single-resolution
comparison cannot distinguish "the discretisation is second order and 3% is
loose" from "the scheme is first order and 3% was met by accident". This module
supplies the missing discretisation-error apparatus:

* :func:`observed_order` - triple-grid order estimate with Richardson
  extrapolation and the Celik / ASME V&V 20 grid convergence index
  ``GCI = Fs |(f_fine - f_medium)/f_fine| / (r**p - 1)``.
* :func:`loglog_order` - least-squares order estimate from an error sequence
  (no exact value needed), used as an independent cross-check.
* :func:`analytic_reference_check` - validates the *analytic* baseline itself:
  the closed form is compared against an independent second-order finite
  difference solution of the differential statement
  ``d/dr (r du/dr) = -(G/mu) r`` on a refinement sequence, so a typo or a lost
  factor in the formula cannot survive, and the observed order of the FD
  differences proves that the residual gap is truncation, not formula error.
* :func:`study` - runs the production annular solve over a refinement sequence
  and reports, per functional, the observed order, the Richardson value, the
  GCI and whether the closed-form value lies inside the extrapolated band.

Everything reported here is a *verification* quantity: discretisation error of
the discretisation against an exact solution. It is not physical validation and
says nothing about general 3-D Navier-Stokes accuracy.

Wall-clock note: the sparse solves are cheap (a few hundred unknowns per level
in the default sequence) but multithreaded BLAS thrashes on a heavily loaded
host. Callers should pin threads (``torch.set_num_threads(1)`` plus one BLAS
thread) before calling :func:`study`; the numbers are unaffected either way.
"""
from __future__ import annotations

from dataclasses import asdict
import math

import numpy as np

from ..annular import AnnularConfig, analytic_flow, analytic_velocity, solve
from .engineering_audit import annular_metrics

__all__ = [
    "observed_order",
    "loglog_order",
    "richardson_extrapolate",
    "analytic_reference_check",
    "study",
]


def richardson_extrapolate(values, ratio, order):
    """Richardson extrapolation from the two finest of at least three values."""
    values = np.asarray(values, dtype=float)
    f_medium, f_fine = values[-2], values[-1]
    return float(f_fine + (f_fine - f_medium) / (ratio ** order - 1.0))


def observed_order(values, ratio, safety=1.25, asymptotic_tolerance=0.25):
    """Triple-grid observed order, Richardson value and fine-grid GCI.

    ``values`` holds at least three solutions of the same functional on
    successively refined grids, ordered coarse -> fine, with uniform refinement
    ratio ``ratio``. A non-monotone sequence or a round-off plateau carries no
    order information, so it raises instead of returning a spurious exponent.
    """
    values = np.asarray(values, dtype=float)
    if values.size < 3:
        raise ValueError("at least three grid levels are required")
    if not np.all(np.isfinite(values)):
        raise ValueError("non-finite functional value on a grid level")
    if not ratio > 1.0:
        raise ValueError("refinement ratio must exceed one")
    coarse, medium, fine = values[-3], values[-2], values[-1]
    medium_coarse = float(medium - coarse)
    fine_medium = float(fine - medium)
    scale = max(abs(float(coarse)), abs(float(medium)), abs(float(fine)), 1e-300)
    if abs(medium_coarse) <= 1e-13 * scale:
        raise ValueError("change between the two coarsest levels is at round-off level; order is not observable")
    if fine_medium == 0.0:
        raise ValueError("the two finest levels are identical; order is not observable")
    signed = medium_coarse / fine_medium
    if not signed > 0.0:
        raise ValueError("functional changes are not monotone under refinement; no asymptotic order")
    order = math.log(signed) / math.log(ratio)
    extrapolated = richardson_extrapolate(values, ratio, order)
    gci_fine = safety * abs(fine_medium) / max(abs(float(fine)), 1e-300) / (ratio ** order - 1.0)
    consistent = None
    if values.size >= 4:
        coarser_medium = float(coarse - values[-4])
        if coarser_medium != 0.0 and coarser_medium / medium_coarse > 0.0:
            order_coarser = math.log(coarser_medium / medium_coarse) / math.log(ratio)
            consistent = bool(abs(order - order_coarser) <= asymptotic_tolerance * max(abs(order), 1e-12))
    return {
        "ratio": float(ratio),
        "order": float(order),
        "safety_factor": float(safety),
        "richardson_extrapolated": extrapolated,
        "finest_value": float(fine),
        "medium_coarse_difference": medium_coarse,
        "fine_medium_difference": fine_medium,
        "gci_fine_relative": float(gci_fine),
        "order_consistent_with_coarser_triple": consistent,
    }


def loglog_order(spacings, errors):
    """Least-squares slope of ``log(error)`` against ``log(spacing)``."""
    spacings = np.asarray(spacings, dtype=float)
    errors = np.abs(np.asarray(errors, dtype=float))
    if spacings.size < 2 or spacings.size != errors.size:
        raise ValueError("matching spacing and error sequences of length >= 2 required")
    if np.any(spacings <= 0.0) or not np.all(np.isfinite(spacings)):
        raise ValueError("spacings must be positive and finite")
    if np.any(errors <= 0.0) or not np.all(np.isfinite(errors)):
        raise ValueError("errors must be positive and finite for a log-log fit")
    slope, intercept = np.polyfit(np.log(spacings), np.log(errors), 1)
    residuals = np.log(errors) - (slope * np.log(spacings) + intercept)
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "levels": int(spacings.size),
        "maximum_fit_residual_in_log": float(np.max(np.abs(residuals))),
    }


def _fd_annular_profile(config, points):
    """Second-order FD solution of ``d/dr (r du/dr) = -(G/mu) r`` with u(Ri)=u(Ro)=0."""
    from scipy.linalg import solve_banded

    ri, ro = config.inner_radius_m, config.outer_radius_m
    gradient, mu = config.pressure_gradient_pa_m, config.viscosity_pa_s
    r = np.linspace(ri, ro, points)
    h = r[1] - r[0]
    interior = r[1:-1]
    lower = (0.5 * (interior + r[:-2])) / (interior * h * h)
    upper = (0.5 * (interior + r[2:])) / (interior * h * h)
    ab = np.zeros((3, interior.size))
    ab[0, 1:] = upper[:-1]
    ab[1, :] = -(lower + upper)
    ab[2, :-1] = lower[1:]
    u = solve_banded((1, 1), ab, np.full(interior.size, -gradient / mu))
    return r, np.concatenate(([0.0], u, [0.0])), h


def analytic_reference_check(config, resolutions=(301, 601, 1201)):
    """Independently re-derive the annular Poiseuille profile used as reference.

    The backend's closed form is
    ``u(r) = G/(4 mu) [Ro^2 - r^2 - (Ro^2 - Ri^2) ln(Ro/r) / ln(Ro/Ri)]``.
    It is compared here against a finite-difference solution of the *differential*
    statement on a refinement sequence. The differences must decay with the
    second order of the FD scheme, which is what proves the remaining gap is
    truncation rather than a wrong formula.
    """
    rows = []
    for points in resolutions:
        r, u, h = _fd_annular_profile(config, points)
        formula = analytic_velocity(r, config)
        scale = float(np.max(np.abs(formula)))
        rows.append({
            "fd_points": int(points),
            "fd_spacing_m": float(h),
            "maximum_absolute_deviation_m_s": float(np.max(np.abs(u - formula))),
            "maximum_relative_deviation": float(np.max(np.abs(u - formula)) / scale),
        })
    slopes = [
        math.log(rows[i]["maximum_absolute_deviation_m_s"] / rows[i + 1]["maximum_absolute_deviation_m_s"])
        / math.log(rows[i]["fd_spacing_m"] / rows[i + 1]["fd_spacing_m"])
        for i in range(len(rows) - 1)
    ]
    r, u, h = _fd_annular_profile(config, resolutions[-1])
    formula = analytic_velocity(r, config)
    flow_formula = analytic_flow(config)
    flow_quad = float(np.trapezoid(2.0 * math.pi * r * formula, r))
    # Exact identity check: with K=(Ro^2-Ri^2)/ln(Ro/Ri), d/dr(r du/dr) = -(G/mu) r.
    exact = -config.pressure_gradient_pa_m / config.viscosity_pa_s * r
    du = config.pressure_gradient_pa_m / (4.0 * config.viscosity_pa_s) * (
        -2.0 * r + (config.outer_radius_m ** 2 - config.inner_radius_m ** 2)
        / math.log(config.outer_radius_m / config.inner_radius_m) / r)
    identity = np.gradient(r * du, r, edge_order=2) - exact
    return {
        "fd_resolutions": rows,
        "fd_observed_orders": [float(v) for v in slopes],
        "finest_fd_max_relative_deviation": rows[-1]["maximum_relative_deviation"],
        "flow_rate_formula_m3_s": float(flow_formula),
        "flow_rate_quadrature_m3_s": flow_quad,
        "flow_rate_relative_deviation": float(abs(flow_quad - flow_formula) / abs(flow_formula)),
        "differential_identity_max_relative_residual": float(
            np.max(np.abs(identity)) / np.max(np.abs(exact))),
        "reference_is_self_check": False,
    }


_GRIDS = ((32, 8), (64, 16), (128, 32), (256, 64), (512, 128))


def study(levels=4, safety=1.25, grids=_GRIDS, linear_backend="scipy"):
    """Run the production annular solve over a refinement sequence and verify it.

    Each level doubles both ``ntheta`` and ``nr``, so the refinement ratio is 2.
    Every level uses the production residual gates and the production fitted
    geometry; nothing is relaxed here.
    """
    if not 3 <= levels <= len(grids):
        raise ValueError(f"levels must lie in [3, {len(grids)}]")
    rows = []
    for ntheta, nr in grids[:levels]:
        config = AnnularConfig(ntheta=ntheta, nr=nr)
        raw = solve(config, linear_backend=linear_backend)
        metrics = annular_metrics(raw["fields"], raw["config"])
        computed = raw["fields"]["axial_velocity_m_s"].ravel()
        volumes = raw["fields"]["cell_volumes_m2"].ravel()
        # Independent functional: volume-weighted integral of the computed profile.
        flow_computed = float(np.sum(computed * volumes))
        flow_reference = float(analytic_flow(config))
        rows.append({
            "grid": [int(ntheta), int(nr)],
            "spacing_m": float((config.outer_radius_m - config.inner_radius_m) / nr),
            "cells": int(computed.size),
            "relative_l2": float(metrics["velocity_volume_l2"]),
            "relative_linf": float(metrics["velocity_linf"]),
            "flow_rate_computed_m3_s": flow_computed,
            "flow_rate_reference_m3_s": flow_reference,
            "flow_rate_relative_error": float(abs(flow_computed - flow_reference) / abs(flow_reference)),
            "inner_wall_force_relative_error": float(metrics["inner_wall_force_relative_error"]),
            "outer_wall_force_relative_error": float(metrics["outer_wall_force_relative_error"]),
            "true_linear_residual": float(metrics["true_linear_residual"]),
            "momentum_force_balance": float(metrics["momentum_force_balance"]),
            "area_relative_error": float(metrics["area_relative_error"]),
            "elapsed_s": float(raw["elapsed_s"]),
            "linear_iterations": int(raw["linear_iterations"]),
            "scope": raw["diagnostics"]["scope"],
        })

    spacings = [row["spacing_m"] for row in rows]
    l2_slope = loglog_order(spacings, [row["relative_l2"] for row in rows])
    linf_slope = loglog_order(spacings, [row["relative_linf"] for row in rows])
    # Per-level order from the error sequence itself (no Richardson assumptions):
    # p_i = log(e_i / e_{i+1}) / log(2).
    pairwise = [
        math.log(rows[i]["relative_l2"] / rows[i + 1]["relative_l2"]) / math.log(2.0)
        for i in range(len(rows) - 1)
    ]
    flow_order = observed_order([row["flow_rate_computed_m3_s"] for row in rows], 2.0, safety=safety)
    # Wall traction is an error sequence, not a functional with an "exact" value,
    # so only the order (no GCI) is meaningful for it.
    wall_errors = [row["inner_wall_force_relative_error"] for row in rows]
    wall_pairs = [
        math.log(wall_errors[i] / wall_errors[i + 1]) / math.log(2.0) for i in range(len(wall_errors) - 1)
    ]
    reference = analytic_reference_check(AnnularConfig(ntheta=grids[0][0], nr=grids[0][1]))
    exact_flow = rows[-1]["flow_rate_reference_m3_s"]
    deviation = abs(flow_order["richardson_extrapolated"] - exact_flow) / abs(exact_flow)
    band = max(flow_order["gci_fine_relative"], 1e-15)
    return {
        "case": "annular-poiseuille-convergence",
        "levels": int(levels),
        "rows": rows,
        "l2_loglog_slope": l2_slope,
        "linf_loglog_slope": linf_slope,
        "l2_pairwise_orders": [float(v) for v in pairwise],
        "flow_rate_order": flow_order,
        "inner_wall_force_pairwise_orders": [float(v) for v in wall_pairs],
        "inner_wall_force_loglog_slope": loglog_order(spacings, wall_errors)["slope"],
        "finest_spacing_m": min(spacings),
        "analytic_flow_rate_m3_s": exact_flow,
        "extrapolated_vs_analytic_relative_deviation": float(deviation),
        "extrapolation_gci_band_relative": float(band),
        "extrapolation_agrees_with_analytic": bool(deviation <= band),
        "analytic_reference": reference,
        "coarse_grid_3pct_band_met": bool(all(row["relative_l2"] < 0.03 for row in rows)),
        "verification_statement": (
            "Discretisation-error verification against the closed-form annular Poiseuille "
            "solution; not physical validation and not a general 3-D Navier-Stokes result."
        ),
        "config": asdict(AnnularConfig(ntheta=grids[0][0], nr=grids[0][1])),
    }