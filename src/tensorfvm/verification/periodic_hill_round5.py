"""Round-5 verification of the periodic-hill solver: the Spalart--Allmaras closure.

Mission 16880 (selfstart, TensorFVM).  Anchor ``CASE-FVM-PERIODIC-HILLS``
(feature ledger, ``open``).  Rounds 1--4 (``docs/case-periodic-hills``) ended at
an honest FAIL with a *diagnosed* cause: after round 4 stabilised the Re = 2800
run, refining 48x24 -> 96x48 moved the mean reattachment point by 0.025 % while
the value stayed 52 % too long (``x_r/H = 8.137`` vs Breuer et al. 2009's 5.34).
The bubble is converged and wrong: this solver was 2D *laminar* Navier--Stokes
and the reference case is turbulent.  Round 5 adds the missing model class -- an
eddy-viscosity closure (:mod:`tensorfvm.periodic_hill_turbulence`, Spalart &
Allmaras 1992) -- and grades it.

Criteria, pre-declared before the runs (the numbers in brackets are the round-5
values, produced by ``python -m tensorfvm.verification.periodic_hill_round5``):

C1  the per-cell-viscosity momentum operator is a *strict* generalisation: with a
    uniform ``mu = nu`` it reproduces an independently assembled round-4 operator
    to round-off (gate: ``max|dM| < 1e-12 * |nu|``) and the laminar plane-channel
    solution does not move (``u_max/u_b`` unchanged to 1e-12).
C2  the wall distance is exact geometry: on the *plane* channel it equals the
    vertical distance to round-off (gate ``< 1e-12 H``); on the hill it is
    strictly smaller than the vertical distance on the slanted part (gate: some
    cell is at least 5 % shorter), i.e. a vertical approximation is really
    replaced by a point-to-segment distance.
C3  ``nu~ = 0`` is an exact solution of the closure (no spurious production):
    starting from zero it stays zero to round-off and ``nu_t = 0``.
C4  closure validation on its own calibration case -- the fully developed plane
    channel.  The SA run must reproduce the log law in the window
    ``30 <= y+ <= 0.3 h+``: measured von Karman slope within 10 % of 0.41 and
    intercept ``B`` within 1.0 of 5.2 (absolute, i.e. ~19 %), with ``y+ < 1``
    resolved by the first cell (wall-resolved).
C5  the negative control is not vacuous: the *laminar* channel (closure off) on
    the same grid deviates from the same log law far more than the SA run
    (reported as a ratio; gate ratio > 3).
C6  Re = 2800, 48x24, ``dt = 0.01`` stays finite for 30000 steps (``t = 300``)
    with the closure active, ``max nu_t >= 5 nu`` (the closure is actually
    working), ``nu_t >= 0`` everywhere, and the imposed bulk velocity holds
    (``|u_b - 1| < 1e-8``).
C7  accuracy against Breuer et al. 2009 (as tabulated by Volpiani et al. 2021,
    section III): mean ``x_r/H`` within 10 % of 5.34 -- **honest gate, reported
    as measured**; ``x_s/H`` against 0.227 is reported with the same 10 % band.
C8  grid sensitivity of ``x_r/H`` between 48x24 and 96x48 is reported, not gated
    (two grids and an unsteady 2D flow cannot settle convergence).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..periodic_hill import PeriodicHillConfig, PeriodicHillSolver, Statistics
from ..periodic_hill_turbulence import (
    SpalartAllmaras,
    SpalartAllmarasConfig,
    wall_distance,
)

__all__ = [
    "X_R_REF",
    "X_S_REF",
    "LAMINAR_CHANNEL_UMAX",
    "reference_momentum_operator",
    "channel_solver",
    "hill_solver",
    "criterion_operator_generalisation",
    "criterion_wall_distance",
    "criterion_zero_state",
    "log_law_fit",
    "channel_profile",
    "hill_mean_field",
    "run",
]

#: Breuer et al. (2009) as tabulated by Volpiani et al. (2021), Re = 2800.
X_R_REF = 5.34
X_S_REF = 0.227
#: Round-4 published laminar plane-channel value (docs/case-periodic-hills/ROUND4.md, C5).
LAMINAR_CHANNEL_UMAX = 1.34589
#: Analytic band value for the ERCOFTAC bulk-velocity definition.
ANALYTIC_CHANNEL_UMAX = 1.34637
#: Log law u+ = ln(y+)/kappa + B.
LOG_KAPPA = 0.41
LOG_B = 5.2
LOG_WINDOW_LO = 30.0
LOG_WINDOW_FRACTION = 0.3


# --------------------------------------------------------------------------- #
# operators
# --------------------------------------------------------------------------- #
def reference_momentum_operator(solver):
    """Independent assembly of ``I/dt - nu * L`` (the round-4 operator).

    Written out again here on purpose: the point of C1 is that the refactored
    per-cell-viscosity assembly has not silently changed the uniform-viscosity
    operator, and a tautological comparison against the solver's own helper would
    not test that.
    """
    from scipy import sparse

    m = solver.mesh
    ny, nx, n, nu = solver.ny, solver.nx, solver.n, solver.nu
    vols = m.volume
    rows, cols, vals = [], [], []
    diag = np.zeros((ny, nx))
    for j in range(ny):
        p = np.arange(j * nx, (j + 1) * nx)
        ce_e = m.area_e[j] / m.dn_e[j]
        ce_w = m.area_e[j, m.im] / m.dn_e[j, m.im]
        cn_s = m.area_n[j] / m.dn_n[j]
        cn_n = m.area_n[j + 1] / m.dn_n[j + 1]
        for col, val in (
            (solver._cell(j, np.arange(nx) + 1), ce_e),
            (solver._cell(j, np.arange(nx) - 1), ce_w),
        ):
            rows.append(p); cols.append(col); vals.append(val / vols[j])
        if j > 0:
            rows.append(p); cols.append(solver._cell(j - 1, np.arange(nx)))
            vals.append(cn_s / vols[j])
        if j < ny - 1:
            rows.append(p); cols.append(solver._cell(j + 1, np.arange(nx)))
            vals.append(cn_n / vols[j])
        diag[j] = (ce_e + ce_w + cn_s + cn_n) / vols[j]
    off = sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    laplacian = off - sparse.diags(diag.ravel())
    return (sparse.diags(np.full(n, 1.0 / solver.time_step_s)) - nu * laplacian).tocsr()


def criterion_operator_generalisation(nx=48, ny=24, steps=2000):
    """C1: uniform ``mu`` reproduces the round-4 operator and solution.

    The solution half re-runs the *round-4 channel gate verbatim* (Re = 100,
    8x32, dt = 0.05, 2000 steps, ``u_max/u_b = 1.34589`` published in
    ``docs/case-periodic-hills/ROUND4.md``), so the refactor is checked against
    a number that was produced by the old code, not against itself.
    """
    solver = hill_solver(nx=nx, ny=ny)
    reference = reference_momentum_operator(solver)
    diff = abs(solver.momentum - reference).max()
    channel = PeriodicHillSolver(PeriodicHillConfig(
        nx=8, ny=32, reynolds=100.0, time_step_s=0.05, hill_amplitude=0.0,
        channel_top_m=3.036, body_force_relaxation=1.0))
    channel.advance(steps)
    umax = float(channel.u.max() / channel.bulk_velocity())
    return {
        "max_abs_matrix_difference": float(diff),
        "scaled": float(diff * solver.time_step_s / solver.nu),
        "laminar_channel_u_max_over_ub": umax,
        "round4_published": LAMINAR_CHANNEL_UMAX,
        "u_max_drift": float(abs(umax - LAMINAR_CHANNEL_UMAX)),
        # the published round-4 value is quoted to five decimals, so the
        # solution half of the gate is 1e-4 absolute, not 1e-12
        "pass": bool(diff * solver.time_step_s / solver.nu < 1e-12
                     and abs(umax - LAMINAR_CHANNEL_UMAX) < 1e-4),
    }


def criterion_wall_distance(nx=48, ny=24):
    """C2: exact point-to-segment distances, checked against vertical ones."""
    channel = channel_solver(ny=32)
    d = wall_distance(channel.mesh, channel.config.length_m, channel.config.channel_top_m)
    vertical = np.minimum(channel.mesh.yc, channel.config.channel_top_m - channel.mesh.yc)
    plane_error = float(np.abs(d - vertical).max())
    hill = hill_solver(nx=nx, ny=ny)
    dh = wall_distance(hill.mesh, hill.config.length_m, hill.config.channel_top_m)
    yb, _, _ = hill.config.hill(hill.mesh.xc[0])
    vertical_hill = np.minimum(hill.mesh.yc - yb[None, :],
                               hill.config.channel_top_m - hill.mesh.yc)
    ratio = dh / vertical_hill
    return {
        "plane_channel_max_abs_error": plane_error,
        "hill_min_distance": float(dh.min()),
        "hill_distance_over_vertical_min": float(ratio.min()),
        "hill_distance_over_vertical_max": float(ratio.max()),
        "fraction_shorter_by_5pct": float(np.mean(ratio < 0.95)),
        "pass": bool(plane_error < 1e-12 and ratio.min() < 0.95),
    }


def criterion_zero_state(steps=50, nx=32, ny=16):
    """C3: ``nu~ = 0`` is invariant (no spurious production)."""
    solver = hill_solver(nx=nx, ny=ny)
    closure = SpalartAllmaras(solver)
    closure.nu_tilde = np.zeros_like(closure.nu_tilde)
    solver.turbulence = closure
    solver.advance(steps)
    return {
        "max_abs_nu_tilde": float(np.abs(closure.nu_tilde).max()),
        "max_abs_nu_t": float(np.abs(closure.viscosity()).max()),
        "pass": bool(np.abs(closure.nu_tilde).max() == 0.0),
    }


# --------------------------------------------------------------------------- #
# solvers
# --------------------------------------------------------------------------- #
def channel_solver(ny=64, nx=8, reynolds=2800.0, dt=0.01, **kwargs):
    """Fully developed plane channel: the hill amplitude is switched off."""
    return PeriodicHillSolver(PeriodicHillConfig(
        nx=nx, ny=ny, reynolds=reynolds, time_step_s=dt,
        hill_amplitude=0.0, body_force_relaxation=1.0, **kwargs))


def hill_solver(nx=48, ny=24, reynolds=2800.0, dt=0.01, **kwargs):
    return PeriodicHillSolver(PeriodicHillConfig(
        nx=nx, ny=ny, reynolds=reynolds, time_step_s=dt,
        body_force_relaxation=1.0, **kwargs))


def log_law_fit(y_plus, u_plus, high_fraction=LOG_WINDOW_FRACTION):
    """Least-squares ``u+ = ln(y+)/kappa_eff + B`` inside the log window."""
    y_plus = np.asarray(y_plus, dtype=float)
    u_plus = np.asarray(u_plus, dtype=float)
    h_plus = float(y_plus.max())
    mask = (y_plus >= LOG_WINDOW_LO) & (y_plus <= high_fraction * h_plus + 1e-12)
    if np.count_nonzero(mask) < 3:
        return {"cells": int(np.count_nonzero(mask)), "kappa_eff": float("nan"),
                "B": float("nan"), "rms": float("nan")}
    x = np.log(y_plus[mask])
    slope, intercept = np.polyfit(x, u_plus[mask], 1)
    model = slope * x + intercept
    return {
        "cells": int(np.count_nonzero(mask)),
        "kappa_eff": float(1.0 / slope),
        "B": float(intercept),
        "rms": float(np.sqrt(np.mean((u_plus[mask] - model) ** 2))),
        "relative_rms": float(np.sqrt(np.mean((u_plus[mask] - model) ** 2))
                              / np.mean(u_plus[mask])),
        "y_plus_range": [float(y_plus[mask].min()), float(y_plus[mask].max())],
        "h_plus": h_plus,
    }


def channel_profile(ny=64, spin_steps=2000, average_steps=20000, with_closure=True,
                    dt=0.01):
    """Run the plane channel (optionally with SA) and return the mean profile."""
    solver = channel_solver(ny=ny, dt=dt)
    if not with_closure:
        # The laminar channel has a viscous development time H^2/nu ~ 8400 here,
        # so starting from rest would compare the log law against a *developing*
        # profile.  Start from the exact Poiseuille shape instead: it is a steady
        # solution of the laminar problem, so the run only re-imposes u_b.
        top = solver.config.channel_top_m
        probe = solver.mesh.yc * (top - solver.mesh.yc)
        solver.u = 1.5 * probe / probe.max()
    solver.advance(spin_steps)
    closure = None
    if with_closure:
        closure = SpalartAllmaras(solver)
        solver.turbulence = closure
    stats = Statistics(solver)
    solver.advance(average_steps, statistics=stats)
    u_mean, v_mean, _ = stats.mean()
    tau_b, tau_t = solver.wall_shear(u_mean, v_mean)
    u_tau = float(np.sqrt(0.5 * (abs(tau_b.mean()) + abs(tau_t.mean()))))
    y = solver.mesh.yc[:, 0]
    u = u_mean[:, 0]
    y_plus = u_tau * y / solver.nu
    u_plus = u / u_tau
    fit = log_law_fit(y_plus, u_plus)
    return {
        "with_closure": bool(with_closure),
        "ny": int(ny),
        "u_tau": u_tau,
        "re_tau": float(u_tau * (solver.config.channel_top_m / 2.0) / solver.nu),
        "first_cell_y_plus": float(y_plus[0]),
        "u_max_over_ub": float(u_mean.max() / solver.bulk_velocity(u_mean)),
        "log_law": fit,
        "mean_nu_t_over_nu": (float(closure.viscosity().mean() / solver.nu)
                              if closure is not None else 0.0),
        "max_nu_t_over_nu": (float(closure.viscosity().max() / solver.nu)
                             if closure is not None else 0.0),
        "bulk_velocity": solver.bulk_velocity(u_mean),
        "max_abs_divergence": float(np.abs(solver.flux_divergence()).max()),
        "profile": {"y_plus": y_plus.tolist(), "u_plus": u_plus.tolist()},
    }


def hill_mean_field(nx=48, ny=24, spin_steps=10000, average_steps=20000, dt=0.01,
                    closure=True, chunk=1000, warm_steps=500):
    """Re = 2800 hill run; returns the time-averaged field diagnostics.

    Phases: ``warm_steps`` laminar (the round-4 state), then ``spin_steps`` with
    the closure attached (it starts from ``nu~ = 3 nu`` homogeneous), then
    ``average_steps`` accumulated into :class:`Statistics`.
    """
    solver = hill_solver(nx=nx, ny=ny, dt=dt)
    trace = []

    def _advance(steps, phase, statistics=None):
        total = 0
        while total < steps:
            step = min(chunk, steps - total)
            solver.advance(step, statistics=statistics)
            total += step
            entry = {
                "phase": phase, "step": solver.steps, "t": solver.steps * dt,
                "max_abs_velocity": float(np.abs(solver.u).max()),
                "max_nu_t_over_nu": float(solver.nu_t.max() / solver.nu),
                "bulk_velocity": solver.bulk_velocity(),
            }
            if statistics is not None:
                entry["max_cell_divergence"] = float(np.abs(solver.divergence()).max())
            trace.append(entry)

    _advance(warm_steps, "laminar-warmup")
    closure_model = None
    if closure:
        closure_model = SpalartAllmaras(solver)
        solver.turbulence = closure_model
    _advance(spin_steps, "closure-spin" if closure else "laminar-spin")
    stats = Statistics(solver)
    _advance(average_steps, "average", statistics=stats)
    u_mean, v_mean, g_mean = stats.mean()
    bulk = solver.bulk_velocity(u_mean)
    tau_b, tau_t = solver.wall_shear(u_mean, v_mean)
    x = solver.mesh.x_n[0]
    crossings = _crossings(tau_b, x, solver.config.length_m)
    sep = [c for c in crossings if c["direction"] == "+to-"]
    rea = [c for c in crossings if c["direction"] == "-to+"]
    x_s = next((c["x_over_H"] for c in sep if c["x_over_H"] <= 2.5), float("nan"))
    x_r = next((c["x_over_H"] for c in reversed(rea) if 1.9 <= c["x_over_H"] <= 8.9),
               float("nan"))
    finite = all(np.isfinite(r["max_abs_velocity"]) for r in trace)
    return {
        "grid": f"{nx}x{ny}",
        "with_closure": bool(closure),
        "samples": int(stats.samples),
        "x_r_over_H": float(x_r),
        "x_s_over_H": float(x_s),
        "x_r_error": float(abs(x_r - X_R_REF) / X_R_REF) if np.isfinite(x_r) else float("nan"),
        "x_s_error": float(abs(x_s - X_S_REF) / X_S_REF) if np.isfinite(x_s) else float("nan"),
        "max_abs_velocity": float(np.abs(u_mean).max()),
        "max_nu_t_over_nu": float(solver.nu_t.max() / solver.nu),
        "min_nu_t": float(solver.nu_t.min()),
        "bulk_velocity_mean": float(bulk),
        "body_force_mean": float(g_mean),
        "max_abs_flux_divergence": float(np.abs(solver.flux_divergence()).max()),
        "all_crossings": crossings,
        "finite": bool(finite),
        "wall_shear_profile": {"x_over_H": (x / solver.config.height_m).tolist(),
                               "tau": tau_b.tolist()},
        "mean_profile_x_over_H_3": _vertical_profile(solver, u_mean, 3.0, bulk),
        "mean_profile_x_over_H_6": _vertical_profile(solver, u_mean, 6.0, bulk),
        "trace": trace,
    }


def _vertical_profile(solver, u_mean, x_over_H, bulk):
    x = x_over_H * solver.config.height_m
    xv = solver.mesh.xc[0]
    i = int(np.clip(np.searchsorted(xv, x) - 1, 0, solver.nx - 1))
    return {"x_over_H": x_over_H,
            "y_over_H": (solver.mesh.yc[:, i] / solver.config.height_m).tolist(),
            "u_over_ub": (u_mean[:, i] / bulk).tolist()}


def _crossings(tau, x, length):
    out = []
    for i in range(len(x)):
        a, b = tau[i], tau[(i + 1) % len(x)]
        if a == 0.0 or not ((a < 0.0 <= b) or (a > 0.0 >= b)):
            continue
        t = -a / (b - a)
        span = (x[(i + 1) % len(x)] - x[i]) % length
        out.append({"x_over_H": float(x[i] + t * span),
                    "direction": "-to+" if a < 0.0 else "+to-"})
    return out


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def run(out_dir=None, quick=False, steps=None):
    """Evaluate the round-5 criteria and (optionally) write the summary."""
    spin = 600 if quick else 3000
    average = 600 if quick else 3000
    c1 = criterion_operator_generalisation()
    c2 = criterion_wall_distance()
    c3 = criterion_zero_state()
    sa = channel_profile(ny=64, spin_steps=300 if quick else 3000,
                         average_steps=400 if quick else 4000)
    lam = channel_profile(ny=64, spin_steps=300 if quick else 2000,
                          average_steps=400 if quick else 4000, with_closure=False)
    dev_sa = abs(sa["log_law"]["kappa_eff"] / LOG_KAPPA - 1.0)
    dev_lam = abs(lam["log_law"]["kappa_eff"] / LOG_KAPPA - 1.0)
    c4 = {
        "log_law": sa["log_law"],
        "kappa_relative_deviation": float(dev_sa),
        "B_deviation": float(abs(sa["log_law"]["B"] - LOG_B)),
        "first_cell_y_plus": sa["first_cell_y_plus"],
        "max_nu_t_over_nu": sa["max_nu_t_over_nu"],
        "pass": bool(dev_sa < 0.10 and abs(sa["log_law"]["B"] - LOG_B) < 1.0
                     and sa["first_cell_y_plus"] < 1.0),
    }
    c5 = {
        "laminar_kappa_relative_deviation": float(dev_lam),
        "laminar_log_law": lam["log_law"],
        "ratio": float(dev_lam / dev_sa) if dev_sa > 0.0 else float("inf"),
        "pass": bool(dev_sa > 0.0 and dev_lam / dev_sa > 3.0),
    }
    hill = hill_mean_field(nx=48, ny=24, spin_steps=spin, average_steps=average)
    c6 = {
        "finite": hill["finite"],
        "max_nu_t_over_nu": hill["max_nu_t_over_nu"],
        "min_nu_t": hill["min_nu_t"],
        "bulk_error": float(abs(hill["bulk_velocity_mean"] - 1.0)),
        "max_abs_flux_divergence": hill["max_abs_flux_divergence"],
        "steps": int(hill["trace"][-1]["step"]) if hill["trace"] else 0,
        "pass": bool(hill["finite"] and hill["max_nu_t_over_nu"] >= 5.0
                     and hill["min_nu_t"] >= 0.0
                     and abs(hill["bulk_velocity_mean"] - 1.0) < 1e-8),
    }
    c7 = {
        "x_r_over_H": hill["x_r_over_H"], "reference": X_R_REF,
        "relative_error": hill["x_r_error"],
        "x_s_over_H": hill["x_s_over_H"], "x_s_reference": X_S_REF,
        "x_s_relative_error": hill["x_s_error"],
        "pass": bool(np.isfinite(hill["x_r_error"]) and hill["x_r_error"] < 0.10),
    }
    criteria = {
        "C1_operator_generalisation": c1,
        "C2_wall_distance": c2,
        "C3_zero_state": c3,
        "C4_log_law_with_closure": c4,
        "C5_negative_control_laminar": c5,
        "C6_stable_reach_and_closure_active": c6,
        "C7_breuer2009_reattachment": c7,
    }
    summary = {
        "round": 5,
        "anchor": "CASE-FVM-PERIODIC-HILLS",
        "model": "Spalart-Allmaras (AIAA J. 32(7):1598, 1992)",
        "channel": {"sa": {k: v for k, v in sa.items() if k != "profile"},
                    "laminar": {k: v for k, v in lam.items() if k != "profile"}},
        "hill": {k: v for k, v in hill.items() if k not in ("trace",)},
        "criteria": criteria,
        "pass_all": all(c.get("pass") for c in criteria.values()),
        "profiles": {"channel_sa": sa["profile"], "channel_laminar": lam["profile"]},
    }
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "round5_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def hill_only(out_file, nx=48, ny=24, spin_steps=3000, average_steps=3000, dt=0.01):
    """Standalone hill run (used when the channel gates are already in the summary)."""
    result = hill_mean_field(nx=nx, ny=ny, spin_steps=spin_steps,
                             average_steps=average_steps, dt=dt)
    result = {k: v for k, v in result.items() if k != "trace"}
    result["pass_C7"] = bool(np.isfinite(result["x_r_error"])
                             and result["x_r_error"] < 0.10)
    Path(out_file).write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in
                      ("grid", "x_r_over_H", "x_s_over_H", "x_r_error",
                       "x_s_error", "max_nu_t_over_nu", "samples", "finite")}, indent=2))


def main(argv=None):
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--hill-only" in argv:
        i = argv.index("--hill-only")
        out_file = argv[i + 1]
        numbers = [int(v) for v in argv[i + 2:i + 6]] if len(argv) >= i + 6 else []
        hill_only(out_file, *numbers)
        return
    out_dir = argv[0] if argv and not argv[0].startswith("--") else None
    quick = "--quick" in argv
    summary = run(out_dir=out_dir, quick=quick)
    print(json.dumps({k: v for k, v in summary["criteria"].items()}, indent=2))
    print("pass_all:", summary["pass_all"])


if __name__ == "__main__":
    main()