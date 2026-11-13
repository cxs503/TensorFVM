"""Round 6 of ``CASE-FVM-PERIODIC-HILLS``: does the closure ever equilibrate?

Round 5 attached a Spalart--Allmaras closure to the Re = 2800 periodic-hill run
and found that ``nu_t`` was *still growing* when the budget ran out
(``nu_t/nu`` 2.1 -> 21.1 between ``t = 7.5`` and ``t = 35``), so that the single
quoted reattachment length (``x_r/H = 3.07`` quick / 7.41 long) was a property
of the averaging window rather than of the solution.  It also left its own
channel validation (C4/C5) **withheld**: the SA plane-channel gate had not
converged, so the closure was never validated on its calibration case, and the
pre-declared gate ``first_cell_y_plus < 1`` was in fact unsatisfiable on the
``ny = 64`` grid (see :func:`channel_y_plus_requirement`).

Round 6 therefore answers three questions with measurements instead of
assertions, and pre-declares the gates before them:

C1  **Equilibration.** Is there a time interval in which ``nu_t`` (and with it
    ``x_r/H``) is statistically stationary?  Gate: relative drift of block-mean
    ``max nu_t/nu`` over the last 50 time units < 2 %, and an exponential fit
    ``v(t) = v_inf - A exp(-t/tau)`` whose ``tau`` is resolved inside the run
    (< 100 time units) with ``|v_end - v_inf| / v_inf < 5 %``.  This is a
    *diagnostic* gate: failing it is a valid, publishable outcome (it says the
    budget is still too short, or that no steady state exists).
C2  **Order of the closure convection.** Round 5 used first-order upwind
    unconditionally.  The blend is now a knob
    (``SpalartAllmarasConfig.convection_blend``); C2 requires (a) ``blend = 0``
    to reproduce round 5 *bit-identically* (regression), and (b) the second
    order (``blend = 1``) run to be reported on the same equilibrated window,
    whatever it shows.
C3  **The ledger gate itself**: mean ``x_r/H`` within 10 % of Breuer et al.
    (2009) ``5.34`` (``x_s/H`` against ``0.227``), evaluated on the C1
    equilibrated window only -- never on a transient.
C4  **Mean profile at ``y/H = 0.5``** against the digitised Breuer values, with
    a geometric admission filter: the station is only admissible if the
    ``y/H = 0.5`` line is at least one cell above the local hill surface (near
    the crest it is *inside* the hill body, which is why two of the tabulated
    stations cannot be used -- quantified by :func:`admissible_stations`).
C5  **Reference provenance.** The repository carried two different
    reattachment references for the same case -- ``5.34`` in the feature ledger
    and in rounds 3--5, ``4.36`` in
    ``tests/test_periodic_hill_breuer2009.py`` -- and the two 10 % bands are
    disjoint (``[4.80, 5.87]`` vs ``[3.92, 4.80]``).  C5 makes the number
    single-sourced (:data:`REFERENCE`) and machine-checked
    (:func:`scan_reference_consistency`); the competing value is recorded, not
    deleted (:data:`COMPETING_X_R`).
C6  **Invariants** on every run: finite field, ``nu_t >= 0``, imposed bulk
    velocity held (``|u_b - 1| < 1e-8``), flux divergence at round-off.

Everything above is measured on the 48x24 terrain-following mesh of rounds 4--5
unless stated; the run driver is :mod:`scripts.round6_hill_long` (one
configuration per process, so the four runs can be sharded).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..periodic_hill import PeriodicHillConfig, PeriodicHillSolver, Statistics, hill_profile
from ..periodic_hill_turbulence import (
    SpalartAllmaras,
    SpalartAllmarasConfig,
    eddy_viscosity,
    wall_distance,
)

__all__ = [
    "REFERENCE",
    "COMPETING_X_R",
    "BREUER_PROFILE_Y05",
    "REFERENCE_TOLERANCE",
    "plateau_fit",
    "drift",
    "admissible_stations",
    "profile_at_y_over_h",
    "profile_error",
    "closure_viscosity",
    "scan_reference_consistency",
    "channel_y_plus_requirement",
    "run_block",
    "run_blocked",
    "channel_convergence",
]

# --------------------------------------------------------------------------- #
# reference data (single source of truth -- see C5)
# --------------------------------------------------------------------------- #
#: Breuer et al. (2009) as tabulated by Volpiani et al. (2021), section III.
#: This is the value the feature-ledger acceptance line cites, and it is the
#: only value the rounds 2--6 gates may use.
REFERENCE = {
    "case": "ERCOFTAC UFR 3-30 periodic hills, Re_b = 2800",
    "x_r_over_H": 5.34,
    "x_s_over_H": 0.227,
    "source": "Breuer et al. 2009, Computers & Fluids 38:433, "
              "doi:10.1016/j.compfluid.2008.05.002 (values as tabulated by "
              "Volpiani et al. 2021, section III)",
    "authority": "feature_ledger CASE-FVM-PERIODIC-HILLS acceptance line",
}

#: The competing value found in ``tests/test_periodic_hill_breuer2009.py``
#: before round 6.  Recorded rather than deleted: if a *primary* reading of
#: Breuer et al. (2009) Fig./Table ever shows 4.36 for Re_b = 2800, this entry
#: is the reconciliation point.  10 % bands: 5.34 -> [4.806, 5.874],
#: 4.36 -> [3.924, 4.796]: disjoint, so the choice is load-bearing.
COMPETING_X_R = {
    "value": 4.36,
    "where": "tests/test_periodic_hill_breuer2009.py (merged m16691, m16819)",
    "provenance": "local comment only ('LES mean, Table 2'); no page/figure "
                  "reference, no tabulated source; likely a different "
                  "Reynolds number or a transcription of the DNS/LES range "
                  "mean quoted to three digits",
    "band": [0.9 * 4.36, 1.1 * 4.36],
    "status": "retired in round 6 (superseded by REFERENCE['x_r_over_H'])",
}

#: Mean streamwise velocity ``u/U_b`` at ``y/H = 0.5`` (Breuer et al. 2009,
#: Fig. 8, LES).  Stations left of ``x/H ~ 0.9`` are physically unreachable on
#: this geometry (the line ``y/H = 0.5`` lies inside the hill body) and are
#: filtered by :func:`admissible_stations` rather than silently interpolated.
BREUER_PROFILE_Y05 = {
    0.05: -0.15,
    0.50: -0.10,
    1.00: -0.02,
    1.50: 0.05,
    2.00: 0.25,
    3.00: 0.65,
    4.00: 0.85,
    5.00: 0.95,
    6.00: 1.00,
    7.00: 1.02,
    8.00: 1.03,
}

REFERENCE_TOLERANCE = 0.10


# --------------------------------------------------------------------------- #
# C1: plateau diagnostics
# --------------------------------------------------------------------------- #
def _exponential_fit(t, v):
    """Fit ``v(t) = v_inf - A exp(-t / tau)`` by scanning ``tau``.

    A non-linear least squares would work too, but the scan is deterministic,
    monotone in ``tau`` and cannot land in a local minimum: for every trial
    ``tau`` the problem in ``(v_inf, A)`` is linear, so the residual is a
    one-dimensional function that can be minimised by a grid + golden refine.
    """
    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)
    if t.size < 4 or not np.all(np.isfinite(v)):
        return {"v_inf": float("nan"), "A": float("nan"), "tau": float("nan"),
                "rms": float("nan"), "ok": False}
    span = float(t.max() - t.min())
    if span <= 0.0:
        return {"v_inf": float("nan"), "A": float("nan"), "tau": float("nan"),
                "rms": float("nan"), "ok": False}
    taus = np.geomspace(max(span / 50.0, 1e-3), span * 50.0, 240)
    best = None
    for tau in taus:
        basis = np.column_stack([np.ones_like(t), -np.exp(-(t - t.min()) / tau)])
        coef, *_ = np.linalg.lstsq(basis, v, rcond=None)
        rms = float(np.sqrt(np.mean((basis @ coef - v) ** 2)))
        if best is None or rms < best[0]:
            best = (rms, tau, coef)
    rms, tau, coef = best
    v_inf, amp = float(coef[0]), float(coef[1])
    # a decaying trend needs a positive amplitude and a resolved rise in-window
    rise = abs(amp) * (1.0 - math.exp(-span / tau))
    ok = bool(np.isfinite(v_inf) and amp > 0.0 and rise > 0.0
              and rms <= 0.10 * max(abs(v_inf), 1e-12))
    return {"v_inf": v_inf, "A": amp, "tau": float(tau), "rms": rms, "ok": ok}


def drift(t, v, window=50.0):
    """Relative drift of ``v`` over the last ``window`` time units."""
    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)
    if t.size == 0 or v.size == 0:
        return {"relative": float("nan"), "absolute": float("nan"),
                "slope": float("nan"), "window": [], "start": float("nan"),
                "end": float("nan")}
    mask = t >= (t.max() - window)
    if np.count_nonzero(mask) < 2:
        mask = np.zeros_like(t, dtype=bool)
        mask[-2:] = True
    tt, vv = t[mask], v[mask]
    scale = max(abs(float(vv[-1])), 1e-30)
    # least-squares slope, not endpoint difference: block means are noisy
    slope = float(np.polyfit(tt, vv, 1)[0])
    span = float(tt[-1] - tt[0]) if tt[-1] > tt[0] else float("nan")
    return {
        "relative": float(slope * span / scale) if np.isfinite(span) else float("nan"),
        "absolute": float(slope),
        "slope": slope,
        "window": [float(tt[0]), float(tt[-1])],
        "start": float(vv[0]),
        "end": float(vv[-1]),
    }


def plateau_fit(t, v, tail_fraction=0.25, window=50.0, drift_gate=0.02,
                asymptote_gate=0.05, tau_gate=100.0):
    """C1: is ``v(t)`` stationary inside the run, and where would it settle?"""
    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)
    n = t.size
    keeper = max(int(math.ceil(n * tail_fraction)), 4) if n else 0
    t_tail, v_tail = t[-keeper:], v[-keeper:]
    fit = _exponential_fit(t_tail, v_tail)
    d = drift(t_tail, v_tail, window=window)
    end = float(v_tail[-1]) if v_tail.size else float("nan")
    # An equilibrated signal is *stationary*: small drift and small gap to the
    # asymptote.  The asymptote comes from the exponential fit only when the fit
    # resolves a real trend; a flat (noise-only) record has no trend to resolve
    # and its asymptote is its tail mean -- requiring a decaying amplitude here
    # would flag an already-settled run as unconverged.
    mean_tail = float(np.mean(v_tail)) if v_tail.size else float("nan")
    span_tail = float(t_tail[-1] - t_tail[0]) if v_tail.size > 1 else 0.0
    rise = (abs(fit["A"]) * (1.0 - math.exp(-span_tail / fit["tau"]))
            if fit["ok"] else 0.0)
    trend_resolved = bool(fit["ok"] and rise > 0.05 * max(abs(mean_tail), 1e-30))
    asymptote = float(fit["v_inf"]) if trend_resolved else mean_tail
    asymptote_error = (abs(end - asymptote) / max(abs(asymptote), 1e-30)
                       if np.isfinite(end) and np.isfinite(asymptote) else float("inf"))
    tau = fit["tau"]
    equilibrated = bool(
        np.isfinite(d["relative"]) and abs(d["relative"]) < drift_gate
        and asymptote_error < asymptote_gate
        and (not trend_resolved or tau < tau_gate)
    )
    return {
        "samples": int(n),
        "tail_window": [float(t_tail[0]), float(t_tail[-1])] if n else [],
        "end": end,
        "drift": d,
        "fit": fit,
        "trend_resolved": trend_resolved,
        "asymptote": float(asymptote),
        "asymptote_relative_error": float(asymptote_error),
        "tau_gate": float(tau_gate),
        "equilibrated": equilibrated,
    }


# --------------------------------------------------------------------------- #
# C4: profile at y/H = 0.5
# --------------------------------------------------------------------------- #
def admissible_stations(height=1.0, amplitude=1.0, length=9.0,
                        min_clearance=1.0, ny=24, channel_top=3.036,
                        profile=None):
    """Stations where ``y/H = 0.5`` is at least ``min_clearance`` cells above the hill.

    ``min_clearance`` is in units of the *local* first-cell height
    ``(channel_top - h(x)) / ny``; the point of the filter is that a station
    whose sampling line is inside the solid body carries no information.
    """
    profile = BREUER_PROFILE_Y05 if profile is None else profile
    out = {"admissible": [], "rejected": []}
    for xh in sorted(profile):
        if amplitude:
            contour = hill_profile(np.array([xh * height]), height, length)
            h = float(np.atleast_1d(contour[0])[0])
        else:
            h = 0.0
        cell = (channel_top - h) / float(ny)   # local first-cell height
        clearance = (0.5 * height - h) / cell
        entry = {"x_over_H": float(xh), "hill_height": h,
                 "first_cell_height": cell, "clearance_cells": float(clearance)}
        (out["admissible"] if clearance >= min_clearance else out["rejected"]).append(entry)
    return out


def profile_at_y_over_h(solver, u_mean, stations, height=1.0, bulk=1.0):
    """Interpolate the mean streamwise velocity onto ``y/H = 0.5`` at stations."""
    xc = solver.mesh.xc[0]
    yc = solver.mesh.yc
    y_target = 0.5 * height
    values = {}
    for xh in stations:
        x_target = xh * height
        i = int(np.clip(np.searchsorted(xc, x_target) - 1, 0, solver.nx - 1))
        column_y = yc[:, i]
        column_u = np.asarray(u_mean)[:, i]
        if not (column_y[0] <= y_target <= column_y[-1]):
            continue
        values[float(xh)] = float(np.interp(y_target, column_y, column_u) / bulk)
    return values


def profile_error(computed, profile=None, tolerance=REFERENCE_TOLERANCE):
    """Compare interpolated ``u/U_b(x/H)`` with the reference table."""
    profile = BREUER_PROFILE_Y05 if profile is None else profile
    rows = []
    for xh, ref in sorted(profile.items()):
        if xh not in computed:
            continue
        got = computed[xh]
        err = abs(got - ref) / max(abs(ref), 1e-30) if abs(ref) >= 0.1 else abs(got - ref)
        rows.append({"x_over_H": xh, "reference": ref, "computed": got,
                     "error": float(err), "pass": bool(err < tolerance)})
    worst = max((r["error"] for r in rows), default=float("inf"))
    rms = float(np.sqrt(np.mean([r["error"] ** 2 for r in rows]))) if rows else float("nan")
    return {"stations": rows, "worst_error": float(worst), "rms_error": rms,
            "pass": bool(rows and all(r["pass"] for r in rows))}


# --------------------------------------------------------------------------- #
# C2: closure viscosity helper (used by the regression test)
# --------------------------------------------------------------------------- #
def closure_viscosity(nu_tilde, nu, config=None):
    """Thin wrapper so tests can compare round-5 and round-6 transport paths."""
    return eddy_viscosity(np.asarray(nu_tilde, dtype=float), float(nu),
                          config or SpalartAllmarasConfig())


# --------------------------------------------------------------------------- #
# C5: provenance scan
# --------------------------------------------------------------------------- #
_REF_PATTERN = re.compile(r"\b(4\.36|5\.34)\b")


def scan_reference_consistency(root=None):
    """Find every place that quotes a reattachment reference for this case.

    Round 6's C5 is a *machine check*: the ledger value lives in exactly one
    module (:data:`REFERENCE`) and every other mention is either that import or
    the recorded competing value.
    """
    root = Path(root or Path(__file__).resolve().parents[3])
    hits = []
    skip_dirs = {".git", "docs", "runs", "__pycache__", ".pytest_cache"}
    for path in root.rglob("*.py"):
        if any(part in skip_dirs for part in path.parts):
            continue
        try:
            text = path.read_text()
        except (OSError, UnicodeDecodeError):
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if _REF_PATTERN.search(line) and "x_r" in line.lower() or (
                    _REF_PATTERN.search(line) and "reattach" in line.lower()):
                hits.append({"file": str(path.relative_to(root)),
                             "line": lineno,
                             "values": sorted(set(_REF_PATTERN.findall(line))),
                             "text": line.strip()[:160]})
    canonical = f'{float(REFERENCE["x_r_over_H"]):.2f}'
    offending = []
    for hit in hits:
        if set(hit["values"]) <= {canonical}:
            continue
        # the recorded competing value may only live in its single documented
        # place (:data:`COMPETING_X_R`), never as a live gate constant
        if hit["file"].endswith("verification/periodic_hill_round6.py"):
            continue
        # prose that *records* the retired value is provenance, not a gate
        if "COMPETING" in hit["text"] or "retired" in hit["text"]:
            continue
        offending.append(hit)
    return {"canonical": float(canonical), "hits": hits,
            "n_hits": len(hits), "n_offending": len(offending),
            "offending": offending}


# --------------------------------------------------------------------------- #
# channel: was round-5 C4 ever satisfiable?
# --------------------------------------------------------------------------- #
def channel_y_plus_requirement(reynolds=2800.0, bulk=1.0, u_tau=None,
                               channel_top=3.036, y_plus_target=1.0):
    """Number of wall-normal cells needed for ``y+ < 1`` on the plane channel.

    Round 5's C4 gated on ``first_cell_y_plus < 1`` at ``ny = 64``.  With the
    channel height this solver's configuration gives (``channel_top_m = 3.036``
    H, ``nu = U H / Re``) the first cell sitss at ``y+ = 133 u_tau``, so the gate
    demanded ``u_tau < 7.5e-3`` -- an order of magnitude below the physical
    friction velocity of a ``Re_h = 8500`` channel (``u_tau ~ 0.06``).  This
    function makes the shortfall explicit instead of leaving it as a comment.
    """
    nu = bulk * 1.0 / float(reynolds)
    u_tau = 0.06 * bulk if u_tau is None else float(u_tau)
    first_cell = (channel_top / 2.0) / 64.0
    y_plus_64 = u_tau * first_cell / nu
    needed = int(math.ceil(u_tau * (channel_top / 2.0) / (nu * y_plus_target)))
    return {"u_tau_assumed": u_tau, "nu": nu, "first_cell_height": first_cell,
            "y_plus_at_ny64": float(y_plus_64),
            "ny_for_target": needed,
            "feasible_at_ny64": bool(y_plus_64 < y_plus_target)}


# --------------------------------------------------------------------------- #
# heavy runs (driver side; see scripts/round6_hill_long.py)
# --------------------------------------------------------------------------- #
def hill_solver(nx=48, ny=24, reynolds=2800.0, dt=0.01, **kwargs):
    return PeriodicHillSolver(PeriodicHillConfig(
        nx=nx, ny=ny, reynolds=reynolds, time_step_s=dt,
        body_force_relaxation=1.0, **kwargs))


def channel_solver(ny=64, nx=8, reynolds=2800.0, dt=0.01, **kwargs):
    return PeriodicHillSolver(PeriodicHillConfig(
        nx=nx, ny=ny, reynolds=reynolds, time_step_s=dt,
        hill_amplitude=0.0, body_force_relaxation=1.0, **kwargs))


def _hill_metrics(solver, stats):
    """Mean separation/reattachment from an accumulated statistics window."""
    from .periodic_hill_round5 import _crossings

    u_mean, v_mean, _ = stats.mean()
    tau_b, _ = solver.wall_shear(u_mean, v_mean)
    x = solver.mesh.x_n[0]
    crossings = _crossings(tau_b, x, solver.config.length_m)
    sep = [c for c in crossings if c["direction"] == "+to-"]
    rea = [c for c in crossings if c["direction"] == "-to+"]
    x_s = next((c["x_over_H"] for c in sep if c["x_over_H"] <= 2.5), float("nan"))
    x_r = next((c["x_over_H"] for c in reversed(rea) if 1.9 <= c["x_over_H"] <= 8.9),
               float("nan"))
    return u_mean, v_mean, tau_b, float(x_s), float(x_r)


def run_block(solver, closure, steps, dt):
    """One statistics block; returns the block-mean diagnostics."""
    stats = Statistics(solver)
    solver.advance(steps, statistics=stats)
    u_mean, v_mean, tau_b, x_s, x_r = _hill_metrics(solver, stats)
    bulk = solver.bulk_velocity(u_mean)
    nu_t = closure.viscosity() if closure is not None else np.zeros_like(solver.u)
    return {
        "t": solver.steps * dt,
        "steps": int(solver.steps),
        "x_s_over_H": x_s,
        "x_r_over_H": x_r,
        "max_nu_t_over_nu": float(nu_t.max() / solver.nu),
        "mean_nu_t_over_nu": float(nu_t.mean() / solver.nu),
        "max_abs_velocity": float(np.abs(solver.u).max()),
        "bulk_velocity": float(bulk),
        "wall_shear_min": float(np.min(tau_b)),
        "u_mean": u_mean,
        "nu_t": nu_t,
    }


def run_blocked(config="sa-upwind", nx=48, ny=24, warm_steps=500, total_steps=30000,
                block_steps=2000, dt=0.01, reynolds=2800.0,
                convection_blend=None, progress=None):
    """Run one closure configuration in statistics blocks and grade C1/C3/C4/C6."""
    solver = hill_solver(nx=nx, ny=ny, reynolds=reynolds, dt=dt)
    if warm_steps:
        solver.advance(warm_steps)
    closure = None
    if config != "laminar":
        blend = ({"sa-upwind": 0.0, "sa-second": 1.0}.get(config)
                 if convection_blend is None else convection_blend)
        if blend is None:
            raise ValueError(f"unknown configuration {config!r}")
        closure = SpalartAllmaras(
            solver, SpalartAllmarasConfig(convection_blend=float(blend)))
        solver.turbulence = closure
    blocks = []
    remaining = int(total_steps)
    while remaining > 0:
        step = min(block_steps, remaining)
        record = run_block(solver, closure, step, dt)
        blocks.append(record)
        remaining -= step
        if progress is not None:
            progress(record)

    t = np.array([b["t"] for b in blocks])
    nu_series = np.array([b["max_nu_t_over_nu"] for b in blocks])
    xr_series = np.array([b["x_r_over_H"] for b in blocks])
    nu_plateau = plateau_fit(t, nu_series)
    xr_plateau = plateau_fit(t, xr_series)

    # C3/C4 are evaluated on the *last block* only when C1 says the window is
    # stationary; otherwise they are reported but marked un-equilibrated.
    last = blocks[-1]
    stations = [e["x_over_H"] for e in
                admissible_stations(ny=ny)["admissible"]]
    profile = profile_at_y_over_h(solver, last["u_mean"], stations,
                                  height=solver.config.height_m,
                                  bulk=last["bulk_velocity"])
    perr = profile_error(profile)
    x_r = float(last["x_r_over_H"])
    x_s = float(last["x_s_over_H"])
    finite = all(np.isfinite(b["max_abs_velocity"]) for b in blocks)
    summary = {
        "config": config,
        "grid": f"{nx}x{ny}",
        "dt": dt,
        "reynolds": reynolds,
        "convection_blend": (None if closure is None
                             else float(closure.config.convection_blend)),
        "steps": int(solver.steps),
        "t_end": float(t[-1]),
        "block_steps": int(block_steps),
        "blocks": [{k: v for k, v in b.items() if k not in ("u_mean", "nu_t")}
                   for b in blocks],
        "nu_t_plateau": nu_plateau,
        "x_r_plateau": xr_plateau,
        "final": {
            "x_r_over_H": x_r,
            "x_s_over_H": x_s,
            "x_r_reference": REFERENCE["x_r_over_H"],
            "x_r_relative_error": abs(x_r / REFERENCE["x_r_over_H"] - 1.0)
            if np.isfinite(x_r) else float("nan"),
            "x_s_reference": REFERENCE["x_s_over_H"],
            "x_s_relative_error": abs(x_s / REFERENCE["x_s_over_H"] - 1.0)
            if np.isfinite(x_s) else float("nan"),
            "max_nu_t_over_nu": last["max_nu_t_over_nu"],
            "mean_nu_t_over_nu": last["mean_nu_t_over_nu"],
            "bulk_velocity": last["bulk_velocity"],
            "max_abs_velocity": last["max_abs_velocity"],
        },
        "profile_y_over_h_half": {
            "admissible": stations,
            "computed": profile,
            "error": perr,
        },
        "criteria": {
            "C1_equilibrated": bool(nu_plateau["equilibrated"]
                                    and (config == "laminar"
                                         or xr_plateau["equilibrated"])),
            "C3_x_r_within_10pct": bool(
                np.isfinite(x_r)
                and abs(x_r / REFERENCE["x_r_over_H"] - 1.0) < REFERENCE_TOLERANCE
                and nu_plateau["equilibrated"]),
            "C4_profile_within_10pct": bool(perr["pass"]
                                            and nu_plateau["equilibrated"]),
            "C6_finite": bool(finite),
            "C6_bulk_held": bool(abs(last["bulk_velocity"] - 1.0) < 1e-8),
        },
        "trace": [(float(b["t"]), float(b["max_nu_t_over_nu"]),
                   float(b["x_r_over_H"])) for b in blocks],
    }
    return summary


def channel_convergence(ny=64, nx=8, dt=0.01, reynolds=2800.0,
                        spin_steps=2000, total_steps=40000, block_steps=5000):
    """Round-5 C4 revisited: does the SA plane channel reach a log law at all?

    The round-5 gate (`kappa` within 10 % of 0.41, `B` within 1.0 of 5.2, first
    cell `y+ < 1`) is re-evaluated here with the *satisfiable* part kept and the
    `y+ < 1` sub-gate replaced by a reported value plus
    :func:`channel_y_plus_requirement` (which shows it needs `ny ~ 500`).
    """
    from .periodic_hill_round5 import LOG_B, LOG_KAPPA, log_law_fit

    solver = channel_solver(ny=ny, nx=nx, reynolds=reynolds, dt=dt)
    solver.advance(spin_steps)
    closure = SpalartAllmaras(solver)
    solver.turbulence = closure
    blocks = []
    remaining = int(total_steps)
    while remaining > 0:
        step = min(block_steps, remaining)
        stats = Statistics(solver)
        solver.advance(step, statistics=stats)
        u_mean, v_mean, _ = stats.mean()
        tau_b, tau_t = solver.wall_shear(u_mean, v_mean)
        u_tau = float(np.sqrt(0.5 * (abs(tau_b.mean()) + abs(tau_t.mean()))))
        y = solver.mesh.yc[:, 0]
        y_plus = u_tau * y / solver.nu
        u_plus = u_mean[:, 0] / u_tau
        fit = log_law_fit(y_plus, u_plus)
        blocks.append({
            "t": solver.steps * dt,
            "u_tau": u_tau,
            "re_tau": float(u_tau * (solver.config.channel_top_m / 2.0) / solver.nu),
            "first_cell_y_plus": float(y_plus[0]),
            "max_nu_t_over_nu": float(closure.viscosity().max() / solver.nu),
            "mean_nu_t_over_nu": float(closure.viscosity().mean() / solver.nu),
            "kappa_eff": fit["kappa_eff"],
            "B": fit["B"],
            "kappa_relative_deviation": (abs(fit["kappa_eff"] / LOG_KAPPA - 1.0)
                                         if np.isfinite(fit["kappa_eff"])
                                         else float("nan")),
            "u_max_over_ub": float(u_mean.max() / solver.bulk_velocity(u_mean)),
        })
        remaining -= step
    t = np.array([b["t"] for b in blocks])
    u_tau_plateau = plateau_fit(t, np.array([b["u_tau"] for b in blocks]))
    last = blocks[-1]
    return {
        "ny": int(ny),
        "dt": dt,
        "blocks": blocks,
        "u_tau_plateau": u_tau_plateau,
        "y_plus_requirement": channel_y_plus_requirement(reynolds=reynolds,
                                                         u_tau=last["u_tau"]),
        "log_law_kappa_gate": bool(np.isfinite(last["kappa_relative_deviation"])
                                   and last["kappa_relative_deviation"] < 0.10),
        "log_law_B_gate": bool(np.isfinite(last["B"]) and abs(last["B"] - LOG_B) < 1.0),
        "c4_verdict": ("pass" if (u_tau_plateau["equilibrated"]
                                  and last["kappa_relative_deviation"] < 0.10
                                  and abs(last["B"] - LOG_B) < 1.0)
                       else ("withheld-unconverged"
                             if not u_tau_plateau["equilibrated"] else "fail")),
    }