"""Oblique (rotated) dam-break verification of the 2-D shallow-water operator.

Why this module exists
----------------------
``CASE-FVM-DAMBREAK-STOKER`` grades the Ritter/Stoker closed forms on the 1-D
solver (``dam_break.py``) and, since ``dam_break2d.py``, on the 2-D solver too.
But the 2-D evidence has one structural blind spot: **the dam line was always
aligned with the grid**.  With ``h`` uniform in ``y`` and the discontinuity on a
``x = const`` face, the transverse flux is *identically* zero, so the run
exercises a 1-D operator embedded in a 2-D array.  A defect that only shows up
when the wave crosses the grid obliquely -- a transverse flux that is not
consistent, a ``dx``/``dy`` mixing in the reconstruction, a directional bias in
the HLL wave-speed estimate, a staircase-induced front lag -- cannot appear in
any of those numbers.

This module closes that gap with the *same* exact solutions, rotated.  The
shallow-water equations are rotation invariant, so the solution on a domain
whose dam line has normal ``n = (cos t, sin t)`` is exactly

    h(X, t) = h_1d(s / t),   s = n . (X - X_c),

i.e. the Ritter (dry bed) / Stoker (wet bed) profile evaluated in the normal
coordinate.  Nothing new is solved analytically; what is verified is that the
**2-D discrete operator reproduces the rotated exact solution**, which is a
statement about the implementation (transverse consistency and grid-direction
independence), and the report says so.

Measurement (declared before any run)
-------------------------------------
A diagonal discontinuity on a Cartesian grid is a staircase, so the reading must
not be a single grid line.  The numerical field is reduced to a **normal
profile** by (a) keeping the cells inside a strip ``|p| <= STRIP_HALF_WIDTH``
around the dam line (``p`` is the along-dam coordinate, so the strip stays away
from the domain walls) and (b) averaging the cells of each bin of width
``bin_width`` in ``s``.  The graded quantities are then the *same* ones as the
1-D case, with the *same* estimators:

``front``  dry bed: :func:`~tensorfvm.shallow_water.dam_break.fan_front_estimate`
           on the binned normal profile, graded against ``2 c0 t``;
           wet bed: :func:`~tensorfvm.shallow_water.dam_break.data_driven_plateau`
           (sub-cell mid-level crossing) graded against ``U t``, plus the
           still-water level step ``h2/h1``.
``profile`` relative L1 / Linf of the binned normal profile against the exact
           profile evaluated at the bin centres.
``monotone`` interior local extrema above ``EXTREMUM_FRACTION * h0`` (zero is
           required: the acceptance sentence says "no spurious oscillation").

The angle sweep ``ANGLES_DEG`` always contains the axis-aligned control
``0`` (the configuration the older evidence used) and two oblique angles whose
dam line has a rational staircase period on the grid (``atan(1/2)`` and
``45 deg``), so the result cannot be blamed on an irrational-alignment artefact.

Honesty statement
-----------------
* The exact reference is analytic and independent of the solver.
* The *estimators* are the case's own; :func:`estimator_zero_bias` applies them
  to the exact profile to show they carry no position bias (a few 1e-3 for the
  binned profile, reported, not assumed).
* If a coarse grid fails a gate, the failure is reported (see the artifacts
  ``honest_fail`` list and ``docs/case-fvm-dambreak-stoker-oblique/REPORT.md``);
  the module never widens a gate to accommodate a run.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from .dam_break import (
    EXTREMUM_FRACTION,
    TOLERANCE,
    DamBreakCase,
    RITTER_DRY,
    STOKER_WET,
    _level_crossing,
    _local_extrema,
    local_extrema_details,
    _relative,
    data_driven_plateau,
    fan_front_estimate,
)
from .exact import (
    ritter_front_position,
    ritter_state,
    stoker_constant_state,
    stoker_state,
    t_star_to_time,
)
from .swe2d import ShallowWater2D

__all__ = [
    "ANGLES_DEG",
    "DOMAIN",
    "OBLIQUE_CASES",
    "OBLIQUE_AGREEMENT_MAX",
    "STRIP_HALF_WIDTH",
    "dam_initial_depth",
    "dam_normal",
    "estimator_zero_bias",
    "grade_profile",
    "isotropy_rows",
    "isotropy_study",
    "normal_coordinate",
    "normal_strip_profile",
    "run_oblique_case",
    "verdict_oblique",
]

#: square side of the computational domain (m); the dam line passes through the
#: centre and the graded strip stays ``STRIP_HALF_WIDTH`` away from it, so the
#: closest wall is at least ``DOMAIN / (2 max(|cos|,|sin|)) - STRIP_HALF_WIDTH``
#: = 5.99 m from the dam line for the 45 deg case -- well beyond the 4.0 m the
#: dry-bed front travels by ``t* = 2`` (checked by ``wall_clearance`` in the
#: payload, not by hand).
DOMAIN = 12.0

#: half width of the along-dam strip that is graded (m).
STRIP_HALF_WIDTH = 2.5

#: dam-line angles (degrees) swept by the study: axis-aligned control, a
#: 2:1 staircase (``atan 1/2``) and the 45 deg staircase.
ANGLES_DEG = (0.0, 26.56505117707799, 45.0)

#: An oblique run must not be **worse** than the axis-aligned control by more
#: than this, in headline-error percentage points (declared at 1 pp = 20 % of
#: the acceptance tolerance, before any run).  The gate is deliberately
#: one-sided: the strip average of a staircase is a *smoothing* filter, so an
#: oblique reading can legitimately come out better than the control, and
#: demanding equality would penalise the estimator rather than the operator.
#: The reverse direction is reported instead (``improvement_vs_control``) and is
#: flagged in the report as a sensitivity limitation of the binned estimator,
#: together with the measured cells-per-bin, because a reading that is "too
#: good" is evidence about the filter, not about the physics.
OBLIQUE_AGREEMENT_MAX = 1e-2

#: the two cases of the anchor, reused unchanged (h0, h1, g, t*_end).
OBLIQUE_CASES = {"ritter-dry": RITTER_DRY, "stoker-wet": STOKER_WET}


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def dam_normal(angle_deg: float) -> tuple[float, float]:
    """Unit normal of the dam line, pointing downstream (``+s``)."""
    theta = math.radians(float(angle_deg))
    return math.cos(theta), math.sin(theta)


def normal_coordinate(solver: ShallowWater2D, angle_deg: float,
                      centre: tuple[float, float] | None = None):
    """``(s, p)``: normal and along-dam coordinates of every cell centre.

    ``s`` is measured from the dam line through ``centre`` (default: the domain
    centre) and grows downstream; ``p`` is measured along the dam line.
    """
    x, y = solver.cell_centers()
    if centre is None:
        centre = (0.5 * (solver.x_min + solver.x_max),
                  0.5 * (solver.y_min + solver.y_max))
    cos, sin = dam_normal(angle_deg)
    dx, dy = x - centre[0], y - centre[1]
    return cos * dx + sin * dy, -sin * dx + cos * dy


def dam_initial_depth(solver: ShallowWater2D, case: DamBreakCase,
                      angle_deg: float,
                      centre: tuple[float, float] | None = None) -> np.ndarray:
    """Dam-break initial depth for a dam line at ``angle_deg``."""
    s, _ = normal_coordinate(solver, angle_deg, centre)
    return np.where(s <= 0.0, case.h0, case.h1)


# --------------------------------------------------------------------------- #
# profile reduction
# --------------------------------------------------------------------------- #
def normal_strip_profile(solver: ShallowWater2D, case: DamBreakCase,
                         angle_deg: float,
                         strip_half_width: float = STRIP_HALF_WIDTH,
                         bin_width: float | None = None,
                         centre: tuple[float, float] | None = None) -> dict:
    """Reduce the 2-D field to a binned **normal** profile ``(s, h, u_n)``.

    Cells with ``|p| > strip_half_width`` are discarded (they are the part of
    the dam line closest to the walls); the remaining cells are area-averaged in
    bins of ``bin_width`` (default ``dx``) in ``s``.  Empty bins are dropped, so
    ``s`` is not uniformly spaced at the strip ends -- the graded arrays carry
    only occupied bins.
    """
    s, p = normal_coordinate(solver, angle_deg, centre)
    h, u, v = solver.primitive()
    cos, sin = dam_normal(angle_deg)
    mask = np.abs(p) <= float(strip_half_width)
    if not np.any(mask):
        raise ValueError("strip contains no cells; widen it or refine the grid")
    bw = float(solver.dx if bin_width is None else bin_width)
    if bw <= 0.0:
        raise ValueError("bin_width must be positive")
    s_sel = s[mask]
    # bin edges are offset by half a bin from the lowest cell centre so that a
    # coordinate field aligned with the grid (0 deg: ``s = x - xc``) lands in the
    # *middle* of its own bin instead of on an edge, where floating-point
    # rounding could split a column between two bins.
    s_lo = float(s_sel.min()) - 0.5 * bw
    nb = int(math.floor((float(s_sel.max()) - s_lo) / bw)) + 1
    k = np.clip(np.floor((s_sel - s_lo) / bw).astype(int), 0, nb - 1)
    count = np.bincount(k, minlength=nb)
    h_sum = np.bincount(k, weights=h[mask], minlength=nb)
    un_sum = np.bincount(k, weights=(u[mask] * cos + v[mask] * sin),
                         minlength=nb)
    keep = count > 0
    return {
        "s": s_lo + (np.arange(nb) + 0.5) * bw,
        "h": np.where(keep, h_sum / np.maximum(count, 1), 0.0),
        "u_n": np.where(keep, un_sum / np.maximum(count, 1), 0.0),
        "cells": count,
        "mask": mask,
        "bin_width": bw,
        "strip_half_width": float(strip_half_width),
        "keep": keep,
        "s_field": s,
        "p_field": p,
    }


def exact_normal_profile(case: DamBreakCase, s, t: float):
    """Exact depth/velocity at the normal coordinates ``s``."""
    if case.kind == "dry":
        return ritter_state(s, t, case.h0, case.gravity, 0.0)
    return stoker_state(s, t, case.h0, case.h1, case.gravity, 0.0)


# --------------------------------------------------------------------------- #
# grading (same estimators as the 1-D anchor)
# --------------------------------------------------------------------------- #
def grade_profile(case: DamBreakCase, profile: dict, t: float) -> dict:
    """Grade one binned normal profile against the rotated exact solution."""
    keep = profile["keep"]
    s = np.asarray(profile["s"], dtype=float)[keep]
    h = np.asarray(profile["h"], dtype=float)[keep]
    h_ex, u_ex = exact_normal_profile(case, s, t)
    h_ex = np.asarray(h_ex, dtype=float)
    mass_ex = float(h_ex.sum())
    out = {
        "t": float(t),
        "t_star": float(t / math.sqrt(case.h0 / case.gravity)),
        "bins": int(s.size),
        "mean_cells_per_bin": float(np.asarray(profile["cells"], dtype=float)[keep].mean()),
        "depth_l1": (float(np.abs(h - h_ex).sum() / mass_ex)
                     if mass_ex > 0.0 else float("nan")),
        "depth_linf": float(np.abs(h - h_ex).max() / case.h0),
        "local_extrema": _local_extrema(h, EXTREMUM_FRACTION * case.h0),
        "local_extrema_details": local_extrema_details(
            h, EXTREMUM_FRACTION * case.h0, x=s),
        "h_max_over_h0": float(h.max() / case.h0),
        "h_min": float(h.min()),
        "velocity_linf": float(np.abs(profile["u_n"][keep] - u_ex).max()
                               / max(math.sqrt(case.gravity * case.h0), 1e-300)),
    }
    if case.kind == "dry":
        fan = fan_front_estimate(s, h, t, case.gravity, case.h0)
        exact_front = ritter_front_position(t, case.h0, case.gravity, 0.0)
        out.update({
            "front_definition": "fan_front_estimate on the binned normal profile",
            "front": fan["front"],
            "front_exact": float(exact_front),
            "front_error": _relative(fan["front"], exact_front),
            "front_fan_cells": fan["cells"],
            "front_fan_spread": fan["spread"],
            "front_fan_slope_error": fan["slope_error"],
        })
        headline = out["front_error"]
    else:
        sol = stoker_constant_state(case.h0, case.h1, case.gravity)
        dd = data_driven_plateau(s, h)
        exact_front = sol["shock_speed"] * t
        step_err = (abs(dd["h2"] / dd["h1"] - sol["step_ratio"])
                    / sol["step_ratio"]) if dd["h1"] else float("nan")
        level = 0.5 * (case.h1 + sol["h2"])
        x_level = _level_crossing(s, h, level)
        out.update({
            "front_level": x_level,
            "front_level_error": _relative(x_level, exact_front),
            "front_level_fraction_of_step": level / case.h1,
            "front_definition": "data_driven_plateau (sub-cell) on the binned profile",
            "front": dd["bore_x"],
            "front_exact": float(exact_front),
            "front_error": _relative(dd["bore_x"], exact_front),
            "step_ratio": dd["h2"] / dd["h1"] if dd["h1"] else float("nan"),
            "step_ratio_exact": sol["step_ratio"],
            "step_ratio_error": step_err,
            "h2_error": _relative(dd["h2"], sol["h2"]),
            "bore_valid": dd["bore_valid"],
            "bore_method": dd["bore_method"],
        })
        headline = out["front_error"]
    out["headline_error"] = float(headline)
    # series rule: dry bed -> fan estimator; wet bed -> mid-level crossing at the
    # analytic mid level, i.e. exactly dam_break2d.sample_2d / the 1-D anchor.
    if case.kind == "dry":
        out["series_error"] = float(headline)
    else:
        out["series_error"] = float(out["front_level_error"])
    out["pass"] = bool(headline <= TOLERANCE
                       and out["depth_l1"] <= TOLERANCE
                       and out["local_extrema"] == 0)
    out["tolerance"] = TOLERANCE
    return out


def estimator_zero_bias(case: DamBreakCase, t: float, n: int = 400,
                        domain: float = DOMAIN) -> dict:
    """Apply the grading estimators to the **exact** profile (zero-bias check).

    Sampling the exact solution on the same normal grid as the solver and
    feeding it to the estimators isolates the estimator's own bias from the
    solver's error: whatever the estimators return here is a floor that every
    numerical reading carries and must be reported, not subtracted.
    """
    ds = domain / float(n)
    s = (np.arange(n) + 0.5) * ds - 0.5 * domain
    h, u = exact_normal_profile(case, s, t)
    h = np.asarray(h, dtype=float)
    fake = {"s": s, "h": h, "u_n": np.asarray(u, dtype=float),
            "keep": np.ones(n, dtype=bool), "cells": np.ones(n)}
    return grade_profile(case, fake, t)


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def wall_clearance(angle_deg: float, domain: float = DOMAIN,
                   strip_half_width: float = STRIP_HALF_WIDTH) -> dict:
    """Distance from the dam line to the nearest wall inside the graded strip."""
    cos, sin = dam_normal(angle_deg)
    half = 0.5 * domain
    # distance along +n from the dam line to x = domain or y = domain (whichever
    # is hit first) at the strip edge p = +strip_half_width:
    limits = [half / c - strip_half_width * abs(s) / c
              for c, s in ((cos, sin), (sin, cos)) if c > 1e-12]
    upstream = [half / c - strip_half_width * abs(s) / c
                for c, s in ((cos, sin), (sin, cos)) if c > 1e-12]
    return {"downstream": float(min(limits)), "upstream": float(min(upstream))}


def run_oblique_case(case: DamBreakCase, angle_deg: float, n: int,
                     domain: float = DOMAIN, cfl: float = 0.4,
                     order: int = 2,
                     strip_half_width: float = STRIP_HALF_WIDTH,
                     sample_t_stars=(0.5, 1.0, 1.5, 2.0)) -> dict:
    """Integrate one oblique dam break and grade it at ``t* = 2`` (plus series).

    The wave-position *series* (acceptance sentence: ``x_f(t)``) is sampled at
    the same estimators, on grids whose front has travelled at least
    ``SERIES_MIN_CELLS`` cells -- the same a-priori floor the 1-D benchmark uses
    (imported, not re-fitted).
    """
    from .dam_break import SERIES_MIN_CELLS

    t_end = t_star_to_time(case.t_star_end, case.h0, case.gravity)
    solver = ShallowWater2D(n, n, 0.0, domain, 0.0, domain,
                            gravity=case.gravity, cfl=cfl, order=order)
    centre = (0.5 * domain, 0.5 * domain)
    h_init = dam_initial_depth(solver, case, angle_deg, centre)
    solver.set_state(h_init)
    volume0 = solver.total_volume()
    clipped0, dry0 = solver.clipped_cells, solver.dry_flips

    gates = [t_star_to_time(t_star, case.h0, case.gravity)
             for t_star in sample_t_stars]
    samples: list[dict] = []
    state = {"next": 0}

    def sample(sol: ShallowWater2D) -> None:
        while state["next"] < len(gates) and sol.time >= gates[state["next"]] - 1e-15:
            prof = normal_strip_profile(sol, case, angle_deg, strip_half_width,
                                        None, centre)
            graded = grade_profile(case, prof, sol.time)
            graded["t_star_target"] = float(sample_t_stars[state["next"]])
            exact_front = graded["front_exact"]
            graded["travelled_cells"] = float(exact_front / sol.dx)
            graded["resolved"] = bool(exact_front / sol.dx >= SERIES_MIN_CELLS)
            samples.append(graded)
            state["next"] += 1

    wall = time.perf_counter()
    steps = solver.advance(t_end, callback=sample)
    wall = time.perf_counter() - wall

    profile = normal_strip_profile(solver, case, angle_deg, strip_half_width,
                                   None, centre)
    metrics = grade_profile(case, profile, solver.time)
    metrics["mass_drift"] = _relative(solver.total_volume(), volume0)
    # mass inside the *graded* strip against the exact (infinite-domain) mass:
    # the global drift caused by the transmissive walls must not show up here.
    strip = profile["mask"]
    h_ex_strip, _ = exact_normal_profile(case, profile["s_field"][strip],
                                         solver.time)
    strip_num = float(solver.h[strip].sum())
    strip_ex = float(np.asarray(h_ex_strip, dtype=float).sum())
    metrics["strip_mass_error"] = _relative(strip_num, strip_ex)
    metrics["clipped_cells"] = int(solver.clipped_cells - clipped0)
    metrics["dry_flips"] = int(solver.dry_flips - dry0)

    graded_series = [smp for smp in samples if smp["resolved"]]
    series = {
        "samples": samples,
        "graded_samples": len(graded_series),
        "total_samples": len(samples),
        "max_graded_error": (max(s["series_error"] for s in graded_series)
                             if graded_series else float("nan")),
        "max_headline_error": (max(s["headline_error"] for s in graded_series)
                               if graded_series else float("nan")),
        "rule": ("dry: fan_front_estimate; wet: mid-level crossing at the "
                 "analytic mid level (same rule as dam_break2d.sample_2d); "
                 "the oracle-free data-driven bore is reported as "
                 "max_headline_error"),
        "ungraded_t_stars": [s["t_star_target"] for s in samples
                             if not s["resolved"]],
    }
    series["pass"] = bool(graded_series
                          and len(graded_series) >= 0.5 * len(samples)
                          and series["max_graded_error"] <= TOLERANCE)

    s_eval = np.asarray(profile["s"], dtype=float)[profile["keep"]]
    h_eval = np.asarray(profile["h"], dtype=float)[profile["keep"]]
    h_ex, _ = exact_normal_profile(case, s_eval, solver.time)
    return {
        "case": case.key,
        "angle_deg": float(angle_deg),
        "discretisation": {
            "n": int(n), "domain": float(domain), "dx": solver.dx,
            "cfl": cfl, "order": int(order), "steps": int(steps),
            "t_end": float(solver.time), "t_star_end": float(case.t_star_end),
            "wall_time_s": wall, "volume_initial": volume0,
            "volume_final": solver.total_volume(),
            "strip_half_width": float(strip_half_width),
            "bin_width": solver.dx,
        },
        "wall_clearance": wall_clearance(angle_deg, domain, strip_half_width),
        "metrics": metrics,
        "series": series,
        "verdict": verdict_oblique(metrics, series),
        "profile": {
            "s": s_eval.tolist(),
            "h_numeric": h_eval.tolist(),
            "h_exact": np.asarray(h_ex, dtype=float).tolist(),
            "u_n": np.asarray(profile["u_n"], dtype=float)[profile["keep"]].tolist(),
        },
    }


def verdict_oblique(metrics: dict, series: dict | None = None) -> dict:
    """Fold the graded metrics into one verdict (no threshold tuning)."""
    failing = []
    if not metrics["headline_error"] <= TOLERANCE:
        failing.append("front<5%")
    if not metrics["depth_l1"] <= TOLERANCE:
        failing.append("depth_l1<5%")
    if metrics["local_extrema"] != 0:
        failing.append("monotone(no_extrema)")
    if metrics.get("step_ratio_error") is not None and not (
            metrics["step_ratio_error"] <= TOLERANCE):
        failing.append("level_step<5%")
    out = {"failing": failing, "pass": not failing,
           "headline_error": metrics["headline_error"],
           "tolerance": TOLERANCE}
    if series is not None:
        out["series_pass"] = bool(series["pass"])
        out["series_max_graded_error"] = series["max_graded_error"]
    return out


def isotropy_rows(case: DamBreakCase, runs) -> dict:
    """Compare a list of angle runs with their 0 deg (first) control.

    The control is the axis-aligned run **on the same square grid**, so the
    difference isolates the grid-rotation effect (it is not the 1-D benchmark,
    which lives on a different mesh).
    """
    control = runs[0]
    rows = []
    for run in runs:
        row = {
            "angle_deg": run["angle_deg"],
            "headline_error": run["metrics"]["headline_error"],
            "depth_l1": run["metrics"]["depth_l1"],
            "local_extrema": run["metrics"]["local_extrema"],
            "series_max_graded_error": run["series"]["max_graded_error"],
            "delta_headline_vs_control": (run["metrics"]["headline_error"]
                                          - control["metrics"]["headline_error"]),
            "delta_depth_l1_vs_control": (run["metrics"]["depth_l1"]
                                          - control["metrics"]["depth_l1"]),
            "mass_drift": run["metrics"]["mass_drift"],
            "strip_mass_error": run["metrics"]["strip_mass_error"],
            "pass": run["verdict"]["pass"],
        }
        if case.kind != "dry":
            row["step_ratio_error"] = run["metrics"]["step_ratio_error"]
            row["delta_step_ratio_vs_control"] = (
                run["metrics"]["step_ratio_error"]
                - control["metrics"]["step_ratio_error"])
        row["improvement_vs_control"] = float(
            -row["delta_headline_vs_control"])
        row["direction_consistent"] = bool(
            row["delta_headline_vs_control"] <= OBLIQUE_AGREEMENT_MAX)
        rows.append(row)
    worst = max(r["delta_headline_vs_control"] for r in rows)
    best = min(r["delta_headline_vs_control"] for r in rows)
    return {
        "case": case.key,
        "n": int(runs[0]["discretisation"]["n"]),
        "angles_deg": [float(r["angle_deg"]) for r in runs],
        "agreement_max": OBLIQUE_AGREEMENT_MAX,
        "worst_delta_headline": float(worst),
        "best_delta_headline": float(best),
        "all_direction_consistent": bool(
            all(r["direction_consistent"] for r in rows)),
        "all_pass": bool(all(r["pass"] for r in rows)),
        "rows": rows,
    }


def isotropy_study(case: DamBreakCase, n: int,
                   angles=ANGLES_DEG, domain: float = DOMAIN,
                   cfl: float = 0.4, order: int = 2) -> dict:
    """Run the angle sweep on one grid and grade the rotation consistency."""
    runs = [run_oblique_case(case, angle, n, domain, cfl, order)
            for angle in angles]
    return isotropy_rows(case, runs)


# --------------------------------------------------------------------------- #
# reporting helpers
# --------------------------------------------------------------------------- #
def comparison_figure(payload: dict, path, grids: dict | None = None) -> str:
    """Profiles (numeric vs exact, per angle) + the error-vs-angle panel."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cases = list(payload["cases"].keys())
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.5))
    for col, key in enumerate(cases):
        block = payload["cases"][key]
        ax = axes[0, col]
        ref = None
        for run in block["runs"]:
            prof = run["profile"]
            s = np.asarray(prof["s"])
            ax.plot(s, np.asarray(prof["h_numeric"]) / block["h0"],
                    lw=1.1, label=f"num {run['angle_deg']:.2f} deg")
            ref = (s, np.asarray(prof["h_exact"]) / block["h0"])
        ax.plot(ref[0], ref[1], "k--", lw=1.4, label="exact (rotated)")
        ax.set_title(f"{key}: normal profile at t*={block['t_star']:.1f}")
        ax.set_xlabel("s = n.(X - Xc)  [m]")
        ax.set_ylabel("h / h0")
        ax.set_xlim(-3.0, 4.6)
        ax.legend(fontsize=7, loc="upper right")

        ax = axes[1, col]
        angles = [r["angle_deg"] for r in block["runs"]]
        ax.plot(angles, [100.0 * r["metrics"]["headline_error"] for r in block["runs"]],
                "o-", label="front error")
        ax.plot(angles, [100.0 * r["metrics"]["depth_l1"] for r in block["runs"]],
                "s-", label="depth L1")
        if block["case"] != "ritter-dry":
            ax.plot(angles, [100.0 * r["metrics"]["step_ratio_error"]
                             for r in block["runs"]], "^-", label="level step")
        ax.axhline(100.0 * TOLERANCE, color="r", ls=":", lw=1.2,
                   label="acceptance 5 %")
        ax.set_yscale("log")
        ax.set_title(f"{key}: graded errors vs dam-line angle (n={block['n']})")
        ax.set_xlabel("dam-line angle [deg]")
        ax.set_ylabel("relative error [%]")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle("Oblique dam-break verification: rotated Ritter/Stoker exact "
                 "solution on a Cartesian grid")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def write_artifacts_oblique(outdir, payload: dict) -> dict:
    """Write ``metrics.json`` / ``SUMMARY.md`` / the figure and return paths."""
    import json

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    metrics_path = outdir / "metrics.json"
    slim = {k: v for k, v in payload.items() if k != "profiles_full"}
    metrics_path.write_text(json.dumps(slim, indent=1, sort_keys=True))
    fig = comparison_figure(payload, outdir / "oblique_profiles.png")
    summary = payload.get("SUMMARY", "")
    (outdir / "SUMMARY.md").write_text(summary)
    return {"metrics": str(metrics_path), "figure": fig,
            "summary": str(outdir / "SUMMARY.md")}