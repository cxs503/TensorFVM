"""Two-dimensional confirmation of the Ritter/Stoker dam-break anchors.

The anchor ``CASE-FVM-DAMBREAK-STOKER`` grades the dam-break wave position
``x_f(t)`` and the still-water level step against the closed forms of Ritter
(1892) and Stoker (1957) at ``t* = 2``.  Until now that evidence came from
:class:`~tensorfvm.shallow_water.swe1d.ShallowWater1D` **only**: the 2-D solver
(:class:`~tensorfvm.shallow_water.swe2d.ShallowWater2D`) was verified against
the Thacker basin (well-balancedness, isotropy, shoreline), never against the
anchor's own exact solutions.  A defect in the 2-D operator (axis mix-up, a
``dx``/``dy`` scaling error, a transverse flux that does not vanish on a
uniform state, a boundary treatment that leaks) would therefore not show up in
any dam-break number.

This module closes that gap in the cheapest defensible way: it re-runs the
*identical* initial-boundary-value problem in a channel that is uniform in
``y``, reduces the field to the ``y``-averaged ``x``-profile and grades the
**same predeclared metrics with the same estimators** as the 1-D benchmark
(:func:`~tensorfvm.shallow_water.dam_break.fan_front_estimate` for the dry-bed
front, :func:`~tensorfvm.shallow_water.dam_break.data_driven_plateau` for the
wet-bed bore/plateau).

Scope and honesty statement
---------------------------
Because the initial data are exactly uniform in ``y`` and the bed is flat, the
2-D problem is *mathematically the same* problem as the 1-D one.  The
agreement obtained here is therefore evidence about the **implementation** of
the 2-D discrete operator (does it reduce to the 1-D answer, does it keep the
transverse direction quiet), not independent physical evidence, and it is
reported as such.  Two failure modes are probed:

``symmetry``
    With unperturbed initial data the transverse defect ``max|h - <h>_y|/h0``
    and the transverse velocity must sit at round-off.  This is *structural*
    (the transverse flux is exactly zero for a ``y``-uniform state), so it is
    graded at ``1e-12`` and explicitly **not** counted as accuracy evidence.
``modulation``
    A transverse depth modulation of amplitude :data:`MODULATION_AMPLITUDE`
    breaks the exact solution's assumptions on purpose.  It must not be
    amplified by the scheme (peak transverse defect ``<=
    TRANSVERSE_GROWTH_MAX * amplitude``) and must not push the graded wave
    position outside ``MODULATION_CONTAMINATION_MAX`` (half the acceptance
    tolerance).  Both limits are declared here, before any run.

All the numbers quoted in the docstrings and in
``docs/case-fvm-dambreak-stoker-2d/REPORT.md`` are produced by
``scripts/run_shallow_water_dam_break_2d.py``; nothing is transcribed.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from .dam_break import (
    EXTREMUM_FRACTION,
    FAN_WINDOW,
    SERIES_MIN_CELLS,
    TOLERANCE,
    DamBreakCase,
    _level_crossing,
    _local_extrema,
    _relative,
    _wetted_edge,
    data_driven_plateau,
    fan_front_estimate,
    series_metrics,
)
from .exact import stoker_constant_state, t_star_to_time
from .swe2d import ShallowWater2D

__all__ = [
    "DEFAULT_NY",
    "MODULATION_AMPLITUDE",
    "MODULATION_CONTAMINATION_MAX",
    "TRANSVERSE_GROWTH_MAX",
    "TRANSVERSE_ROUNDOFF",
    "FAN_RESOLVED_CELLS",
    "channel_profile",
    "compare_with_1d",
    "convergence_2d",
    "data_driven_series_metrics",
    "metrics_2d",
    "run_2d_case",
    "sample_2d",
    "transverse_defect",
    "transverse_modulation_study",
    "verdict_2d",
    "write_artifacts_2d",
]

#: cells across the channel; ``dy`` is deliberately much larger than ``dx`` so
#: an axis-scaling defect cannot hide behind equal spacings.
DEFAULT_NY = 8

#: transverse width of the verification channel (m)
CHANNEL_WIDTH = 1.0

#: grade of the *structural* transverse symmetry (fraction of ``h0``)
TRANSVERSE_ROUNDOFF = 1e-12

#: amplitude (fraction of ``h0``) of the transverse depth modulation probe
MODULATION_AMPLITUDE = 1e-2

#: a modulation of amplitude ``a`` must not be amplified: peak transverse
#: defect ``<= TRANSVERSE_GROWTH_MAX * a``.  Declared as "bounded, no
#: instability" -- not tuned to a measured number.
TRANSVERSE_GROWTH_MAX = 2.0

#: the same modulation must not move the graded wave position by more than
#: half the acceptance tolerance (it *is* a physical perturbation, so a
#: non-zero effect is expected; a large one would mean the metric is noise).
MODULATION_CONTAMINATION_MAX = 0.5 * TOLERANCE

#: cells the exact rarefaction **fan** must span before the oracle-free bore
#: detector (:func:`~tensorfvm.shallow_water.dam_break.data_driven_plateau`) is
#: graded.
#:
#: Same a-priori rule as :data:`~tensorfvm.shallow_water.dam_break.SERIES_MIN_CELLS`,
#: applied to the fan instead of the front: while the fan is narrower than the
#: smearing of the initial discontinuity, its numerical signature is the
#: *steepest* feature of the profile (jump ``h0 - h2`` compressed into a few
#: cells) and the max-gradient detector latches onto it instead of the bore --
#: the reading is then ~``1 + U t_fan / x_shock`` off, i.e. 100 ... 140 %.
#: Measured crossover for h1/h0 = 0.5: the first sample whose reading lands
#: inside 5 % has an exact fan width of 2.83 cells and the reading stays inside
#: 5 % for good from 3.76 cells -- **identically on nx = 200/400/800**
#: (the failure is physical, not a grid artefact: both gradients scale as
#: ``1/dx``).  Four cells is therefore the predeclared gate.
FAN_RESOLVED_CELLS = 4.0

#: the 2-D and 1-D solvers are separate implementations with different
#: transverse treatment, so their graded errors are compared with a tolerance
#: -- 0.5 percentage points, i.e. 10 % of the acceptance tolerance -- rather
#: than demanded identical.  A real operator defect (axis swap, dx/dy scaling,
#: leaking boundary) moves the headline error by far more than that.
ONE_D_AGREEMENT_MAX = 5e-3


# --------------------------------------------------------------------------- #
# profile extraction
# --------------------------------------------------------------------------- #
def channel_profile(solver: ShallowWater2D):
    """``(x, y-average, mid-row)`` of the depth field.

    The ``y``-average is the graded profile (it is what the 1-D answer should
    be); the mid-row is returned as a diagnostic so the report can show that
    the two agree to round-off on the symmetric run.
    """
    x, _ = solver.cell_centers()
    xc = x[:, 0]
    return xc, solver.h.mean(axis=1), solver.h[:, solver.ny // 2]


def transverse_defect(h, h0: float) -> float:
    """``max |h - <h>_y| / h0`` -- the amplitude of the transverse mode."""
    h = np.asarray(h, dtype=float)
    return float(np.abs(h - h.mean(axis=1, keepdims=True)).max() / h0)


def _dam_initial_state(case: DamBreakCase, solver: ShallowWater2D,
                       modulation: float = 0.0, modes: int = 1) -> np.ndarray:
    """Dam-break initial depth, uniformly in ``y`` unless modulated."""
    x, y = solver.cell_centers()
    h = np.where(x <= case.x_dam, case.h0, case.h1)
    if modulation:
        phase = 2.0 * math.pi * modes * (y - solver.y_min) / (
            solver.y_max - solver.y_min)
        h = h * (1.0 + modulation * np.cos(phase))
    return h


# --------------------------------------------------------------------------- #
# wave-position series (identical grading rule as the 1-D benchmark)
# --------------------------------------------------------------------------- #
def sample_2d(case: DamBreakCase, solver: ShallowWater2D) -> dict:
    """Wave position of the current 2-D state on the ``y``-averaged profile.

    The keys mirror :func:`tensorfvm.shallow_water.dam_break.front_sample` so
    that :func:`series_metrics` grades the 2-D series with the *same* rule.  On
    the wet bed the headline is the oracle-free data-driven bore
    (:func:`data_driven_plateau`, measured level, sub-cell crossing); the 1-D
    benchmark's analytic-level crossing is reported alongside as ``position``.
    """
    xc, prof, _ = channel_profile(solver)
    t = solver.time
    exact = case.exact_front(t)
    travelled = (exact - case.x_dam) / float(solver.dx)
    resolved = bool(travelled >= SERIES_MIN_CELLS)
    if case.kind == "dry":
        pos = _wetted_edge(xc, prof)
        fan = fan_front_estimate(xc, prof, t, case.gravity, case.h0)
        headline, cells, spread = fan["front"], fan["cells"], fan["spread"]
    else:
        sol = stoker_constant_state(case.h0, case.h1, case.gravity)
        pos = _level_crossing(xc, prof, 0.5 * (case.h1 + sol["h2"]))
        headline, cells, spread = pos, None, None
    out = {
        "t": t,
        "t_star": t * math.sqrt(case.gravity / case.h0),
        "position": pos,
        "exact": exact,
        "error": _relative(pos, exact),
        "headline": headline,
        "headline_error": (_relative(headline, exact)
                           if headline == headline else float("nan")),
        "headline_cells": cells,
        "headline_spread": spread,
        "resolved": resolved,
        "travelled_cells": travelled,
        "transverse_defect": transverse_defect(solver.h, case.h0),
    }
    if case.kind != "dry":
        # oracle-free companion reading + its own a-priori resolution gate
        sol = stoker_constant_state(case.h0, case.h1, case.gravity)
        fan_width = (sol["tail_speed"] - sol["head_speed"]) * t
        dd = data_driven_plateau(xc, prof)
        out["data_driven"] = dd["bore_x"]
        out["data_driven_error"] = (
            _relative(dd["bore_x"], exact) if dd["bore_x"] == dd["bore_x"]
            else float("nan"))
        out["fan_cells"] = fan_width / float(solver.dx)
        out["data_driven_resolved"] = bool(
            resolved and fan_width / float(solver.dx) >= FAN_RESOLVED_CELLS)
    return out


def data_driven_series_metrics(samples) -> dict:
    """Grade the oracle-free bore series with its own resolution gate.

    Mirrors :func:`~tensorfvm.shallow_water.dam_break.series_metrics` but
    grades the data-driven bore only on samples where the exact rarefaction fan
    spans at least :data:`FAN_RESOLVED_CELLS` cells (see that constant for the
    measured failure mode it excludes).  Samples that fail the gate stay visible
    under ``unresolved_*``.
    """
    graded = [s for s in samples
              if s.get("data_driven_resolved")
              and s.get("data_driven_error") == s.get("data_driven_error")]
    unresolved = [s for s in samples
                  if s.get("data_driven_error") == s.get("data_driven_error")
                  and s not in graded]
    best = max(graded, key=lambda s: s["data_driven_error"]) if graded else None
    worst = (max(unresolved, key=lambda s: s["data_driven_error"])
             if unresolved else None)
    return {
        "min_fan_cells": FAN_RESOLVED_CELLS,
        "min_cells": SERIES_MIN_CELLS,
        "samples": len(samples),
        "samples_graded": len(graded),
        "max_error": None if best is None else float(best["data_driven_error"]),
        "max_error_t_star": None if best is None else float(best["t_star"]),
        "unresolved_samples": len(unresolved),
        "unresolved_max_error": (None if worst is None
                                 else float(worst["data_driven_error"])),
        "tolerance": TOLERANCE,
        "pass": (best is not None
                 and float(best["data_driven_error"]) <= TOLERANCE),
    }


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def metrics_2d(case: DamBreakCase, solver: ShallowWater2D,
               volume0: float) -> dict:
    """All anchor metrics evaluated on the ``y``-averaged 2-D profile."""
    xc, prof, mid = channel_profile(solver)
    t = solver.time
    h_ex, _ = case.exact(xc, t)
    dx = float(solver.dx)
    tv_num = float(np.abs(np.diff(prof)).sum())
    tv_exact = float(case.h0) if case.kind == "dry" else float(case.h0 - case.h1)
    _, u, v = solver.primitive()
    out = {
        "t": t,
        "t_star": t * math.sqrt(case.gravity / case.h0),
        "depth_l1": float(np.abs(prof - h_ex).sum() / max(h_ex.sum(), 1e-300)),
        "depth_linf": float(np.abs(prof - h_ex).max() / case.h0),
        "depth_l1_mid_row": float(
            np.abs(mid - h_ex).sum() / max(h_ex.sum(), 1e-300)),
        "tv_numeric": tv_num,
        "tv_exact": tv_exact,
        "tv_ratio": tv_num / tv_exact,
        "local_extrema": _local_extrema(prof, EXTREMUM_FRACTION * case.h0),
        "mass_drift": _relative(solver.total_volume(), volume0),
        "h_min": float(solver.h.min()),
        "h_max_over_h0": float(solver.h.max() / case.h0),
        "clipped_cells": int(solver.clipped_cells),
        "dry_flips": int(solver.dry_flips),
        # --- transverse (y) diagnostics -------------------------------------
        "transverse_defect": transverse_defect(solver.h, case.h0),
        "transverse_defect_mid_vs_avg": float(
            np.abs(mid - prof).max() / case.h0),
        "v_max": float(np.abs(v).max()),
        "v_max_over_c0": float(np.abs(v).max()
                               / math.sqrt(case.gravity * case.h0)),
        "u_transverse_spread": float(
            np.abs(u - u.mean(axis=1, keepdims=True)).max()),
    }
    if case.kind == "dry":
        fan = fan_front_estimate(xc, prof, t, case.gravity, case.h0)
        x_f = case.exact_front(t)
        wetted = _wetted_edge(xc, prof)
        out.update({
            "front_exact": x_f,
            "front_fan": fan["front"],
            "front_fan_error": _relative(fan["front"], x_f),
            "front_fan_cells": fan["cells"],
            "front_fan_spread": fan["spread"],
            "front_fan_slope_error": fan["slope_error"],
            "front_fan_window": list(FAN_WINDOW),
            "front_wetted": wetted,
            "front_wetted_error": _relative(wetted, x_f),
            "headline_error": _relative(fan["front"], x_f),
        })
    else:
        sol = stoker_constant_state(case.h0, case.h1, case.gravity)
        x_shock = case.x_dam + sol["shock_speed"] * t
        dd = data_driven_plateau(xc, prof)
        out.update({
            "shock_exact": x_shock,
            # oracle-free (data-driven) readings: headline
            "bore_x": dd["bore_x"],
            "bore_x_error": _relative(dd["bore_x"], x_shock),
            "bore_x_cell": dd["bore_x_cell"],
            "bore_x_cell_error": _relative(dd["bore_x_cell"], x_shock),
            "bore_level": dd["bore_level"],
            "bore_method": dd["bore_method"],
            "h2": dd["h2"],
            "h2_error": _relative(dd["h2"], sol["h2"]),
            "h1": dd["h1"],
            "h1_error": _relative(dd["h1"], sol["h1"]),
            "step_ratio": dd["h2"] / dd["h1"] if dd["h1"] else float("nan"),
            "step_ratio_error": (_relative(dd["h2"] / dd["h1"], sol["step_ratio"])
                                 if dd["h1"] else float("nan")),
            "n_plateau": dd["n_plateau"],
            "h2_exact": sol["h2"],
            "h1_exact": sol["h1"],
            "step_ratio_exact": sol["step_ratio"],
            # analytic-level crossing, kept for parity with the 1-D report
            "shock_position": _level_crossing(
                xc, prof, 0.5 * (case.h1 + sol["h2"])),
            "headline_error": _relative(dd["bore_x"], x_shock),
        })
    return out


def verdict_2d(case: DamBreakCase, metrics: dict) -> dict:
    """Pass/fail of the anchor metrics plus the transverse checks."""
    if case.kind == "dry":
        headline = {"front_fan<5%": metrics["front_fan_error"]}
    else:
        headline = {
            "bore_data_driven<5%": metrics["bore_x_error"],
            "step_ratio<5%": metrics["step_ratio_error"],
            "plateau_h2<5%": metrics["h2_error"],
        }
    checks = dict(headline)
    checks["depth_l1<5%"] = metrics["depth_l1"]
    checks["monotone(no_extrema)"] = 0.0 if metrics["local_extrema"] == 0 else 1.0
    checks["tv_no_growth<0.5%"] = max(0.0, metrics["tv_ratio"] - 1.0)
    checks["transverse_symmetry<=1e-12"] = metrics["transverse_defect"]
    checks["v_transverse<=1e-12"] = metrics["v_max"]
    checks["vanish_mass"] = 0.0 if metrics["h_min"] >= 0.0 else 1.0
    limits = {key: TOLERANCE for key in checks}
    limits["monotone(no_extrema)"] = 0.0
    limits["tv_no_growth<0.5%"] = 5e-3
    limits["transverse_symmetry<=1e-12"] = TRANSVERSE_ROUNDOFF
    limits["v_transverse<=1e-12"] = TRANSVERSE_ROUNDOFF
    limits["vanish_mass"] = 0.0
    verdict = {
        key: {"value": float(value), "limit": float(limits[key]),
              "pass": bool(float(value) <= float(limits[key]))}
        for key, value in checks.items()
    }
    verdict["all_pass"] = bool(all(item["pass"] for item in verdict.values()))
    return verdict


# --------------------------------------------------------------------------- #
# runs
# --------------------------------------------------------------------------- #
def run_2d_case(case: DamBreakCase, nx: int = 1600, ny: int = DEFAULT_NY,
                cfl: float = 0.4, order: int = 2,
                t_star: float | None = None, modulation: float = 0.0,
                modes: int = 1) -> dict:
    """Integrate the dam-break problem with the 2-D solver and grade it."""
    t_star_end = case.t_star_end if t_star is None else float(t_star)
    t_end = t_star_to_time(t_star_end, case.h0, case.gravity)
    solver = ShallowWater2D(nx, ny, case.x_min, case.x_max, 0.0, CHANNEL_WIDTH,
                            gravity=case.gravity, cfl=cfl, order=order)
    solver.set_state(_dam_initial_state(case, solver, modulation, modes))
    volume0 = solver.total_volume()
    samples: list[dict] = []
    gate = {"next": 0.0}
    defect_peak = {"value": transverse_defect(solver.h, case.h0)}

    def sample(sol: ShallowWater2D) -> None:
        defect = transverse_defect(sol.h, case.h0)
        defect_peak["value"] = max(defect_peak["value"], defect)
        if sol.time < gate["next"]:
            return
        samples.append(sample_2d(case, sol))
        gate["next"] = sol.time + t_end / 25.0

    wall = time.perf_counter()
    steps = solver.advance(t_end, callback=sample)
    wall = time.perf_counter() - wall

    metrics = metrics_2d(case, solver, volume0)
    metrics["transverse_defect_peak"] = defect_peak["value"]
    metrics["time_to_front_series"] = samples
    metrics["front_series"] = series_metrics(case, samples)
    metrics["front_series_max_error"] = metrics["front_series"]["max_error"]
    if case.kind != "dry":
        metrics["front_series_data_driven"] = data_driven_series_metrics(samples)
    xc, prof, mid = channel_profile(solver)
    h_ex, _ = case.exact(xc, solver.time)
    return {
        "case": case.summary(),
        "discretisation": {
            "nx": solver.nx,
            "ny": solver.ny,
            "dx": solver.dx,
            "dy": solver.dy,
            "channel_width": CHANNEL_WIDTH,
            "cfl": cfl,
            "order": order,
            "steps": steps,
            "t_end": solver.time,
            "t_star_end": t_star_end,
            "wall_time_s": wall,
            "volume_initial": volume0,
            "volume_final": solver.total_volume(),
            "modulation": modulation,
            "modes": modes,
        },
        "metrics": metrics,
        "verdict": verdict_2d(case, metrics),
        "profile": {
            "x": xc.tolist(),
            "h_numeric_y_average": prof.tolist(),
            "h_numeric_mid_row": mid.tolist(),
            "h_exact": np.asarray(h_ex, dtype=float).tolist(),
        },
    }


def convergence_2d(case: DamBreakCase, resolutions=(200, 400, 800),
                   ny: int = DEFAULT_NY, cfl: float = 0.4,
                   order: int = 2) -> dict:
    """Grid refinement along ``x`` at fixed transverse resolution."""
    rows = []
    for nx in resolutions:
        run = run_2d_case(case, nx=nx, ny=ny, cfl=cfl, order=order)
        m = run["metrics"]
        rows.append({
            "nx": nx,
            "dy": run["discretisation"]["dy"],
            "dx": run["discretisation"]["dx"],
            "depth_l1": m["depth_l1"],
            "headline_error": m["headline_error"],
            "front_series_max_error": m.get("front_series_max_error"),
            "tv_ratio": m["tv_ratio"],
            "local_extrema": m["local_extrema"],
            "transverse_defect": m["transverse_defect"],
            "v_max": m["v_max"],
            "wall_time_s": run["discretisation"]["wall_time_s"],
        })
    dx = np.array([row["dx"] for row in rows])
    l1 = np.array([row["depth_l1"] for row in rows])
    head = np.array([row["headline_error"] for row in rows])
    fit_l1 = float(np.polyfit(np.log(dx), np.log(l1), 1)[0]) if len(rows) > 2 else float("nan")
    positive = head > 0.0
    fit_head = (float(np.polyfit(np.log(dx[positive]), np.log(head[positive]), 1)[0])
                if int(np.count_nonzero(positive)) > 2 else float("nan"))
    return {
        "case": case.key,
        "ny": ny,
        "cfl": cfl,
        "order": order,
        "rows": rows,
        "observed_order_l1": fit_l1,
        "observed_order_headline": fit_head,
    }


def transverse_modulation_study(case: DamBreakCase, nx: int = 400,
                                ny: int = DEFAULT_NY,
                                amplitude: float = MODULATION_AMPLITUDE,
                                modes: int = 1) -> dict:
    """Response to a transverse depth modulation (see module docstring).

    Baseline (``amplitude = 0``) and modulated runs at the same grid; the
    modulation is applied to the initial depth only.  Reported: how much the
    transverse mode grows relative to its initial amplitude, the peak
    transverse velocity, and the shift of the graded wave-position error.
    """
    base = run_2d_case(case, nx=nx, ny=ny)
    pert = run_2d_case(case, nx=nx, ny=ny, modulation=amplitude, modes=modes)
    growth = pert["metrics"]["transverse_defect_peak"] / amplitude
    err_base = base["metrics"]["headline_error"]
    err_pert = pert["metrics"]["headline_error"]
    return {
        "case": case.key,
        "nx": nx,
        "ny": ny,
        "amplitude": amplitude,
        "modes": modes,
        "baseline_headline_error": err_base,
        "modulated_headline_error": err_pert,
        "headline_error_shift": err_pert - err_base,
        "transverse_defect_peak_baseline": base["metrics"]["transverse_defect_peak"],
        "transverse_defect_peak": pert["metrics"]["transverse_defect_peak"],
        "growth_factor": growth,
        "v_max_modulated": pert["metrics"]["v_max"],
        "mass_drift_modulated": pert["metrics"]["mass_drift"],
        "series": {
            "t_star": [s["t_star"] for s in pert["metrics"]["time_to_front_series"]],
            "baseline_defect": [
                s["transverse_defect"]
                for s in base["metrics"]["time_to_front_series"]],
            "modulated_defect": [
                s["transverse_defect"]
                for s in pert["metrics"]["time_to_front_series"]],
        },
        "criteria": {
            "growth<=%.1f" % TRANSVERSE_GROWTH_MAX: {
                "value": growth, "limit": TRANSVERSE_GROWTH_MAX,
                "pass": bool(growth <= TRANSVERSE_GROWTH_MAX)},
            "headline_err<=%.3f" % MODULATION_CONTAMINATION_MAX: {
                "value": err_pert, "limit": MODULATION_CONTAMINATION_MAX,
                "pass": bool(err_pert <= MODULATION_CONTAMINATION_MAX)},
        },
    }


def compare_with_1d(case: DamBreakCase, nx: int = 800,
                    ny: int = DEFAULT_NY, cfl: float = 0.4, order: int = 2,
                    t_star: float | None = None) -> dict:
    """Run the same problem through both solvers at the same ``x`` resolution.

    The 2-D field is reduced to its ``y``-average profile, so this compares the
    two *discrete operators* on one problem: agreement is expected up to
    implementation detail, and a large difference would localise a 2-D operator
    defect.  Both sides use the same estimators (dry bed: fan front and depth
    L1; wet bed: oracle-free bore, step ratio, depth L1).
    """
    from .dam_break import run_case

    two = run_2d_case(case, nx=nx, ny=ny, cfl=cfl, order=order, t_star=t_star)
    one = run_case(case, n=nx, cfl=cfl, order=order, t_star=t_star)
    m2, m1 = two["metrics"], one["metrics"]
    if case.kind == "dry":
        pairs = {"front_fan_error": (m2["front_fan_error"], m1["front_fan_error"]),
                 "depth_l1": (m2["depth_l1"], m1["depth_l1"])}
    else:
        pairs = {"bore_error": (m2["bore_x_error"], m1["dd"]["bore_x_error"]),
                 "step_ratio_error": (m2["step_ratio_error"],
                                      m1["dd"]["step_ratio_error"]),
                 "depth_l1": (m2["depth_l1"], m1["depth_l1"])}
    rows = {name: {"two_d": float(a), "one_d": float(b),
                   "difference": float(abs(a - b))}
            for name, (a, b) in pairs.items()}
    worst = max(rows.values(), key=lambda row: row["difference"])
    return {
        "case": case.key,
        "nx": nx,
        "ny": ny,
        "rows": rows,
        "max_difference": worst["difference"],
        "limit": ONE_D_AGREEMENT_MAX,
        "pass": bool(worst["difference"] <= ONE_D_AGREEMENT_MAX),
    }



# --------------------------------------------------------------------------- #
# artifacts
# --------------------------------------------------------------------------- #
def _summary_markdown(payload: dict) -> str:
    meta = payload["meta"]
    lines = [
        "# Ritter / Stoker dam-break in the 2-D solver -- result summary",
        "",
        "Generated by `scripts/run_shallow_water_dam_break_2d.py` on %s."
        % meta["generated_at"],
        "",
        "Acceptance threshold: headline metric < %g %% relative error; waveform"
        " monotone; transverse symmetry at round-off (%g)."
        % (100.0 * meta["tolerance"], TRANSVERSE_ROUNDOFF),
        "",
        "## Grid (x) convergence at fixed ny = %d" % meta["ny"],
        "",
        "| case | nx | dy [m] | dx [m] | depth L1 | headline error | series max"
        " | TV ratio | extrema | transverse defect | v_max | wall [s] |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for case_key, conv in payload["convergence"].items():
        for row in conv["rows"]:
            lines.append(
                "| %s | %d | %.5f | %.5f | %.3e | %.3f %% | %s | %.6f | %d |"
                " %.1e | %.1e | %.2f |"
                % (case_key, row["nx"], row["dy"], row["dx"], row["depth_l1"],
                   100.0 * row["headline_error"],
                   "-" if row["front_series_max_error"] is None
                   else "%.3f %%" % (100.0 * row["front_series_max_error"]),
                   row["tv_ratio"], row["local_extrema"],
                   row["transverse_defect"], row["v_max"], row["wall_time_s"]))
        lines.append("| %s | observed order L1 = **%.3f**, headline = %.3f |"
                     % (case_key, conv["observed_order_l1"],
                        conv["observed_order_headline"]))
        lines.append("")
    lines += [
        "## Graded run at nx = %d" % meta["graded_nx"],
        "",
        "| case | metric | reference (closed form) | 2-D solver | relative error |"
        " < 5 % |",
        "|---|---|---|---|---|---|",
    ]
    for case_key, run in payload["runs"].items():
        m = run["metrics"]
        if run["case"]["kind"] == "dry":
            rows = [
                ("front position x_f (fan, headline) [m]", m["front_exact"],
                 m["front_fan"], m["front_fan_error"]),
                ("front position (wetted edge, diagnostic) [m]",
                 m["front_exact"], m["front_wetted"], m["front_wetted_error"]),
            ]
        else:
            rows = [
                ("bore position x_s (data-driven, headline) [m]",
                 m["shock_exact"], m["bore_x"], m["bore_x_error"]),
                ("bore position (cell centre, negative control) [m]",
                 m["shock_exact"], m["bore_x_cell"], m["bore_x_cell_error"]),
                ("post-bore depth h2 [m]", m["h2_exact"], m["h2"], m["h2_error"]),
                ("still-water step h2/h1 [-]", m["step_ratio_exact"],
                 m["step_ratio"], m["step_ratio_error"]),
            ]
        for label, ref, val, err in rows:
            lines.append("| %s | %s | %.6f | %.6f | %.3f %% | %s |"
                         % (case_key, label, ref, val, 100.0 * err,
                            "yes" if err <= meta["tolerance"] else "no"))
        lines.append("| %s | depth L1 | - | - | %.3f %% | %s |"
                     % (case_key, 100.0 * m["depth_l1"],
                        "yes" if m["depth_l1"] <= meta["tolerance"] else "no"))
        lines.append("")
        lines.append(
            "* x_f(t) series: graded %d/%d (front must travel >= %g cells); max"
            " error %s"
            % (m["front_series"]["samples_graded"], m["front_series"]["samples"],
               m["front_series"]["min_cells"],
               "-" if m["front_series"]["max_error"] is None
               else "%.3f %%" % (100.0 * m["front_series"]["max_error"])))
        lines.append(
            "* transverse defect %.2e (limit %.0e), v_max %.2e m/s, mass drift"
            " %.2e, h_min %.3e m"
            % (m["transverse_defect"], TRANSVERSE_ROUNDOFF, m["v_max"],
               m["mass_drift"], m["h_min"]))
        dd_series = m.get("front_series_data_driven")
        if dd_series:
            lines.append(
                "* oracle-free bore series: graded %d/%d (exact fan must span"
                " >= %g cells); max error %s; excluded samples max %s"
                % (dd_series["samples_graded"], dd_series["samples"],
                   dd_series["min_fan_cells"],
                   "-" if dd_series["max_error"] is None
                   else "%.3f %%" % (100.0 * dd_series["max_error"]),
                   "-" if dd_series["unresolved_max_error"] is None
                   else "%.1f %%" % (100.0 * dd_series["unresolved_max_error"])))
        lines.append("* interior local extrema above %g m: %d; TV ratio %.6f"
                     % (EXTREMUM_FRACTION * run["case"]["h0"], m["local_extrema"],
                        m["tv_ratio"]))
        lines.append("* all headline checks pass: **%s**"
                     % ("yes" if run["verdict"]["all_pass"] else "no"))
        lines.append("")
    lines += [
        "## Transverse modulation probe",
        "",
        "| case | nx | amplitude | baseline headline err | modulated err |"
        " shift | peak defect / amplitude | v_max | verdict |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for case_key, study in payload["modulation"].items():
        lines.append("| %s | %d | %.3f | %.3f %% | %.3f %% | %+.3f pp | %.3f |"
                     " %.2e | %s |"
                     % (case_key, study["nx"], study["amplitude"],
                        100.0 * study["baseline_headline_error"],
                        100.0 * study["modulated_headline_error"],
                        100.0 * study["headline_error_shift"],
                        study["growth_factor"], study["v_max_modulated"],
                        "pass" if all(item["pass"] for item
                                      in study["criteria"].values()) else "FAIL"))
    lines.append("")
    if payload.get("consistency"):
        lines += [
            "## 2-D versus 1-D operator (same problem, same x resolution)",
            "",
            "| case | nx | metric | 2-D | 1-D | |difference| | <= 0.5 pp |",
            "|---|---|---|---|---|---|---|",
        ]
        for case_key, comp in payload["consistency"].items():
            for name, row in comp["rows"].items():
                lines.append(
                    "| %s | %d | %s | %.4f %% | %.4f %% | %.4f pp | %s |"
                    % (case_key, comp["nx"], name,
                       100.0 * row["two_d"], 100.0 * row["one_d"],
                       100.0 * row["difference"],
                       "yes" if row["difference"] <= comp["limit"] else "no"))
        lines.append("")
    return "\n".join(lines)


def write_artifacts_2d(outdir, payload: dict) -> dict:
    """Write the JSON artifacts, the CSV profiles and the markdown summary."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, key in (("metrics_2d.json", "runs"),
                      ("convergence_2d.json", "convergence"),
                      ("modulation_2d.json", "modulation"),
                      ("exactness_2d.json", "exactness"),
                      ("consistency_2d.json", "consistency")):
        if key not in payload:
            continue
        path = out / name
        path.write_text(json.dumps(payload[key], indent=2, sort_keys=True)
                        + "\n", encoding="utf-8")
        written[name] = name
    for case_key, run in payload["runs"].items():
        prof = run["profile"]
        name = "profile_2d_%s_nx%d.csv" % (case_key, run["discretisation"]["nx"])
        path = out / name
        with path.open("w", encoding="utf-8") as handle:
            handle.write("x,h_numeric_y_average,h_numeric_mid_row,h_exact\n")
            for row in zip(prof["x"], prof["h_numeric_y_average"],
                           prof["h_numeric_mid_row"], prof["h_exact"]):
                handle.write("%.10g,%.10g,%.10g,%.10g\n" % row)
        written[name] = name
    summary = _summary_markdown(payload)
    (out / "SUMMARY_2D.md").write_text(summary, encoding="utf-8")
    written["SUMMARY_2D.md"] = "SUMMARY_2D.md"
    manifest = {"meta": payload["meta"], "artifacts": written}
    (out / "manifest_2d.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    written["manifest_2d.json"] = "manifest_2d.json"
    return written