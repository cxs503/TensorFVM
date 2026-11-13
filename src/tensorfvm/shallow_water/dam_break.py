"""Dam-break benchmark: Ritter (dry bed) and Stoker (wet bed) at ``t* = 2``.

The benchmark is the shallow-water analogue of the Sod shock tube: a dam at
``x = x_dam`` separates still water of depth ``h0`` (left) from depth ``h1``
(right).  Two exact references are used, both evaluated in closed form:

* ``ritter-dry``  -- ``h1 = 0``; front ``x_f = 2 t sqrt(g h0)`` (Ritter 1892).
* ``stoker-wet``  -- ``h0 > h1 > 0``; rarefaction + constant state + bore of
  speed ``U`` (Stoker 1957, chapter 10).

Metrics (predeclared, see ``docs/shallow-water-dam-break/REPORT.md``):

``front_fan`` (dry bed headline) / ``shock_position`` (wet bed headline)
    Wave position.  Dry bed: mean over ``FAN_WINDOW`` of the per-cell exact
    identity ``x_f = x_i + 3 t sqrt(g h_i)`` of the Ritter fan
    (:func:`fan_front_estimate`) -- the estimator inherits the *shape* of the
    closed form instead of a depth threshold.  Wet bed: the sub-cell position
    where the depth crosses ``(h1 + h2) / 2``, linearly interpolated between
    cell centres.  Relative error against the closed form.

``front_wetted`` (dry bed, legacy diagnostic)
    Outermost cell with ``h > 0``.  It looks threshold-free but is in fact a
    depth-floor detector: the tip is a zero-depth cusp, so the reading is fixed
    by the numerical depth floor and does **not** converge under refinement
    (1.56 % -> 2.66 % -> 2.89 % -> 2.46 % for n = 400 ... 3200, while the fan
    estimator on the same states goes 0.106 % -> 0.039 % -> 0.017 % -> 0.008 %).
    Reported in every run and in the verdict for continuity, not as a headline.

``step_ratio``
    The still-water level step across the bore, ``h2 / h1`` measured as the
    mean depth over the middle half of the exact plateau divided by the mean
    depth in the undisturbed downstream region.  Relative error.

``depth_l1`` / ``depth_linf``
    Relative ``L1`` error of the depth profile over the whole domain,
    ``sum|h - h_exact| dx / sum h_exact dx``, and ``max|h - h_exact| / h0``.

``monotonicity``
    Total variation of the numerical depth against the exact TV (``h0 - h1``
    for the wet bed, ``h0`` for the dry bed) plus the number of interior local
    extrema of amplitude ``> 1e-4 h0``: the "waveform must stay monotone, no
    spurious oscillation" part of the acceptance criterion.

``scaled_front_error`` (dry bed only, secondary)
    Position where the depth crosses ``h0/100`` plus the exact fan offset
    ``3 t sqrt(g h_level)`` back to the tip.  Reported because the tip is a
    zero-depth cusp that any finite-volume scheme smears; the wetted-edge
    measure above is the primary one.

All relative errors are non-negative ``|measured - exact| / exact``.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass

import numpy as np

from .exact import (
    check_stoker_solution,
    ritter_front_position,
    ritter_state,
    stoker_constant_state,
    stoker_state,
    t_star_to_time,
)
from .swe1d import ShallowWater1D

__all__ = [
    "DamBreakCase",
    "RITTER_DRY",
    "STOKER_WET",
    "CASES",
    "run_case",
    "series_metrics",
    "convergence_study",
    "sensitivity_study",
    "front_definition_study",
    "exactness_checks",
    "write_artifacts",
    "figure_profiles_and_convergence",
    "TOLERANCE",
]

#: acceptance threshold on every headline metric (task criterion: < 5 %)
TOLERANCE = 0.05

#: depth level (fraction of h0) of the secondary dry-bed front metric
FRONT_LEVEL_FRACTION = 1e-2

#: amplitude (fraction of h0) above which a local extremum counts as oscillation
EXTREMUM_FRACTION = 1e-4

#: depth-fraction window ``(h / h0)`` used by :func:`fan_front_estimate`.
#:
#: The window must sit **inside the resolved rarefaction** -- deep enough that
#: the numerical depth is not dominated by the smeared zero-depth cusp, shallow
#: enough that it is not buried in the still-water plateau.  The bounds are
#: declared here (not inferred from a run) and the sensitivity to them is
#: measured in ``docs/verification-dam-break-front-metric/REPORT.md``.
FAN_WINDOW = (1e-2, 2e-1)

#: cell count the exact front must have travelled before a sample of the
#: ``x_f(t)`` series is *graded*.
#:
#: This is an a-priori resolution criterion, declared before any run: while the
#: front/bore still sits inside the smearing region of the initial discontinuity
#: (a few cells), its measured position is a statement about the smearing, not
#: about the wave.  Samples below the threshold are still recorded and reported
#: (as ``unresolved_max_error``); they are simply not admitted to the pass/fail
#: series metric.
SERIES_MIN_CELLS = 4.0


@dataclass(frozen=True)
class DamBreakCase:
    """Geometry and reference solution of one dam-break case."""

    key: str
    kind: str            # "dry" | "wet"
    h0: float
    h1: float
    gravity: float
    x_dam: float
    x_min: float
    x_max: float
    t_star_end: float
    reference: str

    @property
    def t_end(self) -> float:
        return t_star_to_time(self.t_star_end, self.h0, self.gravity)

    def exact(self, x, t=None):
        """Exact ``(h, u)`` at ``x`` and time ``t`` (defaults to ``t_end``)."""
        t = self.t_end if t is None else t
        if self.kind == "dry":
            return ritter_state(x, t, self.h0, self.gravity, self.x_dam)
        return stoker_state(x, t, self.h0, self.h1, self.gravity, self.x_dam)

    def exact_front(self, t=None) -> float:
        """Position of the leading wave (front or bore) at ``t``."""
        t = self.t_end if t is None else t
        if self.kind == "dry":
            return ritter_front_position(t, self.h0, self.gravity, self.x_dam)
        sol = stoker_constant_state(self.h0, self.h1, self.gravity)
        return self.x_dam + sol["shock_speed"] * t

    def initial_mass(self) -> float:
        """Volume per unit width of the initial condition.

        The left inflow boundary is placed at ``x_min`` for an analytic
        estimate only when it lies upstream of the dam; the solver's own
        volume is the reference for the drift metric.
        """
        return self.h0 * (self.x_dam - self.x_min) + self.h1 * (self.x_max - self.x_dam)

    def summary(self) -> dict:
        return {
            "key": self.key,
            "kind": self.kind,
            "h0": self.h0,
            "h1": self.h1,
            "gravity": self.gravity,
            "x_dam": self.x_dam,
            "x_min": self.x_min,
            "x_max": self.x_max,
            "t_star_end": self.t_star_end,
            "t_end": self.t_end,
            "reference": self.reference,
        }


RITTER_DRY = DamBreakCase(
    key="ritter-dry", kind="dry", h0=1.0, h1=0.0, gravity=9.81, x_dam=0.0,
    x_min=-4.0, x_max=6.0, t_star_end=2.0,
    reference="Ritter (1892) Z. Ver. Deutsch. Ing. 36:947-954",
)

STOKER_WET = DamBreakCase(
    key="stoker-wet", kind="wet", h0=1.0, h1=0.5, gravity=9.81, x_dam=0.0,
    x_min=-4.0, x_max=6.0, t_star_end=2.0,
    reference="Stoker (1957) Water Waves, ch. 10 (bore on a wet bed)",
)

CASES = {case.key: case for case in (RITTER_DRY, STOKER_WET)}


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def _relative(measured: float, exact: float) -> float:
    return abs(measured - exact) / abs(exact) if exact != 0.0 else abs(measured)


def _wetted_edge(x: np.ndarray, h: np.ndarray, dry_depth: float = 0.0) -> float:
    """Outermost cell centre whose depth exceeds ``dry_depth``."""
    wet = np.nonzero(h > dry_depth)[0]
    if wet.size == 0:
        return float("nan")
    return float(x[wet[-1]])


def _level_crossing(x: np.ndarray, h: np.ndarray, level: float) -> float:
    """Sub-cell abscissa where ``h`` falls through ``level`` (linear, rightmost)."""
    above = np.nonzero(h > level)[0]
    if above.size == 0:
        return float("nan")
    j = int(above[-1])
    if j + 1 >= h.size:
        return float(x[-1])
    h0, h1 = h[j], h[j + 1]
    if h0 == h1:
        return float(x[j])
    return float(x[j] + (h0 - level) / (h0 - h1) * (x[j + 1] - x[j]))


def _local_extrema(h: np.ndarray, amplitude: float) -> int:
    """Interior local maxima/minima whose prominence exceeds ``amplitude``."""
    if h.size < 3:
        return 0
    left = h[1:-1] - h[:-2]
    right = h[2:] - h[1:-1]
    peak = (left > 0) & (right < 0) & (np.abs(left) > amplitude) & (np.abs(right) > amplitude)
    dip = (left < 0) & (right > 0) & (np.abs(left) > amplitude) & (np.abs(right) > amplitude)
    return int(np.count_nonzero(peak | dip))


def fan_front_estimate(x, h, t: float, gravity: float, h0: float,
                       window=FAN_WINDOW) -> dict:
    """Dry-bed front position from the **exact rarefaction shape**.

    Behind the tip the Ritter (1892) solution is a straight line in
    ``c = sqrt(g h)``::

        c(x, t) = (2 c0 - x / t) / 3,        c0 = sqrt(g h0),

    so every cell inside the fan individually carries the front,

    .. math:: x_f = 2 c0 t = x_i + 3 t \sqrt{g h_i},

    *exactly* -- no threshold, no extrapolation length.  Averaging that identity
    over a window of well-resolved fan cells is therefore a front estimator whose
    error measures the **shape** error of the numerical fan, not the position of
    a user-chosen depth contour.  This is the property that a ``h > 0`` wetted
    edge does *not* have: the tip is a zero-depth cusp with ``h ~ (x_f - x)^2``,
    so "outermost wetted cell" is a depth-floor detector whose reading depends on
    the numerical floor rather than on ``dx`` (measured non-convergence: 1.56 % ->
    2.66 % -> 2.89 % -> 2.46 % for n = 400 ... 3200).

    Parameters
    ----------
    x, h
        Cell centres and depths of the numerical solution.
    t, gravity, h0
        Time of the state, gravity, undisturbed upstream depth.
    window
        ``(lo, hi)`` in units of ``h / h0``; cells outside are ignored.

    Returns
    -------
    dict with ``front`` (mean estimate, m), ``cells`` (window size),
    ``spread`` (std of the per-cell estimates, m -- a resolution diagnostic),
    ``slope`` / ``slope_error`` (fitted ``dc/dx`` vs the exact ``-1/(3t)``).
    """
    x = np.asarray(x, dtype=float)
    h = np.asarray(h, dtype=float)
    nan = float("nan")
    out = {"front": nan, "cells": 0, "spread": nan, "slope": nan,
           "slope_error": nan, "window": [float(window[0]), float(window[1])]}
    if t <= 0.0 or h.size < 3:
        return out
    c = np.sqrt(np.maximum(gravity * h, 0.0))
    frac = h / h0
    sel = (frac >= window[0]) & (frac <= window[1])
    if int(np.count_nonzero(sel)) < 3:
        return out
    xs = x[sel]
    per_cell = xs + 3.0 * t * c[sel]
    out["front"] = float(np.mean(per_cell))
    out["cells"] = int(per_cell.size)
    out["spread"] = float(np.std(per_cell))
    slope = float(np.polyfit(xs, c[sel], 1)[0])
    out["slope"] = slope
    out["slope_error"] = abs(slope + 1.0 / (3.0 * t)) / (1.0 / (3.0 * t))
    return out


def front_sample(case: DamBreakCase, solver: ShallowWater1D) -> dict:
    """Wave position of the current state vs the exact position at that time.

    The sample carries **two** definitions on purpose:

    ``position`` / ``error``
        Legacy definition (dry: outermost wet cell, wet: mid-level crossing).
        Kept because downstream artefacts quote it; for the dry bed it is a
        depth-floor detector that does not converge (see
        :func:`fan_front_estimate`), so it must not be graded.
    ``headline`` / ``headline_error``
        The graded definition: the shape-consistent fan estimator for the dry
        bed (consistent with the headline metric of :func:`_dry_metrics`) and
        the mid-level crossing for the wet bed.  Grading a *series* with the
        legacy estimator was a defect: at ``t* = 0.005`` it reads a 212 %
        error on the dry bed because the numerical wetting front runs several
        cells ahead of the exact zero-depth tip (see
        ``docs/case-fvm-dambreak-stoker/AUDIT.md``).

    ``resolved`` is the a-priori resolution flag (``SERIES_MIN_CELLS``).
    """
    t = solver.time
    exact = case.exact_front(t)
    dx = float(solver.dx)
    travelled = (exact - case.x_dam) / dx
    resolved = bool(travelled >= SERIES_MIN_CELLS)
    if case.kind == "dry":
        pos = _wetted_edge(solver.x, solver.h)
        fan = fan_front_estimate(solver.x, solver.h, t, case.gravity, case.h0)
        headline = fan["front"]
        cells = fan["cells"]
        spread = fan["spread"]
    else:
        sol = stoker_constant_state(case.h0, case.h1, case.gravity)
        pos = _level_crossing(solver.x, solver.h, 0.5 * (case.h1 + sol["h2"]))
        headline = pos
        cells = None
        spread = None
    return {
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
    }


def series_metrics(case: DamBreakCase, samples,) -> dict:
    """Grade the *whole* ``x_f(t)`` series, not just its final state.

    The acceptance criterion is a statement about the wave position *as a
    function of time* (``x_f(t)`` vs analytic); grading only ``t* = 2`` would
    leave the early transient unverified.  Graded samples are those with
    ``resolved`` true (see :data:`SERIES_MIN_CELLS`); the rest are reported
    under ``unresolved_*`` so a regression in the first few steps is still
    visible instead of silently dropped.
    """
    graded = [s for s in samples
              if s["resolved"] and s["headline_error"] == s["headline_error"]]
    # every sample that is not graded stays visible -- including the ones whose
    # headline estimator is undefined (too few fan cells): dropping them
    # silently would hide the first steps of the transient.
    unresolved = [s for s in samples if s not in graded]
    named = [s for s in unresolved if s["headline_error"] == s["headline_error"]]
    best = max(graded, key=lambda s: s["headline_error"]) if graded else None
    worst_unres = max(named, key=lambda s: s["headline_error"]) if named else None
    legacy = [s for s in samples if s["error"] == s["error"]]
    worst_legacy = max(legacy, key=lambda s: s["error"]) if legacy else None
    return {
        "min_cells": SERIES_MIN_CELLS,
        "samples": len(samples),
        "samples_graded": len(graded),
        "max_error": None if best is None else float(best["headline_error"]),
        "max_error_t_star": None if best is None else float(best["t_star"]),
        "unresolved_samples": len(unresolved),
        "unresolved_undefined": len(unresolved) - len(named),
        "unresolved_max_error": (None if worst_unres is None
                                 else float(worst_unres["headline_error"])),
        "unresolved_max_error_t_star": (None if worst_unres is None
                                        else float(worst_unres["t_star"])),
        "legacy_max_error": (None if worst_legacy is None
                             else float(worst_legacy["error"])),
        "legacy_max_error_t_star": (None if worst_legacy is None
                                    else float(worst_legacy["t_star"])),
        "legacy_definition": ("wetted_edge (dry) / mid_level_crossing (wet)"),
        "headline_definition": ("fan_front_estimate (dry) / mid_level_crossing (wet)"),
        "tolerance": TOLERANCE,
        "pass": (best is not None and float(best["headline_error"]) <= TOLERANCE),
    }


def _dry_metrics(case: DamBreakCase, solver: ShallowWater1D, volume0: float) -> dict:
    t = solver.time
    h_ex, _ = case.exact(solver.x, t)
    x_f = case.exact_front(t)
    front_wetted = _wetted_edge(solver.x, solver.h)
    level = FRONT_LEVEL_FRACTION * case.h0
    x_level = _level_crossing(solver.x, solver.h, level)
    # exact fan: h = (x_f - x)^2 / (9 g t^2)  =>  x_f = x + 3 t sqrt(g h)
    x_level_scaled = x_level + 3.0 * t * math.sqrt(case.gravity * level)
    fan = fan_front_estimate(solver.x, solver.h, t, case.gravity, case.h0)
    tv_num = float(np.abs(np.diff(solver.h)).sum())
    ahead = solver.x > x_f
    return {
        "front_exact": x_f,
        # headline metric: shape-consistent fan estimator (see fan_front_estimate)
        "front_fan": fan["front"],
        "front_fan_error": _relative(fan["front"], x_f),
        "front_fan_cells": fan["cells"],
        "front_fan_spread": fan["spread"],
        "front_fan_slope": fan["slope"],
        "front_fan_slope_error": fan["slope_error"],
        "front_fan_window": fan["window"],
        # legacy/diagnostic definitions -- kept for continuity, NOT headline
        "front_wetted": front_wetted,
        "front_wetted_error": _relative(front_wetted, x_f),
        # signed shortfall of the wetted edge (negative => spurious wetting ahead)
        "wetted_shortfall": x_f - front_wetted,
        "precursor_length": float(max(0.0, front_wetted - x_f)),
        "precursor_depth": float(solver.h[ahead].max() / case.h0) if np.any(ahead) else 0.0,
        "front_level": x_level,
        "front_level_scaled": x_level_scaled,
        "front_level_scaled_error": _relative(x_level_scaled, x_f),
        "front_level_fraction": FRONT_LEVEL_FRACTION,
        "depth_l1": float(np.abs(solver.h - h_ex).sum() / max(h_ex.sum(), 1e-300)),
        "depth_linf": float(np.abs(solver.h - h_ex).max() / case.h0),
        "tv_numeric": tv_num,
        "tv_exact": float(case.h0),
        "tv_ratio": tv_num / case.h0,
        "local_extrema": _local_extrema(solver.h, EXTREMUM_FRACTION * case.h0),
        "mass_drift": _relative(solver.total_volume(), volume0),
        "h_max_over_h0": float(solver.h.max() / case.h0),
        "h_min": float(solver.h.min()),
    }


def _wet_metrics(case: DamBreakCase, solver: ShallowWater1D, volume0: float) -> dict:
    t = solver.time
    h_ex, _ = case.exact(solver.x, t)
    sol = stoker_constant_state(case.h0, case.h1, case.gravity)
    x_shock = case.x_dam + sol["shock_speed"] * t
    x_fan_end = case.x_dam + (sol["u2"] - sol["c2"]) * t
    level = 0.5 * (sol["h1"] + sol["h2"])
    x_meas = _level_crossing(solver.x, solver.h, level)
    # plateau: middle half of the exact constant state, shock smearing excluded
    lo = x_fan_end + 0.25 * (x_shock - x_fan_end)
    hi = x_fan_end + 0.75 * (x_shock - x_fan_end)
    plateau = (solver.x >= lo) & (solver.x <= hi)
    h2_meas = float(solver.h[plateau].mean()) if np.any(plateau) else float("nan")
    still = solver.x >= x_shock + 0.5 * (case.x_max - x_shock)
    h1_meas = float(solver.h[still].mean()) if np.any(still) else float("nan")
    tv_num = float(np.abs(np.diff(solver.h)).sum())
    return {
        "shock_exact": x_shock,
        "shock_position": x_meas,
        "shock_level": level,
        "shock_error": _relative(x_meas, x_shock),
        "fan_end_exact": x_fan_end,
        "h2_exact": sol["h2"],
        "h2_measured": h2_meas,
        "h2_error": _relative(h2_meas, sol["h2"]),
        "h1_measured": h1_meas,
        "step_ratio_exact": sol["step_ratio"],
        "step_ratio_measured": h2_meas / h1_meas,
        "step_ratio_error": _relative(h2_meas / h1_meas, sol["step_ratio"]),
        "shock_speed_exact": sol["shock_speed"],
        "u2_exact": sol["u2"],
        "c2_exact": sol["c2"],
        "depth_l1": float(np.abs(solver.h - h_ex).sum() / max(h_ex.sum(), 1e-300)),
        "depth_linf": float(np.abs(solver.h - h_ex).max() / case.h0),
        "tv_numeric": tv_num,
        "tv_exact": float(case.h0 - case.h1),
        "tv_ratio": tv_num / (case.h0 - case.h1),
        "local_extrema": _local_extrema(solver.h, EXTREMUM_FRACTION * case.h0),
        "mass_drift": _relative(solver.total_volume(), volume0),
        "h_max_over_h0": float(solver.h.max() / case.h0),
        "h_min": float(solver.h.min()),
    }


def _verdict(case: DamBreakCase, metrics: dict) -> dict:
    """Headline checks against the predeclared 5 % tolerance."""
    if case.kind == "dry":
        # ``front_wetted<5%`` is retained as a legacy diagnostic: the wetted edge
        # is a depth-floor detector (it does not converge under refinement, see
        # docs/verification-dam-break-front-metric/REPORT.md), so the headline is
        # the shape-consistent fan estimator.
        checks = {
            "front_fan<5%": metrics["front_fan_error"],
            "front_wetted<5%": metrics["front_wetted_error"],
            "depth_l1<5%": metrics["depth_l1"],
            "monotone(no_extrema)": 0.0 if metrics["local_extrema"] == 0 else 1.0,
            "tv_no_growth<0.5%": max(0.0, metrics["tv_ratio"] - 1.0),
        }
    else:
        checks = {
            "shock_position<5%": metrics["shock_error"],
            "step_ratio<5%": metrics["step_ratio_error"],
            "plateau_h2<5%": metrics["h2_error"],
            "depth_l1<5%": metrics["depth_l1"],
            "monotone(no_extrema)": 0.0 if metrics["local_extrema"] == 0 else 1.0,
            "tv_no_growth<0.5%": max(0.0, metrics["tv_ratio"] - 1.0),
        }
    limits = {key: TOLERANCE for key in checks}
    limits["tv_no_growth<0.5%"] = 5e-3
    verdict = {
        key: {
            "value": value,
            "limit": limits[key],
            "pass": bool(value <= limits[key]),
        }
        for key, value in checks.items()
    }
    verdict["all_pass"] = bool(all(item["pass"] for item in verdict.values()))
    return verdict


# --------------------------------------------------------------------------- #
# runs
# --------------------------------------------------------------------------- #
def run_case(case: DamBreakCase, n: int = 1600, cfl: float = 0.4, order: int = 2,
             t_star: float | None = None) -> dict:
    """Integrate one dam-break case and compare it with the exact solution.

    ``t_star`` overrides the case's own end time (used for the time series in
    the report).  The returned dict is JSON-serialisable except for the
    ``profile`` arrays, which are returned as plain lists.
    """
    t_star_end = case.t_star_end if t_star is None else float(t_star)
    t_end = t_star_to_time(t_star_end, case.h0, case.gravity)
    solver = ShallowWater1D(n, case.x_min, case.x_max, gravity=case.gravity,
                            cfl=cfl, order=order)
    initial = np.where(solver.x < case.x_dam, case.h0, case.h1)
    solver.set_state(initial)
    volume0 = solver.total_volume()
    samples: list[dict] = []
    gate = {"next": 0.0}

    def sample(sol: ShallowWater1D) -> None:
        if sol.time < gate["next"]:
            return
        samples.append(front_sample(case, sol))
        gate["next"] = sol.time + t_end / 25.0

    wall = time.perf_counter()
    steps = solver.advance(t_end, callback=sample)
    wall = time.perf_counter() - wall

    h_ex, u_ex = case.exact(solver.x, solver.time)
    metrics = (_dry_metrics if case.kind == "dry" else _wet_metrics)(
        case, solver, volume0)
    metrics["time_to_front_series"] = samples
    metrics["front_series"] = series_metrics(case, samples)
    metrics["front_series_max_error"] = metrics["front_series"]["max_error"]
    run = {
        "case": case.summary(),
        "discretisation": {
            "n": n,
            "dx": solver.dx,
            "cfl": cfl,
            "order": order,
            "steps": steps,
            "t_end": solver.time,
            "t_star_end": t_star_end,
            "wall_time_s": wall,
            "volume_initial": volume0,
            "volume_final": solver.total_volume(),
            "clipped_cells": solver.clipped_cells,
            "dry_momentum_flips": solver.dry_flips,
        },
        "metrics": metrics,
        "verdict": _verdict(case, metrics),
        "profile": {
            "x": solver.x.tolist(),
            "h_numeric": solver.h.tolist(),
            "u_numeric": solver.primitive()[1].tolist(),
            "h_exact": np.asarray(h_ex, dtype=float).tolist(),
            "u_exact": np.asarray(u_ex, dtype=float).tolist(),
        },
    }
    return run


def convergence_study(case: DamBreakCase, resolutions=(100, 200, 400, 800, 1600, 3200),
                      cfl: float = 0.4, order: int = 2) -> dict:
    """Grid-refinement study; returns the errors and the observed order."""
    rows = []
    for n in resolutions:
        run = run_case(case, n=n, cfl=cfl, order=order)
        m = run["metrics"]
        headline = ({"front_fan": m["front_fan_error"],
                     "front_wetted_diagnostic": m["front_wetted_error"]}
                    if case.kind == "dry"
                    else {"shock": m["shock_error"], "step": m["step_ratio_error"]})
        rows.append({
            "n": n,
            "dx": run["discretisation"]["dx"],
            "depth_l1": m["depth_l1"],
            "headline": headline,
            "tv_ratio": m["tv_ratio"],
            "local_extrema": m["local_extrema"],
        })
    dx = np.array([row["dx"] for row in rows])
    l1 = np.array([row["depth_l1"] for row in rows])
    fit = np.polyfit(np.log(dx), np.log(l1), 1) if len(rows) > 2 else [float("nan"), 0.0]
    headline_errors = np.array([
        next(iter(row["headline"].values())) for row in rows
    ])
    positive = headline_errors > 0.0
    fit_head = (np.polyfit(np.log(dx[positive]), np.log(headline_errors[positive]), 1)[0]
                if int(np.count_nonzero(positive)) > 2 else float("nan"))
    return {
        "case": case.key,
        "order": order,
        "cfl": cfl,
        "rows": rows,
        "observed_order_l1": float(fit[0]),
        "observed_order_headline": float(fit_head),
        "headline_error_at_finest": float(headline_errors[-1]),
        "headline_error_at_coarsest": float(headline_errors[0]),
    }


def _max_over_accepted(rows, key: str, accepted_n: int = 800) -> float:
    """Max of ``row[key]`` over the grids the benchmark accepts (``n >= 800``).

    A coarse-only study has no accepted grid; that returns NaN instead of raising
    ``max() arg is an empty sequence`` (found by the metric-repair tests).
    """
    values = [row[key] for row in rows if row["n"] >= accepted_n]
    return float(max(values)) if values else float("nan")


def front_definition_study(resolutions=(400, 800, 1600, 3200)) -> dict:
    """How much does "the front" depend on how it is measured? (dry bed)

    The Ritter tip is a zero-depth cusp, so a finite-volume scheme spreads it
    over a few cells and the reported "front error" depends on the definition.
    Three definitions are evaluated on every grid:

    ``wetted_edge``
        outermost cell with ``h > 0`` -- threshold-free in appearance only: it is
        a *depth-floor* detector (the reading is set by the numerical depth floor,
        not by ``dx``), so it does **not** converge.  Kept as a diagnostic.
    ``level_raw``
        abscissa where ``h`` crosses ``level * h0`` (a level inside the smeared
        tail is *not* the front, so this is reported to keep that visible).
    ``level_scaled``
        the same crossing plus the exact fan offset ``3 t sqrt(g * level * h0)``
        back to the tip.
    ``fan_shape``
        mean of the per-cell exact identity ``x_f = x_i + 3 t sqrt(g h_i)`` over
        ``FAN_WINDOW`` -- the headline metric (:func:`fan_front_estimate`).
    """
    levels = (FRONT_LEVEL_FRACTION, 1e-3)
    rows = []
    for n in resolutions:
        run = run_case(RITTER_DRY, n=n)
        m = run["metrics"]
        row = {
            "n": n,
            "dx": run["discretisation"]["dx"],
            "front_exact": m["front_exact"],
            "wetted_edge": m["front_wetted"],
            "wetted_edge_error": m["front_wetted_error"],
            "fan_shape": m["front_fan"],
            "fan_shape_error": m["front_fan_error"],
            "fan_shape_spread": m["front_fan_spread"],
            "level_raw_error": {},
            "level_scaled_error": {},
        }
        solver = ShallowWater1D(n, RITTER_DRY.x_min, RITTER_DRY.x_max,
                                gravity=RITTER_DRY.gravity, cfl=0.4, order=2)
        solver.set_state(np.where(solver.x < RITTER_DRY.x_dam, RITTER_DRY.h0,
                                  RITTER_DRY.h1))
        solver.advance(run["discretisation"]["t_end"])
        t = solver.time
        x_f = m["front_exact"]
        for level in levels:
            x_raw = _level_crossing(solver.x, solver.h, level * RITTER_DRY.h0)
            x_scaled = x_raw + 3.0 * t * math.sqrt(RITTER_DRY.gravity
                                                   * level * RITTER_DRY.h0)
            key = f"{level:g}"
            row["level_raw_error"][key] = _relative(x_raw, x_f)
            row["level_scaled_error"][key] = _relative(x_scaled, x_f)
        rows.append(row)
    return {
        "case": RITTER_DRY.key,
        "max_wetted_edge_error_at_accepted_grid": _max_over_accepted(rows, "wetted_edge_error"),
        "max_fan_shape_error_at_accepted_grid": _max_over_accepted(rows, "fan_shape_error"),
        "rows": rows,
    }


def sensitivity_study(n: int = 800) -> dict:
    """Parameter sensitivity of the headline metrics (task section 5).

    Three knobs are swept one at a time around the benchmark point
    (``n = 800``, ``CFL = 0.4``, ``order = 2``, ``h0/h1 = 2``):

    * time-step constraint: ``CFL in {0.2, 0.4, 0.8}``;
    * spatial order: MUSCL minmod (2) vs piecewise constant (1);
    * bore strength: ``h1 in {0.8, 0.5, 0.25}`` (``h0/h1 = 1.25, 2, 4``).

    Each row reports the headline metrics of both cases, so a knob that only
    flatters one case is visible.
    """
    rows = []

    def row(label: str, **kwargs) -> dict:
        h1 = kwargs.pop("h1", None)
        case_wet = STOKER_WET
        if h1 is not None:
            case_wet = DamBreakCase(
                key=f"stoker-wet-h1-{h1}", kind="wet", h0=STOKER_WET.h0, h1=h1,
                gravity=STOKER_WET.gravity, x_dam=STOKER_WET.x_dam,
                x_min=STOKER_WET.x_min, x_max=STOKER_WET.x_max,
                t_star_end=STOKER_WET.t_star_end, reference=STOKER_WET.reference)
        dry = run_case(RITTER_DRY, n=n, **kwargs)["metrics"]
        wet = run_case(case_wet, n=n, **kwargs)["metrics"]
        entry = {
            "variant": label,
            "n": n,
            "cfl": kwargs.get("cfl", 0.4),
            "order": kwargs.get("order", 2),
            "h1": case_wet.h1,
            "depth_ratio": case_wet.h0 / case_wet.h1,
            # headline (fan) first; the wetted edge stays as a diagnostic because
            # it is a depth-floor detector, not a solution error
            "dry_front_fan_error": dry["front_fan_error"],
            "dry_front_error": dry["front_wetted_error"],
            "dry_series_max_error": dry.get("front_series_max_error"),
            "dry_depth_l1": dry["depth_l1"],
            "wet_shock_error": wet["shock_error"],
            "wet_step_ratio_error": wet["step_ratio_error"],
            "wet_depth_l1": wet["depth_l1"],
            "wet_tv_ratio": wet["tv_ratio"],
            "wet_local_extrema": wet["local_extrema"],
        }
        rows.append(entry)
        return entry

    for cfl in (0.2, 0.4, 0.8):
        row(f"cfl={cfl}", cfl=cfl)
    row("order=1", order=1)
    for h1 in (0.8, 0.5, 0.25):
        row(f"h1={h1}", h1=h1)
    worst = max(
        max(entry["dry_front_fan_error"], entry["wet_shock_error"],
            entry["wet_step_ratio_error"])
        for entry in rows)
    return {"reference_point": {"n": n, "cfl": 0.4, "order": 2, "h0/h1": 2.0},
            "rows": rows, "worst_headline_error": worst,
            "tolerance": TOLERANCE}


def exactness_checks(gravity: float = 9.81) -> dict:
    """Certify the closed forms without transcribing any tabulated number.

    * Stoker: Riemann invariant and Rankine-Hugoniot residuals at the bracketed
      root (relative, machine precision expected).
    * Ritter: identity checks on the fan (depth/velocity at the dam, front
      speed) and on the mass in the fan, ``int h dx = h0 c0 t`` over
      ``[x_head, x_f]``, verified by quadrature of the evaluated profile.
    """
    h0, h1 = 1.0, 0.5
    g = gravity
    stoker = check_stoker_solution(h0, h1, g)
    sol = stoker_constant_state(h0, h1, g)
    t = t_star_to_time(2.0, h0, g)
    c0 = math.sqrt(g * h0)
    xd = 0.0
    # Ritter fan identities at t = t_end
    h_dam, u_dam = ritter_state(np.array([xd]), t, h0, g, xd)
    x_f = ritter_front_position(t, h0, g, xd)
    x_head = xd - c0 * t
    xs = np.linspace(x_head, x_f, 2_000_001)
    h_fan, _ = ritter_state(xs, t, h0, g, xd)
    mass_fan = float(np.trapezoid(h_fan, xs)) if hasattr(np, "trapezoid") else float(
        np.trapz(h_fan, xs))
    return {
        "stoker_h0_h1": [h0, h1],
        "stoker_gravity": g,
        "stoker": stoker,
        "stoker_state": {key: float(val) for key, val in sol.items()
                         if isinstance(val, (int, float))},
        "ritter": {
            "h_at_dam_over_h0": float(h_dam[0] / h0),
            "h_at_dam_expected": 4.0 / 9.0,
            "h_at_dam_residual": float(h_dam[0] / h0 - 4.0 / 9.0),
            "u_at_dam_over_c0": float(u_dam[0] / c0),
            "u_at_dam_expected": 2.0 / 3.0,
            "u_at_dam_residual": float(u_dam[0] / c0 - 2.0 / 3.0),
            "front_over_c0t": float(x_f / (c0 * t)),
            "front_expected": 2.0,
            "fan_mass": mass_fan,
            "fan_mass_expected": h0 * c0 * t,
            "fan_mass_relative": abs(mass_fan - h0 * c0 * t) / (h0 * c0 * t),
        },
    }


# --------------------------------------------------------------------------- #
# artefacts
# --------------------------------------------------------------------------- #
def figure_profiles_and_convergence(runs: dict, studies: dict, path,
                                    t_star: float = 2.0) -> str:
    """Two-panel figure: profiles at ``t*`` and grid convergence of the errors."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    colors = {"ritter-dry": "tab:blue", "stoker-wet": "tab:red"}
    for key, run in runs.items():
        ax = axes[0]
        x = np.asarray(run["profile"]["x"])
        h = np.asarray(run["profile"]["h_numeric"])
        he = np.asarray(run["profile"]["h_exact"])
        ax.plot(x, he, "-", lw=1.2, color=colors[key], alpha=0.55,
                label=f"{key}: exact ({run['case']['reference'].split('(')[0].strip()})")
        ax.plot(x, h, "--", lw=1.2, color=colors[key], label=f"{key}: FVM n={run['discretisation']['n']}")
    ax = axes[0]
    ax.set_xlim(-3.0, 4.6)
    ax.set_ylim(0.0, 1.1)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("depth h [m]")
    ax.set_title(f"Dam break at t* = {t_star:g} (h0 = 1 m, g = 9.81 m/s²)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[1]
    for key, study in studies.items():
        rows = study["rows"]
        dx = [row["dx"] for row in rows]
        ax.loglog(dx, [row["depth_l1"] for row in rows], "o-",
                  color=colors[key], label=f"{key}: depth L1 (order {study['observed_order_l1']:.2f})")
        headline = [next(iter(row["headline"].values())) for row in rows]
        ax.loglog(dx, headline, "s--", color=colors[key], alpha=0.7,
                  label=f"{key}: wave-position error")
    ax.axhline(TOLERANCE, color="k", ls=":", lw=1.0)
    ax.text(ax.get_xlim()[0], TOLERANCE * 1.1, " 5 % criterion", fontsize=8)
    ax.set_xlabel("dx [m]")
    ax.set_ylabel("relative error [-]")
    ax.set_title("Grid convergence (SSP-RK3 + MUSCL minmod + HLL)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def write_artifacts(outdir, payload: dict) -> dict:
    """Write the JSON/CSV artefacts of a benchmark campaign into ``outdir``."""
    from pathlib import Path

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    written = {}
    meta = {key: val for key, val in payload.items()
            if key not in ("runs", "studies", "profiles")}
    metrics = {}
    for key, run in payload["runs"].items():
        row = {
            "case": run["case"],
            "discretisation": run["discretisation"],
            "metrics": {k: v for k, v in run["metrics"].items()
                        if k != "time_to_front_series"},
            "verdict": run["verdict"],
            # raw x_f(t) samples plus the *graded* summary of the series; the
            # two are separate keys because "front_series" used to be the raw
            # list while the pass/fail statement lived nowhere
            "front_series_samples": run["metrics"]["time_to_front_series"],
            "front_series_summary": run["metrics"]["front_series"],
        }
        metrics[key] = row
        # profiles as csv (x, h, exact h, u, exact u)
        p = run["profile"]
        csv_path = out / f"profile_{key}_n{run['discretisation']['n']}.csv"
        with csv_path.open("w") as fh:
            fh.write("x,h_numeric,h_exact,u_numeric,u_exact\n")
            for row_i in zip(p["x"], p["h_numeric"], p["h_exact"],
                             p["u_numeric"], p["u_exact"]):
                fh.write(",".join(f"{value!r}" for value in row_i) + "\n")
        written[csv_path.name] = str(csv_path)
    path = out / "metrics.json"
    path.write_text(json.dumps({"meta": meta, "cases": metrics}, indent=2) + "\n")
    written["metrics.json"] = str(path)
    path = out / "convergence.json"
    path.write_text(json.dumps(payload["studies"], indent=2) + "\n")
    written["convergence.json"] = str(path)
    path = out / "exactness.json"
    path.write_text(json.dumps(payload["exactness"], indent=2) + "\n")
    written["exactness.json"] = str(path)
    return written