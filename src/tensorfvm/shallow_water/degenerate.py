"""Degenerate limits of the wet-bed dam-break benchmark (``h1/h0 -> 0`` and ``-> 1``).

Why this module exists
----------------------
``dam_break.STOKER_WET`` grades the anchor ``CASE-FVM-DAMBREAK-STOKER`` on two
quantities -- the bore position ``x_bore(t)`` and the still-water level step
``h2 - h1`` across the bore -- at **one** depth ratio, ``h1/h0 = 0.5``.  The
robustness package (``shallow_water.robustness``) perturbs that ratio to
``0.2 / 0.8``; both ends of the interval were never graded, and they are exactly
where the closed form and its measurement operators degenerate:

``h1/h0 -> 0`` (near-dry)
    The bore becomes a front running over a vanishing layer.  The exact
    solution has a *singular* limit: the constant state collapses as

        ``h2/h0 = sqrt(8 h1/h0) * (1 + O((h1/h0) ** 0.25))``,
        ``u2/(2 c0) = 1 - 8 ** 0.25 (h1/h0) ** 0.25 + O(eps)``,
        ``U/(2 c0) -> 1``   (the Ritter front speed ``2 c0``),

    while the tested metric ``h2/h1`` diverges like ``eps ** -0.5``.  So the
    *level-step metric is not a fixed yardstick*: at ``h1/h0 = 1e-6`` the exact
    ratio is ``2680`` and the oracle plateau window ``(x_fan_end, x_shock)`` is
    only ``O(8 dx)`` wide at ``n = 800`` -- narrower than the numerically
    smeared bore.  The window then reads the *undisturbed* depth and reports a
    99.96 % amplitude error without ever measuring the bore.

``h1/h0 -> 1`` (weak bore)
    The acoustic limit: the step height tends to exactly half the depth
    difference, ``h2 - h1 -> (h0 - h1)/2``, ``u2 -> c0 (h0-h1)/(2 h0)`` and the
    bore speed ``U -> c0`` (not ``2 c0``).  The level crossing used as the
    position metric is placed at ``(h1 + h2)/2``, i.e. within ``(h0-h1)/4`` of
    the flat upstream level, so the position reading is expected to lose
    conditioning here -- quantitatively, not by assertion.

The bore speed is therefore **non-monotone** in the depth ratio: it falls from
``c0`` at ``h1/h0 -> 1`` to an interior minimum and rises to ``2 c0`` at
``h1/h0 -> 0``.  A single-ratio benchmark cannot see that.

What is delivered
-----------------
1. **Derived asymptotic laws, certified against the closed form** (not fitted
   tables): the two limits above, with the exponent of every vanishing term
   *fitted* from the exact solver over 12 decades and compared with the
   predicted ``1/2``, ``1/4``, ``1``.
2. **Certification at the limits**: the jump/invariant residuals of
   :func:`~tensorfvm.shallow_water.exact.check_stoker_solution` are re-checked
   at ``h1/h0 = 1e-8 ... 1 - 1e-6``, so the reference itself is proven to stay
   exact where the benchmark was never run.
3. **The validity domain of the anchor's two acceptance quantities**: a
   ``h1/h0 x n`` map of ``PASS / FAIL / unresolved`` for the bore position and
   the level step, with the minimum grid ``n_min(h1/h0)`` each metric needs for
   the declared 5 % tolerance.  A reading whose window sees less than
   :data:`~tensorfvm.shallow_water.dam_break.PLATEAU_MIN_JUMP_FRACTION` of the
   exact step is labelled *unresolved* (insufficient evidence), not FAIL -- the
   solver error is then a separate, honestly reported number.

Everything here reuses the merged solver and the merged metric code
(``dam_break.run_case``); not one line of the discretisation changed.

References
----------
Ritter (1892) *Z. Ver. Deutsch. Ing.* **36**, 947 (dry-bed front, the
``h1/h0 -> 0`` limit); Stoker (1957) *Water Waves* ch. 10 (wet-bed
rarefaction + bore, the family being degenerated here); Whitham (1974) *Linear
and Nonlinear Waves* §3.1 (acoustic limit of a weak bore).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .dam_break import (
    PLATEAU_MIN_JUMP_FRACTION,
    TOLERANCE,
    DamBreakCase,
    run_case,
)
from .exact import (
    check_stoker_solution,
    stoker_constant_state,
    t_star_to_time,
)

__all__ = [
    "EPS_LADDER",
    "GRID_LADDER",
    "DEEP_GRID_CASES",
    "DRY_FLOOR_EPS",
    "near_dry_prediction",
    "weak_bore_prediction",
    "law_table",
    "certification_table",
    "fit_exponents",
    "fit_exponents_local",
    "bore_speed_minimum",
    "sweep",
    "validity_map",
    "analytical_laws",
    "write_artifacts",
    "figure_laws",
    "figure_validity",
    "figure_profiles",
    "main",
]

#: depth ratios ``h1/h0`` graded by the audit: log-spaced from the weak-bore end
#: (``0.9999``) down to ``1e-6`` across the value the anchor uses (``0.5``).
EPS_LADDER = (0.9999, 0.999, 0.99, 0.95, 0.9, 0.8, 0.5, 0.25, 0.1, 0.03, 0.01,
              3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6)

#: grids of the resolution study.  ``800`` is the anchor's declared evidence
#: floor (``dam_break.ACCEPTED_MIN_N``), ``1600`` is the grid of the published
#: report; ``3200`` is the finest ladder rung affordable for the whole sweep.
GRID_LADDER = (800, 1600, 3200)

#: extra rungs, run only for the depth ratios where the accepted grids are not
#: yet enough to resolve the bore (finest run of the study, ~30 s each).
DEEP_GRID_CASES = {1e-4: 6400, 1e-5: 6400, 1e-6: 6400}

#: one probe *below* the solver's dry floor ``sweep1d.ShallowWater1D.dry_depth
#: = 1e-12``: there the exact reference is still a wet-bed solution while the
#: scheme sees a dry bed, so the two definitions disagree by construction.  The
#: probe documents the hard lower bound of the comparison, it is not graded.
DRY_FLOOR_EPS = 1e-13

#: domain of the wet-bed benchmark (identical to ``dam_break.STOKER_WET``)
X_MIN, X_MAX, X_DAM, T_STAR_END, GRAVITY, H0 = -4.0, 6.0, 0.0, 2.0, 9.81, 1.0


# --------------------------------------------------------------------------- #
# cases
# --------------------------------------------------------------------------- #
def wet_case(eps: float, h0: float = H0, gravity: float = GRAVITY) -> DamBreakCase:
    """``DamBreakCase`` with downstream depth ``h1 = eps * h0`` (same box)."""
    return DamBreakCase(
        key=f"stoker-wet-eps{eps:g}", kind="wet", h0=h0, h1=eps * h0,
        gravity=gravity, x_dam=X_DAM, x_min=X_MIN, x_max=X_MAX,
        t_star_end=T_STAR_END,
        reference="Stoker (1957) Water Waves ch. 10 (wet bed, degenerate end)",
    )


# --------------------------------------------------------------------------- #
# derived asymptotics
# --------------------------------------------------------------------------- #
def near_dry_prediction(eps: float, h0: float = H0,
                        gravity: float = GRAVITY) -> dict:
    """Leading-order ``h1/h0 -> 0`` asymptotics of the Stoker constant state.

    Derivation (no fitting): with ``u2 = 2 (c0 - c2)`` and the bore momentum
    jump ``u2^2 = g (h2^2 - h1^2)(h2 - h1) / (2 h1 h2)``, put ``x = sqrt(h2)``,
    ``s = sqrt(h1)``.  For ``h1 -> 0`` the left side tends to ``4 c0^2 = 4 g h0``,
    so the right side must too; with ``x >> s`` it reads ``x^4 / (2 s^2)``, hence
    ``h2 = sqrt(8 h0 h1)`` and ``c2 = sqrt(g h2) ~ h1 ** 0.25``, i.e. the
    deviation of the plateau velocity from the Ritter tip velocity
    ``2 c0`` vanishes like ``eps ** 0.25``.
    """
    c0 = math.sqrt(gravity * h0)
    h2 = math.sqrt(8.0 * h0 * eps * h0)
    u2 = 2.0 * c0 * (1.0 - 8.0 ** 0.25 * eps ** 0.25)
    return {"h2": h2, "h2_over_sqrt8_eps_h0": h2 / math.sqrt(8.0 * h0 * eps * h0),
            "u2": u2, "u2_over_2c0": u2 / (2.0 * c0),
            "shock_speed": 2.0 * c0, "shock_over_2c0": 1.0, "c0": c0}


def weak_bore_prediction(eps: float, h0: float = H0,
                         gravity: float = GRAVITY) -> dict:
    """Leading-order ``h1/h0 -> 1`` (acoustic) asymptotics.

    With ``delta = 1 - eps``: ``h2 = h0 - (h0 - h1)/2 + O(delta^2)``,
    ``u2 = c0 delta/2 + O(delta^2)`` and ``U = c0 (1 + O(delta))``.  The two
    limits of the bore speed differ by a factor two (``2 c0`` on the dry side,
    ``c0`` on the acoustic side), which is why ``U(eps)`` is not monotone.
    """
    c0 = math.sqrt(gravity * h0)
    delta = 1.0 - eps
    return {"delta": delta, "h2": h0 - 0.5 * (h0 - eps * h0),
            "step_over_depth_difference": 0.5,
            "u2": 0.5 * c0 * delta, "u2_over_c0_delta": 0.5,
            "shock_speed": c0, "shock_over_c0": 1.0}


def _loglog_slope(xs, ys) -> dict:
    lx = np.log(np.asarray(xs, dtype=float))
    ly = np.log(np.asarray(ys, dtype=float))
    slope, intercept = np.polyfit(lx, ly, 1)
    resid = ly - (slope * lx + intercept)
    return {"slope": float(slope), "intercept": float(intercept),
            "max_abs_log_residual": float(np.max(np.abs(resid)))}


def law_table(n_points: int = 12, h0: float = H0,
              gravity: float = GRAVITY) -> dict:
    """Exact states at the two degenerate ends against the derived laws."""
    dry_eps = np.logspace(-8.0, -3.0, n_points)
    rows_dry = []
    for eps in dry_eps:
        sol = stoker_constant_state(h0, eps * h0, gravity)
        pred = near_dry_prediction(float(eps), h0, gravity)
        c0 = pred["c0"]
        rows_dry.append({
            "eps": float(eps),
            "h2_exact": sol["h2"],
            "h2_law": pred["h2"],
            "h2_ratio_to_law": sol["h2"] / pred["h2"],
            "u2_deficit_exact": 1.0 - sol["u2"] / (2.0 * c0),
            "u2_deficit_law": 8.0 ** 0.25 * float(eps) ** 0.25,
            "u2_deficit_ratio": ((1.0 - sol["u2"] / (2.0 * c0))
                                 / (8.0 ** 0.25 * float(eps) ** 0.25)),
            "shock_over_2c0": sol["shock_speed"] / (2.0 * c0),
            "step_ratio_exact": sol["step_ratio"],
        })
    bore_eps = 1.0 - np.logspace(-6.0, -1.0, n_points)
    rows_bore = []
    for eps in bore_eps:
        sol = stoker_constant_state(h0, float(eps) * h0, gravity)
        delta = 1.0 - float(eps)
        step = sol["h2"] - sol["h1"]
        rows_bore.append({
            "eps": float(eps),
            "delta": delta,
            "step_over_depth_difference": step / (h0 - sol["h1"]),
            "step_deviation": abs(step / (h0 - sol["h1"]) - 0.5),
            "u2_over_c0_delta": sol["u2"] / (math.sqrt(gravity * h0) * delta),
            "shock_over_c0": sol["shock_speed"] / math.sqrt(gravity * h0),
            "step_ratio_exact": sol["step_ratio"],
        })
    return {"near_dry": rows_dry, "weak_bore": rows_bore}


def fit_exponents(h0: float = H0, gravity: float = GRAVITY) -> dict:
    """Fit the exponent of every vanishing term and compare with the prediction."""
    eps = np.logspace(-8.0, -3.0, 13)
    h2 = []
    u2_deficit = []
    shock_deficit = []
    c0 = math.sqrt(gravity * h0)
    for e in eps:
        sol = stoker_constant_state(h0, float(e) * h0, gravity)
        h2.append(sol["h2"] / h0)
        u2_deficit.append(1.0 - sol["u2"] / (2.0 * c0))
        shock_deficit.append(1.0 - sol["shock_speed"] / (2.0 * c0))
    delta = 1.0 - np.logspace(-6.0, -1.0, 13)
    step_dev = []
    u2_bore_dev = []
    for e in delta:
        sol = stoker_constant_state(h0, float(e) * h0, gravity)
        d = 1.0 - float(e)
        step_dev.append(abs((sol["h2"] - sol["h1"]) / (h0 - sol["h1"]) - 0.5))
        u2_bore_dev.append(abs(sol["u2"] / (c0 * d) - 0.5))
    fits = {
        "h2_over_h0_vs_eps": _loglog_slope(eps, h2),
        "u2_deficit_vs_eps": _loglog_slope(eps, u2_deficit),
        "shock_deficit_vs_eps": _loglog_slope(eps, shock_deficit),
        "weak_step_deviation_vs_delta": _loglog_slope(1.0 - delta, step_dev),
        "weak_u2_deviation_vs_delta": _loglog_slope(1.0 - delta, u2_bore_dev),
    }
    predicted = {
        "h2_over_h0_vs_eps": 0.5,
        "u2_deficit_vs_eps": 0.25,
        "shock_deficit_vs_eps": 0.25,
        "weak_step_deviation_vs_delta": 1.0,
        "weak_u2_deviation_vs_delta": 1.0,
    }
    for key, value in predicted.items():
        fits[key]["predicted"] = value
        fits[key]["deviation"] = fits[key]["slope"] - value
    return fits


def fit_exponents_local(window_decades: float = 1.5,
                        h0: float = H0, gravity: float = GRAVITY) -> dict:
    """Same fit as :func:`fit_exponents` but on the deepest ``window_decades``.

    Used to show that the global exponent is biased by the ``eps ** 0.25``
    correction and that the *local* exponent converges to the predicted 1/2.
    """
    eps = np.logspace(-11.0, -11.0 + window_decades, 7)
    h2 = [stoker_constant_state(h0, float(e) * h0, gravity)["h2"] / h0
          for e in eps]
    return {"window": [float(eps[0]), float(eps[-1])],
            "h2_over_h0_vs_eps": _loglog_slope(eps, h2)}


def bore_speed_minimum(h0: float = H0, gravity: float = GRAVITY,
                       n: int = 4000) -> dict:
    """Locate the interior minimum of ``U/(2 c0)`` in ``h1/h0``.

    ``U -> 2 c0`` on the near-dry side and ``U -> c0`` on the acoustic side, so
    the curve must turn around; a single-ratio benchmark would report the local
    value as if it were the whole family.
    """
    eps = np.logspace(-9.0, -1e-6, n)
    speed = np.array([stoker_constant_state(h0, float(e) * h0, gravity)["shock_speed"]
                      for e in eps])
    c0 = math.sqrt(gravity * h0)
    i = int(np.argmin(speed))
    return {"eps_at_minimum": float(eps[i]),
            "u_over_2c0_at_minimum": float(speed[i] / (2.0 * c0)),
            # eps[0] is the near-dry (strong) end of the ladder, eps[-1] the
            # acoustic (weak) end: 1.0 -> 2 c0, 0.5 -> c0.
            "u_over_2c0_at_strong_end": float(speed[0] / (2.0 * c0)),
            "u_over_2c0_at_weak_end": float(speed[-1] / (2.0 * c0))}


def analytical_laws(h0: float = H0, gravity: float = GRAVITY) -> dict:
    """Everything that needs no numerical run: laws, fits, limits, minimum."""
    return {
        "t_star_end": T_STAR_END,
        "h0": h0,
        "gravity": gravity,
        "law_table": law_table(h0=h0, gravity=gravity),
        "fits": fit_exponents(h0=h0, gravity=gravity),
        "bore_speed_minimum": bore_speed_minimum(h0=h0, gravity=gravity),
        "predictions": {
            "near_dry_h2": "sqrt(8 h0 h1)",
            "near_dry_u2_deficit": "8**0.25 * eps**0.25",
            "near_dry_shock": "2 c0",
            "weak_step": "(h0 - h1) / 2",
            "weak_u2": "c0 * (h0 - h1) / (2 h0)",
            "weak_shock": "c0",
        },
    }


def certification_table(h0: float = H0, gravity: float = GRAVITY) -> dict:
    """Jump/invariant residuals of the closed form at both degenerate ends.

    The merged benchmark certifies the closed form at ``h1/h0 = 0.5`` only; if
    the root bracket or the bisection lost precision near the ends, every
    "error" measured there would be a measurement of the *reference's* noise.
    """
    rows = []
    eps_list = (1e-8, 1e-6, 1e-4, 1e-2, 0.5, 0.9, 0.99, 0.9999, 1.0 - 1e-6)
    for eps in eps_list:
        check = check_stoker_solution(h0, eps * h0, gravity)
        sol = stoker_constant_state(h0, eps * h0, gravity)
        rows.append({
            "eps": float(eps),
            "riemann_invariant_relative": check["riemann_invariant_relative"],
            "momentum_jump_absolute": check["momentum_jump_residual"],
            "mass_jump_relative": check["mass_jump_relative"],
            "momentum_jump_relative": check["momentum_jump_relative"],
            "shock_speed": sol["shock_speed"],
            "h2": sol["h2"],
            "u2": sol["u2"],
            "step_ratio": sol["step_ratio"],
            "plateau_cells_at_n800": (
                (sol["shock_speed"] - (sol["u2"] - sol["c2"])) * t_star_to_time(
                    T_STAR_END, h0, gravity) / ((X_MAX - X_MIN) / 800.0)),
        })
    worst = max(max(r["riemann_invariant_relative"], r["mass_jump_relative"],
                    r["momentum_jump_relative"]) for r in rows)
    worst_abs = max(abs(r["momentum_jump_absolute"]) for r in rows)
    # Declared gate: 1e-8 *relative*, not machine epsilon.  On the acoustic side
    # the momentum-jump scale ``|U h2 u2|`` vanishes like ``delta ** 3`` while
    # float64 round-off in ``u2`` stays at ``eps_mach * c0``, so the *relative*
    # residual must grow as ``eps_mach / delta ** 2``: measured 1.3e-11 at
    # ``delta = 1e-4`` and 4.0e-9 at ``delta = 1e-6``, exactly the predicted
    # amplification factor 1e4.  The absolute residual stays at round-off
    # (``momentum_jump_absolute``), which is what certifies the closed form.
    return {"rows": rows, "worst_relative_residual": float(worst),
            "worst_absolute_momentum_residual": float(worst_abs),
            "tolerance": 1e-8, "pass": bool(worst <= 1e-8)}


# --------------------------------------------------------------------------- #
# numerical sweep (reuses dam_break.run_case unchanged)
# --------------------------------------------------------------------------- #
def _run_one(args) -> dict:
    eps, n = args
    case = wet_case(eps)
    run = run_case(case, n=n)
    m = run["metrics"]
    sol = stoker_constant_state(case.h0, case.h1, case.gravity)
    return {
        "eps": float(eps),
        "n": int(n),
        "dx": run["discretisation"]["dx"],
        "wall_time_s": run["discretisation"]["wall_time_s"],
        "clipped_cells": run["discretisation"]["clipped_cells"],
        "dry_momentum_flips": run["discretisation"]["dry_momentum_flips"],
        "shock_error": m["shock_error"],
        "step_ratio_error": m["step_ratio_error"],
        "step_ratio_exact": sol["step_ratio"],
        "h2_error": m["h2_error"],
        "h2_exact": sol["h2"],
        "plateau_jump_fraction": m["plateau_jump_fraction"],
        "step_insufficient_evidence": m["step_insufficient_evidence"],
        "plateau_window_cells": m["plateau_window_cells"],
        "exact_plateau_cells": ((sol["shock_speed"] - (sol["u2"] - sol["c2"]))
                                * t_star_to_time(T_STAR_END, case.h0,
                                                 case.gravity)
                                / run["discretisation"]["dx"]),
        "depth_l1": m["depth_l1"],
        "local_extrema": m["local_extrema"],
        "h_min": m["h_min"],
        "mass_drift": m["mass_drift"],
        "bore_speed_over_2c0": sol["shock_speed"] / (2.0 * math.sqrt(
            case.gravity * case.h0)),
        "position_pass": bool(m["shock_error"] <= TOLERANCE),
        "step_pass": bool((not m["step_insufficient_evidence"])
                          and m["step_ratio_error"] <= TOLERANCE),
    }


def sweep(eps_ladder=EPS_LADDER, grid_ladder=GRID_LADDER,
          deep_grid=DEEP_GRID_CASES, workers: int = 6,
          dry_floor_probe: bool = True) -> dict:
    """Grade every ``(h1/h0, n)`` pair; returns the raw rows plus the map."""
    jobs = [(eps, n) for eps in eps_ladder for n in grid_ladder]
    jobs += [(eps, n) for eps, n in deep_grid.items()]
    t0 = time.perf_counter()
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_run_one, jobs))
    else:
        rows = [_run_one(job) for job in jobs]
    wall = time.perf_counter() - t0
    probe = None
    if dry_floor_probe:
        try:
            probe = _run_one((DRY_FLOOR_EPS, 1600))
        except (ValueError, ZeroDivisionError) as exc:  # documented failure mode
            probe = {"eps": DRY_FLOOR_EPS, "error": repr(exc)}
    return {"rows": rows, "jobs": len(jobs), "wall_time_s": wall,
            "dry_floor_probe": probe}


def validity_map(rows) -> dict:
    """Minimum grid per metric per depth ratio, and the region each fails in."""
    eps_values = sorted({r["eps"] for r in rows})
    grids = sorted({r["n"] for r in rows})
    position = {}
    step = {}
    for eps in eps_values:
        mine = [r for r in rows if r["eps"] == eps]
        ok_pos = [r["n"] for r in mine if r["position_pass"]]
        ok_step = [r["n"] for r in mine
                   if r["step_pass"] and not r["step_insufficient_evidence"]]
        unresolved = [r["n"] for r in mine if r["step_insufficient_evidence"]]
        position[f"{eps:g}"] = {
            "n_min": min(ok_pos) if ok_pos else None,
            "worst_error": max(r["shock_error"] for r in mine),
        }
        step[f"{eps:g}"] = {
            "n_min": min(ok_step) if ok_step else None,
            "worst_error": max(r["step_ratio_error"] for r in mine),
            "unresolved_grids": unresolved,
            "exact_step_ratio": mine[0]["step_ratio_exact"],
            "exact_plateau_cells_at_800": mine[0]["exact_plateau_cells"],
        }
    # where does each metric stop being gradeable at the anchor's accepted grid?
    accepted = 1600
    pos_fail = [eps for eps in eps_values
                if not any(r["position_pass"] for r in rows
                           if r["eps"] == eps and r["n"] == accepted)]
    step_unres = [eps for eps in eps_values
                  if any(r["step_insufficient_evidence"] for r in rows
                         if r["eps"] == eps and r["n"] == accepted)]
    return {
        "grids": grids,
        "eps_values": eps_values,
        "position": position,
        "step": step,
        "accepted_grid": accepted,
        "position_fails_at_accepted_grid": pos_fail,
        "step_unresolved_at_accepted_grid": step_unres,
        "tolerance": TOLERANCE,
        "plateau_min_jump_fraction": PLATEAU_MIN_JUMP_FRACTION,
    }


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def figure_laws(laws: dict, path) -> str:
    """Two panels: the near-dry and weak-bore laws with their predictions."""
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))
    dry = laws["law_table"]["near_dry"]
    eps = [r["eps"] for r in dry]
    ax = axes[0]
    ax.loglog(eps, [r["h2_exact"] / laws["h0"] for r in dry], "o-",
              label="exact $h_2/h_0$")
    ax.loglog(eps, [math.sqrt(8.0 * e) for e in eps], "k--",
              label=r"law $\sqrt{8\,h_1/h_0}$")
    ax.loglog(eps, [r["u2_deficit_exact"] for r in dry], "s-",
              label=r"exact $1-u_2/(2c_0)$")
    ax.loglog(eps, [r["u2_deficit_law"] for r in dry], "k:",
              label=r"law $8^{1/4}(h_1/h_0)^{1/4}$")
    ax.set_xlabel(r"$h_1/h_0$")
    ax.set_ylabel("dimensionless")
    ax.set_title("near-dry limit: singular $h_2 \\sim \\sqrt{h_1}$")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    ax = axes[1]
    bore = laws["law_table"]["weak_bore"]
    delta = [r["delta"] for r in bore]
    ax.semilogx(delta, [r["step_over_depth_difference"] for r in bore], "o-",
                label=r"exact $(h_2-h_1)/(h_0-h_1)$")
    ax.axhline(0.5, color="k", ls="--", label=r"law $1/2$")
    ax.semilogx(delta, [r["shock_over_c0"] for r in bore], "s-",
                label=r"exact $U/c_0$")
    ax.axhline(1.0, color="k", ls=":", label=r"law $U \to c_0$")
    ax.set_xlabel(r"$\delta = 1 - h_1/h_0$")
    ax.set_title("weak-bore (acoustic) limit")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    ax = axes[2]
    grid = np.logspace(-9.0, -1e-6, 400)
    speed = [stoker_constant_state(laws["h0"], float(e) * laws["h0"],
                                   laws["gravity"])["shock_speed"]
             / (2.0 * math.sqrt(laws["gravity"] * laws["h0"])) for e in grid]
    ax.semilogx(grid, speed, "-", color="tab:red")
    mn = laws["bore_speed_minimum"]
    ax.plot([mn["eps_at_minimum"]], [mn["u_over_2c0_at_minimum"]], "ko",
            label=(f"minimum at $h_1/h_0$={mn['eps_at_minimum']:.3g}, "
                   f"$U/(2c_0)$={mn['u_over_2c0_at_minimum']:.3f}"))
    ax.axhline(1.0, color="k", ls=":", label=r"$U \to 2c_0$ (dry side)")
    ax.axhline(0.5, color="k", ls="--", label=r"$U \to c_0$ (acoustic side)")
    ax.set_xlabel(r"$h_1/h_0$")
    ax.set_ylabel(r"$U/(2c_0)$")
    ax.set_title("bore speed is non-monotone in the depth ratio")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def figure_validity(sweep_result: dict, vmap: dict, path) -> str:
    """Error vs ``h1/h0`` for both metrics and every grid, with the 5 % line."""
    plt = _plt()
    rows = sweep_result["rows"]
    grids = vmap["grids"]
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6), sharex=True)
    markers = {800: "o", 1600: "s", 3200: "^", 6400: "D"}
    for ax, key, label in ((axes[0], "shock_error", "bore position error"),
                           (axes[1], "step_ratio_error", "level-step error")):
        for n in grids + [6400]:
            mine = sorted((r for r in rows if r["n"] == n), key=lambda r: r["eps"])
            if not mine:
                continue
            xs = [r["eps"] for r in mine]
            ys = [max(r[key], 1e-12) for r in mine]
            ax.loglog(xs, ys, markers[n] + "-", ms=4, label=f"n = {n}")
        ax.axhline(TOLERANCE, color="k", ls="--", lw=1.0)
        ax.text(min(r["eps"] for r in rows), TOLERANCE * 1.15,
                " 5 % criterion", fontsize=8)
        ax.set_xlabel(r"$h_1/h_0$")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    axes[1].set_title("level-step reading collapses where the bore is unresolved")
    axes[0].set_title("bore position stays first order at every ratio")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def figure_profiles(path, n_pairs=((800, 3200),), eps: float = 1e-5) -> str:
    """Profiles at a near-dry ratio: the smeared bore and the oracle window."""
    plt = _plt()
    case = wet_case(eps)
    sol = stoker_constant_state(case.h0, case.h1, case.gravity)
    t = t_star_to_time(T_STAR_END, case.h0, case.gravity)
    x_shock = case.x_dam + sol["shock_speed"] * t
    x_fan = case.x_dam + (sol["u2"] - sol["c2"]) * t
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6))
    ax = axes[0]
    for n in n_pairs[0]:
        run = run_case(case, n=n)
        x = np.asarray(run["profile"]["x"])
        h = np.asarray(run["profile"]["h_numeric"])
        ax.semilogy(x, np.maximum(h, 1e-12), "-", lw=1.0,
                    label=f"FVM n = {n} (h$_2$ err "
                          f"{run['metrics']['h2_error']:.2%})")
    xs = np.linspace(x_fan - 0.6, x_shock + 0.3, 2000)
    he, _ = case.exact(xs, t)
    ax.semilogy(xs, np.maximum(he, 1e-12), "k--", lw=1.2,
                label="Stoker exact")
    ax.axvspan(x_fan + 0.25 * (x_shock - x_fan), x_fan + 0.75 * (x_shock - x_fan),
               color="tab:orange", alpha=0.2, label="oracle plateau window")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("h [m]")
    ax.set_title(f"$h_1/h_0$ = {eps:g}: bore is a thin-layer front")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="lower left")

    ax = axes[1]
    xs = np.linspace(3.60, 3.86, 4000)
    he, _ = case.exact(xs, t)
    ax.plot(xs, he / sol["h2"], "k--", lw=1.4, label="exact / $h_2$")
    for n in n_pairs[0]:
        run = run_case(case, n=n)
        x = np.asarray(run["profile"]["x"])
        h = np.asarray(run["profile"]["h_numeric"])
        ax.plot(x, h / sol["h2"], ".-", ms=3, lw=0.9, label=f"FVM n = {n}")
    ax.axhline(1.0, color="grey", ls=":", lw=0.8)
    ax.axhline(sol["h1"] / sol["h2"], color="grey", ls=":", lw=0.8)
    ax.set_ylim(-0.05, 1.2)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("h / $h_2$")
    ax.set_title("zoom on the bore: mass lost at n = 800")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


# --------------------------------------------------------------------------- #
# artefacts
# --------------------------------------------------------------------------- #
def write_artifacts(outdir, payload: dict) -> dict:
    """Write the JSON artefacts (no number in the report is typed by hand)."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, datum in payload.items():
        path = out / f"{name}.json"
        path.write_text(json.dumps(datum, indent=2, sort_keys=False) + "\n")
        written[path.name] = str(path)
    return written


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="src/tensorfvm/verification/artifacts/"
                                        "shallow-water-dam-break-degenerate")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--quick", action="store_true",
                        help="only the anchor ratio + one strong bore, 2 grids")
    args = parser.parse_args(argv)

    eps_ladder = (0.9, 0.5, 1e-3, 1e-5) if args.quick else EPS_LADDER
    grid_ladder = (800, 1600) if args.quick else GRID_LADDER
    deep = {} if args.quick else DEEP_GRID_CASES

    laws = analytical_laws()
    cert = certification_table()
    swept = sweep(eps_ladder, grid_ladder, deep, workers=args.workers)
    vmap = validity_map(swept["rows"])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    figs = {}
    figs["fig_laws"] = figure_laws(laws, out / "fig_laws.png")
    figs["fig_validity"] = figure_validity(swept, vmap, out / "fig_validity.png")
    figs["fig_profiles"] = figure_profiles(out / "fig_profiles.png")
    payload = {
        "meta": {
            "anchor": "CASE-FVM-DAMBREAK-STOKER",
            "t_star_end": T_STAR_END,
            "h0": H0,
            "gravity": GRAVITY,
            "eps_ladder": list(eps_ladder),
            "grid_ladder": list(grid_ladder),
            "deep_grid": {f"{k:g}": v for k, v in deep.items()},
            "dry_floor_eps": DRY_FLOOR_EPS,
            "tolerance": TOLERANCE,
            "plateau_min_jump_fraction": PLATEAU_MIN_JUMP_FRACTION,
            "figures": figs,
        },
        "laws": laws,
        "certification": cert,
        "sweep": swept,
        "validity": vmap,
    }
    written = write_artifacts(out, payload)
    print(json.dumps({"artifacts": written, "jobs": swept["jobs"],
                      "wall_time_s": round(swept["wall_time_s"], 1),
                      "certification_pass": cert["pass"]}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())