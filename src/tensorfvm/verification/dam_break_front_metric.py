"""Verification surface for the **dry-bed front metric** of the dam-break benchmark.

Why this file exists
--------------------
:mod:`tensorfvm.shallow_water.dam_break` grades the Ritter (dry-bed) case on a
"wave position" metric.  The shipped metric, ``front_wetted`` -- the outermost
cell with ``h > 0`` -- *looks* threshold-free but is not: the Ritter tip is a
zero-depth cusp with ``h = (x_f - x)**2 / (9 g t**2)``, so "the last wet cell"
is a **depth-floor detector**.  Its reading is set by the numerical depth floor
(measured ``2.7e-04 ... 3.7e-04 h0``, roughly independent of ``dx``), not by the
grid, and the metric therefore **does not converge**:

=================  ========  ========  ========  ========  ========
n                    200       400       800      1600      3200
=================  ========  ========  ========  ========  ========
``front_wetted``    1.875 %   1.562 %   2.656 %   2.891 %   2.461 %
``front_fan``       0.332 %   0.106 %   0.039 %   0.017 %   0.008 %
=================  ========  ========  ========  ========  ========

A non-convergent headline metric is a verification defect even when its value
happens to sit under the 5 % tolerance: it cannot distinguish a better scheme
from a coarser floor, and it *inverts* the scheme-order comparison (at
``n = 800`` piecewise-constant gives 1.41 % and MUSCL 2.66 %", i.e. the worse
scheme looks better).  The replacement metric is the shape-consistent fan
estimator :func:`tensorfvm.shallow_water.dam_break.fan_front_estimate`, which
uses the exact rarefaction identity ``x_f = x_i + 3 t sqrt(g h_i)`` cell by cell.

Criteria (pre-declared, thresholds fixed before the runs)
---------------------------------------------------------
``F1`` fixed point on the exact profile -- the estimator applied to the *exact*
       Ritter profile returns ``x_f`` to ``< 1e-10`` relative (it is an identity,
       not a fit; anything else means the algebra is wrong).
``F2`` convergence of the headline metric -- monotone decreasing over
       ``n = 200 ... 3200``, observed order ``>= 0.8``, and ``<= 0.05 %`` at
       ``n = 1600`` (100x margin under the 5 % acceptance).
``F3`` the defect itself -- ``front_wetted`` error is **not** reduced by
       refinement (``err(3200) / err(400) >= 1``), while its implied depth floor
       ``(delta / (3 t))**2 / g`` stays within a factor 5 over ``n >= 400``
       (``delta = x_f - x_wetted``).  Both must hold for the floor story to be
       the explanation rather than a hand-wave.
``F4`` non-vacuity (negative control) -- a run with ``g' = 1.05 g``, graded
       against the nominal-gravity front, must be reported as ``sqrt(1.05) - 1
       = 2.47 %`` within 5 % relative; a metric that cannot see a 5 % gravity
       error is void.
``F5`` order discrimination restored -- ``front_fan`` error of order 1 must be
       at least 3x the order-2 error at ``n = 800``, whereas ``front_wetted``
       reports the *opposite* ordering in the same pair.
``F6`` window robustness -- every window of ``FAN_WINDOWS`` is monotone
       decreasing over the last three grids and below 5 % from ``n = 800`` on
       (the tail-inclusive windows are biased, never divergent).
``F7`` time-step insensitivity -- ``front_fan`` error varies by less than 2x
       across ``CFL = 0.2 / 0.4 / 0.8`` at ``n = 800``.
``F8`` conservation untouched -- mass drift ``<= 1e-12``, no clipped cells and
       ``h >= 0`` on every grid (the metric change must not move the solution).
``F9`` wet-bed branch unchanged -- the Stoker headline numbers still meet the
       5 % gate and its verdict keys are exactly the historical set.
``F10`` API back-compatibility -- ``front_wetted_error`` is still reported and
       still reproduces the published ``2.891 %`` at ``n = 1600`` (``+-0.2 %``),
       so the historical claim stays auditable instead of disappearing.

Run::

    PYTHONPATH=src python -m tensorfvm.verification.dam_break_front_metric
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..shallow_water.dam_break import (
    FAN_WINDOW,
    RITTER_DRY,
    STOKER_WET,
    fan_front_estimate,
    run_case,
)
from ..shallow_water.swe1d import ShallowWater1D

#: grids of the convergence ladder (dx halves at every step)
RESOLUTIONS = (200, 400, 800, 1600, 3200)

#: windows swept for F6; the first one is the headline (module constant)
FAN_WINDOWS = ((1e-2, 2e-1), (1e-3, 5e-2), (1e-4, 1e-1))

#: published numbers this driver re-checks (docs/shallow-water-dam-break)
PUBLISHED_WETTED_ERROR_N1600 = 0.02891


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _fan_error(row: dict, window=FAN_WINDOW, gravity: float | None = None) -> dict:
    """Fan estimator on a stored run, graded against that run's exact front."""
    profile = row["profile"]
    x = np.asarray(profile["x"], dtype=float)
    h = np.asarray(profile["h_numeric"], dtype=float)
    t = row["discretisation"]["t_end"]
    g = RITTER_DRY.gravity if gravity is None else gravity
    est = fan_front_estimate(x, h, t, g, RITTER_DRY.h0, window)
    x_f = row["metrics"]["front_exact"]
    return {"front": est["front"], "error": abs(est["front"] - x_f) / x_f,
            "cells": est["cells"], "spread": est["spread"]}


def _order(errors, dxs) -> float:
    """Least-squares slope of ``log(err)`` against ``log(dx)``."""
    e = np.asarray(errors, dtype=float)
    d = np.asarray(dxs, dtype=float)
    keep = e > 0.0
    if int(np.count_nonzero(keep)) < 3:
        return float("nan")
    return float(np.polyfit(np.log(d[keep]), np.log(e[keep]), 1)[0])


def _gravity_control(n: int = 800, factor: float = 1.05) -> dict:
    """Run the same case with ``g' = factor * g`` and grade it at nominal ``g``."""
    g2 = factor * RITTER_DRY.gravity
    t_end = RITTER_DRY.t_star_end * math.sqrt(RITTER_DRY.h0 / RITTER_DRY.gravity)
    solver = ShallowWater1D(n, RITTER_DRY.x_min, RITTER_DRY.x_max, gravity=g2,
                            cfl=0.4, order=2)
    solver.set_state(np.where(solver.x < RITTER_DRY.x_dam, RITTER_DRY.h0,
                              RITTER_DRY.h1))
    solver.advance(t_end)
    est = fan_front_estimate(solver.x, solver.h, solver.time, g2, RITTER_DRY.h0)
    nominal = 2.0 * math.sqrt(RITTER_DRY.gravity * RITTER_DRY.h0) * solver.time
    predicted = math.sqrt(factor) - 1.0
    measured = (est["front"] - nominal) / nominal
    return {
        "n": n,
        "gravity_factor": factor,
        "front_nominal": nominal,
        "front_estimated": est["front"],
        "measured_relative": measured,
        "predicted_relative": predicted,
        "relative_deviation": abs(measured - predicted) / predicted,
    }


# --------------------------------------------------------------------------- #
# criteria
# --------------------------------------------------------------------------- #
def _criteria(runs: dict, order_runs: dict, cfl_runs: dict, control: dict) -> list[dict]:
    criteria: list[dict] = []
    dxs = [runs[n]["discretisation"]["dx"] for n in RESOLUTIONS]

    # ---- F1: identity on the exact profile
    t_probe = 0.7
    x_probe = np.linspace(-1.0, 3.9, 4001)
    x_f_probe = 2.0 * math.sqrt(RITTER_DRY.gravity * RITTER_DRY.h0) * t_probe
    c = (2.0 * math.sqrt(RITTER_DRY.gravity * RITTER_DRY.h0) - x_probe / t_probe) / 3.0
    h_probe = np.clip(c, 0.0, None) ** 2 / RITTER_DRY.gravity
    est = fan_front_estimate(x_probe, h_probe, t_probe, RITTER_DRY.gravity,
                             RITTER_DRY.h0)
    f1 = abs(est["front"] - x_f_probe) / x_f_probe
    criteria.append({
        "id": "F1", "what": "estimator is the exact fan identity (not a fit)",
        "value": f1, "limit": 1e-10, "pass": bool(f1 < 1e-10),
        "detail": {"front": est["front"], "exact": x_f_probe, "cells": est["cells"]},
    })

    # ---- F2: convergence of the headline
    fan_errors = [runs[n]["metrics"]["front_fan_error"] for n in RESOLUTIONS]
    monotone = all(b < a for a, b in zip(fan_errors, fan_errors[1:]))
    p_fan = _order(fan_errors[1:], dxs[1:])
    f2 = bool(monotone and p_fan >= 0.8 and fan_errors[3] <= 5e-4)
    criteria.append({
        "id": "F2", "what": "headline (front_fan) converges: order >= 0.8, <= 0.05 % at n=1600",
        "value": p_fan, "limit": 0.8, "pass": f2,
        "detail": {"errors": fan_errors, "monotone": monotone,
                   "error_n1600": fan_errors[3], "order_from_n400": p_fan},
    })

    # ---- F3: the defect (non-convergence of the wetted edge)
    wetted = [runs[n]["metrics"]["front_wetted_error"] for n in RESOLUTIONS]
    floors = []
    for n in RESOLUTIONS:
        m = runs[n]["metrics"]
        delta = max(m["front_exact"] - m["front_wetted"], 0.0)
        t = runs[n]["discretisation"]["t_end"]
        floors.append((delta / (3.0 * t)) ** 2 / RITTER_DRY.gravity / RITTER_DRY.h0)
    floor_ratio = max(floors[1:]) / min(floors[1:])
    f3 = bool(wetted[-1] / wetted[1] >= 1.0 and floor_ratio <= 5.0)
    criteria.append({
        "id": "F3", "what": "front_wetted does NOT converge and its implied depth floor is dx-independent",
        "value": wetted[-1] / wetted[1], "limit": 1.0, "pass": f3,
        "detail": {"wetted_errors": wetted, "error_ratio_3200_over_400": wetted[-1] / wetted[1],
                   "implied_depth_floor_fraction": floors, "floor_ratio_n_ge_400": floor_ratio},
    })

    # ---- F4: negative control
    f4 = control["relative_deviation"] <= 0.05
    criteria.append({
        "id": "F4", "what": "a 5 % gravity error is reported as 2.47 % front error",
        "value": control["relative_deviation"], "limit": 0.05, "pass": f4,
        "detail": control,
    })

    # ---- F5: order discrimination
    e1 = order_runs[1]["metrics"]["front_fan_error"]
    e2 = order_runs[2]["metrics"]["front_fan_error"]
    w1 = order_runs[1]["metrics"]["front_wetted_error"]
    w2 = order_runs[2]["metrics"]["front_wetted_error"]
    ratio = e1 / e2
    f5 = bool(ratio >= 3.0 and w1 < w2)
    criteria.append({
        "id": "F5", "what": "fan metric ranks order-1 worse than order-2 (wetted edge ranks it backwards)",
        "value": ratio, "limit": 3.0, "pass": f5,
        "detail": {"n": 800, "fan_error_order1": e1, "fan_error_order2": e2,
                   "wetted_error_order1": w1, "wetted_error_order2": w2,
                   "inversion_reproduced": bool(w1 < w2)},
    })

    # ---- F6: window robustness
    window_rows = {}
    for window in FAN_WINDOWS:
        errs = [_fan_error(runs[n], window)["error"] for n in RESOLUTIONS]
        window_rows[f"{window[0]:g}-{window[1]:g}"] = {
            "errors": errs, "monotone_last3": bool(errs[2] > errs[3] > errs[4]),
            "max_from_n800": max(errs[2:]),
        }
    f6 = all(row["monotone_last3"] and row["max_from_n800"] < 0.05
             for row in window_rows.values())
    criteria.append({
        "id": "F6", "what": "all windows converge and stay < 5 % from n=800",
        "value": max(row["max_from_n800"] for row in window_rows.values()),
        "limit": 0.05, "pass": bool(f6), "detail": window_rows,
    })

    # ---- F7: CFL insensitivity
    cfl_errors = {str(cfl): cfl_runs[cfl]["metrics"]["front_fan_error"]
                  for cfl in sorted(cfl_runs)}
    spread = max(cfl_errors.values()) / min(cfl_errors.values())
    criteria.append({
        "id": "F7", "what": "front_fan varies < 2x over CFL 0.2/0.4/0.8 (n=800)",
        "value": spread, "limit": 2.0, "pass": bool(spread < 2.0),
        "detail": cfl_errors,
    })

    # ---- F8: conservation untouched
    drifts = [runs[n]["metrics"]["mass_drift"] for n in RESOLUTIONS]
    clipped = [runs[n]["discretisation"]["clipped_cells"] for n in RESOLUTIONS]
    hmin = [runs[n]["metrics"]["h_min"] for n in RESOLUTIONS]
    f8 = bool(max(drifts) <= 1e-12 and max(clipped) == 0 and min(hmin) >= 0.0)
    criteria.append({
        "id": "F8", "what": "mass drift <= 1e-12, no clipping, h >= 0 on all grids",
        "value": max(drifts), "limit": 1e-12, "pass": f8,
        "detail": {"mass_drift": drifts, "clipped_cells": clipped, "h_min": hmin},
    })

    # ---- F9: wet branch unchanged
    wet = runs["wet"]
    wm = wet["metrics"]
    expected_keys = {"shock_position<5%", "step_ratio<5%", "plateau_h2<5%",
                     "depth_l1<5%", "monotone(no_extrema)", "tv_no_growth<0.5%"}
    f9 = bool(set(wet["verdict"]) - {"all_pass"} == expected_keys and all(
        item["pass"] for key, item in wet["verdict"].items() if key != "all_pass"))
    criteria.append({
        "id": "F9", "what": "Stoker branch unchanged: same verdict keys, all pass",
        "value": wm["shock_error"], "limit": 0.05, "pass": f9,
        "detail": {"keys": sorted(k for k in wet["verdict"] if k != "all_pass"),
                   "shock_error": wm["shock_error"],
                   "step_ratio_error": wm["step_ratio_error"],
                   "plateau_error": wm["h2_error"]},
    })

    # ---- F10: API back-compatibility
    legacy = runs[1600]["metrics"]["front_wetted_error"]
    dev = abs(legacy - PUBLISHED_WETTED_ERROR_N1600) / PUBLISHED_WETTED_ERROR_N1600
    criteria.append({
        "id": "F10", "what": "front_wetted_error still reproduces the published 2.891 % (n=1600)",
        "value": legacy, "limit": PUBLISHED_WETTED_ERROR_N1600 * (1.0 + 0.2),
        "pass": bool(dev <= 0.2),
        "detail": {"measured": legacy, "published": PUBLISHED_WETTED_ERROR_N1600,
                   "relative_deviation": dev,
                   "dry_verdict_keys": sorted(k for k in runs[1600]["verdict"]
                                              if k != "all_pass")},
    })
    return criteria


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def run(resolutions=RESOLUTIONS) -> dict:
    """Run the whole surface; returns the JSON payload."""
    runs = {}
    for n in resolutions:
        runs[n] = run_case(RITTER_DRY, n=n)
    runs[1600] = runs.get(1600) or run_case(RITTER_DRY, n=1600)
    runs["wet"] = run_case(STOKER_WET, n=1600)
    order_runs = {1: run_case(RITTER_DRY, n=800, order=1),
                  2: runs.get(800) or run_case(RITTER_DRY, n=800)}
    cfl_runs = {cfl: (runs.get(800) if cfl == 0.4 else run_case(RITTER_DRY, n=800, cfl=cfl))
                for cfl in (0.2, 0.4, 0.8)}
    control = _gravity_control()
    criteria = _criteria(runs, order_runs, cfl_runs, control)
    ladder = [{
        "n": n,
        "dx": runs[n]["discretisation"]["dx"],
        "steps": runs[n]["discretisation"]["steps"],
        "wall_time_s": runs[n]["discretisation"]["wall_time_s"],
        "front_exact": runs[n]["metrics"]["front_exact"],
        "front_fan": runs[n]["metrics"]["front_fan"],
        "front_fan_error": runs[n]["metrics"]["front_fan_error"],
        "front_fan_cells": runs[n]["metrics"]["front_fan_cells"],
        "front_fan_spread": runs[n]["metrics"]["front_fan_spread"],
        "front_fan_slope_error": runs[n]["metrics"]["front_fan_slope_error"],
        "front_wetted": runs[n]["metrics"]["front_wetted"],
        "front_wetted_error": runs[n]["metrics"]["front_wetted_error"],
        "precursor_length": runs[n]["metrics"]["precursor_length"],
        "precursor_depth": runs[n]["metrics"]["precursor_depth"],
        "depth_l1": runs[n]["metrics"]["depth_l1"],
    } for n in resolutions]
    n_pass = sum(1 for c in criteria if c["pass"])
    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "case": RITTER_DRY.summary(),
        "window": list(FAN_WINDOW),
        "criteria": criteria,
        "ladder": ladder,
        "order": {"front_fan_from_n400": _order(
            [row["front_fan_error"] for row in ladder[1:]],
            [row["dx"] for row in ladder[1:]])},
        "n_pass": n_pass,
        "n_total": len(criteria),
        "all_pass": bool(n_pass == len(criteria)),
    }


def figure(payload: dict, path) -> Path:
    """Three panels: tip profile, metric convergence, implied depth floor."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run1600 = run_case(RITTER_DRY, n=1600)
    x = np.asarray(run1600["profile"]["x"])
    h = np.asarray(run1600["profile"]["h_numeric"])
    h_ex = np.asarray(run1600["profile"]["h_exact"])
    t = run1600["discretisation"]["t_end"]
    x_f = run1600["metrics"]["front_exact"]
    x_w = run1600["metrics"]["front_wetted"]
    lo, hi = FAN_WINDOW
    ladder = payload["ladder"]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))

    ax = axes[0]
    tail = (x > x_f - 1.2) & (x < x_f + 0.25)
    ax.plot(x[tail], h[tail] / RITTER_DRY.h0, "o-", ms=2.5, lw=0.8,
            label="FVM n=1600")
    ax.plot(x[tail], h_ex[tail] / RITTER_DRY.h0, "-", lw=1.0, label="Ritter exact")
    ax.axvspan(x_f - 0.35, x_f - 0.35, color="none")
    ax.axhline(lo, color="0.6", ls=":", lw=0.8)
    ax.axhline(hi, color="0.6", ls=":", lw=0.8)
    ax.axvline(x_f, color="C2", lw=1.0, label=f"$x_f$ = {x_f:.3f} m")
    ax.axvline(x_w, color="C3", lw=1.0, label=f"wetted edge = {x_w:.3f} m")
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 3.0)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("h / h0")
    ax.set_title("Ritter tip: zero-depth cusp smeared over ~30 cells")
    ax.legend(fontsize=7, loc="lower right")

    ax = axes[1]
    dxs = [row["dx"] for row in ladder]
    ax.plot(dxs, [row["front_fan_error"] * 100 for row in ladder], "o-",
            label="front_fan (headline)")
    ax.plot(dxs, [row["front_wetted_error"] * 100 for row in ladder], "s--",
            label="front_wetted (legacy)")
    ref = np.array(dxs)
    ax.plot(ref, 4.0 * ref / dxs[0], ":", color="0.5", label="$\\propto \\Delta x$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$\\Delta x$ [m]")
    ax.set_ylabel("front error [%]")
    ax.set_title(f"front error vs $\\Delta x$ (fan order {payload['order']['front_fan_from_n400']:.2f})")
    ax.legend(fontsize=7)

    ax = axes[2]
    f3 = next(c for c in payload["criteria"] if c["id"] == "F3")
    floors = f3["detail"]["implied_depth_floor_fraction"]
    ax.plot(range(len(ladder)), floors, "o-")
    ax.set_yscale("log")
    ax.set_xticks(range(len(ladder)))
    ax.set_xticklabels([str(row["n"]) for row in ladder])
    ax.set_xlabel("grid n")
    ax.set_ylabel("implied depth floor [h0]")
    ax.set_title("wetted edge = depth-floor detector (floor $\\approx$ const)")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def write_artifacts(outdir="docs/verification-dam-break-front-metric") -> dict:
    payload = run()
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    fig_path = figure(payload, out / "fig_dam_break_front_metric.png")
    payload["figure"] = str(fig_path)
    (out / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="docs/verification-dam-break-front-metric")
    args = parser.parse_args(argv)
    payload = write_artifacts(args.out)
    for criterion in payload["criteria"]:
        print(f"{criterion['id']:>4} {'PASS' if criterion['pass'] else 'FAIL'}  "
              f"{criterion['what']}  -> {criterion['value']:.4g} "
              f"(limit {criterion['limit']:.4g})")
    print(f"{payload['n_pass']}/{payload['n_total']} PASS")
    return 0 if payload["all_pass"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())