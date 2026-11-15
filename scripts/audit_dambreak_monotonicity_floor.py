"""Which cell fails the dam-break monotonicity gate, and is it a solver defect?

Motivation
----------
The anchor ``CASE-FVM-DAMBREAK-STOKER`` carries one outstanding honest-FAIL
(``docs/case-fvm-dambreak-stoker-oblique/REPORT.md`` §5): on the acceptance grid
``n = 240``, axis-aligned, wet bed, every *magnitude* metric is far inside the
5 % tolerance (bore position 0.292 %, ``depth_l1`` 0.266 %, level step 0.029 %)
yet the waveform clause -- "monotone, no spurious oscillation" -- fails with
``local_extrema = 1``.  The published record therefore says the solver is not
monotone but not *how big* the offending oscillation is, nor *where* it sits
relative to the analytic structure, nor what it does under refinement.

That is the verification debt this driver pays.  It does not change the gate
(``EXTREMUM_FRACTION = 1e-4 h0``, imported from :mod:`.dam_break`, unchanged)
and it does not re-tune any estimator.  It only *locates* the failure:

* every beyond-gate extremum is reported with its amplitude, depth and normal
  coordinate (:func:`~tensorfvm.shallow_water.dam_break.local_extrema_details`);
* each one is classified against the closed-form Stoker regions
  (upstream / rarefaction fan / constant plateau / downstream), using
  :func:`~tensorfvm.shallow_water.exact.stoker_region_bounds`;
* the **same detector** is applied to the exact solution sampled on the same
  bins.  Whatever it reports there is a detector floor that every numerical
  reading inherits; what it reports *beyond* that floor is produced by the
  solver (the zero-bias split used by ``estimator_zero_bias``);
* the amplitude sequence over a refinement ladder separates "a real defect that
  persists" from "a threshold-proximity event on one grid".

Declared before any run (never fitted afterwards)
-------------------------------------------------
``GRIDS_2D``/``GRIDS_1D`` the ladder; ``KINK_BAND_CELLS`` how close (in cells)
an extremum must sit to an analytic region boundary to be called
*kink-seeded*; ``DECAY_MIN_GRIDS`` the number of grids an amplitude sequence
needs before a convergence statement is made at all.  The magnitude metrics are
re-graded on every grid so that a pass/fail statement never rests on one grid.

Outputs
-------
``src/tensorfvm/verification/artifacts/shallow-water-dam-break-monotonicity/``
(``monotonicity_floor.json`` + ``monotonicity_floor.png``), written
programmatically; the report quotes the JSON, never re-types the numbers.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensorfvm.shallow_water import (  # noqa: E402
    STOKER_WET,
    local_extrema_details,
    run_case,
)
from tensorfvm.shallow_water.dam_break import EXTREMUM_FRACTION  # noqa: E402
from tensorfvm.shallow_water.dam_break2d_oblique import (  # noqa: E402
    TOLERANCE,
    run_oblique_case,
)
from tensorfvm.shallow_water.exact import stoker_region_bounds  # noqa: E402

CASE = STOKER_WET
ANGLE_DEG = 0.0

#: refinement ladders (declared, not fitted)
GRIDS_2D = (120, 240, 480)
GRIDS_1D = (400, 800, 1600)

#: an extremum within this many cells of an analytic region boundary is called
#: *kink-seeded*: the closed form has a slope (or curvature) discontinuity
#: there, so a limiter reconstruction straddling it is where a one-cell
#: overshoot is expected to be born.
KINK_BAND_CELLS = 12.0

#: minimum number of grids carrying a beyond-gate extremum before any decay
#: statement is attempted.  One grid says nothing about convergence.
DECAY_MIN_GRIDS = 2

ARTIFACT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "src", "tensorfvm", "verification",
    "artifacts", "shallow-water-dam-break-monotonicity")


def region_of(xi: float, bounds: dict) -> str:
    """Name the exact Stoker region a similarity coordinate belongs to."""
    if xi <= bounds["head"]:
        return "upstream"
    if xi < bounds["tail"]:
        return "fan"
    if xi <= bounds["shock"]:
        return "plateau"
    return "downstream"


def classify(rec: dict, bounds: dict, t: float, dx: float) -> dict:
    """Attach the analytic region and the distance (in cells) to its walls."""
    s = rec["x"]
    xi = s / t
    walls = {"head": bounds["head"], "tail": bounds["tail"],
             "shock": bounds["shock"]}
    nearest_name, nearest_dx = min(
        ((name, abs(xi - xi_wall) * t / dx) for name, xi_wall in walls.items()),
        key=lambda item: item[1])
    out = dict(rec)
    out.update({
        "xi": xi,
        "region": region_of(xi, bounds),
        "nearest_wall": nearest_name,
        "nearest_wall_cells": float(nearest_dx),
        "kink_seeded": bool(nearest_dx <= KINK_BAND_CELLS),
        # depth deficit against the constant plateau state, in h0 units
        "value_over_h2": rec["value"] / bounds["h2"],
    })
    return out


def _oracle(bounds: dict, xs, h_exact, amplitude: float, t: float, dx: float):
    """The same detector applied to the exact profile on the same bins."""
    return [classify(rec, bounds, t, dx)
            for rec in local_extrema_details(h_exact, amplitude, x=xs)]


def run_2d(n: int, bounds: dict, amplitude: float) -> dict:
    wall0 = time.perf_counter()
    res = run_oblique_case(CASE, ANGLE_DEG, n)
    wall = time.perf_counter() - wall0
    disc = res["discretisation"]
    m = res["metrics"]
    prof = res["profile"]
    s = np.asarray(prof["s"], dtype=float)
    h_n = np.asarray(prof["h_numeric"], dtype=float)
    h_x = np.asarray(prof["h_exact"], dtype=float)
    dx = disc["dx"]
    t = disc["t_end"]
    numer = [classify(rec, bounds, t, dx)
             for rec in local_extrema_details(h_n, amplitude, x=s)]
    oracle = _oracle(bounds, s, h_x, amplitude, t, dx)
    return {
        "case": res["case"], "angle_deg": res["angle_deg"], "n": n, "dx": dx,
        "bins": int(s.size), "t_end": t,
        "wall_time_s": float(wall), "steps": disc["steps"],
        "gate": {"amplitude_fraction_of_h0": EXTREMUM_FRACTION,
                 "amplitude_absolute": amplitude},
        "magnitude_metrics": {
            "bore_position_error": m["front_error"],
            "depth_l1": m["depth_l1"],
            "level_step_error": m["step_ratio_error"],
            "h2_error": m["h2_error"],
            "tolerance": TOLERANCE,
        },
        "extrema_count": len(numer),
        "extrema": numer,
        "oracle_extrema_count": len(oracle),
        "oracle_extrema": oracle,
        "profile": {"s": s.tolist(), "h_numeric": h_n.tolist(),
                    "h_exact": h_x.tolist()},
    }


def run_1d(n: int, bounds: dict, amplitude: float) -> dict:
    wall0 = time.perf_counter()
    res = run_case(CASE, n=n)
    wall = time.perf_counter() - wall0
    disc = res["discretisation"]
    prof = res["profile"]
    x = np.asarray(prof["x"], dtype=float)
    h_n = np.asarray(prof["h_numeric"], dtype=float)
    h_x = np.asarray(prof["h_exact"], dtype=float)
    dx = disc["dx"]
    t = disc["t_end"]
    numer = [classify(rec, bounds, t, dx)
             for rec in local_extrema_details(h_n, amplitude, x=x)]
    m = res["metrics"]
    return {
        "case": res["case"], "n": n, "dx": dx, "bins": int(x.size),
        "wall_time_s": float(wall), "steps": disc["steps"],
        "magnitude_metrics": {
            "bore_position_error": m["shock_error"],
            "depth_l1": m["depth_l1"],
            "level_step_error": m["step_ratio_error"],
            "h2_error": m["h2_error"],
            "tolerance": TOLERANCE,
        },
        "extrema_count": len(numer),
        "extrema": numer,
        "oracle_extrema_count": len(_oracle(bounds, x, h_x, amplitude, t, dx)),
        "profile": {"s": x.tolist(), "h_numeric": h_n.tolist(),
                    "h_exact": h_x.tolist()},
    }


def decay_statement(rows) -> dict:
    """Amplitude against ``dx`` over the grids that actually carry an extremum."""
    live = [r for r in rows if r["extrema_count"] > 0]
    if len(live) < DECAY_MIN_GRIDS:
        return {"assessed": False,
                "reason": (f"only {len(live)} grid(s) carry a beyond-gate "
                           f"extremum; DECAY_MIN_GRIDS = {DECAY_MIN_GRIDS}")}
    amps = [max(r["extrema"][0]["amplitude"], 0.0) if r["extrema_count"] == 1
            else max(rec["amplitude"] for rec in r["extrema"]) for r in live]
    out = {"assessed": True, "grids": [r["n"] for r in live],
           "amplitude_h0": [a for a in amps],
           "per_grid": [
               {"n": r["n"], "dx": r["dx"], "count": r["extrema_count"],
                "max_amplitude": a,
                "max_amplitude_over_dx": a / r["dx"],
                "over_gate": a / r["gate"]["amplitude_absolute"]}
               for r, a in zip(live, amps)]}
    if len(live) >= 2:
        p = math.log(amps[0] / amps[-1]) / math.log(live[0]["dx"] / live[-1]["dx"])
        out["observed_order"] = float(p)
        out["order_note"] = ("p > 0: the oscillation shrinks under refinement "
                             "(discretisation noise), p <= 0: it does not")
    return out


def figure(rows2d, path: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bounds = stoker_region_bounds(CASE.h0, CASE.h1, CASE.gravity)
    t = rows2d[0]["t_end"]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))

    ax = axes[0]
    colours = plt.cm.viridis(np.linspace(0.15, 0.8, len(rows2d)))
    for row, colour in zip(rows2d, colours):
        s = np.asarray(row["profile"]["s"])
        h = np.asarray(row["profile"]["h_numeric"])
        keep = (s > bounds["head"] * t - 0.3) & (s < bounds["shock"] * t + 0.6)
        ax.plot(s[keep], h[keep], color=colour, lw=1.1,
                label=f"n={row['n']} (dx={row['dx']:.4f} m)")
        for rec in row["extrema"]:
            ax.plot([rec["x"]], [rec["value"]], "o", ms=5.5, mfc="none",
                    mec=colour, mew=1.4)
    hx = np.asarray(rows2d[0]["profile"]["h_exact"])
    s0 = np.asarray(rows2d[0]["profile"]["s"])
    keep0 = (s0 > bounds["head"] * t - 0.3) & (s0 < bounds["shock"] * t + 0.6)
    ax.plot(s0[keep0], hx[keep0], color="0.25", lw=2.2, ls="--",
            label="Stoker exact")
    for name, key in (("fan->plateau", "tail"), ("bore", "shock")):
        ax.axvline(bounds[key] * t, color="crimson", lw=0.9, alpha=0.55)
        ax.annotate(name, (bounds[key] * t, bounds["h2"] + 0.012),
                    rotation=90, fontsize=7.5, color="crimson", ha="right")
    ax.set_xlabel("$s$ along the dam normal  [m]")
    ax.set_ylabel("$h$  [m]")
    ax.set_title("Depth across the Stoker wave (circles = beyond-gate extrema)")
    ax.legend(fontsize=7.5, loc="lower left")
    ax.grid(alpha=0.25)

    ax = axes[1]
    for row, colour in zip(rows2d, colours):
        if not row["extrema"]:
            continue
        ds = row["dx"]
        amp = np.array([rec["amplitude"] for rec in row["extrema"]]) / ds
        ax.plot(np.full(amp.shape, ds), amp, "o", color=colour,
                label=f"n={row['n']}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$\\Delta x$  [m]")
    ax.set_ylabel("extremum amplitude / $\\Delta x$")
    ax.set_title("Oscillation amplitude in cell units")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=7.5)

    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grids-2d", type=int, nargs="+", default=list(GRIDS_2D))
    parser.add_argument("--grids-1d", type=int, nargs="+", default=list(GRIDS_1D))
    parser.add_argument("--outdir", default=ARTIFACT_DIR)
    args = parser.parse_args(argv)

    amplitude = EXTREMUM_FRACTION * CASE.h0
    bounds = stoker_region_bounds(CASE.h0, CASE.h1, CASE.gravity)
    payload = {
        "case": CASE.key, "angle_deg": ANGLE_DEG,
        "case_params": {"h0": CASE.h0, "h1": CASE.h1, "gravity": CASE.gravity,
                        "t_star_end": CASE.t_star_end},
        "reference": ("Ritter (1892) Z. Ver. Deutsch. Ing. 36:947; "
                      "Stoker (1957) Water Waves ch. 10"),
        "declared": {
            "gate_amplitude_fraction_of_h0": EXTREMUM_FRACTION,
            "gate_amplitude_absolute": amplitude,
            "tolerance": TOLERANCE,
            "kink_band_cells": KINK_BAND_CELLS,
            "decay_min_grids": DECAY_MIN_GRIDS,
            "grids_2d": args.grids_2d, "grids_1d": args.grids_1d,
        },
        "stoker_regions": bounds,
    }
    rows2d = [run_2d(n, bounds, amplitude) for n in args.grids_2d]
    rows1d = [run_1d(n, bounds, amplitude) for n in args.grids_1d]
    payload["runs_2d"] = rows2d
    payload["runs_1d"] = rows1d
    payload["decay_2d"] = decay_statement(rows2d)
    payload["decay_1d"] = decay_statement(rows1d)

    # verdict, computed from the rows -- never typed by hand
    worst = max(rows2d, key=lambda r: r["extrema_count"])
    payload["verdict"] = {
        "grids_with_extrema_2d": [r["n"] for r in rows2d if r["extrema_count"]],
        "max_extrema_2d": worst["extrema_count"],
        "worst_grid_2d": worst["n"],
        "oracle_extrema_2d": [r["oracle_extrema_count"] for r in rows2d],
        "all_magnitude_metrics_pass": bool(
            all(r["magnitude_metrics"][k] <= TOLERANCE
                for r in rows2d + rows1d
                for k in ("bore_position_error", "depth_l1",
                          "level_step_error"))),
        "extrema_are_kink_seeded": bool(
            all(rec["kink_seeded"] for r in rows2d + rows1d
                for rec in r["extrema"])),
        "regions_2d": sorted({rec["region"] for r in rows2d
                              for rec in r["extrema"]}),
    }

    os.makedirs(args.outdir, exist_ok=True)
    json_path = os.path.join(args.outdir, "monotonicity_floor.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)
    png_path = figure(rows2d, os.path.join(args.outdir, "monotonicity_floor.png"))
    payload["artifacts"] = {"json": json_path, "figure": png_path}

    v = payload["verdict"]
    print(f"2-D grids with beyond-gate extrema: {v['grids_with_extrema_2d']}")
    print(f"worst grid n={v['worst_grid_2d']} count={v['max_extrema_2d']}")
    print(f"oracle (exact profile) counts: {v['oracle_extrema_2d']}")
    print(f"all magnitude metrics pass: {v['all_magnitude_metrics_pass']}")
    print(f"all extrema kink-seeded: {v['extrema_are_kink_seeded']}")
    print(f"regions: {v['regions_2d']}")
    print(f"decay 2-D: {payload['decay_2d']}")
    print(f"decay 1-D: {payload['decay_1d']}")
    print(f"wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())