"""Long-window (2-period) health audit of ``CASE-FVM-THACKER-BASIN``.

Why this module exists
----------------------
``docs/case-fvm-thacker-basin/REPORT.md`` grades the basin only over *half* an
oscillation period: at ``--periods 2`` the reference run developed
non-physical velocities (``18.2 m/s`` at ``n = 40``, step 200) and was frozen
out of the graded window.  That debt -- "2-period long-time momentum
injection" -- was carried in ``docs/VERIFICATION_DEBT_LEDGER.md`` (D16) with a
diagnosis but no *measurement protocol*: nothing in the repository reports how
far off the 2-period solution is, where the spurious velocity sits, or whether
a candidate cure actually fixes the accuracy.

This module supplies that protocol.  It integrates the frozen benchmark
(``thacker_basin.DEFAULTS``) over a configurable number of periods and reports

``centre_depth``
    Relative ``L2`` (``RMS error / half-range of the exact centre-depth
    oscillation``) and ``Linf`` of the centre-depth history -- the same
    functional as the graded ``C4``, only over a longer window.
``velocity``
    ``max |u|`` over the run, its time and cell, compared with the *exact*
    envelope ``max_wet |beta(t)| r = |beta(t)| s(t)`` (the analytical solution
    is a linear-in-``(x, y)`` velocity field, so its maximum is analytic).
    A run whose numerical maximum exceeds that envelope by a large factor has
    momentum that the exact solution does not contain.
``spike``
    The state of the offending cell: depth, ``h/H``, radius, exact shoreline
    radius at that instant.  This is what distinguishes the two candidate
    explanations -- "coarse grid smears the shoreline" (spike *inside* the wet
    region at ``h`` comparable to the local depth) versus "thin film outside
    the shoreline accelerated by an unbalanced source" (spike at ``r > s(t)``
    with ``h/H`` below a percent).
``spurious_mass``
    Water mass held outside the exact shoreline, relative to the conserved
    total.  Volume conservation alone cannot see this: the ledger run conserves
    volume to ``4e-14`` while carrying mass far outside the exact basin.
``clip_study``
    Optional cure probe: momentum is zeroed in cells with ``h < thin * H``
    after every stage.  Reported, not assumed -- it is a regularisation with a
    threshold, so the study is what decides whether it is a cure or a
    cosmetic change to ``max |u|``.

Honest scope: this module *measures and attributes* the long-window defect; it
does not claim to fix it.  The clip study is included only to rule a candidate
out (or in) with numbers.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from ..shallow_water.swe2d import ShallowWater2D
from . import thacker_basin as tb

__all__ = ["run", "figure", "main", "DEFAULTS"]

#: Thin-film thresholds (``h / H``) probed by the cure study.
CLIP_FRACTIONS = (0.0, 1e-2, 3e-2)


class _ClippedSolver(ShallowWater2D):
    """``ShallowWater2D`` with an optional thin-film momentum clip."""

    def __init__(self, *args, thin: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.thin = float(thin)
        self.thin_clips = 0

    def _sanitise(self) -> None:  # noqa: D401 - see ShallowWater2D
        super()._sanitise()
        if self.thin > 0.0:
            mask = self.h < self.thin
            hits = int(np.count_nonzero(mask & ((self.hu != 0.0)
                                                | (self.hv != 0.0))))
            if hits:
                self.thin_clips += hits
                self.hu = np.where(mask, 0.0, self.hu)
                self.hv = np.where(mask, 0.0, self.hv)


def _solver(basin, n: int, params: dict, thin: float = 0.0):
    L = params["domain_factor"] * basin.a
    X, Y = tb.grid(basin, n, params["domain_factor"])
    return _ClippedSolver(n, n, -L, L, -L, L, bed=basin.bed(X, Y),
                          gravity=basin.gravity, cfl=params["cfl"],
                          order=params["order"],
                          dry_depth=params["dry_depth"],
                          well_balanced=True, thin=thin)


def envelope_speed(basin, t: float) -> float:
    """Exact ``max |u|`` over the wet region: ``|beta(t)| * s(t)``."""
    e = basin.envelope(t)
    return abs(e["beta"]) * e["s"]


def run(n: int = 40, periods: float = 2.0, thin: float = 0.0,
        samples: int = 81) -> dict:
    """Integrate ``periods`` oscillations at ``n x n`` cells and measure them."""
    params = dict(tb.DEFAULTS)
    basin = tb.build_basin(**{k: params[k] for k in
                              ("H", "a", "amplitude", "gravity")})
    solver = _solver(basin, n, params, thin=thin)
    X, Y = tb.grid(basin, n, params["domain_factor"])
    R = np.hypot(X, Y)
    solver.set_state(*basin.initial_state(X, Y))

    t_end = periods * basin.period
    volume0 = solver.total_volume()
    volume_exact = basin.volume
    centre_num, centre_exact, times = [], [], []
    vmax = {"speed": 0.0, "t": 0.0, "cell": None}
    state_at_spike = {}
    next_sample = [0.0]
    dt_sample = t_end / max(samples - 1, 1)

    def callback(sol):
        speed = sol.max_spurious_velocity()
        if speed > vmax["speed"]:
            h, u, v = sol.primitive()
            i, j = np.unravel_index(np.argmax(np.hypot(u, v)), h.shape)
            vmax.update(speed=float(speed), t=float(sol.time),
                        cell=(int(i), int(j)))
            state_at_spike.update(
                h=float(h[i, j]), h_over_H=float(h[i, j] / basin.H),
                u=float(u[i, j]), v=float(v[i, j]),
                radius=float(R[i, j]),
                shoreline=float(basin.shoreline_radius(sol.time)),
                eta=float((h + sol.bed)[i, j]),
                bed=float(sol.bed[i, j]),
                speed_exact=float(envelope_speed(basin, sol.time)),
            )
        while sol.time >= next_sample[0] or sol.time >= t_end:
            times.append(sol.time)
            centre_num.append(float(sol.h[solver.nx // 2, solver.ny // 2]))
            centre_exact.append(float(basin.centre_depth(sol.time)))
            next_sample[0] += dt_sample
            if sol.time >= t_end:
                break

    wall = time.perf_counter()
    solver.advance(t_end, callback=callback, max_steps=1_000_000)
    wall = time.perf_counter() - wall

    exact = np.array(centre_exact)
    num = np.array(centre_num)
    half_range = 0.5 * (exact.max() - exact.min())
    err = num - exact
    h, u, v = solver.primitive()
    radius = R
    shoreline = basin.shoreline_radius(solver.time)
    outside = radius > shoreline
    total = solver.total_volume()

    film = h < CLIP_FRACTIONS[1] * basin.H
    return {
        "n": int(n),
        "periods": float(periods),
        "t_end": float(t_end),
        "steps": int(solver.steps),
        "wall_time_s": float(wall),
        "centre_depth": {
            "l2_relative": float(np.sqrt(np.mean(err ** 2)) / half_range),
            "linf_relative": float(np.max(np.abs(err)) / half_range),
            "tolerance": float(params["tolerance_centre"]),
            "half_range": float(half_range),
        },
        "velocity": {
            "max": vmax["speed"],
            "t": vmax["t"],
            "t_over_period": vmax["t"] / basin.period,
            "cell": vmax["cell"],
            "speed_exact_there": state_at_spike.get("speed_exact"),
            "ratio_to_exact": (None if not state_at_spike.get("speed_exact")
                               else vmax["speed"] / state_at_spike["speed_exact"]),
            "tolerance": float(params["tolerance_centre"]),
        },
        "spike": state_at_spike,
        "spurious_mass": {
            "relative": float(h[outside].sum() / h.sum()) if h.sum() > 0 else 0.0,
            "outside_shoreline": float(shoreline),
            "final_shoreline_exact": float(shoreline),
        },
        "film": {
            "threshold_fraction": CLIP_FRACTIONS[1],
            "cells": int(np.count_nonzero(film)),
            "mass_relative": float(h[film].sum() / h.sum()) if h.sum() > 0 else 0.0,
        },
        "volume": {
            "exact": float(volume_exact),
            "initial": float(volume0),
            "final": float(total),
            "drift": float(abs(total - volume0) / volume0),
            "relative_to_exact": float(abs(total - volume_exact) / volume_exact),
        },
        "thin": float(thin),
        "thin_clips": int(getattr(solver, "thin_clips", 0)),
        "health": solver.summary(),
        "series": {"t": times, "centre_num": centre_num,
                   "centre_exact": centre_exact},
    }


def figure(metrics: dict, clip_study: list, path) -> str:
    """Two-panel figure: centre-depth history and the velocity spike."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    t = np.array(metrics["series"]["t"])
    ax.plot(t, metrics["series"]["centre_exact"], "k-", lw=2,
            label="exact (Thacker 1981)")
    ax.plot(t, metrics["series"]["centre_num"], "C0--", lw=1.4,
            ls="-", label="numerical, n=%d" % metrics["n"])
    ax.set_xlabel("t [s]")
    ax.set_ylabel("centre depth h_c [m]")
    ax.set_title("centre-depth history over %.1f periods" % metrics["periods"])
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    labels = [("baseline" if not s["thin"] else "clip h<%.0e H" % s["thin"])
              for s in clip_study]
    vals = [s["velocity"]["max"] for s in clip_study]
    l2 = [100.0 * s["centre_depth"]["l2_relative"] for s in clip_study]
    x = np.arange(len(labels))
    ax.bar(x - 0.2, vals, width=0.4, color="C3", label="max |u| [m/s]")
    ax.bar(x + 0.2, l2, width=0.4, color="C0", label="centre-depth L2 [%]")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yscale("log")
    ax.set_title("spurious velocity vs accuracy")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/case-fvm-thacker-2period")
    ap.add_argument("--resolution", type=int, default=40)
    ap.add_argument("--periods", type=float, default=2.0)
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    study = [run(n=args.resolution, periods=args.periods, thin=f)
             for f in CLIP_FRACTIONS]
    baseline = study[0]
    fig_path = figure(baseline, study, out / "fig_thacker_2period.png")
    payload = {"baseline": {k: v for k, v in baseline.items() if k != "series"},
               "clip_study": [{k: v for k, v in s.items() if k != "series"}
                              for s in study],
               "figure": fig_path}
    (out / "metrics.json").write_text(json.dumps(payload, indent=2,
                                                 sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())