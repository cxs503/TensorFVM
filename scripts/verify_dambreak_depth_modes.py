#!/usr/bin/env python
"""Oracle-free wet-bed plateau / bore reading: grid study, artifact and figure.

The Stoker (wet-bed) benchmark publishes ``h2`` / ``step_ratio`` measured inside
a window placed from the *closed form* (``AUDIT.md`` F4), which makes the two
quantities near-tautological for the plateau position, and it reads the bore
position with a gradient walk whose cut has to be retried per grid (F9).  This
script grades the third, fully oracle-free mechanism
(:func:`tensorfvm.shallow_water.dam_break.plateau_from_depth_modes`): every
level of the profile is a mode of its depth *distribution*, so ``h1``, ``h2``
and the bore position come from the numerical profile alone.

Outputs
-------
``src/tensorfvm/verification/artifacts/shallow-water-dam-break/depth_modes.json``
    per-grid readings of all three mechanisms plus the worst-case errors.
``docs/case-fvm-dambreak-stoker/depth_modes.png``
    left: numerical profiles with the detected modal levels and the detected
    bore; right: log-log error of ``h2`` and of the bore position against ``dx``
    for the three mechanisms, with the declared 5 % criterion.

Usage::

    PYTHONPATH=src python scripts/verify_dambreak_depth_modes.py
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from tensorfvm.shallow_water.dam_break import (
    ACCEPTED_MIN_N,
    PLATEAU_MODE_BINS,
    PLATEAU_MODE_MIN_FRACTION,
    PLATEAU_MODE_TOL_FRACTION,
    STOKER_WET,
    TOLERANCE,
    depth_mode_study,
    plateau_from_depth_modes,
)
from tensorfvm.shallow_water.exact import (
    stoker_constant_state,
    t_star_to_time,
)
from tensorfvm.shallow_water.swe1d import ShallowWater1D

ARTIFACT = Path("src/tensorfvm/verification/artifacts/shallow-water-dam-break")
DOCS = Path("docs/case-fvm-dambreak-stoker")


def _profile(n: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Integrate the Stoker case to ``t* = 2`` and return ``(x, h, t)``."""
    t_end = t_star_to_time(STOKER_WET.t_star_end, STOKER_WET.h0,
                           STOKER_WET.gravity)
    solver = ShallowWater1D(n, STOKER_WET.x_min, STOKER_WET.x_max,
                            gravity=STOKER_WET.gravity, cfl=0.4, order=2)
    solver.set_state(np.where(solver.x < STOKER_WET.x_dam,
                              STOKER_WET.h0, STOKER_WET.h1))
    solver.advance(t_end)
    return solver.x.copy(), solver.h.copy(), solver.time


def _figure(study: dict, path: Path) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sol = stoker_constant_state(STOKER_WET.h0, STOKER_WET.h1, STOKER_WET.gravity)
    figs = [(400, "tab:orange"), (1600, "tab:blue")]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ax = axes[0]
    for n, colour in figs:
        x, h, t = _profile(n)
        read = plateau_from_depth_modes(x, h)
        x_shock = STOKER_WET.x_dam + sol["shock_speed"] * t
        ax.plot(x, h, "-", lw=1.2, color=colour, label=f"FVM n = {n}")
        ax.axhline(read["h2"], color=colour, ls=":", lw=1.0, alpha=0.8)
        ax.plot([read["bore_x"]], [read["bore_level"]], "v", ms=7,
                color=colour, label=f"n = {n}: bore x = {read['bore_x']:.4f} m")
        ax.plot([x_shock], [sol["h2"]], "*", ms=9, color="k",
                label="exact bore x" if n == figs[0][0] else None)
    ax.axhline(sol["h1"], color="grey", ls="--", lw=1.0,
               label=f"exact h1 = {sol['h1']:.4f} m")
    ax.axhline(sol["h2"], color="k", ls="--", lw=1.0,
               label=f"exact h2 = {sol['h2']:.4f} m")
    ax.set_xlim(-1.5, 3.0)
    ax.set_ylim(0.45, 1.05)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("depth h [m]")
    ax.set_title("Stoker wet bed, t* = 2: depth modes = levels (oracle-free)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, loc="upper right")

    ax = axes[1]
    rows = study["rows"]
    dx = np.array([row["dx"] for row in rows])
    styles = {
        "depth_mode": ("depth mode (this work, no analytic input)", "o-"),
        "oracle_window": ("oracle window (F4, near-tautological)", "s--"),
        "gradient_walk": ("gradient walk (F9, cut retried)", "^:"),
    }
    for key, (label, style) in styles.items():
        h2 = [row[key]["h2_error"] for row in rows]
        ax.loglog(dx, h2, style, color="tab:blue", label=f"h2: {label}")
    bore = [row["depth_mode"]["bore_x_error"] for row in rows]
    ax.loglog(dx, bore, "d-", color="tab:red",
              label="bore position: depth mode")
    ax.axhline(TOLERANCE, color="k", ls=":", lw=1.0)
    ax.text(dx.max(), TOLERANCE * 1.15, " 5 % criterion", fontsize=8,
            ha="right")
    ax.axvline(1.0 / ACCEPTED_MIN_N * (STOKER_WET.x_max - STOKER_WET.x_min),
               color="grey", ls="-.", lw=0.8)
    ax.set_xlabel("dx [m]")
    ax.set_ylabel("relative error [-]")
    ax.set_title("Amplitude and position, three independent readings")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7.0, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolutions", default="200,400,800,1600,3200")
    parser.add_argument("--out", default=str(ARTIFACT))
    parser.add_argument("--figure", default=str(DOCS / "depth_modes.png"))
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args()
    resolutions = tuple(int(v) for v in args.resolutions.split(","))

    wall = time.perf_counter()
    study = depth_mode_study(resolutions=resolutions)
    wall = time.perf_counter() - wall
    study["meta"] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": ("PYTHONPATH=src python scripts/verify_dambreak_depth_modes.py "
                    f"--resolutions {args.resolutions}"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "wall_time_s": wall,
        "bins": PLATEAU_MODE_BINS,
        "mode_min_fraction": PLATEAU_MODE_MIN_FRACTION,
        "mode_tolerance": PLATEAU_MODE_TOL_FRACTION,
        "tolerance": TOLERANCE,
        "accepted_min_n": ACCEPTED_MIN_N,
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "depth_modes.json"
    path.write_text(json.dumps(study, indent=2) + "\n")
    print(f"wrote {path}")

    if not args.no_figure:
        print(f"wrote {_figure(study, Path(args.figure))}")

    print(f"{'n':>6} {'dx':>9} {'h2 err %':>10} {'bore err %':>11} "
          f"{'bore [cell]':>12} {'modes':>6} {'plateau cells':>14}  method")
    for row in study["rows"]:
        mode = row["depth_mode"]
        print(f"{row['n']:>6} {row['dx']:>9.6f} "
              f"{100.0 * mode['h2_error']:>10.4f} "
              f"{100.0 * mode['bore_x_error']:>11.4f} "
              f"{mode['bore_x_cells']:>12.3f} {mode['n_modes']:>6} "
              f"{mode['plateau_cells']:>14}  {mode['method']}")
    worst = study["max_error_depth_mode_accepted"]
    print("worst depth-mode error on n >= %d: h2 %.4f %%, step_ratio %.4f %%, "
          "bore %.4f %%"
          % (ACCEPTED_MIN_N, 100.0 * worst["h2_error"],
             100.0 * worst["step_ratio_error"],
             100.0 * worst["bore_x_error"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())