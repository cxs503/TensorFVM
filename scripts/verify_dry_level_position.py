#!/usr/bin/env python
"""Oracle-free dry-bed **position** verification for ``CASE-FVM-DAMBREAK-STOKER``.

Why this exists
---------------
The dry-bed headline metric of the dam-break benchmark is ``front_fan``, built
on the *exact similarity identity* ``x_f = x_i + 3 t sqrt(g h_i)``.  It is a good
test of the fan **shape** (0.017 % at ``n = 1600``) but it locates the front from
the closed form, so it cannot by itself establish that the numerical profile sits
at the right **position** -- and the ledger acceptance for
``CASE-FVM-DAMBREAK-STOKER`` is a position statement (``x_f(t)`` vs analytic,
``< 5 %``).  The only position-only definitions in the module were:

* the wetted edge -- a depth-floor detector whose reading is set by the
  numerical floor, not by ``dx`` (measured non-convergent: 1.56 % -> 2.66 % ->
  2.89 % -> 2.46 % for ``n = 400 ... 3200``);
* ``level_scaled`` -- which adds the exact fan offset, i.e. imports the answer.

This script supplies the missing evidence: the sub-cell abscissa where the
numerical depth falls through ``level * h0`` is graded against its own closed
form ``x_dam + c0 t (2 - 3 sqrt(level))``
(:func:`tensorfvm.shallow_water.exact.ritter_level_position`).  The measurement
uses the profile and ``level`` only; the reference is a *different* closed-form
point than the tip, so agreement is not a restatement of ``x_f``.

Usage
-----
    PYTHONPATH=src python scripts/verify_dry_level_position.py
    PYTHONPATH=src python scripts/verify_dry_level_position.py --no-figure
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys

import numpy as np

#: declared before any run: the grids of the published study
RESOLUTIONS = (100, 200, 400, 800, 1600, 3200)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
OUT_DIR = os.path.join(REPO_ROOT, "src", "tensorfvm", "verification",
                       "artifacts", "shallow-water-dam-break-level-position")


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=REPO_ROOT, text=True).strip()
    except Exception:                                    # pragma: no cover
        return "unavailable"


def _figure(study: dict, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from tensorfvm.shallow_water.dam_break import (LEVEL_POSITION_SERIES_LEVEL,
                                                   RITTER_DRY, _level_series)
    from tensorfvm.shallow_water.exact import (ritter_level_position,
                                               ritter_state)

    rows = study["rows"]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))

    # (a) contour error vs dx, log-log, with the first-order reference slope
    ax = axes[0]
    dx = np.array([r["dx"] for r in rows])
    for level in study["levels"]:
        key = f"{float(level):g}"
        err = np.array([max(r["levels"][key]["error"], 1e-12) for r in rows])
        ax.loglog(dx, err, "o-", label=f"h/h0 = {key}")
    ref = 0.5 * dx / dx[0]                      # unit-slope guide
    err0 = max(rows[0]["levels"]["0.25"]["error"], 1e-12)
    ax.loglog(dx, err0 * ref, "k--", lw=1.0, label="first order (p = 1)")
    ax.axhline(study["tolerance"], color="crimson", ls=":", lw=1.5,
               label=f"gate {study['tolerance']:.0%}")
    ax.set_xlabel("dx [m]")
    ax.set_ylabel(r"$|x_{measured}-x_{exact}|/x_{exact}$")
    ax.set_title("(a) contour position error vs dx")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, ncol=2)

    # (b) the profile at t* = 2 with the declared contours
    ax = axes[1]
    n = 1600
    solver, _ = _level_series(case=RITTER_DRY, n=n, cfl=study["cfl"],
                              order=study["order"],
                              level=LEVEL_POSITION_SERIES_LEVEL)
    h_ex, _ = ritter_state(solver.x, solver.time, RITTER_DRY.h0,
                           RITTER_DRY.gravity, RITTER_DRY.x_dam)
    ax.plot(solver.x, solver.h / RITTER_DRY.h0, "b-", lw=1.6,
            label=f"numerical (n = {n})")
    ax.plot(solver.x, np.asarray(h_ex) / RITTER_DRY.h0, "k--", lw=1.2,
            label="Ritter exact")
    for level in study["levels"]:
        xl = float(level)
        ax.axhline(xl, color="grey", lw=0.5, alpha=0.6)
        ax.axvline(ritter_level_position(xl, solver.time, RITTER_DRY.h0,
                                         RITTER_DRY.gravity, RITTER_DRY.x_dam),
                   color="orange", lw=0.9, ls=":")
    ax.set_xlim(RITTER_DRY.x_dam - 1.0, RITTER_DRY.x_dam + 6.5)
    ax.set_ylim(0.0, 1.15)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("h / h0")
    ax.set_title(r"(b) profile at $t^*=2$; orange = exact contour positions")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # (c) the oracle-free x(t) series against the closed form
    ax = axes[2]
    for r in rows:
        if r["n"] not in (200, 800, 3200):
            continue
        s = [x for x in r["series"] if x["resolved"]]
        ax.plot([x["t_star"] for x in s], [x["measured"] for x in s], "-",
                lw=1.3, label=f"n = {r['n']}")
    s0 = [x for x in rows[-1]["series"]]
    ax.plot([x["t_star"] for x in s0], [x["exact"] for x in s0], "k--", lw=1.6,
            label="exact contour")
    ax.set_xlabel(r"$t^* = t\sqrt{g/h_0}$")
    ax.set_ylabel(r"$x_{contour}$ [m]")
    ax.set_title(r"(c) $h/h_0=0.04$ contour position vs time")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-figure", action="store_true")
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()

    from tensorfvm.shallow_water.dam_break import level_position_study

    study = level_position_study(RESOLUTIONS)
    os.makedirs(args.out, exist_ok=True)
    payload = {
        "meta": {
            "git_commit": _git_commit(),
            "command": "PYTHONPATH=src python scripts/verify_dry_level_position.py",
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "study": study,
    }
    json_path = os.path.join(args.out, "study.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)

    if not args.no_figure:
        _figure(study, os.path.join(args.out, "figure_level_position.png"))

    print(f"wrote {json_path}")
    print(f"  max contour error at n >= {study['accepted_min_n']}: "
          f"{study['max_error_at_accepted_grid']:.4%}")
    print(f"  max series  error at n >= {study['accepted_min_n']}: "
          f"{study['max_series_error_at_accepted_grid']:.4%}")
    print(f"  min observed order (contour): "
          f"{study['min_observed_order_at_accepted_grid']:.3f}")
    ok = (study["max_error_at_accepted_grid"] <= study["tolerance"]
          and study["max_series_error_at_accepted_grid"] <= study["tolerance"])
    print("GATE", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())