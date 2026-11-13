"""Sub-cell wet-bed bore position: refinement + detectability study (F7).

Background
----------
``docs/case-fvm-dambreak-stoker/AUDIT.md`` finding F4 recorded that the
oracle-guided wet-bed metrics are insensitive to a misplaced bore, and the fix
(``data_driven_plateau``) anchored the measurement window to the *measured*
bore.  That fix read the bore position as the **cell centre of the maximum
gradient**, which has the same defect class as the dry-bed wetted edge (F1): it
is quantised to ``dx``, so

* its refinement error is non-monotone with an ``O(dx/2)`` floor
  (0.73 % -> 0.068 % -> 0.399 % -> 0.233 % for n = 200 ... 1600), and
* it cannot express a sub-cell displacement at all -- the property a metric
  needs if it is to be a *position* check rather than a coarse alarm.

This script measures the fix (``bore_x`` = crossing of the mid level between the
two measured states, linearly interpolated) against that legacy reading on three
independent probes:

1. **refinement**   -- the sub-cell error must stay below
   ``BORE_SUBCELL_MAX_CELLS`` (half a cell) at *every* rung, while the legacy
   reading sits at the quantisation floor;
2. **translation**  -- a synthetic fractional-cell translation of the depth
   profile (amplitude untouched) must be recovered to a small fraction of a
   cell, proving the reading is a position measurement with sub-cell authority;
3. **amplitude mutation** -- uniformly scaling the plateau (position untouched)
   must move ``h2``/% while leaving ``bore_x`` alone, proving that the
   ``h2``/``step_ratio`` pair and ``bore_x`` are *independent* evidence
   (they are near-tautologies only when both are pinned to analytic positions).

Usage::

    PYTHONPATH=src python scripts/verify_dambreak_bore_subcell.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import json

import numpy as np

from tensorfvm.shallow_water.dam_break import (
    BORE_SUBCELL_MAX_CELLS,
    STOKER_WET,
    data_driven_plateau,
    run_case,
)
from tensorfvm.shallow_water.exact import stoker_constant_state

RESOLUTIONS = (200, 400, 800, 1600)
TRANSLATIONS = tuple(round(0.1 * k, 1) for k in range(1, 10))   # in units of dx
AMPLITUDE_SCALE = 1.5e-2                                        # +1.5 % plateau
ARTIFACT_DIR = (Path(__file__).resolve().parents[1]
                / "src/tensorfvm/verification/artifacts/shallow-water-dam-break")
DOCS_DIR = Path(__file__).resolve().parents[1] / "docs/case-fvm-dambreak-stoker"


def _delta(raw: float, exact_val: float) -> float:
    return abs(raw - exact_val) / abs(exact_val) if exact_val else abs(raw)


def refinement() -> dict:
    sol = stoker_constant_state(STOKER_WET.h0, STOKER_WET.h1, STOKER_WET.gravity)
    x_exact = STOKER_WET.exact_front()
    rows = []
    for n in RESOLUTIONS:
        run = run_case(STOKER_WET, n=n)
        dx = run["discretisation"]["dx"]
        dd = run["metrics"]["dd"]
        rows.append({
            "n": n,
            "dx": dx,
            "bore_x_exact": x_exact,
            "bore_x": run["metrics"]["dd"]["bore_x"],
            "bore_x_cell": dd["bore_x_cell"],
            "bore_method": dd["bore_method"],
            "level": dd["bore_level"],
            "error": _delta(dd["bore_x"], x_exact),
            "error_cells": _delta(dd["bore_x"], x_exact) * x_exact / dx,
            "error_cell_reading": _delta(dd["bore_x_cell"], x_exact),
            "error_cell_reading_cells": _delta(dd["bore_x_cell"], x_exact) * x_exact / dx,
            "h2_error": dd["h2_error"],
            "step_ratio_error": dd["step_ratio_error"],
            "n_plateau": dd["n_plateau"],
            "shock_error": run["metrics"]["shock_error"],
            "verdict_all_pass": run["verdict"]["all_pass"],
            "verdict_failing": [key for key, item in run["verdict"].items()
                                if key != "all_pass" and not item["pass"]],
        })
    worst = max(row["error_cells"] for row in rows)
    worst_cell = max(row["error_cell_reading_cells"] for row in rows)
    return {
        "x_shock_exact": x_exact,
        "stoker": dict(sol),
        "rows": rows,
        "subcell_worst_cells": worst,
        "cell_reading_worst_cells": worst_cell,
        "limit_cells": BORE_SUBCELL_MAX_CELLS,
        "pass": bool(worst <= BORE_SUBCELL_MAX_CELLS),
        "improvement_factor": (worst_cell / worst) if worst else float("inf"),
    }


def translation(n: int = 800) -> dict:
    """Fractional-cell translation of the profile: same amplitude, new place."""
    run = run_case(STOKER_WET, n=n)
    x = np.asarray(run["profile"]["x"], dtype=float)
    h = np.asarray(run["profile"]["h_numeric"], dtype=float)
    dx = x[1] - x[0]
    base = data_driven_plateau(x, h)
    rows = []
    for frac in TRANSLATIONS:
        delta = frac * dx
        shifted = np.interp(x - delta, x, h)     # translate the profile right
        dd = data_driven_plateau(x, shifted)
        rows.append({
            "translation_cells": frac,
            "translation_m": delta,
            "residual_subcell_cells": (dd["bore_x"] - base["bore_x"] - delta) / dx,
            "residual_cell_reading_cells": (dd["bore_x_cell"] - base["bore_x_cell"] - delta) / dx,
            "h2_change": _delta(dd["h2"], base["h2"]),
        })
    worst_sub = max(abs(row["residual_subcell_cells"]) for row in rows)
    worst_cell = max(abs(row["residual_cell_reading_cells"]) for row in rows)
    return {
        "n": n,
        "dx": dx,
        "base": {key: base[key] for key in
                 ("bore_x", "bore_x_cell", "bore_level", "bore_method", "h2", "h1")},
        "rows": rows,
        "worst_subcell_residual_cells": worst_sub,
        "worst_cell_residual_cells": worst_cell,
        "pass": bool(worst_sub < 0.1),
    }


def amplitude_mutation(n: int = 800) -> dict:
    """Scale the plateau uniformly: amplitude changes, the bore must not move."""
    run = run_case(STOKER_WET, n=n)
    x = np.asarray(run["profile"]["x"], dtype=float)
    h = np.asarray(run["profile"]["h_numeric"], dtype=float)
    dx = x[1] - x[0]
    base = data_driven_plateau(x, h)
    # plateau span measured (no analytic position): [plateau left edge, bore]
    plateau = (x > base["bore_x"] - 0.8 * (base["bore_x"] - x.min() + dx)) & (x < base["bore_x"])
    mutated = h.copy()
    mutated[plateau] *= (1.0 + AMPLITUDE_SCALE)
    dd = data_driven_plateau(x, mutated)
    return {
        "n": n,
        "amplitude_scale": AMPLITUDE_SCALE,
        "cells_mutated": int(np.count_nonzero(plateau)),
        "h2_change": _delta(dd["h2"], base["h2"]),
        "bore_shift_cells": (dd["bore_x"] - base["bore_x"]) / dx,
        "pass": bool(_delta(dd["h2"], base["h2"]) > 1e-4
                     and abs(dd["bore_x"] - base["bore_x"]) / dx < 0.05),
    }


def figure(study: dict, trans: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    ax = axes[0]
    dx = [row["dx"] for row in study["rows"]]
    ax.loglog(dx, [row["error_cells"] for row in study["rows"]], "o-",
              label="sub-cell (mid-level crossing, F7 fix)")
    ax.loglog(dx, [row["error_cell_reading_cells"] for row in study["rows"]], "s--",
              label="legacy cell-centre (max-gradient cell)")
    ax.axhline(study["limit_cells"], color="k", ls=":", lw=1.0,
               label=f"predeclared limit = {study['limit_cells']:g} cell")
    ax.set_xlabel("dx [m]")
    ax.set_ylabel("bore-position error [cells]")
    ax.set_title("Stoker bore position: refinement (t* = 2)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7.5)

    ax = axes[1]
    frac = [row["translation_cells"] for row in trans["rows"]]
    ax.plot(frac, [row["residual_subcell_cells"] for row in trans["rows"]], "o-",
            label="sub-cell reading")
    ax.plot(frac, [row["residual_cell_reading_cells"] for row in trans["rows"]], "s--",
            label="legacy cell-centre reading")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel("imposed translation [cells]")
    ax.set_ylabel("recovered - imposed [cells]")
    ax.set_title(f"Detectability of a sub-cell bore displacement (n = {trans['n']})")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    study = refinement()
    trans = translation()
    amp = amplitude_mutation()
    payload = {
        "case": STOKER_WET.summary(),
        "criterion": {
            "subcell_max_cells": BORE_SUBCELL_MAX_CELLS,
            "declared": "bore position must resolve below half a cell at every rung",
            "translation_tolerance_cells": 0.1,
            "amplitude_scale": AMPLITUDE_SCALE,
        },
        "refinement": study,
        "translation": trans,
        "amplitude_mutation": amp,
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    target = ARTIFACT_DIR / "bore_subcell.json"
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    fig_path = DOCS_DIR / "bore_subcell.png"
    figure(study, trans, fig_path)

    print(f"refinement  pass={study['pass']}  worst sub-cell={study['subcell_worst_cells']:.3f} cells"
          f"  worst cell-centre={study['cell_reading_worst_cells']:.3f} cells"
          f"  improvement {study['improvement_factor']:.1f}x")
    print(f"translation pass={trans['pass']}  worst residual={trans['worst_subcell_residual_cells']:.4f} cells"
          f"  (cell-centre {trans['worst_cell_residual_cells']:.3f} cells)")
    print(f"amplitude   pass={amp['pass']}  h2 change={amp['h2_change']:.2e}"
          f"  bore shift={amp['bore_shift_cells']:.4f} cells")
    print(f"wrote {target}")
    print(f"wrote {fig_path}")
    return 0 if (study["pass"] and trans["pass"] and amp["pass"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())