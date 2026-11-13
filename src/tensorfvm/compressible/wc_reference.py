"""Quantitative check of the Woodward-Colella double blast against literature.

The reference profile is the "Reference" curve of Figure 21 of Dumbser et al.
(J. Comput. Phys. 2019; arXiv:1902.07773), which that paper's Sec. 5.1.4
identifies as the pure RPJ solution on 3200 cells at ``T = 0.038``.  It is
recovered from the PDF's vector paths by
``docs/woodward-colella-double-blast/reference/digitize_reference.py`` and
stored as ``reference_density.csv``; the two independent digitisation passes
(400-cell and 800-cell panels) agree to 3e-5 in density, which is the
digitisation error budget.

This module runs the local case and reports, per metric, the deviation from the
published reference plus a pass/fail against the ledger tolerance (<10%).

Metrics (all on the published window ``x in [0.4, 1]``):

* ``peak_density``      — maximum density and its relative error,
* ``peak_position``     — x of that maximum, absolute error,
* ``l1_relative``       — ``∫|ρ − ρ_ref| dx / ∫ρ_ref dx`` (resolution-honest
  integral norm; pointwise values at shocks are smeared over a few cells),
* ``low_plateau``       — the shallow near-vacuum minimum, measured on the
  interior ``x in [0.42, 0.98]`` so the window edge cannot fake a minimum,
  reported both as a relative error and as a fraction of the peak density,
* ``faces``             — the four steepest density faces, absolute x error.

CLI: ``python -m tensorfvm.compressible.wc_reference --nx 800``
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .cases import DoubleBlastConfig
from .double_blast import _run

TOLERANCE_PERCENT = 10.0
REFERENCE_X = (0.4, 1.0)
LOW_PLATEAU_X = (0.42, 0.98)
POSITION_TOLERANCE = 0.02
FACE_MIN_SEPARATION = 0.02
N_FACES = 4

DEFAULT_REFERENCE = (Path(__file__).resolve().parents[3] / "docs"
                     / "woodward-colella-double-blast" / "reference"
                     / "reference_density.csv")


def load_reference(path: str | Path | None = None) -> dict:
    """Read the digitised published reference (mean of the two panels)."""
    path = Path(path or DEFAULT_REFERENCE)
    raw = np.loadtxt(path, delimiter=",", skiprows=1)
    x, a, b = raw[:, 0], raw[:, 1], raw[:, 2]
    return {"path": str(path), "x": x, "rho": 0.5 * (a + b),
            "panel_spread": float(np.max(np.abs(a - b)))}


def profile(nx: int, t_end: float | None = None) -> dict:
    """Cell-centred density at the final time of the 1-D case."""
    run = _run(nx, t_end if t_end is not None else DoubleBlastConfig().t_end)
    x = np.asarray(run["x"]).ravel()
    rho = np.asarray(run["rho"]).ravel()
    order = np.argsort(x)
    return {"x": x[order], "rho": rho[order], "run": run}


def faces(x, rho) -> list[float]:
    """x of the steepest density faces (shock/contact), far enough apart."""
    d = np.abs(np.gradient(rho, x))
    idx = [i for i in range(1, d.size - 1) if d[i] > d[i - 1] and d[i] >= d[i + 1]]
    idx.sort(key=lambda i: -d[i])
    keep: list[int] = []
    for i in idx:
        if all(abs(x[i] - x[j]) > FACE_MIN_SEPARATION for j in keep):
            keep.append(i)
        if len(keep) == N_FACES:
            break
    return sorted(float(x[i]) for i in keep)


def metrics_from_profiles(xr, rr, xm, rm) -> dict:
    """All comparison metrics for one grid pair (no solver call)."""
    lo, hi = REFERENCE_X
    sr = (xr >= lo) & (xr <= hi)
    xr, rr = xr[sr], rr[sr]
    sm = (xm >= lo) & (xm <= hi)
    xm, rm = xm[sm], rm[sm]

    rho_ref_on_mine = np.interp(xm, xr, rr)
    l1 = float(np.trapezoid(np.abs(rm - rho_ref_on_mine), xm) / np.trapezoid(rr, xr))
    peak_ref, peak = float(rr.max()), float(rm.max())
    peak_ref_x = float(xr[int(rr.argmax())])
    peak_x = float(xm[int(rm.argmax())])

    plo, phi = LOW_PLATEAU_X
    low_ref = float(rr[(xr >= plo) & (xr <= phi)].min())
    low = float(rm[(xm >= plo) & (xm <= phi)].min())

    fr, fm = faces(xr, rr), faces(xm, rm)
    face_err = [abs(a - b) for a, b in zip(fm, fr)] if len(fm) == len(fr) else None

    out = {
        "window_x": [lo, hi],
        "peak_density": {"ref": peak_ref, "got": peak,
                         "rel_error_percent": abs(peak - peak_ref) / peak_ref * 100},
        "peak_position": {"ref": peak_ref_x, "got": peak_x,
                          "abs_error": abs(peak_x - peak_ref_x)},
        "l1_relative": {"value": l1, "percent": 100 * l1},
        "low_plateau": {"ref": low_ref, "got": low,
                        "abs_error": abs(low - low_ref),
                        "rel_error_percent": abs(low - low_ref) / low_ref * 100,
                        "percent_of_peak_density": abs(low - low_ref) / peak_ref * 100},
        "faces_ref": fr, "faces": fm, "faces_abs_error": face_err,
    }
    checks = {
        "peak_density_lt_10pct": out["peak_density"]["rel_error_percent"] < TOLERANCE_PERCENT,
        "peak_position_lt_tol": out["peak_position"]["abs_error"] < POSITION_TOLERANCE,
        "l1_lt_10pct": out["l1_relative"]["percent"] < TOLERANCE_PERCENT,
        "faces_lt_tol": bool(face_err and max(face_err) < POSITION_TOLERANCE),
        "low_plateau_lt_10pct": out["low_plateau"]["rel_error_percent"] < TOLERANCE_PERCENT,
    }
    out["checks"] = checks
    out["verdict"] = "PASS" if all(checks.values()) else "FAIL"
    out["tolerance_percent"] = TOLERANCE_PERCENT
    out["position_tolerance"] = POSITION_TOLERANCE
    return out


def metrics(nx: int = 800, reference: dict | None = None, t_end: float | None = None) -> dict:
    ref = reference or load_reference()
    mine = profile(nx, t_end)
    out = metrics_from_profiles(ref["x"], ref["rho"], mine["x"], mine["rho"])
    out.update({"nx": nx, "dx": 1.0 / nx, "reference": ref["path"],
                "reference_source": "Dumbser et al. 2019 Fig. 21 'Reference' "
                                    "(pure RPJ, 3200 cells), digitised from vector paths",
                "reference_panel_spread": ref["panel_spread"]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, nargs="+", default=[800])
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    ref = load_reference()
    rows = []
    for nx in args.nx:
        m = metrics(nx, reference=ref)
        rows.append(m)
        print(json.dumps(m, indent=2))
    print("\nnx    peak_rel%  peak_x_err   L1%   low_plateau%  verdict")
    for m in rows:
        print(f"{m['nx']:<5d} {m['peak_density']['rel_error_percent']:8.3f} "
              f"{m['peak_position']['abs_error']:11.4f} "
              f"{m['l1_relative']['percent']:6.3f} "
              f"{m['low_plateau']['rel_error_percent']:12.3f}  {m['verdict']}")
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()