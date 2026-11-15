"""Attribution of the round-2 certified-<1 % residual (anchor XVAL-FVM-M18737-ROUND2).

Why this file exists
-------------------
``CERTIFIED-REF-CLOSURE.md`` closed the round-2 question with an honest FAIL:
the TensorFVM MUSCL/HLLC nx=3200 solution sits **L1 = 1.776 %** away from the
independent WENO5@12800 reference (peak 0.758 %), so "certified <1 %" was not
claimed.  Two statements in that closure were asserted rather than measured:

* "the residual is dragged by the contact discontinuity / plateau region";
* "uniform refinement gains are already flat between 6400 and 12800".

The binding question for the *next* round is attribution: of the 1.776 %,
which part is still **reference-side** (the WENO5 reference's own error) and
which part is **graded-side** (our second-order MUSCL/HLLC error)?  If the
reference is still moving, the honest bar is "improve the reference"; if it is
not, the honest bar is "refine or re-scheme TensorFVM at nx=3200".

Three measurements answer it, all from stored profiles (no solver call):

1. **Reference floor.**  The audited WENO5 code is pushed one more rung
   (nx=25600, 2x finer than the 12800 reference).  ``WENO5@12800 vs
   WENO5@25600`` bounds what the reference itself still moves by.  This
   directly tests the "gains are flat" claim.
2. **Metric band.**  ``l1_percent`` interpolates the finer profile onto the
   coarser grid; that is *one* of two defensible conventions.  Both directions
   are reported, so the headline number carries an interpolation band instead
   of a false number of digits.
3. **Region attribution.**  The comparison window [0.4, 1] is split into
   shock / rarefied-plateau / smooth regions from the finest reference's own
   density gradient and pressure.  The graded-side error and the
   reference-side difference are integrated in the *same* regions, so the
   shares are comparable; a region where the graded-side error does not exceed
   the reference-side difference is a region where we are measuring the
   reference, not ourselves.

The graded-side share is additionally cross-checked against TensorFVM's own
ladder convergence (800/1600/3200), which is reference-free: if the measured
graded-side error is consistent with our own observed order, the attribution
is self-consistent.

Usage::

    PYTHONPATH=src python docs/woodward-colella-double-blast/xval/wc_ref_attribution.py
    PYTHONPATH=src python docs/woodward-colella-double-blast/xval/wc_ref_attribution.py --json out.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from tensorfvm.compressible import wc_reference as wcr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                      # docs/woodward-colella-double-blast
DATA = ROOT / "ladder" / "data"
ARTIFACTS = ROOT / "artifacts"

WINDOW = wcr.REFERENCE_X                # (0.4, 1.0)
CERTIFIED_PERCENT = 1.0
LEDGER_PERCENT = 10.0

#: finest independent reference available, and the rung used to measure its floor
REFERENCE_RUNG = 25600
REFERENCE_FALLBACK = 12800
FLOOR_PAIR = (12800, 6400)              # rungs whose difference bounds the floor

#: region thresholds, declared before the analysis
SHOCK_GRAD_FRAC = 0.20                  # |drho/dx| >= 20 % of the window maximum
PLATEAU_P_MAX = 0.10                    # rarefied plateau: p < 0.1 (initial p=0.01)


def _load(tag: str) -> Dict:
    """Load a stored profile (``ladder_profile_<tag>.npz`` or a legacy artifact)."""
    p = DATA / f"ladder_profile_{tag}.npz"
    if not p.exists():
        bare = tag.replace("_hllc", "")
        p = ARTIFACTS / f"wc_blast_profile_{bare}.npz"
    if not p.exists():
        raise FileNotFoundError(f"profile {tag} not found: {p}")
    d = np.load(p)
    x = np.asarray(d["x_got"] if "x_got" in d else d["x"], float)
    rho = np.asarray(d["rho_got"] if "rho_got" in d else d["rho"], float)
    pr = np.asarray(d["p"], float) if "p" in d else None
    o = np.argsort(x)
    return {"x": x[o], "rho": rho[o], "p": None if pr is None else pr[o],
            "source": str(p.relative_to(ROOT))}


def _on(x_nodes, xa, ra, xb, rb) -> Tuple[np.ndarray, np.ndarray]:
    """Both profiles sampled at ``x_nodes`` (linear interpolation)."""
    return np.interp(x_nodes, xa, ra), np.interp(x_nodes, xb, rb)


def l1_band(pa: Dict, pb: Dict) -> Dict:
    """Relative L1 (%) of two profiles on the window, in *both* conventions.

    ``graded_on_ref`` samples at the finer grid (``pb``), ``ref_on_graded`` at the
    coarser one (``pa``).  The two numbers are the metric's interpolation band.
    """
    lo, hi = WINDOW
    x_fine = pb["x"][(pb["x"] >= lo) & (pb["x"] <= hi)]
    x_coarse = pa["x"][(pa["x"] >= lo) & (pa["x"] <= hi)]
    out = {}
    for tag, xn in (("graded_on_ref", x_fine), ("ref_on_graded", x_coarse)):
        a, b = _on(xn, pa["x"], pa["rho"], pb["x"], pb["rho"])
        den = float(np.trapezoid(np.abs(b), xn))
        out[tag] = 100.0 * float(np.trapezoid(np.abs(a - b), xn)) / den
    out["l1_percent"] = float(np.mean([out["graded_on_ref"], out["ref_on_graded"]]))
    out["l1_band"] = [float(min(out["graded_on_ref"], out["ref_on_graded"])),
                      float(max(out["graded_on_ref"], out["ref_on_graded"]))]
    # peak + window mass at the reference grid (smooth functionals, one convention)
    a, b = _on(x_fine, pa["x"], pa["rho"], pb["x"], pb["rho"])
    out["peak_percent"] = 100.0 * abs(float(a.max()) - float(b.max())) / float(b.max())
    out["mass_percent"] = 100.0 * abs(float(np.trapezoid(a, x_fine)) -
                                      float(np.trapezoid(b, x_fine))) / float(
        np.trapezoid(b, x_fine))
    return out


def regions(ref: Dict) -> Dict[str, np.ndarray]:
    """Split the window of the finest reference into shock / plateau / smooth.

    Masks are boolean arrays on ``ref``'s own grid, so every per-region integral
    below is taken on the same grid and the shares are comparable.
    """
    lo, hi = WINDOW
    x, rho, p = ref["x"], ref["rho"], ref["p"]
    m = (x >= lo) & (x <= hi)
    x, rho, p = x[m], rho[m], p[m]
    g = np.abs(np.gradient(rho, x))
    shock = g >= SHOCK_GRAD_FRAC * float(g.max())
    plateau = (p < PLATEAU_P_MAX) & ~shock if p is not None else np.zeros_like(shock)
    smooth = ~(shock | plateau)
    return {"x": x, "shock": shock, "plateau": plateau, "smooth": smooth}


def region_shares(pa: Dict, pb: Dict, reg: Dict) -> Dict:
    """Per-region share of ``int |drho| dx`` and its relative-to-reference size."""
    x = reg["x"]
    a, b = _on(x, pa["x"], pa["rho"], pb["x"], pb["rho"])
    d = np.abs(a - b)
    den_total = float(np.trapezoid(np.abs(b), x))
    tot = float(np.trapezoid(d, x))
    out = {}
    for name in ("shock", "plateau", "smooth"):
        m = reg[name]
        val = float(np.trapezoid(np.where(m, d, 0.0), x))
        out[name] = {"l1_percent_of_ref": 100.0 * val / den_total,
                     "share_of_residual": (100.0 * val / tot) if tot > 0 else 0.0,
                     "window_fraction": float(np.trapezoid(m.astype(float), x) /
                                              (x[-1] - x[0]))}
    out["total_l1_percent_of_ref"] = 100.0 * tot / den_total
    return out


def _order(e_coarse: float, e_fine: float, r: float = 2.0) -> float:
    return float(np.log(e_coarse / e_fine) / np.log(r))


def run() -> Dict:
    have_tight = (DATA / f"ladder_profile_nx{REFERENCE_RUNG}_weno5.npz").exists()
    ref_nx = REFERENCE_RUNG if have_tight else REFERENCE_FALLBACK
    ref = _load(f"nx{ref_nx}_weno5")
    flo_pair = (_load(f"nx{FLOOR_PAIR[0]}_weno5"), _load(f"nx{FLOOR_PAIR[1]}_weno5"))
    reg = regions(ref)

    # --- 1. reference floor: how much does the reference family still move? ----
    floor_now = l1_band(flo_pair[0], flo_pair[1])          # 12800 vs 6400
    floor_tight = l1_band(_load(f"nx12800_weno5"), ref) if have_tight else None
    e_6400, e_12800 = floor_now["l1_percent"], (floor_tight["l1_percent"]
                                                if floor_tight else float("nan"))
    ref_order = _order(e_6400, e_12800) if have_tight else float("nan")
    # Richardson estimate of the reference's own error at its finest rung
    ref_self_error = (e_12800 / (2 ** ref_order - 1)) if have_tight else float("nan")

    # --- 2. grade the TensorFVM ladder against the tightest reference ----------
    graded = {}
    for nx in (800, 1600, 3200):
        graded[f"nx{nx}_hllc"] = l1_band(_load(f"nx{nx}_hllc"), ref)
    e_grade = graded["nx3200_hllc"]["l1_percent"]
    band = graded["nx3200_hllc"]["l1_band"]

    # --- 3. region attribution (graded side vs reference side) ----------------
    graded_regions = region_shares(_load("nx3200_hllc"), ref, reg)
    ref_regions = (region_shares(_load("nx12800_weno5"), ref, reg)
                   if have_tight else None)

    # --- 4. reference-free cross-check: TensorFVM's own ladder order -----------
    fvm_self = {}
    for lo, hi in ((800, 1600), (1600, 3200)):
        d = l1_band(_load(f"nx{lo}_hllc"), _load(f"nx{hi}_hllc"))
        fvm_self[f"{lo}->{hi}"] = d["l1_percent"]
    e_self = fvm_self["1600->3200"]
    p_fvm = _order(fvm_self["800->1600"], e_self) if e_self > 0 else float("nan")
    # own discretisation error at nx=3200 implied by that order
    own_error = e_self / (2 ** p_fvm - 1) if np.isfinite(p_fvm) else float("nan")

    graded_side_low = max(e_grade - (floor_tight["l1_percent"] if floor_tight else 0.0), 0.0)
    verdict = {
        "reference_rung": ref_nx,
        "reference_is_tightened": bool(have_tight),
        "graded_solution": "TensorFVM MUSCL-Hancock/MC + HLLC, nx=3200",
        "l1_percent": e_grade,
        "l1_band_percent": band,
        "l1_within_metric_band_under_1": bool(band[1] < CERTIFIED_PERCENT),
        "peak_percent": graded["nx3200_hllc"]["peak_percent"],
        "mass_percent": graded["nx3200_hllc"]["mass_percent"],
        "certified_under_1_percent": bool(e_grade < CERTIFIED_PERCENT),
        "ledger_under_10_percent": bool(e_grade < LEDGER_PERCENT),
        "reference_floor": {
            "pair_12800_vs_6400_l1_percent": floor_now["l1_percent"],
            "tight_pair_l1_percent": None if not floor_tight else floor_tight["l1_percent"],
            "observed_order": ref_order,
            "implied_self_error_of_finest_rung_percent": ref_self_error,
        },
        "attribution": {
            "graded_side_bracket_percent": [graded_side_low, e_grade],
            "reference_side_upper_bound_percent": (floor_tight["l1_percent"]
                                                  if floor_tight else floor_now["l1_percent"]),
            "graded_side_dominates": bool(graded_side_low > 0.5 * e_grade),
            "own_discretisation_error_percent": own_error,
            "own_order": p_fvm,
            "own_ladder_self_l1_percent": fvm_self,
            "consistent_with_own_error": bool(
                np.isfinite(own_error) and
                abs(own_error - graded_side_low) <= 0.5 * max(graded_side_low, 1e-12)),
        },
        "regions": graded_regions,
        "reference_side_regions": ref_regions,
        "predicted_nx_for_under_1_percent": (
            int(round(3200.0 * (e_grade / CERTIFIED_PERCENT) ** (1.0 / p_fvm)))
            if np.isfinite(p_fvm) and p_fvm > 0 else None),
    }
    return verdict


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    v = run()
    print(json.dumps(v, indent=2, default=float))
    if args.json:
        args.json.write_text(json.dumps(v, indent=2, default=float) + "\n")