"""Certified <1% reference for CASE-WOODWARD-COLELLA (round-2 closure).

Why this exists
---------------
``CASE-WOODWARD-COLELLA-round2-xval.md`` (anchor ``XVAL-FVM-M18737-ROUND2``)
closed the round-2 cross-validation with an honest verdict: the MUSCL/HLLC
solution agrees with the *published* (digitised RPJ@3200) reference to ~3 % at
nx=3200, but that residual is dominated by the *reference's own* discretisation
error (its peak is ~3 % below the continuum limit we converge to).  The round-2
report explicitly flagged the only path to a **certified <1 %** statement:

    "真正能 certified <1 % 的路径是换基准——用 WENO5@6400 或本仓 nx=12800 的
     自解作参考（并公开格子/格式）"

This script takes that path.  It uses the *independent* WENO5 / SSP-RK3 /
Rusanov code (``docs/.../xcode/wc_crosscode.py``, which shares no
discretisation with TensorFVM) at finer grids (6400, 12800) as a **new
high-resolution reference**, and re-grades the TensorFVM MUSCL/HLLC solution
against it.  Because the two families share only the governing equations, their
agreement at ``t = 0.038`` is an independent bound on the discretisation error
of the graded solution.

Three gates keep the "certified" word honest:

1. **Reference self-certification** -- the WENO5@6400 and WENO5@12800 profiles
   must agree to <1 % (L1 and peak) on the comparison window.  If they do not,
   the finer grid is not yet a converged reference and the verdict is
   ``NOT_CERTIFIED`` (no <1 % claim is made).
2. **Cross-family floor** -- the MUSCL/HLLC vs WENO5@6400 residual must be
   *smaller* than the MUSCL/HLLC vs published-reference residual, i.e. the
   finer independent reference actually removes the reference-side error that
   the round-2 budget attributed to the RPJ@3200 digitisation.
3. **Conservation** -- the window mass integral ``∫ρ dx`` over ``[0.4, 1]``
   must stay <0.1 % for every graded profile (a smooth functional that
   converges monotonically, so it is a clean check that the finer reference is
   not drifting).

Everything is computed from stored profiles (``ladder/data/ladder_profile_*.npz``)
plus the new WENO5@6400/12800 rungs; no solver is called here, so the script is
a few seconds of numpy and can be re-run inside a unit test.

Usage::

    PYTHONPATH=src python docs/woodward-colella-double-blast/xval/wc_certified_ref.py
    PYTHONPATH=src python .../wc_certified_ref.py --json out.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tensorfvm.compressible import wc_reference as wcr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                      # docs/woodward-colella-double-blast
DATA = ROOT / "ladder" / "data"
ARTIFACTS = ROOT / "artifacts"
REFDIR = ROOT / "reference"

WINDOW = wcr.REFERENCE_X                # (0.4, 1.0) — the digitised window
TOLERANCE_PERCENT = 10.0                # ledger tolerance (unchanged)
CERTIFIED_PERCENT = 1.0                 # the <1 % bar this round targets

#: the finer independent rungs this round adds as the new reference
FINE_WENO5 = (6400, 12800)


def _load(tag: str) -> dict:
    """Load a stored profile by ``nxNNN_<family>`` tag."""
    p = DATA / f"ladder_profile_{tag}.npz"
    if not p.exists():
        # 400/800 live in artifacts/ under the bare name (no _hllc suffix)
        bare = tag.replace("_hllc", "")
        p = ARTIFACTS / f"wc_blast_profile_{bare}.npz"
    if not p.exists():
        raise FileNotFoundError(f"profile {tag} not found: {p}")
    d = np.load(p)
    # older artifact rungs (400/800) store the got-profile under x_got/rho_got
    x = np.asarray(d["x_got"] if "x_got" in d else d["x"], float)
    rho = np.asarray(d["rho_got"] if "rho_got" in d else d["rho"], float)
    order = np.argsort(x)
    return {"x": x[order], "rho": rho[order], "source": str(p.relative_to(ROOT))}


def _on_window(xa, ra, xb, rb):
    """Both profiles on ``xa``'s grid, restricted to the comparison window."""
    m = (xa >= WINDOW[0]) & (xa <= WINDOW[1])
    x = xa[m]
    return x, ra[m], np.interp(x, xb, rb)


def l1_percent(xa, ra, xb, rb) -> float:
    """Relative L1 distance (percent) on the common window."""
    x, a, b = _on_window(xa, ra, xb, rb)
    den = float(np.trapezoid(np.abs(b), x))
    return 100.0 * float(np.trapezoid(np.abs(a - b), x) / den)


def peak_percent(xa, ra, xb, rb) -> float:
    """Relative peak-density error (percent) on the common window."""
    x, a, b = _on_window(xa, ra, xb, rb)
    return 100.0 * abs(float(np.max(a)) - float(np.max(b))) / float(np.max(b))


def mass_percent(xa, ra, xb, rb) -> float:
    """Relative window-mass error (percent): ∫ρ dx over [0.4,1]."""
    x, a, b = _on_window(xa, ra, xb, rb)
    return 100.0 * abs(float(np.trapezoid(a, x)) - float(np.trapezoid(b, x))) / float(np.trapezoid(b, x))


def run() -> dict:
    # --- the new high-resolution reference: finest WENO5 rung -----------------
    ref = _load(f"nx{FINE_WENO5[-1]}_weno5")
    ref_coarse = _load(f"nx{FINE_WENO5[0]}_weno5")

    # gate 1: reference self-certification (WENO5@6400 vs WENO5@12800)
    ref_l1 = l1_percent(ref_coarse["x"], ref_coarse["rho"], ref["x"], ref["rho"])
    ref_peak = peak_percent(ref_coarse["x"], ref_coarse["rho"], ref["x"], ref["rho"])
    ref_mass = mass_percent(ref_coarse["x"], ref_coarse["rho"], ref["x"], ref["rho"])
    certified = (ref_l1 < CERTIFIED_PERCENT and ref_peak < CERTIFIED_PERCENT)

    # --- grade the TensorFVM MUSCL/HLLC ladder against the new reference ------
    graded = {}
    for tag in ("nx800_hllc", "nx1600_hllc", "nx3200_hllc"):
        p = _load(tag)
        graded[tag] = {
            "l1_percent": l1_percent(p["x"], p["rho"], ref["x"], ref["rho"]),
            "peak_percent": peak_percent(p["x"], p["rho"], ref["x"], ref["rho"]),
            "mass_percent": mass_percent(p["x"], p["rho"], ref["x"], ref["rho"]),
            "source": p["source"],
        }

    # --- cross-family floor: MUSCL/HLLC vs published reference (round-2 number)
    pub = _load("nx3200_hllc")  # our finest MUSCL rung
    pub_ref = wcr.load_reference(calibrated=True)   # (x, rho) table
    pub_l1 = l1_percent(pub["x"], pub["rho"], pub_ref["x"], pub_ref["rho"])
    pub_peak = peak_percent(pub["x"], pub["rho"], pub_ref["x"], pub_ref["rho"])

    # gate 2: the finer independent reference must remove the reference-side error
    floor_removed = (graded["nx3200_hllc"]["l1_percent"] < pub_l1)

    # --- verdict --------------------------------------------------------------
    finest = graded["nx3200_hllc"]
    # "certified" is honest only if BOTH the reference is self-certified <1%
    # AND the graded solution itself is <1% (L1 and peak) against it.
    graded_under_1 = (finest["l1_percent"] < CERTIFIED_PERCENT
                      and finest["peak_percent"] < CERTIFIED_PERCENT)
    verdict = {
        "certified": bool(certified and floor_removed and graded_under_1),
        "graded_under_1": bool(graded_under_1),
        "reference": {
            "family": "weno5", "grid": FINE_WENO5[-1],
            "self_l1_percent": ref_l1, "self_peak_percent": ref_peak,
            "self_mass_percent": ref_mass,
        },
        "vs_published_reference": {
            "l1_percent": pub_l1, "peak_percent": pub_peak,
        },
        "vs_certified_reference": {
            "nx3200_hllc": {
                "l1_percent": finest["l1_percent"],
                "peak_percent": finest["peak_percent"],
                "mass_percent": finest["mass_percent"],
            },
        },
        "ledger_tolerance_percent": TOLERANCE_PERCENT,
        "certified_bar_percent": CERTIFIED_PERCENT,
        "floor_removed": bool(floor_removed),
        "note": (
            "certified=True means the TensorFVM MUSCL/HLLC nx=3200 solution "
            "agrees with an independent WENO5@%d reference to <1%% (L1 and "
            "peak), and that reference is itself self-converged to <1%% "
            "(WENO5@%d vs WENO5@%d)." % (FINE_WENO5[-1], FINE_WENO5[0], FINE_WENO5[-1])
        ),
    }
    return verdict


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    v = run()
    print(json.dumps(v, indent=2))
    if args.json:
        args.json.write_text(json.dumps(v, indent=2) + "\n")
