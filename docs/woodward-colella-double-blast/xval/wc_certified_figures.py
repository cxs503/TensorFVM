"""Figures for the round-2 certified-<1% reference closure.

Generates the comparison plot for ``CASE-WOODWARD-COLELLA``: the TensorFVM
MUSCL/HLLC nx=3200 solution vs the *certified* independent WENO5@12800
reference (and vs the published digitised RPJ@3200 reference), plus the
reference self-convergence curve (WENO5@6400 vs WENO5@12800).

Requires the finer WENO5 rungs (``ladder/data/ladder_profile_nx{6400,12800}_weno5.npz``)
to exist (produced by ``wc_ladder_weno5.py --sizes 6400,12800``).

Usage::

    PYTHONPATH=src python docs/woodward-colella-double-blast/xval/wc_certified_figures.py \
        --outdir artifacts
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorfvm.compressible import wc_reference as wcr
from wc_certified_ref import _load, FINE_WENO5, WINDOW

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ARTIFACTS = ROOT / "artifacts"


def main(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    # --- the three profiles ------------------------------------------------
    muscl = _load("nx3200_hllc")                       # TensorFVM MUSCL/HLLC
    weno_fine = _load(f"nx{FINE_WENO5[-1]}_weno5")     # certified reference
    weno_coarse = _load(f"nx{FINE_WENO5[0]}_weno5")    # self-certification pair
    pub = wcr.load_reference(calibrated=True)          # published (digitised)

    # --- fig 1: MUSCL/HLLC vs certified WENO5 reference (full window) -------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    m = (muscl["x"] >= WINDOW[0]) & (muscl["x"] <= WINDOW[1])
    ax.plot(muscl["x"][m], muscl["rho"][m], lw=1.2, color="tab:blue",
            label=f"TensorFVM MUSCL/HLLC nx=3200")
    ax.plot(weno_fine["x"], weno_fine["rho"], lw=1.0, color="tab:red",
            label=f"independent WENO5/SSP-RK3/Rusanov nx={FINE_WENO5[-1]} (certified ref)")
    ax.plot(pub["x"], pub["rho"], lw=0.8, ls="--", color="k",
            label="published RPJ@3200 (digitised)")
    ax.set_xlabel("x"); ax.set_ylabel(r"$\rho$")
    ax.set_title("CASE-WOODWARD-COLELLA: MUSCL/HLLC vs certified WENO5 reference")
    ax.legend(fontsize=8)
    ax.set_xlim(WINDOW)
    fig.tight_layout()
    fig.savefig(outdir / "fig_certified_ref_profiles.png", dpi=150)
    plt.close(fig)

    # --- fig 2: reference self-convergence (WENO5@6400 vs WENO5@12800) ------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    m = (weno_coarse["x"] >= WINDOW[0]) & (weno_coarse["x"] <= WINDOW[1])
    ax.plot(weno_coarse["x"][m], weno_coarse["rho"][m], lw=1.0, color="tab:orange",
            label=f"WENO5 nx={FINE_WENO5[0]}")
    ax.plot(weno_fine["x"], weno_fine["rho"], lw=1.0, color="tab:red",
            label=f"WENO5 nx={FINE_WENO5[-1]}")
    # residual
    ax2 = ax.twinx()
    res = np.abs(np.interp(weno_fine["x"], weno_coarse["x"], weno_coarse["rho"])
                 - weno_fine["rho"])
    ax2.plot(weno_fine["x"], 100 * res / weno_fine["rho"], lw=0.8, color="tab:green",
             label="|Δρ|/ρ (%)")
    ax2.set_ylabel("|Δρ|/ρ (%)"); ax2.set_ylim(0, 2)
    ax.set_xlabel("x"); ax.set_ylabel(r"$\rho$")
    ax.set_title("WENO5 reference self-convergence (certification gate)")
    ax.legend(fontsize=8, loc="upper left"); ax2.legend(fontsize=8, loc="upper right")
    ax.set_xlim(WINDOW)
    fig.tight_layout()
    fig.savefig(outdir / "fig_certified_ref_selfconv.png", dpi=150)
    plt.close(fig)

    print(f"wrote {outdir/'fig_certified_ref_profiles.png'} and "
          f"{outdir/'fig_certified_ref_selfconv.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", type=Path, default=ARTIFACTS)
    args = ap.parse_args()
    main(args.outdir)
