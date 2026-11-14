"""Figures for the Woodward-Colella round-2 cross-validation.

Reads ``xval_round2.json`` (produced by ``wc_xval.py``) plus the stored profiles
and renders three panels sets used by
``docs/CASE-WOODWARD-COLELLA-round2-xval.md``:

* ``fig_xval_ladder.png``   -- peak / L1 error against the reference vs grid, and
  the reference-free grid-to-grid (self) differences.
* ``fig_xval_residual.png`` -- where the residual lives: profiles at the finest
  grid against the calibrated reference (whole window + peak zoom + the
  x~0.76 discontinuity), plus the band decomposition of the L1 integral.
* ``fig_xval_floor.png``    -- peak-amplitude extrapolation to the continuum and
  the resolved-gradient comparison (our solution gets sharper than the
  digitised reference), plus the smoothing-floor scan.

    PYTHONPATH=src python docs/woodward-colella-double-blast/xval/wc_xval_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorfvm.compressible import wc_reference as wcr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIG = ROOT / "artifacts"


def load() -> tuple[dict, dict]:
    a = json.loads((HERE / "xval_round2.json").read_text())
    raw = wcr.load_reference()
    cal = wcr.load_reference(ROOT / "reference" / "reference_density_calibrated.csv")
    return a, {"raw": raw, "cal": cal}


def profiles(a: dict) -> dict:
    import sys
    sys.path.insert(0, str(HERE))
    from wc_xval import stored_profiles
    return stored_profiles()


def fig_ladder(a: dict) -> Path:
    fvm = [(r["nx"], r["peak_percent_calibrated"], r["l1_percent_calibrated"])
           for r in a["profiles"].values() if r["family"] == "fvm"]
    fvm.sort()
    nx = np.array([r[0] for r in fvm], float)
    peak = np.array([r[1] for r in fvm])
    l1 = np.array([r[2] for r in fvm])

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].loglog(nx, peak, "o-", label="peak density |rel err|")
    ax[0].loglog(nx, l1, "s-", label="profile L1")
    ax[0].axhline(3.0, color="0.4", ls="--", lw=1, label="3 % (earlier claim)")
    ax[0].axhline(a["ledger_tolerance_percent"], color="crimson", ls=":",
                  lw=1.4, label="10 % ledger tolerance")
    ax[0].set_xlabel("nx")
    ax[0].set_ylabel("error vs calibrated reference  [%]")
    ax[0].set_title("(a) error vs published reference")
    ax[0].legend(fontsize=8)
    ax[0].grid(True, which="both", alpha=0.3)

    sc = a["self_convergence"]["fvm"]
    pairs = [f"{d['coarse_nx']}→{d['fine_nx']}" for d in sc["differences"]]
    vals = [d["l1_window_percent"] for d in sc["differences"]]
    ax[1].semilogy(range(len(vals)), vals, "^-", color="tab:green",
                   label="grid-to-grid L1 (reference-free)")
    for i, v in enumerate(vals):
        ax[1].annotate(f"{v:.2f} %", (i, v), textcoords="offset points",
                       xytext=(0, 6), ha="center", fontsize=8)
    ax[1].set_xticks(range(len(vals)))
    ax[1].set_xticklabels(pairs, fontsize=8)
    ax[1].set_ylabel("∫|ρ_nx − ρ_2nx| dx / ∫ρ dx  [%]")
    ax[1].set_title("(b) our own discretisation error (no reference needed)")
    ax[1].grid(True, which="both", alpha=0.3)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    out = FIG / "fig_xval_ladder.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def fig_residual(a: dict, refs: dict, prof: dict) -> Path:
    tag = a["finest_tag"]
    p = prof[tag]
    x, rho = p["x"], p["rho"]
    cal = refs["cal"]
    s = (x >= 0.4) & (x <= 1.0)

    fig = plt.figure(figsize=(12, 4.4))
    ax = fig.add_subplot(1, 3, 1)
    ax.plot(cal["x"], cal["rho"], "k-", lw=1.4, label="published reference (calibrated)")
    ax.plot(x[s], rho[s], "-", color="tab:blue", lw=0.9, label=f"TensorFVM nx={p['nx']}")
    ax.set_xlabel("x")
    ax.set_ylabel("ρ")
    ax.set_title("(a) full comparison window")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(1, 3, 2)
    m = (x >= 0.74) & (x <= 0.82)
    mc = (cal["x"] >= 0.74) & (cal["x"] <= 0.82)
    ax.plot(cal["x"][mc], cal["rho"][mc], "k-", lw=1.4, label="reference")
    ax.plot(x[m], rho[m], "-", color="tab:blue", lw=1.0, label=f"nx={p['nx']}")
    ax.set_xlabel("x")
    ax.set_title("(b) peak / collision region")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(1, 3, 3)
    bands = a["band_contributions_finest"]
    labels = list(bands)
    vals = [bands[b]["l1_contribution_percent"] for b in labels]
    ax.barh(range(len(labels)), vals, color="tab:purple")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("share of the L1 integral  [%]")
    ax.set_title("(c) where the residual sits")
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.2f}", (v, i), xytext=(3, 0), textcoords="offset points",
                    va="center", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out = FIG / "fig_xval_residual.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def fig_floor(a: dict, refs: dict, prof: dict) -> Path:
    cal = refs["cal"]
    fig, ax = plt.subplots(1, 3, figsize=(12, 4.2))

    pe = a["peak_extrapolation_fvm"]
    ax[0].semilogx(pe["nx"], pe["peak"], "o-", color="tab:blue", label="ρ_max(nx)")
    if pe.get("peak_infinity"):
        ax[0].axhline(pe["peak_infinity"], color="tab:blue", ls="--", lw=1.2,
                      label=f"extrapolated {pe['peak_infinity']:.3f} (order {pe['order']:.2f})")
    ax[0].axhline(cal["rho"].max(), color="k", ls=":", lw=1.4,
                  label=f"reference peak {cal['rho'].max():.3f}")
    ax[0].set_xlabel("nx")
    ax[0].set_ylabel("peak density")
    ax[0].set_title("(a) peak amplitude → continuum")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3, which="both")

    fvm = sorted(((r["nx"], r["max_gradient"]) for r in a["profiles"].values()
                  if r["family"] == "fvm"))
    nx = np.array([v[0] for v in fvm], float)
    g = np.array([v[1] for v in fvm], float)
    ax[1].loglog(nx, g, "o-", color="tab:orange", label="max |dρ/dx|, TensorFVM")
    ax[1].axhline(a["reference"]["calibrated"]["max_gradient"], color="k", ls=":",
                  lw=1.4, label="max |dρ/dx|, digitised reference")
    ax[1].set_xlabel("nx")
    ax[1].set_ylabel("steepest gradient  [ρ per unit x]")
    ax[1].set_title("(b) resolving power vs the reference")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3, which="both")

    sf = a["smoothing_floor_finest"]
    w = np.asarray(sf["widths"])
    l = np.asarray(sf["l1_percent"])
    ax[2].semilogx(np.maximum(w, 1e-5), l, "-", color="tab:red")
    ax[2].plot([sf["best_width"]], [sf["best_percent"]], "o", color="tab:red",
               label=f"min {sf['best_percent']:.2f} % @ w={sf['best_width']:.4f}")
    ax[2].axhline(sf["base_percent"], color="0.4", ls="--", lw=1,
                  label=f"no smoothing {sf['base_percent']:.2f} %")
    ax[2].set_xlabel("top-hat smoothing width applied to our profile")
    ax[2].set_ylabel("L1 vs reference  [%]")
    ax[2].set_title("(c) is the residual sharpness? (floor scan)")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=0.3, which="both")
    fig.tight_layout()
    out = FIG / "fig_xval_floor.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def main() -> None:
    a, refs = load()
    prof = profiles(a)
    for p in (fig_ladder(a), fig_residual(a, refs, prof), fig_floor(a, refs, prof)):
        print(p.relative_to(ROOT.parent.parent))


if __name__ == "__main__":
    main()