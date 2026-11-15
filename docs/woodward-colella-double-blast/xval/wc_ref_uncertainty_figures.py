"""Figures for the reference-side uncertainty budget (W&C double blast).

Three panels, all built from committed profiles (no solver call):

(a) successive-grid L1 distances for both families vs cell size (log-log) with
    the fitted observed orders -- the evidence for (and the price of) quoting an
    error bar at all;
(b) zoom on the collision peak: the graded MUSCL/HLLC profile, the WENO5@12800
    reference, and both Richardson-extrapolated continuum stand-ins.  This is
    where the residual lives, so it shows *which* of the two families moves;
(c) the honest interval: measured L1 per graded rung, the reference-corrected
    estimate, the +/-eps reference band, and the 1 % / 2 % bars.  The point of
    the panel is visual: the reference band is wider than the distance between
    the 1 % line and the point estimate.

Usage::

    PYTHONPATH=src python docs/woodward-colella-double-blast/xval/wc_ref_uncertainty_figures.py
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from pathlib import Path  # noqa: E402

import wc_ref_uncertainty as unc  # noqa: E402  (same directory, on sys.path via pytest/script)

OUT = Path(__file__).resolve().parent.parent / "artifacts" / "fig_wc_ref_uncertainty.png"


def main() -> None:
    ref = unc.reference_uncertainty()
    grading = unc.grade(ref)
    eps = ref["eps_ref_percent"]

    fig, axes = plt.subplots(1, 3, figsize=(17.0, 4.8))

    # (a) self-convergence of both families -----------------------------------
    ax = axes[0]
    for family, style in (("weno5", "o-"), ("hllc", "s-")):
        ladders = unc.WENO5_LADDER if family == "weno5" else unc.MUSCL_LADDER
        have = [(nx, unc._load(nx, family)) for nx in ladders]
        have = [(nx, p) for nx, p in have if p is not None]
        hs, ds = [], []
        for (na, pa), (nb, pb) in zip(have[:-1], have[1:]):
            if pb["nx"] != 2 * pa["nx"]:
                continue
            hs.append(1.0 / pb["nx"])
            ds.append(unc.l1_percent(pb, pa))
        if hs:
            ax.loglog(hs, ds, style, label=f"{family} successive-grid L1")
    if ref.get("status") == "OK":
        p = ref["observed_order"]
        h_fine = 1.0 / ref["reference_nx"]
        d_fine = ref["successive"][-1]["l1_percent"]
        hh = np.array([h_fine, 4 * h_fine])
        ax.loglog(hh, d_fine * (hh / h_fine) ** p, "k--", lw=1,
                  label=f"WENO5 slope p={p:.2f}")
    ax.set_xlabel("cell size h = 1/nx")
    ax.set_ylabel("L1 distance to next-coarser grid (%)")
    ax.set_title("(a) Self-convergence: the price of an error bar")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    # (b) peak zoom with both extrapolated fields ----------------------------
    ax = axes[1]
    ref_fine = ref["ref_fine"]
    rho_inf = ref["rho_inf"]
    peak_x = float(ref_fine["x"][np.argmax(np.where(
        (ref_fine["x"] >= unc.WINDOW[0]) & (ref_fine["x"] <= unc.WINDOW[1]),
        ref_fine["rho"], -np.inf))])
    lo, hi = peak_x - 0.07, peak_x + 0.07
    for prof, label, style in ((ref_fine, "WENO5@12800 (reference)", "b-"),
                               (rho_inf, "WENO5 Richardson-extrapolated", "c--")):
        m = (prof["x"] >= lo) & (prof["x"] <= hi)
        ax.plot(prof["x"][m], prof["rho"][m], style, lw=1.0 if style == "b-" else 1.4,
                label=label)
    finest = unc._load(max(int(k) for k in grading["graded"]), "hllc")
    m = (finest["x"] >= lo) & (finest["x"] <= hi)
    ax.plot(finest["x"][m], finest["rho"][m], "r-", lw=1.0,
            label=f"MUSCL/HLLC@{finest['nx']} (graded)")
    cross = grading.get("cross_family_extrapolated") or {}
    if cross:
        a, b = int(cross["graded_nx"]) // 2, int(cross["graded_nx"])
        m_inf = unc.extrapolate_field(unc._load(b, "hllc"), unc._load(a, "hllc"),
                                      cross["muscl_observed_order"])
        mm = (m_inf["x"] >= lo) & (m_inf["x"] <= hi)
        ax.plot(m_inf["x"][mm], m_inf["rho"][mm], "m--", lw=1.2,
                label="MUSCL Richardson-extrapolated")
    ax.set_xlabel("x")
    ax.set_ylabel("density ρ")
    ax.set_title("(b) Collision peak: reference vs both continuum stand-ins")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (c) the honest interval ------------------------------------------------
    ax = axes[2]
    nxs = sorted(int(k) for k in grading["graded"])
    xs = np.arange(len(nxs))
    d_ref = [grading["graded"][n]["l1_vs_reference_percent"] for n in nxs]
    d_cor = [grading["graded"][nx]["l1_vs_extrapolated_percent"] for nx in nxs]
    ax.bar(xs - 0.18, d_ref, width=0.36, color="0.65", label="vs WENO5@12800 (raw)")
    ax.bar(xs + 0.18, d_cor, width=0.36, color="tab:blue",
           label="vs extrapolated reference")
    ax.errorbar(xs + 0.18, d_cor, yerr=eps, fmt="none", ecolor="tab:red", capsize=4,
                label=f"reference side ±{eps:.2f} %")
    ax.axhline(1.0, color="tab:green", ls="--", lw=1, label="1 % bar")
    ax.axhline(2.0, color="tab:orange", ls=":", lw=1, label="2 % bar")
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"nx={n}" for n in nxs])
    ax.set_ylabel("L1 error (%)")
    ax.set_title("(c) Honest interval: point estimate vs provable bound")
    ax.legend(fontsize=7)
    ax.grid(True, axis="y", alpha=0.3)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()