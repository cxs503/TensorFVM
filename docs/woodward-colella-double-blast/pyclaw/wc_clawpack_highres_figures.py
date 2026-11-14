"""Figures for the Clawpack high-resolution cross-validation (third code).

Everything plotted is read back from ``highres/results.json`` and
``highres/profiles.npz`` — no number is typed in by hand.

    PYTHONPATH=src python docs/woodward-colella-double-blast/pyclaw/wc_clawpack_highres_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "highres"
REF = HERE.parent / "reference"
ASSETS = HERE.parent / "artifacts"
LADDER = HERE.parent / "ladder" / "data"


def _ref(name):
    raw = np.loadtxt(REF / name, delimiter=",", skiprows=1)
    return raw[:, 0], 0.5 * (raw[:, 1] + raw[:, 2])


def _tfvm(nx):
    path = (ASSETS / f"wc_blast_profile_nx{nx}.npz") if nx <= 800 else \
           (LADDER / f"ladder_profile_nx{nx}_hllc.npz")
    d = np.load(path)
    x = d["x"] if "x" in d else d["x_got"]
    r = d["rho"] if "rho" in d else d["rho_got"]
    return np.asarray(x), np.asarray(r)


def main() -> int:
    res = json.loads((OUT / "results.json").read_text())
    an = res["analysis"]
    prof = np.load(OUT / "profiles.npz")
    tf_rows = an["tfvm_vs_truth"]["roe"]
    nx = [r["nx"] for r in tf_rows]
    peak = [r["peak_percent"] for r in tf_rows]
    l1 = [r["l1_percent"] for r in tf_rows]
    truth = an["truth"]["roe"]
    cross = an["cross_code"]
    dig = an["vs_digitised"]

    fig = plt.figure(figsize=(15.5, 9.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.05], hspace=0.32, wspace=0.28)

    # (a) graded ladder against the third-code truth -----------------------
    ax = fig.add_subplot(gs[0, 0])
    ax.semilogx(nx, peak, "o-", color="#c0392b", label="peak density")
    ax.semilogx(nx, l1, "s-", color="#2980b9", label=r"profile $L_1$")
    ax.axhline(3, color="#27ae60", ls="--", lw=1.2, label="3 % line")
    ax.axhline(10, color="#7f8c8d", ls=":", lw=1.4, label="ledger 10 %")
    ax.axhline(15, color="#8e44ad", ls="-.", lw=1.0, label="owner 15 % reject")
    ax.set_xlabel("TensorFVM cells $n_x$")
    ax.set_ylabel("error vs Clawpack $n_x$=%d  [%%]" % truth["nx"])
    ax.set_title("(a) graded ladder vs third-code truth\n(truth error bar %.2f %%)"
                 % truth["error_bar_l1_percent"], fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    for x_, y_ in zip(nx, l1):
        ax.annotate("%.2f" % y_, (x_, y_), textcoords="offset points",
                    xytext=(0, -13), fontsize=7, ha="center", color="#2980b9")

    # (b) error-budget bar chart ------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    labels = ["TensorFVM@3200\nvs truth ($L_1$)",
              "Clawpack truth\nown error bar",
              "cross-code\n@3200 ($L_1$)",
              "digitised curve\nvs truth ($L_1$)"]
    vals = [l1[-1], truth["error_bar_l1_percent"],
            cross[-1]["l1_percent"], dig["calibrated"]["l1_percent"]]
    cols = ["#2980b9", "#16a085", "#8e44ad", "#e67e22"]
    bars = ax.bar(range(4), vals, color=cols, alpha=0.85)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.06, "%.2f %%" % v,
                ha="center", fontsize=9)
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("[%]")
    ax.set_title("(b) who owns the 3 % residual\n(the printed curve, not the solver)",
                 fontsize=10)
    ax.grid(alpha=0.3, axis="y")

    # (c) Clawpack self-convergence ---------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    sc = an["self_convergence"]["roe"][0]
    ax.bar([0, 1], [sc["coarse_vs_fine_l1_percent"], sc["fine_vs_coarse_l1_percent"]],
           color="#16a085", alpha=0.85, width=0.55)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["$n_x$=%d vs %d" % (sc["coarser"], sc["finer"]),
                        "reverse direction"], fontsize=8)
    ax.set_ylabel(r"$L_1$ [%]")
    ax.set_title("(c) third code self-convergence\n"
                 "peak %.5f -> %.5f (%.3f %%)"
                 % (sc["peak_coarse"], sc["peak_fine"], sc["peak_rel_percent"]),
                 fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    ax.text(0.5, max(sc["coarse_vs_fine_l1_percent"],
                     sc["fine_vs_coarse_l1_percent"]) * 0.55,
            "steps to $t_{end}$:\n%s"
            % ", ".join("%d: %d" % (r["nx"], r["steps"]) for r in res["rows"]),
            ha="center", fontsize=7.5)

    # (d) profiles ---------------------------------------------------------
    xr, rr = prof["x_%d_roe" % truth["nx"]], prof["rho_%d_roe" % truth["nx"]]
    xm, rm = _tfvm(3200)
    xd, rd = _ref("reference_density_calibrated.csv")
    ax = fig.add_subplot(gs[1, :2])
    ax.plot(xr, rr, "-", color="#16a085", lw=2.4,
            label="Clawpack %d cells (third code, truth)" % truth["nx"])
    ax.plot(xm, rm, "-", color="#2980b9", lw=1.1,
            label="TensorFVM 3200 cells (graded)")
    ax.plot(xd, rd, "k.", ms=1.2, alpha=0.55,
            label="digitised published curve (calibrated)")
    ax.set_xlabel("$x$")
    ax.set_ylabel(r"$\rho$")
    ax.set_title("(d) $t=0.038$: two independent codes agree to %.2f %% $L_1$ "
                 "(matched 3200 cells) while both sit %.2f %% off the digitised curve"
                 % (cross[-1]["l1_percent"], dig["calibrated"]["l1_percent"]),
                 fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    ax.set_xlim(0.4, 1.0)

    axin = ax.inset_axes([0.62, 0.42, 0.34, 0.52])
    axin.plot(xr, rr, "-", color="#16a085", lw=2.2)
    axin.plot(xm, rm, "-", color="#2980b9", lw=1.0)
    axin.plot(xd, rd, "k.", ms=2.0, alpha=0.7)
    axin.set_xlim(0.72, 0.84)
    axin.set_ylim(4.6, 6.9)
    axin.tick_params(labelsize=7)
    axin.set_title("peak region", fontsize=8)

    # (e) peak amplitude / truth bracket ----------------------------------
    ax = fig.add_subplot(gs[1, 2])
    peaks = {r["nx"]: float(_tfvm(r["nx"])[1].max()) for r in tf_rows}
    ax.plot(list(peaks), list(peaks.values()), "o-", color="#2980b9",
            label="TensorFVM (stored profiles)")
    ax.plot([3200, 6400], [sc["peak_coarse"], sc["peak_fine"]], "s-",
            color="#16a085", label="Clawpack")
    ax.axhline(dig["calibrated"]["truth_peak"], color="k", ls="--", lw=1.0,
               label="digitised curve peak %.4f" % dig["calibrated"]["truth_peak"])
    ax.set_xscale("log")
    ax.set_xlabel("cells $n_x$")
    ax.set_ylabel(r"$\rho_{max}$")
    ax.set_title("(e) peak amplitude: the two codes converge\nfrom the same side, "
                 "the printed curve sits low", fontsize=10)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3, which="both")

    fig.suptitle("CASE-WOODWARD-COLELLA — third-code (Clawpack) high-resolution "
                 "reference: the round-2 3 % floor is reference-side, not solver-side",
                 fontsize=12)
    out = OUT / "fig_wc_clawpack_highres.png"
    fig.savefig(out, dpi=135, bbox_inches="tight")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())