"""Figure for the graded-family refinement closure (WC double blast).

Two panels, both injected straight from ``wc_graded_refinement.run()`` (no
hand-copied numbers):

(a) window-L1 of the MUSCL/HLLC ladder against the self-certified WENO5@12800
    reference, log-log, with the power-law prediction for the rungs that were
    still being computed when this figure was generated (open markers);
(b) peak-density error on the same ladder, with the 1 % bar drawn for reference
    -- it shows why peak and L1 are *not* interchangeable as "certified" gates.

Usage::

    PYTHONPATH=src python docs/woodward-colella-double-blast/xval/wc_graded_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import wc_graded_refinement as g

HERE = Path(__file__).resolve().parent
OUT = HERE / "fig_graded_refinement.png"
BAR = g.CERTIFIED_PERCENT


def main() -> None:
    v = g.run()
    lad = {int(n): r for n, r in v["ladder"].items()}
    ns = sorted(lad)
    l1 = [lad[n]["l1_percent"] for n in ns]
    pk = [lad[n]["peak_percent"] for n in ns]

    preds = {int(n): e for n, e in v["asymptote"]["predicted_l1_percent"].items()}
    p = v["asymptote"]["fixed_order"]["order_used"]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    ax = axes[0]
    ax.loglog(ns, l1, "o-", color="C0", label="MUSCL/HLLC vs WENO5@12800")
    if preds:
        px = sorted(preds)
        ax.loglog(px, [preds[k] for k in px], "s--", mfc="none", color="C1",
                  label=f"power-law prediction (p={p:.2f})")
    ax.axhline(BAR, color="k", ls=":", lw=1.2, label="1 % bar")
    for n, e in zip(ns, l1):
        ax.annotate(f"{e:.2f}%", (n, e), textcoords="offset points",
                    xytext=(6, 6), fontsize=8)
    ax.set_xlabel("nx (cells)")
    ax.set_ylabel("window L1 vs certified reference [%]")
    ax.set_title("(a) L1 refinement of the graded family")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.semilogx(ns, pk, "o-", color="C3", label="peak density error")
    ax.axhline(BAR, color="k", ls=":", lw=1.2, label="1 % bar")
    for n, e in zip(ns, pk):
        ax.annotate(f"{e:.2f}%", (n, e), textcoords="offset points",
                    xytext=(6, 6), fontsize=8)
    ax.set_xlabel("nx (cells)")
    ax.set_ylabel("peak density error [%]")
    ax.set_title("(b) Peak error is non-monotone (shock sampling)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle(
        f"WC double blast — graded-family refinement closure  "
        f"(status {v['status']}, floor class {v['asymptote']['floor_class']})",
        fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()