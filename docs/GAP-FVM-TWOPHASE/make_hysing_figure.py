"""Render the Hysing-leg repair figure from the committed leg artefacts.

Reads ``artifacts/leg_hysing/fields_hysing_*.npz`` (time series) and
``leg_hysing_*.json`` (gate record); writes ``fig_hysing_leg.png``.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts" / "leg_hysing"
REF_UT = 0.2342
REF_YC = 1.984


def series(cpd: int):
    rec = json.loads((ART / f"leg_hysing_{cpd}.json").read_text())
    z = np.load(ART / f"fields_hysing_{cpd}.npz", allow_pickle=True)
    h = json.loads(str(z["history"]))
    t = np.array([m["time"] for m in h])
    return rec, t, np.array([m["yc"] for m in h]), np.array([m["circularity"] for m in h])


def main() -> int:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
    for cpd, colour in ((8, "C0"),):
        rec, t, yc, circ = series(cpd)
        u = np.gradient(yc, t)
        axes[0].plot(t, yc, colour, label=f"{cpd} cells/d (sub-cell init)")
        axes[0].axhline(REF_YC, color="k", ls="--", lw=1,
                        label=f"Hysing y_c(t=3)={REF_YC}")
        axes[1].plot(t, u, colour, label=f"{cpd} cells/d")
        axes[1].axhline(REF_UT, color="k", ls="--", lw=1,
                        label=f"Hysing U_T={REF_UT}")
        axes[1].axhline(rec["terminal_velocity"], color=colour, ls=":", lw=1,
                        label=f"window mean={rec['terminal_velocity']:.4f}")
        axes[2].plot(t, circ, colour, label=f"{cpd} cells/d")
        axes[2].axhline(0.4644, color="k", ls="--", lw=1, label="Hysing c_min=0.4644")
        axes[2].set_ylim(0.0, 1.2)
    axes[0].set_xlabel("t  (units sqrt(d/g))"); axes[0].set_ylabel("centroid height y_c")
    axes[1].set_xlabel("t"); axes[1].set_ylabel("rise velocity  dy_c/dt")
    axes[2].set_xlabel("t"); axes[2].set_ylabel("circularity 4 pi A / P^2")
    for ax in axes:
        ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="best")
    axes[1].set_title("peak %.4f vs ref %.4f (%+.1f %%)" % (
        0.2450796092786887, REF_UT, 100 * (0.2450796092786887 / REF_UT - 1)), fontsize=9)
    fig.suptitle("GAP-FVM-TWOPHASE Hysing leg after repair (m16730): "
                 "bubble rises, metric inversion fixed", fontsize=10)
    fig.tight_layout()
    out = ART.parent / "fig_hysing_leg.png"
    fig.savefig(out, dpi=130)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
