"""Figures for CASE-FVM-DAMBREAK-STOKER (reads artifacts/reproduction.json).

* (a) Ritter dry bed: finite-volume profile vs the closed form vs the generic
  Riemann solution (independent code path).
* (b) Stoker wet bed: same, with a zoom on the bore.
* (c) Grid convergence of :math:`L_1` depth error (both cases) together with
  the dry-bed front metrics -- the honest picture of what converges.
* (d) Weak-bore conditioning: relative error of the bore speed against a
  60-digit reference for the subtractive and the cancellation-free forms.

Usage::

    PYTHONPATH=src python docs/case-fvm-dambreak-stoker/make_figure.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts" / "reproduction.json"


def main() -> Path:
    data = json.loads(ARTIFACTS.read_text())
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.5))

    # (a) Ritter dry bed
    ax = axes[0][0]
    ritter = next(p for p in data["profiles"] if p["case"] == "ritter-dry")
    prof = ritter["profile"]
    ax.plot(prof["x"], prof["h_exact"], "k-", lw=2.0, label="Ritter (1892) exact")
    ax.plot(prof["x"], prof["h_generic_riemann"], "o", mfc="none", ms=5,
            markevery=6, label="generic SWE Riemann solver")
    ax.plot(prof["x"], prof["h_numeric"], "r-", lw=1.2, label="MUSCL-HLL n=800")
    ax.axvline(prof["exact_front"], color="0.5", ls=":", lw=1.0)
    ax.annotate(f"$x_f$ = {prof['exact_front']:.3f} m",
                xy=(prof["exact_front"], 0.6), xytext=(2.0, 0.75),
                arrowprops={"arrowstyle": "->", "lw": 0.8}, fontsize=9)
    m = data["headline"]["ritter-dry"]["metrics"]
    ax.set_title("(a) Ritter dry bed, $t^*=2$\n"
                 f"fan front error {100*m['front_fan_error']:.3f} %, "
                 f"$L_1$ depth {100*m['depth_l1']:.3f} %", fontsize=10)
    ax.set_xlabel("$x$ [m]")
    ax.set_ylabel("$h$ [m]")
    ax.set_ylim(-0.03, 1.1)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)

    # (b) Stoker wet bed + bore zoom
    ax = axes[0][1]
    stoker = next(p for p in data["profiles"] if p["case"] == "stoker-wet")
    prof = stoker["profile"]
    ax.plot(prof["x"], prof["h_exact"], "k-", lw=2.0, label="Stoker (1957) exact")
    ax.plot(prof["x"], prof["h_generic_riemann"], "o", mfc="none", ms=5,
            markevery=6, label="generic SWE Riemann solver")
    ax.plot(prof["x"], prof["h_numeric"], "r-", lw=1.2, label="MUSCL-HLL n=800")
    ax.axvline(prof["exact_front"], color="0.5", ls=":", lw=1.0)
    m = data["headline"]["stoker-wet"]["metrics"]
    ax.set_title("(b) Stoker wet bed ($h_1/h_0=0.5$), $t^*=2$\n"
                 f"bore error {100*m['shock_error']:.3f} %, "
                 f"level-step error {100*m['step_ratio_error']:.2e} %", fontsize=10)
    ax.set_xlabel("$x$ [m]")
    ax.set_ylabel("$h$ [m]")
    ax.set_ylim(0.4, 1.05)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)

    inset = ax.inset_axes([0.55, 0.35, 0.42, 0.42])
    x = np.asarray(prof["x"])
    window = (x > m["shock_exact"] - 0.6) & (x < m["shock_exact"] + 0.6)
    inset.plot(x[window], np.asarray(prof["h_exact"])[window], "k-", lw=2.0)
    inset.plot(x[window], np.asarray(prof["h_numeric"])[window], "r-", lw=1.2)
    inset.axvline(m["shock_exact"], color="0.5", ls=":", lw=1.0)
    inset.set_title(f"bore: $h_2/h_1$ = {m['step_ratio_measured']:.4f} "
                    f"(exact {m['step_ratio_exact']:.4f})", fontsize=7)
    inset.tick_params(labelsize=6)

    # (c) convergence
    ax = axes[1][0]
    styles = {"ritter-dry": ("C3", "Ritter dry bed"),
              "stoker-wet": ("C0", "Stoker wet bed")}
    for key, (colour, label) in styles.items():
        study = data["convergence"][key]
        n = np.array([row["n"] for row in study["rows"]], dtype=float)
        l1 = np.array([row["depth_l1"] for row in study["rows"]])
        ax.loglog(1.0 / n, l1, "o-", color=colour,
                  label=f"{label}: $L_1$ (order {study['observed_order_l1']:.2f})")
    study = data["convergence"]["ritter-dry"]
    n = np.array([row["n"] for row in study["rows"]], dtype=float)
    fan = np.array([row["headline"]["front_fan"] for row in study["rows"]])
    wet = np.array([row["headline"]["front_wetted_diagnostic"]
                    for row in study["rows"]])
    ax.loglog(1.0 / n, fan, "s--", color="C2", lw=1.0,
              label="dry front, fan estimator (headline)")
    ax.loglog(1.0 / n, wet, "^:", color="C7", lw=1.0,
              label="dry front, wetted edge (diagnostic)")
    ref = l1_ref = study["rows"][0]["depth_l1"]
    ax.loglog(1.0 / n, ref * (n[0] / n), "k--", lw=0.8, label="slope 1 reference")
    ax.set_xlabel(r"$\Delta x$ [m]")
    ax.set_ylabel("relative error")
    ax.set_title("(c) grid convergence: $L_1$ converges at order ~1;\n"
                 "the wetted-edge diagnostic does not converge", fontsize=10)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3, which="both")

    # (d) weak-bore conditioning
    ax = axes[1][1]
    rows = data["weak_bore_conditioning"]
    gap = np.array([row["depth_gap"] for row in rows])
    ax.loglog(gap, [row["package_relative_error"] for row in rows], "o-",
              color="C0", label="cancellation-free form (package)")
    ax.loglog(gap, [max(row["subtractive_relative_error"], 1e-18)
                    for row in rows], "s-", color="C3",
              label=r"subtractive $h_2 u_2/(h_2-h_1)$")
    ax.axhline(1.0, color="0.6", lw=0.8, ls=":")
    ax.set_xlabel(r"depth gap $(h_0-h_1)/h_0$")
    ax.set_ylabel("relative error of bore speed $U$")
    ax.set_title("(d) weak-bore limit: bore speed vs 60-digit reference",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    fig.suptitle("CASE-FVM-DAMBREAK-STOKER — independent verification of the "
                 "Ritter/Stoker references", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = HERE / "crosscheck.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    main()