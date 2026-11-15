"""Figures for the transport-norm (W1) certification (anchor XVAL-FVM-M18737-ROUND2).

Four panels, all driven directly by ``wc_w1_norm.build()`` (zero hand-copied
numbers):

(a) both norms vs the mesh size on a log-log axis, with the observed slopes --
    the point of the exercise: ``d_W1`` sits ~100x below ``d_L1`` at every grid;
(b) the accumulated difference field ``D(x) = int e`` for the graded rungs --
    where the transport error actually lives (x ~ 0.765, the collision region),
    and that it is a net-mass-flux defect, not a smeared-shock sawtooth;
(c) the certification ladder against the 1 % bar: pointwise L1 vs transport norm
    + reference error bar, with the honest bound;
(d) the injected-error probes, measured vs closed form, on a y = x line --
    the evidence that ``d_W1`` is not a metric that measures nothing.

Usage::

    PYTHONPATH=src python docs/XVAL-FVM-M18737-ROUND2/wc_w1_norm_figures.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"

spec = importlib.util.spec_from_file_location("wc_w1_norm", HERE / "wc_w1_norm.py")
w1 = importlib.util.module_from_spec(spec)
sys.modules.setdefault("wc_w1_norm", w1)
spec.loader.exec_module(w1)


def _load_all():
    prof = {}
    for fam, ladder in (("hllc", w1.MUSCL_LADDER), ("weno5", w1.WENO_LADDER)):
        for nx in ladder:
            p = w1.load(nx, fam)
            if p is not None:
                prof[(fam, nx)] = p
    return prof


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    out = w1.build(verbose=False)
    prof = _load_all()
    ref = prof[("weno5", w1.REF_FINEST)]
    span = w1.WINDOW[1] - w1.WINDOW[0]

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.4))

    # ---------------------------------------------------------------- (a) norms
    ax = axes[0, 0]
    g_rows = out["graded_family"]["rows"]
    dx_g = np.array([span / r["nx"] for r in g_rows])
    l1_g = np.array([r["d_l1"] for r in g_rows])
    w1_g = np.array([r["d_w1"] for r in g_rows])
    ax.loglog(dx_g, l1_g, "o-", color="#c0392b", lw=2, label="pointwise L1 (round-2 metric)")
    ax.loglog(dx_g, w1_g, "s-", color="#1f6fb4", lw=2, label="transport norm $d_{W1}$")
    for arr, col, name in ((l1_g, "#c0392b", "L1"), (w1_g, "#1f6fb4", "W1")):
        p = np.polyfit(np.log(dx_g), np.log(arr), 1)[0]
        ax.annotate(f"{name}: $\\propto h^{{{p:.2f}}}$", xy=(dx_g[-1], arr[-1]),
                    xytext=(6, -12 if name == "W1" else 8), textcoords="offset points",
                    color=col, fontsize=9)
    for p in out["reference_family"]["pairs"]:
        ax.loglog(span / p["fine_nx"], p["d_w1"], "^", color="#1f6fb4", ms=7, mfc="none")
    ax.loglog([], [], "^", color="#1f6fb4", mfc="none", label="WENO5 self-convergence (W1)")
    ax.set_xlabel("mesh size $h$  (window $[0.4,1]$, span 0.6)")
    ax.set_ylabel("distance to WENO5@12800  [% ]")
    ax.set_title("(a) same ladder, two norms: a ~100x constant, not an order")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

    # -------------------------------------------------- (b) accumulated diff
    ax = axes[0, 1]
    for r in g_rows:
        f = w1.cumulative_diff_field(ref, prof[("hllc", r["nx"])])
        ax.plot(f["x"], f["D"], lw=1.6, label=f"MUSCL nx={r['nx']}")
    for p in out["reference_family"]["pairs"][-1:]:
        f = w1.cumulative_diff_field(prof[("weno5", p["fine_nx"])],
                                     prof[("weno5", p["coarse_nx"])])
        ax.plot(f["x"], f["D"], "--", color="k", lw=1.2,
                label=f"WENO5 {p['coarse_nx']}v{p['fine_nx']}")
    ax.axhline(0.0, color="grey", lw=0.8)
    ax.set_xlabel("x")
    ax.set_ylabel(r"$D(x)=\int_{0.4}^{x}(\rho_c-\rho_f)\,dx'$   [mass units]")
    ax.set_title("(b) where the transport error lives: x~0.765 (collision region)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # ------------------------------------------------- (c) certification ladder
    ax = axes[1, 0]
    nx = [r["nx"] for r in g_rows]
    ax.plot(nx, [r["d_l1"] for r in g_rows], "o-", color="#c0392b", lw=2, label="pointwise L1")
    ax.plot(nx, [r["d_w1"] for r in g_rows], "s-", color="#1f6fb4", lw=2, label="$d_{W1}$ (transport)")
    eps = out["reference_family"]["eps_ref_w1"]
    ax.errorbar(nx, [r["d_w1"] for r in g_rows], yerr=eps, fmt="none", ecolor="#1f6fb4",
                capsize=4, label=f"$\\pm\\epsilon_{{ref}}$={eps:.4f} %")
    ax.axhline(1.0, color="k", ls="--", lw=1.2, label="1 % bar")
    ax.axhline(out["verdict"]["P4"]["honest_bound"], color="#1f6fb4", ls=":", lw=1.2,
               label=f"honest bound {out['verdict']['P4']['honest_bound']:.4f} %")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(nx)
    ax.set_xticklabels([str(v) for v in nx])
    ax.set_xlabel("graded mesh nx (reference WENO5@12800)")
    ax.set_ylabel("distance  [% ]")
    ax.set_title("(c) <1 % is provable in the transport norm, not pointwise")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    # ------------------------------------------------------------- (d) probes
    ax = axes[1, 1]
    probes = out["probes_post_hoc"]
    pts = []
    for key in ("shift_1cell_nx3200", "bump_shift_1cell_exact",
                "mass_1pct_misplaced", "amplitude_1pct"):
        p = probes[key]
        pts.append((key, p["measured"], p["analytic"]))
    meas = [q[1] for q in pts]
    anal = [q[2] for q in pts]
    ax.loglog(anal, meas, "o", ms=9, color="#6c5ce7")
    lim = [min(anal + meas) * 0.5, max(anal + meas) * 2.0]
    ax.loglog(lim, lim, "k--", lw=1, label="y = x (closed form)")
    for key, m, a in pts:
        ax.annotate(key.replace("_", " "), xy=(a, m), xytext=(7, -10),
                    textcoords="offset points", fontsize=7.5, color="#444")
    ctl = probes["published_curve_control"]
    ax.axhline(ctl["measured"], color="#e17055", ls="-.", lw=1.2,
               label=f"negative control: published curve {ctl['measured']:.3f} %")
    ax.axhline(out["verdict"]["P4"]["honest_bound"], color="#0984e3", ls=":", lw=1.2,
               label=f"our honest bound {out['verdict']['P4']['honest_bound']:.4f} %")
    ax.set_xlabel("injected error, closed form  [% ]")
    ax.set_ylabel("measured $d_{W1}$  [% ]")
    ax.set_title("(d) metric validation: injected errors read back at closed form")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

    fig.suptitle("CASE-WOODWARD-COLELLA — transport-norm (W1) certification of the graded FVM solution",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    png = FIG / "fig_w1_norm.png"
    fig.savefig(png, dpi=150)
    (FIG / "w1_norm_budget.json").write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(f"wrote {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())