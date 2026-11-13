"""Figures for the CASE-WOODWARD-COLELLA low-density-region diagnosis.

Reads only the artifacts written by ``wc_ladder.py`` / ``wc_ladder_weno5.py`` and
``wc_plateau_convergence.py`` (never the solver), so a plot can be redone or
disputed without recomputing anything.

    python docs/woodward-colella-double-blast/ladder/wc_plateau_figures.py
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent
DATA = _HERE / "data"
TOL = 10.0


def _load_analysis():
    spec = importlib.util.spec_from_file_location(
        "wc_plateau_convergence", _HERE / "wc_plateau_convergence.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["wc_plateau_convergence"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--out", type=Path,
                    default=_HERE.parent / "artifacts" / "fig_wc_plateau_diagnosis.png")
    args = ap.parse_args()

    wpc = _load_analysis()
    res = wpc.analyse(args.data, _HERE.parent / "artifacts", outdir=args.data)
    ref = wpc.wcr.load_reference()
    rows = res["rows"]

    fvm = sorted([r for r in rows.values() if r["family"] == "fvm" and r["baseline"]],
                 key=lambda r: r["nx"])
    wen = sorted([r for r in rows.values() if r["family"] == "weno5"],
                 key=lambda r: r["nx"])

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0))

    # (a) the low-density region: the parallel-offset picture ----------------
    ax = axes[0, 0]
    ax.plot(ref["x"], ref["rho"], "k-", lw=2.0,
            label="Reference (RPJ 3200, digitised)")
    cmap = plt.get_cmap("viridis")
    for i, r in enumerate(fvm):
        d = np.load(r["path"])
        x, rho = (d["x"], d["rho"]) if "x" in d.files else (d["x_got"], d["rho_got"])
        ax.plot(x, rho, "-", color=cmap(i / max(1, len(fvm) - 1)), lw=1.1,
                label=f"TensorFVM nx={r['nx']}")
    for r in wen:
        d = np.load(r["path"])
        ax.plot(d["x"], d["rho"], "--", lw=1.2, label=f"WENO5 nx={r['nx']}")
    lo = wpc.ORIGINAL_WINDOW[0]
    ax.axvline(lo, color="tab:red", ls=":", lw=1.2, label="metric window edge x=0.42")
    ax.axvline(ref["x"].min(), color="tab:gray", ls="-.", lw=1.0,
               label=f"digitised domain start x={ref['x'].min():.3f}")
    ax.set_xlim(0.38, 0.62)
    ax.set_xlabel("$x$")
    ax.set_ylabel(r"$\rho$")
    ax.set_title("(a) shallow low-density region: parallel offset")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.3)

    # (b) the "window minimum" is a window truncation ------------------------
    ax = axes[0, 1]
    ax.plot(ref["x"], ref["rho"], "k-", lw=1.8, label="Reference (digitised)")
    m = wpc.window_min(ref["x"], ref["rho"], wpc.ORIGINAL_WINDOW)
    ax.plot([m[1]], [m[0]], "rv", ms=9,
            label=f"window min {m[0]:.4f} at x={m[1]:.4f}")
    ax.axvline(lo, color="tab:red", ls=":", lw=1.2)
    ax.annotate("window minimum sits ON the window border\n(monotone ramp -> truncation, not a trough)",
                xy=(m[1], m[0]), xytext=(0.52, 1.2), fontsize=8,
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_xlabel("$x$")
    ax.set_ylabel(r"$\rho$")
    ax.set_title("(b) why the published 'low_plateau' stat is ill-posed")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.3)

    # (c) error vs grid for every statistic --------------------------------
    ax = axes[1, 0]
    keys = [("window_min_rel_percent", "window min (ill-posed)", "tab:red", "o"),
            ("plateau_mean_rel_percent", "plateau mean (well-posed)", "tab:blue", "s"),
            ("probe_rel_percent", "probe x=0.50", "tab:green", "^"),
            ("window_mass_rel_percent", "window mass", "tab:purple", "D"),
            ("peak_density_rel_percent", "peak density", "tab:brown", "v"),
            ("l1_percent", "profile L1", "tab:orange", "P")]
    for fam, style in (("fvm", "-"), ("weno5", "--")):
        lad = sorted([r for r in rows.values()
                      if r["family"] == fam and (r["baseline"] or fam != "fvm")],
                     key=lambda r: r["nx"])
        if len(lad) < 1:
            continue
        nx = [r["nx"] for r in lad]
        for key, label, color, marker in keys:
            vals = ([r["published"][key] if key in r["published"]
                     else r[key.split("_rel")[0]]["rel_error_percent"] for r in lad])
            ax.plot(nx, vals, style + marker, color=color, ms=5,
                    label=f"{label} [{fam}]" if fam == "fvm" else None)
    ax.axhline(TOL, color="k", ls="--", lw=1.1, label=f"{TOL:g} % acceptance bound")
    ax.set_xscale("log")
    ax.set_xlabel("cells $n$")
    ax.set_ylabel("error vs digitised reference  [%]")
    ax.set_title("(c) grid ladder: what converges and what does not")
    ax.legend(fontsize=6.5, ncol=1)
    ax.grid(alpha=0.3, which="both")

    # (d) scheme sensitivity at fixed resolution ---------------------------
    ax = axes[1, 1]
    labels, dp, dm = [], [], []
    for nx, e in res["scheme_sensitivity"]["fixed_resolution_variants"].items():
        for tag, v in e["variants"].items():
            labels.append(tag.replace(f"nx{nx}_", ""))
            dp.append(v["delta_percent"])
            dm.append(v["window_min_delta_percent"])
    for nx, e in res["scheme_sensitivity"]["cross_family"].items():
        labels.append(f"WENO5 vs FVM @nx={nx}")
        dp.append(e["plateau_mean_diff_percent"])
        dm.append(abs(e["weno5"]["window_min"] - e["fvm"]["window_min"])
                  / e["fvm"]["window_min"] * 100)
    idx = np.arange(len(labels))
    ax.bar(idx - 0.2, dp, width=0.4, label=r"$\Delta$ plateau mean [%]")
    ax.bar(idx + 0.2, dm, width=0.4, label=r"$\Delta$ window min [%]")
    ax.set_xticks(idx)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=7)
    ax.set_ylabel("change of the statistic [%]")
    ax.set_title("(d) scheme sensitivity (limiter/CFL, and MUSCL vs WENO5)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("CASE-WOODWARD-COLELLA — low-density region: metric diagnosis and grid/scheme ladder",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()