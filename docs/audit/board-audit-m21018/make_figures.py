#!/usr/bin/env python3
"""Figures for the board audit of m21018 (GAP-DIFFLBM-EXT-LIT-CYL, TensorFVM).

Every number is read from the collected raw artifacts -- nothing is transcribed
by hand:

  artifacts/cpd{12,20,32}/result.json     computed Cd_mean / St / Cl amp
  artifacts/cpd{12,20,32}/benchmark.json  reference + tolerance + pass flags
  artifacts/cpd32/force_history_head.csv  first rows of the diverging run
  artifacts/cpd20/force_history.csv       the shedding signal behind the numbers

Usage: python3 docs/audit/board-audit-m21018/make_figures.py
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
FIG = ROOT / "figures"
GRIDS = (12, 20, 32)


def load(cpd: int, name: str) -> dict:
    return json.loads((ART / f"cpd{cpd}" / name).read_text())


def fig_divergence() -> Path:
    """cpd32 (960x640) drag budget vs time: exponential growth -> inf -> NaN."""
    rows = list(csv.DictReader((ART / "cpd32" / "force_history_head.csv").open()))
    t, drag, nan_from = [], [], None
    for r in rows:
        tt, dd = float(r["time"]), float(r["drag"])
        if not math.isfinite(dd) or abs(dd) > 1e30:
            nan_from = tt if nan_from is None else nan_from
            continue
        t.append(tt)
        drag.append(abs(dd) if abs(dd) > 0 else float("nan"))
    res = load(32, "result.json")
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150)
    ax.semilogy(t, drag, "o-", color="#c0392b", ms=4, lw=1.2,
                label="|Cd| from the control-volume momentum budget")
    if nan_from is not None:
        ax.axvline(nan_from, color="k", ls="--", lw=1.0)
        ax.annotate(f"non-finite from t={nan_from:.2f}\n"
                    f"({sum(1 for r in rows) - len(t)} of {len(rows)} sampled rows non-finite)\n"
                    "run keeps stepping, gate calls it\n\"no vortex shedding\"",
                    xy=(nan_from, ax.get_ylim()[1] * 1e-3), xytext=(0.5, 0.45),
                    textcoords="axes fraction", fontsize=7.5,
                    arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_xlabel("t  (tU/D)")
    ax.set_ylabel("|Cd|")
    ax.set_title("m21018 cpd32 (960x640): force budget diverges at step ~15\n"
                 f"steps={res['steps']}  reported cd_mean={res['cd_mean']}  "
                 f"verdict=\"{res['gate_reasons'][0][:38]}...\"", fontsize=8.5)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    out = FIG / "fig1_cpd32_divergence.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_ladder() -> Path:
    """Relative error of the two registered benchmark metrics vs the tolerance band."""
    ref = load(20, "benchmark.json")["reference"]["values"]
    band = load(20, "benchmark.json")["reference"]["tolerance_band"]
    tol_cd = band["cd_mean"] / ref["cd_mean"]
    tol_st = band["st"] / ref["st"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), dpi=150)
    for ax, key, tol, label in ((axes[0], "cd_mean", tol_cd, "Cd_mean"),
                                (axes[1], "st", tol_st, "St")):
        xs, ys = [], []
        for cpd in GRIDS:
            r = load(cpd, "result.json")
            v = r.get(key)
            if v is None or not math.isfinite(v):
                continue
            xs.append(cpd)
            ys.append(100 * (v / ref[key] - 1))
        ax.axhspan(-100 * tol, 100 * tol, color="#2ecc71", alpha=0.18,
                   label=f"tolerance +/-{100 * tol:.1f} %")
        ax.plot(xs, ys, "o--", color="#2c3e50", label="got vs reference")
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:+.2f} %", (x, y), textcoords="offset points",
                        xytext=(4, 5), fontsize=7.5)
        bad = [cpd for cpd in GRIDS if (lambda r: r.get(key) is None or not math.isfinite(r.get(key)))(
            load(cpd, "result.json"))]
        if bad:
            ax.annotate("32 cpd: not finite\n(run diverged)", xy=(32, 0.0),
                        textcoords="offset points", xytext=(-52, 18), fontsize=7,
                        color="#c0392b", arrowprops=dict(arrowstyle="->", lw=0.7,
                                                         color="#c0392b"))
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel("cells / D")
        ax.set_ylabel("relative error  (%)")
        ax.set_xticks(list(GRIDS))
        ax.set_title(f"{label}: ref {ref[key]:.3f}", fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("m21018 benchmark error vs registered tolerance "
                 "(tensor_benchmark_cases: cd_mean 5 %, st 3 %)", fontsize=9)
    out = FIG / "fig2_ladder_errors.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_signal() -> Path:
    """cpd20 lift/drag record -- the physical shedding the numbers come from."""
    t, cd, cl = [], [], []
    with (ART / "cpd20" / "force_history.csv").open() as fh:
        for r in csv.DictReader(fh):
            try:
                d, l = float(r["drag"]), float(r["lift"])
            except ValueError:
                continue
            if math.isfinite(d) and math.isfinite(l):
                t.append(float(r["time"]))
                cd.append(d)
                cl.append(l)
    res = load(20, "result.json")
    w = res.get("statistics_window") or [t[0], t[-1]]
    fig, ax = plt.subplots(figsize=(7.4, 3.6), dpi=150)
    ax.plot(t, cd, lw=0.8, color="#2980b9", label="Cd")
    ax.plot(t, cl, lw=0.8, color="#e67e22", label="Cl")
    ax.axvspan(w[0], w[1], color="#95a5a6", alpha=0.2,
               label=f"statistics window {w[0]:.1f}-{w[1]:.1f}")
    ax.axhline(res["cd_mean"], color="#2980b9", ls=":", lw=0.9,
               label=f"Cd_mean = {res['cd_mean']:.4f}")
    ax.set_xlabel("t  (tU/D)")
    ax.set_ylabel("force coefficient")
    ax.set_title(f"m21018 cpd20 (600x400): shedding St={res['st']:.4f} "
                 f"(FFT {res['st_fft']:.4f}), Cl half-amplitude {res['cl_amplitude']:.4f}",
                 fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper left")
    out = FIG / "fig3_cpd20_signal.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    for p in (fig_divergence(), fig_ladder(), fig_signal()):
        print(p, p.stat().st_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())