"""Figures for the forecast-v4 report (``docs/xval-round2/FORECAST-V4.md``).

    PYTHONPATH=src python docs/xval-round2/causal_forecast_v4_figures.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorfvm.convergence_diagnosis import DiagnosisConfig
from tensorfvm.verification import convergence_forecast_v4 as v4mod

HERE = Path(__file__).resolve().parent
DATA = HERE / "causal_forecast_v4.json"
HIST = HERE.parent / "convergence-diagnosis-v2" / "histories"
LEADS = ("0.5", "0.25", "0.1")
ARMS = ("v1", "v3", "v4")
COLOR = {"v1": "#4c72b0", "v3": "#dd8452", "v4": "#55a868"}


def fig_arms(payload: dict) -> Path:
    agg = payload["aggregate"]
    n = agg["n_decaying"]
    labels = [f"lead {float(l):.0%}" for l in LEADS] + ["first publish"]
    x = np.arange(len(labels))
    w = 0.8 / len(ARMS)
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    for i, arm in enumerate(ARMS):
        vals = ([agg["by_lead"][l][arm]["pass"] for l in LEADS]
                + [agg["first_publish"][arm]["pass"]])
        worst = ([agg["by_lead"][l][arm]["worst"] for l in LEADS]
                 + [agg["first_publish"][arm]["worst"]])
        bars = ax.bar(x + (i - 1) * w, vals, w, label=arm, color=COLOR[arm],
                      edgecolor="black", linewidth=0.5)
        for b, v, ws in zip(bars, vals, worst):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.08,
                    f"{v}/{n}\n{100 * ws:.0f}%", ha="center", va="bottom",
                    fontsize=7)
    ax.axhline(n, color="grey", ls=":", lw=0.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel(f"cases inside 15 % (of {n})")
    ax.set_title("Point forecast: cases passing 15 % at each lead\n"
                 "(second line = worst-case relative error)")
    ax.set_ylim(0, n + 1.6)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = HERE / "fig_forecast_v4_arms.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def fig_coverage(payload: dict) -> Path:
    agg = payload["aggregate"]
    confs = ("0.68", "0.90")
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ax = axes[0]
    x = np.arange(len(confs))
    w = 0.35
    for i, arm in enumerate(("v3", "v4")):
        vals = [agg["coverage_all_prefixes"][arm][c]["coverage"] or 0.0 for c in confs]
        widths = [agg["coverage_all_prefixes"][arm][c]["median_case_rel_width"]
                  for c in confs]
        bars = ax.bar(x + (i - 0.5) * w, vals, w, label=arm, color=COLOR[arm],
                      edgecolor="black", linewidth=0.5)
        for b, v, wd in zip(bars, vals, widths):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02,
                    f"{100 * v:.0f}%\nw={100 * (wd or 0):.0f}%", ha="center",
                    fontsize=8)
    ax.axhline(1.0, color="grey", ls=":", lw=0.8)
    ax.set_xticks(x, [f"nominal {float(c):.0%}" for c in confs])
    ax.set_ylabel("measured coverage (all prefixes)")
    ax.set_ylim(0, 1.25)
    ax.set_title("Calibrated vs parameter-only interval\n(measured on every published prefix)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    x = np.arange(len(LEADS))
    for i, c in enumerate(confs):
        vals = [agg["by_lead"][l]["coverage_strict_lead"]["v4"][c]["coverage"] or 0.0
                for l in LEADS]
        ax.plot(x, vals, "o-", color=COLOR["v4"], alpha=1.0 - 0.35 * i,
                label=f"v4, nominal {float(c):.0%}")
        vals3 = [agg["by_lead"][l]["coverage_strict_lead"]["v3"][c]["coverage"] or 0.0
                 for l in LEADS]
        ax.plot(x, vals3, "s--", color=COLOR["v3"], alpha=1.0 - 0.35 * i,
                label=f"v3, nominal {float(c):.0%}")
    ax.set_xticks(x, [f"lead {float(l):.0%}" for l in LEADS])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("coverage at the scored prefix")
    ax.set_title("Coverage at the single prefix used for scoring")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = HERE / "fig_forecast_v4_coverage.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def _series(case: str):
    hist = json.loads((HIST / f"{case}.json").read_text())
    its = [h["iteration"] for h in hist["history"]]
    res = [h["residual"] for h in hist["history"]]
    cfg = DiagnosisConfig(tolerance=hist["tolerance"], rel_tol=hist["rel_tol"],
                          max_iter=2000, window=5, min_points=6,
                          ratio_spread_tol=0.20, min_decades=1.0,
                          rate_stability_tol=0.05)
    out: dict = {}
    series = v4mod.causal_series(res, its, cfg, errors_out=out)
    return series, out["errors"], hist


def fig_trajectory(payload: dict) -> Path:
    cases = ["cd-pe10-gmres", "poisson-n32-tol1e-8"]
    stops = {r["case"]: r["stopping_iteration"] for r in payload["cases"]}
    fig, axes = plt.subplots(1, len(cases), figsize=(12.0, 4.2))
    for ax, case in zip(axes, cases):
        series, _, _ = _series(case)
        ks = [s["k"] for s in series if s["ready"]]
        fc = [s["forecast"] for s in series if s["ready"]]
        lo68 = [(s["interval"].get("0.68") or {}).get("lo") for s in series]
        hi68 = [(s["interval"].get("0.68") or {}).get("hi") for s in series]
        kk = [s["k"] for s in series]
        lo68 = [v if v is not None else np.nan for v in lo68]
        hi68 = [v if v is not None else np.nan for v in hi68]
        ax.fill_between(kk, lo68, hi68, color=COLOR["v4"], alpha=0.25,
                        label="v4 68 % band")
        ax.plot(ks, fc, color=COLOR["v4"], lw=1.4, label="v4 point forecast")
        ax.axhline(stops[case], color="black", lw=1.0, ls="--",
                   label=f"true stop = {stops[case]}")
        ax.set_xlabel("prefix iteration k (forecast issued here)")
        ax.set_ylabel("predicted stopping iteration")
        ax.set_title(case)
        ax.set_ylim(0, 1.15 * max(stops[case], 100))
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(alpha=0.3)
    fig.suptitle("v4 forecast and calibrated 68 % band along the solve "
                 "(band wide while the model has no skill, tight near the stop)")
    fig.tight_layout()
    out = HERE / "fig_forecast_v4_trajectory.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def fig_scalelaw(payload: dict) -> Path:
    cases = [r for r in payload["cases"] if r.get("scale_law") and r["decaying"]]
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for i, r in enumerate(cases):
        prof = r["scale_law"]["constant"]["profile"]
        hs = np.array(sorted(int(h) for h in prof), dtype=float)
        rms = np.array([prof[str(int(h))]["rms"] for h in hs])
        if hs.size < 3:
            continue
        p = r["scale_law"]["constant"]["exponent"]
        ax.plot(hs, rms, "o", ms=4, label=f"{r['case']} (p={p:.2f})")
        ax.plot(hs, rms[0] * (hs / hs[0]) ** p, "-", lw=1.0, alpha=0.7)
    href = np.array([1.0, 900.0])
    ax.plot(href, 0.03 * np.sqrt(href), "k--", lw=1.2,
            label="sqrt(h) law assumed by v3-style bands")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("forecast horizon h (iterations ahead)")
    ax.set_ylabel("RMS of log-residual extrapolation error")
    ax.set_title("Measured error growth law: RMS ~ h^p with p in "
                 f"[{min(r['scale_law']['constant']['exponent'] for r in cases):.2f}, "
                 f"{max(r['scale_law']['constant']['exponent'] for r in cases):.2f}]\n"
                 "-- not the sqrt(h) a random-walk argument would suggest")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out = HERE / "fig_forecast_v4_scalelaw.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def main() -> None:
    payload = json.loads(DATA.read_text())
    for fn in (fig_arms, fig_coverage, fig_trajectory, fig_scalelaw):
        print("wrote", fn(payload).name)


if __name__ == "__main__":
    main()