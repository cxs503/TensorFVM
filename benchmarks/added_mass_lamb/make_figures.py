#!/usr/bin/env python3
"""Figures for `benchmarks/added_mass_lamb` (matplotlib, no hand-entered numbers).

Reads the three artefacts produced by ``run_benchmark.py`` and writes:

    figures/convergence.png        |relative error| vs h per case (log-log) + slope guides
    figures/deviation_finest.png   deviation of every component at the finest mesh, +-5 % band
    figures/quadrature_study.png   centroid vs gauss surface rule on identical meshes
    figures/anchor_limits.png      slender-prolate and flat-oblate limit ladders

Usage
-----
    PYTHONPATH=src python benchmarks/added_mass_lamb/make_figures.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent


def load(name: str) -> dict:
    return json.loads((HERE / name).read_text())


def fig_convergence(conv: dict, out: pathlib.Path) -> pathlib.Path:
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for case in conv["cases"]:
        worst = max(case["components"], key=lambda c: abs(c["finest_dev_pct"]))
        h = case["h_rel"]
        err = [abs(e) for e in worst["rel_error_by_level"]]
        ax.loglog(h, err, "o-", ms=3.5, lw=1.2,
                  label=f"{case['label']} (axis {worst['axis']})")
    h0 = conv["cases"][0]["h_rel"]
    if h0:
        ref = [abs(conv["cases"][0]["components"][0]["rel_error_by_level"][-1])]
        hh = [h0[-1], h0[0]]
        ax.loglog(hh, [ref[0] * (x / hh[0]) for x in hh], "k--", lw=0.9, label="slope 1")
        ax.loglog(hh, [ref[0] * (x / hh[0]) ** 2 for x in hh], "k:", lw=0.9, label="slope 2")
    ax.set_xlabel("representative panel size  h = 1/sqrt(N panels)")
    ax.set_ylabel("|relative error| of the worst axis")
    ax.set_title("Source-panel added mass on Lamb ellipsoids: mesh convergence")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize="x-small", ncol=2)
    fig.tight_layout()
    path = out / "convergence.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_deviation(conv: dict, tol: float, out: pathlib.Path) -> pathlib.Path:
    labels, values, colors = [], [], []
    for case in conv["cases"]:
        for comp in case["components"]:
            labels.append(f"{case['label']}\nax{comp['axis']}")
            values.append(comp["finest_dev_pct"])
            colors.append("#c0504d" if abs(comp["finest_dev_pct"]) > tol else "#4f81bd")
    fig, ax = plt.subplots(figsize=(11.0, 4.6))
    idx = range(len(values))
    ax.bar(idx, values, color=colors, width=0.75)
    ax.axhline(tol, color="k", ls="--", lw=1.0, label=f"+{tol:g} % limit")
    ax.axhline(-tol, color="k", ls="--", lw=1.0, label=f"-{tol:g} % limit")
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_xticks(list(idx))
    ax.set_xticklabels(labels, rotation=90, fontsize=6.5)
    ax.set_ylabel("k_panel / k_anchor - 1  [%]")
    ax.set_title("Deviation at the finest mesh, every case and axis (red = outside the band)")
    ax.legend(fontsize="small")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = out / "deviation_finest.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_quadrature(conv: dict, out: pathlib.Path) -> pathlib.Path | None:
    quad = conv.get("quadrature_study") or []
    if not quad:
        return None
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    labels = [f"{q['case']}\n{q['level']}" for q in quad]
    idx = list(range(len(quad)))
    width = 0.38
    cent = [min(q["rel_error_centroid_pct"], key=abs) for q in quad]
    gaus = [min(q["rel_error_gauss_pct"], key=abs) for q in quad]
    ax.bar([i - width / 2 for i in idx], cent, width, label="centroid rule", color="#c0504d")
    ax.bar([i + width / 2 for i in idx], gaus, width, label="gauss rule", color="#4f81bd")
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_xticks(idx)
    ax.set_xticklabels(labels, fontsize="small")
    ax.set_ylabel("worst |relative error| [%]")
    ax.set_title("Surface-integration rule vs self-panel bias (same body, same mesh)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = out / "quadrature_study.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_anchor_limits(anchors: dict, out: pathlib.Path) -> pathlib.Path:
    prolate = next(r for r in anchors["limit_checks"] if r["limit"] == "slender prolate")
    oblate = next(r for r in anchors["limit_checks"] if r["limit"] == "flat oblate disc")
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    bp = [e["b_over_a"] for e in prolate["value"]]
    axes[0].semilogx(bp, [e["k_axial"] for e in prolate["value"]], "o-", label="k_axial -> 0")
    axes[0].semilogx(bp, [e["k_transverse"] for e in prolate["value"]], "s-",
                     label="k_transverse -> 1")
    axes[0].axhline(0.5, color="k", ls=":", lw=0.9, label="sphere k = 0.5")
    axes[0].axhline(1.0, color="k", ls="--", lw=0.9)
    axes[0].set_xlabel("minor / major axis  b/a")
    axes[0].set_ylabel("added-mass coefficient k")
    axes[0].set_title("Slender prolate limit (Lamb closed form)")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize="small")

    co = [e["c_over_a"] for e in oblate["value"]]
    axes[1].semilogx(co, [e["M_normal_over_disc"] for e in oblate["value"]], "o-",
                     label="M_33 / (8/3 rho a^3)")
    axes[1].axhline(1.0, color="k", ls="--", lw=0.9, label="disc limit 1")
    axes[1].set_xlabel("flattening  c/a")
    axes[1].set_ylabel("normal added mass / disc value")
    axes[1].set_title("Flat oblate disc limit (k_3 diverges, M_33 -> 8/3 rho a^3)")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(fontsize="small")
    fig.tight_layout()
    path = out / "anchor_limits.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> int:
    out = HERE / "figures"
    out.mkdir(exist_ok=True)
    anchors = load("anchors.json")
    conv = load("convergence.json")
    tol = float(conv.get("tol_pct", 5.0))
    made = [fig_convergence(conv, out),
            fig_deviation(conv, tol, out),
            fig_anchor_limits(anchors, out)]
    quad = fig_quadrature(conv, out)
    if quad is not None:
        made.append(quad)
    for path in made:
        print(f"[figure] {path} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())