"""Render the P167W2B-N03 VOF verification figures from the gate artefacts.

Reads the gate artefacts (``docs/p167-n03/artifacts/*.json``, written by
:mod:`tensorfvm.verification.vof_cicsam_gate`) and writes the five figures of
``docs/p167-n03/report.md`` into ``docs/p167-n03/figures``.  The rotating-disk
fields are re-integrated here because the gate stores metrics, not fields.

Usage::

    PYTHONPATH=src python3 scripts/render_p167_gate_report.py [--runs DIR] [--out DIR]
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from tensorfvm.vof_cicsam import interface_thickness_cells  # noqa: E402
from tensorfvm.vof_phase0.anchors import (  # noqa: E402
    MARTIN_MOYCE_FRONT_T,
    MARTIN_MOYCE_FRONT_X,
    ritter_front,
)
from tensorfvm.vof_verification.cases import (  # noqa: E402
    Transport,
    sample_faces,
    zalesak_case,
)
from tensorfvm.vof_verification.geometry import zalesak_alpha  # noqa: E402
from tensorfvm.vof_verification.metrics import field_metrics  # noqa: E402

SCHEME_COLOURS = {"cicsam": "tab:blue", "plic": "tab:green", "fct": "tab:orange"}


def load(path_glob: str) -> list[dict]:
    out = []
    for path in sorted(glob.glob(path_glob)):
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        doc["_path"] = path
        out.append(doc)
    return out


def zalesak_fields(nx: int = 64, revolutions: float = 1.0) -> dict:
    """Re-integrate one revolution and return the fields of every scheme."""
    dx = 1.0 / nx
    case = zalesak_case()
    exact = zalesak_alpha(nx, nx, 1.0, 1.0, (0.5, 0.75), 0.15, 0.025)
    u0, _ = sample_faces(case, nx, nx, 0.0)
    dt = 0.5 * dx / float(u0.abs().max().item())
    steps = int(math.ceil(revolutions / dt))
    dt = revolutions / steps
    fields = {"exact": exact.numpy()}
    for scheme in ("cicsam", "plic", "fct"):
        transport = Transport(scheme=scheme, cfl=0.5)
        state = transport.build(nx, nx, dx, dx)
        alpha = exact.clone()
        for k in range(steps):
            u, v = sample_faces(case, nx, nx, k * dt, (k + 1) * dt)
            alpha = transport.step(state, alpha, u, v, dt, dx, dx)
        fields[scheme] = alpha.numpy()
        m = field_metrics(alpha, exact, dx * dx)
        t = interface_thickness_cells(alpha)
        fields[f"{scheme}_metrics"] = {
            "relative_l1": m["relative_l1"],
            "thickness_median": t["median"],
            "thickness_p90": t["p90"],
        }
    return fields


def fig_zalesak_fields(fields: dict, path: str) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.1), sharey=True)
    titles = ["exact", "PLIC", "CICSAM", "FCT"]
    keys = ["exact", "plic", "cicsam", "fct"]
    for ax, key, title in zip(axes, keys, titles):
        field = fields[key]
        im = ax.imshow(field, origin="lower", cmap="Blues", vmin=0.0, vmax=1.0,
                       extent=(0, 1, 0, 1))
        if key != "exact":
            ax.contour(np.linspace(0.5 / field.shape[1], 1 - 0.5 / field.shape[1],
                                   field.shape[1]),
                       np.linspace(0.5 / field.shape[0], 1 - 0.5 / field.shape[0],
                                   field.shape[0]),
                       field, levels=[0.5], colors="crimson", linewidths=0.8)
            m = fields[f"{key}_metrics"]
            ax.set_title(f"{title}\n$L_1$ = {m['relative_l1']:.3f}, "
                         f"$t_{{int}}$ = {m['thickness_p90']:.2f} cells")
        else:
            ax.set_title(title)
        ax.set_xlabel("x / L")
    axes[0].set_ylabel("y / L")
    fig.colorbar(im, ax=axes, label=r"$\alpha$", shrink=0.85)
    fig.suptitle("Zalesak slotted disk after one revolution (64$^2$, CFL 0.5): "
                 "0.5 contour in red")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_zalesak_errors(docs: list[dict], path: str) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for doc in docs:
        nx = doc["cells"][0]
        for scheme, entry in doc["schemes"].items():
            revs = sorted(int(r) for r in entry["revolutions"])
            l1 = [entry["revolutions"][str(r)]["relative_l1"] for r in revs]
            mass = [entry["revolutions"][str(r)]["mass_drift_relative"]
                    for r in revs]
            label = f"{scheme} {nx}$^2$"
            ax1.plot(revs, l1, marker="o", color=SCHEME_COLOURS.get(scheme, "k"),
                     ls="-" if nx == 64 else "--", label=label)
            ax2.plot(revs, [m if m > 0 else 1e-17 for m in mass],
                     marker="o", color=SCHEME_COLOURS.get(scheme, "k"),
                     ls="-" if nx == 64 else "--", label=label)
    ax1.set_xlabel("revolutions")
    ax1.set_ylabel(r"$L_1$ interface error / liquid volume")
    ax1.set_yscale("log")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=7)
    ax1.set_title("Shape error grows with the number of revolutions")
    ax2.axhline(1e-3, color="crimson", ls=":", label="0.1 % criterion")
    ax2.set_xlabel("revolutions")
    ax2.set_ylabel(r"$|V(T)-V(0)|/V(0)$")
    ax2.set_yscale("log")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=7)
    ax2.set_title("Mass ledger: flux form stays at roundoff")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_thickness(docs: list[dict], path: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    labels, med, p90 = [], [], []
    for doc in docs:
        nx = doc["cells"][0]
        for scheme, entry in doc["schemes"].items():
            for revs in sorted(int(r) for r in entry["revolutions"]):
                rec = entry["revolutions"][str(revs)]["thickness"]
                labels.append(f"{scheme}\n{nx}$^2$/rev {revs}")
                med.append(rec["median"])
                p90.append(rec["p90"])
    idx = np.arange(len(labels))
    ax.bar(idx - 0.2, med, width=0.4, label="median", color="tab:blue")
    ax.bar(idx + 0.2, p90, width=0.4, label="90th pct", color="tab:red")
    ax.axhline(2.0, color="k", ls=":", label="2-cell criterion")
    ax.set_xticks(idx)
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel("interface thickness [cells] (1 / max|Δα|)")
    ax.set_title("Interface sharpness after 1/2/4/8 revolutions")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_dam_break(dam: list[dict], path: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(MARTIN_MOYCE_FRONT_T, MARTIN_MOYCE_FRONT_X, "ks-", ms=4,
            label="Martin & Moyce (1952), digitised")
    a = 0.05715
    t = np.linspace(0.0, 4.0, 101)
    ax.plot(t, [(ritter_front(x, 2.0 * a, 9.81, a) - a) / a for x in t], "k:",
            lw=1.0, label="Ritter (1892) shallow-water limit")
    for row in dam:
        fx = np.array(row["front_star"], dtype=float)
        ft = np.array(row["t_star"], dtype=float)
        worst = row["evaluation"]["worst_relative_error"]
        ax.plot(ft, fx, "-", lw=1.6,
                label=f"{row['scheme']} {row['grid'][0]}x{row['grid'][1]} "
                      f"(worst {100 * worst:.0f} %)")
    ax.set_xlabel(r"$t^* = t\sqrt{g/a}$")
    ax.set_ylabel(r"$x^* = (x_f - a)/a$")
    ax.set_xlim(0, 3.2)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("Dam-break surge front: simulation vs experiment")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_hydrostatic(hydro: dict, path: str) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    runs = hydro["runs"] + [hydro["control_not_well_balanced"]]
    labels = [f"{r['cells'][0]}x{r['cells'][1]}\n"
              + ("well-balanced" if r["well_balanced_surface"] else "control")
              for r in runs]
    values = [max(r["pressure_oscillation"], 1e-18) for r in runs]
    colours = ["tab:blue" if r["well_balanced_surface"] else "tab:red"
               for r in runs]
    ax.bar(np.arange(len(runs)), values, color=colours)
    ax.axhline(1e-2, color="k", ls=":", label="1 % criterion")
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(runs)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel(r"$\max|p-p_{hydro}|/(\rho_l g H)$")
    ax.set_title("Hydrostatic rest state: $\\rho g h$ distribution error")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def tables_markdown(runs: str) -> str:
    """Emit the report's number tables straight from the gate artefacts."""
    docs = sorted([d for d in (load(os.path.join(runs, "zalesak_*.json"))
                              + load(os.path.join(runs, "*", "zalesak_*.json")))
                   if "schemes" in d], key=lambda d: d["cells"][0])
    lines = ["### Zalesak rotating disk (solid-body rotation, closed solution)", "",
             "| grid | scheme | revs | steps | mass drift | relative L1 | "
             "sym. difference | peak | thickness median/p90 [cells] |",
             "|---|---|---|---|---|---|---|---|---|"]
    for doc in docs:
        nx = doc["cells"][0]
        for scheme in ("cicsam", "plic", "fct"):
            entry = doc["schemes"].get(scheme)
            if not entry:
                continue
            for revs in sorted(int(r) for r in entry["revolutions"]):
                rec = entry["revolutions"][str(revs)]
                t = rec["thickness"]
                lines.append(
                    f"| {nx}x{nx} | {scheme} | {revs} | {rec['steps']} | "
                    f"{rec['mass_drift_relative']:.2e} | "
                    f"{rec['relative_l1']:.4f} | "
                    f"{rec['symmetric_difference']:.4f} | "
                    f"{rec['peak_retention']:.3f} | "
                    f"{t['median']:.2f} / {t['p90']:.2f} |")
    lines += ["", "### Dam break vs Martin & Moyce (1952)", "",
              "| run | t*=1 | 1.5 | 2.0 | 2.5 | 3.0 | worst | steps | mass drift |",
              "|---|---|---|---|---|---|---|---|---|"]
    dam = sorted([d for d in load(os.path.join(runs, "dam_*.json"))
                  if "front_star" in d],
                 key=lambda d: (d["scheme"], d["grid"][0]))
    ref = None
    for row in dam:
        pts = row["evaluation"]["points"]
        if ref is None:
            ref = "| reference x* | " + " | ".join(
                f"{p['reference']:.2f}" for p in pts) + " | — | — | — |"
            lines.append(ref)
        cells = " | ".join(
            "n/a" if p["measured"] is None
            else f"{p['measured']:.3f} ({100 * p['relative_error']:+.1f} %)"
            for p in pts)
        lines.append(f"| {row['scheme']} {row['grid'][0]}x{row['grid'][1]} | "
                     f"{cells} | "
                     f"{100 * row['evaluation']['worst_relative_error']:.1f} % | "
                     f"{row['steps']} | "
                     f"{row['mass_drift_relative']:.1e} |")
    hydro_path = os.path.join(runs, "hydrostatic.json")
    if os.path.exists(hydro_path):
        with open(hydro_path, encoding="utf-8") as fh:
            hydro = json.load(fh)
        lines += ["", "### Hydrostatic rest state", "",
                  "| grid | surface model | max pressure deviation | "
                  "spurious velocity | mass drift |",
                  "|---|---|---|---|---|"]
        for row in hydro["runs"] + [hydro["control_not_well_balanced"]]:
            lines.append(
                f"| {row['cells'][0]}x{row['cells'][1]} | "
                f"{'well-balanced' if row['well_balanced_surface'] else 'control'} | "
                f"{row['pressure_oscillation']:.3e} | "
                f"{row['spurious_velocity']:.3e} | "
                f"{row['mass_drift_relative']:.1e} |")
    summary = os.path.join(runs, "summary.json")
    if os.path.exists(summary):
        with open(summary, encoding="utf-8") as fh:
            doc = json.load(fh)
        lines += ["", "### Machine verdict", "",
                  "| gate | criterion | measured | threshold | verdict |",
                  "|---|---|---|---|---|"]
        names = {"G1_zalesak_mass": "G1 Zalesak mass ledger",
                 "G2_cicsam_sharpness": "G2 CICSAM sharpness",
                 "G3_cicsam_vs_plic": "G3 CICSAM vs PLIC (report only)",
                 "G4_hydrostatic_pressure": "G4 hydrostatic",
                 "G5_dam_front": "G5 dam-break front"}
        for key, row in doc["criteria"].items():
            value = row.get("value")
            lines.append(
                f"| {names.get(key, key)} | {row.get('detail', '')[:60]} | "
                f"{'—' if value is None else f'{value:.4g}'} | "
                f"{row.get('threshold', '—')} | "
                f"{'PASS' if row.get('passed') else 'FAIL'} |")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="docs/p167-n03/artifacts")
    ap.add_argument("--out", default="docs/p167-n03/figures")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)

    docs = [d for d in (load(os.path.join(args.runs, "zalesak_*.json"))
                        + load(os.path.join(args.runs, "*", "zalesak_*.json")))
            if "schemes" in d]
    docs = sorted(docs, key=lambda d: d["cells"][0])
    if docs:
        fig_zalesak_errors(docs, os.path.join(args.out, "fig2_zalesak_errors.png"))
        fig_thickness(docs, os.path.join(args.out, "fig3_interface_thickness.png"))

    fields = zalesak_fields()
    fig_zalesak_fields(fields, os.path.join(args.out, "fig1_zalesak_fields.png"))
    with open(os.path.join(args.out, "zalesak_fields_metrics.json"), "w",
              encoding="utf-8") as fh:
        json.dump({k: v for k, v in fields.items() if k.endswith("_metrics")},
                  fh, indent=1)

    dam = [d for d in load(os.path.join(args.runs, "dam_*.json"))
           if "front_star" in d]
    if dam:
        fig_dam_break(dam, os.path.join(args.out, "fig4_dam_break_front.png"))

    hydro_path = os.path.join(args.runs, "hydrostatic.json")
    if os.path.exists(hydro_path):
        with open(hydro_path, encoding="utf-8") as fh:
            hydro = json.load(fh)
        fig_hydrostatic(hydro, os.path.join(args.out, "fig5_hydrostatic.png"))

    table_path = os.path.join(os.path.dirname(args.out), "tables.md")
    with open(table_path, "w", encoding="utf-8") as fh:
        fh.write(tables_markdown(args.runs))
    print("figures written to", args.out, "| tables:", table_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())