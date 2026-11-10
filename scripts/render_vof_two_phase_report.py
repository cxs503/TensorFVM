#!/usr/bin/env python
"""Render the figures and the field diagnostics for the VOF verification report.

Produces (into ``docs/vof-two-phase-verification/artifacts/figures``):

``fig1_zalesak_fields.png``   exact / PLIC / FCT volume fraction after one revolution
``fig2_single_vortex.png``    exact / PLIC / difference for the Rider-Kothe anchor
``fig3_double_vortex.png``    exact / PLIC / difference for the two-cell anchor
``fig4_convergence.png``      interface error vs grid size for every scheme
``fig5_hydrostatic.png``      rest-state pressure deviation, shipped vs well-balanced

and ``artifacts/field_diagnostics.json`` with the bulk diagnostics that explain
the residual interface error (mask-area error and centroid offset), which the
scalar norms in ``campaign.json`` cannot show on their own.

Usage::

    python scripts/render_vof_two_phase_report.py [--quick]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensorfvm.vof_verification.cases import (  # noqa: E402
    Transport,
    advance,
    double_vortex_case,
    single_vortex_case,
    zalesak_case,
)

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "docs" / "vof-two-phase-verification" / "artifacts"
FIG = ART / "figures"


def bulk_diagnostics(alpha: torch.Tensor, exact: torch.Tensor,
                     dx: float, dy: float) -> dict:
    """Mask-area error and centroid offset of the bulk (alpha > 1/2) region."""
    num = alpha > 0.5
    ref = exact > 0.5
    area_num = float(num.sum().item()) * dx * dy
    area_ref = float(ref.sum().item()) * dx * dy
    centre = torch.arange(alpha.shape[1], dtype=torch.float64) + 0.5
    cent_y = torch.arange(alpha.shape[0], dtype=torch.float64) + 0.5
    wx = alpha / alpha.sum()
    cx = float((wx.sum(dim=0) * centre).sum().item()) * dx
    cy = float((wx.sum(dim=1) * cent_y).sum().item()) * dy
    we = exact / exact.sum()
    cxe = float((we.sum(dim=0) * centre).sum().item()) * dx
    cye = float((we.sum(dim=1) * cent_y).sum().item()) * dy
    return {
        "bulk_area_error": abs(area_num - area_ref) / area_ref,
        "centroid_offset": ((cx - cxe) ** 2 + (cy - cye) ** 2) ** 0.5,
        "centroid_numeric": [cx, cy],
        "centroid_exact": [cxe, cye],
    }


def _panel(ax, field, title, vmin=0.0, vmax=1.0, cmap="viridis"):
    im = ax.imshow(field, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def _json_figures() -> int:
    """Figures that only need the campaign JSONs (no transport re-runs)."""
    runs = [json.loads(p.read_text())
            for p in sorted((ART / "runs").glob("run_*.json"))]
    cases = ("zalesak", "single_vortex", "double_vortex")
    labels = {"plic": "PLIC", "fct": "FCT (Phase-0)", "donor": "donor cell"}
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))
    for ax, case in zip(axes, cases):
        for scheme in ("plic", "fct", "donor"):
            rows = sorted([r for r in runs
                           if r["case"] == case and r["scheme"] == scheme],
                          key=lambda r: r["cells"][0])
            if not rows:
                continue
            ax.loglog([r["cells"][0] for r in rows],
                      [r["symmetric_difference"] for r in rows], "o-",
                      label=labels[scheme])
        ax.set_title(case.replace("_", " "), fontsize=9)
        ax.set_xlabel("cells in x")
        ax.set_ylabel("symmetric-difference error")
        ax.axhline(0.05, color="k", ls=":", lw=1)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("interface deviation vs resolution (dotted: 5 % criterion)",
                 fontsize=10)
    fig.savefig(FIG / "fig4_convergence.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    sweep_path = ART / "deformation_sensitivity.json"
    if sweep_path.exists():
        sweep = json.loads(sweep_path.read_text())
        rows = sweep["rows"] + [r for r in runs if r["scheme"] == "plic"
                                and r["case"].endswith("vortex")]
        fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))
        for ax, case in zip(axes, ("single_vortex", "double_vortex")):
            sel = [r for r in rows if r["case"] == case]
            for nx, style in ((64, "o-"), (128, "s--")):
                line = sorted([r for r in sel if r["cells"][0] == nx],
                              key=lambda r: r.get("T", 8.0))
                if not line:
                    continue
                ax.semilogy([r.get("T", 8.0) for r in line],
                            [r["symmetric_difference"] for r in line], style,
                            label=f"{nx}x{line[0]['cells'][1]}")
            ax.axhline(0.05, color="k", ls=":", lw=1)
            ax.set_xlabel("reversal window T (deformation magnitude)")
            ax.set_ylabel("symmetric-difference error")
            ax.set_title(case.replace("_", " "), fontsize=9)
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(fontsize=7)
        fig.suptitle("error grows with the accumulated winding (dotted: 5 %)",
                     fontsize=10)
        fig.savefig(FIG / "fig6_deformation_sensitivity.png", dpi=140,
                    bbox_inches="tight")
        plt.close(fig)
    print("json-driven figures written")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="coarser, faster fields")
    ap.add_argument("--skip-fields", action="store_true",
                    help="only re-plot the JSON-driven figures")
    args = ap.parse_args(argv)
    FIG.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(__import__("os").environ.get("TFVM_THREADS", "8")))

    zalesak_n = 64 if args.quick else 128
    vortex_n = 48 if args.quick else 64
    dv_n = 24 if args.quick else 32

    diagnostics: dict = {"fields": {}}
    if args.skip_fields:
        return _json_figures() or 0

    # ---- Zalesak: PLIC vs FCT vs exact ------------------------------------
    z = zalesak_case()
    out_plic = advance(z, zalesak_n, zalesak_n, Transport("plic", 0.75))
    out_fct = advance(z, zalesak_n, zalesak_n, Transport("fct", 0.75))
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    _panel(axes[0], out_plic["exact"].numpy(), "exact (analytic)")
    im = _panel(axes[1], out_plic["alpha"].numpy(), f"PLIC, {zalesak_n}$^2$")
    _panel(axes[2], out_fct["alpha"].numpy(), f"FCT (Phase-0), {zalesak_n}$^2$")
    fig.colorbar(im, ax=axes, shrink=0.8, label=r"$\alpha$")
    fig.savefig(FIG / "fig1_zalesak_fields.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    diagnostics["fields"]["zalesak_plic"] = bulk_diagnostics(
        out_plic["alpha"], out_plic["exact"], 1.0 / zalesak_n, 1.0 / zalesak_n)

    # ---- single vortex ----------------------------------------------------
    sv = single_vortex_case()
    out_sv = advance(sv, vortex_n, vortex_n, Transport("plic", 0.75))
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    _panel(axes[0], out_sv["exact"].numpy(), "exact (initial = analytic at T)")
    im = _panel(axes[1], out_sv["alpha"].numpy(), f"PLIC after reversal, {vortex_n}$^2$")
    diff = out_sv["alpha"] - out_sv["exact"]
    im2 = axes[2].imshow(diff.numpy(), origin="lower", cmap="coolwarm",
                         vmin=-0.5, vmax=0.5, interpolation="nearest")
    axes[2].set_title(r"$\alpha_{num} - \alpha_{exact}$", fontsize=9)
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    fig.colorbar(im2, ax=axes, shrink=0.8, label=r"$\Delta\alpha$")
    fig.savefig(FIG / "fig2_single_vortex.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    diagnostics["fields"]["single_vortex_plic"] = bulk_diagnostics(
        out_sv["alpha"], out_sv["exact"], 1.0 / vortex_n, 1.0 / vortex_n)

    # ---- double vortex ----------------------------------------------------
    dv = double_vortex_case()
    out_dv = advance(dv, 2 * dv_n, dv_n, Transport("plic", 0.75))
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.0))
    _panel(axes[0], out_dv["exact"].numpy(), "exact (initial = analytic at T)")
    _panel(axes[1], out_dv["alpha"].numpy(), f"PLIC after reversal, "
                                             f"{2 * dv_n}x{dv_n}")
    im2 = axes[2].imshow((out_dv["alpha"] - out_dv["exact"]).numpy(),
                         origin="lower", cmap="coolwarm", vmin=-0.5, vmax=0.5,
                         interpolation="nearest")
    axes[2].set_title(r"$\alpha_{num} - \alpha_{exact}$", fontsize=9)
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    fig.colorbar(im2, ax=axes, shrink=0.8, label=r"$\Delta\alpha$")
    fig.savefig(FIG / "fig3_double_vortex.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    diagnostics["fields"]["double_vortex_plic"] = bulk_diagnostics(
        out_dv["alpha"], out_dv["exact"], 2.0 / (2 * dv_n), 1.0 / dv_n)

    # ---- convergence curves ----------------------------------------------
    runs = []
    for path in sorted((ART / "runs").glob("run_*.json")):
        runs.append(json.loads(path.read_text()))
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))
    cases = ("zalesak", "single_vortex", "double_vortex")
    labels = {"plic": "PLIC", "fct": "FCT (Phase-0)", "donor": "donor cell"}
    for ax, case in zip(axes, cases):
        for scheme in ("plic", "fct", "donor"):
            rows = sorted([r for r in runs
                           if r["case"] == case and r["scheme"] == scheme],
                          key=lambda r: r["cells"][0])
            if not rows:
                continue
            cells = [r["cells"][0] for r in rows]
            sym = [r["symmetric_difference"] for r in rows]
            ax.loglog(cells, sym, "o-", label=labels[scheme])
        ax.set_title(case.replace("_", " "), fontsize=9)
        ax.set_xlabel("cells in x")
        ax.set_ylabel("symmetric-difference error")
        ax.axhline(0.05, color="k", ls=":", lw=1)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("interface deviation vs resolution (dotted: 5 % criterion)", fontsize=10)
    fig.savefig(FIG / "fig4_convergence.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # ---- hydrostatic ------------------------------------------------------
    hydro_path = ART / "hydrostatic.json"
    if hydro_path.exists():
        hydro = json.loads(hydro_path.read_text())
        rows = [c for c in hydro["cases"] if c["density_ratio"] == 1000.0]
        fig, ax = plt.subplots(figsize=(5.4, 3.8))
        for depth, style in zip(sorted({c["depth"] for c in rows}), ("o-", "s-", "^-")):
            sel = [c for c in rows if c["depth"] == depth]
            sel.sort(key=lambda c: c["cells"][0])
            n = [c["cells"][0] for c in sel]
            ax.loglog(n, [c["shipped"]["pressure_oscillation"] for c in sel],
                      style, label=f"shipped void, $H_l$={depth}")
            ax.loglog(n, [max(c["well_balanced"]["pressure_oscillation"], 1e-18)
                          for c in sel], "k:", marker="x",
                      label="well-balanced" if depth == min({c["depth"] for c in rows})
                      else None)
        ax.axhline(0.005, color="r", ls="--", lw=1, label="0.5 % criterion")
        ax.set_xlabel("cells per side")
        ax.set_ylabel(r"max $|p-p_{hydro}| / (\rho_l g H_l)$")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=6)
        fig.savefig(FIG / "fig5_hydrostatic.png", dpi=140, bbox_inches="tight")
        plt.close(fig)
    diagnostics["grids"] = {"zalesak": zalesak_n, "single_vortex": vortex_n,
                            "double_vortex": [2 * dv_n, dv_n]}
    (ART / "field_diagnostics.json").write_text(json.dumps(diagnostics, indent=1))
    print(json.dumps(diagnostics, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())