"""Figures for the k-omega SST RANS adjoint report (reads result.json)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("docs/sst-rans-adjoint")


def main(out_dir: Path = OUT):
    out_dir = Path(out_dir)
    res = json.loads((out_dir / "result.json").read_text())
    figs = []

    # ---- 1. gradient comparison (laminar + SST nut) ---------------------
    lam = res["blocks"]["laminar"]
    sst = res["blocks"]["sst_nut"]
    q = np.arange(1, 9)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].bar(q - 0.2, lam["gradient_richardson"], 0.4, label="central diff. + Richardson")
    axes[0].bar(q + 0.2, lam["gradient_adjoint"], 0.4, label="discrete adjoint (exact)")
    axes[0].set_title(f"laminar (max rel. err {lam['max_rel_err_vs_richardson']:.1e})")
    axes[1].bar(q - 0.2, sst["gradient_fd_full_richardson"], 0.4, label="central diff. + Richardson")
    axes[1].bar(q + 0.2, sst["gradient_full_differential"], 0.4, label="full-diff. adjoint")
    axes[1].set_title(f"SST $\\nu_t(\\theta)$ (max rel. err {sst['max_rel_err_full_vs_fd']:.1e})")
    for ax in axes:
        ax.set_xlabel("shape parameter $q$ (Fourier mode)")
        ax.set_ylabel("$\\partial C_D/\\partial\\theta_q$")
        ax.set_xticks(q)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle("Adjoint vs re-converged central differences, 8 shape parameters", fontsize=10)
    fig.tight_layout()
    f1 = out_dir / "fig1_gradient_verification.png"
    fig.savefig(f1, dpi=140)
    figs.append(f1.name)

    # ---- 2. frozen turbulence bias --------------------------------------
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    w = 0.38
    ax.bar(q - w / 2, sst["gradient_full_differential"], w, label="full differential (incl. $d\\nu_t/d\\theta$)")
    ax.bar(q + w / 2, sst["gradient_frozen"], w, label="frozen turbulence")
    ax2 = ax.twinx()
    ax2.plot(q, 100 * np.asarray(sst["frozen_bias_rel"]), "k.-", label="relative bias [%]")
    ax2.set_ylabel("frozen-turbulence bias [%]")
    ax.set_xlabel("shape parameter $q$")
    ax.set_ylabel("$\\partial C_D/\\partial\\theta_q$")
    ax.set_xticks(q)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")
    ax.set_title("Frozen-turbulence bias (prescribed SST-$\\nu_t(\\theta)$ closure)", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    f2 = out_dir / "fig2_frozen_bias.png"
    fig.savefig(f2, dpi=140)
    figs.append(f2.name)

    # ---- 3. grid convergence -------------------------------------------
    rows = res["blocks"]["grid"]["rows"]
    hs = np.array([r["h"] for r in rows])
    Js = np.array([abs(r["J"]) for r in rows])
    gs = np.array([r["grad_norm"] for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    pJ = res["blocks"]["grid"]["observed_order_CD"]
    pG = res["blocks"]["grid"]["observed_order_grad_norm"]
    axes[0].loglog(hs, Js, "o-", label=f"$C_D$ (order {pJ:.2f})")
    axes[0].loglog(hs, Js[-1] * (hs / hs[-1]) ** pJ, "k--", lw=0.8, label=f"$h^{{{pJ:.1f}}}$")
    axes[0].set_xlabel("$h$")
    axes[0].set_ylabel("$C_D$")
    axes[0].legend(fontsize=8)
    axes[1].loglog(hs, gs, "s-", label=f"$\\|\\partial C_D/\\partial\\theta\\|$ (order {pG:.2f})")
    axes[1].loglog(hs, gs[-1] * (hs / hs[-1]) ** pG, "k--", lw=0.8, label=f"$h^{{{pG:.1f}}}$")
    axes[1].set_xlabel("$h$")
    axes[1].set_ylabel("gradient norm")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.3, which="both")
    fig.suptitle("Grid convergence (SST $\\nu_t(\\theta)$ closure)", fontsize=10)
    fig.tight_layout()
    f3 = out_dir / "fig3_grid_convergence.png"
    fig.savefig(f3, dpi=140)
    figs.append(f3.name)

    # ---- 4. coupled SST solve histories (honest failure record) --------
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for rec in res["blocks"]["coupled_sst"]:
        if "history" not in rec:
            continue
        hist = np.asarray(rec["history"], float)
        ax.semilogy(np.arange(hist.size), hist, "-",
                    label=f"Re={rec['re']:.0e} {rec['solver']} (rel {rec['relative_residual']:.1e})")
    ax.set_xlabel("iteration")
    ax.set_ylabel("$\\|R\\|_2$")
    ax.set_title("Coupled k-$\\omega$ SST steady solve: no machine-precision steady state", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    f4 = out_dir / "fig4_coupled_sst_history.png"
    fig.savefig(f4, dpi=140)
    figs.append(f4.name)

    # ---- 5. nu_t field of the prescribed SST closure -------------------
    from tensorfvm.adjoint_shape.sst_study import make_theta, sst_profile_nut

    theta = np.asarray(res["config"]["theta_base"])
    nut = sst_profile_nut()
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    # reconstruct the grid used for the main case
    nx, ny = res["config"]["nx"], res["config"]["ny"]
    from tensorfvm.adjoint_shape.sst_rans import make_case

    case = make_case(nx=nx, ny=ny, re=res["config"]["re"], u_mean=1.0,
                     radius0=res["config"]["radius0"],
                     center=tuple(res["config"]["center"]), turbulent=False)
    nutf = nut(case._theta_t(theta), case).detach().numpy() / case.nu
    im = ax.imshow(nutf, origin="lower",
                   extent=[0, case.grid.length, 0, case.grid.height], aspect="equal")
    ax.contour(np.linspace(0, case.grid.length, nx),
               np.linspace(0, case.grid.height, ny), nutf, levels=6, colors="w", linewidths=0.5)
    fig.colorbar(im, ax=ax, label="$\\nu_t/\\nu$")
    ax.set_title("SST-closure $\\nu_t$ on the prescribed profile (main case)", fontsize=9)
    fig.tight_layout()
    f5 = out_dir / "fig5_nut_field.png"
    fig.savefig(f5, dpi=140)
    figs.append(f5.name)
    print("figures:", figs)
    return figs


if __name__ == "__main__":  # pragma: no cover
    main()