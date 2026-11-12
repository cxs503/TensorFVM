"""Regenerate the missing result figures from the delivered run artifacts (m16077 audit).

The m16077 report (tensor_reports id 15224) lists two figures with captions but no
PNG payload, and `scripts/ethier-steinman-3d/figures/*.png` never entered the merge
commit e6e254fd.  This script rebuilds both figures straight from the committed
npz artifacts, which doubles as an audit check that the artifacts are real and
internally consistent.

Outputs: figures/convergence.png, figures/re1-slice-n6.png
"""

from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REFS = HERE / "refs"
OUT = HERE / "figures"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    study = json.loads((REFS / "study.json").read_text())
    rows = [r for r in study["records"] if r.get("converged") and r.get("case") == "re1"]
    rows.sort(key=lambda r: r["n"])
    n = np.array([r["n"] for r in rows], dtype=float)
    dx = 2.0 / n
    e_linf = np.array([r["velocity_relative_linf"] for r in rows])
    e_l2 = np.array([r["velocity_relative_l2"] for r in rows])

    # --- figure 1: convergence ladder against the pre-registered gate -------------
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.loglog(dx, e_linf, "o-", label="velocity $L_\\infty$ relative error (gated)", lw=1.6)
    ax.loglog(dx, e_l2, "s--", label="velocity $L_2$ relative error", lw=1.2, alpha=0.8)
    ax.axhline(1e-2, color="crimson", ls=":", lw=1.4, label="mission gate < 1%")
    ax.axhline(3e-2, color="darkorange", ls=":", lw=1.0, label="3% acceptability bar")
    # slope guide from the measured pair
    order = np.log(e_linf[0] / e_linf[-1]) / np.log(dx[0] / dx[-1])
    ax.loglog(
        dx,
        e_linf[-1] * (dx / dx[-1]) ** order,
        "k--",
        lw=0.9,
        alpha=0.6,
        label=f"measured order p={order:.2f}",
    )
    ax.set_xlabel("cell size $dx$ (-1..1 box, $n^3$ cells)")
    ax.set_ylabel("relative error vs exact field (-)")
    ax.set_title("Ethier-Steinman 3-D, Re=1, frozen t=0.25 s\n(rebuilt from committed npz artifacts)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "convergence.png", dpi=140)
    plt.close(fig)

    # --- figure 2: mid-plane slice, computed vs exact, n=6 ------------------------
    z = np.load(REFS / "re1-n6.npz")
    c = z["cell_centers"]
    u = z["velocity"]
    uref = z["reference_velocity"]
    tol = 1e-9
    mask = np.abs(c[:, 0] - np.round(c[:, 0] / (2.0 / 6)) * (2.0 / 6)) < tol
    mask &= np.abs(c[:, 0]) < 1.0 / 6.0  # x ~ 0 plane
    idx = np.where(mask)[0]
    if idx.size == 0:  # fall back: cells nearest to x=0
        idx = np.argsort(np.abs(c[:, 0]))[: len(c) // 6]
    y, zc = c[idx, 1], c[idx, 2]

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6))
    vmax = np.max(np.abs(uref[idx, 0]))
    for ax, field, title in (
        (axes[0], uref[idx, 0], "exact $u_x$"),
        (axes[1], u[idx, 0], "computed $u_x$"),
        (axes[2], u[idx, 0] - uref[idx, 0], "error $u_x^{h}-u_x^{exact}$"),
    ):
        sc = ax.scatter(y, zc, c=field, s=48, cmap="coolwarm")
        if title.startswith("error"):
            sc.set_clim(-np.max(np.abs(field)), np.max(np.abs(field)))
        else:
            sc.set_clim(-vmax, vmax)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("y")
        ax.set_ylabel("z")
        fig.colorbar(sc, ax=ax, shrink=0.85)
    fig.suptitle("Re=1, n=6, cell plane nearest x=0 (from committed npz)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "re1-slice-n6.png", dpi=140)
    plt.close(fig)

    print("order_from_artifacts", round(float(order), 4))
    print("wrote", OUT / "convergence.png", OUT / "re1-slice-n6.png")


if __name__ == "__main__":
    main()