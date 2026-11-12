#!/usr/bin/env python3
"""Figures for the m15458 board audit, built from ``cavity_audit.json`` only.

Every number plotted is read back from the audit JSON (which itself came from the
stored fields), so nothing is transcribed by hand.

Usage::

    PYTHONPATH=src python3 docs/audit-m15554/audit_figures.py \
        --audit docs/audit-m15554/cavity_audit.json \
        --bundle /root/tp-agent/work/m15458/artifacts/CASE-CAVITY-RE1000-FVM
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def fig_profiles(audit: dict, path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), dpi=140)
    cols = {32: "0.55", 64: "tab:orange", 128: "tab:blue"}
    for n, row in sorted(audit["rows"].items(), key=lambda kv: int(kv[0])):
        u = row["profiles"]["u"]
        axes[0].plot(u["computed"], u["coords"], color=cols[int(n)], lw=1.4,
                     label=f"TensorFVM n={n} (audit re-read)")
        v = row["profiles"]["v"]
        axes[1].plot(v["coords"], v["computed"], color=cols[int(n)], lw=1.4, label=f"n={n}")
    u64 = audit["rows"]["64"]["profiles"]["u"]
    axes[0].plot(u64["reference"], u64["coords"], "ks", ms=4.2, mfc="none",
                 label="Ghia et al. (1982) Table I")
    v64 = audit["rows"]["64"]["profiles"]["v"]
    axes[1].plot(v64["coords"], v64["reference"], "ks", ms=4.2, mfc="none",
                 label="Ghia et al. (1982) Table II")
    axes[0].set_xlabel("u / U   (x = L/2)"); axes[0].set_ylabel("y / L")
    axes[1].set_xlabel("x / L"); axes[1].set_ylabel("v / U   (y = L/2)")
    for ax in axes:
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    axes[0].set_title("vertical centreline", fontsize=10)
    axes[1].set_title("horizontal centreline", fontsize=10)
    fig.suptitle("CASE-CAVITY-RE1000-FVM — audit re-read of the stored fields vs Ghia", fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_field(bundle: str, n: int, path: str) -> dict:
    blob = torch.load(os.path.join(bundle, "fields", f"Re1000_n{n}.pt"), map_location="cpu",
                      weights_only=False)
    u = blob["u"].double().numpy()
    v = blob["v"].double().numpy()
    h = 1.0 / n
    ux = (u[1:, :] - u[:-1, :]) / h          # du/dy at cell x-faces' rows
    vy = (v[:, 1:] - v[:, :-1]) / h          # dv/dx
    omega = vy - ux                          # cell-centred vorticity
    xs = (np.arange(n) + 0.5) * h
    X, Y = np.meshgrid(xs, xs)
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.6), dpi=140)
    im0 = axes[0].pcolormesh(X, Y, omega, cmap="RdBu_r", shading="auto",
                             vmin=-np.abs(omega).max(), vmax=np.abs(omega).max())
    fig.colorbar(im0, ax=axes[0], shrink=0.85)
    ucell = 0.5 * (u[1:, :] + u[:-1, :])
    im1 = axes[1].pcolormesh(X, Y, ucell, cmap="viridis", shading="auto")
    fig.colorbar(im1, ax=axes[1], shrink=0.85)
    axes[1].set_title(f"u / U (n={n})", fontsize=10)
    axes[0].set_title(f"vorticity ω_z·L/U (n={n})", fontsize=10)
    for ax in axes:
        ax.set_xlabel("x / L"); ax.set_ylabel("y / L"); ax.set_aspect("equal")
    fig.suptitle("CASE-CAVITY-RE1000-FVM — stored field at the finest converged resolution",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return {"omega_min": float(omega.min()), "omega_max": float(omega.max()),
            "u_centre_max": float(ucell.max())}


def fig_convergence(audit: dict, path: str) -> None:
    ns = np.array(audit["convergence"]["n"], dtype=float)
    wall = np.array(audit["convergence"]["u_max_abs_error_pct_of_U"])
    inner = np.array(audit["convergence"]["u_interior_max_abs_error_pct_of_U"])
    fig, ax = plt.subplots(figsize=(6.4, 4.3), dpi=140)
    ax.loglog(ns, wall, "o-", label="max|Δu| incl. wall endpoint (Hermite/L1 extrapolation)")
    ax.loglog(ns, inner, "s-", label="max|Δu| interior (0 < y < 1)")
    ax.axhline(3.0, color="tab:red", ls="--", lw=1, label="3 % criterion")
    ax.axhline(5.0, color="tab:gray", ls=":", lw=1, label="5 % ledger band")
    for n, e in zip(ns, inner):
        ax.annotate(f"{e:.2f}%", (n, e), textcoords="offset points", xytext=(4, -12), fontsize=8)
    ax.set_xlabel("n (cells per direction)"); ax.set_ylabel("pointwise error [% of U]")
    ax.set_title("CASE-CAVITY-RE1000-FVM — error vs resolution (audit re-read)", fontsize=10)
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", default=os.path.join(HERE, "cavity_audit.json"))
    ap.add_argument("--bundle", required=True)
    a = ap.parse_args()
    with open(a.audit) as fh:
        audit = json.load(fh)
    fig_profiles(audit, os.path.join(HERE, "fig_audit_centrelines.png"))
    stats = fig_field(a.bundle, max(int(k) for k in audit["rows"]),
                      os.path.join(HERE, "fig_audit_fields.png"))
    fig_convergence(audit, os.path.join(HERE, "fig_audit_convergence.png"))
    with open(os.path.join(HERE, "fig_audit_field_stats.json"), "w") as fh:
        json.dump(stats, fh, indent=1)
    print("figures written:", sorted(f for f in os.listdir(HERE) if f.endswith(".png")), stats)