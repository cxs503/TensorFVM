"""Figures for the compressible multi-material verification report.

Every figure is generated from the numbers the verification legs produced; no
curve is drawn from a hard-coded array.  The functions return the PNG paths so
the report can embed them and the JSON artifact can record what was plotted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

__all__ = ["make_all"]


def _save(fig, outdir: Path, name: str) -> str:
    path = outdir / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def fig_sod(leg: Dict, outdir: Path) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.4))
    data = leg.get("profile")
    if data is None:
        raise KeyError("leg_sod must store a 'profile' for plotting")
    for ax, key, label in zip(axes, ("rho", "u", "p"), (r"$\rho$", r"$u$", r"$p$")):
        ax.plot(data["x"], data[f"{key}_exact"], "-", color="k", lw=1.2, label="exact")
        ax.plot(data["x"], data[key], "o", ms=2.6, color="#1f77b4", label="MUSCL-Hancock + HLLC")
        ax.set_xlabel("x")
        ax.set_ylabel(label)
        ax.set_title(f"Sod, t = {data['t']:.2f}, n = {data['nx']}")
        ax.legend(frameon=False, fontsize=8)
    return _save(fig, outdir, "fig_sod_profiles.png")


def fig_lax(leg: Dict, outdir: Path) -> str:
    """Lax shock tube: numerical vs exact profiles + the L1 grid-convergence ladder."""
    data = leg.get("profile")
    if data is None:
        raise KeyError("leg_lax must store a 'profile' for plotting")
    rows = leg["rows"]
    fig, axes = plt.subplots(1, 4, figsize=(17.0, 3.5))
    for ax, key, label in zip(axes[:3], ("rho", "u", "p"), (r"$\rho$", r"$u$", r"$p$")):
        ax.plot(data["x"], data[f"{key}_exact"], "-", color="k", lw=1.3, label="exact")
        ax.plot(data["x"], data[key], "o", ms=2.4, color="#1f77b4",
                label="MUSCL-Hancock + HLLC (MC)")
        ax.set_xlabel("x")
        ax.set_ylabel(label)
        ax.set_title(f"Lax, t = {data['t']:.2f}, nx = {data['nx']}")
        ax.legend(frameon=False, fontsize=8)
    ax = axes[3]
    for key, color in (("rho", "#1f77b4"), ("u", "#2ca02c"), ("p", "#d62728")):
        ax.loglog([r["nx"] for r in rows], [max(r["l1_rel"][key], 1e-16) for r in rows],
                  "o-", color=color, label=r"$\rho$" if key == "rho" else
                  (r"$u$" if key == "u" else r"$p$"))
    tol = leg["thresholds"]["l1_rel_error"]
    ax.axhline(tol, ls="--", color="k", lw=1.0, label=f"{100 * tol:g} % acceptance")
    ax.set_xlabel("cells")
    ax.set_ylabel("relative L1 error")
    orders = leg.get("observed_order", {}).get("rho") or []
    ax.set_title("grid convergence, order "
                 + ", ".join(f"{o:.2f}" for o in orders if o == o))
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    return _save(fig, outdir, "fig_lax_profiles_vs_exact.png")


def fig_interface(leg: Dict, outdir: Path) -> str:
    rows = leg["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    ax = axes[0]
    for flux, color in (("hllc", "#d62728"), ("gfm", "#2ca02c")):
        sub = [r for r in rows if r["flux_mode"] == flux]
        ax.semilogy([r["nx"] for r in sub], [max(r["pressure_pp"], 1e-16) for r in sub],
                    "o-", color=color, label=flux.upper())
    ax.set_xlabel("cells")
    ax.set_ylabel("peak-to-peak pressure error at contact")
    ax.set_title("Ghost-fluid vs plain multi-material HLLC")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    sub = [r for r in rows if r["flux_mode"] == "gfm"]
    ax.loglog([r["dx"] for r in sub], [r["interface_error"] for r in sub],
              "s-", color="#1f77b4")
    for r in sub:
        ax.annotate(f"n={r['nx']}", (r["dx"], r["interface_error"]),
                    textcoords="offset points", xytext=(4, 4), fontsize=8)
    orders = leg.get("observed_order") or []
    if orders:
        ax.set_title("interface position error, order "
                     + ", ".join(f"{o:.2f}" for o in orders))
    ax.set_xlabel(r"$\Delta x$")
    ax.set_ylabel(r"$|x_I^{num} - x_I^{exact}|$")
    ax.grid(alpha=0.3, which="both")
    return _save(fig, outdir, "fig_interface_convergence.png")


def fig_rm(leg: Dict, outdir: Path) -> str:
    rows = leg["rows"]
    analytic = leg["analytic"]
    finest = rows[-1]
    snaps = finest.get("snapshots") or []
    fig = plt.figure(figsize=(13.0, 6.4))
    if snaps:
        for k, snap in enumerate(snaps[:3]):
            ax = fig.add_subplot(2, 3, k + 1)
            Y = np.asarray(snap["Y"])
            ax.imshow(Y.T, origin="lower", cmap="RdBu_r", vmin=0, vmax=1,
                      extent=[-1.8, 1.5, 0, 0.5], aspect="auto")
            ax.set_title(f"SF$_6$ mass fraction, t = {snap['t']:.2f}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
    ax_a = fig.add_subplot(2, 3, 4)
    ax_w = fig.add_subplot(2, 3, 5)
    ax_o = fig.add_subplot(2, 3, 6)
    t_tr = analytic["shock_transit_time"]
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(rows)))
    for row, color in zip(rows, colors):
        ts = row["width_series"]["t"]
        ax_a.plot(ts, row["width_series"]["amplitude"], color=color,
                  label=f"ny={row['ny']}")
        ax_w.plot(ts, row["width_series"]["width"], color=color, label=f"ny={row['ny']}")
    tt = np.linspace(t_tr, max(r["t_end"] for r in rows), 40)
    ax_a.plot(tt, analytic["a0_post"] + analytic["linear_growth_rate"] * (tt - t_tr),
              "k--", lw=1.0, label="Richtmyer linear")
    ax_a.set_xlabel("t")
    ax_a.set_ylabel("amplitude a(t)")
    ax_a.legend(frameon=False, fontsize=7)
    ax_a.set_title("single-mode amplitude")
    ax_w.plot(tt, 2 * analytic["a0_post"] + 2 * analytic["linear_growth_rate"] * (tt - t_tr),
              "k--", lw=1.0, label="linear W(t)")
    ax_w.set_xlabel("t")
    ax_w.set_ylabel("mixing width W(t)")
    ax_w.legend(frameon=False, fontsize=7)
    ax_w.set_title("mixing-zone width (1 %-99 %)")
    for row in rows:
        ax_o.semilogx(row["ny"], abs(row["amplitude_growth_rate"]
                                     - analytic["linear_growth_rate"])
                      / analytic["linear_growth_rate"], "o")
    ax_o.axhline(0.15, color="r", ls="--", lw=1.0, label="15 % gate")
    ax_o.set_xlabel("transverse cells ny")
    ax_o.set_ylabel("growth-rate error")
    ax_o.set_title("linear growth vs Richtmyer")
    ax_o.legend(frameon=False, fontsize=8)
    ax_o.grid(alpha=0.3, which="both")
    fig.tight_layout()
    return _save(fig, outdir, "fig_rm_instability.png")


def fig_bubble(leg: Dict, outdir: Path) -> str:
    snaps = leg.get("snapshots") or []
    series = leg.get("series") or {}
    fig = plt.figure(figsize=(13.0, 6.6))
    n_show = min(4, len(snaps))
    ext = [-3.0, 6.0, -2.5, 2.5]
    for k in range(n_show):
        ax = fig.add_subplot(2, 4, k + 1)
        w = np.asarray(snaps[k]["vorticity"])
        lim = float(np.abs(w).max())
        ax.imshow(w.T, origin="lower", cmap="gray_r", vmin=-lim, vmax=lim,
                  extent=ext, aspect="auto")
        ax.set_title(f"vorticity, t = {snaps[k]['t']:.2f}")
        ax.set_xlabel("x/R")
        if k == 0:
            ax.set_ylabel("y/R")
    if n_show:
        ax = fig.add_subplot(2, 4, n_show + 1)
        rho = np.asarray(snaps[n_show - 1]["rho"])
        ax.imshow(rho.T, origin="lower", cmap="magma", extent=ext, aspect="auto")
        ax.set_title(f"density, t = {snaps[n_show - 1]['t']:.2f}")
        ax.set_xlabel("x/R")
    if series:
        ax = fig.add_subplot(2, 2, 3)
        ax.plot(series["t"], series["circulation_upper"], color="#1f77b4")
        ax.axvline(leg["formation_reference"], color="r", ls="--", lw=1.0,
                   label="exact crossing time")
        ax.axvline(leg["formation_time"], color="g", ls=":", lw=1.2,
                   label="measured 95 % deposition")
        ax.set_xlabel("t")
        ax.set_ylabel(r"$\int_{y>0}\omega\,dA$")
        ax.legend(frameon=False, fontsize=8)
        ax.set_title("baroclinic vorticity deposition")
        ax = fig.add_subplot(2, 2, 4)
        ax.plot(series["t"], series["bubble_width"], color="#ff7f0e")
        ax.set_xlabel("t")
        ax.set_ylabel("bubble streamwise extent")
        ax.set_title("bubble deformation (Haas & Sturtevant topology)")
    fig.tight_layout()
    return _save(fig, outdir, "fig_shock_bubble.png")


def fig_vortex(leg: Dict, outdir: Path) -> str:
    main = leg["main"]
    snaps = main.get("snapshots") or []
    fig = plt.figure(figsize=(12.6, 3.9))
    n_show = min(3, len(snaps))
    for k in range(n_show):
        ax = fig.add_subplot(1, 3, k + 1)
        w = np.asarray(snaps[k]["vorticity"])
        lim = float(np.abs(w).max())
        ax.imshow(w.T, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
        sep = main["series"]["separation"][k] if k < len(main["series"]["separation"]) else None
        title = f"$\\omega_z$, t = {snaps[k]['t']:.2f}"
        if sep is not None:
            title += f", core sep = {sep:.2f}"
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    ax = fig.add_subplot(1, 3, 3)
    series = main["series"]
    ax.plot(series["t"], series["angle"], "o", ms=2.5, color="#1f77b4",
            label="principal axis (mod $\\pi$)")
    omega = main["rotation_rate_theory"]
    t = np.asarray(series["t"])
    ax.plot(t, np.unwrap(np.asarray(series["angle"]))[0] + omega * t, "k--", lw=1.0,
            label=f"theory $\\Omega=\\Gamma/\\pi b^2$ = {omega:.3f}")
    ax.set_xlabel("t")
    ax.set_ylabel("axis angle (rad)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(f"rotation error {100 * main['rotation_error']:.1f} %")
    fig.tight_layout()
    return _save(fig, outdir, "fig_vortex_merger.png")


def make_all(result: Dict, outdir: Path) -> Dict[str, List[str]]:
    """Generate every figure whose leg data is present; return the paths."""
    outdir = Path(outdir)
    produced: Dict[str, List[str]] = {}
    legs = result.get("legs", {})
    if "sod" in legs:
        produced.setdefault("sod", []).append(fig_sod(legs["sod"], outdir))
    if "lax" in legs:
        produced.setdefault("lax", []).append(fig_lax(legs["lax"], outdir))
    if "interface" in legs:
        produced.setdefault("interface", []).append(fig_interface(legs["interface"], outdir))
    if "rm" in legs:
        produced.setdefault("rm", []).append(fig_rm(legs["rm"], outdir))
    if "bubble" in legs:
        produced.setdefault("bubble", []).append(fig_bubble(legs["bubble"], outdir))
    if "vortex" in legs:
        produced.setdefault("vortex", []).append(fig_vortex(legs["vortex"], outdir))
    return produced