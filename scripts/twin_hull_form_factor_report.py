"""Figures for the twin-hull form-factor / viscous-decomposition report.

Usage:  PYTHONPATH=src python3 scripts/twin_hull_form_factor_report.py [--out DIR]

Reads the artifact JSONs produced by
``python3 -m tensorfvm.twin_hull_form_factor`` and recomputes the two
verification panels (sphere, flat plate) and the surface-pressure comparison, so
every figure is generated from the code under test rather than transcribed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tensorfvm.integral_boundary_layer import ittc57_cf, march_boundary_layer  # noqa: E402
from tensorfvm.panel_potential import (  # noqa: E402
    IMAGE_SPECS,
    solve_sources,
    sphere_cp_analytic,
    surface_velocity,
)
from tensorfvm.twin_hull_form_factor import (  # noqa: E402
    _streamwise_geometry,
    holtrop_mennen_form_factor,
)
from tensorfvm.twin_hull_geometry import (  # noqa: E402
    DemihullGeometry,
    KVLCC2Principal,
    calibrate_form,
    sphere_panels,
)

ART = Path("artifacts/twin-hull-form-factor")


def fig_geometry(geo: DemihullGeometry, out: Path):
    pr = geo.principal
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    x = (geo.nodes[:, 0, 0])
    ax[0].plot((x + 0.5 * pr.L) / pr.L, np.max(np.abs(geo.nodes[:, :, 1]), axis=1), "k-")
    ax[0].set_xlabel("x / L"); ax[0].set_ylabel("half breadth  b/B [-]".replace("b/", "b [m] / B"))
    ax[0].set_title("waterline (panel nodes)")
    ax[0].grid(alpha=0.3)
    i = geo.n_stations // 2
    ax[1].plot(geo.nodes[i, :, 1], geo.nodes[i, :, 2], "k.-", ms=2, lw=0.8)
    ax[1].set_aspect("equal"); ax[1].set_title("midship section (panel nodes)")
    ax[1].set_xlabel("y [m]"); ax[1].set_ylabel("z [m]"); ax[1].grid(alpha=0.3)
    c = geo.panels.centroids
    ax[2].plot((c[:, 0] + 0.5 * pr.L) / pr.L, c[:, 1], ",", alpha=0.15, ms=2)
    ax[2].set_title(f"panel centroids ({geo.panels.n_panels})")
    ax[2].set_xlabel("x / L"); ax[2].set_ylabel("y [m]"); ax[2].grid(alpha=0.3)
    fig.suptitle("KVLCC2 demihull, parametric reconstruction (Cb = 0.8098, LCB = +3.48 % L)")
    fig.tight_layout()
    fig.savefig(out / "fig1_geometry.png", dpi=130)
    plt.close(fig)


def fig_sphere(out: Path):
    U = np.array([0.0, 0.0, 1.0])
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    for n_theta, n_phi, style in ((16, 12, "o"), (36, 26, "s")):
        panels = sphere_panels(1.0, n_theta, n_phi)
        active = panels.active_mask()
        sigma = solve_sources(panels, U, order=2)
        _, cp = surface_velocity(panels.centroids, panels.normals, sigma, panels, U, order=2)
        sphere_cp_analytic(panels, U)  # analytic reference plotted once below
        theta = np.degrees(np.arccos(np.clip(panels.centroids[:, 2] /
                                             np.linalg.norm(panels.centroids, axis=1), -1, 1)))
        order = np.argsort(theta)
        sel = order[active[order]]
        ax.plot(theta[sel], cp[sel], style, ms=3, alpha=0.6,
                label=f"panel solution {n_theta}x{n_phi}")
    th = np.linspace(0, 180, 200)
    ax.plot(th, 1 - 2.25 * np.sin(np.radians(th)) ** 2, "k-", lw=1.6,
            label=r"analytic $1-\frac{9}{4}\sin^2\theta$")
    ax.set_xlabel(r"polar angle $\theta$ [deg]"); ax.set_ylabel(r"$C_p$")
    ax.set_title("source-panel verification: sphere in uniform flow")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "fig2_sphere_verification.png", dpi=130)
    plt.close(fig)


def fig_boundary_layer(out: Path):
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8))
    n = 400
    x = np.linspace(0, 1, n)
    for Re, ls in ((1.0e6, ":"), (1.0e7, "-"), (1.0e8, "--")):
        ds = np.full((n, 1), 1.0 / n)
        r = march_boundary_layer(np.ones((n, 1)), ds, ds.copy(), nu=1.0 / Re, U=1.0,
                                 Re=Re, force_turbulent=True)
        ax[0].plot(x, r.cf[:, 0], ls, label=f"Head, Re={Re:.0e}: $C_F$={r.C_F:.5f}")
        ax[0].axhline(ittc57_cf(Re), color="k", lw=0.5, alpha=0.35)
    ax[0].set_yscale("log")
    ax[0].set_xlabel("x / L"); ax[0].set_ylabel(r"$c_f$")
    ax[0].set_title("flat plate: Head's method vs ITTC-1957 (thin black lines)")
    ax[0].legend(fontsize=7); ax[0].grid(alpha=0.3)

    pr = KVLCC2Principal(); fm = calibrate_form(pr)
    geo = DemihullGeometry(pr, fm, n_stations=36, n_girth_half=14)
    U = 1.4286
    panels = geo.solver_panels()
    sig = solve_sources(panels, np.array([-U, 0.0, 0.0]), order=2, images=IMAGE_SPECS["single_hull"])
    speed, _ = surface_velocity(panels.centroids, panels.normals, sig, panels,
                                np.array([-U, 0.0, 0.0]), order=2, images=IMAGE_SPECS["single_hull"])
    areas, ds_b = _streamwise_geometry(geo, geo.panels)
    valid = areas > 1e-3 * areas.max()
    bl = march_boundary_layer(speed[:geo.n_hull_panels][::-1].reshape(geo.shape), ds_b, areas,
                              nu=1.0e-6, U=U, Re=U * pr.L / 1.0e-6, valid=valid)
    xb = (geo.station_centres_x + 0.5 * pr.L) / pr.L
    mid = geo.shape[1] // 2
    ax[1].plot(xb, bl.cf[:, mid], "b-", label=r"$c_f$, mid girth")
    ax[1].plot(xb, bl.cf[:, 2], "b--", label=r"$c_f$, waterline-side girth")
    ax[1].axhline(ittc57_cf(U * pr.L / 1e-6), color="k", ls=":", lw=1.2,
                  label="ITTC-1957 mean level")
    ax[1].set_xlabel("x / L"); ax[1].set_ylabel(r"$c_f$")
    ax2 = ax[1].twinx()
    ax2.plot(xb, bl.delta_star[:, mid] * 1000, "r-.", lw=1.0, label=r"$\delta^*$ [mm]")
    ax2.set_ylabel(r"$\delta^*$ [mm]", color="r")
    ax[1].set_title(r"single hull, Re = 1e7: " + f"$C_F$={bl.C_F:.5f}")
    ax[1].legend(fontsize=7, loc="upper left"); ax2.legend(fontsize=7, loc="upper right")
    ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "fig3_boundary_layer.png", dpi=130)
    plt.close(fig)


def fig_grid(grid: dict, out: Path):
    conv = grid["convergence"]
    levels = grid["levels"]
    h = [1.0 / lv["level"]["n_stations"] for lv in levels]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for key, style in (("C_F", "o-"), ("C_VP", "s--"), ("C_V", "^-")):
        vals = [lv[key] for lv in levels]
        ax[0].plot(h, vals, style, label=key)
    ax[0].set_xscale("log"); ax[0].set_xlabel("h  (1 / N_stations)")
    ax[0].set_ylabel("coefficient"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    ax[0].set_title("grid refinement (both directions, ratio 1.5)")
    keys = ["C_F", "C_VP", "C_V", "k"]
    rel = [conv[k]["max_relative_change"] for k in keys]
    ax[1].bar(range(len(keys)), rel, color=["tab:blue", "tab:orange", "tab:green", "tab:red"])
    ax[1].axhline(0.02, color="k", ls="--", lw=1, label="2 % criterion")
    ax[1].set_xticks(range(len(keys))); ax[1].set_xticklabels(keys)
    ax[1].set_ylabel("max relative change between grids")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    ax[1].set_title("observed order p: C_F = %.2f, C_V = n/a"
                    % (conv["C_F"]["observed_order"] or float("nan")))
    fig.tight_layout(); fig.savefig(out / "fig4_grid_convergence.png", dpi=130)
    plt.close(fig)


def fig_interference(sweep: dict, emp: dict, out: Path):
    cases = sweep["cases"]
    s = [c["separation_ratio"] for c in cases]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(s, [c["interference_factor"] for c in cases], "ko-", label="total $i_f$")
    ax[0].plot(s, [c["interference_factor_friction"] for c in cases], "b^--", label="friction part")
    ax[0].plot(s, [c["interference_factor_form"] for c in cases], "rs--", label="form part")
    ax[0].axhline(0.0, color="k", lw=0.6)
    ax[0].set_xlabel("separation  s / L"); ax[0].set_ylabel("interference factor  $i_f$")
    ax[0].set_title("viscous interference of the demihull")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    ax[1].plot(s, [c["one_plus_k"] for c in cases], "ko-", label="twin-hull $(1+k)(1+i_f)$")
    ax[1].axhline(1.0 + emp["k"], color="k", ls="--", lw=1.2,
                  label=f"ITTC-78/Holtrop-Mennen $1+k$ = {1 + emp['k']:.3f}")
    ax[1].set_xlabel("separation  s / L"); ax[1].set_ylabel("form factor  $1+k$")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    ax[1].set_title("form factor vs empirical reference")
    fig.tight_layout(); fig.savefig(out / "fig5_interference.png", dpi=130)
    plt.close(fig)


def fig_pressure(out: Path):
    pr = KVLCC2Principal(); fm = calibrate_form(pr)
    geo = DemihullGeometry(pr, fm, n_stations=36, n_girth_half=14)
    U = 1.4286
    result = {}
    for label, sep in (("single hull", None), ("catamaran s/L = 0.3", 0.3)):
        if sep is None:
            panels = geo.solver_panels()
            images = IMAGE_SPECS["single_hull"]
        else:
            from tensorfvm.twin_hull_geometry import catamaran_panels
            panels, _ = catamaran_panels(geo, sep)
            images = IMAGE_SPECS["twin_hull"]
        sigma = solve_sources(panels, np.array([-U, 0.0, 0.0]), order=2, images=images)
        _, cp = surface_velocity(panels.centroids, panels.normals, sigma, panels,
                                 np.array([-U, 0.0, 0.0]), order=2, images=images)
        cp = cp[:geo.n_hull_panels].reshape(geo.shape)
        result[label] = cp
    xb = (geo.station_centres_x + 0.5 * pr.L) / pr.L
    j_inner = 2
    j_outer = geo.shape[1] - 3
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for label, cp in result.items():
        style = "-" if label == "single hull" else "--"
        ax.plot(xb, cp[:, j_inner], style, color="tab:red",
                label=f"{label}: inner shell")
        ax.plot(xb, cp[:, j_outer], style, color="tab:blue",
                label=f"{label}: outer shell")
    ax.set_xlabel("x / L"); ax.set_ylabel(r"$C_p$")
    ax.set_title("double-body surface pressure, inner vs outer shell")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "fig6_pressure.png", dpi=130)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/twin-hull-form-factor/figures")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    grid = json.loads((ART / "grid_independence.json").read_text())
    sweep = json.loads((ART / "separation_sweep.json").read_text())
    emp = holtrop_mennen_form_factor()
    pr = KVLCC2Principal(); fm = calibrate_form(pr)
    geo = DemihullGeometry(pr, fm, n_stations=36, n_girth_half=14)

    fig_geometry(geo, out)
    fig_sphere(out)
    fig_boundary_layer(out)
    fig_grid(grid, out)
    fig_interference(sweep, emp, out)
    fig_pressure(out)
    print("figures written to", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())