#!/usr/bin/env python3
"""Verification driver for the variable-density two-phase projection solver.

Mission m2564.  Three acceptance anchors are exercised end-to-end against the
``tensorfvm`` two-phase solver (``src/tensorfvm/two_phase_solver.py``):

A. **Static water column** (liquid/gas column at rest under gravity).
   A well-balanced variable-density projection must keep the velocity at the
   machine-noise level and drive the discrete divergence to zero.
   Metrics: ``max|u|``, ``max|div u|`` per step, and the pressure profile
   against the piecewise-analytic hydrostatic field.

B. **Static droplet** (Young-Laplace).  A circular liquid drop held by surface
   tension in a box at rest must build the Laplace pressure jump
   ``dp = sigma / R``.  Metrics: ``|dp - sigma/R| / (sigma/R)``.

C. **Fully closed domain rejected at configuration time.**  With
   ``top_atmosphere=False`` every boundary is an impermeable wall, the pressure
   Poisson equation is pure-Neumann (singular up to an additive constant) and
   the projection has no gauge.  The configuration must raise ``ValueError``.

Usage
-----
    python verify_two_phase_solver.py --tag after          # run + JSON + PNG
    python verify_two_phase_solver.py --tag before --no-figs

Outputs (written next to this script):
    results_<tag>.json    programmatic record of every metric (no hand copy)
    fig_<tag>_column.png  static-column panel
    fig_<tag>_droplet.png droplet panel
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from tensorfvm.two_phase_solver import (  # noqa: E402
    GridConfig,
    TwoPhaseConfig,
    TwoPhaseSolver,
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def stratified_alpha(solver: TwoPhaseSolver, interface_y: float) -> torch.Tensor:
    """Sharp (cell-wise) alpha field: 1 below ``interface_y``, 0 above."""
    y = (torch.arange(solver.ny, dtype=torch.float64) + 0.5) * solver.dy
    col = (y < interface_y).to(torch.float64).reshape(-1, 1)
    return col.expand(solver.ny, solver.nx).clone()


def circular_alpha(nx: int, ny: int, length: float, height: float,
                   cx: float, cy: float, radius: float,
                   dx: float, eps_factor: float = 0.75) -> torch.Tensor:
    """Diffuse circular droplet: alpha = 0.5*(1 - tanh((r-R)/eps))."""
    x = (torch.arange(nx, dtype=torch.float64) + 0.5) * dx
    y = (torch.arange(ny, dtype=torch.float64) + 0.5) * (height / ny)
    r = torch.sqrt((x[None, :] - cx) ** 2 + (y[:, None] - cy) ** 2)
    eps = eps_factor * dx
    return 0.5 * (1.0 - torch.tanh((r - radius) / eps))


def analytic_hydrostatic(y: float, interface_y: float, height: float,
                         rho_l: float, rho_g: float, g: float) -> float:
    """Piecewise-analytic hydrostatic pressure with p(height) = 0."""
    if y < interface_y:
        return rho_l * g * (interface_y - y) + rho_g * g * (height - interface_y)
    return rho_g * g * (height - y)


# --------------------------------------------------------------------------- #
# case A: static water column
# --------------------------------------------------------------------------- #
def case_static_column(n: int = 40, dt: float = 1e-3, steps: int = 10,
                       interface_y: float = 0.5) -> dict:
    cfg = GridConfig(nx=n, ny=n, length=1.0, height=1.0, time_step=dt)
    tp = TwoPhaseConfig(enabled=True, rho_liquid=1000.0, rho_gas=1.0,
                        mu_liquid=1.0e-3, mu_gas=1.0e-5,
                        gravity=9.81, top_atmosphere=True)
    solver = TwoPhaseSolver(cfg, tp)
    solver.set_alpha(stratified_alpha(solver, interface_y))
    solver.initialize_hydrostatic_pressure(interface_y)

    history = []
    with torch.no_grad():
        for _ in range(steps):
            solver.step(dt)
            history.append({
                "step": len(history) + 1,
                "t": solver.time,
                "max_u": float(torch.abs(solver.u).max()),
                "max_v": float(torch.abs(solver.v).max()),
                "max_div": float(torch.abs(solver._divergence()).max()),
                "p_max": float(solver.p.max()),
                "p_min": float(solver.p.min()),
            })

    y = (torch.arange(n, dtype=torch.float64) + 0.5) * solver.dy
    p_analytic = torch.tensor(
        [analytic_hydrostatic(float(yy), interface_y, cfg.height,
                              tp.rho_liquid, tp.rho_gas, tp.gravity) for yy in y])
    p_col = solver.p[:, n // 2]
    scale = tp.rho_liquid * tp.gravity * interface_y
    p_err = float((p_col - p_analytic).abs().max()) / scale

    return {
        "params": {"n": n, "dt": dt, "steps": steps, "interface_y": interface_y,
                   "gravity": tp.gravity, "rho_liquid": tp.rho_liquid,
                   "rho_gas": tp.rho_gas},
        "history": history,
        "max_u_final": history[-1]["max_u"],
        "max_v_final": history[-1]["max_v"],
        "max_div_final": history[-1]["max_div"],
        "max_div_step1": history[0]["max_div"],
        "p_max_final": history[-1]["p_max"],
        "p_scale_rho_g_dy": tp.rho_liquid * tp.gravity * solver.dy,
        "p_profile_rel_error_vs_analytic": p_err,
        "p_profile": [float(v) for v in p_col],
        "p_analytic": [float(v) for v in p_analytic],
        "y": [float(v) for v in y],
    }


# --------------------------------------------------------------------------- #
# case B: static droplet (Young-Laplace)
# --------------------------------------------------------------------------- #
def case_static_droplet(n: int = 64, radius: float = 0.25, sigma: float = 0.0728,
                        rho_l: float = 1000.0, rho_g: float = 1000.0,
                        mu: float = 1.0e-3, dt: float = 1e-4, steps: int = 5,
                        eps_factor: float = 0.75) -> dict:
    cfg = GridConfig(nx=n, ny=n, length=1.0, height=1.0, time_step=dt)
    tp = TwoPhaseConfig(enabled=True, rho_liquid=rho_l, rho_gas=rho_g,
                        mu_liquid=mu, mu_gas=mu, sigma=sigma,
                        surface_tension=True, gravity=0.0, top_atmosphere=True)
    solver = TwoPhaseSolver(cfg, tp)
    solver.set_alpha(circular_alpha(n, n, 1.0, 1.0, 0.5, 0.5, radius,
                                    solver.dx, eps_factor))

    with torch.no_grad():
        for _ in range(steps):
            solver.step(dt)

    p = solver.p
    cy, cx = n // 2, n // 2
    p_center = float(p[cy - 1:cy + 1, cx - 1:cx + 1].mean())
    p_far = float(p[2:5, 2:5].mean())
    dp = p_center - p_far
    dp_ref = sigma / radius
    return {
        "params": {"n": n, "radius": radius, "sigma": sigma, "rho_liquid": rho_l,
                   "rho_gas": rho_g, "mu": mu, "dt": dt, "steps": steps,
                   "eps_factor": eps_factor},
        "dp_num": dp,
        "dp_ref": dp_ref,
        "rel_error": abs(dp - dp_ref) / dp_ref,
        "p_center": p_center,
        "p_far": p_far,
        "max_u_final": float(torch.abs(solver.u).max()),
        "max_div_final": float(torch.abs(solver._divergence()).max()),
        "p_field": [[float(v) for v in row] for row in p],
        "p_slice": [float(v) for v in p[cy, :]],
        "x_slice": [float((i + 0.5) * solver.dx) for i in range(n)],
        "alpha_slice": [float(v) for v in solver.alpha[cy, :]],
    }


# --------------------------------------------------------------------------- #
# case C: closed domain must be rejected at configuration time
# --------------------------------------------------------------------------- #
def case_closed_domain() -> dict:
    rejected = False
    message = ""
    try:
        TwoPhaseConfig(enabled=True, top_atmosphere=False)
    except ValueError as exc:  # expected
        rejected = True
        message = str(exc)
    return {"rejected": rejected, "message": message}


# --------------------------------------------------------------------------- #
# case D: operator/corrector consistency of one projection solve
# --------------------------------------------------------------------------- #
def case_projection_consistency(n: int = 32, dt: float = 1e-3) -> dict:
    """Inject a non-solenoidal u* and check div(u_new) after the projection."""
    cfg = GridConfig(nx=n, ny=n, length=1.0, height=1.0, time_step=dt)
    tp = TwoPhaseConfig(enabled=True, rho_liquid=1000.0, rho_gas=1.0,
                        mu_liquid=1.0e-3, mu_gas=1.0e-5, gravity=0.0,
                        top_atmosphere=True)
    solver = TwoPhaseSolver(cfg, tp)
    solver.set_alpha(stratified_alpha(solver, 0.5))
    with torch.no_grad():
        solver._boundaries()
        u_star = solver.u.clone()
        v_star = solver.v.clone()
        # smooth, deliberately non-divergence-free field
        for j in range(1, n - 1):
            for i in range(1, n - 1):
                u_star[j, i] = 1e-3 * math.sin(3.0 * math.pi * (i + 0.5) / n)
                v_star[j, i] = 1e-3 * math.cos(2.0 * math.pi * (j + 0.5) / n)
        div_star = float(torch.abs(
            (u_star[:, 1:] - u_star[:, :-1]) / solver.dx
            + (v_star[1:] - v_star[:-1]) / solver.dy).max())
        p = solver._projection_pressure(u_star, v_star, dt)
        face_inv = getattr(solver, "_face_inv_density", None)
        if face_inv is not None:
            inv_rho_u, inv_rho_v = face_inv()
        else:  # pre-fix layout: same formula, inlined
            inv_rho_u = 2.0 / (solver.rho_field[:, :-1] + solver.rho_field[:, 1:])
            inv_rho_v = 2.0 / (solver.rho_field[:-1, :] + solver.rho_field[1:, :])
        u_new = u_star.clone()
        v_new = v_star.clone()
        u_new[:, 1:-1] = u_star[:, 1:-1] - dt * inv_rho_u * (p[:, 1:] - p[:, :-1]) / solver.dx
        v_new[1:-1, :] = v_star[1:-1, :] - dt * inv_rho_v * (p[1:, :] - p[:-1, :]) / solver.dy
        v_new[-1, :] = v_new[-2, :]
        div_new = float(torch.abs(
            (u_new[:, 1:] - u_new[:, :-1]) / solver.dx
            + (v_new[1:] - v_new[:-1]) / solver.dy).max())
        # hydrostatic check: the discrete field rho*g*(h-y) must be reproduced
        y = (torch.arange(n, dtype=torch.float64) + 0.5) * solver.dy
        p_hyd = (1000.0 * 9.81 * (1.0 - y)).reshape(-1, 1).expand(n, n)
    return {"div_star_max": div_star, "div_new_max": div_new,
            "ratio": div_new / div_star if div_star > 0 else float("nan"),
            "p_max_after_projection": float(p.abs().max()),
            "p_hydrostatic_scale": float(p_hyd.max())}


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def make_figures(tag: str, results: dict) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    col = results["static_column"]
    hist = col["history"]
    drop = results["static_droplet"]
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.2))
    steps = [h["step"] for h in hist]
    ax[0].semilogy(steps, [max(h["max_div"], 1e-18) for h in hist], "o-",
                   label="max|div u|")
    ax[0].semilogy(steps, [max(h["max_u"], 1e-18) for h in hist], "s-",
                   label="max|u|")
    ax[0].set_xlabel("step")
    ax[0].set_ylabel("magnitude (SI)")
    ax[0].set_title("A: static column residual")
    ax[0].grid(True, which="both", alpha=0.3)
    ax[0].legend(fontsize=7)

    ax[1].plot(col["p_profile"], col["y"], "-", label="solver p")
    ax[1].plot(col["p_analytic"], col["y"], "--", label="analytic")
    ax[1].set_xlabel("p [Pa]")
    ax[1].set_ylabel("y [m]")
    ax[1].set_title(f"A: hydrostatic p (rel.err "
                    f"{col['p_profile_rel_error_vs_analytic'] * 100:.2f}%)")
    ax[1].grid(True, alpha=0.3)
    ax[1].legend(fontsize=7)

    ax[2].plot(drop["x_slice"], drop["p_slice"], "-", label="p (y=0.5)")
    ax[2].plot(drop["x_slice"], drop["alpha_slice"], ":", label="alpha")
    ax[2].axhline(drop["dp_ref"], color="k", ls="--", lw=0.8,
                  label="sigma/R=%.3f Pa" % drop["dp_ref"])
    ax[2].set_xlabel("x [m]")
    ax[2].set_title("B: droplet p and alpha")
    ax[2].grid(True, alpha=0.3)
    ax[2].legend(fontsize=7)
    fig.suptitle("two-phase projection verification [%s]" % tag, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p1 = os.path.join(HERE, "fig_%s_column.png" % tag)
    fig.savefig(p1, dpi=110)
    plt.close(fig)
    paths.append(p1)

    import numpy as np
    fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.2))
    im = ax[0].imshow(np.array(drop["p_field"]), origin="lower",
                      extent=(0, 1, 0, 1), aspect="equal", cmap="coolwarm")
    ax[0].contour(np.linspace(0.5 / len(drop["p_field"]), 1 - 0.5 / len(drop["p_field"]),
                              len(drop["p_field"])),
                  np.linspace(0.5 / len(drop["p_field"]), 1 - 0.5 / len(drop["p_field"]),
                              len(drop["p_field"])),
                  np.array(drop["p_field"]), levels=[0.5 * drop["dp_ref"]],
                  colors="k", linewidths=0.6)
    ax[0].set_title("B: pressure field [Pa]")
    fig.colorbar(im, ax=ax[0], fraction=0.046)
    ax[1].bar(["numerical\ndp", "sigma/R"],
              [drop["dp_num"], drop["dp_ref"]],
              color=["tab:blue", "tab:gray"])
    ax[1].set_ylabel("dp [Pa]")
    ax[1].set_title("Laplace jump, rel.err=%.2f%%" % (drop["rel_error"] * 100))
    ax[1].grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    p2 = os.path.join(HERE, "fig_%s_droplet.png" % tag)
    fig.savefig(p2, dpi=110)
    plt.close(fig)
    paths.append(p2)
    return paths


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="after")
    ap.add_argument("--no-figs", action="store_true")
    ap.add_argument("--n-column", type=int, default=40)
    ap.add_argument("--n-droplet", type=int, default=64)
    ap.add_argument("--steps-column", type=int, default=10)
    ap.add_argument("--steps-droplet", type=int, default=5)
    args = ap.parse_args()

    results = {
        "tag": args.tag,
        "static_column": case_static_column(n=args.n_column, steps=args.steps_column),
        "static_droplet": case_static_droplet(n=args.n_droplet,
                                              steps=args.steps_droplet),
        "closed_domain": case_closed_domain(),
        "projection_consistency": case_projection_consistency(),
    }
    out = os.path.join(HERE, "results_%s.json" % args.tag)
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)

    print(json.dumps({
        "tag": args.tag,
        "column_max_u": results["static_column"]["max_u_final"],
        "column_max_div": results["static_column"]["max_div_final"],
        "column_p_max": results["static_column"]["p_max_final"],
        "column_p_scale": results["static_column"]["p_scale_rho_g_dy"],
        "column_p_err": results["static_column"]["p_profile_rel_error_vs_analytic"],
        "droplet_dp": results["static_droplet"]["dp_num"],
        "droplet_ref": results["static_droplet"]["dp_ref"],
        "droplet_rel_err": results["static_droplet"]["rel_error"],
        "closed_domain_rejected": results["closed_domain"]["rejected"],
        "proj_div_star": results["projection_consistency"]["div_star_max"],
        "proj_div_new": results["projection_consistency"]["div_new_max"],
    }, indent=2))

    if not args.no_figs:
        for p in make_figures(args.tag, results):
            print("figure:", p)
    print("json:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())