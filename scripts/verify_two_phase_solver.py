"""Reproducible evidence for the two-phase VOF solver verification anchors.

Writes ``evidence.json`` and two figures next to this file (into
``scripts/verification-two-phase-solver/``).  Everything is deterministic and
runs in a few seconds on CPU; nothing here is a benchmark of performance.

Usage:  python scripts/verify_two_phase_solver.py
"""

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from tensorfvm.two_phase_solver import GridConfig, TwoPhaseConfig, TwoPhaseSolver

OUT = Path(__file__).resolve().parent / "verification-two-phase-solver"
G = 9.81
RHO_L, RHO_G = 1000.0, 1.0


def stratified(n=32, top_atmosphere=True):
    tp = TwoPhaseConfig(enabled=True, rho_liquid=RHO_L, rho_gas=RHO_G,
                        mu_liquid=1e-3, mu_gas=1e-3, gravity=G,
                        top_atmosphere=top_atmosphere)
    s = TwoPhaseSolver(GridConfig(nx=n, ny=n, length=1.0, height=1.0), tp)
    y = (torch.arange(n, dtype=torch.float64) + 0.5).reshape(-1, 1) * s.dy
    yy = y.expand(n, n)
    s.set_alpha(torch.where(yy < 0.5, torch.ones_like(yy), torch.zeros_like(yy)))
    s.initialize_hydrostatic_pressure(0.5)
    return s, yy, y.flatten()


def drop(n=32, radius=0.2, sigma=0.07):
    tp = TwoPhaseConfig(enabled=True, rho_liquid=RHO_L, rho_gas=RHO_G,
                        mu_liquid=1e-2, mu_gas=1e-2, gravity=0.0,
                        sigma=sigma, surface_tension=True)
    s = TwoPhaseSolver(GridConfig(nx=n, ny=n, length=1.0, height=1.0), tp)
    x = (torch.arange(n, dtype=torch.float64) + 0.5) * s.dx
    y = (torch.arange(n, dtype=torch.float64) + 0.5) * s.dy
    rr = torch.sqrt((x[None, :] - 0.5) ** 2 + (y[:, None] - 0.5) ** 2)
    s.set_alpha(torch.clamp((radius - rr) / max(s.dx, s.dy) + 0.5, 0.0, 1.0))
    return s, rr


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ev = {"cases": {}}

    # --- anchor 1: analytic hydrostatic initialisation -------------------
    s, yy, ycol = stratified(n=32)
    exact = torch.where(yy < 0.5, RHO_L * G * (0.5 - yy), RHO_G * G * (1.0 - yy))
    interior = (s.p[:-1] - exact[:-1]).abs().max().item()
    ev["cases"]["hydrostatic_initialisation"] = {
        "cells": "32x32", "max_abs_error_pa": interior,
        "relative_error": interior / exact[:-1].abs().max().item(),
        "top_row_gauge_pa": float(s.p[-1, 0]),
        "analytic_top_row_pa": float(exact[-1, 0]),
        "note": "top row is pinned to the atmospheric gauge, excluded from the error",
    }
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    ax.plot(exact[:, 0].numpy(), ycol.numpy(), "k-", lw=2, label="analytic")
    ax.plot(s.p[:, 0].numpy(), ycol.numpy(), "o", ms=3, label="initialised p")
    ax.axhline(0.5, color="0.6", ls=":", label="interface")
    ax.set(xlabel="pressure (Pa)", ylabel="y (m)",
           title="Hydrostatic initialisation vs analytic")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "hydrostatic-initialisation.png", dpi=130)
    plt.close(fig)

    # --- anchor 2: rest state under gravity -----------------------------
    s, _, _ = stratified(n=16)
    residual0 = s.hydrostatic_balance_residual()
    trace = []
    diverged_at = None
    dt = 1e-5
    for k in range(10):
        try:
            s.step(dt)
        except RuntimeError as exc:
            diverged_at = {"step": k, "error": str(exc)}
            break
        trace.append({"step": k + 1, "max_u": float(s.u.abs().max()),
                      "max_v": float(s.v.abs().max()),
                      "max_div": float(torch.abs(s._divergence()).max())})
    ev["cases"]["rest_state"] = {
        "cells": "16x16", "dt_s": dt, "steps_requested": 10,
        "initial_balance_residual": residual0,
        "max_velocity_after": trace[-1] if trace else None,
        "divergence_trace": trace, "diverged": diverged_at,
        "exact": {"max_u": 0.0, "max_v": 0.0},
    }
    if trace:
        fig, ax = plt.subplots(figsize=(5.0, 4.2))
        ax.semilogy([t["step"] for t in trace],
                    [max(t["max_u"], t["max_v"]) for t in trace], "o-")
        ax.set(xlabel="projection step", ylabel="max |velocity| (m/s)",
               title="Resting column: velocity growth (exact = 0)")
        ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(OUT / "rest-state-growth.png", dpi=130)
        plt.close(fig)

    # --- anchor 3: 2D Young-Laplace -------------------------------------
    laplace = []
    for n in (32, 64):
        s, rr = drop(n=n, radius=0.2, sigma=0.07)
        for _ in range(10):
            s.step(1e-5 * (32.0 / n))
        pin = float(s.p[rr < 0.1].mean())
        pout = float(s.p[rr > 0.32].mean())
        from tensorfvm.two_phase_vof import csf_surface_tension
        force = csf_surface_tension(s.alpha, 0.07, s.dx, s.dy)
        laplace.append({
            "cells": f"{n}x{n}", "dp_measured_pa": pin - pout,
            "dp_exact_pa": 0.07 / 0.2, "max_u": float(s.u.abs().max()),
            "csf_force_max_n_m3": float(force.abs().max()),
            "drop_area_error": abs(float(s.alpha.sum()) * s.dx * s.dy
                                   - math.pi * 0.04) / (math.pi * 0.04),
        })
    ev["cases"]["young_laplace"] = {
        "radius_m": 0.2, "sigma_n_m": 0.07, "exact_dp_pa": 0.35,
        "runs": laplace,
        "note": "csf_force_max_n_m3 is non-zero: the CSF operator itself is "
                "active, but it is never added to the transient momentum "
                "predictor, so dp remains 0",
    }
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    ax.bar([r["cells"] for r in laplace], [r["dp_measured_pa"] for r in laplace],
           label="measured dp", width=.45)
    ax.axhline(0.35, color="k", ls="--", label="exact sigma/R = 0.35 Pa")
    ax.set(xlabel="grid", ylabel="pressure jump (Pa)",
           title="Static drop pressure jump vs Young-Laplace")
    ax.legend(); ax.grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(OUT / "laplace-jump.png", dpi=130)
    plt.close(fig)

    (OUT / "evidence.json").write_text(json.dumps(ev, indent=2))
    print(json.dumps(ev, indent=2))


if __name__ == "__main__":
    main()