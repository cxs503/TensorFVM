"""Evidence generator for the two-phase Chorin-projection repair.

Runs four analytic anchors on ``tensorfvm.two_phase_solver`` and writes
``results.json`` plus three figures next to this file.

Anchors (all references are analytic, no literature numbers needed):

1. hydrostatic initialisation       p = rho_l g (h_i - y) + rho_g g (H - h_i)
                                   for y < h_i,  p = rho_g g (H - y) otherwise
                                   (continuous at the interface)
2. stratified column at rest       u = v = 0 for all t (exact steady state)
3. sealed domain                   pressure Poisson is all-Neumann -> must raise
4. static drop (Young-Laplace)     dp = sigma / R  (2D circular interface)

Usage:  PYTHONPATH=src python scripts/verify_two_phase_projection.py
"""

from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from tensorfvm.two_phase_solver import GridConfig, TwoPhaseConfig, TwoPhaseSolver

HERE = pathlib.Path(__file__).resolve().parent / "verification-two-phase-projection"
HERE.mkdir(parents=True, exist_ok=True)
DTYPE = torch.float64


def stratified(n: int, gravity: float = 9.81, mu: float = 1e-3):
    tp = TwoPhaseConfig(enabled=True, rho_liquid=1000.0, rho_gas=1.0,
                        mu_liquid=mu, mu_gas=mu, gravity=gravity)
    s = TwoPhaseSolver(GridConfig(nx=n, ny=n, length=1.0, height=1.0), tp)
    y = (torch.arange(n, dtype=DTYPE) + 0.5) * s.dy
    yy = y.reshape(-1, 1).expand(n, n)
    s.set_alpha(torch.where(yy < 0.5, torch.ones_like(yy), torch.zeros_like(yy)))
    s.initialize_hydrostatic_pressure(0.5)
    return s


def drop(n: int, radius: float = 0.2, sigma: float = 0.07, mu: float = 1e-2):
    tp = TwoPhaseConfig(enabled=True, rho_liquid=1000.0, rho_gas=1.0,
                        mu_liquid=mu, mu_gas=mu, gravity=0.0,
                        sigma=sigma, surface_tension=True)
    s = TwoPhaseSolver(GridConfig(nx=n, ny=n, length=1.0, height=1.0), tp)
    x = (torch.arange(n, dtype=DTYPE) + 0.5) * s.dx
    y = (torch.arange(n, dtype=DTYPE) + 0.5) * s.dy
    rr = torch.sqrt((x[None, :] - 0.5) ** 2 + (y[:, None] - 0.5) ** 2)
    alpha = torch.clamp((radius - rr) / max(s.dx, s.dy) + 0.5, 0.0, 1.0)
    s.set_alpha(alpha)
    return s, rr


def anchor_hydrostatic():
    n = 32
    s = stratified(n)
    g, rho_l, rho_g, h_i = 9.81, 1000.0, 1.0, 0.5
    y = ((torch.arange(n, dtype=DTYPE) + 0.5) * s.dy).reshape(-1, 1)
    p_iface = rho_g * g * (1.0 - h_i)
    exact = torch.where(y < h_i,
                        p_iface + rho_l * g * (h_i - y),
                        rho_g * g * (1.0 - y)).expand(n, n)
    err = float((s.p - exact).abs().max())
    scale = float(exact.abs().max())
    # interface continuity of the analytic reference
    p_liq = float(p_iface)
    p_gas = float(rho_g * g * (1.0 - h_i))
    bal = s.hydrostatic_balance_residual()
    return {
        "cells": n * n,
        "max_abs_error_Pa": err,
        "pressure_scale_Pa": scale,
        "relative_error": err / scale,
        "interface_jump_Pa": abs(p_liq - p_gas),
        "top_cell_p_Pa": float(s.p[-1, 0]),
        "analytic_top_cell_p_Pa": rho_g * g * s.dy / 2,
        "balance_residual_max_Pa": bal["max_residual"],
        "balance_residual_rms_Pa": bal["rms_residual"],
    }


def anchor_rest_state(steps: int = 40):
    s = stratified(16)
    hist = []
    for k in range(steps):
        item = s.step(1e-5)
        hist.append({
            "step": k + 1,
            "max_u": float(s.u.abs().max()),
            "max_v": float(s.v.abs().max()),
            "max_div": float(item["continuity"]),
            "max_p": float(s.p.abs().max()),
        })
    first10 = hist[:10]
    return {
        "steps": steps,
        "max_velocity_first_10_steps": max(max(h["max_u"], h["max_v"]) for h in first10),
        "max_div_first_10_steps": max(h["max_div"] for h in first10),
        "max_velocity_step10": max(hist[9]["max_u"], hist[9]["max_v"]),
        "max_velocity_step40": max(hist[-1]["max_u"], hist[-1]["max_v"]),
        "history": hist,
    }


def anchor_sealed_domain():
    s = stratified(16)
    s.tp.top_atmosphere = False
    s._v_active[-1, :] = False
    try:
        s.step(1e-5)
    except ValueError as exc:
        return {"raised": True, "message": str(exc)[:120]}
    return {"raised": False, "message": ""}


def anchor_laplace():
    out = {}
    sigma, radius = 0.07, 0.2
    for n in (32, 64):
        s, rr = drop(n)
        for _ in range(10):
            s.step(1e-5)
        inside = float(s.p[rr < 0.5 * radius].mean())
        outside = float(s.p[rr > 1.6 * radius].mean())
        jump = inside - outside
        exact = sigma / radius
        area = float(s.alpha.sum()) * s.dx * s.dy
        out[f"n{n}"] = {
            "pressure_jump_Pa": jump,
            "analytic_jump_Pa": exact,
            "relative_error": abs(jump - exact) / exact,
            "max_velocity": float(max(s.u.abs().max(), s.v.abs().max())),
            "area_error": abs(area - 3.141592653589793 * radius ** 2)
            / (3.141592653589793 * radius ** 2),
        }
    return out


def anchor_projection_exactness():
    """Divergence before/after one isolated projection on a random velocity."""
    n, dt = 16, 1e-5
    s = stratified(n, gravity=0.0, mu=0.0)
    torch.manual_seed(3)
    s.u.normal_()
    s.v.normal_()
    s._boundaries(projection=True)
    u_star, v_star = s.u.clone(), s.v.clone()
    div_before = float(s._divergence().abs().max())
    dp = s._projection_pressure(u_star, v_star, dt)
    ir_u = 2.0 / (s.rho_field[:, :-1] + s.rho_field[:, 1:])
    ir_v = 2.0 / (s.rho_field[:-1, :] + s.rho_field[1:, :])
    s.u[:, 1:-1] = u_star[:, 1:-1] - dt * ir_u * (dp[:, 1:] - dp[:, :-1]) / s.dx
    s.v[1:-1, :] = v_star[1:-1, :] - dt * ir_v * (dp[1:, :] - dp[:-1, :]) / s.dy
    if s.tp.top_atmosphere:
        ir_t = 1.0 / s.rho_field[-1, :]
        s.v[-1, :] = v_star[-1, :] + dt * ir_t * dp[-1, :] / (0.5 * s.dy)
    div_after = float(s._divergence().abs().max())
    return {"div_before": div_before, "div_after": div_after,
            "amplification": div_after / div_before}


def figures(hydro, rest, lap):
    # 1. hydrostatic profile
    s = stratified(32)
    y = ((torch.arange(32, dtype=DTYPE) + 0.5) * s.dy).numpy()
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.plot(s.p[:, 0].numpy(), y, "o", ms=3, label="discrete p")
    g, rho_l, rho_g, h_i = 9.81, 1000.0, 1.0, 0.5
    yy = y
    ref = [rho_g * g * (1 - h_i) + rho_l * g * (h_i - q) if q < h_i
           else rho_g * g * (1 - q) for q in yy]
    ax.plot(ref, yy, "-", lw=1.0, label="analytic (continuous)")
    ax.set_xlabel("p [Pa]")
    ax.set_ylabel("y [m]")
    ax.set_title("Hydrostatic initialisation (32$^2$)")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(HERE / "hydrostatic-profile.png", dpi=140)
    plt.close(fig)

    # 2. rest state growth
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    steps = [h["step"] for h in rest["history"]]
    ax.semilogy(steps, [max(h["max_u"], h["max_v"]) for h in rest["history"]],
                "-", lw=1.2, label="max|u|,|v|")
    ax.semilogy(steps, [max(h["max_div"], 1e-300) for h in rest["history"]],
                "--", lw=1.0, label="max|div u|")
    ax.axvline(10, color="0.5", lw=0.8, ls=":")
    ax.set_xlabel("step")
    ax.set_ylabel("magnitude")
    ax.set_title("Stratified column at rest (exact: 0)")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(HERE / "rest-state-growth.png", dpi=140)
    plt.close(fig)

    # 3. Laplace jump
    keys = sorted(lap)
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    measured = [lap[k]["pressure_jump_Pa"] for k in keys]
    exact = [lap[k]["analytic_jump_Pa"] for k in keys]
    xpos = range(len(keys))
    ax.bar([p - 0.18 for p in xpos], measured, width=0.36, label="measured")
    ax.bar([p + 0.18 for p in xpos], exact, width=0.36, label=r"$\sigma/R$")
    ax.set_xticks(list(xpos))
    ax.set_xticklabels([k.replace("n", "") + "$^2$" for k in keys])
    ax.set_ylabel(r"$\Delta p$ [Pa]")
    ax.set_title("Static drop: Young-Laplace jump")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(HERE / "laplace-jump.png", dpi=140)
    plt.close(fig)


def main():
    result = {
        "hydrostatic_initialisation": anchor_hydrostatic(),
        "rest_state": anchor_rest_state(),
        "sealed_domain": anchor_sealed_domain(),
        "young_laplace": anchor_laplace(),
        "projection_exactness": anchor_projection_exactness(),
    }
    (HERE / "results.json").write_text(json.dumps(result, indent=2))
    figures(result["hydrostatic_initialisation"], result["rest_state"],
            result["young_laplace"])
    summary = {k: v for k, v in result.items() if k != "rest_state"}
    summary["rest_state_max_vel_step10"] = result["rest_state"]["max_velocity_step10"]
    summary["rest_state_max_vel_step40"] = result["rest_state"]["max_velocity_step40"]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()