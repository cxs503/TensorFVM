"""Audit the two-phase CSF surface-tension chain against analytic capillary references.

Why this audit exists (verification debt)
-----------------------------------------
`tests/test_two_phase_solver_verification.py` carried two ``xfail`` cases, and the
Young--Laplace one states the reason directly: *"surface_tension is never added to
the transient projection momentum predictor, so a static drop carries dp = 0"*.
So `csf_surface_tension()` had unit tests for its own output but the force never
entered the momentum equation, and no grid-convergence/error evidence existed for
the capillary chain.

This script measures three things that all have analytic references:

A. discrete curvature of an exact circular interface vs ``kappa = 1/R``;
B. net capillary force of a closed interface (analytic value: zero, because
   ``|closed kappa*n ds = 0``) -- the discrete CSF imbalance;
C. a static droplet integrated with the two-phase solver: the Young--Laplace
   pressure jump vs ``dp = sigma/R`` (2D), the spurious-current magnitude, and
   whether the projection step holds the divergence at zero.

Criteria use the platform evidence bar: >3% deviation from an analytic reference
is a FAIL, and every criterion is machine-checked.  The script exits non-zero if
any criterion fails; it never reports a pass it did not measure.

Usage::

    python scripts/audit_two_phase_surface_tension.py --output docs/two-phase-surface-tension-audit
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from tensorfvm.two_phase_solver import GridConfig, TwoPhaseConfig, TwoPhaseSolver
from tensorfvm.two_phase_vof import csf_surface_tension, interface_curvature


def drop_alpha(n: int, radius: float, length: float = 1.0):
    """Cell-centred alpha of a circular drop, smeared over one cell (VOF init)."""
    dx = length / n
    x = (torch.arange(n, dtype=torch.float64) + 0.5) * dx
    r = torch.sqrt((x[None, :] - 0.5) ** 2 + (x[:, None] - 0.5) ** 2)
    alpha = torch.clamp((radius - r) / dx + 0.5, 0.0, 1.0)
    return alpha, r, dx


def curvature_study(n: int, radius: float) -> dict:
    alpha, r, dx = drop_alpha(n, radius)
    kappa = interface_curvature(alpha, dx, dx)
    band = (alpha > 0.05) & (alpha < 0.95)
    exact = 1.0 / radius
    # curvature is only defined on the interface band; report the volume-weighted
    # band mean against the exact circular value and the band maximum (noise).
    weights = (alpha * (1 - alpha))[band]
    weighted = float((kappa[band] * weights).sum() / weights.sum())
    return dict(
        n=n, dx=dx, exact=exact,
        kappa_band_mean=weighted,
        kappa_band_max=float(kappa[band].abs().max()),
        relative_error=(weighted - exact) / exact,
        band_cells=int(band.sum()),
    )


def force_balance(n: int, radius: float, sigma: float) -> dict:
    """Net CSF force of a closed interface must vanish (integral of kappa*n ds)."""
    alpha, _, dx = drop_alpha(n, radius)
    force = csf_surface_tension(alpha, sigma, dx, dx)
    net = force.sum(dim=(0, 1)) * dx * dx
    gross = force.abs().sum() * dx * dx
    return dict(
        n=n,
        net_force_n=float(torch.linalg.vector_norm(net)),
        gross_force_n=float(gross),
        imbalance=(float(torch.linalg.vector_norm(net)) / float(gross))
        if gross > 0 else 0.0,
    )


def laplace_study(n: int, radius: float, sigma: float, mu: float,
                  steps: int, dt: float) -> dict:
    tp = TwoPhaseConfig(
        enabled=True, rho_liquid=1000.0, rho_gas=1.0,
        mu_liquid=mu, mu_gas=mu, gravity=0.0,
        sigma=sigma, surface_tension=True, top_atmosphere=True,
    )
    solver = TwoPhaseSolver(GridConfig(nx=n, ny=n, length=1.0, height=1.0), tp)
    alpha, r, _ = drop_alpha(n, radius)
    solver.set_alpha(alpha)
    history = []
    for _ in range(steps):
        item = solver.step(dt)
        history.append(dict(iteration=item["iteration"],
                            divergence=item["continuity"],
                            velocity_change=item["velocity_change"],
                            volume_drift=item["volume_drift"]))
    inside = float(solver.p[r < 0.5 * radius].mean())
    outside = float(solver.p[r > 1.6 * radius].mean())
    jump = inside - outside
    exact = sigma / radius
    umax = max(float(solver.u.abs().max()), float(solver.v.abs().max()))
    finite = bool(torch.isfinite(solver.u).all() and torch.isfinite(solver.v).all()
                  and torch.isfinite(solver.p).all())
    return dict(
        n=n, radius_m=radius, sigma_n_m=sigma, mu_pa_s=mu, dt_s=dt, steps=steps,
        pressure_jump_pa=jump, exact_jump_pa=exact,
        relative_error=jump / exact - 1.0 if exact else None,
        umax_m_s=umax,
        capillary_number=mu * umax / sigma,
        capillary_velocity_scale=sigma / mu,
        max_divergence=max(h["divergence"] for h in history if math.isfinite(h["divergence"])),
        final_divergence=history[-1]["divergence"],
        finite=finite, history=history,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("docs/two-phase-surface-tension-audit"))
    parser.add_argument("--radius", type=float, default=0.2)
    parser.add_argument("--sigma", type=float, default=0.07)
    parser.add_argument("--mu", type=float, default=1e-2)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--dt", type=float, default=1e-5)
    parser.add_argument("--grids", type=int, nargs="+", default=[32, 48, 64])
    args = parser.parse_args(argv)

    torch.set_num_threads(1)
    started = time.time()
    curvature = [curvature_study(n, args.radius) for n in args.grids]
    balance = [force_balance(n, args.radius, args.sigma) for n in args.grids]
    laplace = [laplace_study(n, args.radius, args.sigma, args.mu, args.steps, args.dt)
               for n in args.grids]

    # Analytic anchors: kappa = 1/R, zero net force, dp = sigma/R (2D).
    curvature_ok = {r["n"]: abs(r["relative_error"]) < 0.03 for r in curvature}
    balance_ok = {r["n"]: r["imbalance"] < 0.03 for r in balance}
    laplace_ok = {r["n"]: (r["finite"] and r["relative_error"] is not None
                           and abs(r["relative_error"]) < 0.03) for r in laplace}
    stability_ok = {r["n"]: bool(r["finite"]) for r in laplace}
    spurious_ok = {r["n"]: r["capillary_number"] < 1e-2 for r in laplace}
    criteria = dict(
        C1_curvature_within_3pct=curvature_ok,
        C2_net_csf_force_within_3pct=balance_ok,
        C3_young_laplace_within_3pct=laplace_ok,
        C4_capillary_number_below_0p01=spurious_ok,
        C5_finite_iterates=stability_ok,
    )
    passing = all(all(v.values()) for v in criteria.values())
    summary = dict(
        schema="tensorfvm.two-phase-surface-tension-audit/1",
        generated_s=round(time.time() - started, 2),
        parameters=dict(radius_m=args.radius, sigma_n_m=args.sigma,
                        mu_pa_s=args.mu, dt_s=args.dt, steps=args.steps,
                        grids=args.grids, rho_liquid=1000.0, rho_gas=1.0),
        analytic_references=dict(
            curvature="kappa = 1/R for a circle",
            net_force="closed interface: |closed kappa*n ds = 0",
            pressure_jump="2D Young-Laplace: dp = sigma/R",
        ),
        curvature=curvature, force_balance=balance, laplace=laplace,
        criteria=criteria, all_criteria_passed=passing,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))

    # figure: analytic vs discrete
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
        ns = [r["n"] for r in curvature]
        ax[0].loglog(ns, [abs(r["relative_error"]) for r in curvature], "o-",
                     label="discrete $\\kappa$ vs $1/R$")
        ax[0].loglog(ns, [r["imbalance"] for r in balance], "s--",
                     label="net CSF force / gross")
        ax[0].axhline(0.03, color="r", ls=":", label="3% evidence bar")
        ax[0].set_xlabel("grid n"); ax[0].set_ylabel("relative measure")
        ax[0].legend(fontsize=7); ax[0].set_title("curvature & CSF balance")
        ax[1].semilogy(ns, [abs(r["relative_error"]) for r in laplace], "o-",
                       label="$|\\Delta p-\\sigma/R|/(\\sigma/R)$")
        ax[1].semilogy(ns, [max(r["capillary_number"], 1e-16) for r in laplace],
                       "s--", label="$Ca=\\mu|u|_{max}/\\sigma$")
        ax[1].semilogy(ns, [max(r["final_divergence"], 1e-16) for r in laplace],
                       "^-.", label="final $|\\nabla\\cdot u|$")
        ax[1].axhline(0.03, color="r", ls=":", label="3% evidence bar")
        ax[1].set_xlabel("grid n"); ax[1].legend(fontsize=7)
        ax[1].set_title("static droplet, Young-Laplace")
        fig.tight_layout(); fig.savefig(args.output / "fig_surface_tension_audit.png", dpi=130)
        plt.close(fig)
    except Exception as error:  # pragma: no cover - figure is evidence, not a gate
        print("figure skipped:", error)

    lines = ["# Two-phase CSF surface-tension audit", "",
             f"parameters: {json.dumps(summary['parameters'])}", "",
             "| n | kappa rel err | net/gross CSF | dP (Pa) | exact | rel err | |u|max | Ca | div_final |",
             "|---|---|---|---|---|---|---|---|---|"]
    for c, b, l in zip(curvature, balance, laplace):
        lines.append(
            f"| {c['n']} | {c['relative_error']:+.4f} | {b['imbalance']:.4f} | "
            f"{l['pressure_jump_pa']:.6g} | {l['exact_jump_pa']:.6g} | "
            f"{l['relative_error']:+.4f} | {l['umax_m_s']:.3g} | "
            f"{l['capillary_number']:.3g} | {l['final_divergence']:.3g} |")
    lines += ["", "criteria: " + json.dumps(criteria, indent=2),
              "", f"all criteria passed: {passing}", ""]
    (args.output / "report.md").write_text("\n".join(lines))
    print("\n".join(lines))
    return 0 if passing else 2


if __name__ == "__main__":
    raise SystemExit(main())