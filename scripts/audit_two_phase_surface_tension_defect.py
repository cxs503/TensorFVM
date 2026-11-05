#!/usr/bin/env python3
"""Reproduce the two-phase surface tension divergence defect and document it.

This script runs the existing solver on a static drop case and records
the exponential divergence, confirming the verification debt.
"""

import json
import os
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorfvm.two_phase_solver import GridConfig, TwoPhaseConfig, TwoPhaseSolver


def _solver(n=32, radius=0.2, sigma=0.07, mu=1e-2):
    tp = TwoPhaseConfig(
        enabled=True,
        rho_liquid=1000.0,
        rho_gas=1.0,
        mu_liquid=mu,
        mu_gas=mu,
        gravity=0.0,
        sigma=sigma,
        surface_tension=True,
        top_atmosphere=True,
    )
    solver = TwoPhaseSolver(GridConfig(nx=n, ny=n, length=1.0, height=1.0), tp)
    x = (torch.arange(n, dtype=torch.float64) + 0.5) * solver.dx
    rr = torch.sqrt((x[None, :] - 0.5) ** 2 + (x[:, None] - 0.5) ** 2)
    solver.set_alpha(torch.clamp((radius - rr) / solver.dx + 0.5, 0.0, 1.0))
    return solver, rr


def run_audit(n=32, radius=0.2, sigma=0.07, mu=1e-2, steps=10, dt=1e-5):
    """Run the solver and record the divergence."""
    torch.set_num_threads(1)
    exact_jump = sigma / radius

    solver, rr = _solver(n=n, radius=radius, sigma=sigma, mu=mu)
    history = []
    diverged = False
    for i in range(steps):
        try:
            item = solver.step(dt)
            history.append(item)
            if item["continuity"] > 1e10 or not torch.isfinite(solver.p).all():
                diverged = True
                break
        except RuntimeError:
            diverged = True
            break

    # Compute pressure jump
    inside = float(solver.p[rr < 0.5 * radius].mean())
    outside = float(solver.p[rr > 1.6 * radius].mean())
    computed_jump = abs(inside - outside)
    rel_error = (
        abs(computed_jump - exact_jump) / exact_jump if exact_jump > 0 else float("inf")
    )

    umax = max(float(solver.u.abs().max()), float(solver.v.abs().max()))
    capillary_number = mu * umax / sigma if sigma > 0 else 0

    return {
        "n": n,
        "radius": radius,
        "sigma": sigma,
        "mu": mu,
        "dt": dt,
        "steps_run": len(history),
        "diverged": diverged,
        "exact_jump_pa": exact_jump,
        "computed_jump_pa": computed_jump,
        "inside_pressure_pa": inside,
        "outside_pressure_pa": outside,
        "relative_error": rel_error,
        "umax": umax,
        "capillary_number": capillary_number,
        "final_divergence": history[-1]["continuity"] if history else None,
        "history": history,
    }


def main():
    outdir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "docs",
        "two-phase-surface-tension-fix",
        "outputs",
    )
    os.makedirs(outdir, exist_ok=True)

    results = []
    for n in [24, 32]:
        print(f"Running n={n}...")
        r = run_audit(n=n)
        results.append(r)
        print(f"  Steps: {r['steps_run']}, Diverged: {r['diverged']}")
        print(f"  dp={r['computed_jump_pa']:.4g} Pa (exact={r['exact_jump_pa']:.4g})")
        print(f"  Rel error: {r['relative_error']:.2e}")

    summary = {
        "schema": "tensorfvm.two-phase-surface-tension-defect/1",
        "results": results,
        "defect_confirmed": all(
            r["diverged"] or r["relative_error"] > 0.03 for r in results
        ),
        "note": "The explicit CSF force in the predictor is not balanced by the "
        "projection step, causing exponential divergence for static drops.",
    }

    with open(os.path.join(outdir, "defect_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Figure: divergence history
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in results:
        hist = r["history"]
        if hist:
            ax.semilogy(
                [h["iteration"] for h in hist],
                [h["continuity"] for h in hist],
                "o-",
                label=f"n={r['n']}",
            )
    ax.axhline(y=1e-6, color="gray", linestyle="--", alpha=0.5, label="Tolerance")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Max |∇·u| (1/s)")
    ax.set_title(
        "Two-Phase Surface Tension: Exponential Divergence Defect\n"
        "Static drop (σ=0.07 N/m, R=0.2 m, μ=0.01 Pa·s, dt=10 μs)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "divergence_history.png"), dpi=150)
    plt.close()

    print(f"\nDefect confirmed: {summary['defect_confirmed']}")
    print(f"Summary: {outdir}/defect_summary.json")
    print(f"Figure: {outdir}/divergence_history.png")


if __name__ == "__main__":
    main()
