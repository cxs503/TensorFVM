#!/usr/bin/env python3
"""Audit and fix attempt for the two-phase surface tension verification debt.

This script:
1. Reproduces the divergence defect with the current solver
2. Applies a corrected projection that accounts for CSF in the Poisson RHS
3. Verifies the Young-Laplace jump dp = sigma/R within 3%
4. Generates a comparison figure and summary JSON
"""
import json
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tensorfvm.two_phase_solver import GridConfig, TwoPhaseConfig, TwoPhaseSolver


def _solver(n=32, radius=0.2, sigma=0.07, mu=1e-2, surface_tension=True):
    tp = TwoPhaseConfig(
        enabled=True, rho_liquid=1000.0, rho_gas=1.0, mu_liquid=mu, mu_gas=mu,
        gravity=0.0, sigma=sigma, surface_tension=surface_tension,
        top_atmosphere=True,
    )
    solver = TwoPhaseSolver(GridConfig(nx=n, ny=n, length=1.0, height=1.0), tp)
    x = (torch.arange(n, dtype=torch.float64) + 0.5) * solver.dx
    rr = torch.sqrt((x[None, :] - 0.5) ** 2 + (x[:, None] - 0.5) ** 2)
    solver.set_alpha(torch.clamp((radius - rr) / solver.dx + 0.5, 0.0, 1.0))
    return solver, rr


def _csf_face_forces_fixed(solver):
    """Compute CSF face forces (same as original _csf_face_forces)."""
    if not (solver.tp.surface_tension or solver.props.sigma == 0.0):
        return None, None
    f = solver.props.csf(solver.alpha, solver.dx, solver.dy)
    f_u = 0.5 * (f[:, :-1, 0] + f[:, 1:, 0])
    f_v = 0.5 * (f[:-1, :, 1] + f[1:, :, 1])
    return f_u, f_v


def _projection_step_fixed(solver, dt):
    """Corrected projection step that accounts for CSF in the Poisson RHS.

    The key change: the CSF divergence term is added to the RHS of the
    pressure Poisson equation, so the pressure correction balances the
    CSF force.
    """
    solver._boundaries(projection=True)
    rho = solver.rho_field
    mu = solver.mu_field
    u_old = solver.u.clone()
    v_old = solver.v.clone()

    # Predictor WITHOUT CSF force
    d2udx2 = (solver.u[:, :-2] - 2 * solver.u[:, 1:-1] + solver.u[:, 2:]) / solver.dx**2
    d2udy2_full = (solver.u[:-2, :] - 2 * solver.u[1:-1, :] + solver.u[2:, :]) / solver.dy**2
    mu_u_face = 0.5 * (mu[:, :-1] + mu[:, 1:])
    rho_u_face = 0.5 * (rho[:, :-1] + rho[:, 1:])
    dpdx_old = (solver.p[:, 1:] - solver.p[:, :-1]) / solver.dx
    dpdy_old = (solver.p[1:, :] - solver.p[:-1, :]) / solver.dy

    du_dt = (mu_u_face[1:-1, :] * (d2udx2[1:-1, :] + d2udy2_full[:, 1:-1])
             - dpdx_old[1:-1, :]) / rho_u_face[1:-1, :]
    solver.u[1:-1, 1:-1] = u_old[1:-1, 1:-1] + dt * du_dt

    d2vdx2_full = (solver.v[:, :-2] - 2 * solver.v[:, 1:-1] + solver.v[:, 2:]) / solver.dx**2
    d2vdy2_full = (solver.v[:-2, :] - 2 * solver.v[1:-1, :] + solver.v[2:, :]) / solver.dy**2
    mu_v_face = 0.5 * (mu[:-1, :] + mu[1:, :])
    rho_v_face = 0.5 * (rho[:-1, :] + rho[1:, :])
    dv_dt = (mu_v_face[:, 1:-1]
             * (d2vdx2_full[1:-1, :] + d2vdy2_full[:, 1:-1])
             - dpdy_old[:, 1:-1]) / rho_v_face[:, 1:-1]
    grav_v = -solver.tp.gravity
    dv_dt = dv_dt + grav_v
    solver.v[1:-1, 1:-1] = v_old[1:-1, 1:-1] + dt * dv_dt

    solver._boundaries(projection=True)

    # Compute CSF forces for the Poisson RHS
    f_u, f_v = _csf_face_forces_fixed(solver)

    # Projection with CSF correction in RHS
    u_star = solver.u.clone()
    v_star = solver.v.clone()

    # Modified RHS: (1/dt) div(u*) - div(f_csf/rho)
    ny, nx = solver.ny, solver.nx
    inv_rho_u = 2.0 / (rho[:, :-1] + rho[:, 1:]).clamp_min(1e-10)
    inv_rho_v = 2.0 / (rho[:-1, :] + rho[1:, :]).clamp_min(1e-10)

    # Divergence of u*
    div_u = ((u_star[:, 1:] - u_star[:, :-1]) / solver.dx
             + (v_star[1:] - v_star[:-1]) / solver.dy)

    # Divergence of CSF force (if present)
    if f_u is not None and f_v is not None:
        div_f = ((f_u[:, 1:] - f_u[:, :-1]) / solver.dx
                 + (f_v[1:, :] - f_v[:-1, :]) / solver.dy)
        rhs = -(solver.dx * solver.dy * (div_u / dt - div_f)).masked_fill(~solver.fluid, 0)
    else:
        rhs = -(solver.dx * solver.dy * div_u / dt).masked_fill(~solver.fluid, 0)

    # Solve Poisson (simplified - uses same solver as original)
    rho_ref = solver.props.rho_liquid
    cu = rho_ref * solver.dy * (dt * inv_rho_u)
    cv = rho_ref * solver.dx * (dt * inv_rho_v)
    ce, cw, cn, cs = (torch.zeros_like(solver.p) for _ in range(4))
    ce[:, :-1], cw[:, 1:] = cu[:, 1:-1], cu[:, 1:-1]
    cn[:-1], cs[1:] = cv[1:-1], cv[1:-1]
    if solver.tp.top_atmosphere:
        cn[-1, :] = 0
    diagonal = ce + cw + cn + cs
    diagonal = torch.where(diagonal > 0, diagonal, torch.ones_like(diagonal))
    diagonal = torch.where(solver.fluid, diagonal, torch.ones_like(diagonal))
    coeffs = [ce.clone(), cw, cn, cs]

    def matvec(q):
        return diagonal * q - sum(
            torch.roll(q, shifts=s, dims=d) * c
            for (s, d), c in [((0, 1), ce), ((0, -1), cw), ((1, 0), cn), ((-1, 0), cs)]
        )

    # Actually use the neighbor sum approach
    def neighbor_sum(q, coeffs_list):
        result = torch.zeros_like(q)
        # ce: east (shift left in col index)
        result[:, :-1] += coeffs_list[0][:, :-1] * q[:, 1:]
        # cw: west (shift right in col index)
        result[:, 1:] += coeffs_list[1][:, 1:] * q[:, :-1]
        # cn: north (shift down in row index)
        result[:-1, :] += coeffs_list[2][:-1, :] * q[1:, :]
        # cs: south (shift up in row index)
        result[1:, :] += coeffs_list[3][1:, :] * q[:-1, :]
        return result

    def matvec2(q):
        return diagonal * q - neighbor_sum(q, coeffs)

    dp = torch.zeros_like(rhs)
    r = rhs.clone()
    norm = torch.linalg.vector_norm(rhs).item()
    if norm > 0:
        z = r / diagonal
        direction, rz = z.clone(), torch.sum(r * z)
        target = max(norm * 1e-10, 1e-14)
        for _ in range(max(200, nx * ny)):
            ad = matvec2(direction)
            denom = torch.sum(direction * ad)
            if denom.item() <= 0 or not torch.isfinite(denom):
                break
            step = rz / denom
            dp += step * direction
            r -= step * ad
            if torch.linalg.vector_norm(r).item() <= target:
                break
            z = r / diagonal
            new_rz = torch.sum(r * z)
            direction = z + (new_rz / rz) * direction
            rz = new_rz

    # Velocity correction
    dpdx = (dp[:, 1:] - dp[:, :-1]) / solver.dx
    dpdy = (dp[1:, :] - dp[:-1, :]) / solver.dy
    solver.u[1:-1, 1:-1] -= dt * inv_rho_u[1:-1, :] * dpdx
    solver.v[1:-1, 1:-1] -= dt * inv_rho_v[1:-1, :] * dpdy

    # Add CSF force to velocity AFTER correction (semi-implicit)
    if f_u is not None and f_v is not None:
        solver.u[1:-1, 1:-1] += dt * f_u[1:-1, :] / rho_u_face[1:-1, :]
        solver.v[1:-1, 1:-1] += dt * f_v[1:-1, :] / rho_v_face[1:-1, :]

    solver.p += dp
    solver._boundaries(projection=True)

    # VOF advection
    solver.alpha = solver.vof.advect(solver.alpha, solver.u, solver.v, dt)
    solver._update_properties()
    solver.time += dt

    div_max = torch.max(torch.abs(solver._divergence())).item()
    vel_change = max(
        torch.max(torch.abs(solver.u - u_old)).item(),
        torch.max(torch.abs(solver.v - v_old)).item(),
    )
    v_drift = solver.vof.volume_drift(solver.alpha, solver.vof.total_volume(solver.alpha))

    return {
        "iteration": len(solver.history) + 1,
        "time": solver.time,
        "continuity": div_max,
        "velocity_change": vel_change,
        "volume_drift": v_drift,
    }


def run_comparison(n=32, radius=0.2, sigma=0.07, mu=1e-2, steps=10, dt=1e-5):
    """Run both original and fixed solvers and compare."""
    torch.set_num_threads(1)
    exact_jump = sigma / radius

    # Original solver (diverges)
    solver_orig, rr = _solver(n=n, radius=radius, sigma=sigma, mu=mu)
    history_orig = []
    diverged_orig = False
    for i in range(steps):
        try:
            item = solver_orig.step(dt)
            history_orig.append(item)
            if item["continuity"] > 1e10:
                diverged_orig = True
                break
        except RuntimeError:
            diverged_orig = True
            break

    # Fixed solver
    solver_fix, rr = _solver(n=n, radius=radius, sigma=sigma, mu=mu)
    history_fix = []
    for i in range(steps):
        item = _projection_step_fixed(solver_fix, dt)
        history_fix.append(item)
        if item["continuity"] > 1e10 or not torch.isfinite(solver_fix.p).all():
            break

    # Compute pressure jump for fixed solver
    inside = float(solver_fix.p[rr < 0.5 * radius].mean())
    outside = float(solver_fix.p[rr > 1.6 * radius].mean())
    computed_jump = abs(inside - outside)
    rel_error = abs(computed_jump - exact_jump) / exact_jump

    # Capillary number
    umax = max(float(solver_fix.u.abs().max()), float(solver_fix.v.abs().max()))
    capillary_number = mu * umax / sigma if sigma > 0 else 0

    return {
        "n": n,
        "radius": radius,
        "sigma": sigma,
        "mu": mu,
        "dt": dt,
        "steps": steps,
        "exact_jump_pa": exact_jump,
        "original": {
            "diverged": diverged_orig,
            "final_divergence": history_orig[-1]["continuity"] if history_orig else None,
            "history": history_orig,
        },
        "fixed": {
            "computed_jump_pa": computed_jump,
            "inside_pressure_pa": inside,
            "outside_pressure_pa": outside,
            "relative_error": rel_error,
            "umax": umax,
            "capillary_number": capillary_number,
            "final_divergence": history_fix[-1]["continuity"] if history_fix else None,
            "history": history_fix,
        },
        "criteria": {
            "C1_young_laplace_within_3pct": rel_error < 0.03,
            "C2_capillary_number_below_0p01": capillary_number < 0.01,
            "C3_finite_iterates": torch.isfinite(solver_fix.p).all().item(),
            "C4_divergence_bounded": (history_fix[-1]["continuity"] < 1e-6
                                       if history_fix else False),
        },
    }


def main():
    outdir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(outdir, exist_ok=True)

    results = []
    for n in [24, 32]:
        print(f"Running n={n}...")
        r = run_comparison(n=n)
        results.append(r)
        print(f"  Fixed: dp={r['fixed']['computed_jump_pa']:.4g} Pa "
              f"(exact={r['exact_jump_pa']:.4g}, err={r['fixed']['relative_error']:.2%})")
        print(f"  Ca={r['fixed']['capillary_number']:.2e}, "
              f"div={r['fixed']['final_divergence']:.2e}")

    # Summary
    summary = {
        "schema": "tensorfvm.two-phase-surface-tension-fix/1",
        "results": results,
        "all_criteria_passed": all(
            all(r["criteria"].values()) for r in results
        ),
    }

    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Divergence history
    for r in results:
        ax = axes[0]
        orig_hist = r["original"]["history"]
        fix_hist = r["fixed"]["history"]
        if orig_hist:
            ax.semilogy([h["iteration"] for h in orig_hist],
                        [h["continuity"] for h in orig_hist],
                        'r--o', label=f"Original n={r['n']}")
        if fix_hist:
            ax.semilogy([h["iteration"] for h in fix_hist],
                        [h["continuity"] for h in fix_hist],
                        'b-o', label=f"Fixed n={r['n']}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Max |∇·u|")
    ax.set_title("Divergence History")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Pressure jump comparison
    ax = axes[1]
    ns = [r["n"] for r in results]
    exact = [r["exact_jump_pa"] for r in results]
    computed = [r["fixed"]["computed_jump_pa"] for r in results]
    x = np.arange(len(ns))
    width = 0.35
    ax.bar(x - width/2, exact, width, label="Exact (σ/R)", color='gray', alpha=0.7)
    ax.bar(x + width/2, computed, width, label="Fixed solver", color='blue', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"n={n}" for n in ns])
    ax.set_ylabel("Pressure jump (Pa)")
    ax.set_title("Young-Laplace Jump Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "comparison.png"), dpi=150)
    plt.close()

    print(f"\nSummary written to {outdir}/summary.json")
    print(f"Figure written to {outdir}/comparison.png")
    print(f"All criteria passed: {summary['all_criteria_passed']}")


if __name__ == "__main__":
    main()
