"""Verification anchors for the repaired transient two-phase projection path.

Produces ``evidence.json`` (programmatic, zero hand-copy) plus three figures in
``scripts/verification-two-phase-dynamics/``.

Anchors (all analytic, no external data):
1. stratified liquid/gas column at rest under gravity  ->  u = v = 0 exactly;
2. static 2D drop with surface tension                 ->  dp = sigma / R;
3. fully sealed domain                                 ->  ValueError (no datum).

Run:  PYTHONPATH=src python scripts/verify_two_phase_dynamics.py
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))

from tensorfvm.two_phase_solver import (  # noqa: E402
    GridConfig,
    TwoPhaseConfig,
    TwoPhaseSolver,
)

torch.set_num_threads(1)

OUT = pathlib.Path(__file__).resolve().parent / 'verification-two-phase-dynamics'
RHO_L, RHO_G, MU_G = 1000.0, 1.0, 1e-3


def build(n=16, gravity=9.81, sigma=0.0, surface_tension=False, mu=1e-3,
          top_atmosphere=True, drop_radius=None, dt=1e-5):
    tp = TwoPhaseConfig(
        enabled=True, rho_liquid=RHO_L, rho_gas=RHO_G,
        mu_liquid=mu, mu_gas=MU_G, gravity=gravity,
        sigma=sigma, surface_tension=surface_tension,
        top_atmosphere=top_atmosphere,
    )
    solver = TwoPhaseSolver(GridConfig(nx=n, ny=n, length=1.0, height=1.0), tp)
    x = (torch.arange(n, dtype=torch.float64) + 0.5) * solver.dx
    y = (torch.arange(n, dtype=torch.float64) + 0.5) * solver.dy
    if drop_radius is None:
        yy = y.reshape(-1, 1).expand(n, n)
        solver.set_alpha(torch.where(yy < 0.5, torch.ones_like(yy),
                                     torch.zeros_like(yy)))
        solver.initialize_hydrostatic_pressure(0.5)
        return solver, dt
    rr = torch.sqrt((x[None, :] - 0.5) ** 2 + (y[:, None] - 0.5) ** 2)
    alpha = torch.clamp((drop_radius - rr) / max(solver.dx, solver.dy) + 0.5, 0.0, 1.0)
    solver.set_alpha(alpha)
    # hydrostatic reference pressure of the gas above the drop is irrelevant for
    # the jump; start from a zero pressure field so the jump is the only signal.
    solver.p.zero_()
    return solver, dt


def rest_column(n=16, steps=20, dt=5e-6):
    """Anchor 1: gravity balanced by the analytic pressure gradient."""
    solver, _ = build(n=n, dt=dt)
    history = []
    for k in range(steps):
        solver.step(dt)
        history.append({
            'step': k + 1,
            'max_abs_u': float(solver.u.abs().max()),
            'max_abs_v': float(solver.v.abs().max()),
            'max_abs_div': float(solver._divergence().abs().max()),
            'max_abs_p': float(solver.p.abs().max()),
        })
    exact_p = RHO_L * 9.81 * (0.5 - 0.5 * solver.dy)
    return {
        'n': n, 'steps': steps, 'dt': dt,
        'max_abs_u_final': history[-1]['max_abs_u'],
        'max_abs_v_final': history[-1]['max_abs_v'],
        'max_abs_div_final': history[-1]['max_abs_div'],
        'p_scale_reference': exact_p,
        'history': history,
    }


def laplace_drop(n=32, sigma=0.07, radius=0.2, steps=10, dt=2e-6, mu=1e-2):
    """Anchor 2: 2D Young-Laplace pressure jump of a static drop."""
    solver, _ = build(n=n, gravity=0.0, sigma=sigma, surface_tension=True,
                      mu=mu, drop_radius=radius, dt=dt)
    x = (torch.arange(n, dtype=torch.float64) + 0.5) * solver.dx
    y = (torch.arange(n, dtype=torch.float64) + 0.5) * solver.dy
    rr = torch.sqrt((x[None, :] - 0.5) ** 2 + (y[:, None] - 0.5) ** 2)
    for _ in range(steps):
        solver.step(dt)
    inside = float(solver.p[rr < 0.5 * radius].mean())
    outside = float(solver.p[rr > 1.6 * radius].mean())
    measured = inside - outside
    exact = sigma / radius
    area = float(solver.alpha.sum()) * solver.dx * solver.dy
    force = solver._csf_force()
    return {
        'n': n, 'sigma': sigma, 'radius': radius, 'steps': steps, 'dt': dt,
        'delta_p_measured': measured, 'delta_p_exact': exact,
        'relative_error': abs(measured - exact) / exact,
        'max_abs_u': float(solver.u.abs().max()),
        'max_abs_v': float(solver.v.abs().max()),
        'max_csf_acceleration': float(max(force[0].abs().max(), force[1].abs().max())),
        'drop_area': area, 'drop_area_exact': math.pi * radius ** 2,
        'drop_area_relative_error': abs(area - math.pi * radius ** 2) / (math.pi * radius ** 2),
        'pressure_profile': {
            'r': rr[:, n // 2].tolist(), 'p': solver.p[:, n // 2].tolist(),
        },
    }


def sealed_domain(n=16):
    """Anchor 3: a sealed domain has no pressure datum and must be rejected."""
    solver, _ = build(n=n, gravity=9.81, top_atmosphere=False)
    try:
        solver.step(1e-5)
    except ValueError as exc:
        return {'rejected': True, 'error': str(exc)}
    return {'rejected': False, 'error': None,
            'max_abs_u': float(solver.u.abs().max())}


def single_phase_regression():
    """The two-phase switch must leave the single-phase path untouched."""
    tp = TwoPhaseConfig(enabled=False)
    solver = TwoPhaseSolver(GridConfig(nx=16, ny=16, density=1.0,
                                       inlet_velocity=1.0, max_iterations=20), tp)
    for _ in range(5):
        solver.step()
    return {'max_abs_u': float(solver.u.abs().max()),
            'finite': bool(torch.isfinite(solver.u).all())}


def figures(rest, drop):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    OUT.mkdir(parents=True, exist_ok=True)

    steps = [row['step'] for row in rest['history']]
    vmax = [row['max_abs_v'] for row in rest['history']]
    divv = [row['max_abs_div'] for row in rest['history']]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.semilogy(steps, [max(v, 1e-30) for v in vmax], 'o-', label=r'$\max|v|$')
    ax.semilogy(steps, [max(d, 1e-30) for d in divv], 's-', label=r'$\max|\nabla\cdot u|\,$[1/s]')
    ax.axhline(1e-9, color='k', ls='--', lw=1, label='acceptance 1e-9 m/s')
    ax.set_xlabel('step'); ax.set_ylabel('magnitude'); ax.set_yscale('log')
    ax.set_title('Rest column: exact solution is u = v = 0')
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(OUT / 'rest-column.png', dpi=140); plt.close(fig)

    r = drop['pressure_profile']['r']; p = drop['pressure_profile']['p']
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(r, p, 'o-', ms=3, label='TensorFVM $p(y=H/2)$')
    ax.axhline(drop['delta_p_exact'], color='k', ls='--', lw=1,
               label=r'analytic $\Delta p = \sigma/R$')
    ax.set_xlabel('r [m]'); ax.set_ylabel('p [Pa]')
    ax.set_title('Static drop: Young-Laplace jump (n=%d)' % drop['n'])
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(OUT / 'laplace-pressure.png', dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.0, 3.2))
    bars = ['measured', 'analytic']
    vals = [drop['delta_p_measured'], drop['delta_p_exact']]
    ax.bar(bars, vals, color=['tab:blue', 'tab:gray'])
    for i, v in enumerate(vals):
        ax.text(i, v, '%.4f' % v, ha='center', va='bottom', fontsize=8)
    ax.set_ylabel(r'$\Delta p$ [Pa]')
    ax.set_title('Young-Laplace: error %.2f %%' % (100 * drop['relative_error']))
    fig.tight_layout(); fig.savefig(OUT / 'laplace-bar.png', dpi=140); plt.close(fig)


def main():
    results = {
        'rest_column': rest_column(),
        'laplace_drop': laplace_drop(),
        'sealed_domain': sealed_domain(),
        'single_phase_regression': single_phase_regression(),
    }
    results['verdict'] = {
        'rest_column_below_1e-9': results['rest_column']['max_abs_v_final'] < 1e-9
        and results['rest_column']['max_abs_u_final'] < 1e-9,
        'laplace_below_15pct': results['laplace_drop']['relative_error'] < 0.15,
        'laplace_below_3pct': results['laplace_drop']['relative_error'] < 0.03,
        'sealed_rejected': results['sealed_domain']['rejected'],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'evidence.json').write_text(json.dumps(results, indent=1))
    figures(results['rest_column'], results['laplace_drop'])
    summary = {k: results['verdict'][k] for k in results['verdict']}
    print(json.dumps({
        'verdict': summary,
        'rest_max_v': results['rest_column']['max_abs_v_final'],
        'rest_max_u': results['rest_column']['max_abs_u_final'],
        'laplace_dp': results['laplace_drop']['delta_p_measured'],
        'laplace_exact': results['laplace_drop']['delta_p_exact'],
        'laplace_rel_err': results['laplace_drop']['relative_error'],
        'sealed': results['sealed_domain'],
    }, indent=1))


if __name__ == '__main__':
    main()