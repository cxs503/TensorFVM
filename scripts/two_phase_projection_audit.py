"""Audit the two-phase transient projection path.

Produces ``outputs/two-phase-projection-audit/metrics.json`` and a two-panel
figure.  Every number quoted in ``docs/two-phase-projection-audit.md`` comes
from this script.
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from tensorfvm.two_phase_solver import GridConfig, TwoPhaseConfig, TwoPhaseSolver

OUT = Path('outputs/two-phase-projection-audit')
OUT.mkdir(parents=True, exist_ok=True)


def column(n=32, mu=1e-3):
    tp = TwoPhaseConfig(enabled=True, rho_liquid=1000.0, rho_gas=1.0,
                        mu_liquid=mu, mu_gas=mu, gravity=9.81)
    s = TwoPhaseSolver(GridConfig(nx=n, ny=n, length=1.0, height=1.0), tp)
    y = (torch.arange(n, dtype=torch.float64) + 0.5) * s.dy
    yy = y.reshape(-1, 1).expand(n, n)
    s.set_alpha(torch.where(yy < 0.5, torch.ones_like(yy), torch.zeros_like(yy)))
    s.initialize_hydrostatic_pressure(0.5)
    residual = s.hydrostatic_balance_residual()
    history = []
    for _ in range(10):
        s.step(1e-5)
        history.append(float(s.v.abs().max()))
    return dict(max_force_residual=residual['max_residual'],
                rms_force_residual=residual['rms_residual'],
                final_vmax=history[-1], final_umax=float(s.u.abs().max()),
                v_history=history, pressure=[float(v) for v in s.p[:, 0]],
                gravity=9.81, rho_l=1000.0, rho_g=1.0, dy=s.dy)


def drop(n=48, radius=0.2, sigma=0.07, mu=1e-2, steps=20):
    tp = TwoPhaseConfig(enabled=True, rho_liquid=1000.0, rho_gas=1.0,
                        mu_liquid=mu, mu_gas=mu, gravity=0.0,
                        sigma=sigma, surface_tension=True)
    s = TwoPhaseSolver(GridConfig(nx=n, ny=n, length=1.0, height=1.0), tp)
    x = (torch.arange(n, dtype=torch.float64) + 0.5) * s.dx
    y = (torch.arange(n, dtype=torch.float64) + 0.5) * s.dy
    rr = torch.sqrt((x[None, :] - 0.5) ** 2 + (y[:, None] - 0.5) ** 2)
    s.set_alpha(torch.clamp((radius - rr) / max(s.dx, s.dy) + 0.5, 0.0, 1.0))
    v_history = []
    for _ in range(steps):
        s.step(1e-5)
        v_history.append(float(s.v.abs().max()))
    inside = float(s.p[rr < 0.5 * radius].mean())
    outside = float(s.p[rr > 1.6 * radius].mean())
    return dict(radius=radius, sigma=sigma, dp=inside - outside, exact=sigma / radius,
                rel_error=abs((inside - outside) - sigma / radius) / (sigma / radius),
                v_history=v_history, sigma_enabled_force=(lambda fv: float(fv.abs().max()) if fv is not None else 0.0)(s._csf_face_forces()[1]))


def divergence_removal(n=32):
    tp = TwoPhaseConfig(enabled=True, rho_liquid=1000.0, rho_gas=1000.0,
                        mu_liquid=1e-2, mu_gas=1e-2, gravity=0.0)
    s = TwoPhaseSolver(GridConfig(nx=n, ny=n, length=1.0, height=1.0), tp)
    torch.manual_seed(0)
    u = torch.randn(n, n + 1, dtype=torch.float64) * 1e-3
    v = torch.randn(n + 1, n, dtype=torch.float64) * 1e-3
    u[:, 0] = 0.0
    u[:, -1] = 0.0
    v[0, :] = 0.0
    div0 = (u[:, 1:] - u[:, :-1]) / s.dx + (v[1:] - v[:-1]) / s.dy
    phi = s._projection_pressure(u, v, 1e-5)
    iu = 2.0 / (s.rho_field[:, :-1] + s.rho_field[:, 1:])
    iv = 2.0 / (s.rho_field[:-1, :] + s.rho_field[1:, :])
    un, vn = u.clone(), v.clone()
    un[:, 1:-1] = u[:, 1:-1] - 1e-5 * iu * (phi[:, 1:] - phi[:, :-1]) / s.dx
    vn[1:-1, :] = v[1:-1, :] - 1e-5 * iv * (phi[1:] - phi[:-1]) / s.dy
    div1 = (un[:, 1:] - un[:, :-1]) / s.dx + (vn[1:] - vn[:-1]) / s.dy
    return dict(div_before_max=float(div0.abs().max()),
                div_after_max=float(div1.abs().max()),
                div_before_median=float(div0.abs().median()),
                div_after_median=float(div1.abs().median()),
                cg=s.cg_info)


def main():
    metrics = dict(column=column(), drops=[drop(radius=r) for r in (0.16, 0.20, 0.24)],
                   divergence=divergence_removal())
    (OUT / 'metrics.json').write_text(json.dumps(metrics, indent=1))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    col = metrics['column']
    axes[0].plot(col['v_history'], 'o-', label='|v|max  (rest column)')
    axes[0].axhline(1e-9, ls='--', c='k', label='acceptance 1e-9')
    axes[0].set_yscale('log')
    axes[0].set_xlabel('step'); axes[0].set_ylabel('|v|max [m/s]')
    axes[0].set_title('Hydrostatic column at rest (10 steps)')
    axes[0].legend()
    for d in metrics['drops']:
        axes[1].plot(d['v_history'], label='R=%.2f  dp/exact=%.2e'
                                         % (d['radius'], d['dp'] / d['exact']))
    axes[1].set_yscale('log')
    axes[1].set_xlabel('step'); axes[1].set_ylabel('|v|max [m/s]')
    axes[1].set_title('Static drop with CSF: parasitic currents (open defect)')
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / 'projection-audit.png', dpi=130)
    print(json.dumps({k: v for k, v in metrics.items() if k != 'column'}, indent=1)[:1200])
    print('column: vmax=%.3e umax=%.3e force_resid=%.3e'
          % (col['final_vmax'], col['final_umax'], col['max_force_residual']))


if __name__ == '__main__':
    main()
