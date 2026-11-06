"""Audit of the balanced-force CSF (surface-tension) chain in the two-phase solver.

Run with ``PYTHONPATH=src python scripts/audit_two_phase_csf_balance.py``.

Blocks (every number is written by this script; nothing is transcribed):

1. **Curvature vs the analytic circle.** A circular drop of radius ``R``
   (``kappa = 1/R``) is sampled on ``n x n`` grids; the discrete curvature of
   :func:`tensorfvm.two_phase_vof.interface_curvature` is compared with ``1/R``
   for an unsmoothed grid-scale profile and for the Brackbill--Kothe--Zemach
   smoothed one.
2. **Closed-interface force balance.** ``\\oint kappa n ds = 0``: the discrete
   CSF force integrated over the whole domain must sum to zero (it is an
   internal force).
3. **Static drop, Young--Laplace.** ``dp = sigma/R`` in 2D.  Measured
   ``p(inside)-p(outside)`` after six steps of ``dt = 1e-5 s`` for the legacy
   (cell-force average) and the balanced (face force) discretisations, on a
   grid ladder, for two interface descriptions: a grid-scale ramp and a
   resolved ``tanh`` profile.
4. **Spurious currents.** ``max|u|`` of the static drop, normalised by the
   capillary velocity scale ``sigma/mu``.

Acceptance gate: ``|dp/(sigma/R) - 1| < 3 %``.
"""
import argparse
import json
from pathlib import Path

import torch

from tensorfvm.two_phase_solver import GridConfig, TwoPhaseConfig, TwoPhaseSolver
from tensorfvm.two_phase_vof import interface_curvature, smooth_alpha

GATE = 0.03
RADIUS = 0.2
SIGMA = 0.07
MU = 1e-2
RHO_L, RHO_G = 1000.0, 1.0
DT = 1e-5
STEPS = 6
GRIDS = (24, 32, 48, 64, 96)
SMOOTHING = 3


def _drop(n, profile):
    tp = TwoPhaseConfig(
        enabled=True, rho_liquid=RHO_L, rho_gas=RHO_G, mu_liquid=MU, mu_gas=MU,
        gravity=0.0, sigma=SIGMA, surface_tension=True, top_atmosphere=True,
    )
    solver = TwoPhaseSolver(GridConfig(nx=n, ny=n, length=1.0, height=1.0), tp)
    x = (torch.arange(n, dtype=torch.float64) + 0.5) * solver.dx
    rr = torch.sqrt((x[None, :] - 0.5) ** 2 + (x[:, None] - 0.5) ** 2)
    if profile == 'tanh':
        alpha = 0.5 * (1.0 - torch.tanh((rr - RADIUS) / solver.dx))
    else:
        alpha = torch.clamp((RADIUS - rr) / solver.dx + 0.5, 0.0, 1.0)
    solver.set_alpha(alpha)
    return solver, rr


def _grad_magnitude(alpha, dx, dy):
    gx = torch.zeros_like(alpha); gy = torch.zeros_like(alpha)
    gx[:, 1:-1] = (alpha[:, 2:] - alpha[:, :-2]) / (2 * dx)
    gy[1:-1, :] = (alpha[2:, :] - alpha[:-2, :]) / (2 * dy)
    return torch.sqrt(gx ** 2 + gy ** 2)


def curvature_block():
    rows = []
    for n in GRIDS:
        solver, _ = _drop(n, 'ramp')
        for smoothing, label in ((0, 'unsmoothed'), (SMOOTHING, 'smoothed')):
            a = smooth_alpha(solver.alpha, smoothing)
            kappa = interface_curvature(a, solver.dx, solver.dy)
            band = (a > 1e-3) & (a < 1 - 1e-3)
            g = _grad_magnitude(a, solver.dx, solver.dy)
            exact = 1.0 / RADIUS
            rows.append(dict(
                n=n, dx=solver.dx, profile='ramp', smoothing=smoothing, scheme=label,
                exact=exact,
                band_mean=float(kappa[band].mean()),
                band_max_abs=float(kappa[band].abs().max()),
                grad_weighted=float((kappa * g)[band].sum() / g[band].sum()),
                band_mean_error=float(kappa[band].mean() / exact - 1.0),
                grad_weighted_error=float((kappa * g)[band].sum() / g[band].sum() / exact - 1.0),
                band_cells=int(band.sum()),
            ))
    return rows


def force_balance_block():
    rows = []
    for n in GRIDS:
        solver, _ = _drop(n, 'ramp')
        for balance in ('legacy', 'balanced'):
            f = solver.props.csf(solver.alpha, solver.dx, solver.dy)
            if balance == 'balanced':
                kappa = interface_curvature(smooth_alpha(solver.alpha, SMOOTHING),
                                            solver.dx, solver.dy)
                gx = torch.zeros_like(solver.alpha); gy = torch.zeros_like(solver.alpha)
                gx[:, 1:-1] = (solver.alpha[:, 2:] - solver.alpha[:, :-2]) / (2 * solver.dx)
                gy[1:-1, :] = (solver.alpha[2:, :] - solver.alpha[:-2, :]) / (2 * solver.dy)
                f = SIGMA * kappa[..., None] * torch.stack([gx, gy], dim=-1)
            net = float(torch.linalg.vector_norm(f.sum(dim=(0, 1))))
            gross = float(f.abs().sum())
            rows.append(dict(n=n, scheme=balance, net_force=net, gross_force=gross,
                             imbalance=net / gross if gross else 0.0))
    return rows


def laplace_block():
    rows = []
    for profile in ('ramp', 'tanh'):
        for n in GRIDS:
            row = dict(profile=profile, n=n, dx=1.0 / n)
            for balance in ('legacy', 'balanced'):
                solver, rr = _drop(n, profile)
                solver.tp.csf_balance = balance
                div = []
                for _ in range(STEPS):
                    div.append(solver.step(DT)['continuity'])
                inside = float(solver.p[rr < 0.5 * RADIUS].mean())
                outside = float(solver.p[rr > 1.6 * RADIUS].mean())
                dp = inside - outside
                exact = SIGMA / RADIUS
                row[balance] = dict(
                    dp=dp, exact=exact, relative_error=dp / exact - 1.0,
                    p_max=float(solver.p.abs().max()),
                    u_max=float(solver.u.abs().max()),
                    u_max_over_capillary=float(solver.u.abs().max()) / (SIGMA / MU),
                    div_max=max(div),
                )
            rows.append(row)
    return rows


def analyze(rows):
    verdict = {}
    for profile in ('ramp', 'tanh'):
        for balance in ('legacy', 'balanced'):
            errs = [abs(r[balance]['relative_error'])
                    for r in rows if r['profile'] == profile]
            worst = max(errs)
            verdict[f'{profile}/{balance}'] = dict(
                worst_error=worst, mean_error=sum(errs) / len(errs),
                passes_gate=bool(worst < GATE))
    return verdict


def _figure(curvature, laplace, output):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - matplotlib optional
        return None
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
    ax = axes[0]
    for scheme in ('unsmoothed', 'smoothed'):
        pts = [r for r in curvature if r['scheme'] == scheme]
        ax.semilogy([r['dx'] for r in pts], [abs(r['band_mean_error']) for r in pts],
                    'o-', label=scheme)
    ax.set_xlabel('dx [m]'); ax.set_ylabel(r'$|\langle\kappa\rangle/(1/R)-1|$')
    ax.set_title('curvature of a circle, R=0.2'); ax.legend(); ax.grid(True, which='both', alpha=.3)

    ax = axes[1]
    for profile, style in (('ramp', '--'), ('tanh', '-')):
        for balance, colour in (('legacy', 'tab:red'), ('balanced', 'tab:blue')):
            pts = [r for r in laplace if r['profile'] == profile]
            ax.plot([r['dx'] for r in pts],
                    [100 * r[balance]['relative_error'] for r in pts],
                    style, marker='o', color=colour, label=f'{profile}/{balance}')
    ax.axhspan(-3, 3, color='0.85', zorder=0, label='+/-3% gate')
    ax.set_xlabel('dx [m]'); ax.set_ylabel(r'$dp/(\sigma/R)-1$ [%]')
    ax.set_title('2D Young-Laplace jump'); ax.legend(fontsize=7); ax.grid(alpha=.3)

    ax = axes[2]
    for profile, style in (('ramp', '--'), ('tanh', '-')):
        for balance, colour in (('legacy', 'tab:red'), ('balanced', 'tab:blue')):
            pts = [r for r in laplace if r['profile'] == profile]
            ax.loglog([r['dx'] for r in pts], [r[balance]['u_max_over_capillary'] for r in pts],
                      style, marker='o', color=colour, label=f'{profile}/{balance}')
    ax.set_xlabel('dx [m]'); ax.set_ylabel(r'$max|u|/(\sigma/\mu)$')
    ax.set_title('spurious currents'); ax.legend(fontsize=7); ax.grid(True, which='both', alpha=.3)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=130)
    plt.close(fig)
    return output


def _report(summary, path):
    lines = ['# Two-phase CSF balance: measured evidence', '',
             'Generated by `scripts/audit_two_phase_csf_balance.py`; every number below is',
             'written by that script (no transcription).', '',
             f'Gate: |dp/(sigma/R) - 1| < {GATE:.0%}. Drop R = {RADIUS} m, sigma = {SIGMA} N/m,',
             f'mu = {MU} Pa s, rho_l/rho_g = {RHO_L}/{RHO_G}, dt = {DT} s, {STEPS} steps.', '',
             '## Curvature of a circle (kappa = 1/R = 5 1/m)', '',
             '| n | dx | scheme | <kappa> band | error | grad-weighted error |',
             '|---|---|---|---|---|---|']
    for r in summary['curvature']:
        lines.append('| {n} | {dx:.5f} | {scheme} | {band_mean:.4f} | {band_mean_error:+.2%} | '
                     '{grad_weighted_error:+.2%} |'.format(**r))
    lines += ['', '## Closed-interface force balance', '',
              '| n | scheme | net | gross | imbalance |', '|---|---|---|---|---|']
    for r in summary['force_balance']:
        lines.append('| {n} | {scheme} | {net_force:.3e} | {gross_force:.4f} | {imbalance:.2e} |'
                     .format(**r))
    lines += ['', '## Static drop: Young-Laplace jump and spurious currents', '',
              '| profile | n | scheme | dp [Pa] | error | max|u| | max|u|/(sigma/mu) | max div |',
              '|---|---|---|---|---|---|---|---|']
    for r in summary['laplace']:
        for balance in ('legacy', 'balanced'):
            d = r[balance]
            lines.append('| {p} | {n} | {b} | {dp:.5f} | {e:+.2%} | {u:.2e} | {ur:.2e} | {dv:.1e} |'
                         .format(p=r['profile'], n=r['n'], b=balance, dp=d['dp'],
                                 e=d['relative_error'], u=d['u_max'],
                                 ur=d['u_max_over_capillary'], dv=d['div_max']))
    lines += ['', '## Verdict per series', '',
              '| series | worst |dp| error | mean error | passes 3% gate |', '|---|---|---|---|']
    for key, v in summary['verdict'].items():
        lines.append(f"| {key} | {v['worst_error']:.2%} | {v['mean_error']:.2%} | "
                     f"{'yes' if v['passes_gate'] else 'NO'} |")
    Path(path).write_text('\n'.join(lines) + '\n')


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='docs/two-phase-csf-balance')
    args = parser.parse_args(argv)
    out = Path(args.output)

    curvature = curvature_block()
    forces = force_balance_block()
    laplace = laplace_block()
    summary = dict(
        schema='tensorfvm.two-phase-csf-balance/1',
        parameters=dict(radius_m=RADIUS, sigma_n_m=SIGMA, mu_pa_s=MU,
                        rho_liquid=RHO_L, rho_gas=RHO_G, dt_s=DT, steps=STEPS,
                        grids=list(GRIDS), csf_smoothing=SMOOTHING, gate=GATE),
        curvature=curvature, force_balance=forces, laplace=laplace,
        verdict=analyze(laplace),
    )
    out.mkdir(parents=True, exist_ok=True)
    fig = _figure(curvature, laplace, out / 'csf-balance.png')
    summary['figure'] = str(fig) if fig else None
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    _report(summary, out / 'report.md')
    print(json.dumps(summary['verdict'], indent=1))
    for r in laplace:
        print(r['profile'], r['n'],
              'legacy %+.2f%%' % (100 * r['legacy']['relative_error']),
              'balanced %+.2f%%' % (100 * r['balanced']['relative_error']))
    return summary


if __name__ == '__main__':
    main()