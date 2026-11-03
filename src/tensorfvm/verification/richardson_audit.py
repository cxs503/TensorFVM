"""Self-verification audit of the Richardson/GCI verification tool itself.

Run with ``python -m tensorfvm.verification.richardson_audit --output DIR``.

Three independent blocks, all programmatically generated (no hand-copied number):

1. **Tool against closed form** -- synthetic refinement series
   ``f(h) = f_exact + C h^p`` with known ``p`` and ``f_exact`` must return the
   order and the limit to round-off; non-uniform refinement ratios are included
   because the legacy ``core.observed_orders`` cannot represent them.
2. **Negative controls** -- a series with a non-negligible higher-order term
   (outside the asymptotic range), an inconsistent (sign-alternating) series and
   a non-refining series must be rejected or flagged, never reported as a
   confident band.
3. **Application instance** -- the annular fully developed laminar backend
   (:mod:`tensorfvm.annular`, closed-form Bessel-free Poiseuille solution) is
   refined ``32x8 -> 64x16 -> 128x32 -> 256x64``; the tool must recover the
   formal second order, extrapolate below the 3% acceptance gate and report a
   band that covers the analytic value.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from .core import observed_orders
from .richardson import (power_law_series, reference_agreement,
                         richardson_gci)

GATE = .03

SYNTHETIC_ORDERS = (1., 1.5, 2., 3.)
SYNTHETIC_UNIFORM = (1./32, 1./16, 1./8)
SYNTHETIC_NONUNIFORM = (1./32, 1./20, 1./8)


def synthetic_block():
    rows = []
    for spacings, label in ((SYNTHETIC_UNIFORM, 'uniform r=2'),
                            (SYNTHETIC_NONUNIFORM, 'non-uniform r=1.6,2.5')):
        for order in SYNTHETIC_ORDERS:
            values = power_law_series(1., .7, order, spacings)
            study = richardson_gci(*values, spacings, theory_order=order)
            # The legacy helper returns per-pair orders for a metric whose limit
            # is zero (an error).  Applying it to the raw metric Q(h) -- the
            # normal Richardson situation -- is exactly the gap this tool closes.
            legacy = observed_orders(
                [dict(spacing_m=h, metrics=[dict(name='m', value=v)])
                 for h, v in zip(spacings, values)], 'm')
            rows.append(dict(block='synthetic', case=label, spacing_m=list(spacings),
                             values=values, theory_order=order,
                             order=study.order, order_error=abs(study.order-order),
                             legacy_orders=legacy,
                             legacy_order_error_raw_metric=abs(legacy[0]-order),
                             extrapolated=study.extrapolated,
                             extrapolated_error=abs(study.extrapolated-1.),
                             asymptotic=study.asymptotic, gci=study.gci,
                             band_covered=reference_agreement(study, 1.)['band_covered']))
    return rows


def negative_block():
    rows = []
    # (a) higher-order term of the same size as the primary term: pre-asymptotic.
    for h0 in (.4, .1, .025):
        spacings = [h0/4, h0/2, h0]
        values = power_law_series(1., .5, 2., spacings, secondary=4., secondary_order=4.)
        study = richardson_gci(*values, spacings, theory_order=2.)
        rows.append(dict(block='negative', case=f'pre-asymptotic h_max={h0}',
                         spacing_m=spacings, values=values, theory_order=2.,
                         order=study.order, convergence=study.convergence,
                         asymptotic=study.asymptotic,
                         extrapolated=study.extrapolated,
                         extrapolated_error=abs(study.extrapolated-1.),
                         gci=study.gci,
                         band_covered=reference_agreement(study, 1.)['band_covered']))
    # (b) sign-alternating error: two grids already disagree about the limit.
    for values in ((.1, .08, .09), (.1, .09, .095)):
        study = richardson_gci(*values, SYNTHETIC_UNIFORM, theory_order=2.)
        rows.append(dict(block='negative', case=f'oscillatory {values}',
                         spacing_m=list(SYNTHETIC_UNIFORM), values=list(values),
                         theory_order=2., order=study.order,
                         convergence=study.convergence, asymptotic=study.asymptotic,
                         extrapolated=study.extrapolated, gci=study.gci,
                         safety_factor=study.safety_factor))
    return rows


def annular_block(grids=((32, 8), (64, 16), (128, 32), (256, 64))):
    """Refine the production annular diffusion backend against its closed form.

    The verified quantity is the axial flow rate ``Q = sum_cells u_c V_c``; the
    reference is the closed-form annular Poiseuille flow rate of
    :func:`tensorfvm.annular.analytic_flow`, not a computed field.
    """
    import torch
    from types import SimpleNamespace
    from ..annular import AnnularConfig, analytic_flow, solve
    torch.set_num_threads(1)
    rows = []
    for ntheta, nr in grids:
        config = AnnularConfig(ntheta=ntheta, nr=nr)
        result = solve(config)
        fields = result['fields']
        reference = np.asarray(fields['reference_axial_velocity_m_s'], dtype=float)
        velocity = np.asarray(fields['axial_velocity_m_s'], dtype=float)
        volume = np.asarray(fields['cell_volumes_m2'], dtype=float)
        flow_rate = float((velocity.ravel()*volume.ravel()).sum())
        analytic = float(analytic_flow(SimpleNamespace(
            inner_radius_m=config.inner_radius_m, outer_radius_m=config.outer_radius_m,
            pressure_gradient_pa_m=config.pressure_gradient_pa_m,
            viscosity_pa_s=config.viscosity_pa_s)))
        rows.append(dict(block='annular', case=f'{ntheta}x{nr}',
                         ntheta=ntheta, nr=nr, spacing_m=1./ntheta,
                         flow_rate_m3_s=flow_rate, analytic_flow_rate_m3_s=analytic,
                         relative_flow_error=abs(flow_rate-analytic)/abs(analytic),
                         relative_l2=float(np.linalg.norm(velocity-reference)/np.linalg.norm(reference)),
                         relative_linf=float(np.abs(velocity-reference).max()/np.abs(reference).max()),
                         linear_iterations=result['linear_iterations'],
                         true_residual=result['diagnostics']['true_production_relative_residual'],
                         passed_single_grid_3pct_gate=bool(
                             abs(flow_rate-analytic)/abs(analytic) < GATE)))
    return rows


def analyze(rows):
    """Run the three-grid analysis on the annular flow rate, fine grid first."""
    annular = sorted((r for r in rows if r['block'] == 'annular'),
                     key=lambda r: r['spacing_m'])
    h = [r['spacing_m'] for r in annular]
    q = [r['flow_rate_m3_s'] for r in annular]
    analytic = annular[0]['analytic_flow_rate_m3_s']
    studies = []
    for i in range(len(h)-2):
        triple_h, triple_q = h[i:i+3], q[i:i+3]
        study = richardson_gci(*triple_q, triple_h, theory_order=2.)
        agreement = reference_agreement(study, analytic)
        studies.append(dict(case=' -> '.join(r['case'] for r in annular[i:i+3]),
                            spacing_m=triple_h, flow_rate_m3_s=triple_q,
                            order=study.order, convergence=study.convergence,
                            asymptotic=study.asymptotic,
                            extrapolated_flow_rate_m3_s=study.extrapolated,
                            extrapolated_error=agreement['relative_error'],
                            gci=study.gci, safety_factor=study.safety_factor,
                            band=study.band,
                            band_covered=agreement['band_covered'],
                            efficiency_index=agreement['efficiency_index'],
                            improvement_factor=float(
                                annular[i]['relative_flow_error']/agreement['relative_error'])
                            if agreement['relative_error'] > 0 else float('inf')))
    return studies


def _clean(value):
    """Replace non-finite floats by ``None`` so the JSON stays standards-compliant."""
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def publish(output, rows, studies):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output.mkdir(parents=True, exist_ok=True)
    figures = output/'figures'
    figures.mkdir(exist_ok=True)

    annular = sorted((r for r in rows if r['block'] == 'annular'),
                     key=lambda r: r['spacing_m'])
    fig, ax = plt.subplots(figsize=(6, 4.2))
    hs = np.array([r['spacing_m'] for r in annular])
    es = np.array([r['relative_flow_error'] for r in annular])*100
    ax.loglog(hs, es, 'o-', color='C0', label='computed flow-rate error vs closed form')
    finest = studies[-1]
    span = np.array([hs.min(), hs.max()])
    ax.loglog(span, es[0]*(span/hs.min())**finest['order'], '--', color='C0',
              label=f"slope $p$={finest['order']:.3f}")
    ax.axhline(GATE*100, color='k', ls=':', label='3% acceptance gate')
    ax.set(xlabel='grid spacing h = 1/n_theta (m)', ylabel='relative flow-rate error (%)',
           title='Annular Poiseuille: refinement, observed order, gate')
    ax.grid(which='both', alpha=.3)
    ax.legend(fontsize=8)
    fig.savefig(figures/'annular-refinement.png', dpi=140, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.2))
    labels = [f"p={r['theory_order']:g}\n{r['case'].split()[0]}" for r in rows
              if r['block'] == 'synthetic']
    general = [max(r['order_error'], 1e-16) for r in rows if r['block'] == 'synthetic']
    naive = [max(r['legacy_order_error_raw_metric'], 1e-16)
             for r in rows if r['block'] == 'synthetic']
    x = np.arange(len(labels))
    ax.bar(x-.2, general, .4, label='general three-grid solver (this module)', color='C0')
    ax.bar(x+.2, naive, .4, color='C2',
           label='legacy core.observed_orders applied to the raw metric')
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set(ylabel='|p_observed - p_theory|',
           title='Order-recovery error on synthetic data of known order')
    ax.grid(axis='y', which='both', alpha=.3)
    ax.legend(fontsize=8)
    fig.savefig(figures/'order-recovery.png', dpi=140, bbox_inches='tight')
    plt.close(fig)

    (output/'summary.json').write_text(json.dumps(_clean(dict(rows=rows, studies=studies)),
                                                  indent=2, allow_nan=False))
    lines = ['# Richardson/GCI tool: self-verification and annular application', '',
             '## Annular Poiseuille refinement (closed-form reference)', '',
             '| grid | h (m) | flow rate $Q$ (m3/s) | analytic $Q$ | relative $Q$ error (%) | field $L_2$ (%) | field $L_\infty$ (%) | linear its | true residual | single-grid 3% gate |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in annular:
        lines.append(f"| {r['case']} | {r['spacing_m']:.5f} | {r['flow_rate_m3_s']:.10e} | "
                     f"{r['analytic_flow_rate_m3_s']:.10e} | {100*r['relative_flow_error']:.4f} | "
                     f"{100*r['relative_l2']:.4f} | {100*r['relative_linf']:.4f} | "
                     f"{r['linear_iterations']} | {r['true_residual']:.2e} | "
                     f"{r['passed_single_grid_3pct_gate']} |")
    lines += ['', '| triple (fine first) | p observed | asymptotic | extrapolated $Q$ error (%) | GCI (%) | band covers analytic $Q$ | improvement over fine grid |',
              '|---|---:|---|---:|---:|---|---:|']
    for s in studies:
        lines.append(f"| {s['case']} | {s['order']:.4f} | {s['asymptotic']} | "
                     f"{100*s['extrapolated_error']:.6f} | {100*s['gci']:.4f} | "
                     f"{s['band_covered']} | {s['improvement_factor']:.1f}x |")
    (output/'tables.md').write_text('\n'.join(lines)+'\n')
    return figures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--grids', nargs='+', default=['32x8', '64x16', '128x32', '256x64'])
    args = parser.parse_args(argv)
    grids = [tuple(int(v) for v in g.split('x')) for g in args.grids]
    rows = synthetic_block()+negative_block()+annular_block(grids)
    studies = analyze(rows)
    publish(args.output, rows, studies)
    print((args.output/'tables.md').read_text())


if __name__ == '__main__':
    main()