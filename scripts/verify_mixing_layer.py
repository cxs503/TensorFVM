"""Producer for the temporal mixing-layer verification bundle.

Every number published in ``docs/repro-mixing-layer-temporal/`` comes from this
script; nothing is transcribed by hand.

The case: two-stream shear layer ``u0 = U tanh(y/delta)`` (free streams +-U,
``dU = 2U``) on a box that is periodic in x and free-slip in y, evolved in time
with :class:`tensorfvm.mixing_layer.TemporalMixingLayer`.

References used for the comparison (all computed here, none copied):

* **exact**: for a unidirectional field ``u(y,t)`` with ``v = 0`` and no external
  force the incompressible Navier--Stokes equations reduce to the heat equation,
  so ``u(y,t) = int G(y-y',t) tanh(y'/delta) dy'`` is an exact solution.  The
  diffusion phase is checked against it.
* **linear stability**: the Rayleigh (inviscid) and Orr--Sommerfeld (viscous)
  eigenproblems are solved by :mod:`tensorfvm.mixing_layer` with an independent
  discretisation (uniform-grid second-order, interior generalised eigenproblem),
  cross-checked against each other in the ``Re -> infinity`` limit and against
  their own grid refinement.

Stages can be run separately (each writes its own JSON) so the producer can be
chunked; ``--stage all`` runs them in order::

    PYTHONPATH=src python3 scripts/verify_mixing_layer.py --output /tmp/mixing --stage all
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from tensorfvm.mixing_layer import (
    MixingLayerConfig,
    TemporalMixingLayer,
    growth_rate_fit,
    heat_kernel_profile,
    layer_metrics,
    linear_fit,
    orr_sommerfeld_modes,
    rayleigh_modes,
)

HALF_VELOCITY = 1.0
THICKNESS = 1.0
VISCOSITY = 2e-3                      # Re_delta = U delta / nu = 500
PRODUCTION_ALPHA = 0.5                # alpha * delta
A0 = 1e-5                             # linear-phase seed (fraction of dU)
A0_NONLINEAR = 2e-2
LINEAR_HORIZON = 9.0
NONLINEAR_HORIZON = 50.0
GROWTH_WINDOW = (0.20 * NONLINEAR_HORIZON, 0.56 * NONLINEAR_HORIZON)
STABILITY_NODES = 301
ALPHA_SWEEP = np.round(np.arange(0.20, 0.81, 0.05), 3)
STAGES = ('stability', 'diffusion', 'linear', 'nonlinear', 'similarity', 'report')


# --------------------------------------------------------------------- helpers
def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    return path


def _read_json(path: Path):
    if not path.exists():
        raise SystemExit('missing stage artefact: %s' % path)
    return json.loads(path.read_text())


def linear_reference(alpha: float, delta: float) -> dict:
    """Orr--Sommerfeld and Rayleigh growth rates for a layer of scale ``delta``."""
    reynolds = HALF_VELOCITY * delta / VISCOSITY
    os_modes = orr_sommerfeld_modes(alpha, reynolds, thickness=delta,
                                    nodes=STABILITY_NODES, keep=1)
    ray_modes = rayleigh_modes(alpha, thickness=delta, nodes=STABILITY_NODES, keep=1)
    return {
        'alpha': float(alpha), 'delta': float(delta),
        'reynolds_delta': float(reynolds),
        'viscous_growth': float(os_modes[0].growth) if os_modes else None,
        'inviscid_growth': float(ray_modes[0].growth) if ray_modes else None,
        'viscous_c': [float(os_modes[0].c.real), float(os_modes[0].c.imag)]
                     if os_modes else None,
    }


def build_solver(nx: int, ny: int, alpha: float) -> TemporalMixingLayer:
    cfg = MixingLayerConfig(nx=nx, ny=ny, wavelength=2 * math.pi / alpha,
                            half_velocity=HALF_VELOCITY, thickness=THICKNESS,
                            viscosity=VISCOSITY)
    return TemporalMixingLayer(cfg)


def sample_until(solver: TemporalMixingLayer, horizon: float, dt: float,
                 samples: int, store_profile: bool = False) -> list[dict]:
    history: list[dict] = []
    interval = horizon / samples
    next_sample = 0.0
    while solver.time < horizon - 1e-12:
        solver.step(min(dt, horizon - solver.time))
        if solver.time + 1e-12 >= next_sample:
            record = solver.diagnostics()
            if store_profile:
                record['mean_profile'] = solver.mean_profile().tolist()
            history.append(record)
            next_sample = next_sample + interval
    return history


# ------------------------------------------------------------------ study parts
def study_stability(out: Path) -> dict:
    """Rayleigh / Orr--Sommerfeld eigenvalues: sweep, refinement, inviscid limit."""
    sweep = []
    for alpha in ALPHA_SWEEP:
        ray = rayleigh_modes(float(alpha), nodes=STABILITY_NODES, keep=1)
        sweep.append({'alpha': float(alpha),
                      'inviscid_growth': float(ray[0].growth) if ray else None,
                      'inviscid_c_imag': float(ray[0].c.imag) if ray else None})
    refined = []
    for nodes in (301, 401, 601):
        ray = rayleigh_modes(0.5, nodes=nodes, keep=1)
        refined.append({'nodes': nodes, 'growth': float(ray[0].growth) if ray else None})
    inviscid_limit = []
    for reynolds in (500.0, 2000.0, 1.0e4, 1.0e5):
        os_mode = orr_sommerfeld_modes(0.5, reynolds, nodes=STABILITY_NODES, keep=1)
        inviscid_limit.append({'reynolds': reynolds,
                               'growth': float(os_mode[0].growth) if os_mode else None})
    peak = max((s for s in sweep if s['inviscid_growth']),
               key=lambda s: s['alpha'] * s['inviscid_growth'])
    payload = {
        'alpha_sweep': sweep,
        'alpha_sweep_argmax': {'alpha': peak['alpha'],
                               'growth': peak['inviscid_growth'],
                               'amplification': peak['alpha'] * peak['inviscid_growth']},
        'rayleigh_refinement_at_alpha_0p5': refined,
        'orr_sommerfeld_reynolds_limit_at_alpha_0p5': inviscid_limit,
        'production_reference': linear_reference(0.5, 1.0),
    }
    _write_json(out / 'stability.json', payload)
    return payload


def study_diffusion(out: Path) -> dict:
    """Pre-instability viscous spreading against the exact heat-kernel solution."""
    solver = build_solver(96, 256, PRODUCTION_ALPHA)
    horizon = 6.0
    dt = solver.stable_time_step()
    history = []
    interval = horizon / 12
    next_sample = 0.0
    while solver.time < horizon - 1e-12:
        solver.step(min(dt, horizon - solver.time))
        if solver.time + 1e-12 >= next_sample:
            profile = solver.mean_profile()
            exact = heat_kernel_profile(solver.yc, solver.time)
            metrics = layer_metrics(profile, solver.yc, solver.dy)
            exact_metrics = layer_metrics(exact, solver.yc, solver.dy)
            history.append({
                'time': solver.time,
                'theta_error': abs(metrics['momentum_thickness']
                                   - exact_metrics['momentum_thickness'])
                              / exact_metrics['momentum_thickness'],
                'delta_omega_error': abs(metrics['vorticity_thickness']
                                         - exact_metrics['vorticity_thickness'])
                                    / exact_metrics['vorticity_thickness'],
                'profile_error': float(np.abs(profile - exact).max() / (2 * HALF_VELOCITY)),
                'momentum_thickness': metrics['momentum_thickness'],
                'momentum_thickness_exact': exact_metrics['momentum_thickness'],
            })
            next_sample = next_sample + interval
    payload = {'grid': {'nx': 96, 'ny': 256, 'dt': dt, 'horizon': horizon},
               'history': history,
               'max_theta_error': max(h['theta_error'] for h in history),
               'max_profile_error': max(h['profile_error'] for h in history),
               'max_delta_omega_error': max(h['delta_omega_error'] for h in history)}
    _write_json(out / 'diffusion.json', payload)
    return payload


def study_linear(out: Path) -> dict:
    """Linear-phase growth: grid ladder, time-step ladder and the alpha sweep."""
    mode = rayleigh_modes(PRODUCTION_ALPHA, nodes=STABILITY_NODES, keep=1)[0]
    window = (2.0, LINEAR_HORIZON)
    ladder = []
    for nx, ny, factor in ((48, 128, 1.0), (96, 256, 1.0), (144, 384, 1.0)):
        solver = build_solver(nx, ny, PRODUCTION_ALPHA)
        solver.seed_stability_mode(A0, mode)
        dt = solver.stable_time_step() * factor
        t0 = time.time()
        history = sample_until(solver, LINEAR_HORIZON, dt, 36)
        wall = time.time() - t0
        t = [h['time'] for h in history]
        a = [h['fundamental_amplitude'] for h in history]
        e = [h['fluctuation_energy'] for h in history]
        rate, residual = growth_rate_fit(t, a, window)
        energy_rate = 0.5 * growth_rate_fit(t, e, window)[0]
        idx = int(np.argmin(np.abs(np.array(t) - 0.5 * sum(window))))
        delta = history[idx]['vorticity_thickness'] / 2
        reference = linear_reference(PRODUCTION_ALPHA, delta)
        frozen = linear_reference(PRODUCTION_ALPHA, THICKNESS)
        ladder.append({
            'grid': [nx, ny], 'dt': dt, 'wall_seconds': wall, 'window': list(window),
            'rate': rate, 'rate_log_residual': residual, 'energy_rate': energy_rate,
            'instantaneous_delta': delta,
            'reference_viscous': reference['viscous_growth'],
            'reference_inviscid': reference['inviscid_growth'],
            'reference_frozen_profile': frozen['viscous_growth'],
            'error_vs_instantaneous': rate / reference['viscous_growth'] - 1,
            'error_vs_frozen': rate / frozen['viscous_growth'] - 1,
            'max_divergence': max(h['max_divergence'] for h in history),
            'history': history,
        })
    dt_ladder = []
    base_dt = build_solver(96, 256, PRODUCTION_ALPHA).stable_time_step()
    for factor in (1.0, 0.75, 0.5):
        solver = build_solver(96, 256, PRODUCTION_ALPHA)
        solver.seed_stability_mode(A0, mode)
        history = sample_until(solver, LINEAR_HORIZON, base_dt * factor, 36)
        rate = growth_rate_fit([h['time'] for h in history],
                               [h['fundamental_amplitude'] for h in history], window)[0]
        dt_ladder.append({'dt': base_dt * factor, 'rate': rate})
    alpha_scan = []
    for alpha in (0.30, 0.40, 0.50, 0.60, 0.70):
        alpha_mode = rayleigh_modes(alpha, nodes=STABILITY_NODES, keep=1)
        if not alpha_mode:
            continue
        solver = build_solver(64, 192, alpha)
        solver.seed_stability_mode(A0, alpha_mode[0])
        history = sample_until(solver, 7.0, solver.stable_time_step(), 28)
        t = [h['time'] for h in history]
        rate = growth_rate_fit(t, [h['fundamental_amplitude'] for h in history],
                               (2.0, 7.0))[0]
        idx = int(np.argmin(np.abs(np.array(t) - 4.5)))
        delta = history[idx]['vorticity_thickness'] / 2
        reference = linear_reference(alpha, delta)
        alpha_scan.append({'alpha': alpha, 'rate': rate,
                           'reference_viscous': reference['viscous_growth'],
                           'reference_inviscid': reference['inviscid_growth'],
                           'error': rate / reference['viscous_growth'] - 1,
                           'amplification': rate})
    payload = {'seed': {'amplitude': A0, 'mode': 'Rayleigh eigenfunction at alpha=0.5'},
               'grid_ladder': ladder, 'time_step_ladder': dt_ladder,
               'alpha_scan': alpha_scan,
               'alpha_scan_argmax': max(alpha_scan, key=lambda item: item['rate'])['alpha']}
    _write_json(out / 'linear.json', payload)
    return payload


def study_nonlinear(out: Path) -> dict:
    """Post-roll-up spreading rate and grid/time-step sensitivity."""
    mode = rayleigh_modes(PRODUCTION_ALPHA, nodes=STABILITY_NODES, keep=1)[0]
    cases = []
    specs = ((64, 192, 1.0, 'coarse'), (96, 256, 1.0, 'production'),
             (64, 192, 0.5, 'coarse-dt-halved'))
    for nx, ny, factor, label in specs:
        solver = build_solver(nx, ny, PRODUCTION_ALPHA)
        solver.seed_stability_mode(A0_NONLINEAR, mode)
        dt = solver.stable_time_step() * factor
        t0 = time.time()
        history = sample_until(solver, NONLINEAR_HORIZON, dt, 30)
        wall = time.time() - t0
        t = np.array([h['time'] for h in history])
        theta = np.array([h['momentum_thickness'] for h in history])
        window = (NONLINEAR_HORIZON * 0.45, NONLINEAR_HORIZON * 0.95)
        mask = (t >= window[0]) & (t <= window[1])
        _, slope = linear_fit(t[mask], theta[mask])
        yy = np.linspace(-8, 8, 1601)
        exact_theta = np.array([layer_metrics(heat_kernel_profile(yy, tt), yy, yy[1] - yy[0])
                                ['momentum_thickness'] for tt in t[mask]])
        _, baseline = linear_fit(t[mask], exact_theta)
        cases.append({
            'label': label, 'grid': [nx, ny], 'dt': dt, 'wall_seconds': wall,
            'window': list(window), 'spreading_rate': float(slope),
            'spreading_rate_normalised': float(slope / (2 * HALF_VELOCITY)),
            'diffusive_baseline': float(baseline),
            'spreading_rate_minus_baseline': float(slope - baseline),
            'final_momentum_thickness': float(theta[-1]),
            'final_vorticity_thickness': float(history[-1]['vorticity_thickness']),
            'max_divergence': max(h['max_divergence'] for h in history),
            'history': history,
            'fields': {'y': solver.yc.tolist(),
                       'vorticity': solver.vorticity().tolist(),
                       'mean_profile': solver.mean_profile().tolist()},
        })
    payload = {'seed_amplitude': A0_NONLINEAR, 'horizon': NONLINEAR_HORIZON,
               'cases': cases}
    _write_json(out / 'nonlinear.json', payload)
    return payload


def study_self_similarity(out: Path) -> dict:
    """Collapse of the mean profile in the late (self-similar) window."""
    mode = rayleigh_modes(PRODUCTION_ALPHA, nodes=STABILITY_NODES, keep=1)[0]
    solver = build_solver(96, 256, PRODUCTION_ALPHA)
    solver.seed_stability_mode(A0_NONLINEAR, mode)
    dt = solver.stable_time_step()
    stored = []
    for fraction in (0.45, 0.70, 0.95):
        target = NONLINEAR_HORIZON * fraction
        while solver.time < target - 1e-12:
            solver.step(min(dt, target - solver.time))
        domega = solver.vorticity_thickness()
        stored.append({'time': solver.time, 'delta_omega': domega,
                       'eta': (solver.yc / domega).tolist(),
                       'normalised_profile': (solver.mean_profile()
                                              / (2 * HALF_VELOCITY)).tolist()})
    eta_grid = np.linspace(-1.0, 1.0, 81)
    curves = [np.interp(eta_grid, s['eta'], s['normalised_profile']) for s in stored]
    spread = float(np.max(np.abs(np.array(curves) - np.array(curves)[-1])))
    payload = {'times': [s['time'] for s in stored],
               'delta_omega': [s['delta_omega'] for s in stored],
               'eta_grid': eta_grid.tolist(),
               'curves': [c.tolist() for c in curves],
               'max_collapse_deviation': spread}
    _write_json(out / 'self_similarity.json', payload)
    return payload


# ---------------------------------------------------------------------- figures
def make_figures(out: Path, stability: dict, diffusion: dict, linear: dict,
                 nonlinear: dict, similarity: dict) -> list[dict]:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figures = []
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0), dpi=160)
    sweep = [s for s in stability['alpha_sweep'] if s['inviscid_growth']]
    axes[0].plot([s['alpha'] for s in sweep], [s['inviscid_growth'] for s in sweep],
                 'k-', lw=1.6, label='Rayleigh (inviscid) eigen-solve')
    axes[0].plot([s['alpha'] for s in linear['alpha_scan']],
                 [s['reference_viscous'] for s in linear['alpha_scan']], 's', ms=6,
                 mfc='none', label='Orr--Sommerfeld (Re=500)')
    axes[0].plot([s['alpha'] for s in linear['alpha_scan']],
                 [s['rate'] for s in linear['alpha_scan']], 'o', ms=4,
                 label='FVM linear phase')
    axes[0].set_xlabel(r'$\alpha\delta$')
    axes[0].set_ylabel(r'$\sigma\delta/U$')
    axes[0].set_title('Temporal growth rate vs wavenumber')
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)
    case = linear['grid_ladder'][1]
    t = [h['time'] for h in case['history']]
    a = [h['fundamental_amplitude'] for h in case['history']]
    axes[1].semilogy(t, a, 'o-', ms=3, label='FVM (96x256)')
    rate = case['rate']
    offset = math.log(a[0]) - rate * t[0]
    axes[1].semilogy(t, [math.exp(offset + rate * ti) for ti in t], 'k--',
                     label=r'fit $\sigma\delta/U=%.4f$' % rate)
    axes[1].axvspan(case['window'][0], case['window'][1], color='0.9', zorder=0)
    axes[1].set_xlabel(r'$tU/\delta$')
    axes[1].set_ylabel(r'$|\hat v|$ (fundamental)')
    axes[1].set_title('Linear phase: exponential growth')
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3, which='both')
    fig.tight_layout()
    path = out / 'fig1_linear_growth.png'
    fig.savefig(path)
    plt.close(fig)
    figures.append({'name': path.name,
                    'caption': 'Left: temporal growth rate vs wavenumber, independent '
                               'Rayleigh/Orr-Sommerfeld reference against the FVM linear '
                               'phase. Right: exponential growth of the streamwise '
                               'fundamental with the least-squares fit.'})

    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=160)
    tt = [h['time'] for h in diffusion['history']]
    ax.plot(tt, [h['momentum_thickness'] for h in diffusion['history']], 'o', ms=4,
            label='FVM momentum thickness')
    ax.plot(tt, [h['momentum_thickness_exact'] for h in diffusion['history']], 'k--',
            label='exact heat-kernel solution')
    ax.set_xlabel(r'$tU/\delta$')
    ax.set_ylabel(r'$\theta/\delta$')
    ax.set_title('Diffusion phase vs exact solution')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = out / 'fig2_diffusion_exact.png'
    fig.savefig(path)
    plt.close(fig)
    figures.append({'name': path.name,
                    'caption': 'Viscous spreading of the undisturbed layer against the exact '
                               'heat-kernel solution of the same initial profile.'})

    fig, ax = plt.subplots(figsize=(6.6, 4.0), dpi=160)
    for case in nonlinear['cases']:
        ax.plot([h['time'] for h in case['history']],
                [h['momentum_thickness'] for h in case['history']], lw=1.2,
                label='%s (%dx%d, dt=%g)' % (case['label'], case['grid'][0],
                                             case['grid'][1], case['dt']))
    ax.set_xlabel(r'$tU/\delta$')
    ax.set_ylabel(r'$\theta/\delta$')
    ax.set_title('Nonlinear spreading of the mixing layer')
    ax.legend(fontsize=6)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = out / 'fig3_nonlinear_spreading.png'
    fig.savefig(path)
    plt.close(fig)
    figures.append({'name': path.name,
                    'caption': 'Momentum-thickness history for the grid/time-step ladder in '
                               'the nonlinear regime.'})

    case = [c for c in nonlinear['cases'] if c['label'] == 'production'][0]
    vort = np.array(case['fields']['vorticity'])
    y = np.array(case['fields']['y'])
    x = np.arange(vort.shape[0]) * (2 * math.pi / PRODUCTION_ALPHA) / vort.shape[0]
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
    levels = np.linspace(-1.6, 1.6, 25)
    cs = ax.contourf(x, y, vort.T, levels=levels, cmap='RdBu_r', extend='both')
    ax.set_xlabel(r'$x/\delta$')
    ax.set_ylabel(r'$y/\delta$')
    ax.set_title(r'Vorticity $\omega_z\delta/U$ at $tU/\delta=%.0f$'
                 % case['history'][-1]['time'])
    fig.colorbar(cs, ax=ax)
    fig.tight_layout()
    path = out / 'fig4_vorticity.png'
    fig.savefig(path)
    plt.close(fig)
    figures.append({'name': path.name,
                    'caption': 'Out-of-plane vorticity of the rolled-up mixing layer at the '
                               'end of the production run.'})

    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=160)
    for curve, tt in zip(similarity['curves'], similarity['times']):
        ax.plot(similarity['eta_grid'], curve, label=r'$tU/\delta=%.0f$' % tt)
    ax.set_xlabel(r'$y/\delta_\omega$')
    ax.set_ylabel(r'$\bar u/\Delta U$')
    ax.set_title('Self-similar mean-profile collapse')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = out / 'fig5_self_similarity.png'
    fig.savefig(path)
    plt.close(fig)
    figures.append({'name': path.name,
                    'caption': 'Collapse of the normalised mean profile against the '
                               'vorticity thickness in the late window.'})
    return figures


# ------------------------------------------------------------------------ table
def write_table(out: Path, stability: dict, diffusion: dict, linear: dict,
                nonlinear: dict, similarity: dict) -> Path:
    prod = stability['production_reference']
    frozen = linear['grid_ladder'][1]
    inviscid_limit = stability['orr_sommerfeld_reynolds_limit_at_alpha_0p5'][-1]['growth']
    rayleigh_ref = stability['rayleigh_refinement_at_alpha_0p5'][0]['growth']
    checks = [
        ('linear growth rate, alpha=0.5, Re=500, instantaneous layer scale',
         'Orr--Sommerfeld eigen-solve (independent discretisation)',
         prod['viscous_growth'], frozen['rate'], frozen['error_vs_instantaneous'], 0.10),
        ('linear growth rate, alpha=0.5, Re=500, frozen initial profile',
         'Orr--Sommerfeld eigen-solve (independent discretisation)',
         frozen['reference_frozen_profile'], frozen['rate'], frozen['error_vs_frozen'], 0.10),
        ('inviscid limit of the Orr--Sommerfeld solve (Re=1e5)',
         'Rayleigh eigen-solve', rayleigh_ref, inviscid_limit,
         abs(inviscid_limit / rayleigh_ref - 1), 0.01),
        ('momentum thickness during the diffusion phase',
         'exact heat-kernel solution of the same initial profile', None, None,
         diffusion['max_theta_error'], 0.05),
        ('mean velocity profile during the diffusion phase',
         'exact heat-kernel solution of the same initial profile', None, None,
         diffusion['max_profile_error'], 0.05),
        ('Rayleigh grid refinement (301 -> 601 nodes)',
         'self-convergence of the eigen-solve',
         rayleigh_ref,
         stability['rayleigh_refinement_at_alpha_0p5'][-1]['growth'],
         abs(stability['rayleigh_refinement_at_alpha_0p5'][-1]['growth'] / rayleigh_ref - 1),
         0.005),
    ]
    rows = ['# Machine-generated comparison table', '',
            'All values are produced by `scripts/verify_mixing_layer.py`; nothing is hand-typed.',
            '',
            '| item | reference | ref | got | rel. error | verdict |',
            '|---|---|---:|---:|---:|---|']
    for label, source, ref, got, err, tol in checks:
        verdict = 'PASS' if err is not None and err <= tol else 'FAIL'
        rows.append('| %s | %s | %s | %s | %.3f%% | %s (tol %.1f%%) |' % (
            label, source,
            '%.5f' % ref if ref is not None else '-',
            '%.5f' % got if got is not None else '-',
            100 * err, verdict, 100 * tol))
    by_label = {c['label']: c['spreading_rate'] for c in nonlinear['cases']}
    err = abs(by_label['production'] / by_label['coarse'] - 1)
    rows.append('| nonlinear spreading rate dtheta/dt, production vs coarse grid | '
                'self-convergence over the grid/time-step ladder | %.5f (coarse) | %.5f '
                '(production) | %.2f%% | %s (tol 10%%) |'
                % (by_label['coarse'], by_label['production'], 100 * err,
                   'PASS' if err <= 0.10 else 'FAIL'))
    rows.append('| self-similar mean-profile collapse | profile at t=22.5 vs t=47.5 | - | - | '
                '%.4f (max deviation) | %s (tol 0.0500) |'
                % (similarity['max_collapse_deviation'],
                   'PASS' if similarity['max_collapse_deviation'] <= 0.05 else 'FAIL'))
    rows.append('')
    prod_case = [c for c in nonlinear['cases'] if c['label'] == 'production'][0]
    rows.append('Nonlinear spreading measured on the growth window '
                'tU/delta in [%.1f, %.1f] (the layer saturates near tU/delta ~ 28 in the '
                'single-wavelength box and then oscillates).' % GROWTH_WINDOW)
    rows.append('')
    rows.append('Production case: 96x256 cells, dt=%.4f (CFL 0.4), Re_delta=500, '
                'alpha*delta=0.5, linear seed %.0e, nonlinear seed %.0e.'
                % (prod_case['dt'], A0, A0_NONLINEAR))
    rows.append('')
    rows.append('External literature anchor (published spreading / entrainment rate): '
                '**WITHHELD** -- the offline work cell cannot reach the reference text '
                '(no internet egress; the platform knowledge engine at :8021 refused '
                'connections), so no literature number is quoted or transcribed here.')
    path = out / 'table.md'
    path.write_text('\n'.join(rows) + '\n')
    return path


# ------------------------------------------------------------------------- main
def run_report(out: Path) -> Path:
    nonlinear = _read_json(out / 'nonlinear.json')
    # Recompute the spreading rate from the stored histories on the growth window
    # (the layer saturates around tU/delta ~ 28 and then oscillates, see the report).
    window = GROWTH_WINDOW
    for case in nonlinear['cases']:
        t = np.array([h['time'] for h in case['history']])
        theta = np.array([h['momentum_thickness'] for h in case['history']])
        mask = (t >= window[0]) & (t <= window[1])
        _, slope = linear_fit(t[mask], theta[mask])
        yy = np.linspace(-8, 8, 1601)
        exact = np.array([layer_metrics(heat_kernel_profile(yy, tt), yy, yy[1] - yy[0])
                          ['momentum_thickness'] for tt in t[mask]])
        _, baseline = linear_fit(t[mask], exact)
        case['spreading_window'] = list(window)
        case['spreading_rate'] = float(slope)
        case['spreading_rate_normalised'] = float(slope / (2 * HALF_VELOCITY))
        case['diffusive_baseline'] = float(baseline)
        case['spreading_rate_minus_baseline'] = float(slope - baseline)
    _write_json(out / 'nonlinear.json', nonlinear)
    stability = _read_json(out / 'stability.json')
    diffusion = _read_json(out / 'diffusion.json')
    linear = _read_json(out / 'linear.json')
    nonlinear = _read_json(out / 'nonlinear.json')
    similarity = _read_json(out / 'self_similarity.json')
    figures = make_figures(out, stability, diffusion, linear, nonlinear, similarity)
    table = write_table(out, stability, diffusion, linear, nonlinear, similarity)
    summary = {
        'case': 'temporal plane mixing layer (TensorFVM)',
        'params': {'half_velocity': HALF_VELOCITY,
                   'velocity_difference': 2 * HALF_VELOCITY,
                   'thickness': THICKNESS, 'viscosity': VISCOSITY,
                   'reynolds_delta': HALF_VELOCITY * THICKNESS / VISCOSITY,
                   'alpha_delta': PRODUCTION_ALPHA,
                   'wavelength_over_thickness': 2 * math.pi / PRODUCTION_ALPHA,
                   'height_over_thickness': 16.0},
        'stability': {'production_reference': stability['production_reference'],
                      'alpha_sweep_argmax': stability['alpha_sweep_argmax'],
                      'rayleigh_refinement': stability['rayleigh_refinement_at_alpha_0p5'],
                      'inviscid_limit': stability['orr_sommerfeld_reynolds_limit_at_alpha_0p5']},
        'diffusion_max_errors': {'theta': diffusion['max_theta_error'],
                                 'profile': diffusion['max_profile_error'],
                                 'delta_omega': diffusion['max_delta_omega_error']},
        'linear_errors': {
            'vs_instantaneous_reference': [c['error_vs_instantaneous']
                                           for c in linear['grid_ladder']],
            'vs_frozen_reference': [c['error_vs_frozen'] for c in linear['grid_ladder']],
            'rate': [c['rate'] for c in linear['grid_ladder']],
            'time_step_ladder': linear['time_step_ladder'],
        },
        'nonlinear_spreading': [
            {k: c[k] for k in ('label', 'grid', 'dt', 'spreading_rate',
                               'spreading_rate_normalised', 'diffusive_baseline',
                               'spreading_rate_minus_baseline', 'max_divergence')}
            for c in nonlinear['cases']],
        'self_similarity_deviation': similarity['max_collapse_deviation'],
        'external_literature_anchor': {
            'status': 'WITHHELD',
            'reason': 'offline work cell: no internet egress, knowledge engine :8021 '
                      'unreachable; no reference text available to quote a published '
                      'spreading rate without transcribing it by hand',
        },
        'figures': figures,
        'table': table.name,
    }
    _write_json(out / 'summary.json', summary)
    manifest = {p.name: _sha256(p) for p in sorted(out.iterdir())
                if p.is_file() and p.name != 'manifest.json'}
    _write_json(out / 'manifest.json', manifest)
    return out / 'summary.json'


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='output directory')
    parser.add_argument('--stage', default='all',
                        help='comma separated stages: %s or all' % ','.join(STAGES))
    args = parser.parse_args(argv)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    stages = list(STAGES) if args.stage == 'all' else [s.strip() for s in
                                                       args.stage.split(',') if s.strip()]
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        raise SystemExit('unknown stage(s): %s' % unknown)
    functions = {'stability': study_stability, 'diffusion': study_diffusion,
                 'linear': study_linear, 'nonlinear': study_nonlinear,
                 'similarity': study_self_similarity}
    started = time.time()
    for stage in stages:
        t0 = time.time()
        if stage == 'report':
            summary_path = run_report(out)
            print('report ->', summary_path)
        else:
            functions[stage](out)
        print('stage %-11s %6.1f s' % (stage, time.time() - t0))
    if 'report' in stages:
        print((out / 'table.md').read_text())
    print('total %.1f s' % (time.time() - started))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())