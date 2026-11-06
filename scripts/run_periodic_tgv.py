"""Physical-accuracy study of the periodic MAC kernel on the exact Taylor-Green vortex.

Writes ``study.json`` (numbers only, no hand transcribed values), per-case gzip
artifacts and the report figures into one output directory.
"""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

from tensorfvm.verification import periodic_tgv

SOURCES = ('src/tensorfvm/periodic_mac.py', 'src/tensorfvm/runtime.py',
           'src/tensorfvm/verification/periodic_tgv.py', 'scripts/run_periodic_tgv.py')


def jsonable(value):
    if hasattr(value, 'tolist') or hasattr(value, 'item'):
        return jsonable(value.tolist() if hasattr(value, 'tolist') else value.item())
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_gzip(path, value):
    path.write_bytes(gzip.compress(
        json.dumps(jsonable(value), separators=(',', ':'), allow_nan=False).encode(), mtime=0))


def render_figures(study, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    written = []

    # 1. field comparison at the finest grid: vorticity and pressure
    (name, fields), = study['fields'].items()
    x = np.asarray(fields['centres_x'])[0, 0, :]
    y = np.asarray(fields['centres_y'])[0, :, 0]
    for key, exact_key, label, tag in (
            ('vorticity', 'vorticity_exact', r'$\omega_z$', 'vorticity'),
            ('pressure', 'pressure_exact', 'p', 'pressure')):
        numeric = np.asarray(fields[f'{key}_numeric'])[0]
        exact = np.asarray(fields[exact_key])[0]
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
        vmin, vmax = float(exact.min()), float(exact.max())
        for ax, data, title in ((axes[0], exact, f'exact {label}'),
                                (axes[1], numeric, f'FVM {label}')):
            image = ax.pcolormesh(x, y, data, cmap='RdBu_r', vmin=vmin, vmax=vmax, shading='auto')
            fig.colorbar(image, ax=ax, fraction=0.046)
            ax.set_title(f'{title} ({name})')
            ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_aspect('equal')
        err = numeric - exact
        limit = float(np.abs(err).max()) or 1.0
        image = axes[2].pcolormesh(x, y, err, cmap='PuOr', vmin=-limit, vmax=limit, shading='auto')
        fig.colorbar(image, ax=axes[2], fraction=0.046)
        axes[2].set_title('FVM - exact  (rel. $L_2$ = %.2e)'
                          % study['refinement'][-1][f'{key}_error'])
        axes[2].set_xlabel('x'); axes[2].set_ylabel('y'); axes[2].set_aspect('equal')
        fig.tight_layout()
        path = output / f'tgv-{tag}-fields.png'
        fig.savefig(path, dpi=140); plt.close(fig)
        written.append(path.name)

    # 2. refinement: observed order
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    refinement = study['refinement']
    n = np.asarray([c['n'] for c in refinement], dtype=float)
    for key, marker, label in (('velocity_error', 'o', 'velocity'),
                               ('pressure_error', 's', 'pressure'),
                               ('vorticity_error', '^', 'vorticity $\\omega_z$'),
                               ('dissipation_error', 'd', 'dissipation rate')):
        values = np.asarray([c[key] for c in refinement], dtype=float)
        ax.loglog(n, values, marker + '-', label=label)
    reference = refinement[-1]
    scale = np.asarray([reference['velocity_error'] * (reference['n'] / v) ** 2 for v in n])
    ax.loglog(n, scale, 'k--', lw=1, label=r'$O(h^2)$ reference')
    ax.set_xlabel('grid points per direction n')
    ax.set_ylabel('relative $L_2$ error')
    ax.set_title('Taylor-Green refinement (dt $\\propto$ h)')
    ax.grid(True, which='both', alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout()
    path = output / 'tgv-refinement.png'
    fig.savefig(path, dpi=140); plt.close(fig)
    written.append(path.name)

    # 3. energy and dissipation traces of the finest grid
    trace = study['traces'][refinement[-1]['name']]
    t = np.asarray([r['time_s'] for r in trace])
    energy = np.asarray([r['energy_numeric_J'] for r in trace])
    energy_exact = np.asarray([r['energy_exact_discrete_J'] for r in trace])
    eps = np.asarray([r['dissipation_numeric_W'] for r in trace])
    eps_exact = np.asarray([r['dissipation_exact_W'] for r in trace])
    visc = np.asarray([-r['viscous_power_numeric_W'] for r in trace])
    conv = np.asarray([abs(r['convective_power_numeric_W']) for r in trace])
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    axes[0].plot(t, energy_exact, 'k-', lw=2, label='exact $E(t)=\\rho V/4\\,e^{-4\\nu t}$')
    axes[0].plot(t, energy, 'r--', label='FVM')
    axes[0].set_xlabel('t [s]'); axes[0].set_ylabel('kinetic energy [J]')
    axes[0].legend(fontsize=8); axes[0].set_title('energy decay')
    axes[1].plot(t, eps_exact, 'k-', lw=1, label='exact $\\epsilon=4\\nu E$')
    axes[1].plot(t, eps, 'r--', label='FVM $-\\Delta E/\\Delta t$')
    axes[1].plot(t, visc, 'b:', label='FVM viscous power')
    axes[1].set_xlabel('t [s]'); axes[1].set_ylabel('dissipation rate [W]')
    axes[1].legend(fontsize=8); axes[1].set_title('dissipation closure')
    axes[2].semilogy(t, np.maximum(conv, 1e-18), 'b-')
    axes[2].axhline(refinement[-1]['power_scale_W'], color='k', ls='--', lw=1,
                    label='scale $4\\nu E_0$ = %.3g W' % refinement[-1]['power_scale_W'])
    axes[2].set_xlabel('t [s]'); axes[2].set_ylabel('$|P_{conv}|$ [W]')
    axes[2].legend(fontsize=8); axes[2].set_title('convective work (must vanish)')
    fig.tight_layout()
    path = output / 'tgv-budgets.png'
    fig.savefig(path, dpi=140); plt.close(fig)
    written.append(path.name)

    # 4. honest limit: error vs Reynolds number at fixed n
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    labels, values = [], []
    for case in refinement:
        labels.append('$\\nu$=%.3g\nn=%d' % (study['viscosity_pa_s'], case['n']))
        values.append(case['velocity_error'])
    limit = study['limit_case']
    labels.append('$\\nu$=%.3g\nn=%d' % (limit['viscosity_pa_s'], limit['n']))
    values.append(limit['velocity_error'] if limit['velocity_error'] is not None else float('inf'))
    ax.bar(range(len(values)), values, color=['C0'] * len(refinement) + ['C3'])
    ax.axhline(study['tolerance'], color='k', ls='--', lw=1, label='3 % acceptance gate')
    ax.set_yscale('log'); ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel('relative $L_2$ velocity error')
    ax.set_title('Taylor-Green: qualified regime and its edge')
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = output / 'tgv-limit.png'
    fig.savefig(path, dpi=140); plt.close(fig)
    written.append(path.name)
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--grids', type=int, nargs='+', default=[16, 32, 64])
    parser.add_argument('--viscosity', type=float, default=.05)
    parser.add_argument('--time-end', type=float, default=1.0)
    parser.add_argument('--limit-viscosity', type=float, default=.002)
    parser.add_argument('--no-figures', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    study = periodic_tgv.run_study(tuple(args.grids), viscosity_pa_s=args.viscosity,
                                   time_end_s=args.time_end,
                                   limit_viscosity_pa_s=args.limit_viscosity)
    for name, trace in study['traces'].items():
        write_gzip(args.output / f'trace-{name}.json.gz', trace)
    root = Path(__file__).resolve().parents[1]
    study['source_sha256'] = {p: hashlib.sha256((root / p).read_bytes()).hexdigest()
                              for p in SOURCES}
    if not args.no_figures:
        study['figures'] = render_figures(study, args.output)
    (args.output / 'study.json').write_text(json.dumps(jsonable(study), indent=2) + '\n')
    print(json.dumps({k: v for k, v in study.items()
                      if k not in ('traces', 'fields')}, indent=2), flush=True)


if __name__ == '__main__':
    main()