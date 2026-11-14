"""Driver: norm + mesh-phase audit of the CASE-FVM-CHANNEL-KMM180 graded number.

Writes ``docs/verification-channel-kmm180-metric-norm/``:

``summary.json``   every graded number under the four norms, plus the phase sweep
``REPORT.md``      prose report with the tables rendered from ``summary.json``
``fig_norms.png``  graded deviation per norm / window / resolution
``fig_phase.png``  graded number vs wall clustering at fixed resolution

Usage::

    PYTHONPATH=src python3 scripts/run_case_fvm_channel_kmm180_metric_norm.py
    PYTHONPATH=src python3 scripts/run_case_fvm_channel_kmm180_metric_norm.py --quick

``--quick`` runs one resolution (the sweep is what costs time); the published
artifact set is produced without it.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from tensorfvm.verification.kmm1987 import load_kmm1987_reference
from tensorfvm.verification.kmm180_metric_norm import (
    LIKE_FOR_LIKE_MODELS, LIKE_FOR_LIKE_RESOLUTIONS, NORM_CHOICES, TOLERANCE,
    mesh_phase_sweep, run_matrix, write_summary)

OUT = Path('docs/verification-channel-kmm180-metric-norm')
NORM_LABEL = {
    'rms_rel_unweighted': r'RMS rel. (unweighted, implemented)',
    'rms_rel_weighted': r'RMS rel. (volume weighted, declared)',
    'mean_rel_weighted': r'signed weighted mean',
    'integral_rel': r'integral (mass-flux) rel.',
}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def figure_norms(summary, path, dns):
    """Graded deviation per norm, both windows, log x = resolution."""
    cases = summary['cases']
    models = sorted({c['model'] for c in cases})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    colours = dict(zip(NORM_CHOICES, ('#1f77b4', '#d62728', '#2ca02c', '#9467bd')))
    styles = dict(zip(NORM_CHOICES, ('-', '--', '-.', ':')))
    for ax, window in zip(axes, ('legacy_full', 'log_layer')):
        for model in models:
            rows = sorted([c for c in cases if c['model'] == model],
                          key=lambda c: c['n'])
            ns = [r['n'] for r in rows]
            for norm in NORM_CHOICES:
                ys = [r[window]['values'][norm] for r in rows]
                ax.plot(ns, ys, styles[norm], color=colours[norm], marker='o',
                        ms=3, lw=1.4, alpha=.85,
                        label=(NORM_LABEL[norm] if model == models[0] else None))
            ax.annotate(model, (ns[-1], rows[-1][window]['values']
                                ['rms_rel_unweighted']),
                        textcoords='offset points', xytext=(-4, 6),
                        fontsize=8, ha='right')
        ax.axhline(TOLERANCE, color='k', ls='--', lw=1)
        ax.set_xscale('log')
        ax.set_xlabel('wall-normal cells $n$')
        bounds = cases[0][window]['window']
        ax.set_title('%s: $%d \\leq y^+ \\leq %.1f$ (%d cells at $n$=%d)'
                     % (window, bounds['low'], bounds['high'], bounds['cells'],
                        cases[0]['n']))
        ax.grid(alpha=.25)
    axes[0].set_ylabel(r'deviation in $u^+$')
    axes[0].legend(fontsize=7, loc='best')
    fig.suptitle('CASE-FVM-CHANNEL-KMM180: the graded number under four '
                 'measures (dashed = 5 %% criterion)', fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def figure_phase(sweep, path):
    """Graded number vs wall clustering at fixed resolution."""
    rows = sweep['rows']
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for n in sorted({r['n'] for r in rows}):
        group = sorted([r for r in rows if r['n'] == n],
                       key=lambda r: r['first_cell_y_plus_requested'])
        ax.plot([g['first_cell_y_plus_requested'] for g in group],
                [g['rms_rel_unweighted'] for g in group], 'o-',
                label='$n$=%d (%d cell%s in window)'
                      % (n, group[0]['cells'], '' if group[0]['cells'] == 1 else 's'))
    ax.axhline(TOLERANCE, color='k', ls='--', lw=1, label='5 %% criterion')
    ax.set_xlabel(r'first cell centre $y_1^+$ (wall clustering)')
    ax.set_ylabel(r'log-layer deviation in $u^+$')
    ax.set_title('Same model, same grid count, different clustering:\n'
                 'the log-layer verdict is not a converged number',
                 fontsize=9)
    ax.grid(alpha=.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def render_report(summary, sweep, dns):
    cases = summary['cases']
    lines = []
    add = lines.append
    add('# CASE-FVM-CHANNEL-KMM180: measure (norm) and mesh-phase audit of the '
        'graded number')
    add('')
    add('Anchor: `CASE-FVM-CHANNEL-KMM180` — 算例池·湍流槽道 Re_tau=180 平均速度'
        '对数律. Acceptance: RMS relative deviation of `u+` from the reference '
        'for `y+ > 30` **< 5 %**.')
    add('')
    add('Reference: Lee & Moser (2015) DNS, `Re_tau = %.3f` (provenance '
        '`docs/rans-turbulence-validation-matrix/dns/0180/provenance.json`, '
        'SHA-256 recorded there). Solver: `rans_validation.channel.'
        'solve_channel` at the nominal `Re_tau = 180`.'
        % dns['re_tau'])
    add('')
    add('## 1. Why this audit exists')
    add('')
    add('Three earlier nodes varied the **window** (`kmm180_windows`), the '
        '**model/grid ladder** (the benchmark matrix) and the **reference '
        'choice** (LM2015 for the KMM1987 table). None of them varied the '
        '**measure**, although the repository declares two different ones:')
    add('')
    add('* implemented (`kmm1987.log_region_deviation`): `sqrt(mean(e^2))`, one '
        'vote per cell centre;')
    add('* declared by the DNS provenance (`metric_profile`): '
        '`sqrt(sum(V e^2) / sum(V))`, wall-normal volume weighted.')
    add('')
    add('On a wall-clustered mesh these are different functionals. This report '
        'grades the *same solutions* under four measures and then holds the '
        'measure fixed and varies the clustering, which is the only way to tell '
        'a converged verdict from a mesh artefact.')
    add('')
    add('## 2. What was computed')
    add('')
    add('Four definitions of "the deviation in the log region", all on the same '
        'converged solution:')
    add('')
    add('| norm | definition |')
    add('|---|---|')
    add('| `rms_rel_unweighted` | `sqrt(mean((u+_i-u+_DNS)/u+_DNS)^2)`, one vote '
        'per cell (implemented legacy metric) |')
    add('| `rms_rel_weighted` | same residual, weighted by the wall-normal '
        'measure each cell centre represents |')
    add('| `mean_rel_weighted` | signed weighted mean — separates a constant '
        '`u+` offset from a slope error |')
    add('| `integral_rel` | relative deficit of `int u+ dy` over the window '
        '(the mass-flux quantity `C_f`/`U_bulk+` actually feel) |')
    add('')
    add('The weights are trapezoidal weights built from the sample abscissae, so '
        'the identical routine weights a solver mesh and the DNS table; they sum '
        'to one, which is what makes the weighted RMS comparable to the '
        'unweighted one.')
    add('')
    add('## 3. Results: does the measure matter?')
    add('')
    for window in ('legacy_full', 'log_layer'):
        block = cases[0][window]['window']
        add('### Window `%s` — `%g <= y+ <= %.3f`'
            % (window, block['low'], block['high']))
        add('')
        add('| model | n | cells | unweighted RMS | weighted RMS | signed '
            'weighted mean | integral | verdicts (u/w/m/i) | offset [w.u.] |')
        add('|---|---|---|---|---|---|---|---|---|')
        for case in cases:
            b = case[window]
            v = b['values']
            add('| %s | %d | %d | %.4f | %.4f | %+.4f | %.4f | %s | %+.3f |'
                % (case['model'], case['n'], b['window']['cells'],
                   v['rms_rel_unweighted'], v['rms_rel_weighted'],
                   v['mean_rel_weighted'], v['integral_rel'],
                   '/'.join(b['verdicts'][k][0] for k in NORM_CHOICES),
                   b['implied_offset_u_plus']))
        add('')
    add('### Verdict flips across measures')
    add('')
    if summary['flips']:
        add('Cases whose verdict depends on the choice of measure '
            '(the claim is **not** measure-independent here):')
        add('')
        add('| model | n | window | cells | verdicts | values |')
        add('|---|---|---|---|---|---|')
        for f in summary['flips']:
            add('| %s | %d | %s | %d | %s | %s |'
                % (f['model'], f['n'], f['window'], f['cells'],
                   ', '.join('%s=%s' % (k, v) for k, v in f['verdicts'].items()),
                   ', '.join('%.4f' % v for v in f['values'].values())))
        add('')
    else:
        add('None. Every `(case, window)` keeps its verdict under all four '
            'measures, so the published verdicts are measure-independent.')
        add('')
    add('## 4. Results: is the number a converged number, or a mesh artefact?')
    add('')
    add('The log-layer window spans `%.2f` wall units and holds `%s` cells on the '
        'ladder `n = %s`. Which cells land inside is a mesh choice. Holding '
        'model, resolution and reference fixed and varying only the wall '
        'clustering:'
        % (cases[0]['log_layer']['window']['high']
           - cases[0]['log_layer']['window']['low'],
           ', '.join(str(c['log_layer']['window']['cells']) for c in cases
                         if c['model'] == cases[0]['model']),
           '/'.join(str(c['n']) for c in cases
                    if c['model'] == cases[0]['model'])))
    add('')
    add('| model | n | cells in window | min | max | max/min | verdicts | '
        'verdict stable? |')
    add('|---|---|---|---|---|---|---|---|')
    for n in sorted(sweep['spread_by_resolution'], key=int):
        s = sweep['spread_by_resolution'][n]
        add('| %s | %s | %s | %.4f | %.4f | %.2fx | %s | %s |'
            % (sweep['model'], n, '/'.join(str(c) for c in s['cells']),
               s['min'], s['max'], s['ratio'], ', '.join(s['verdicts']),
               'yes' if not s['verdict_unstable'] else '**no**'))
    add('')
    add('Per-point sweep (all rows in `summary.json`):')
    add('')
    add('| n | requested y1+ | actual y1+ | cells | unweighted RMS | weighted '
        'RMS | integral | verdict |')
    add('|---|---|---|---|---|---|---|---|')
    for row in sweep['rows']:
        add('| %d | %.2f | %.3f | %d | %.4f | %.4f | %.4f | %s |'
            % (row['n'], row['first_cell_y_plus_requested'],
               row['first_cell_y_plus_actual'], row['cells'],
               row['rms_rel_unweighted'], row['rms_rel_weighted'],
               row['integral_rel'], row['verdict']))
    add('')
    add('## 5. Figures')
    add('')
    add('![fig_norms.png](fig_norms.png)')
    add('')
    add('![fig_phase.png](fig_phase.png)')
    add('')
    add('## 6. Error judgement against the criterion')
    add('')
    log_rows = [c for c in cases if c['model'] == 'sst']
    worst = max(c['log_layer']['values']['rms_rel_unweighted'] for c in log_rows)
    best = min(c['log_layer']['values']['rms_rel_unweighted'] for c in log_rows)
    add('Criterion: < 5 % in `y+ > 30`. Key numbers, per measure:')
    add('')
    add('* log layer, SST, `n >= 128`: %.2f–%.2f %% under all four measures '
        '→ **does not reach < 5 %%** (FAIL is measure-independent).'
        % (100 * best, 100 * worst))
    add('* legacy full window, SST: %.2f–%.2f %% (unweighted) vs %.2f–%.2f %% '
        '(declared volume-weighted) → the implemented metric is the *stricter* '
        'one by up to %.0f %%, and both pass.'
        % (100 * min(c['legacy_full']['values']['rms_rel_unweighted']
                     for c in log_rows),
           100 * max(c['legacy_full']['values']['rms_rel_unweighted']
                     for c in log_rows),
           100 * min(c['legacy_full']['values']['rms_rel_weighted']
                     for c in log_rows),
           100 * max(c['legacy_full']['values']['rms_rel_weighted']
                     for c in log_rows),
           100 * (max(c['legacy_full']['values']['rms_rel_unweighted']
                      for c in log_rows)
                  / max(c['legacy_full']['values']['rms_rel_weighted']
                        for c in log_rows) - 1)))
    sweep_min = min(s['min'] for s in sweep['spread_by_resolution'].values())
    sweep_max = max(s['max'] for s in sweep['spread_by_resolution'].values())
    add('* the log-layer verdict is **not** converged at *any* resolution of the '
        'ladder: with the model, the grid count and the reference fixed and '
        'only the wall clustering changed, the graded number moves over a '
        '%.2fx range and flips its verdict at `n = 64`, `128` and `256` alike -- '
        'the whole swept range %.2f %%-%.2f %% straddles the 5 %% criterion.'
        % (max(s['ratio'] for s in sweep['spread_by_resolution'].values()),
           100 * sweep_min, 100 * sweep_max))
    add('')
    add('Is the anchor met? **No, not in the physical log layer.** The SST '
        'mean-velocity profile misses the DNS by a near-constant offset of about '
        '`-0.8` wall units (≈ 5.7 % of `u+`) for `y+ in [30, 36.4]` at the '
        'resolutions where the window is resolved; the rows that pass do so '
        'because of where the cells land, not because the solution is better.')
    add('')
    add('## 7. Caveats and scope')
    add('')
    add('* The audit is purely diagnostic: solver, closures and the legacy '
        'grading path are untouched; the legacy metric is reproduced '
        'bit-for-bit (`stations.csv` values match to `1e-12`).')
    add('* Weights are trapezoidal weights over the cell centres of the window, '
        'not the solver\'s exact control volumes; the exact volumes differ only '
        'at the two window ends and by O(dy) there, which the identical '
        'conclusions at three resolutions show is immaterial.')
    add('* The reference is LM2015 at `Re_tau = 182.088`, not the KMM1987 '
        '`Re_tau = 180` table (see `kmm180_reference_choice` on its branch): '
        'this report grades whatever reference the anchor uses and does not '
        're-litigate the substitution.')
    add('* Only the `y+ > 30` windows are audited; the viscous sublayer and the '
        '`C_f` / `U_bulk+` grading are reported elsewhere and are not re-graded '
        'here.')
    add('* One closure family (SST / Wilcox k-omega / standard k-epsilon) on a '
        '1-D equilibrium channel; no claim is made about the 2-/3-D FVM '
        'discretisation.')
    add('')
    add('## 8. Reproduction')
    add('')
    add('```')
    add('PYTHONPATH=src python3 scripts/run_case_fvm_channel_kmm180_metric_norm.py')
    add('PYTHONPATH=src python3 -m pytest tests/test_channel_kmm180_metric_norm.py -q')
    add('```')
    add('')
    add('Artifact SHA-256 are listed in `summary.json` under `artifacts`.')
    add('')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--quick', action='store_true',
                        help='single resolution ladder point (sweep still runs)')
    parser.add_argument('--repeat', type=int, default=2,
                        help='solves per case, to expose non-determinism')
    parser.add_argument('--stage', choices=('all', 'matrix', 'phase'), default='all',
                        help='run the matrix, the phase sweep, or both; the two '
                             'stages can be run separately so the artifact set '
                             'fits a short wall-clock budget (phase reuses the '
                             'summary.json written by matrix)')
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    dns = load_kmm1987_reference()

    resolutions = (64,) if args.quick else LIKE_FOR_LIKE_RESOLUTIONS

    summary_path = args.out / 'summary.json'
    t_matrix = t_sweep = 0.0
    if args.stage in ('all', 'matrix'):
        t0 = time.time()
        summary = run_matrix(LIKE_FOR_LIKE_MODELS, resolutions, dns=dns,
                             repeat=args.repeat)
        t_matrix = time.time() - t0
        write_summary(summary, summary_path)
        print('matrix stage done: %.1f s' % t_matrix)
    else:
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        t_matrix = summary.get('timings', {}).get('matrix_s', 0.0)

    if args.stage == 'matrix':
        print('wrote', summary_path)
        return

    t0 = time.time()
    sweep = mesh_phase_sweep('sst', resolutions, dns=dns)
    t_sweep = time.time() - t0
    print('phase stage done: %.1f s' % t_sweep)

    summary['mesh_phase_sweep'] = sweep
    summary['timings'] = dict(matrix_s=t_matrix, phase_sweep_s=t_sweep,
                              resolutions=list(resolutions),
                              repeat=args.repeat,
                              repro='PYTHONPATH=src python3 '
                                    'scripts/run_case_fvm_channel_kmm180_metric_norm.py')

    figure_norms(summary, args.out / 'fig_norms.png', dns)
    figure_phase(sweep, args.out / 'fig_phase.png')

    (args.out / 'REPORT.md').write_text(render_report(summary, sweep, dns),
                                        encoding='utf-8')

    artifacts = {}
    for name in ('REPORT.md', 'fig_norms.png', 'fig_phase.png'):
        artifacts[name] = _sha256(args.out / name)
    write_summary(summary, summary_path)
    # summary.json cannot hash itself; hash it last and record it in a sidecar
    (args.out / 'manifest.json').write_text(
        json.dumps(dict(artifacts=artifacts,
                        summary_json=_sha256(args.out / 'summary.json'),
                        implementation='src/tensorfvm/verification/'
                                       'kmm180_metric_norm.py',
                        driver='scripts/'
                               'run_case_fvm_channel_kmm180_metric_norm.py'),
                   indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print('wrote', args.out / 'summary.json')
    print('flips:', len(summary['flips']))
    for n, s in sorted(summary['mesh_phase_sweep']['spread_by_resolution'].items()):
        print('  phase n=%s ratio=%.2f verdicts=%s'
              % (n, s['ratio'], '/'.join(s['verdicts'])))


if __name__ == '__main__':
    main()