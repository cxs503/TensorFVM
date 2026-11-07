"""Run the RANS turbulence-model validation matrix and write its artifacts.

    python3 scripts/run_rans_validation_matrix.py [--output DIR] [--quick]

Artifacts: ``matrix.json`` (machine-readable, every graded number), ``table.md``
(generated tables) and three figures.  ``--quick`` shrinks the refinement ladder
and the sensitivity mesh; the shipped artifacts use the full settings.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from tensorfvm.rans_validation.matrix import (CRITERIA, DNS_REFERENCE, PRIMARY_FRACTION,  # noqa: E402
                                             PRIMARY_REFERENCE, available_dns, match_dns, run_matrix,
                                             summarise, write_report)
from tensorfvm.rans_validation.reference import log_law, reichardt  # noqa: E402

DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / 'docs' / 'rans-turbulence-validation-matrix'


def plot_re_tau(report):
    """Two representative Reynolds numbers: the lowest and highest in the matrix."""
    values = sorted({case['re_tau'] for case in report['cases']})
    return values[0], values[-1]


def figure_profiles(report, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    colours = {'sst': 'C0', 'komega': 'C1', 'kepsilon': 'C2'}
    profiles = available_dns()
    for re_tau, axis in zip(plot_re_tau(report), axes):
        grid = np.logspace(np.log10(1.), np.log10(re_tau), 400)
        axis.semilogx(grid, reichardt(grid), 'k-', lw=2, label='Reichardt (1951)')
        axis.semilogx(grid, log_law(grid), 'k--', lw=1.2, label=r'log law $\kappa=0.41$, $B=5.2$')
        dns = match_dns(re_tau, profiles)
        if dns is not None:
            axis.semilogx(dns['y_plus'][1:], dns['u_plus'][1:], 'ko', ms=3, mfc='none',
                          label='DNS Lee & Moser (%s)' % dns['case'])
        for case in report['cases']:
            if case['re_tau'] != re_tau:
                continue
            profile = case['profiles']
            axis.semilogx(np.maximum(profile['y_plus'], 1e-3), profile['u_plus'], color=colours[case['model']],
                          lw=1.4, label='%s (%s)' % (case['model'], case['wall_treatment'].replace('_', '-')))
        axis.set_xlim(1, re_tau)
        axis.set_xlabel(r'$y^+$')
        axis.set_ylabel(r'$U^+$')
        axis.set_title(r'$\mathrm{Re}_\tau = %d$' % int(re_tau))
        axis.grid(alpha=.3)
        axis.legend(fontsize=7, loc='upper left')
    figure.suptitle('Equilibrium half-channel mean velocity: models vs wall laws')
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)
    return path


def figure_eddy_viscosity(report, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    colours = {'sst': 'C0', 'komega': 'C1', 'kepsilon': 'C2'}
    for re_tau, axis in zip(plot_re_tau(report), axes):
        grid = np.logspace(np.log10(1.), np.log10(re_tau), 400)
        axis.loglog(grid, .41 * grid, 'k--', lw=1.2, label=r'$\kappa u_\tau y/\nu$ ($\kappa=0.41$)')
        for case in report['cases']:
            if case['re_tau'] != re_tau:
                continue
            profile = case['profiles']
            axis.loglog(np.maximum(profile['y_plus'], 1e-3), np.maximum(profile['eddy_viscosity_ratio'], 1e-8),
                        color=colours[case['model']], lw=1.4, label=case['model'])
        axis.set_xlim(1, re_tau)
        axis.set_xlabel(r'$y^+$')
        axis.set_ylabel(r'$\nu_t/\nu$')
        axis.set_title(r'$\mathrm{Re}_\tau = %d$' % int(re_tau))
        axis.grid(alpha=.3, which='both')
        axis.legend(fontsize=8, loc='lower right')
    figure.suptitle('Eddy-viscosity ratio: the log layer expects a straight line of slope one')
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)
    return path


def figure_deviation(report, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels, primary, alternative, dns_bars = [], [], [], []
    for case in report['cases']:
        labels.append('%s\nRe=%.0f' % (case['model'], case['re_tau']))
        primary.append(100 * case['deviations'][PRIMARY_REFERENCE]['%.2f' % PRIMARY_FRACTION]['rms_relative'])
        alternative.append(100 * case['deviations']['reichardt']['%.2f' % PRIMARY_FRACTION]['rms_relative'])
        row = case['deviations'].get(DNS_REFERENCE, {}).get('%.2f' % PRIMARY_FRACTION)
        dns_bars.append(np.nan if row is None else 100 * row['rms_relative'])
    position = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(10, 4.2))
    axis.bar(position - .26, primary, .25, label='vs log law')
    axis.bar(position, alternative, .25, label='vs Reichardt composite')
    axis.bar(position + .26, dns_bars, .25, label='vs DNS (Lee & Moser)')
    axis.axhline(100 * CRITERIA['log_region_deviation'], color='crimson', ls='--', lw=1.5,
                 label='%.0f%% criterion' % (100 * CRITERIA['log_region_deviation']))
    axis.set_xticks(position)
    axis.set_xticklabels(labels, fontsize=8)
    axis.set_ylabel('log-region RMS deviation [%]')
    axis.set_title('Log-region deviation, window $30 \\leq y^+ \\leq %.1f\\,\\mathrm{Re}_\\tau$ (grey bar = no DNS at that $Re_\\tau$)'
                   % PRIMARY_FRACTION)
    axis.grid(alpha=.3, axis='y')
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)
    return path


def figure_sensitivity(report, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    for sweep in report['sensitivity']:
        axis.plot([100 * row['change'] for row in sweep['rows']],
                  [100 * row['Cf_relative_change'] for row in sweep['rows']],
                  marker='o', label='%s: %s' % (sweep['model'], sweep['constant']))
    axis.axhline(15, color='crimson', ls='--', lw=1.4, label='+/-15% criterion')
    axis.axhline(-15, color='crimson', ls='--', lw=1.4)
    axis.set_xlabel('constant perturbation [%]')
    axis.set_ylabel(r'$\Delta C_f / C_f$ [%]')
    axis.set_title('Skin-friction sensitivity to the $C_\\mu$-equivalent constant')
    axis.grid(alpha=.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)
    return path


def figure_reynolds_stress(report, path):
    """Modelled -u'v'/u_tau^2 and k/u_tau^2 against the DNS momentum-balance value."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from tensorfvm.rans_validation.reference import dns_reynolds_shear
    profiles = available_dns()
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    colours = {'sst': 'C0', 'komega': 'C1', 'kepsilon': 'C2'}
    for re_tau, axis in zip(plot_re_tau(report), axes):
        dns = match_dns(re_tau, profiles)
        if dns is not None:
            axis.semilogx(dns['y_plus'][1:], dns_reynolds_shear(dns)[1:], 'k-', lw=2,
                          label="DNS (from mean momentum balance)")
        for case in report['cases']:
            if case['re_tau'] != re_tau:
                continue
            axis.semilogx(np.maximum(case['profiles']['y_plus'], 1e-3), case['profiles']['reynolds_shear_stress'],
                          color=colours[case['model']], lw=1.4, label=case['model'])
        axis.axhline(1 / np.sqrt(.09), color='grey', ls=':', lw=1.2,
                     label=r'equilibrium $k^+ = 1/\sqrt{\beta^*}$')
        axis.set_xlim(1, re_tau)
        axis.set_xlabel(r'$y^+$')
        axis.set_ylabel(r"$-\overline{u'v'}/u_\tau^2$")
        axis.set_title(r'$\mathrm{Re}_\tau = %.1f$' % re_tau)
        axis.grid(alpha=.3)
        axis.legend(fontsize=7)
    figure.suptitle('Turbulent shear stress: fixed by the momentum balance, so the models agree by construction')
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)
    return path


def figure_turbulence_energy(report, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    colours = {'sst': 'C0', 'komega': 'C1', 'kepsilon': 'C2'}
    for case in report['cases']:
        profile = case['profiles']
        axis.semilogx(np.maximum(profile['y_plus'], 1e-3), profile['k_over_u_tau_squared'],
                      color=colours[case['model']], ls='-' if case['re_tau'] < 300 else '--',
                      lw=1.3, label='%s Re=%.0f' % (case['model'], case['re_tau']))
    axis.axhline(1 / np.sqrt(.09), color='grey', ls=':', lw=1.2, label=r'$1/\sqrt{\beta^*}=3.33$')
    axis.set_xlabel(r'$y^+$')
    axis.set_ylabel(r'$k/u_\tau^2$')
    axis.set_title('Modelled turbulence kinetic energy (log-layer equilibrium value dashed)')
    axis.grid(alpha=.3)
    axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)
    return path


def table_markdown(report):
    lines = ['# Generated tables', '',
             'Generated by `scripts/run_rans_validation_matrix.py`. Do not edit by hand.', '',
             '## 1. Graded matrix (window $30 \\leq y^+ \\leq %.1f\\,\\mathrm{Re}_\\tau$)' % PRIMARY_FRACTION, '',
             '| model | Re_tau | cells | Cf | Cf vs DNS | U_b+ | kappa implied | kappa local | RMS vs DNS | RMS vs log law | RMS vs Reichardt | max vs Reichardt |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for case in report['cases']:
        primary = case['deviations'][PRIMARY_REFERENCE]['%.2f' % PRIMARY_FRACTION]
        alternative = case['deviations']['reichardt']['%.2f' % PRIMARY_FRACTION]
        dns_row = case['deviations'].get(DNS_REFERENCE, {}).get('%.2f' % PRIMARY_FRACTION)
        local = case['local_kappa']['kappa_local'] if case['local_kappa'] else float('nan')
        cf = case['Cf_deviation_vs_dns']
        lines.append('| %s | %.3f | %d | %.5f | %s | %.3f | %.4f | %.4f | %s | %.2f%% | %.2f%% | %.2f%% |' % (
            case['model'], case['re_tau'], case['cells'], case['Cf'],
            '—' if cf is None else '%+.2f%%' % (100 * cf), case['bulk_u_plus'],
            case['implied_kappa'], local,
            '—' if dns_row is None else '%.2f%%' % (100 * dns_row['rms_relative']),
            100 * primary['rms_relative'], 100 * alternative['rms_relative'], 100 * alternative['max_relative']))
    lines += ['', '## 2. Grid-refinement ladder', '',
              '| model | Re_tau | cells | Cf | RMS vs log law | Cf change vs previous | converged |',
              '|---|---:|---:|---:|---:|---:|---|']
    for ladder in report['grid']:
        rows = ladder['rows']
        for index, row in enumerate(rows):
            step = '—' if index == 0 else '%.3f%%' % (100 * abs(rows[index]['Cf'] / rows[index - 1]['Cf'] - 1))
            lines.append('| %s | %d | %d | %.6f | %.2f%% | %s | %s |' % (
                ladder['model'], int(ladder['re_tau']), row['cells'], row['Cf'],
                100 * row['log_region_rms_relative'], step, row['converged']))
    lines += ['', '## 3. Constant sensitivity ($C_\\mu$-equivalent swept by up to +/-20%)', '',
              '| model | constant | base value | base Cf | max abs(dCf/Cf) for +/-10% | dlnCf/dlnC | verdict |',
              '|---|---|---:|---:|---:|---:|---|']
    for sweep in report['sensitivity']:
        lines.append('| %s | %s | %.4f | %.5f | %.2f%% | %.2f | %s |' % (
            sweep['model'], sweep['constant'], sweep['base_value'], sweep['base_Cf'],
            100 * sweep['max_abs_change_within_ten_percent'], sweep['sensitivity_dCfdC_over_Cf'],
            'PASS' if sweep['max_abs_change_within_ten_percent'] < CRITERIA['constant_sensitivity'] else 'FAIL'))
    lines += ['', '## 4. Log-window sensitivity of the graded deviation', '',
              '| model | Re_tau | reference | window 0.15 Re_tau | window 0.20 Re_tau | window 0.30 Re_tau |',
              '|---|---:|---|---:|---:|---:|']
    for case in report['cases']:
        for name in ('dns', 'log_law', 'reichardt'):
            if name not in case['deviations']:
                continue
            cells = []
            for fraction in ('0.15', '0.20', '0.30'):
                row = case['deviations'][name][fraction]
                cells.append('—' if row is None else '%.2f%% (%d cells)' % (100 * row['rms_relative'], row['cells']))
            lines.append('| %s | %d | %s | %s | %s | %s |' % (case['model'], int(case['re_tau']), name, *cells))
    lines += ['', '## 5. Verdicts', '', '| criterion | model | quantity | value | threshold | verdict |', '|---|---|---|---:|---:|---|']
    for verdict in report['verdicts']:
        if verdict['criterion'] == 'log_region_deviation':
            primary = verdict['primary']
            lines.append('| log-region deviation (primary) | %s (Re_tau=%.0f) | RMS vs %s | %.2f%% | %.0f%% | %s |' % (
                verdict['model'], verdict['re_tau'], primary['reference'], 100 * primary['rms_relative'],
                100 * primary['threshold'], 'PASS' if primary['passed'] else 'FAIL'))
            for alternative in verdict['alternatives'].values():
                lines.append('| log-region deviation (alternative) | %s (Re_tau=%.0f) | RMS vs %s | %.2f%% | %.0f%% | %s |' % (
                    verdict['model'], verdict['re_tau'], alternative['reference'],
                    100 * alternative['rms_relative'], 100 * CRITERIA['log_region_deviation'],
                    'PASS' if alternative['passed'] else 'FAIL'))
        else:
            lines.append('| constant sensitivity | %s | max abs(dCf/Cf), %s +/-10%% | %.2f%% | %.0f%% | %s |' % (
                verdict['model'], verdict['constant'], 100 * verdict['max_abs_Cf_change_within_ten_percent'],
                100 * verdict['threshold'], 'PASS' if verdict['passed'] else 'FAIL'))
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--quick', action='store_true', help='smaller ladders and sensitivity mesh')
    arguments = parser.parse_args(argv)
    output = arguments.output
    output.mkdir(parents=True, exist_ok=True)
    if arguments.quick:
        report = run_matrix(levels={'low_re': (32, 64), 'wall_function': (16, 32)},
                            cells={'low_re': 64, 'wall_function': 32}, sensitivity_cells=64)
    else:
        report = run_matrix()
    write_report(report, output / 'matrix.json')
    (output / 'table.md').write_text(table_markdown(report), encoding='utf-8')
    figure_profiles(report, output / 'fig1_velocity_profiles.png')
    figure_eddy_viscosity(report, output / 'fig2_eddy_viscosity.png')
    figure_deviation(report, output / 'fig3_log_region_deviation.png')
    figure_sensitivity(report, output / 'fig4_constant_sensitivity.png')
    figure_reynolds_stress(report, output / 'fig5_reynolds_shear_stress.png')
    figure_turbulence_energy(report, output / 'fig6_turbulence_energy.png')
    print(summarise(report))
    print('\nartifacts written to %s' % output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())