"""The graded turbulence-model validation matrix.

What the matrix contains
------------------------
Three closures (SST-2003, Wilcox k-omega, standard k-epsilon) on the equilibrium
half channel, each with

* a grid-refinement ladder, so the reported numbers are shown to be
  discretisation-independent rather than single-mesh accidents;
* the wall-law deviation in the log region against the original DNS of Lee &
  Moser (2015), against the classic logarithmic law and against the Reichardt
  (1951) composite law, over three window widths;
* the local log-law indicator ``Xi = y+ dU+/dy+`` (whose reciprocal is the slope
  the model actually produces) next to the slope its constant set implies in
  closed form;
* the skin-friction coefficient against the DNS value;
* the ``C_mu``-equivalent constant sweep used for the robustness criterion.

Reference discipline
--------------------
A DNS-graded case is run at the wall Reynolds number *reported in the DNS
header* (182.088 and 543.496 for the imported Lee & Moser cases ``0180`` and
``0550``), not at the nominal case label, so the comparison needs no Reynolds
interpolation.  Where no original DNS profile covers the Reynolds number -- the
mandated ``Re_tau = 590`` is not in the Lee & Moser 2015 series, which jumps from
550 to 1000 -- the case is graded against the logarithmic law and the Reichardt
composite law and the missing DNS is stated in the verdict rather than imputed.
"""
import json
from pathlib import Path

import numpy as np

from .channel import solve_channel
from .models import MODELS, WALL_TREATMENT
from .reference import (REFERENCE_LAWS, dns_deviation, load_dns_profile,
                        log_law_indicator, log_region_deviation, shear_stress_deviation)
from .sensitivity import constant_sweep

PRIMARY_REFERENCE = 'log_law'
DNS_REFERENCE = 'dns'
DNS_ROOT = Path(__file__).resolve().parents[3] / 'docs' / 'rans-turbulence-validation-matrix' / 'dns'
MANDATE_RE_TAU = 590.
REFERENCE_LABELS = {'dns': 'original DNS (Lee & Moser 2015)',
                    'log_law': 'logarithmic law kappa=0.41, B=5.2',
                    'reichardt': 'Reichardt (1951) composite law'}
FRACTIONS = (.15, .2, .3)
PRIMARY_FRACTION = .2
DEFAULT_LEVELS = {'low_re': (64, 128, 256, 512), 'wall_function': (16, 32, 64, 128)}
DEFAULT_CELLS = {'low_re': 512, 'wall_function': 64}
CRITERIA = {'log_region_deviation': .05, 'constant_sensitivity': .15}


def available_dns(dns_root=None):
    """Imported DNS profiles keyed by their reported wall Reynolds number."""
    root = Path(DNS_ROOT if dns_root is None else dns_root)
    profiles = {}
    if not root.is_dir():
        return profiles
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        if not (directory / 'mean.csv').is_file():
            continue
        profile = load_dns_profile(directory)
        profiles[round(profile['re_tau'], 3)] = profile
    return profiles


def default_re_tau_values(dns_root=None):
    """The DNS-anchored Reynolds numbers plus the mandated one."""
    return tuple(available_dns(dns_root)) + (MANDATE_RE_TAU,)


def solve_case(model, re_tau, n):
    """Solve one matrix cell with the wall treatment the model requires."""
    options = {}
    if WALL_TREATMENT[model] == 'wall_function':
        options['first_cell_y_plus'] = 30.
    return solve_channel(model, n, re_tau=re_tau, **options)


def match_dns(re_tau, profiles):
    """DNS profile whose reported ``Re_tau`` matches this case within 0.1 per cent."""
    for value, profile in profiles.items():
        if abs(value / float(re_tau) - 1) < 1e-3:
            return profile
    return None


def case_summary(result, *, fractions=FRACTIONS, dns=None):
    """Deviations, local slope and the implied slope for one converged case."""
    summary = dict(model=result['model'], re_tau=result['re_tau'], cells=result['n'],
                   wall_treatment=result['wall_treatment'], converged=result['converged'],
                   iterations=result['iterations'], Cf=result['Cf'],
                   bulk_u_plus=result['bulk_u_plus'], first_cell_y_plus=result['first_cell_y_plus'],
                   inner_boundary_y_plus=result['inner_boundary_y_plus'],
                   implied_kappa=result['implied_kappa'],
                   relative_local_residual=result['relative_local_residual'],
                   dns_case=None if dns is None else dns['case'],
                   Cf_deviation_vs_dns=None if dns is None else float(result['Cf'] / dns['Cf'] - 1),
                   deviations={})
    try:
        summary['local_kappa'] = log_law_indicator(result['y_plus'], result['u_plus'], re_tau=result['re_tau'],
                                                   high_fraction=PRIMARY_FRACTION)
    except ValueError:
        summary['local_kappa'] = None
    for name, law in REFERENCE_LAWS.items():
        per_fraction = {}
        for fraction in fractions:
            try:
                deviation = log_region_deviation(result['y_plus'], result['u_plus'], law,
                                                 re_tau=result['re_tau'], high_fraction=fraction)
            except ValueError:
                per_fraction['%.2f' % fraction] = None
                continue
            per_fraction['%.2f' % fraction] = dict(
                rms_relative=deviation['rms_relative'], max_relative=deviation['max_relative'],
                mean_relative=deviation['mean_relative'], cells=deviation['cells'],
                window=[deviation['low'], deviation['high']])
        summary['deviations'][name] = per_fraction
    if dns is not None:
        per_fraction = {}
        for fraction in fractions:
            try:
                deviation = dns_deviation(result['y_plus'], result['u_plus'], dns, high_fraction=fraction,
                                          re_tau=result['re_tau'])
            except ValueError:
                per_fraction['%.2f' % fraction] = None
                continue
            per_fraction['%.2f' % fraction] = dict(
                rms_relative=deviation['rms_relative'], max_relative=deviation['max_relative'],
                mean_relative=deviation['mean_relative'], cells=deviation['cells'],
                window=[deviation['low'], deviation['high']])
        summary['deviations'][DNS_REFERENCE] = per_fraction
        try:
            shear = shear_stress_deviation(result['y_plus'], result['nut'] * result['shear'], dns,
                                           high_fraction=PRIMARY_FRACTION, re_tau=result['re_tau'])
        except ValueError:
            shear = None
        summary['reynolds_shear'] = None if shear is None else dict(
            modelled=dict(rms_over_peak=shear['normalised_rms'], max_over_peak=shear['normalised_max'],
                          cells=shear['cells']),
            dns_peak=shear['dns_peak'],
            identity_residual=float(np.max(np.abs((result['nut'] + result['nu']) * result['shear'] - (1 - result['y'])))))
    else:
        summary['reynolds_shear'] = None
        summary['reynolds_shear_identity_residual'] = float(
            np.max(np.abs((result['nut'] + result['nu']) * result['shear'] - (1 - result['y']))))
    summary['log_layer_k_over_u_tau_squared'] = None
    in_log = (result['y_plus'] >= 30.) & (result['y_plus'] <= PRIMARY_FRACTION * result['re_tau'])
    if np.any(in_log):
        summary['log_layer_k_over_u_tau_squared'] = float(np.mean(result['k'][in_log]))
    return summary


def grid_ladder(model, re_tau, levels=None, dns=None):
    """Skin friction and primary deviation across the refinement ladder."""
    levels = DEFAULT_LEVELS[WALL_TREATMENT[model]] if levels is None else levels
    rows = []
    for n in levels:
        result = solve_case(model, re_tau, n)
        reference = DNS_REFERENCE if dns is not None else PRIMARY_REFERENCE
        try:
            if dns is not None:
                deviation = dns_deviation(result['y_plus'], result['u_plus'], dns,
                                          high_fraction=PRIMARY_FRACTION, re_tau=re_tau)['rms_relative']
            else:
                deviation = log_region_deviation(result['y_plus'], result['u_plus'],
                                                 REFERENCE_LAWS[PRIMARY_REFERENCE], re_tau=re_tau,
                                                 high_fraction=PRIMARY_FRACTION)['rms_relative']
        except ValueError:
            # A coarse ladder level can leave fewer than three cells in the window;
            # that level then carries no deviation instead of a fabricated value.
            deviation = None
        rows.append(dict(cells=int(n), Cf=result['Cf'], bulk_u_plus=result['bulk_u_plus'],
                         log_region_rms_relative=deviation, reference=reference,
                         converged=bool(result['converged']), iterations=result['iterations']))
    finest, previous = rows[-1], rows[-2]
    change = None
    if finest['log_region_rms_relative'] is not None and previous['log_region_rms_relative'] is not None:
        change = float(abs(finest['log_region_rms_relative'] - previous['log_region_rms_relative']))
    return dict(model=model, re_tau=float(re_tau), primary_reference=reference, rows=rows,
                finest_Cf_change=float(abs(finest['Cf'] / previous['Cf'] - 1)),
                finest_deviation_change=change)


def run_matrix(*, re_tau_values=None, models=MODELS, levels=None, cells=None, fractions=FRACTIONS,
               sensitivity=True, sensitivity_cells=256, dns_root=None):
    """Run the whole matrix and return a JSON-serialisable report."""
    levels = DEFAULT_LEVELS if levels is None else levels
    cells = DEFAULT_CELLS if cells is None else cells
    profiles = available_dns(dns_root)
    re_tau_values = default_re_tau_values(dns_root) if re_tau_values is None else re_tau_values
    cases, ladders = [], []
    for model in models:
        for re_tau in re_tau_values:
            dns = match_dns(re_tau, profiles)
            ladder = grid_ladder(model, re_tau, levels[WALL_TREATMENT[model]], dns=dns)
            ladders.append(ladder)
            result = solve_case(model, re_tau, cells[WALL_TREATMENT[model]])
            summary = case_summary(result, fractions=fractions, dns=dns)
            summary['grid_finest_Cf_change'] = ladder['finest_Cf_change']
            summary['grid_finest_deviation_change'] = ladder['finest_deviation_change']
            summary['profiles'] = dict(
                y_plus=result['y_plus'].tolist(), u_plus=result['u_plus'].tolist(),
                k_over_u_tau_squared=result['k'].tolist(),
                eddy_viscosity_ratio=(result['nut'] / result['nu']).tolist(),
                reynolds_shear_stress=(result['nut'] * result['shear']).tolist())
            cases.append(summary)
    sweeps = [constant_sweep(model, re_tau=max(re_tau_values), n=sensitivity_cells) for model in models] if sensitivity else []
    report = dict(
        description='RANS turbulence-model validation matrix, equilibrium half channel',
        normalisation='unit density, unit friction velocity, half-height length scale, nu = 1/Re_tau',
        references={name: label for name, label in REFERENCE_LABELS.items()},
        primary_reference=PRIMARY_REFERENCE, primary_fraction=PRIMARY_FRACTION,
        dns_cases={str(key): dict(case=profile['case'], Cf=profile['Cf'], upstream=profile['upstream'],
                                  raw_sha256=profile['raw_sha256']) for key, profile in profiles.items()},
        criteria=dict(CRITERIA),
        thresholds_note='PASS requires the graded quantity to stay below the threshold',
        cases=cases, grid=ladders, sensitivity=sweeps)
    report['verdicts'] = grade(report)
    return report


def grade(report):
    """Explicit per-criterion verdicts, each carrying its own numbers."""
    verdicts = []
    for case in report['cases']:
        available = case['deviations']
        primary_name = DNS_REFERENCE if DNS_REFERENCE in available else PRIMARY_REFERENCE
        primary = available[primary_name]['%.2f' % PRIMARY_FRACTION]
        alternatives = {name: available[name]['%.2f' % PRIMARY_FRACTION]
                        for name in ('log_law', 'reichardt', DNS_REFERENCE)
                        if name in available and name != primary_name}
        verdicts.append(dict(
            criterion='log_region_deviation', model=case['model'], re_tau=case['re_tau'],
            primary=dict(reference=primary_name, rms_relative=primary['rms_relative'],
                         max_relative=primary['max_relative'],
                         threshold=CRITERIA['log_region_deviation'],
                         passed=bool(primary['rms_relative'] < CRITERIA['log_region_deviation'])),
            alternatives={name: dict(reference=name, rms_relative=row['rms_relative'],
                                     max_relative=row['max_relative'],
                                     passed=bool(row['rms_relative'] < CRITERIA['log_region_deviation']))
                          for name, row in alternatives.items()},
            local_kappa=case['local_kappa'], implied_kappa=case['implied_kappa'], wall_law_kappa=.41,
            Cf_deviation_vs_dns=case['Cf_deviation_vs_dns']))
    for sweep in report['sensitivity']:
        verdicts.append(dict(
            criterion='constant_sensitivity', model=sweep['model'], constant=sweep['constant'],
            max_abs_Cf_change_within_ten_percent=sweep['max_abs_change_within_ten_percent'],
            threshold=CRITERIA['constant_sensitivity'],
            passed=bool(sweep['max_abs_change_within_ten_percent'] < CRITERIA['constant_sensitivity'])))
    return verdicts


def summarise(report):
    """Compact text table of the graded numbers."""
    lines = ['model      Re_tau   n  Cf        Ub+      Cf/DNS-1  kappa_impl  kappa_loc  rms(DNS)  rms(loglaw)  rms(Reich)  max(Reich)']
    for case in report['cases']:
        row = lambda name, key='rms_relative': (
            case['deviations'][name]['%.2f' % PRIMARY_FRACTION][key]
            if name in case['deviations'] and case['deviations'][name]['%.2f' % PRIMARY_FRACTION] else float('nan'))
        local = case['local_kappa']['kappa_local'] if case['local_kappa'] else float('nan')
        cf = case['Cf_deviation_vs_dns']
        lines.append('%-10s %6.3f %3d  %.5f  %6.3f  %8s  %10.4f  %9.4f  %7.2f%%  %10.2f%%  %9.2f%%  %9.2f%%' % (
            case['model'], case['re_tau'], case['cells'], case['Cf'], case['bulk_u_plus'],
            'n/a' if cf is None else '%+.2f%%' % (100 * cf), case['implied_kappa'], local,
            100 * row(DNS_REFERENCE), 100 * row('log_law'), 100 * row('reichardt'), 100 * row('reichardt', 'max_relative')))
    lines.append('')
    lines.append('constant sweep (graded: |dCf/Cf| < %.0f%% for a +/-10%% constant change)' % (100 * CRITERIA['constant_sensitivity']))
    for sweep in report['sensitivity']:
        lines.append('  %-10s %-11s base=%.4f  max|dCf/Cf|=%.2f%%  dlnCf/dlnC=%.2f  %s' % (
            sweep['model'], sweep['constant'], sweep['base_value'],
            100 * sweep['max_abs_change_within_ten_percent'], sweep['sensitivity_dCfdC_over_Cf'],
            'PASS' if sweep['max_abs_change_within_ten_percent'] < CRITERIA['constant_sensitivity'] else 'FAIL'))
    return '\n'.join(lines)


def write_report(report, path):
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=1, sort_keys=False)
    return path