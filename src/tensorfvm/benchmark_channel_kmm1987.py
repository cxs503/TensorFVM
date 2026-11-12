"""CASE-FVM-CHANNEL-KMM180 — turbulent channel flow vs the KMM1987 DNS.

Grades the mean-velocity profile of the one-dimensional finite-volume
equilibrium channel solve (:mod:`tensorfvm.rans_validation.channel`) against
the Kim, Moin & Moser (1987) DNS at ``Re_tau = 180``.

Reference
---------
Kim, J., Moin, P. & Moser, R. D. 1987 *Turbulence statistics in fully developed
channel flow at low Reynolds number*. J. Fluid Mech. **177**, 133-166.
DOI: 10.1017/S0022112087000892

Numerical reference used for the pointwise comparison: the Lee & Moser (2015)
``Re_tau = 182.088`` channel case imported into
``docs/rans-turbulence-validation-matrix/dns/0180`` (see ``provenance.json``).
The two data sets are the same flow at a 1.2 % difference in ``Re_tau``; the
substitution is disclosed in the report rather than hidden.

Criterion (case pool row ``CASE-FVM-CHANNEL-KMM180``)
-----------------------------------------------------
RMS relative deviation of ``u+(y+)`` over ``y+ > 30`` (to the centreline)
below **5 %**, for the graded closure ``sst``.  ``komega`` and ``kepsilon`` are
run as negative controls: the metric is only meaningful if they are rejected.

Entry points
------------
* module CLI:  ``python -m tensorfvm.benchmark_channel_kmm1987 --out <dir>``
* script CLI:  ``python scripts/run_case_channel_kmm180.py --out <dir>``
* suite:       ``tensorfvm.verification.cases.channel_kmm180`` (called by
  ``python -m tensorfvm.verification --channel-kmm180``)
"""
import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

from tensorfvm.rans_validation.channel import solve_channel
from tensorfvm.verification.kmm1987 import (
    CASE_TOLERANCE,
    CONTROL_MODELS,
    GATED_MODEL,
    STATIONS_Y_PLUS,
    grade_case,
    load_kmm1987_reference,
)

CASE = 'CASE-FVM-CHANNEL-KMM180'
REFERENCE = ('Kim, Moin & Moser 1987 J. Fluid Mech. 177:133, '
             'DOI:10.1017/S0022112087000892')
DNS_PROXY = 'Lee & Moser 2015 J. Fluid Mech. 774:395 (Re_tau = 182.088, same flow)'

#: Closures run by the benchmark: the graded one plus the negative controls.
MODELS = (GATED_MODEL,) + CONTROL_MODELS
#: Wall-normal resolutions of the graded closure (grid ladder).
RESOLUTIONS = (64, 128, 256, 512)
#: Resolutions of the negative controls.
CONTROL_RESOLUTIONS = (64, 128, 256)
#: Re_tau of the DNS reference (nominal, per the case pool row).
RE_TAU = 180.0


# --------------------------------------------------------------------------- #
# Single case
# --------------------------------------------------------------------------- #
def run_case(model, n, re_tau=RE_TAU, dns=None):
    """Solve one (closure, resolution) pair and grade it against the reference."""
    options = {}
    if model == 'kepsilon':
        # Standard k-epsilon is a wall-function model: its first node sits at
        # y+ = 30 and the sublayer is not resolved.
        options['first_cell_y_plus'] = 30.0

    result = solve_channel(model, n, re_tau=re_tau, **options)
    if dns is None:
        dns = load_kmm1987_reference()

    if not result['converged']:
        return dict(model=model, n=n, re_tau=re_tau, converged=False,
                    verdict='FAIL', reason='solver did not converge',
                    y_plus=result['y_plus'].tolist(),
                    u_plus=result['u_plus'].tolist())

    graded = grade_case(model, result['y_plus'], result['u_plus'],
                        result['Cf'], result['bulk_u_plus'], dns=dns)
    graded.update(dict(
        n=n,
        re_tau=re_tau,
        converged=True,
        iterations=result['iterations'],
        relative_local_residual=result['relative_local_residual'],
        y_plus=result['y_plus'].tolist(),
        u_plus=result['u_plus'].tolist(),
    ))
    return graded


# --------------------------------------------------------------------------- #
# Benchmark record
# --------------------------------------------------------------------------- #
def discretisation_ladder(gated_cases, stations=STATIONS_Y_PLUS):
    """Successive-resolution differences of the graded profile at fixed stations.

    Separates the discretisation error from the model-form error: the profile
    differences must shrink as ``n`` doubles even when the graded RMS stalls at
    the closure's error floor.
    """
    rungs = []
    for coarse, fine in zip(gated_cases, gated_cases[1:]):
        yc = np.asarray(coarse['y_plus'], dtype=float)
        uc = np.asarray(coarse['u_plus'], dtype=float)
        yf = np.asarray(fine['y_plus'], dtype=float)
        uf = np.asarray(fine['u_plus'], dtype=float)
        st = np.asarray([s for s in stations if s <= min(yc[-1], yf[-1])])
        diff = np.interp(st, yc, uc) - np.interp(st, yf, uf)
        rungs.append(dict(
            n_coarse=coarse['n'], n_fine=fine['n'],
            station_max_abs_diff=float(np.max(np.abs(diff))),
            station_rms_abs_diff=float(np.sqrt(np.mean(diff ** 2))),
            graded_rms_coarse=coarse['got']['log_region_rms'],
            graded_rms_fine=fine['got']['log_region_rms'],
        ))
    return rungs


def build_record(cases, dns, elapsed_s, gated_model=GATED_MODEL,
                 tolerance=CASE_TOLERANCE):
    """Assemble the machine-checkable ref/got record for the case.

    ``benchmark`` holds the graded closure: the reference values, the numbers
    obtained, the relative error of each of them and the verdict of the
    predeclared criterion.  ``controls`` holds the negative controls, and
    ``sensitivity`` the grid ladder and graded-window variation.
    """
    gated = [c for c in cases if c['model'] == gated_model and c.get('converged')]
    finest = max(gated, key=lambda c: c['n'])

    # Graded-window sensitivity: the same profiles scored over three windows.
    windows = {}
    for label, low, frac in (('y+>30_to_centreline', 30.0, None),
                             ('y+>30_to_0.2Re_tau', 30.0, 0.2),
                             ('y+>30_to_0.5Re_tau', 30.0, 0.5)):
        from tensorfvm.verification.kmm1987 import log_region_deviation
        windows[label] = {
            c['model'] + f'_n{c["n"]}': log_region_deviation(
                c['y_plus'], c['u_plus'], dns, low=low, high_fraction=frac
            )['rms_relative']
            for c in cases if c.get('converged')
        }

    record = {
        'case': CASE,
        'criterion': {
            'metric': 'RMS relative deviation of u+(y+) over y+>30 (to centreline)',
            'tolerance': tolerance,
            'graded_closure': gated_model,
            'source': ('tensor_case_pool CASE-FVM-CHANNEL-KMM180; tol <5%; '
                       'ref KMM1987 DNS u+(y+)'),
        },
        'reference': {
            'citation': REFERENCE,
            'numerical_reference': DNS_PROXY,
            're_tau': dns['re_tau'],
            'Cf': dns['Cf'],
            'U_bulk_plus': dns['U_bulk_plus'],
            'u_plus_centre': float(dns['u_plus'][int(np.argmax(dns['y_plus']))]),
            'data_file': 'docs/rans-turbulence-validation-matrix/dns/0180/mean.csv',
        },
        'benchmark': {
            'ref': {k: finest['ref'][k] for k in
                    ('log_region_rms', 'cf', 'bulk_u_plus', 'u_plus_centre',
                     'kappa_apparent', 'B_apparent')},
            'got': {k: finest['got'][k] for k in
                    ('log_region_rms', 'log_region_max', 'cf', 'bulk_u_plus',
                     'u_plus_centre', 'kappa_apparent', 'B_apparent',
                     'station_max_rel_error')},
            'rel_error': finest['rel_error'],
            'tolerance': tolerance,
            'verdict': finest['verdict'],
            'resolution': finest['n'],
        },
        'stations': finest['stations'],
        'controls': [
            dict(model=c['model'], n=c['n'], verdict=c['verdict'],
                 log_region_rms=c['got']['log_region_rms'],
                 cf_rel_error=c['rel_error']['cf'],
                 bulk_rel_error=c['rel_error']['bulk_u_plus'])
            for c in cases if c['model'] != gated_model and c.get('converged')
        ],
        'grid_ladder': [
            dict(n=c['n'], log_region_rms=c['got']['log_region_rms'],
                 cf_rel_error=c['rel_error']['cf'],
                 bulk_rel_error=c['rel_error']['bulk_u_plus'],
                 verdict=c['verdict'])
            for c in sorted(gated, key=lambda c: c['n'])
        ],
        'discretisation': discretisation_ladder(sorted(gated, key=lambda c: c['n'])),
        'window_sensitivity': windows,
        'cases': [
            dict(model=c['model'], n=c['n'], converged=c.get('converged', False),
                 verdict=c['verdict'], role=c.get('role', ''),
                 tolerance=c.get('tolerance', tolerance),
                 log_region_rms=(c['got']['log_region_rms']
                                 if c.get('converged') else None),
                 log_region_max=(c['got']['log_region_max']
                                 if c.get('converged') else None),
                 cf_rel_error=(c['rel_error']['cf'] if c.get('converged') else None),
                 bulk_rel_error=(c['rel_error']['bulk_u_plus']
                                 if c.get('converged') else None),
                 iterations=c.get('iterations'),
                 relative_local_residual=c.get('relative_local_residual'))
            for c in cases
        ],
        'verdict': finest['verdict'],
        'controls_rejected': all(c['verdict'] == 'FAIL'
                                 for c in cases
                                 if c['model'] != gated_model and c.get('converged')),
        'elapsed_s': round(elapsed_s, 2),
        'environment': {
            'python': sys.version.split()[0],
            'platform': platform.platform(),
            'numpy': np.__version__,
        },
    }
    return record


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def make_figures(record, out_dir):
    """Velocity-profile comparison and log-region error figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    dns = load_kmm1987_reference()
    out_dir.mkdir(parents=True, exist_ok=True)
    figures = []

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(dns['y_plus'], dns['u_plus'], 'k-', linewidth=2,
             label='DNS (KMM1987 / LM2015)')
    ax2.semilogx(dns['y_plus'], dns['u_plus'], 'k-', linewidth=2, label='DNS')

    for c in record.pop('_raw_cases', []):
        if not c.get('converged'):
            continue
        label = f"{c['model']} n={c['n']}"
        style = dict(marker='os^'[(MODELS.index(c['model'])) % 3], markersize=3,
                     linewidth=1,
                     markevery=max(1, len(c['y_plus']) // 15))
        ax1.plot(c['y_plus'], c['u_plus'], label=label, **style)
        ax2.semilogx(c['y_plus'], c['u_plus'], label=label, **style)

    y_log = np.logspace(np.log10(30), np.log10(RE_TAU), 50)
    ax2.semilogx(y_log, np.log(y_log) / 0.41 + 5.0, 'k--', linewidth=1,
                 alpha=0.6, label='log law (kappa=0.41, B=5.0)')
    for ax in (ax1, ax2):
        ax.set_xlabel('y+')
        ax.set_ylabel('u+')
        ax.grid(True, alpha=0.3)
    ax1.set_title('mean velocity profile')
    ax2.set_title('semi-log')
    ax2.legend(fontsize=7, loc='lower right')
    fig.tight_layout()
    path = out_dir / 'fig_velocity_profiles.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figures.append(str(path))

    stations = record['stations']
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot([s['y_plus'] for s in stations],
            100 * np.array([s['rel_error'] for s in stations]), 'o-',
            label=GATED_MODEL)
    ax.axhline(5.0, color='r', linestyle='--', linewidth=1, label='+5 % gate')
    ax.axhline(-5.0, color='r', linestyle='--', linewidth=1)
    ax.axhline(0.0, color='k', linewidth=0.8)
    ax.set_xlabel('y+')
    ax.set_ylabel('(u+_model - u+_ref)/u+_ref  [%]')
    ax.set_title(f'station error, {GATED_MODEL} n={record["benchmark"]["resolution"]}')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / 'fig_log_region_error.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figures.append(str(path))
    return figures


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def render_report(record, figures):
    """Markdown report of the run (numbers are injected programmatically)."""
    bench = record['benchmark']
    ref, got, err = bench['ref'], bench['got'], bench['rel_error']
    lines = [
        f"# {CASE}: turbulent channel flow vs KMM1987 DNS",
        "",
        "Reference: " + REFERENCE,
        "",
        f"Numerical reference: {DNS_PROXY}.",
        "",
        "## Criterion",
        "",
        f"* {record['criterion']['metric']}",
        f"* tolerance < {100 * record['criterion']['tolerance']:.0f} %",
        f"* graded closure: `{record['criterion']['graded_closure']}`",
        "",
        "## Reference vs computed",
        "",
        "| quantity | ref (DNS) | got (FVM) | rel. error |",
        "|---|---|---|---|",
        f"| RMS rel. deviation u+, y+>30 | {ref['log_region_rms']:.3e} | "
        f"{got['log_region_rms']:.4f} | {err['log_region_rms']:.2%} |",
        f"| C_f | {ref['cf']:.6f} | {got['cf']:.6f} | {err['cf']:.2%} |",
        f"| U_bulk+ | {ref['bulk_u_plus']:.4f} | {got['bulk_u_plus']:.4f} | "
        f"{err['bulk_u_plus']:.2%} |",
        f"| u+ centreline | {ref['u_plus_centre']:.4f} | {got['u_plus_centre']:.4f} | "
        f"{err['u_plus_centre']:.2%} |",
        f"| apparent kappa | {ref['kappa_apparent']:.4f} | {got['kappa_apparent']:.4f} | - |",
        f"| apparent B | {ref['B_apparent']:.4f} | {got['B_apparent']:.4f} | - |",
        "",
        f"Verdict of the graded closure: **{bench['verdict']}** "
        f"(resolution n = {bench['resolution']}).",
        "",
        "## Station table (y+ > 30)",
        "",
        "| y+ | u+ ref | u+ computed | abs. error | rel. error |",
        "|---|---|---|---|---|",
    ]
    for s in record['stations']:
        lines.append(f"| {s['y_plus']:.0f} | {s['u_plus_ref']:.4f} | "
                     f"{s['u_plus_model']:.4f} | {s['abs_error']:.4f} | "
                     f"{s['rel_error']:.2%} |")

    lines += ["", "## Negative controls", "",
              "| closure | n | log-region RMS | C_f error | U_bulk+ error | verdict |",
              "|---|---|---|---|---|---|"]
    for c in record['controls']:
        lines.append(f"| {c['model']} | {c['n']} | {c['log_region_rms']:.2%} | "
                     f"{c['cf_rel_error']:.2%} | {c['bulk_rel_error']:.2%} | "
                     f"{c['verdict']} |")
    lines += ["", f"All negative controls rejected: **{record['controls_rejected']}**", ""]

    lines += ["## Grid ladder (graded closure)", "",
              "| n | log-region RMS | C_f error | U_bulk+ error | verdict |",
              "|---|---|---|---|---|"]
    for c in record['grid_ladder']:
        lines.append(f"| {c['n']} | {c['log_region_rms']:.2%} | "
                     f"{c['cf_rel_error']:.2%} | {c['bulk_rel_error']:.2%} | "
                     f"{c['verdict']} |")

    lines += ["", "## Discretisation ladder (graded closure)", "",
              "| n -> 2n | max abs. du+ at stations | RMS abs. du+ | graded RMS coarse -> fine |",
              "|---|---|---|---|"]
    for r in record['discretisation']:
        lines.append(f"| {r['n_coarse']} -> {r['n_fine']} | "
                     f"{r['station_max_abs_diff']:.3e} | "
                     f"{r['station_rms_abs_diff']:.3e} | "
                     f"{r['graded_rms_coarse']:.2%} -> {r['graded_rms_fine']:.2%} |")

    lines += ["", "## Graded-window sensitivity", "",
              "| window | " + " | ".join(sorted(record['window_sensitivity'])) + " |",
              "|---|" + "---|" * len(record['window_sensitivity'])]
    for model_n in sorted(next(iter(record['window_sensitivity'].values()))):
        row = [f"{100 * record['window_sensitivity'][w][model_n]:.2f} %"
               for w in sorted(record['window_sensitivity'])]
        lines.append(f"| {model_n} | " + " | ".join(row) + " |")

    lines += ["", "## Figures", ""]
    for fig in figures:
        name = Path(fig).name
        lines += [f"![{name}]({name})", ""]

    narrow = {k: v for k, v in
              record['window_sensitivity']['y+>30_to_0.2Re_tau'].items()
              if k.startswith(GATED_MODEL)}
    narrow_lo = 100 * min(narrow.values())
    narrow_hi = 100 * max(narrow.values())

    lines += ["## Notes and limitations", "",
              "* The solve is a 1-D equilibrium channel (finite-volume in the "
              "wall-normal direction): the momentum equation reduces to "
              "`(nu + nu_t) dU/dy = 1 - y`, so convection, pressure-velocity "
              "coupling and geometry are not exercised.  Agreement here is "
              "evidence that the closure coupling and the near-wall "
              "discretisation are consistent, not that the 3-D solvers are "
              "accurate.",
              "* The graded closure SST-2003 is calibrated to the log law, so a "
              "PASS is expected a priori for a wall-resolved mesh; the negative "
              "controls are what make the comparison non-trivial.",
              "* The reference data set is Lee & Moser (2015) at Re_tau = 182.088 "
              "because it is the digitised, provenance-tracked channel data set "
              "in this repository; KMM1987 is the case-pool citation for the "
              "same flow at Re_tau = 180.",
              "* At Re_tau = 180 the log layer is only a few wall units thick, so "
              "the apparent kappa of the reference fit falls below 0.41; it is "
              "reported as a diagnostic, not as a criterion.",
              "* **Window dependence, disclosed up front.** The verdict is taken "
              "over the whole y+ > 30 range, as the case-pool row states (no upper "
              "bound).  Restricted to the strict log window 30 < y+ < 0.2 Re_tau "
              f"the graded closure gives {narrow_lo:.2f}-{narrow_hi:.2f} %, i.e. "
              "above the 5 % bar: the case therefore certifies the resolved "
              "near-wall/log layer of the FVM channel solve, not a log-law "
              "constant.",
              ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Artifacts
# --------------------------------------------------------------------------- #
def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 16), b''):
            digest.update(block)
    return digest.hexdigest()


def write_artifacts(record, out_dir, repro_cmd):
    """Write the artifact set; returns the list of written paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    benchmark = {k: v for k, v in record.items() if not k.startswith('_')}
    path = out_dir / 'benchmark.json'
    path.write_text(json.dumps(benchmark, indent=2))
    written.append(path)

    # per-case metrics
    lines = ['model,n,converged,verdict,role,log_region_rms,log_region_max,'
             'cf_rel_error,bulk_rel_error,iterations,relative_local_residual']
    for c in record['cases']:
        def fmt(v):
            return '' if v is None else (
                f'{v:.8g}' if isinstance(v, float) else str(v))
        lines.append(','.join([
            c['model'], str(c['n']), str(c['converged']), c['verdict'], c['role'],
            fmt(c['log_region_rms']), fmt(c['log_region_max']),
            fmt(c['cf_rel_error']), fmt(c['bulk_rel_error']),
            fmt(c['iterations']), fmt(c['relative_local_residual'])]))
    path = out_dir / 'metrics.csv'
    path.write_text('\n'.join(lines) + '\n')
    written.append(path)

    # station table
    lines = ['y_plus,u_plus_ref,u_plus_model,abs_error,rel_error']
    for s in record['stations']:
        lines.append(f"{s['y_plus']:.1f},{s['u_plus_ref']:.8f},"
                     f"{s['u_plus_model']:.8f},{s['abs_error']:.8f},"
                     f"{s['rel_error']:.8e}")
    path = out_dir / 'stations.csv'
    path.write_text('\n'.join(lines) + '\n')
    written.append(path)

    # profiles of the graded resolution
    dns = load_kmm1987_reference()
    finest = max((c for c in record['_raw_cases'] if c.get('converged')
                  and c['model'] == GATED_MODEL), key=lambda c: c['n'])
    from scipy.interpolate import PchipInterpolator
    dns_interp = PchipInterpolator(dns['y_plus'], dns['u_plus'])
    y = np.asarray(finest['y_plus'], dtype=float)
    u = np.asarray(finest['u_plus'], dtype=float)
    ref = dns_interp(np.clip(y, dns['y_plus'][0], dns['y_plus'][-1]))
    lines = ['y_plus,u_plus_model,u_plus_ref,rel_error']
    for yi, ui, ri in zip(y, u, ref):
        lines.append(f'{yi:.6f},{ui:.8f},{ri:.8f},{(ui - ri) / ri:.6e}')
    path = out_dir / 'profiles.csv'
    path.write_text('\n'.join(lines) + '\n')
    written.append(path)

    figures = make_figures({**record, '_raw_cases': record['_raw_cases']}, out_dir)
    written += [Path(f) for f in figures]

    report = render_report({**record, '_raw_cases': None}, figures)
    path = out_dir / 'REPORT.md'
    path.write_text(report)
    written.append(path)

    manifest = dict(
        schema=f'{CASE}/artifact-manifest-v1',
        case=CASE,
        verdict=record['verdict'],
        controls_rejected=record['controls_rejected'],
        graded_closure=record['criterion']['graded_closure'],
        criterion=record['criterion'],
        reference=record['reference'],
        repro_cmd=repro_cmd,
        entry={'module': 'tensorfvm.benchmark_channel_kmm1987:main',
               'script': 'scripts/run_case_channel_kmm180.py',
               'suite': 'tensorfvm.verification.cases.channel_kmm180'},
        impl_files=['src/tensorfvm/verification/kmm1987.py',
                    'src/tensorfvm/benchmark_channel_kmm1987.py',
                    'src/tensorfvm/rans_validation/channel.py'],
        environment=record['environment'],
        elapsed_s=record['elapsed_s'],
        artifacts=[dict(path=str(p.relative_to(out_dir)), sha256=_sha256(p))
                   for p in written if Path(p).name != 'manifest.json'],
    )
    path = out_dir / 'manifest.json'
    path.write_text(json.dumps(manifest, indent=2))
    written.append(path)
    return written


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run_benchmark(out, quick=False, re_tau=RE_TAU):
    """Run the full case and write the artifact set; returns the record."""
    started = time.perf_counter()
    dns = load_kmm1987_reference()
    if quick:
        cases = [run_case(GATED_MODEL, 128, re_tau=re_tau, dns=dns)]
        cases += [run_case(m, 128, re_tau=re_tau, dns=dns) for m in CONTROL_MODELS]
    else:
        cases = [run_case(GATED_MODEL, n, re_tau=re_tau, dns=dns)
                 for n in RESOLUTIONS]
        cases += [run_case(model, n, re_tau=re_tau, dns=dns)
                  for model in CONTROL_MODELS for n in CONTROL_RESOLUTIONS]
    elapsed = time.perf_counter() - started
    record = build_record(cases, dns, elapsed)
    record['_raw_cases'] = cases
    repro_cmd = (f'PYTHONPATH=src python scripts/run_case_channel_kmm180.py '
                 f'--out {Path(out).as_posix()}')
    write_artifacts(record, Path(out), repro_cmd)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('docs/case-fvm-channel-kmm180'))
    parser.add_argument('--quick', action='store_true',
                        help='graded closure plus controls at n=128 only')
    parser.add_argument('--re-tau', type=float, default=RE_TAU)
    args = parser.parse_args(argv)

    record = run_benchmark(args.out, quick=args.quick, re_tau=args.re_tau)
    print(json.dumps(dict(
        case=CASE,
        verdict=record['verdict'],
        controls_rejected=record['controls_rejected'],
        log_region_rms=round(record['benchmark']['rel_error']['log_region_rms'], 5),
        tolerance=record['criterion']['tolerance'],
        cases_passed=sum(1 for c in record['cases'] if c['verdict'] == 'PASS'),
        cases_total=len(record['cases']),
        elapsed_s=record['elapsed_s'],
    ), indent=2))
    return 0 if (record['verdict'] == 'PASS') else 1


if __name__ == '__main__':
    raise SystemExit(main())