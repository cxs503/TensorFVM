"""Durable artifact set for the anchor ``CASE-FVM-CHANNEL-KMM180``.

Why this module exists
----------------------
``tensorfvm.benchmark_channel_kmm1987`` grades the SST / k-omega / k-epsilon
channel solves against the KMM1987 DNS and writes ``results.json``,
``results.csv`` and ``REPORT.md``.  Those files are shaped for a human reader:
the per-cell profiles are inlined in the JSON and the two grading windows are
not separated, so a consumer (catalog, dashboard, another agent) cannot diff or
re-derive the numbers without parsing prose.

This module emits the anchor's durable artifact set in the repo-wide layout
(``benchmark.json`` / ``metrics.csv`` / ``profiles.csv`` / ``stations.csv`` --
same schema family as ``docs/case-channel-periodic``):

* ``benchmark.json`` -- case metadata, DNS reference, both grading windows and
  one record per ``(model, n)`` carrying every graded number;
* ``metrics.csv``    -- one row per ``(case, station, metric)`` with
  reference / computed / relative error / tolerance / verdict;
* ``profiles.csv``   -- per-cell ``y+``, model ``u+``, DNS ``u+`` and relative
  error for every case;
* ``stations.csv``   -- the grade stations themselves: bounds, cell counts, DNS
  values at the bounds and the per-case verdict.

Grading windows
---------------
Two windows are reported and neither may be quoted alone (see
``docs/oss_solutions/CASE-FVM-CHANNEL-KMM180.json``):

* ``legacy_full`` -- ``30 <= y+ <= centreline``.  The literal reading of the
  anchor text and the verdict of the legacy validation matrix.
* ``log_layer``   -- ``30 <= y+ <= 0.2 Re_tau``.  At ``Re_tau ~ 180`` this is
  the only interval that is actually logarithmic (a few wall units wide).

The ``legacy_full`` window can PASS while the log layer FAILS, because a
near-constant ``u+`` offset is divided down by the larger outer-layer ``u+``
(see :mod:`tensorfvm.verification.kmm180_windows`).  Both verdicts are
therefore written into every artifact.

Nothing here is typed in by hand: every number is computed from the solver
output and the imported DNS profile.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

from .kmm180_windows import LOG_LAYER_FRACTION, LOW_Y_PLUS
from .kmm1987 import TOLERANCE_LOG_REGION, load_kmm1987_reference

SCHEMA = 'tensorfvm.case-fvm-channel-kmm180/1'

#: Station identifiers, in report order.
LEGACY_STATION = 'legacy_full'
LOG_LAYER_STATION = 'log_layer'
STATIONS = (LEGACY_STATION, LOG_LAYER_STATION)

#: Files ``write_artifacts`` owns (``REPORT.md`` / ``manifest.json`` are written
#: by the benchmark driver, which needs the artifact numbers first).
OWNED_FILES = ('benchmark.json', 'metrics.csv', 'profiles.csv', 'stations.csv')

#: A window needs at least this many model cells to be graded at all (matches
#: the guard in :func:`tensorfvm.verification.kmm1987.log_region_deviation`).
MIN_WINDOW_CELLS = 3


def windows(dns):
    """The two grading windows as ``{station: {low, high, definition}}``."""
    centre = float(np.max(dns['y_plus']))
    return {
        LEGACY_STATION: dict(
            low=LOW_Y_PLUS,
            high=centre,
            definition='y+ in [30, centreline]; literal reading of the anchor text',
        ),
        LOG_LAYER_STATION: dict(
            low=LOW_Y_PLUS,
            high=LOG_LAYER_FRACTION * float(dns['re_tau']),
            definition='y+ in [30, 0.2*Re_tau]; the only interval that is '
                       'logarithmic at this Re_tau',
        ),
    }


def case_key(case):
    """Stable identifier of a case record, e.g. ``sst_n128``."""
    return f"{case['model']}_n{case['n']}"


def case_profile(case, dns):
    """Model/DNS ``u+`` on the model cells plus the relative error.

    Returns ``None`` for a case that did not converge (it carries no profile).
    """
    if not case.get('converged'):
        return None
    y_plus = np.asarray(case['y_plus'], dtype=float)
    u_plus = np.asarray(case['u_plus'], dtype=float)
    dns_u = PchipInterpolator(dns['y_plus'], dns['u_plus'])(y_plus)
    # A cell sitting exactly on the wall has DNS u+ = 0, so the relative error
    # is undefined there (0/0); those cells stay NaN and are never inside a
    # graded window (both windows start at y+ = 30).
    with np.errstate(invalid='ignore', divide='ignore'):
        relative = (u_plus - dns_u) / dns_u
    return dict(y_plus=y_plus, u_plus=u_plus, dns_u_plus=dns_u, relative=relative)


def window_of(case, dns, station):
    """RMS/max/mean relative deviation of one case over one station."""
    profile = case_profile(case, dns)
    bound = windows(dns)[station]
    if profile is None:
        return dict(low=bound['low'], high=bound['high'], cells=0,
                    rms_relative=None, max_relative=None, mean_relative=None,
                    mean_offset=None, mean_u_plus_dns=None, graded=False,
                    verdict='FAIL', reason='solver did not converge')
    mask = (profile['y_plus'] >= bound['low']) & (profile['y_plus'] <= bound['high'])
    cells = int(np.count_nonzero(mask))
    if cells < MIN_WINDOW_CELLS:
        return dict(low=bound['low'], high=bound['high'], cells=cells,
                    rms_relative=None, max_relative=None, mean_relative=None,
                    mean_offset=None, mean_u_plus_dns=None, graded=False,
                    verdict='FAIL',
                    reason=f'window holds {cells} cells (< {MIN_WINDOW_CELLS})')
    rel = profile['relative'][mask]
    rms = float(np.sqrt(np.mean(rel ** 2)))
    return dict(
        low=bound['low'],
        high=bound['high'],
        cells=cells,
        rms_relative=rms,
        max_relative=float(np.max(np.abs(rel))),
        mean_relative=float(np.mean(rel)),
        mean_offset=float(np.mean(profile['u_plus'][mask] - profile['dns_u_plus'][mask])),
        mean_u_plus_dns=float(np.mean(profile['dns_u_plus'][mask])),
        graded=True,
        verdict='PASS' if rms < TOLERANCE_LOG_REGION else 'FAIL',
        reason='',
    )


def station_rows(cases, dns=None):
    """One row per ``(case, station)`` for ``stations.csv``."""
    dns = load_kmm1987_reference() if dns is None else dns
    interp = PchipInterpolator(dns['y_plus'], dns['u_plus'])
    bounds = windows(dns)
    rows = []
    for case in cases:
        for station in STATIONS:
            window = window_of(case, dns, station)
            rows.append(dict(
                station=station,
                case=case_key(case),
                model=case['model'],
                n=case['n'],
                definition=bounds[station]['definition'],
                low_y_plus=window['low'],
                high_y_plus=window['high'],
                cells=window['cells'],
                dns_u_plus_at_low=float(interp(window['low'])),
                dns_u_plus_at_high=float(interp(window['high'])),
                mean_u_plus_dns=window['mean_u_plus_dns'],
                rms_relative=window['rms_relative'],
                max_relative=window['max_relative'],
                mean_offset_u_plus=window['mean_offset'],
                tolerance=TOLERANCE_LOG_REGION,
                graded=window['graded'],
                verdict=window['verdict'],
            ))
    return rows


def metric_rows(cases, dns=None):
    """One row per ``(case, station, metric)`` for ``metrics.csv``.

    ``Cf`` and ``U_bulk+`` are reported next to the graded window metrics but
    are *not* graded here (``graded=False``): the anchor criterion is a
    statement about ``u+``, and writing a ``Cf`` pass/fail that disagrees with
    the headline verdict would be an invented criterion.
    """
    dns = load_kmm1987_reference() if dns is None else dns
    rows = []
    for case in cases:
        key = case_key(case)
        base = dict(case=key, model=case['model'], n=case['n'])
        if not case.get('converged'):
            rows.append(dict(**base, station='', metric='converged',
                             reference=True, computed=False, relative_error=None,
                             unit='-', tolerance=None, graded=True, passed=False,
                             verdict='FAIL'))
            continue
        rows.append(dict(**base, station='', metric='Cf',
                         reference=float(dns['Cf']), computed=float(case['Cf']),
                         relative_error=float(abs(case['Cf'] - dns['Cf']) / dns['Cf']),
                         unit='-', tolerance=None, graded=False, passed=None,
                         verdict=''))
        rows.append(dict(**base, station='', metric='bulk_u_plus',
                         reference=float(dns['U_bulk_plus']),
                         computed=float(case['bulk_u_plus']),
                         relative_error=float(
                             abs(case['bulk_u_plus'] - dns['U_bulk_plus'])
                             / dns['U_bulk_plus']),
                         unit='-', tolerance=None, graded=False, passed=None,
                         verdict=''))
        for station in STATIONS:
            window = window_of(case, dns, station)
            for metric, value in (('log_rms_relative', window['rms_relative']),
                                  ('log_max_relative', window['max_relative'])):
                graded = metric == 'log_rms_relative' and window['graded']
                rows.append(dict(
                    **base, station=station, metric=metric,
                    reference=0.0, computed=value, relative_error=value,
                    unit='1',
                    tolerance=TOLERANCE_LOG_REGION if graded else None,
                    graded=graded,
                    passed=(window['verdict'] == 'PASS') if graded else None,
                    verdict=window['verdict'] if graded else '',
                ))
    return rows


def profile_rows(cases, dns=None):
    """Per-cell profile rows for ``profiles.csv``."""
    dns = load_kmm1987_reference() if dns is None else dns
    rows = []
    for case in cases:
        profile = case_profile(case, dns)
        if profile is None:
            continue
        for y_plus, u_plus, dns_u, rel in zip(profile['y_plus'], profile['u_plus'],
                                              profile['dns_u_plus'], profile['relative']):
            rows.append(dict(case=case_key(case), model=case['model'], n=case['n'],
                             y_plus=float(y_plus), u_plus=float(u_plus),
                             u_plus_dns=float(dns_u), error_relative=float(rel)))
    return rows


def build_benchmark(cases, dns=None, *, elapsed_s=None, matrix_elapsed_s=None,
                    repro_cmd=None, substitution=None):
    """The ``benchmark.json`` payload (every number, no arrays of profiles)."""
    dns = load_kmm1987_reference() if dns is None else dns
    bounds = windows(dns)
    records = []
    for case in cases:
        record = dict(case=case_key(case), model=case['model'], n=case['n'],
                      converged=bool(case.get('converged')))
        if not case.get('converged'):
            record.update(verdict='FAIL', reason=case.get('reason', 'not converged'))
            records.append(record)
            continue
        record.update(
            cells=int(len(case['y_plus'])),
            iterations=int(case['iterations']),
            Cf=float(case['Cf']),
            cf_relative_error=float(abs(case['Cf'] - dns['Cf']) / dns['Cf']),
            bulk_u_plus=float(case['bulk_u_plus']),
            bulk_relative_error=float(
                abs(case['bulk_u_plus'] - dns['U_bulk_plus']) / dns['U_bulk_plus']),
        )
        for station in STATIONS:
            window = window_of(case, dns, station)
            record[station] = {k: window[k] for k in
                               ('low', 'high', 'cells', 'rms_relative',
                                'max_relative', 'mean_relative', 'mean_offset',
                                'mean_u_plus_dns', 'graded', 'verdict')}
        record['verdict'] = record[LEGACY_STATION]['verdict']
        record['overall_verdict'] = (
            'PASS' if record[LEGACY_STATION]['verdict'] == 'PASS'
            and record[LOG_LAYER_STATION]['verdict'] == 'PASS' else 'FAIL')
        records.append(record)

    def _all(station, model=None):
        picked = [r for r in records
                  if r.get(station) and (model is None or r['model'] == model)]
        return bool(picked) and all(r[station]['verdict'] == 'PASS' for r in picked)

    payload = dict(
        schema=SCHEMA,
        anchor='CASE-FVM-CHANNEL-KMM180',
        reference=dict(
            citation='Kim, J., Moin, P. & Moser, R. D. 1987, J. Fluid Mech. 177, 133-166',
            doi='10.1017/S0022112087000892',
            dns_proxy='Lee & Moser 2015, J. Fluid Mech. 774, 395-415 (Re_tau=182.088)',
        ),
        dns=dict(re_tau=float(dns['re_tau']), Cf=float(dns['Cf']),
                 U_bulk_plus=float(dns['U_bulk_plus']),
                 points=int(len(dns['y_plus'])),
                 max_y_plus=float(np.max(dns['y_plus']))),
        tolerance=TOLERANCE_LOG_REGION,
        grading_windows={station: dict(bounds[station]) for station in STATIONS},
        plan=dict(
            models=sorted({case['model'] for case in cases}),
            resolutions=sorted({int(case['n']) for case in cases}),
            wall_treatment=dict(
                sst='low-Re (viscous sublayer resolved)',
                komega='low-Re (viscous sublayer resolved)',
                kepsilon='wall function, first cell at y+ = 30',
            ),
        ),
        cases=records,
        cases_total=len(records),
        all_passed_legacy=_all(LEGACY_STATION),
        all_passed_log_layer=_all(LOG_LAYER_STATION),
        all_passed_overall=all(r.get('overall_verdict') == 'PASS' for r in records),
        headline=_headline(records),
        verdict_qualified=True,
        reference_substitution=substitution,
        elapsed_s=elapsed_s,
        matrix_elapsed_s=matrix_elapsed_s,
        repro_cmd=repro_cmd,
    )
    return payload


def _headline(records):
    """One-line honest verdict: both windows, per model family."""
    parts = []
    for model in sorted({r['model'] for r in records}):
        family = [r for r in records if r['model'] == model and r.get('legacy_full')]
        if not family:
            parts.append(f'{model}: no converged case')
            continue
        legacy = sorted({r[LEGACY_STATION]['verdict'] for r in family})
        log_layer = sorted({r[LOG_LAYER_STATION]['verdict'] for r in family})
        parts.append(
            f"{model}: legacy_full {'/'.join(legacy)}, "
            f"log_layer {'/'.join(log_layer)}")
    return ('QUALIFIED -- ' + '; '.join(parts)
            + '. Both windows are published above and neither may be quoted alone.')


# --------------------------------------------------------------------------- #
# Serialisation
# --------------------------------------------------------------------------- #

_FIELDS = dict(
    metrics=('case', 'model', 'n', 'station', 'metric', 'reference', 'computed',
             'relative_error', 'unit', 'tolerance', 'graded', 'passed', 'verdict'),
    profiles=('case', 'model', 'n', 'y_plus', 'u_plus', 'u_plus_dns', 'error_relative'),
    stations=('station', 'case', 'model', 'n', 'definition', 'low_y_plus', 'high_y_plus',
              'cells', 'dns_u_plus_at_low', 'dns_u_plus_at_high', 'mean_u_plus_dns',
              'rms_relative', 'max_relative', 'mean_offset_u_plus', 'tolerance',
              'graded', 'verdict'),
)


def _fmt(value):
    if value is None:
        return ''
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f'{float(value):.12g}'
    return str(value)


def write_csv(path, rows, fields):
    """Write ``rows`` (dicts) with a deterministic header."""
    path = Path(path)
    with path.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        for row in rows:
            writer.writerow([_fmt(row.get(field)) for field in fields])
    return path


def write_artifacts(out, cases, dns=None, *, elapsed_s=None, matrix_elapsed_s=None,
                    repro_cmd=None, substitution=None):
    """Write ``benchmark.json`` / ``metrics.csv`` / ``profiles.csv`` / ``stations.csv``.

    Returns a dict with the written ``paths`` and the ``benchmark`` payload so
    the caller (benchmark driver or test) can quote the same numbers it just
    wrote without re-deriving them.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    dns = load_kmm1987_reference() if dns is None else dns

    benchmark = build_benchmark(cases, dns, elapsed_s=elapsed_s,
                               matrix_elapsed_s=matrix_elapsed_s,
                               repro_cmd=repro_cmd, substitution=substitution)
    paths = {
        'benchmark.json': out / 'benchmark.json',
        'metrics.csv': write_csv(out / 'metrics.csv', metric_rows(cases, dns),
                                 _FIELDS['metrics']),
        'profiles.csv': write_csv(out / 'profiles.csv', profile_rows(cases, dns),
                                  _FIELDS['profiles']),
        'stations.csv': write_csv(out / 'stations.csv', station_rows(cases, dns),
                                  _FIELDS['stations']),
    }
    paths['benchmark.json'].write_text(json.dumps(benchmark, indent=2) + '\n')
    return dict(paths=paths, benchmark=benchmark)