"""Tests for the CASE-FVM-CHANNEL-KMM180 durable artifact set.

Three layers, in increasing cost:

* **synthetic** -- the metric/station/serialisation helpers must recover a
  manufactured deviation exactly, so a defect in the artifact layer cannot be
  mistaken for a physics result (and a 5.00 % deviation must be graded FAIL, not
  PASS);
* **registration** -- one real solve (SST, 64 cells) is written through
  :func:`tensorfvm.verification.kmm180_artifacts.write_artifacts` and every file
  is read back: ``benchmark.json`` / ``metrics.csv`` / ``profiles.csv`` /
  ``stations.csv`` must agree with each other and with the solver result;
* **committed artifacts** -- ``scripts/run_case_fvm_channel_kmm180.py
  --check`` must be clean on the artifacts in ``docs/case-fvm-channel-kmm180``,
  which couples the committed artifacts, their manifest sha256 and
  ``docs/oss_solutions/CASE-FVM-CHANNEL-KMM180.json``.  Hand-editing an artifact
  or letting the catalog drift is a test failure, not a silent inconsistency.
"""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from tensorfvm.benchmark_channel_kmm1987 import (
    SUBSTITUTION_BAND_PP,
    run_case,
    substitution_summary,
)
from tensorfvm.verification.kmm180_artifacts import (
    LEGACY_STATION,
    LOG_LAYER_STATION,
    metric_rows,
    profile_rows,
    station_rows,
    write_artifacts,
)
from tensorfvm.verification.kmm1987 import TOLERANCE_LOG_REGION, load_kmm1987_reference

REPO_ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = REPO_ROOT / 'docs' / 'case-fvm-channel-kmm180'
SCRIPT = REPO_ROOT / 'scripts' / 'run_case_fvm_channel_kmm180.py'


def _load_script():
    spec = importlib.util.spec_from_file_location('run_case_kmm180', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _read_csv(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope='module')
def dns():
    return load_kmm1987_reference()


def manufactured_case(dns, factor=1.05, model='sst', n=64):
    """A case record whose u+ is ``factor`` x DNS on every model cell."""
    y_plus = np.asarray(dns['y_plus'], dtype=float)
    u_plus = np.asarray(dns['u_plus'], dtype=float) * factor
    return dict(model=model, n=n, re_tau=180.0, converged=True, iterations=1,
                relative_local_residual=0.0, Cf=float(dns['Cf']) * factor,
                bulk_u_plus=float(dns['U_bulk_plus']) * factor,
                verdict='FAIL', verdict_log_layer='FAIL', overall_verdict='FAIL',
                log_region_rms=0.0, log_region_max=0.0,
                log_layer_rms=0.0, log_layer_max=0.0,
                cf_relative_error=0.0, bulk_relative_error=0.0,
                y_plus=y_plus.tolist(), u_plus=u_plus.tolist())


# --------------------------------------------------------------------------- #
# Synthetic
# --------------------------------------------------------------------------- #

def test_metric_rows_recover_a_manufactured_offset(dns):
    rows = metric_rows([manufactured_case(dns)], dns)
    graded = {(r['station'], r['metric']): r for r in rows if r['graded']}
    for station in (LEGACY_STATION, LOG_LAYER_STATION):
        row = graded[(station, 'log_rms_relative')]
        assert row['computed'] == pytest.approx(0.05, rel=1e-10)
        assert row['tolerance'] == pytest.approx(TOLERANCE_LOG_REGION)
        # exactly at the tolerance is not a PASS
        assert row['verdict'] == 'FAIL'
        assert row['passed'] is False


def test_ungraded_metrics_are_declared_as_such(dns):
    rows = metric_rows([manufactured_case(dns)], dns)
    for row in rows:
        if row['metric'] in ('Cf', 'bulk_u_plus'):
            assert row['graded'] is False
            assert row['passed'] is None
            assert row['verdict'] == ''


def test_station_rows_cover_both_windows(dns):
    rows = station_rows([manufactured_case(dns)], dns)
    assert [r['station'] for r in rows] == [LEGACY_STATION, LOG_LAYER_STATION]
    legacy, log_layer = rows
    assert legacy['high_y_plus'] == pytest.approx(float(np.max(dns['y_plus'])))
    assert log_layer['high_y_plus'] == pytest.approx(0.2 * dns['re_tau'])
    for row in rows:
        assert row['cells'] >= 3
        assert row['rms_relative'] == pytest.approx(0.05, rel=1e-10)
        assert row['mean_offset_u_plus'] > 0.0
    # the log layer is the narrower window by construction
    assert log_layer['cells'] < legacy['cells']


def test_profile_rows_carry_the_relative_error(dns):
    case = manufactured_case(dns)
    rows = profile_rows([case], dns)
    assert len(rows) == len(case['y_plus'])
    # y+ = 0 is a DNS sample point, not a model cell, so the relative error is
    # only defined for y+ > 0.
    checked = [row for row in rows if row['y_plus'] > 1.0]
    assert checked
    for row in checked:
        assert row['error_relative'] == pytest.approx(0.05, rel=1e-9)


def test_non_converged_case_is_recorded_as_fail_not_dropped(dns):
    rows = metric_rows([dict(model='sst', n=64, converged=False, verdict='FAIL')], dns)
    assert len(rows) == 1
    assert rows[0]['metric'] == 'converged'
    assert rows[0]['passed'] is False
    assert station_rows([dict(model='sst', n=64, converged=False)], dns)[0]['verdict'] == 'FAIL'
    assert profile_rows([dict(model='sst', n=64, converged=False)], dns) == []


def test_substitution_summary_flags_a_verdict_flip_and_a_band_breach():
    good = dict(case='sst_n64', re_tau_reference=182.088,
                delta_pp_legacy=0.01, delta_pp_log_layer=0.02,
                verdict_flip_legacy=False, verdict_flip_log_layer=False)
    assert substitution_summary([good])['passed'] is True

    flip = dict(good, case='sst_n128', verdict_flip_log_layer=True)
    summary = substitution_summary([good, flip])
    assert summary['passed'] is False
    assert summary['verdict_flips'] == ['sst_n128']

    wide = dict(good, case='sst_n256', delta_pp_legacy=SUBSTITUTION_BAND_PP + 0.05)
    summary = substitution_summary([wide])
    assert summary['passed'] is False
    assert summary['max_delta_pp'] > summary['band_pp']


# --------------------------------------------------------------------------- #
# Registration: one real solve written through the artifact writer
# --------------------------------------------------------------------------- #

@pytest.fixture(scope='module')
def written(tmp_path_factory, dns):
    out = tmp_path_factory.mktemp('kmm180_artifacts')
    case = run_case('sst', 64)
    artifacts = write_artifacts(out, [case], dns, elapsed_s=1.0, matrix_elapsed_s=1.0,
                                repro_cmd='n/a')
    return dict(out=out, case=case, benchmark=artifacts['benchmark'])


def test_benchmark_record_matches_the_solver_result(written):
    record = written['benchmark']['cases'][0]
    case = written['case']
    assert record['case'] == 'sst_n64'
    assert record['legacy_full']['rms_relative'] == pytest.approx(case['log_region_rms'])
    assert record['log_layer']['rms_relative'] == pytest.approx(case['log_layer_rms'])
    assert record['Cf'] == pytest.approx(case['Cf'])
    assert record['cells'] == len(case['y_plus'])
    # the registered windows are the ones the legacy validator used
    assert record['legacy_full']['verdict'] == case['verdict']
    assert record['log_layer']['verdict'] == case['verdict_log_layer']


def test_written_files_agree_with_each_other(written):
    out, case = written['out'], written['case']
    benchmark = json.loads((out / 'benchmark.json').read_text())['cases'][0]
    metrics = _read_csv(out / 'metrics.csv')
    profiles = _read_csv(out / 'profiles.csv')
    stations = _read_csv(out / 'stations.csv')

    assert len(profiles) == benchmark['cells']
    for row in stations:
        window = benchmark[row['station']]
        assert int(row['cells']) == window['cells']
        assert float(row['rms_relative']) == pytest.approx(window['rms_relative'], abs=1e-12)
        assert row['verdict'] == window['verdict']

    graded = [r for r in metrics if r['graded'] == 'true']
    assert len(graded) == 2                                   # one per station
    for row in graded:
        window = benchmark[row['station']]
        assert float(row['computed']) == pytest.approx(window['rms_relative'], abs=1e-12)
        assert row['verdict'] == window['verdict']
        assert float(row['tolerance']) == pytest.approx(TOLERANCE_LOG_REGION)

    cf_row = next(r for r in metrics if r['metric'] == 'Cf')
    assert float(cf_row['computed']) == pytest.approx(case['Cf'])
    assert float(cf_row['relative_error']) == pytest.approx(
        abs(case['Cf'] - load_kmm1987_reference()['Cf']) / load_kmm1987_reference()['Cf'])


def test_benchmark_json_is_self_describing(written):
    benchmark = written['benchmark']
    assert benchmark['schema'] == 'tensorfvm.case-fvm-channel-kmm180/1'
    assert benchmark['tolerance'] == pytest.approx(TOLERANCE_LOG_REGION)
    assert set(benchmark['grading_windows']) == {LEGACY_STATION, LOG_LAYER_STATION}
    assert benchmark['verdict_qualified'] is True
    # the headline must not read as a bare PASS
    assert 'neither may be quoted alone' in benchmark['headline']


# --------------------------------------------------------------------------- #
# Committed artifacts
# --------------------------------------------------------------------------- #

def test_committed_artifacts_pass_the_gate():
    """Artifacts, manifest sha256 and the catalog entry must agree."""
    module = _load_script()
    assert module.check_artifacts() == []


def test_committed_report_quotes_both_windows():
    report = (CASE_DIR / 'REPORT.md').read_text()
    for needle in ('legacy_full', 'log_layer', 'Reference substitution',
                   'Machine-readable artifacts'):
        assert needle in report, needle
    assert 'Overall verdict' not in report or 'Legacy full-range window' in report


def test_committed_manifest_pins_the_implementation_files():
    manifest = json.loads((CASE_DIR / 'manifest.json').read_text())
    for name in ('kmm180_artifacts.py', 'benchmark_channel_kmm1987.py'):
        assert any(path.endswith(name) for path in manifest['impl_files']), name
    assert set(manifest['sha256']) >= {'benchmark.json', 'metrics.csv',
                                       'profiles.csv', 'stations.csv'}
    for name in manifest['sha256']:
        assert (CASE_DIR / name).is_file()