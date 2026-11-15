"""Tests for the Spalart-Allmaras row integration into the main KMM180 benchmark.

The SA model was previously tested in isolation (docs/case-fvm-channel-kmm180-sa/)
but was not included in the main benchmark.json matrix. This test verifies that:

1. The SA cases are present in benchmark.json with correct structure;
2. The SA numbers match a fresh solve (no hand-edited numbers);
3. The SA verdict is consistent with the published SA directory artifacts;
4. The headline in benchmark.json mentions the SA model.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from tensorfvm.verification.kmm180_sa import solve_sa_channel
from tensorfvm.verification.kmm1987 import load_kmm1987_reference
from tensorfvm.verification.kmm180_windows import window_deviation, LOG_LAYER_FRACTION

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / 'docs' / 'case-fvm-channel-kmm180' / 'benchmark.json'
SA_SUMMARY = ROOT / 'docs' / 'case-fvm-channel-kmm180-sa' / 'summary.json'


@pytest.fixture(scope='module')
def benchmark():
    return json.loads(BENCHMARK.read_text())


@pytest.fixture(scope='module')
def dns():
    return load_kmm1987_reference()


def test_sa_cases_present_in_benchmark(benchmark):
    """The main benchmark must contain SA rows for all three grids."""
    sa_cases = [c for c in benchmark['cases'] if c['model'] == 'sa']
    assert len(sa_cases) == 3, f'Expected 3 SA cases, found {len(sa_cases)}'
    ns = sorted(c['n'] for c in sa_cases)
    assert ns == [64, 128, 256], f'Expected grids [64, 128, 256], got {ns}'


def test_sa_numbers_match_fresh_solve(benchmark, dns):
    """The committed SA numbers must be reproducible: no hand-edited numbers."""
    re_tau = float(dns['re_tau'])
    for case in benchmark['cases']:
        if case['model'] != 'sa':
            continue
        n = case['n']
        r = solve_sa_channel(n, re_tau=180.0, first_cell_y_plus=0.3)
        
        # Check Cf
        assert case['Cf'] == pytest.approx(r['Cf'], rel=1e-9), f'Cf mismatch for sa_n{n}'
        
        # Check bulk_u_plus
        assert case['bulk_u_plus'] == pytest.approx(r['bulk_u_plus'], rel=1e-9), f'bulk_u+ mismatch for sa_n{n}'
        
        # Check grading windows
        full = window_deviation(r['y_plus'], r['u_plus'], dns, high=float(dns['y_plus'].max()))
        log = window_deviation(r['y_plus'], r['u_plus'], dns, high=LOG_LAYER_FRACTION * re_tau)
        
        assert case['legacy_full']['rms_relative'] == pytest.approx(
            full['rms_relative'], rel=1e-9), f'legacy_full RMS mismatch for sa_n{n}'
        assert case['log_layer']['rms_relative'] == pytest.approx(
            log['rms_relative'], rel=1e-9), f'log_layer RMS mismatch for sa_n{n}'


def test_sa_verdict_consistent_with_sa_directory(benchmark):
    """The SA verdict in the main benchmark must match the SA directory artifacts."""
    if not SA_SUMMARY.exists():
        pytest.skip('SA summary artifacts absent')
    
    sa_summary = json.loads(SA_SUMMARY.read_text())
    
    for case in benchmark['cases']:
        if case['model'] != 'sa':
            continue
        n = case['n']
        sa_case_key = f'sa_n{n}'
        if sa_case_key not in sa_summary['cases']:
            continue
        
        sa_case = sa_summary['cases'][sa_case_key]
        assert case['log_layer']['rms_relative'] == pytest.approx(
            sa_case['log_layer_rms'], rel=1e-6), f'log_layer RMS mismatch for {sa_case_key}'
        assert case['legacy_full']['rms_relative'] == pytest.approx(
            sa_case['log_region_rms'], rel=1e-6), f'legacy_full RMS mismatch for {sa_case_key}'


def test_sa_all_pass(benchmark):
    """All SA cases must pass both windows (the SA model is calibrated on the log law)."""
    sa_cases = [c for c in benchmark['cases'] if c['model'] == 'sa']
    for case in sa_cases:
        assert case['legacy_full']['verdict'] == 'PASS', f'sa_n{case["n"]} legacy_full should PASS'
        assert case['log_layer']['verdict'] == 'PASS', f'sa_n{case["n"]} log_layer should PASS'
        assert case['overall_verdict'] == 'PASS', f'sa_n{case["n"]} overall should PASS'


def test_benchmark_headline_mentions_sa(benchmark):
    """The headline must mention the SA model's PASS verdict."""
    assert 'sa' in benchmark['headline'].lower(), 'Headline must mention SA model'
    assert 'PASS' in benchmark['headline'], 'Headline must mention PASS verdict'


def test_sa_cf_error_below_two_equation_models(benchmark):
    """SA Cf error must be below the two-equation models (SA is calibrated on the log law)."""
    sa_cases = [c for c in benchmark['cases'] if c['model'] == 'sa']
    sst_cases = [c for c in benchmark['cases'] if c['model'] == 'sst']
    
    for sa_case in sa_cases:
        n = sa_case['n']
        sst_case = next((c for c in sst_cases if c['n'] == n), None)
        if sst_case is not None:
            assert sa_case['cf_relative_error'] < sst_case['cf_relative_error'], \
                f'sa_n{n} Cf error ({sa_case["cf_relative_error"]:.4f}) should be < sst_n{n} ({sst_case["cf_relative_error"]:.4f})'


def test_sa_log_layer_rms_below_tolerance(benchmark):
    """SA log layer RMS must be well below the 5% tolerance."""
    sa_cases = [c for c in benchmark['cases'] if c['model'] == 'sa']
    for case in sa_cases:
        assert case['log_layer']['rms_relative'] < 0.05, \
            f'sa_n{case["n"]} log_layer RMS ({case["log_layer"]["rms_relative"]:.4f}) should be < 0.05'
        # SA should be well below tolerance (order of magnitude better than two-equation models)
        assert case['log_layer']['rms_relative'] < 0.01, \
            f'sa_n{case["n"]} log_layer RMS ({case["log_layer"]["rms_relative"]:.4f}) should be < 0.01'
