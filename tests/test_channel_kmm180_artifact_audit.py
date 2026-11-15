"""Lightweight audit of CASE-FVM-CHANNEL-KMM180 artifacts.

This test verifies that the committed artifacts are internally consistent
and that the grading logic is correctly applied, without running any
heavy FVM solves.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def benchmark():
    path = REPO_ROOT / 'docs' / 'case-fvm-channel-kmm180' / 'benchmark.json'
    return json.loads(path.read_text())


@pytest.fixture(scope='module')
def case_pool():
    path = REPO_ROOT / 'docs' / 'case-fvm-channel-kmm180-pool' / 'case_pool_entry.json'
    return json.loads(path.read_text())


@pytest.fixture(scope='module')
def dns_provenance():
    path = REPO_ROOT / 'docs' / 'rans-turbulence-validation-matrix' / 'dns' / '0180' / 'provenance.json'
    return json.loads(path.read_text())


def _find_case(cases, model, n):
    """Find a case in the benchmark cases list."""
    for c in cases:
        if c.get('model') == model and c.get('n') == n:
            return c
    return None


def test_benchmark_has_both_windows(benchmark):
    """Every case must have both legacy_full and log_layer grades."""
    for case in benchmark.get('cases', []):
        assert 'legacy_full' in case, f'{case.get("case")} missing legacy_full'
        assert 'log_layer' in case, f'{case.get("case")} missing log_layer'
        assert 'rms_relative' in case['legacy_full']
        assert 'rms_relative' in case['log_layer']


def test_benchmark_verdicts_are_consistent(benchmark):
    """Verdicts must match the RMS vs tolerance comparison."""
    tolerance = benchmark.get('tolerance', 0.05)
    for case in benchmark.get('cases', []):
        legacy_rms = case['legacy_full']['rms_relative']
        log_rms = case['log_layer']['rms_relative']
        verdict = case.get('verdict')

        if verdict is not None:
            expected = 'PASS' if legacy_rms < tolerance else 'FAIL'
            assert verdict == expected, (
                f'{case.get("case")}: verdict {verdict} != expected {expected} '
                f'(RMS={legacy_rms:.4f}, tol={tolerance})')


def test_case_pool_entry_is_well_formed(case_pool):
    """The case pool entry must have required keys."""
    required_keys = ['criteria', 'admitted_models', 'reference']
    for key in required_keys:
        assert key in case_pool, f'Missing key: {key}'


def test_case_pool_reference_has_re_tau(case_pool):
    """The case pool reference must have Re_tau."""
    ref = case_pool.get('reference', {})
    assert 're_tau' in ref, 'Case pool reference missing re_tau'
    assert np.isfinite(ref['re_tau']), 'Case pool reference re_tau not finite'


def test_sst_n64_passes_both_windows(benchmark):
    """SST at n=64 must pass both windows (the best case)."""
    sst_64 = _find_case(benchmark.get('cases', []), 'sst', 64)
    assert sst_64 is not None, 'sst_n64 case not found'
    assert sst_64['legacy_full']['rms_relative'] < 0.05
    assert sst_64['log_layer']['rms_relative'] < 0.05


def test_kepsilon_fails_both_windows(benchmark):
    """k-epsilon must fail both windows (wall function model)."""
    ke_128 = _find_case(benchmark.get('cases', []), 'kepsilon', 128)
    assert ke_128 is not None, 'kepsilon_n128 case not found'
    assert ke_128['legacy_full']['rms_relative'] > 0.05
    assert ke_128['log_layer']['rms_relative'] > 0.05


def test_dns_provenance_has_required_fields(dns_provenance):
    """The DNS provenance must have all required fields."""
    required = ['Re_tau_reported', 'Cf', 'U_bulk_plus', 'u_tau', 'nu', 'half_width']
    for field in required:
        assert field in dns_provenance, f'Missing DNS provenance field: {field}'
        assert np.isfinite(dns_provenance[field]), f'DNS provenance {field} not finite'


def test_synthesis_report_exists():
    """The synthesis report must exist."""
    path = REPO_ROOT / 'docs' / 'case-fvm-channel-kmm180' / 'SYNTHESIS_REPORT.md'
    assert path.exists(), 'SYNTHESIS_REPORT.md not found'
    content = path.read_text()
    assert 'CASE-FVM-CHANNEL-KMM180' in content
    assert '综合证据合成报告' in content
