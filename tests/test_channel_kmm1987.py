"""Tests for CASE-FVM-CHANNEL-KMM180 (turbulent channel vs KMM1987 DNS)."""
import json
from pathlib import Path

import numpy as np
import pytest

from tensorfvm.rans_validation.channel import solve_channel
from tensorfvm.verification.kmm1987 import (
    CASE_TOLERANCE,
    GATED_MODEL,
    STATIONS_Y_PLUS,
    cf_deviation,
    bulk_velocity_deviation,
    grade_case,
    load_kmm1987_reference,
    log_law_fit,
    log_region_deviation,
    station_table,
    validate_kmm1987,
)

REPO = Path(__file__).resolve().parents[1]


class TestReference:
    def test_load_reference(self):
        dns = load_kmm1987_reference()
        assert dns['re_tau'] == pytest.approx(182.088, rel=1e-3)
        assert dns['Cf'] > 0 and dns['U_bulk_plus'] > 0
        assert len(dns['y_plus']) == len(dns['u_plus'])
        assert np.all(np.diff(dns['y_plus']) > 0)

    def test_viscous_sublayer_is_linear(self):
        dns = load_kmm1987_reference()
        mask = (dns['y_plus'] >= 1.0) & (dns['y_plus'] < 3.0)
        err = np.abs(dns['u_plus'][mask] - dns['y_plus'][mask]) / dns['y_plus'][mask]
        assert np.max(err) < 0.03


class TestMetrics:
    def test_perfect_match(self):
        dns = load_kmm1987_reference()
        assert log_region_deviation(dns['y_plus'], dns['u_plus'], dns)['rms_relative'] < 1e-10

    def test_constant_offset(self):
        dns = load_kmm1987_reference()
        dev = log_region_deviation(dns['y_plus'], 1.1 * dns['u_plus'], dns)
        assert dev['rms_relative'] == pytest.approx(0.10, rel=1e-6)

    def test_cf_and_bulk_deviation(self):
        dns = load_kmm1987_reference()
        assert cf_deviation(dns['Cf'], dns)['relative_error'] == pytest.approx(0.0)
        assert bulk_velocity_deviation(dns['U_bulk_plus'], dns)['relative_error'] == pytest.approx(0.0)

    def test_station_table_shape_and_identity(self):
        dns = load_kmm1987_reference()
        table = station_table(dns['y_plus'], dns['u_plus'], dns)
        assert len(table) == len(STATIONS_Y_PLUS)  # all stations below the centreline
        for row in table:
            assert row['y_plus'] >= 30.0
            # model side is a linear interpolant of the same rows, the
            # reference side a shape-preserving cubic: agreement to 1e-3 is the
            # interpolation consistency check, not an exact identity
            assert abs(row['rel_error']) < 1e-3
            assert row['abs_error'] < 5e-3

    def test_log_law_fit_reproduces_a_synthetic_log_law(self):
        y = np.linspace(30.0, 60.0, 40)
        u = np.log(y) / 0.41 + 5.0
        fit = log_law_fit(y, u, high_fraction=2.0)
        assert fit['kappa'] == pytest.approx(0.41, rel=1e-6)
        assert fit['B'] == pytest.approx(5.0, rel=1e-6)


class TestGrading:
    @pytest.fixture(scope='class')
    def dns(self):
        return load_kmm1987_reference()

    def test_graded_closure_passes_the_case_criterion(self, dns):
        result = solve_channel(GATED_MODEL, 128, re_tau=180.0)
        graded = grade_case(GATED_MODEL, result['y_plus'], result['u_plus'],
                            result['Cf'], result['bulk_u_plus'], dns=dns)
        assert graded['role'] == 'gated'
        assert graded['got']['log_region_rms'] < CASE_TOLERANCE
        assert graded['verdict'] == 'PASS'
        # the verdict is the predeclared metric, nothing else
        validation = validate_kmm1987(result['y_plus'], result['u_plus'],
                                      result['Cf'], result['bulk_u_plus'], dns=dns)
        assert graded['got']['log_region_rms'] == pytest.approx(
            validation['log_region']['rms_relative'])

    def test_negative_control_is_rejected(self, dns):
        """k-omega must fail the same criterion, else the metric is vacuous."""
        result = solve_channel('komega', 128, re_tau=180.0)
        graded = grade_case('komega', result['y_plus'], result['u_plus'],
                            result['Cf'], result['bulk_u_plus'], dns=dns)
        assert graded['role'] == 'control'
        assert graded['verdict'] == 'FAIL'
        assert graded['got']['log_region_rms'] > CASE_TOLERANCE

    def test_tolerance_is_honoured(self, dns):
        result = solve_channel(GATED_MODEL, 64, re_tau=180.0)
        strict = grade_case(GATED_MODEL, result['y_plus'], result['u_plus'],
                            result['Cf'], result['bulk_u_plus'], dns=dns,
                            tolerance=0.01)
        assert strict['verdict'] == 'FAIL'


class TestSuiteIntegration:
    def test_verification_suite_caller_runs_the_case(self):
        """The official verification suite drives the case (L3 integration)."""
        from tensorfvm.verification.cases import channel_kmm180
        case = channel_kmm180(n=64, re_tau=180.0)
        assert case['case'] == 'channel-kmm180'
        assert case['converged'] is True
        assert case['verdict'] == 'PASS'
        assert case['passed'] is True
        assert case['reference']['tolerance'] == CASE_TOLERANCE


class TestArtifacts:
    def test_delivered_artifact_set_is_complete_and_consistent(self):
        root = REPO / 'docs' / 'case-fvm-channel-kmm180'
        manifest = json.loads((root / 'manifest.json').read_text())
        assert manifest['case'] == 'CASE-FVM-CHANNEL-KMM180'
        assert manifest['verdict'] == 'PASS'
        assert manifest['controls_rejected'] is True
        assert manifest['criterion']['tolerance'] == CASE_TOLERANCE
        names = {a['path'] for a in manifest['artifacts']}
        assert {'benchmark.json', 'metrics.csv', 'stations.csv', 'profiles.csv',
                'REPORT.md'} <= names
        for artifact in manifest['artifacts']:
            path = root / artifact['path']
            assert path.is_file() and path.stat().st_size > 0
            import hashlib
            assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact['sha256']

        bench = json.loads((root / 'benchmark.json').read_text())
        published = bench['benchmark']
        assert published['verdict'] == 'PASS'
        assert published['got']['log_region_rms'] < published['tolerance']
        # ref vs got must both be present and the error must be the difference
        assert published['ref']['bulk_u_plus'] > 0
        assert published['got']['bulk_u_plus'] > 0
        assert published['rel_error']['bulk_u_plus'] == pytest.approx(
            abs(published['got']['bulk_u_plus'] - published['ref']['bulk_u_plus'])
            / published['ref']['bulk_u_plus'])
        assert len(bench['stations']) >= 5
        assert bench['controls_rejected'] is True
        assert all(c['verdict'] == 'FAIL' for c in bench['controls'])