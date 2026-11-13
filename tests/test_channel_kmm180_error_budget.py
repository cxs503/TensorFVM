"""Tests for the CASE-FVM-CHANNEL-KMM180 wall-normal error budget.

The tests split into three groups:

* **integrator unit tests** on synthetic data with a known answer (no solver),
  which pin the two cumulative integrals and the band bookkeeping;
* **solver-state tests** on one fast converged case per wall treatment, which pin
  the pre-declared checks C3 (reconstruction) and C4 (momentum identity);
* **cross-artifact tests** that re-derive a number already published by the
  anchor and require agreement with it, so the budget cannot drift away from the
  graded case without failing.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from tensorfvm.verification import kmm180_error_budget as budget
from tensorfvm.verification.kmm1987 import load_kmm1987_reference

ROOT = Path(__file__).resolve().parents[1]
PUBLISHED = ROOT / 'docs' / 'case-fvm-channel-kmm180' / 'benchmark.json'


@pytest.fixture(scope='module')
def dns():
    return load_kmm1987_reference()


@pytest.fixture(scope='module')
def sst_case(dns):
    return budget.audit('sst', 64, dns=dns)


@pytest.fixture(scope='module')
def kepsilon_case(dns):
    return budget.audit('kepsilon', 64, dns=dns)


# --------------------------------------------------------------------------- #
# Integrators on synthetic data
# --------------------------------------------------------------------------- #

def test_model_cumulative_is_linear_inside_a_cell():
    faces = np.array([0.0, 1.0, 3.0, 6.0])
    face_u = np.array([0.0, 2.0, 5.0, 9.0])       # cell shears 2, 1.5, 1.333
    shear = np.diff(face_u) / np.diff(faces)
    for fraction in (0.25, 0.5, 0.75):
        y = 1.0 + fraction * 2.0
        expected = face_u[1] + shear[1] * (y - faces[1])
        assert budget.model_cumulative_u_plus(y, faces, face_u) == pytest.approx(expected, rel=1e-14)


def test_model_cumulative_clamps_below_first_face():
    """A wall-function profile has nothing below its inner boundary."""
    faces = np.array([30.0, 31.0, 33.0])
    face_u = np.array([13.0, 13.2, 13.4])
    assert budget.model_cumulative_u_plus(5.0, faces, face_u) == pytest.approx(13.0)


def test_dns_cumulative_is_exact_for_piecewise_linear_shear():
    y_plus = np.array([0.0, 2.0, 5.0])
    shear = 2.0 * y_plus                          # integral 0 -> y is y^2
    for query in (0.0, 1.0, 1.5, 2.0, 3.7, 5.0):
        assert budget.dns_cumulative_u_plus(query, y_plus, shear) == pytest.approx(
            query ** 2, abs=1e-12)


def test_dns_implied_nut_matches_shear_of_a_known_identity():
    re_tau = 100.0
    y_plus = np.array([1.0, 10.0, 30.0])
    nut_plus = np.array([0.5, 2.0, 6.0])
    shear = (1.0 - y_plus / re_tau) / (1.0 + nut_plus)
    assert budget.dns_implied_nut_plus(y_plus, shear, re_tau) == pytest.approx(nut_plus)


def test_dns_implied_nut_is_nan_where_the_identity_is_degenerate():
    y_plus = np.array([0.0, 1.0])
    shear = np.array([1.0, 0.0])                  # zero shear: nu_t undetermined
    implied = budget.dns_implied_nut_plus(y_plus, shear, 180.0)
    assert np.isnan(implied[1])
    assert np.isfinite(implied[0])


def test_bands_are_contiguous_from_the_inner_boundary():
    bands = budget.bands_for(36.42, inner=0.0)
    assert bands[0][0] == 0.0
    assert bands[-1][1] == pytest.approx(36.42)
    for (_, high), (low, _) in zip(bands, bands[1:]):
        assert low == pytest.approx(high)
    wall_function = budget.bands_for(36.42, inner=30.0)
    assert wall_function == [(30.0, 36.42)]


# --------------------------------------------------------------------------- #
# Solver state: pre-declared checks C3 / C4
# --------------------------------------------------------------------------- #

def test_reconstruction_and_identity_checks_pass(sst_case):
    assert sst_case['checks']['reconstruction'] <= budget.RECONSTRUCTION_TOLERANCE
    assert sst_case['checks']['momentum_identity'] <= budget.IDENTITY_TOLERANCE


def test_band_sum_equals_the_total(sst_case):
    band_total = sum(row['delta'] for row in sst_case['bands'])
    wall = sst_case['wall_law_offset'] or 0.0
    assert band_total + wall == pytest.approx(sst_case['total_residual'], abs=1e-12)
    assert sst_case['below_log_delta'] + sst_case['inside_log_delta'] == pytest.approx(
        band_total, abs=1e-12)


def test_budget_closes_against_the_point_value(sst_case, kepsilon_case):
    for case in (sst_case, kepsilon_case):
        assert case['settle_residual'] <= budget.SETTLE_TOLERANCE


def test_wall_function_case_quotes_the_prescribed_law(kepsilon_case):
    assert kepsilon_case['inner_boundary_y_plus'] >= budget.LOW_Y_PLUS - 1e-9
    assert kepsilon_case['wall_law_offset'] is not None
    assert kepsilon_case['wall_law_offset'] < 0.0            # wall law under-predicts u+
    assert kepsilon_case['bands'][0]['low'] >= budget.LOW_Y_PLUS - 1e-9


def test_wall_resolving_case_has_no_wall_law_offset(sst_case):
    assert sst_case['wall_law_offset'] is None
    assert sst_case['bands'][0]['low'] == 0.0


def test_origin_label_follows_the_pre_declared_rule(sst_case):
    for case in (sst_case,):
        expected = ('buffer_origin' if case['share_below_log'] >= budget.DOMINANCE_SHARE
                    else 'log_layer_origin')
        assert case['origin'] == expected


def test_nut_station_outside_the_profile_is_not_fabricated(kepsilon_case, sst_case):
    assert kepsilon_case['nut_stations']['15']['model'] is None
    assert kepsilon_case['nut_stations']['15']['ratio'] is None
    assert sst_case['nut_stations']['15']['model'] is not None


# --------------------------------------------------------------------------- #
# Reference data integrity (C2) and cross-artifact agreement
# --------------------------------------------------------------------------- #

def test_dns_reference_is_internally_consistent(dns):
    assert budget.dns_integral_residual(dns) <= budget.DNS_INTEGRAL_TOLERANCE


def test_window_rms_matches_the_published_reference_substitution(sst_case):
    """The window RMS must equal the number the anchor already published.

    ``docs/case-fvm-channel-kmm180/benchmark.json`` stores the log-layer RMS of
    every quoted case *re-solved at the reference Re_tau*; this module computes
    the same quantity through a different code path (cell-centre profile vs the
    graded validation call), so agreement is a real cross-check.
    """
    published = json.loads(PUBLISHED.read_text())['reference_substitution']['cases']
    row = next(item for item in published
               if item['model'] == sst_case['model'] and item['n'] == sst_case['cells'])
    assert sst_case['rms_relative'] == pytest.approx(row['rms_log_layer_reference'], rel=1e-9)
    expected_verdict = ('PASS' if row['rms_log_layer_reference'] < 0.05 else 'FAIL')
    assert sst_case['verdict'] == expected_verdict


def test_committed_artifacts_match_a_fresh_run():
    """The committed JSON must be reproducible: no hand-edited numbers."""
    committed = json.loads(
        (ROOT / 'docs' / 'case-fvm-channel-kmm180-error-budget' / 'error_budget.json').read_text())
    fresh = budget.audit('sst', 64)
    row = next(case for case in committed['cases']
               if case['model'] == 'sst' and case['cells'] == 64)
    for key in ('rms_relative', 'total_residual', 'below_log_delta', 'inside_log_delta',
                'share_below_log', 'window_cells'):
        assert row[key] == pytest.approx(fresh[key], rel=1e-9), key
    assert committed['checks']['reference_integrand_pass'] is True