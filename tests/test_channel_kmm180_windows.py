"""Window-sensitivity tests for the CASE-FVM-CHANNEL-KMM180 adjudication.

The claim under test is the repo's own: SST reproduces the DNS ``u+(y+)``
within 5 % "in the log region (y+ > 30)".  ``tensorfvm.verification.kmm1987``
grades that claim over ``30 <= y+ <= centreline``; these tests pin down what
that window choice does to the verdict, and register the honest numbers.

Two classes of check:

* synthetic -- the metric and the residual decomposition must recover a known
  constant offset / known slope defect from manufactured profiles;
* registration -- the real solver's numbers, so a silent regression (or a
  quiet change of the grading window) fails the suite instead of passing it.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from tensorfvm.rans_validation.channel import solve_channel
from tensorfvm.verification.kmm180_windows import (
    KAPPA_REF,
    LOG_LAYER_FRACTION,
    LOW_Y_PLUS,
    audit,
    decompose_residual,
    estimate_dilution,
    grade_window,
    log_law_reference_deviation,
    run,
    summarise,
    window_deviation,
)
from tensorfvm.verification.kmm1987 import load_kmm1987_reference, TOLERANCE_LOG_REGION


@pytest.fixture(scope='module')
def dns():
    return load_kmm1987_reference()


# --------------------------------------------------------------------------- #
# Synthetic: the metric must recover a manufactured defect
# --------------------------------------------------------------------------- #

def test_window_deviation_recovers_a_constant_offset(dns):
    model = dns['u_plus'] * 1.05
    dev = window_deviation(dns['y_plus'], model, dns, high=float(np.max(dns['y_plus'])))
    assert dev['rms_relative'] == pytest.approx(0.05, rel=1e-10)
    assert dev['mean_relative'] == pytest.approx(0.05, rel=1e-10)


def test_window_deviation_counts_cells_and_rejects_narrow_windows(dns):
    dev = window_deviation(dns['y_plus'], dns['u_plus'], dns, high=40.0)
    expected = int(np.count_nonzero((dns['y_plus'] >= LOW_Y_PLUS) & (dns['y_plus'] <= 40.0)))
    assert dev['cells'] == expected
    with pytest.raises(ValueError):
        window_deviation(dns['y_plus'], dns['u_plus'], dns, high=LOW_Y_PLUS * 0.5)


def test_decompose_identifies_a_pure_offset(dns):
    model = dns['u_plus'] + 0.5
    dec = decompose_residual(dns['y_plus'], model, dns, high=LOG_LAYER_FRACTION * dns['re_tau'])
    assert dec['residual_mean'] == pytest.approx(0.5, abs=1e-12)
    assert dec['slope_ratio'] == pytest.approx(1.0, abs=1e-9)
    assert dec['offset_share'] == pytest.approx(1.0, abs=1e-9)
    assert dec['kappa_local_model'] == pytest.approx(dec['kappa_local_dns'], rel=1e-9)


def test_decompose_identifies_a_slope_defect(dns):
    """A pure log-slope error must show up as slope_ratio != 1 and slope_a = 0.3."""
    model = dns['u_plus'] + 0.3 * np.log(dns['y_plus'] / LOW_Y_PLUS)
    dec = decompose_residual(dns['y_plus'], model, dns, high=LOG_LAYER_FRACTION * dns['re_tau'])
    assert dec['slope_a'] == pytest.approx(0.3, rel=1e-6)
    assert dec['slope_ratio'] > 1.0
    assert dec['residual_mean'] > 0.0


def test_dilution_prediction_is_a_ratio_greater_than_one(dns):
    est = estimate_dilution(-0.9, mean_u_log=14.2, mean_u_full=16.3)
    assert est['predicted_ratio'] == pytest.approx(16.3 / 14.2, rel=1e-12)
    assert est['predicted_log_rms'] > est['predicted_full_rms']
    assert est['predicted_log_rms'] == pytest.approx(0.9 / 14.2, rel=1e-12)


def test_grade_window_marks_five_percent_boundary(dns):
    base = dict(mean_relative=0.0, mean_offset=0.0, mean_u_plus=14.0,
                cells=10, high=36.0)
    assert grade_window(dict(base, rms_relative=0.049))['verdict'] == 'PASS'
    assert grade_window(dict(base, rms_relative=0.051))['verdict'] == 'FAIL'


def test_dns_itself_is_not_a_perfect_log_law(dns):
    """Context for the 5 % target: the DNS sits ~3 % off the classic log law."""
    ref = log_law_reference_deviation(dns, high=LOG_LAYER_FRACTION * dns['re_tau'])
    assert ref['kappa'] == KAPPA_REF
    assert 0.0 < ref['rms_relative'] < 0.10
    assert ref['mean_offset'] > 0.0


# --------------------------------------------------------------------------- #
# Registration: the real solver
# --------------------------------------------------------------------------- #

@pytest.fixture(scope='module')
def sst_fine(dns):
    return audit('sst', 256, dns=dns)


@pytest.fixture(scope='module')
def sst_coarse(dns):
    return audit('sst', 64, dns=dns)


def test_sst_verdict_flips_between_the_two_windows(sst_fine):
    """The legacy full-range window passes while the log-layer window fails."""
    assert sst_fine['converged']
    assert sst_fine['windows']['full']['rms_relative'] < TOLERANCE_LOG_REGION
    assert sst_fine['log_layer']['rms_relative'] > TOLERANCE_LOG_REGION
    assert sst_fine['verdict_full'] == 'PASS'
    assert sst_fine['verdict_log_layer'] == 'FAIL'
    # the two windows differ by the mean u+ they divide by, measured as a ratio
    assert 0.4 < sst_fine['divergence_vs_full']['ratio'] < 0.7


def test_sst_log_layer_deviation_is_registered(sst_fine):
    """Grid-converged SST number for the log layer: (6.12 +/- 0.3) %."""
    assert 0.058 < sst_fine['log_layer']['rms_relative'] < 0.065
    assert 0.033 < sst_fine['windows']['full']['rms_relative'] < 0.038


def test_sst_residual_is_offset_dominated(sst_fine):
    dec = sst_fine['decomposition']
    assert dec['residual_mean'] < 0.0                      # model low, not high
    assert -1.05 < dec['residual_mean'] < -0.65            # wall units
    assert dec['offset_share'] > 0.85                      # a shift, not a tilt
    assert 1.0 < dec['slope_ratio'] < 1.25                 # mildly steeper locally


def test_under_resolution_can_pass_the_log_layer(sst_coarse, sst_fine):
    """Coarse grid (64 cells) slips under 5 % where the resolved grid cannot."""
    assert sst_coarse['log_layer']['rms_relative'] < TOLERANCE_LOG_REGION
    assert sst_fine['log_layer']['rms_relative'] > sst_coarse['log_layer']['rms_relative']


@pytest.mark.parametrize('model', ['komega', 'kepsilon'])
def test_other_models_fail_both_windows(model, dns):
    row = audit(model, 128, dns=dns)
    assert row['converged']
    assert row['verdict_full'] == 'FAIL'
    assert row['verdict_log_layer'] == 'FAIL'
    assert row['log_layer']['rms_relative'] > 0.05


def test_wall_functioned_kepsilon_has_a_shallow_slope(dns):
    row = audit('kepsilon', 128, dns=dns)
    assert 30.0 <= row['first_cell_y_plus'] < 31.0   # wall-function domain starts at y+=30
    assert row['decomposition']['slope_ratio'] < 0.8


def test_report_is_json_serialisable_and_summarised(sst_fine):
    blob = json.dumps(sst_fine)
    assert json.loads(blob)['anchor' if 'anchor' in sst_fine else 'model'] == sst_fine['model']
    text = summarise(dict(cases=[sst_fine]))
    assert text.splitlines()[0].startswith('model\t')
    assert 'sst' in text.splitlines()[1]


# `run` is exercised through the CLI; assert its schema without solving twice.
def test_run_schema_defaults():
    import inspect
    signature = inspect.signature(run)
    assert list(signature.parameters) == ['models', 'grids']

def test_validate_reports_both_windows_and_overall_verdict(dns):
    """`validate_kmm1987` must not let the diluted window decide alone."""
    from tensorfvm.verification.kmm1987 import validate_kmm1987

    result = solve_channel('sst', 128, re_tau=dns['re_tau'])
    validation = validate_kmm1987(result['y_plus'], result['u_plus'], result['Cf'],
                                  result['bulk_u_plus'], dns=dns)
    assert validation['verdict'] == 'PASS'              # legacy full-range window
    assert validation['verdict_log_layer'] == 'FAIL'    # physical log layer
    assert validation['overall_verdict'] == 'FAIL'      # honest combined verdict
    assert validation['log_region_narrow']['high'] == pytest.approx(
        LOG_LAYER_FRACTION * dns['re_tau'])
