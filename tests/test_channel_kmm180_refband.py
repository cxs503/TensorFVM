"""Contract tests for the Re_tau ~ 180 reference band of CASE-FVM-CHANNEL-KMM180."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tensorfvm.verification import kmm180_refband as refband

ROOT = refband.ROOT
DNS = refband.DNS_ROOT


@pytest.fixture(scope='module')
def references():
    return refband.load_references()


# --------------------------------------------------------------------------- #
# Imported data and provenance
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('key,re_tau', [('mkm', 178.12), ('hj', 180.0)])
def test_imported_provenance_pins_the_committed_csv(key, re_tau):
    sub, _, _ = refband.REFERENCE_SETS[key]
    provenance = json.loads((DNS / sub / 'provenance.json').read_text())
    assert provenance['Re_tau_reported'] == pytest.approx(re_tau)
    raw = DNS / 'cache' / provenance['raw_file']
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == provenance['raw_sha256']
    committed = (DNS / sub / 'mean.csv').read_bytes()
    assert hashlib.sha256(committed).hexdigest() == provenance['csv_sha256']


def test_imported_profiles_are_wall_resolved_and_start_at_the_wall(references):
    for key in ('mkm', 'hj'):
        profile = references[key]
        assert np.all(np.diff(profile['y_plus']) >= 0)
        assert profile['y_plus'][0] == pytest.approx(0.0, abs=1e-12)
        assert profile['u_plus'][0] == pytest.approx(0.0, abs=1e-5)


def test_band_brackets_the_anchor_nominal_reynolds_number(references):
    re_taus = [reference['re_tau'] for reference in references.values()]
    assert min(re_taus) < refband.RE_TAU_NOMINAL < max(re_taus)
    assert len(set(round(value, 3) for value in re_taus)) == 3


# --------------------------------------------------------------------------- #
# Band and window
# --------------------------------------------------------------------------- #
def test_band_envelope_contains_every_reference(references):
    band = refband.build_band(references)
    stack = np.vstack([band['u_plus'][key] for key in refband.BAND_ORDER])
    assert np.allclose(band['lo'], stack.min(axis=0))
    assert np.allclose(band['hi'], stack.max(axis=0))
    assert band['relative_half_width'].shape == band['y_plus'].shape
    assert np.all(band['relative_half_width'] >= 0)


def test_window_is_the_log_layer_of_the_case_not_of_a_reference(references):
    window = refband.band_window(references)
    # The upper bound is fixed by the case (0.2 * Re_tau = 0.2 * 180) so that
    # adding or dropping a reference cannot move the graded interval.
    assert window['high'] == pytest.approx(0.2 * refband.RE_TAU_NOMINAL)
    assert window['low'] == pytest.approx(refband.LOW_Y_PLUS)
    assert window['cells'] >= 3


def test_reference_substitution_uncertainty_is_smaller_than_the_tolerance(references):
    """The band must resolve model error, i.e. be far below the 5 % tolerance."""
    window = refband.band_window(references)
    assert window['rms_relative_half_width'] < 0.2 * refband.TOLERANCE
    assert window['max_relative_half_width'] < refband.TOLERANCE
    for pair, value in window['pairwise_rms'].items():
        assert value['rms'] < 0.2 * refband.TOLERANCE, pair


def test_window_shrinks_to_three_cells_at_the_coarsest_model_grid(references):
    """n=64 carries only three cells in the log layer -- must be disclosed."""
    cases = refband.load_case_profiles(ROOT / 'docs' / 'case-fvm-channel-kmm180' / 'profiles.csv')
    graded = refband.grade_profile(cases['sst_n64']['y_plus'], cases['sst_n64']['u_plus'],
                                  references)
    assert graded['cells'] == 3
    assert graded['sparse_window'] is True


# --------------------------------------------------------------------------- #
# Grading semantics
# --------------------------------------------------------------------------- #
def test_self_grading_a_reference_against_the_band_has_zero_error(references):
    from scipy.interpolate import PchipInterpolator

    # The HJ2006 profile is coarse (49 points) and carries a single point in
    # the log-layer window, so resample it onto a grid that fills the window
    # before self-grading.
    profile = references['hj']
    grid = np.geomspace(1.0, float(profile['y_plus'].max()), 200)
    graded = refband.grade_profile(
        grid, PchipInterpolator(profile['y_plus'], profile['u_plus'])(grid), references)
    assert graded['per_reference']['hj']['rms'] == pytest.approx(0.0, abs=1e-6)
    assert graded['verdict'] is True
    # Band membership is only guaranteed for the envelope extremes: a reference
    # sitting inside the envelope is by construction *not* outside it, but a
    # reference that *is* an extreme is on the boundary, so only sanity-check.
    assert 0.0 <= graded['inside_band_fraction'] <= 1.0


def test_ten_percent_offset_fails_every_reference(references):
    profile = references['lm']
    graded = refband.grade_profile(profile['y_plus'], 1.1 * profile['u_plus'], references)
    assert graded['verdict'] is False
    assert set(graded['per_reference_verdict'].values()) == {False}
    assert graded['worst_rms'] > 0.09
    assert graded['verdict_robust'] is True


def test_verdict_is_the_worst_case_over_the_band(references):
    cases = refband.load_case_profiles(ROOT / 'docs' / 'case-fvm-channel-kmm180' / 'profiles.csv')
    case = cases['sst_n128']
    graded = refband.grade_profile(case['y_plus'], case['u_plus'], references)
    per_reference_rms = [value['rms'] for value in graded['per_reference'].values()]
    assert graded['worst_rms'] == pytest.approx(max(per_reference_rms))
    assert graded['worst_max'] >= graded['worst_rms']
    # The committed SST n=128 profile sits just outside the 5 % tolerance.
    assert graded['verdict'] is False
    assert 0.05 < graded['worst_rms'] < 0.07


def test_committed_profile_grades_reproduce_the_anchor_single_proxy_numbers(references):
    """Grading the same profile against the LM2015 proxy must match the anchor.

    ``docs/case-fvm-channel-kmm180`` publishes 4.6 % (n=64) and 5.7 % (n=128)
    for the SST log-layer window; the band audit must not silently change the
    definition of the metric.
    """
    cases = refband.load_case_profiles(ROOT / 'docs' / 'case-fvm-channel-kmm180' / 'profiles.csv')
    for name, expected in (('sst_n64', 0.046), ('sst_n128', 0.057)):
        case = cases[name]
        graded = refband.grade_profile(case['y_plus'], case['u_plus'], references)
        assert graded['per_reference']['lm']['rms'] == pytest.approx(expected, abs=0.004), name


# --------------------------------------------------------------------------- #
# Audit driver
# --------------------------------------------------------------------------- #
def test_audit_writes_artifacts(tmp_path):
    summary = refband.audit(out_dir=tmp_path)
    assert (tmp_path / 'summary.json').exists()
    assert (tmp_path / 'REPORT.md').exists()
    assert (tmp_path / 'fig_reference_band.png').stat().st_size > 10_000
    assert (tmp_path / 'fig_reference_deviation.png').stat().st_size > 10_000
    assert summary['schema'] == refband.SCHEMA
    assert summary['anchor'] == 'CASE-FVM-CHANNEL-KMM180'
    assert summary['window']['rms_relative_half_width'] < refband.TOLERANCE
    assert summary['all_pass'] is False  # k-omega / k-epsilon stay outside 5 %
    assert set(summary['cases']) == {
        'sst_n64', 'sst_n128', 'sst_n256',
        'komega_n64', 'komega_n128', 'komega_n256',
        'kepsilon_n64', 'kepsilon_n128', 'kepsilon_n256'}
    # Every case record must carry one graded number per reference in the band.
    for record in summary['cases'].values():
        assert set(record['per_reference']) == set(refband.BAND_ORDER)
        assert record['worst_rms'] >= max(v['rms'] for v in record['per_reference'].values()) - 1e-15