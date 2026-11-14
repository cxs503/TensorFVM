"""Tests for the CASE-FVM-CHANNEL-KMM180 measure (norm) audit.

Four layers, on purpose:

1. **Measure identities** -- the weights are a quadrature rule, so they must
   sum to one, reproduce exact cell volumes on a solver mesh and reduce to
   ``1/N`` on a uniform grid.
2. **Analytic grading** -- on synthetic profiles where every norm has a value
   that can be written down by hand (constant offset, exact match, linear
   DNS), the code must return exactly that value.  These are the tests that
   would catch a norm silently swapped for another.
3. **Contract / identities of the audit** -- the unweighted norm must reproduce
   the *published* graded number for a real case bit-for-bit, otherwise this
   module is not auditing the object it claims to audit.
4. **Guard rails** -- unreachable mesh clustering and too-thin windows raise
   instead of returning a meaningless number.
"""
import numpy as np
import pytest

from tensorfvm.rans_validation.channel import solve_channel
from tensorfvm.verification import kmm180_metric_norm as mn
from tensorfvm.verification.kmm1987 import load_kmm1987_reference


def synthetic_dns(y_plus, u_plus):
    """Minimal DNS reference with the keys ``window_metric`` reads."""
    return dict(y_plus=np.asarray(y_plus, dtype=float),
                u_plus=np.asarray(u_plus, dtype=float),
                re_tau=float(np.max(y_plus)))


# --------------------------------------------------------------------------- #
# 1. The measure is a quadrature rule
# --------------------------------------------------------------------------- #

def test_trapezoid_weights_uniform_grid_is_one_over_n_minus_one():
    """A trapezoidal rule over ``N`` samples spans ``N-1`` intervals.

    The interior nodes therefore carry ``1/(N-1)`` each and the two end nodes
    half of it -- *not* ``1/N``.  That half-cell deficit at the ends is exactly
    why the exact control-volume path (:func:`window_volumes`) exists and is
    used whenever the mesh faces are known.
    """
    w = mn.trapezoid_weights(np.linspace(0.0, 1.0, 9))
    assert np.allclose(w[1:-1], 1.0 / 8.0, rtol=0, atol=1e-15)
    assert np.allclose([w[0], w[-1]], 0.5 / 8.0, rtol=0, atol=1e-15)
    assert w.sum() == pytest.approx(1.0, abs=1e-15)


def test_window_volumes_on_a_uniform_mesh_are_equal():
    """The exact finite-volume measure gives every equal cell an equal vote.

    This is the property the trapezoidal fallback cannot have, and it is what
    makes the weighted norm a measure of wall-normal volume rather than of
    node placement.
    """
    faces = np.linspace(0.0, 40.0, 41)
    y = 0.5 * (faces[:-1] + faces[1:])
    w = mn.window_volumes(faces, y, low=30.0, high=36.0)
    assert np.allclose(w, 1.0 / 6.0, rtol=0, atol=1e-15)


def test_trapezoid_weights_are_positive_non_decreasing_with_spacing():
    y = np.array([0.0, 0.1, 0.5, 2.0])
    w = mn.trapezoid_weights(y)
    assert np.all(w > 0)
    assert w.sum() == pytest.approx(1.0, abs=1e-15)
    # the cell spanning [0.1, 2.0] carries more measure than the first stub
    assert w[2] > w[1] > w[0]


def test_trapezoid_weights_match_solver_cell_volumes():
    """Weights built from cell centres approximate the true control volumes.

    The two agree to first order at the window ends and to O(dy^2) inside, so
    a 5 % band on an ``n = 64`` clustered mesh is a meaningful assertion that
    the measure is the wall-normal volume, not something else.
    """
    solution = solve_channel('kepsilon', n=64, re_tau=180.0)
    y_plus = np.asarray(solution['y_plus'], dtype=float)
    mask = (y_plus >= mn.LOW_Y_PLUS) & (y_plus <= y_plus.max())
    volume = np.asarray(solution['volume'], dtype=float)[mask]
    weights = mn.trapezoid_weights(y_plus[mask])
    reference = volume / volume.sum()
    assert np.max(np.abs(weights - reference)) < 0.05


def test_window_mask_raises_on_thin_window():
    with pytest.raises(ValueError, match='fewer than'):
        mn.window_mask([10.0, 20.0, 40.0], low=30.0, high=36.0)


@pytest.mark.parametrize('bad', [
    [0.0, 0.0, 1.0],          # not strictly increasing
    [0.0, -1.0, 2.0],         # decreasing
    [0.0, np.nan, 1.0],       # non-finite
    [1.0],                    # too few samples
])
def test_trapezoid_weights_reject_invalid_grids(bad):
    with pytest.raises(ValueError):
        mn.trapezoid_weights(bad)


# --------------------------------------------------------------------------- #
# 2. Analytic grading
# --------------------------------------------------------------------------- #

def test_exact_match_is_zero_under_every_norm():
    y = np.linspace(30.0, 36.4, 7)
    dns = synthetic_dns(y, 2.5 * np.log(y) + 5.5)
    block = mn.window_metric(y, dns['u_plus'].copy(), dns, high=36.4)
    assert all(value == pytest.approx(0.0, abs=1e-15)
               for value in block['values'].values())
    assert block['unanimous']
    assert set(block['verdicts'].values()) == {'PASS'}


def test_constant_offset_gives_the_analytic_value_under_all_norms():
    """Flat DNS + constant offset: every norm equals ``delta / u_DNS``.

    With a flat reference the relative residual is uniform, so an unweighted
    and a weighted RMS must coincide -- this is the test that pins the
    weighting to the *residual*, not to the profile.
    """
    y = np.linspace(30.0, 36.4, 9)
    dns = synthetic_dns(y, np.full_like(y, 14.0))
    delta = 0.7
    block = mn.window_metric(y, dns['u_plus'] + delta, dns, high=36.4)
    expected = delta / 14.0
    assert block['values']['rms_rel_unweighted'] == pytest.approx(expected, rel=1e-12)
    assert block['values']['rms_rel_weighted'] == pytest.approx(expected, rel=1e-12)
    assert block['values']['mean_rel_weighted'] == pytest.approx(expected, rel=1e-12)
    assert block['values']['integral_rel'] == pytest.approx(expected, rel=1e-12)
    assert block['implied_offset_u_plus'] == pytest.approx(delta, rel=1e-12)


def test_weighted_rms_differs_from_unweighted_on_a_clustered_mesh():
    """The two measures are genuinely different functionals.

    Clustered samples, a residual that is large where the cells are dense:
    the unweighted RMS must exceed the weighted one, and the weighted value
    must equal the hand-computed trapezoidal sum.
    """
    y = np.array([30.0, 30.2, 30.5, 32.0, 36.4])
    dns = synthetic_dns(y, np.full_like(y, 14.0))
    residual = np.array([1.0, 0.9, 0.5, 0.1, 0.0])   # wall units
    block = mn.window_metric(y, dns['u_plus'] + residual, dns, high=36.4)

    gaps = np.diff(y)
    w = np.array([0.5 * gaps[0], 0.5 * (gaps[0] + gaps[1]),
                  0.5 * (gaps[1] + gaps[2]), 0.5 * (gaps[2] + gaps[3]),
                  0.5 * gaps[3]])
    w = w / w.sum()
    expected_weighted = np.sqrt(np.sum(w * (residual / 14.0) ** 2))
    expected_unweighted = np.sqrt(np.mean((residual / 14.0) ** 2))

    assert block['values']['rms_rel_weighted'] == pytest.approx(expected_weighted, rel=1e-12)
    assert block['values']['rms_rel_unweighted'] == pytest.approx(expected_unweighted, rel=1e-12)
    assert expected_unweighted > 2.0 * expected_weighted   # not a rounding-level gap


def test_integral_norm_is_insensitive_to_cell_placement():
    """Refining the sampling of the same profile must not move ``integral_rel``.

    This is the property that makes the integral norm the placement-robust
    companion to the RMS: for a smooth residual the trapezoidal integral
    converges, while the point RMS keeps sampling the residual at new points.
    """
    def residual(y):
        return 0.8 * np.exp(-(y - 30.0) / 2.0)     # wall units, monotone decay

    coarse = np.linspace(30.0, 36.4, 9)
    fine = np.linspace(30.0, 36.4, 257)
    values = []
    for grid in (coarse, fine):
        dns = synthetic_dns(grid, np.full_like(grid, 14.0))
        block = mn.window_metric(grid, dns['u_plus'] + residual(grid), dns,
                                 high=36.4)
        values.append(block['values']['integral_rel'])
    assert abs(values[1] - values[0]) < 0.05 * values[1]


def test_signed_mean_carries_the_sign_of_the_offset():
    y = np.linspace(30.0, 36.4, 6)
    dns = synthetic_dns(y, 2.0 * np.log(y) + 5.0)
    low = mn.window_metric(y, dns['u_plus'] - 0.5, dns, high=36.4)
    high = mn.window_metric(y, dns['u_plus'] + 0.5, dns, high=36.4)
    assert low['values']['mean_rel_weighted'] < 0 < high['values']['mean_rel_weighted']
    assert low['implied_offset_u_plus'] < 0 < high['implied_offset_u_plus']


# --------------------------------------------------------------------------- #
# 3. It audits the published object
# --------------------------------------------------------------------------- #

def test_unweighted_norm_reproduces_the_published_graded_number():
    """``stations.csv`` for ``sst_n64``: 0.0224775651552 and 0.0460953626394.

    If this drifts, the audit is grading a different metric than the anchor's
    report and every conclusion in it is void -- hence the exact comparison.
    """
    case = mn.case_metrics('sst', 64)
    assert case['legacy_full']['values']['rms_rel_unweighted'] == pytest.approx(
        0.0224775651552, rel=1e-9)
    assert case['log_layer']['values']['rms_rel_unweighted'] == pytest.approx(
        0.0460953626394, rel=1e-9)
    assert case['legacy_full']['window']['cells'] == 33
    assert case['log_layer']['window']['cells'] == 3
    assert case['log_layer']['window']['high'] == pytest.approx(36.4176, rel=1e-9)


def test_legacy_window_number_is_measure_sensitive_but_log_layer_is_not():
    """Finding under test: weighting changes the full window, not the log layer.

    At ``n = 64`` the legacy window is dominated by densely clustered near-wall
    cells, so the unweighted RMS sits well above the volume-weighted one; in
    the 3-cell log-layer window the residual is a near-constant offset, so the
    four measures agree to a fraction of a percent.
    """
    case = mn.case_metrics('sst', 64)
    full = case['legacy_full']
    log = case['log_layer']
    full_ratio = (full['values']['rms_rel_unweighted']
                  / full['values']['rms_rel_weighted'])
    log_ratio = (log['values']['rms_rel_unweighted']
                 / log['values']['rms_rel_weighted'])
    assert full_ratio > 1.25
    assert log_ratio == pytest.approx(1.0, abs=0.01)
    assert log['unanimous']


def test_matrix_summary_is_self_consistent():
    summary = mn.run_matrix(('sst',), (64,))
    assert summary['schema'].startswith('tensorfvm.case-fvm-channel-kmm180')
    assert len(summary['cases']) == 1
    case = summary['cases'][0]
    for window in ('legacy_full', 'log_layer'):
        block = case[window]
        assert block['unanimous'] == (len(set(block['verdicts'].values())) == 1)
        assert set(block['values']) == set(mn.NORM_CHOICES)
    for row in summary['headline']:
        assert row['best'] <= row['worst']
    # the anchor's tolerance is the threshold used for every per-norm verdict
    assert summary['tolerance'] == pytest.approx(0.05)


# --------------------------------------------------------------------------- #
# 4. Guard rails
# --------------------------------------------------------------------------- #

def test_first_cell_limit_is_re_tau_over_n():
    assert mn.first_cell_limit(180.0, 64) == pytest.approx(2.8125)
    assert mn.first_cell_limit(180.0, 256) == pytest.approx(0.703125)


def test_phase_sweep_rejects_unreachable_clustering():
    with pytest.raises(ValueError, match='unreachable'):
        mn.mesh_phase_sweep('sst', (64,), (5.0,))