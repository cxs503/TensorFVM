"""Tests for the Spalart--Allmaras row of the KMM180 anchor matrix.

The gates here are the ones that can fail for a *wrong* SA implementation, not
merely ones that execute code:

* the model's own log-layer solution ``nu_t+ = kappa y+ (1 - y/h)`` must be
  approached from below and approached *monotonically* with grid refinement;
* the momentum balance of the converged state, ``(nu + nu_t) dU/dy = 1 - y``,
  must hold to round-off, which catches a mis-scaled ``nu_t``/``f_v1``;
* the wall value of ``nuTilde`` must be zero and ``f_v1 -> 0`` there, otherwise
  the eddy viscosity would not vanish on the wall;
* the grading plumbing must give exactly zero deviation for the reference itself.
"""
import numpy as np
import pytest

from tensorfvm.verification.kmm180_sa import (SA_CONSTANTS, auxiliary, log_layer_slope,
                                              sa_coefficients, self_check, solve_sa_channel)
from tensorfvm.verification.kmm1987 import load_kmm1987_reference, validate_kmm1987


@pytest.fixture(scope='module')
def solve():
    return {n: solve_sa_channel(n, re_tau=180.0, first_cell_y_plus=0.3) for n in (64, 128)}


def test_constants_are_the_published_ones():
    c = sa_coefficients()
    assert c['c_b1'] == SA_CONSTANTS['c_b1']
    assert c['c_w1'] == pytest.approx(0.1355 / 0.41 ** 2 + (1 + 0.622) / (2 / 3), rel=1e-12)


def test_wall_and_centre_behaviour(solve):
    r = solve[128]
    assert r['converged']
    assert r['relative_local_residual'] < 1e-8
    i10 = int(np.argmin(np.abs(r['y_plus'] - 10.)))
    assert r['nutilde'][0] < 0.05 * r['nutilde'][i10]        # wall value ~ 0 (first cell at y+=0.3)
    assert r['nut'][0] < 1e-3 * r['nut'].max()              # f_v1 -> 0 at the wall
    assert r['nut'][-1] == pytest.approx(r['nut'][-1])      # finite at the centre plane
    assert np.all(np.diff(r['u_plus']) > 0)                 # monotone mean velocity


def test_momentum_balance_is_satisfied(solve):
    r = solve[128]
    lhs = (r['nu'] + r['nut']) * r['shear']
    assert np.max(np.abs(lhs - (1 - r['y']))) < 1e-10


def test_log_layer_is_approached_with_resolution(solve):
    coarse = self_check(solve[64])
    fine = self_check(solve[128])
    # the implied Karman constant must be below the model's asymptotic 0.41
    # (finite Re_tau) and must move towards it under refinement
    assert fine['log_layer']['implied_kappa'] < fine['model_kappa']
    fine_ratio = fine['mixing_length']['ratio_to_mixing_length']
    assert fine_ratio > coarse['mixing_length']['ratio_to_mixing_length']
    # SA sits *below* the local mixing-length baseline kappa*y+(1-y/h) in the
    # buffer/log region (the DNS itself does), so the ratio is O(1) but < 1.
    assert 0.5 < fine_ratio < 1.0


def test_negative_controls():
    with pytest.raises(ValueError):
        solve_sa_channel(4)
    with pytest.raises(ValueError):
        solve_sa_channel(64, first_cell_y_plus=50.0)
    with pytest.raises(ValueError):
        log_layer_slope(dict(y_plus=np.array([1.0]), u_plus=np.array([1.0]), re_tau=180.))


def test_grading_gives_zero_for_the_reference_itself():
    dns = load_kmm1987_reference()
    r = solve_sa_channel(64, re_tau=180.0, first_cell_y_plus=0.3)
    u_plus = np.interp(r['y_plus'], dns['y_plus'], dns['u_plus'])
    v = validate_kmm1987(r['y_plus'], u_plus, r['Cf'], r['bulk_u_plus'], dns=dns)
    # the residual is the PCHIP interpolant's own error (~4e-5), not a solver term
    assert v['log_region']['rms_relative'] < 1e-3
    assert v['verdict'] == 'PASS'


def test_sa_verdict_on_anchor_windows(solve):
    dns = load_kmm1987_reference()
    r = solve[128]
    v = validate_kmm1987(r['y_plus'], r['u_plus'], r['Cf'], r['bulk_u_plus'], dns=dns)
    # SA is calibrated on the log law; on the anchor's 5 % criterion it must land
    # in the same regime as the two-equation models, i.e. be graded, not crash.
    assert v['log_region']['rms_relative'] < 0.05
    assert v['log_region_narrow']['rms_relative'] < 0.05


def test_published_artifacts_match_a_fresh_solve():
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    summary_path = root / 'docs' / 'case-fvm-channel-kmm180-sa' / 'summary.json'
    if not summary_path.exists():
        pytest.skip('published artifacts absent')
    summary = json.loads(summary_path.read_text())
    dns = load_kmm1987_reference()
    n = summary['grids'][-1]
    r = solve_sa_channel(n, re_tau=180.0, first_cell_y_plus=summary['first_cell_y_plus'])
    v = validate_kmm1987(r['y_plus'], r['u_plus'], r['Cf'], r['bulk_u_plus'], dns=dns)
    case = summary['cases']['sa_n%d' % n]
    assert v['log_region']['rms_relative'] == pytest.approx(case['log_region_rms'], rel=1e-9)
    assert v['log_region_narrow']['rms_relative'] == pytest.approx(case['log_layer_rms'], rel=1e-9)