"""Tests for the closure-free (``channel_spectral_dns``) PASS audit (KMM180).

The assertions target the audit's *discriminating power*, not "the code ran":

* the passage criterion is not rigged -- a synthetic admissible profile passes;
* the shipped artifacts are rejected on the two structural preconditions
  (zero resolved Reynolds shear, 0.71 ``u_tau^2`` momentum-balance residual)
  while their graded 5 % criterion still reads PASS, which is exactly the
  ``VACUOUS_PASS`` classification;
* the negative control shows the same diagnostics are exact on the discrete
  laminar state of the same solver, so the residual is not a diagnostic bug;
* the wall-unit bookkeeping (``dt+``) matches its definition and the shipped
  formula is shown to differ by exactly ``Re_tau``.
"""
from __future__ import annotations

import numpy as np
import pytest

from tensorfvm.verification.channel_spectral_dns import ChannelConfig, SpectralChannel
from tensorfvm.verification.kmm180_closurefree_audit import (
    CRITERIA,
    ClosureFreeArtifacts,
    blended_initial_profile,
    closurefree_audit,
    dt_plus,
    laminar_negative_control,
    legacy_dt_plus,
    load_closurefree_artifacts,
)


def synthetic(uv_share: float, residual: float, *, drift: float = 0.02,
              graded_error: float = 0.02) -> ClosureFreeArtifacts:
    """Manufacture a profile family where the two preconditions can be dialled."""
    re_tau = 180.0
    y_plus = np.array([0.5, 1.0, 3.0, 8.0, 15.0, 25.0, 31.0, 34.0, 40.0, 60.0,
                       90.0, 120.0, 160.0])
    target = 1.0 - y_plus / re_tau
    uv = -uv_share * target
    du_dy = target + uv + 0.5 * residual * (1.0 + np.sign(target))
    ic = blended_initial_profile(y_plus)
    u_plus = ic + drift * np.linspace(0.0, 1.0, y_plus.size)
    u_dns = u_plus / (1.0 + graded_error)
    return ClosureFreeArtifacts(
        y_plus=y_plus, u_plus=u_plus, u_plus_dns=u_dns,
        du_plus_dy_plus=du_dy, uv_plus=uv,
        uu_plus=np.full_like(y_plus, 1.0), vv_plus=np.full_like(y_plus, 0.4),
        ww_plus=np.full_like(y_plus, 0.5),
        total_shear_plus=du_dy - uv, target_shear_plus=target,
        metrics=dict(re_tau=re_tau, tolerance=0.05, averaging_time_span=6.0,
                     resolution=dict(time_step=1e-3, length_x_plus=1130.97,
                                     dt_plus=dt_plus(1e-3, re_tau=re_tau)),
                     global_balance=dict(bulk_u_plus=15.0)),
    )


# --------------------------------------------------------------------------- #
# Criteria are frozen and the gate can pass
# --------------------------------------------------------------------------- #
def test_criteria_are_the_preregistered_values():
    assert CRITERIA['uv_share_min'] == pytest.approx(0.50)
    assert CRITERIA['balance_residual_max'] == pytest.approx(0.05)
    assert CRITERIA['flow_throughs_min'] == pytest.approx(1.00)
    assert CRITERIA['low_y_plus'] == pytest.approx(30.0)
    assert CRITERIA['log_layer_fraction'] == pytest.approx(0.20)


def test_equilibrium_profile_is_admitted():
    verdict = closurefree_audit(synthetic(uv_share=0.75, residual=1e-3))
    assert verdict.admitted, verdict.notes
    assert verdict.verdict == 'ADMISSIBLE'
    assert verdict.failed == []


def test_zero_reynolds_shear_is_rejected_even_if_graded_error_is_small():
    verdict = closurefree_audit(synthetic(uv_share=0.0, residual=1e-3,
                                          graded_error=0.001))
    assert not verdict.admitted
    assert 'C1 uv_share' in verdict.failed
    # the graded 5 % criterion is still satisfied -> the pass would be vacuous
    assert verdict.indicators['graded_verdict'] == 'PASS'
    assert verdict.verdict == 'VACUOUS_PASS'


def test_imbalanced_state_is_rejected():
    verdict = closurefree_audit(synthetic(uv_share=0.75, residual=0.5))
    assert 'C2 balance_residual' in verdict.failed


def test_short_averaging_is_rejected():
    art = synthetic(uv_share=0.75, residual=1e-3)
    art.metrics['averaging_time_span'] = 0.01      # 0.024 flow-throughs
    verdict = closurefree_audit(art)
    assert 'C3 flow_through_times' in verdict.failed
    # a PASS obtained over a sub-flow-through window is vacuous too
    assert verdict.verdict == 'VACUOUS_PASS'


def test_failing_route_with_failing_grade_is_not_called_vacuous():
    verdict = closurefree_audit(synthetic(uv_share=0.0, residual=0.5,
                                          graded_error=0.20))
    assert verdict.verdict == 'NOT_APPLICABLE'


# --------------------------------------------------------------------------- #
# The shipped artifacts
# --------------------------------------------------------------------------- #
def test_shipped_artifacts_are_loadable_and_consistent():
    art = load_closurefree_artifacts()
    assert art.y_plus.size >= 32
    assert np.all(np.diff(art.y_plus) > 0)
    assert np.allclose(art.total_shear_plus,
                       art.du_plus_dy_plus - art.uv_plus, atol=1e-5)
    assert np.allclose(art.target_shear_plus, 1.0 - art.y_plus / 180.0, atol=1e-5)


def test_shipped_pass_is_vacuous():
    verdict = closurefree_audit()
    assert verdict.verdict == 'VACUOUS_PASS'
    assert set(verdict.failed) == {'C1 uv_share', 'C2 balance_residual'}
    ind = verdict.indicators
    # graded PASS at 5 % ...
    assert ind['graded_verdict'] == 'PASS'
    assert ind['graded_rms_relative'] < ind['graded_criterion']
    # ... obtained in a state with no resolved Reynolds shear and no balance
    assert ind['uv_share_max'] < 1e-3
    assert ind['uv_plus_max_abs'] < 1e-3
    assert ind['balance_residual_max'] > 0.5
    assert ind['balance_residual_fraction'] > 0.5
    # the profile barely moved from its own initial condition
    assert ind['initial_condition_drift_relative'] < 0.05


def test_shipped_artifacts_report_the_wrong_time_step_in_wall_units():
    art = load_closurefree_artifacts()
    dt = art.metrics['resolution']['time_step']
    shipped = art.metrics['resolution']['dt_plus']
    assert shipped == pytest.approx(legacy_dt_plus(dt, re_tau=180.0))
    assert dt_plus(dt, re_tau=180.0) == pytest.approx(shipped / 180.0)


# --------------------------------------------------------------------------- #
# Negative control: the same diagnostics on the discrete laminar state
# --------------------------------------------------------------------------- #
def test_laminar_negative_control_is_exact():
    control = laminar_negative_control(re_tau=180.0, ny=64, nx=8, nz=8, steps=5)
    assert control['uv_plus_max_abs'] == pytest.approx(0.0, abs=1e-14)
    assert control['balance_residual_max'] < 1e-10
    assert control['fixed_point_drift'] < 1e-10
    assert abs(control['wall_shear_error']) < 0.01
    assert control['legacy_dt_plus'] == pytest.approx(180.0 * control['dt_plus'])


def test_resolution_report_time_step_matches_its_definition():
    config = ChannelConfig(re_tau=180.0, nx=8, ny=32, nz=8)
    solver = SpectralChannel(config)
    report = solver.resolution_report()
    assert report['dt_plus'] == pytest.approx(
        solver.time_step * config.u_tau * config.re_tau / config.half_width)


def test_blended_initial_profile_matches_the_solver_initial_condition():
    config = ChannelConfig(re_tau=180.0, nx=8, ny=32, nz=8)
    solver = SpectralChannel(config)
    solver.initialize(noise=0.0, seed=1)
    half = config.ny // 2
    profile = solver.u.mean(axis=(1, 2))
    mirrored = 0.5 * (profile[:half] + profile[::-1][:half])
    assert np.allclose(mirrored, blended_initial_profile(solver.y_plus[:half]),
                       rtol=1e-10, atol=1e-10)