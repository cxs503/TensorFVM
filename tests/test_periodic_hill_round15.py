"""Round-15 tests: channel closure audit, transport-blend plumbing, re-scoring.

The gates of ``CASE-FVM-PERIODIC-HILLS`` are only meaningful if the measuring
instrument is itself verified, so the pure post-processing functions are tested
against *analytic* profiles (a log law, and a profile integrated from the linear
total-stress identity) and the solver is exercised once end-to-end on a tiny
grid so that the wiring cannot rot.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from tensorfvm.verification.periodic_hill_round11 import _closure_block, build_solver
from tensorfvm.verification.periodic_hill_round15 import (
    LOG_B,
    LOG_KAPPA,
    channel_postprocess,
    channel_solver,
    channel_verdict,
    dean_friction,
    run_channel,
    score_hill,
    write_json,
)
from tensorfvm.verification.periodic_hill_reference import X_R_OVER_H, y05_table


# --------------------------------------------------------------------------- #
# 1. the knob that never reached the closure
# --------------------------------------------------------------------------- #
def test_transport_blend_reaches_the_sst_closure():
    """Round 11--14 defect: ``--convection-blend`` only set the momentum blend.

    The artefact recorded both numbers (momentum 0.75, transport 0.0) and the
    round-14 "second-order transport" next step had therefore never been run.
    Round 15 gives the transport its own knob; this test pins it.
    """
    solver = build_solver("sst", nx=8, ny=8, convection_blend=0.25,
                          transport_convection_blend=0.75)
    assert solver.config.convection_blend == pytest.approx(0.25)
    assert solver.turbulence.config.convection_blend == pytest.approx(0.75)
    block = _closure_block("sst", solver)
    assert block["transport_convection_blend"] == pytest.approx(0.75)
    assert block["momentum_convection_blend"] == pytest.approx(0.25)


def test_transport_blend_default_is_unchanged():
    """No regression for the historical artefacts: default transport blend stays 0."""
    solver = build_solver("sst", nx=8, ny=8)
    assert solver.turbulence.config.convection_blend == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# 2. the channel reference
# --------------------------------------------------------------------------- #
def test_dean_friction_definition_is_self_consistent():
    ref = dean_friction(8501.5)
    cf = ref["cf"]
    # C_f = 2 (u_tau/U_b)^2
    assert cf == pytest.approx(2.0 * ref["u_tau_over_ub"] ** 2)
    # Re_tau = (Re_b/2) (u_tau/U_b)
    assert ref["re_tau"] == pytest.approx(0.5 * 8501.5 * ref["u_tau_over_ub"])
    # sanity: ~262 for the 3.036H flat channel at Re_b(h) = 2800
    assert 240.0 < ref["re_tau"] < 290.0


def test_dean_friction_is_monotone_and_validates_input():
    re_tau = [dean_friction(r)["re_tau"] for r in (1e3, 1e4, 1e5)]
    assert re_tau[0] < re_tau[1] < re_tau[2]
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError):
            dean_friction(bad)


# --------------------------------------------------------------------------- #
# 3. the instrument: pure post-processing against analytic profiles
# --------------------------------------------------------------------------- #
def _log_law_column(ny=80, h=1.518, nu=1.0 / 2800.0, u_tau=1.0, multiplier=1.0):
    y = np.linspace(0.005 * h, 0.95 * h, ny)
    y_plus = u_tau * y / nu
    u_plus = np.log(y_plus) / LOG_KAPPA + LOG_B
    nu_t_plus = multiplier * np.maximum(LOG_KAPPA * y_plus - 1.0, 0.0)
    return y, u_plus * u_tau, nu_t_plus * nu


def test_eddy_viscosity_identity_is_recovered_from_a_log_law():
    """``nu_t+ = kappa y+ - 1`` is exact for a log law: ratio must be 1."""
    y, u, nut = _log_law_column()
    post = channel_postprocess(y, u, nut, nu=1.0 / 2800.0, u_tau=1.0, half_height=1.518)
    assert post["nu_t_plus_ratio_median"] == pytest.approx(1.0, rel=1e-9)
    post_verdict = channel_verdict(post, re_tau=dean_friction(8501.5)["re_tau"],
                                   reference=dean_friction(8501.5))
    assert post_verdict["gates"]["C3_eddy_viscosity_identity"] is True
    assert post_verdict["gates"]["C1_log_law"] is True
    assert post_verdict["kappa_eff"] == pytest.approx(LOG_KAPPA, rel=1e-6)
    assert post_verdict["B"] == pytest.approx(LOG_B, rel=1e-6)


def _stress_consistent_column(multiplier=6.0, ny=400):
    """Profile integrated from the linear-stress identity with a chosen nu_t."""
    h, nu, u_tau = 1.518, 1.0 / 2800.0, 1.0
    y = np.linspace(0.0, 0.98 * h, ny)
    y_plus = u_tau * y / nu
    nut = nu * multiplier * np.maximum(LOG_KAPPA * y_plus - 1.0, 0.0)
    dudy = u_tau ** 2 * (1.0 - y / h) / (nu + nut)
    u = np.concatenate(([0.0], np.cumsum(0.5 * (dudy[1:] + dudy[:-1]) * np.diff(y))))
    return y, u, nut


def test_six_fold_over_production_is_detected_and_named():
    """The hill's 342/30 signature must be reported as a closure-level defect."""
    y, u, nut = _stress_consistent_column(multiplier=6.0)
    post = channel_postprocess(y, u, nut, nu=1.0 / 2800.0, u_tau=1.0, half_height=1.518)
    verdict = channel_verdict(post, re_tau=dean_friction(8501.5)["re_tau"],
                              reference=dean_friction(8501.5))
    assert post["nu_t_plus_ratio_median"] == pytest.approx(6.0, rel=1.0)
    assert verdict["gates"]["C2_linear_stress"] is True
    assert verdict["gates"]["C3_eddy_viscosity_identity"] is False
    assert "closure-level" in verdict["interpretation"]


def test_linear_total_stress_identity_is_checked_not_assumed():
    """Integrate ``du/dy = u_tau^2 (1 - y/h) / (nu + nu_t)``: C2 must pass."""
    h, nu, u_tau = 1.518, 1.0 / 2800.0, 0.06165374861815695
    y = np.linspace(0.0, 0.98 * h, 400)
    y_plus = u_tau * y / nu
    nu_t = nu * np.maximum(LOG_KAPPA * y_plus - 1.0, 0.0)
    dudy = u_tau ** 2 * (1.0 - y / h) / (nu + nu_t)
    u = np.concatenate(([0.0], np.cumsum(0.5 * (dudy[1:] + dudy[:-1]) * np.diff(y))))
    post = channel_postprocess(y, u, nu_t, nu=nu, u_tau=u_tau, half_height=h)
    assert post["stress_residual_max_rel"] < 0.05
    reference = dean_friction(8501.5)
    verdict = channel_verdict(post, re_tau=reference["re_tau"], reference=reference)
    assert verdict["gates"]["C2_linear_stress"] is True


# --------------------------------------------------------------------------- #
# 4. end-to-end wiring (tiny grid: the point is the plumbing, not the physics)
# --------------------------------------------------------------------------- #
def test_channel_run_produces_a_serialisable_artefact(tmp_path):
    payload = run_channel("sst", ny=16, nx=8, dt=0.01, steps=200, burn_in=100,
                          block_steps=100, chunk=50, verbose=False)
    assert payload["round"] == 15
    assert len(payload["blocks"]) == 1
    block = payload["blocks"][-1]
    for key in ("y_plus", "u_plus", "nu_t_plus", "log_law_model_nu_t_plus"):
        assert len(block["profile"][key]) == 16
    assert payload["reference"]["re_tau"] == pytest.approx(
        dean_friction(payload["config"]["re_bulk_full_height"])["re_tau"])
    out = write_json(payload, tmp_path / "chan.json")
    reloaded = json.loads(out.read_text())
    assert reloaded["final"]["gates"].keys() == block["verdict"]["gates"].keys()


def test_channel_solver_has_no_hill():
    solver = channel_solver("sst", ny=16, nx=8)
    assert solver.config.hill_amplitude == 0.0
    # a flat bottom wall: no hill bump in the eta coordinate
    dv = np.asarray(solver.mesh.delta_vertices, dtype=float)
    assert np.allclose(dv, dv.flat[0])


# --------------------------------------------------------------------------- #
# 5. hill re-score against the single-sourced reference
# --------------------------------------------------------------------------- #
def _fake_hill_payload(x_r=3.14, offset=0.0):
    xs = np.linspace(0.5, 9.0, 400)
    table = y05_table()
    us = np.interp(xs, sorted(table), [table[k] for k in sorted(table)]) + offset
    return {
        "time_averaged": {"x_r_over_H": x_r, "max_nu_t_ratio": 297.0},
        "horizontal_profile_y_H_0.5": {"x_over_H": list(xs), "u_over_Ub": list(us)},
        "stationarity": {"x_r_over_H": {"ok": True}},
        "closure": {"name": "sst"},
        "config": {"nx": 96, "ny": 48},
    }


def test_hill_rescore_uses_the_dns_reference_and_both_references_are_reported():
    score = score_hill(_fake_hill_payload(x_r=X_R_OVER_H))
    assert score["x_r_over_H_pass"] is True
    assert score["acceptance_pass"] is True
    assert score["x_r_over_H_reference"] == pytest.approx(X_R_OVER_H)
    # 5.34 is still reported so an artefact cannot quote one reference silently
    assert score["x_r_over_H_legacy_reference"] == pytest.approx(5.34)

    bad = score_hill(_fake_hill_payload(x_r=3.14))
    assert bad["x_r_over_H_pass"] is False
    assert bad["x_r_over_H_error"] > 0.4
    assert bad["acceptance_pass"] is False


def test_hill_rescore_profile_band_is_the_round11_convention():
    """|ref| >= 0.1 -> relative 10 %; near-zero stations -> absolute 0.10."""
    score = score_hill(_fake_hill_payload(x_r=X_R_OVER_H, offset=0.0))
    bands = {s["band"] for s in score["profile_stations"]}
    assert bands == {"relative", "absolute"}
    assert score["profile_stations_pass"] == score["profile_stations_admissible"]
    shifted = score_hill(_fake_hill_payload(x_r=X_R_OVER_H, offset=0.05))
    assert shifted["profile_stations_pass"] < score["profile_stations_pass"]