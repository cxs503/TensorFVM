"""M1 drift-flux verification: Wallis closed form, FVM convergence, controls."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorfvm.two_phase.drift_flux import (
    DriftFluxConfig,
    compare_profiles,
    homogeneous_void_fraction,
    solve_drift_flux,
    wallis_drift_velocity,
    wallis_gas_flux,
    wallis_profile,
    wallis_void_fraction,
)


def test_wallis_closed_form_solves_the_drift_flux_relation():
    """The quadratic root must satisfy ``j_g = C0 alpha j + alpha(1-alpha) v_dj``."""
    v_dj, c0 = 0.24, 1.2
    for j_l in (0.0, 0.3, 1.5):
        for j_g in (1e-4, 0.01, 0.14, 0.9):
            alpha = wallis_void_fraction(j_g, j_l, v_dj, c0)
            residual = j_g - (c0 * alpha * (j_g + j_l)
                              + alpha * (1.0 - alpha) * v_dj)
            assert abs(residual) <= 1e-10 * max(j_g, 1e-12)
            assert 0.0 <= alpha <= 1.0


def test_wallis_limits():
    # zero gas -> zero void fraction
    assert wallis_void_fraction(0.0, 0.5, 0.24, 1.2) == 0.0
    # zero drift velocity -> homogeneous (no-slip) value
    j_g, j_l = 0.2, 0.5
    assert wallis_void_fraction(j_g, j_l, 0.0, 1.0) == pytest.approx(
        homogeneous_void_fraction(j_g, j_l, 1.0), rel=1e-14)
    # non-physical inputs are rejected by the correlations
    with pytest.raises(ValueError):
        wallis_drift_velocity("nonsense", 0.05, 998.0, 1.2, 0.0728)


def test_gas_flux_inverts_the_closure():
    """``wallis_gas_flux`` and ``wallis_void_fraction`` are inverse maps."""
    v_dj, c0, j_l, rho_g = 0.24, 1.2, 0.5, 3.5
    alpha = 0.15
    j_g = wallis_gas_flux(alpha, j_l, rho_g, v_dj, c0) / rho_g
    assert wallis_void_fraction(j_g, j_l, v_dj, c0) == pytest.approx(alpha, rel=1e-12)


def test_drift_velocity_correlations_are_physical():
    bubbly = wallis_drift_velocity("bubbly", 0.05, 998.2, 1.2, 0.0728)
    churn = wallis_drift_velocity("churn", 0.05, 998.2, 1.2, 0.0728)
    assert 0.05 < bubbly < 0.5
    assert 0.1 < churn < 0.5
    assert churn == pytest.approx(0.35 * math.sqrt(9.80665 * 0.05 * (998.2 - 1.2)
                                                   / 998.2))


def test_fvm_matches_wallis_within_the_gate():
    cfg = DriftFluxConfig(n_cells=60)
    ref = wallis_profile(cfg)
    num = solve_drift_flux(cfg)
    out = compare_profiles(ref, num)
    assert out["max_rel_error"] <= 0.10          # J1 gate
    assert out["rms_rel_error"] <= 0.02
    assert num.steady
    # per-phase mass closure along the pipe (J5)
    assert num.gas_mass_drift <= 0.01
    assert num.liquid_mass_drift <= 0.01


def test_fvm_refinement_monotonically_reduces_the_error():
    errors = []
    for n in (25, 50, 100):
        cfg = DriftFluxConfig(n_cells=n)
        errors.append(compare_profiles(wallis_profile(cfg),
                                       solve_drift_flux(cfg))["max_rel_error"])
    assert errors[1] < errors[0] and errors[2] < errors[1]
    assert errors[-1] < 0.01


def test_zero_gas_control_degenerates_to_single_phase_liquid_flow():
    cfg = DriftFluxConfig(n_cells=40, gas_mass_flux=0.0)
    out = solve_drift_flux(cfg)
    assert float(np.max(out.alpha)) == 0.0
    assert np.allclose(out.v_liquid, cfg.liquid_superficial_velocity, rtol=1e-14)
    gradient = (out.pressure[0] - out.pressure[-1]) / (cfg.length - cfg.dz)
    assert gradient == pytest.approx(cfg.rho_liquid * cfg.gravity, rel=1e-12)


def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError):
        DriftFluxConfig(n_cells=2).validate()
    with pytest.raises(ValueError):
        DriftFluxConfig(gas_mass_flux=-1.0).validate()