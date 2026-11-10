"""M4 Stefan verification: similarity constant, front accuracy, zero-latent control."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.special import erf

from tensorfvm.two_phase.stefan import (
    StefanConfig,
    conduction_temperature,
    neumann_front,
    neumann_temperature,
    solve_stefan_1d,
    stefan_lambda,
    stefan_lambda_two_phase,
)


def test_similarity_constant_solves_the_transcendental_equation():
    for st in (0.02, 0.123, 1.0, 5.0, 20.0):
        lam = stefan_lambda(st)
        residual = math.sqrt(math.pi) * lam * math.exp(lam * lam) * erf(lam) - st
        assert abs(residual) <= 1e-10 * max(st, 1.0)


def test_similarity_constant_small_stefan_limit():
    st = 1.0e-6
    lam = stefan_lambda(st)
    # lambda -> sqrt(St/2) as St -> 0 (erf(lam) ~ 2 lam/sqrt(pi))
    assert lam == pytest.approx(math.sqrt(st / 2.0), rel=0.01)


def test_two_phase_limit_recovers_the_one_phase_constant():
    st = StefanConfig().stefan_number
    assert stefan_lambda_two_phase(st, 1e-12) == pytest.approx(
        stefan_lambda(st), rel=1e-9)


def test_two_phase_constant_increases_with_liquid_superheat():
    base = stefan_lambda_two_phase(0.2, 0.0)
    hotter = stefan_lambda_two_phase(0.2, 0.5)
    assert hotter > base


def test_front_matches_carslaw_and_jaeger():
    cfg = StefanConfig(n_cells=200, t_end=300.0)
    out = solve_stefan_1d(cfg)
    reference = neumann_front(out["t"], out["lambda"], cfg.alpha)
    error = np.abs(out["front"] - reference) / np.maximum(reference, 1e-12)
    assert float(np.max(error)) <= 0.03       # J4 gate
    assert float(np.mean(error)) < 3e-3


def test_temperature_field_matches_the_similarity_profile():
    cfg = StefanConfig(n_cells=200, t_end=300.0)
    out = solve_stefan_1d(cfg)
    analytic = neumann_temperature(out["x"], out["t"][-1], out["lambda"],
                                   cfg.alpha, cfg.t_fusion, cfg.t_wall)
    scale = cfg.t_fusion - cfg.t_wall
    inside = out["x"] < 0.8 * out["front"][-1]
    assert float(np.max(np.abs(out["temperature"][inside]
                               - analytic[inside])) / scale) < 0.02


def test_energy_ledger_closes():
    out = solve_stefan_1d(StefanConfig(n_cells=150, t_end=200.0))
    assert out["energy_ledger_residual"] <= 0.01
    assert out["energy_change"] < 0.0


def test_zero_latent_heat_control_has_no_phase_change():
    cfg = StefanConfig(n_cells=150, t_end=200.0, latent=0.0)
    out = solve_stefan_1d(cfg)
    assert float(np.max(np.abs(out["front"]))) == 0.0
    analytic = conduction_temperature(out["x"], out["t"][-1], cfg.alpha,
                                      cfg.t_fusion, cfg.t_wall)
    scale = cfg.t_fusion - cfg.t_wall
    assert float(np.max(np.abs(out["temperature"] - analytic)) / scale) <= 1e-3


def test_latent_heat_slows_the_front():
    fast = solve_stefan_1d(StefanConfig(n_cells=120, t_end=200.0, latent=1.0e5))
    slow = solve_stefan_1d(StefanConfig(n_cells=120, t_end=200.0,
                                        latent=1.0e6))
    assert slow["final_front"] < fast["final_front"]


def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError):
        StefanConfig(t_wall=300.0).validate()
    with pytest.raises(ValueError):
        StefanConfig(latent=-1.0).validate()