"""CASE-FVM-CONVDIFF-PHASE gates: phase accuracy of the shipped CV-diffusion kernel."""
import math

import numpy as np
import pytest

from tensorfvm.verification import convdiff_phase as cp


def test_analytic_sine_satisfies_pde():
    assert cp.pde_residual(c=1.0, D=1.0e-2) < 1e-5


def test_ring_stencil_is_uniform():
    coef = cp.ring_coefficients(32, 1.0, speed=0.1, diffusivity=1e-3)
    assert coef["spread_C"] < 1e-12
    assert coef["spread_a"] < 1e-12
    assert coef["spread_volume"] < 1e-12
    assert abs(coef["ds"] / coef["volume_over_area"] - 1.0) < 1e-3


@pytest.mark.parametrize("courant", [0.25, 0.5, 0.75, 1.0])
def test_measured_phase_matches_closed_form_symbol(courant):
    m = cp.march_sine(nx=32, courant=courant, diffusion_number=0.0, steps=40)
    sym = m["symbol_phase_error_per_step"]
    assert abs(m["lag_per_step"] - sym) < 1e-12 + 1e-3 * abs(sym)


def test_phase_exact_at_half_and_unit_courant():
    for C in (0.5, 1.0):
        m = cp.march_sine(nx=32, courant=C, diffusion_number=0.0, steps=40)
        assert abs(m["lag_per_step"]) < 1e-12


def test_phase_lag_third_order_under_refinement():
    sweep = cp.grid_sweep(sizes=(16, 32, 64), courant=0.25, steps=40)
    order = cp.observed_order([abs(g["lag_per_step"]) for g in sweep],
                              [g["ds_over_L"] for g in sweep])
    assert 2.7 < order < 3.3


def test_reference_condition_meets_three_percent():
    m = cp.march_sine(nx=64, courant=0.25, diffusion_number=0.0, steps=200)
    assert abs(m["lag_relative"]) < 0.03


def test_upwind_amplitude_follows_modified_equation():
    m = cp.march_sine(nx=64, courant=0.25, diffusion_number=0.0, steps=200)
    C, th, n = m["courant_measured"], m["theta"], m["steps"]
    predicted = math.exp(-0.5 * C * (1.0 - C) * th ** 2 * n)
    assert abs(m["amplitude"] - predicted) / predicted < 0.03


@pytest.mark.parametrize("number", [1e-3, 1e-2])
def test_diffusive_run_matches_symbol_with_patch_term(number):
    """With diffusion on, the run is only explained by the w-augmented symbol."""
    m = cp.march_sine(nx=64, courant=0.25, diffusion_number=number, steps=100)
    assert m["patch_relaxation_w"] > 1e-3
    sym, lag = m["symbol_phase_error_per_step"], m["lag_per_step"]
    assert abs(lag - sym) < 1e-9 + 0.02 * abs(sym)
    assert abs(m["amplitude"] - m["symbol_abs_per_step"] ** 100) < 1e-6


def test_patch_closure_slows_the_wave():
    """Documented limitation of the shipped closure (not a no-flux boundary)."""
    base = cp.march_sine(nx=64, courant=0.25, diffusion_number=0.0, steps=100)
    diff = cp.march_sine(nx=64, courant=0.25, diffusion_number=1e-2, steps=100)
    assert diff["lag_deg"] > 10.0 * base["lag_deg"]
