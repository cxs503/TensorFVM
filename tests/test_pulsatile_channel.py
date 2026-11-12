"""Tests for the pulsatile (Womersley) channel verification surface.

The closed form itself is graded first (steadiness limit, bulk/wall-gradient
consistency, Stokes-layer decay law, and the momentum-equation residual taken
with *numerical* derivatives); only then is the production
``PeriodicChannelSolver`` graded against it.  Thresholds are the pre-declared
ones from ``tensorfvm.verification.pulsatile_channel``.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensorfvm.channel_periodic import PeriodicChannelConfig, PeriodicChannelSolver
from tensorfvm.verification import pulsatile_channel as pc


# ------------------------------------------------------------------ closed form
def test_closed_form_satisfies_the_momentum_equation():
    """The closed form must leave only discretisation-level residual."""
    for omega in (0.25, 1.0, 4.0):
        res = pc.closed_form_residual(omega=omega)
        assert res["relative"] < 1e-6, (omega, res)


def test_steady_limit_is_the_analytic_parabola():
    omega = 1e-9 * pc.NU / pc.H ** 2
    y = np.linspace(0.0, pc.H, 201)
    u = pc.womersley_profile(y, 0.0, omega=omega)
    parabola = pc.steady_poiseuille(y)
    assert abs(u.max() / parabola.max() - 1.0) < 1e-6
    bulk = pc.womersley_bulk(0.0, omega=omega)
    assert abs(bulk / (pc.G0 * pc.H ** 2 / (12 * pc.NU)) - 1.0) < 1e-6


def test_bulk_and_wall_gradient_formulas_match_numerics():
    omega = 25.0 * pc.NU
    y = np.linspace(0.0, pc.H, 40001)
    for phase in (0.0, 0.25, 0.5, 0.75):
        t = phase * 2 * math.pi / omega
        u = pc.womersley_profile(y, t, omega=omega)
        bulk = float(np.sum(0.5 * (u[:-1] + u[1:])) * (y[1] - y[0])) / pc.H
        assert abs(bulk / pc.womersley_bulk(t, omega=omega) - 1.0) < 1e-6
        h = 1e-6
        grad = (pc.womersley_profile(h, t, omega=omega)
                - pc.womersley_profile(-h, t, omega=omega)) / (2 * h)
        assert abs(grad / pc.womersley_wall_gradient(t, omega=omega) - 1.0) < 1e-5


def test_stokes_layer_decay_law_is_recovered():
    for W, tol in ((40.0, 0.02), (80.0, 0.02)):
        decay = pc.near_wall_decay_slope(omega=(W / pc.H) ** 2 * pc.NU)
        assert decay["relative_error"] < tol, (W, decay)
        assert decay["delta"] == pytest.approx(pc.stokes_layer((W / pc.H) ** 2 * pc.NU))


def test_shear_leads_the_bulk_velocity_at_high_womersley():
    """Physics check on the closed form: the wall shear leads the bulk flow."""
    omega = (10.0 / pc.H) ** 2 * pc.NU
    bulk = pc.exact_amplitude_phase(quantity="bulk", omega=omega)
    shear = pc.exact_amplitude_phase(quantity="shear_lower", omega=omega)
    assert 0.0 < shear["phase_deg"] - bulk["phase_deg"] < 45.0


def test_oscillatory_profile_is_far_from_the_steady_one():
    """Negative control: a steady solver cannot accidentally match the closed form."""
    omega = (5.0 / pc.H) ** 2 * pc.NU
    y = (np.arange(64) + 0.5) * (pc.H / 64)
    steady = pc.steady_poiseuille(y)
    gap = max(abs(np.linalg.norm(pc.womersley_profile(y, f * 2 * math.pi / omega,
                                                      omega=omega) - steady))
              / np.linalg.norm(steady) for f in np.linspace(0, 1, 9))
    assert gap > 0.10


# --------------------------------------------------------------------- solver
def test_forcing_hook_matches_the_constant_force_path():
    """``step(g)`` with ``g`` equal to the configured force is bit-identical to ``step()``."""
    cfg = PeriodicChannelConfig(nx=4, ny=16, length_m=2.0, height_m=pc.H,
                                density_kg_m3=pc.RHO, viscosity_pa_s=pc.NU * pc.RHO,
                                body_force_m_s2=pc.G0, time_step_s=0.05)
    a, b = PeriodicChannelSolver(cfg), PeriodicChannelSolver(cfg)
    for _ in range(20):
        rec_a = a.step()
        b.step(pc.G0)
        assert rec_a["body_force_m_s2"] == pytest.approx(pc.G0)
    assert torch.equal(a.u, b.u) and torch.equal(a.v, b.v)


def test_steady_poiseuille_regression_of_the_production_solver():
    steady = pc.steady_poiseuille_channel(ny=32, max_steps=5000)
    assert steady["bulk_relative_error"] < 2e-3, steady
    finite = pc.steady_poiseuille_channel(ny=64, max_steps=5000)
    assert finite["bulk_relative_error"] < 5e-4, finite


def test_solver_reproduces_the_closed_form_at_w2():
    run = pc.solve_pulsatile(womersley=2.0, ny=32, dt_fraction=1.0 / 100.0,
                             n_cycles=1.0, n_phases=4)
    err = pc.profile_error(run)
    assert err["max_relative_l2"] < 0.03, err
    assert max(s["max_v"] for s in run["samples"]) == 0.0
    assert max(s["max_div"] for s in run["samples"]) < 1e-10


def test_bulk_amplitude_and_phase_at_w5():
    run = pc.solve_pulsatile(womersley=5.0, ny=32, dt_fraction=1.0 / 100.0,
                             n_cycles=2.0, n_phases=8)
    num = pc.amplitude_phase(run["samples"], quantity="bulk", omega=run["omega"])
    exact = pc.exact_amplitude_phase(quantity="bulk", omega=run["omega"])
    assert abs(num["amplitude"] / exact["amplitude"] - 1.0) < 0.05, (num, exact)
    assert abs(math.degrees(math.remainder(num["phase_rad"] - exact["phase_rad"],
                                          2 * math.pi))) < 8.0, (num, exact)


def test_temporal_refinement_reduces_the_error():
    coarse = pc.solve_pulsatile(womersley=5.0, ny=16, dt_fraction=1.0 / 50.0,
                                n_cycles=1.0, n_phases=4)
    fine = pc.solve_pulsatile(womersley=5.0, ny=16, dt_fraction=1.0 / 100.0,
                              n_cycles=1.0, n_phases=4)
    e_coarse = pc.profile_error(coarse)["max_relative_l2"]
    e_fine = pc.profile_error(fine)["max_relative_l2"]
    assert e_fine < e_coarse, (e_coarse, e_fine)


def test_wall_shear_is_within_three_percent_at_fine_grid():
    run = pc.solve_pulsatile(womersley=5.0, ny=64, dt_fraction=1.0 / 200.0,
                             n_cycles=2.0, n_phases=8)
    num = pc.amplitude_phase(run["samples"], quantity="shear_lower", omega=run["omega"])
    exact = pc.exact_amplitude_phase(quantity="shear_lower", omega=run["omega"])
    assert abs(num["amplitude"] / exact["amplitude"] - 1.0) < 0.03, (num, exact)