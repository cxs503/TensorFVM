"""Round-7 verification of the SA closure used by the periodic-hill case.

Every assertion here is against an *external* reference (analytic laminar
channel, the log-layer total-stress eddy viscosity, the geometric series of a
clustered mesh) rather than against a stored output of the code under test --
that is the point of the module: round 6 could only say "the 2D hill run is
over-diffusive", it had no way to attribute the error to the model.

Run with ``PYTHONPATH=src pytest tests/test_sa_equilibrium.py -q`` (whole file
takes a few seconds; the column is O(100) iterations of two tridiagonal solves).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorfvm.periodic_hill_turbulence import SpalartAllmarasConfig
from tensorfvm.verification import sa_equilibrium as sa


# --------------------------------------------------------------------------- #
# geometry / planning
# --------------------------------------------------------------------------- #
def test_stretched_cell_centres_match_geometric_series():
    ny, half, growth = 64, 1.518, 1.05
    y, delta0 = sa.stretched_cell_centres(ny, half, growth)
    assert y.size == ny
    # symmetric about the mid-plane
    np.testing.assert_allclose(y + y[::-1], 2.0 * half, rtol=0, atol=1e-12)
    # monotone
    assert np.all(np.diff(y) > 0.0)
    # first cell height equals the closed-form geometric series value
    nh = ny // 2
    expected = half / np.sum(growth ** np.arange(nh))
    assert delta0 == pytest.approx(expected, rel=1e-12)
    # the half-channel spacings sum to the half height exactly
    spacings = np.diff(np.concatenate([[0.0], y[: nh - 1] + 0.5 * np.diff(y)[: nh - 1],
                                       [half]]))
    assert spacings.sum() == pytest.approx(half, rel=1e-12)


def test_uniform_grid_is_the_growth_one_limit():
    y_uniform, d0 = sa.stretched_cell_centres(64, 1.518, 1.0)
    assert d0 == pytest.approx(1.518 / 32, rel=1e-12)
    np.testing.assert_allclose(np.diff(y_uniform), d0, rtol=1e-12)


def test_wall_resolution_plan_quantifies_the_round6_dead_end():
    """Round 6 said y+ < 1 costs ``ny ~ 256`` (uniform); the clustered plan is ~7x cheaper."""
    plan = sa.wall_resolution_plan(y_plus_target=1.0, growth=1.1)
    assert plan["first_cell_y_plus"] <= 1.0 + 1e-12
    assert plan["ny_uniform"] > 400          # uniform spacing needs ~512 rows
    assert plan["ny"] < 100                  # clustering needs ~70
    assert plan["saving_factor_vs_uniform"] > 5.0
    # the uniform reference grid really is wall-under-resolved
    uniform = sa.wall_resolution_plan(growth=1.0)
    assert uniform["first_cell_y_plus_at_uniform_reference"] > 5.0
    assert uniform["feasible"] if "feasible" in uniform else True


def test_wall_resolution_plan_rejects_invalid_growth():
    with pytest.raises(ValueError):
        sa.wall_resolution_plan(growth=0.9)
    with pytest.raises(ValueError):
        sa.stretched_cell_centres(63, 1.5, 1.05)     # odd ny


# --------------------------------------------------------------------------- #
# the momentum operator (must be exactly conservative)
# --------------------------------------------------------------------------- #
def test_momentum_column_reproduces_analytic_laminar_channel():
    """Uniform-viscosity column vs the exact Poiseuille solution of ``d/dy(mu dU/dy) = -G``."""
    ny, half, nu, u_tau = 64, 1.518, 1.0 / 2800.0, 0.06
    config = SpalartAllmarasConfig(freestream_ratio=0.0)   # nu~ stays 0 -> laminar
    col = sa.solve_channel_column(ny=ny, growth=1.0, u_tau=u_tau, config=config,
                                  iterations=400, relaxation=1.0)
    height = 2.0 * half
    G = u_tau ** 2 / half
    y = col["y"]
    exact = G / (2.0 * nu) * y * (height - y)
    # discrete peak vs the exact parabola: second-order, so a few percent at ny=64
    assert col["U"].max() == pytest.approx(exact.max(), rel=2e-2)
    assert np.max(np.abs(col["U"] - exact)) / exact.max() < 0.05


@pytest.mark.parametrize("ny,growth", [(48, 1.0), (64, 1.0), (96, 1.04)])
def test_column_is_conservative_and_symmetric(ny, growth):
    """The wall shear must reproduce the imposed ``u_tau`` to solver tolerance."""
    col = sa.solve_channel_column(ny=ny, growth=growth, iterations=4000)
    nu, y = col["nu"], col["y"]
    u_tau_num = math.sqrt(nu * col["shear"][0])
    assert u_tau_num == pytest.approx(0.06, rel=1e-4)
    # no-slip symmetry about the mid-plane (a sign error here made the round-7
    # prototype asymmetric: the upper-half shear entered S~ negatively)
    assert abs(col["U"][0] - col["U"][-1]) / col["U"][0] < 1e-8
    assert np.max(np.abs(col["U"] - col["U"][::-1])) / col["U"].max() < 1e-6


# --------------------------------------------------------------------------- #
# the closure vs the literature reference
# --------------------------------------------------------------------------- #
def test_sa_eddy_viscosity_matches_log_layer_reference():
    """Peak ``nu_t/nu`` within 10 % of ``kappa u_tau y (1-y/h)`` -- the round-7 verdict."""
    report = sa.equilibrium_report(ny=192, growth=1.05)
    assert report["over_diffusion_ratio"] == pytest.approx(1.0, abs=0.10)
    # the merged closure must NOT be the 1.8x-over-diffusive object round 6 blamed
    assert report["peak_nu_t_over_nu"] < 30.0
    assert report["u_tau_relative_error"] < 1e-3


def test_peak_eddy_viscosity_is_mesh_independent():
    """y+ from 10 down to 0.1 must not move the equilibrium peak by more than 2 %."""
    peaks = [sa.equilibrium_report(ny=ny, growth=g)["peak_nu_t_over_nu"]
             for ny, g in ((48, 1.0), (96, 1.0), (192, 1.05))]
    assert max(peaks) / min(peaks) - 1.0 < 0.02


def test_reference_eddy_viscosity_formula():
    """``nu_t^ref`` peaks at the mid-plane and vanishes at both walls."""
    y = np.linspace(0.0, 1.0, 101)
    ref = sa.reference_eddy_viscosity(y, 1.0, h=1.0)
    assert ref[0] == 0.0 and ref[-1] == pytest.approx(0.0)
    assert ref.max() == pytest.approx(0.41 * 0.25, rel=1e-2)   # kappa*u_tau*h/4


def test_log_law_fit_recovers_synthetic_kappa():
    y_plus = np.logspace(1.2, 2.4, 60)
    u_plus = (1.0 / 0.41) * np.log(y_plus) + 5.2
    fit = sa.log_law_fit(y_plus, u_plus)
    assert fit["kappa"] == pytest.approx(0.41, rel=1e-6)
    assert fit["B"] == pytest.approx(5.2, rel=1e-6)


def test_hill_case_channel_half_height_convention():
    assert sa.channel_half_height_for_hill_case() == pytest.approx(1.518)
    assert sa.channel_half_height_for_hill_case(3.036) == pytest.approx(1.518)