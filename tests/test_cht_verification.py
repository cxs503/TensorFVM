"""Tests of the conjugate-heat-transfer verification surface.

The fast assertions run everywhere; the coupled cross-flow case is marked slow
(the flow solve needs minutes) and is skipped unless ``CHT_SLOW=1``.
"""

import math
import os

import numpy as np
import pytest

from tensorfvm.cht.energy import Grid2D, solve_steady_energy
from tensorfvm.cht.flatplate import FlatPlateCHT
from tensorfvm.cht.polar import (ConjugateCylinder, PolarVorticitySolver,
                                 periodic_overlap_matrix)
from tensorfvm.cht.similarity import (classical_correlation, pohlhausen_nusselt,
                                      solve_blasius)

SLOW = os.environ.get("CHT_SLOW") == "1"


# --------------------------------------------------------------------- similarity
def test_blasius_wall_shear_matches_classical_value():
    sol = solve_blasius()
    assert sol.wall_shear == pytest.approx(0.332057336215, rel=1e-6)


@pytest.mark.parametrize("pr,expected", [(0.7, 0.29268), (1.0, 0.33206),
                                         (5.0, 0.57669), (7.0, 0.64592)])
def test_pohlhausen_nusselt_matches_literature(pr, expected):
    """``Nu_x / sqrt(Re_x)`` from the ODE must reproduce the tabulated values."""
    value = pohlhausen_nusselt(pr)
    assert value == pytest.approx(expected, rel=2e-3)
    assert abs(classical_correlation(pr) / value - 1.0) < 0.02


# --------------------------------------------------------------------- flat plate
def test_flat_plate_conjugate_nusselt_within_three_percent():
    """Conjugate plate (k_s/k_f = 100) against the Pohlhausen similarity solution."""
    case = FlatPlateCHT(re_l=1.0e5, pr=0.7, solid_ratio=100.0, nx=120, ny_fluid=60,
                        ny_solid=12)
    res = case.solve()
    assert res.metrics["nu_l2_rel_error"] < 0.03
    assert res.metrics["nu_max_rel_error"] < 0.15
    # single-field harmonic-mean interface: the flux is one number by construction
    assert res.metrics["interface_local_flux_imbalance_rel"] < 1e-10
    # the Blasius streamfunction makes the discrete divergence vanish identically
    assert res.metrics["max_discrete_divergence"] < 1e-15


def test_flat_plate_isothermal_control_matches_boundary_layer():
    case = FlatPlateCHT.isothermal(re_l=1.0e5, pr=0.7, nx=120, ny_fluid=60)
    res = case.solve()
    assert res.metrics["nu_l2_rel_error"] < 0.03
    assert np.allclose(res.t_wall, 1.0)


def test_flat_plate_energy_ledger_closes():
    case = FlatPlateCHT(re_l=1.0e5, pr=0.7, solid_ratio=100.0, nx=100, ny_fluid=50,
                        ny_solid=10)
    res = case.solve()
    assert res.ledger["normalized_imbalance"] < 1e-9
    assert res.ledger["boundary_heat_rate"]["south"] > 0.0


# --------------------------------------------------------- interface conservation
def test_two_layer_slab_flux_continuity_is_machine_exact():
    """Harmonic-mean face conductance reproduces the exact piecewise-linear solution."""
    for ratio in (0.1, 1.0, 10.0):
        nx = 40
        xf = np.linspace(0.0, 1.0, 2 * nx + 1)
        dx = np.diff(xf)
        kappa = np.empty((1, 2 * nx))
        kappa[0, :nx] = 1.0
        kappa[0, nx:] = ratio
        T, info = solve_steady_energy(
            dx, np.array([1.0]), kappa, np.zeros((1, 2 * nx + 1)),
            np.zeros((2, 2 * nx)),
            {"west": {"type": "dirichlet", "values": 0.0},
             "east": {"type": "dirichlet", "values": 1.0}})
        cond = 1.0 / (0.5 * dx[nx - 1] / 1.0 + 0.5 * dx[nx] / ratio)
        q = cond * (T[0, nx - 1] - T[0, nx])
        t_iface = T[0, nx - 1] - 0.5 * dx[nx - 1] * q
        # interface temperature measured from the cold (west) wall: the drop is
        # q * R_west, i.e. the *west* resistance over the total -- subtracting the
        # east resistance from 1 gives the drop across the east half instead, and
        # inverts the answer whenever ratio != 1.
        exact = (0.5 / 1.0) / (0.5 / 1.0 + 0.5 / ratio)
        assert t_iface == pytest.approx(exact, abs=1e-9)
        assert info["ledger"].normalized_imbalance < 1e-10


def test_conservative_overlap_matrix_preserves_integral():
    for n_from, n_to in ((48, 48), (48, 32), (48, 96), (30, 44)):
        w = periodic_overlap_matrix(n_from, n_to)
        assert np.allclose(w.sum(axis=1), 1.0, atol=1e-12)
        # a constant field is transferred unchanged ...
        assert np.allclose(w @ np.ones(n_from), 1.0, atol=1e-12)
        # ... and the transferred *integral* is preserved whatever the ratio, i.e.
        # the mean of the target field equals the mean of the source field
        q = np.random.default_rng(0).normal(size=n_from)
        assert math.isclose(float(np.sum(w @ q) / n_to), float(np.mean(q)),
                            rel_tol=1e-12)


def test_orthogonal_grid_rejects_degenerate_spacing():
    with pytest.raises(ValueError):
        Grid2D(np.array([0.0, 1.0, 0.5]), np.array([0.0, 1.0]))


# --------------------------------------------------------------- polar cylinder
def test_polar_poisson_reproduces_potential_flow():
    """``lap(psi) = 0`` with the free stream recovers potential flow around a cylinder."""
    flow = PolarVorticitySolver(re=40.0, n_r=71, n_theta=48, r_out=12.0, wall_cell=1.0e-2)
    outer = flow.r_node[-1] * np.sin(flow.theta)
    psi = flow.solve_psi(np.zeros((flow.n_node, flow.n_theta)), outer)
    a = 0.5
    exact = (flow.r_node[:, None] - a ** 2 / flow.r_node[:, None]) * np.sin(flow.theta)
    assert np.max(np.abs(psi - exact)) / np.max(np.abs(exact)) < 0.12


def test_churchill_bernstein_reference_is_monotone():
    from tensorfvm.cht.runner import churchill_bernstein
    values = [churchill_bernstein(re, 0.7) for re in (40.0, 100.0, 200.0)]
    assert values == sorted(values)
    assert values[0] == pytest.approx(3.35, abs=0.05)


@pytest.mark.skipif(not SLOW, reason="coupled cross-flow CHT needs minutes")
def test_cylinder_conjugate_nusselt_against_churchill_bernstein():
    from tensorfvm.cht.runner import churchill_bernstein
    flow = PolarVorticitySolver(re=40.0, n_r=71, n_theta=72, r_out=15.0,
                                wall_cell=8.0e-3)
    march = flow.march(t_end=60.0, dt=2.0e-3, tol_steady=1e-7, max_steps=40000,
                       t_ramp=4.0)
    case = ConjugateCylinder(re=40.0, pr=0.7, solid_ratio=100.0, flow=flow)
    res = case.couple(march["psi"], relax=0.7, max_iter=200, tol=1e-8)
    reference = churchill_bernstein(40.0, 0.7)
    assert abs(res.nusselt_mean - reference) / reference < 0.15
    assert res.metrics["interface_local_residual_rel"] < 1e-6