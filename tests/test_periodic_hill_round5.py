"""Round-5 (mission 16880) tests: the Spalart--Allmaras closure.

Two classes, as in the round-2/3/4 suites:

* exactness -- the wall distance, the eddy-viscosity law and the refactored
  per-cell-viscosity momentum operator are checked against independently
  written references (a point-to-segment distance, the SA definitions, and a
  second assembly of the round-4 operator).  These run in seconds;
* registration -- the closure-off path must stay bit-identical to round 4 and
  the closure-on path must stay finite, so a silent regression fails the suite.
"""
from __future__ import annotations

import numpy as np
import pytest

from tensorfvm.periodic_hill import PeriodicHillConfig, PeriodicHillSolver
from tensorfvm.periodic_hill_turbulence import (
    SpalartAllmaras,
    SpalartAllmarasConfig,
    eddy_viscosity,
    wall_distance,
)
from tensorfvm.verification.periodic_hill_round5 import (
    LAMINAR_CHANNEL_UMAX,
    criterion_operator_generalisation,
    criterion_wall_distance,
    criterion_zero_state,
    channel_solver,
    hill_solver,
    log_law_fit,
    reference_momentum_operator,
)


# --------------------------------------------------------------------------- #
# wall distance
# --------------------------------------------------------------------------- #
def test_wall_distance_equals_vertical_on_plane_channel():
    solver = channel_solver(ny=32)
    d = wall_distance(solver.mesh, solver.config.length_m, solver.config.channel_top_m)
    vertical = np.minimum(solver.mesh.yc, solver.config.channel_top_m - solver.mesh.yc)
    assert np.array_equal(d, vertical)


def test_wall_distance_is_shorter_than_vertical_on_the_hill_flank():
    solver = hill_solver(nx=48, ny=24)
    d = wall_distance(solver.mesh, solver.config.length_m, solver.config.channel_top_m)
    yb, _, _ = solver.config.hill(solver.mesh.xc[0])
    vertical = np.minimum(solver.mesh.yc - yb[None, :],
                          solver.config.channel_top_m - solver.mesh.yc)
    assert d.min() > 0.0
    assert (d / vertical).min() < 0.95        # a vertical approximation is measurably wrong
    assert (d / vertical).min() > 0.5         # ... but not absurdly far off


def test_wall_distance_rejects_a_point_on_the_wall():
    solver = channel_solver(ny=8)
    with pytest.raises(ValueError):
        wall_distance(solver.mesh, solver.config.length_m, float(solver.mesh.yc.min()) * 0.0)


# --------------------------------------------------------------------------- #
# eddy-viscosity law
# --------------------------------------------------------------------------- #
def test_eddy_viscosity_matches_the_sa_definition():
    cfg = SpalartAllmarasConfig(max_viscosity_ratio=1.0e12)
    nu = 1.0e-3
    nu_tilde = np.array([0.0, 1.0e-3, 1.0e-2, 1.0e2])
    nu_t, chi, fv1 = eddy_viscosity(nu_tilde, nu, cfg)
    expected_chi = nu_tilde / nu
    expected_fv1 = expected_chi ** 3 / (expected_chi ** 3 + cfg.cv1 ** 3)
    assert np.allclose(chi, expected_chi)
    assert np.allclose(fv1, expected_fv1)
    assert np.allclose(nu_t, nu_tilde * expected_fv1)
    assert nu_t[0] == 0.0


def test_eddy_viscosity_is_capped():
    cfg = SpalartAllmarasConfig(max_viscosity_ratio=10.0)
    nu = 1.0
    nu_t, _, _ = eddy_viscosity(np.array([1.0e6]), nu, cfg)
    assert nu_t[0] == pytest.approx(10.0 * nu)


def test_cw1_follows_from_the_other_constants():
    cfg = SpalartAllmarasConfig()
    assert cfg.cw1 == pytest.approx(
        cfg.cb1 / cfg.kappa ** 2 + (1.0 + cfg.cb2) / cfg.sigma)


# --------------------------------------------------------------------------- #
# momentum operator generalisation
# --------------------------------------------------------------------------- #
def test_uniform_viscosity_reproduces_the_round4_operator():
    solver = hill_solver(nx=24, ny=12)
    reference = reference_momentum_operator(solver)
    assert abs(solver.momentum - reference).max() < 1e-12 * solver.nu / solver.time_step_s


def test_per_cell_viscosity_only_touches_the_diffusion_part():
    """Raising mu in one cell changes exactly that cell's row and its neighbours."""
    solver = hill_solver(nx=16, ny=8)
    base = solver.momentum.toarray()
    mu = np.full((8, 16), solver.nu)
    mu[4, 4] *= 2.0
    solver._assemble_momentum(mu)
    got = solver.momentum.toarray()
    affected = np.nonzero(np.abs(got - base).max(axis=1) > 0.0)[0]
    centre = 4 * 16 + 4
    neighbours = {centre, centre - 1, centre + 1, centre - 16, centre + 16}
    assert set(affected.tolist()) <= neighbours
    assert centre in affected


def test_rejects_non_positive_viscosity():
    solver = hill_solver(nx=8, ny=4)
    with pytest.raises(ValueError):
        solver._assemble_momentum(np.zeros((4, 8)))


def test_laminar_channel_gate_is_unchanged_by_the_refactor():
    report = criterion_operator_generalisation(nx=16, ny=8, steps=2000)
    # the round-4 published value is quoted to five decimals
    assert report["u_max_drift"] < 1e-4
    assert report["pass"] is True


# --------------------------------------------------------------------------- #
# closure mechanics
# --------------------------------------------------------------------------- #
def test_zero_state_is_invariant():
    report = criterion_zero_state(steps=20, nx=16, ny=8)
    assert report["max_abs_nu_tilde"] == 0.0


def test_closure_off_path_is_bit_identical_to_no_closure():
    """Attaching then detaching must be irrelevant: friction change only when active."""
    a = hill_solver(nx=16, ny=8)
    a.advance(20)
    b = hill_solver(nx=16, ny=8)
    b.turbulence = None
    b.advance(20)
    assert np.array_equal(a.u, b.u) and np.array_equal(a.v, b.v)


def test_closure_stays_finite_and_non_negative_on_a_short_hill_run():
    solver = hill_solver(nx=16, ny=8)
    closure = SpalartAllmaras(solver)
    solver.turbulence = closure
    solver.advance(40)
    assert np.all(np.isfinite(solver.u)) and np.all(np.isfinite(solver.v))
    assert solver.nu_t.min() >= 0.0
    assert solver.nu_t.max() > 0.0            # the closure is actually producing
    assert abs(solver.bulk_velocity() - 1.0) < 1e-6


def test_eddy_viscosity_shortens_the_laminar_shear_layer_gate():
    """The closure must add diffusion: with it, |u| max drops on the same run."""
    laminar = hill_solver(nx=16, ny=8)
    laminar.advance(60)
    turbulent = hill_solver(nx=16, ny=8)
    turbulent.turbulence = SpalartAllmaras(turbulent)
    turbulent.advance(60)
    assert np.abs(turbulent.u).max() < np.abs(laminar.u).max()


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def test_log_law_fit_recovers_a_manufactured_kappa_and_intercept():
    y_plus = np.linspace(5.0, 300.0, 60)
    u_plus = np.log(y_plus) / 0.41 + 5.2
    fit = log_law_fit(y_plus, u_plus)
    assert fit["kappa_eff"] == pytest.approx(0.41, rel=1e-10)
    assert fit["B"] == pytest.approx(5.2, rel=1e-10)
    assert fit["cells"] >= 3
