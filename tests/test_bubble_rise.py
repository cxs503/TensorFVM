"""Tests for the 2-D bubble-rise benchmark and the solver paths it needs.

Fast unit/integration tests (no long runs): the quantitative benchmark itself is
``src/tensorfvm/verification/bubble_rise.py`` and takes minutes; these tests pin
the pieces it relies on -- the case algebra, the advective operator, the
pressure/gravity split, conservative PLIC transport, the field metrics and the
criteria gate (including that the gate *can* fail).
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorfvm.two_phase_solver import GridConfig, TwoPhaseConfig, TwoPhaseSolver
from tensorfvm.verification import bubble_rise as br


# --------------------------------------------------------------------------- #
# Case algebra and analytic anchors
# --------------------------------------------------------------------------- #


def test_case_groups_round_trip_and_anchors():
    case = br.BubbleCase(name="t", eotvos=1.23, morton=2.63e-11,
                         density_ratio=10.0, cells_per_diameter=12)
    # sigma is built from Eo, mu_l from M: both must round-trip.
    assert case.sigma == pytest.approx(1.0 / 1.23, rel=1e-15)
    delta_rho = 1.0 - 1.0 / case.density_ratio
    morton_back = 1.0 * case.mu_liquid ** 4 * delta_rho / case.sigma ** 3
    assert morton_back == pytest.approx(case.morton, rel=1e-12)
    # Fan-Tsuchiya in bubble units is the dimensional correlation with
    # sigma -> 1/Eo, rho_l g d -> 1.
    u = case.fan_tsuchiya_velocity()
    assert u == pytest.approx(math.sqrt(2.14 * case.sigma + 0.505), rel=1e-14)
    # Hadamard-Rybczynski must reduce to the clean-bubble Stokes value
    # drho g d^2/(12 mu_l) when mu_g -> 0 (a = d/2).
    u_hr = br.hadamard_rybczynski_velocity(1.23, 2.63e-11, 10.0, 1e9)
    stokes = delta_rho * 1.0 * 1.0 / (12.0 * case.mu_liquid)
    assert u_hr == pytest.approx(stokes, rel=1e-8)


def test_clift_regime_windows():
    assert br.clift_regime(0.1, 1e-11, 0.5)["regime"] == "spherical"
    assert br.clift_regime(1.23, 2.6e-11, 500.0)["regime"] == "ellipsoidal"
    assert br.clift_regime(45.0, 1e-11, 5.0e3)["regime"] == "spherical-cap"
    # the raw numbers are echoed, so a disagreement does not lose the data
    rec = br.clift_regime(1.23, 2.6e-11, 500.0)
    assert rec["eotvos"] == 1.23 and rec["reynolds"] == 500.0


# --------------------------------------------------------------------------- #
# Momentum advection operator
# --------------------------------------------------------------------------- #


def _solver(nx=8, ny=16, **tp_kwargs):
    grid = GridConfig(nx=nx, ny=ny, length=1.0, height=2.0)
    base = dict(enabled=True, rho_liquid=1.0, rho_gas=0.1, mu_liquid=1e-2,
                mu_gas=1e-3, sigma=0.0, surface_tension=False, gravity=0.0,
                top_atmosphere=True, bottom_wall=True)
    base.update(tp_kwargs)
    return TwoPhaseSolver(grid, TwoPhaseConfig(**base))


def test_advective_accel_is_zero_for_uniform_flow():
    solver = _solver()
    solver.u.fill_(0.7)
    solver.v.fill_(0.0)
    adv_u, adv_v = solver._advective_accel()
    assert adv_u.shape == (solver.ny - 2, solver.nx - 1)
    assert adv_v.shape == (solver.ny - 1, solver.nx - 2)
    assert float(adv_u.abs().max()) < 1e-14
    assert float(adv_v.abs().max()) < 1e-14


def test_advective_accel_matches_advective_derivative():
    # u = A*y (linear), v = B (uniform): u.dudx + v.dudy = 0 + B*A.
    solver = _solver()
    A, B = 2.0, 0.5
    y_faces = (torch.arange(solver.ny, dtype=torch.float64) + 0.5) * solver.dy
    solver.u.copy_(A * y_faces[:, None].expand_as(solver.u))
    solver.v.fill_(B)
    adv_u, _ = solver._advective_accel()
    assert float(adv_u.min()) == pytest.approx(A * B, rel=1e-12)
    assert float(adv_u.max()) == pytest.approx(A * B, rel=1e-12)


# --------------------------------------------------------------------------- #
# Pressure / gravity split (the defect the benchmark had to fix)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("pressure_form,limit", [("reduced", 1e-9), ("total", 1e-2)])
def test_hydrostatic_pool_at_rest(pressure_form, limit):
    """A liquid pool at rest is an exact solution -- but only with the reduced form.

    With ``pressure_form="total"`` the incremental projection annihilates the
    hydrostatic head (a uniformly-divergent-free ``(1/rho) grad p``), so the pool
    free-falls; that is the measured defect, not a numerical accident.
    """
    solver = _solver(nx=16, ny=32, pressure_form=pressure_form, gravity=1.0)
    solver.set_alpha(torch.ones((solver.ny, solver.nx), dtype=torch.float64))
    for _ in range(20):
        solver.step(solver.stable_time_step())
    vmax = float(solver.v.abs().max())
    if pressure_form == "reduced":
        assert vmax < limit
    else:
        assert vmax > limit


# --------------------------------------------------------------------------- #
# Conservative PLIC interface transport inside the solver
# --------------------------------------------------------------------------- #


def test_plic_transport_bounded_and_conserving_in_solver():
    solver = _solver(nx=16, ny=32, vof_scheme="plic")
    x = (torch.arange(solver.nx) + 0.5) * solver.dx
    y = (torch.arange(solver.ny) + 0.5) * solver.dy
    bubble = (((x[None, :] - 0.5) ** 2 + (y[:, None] - 1.0) ** 2) <= 0.09).double()
    solver.set_alpha(1.0 - bubble)
    volume0 = float(solver.alpha.sum())
    # A divergence-free, spatially varying velocity field drives the sweeps.
    xu = (torch.arange(solver.nx + 1) + 0.5) * solver.dx
    solver.u.copy_(0.4 * torch.sin(math.pi * xu)[None, :].expand(solver.ny, -1))
    solver.v.zero_()
    for _ in range(15):
        solver._advect_vof(1e-3)
        assert float(solver.alpha.min()) >= 0.0
        assert float(solver.alpha.max()) <= 1.0
    drift = abs(float(solver.alpha.sum()) - volume0) / volume0
    assert drift < 1e-10


# --------------------------------------------------------------------------- #
# Field metrics
# --------------------------------------------------------------------------- #


def test_bubble_metrics_on_analytic_disk_and_ellipse():
    nx, ny, dx, dy = 160, 160, 0.25, 0.25
    x = (torch.arange(nx) + 0.5) * dx
    y = (torch.arange(ny) + 0.5) * dy
    X, Y = torch.meshgrid(x, y, indexing="xy")
    disk = (((X - 20.0) ** 2 + (Y - 20.0) ** 2) <= 4.0).double()
    m = br.bubble_metrics(disk, dx, dy)
    assert m["centroid"][0] == pytest.approx(20.0, abs=0.05)
    assert m["centroid"][1] == pytest.approx(20.0, abs=0.05)
    assert m["d_equivalent"] == pytest.approx(4.0, rel=0.02)
    assert m["aspect_ratio"] == pytest.approx(1.0, abs=0.06)
    assert br.second_moment_aspect(disk, dx, dy) == pytest.approx(1.0, abs=0.05)

    ellipse = ((((X - 20.0) / 2.0) ** 2 + ((Y - 20.0) / 1.0) ** 2) <= 1.0).double()
    m2 = br.bubble_metrics(ellipse, dx, dy)
    assert m2["aspect_ratio"] == pytest.approx(2.0, rel=0.08)
    assert br.second_moment_aspect(ellipse, dx, dy) == pytest.approx(2.0, rel=0.08)
    assert m2["circularity"] < m["circularity"]


# --------------------------------------------------------------------------- #
# Criteria gate -- must be able to fail
# --------------------------------------------------------------------------- #


def _fake_run(terminal, aspect=1.3, circularity=0.95, worst=1e-16, total=1e-16, n=16):
    return {
        "case": {"eotvos": 1.23, "morton": 2.63e-11},
        "grid": {"cells_per_diameter": n},
        "geometry_validation": {"relative_error": 1e-3},
        "terminal_velocity": terminal,
        "terminal_velocity_anchor": br.fan_tsuchiya_velocity(1.23),
        "dimensionless": {"eotvos": 1.23, "morton": 2.63e-11, "reynolds": 500.0},
        "regime": {"regime": "ellipsoidal"},
        "shape": {"aspect_ratio": aspect, "aspect_second_moment": aspect,
                  "circularity": circularity, "d_equivalent": 1.0,
                  "centroid": [2.0, 3.0]},
        "conservation": {"worst_step_relative_drift": worst,
                         "total_relative_drift": total},
    }


def test_criteria_gate_accepts_and_rejects():
    anchor = br.fan_tsuchiya_velocity(1.23)
    good = [_fake_run(anchor * 0.9, n=12), _fake_run(anchor * 0.97, n=16),
            _fake_run(anchor * 0.99, n=20)]
    crit = br.evaluate_criteria(good)
    assert crit["C4_terminal_velocity"]["pass"] is True
    assert crit["C1_shape_band"]["pass"] is True
    assert crit["C2_mass_conservation"]["pass"] is True
    assert crit["C3_grid_convergence"]["pass"] is True
    assert crit["all_pass"] is True

    bad = [_fake_run(anchor * 0.2, n=12), _fake_run(anchor * 0.25, n=16),
           _fake_run(anchor * 0.3, aspect=3.0, circularity=0.4, worst=1e-3, n=20)]
    crit_bad = br.evaluate_criteria(bad)
    assert crit_bad["C4_terminal_velocity"]["pass"] is False
    assert crit_bad["C1_shape_band"]["pass"] is False
    assert crit_bad["C2_mass_conservation"]["pass"] is False
    assert crit_bad["all_pass"] is False
    # the declared tolerance is recorded with the number, not implied
    assert crit_bad["C4_terminal_velocity"]["declared_tolerance"] == (
        br.TERMINAL_VELOCITY_TOLERANCE)


# --------------------------------------------------------------------------- #
# Defensive contracts
# --------------------------------------------------------------------------- #


def test_negative_inputs_are_rejected():
    with pytest.raises(ValueError):
        br.BubbleCase(name="x", eotvos=-1.0, morton=1e-11)
    with pytest.raises(ValueError):
        br.BubbleCase(name="x", eotvos=1.0, morton=0.0)
    with pytest.raises(ValueError):
        br.BubbleCase(name="x", eotvos=1.0, morton=1e-11, cells_per_diameter=4)
    with pytest.raises(ValueError):
        br.BubbleCase(name="x", eotvos=1.0, morton=1e-11, domain_height=3.0)
    with pytest.raises(ValueError):
        br.fan_tsuchiya_velocity(0.0)
    with pytest.raises(ValueError):
        TwoPhaseConfig(vof_scheme="plicky")
    with pytest.raises(ValueError):
        TwoPhaseConfig(pressure_form="dynamic")


def test_stable_time_step_respects_its_limits():
    solver = _solver(nx=16, ny=32, sigma=1.0, surface_tension=True,
                     mu_liquid=1.0, mu_gas=0.1)
    solver.set_alpha(torch.ones((solver.ny, solver.nx), dtype=torch.float64))
    solver.u.fill_(2.0)
    dt = solver.stable_time_step(cfl=0.4, capillary_safety=0.5, viscous_safety=0.25)
    h = min(solver.dx, solver.dy)
    dt_cfl = 0.4 * h / 2.0
    dt_cap = 0.5 * math.sqrt(solver.props.rho_liquid * h ** 3
                             / (2.0 * math.pi * solver.props.sigma))
    assert 0.0 < dt <= min(dt_cfl, dt_cap) + 1e-15
    # and a stiff capillary case must return a smaller step than a slack one
    solver2 = _solver(nx=16, ny=32, sigma=100.0, surface_tension=True)
    assert solver2.stable_time_step() < dt
