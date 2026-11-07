"""Kernel-level tests for the independent incompressible verification surface."""

import math

import pytest
import torch

from tensorfvm.inc_bench import cases
from tensorfvm.inc_bench.solver import (SolverOptions, SteadyFlow, cell_velocity,
                                        divergence, limiter_weight, shift, streamfunction)


def test_shift_matches_index_arithmetic():
    a = torch.arange(12.0).reshape(3, 4)
    east = shift(a, 1, -1)
    west = shift(a, 1, 1)
    north = shift(a, 0, -1)
    for i in range(3):
        for j in range(4):
            assert east[i, j] == (a[i, j + 1] if j + 1 < 4 else 0.0)
            assert west[i, j] == (a[i, j - 1] if j - 1 >= 0 else 0.0)
            assert north[i, j] == (a[i + 1, j] if i + 1 < 3 else 0.0)


def test_divergence_of_uniform_field_is_zero():
    ny, nx, dx, dy = 5, 7, 0.25, 0.5
    u = torch.full((ny, nx + 1), 2.0)
    v = torch.zeros((ny + 1, nx))
    assert float(divergence(u, v, dx, dy).abs().max()) == 0.0
    v = torch.full((ny + 1, nx), 3.0)
    assert float(divergence(u, v, dx, dy).abs().max()) == 0.0


def test_limiter_weight_bounded():
    upwind = torch.tensor([[-1e3, -1.0, 0.0, 1.0, 1e3]])
    downstream = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0]])
    weight = limiter_weight(upwind, downstream)
    assert torch.all(weight >= 0.0) and torch.all(weight <= 1.0)
    assert float(limiter_weight(torch.tensor([[1.0]]), torch.tensor([[0.0]]))) <= 1.0


def test_streamfunction_of_rigid_rotation_is_zero_net():
    # Uniform u with a zero transverse velocity gives psi = u * y.
    ny, nx, dy = 6, 4, 0.1
    u = torch.full((ny, nx + 1), 0.5, dtype=torch.float64)
    v = torch.zeros((ny + 1, nx), dtype=torch.float64)
    fluid = torch.ones((ny, nx), dtype=torch.bool)
    psi = streamfunction(u, v, dy, fluid)
    for i in range(ny):
        # cell-centred quadrature: the first cell contributes a full cell height
        assert abs(float(psi[i].mean()) - 0.5 * (i + 1) * dy) < 1e-12


def test_solver_options_reject_bad_relaxation():
    with pytest.raises(ValueError):
        SolverOptions(alpha_u=0.0)
    with pytest.raises(ValueError):
        SolverOptions(alpha_p=1.5)
    with pytest.raises(ValueError):
        SolverOptions(tolerance=0.0)


def test_steady_flow_validates_inputs():
    fluid = torch.ones((4, 4), dtype=torch.bool)
    with pytest.raises(ValueError):
        SteadyFlow(fluid.double(), dx=0.1, dy=0.1, viscosity=1.0)
    with pytest.raises(ValueError):
        SteadyFlow(fluid, dx=-0.1, dy=0.1, viscosity=1.0)
    with pytest.raises(ValueError):
        SteadyFlow(fluid, dx=0.1, dy=0.1, viscosity=1.0, inlet_profile=torch.ones(3))


def test_plane_channel_drag_balance():
    """Fully developed channel: total wall shear must balance the pressure drop.

    This is the first-principles closure that makes a friction coefficient
    meaningful: for steady plane Poiseuille flow the streamwise momentum balance
    gives ``tau_w = -(H/2) dp/dx`` independently of the discretisation.
    """
    ny, nx = 12, 12
    dy = 1.0 / ny
    dx = dy
    fluid = torch.ones((ny, nx), dtype=torch.bool)
    y = (torch.arange(ny) + 0.5) * dy
    profile = 6.0 * y * (1.0 - y)
    options = SolverOptions(max_iterations=1200, tolerance=1e-10, min_iterations=300,
                            check_every=100)
    solver = SteadyFlow(fluid, dx=dx, dy=dy, viscosity=2.0 / 50.0,
                        inlet_profile=profile, outlet=True, options=options)
    solver.solve()
    uc, _ = cell_velocity(solver.u, solver.v)
    # Momentum balance on a developed pair of columns away from the outlet, where
    # the zero gauge pressure is imposed and the gradient is not physical.
    dp = float((solver.p[:, -4] - solver.p[:, -6]).mean()) / (2 * dx)
    tau_wall = 2.0 * solver.nu * float(solver.u[0, -4]) / dy
    balance = abs(tau_wall + 0.5 * dp) / abs(tau_wall)
    assert balance < 0.05, f"wall shear / pressure-drop balance off by {balance:.1%}"
    # The developed profile is the analytic paraboloid to within a few percent.
    developed = float(uc[:, -4].max())
    assert abs(developed - 1.5) / 1.5 < 0.05


def test_cavity_nusselt_and_energy_balance_smoke():
    """Coarse differentially heated cavity: plausible Nu and closed energy budget."""
    setup = cases.natural_cavity(1.0e3, level="coarse")
    options = SolverOptions(max_iterations=600, tolerance=1e-8, min_iterations=200,
                            check_every=100)
    solver = setup.solver(options)
    record = solver.solve()
    metrics = cases.cavity_metrics(solver, setup)
    assert record["iterations"] > 100
    assert 1.0 < metrics["nusselt_hot"] < 2.0, metrics["nusselt_hot"]
    assert metrics["energy_balance_rel"] < 0.05, metrics["energy_balance_rel"]
    assert math.isfinite(metrics["psi_max"]) and metrics["psi_max"] > 0.5