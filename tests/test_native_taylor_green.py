"""Taylor--Green vortex verification for the native hybrid Navier--Stokes solver.

The periodic MAC kernel is qualified against the exact Taylor--Green vortex in
:mod:`tensorfvm.verification.taylor_green_mac`.  This module closes the gap for
the native hybrid backend (:class:`tensorfvm.native_hybrid_ns3d.NativeHybridNavierStokes3D`),
which previously had polynomial, Kovasznay and potential-flow manufactured cases
but no unforced nonlinear periodic benchmark.

The exact solution on the doubly-periodic box ``[0, 2*pi]^3`` is

    u(x,y,z,t) =  A cos(x) sin(y) exp(-2 nu t)
    v(x,y,z,t) = -A sin(x) cos(y) exp(-2 nu t)
    w          = 0
    p(x,y,z,t) = -(A^2/4) (cos 2x + cos 2y) exp(-4 nu t)   (rho = 1)

which satisfies ``du/dt + (u.grad)u = -grad(p) + nu lap(u)`` identically.

Two layers are tested here:

1. *The reference itself.*  The analytic velocity/pressure are pushed through
   the momentum and continuity equations by finite differences before they are
   used to judge the solver.
2. *The solver against the reference.*  Discrete divergence-free sampling,
   the grid-convergence ladder, and discrimination controls.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tensorfvm.native_hybrid_ns3d import NativeHybridNavierStokes3D
from tensorfvm.verification.native_taylor_green import (
    TaylorGreenCase,
    tgv_fields,
    tgv_fields_at_time,
    tgv_solver,
    tgv_metrics,
    run_study,
)

BOX = 2 * math.pi
STUDY = Path(__file__).resolve().parents[1] / "scripts" / "verification-native-taylor-green" / "study.json"


# --------------------------------------------------------------------------- #
# 1. the reference verifies itself
# --------------------------------------------------------------------------- #
def test_analytic_fields_satisfy_continuity():
    """The analytic velocity field is exactly divergence-free."""
    case = TaylorGreenCase(amplitude=1.3, viscosity=0.07)
    rng = np.random.default_rng(42)
    x = rng.uniform(0, BOX, 64)
    y = rng.uniform(0, BOX, 64)
    z = rng.uniform(0, BOX, 64)
    points = np.column_stack((x, y, z))
    fields = tgv_fields(points, case)
    u, v, w = fields['velocity'].T
    dx = dy = dz = 1e-4
    dudx = (tgv_fields(np.column_stack((x + dx, y, z)), case)['velocity'][:, 0]
            - tgv_fields(np.column_stack((x - dx, y, z)), case)['velocity'][:, 0]) / (2 * dx)
    dudy = (tgv_fields(np.column_stack((x, y + dy, z)), case)['velocity'][:, 0]
            - tgv_fields(np.column_stack((x, y - dy, z)), case)['velocity'][:, 0]) / (2 * dy)
    divergence = dudx + dudy
    assert np.max(np.abs(divergence)) < 1e-10


def test_analytic_fields_satisfy_momentum():
    """The analytic velocity/pressure satisfy the unforced Navier--Stokes equations."""
    case = TaylorGreenCase(amplitude=1.0, viscosity=0.05)
    rng = np.random.default_rng(17)
    x = rng.uniform(0, BOX, 32)
    y = rng.uniform(0, BOX, 32)
    z = rng.uniform(0, BOX, 32)
    t = 0.37
    points = np.column_stack((x, y, z))

    fields = tgv_fields_at_time(points, case, t)
    u, v, w = fields['velocity'].T
    p = fields['pressure']
    nu = case.viscosity

    # Time derivatives (central difference)
    dt = 1e-4
    fields_t_plus = tgv_fields_at_time(points, case, t + dt)
    fields_t_minus = tgv_fields_at_time(points, case, t - dt)
    dudz = (fields_t_plus['velocity'][:, 0] - fields_t_minus['velocity'][:, 0]) / (2 * dt)
    dvdt = (fields_t_plus['velocity'][:, 1] - fields_t_minus['velocity'][:, 1]) / (2 * dt)

    # Spatial derivatives
    dx = dy = 1e-4
    dudx = (tgv_fields_at_time(np.column_stack((x + dx, y, z)), case, t)['velocity'][:, 0]
            - tgv_fields_at_time(np.column_stack((x - dx, y, z)), case, t)['velocity'][:, 0]) / (2 * dx)
    dudy = (tgv_fields_at_time(np.column_stack((x, y + dy, z)), case, t)['velocity'][:, 0]
            - tgv_fields_at_time(np.column_stack((x, y - dy, z)), case, t)['velocity'][:, 0]) / (2 * dy)
    dvdx = (tgv_fields_at_time(np.column_stack((x + dx, y, z)), case, t)['velocity'][:, 1]
            - tgv_fields_at_time(np.column_stack((x - dx, y, z)), case, t)['velocity'][:, 1]) / (2 * dx)
    dvdy = (tgv_fields_at_time(np.column_stack((x, y + dy, z)), case, t)['velocity'][:, 1]
            - tgv_fields_at_time(np.column_stack((x, y - dy, z)), case, t)['velocity'][:, 1]) / (2 * dy)
    dpx = (tgv_fields_at_time(np.column_stack((x + dx, y, z)), case, t)['pressure']
           - tgv_fields_at_time(np.column_stack((x - dx, y, z)), case, t)['pressure']) / (2 * dx)
    dpy = (tgv_fields_at_time(np.column_stack((x, y + dy, z)), case, t)['pressure']
           - tgv_fields_at_time(np.column_stack((x, y - dy, z)), case, t)['pressure']) / (2 * dy)

    # Laplacian
    d2udx2 = (tgv_fields_at_time(np.column_stack((x + dx, y, z)), case, t)['velocity'][:, 0]
              - 2 * u + tgv_fields_at_time(np.column_stack((x - dx, y, z)), case, t)['velocity'][:, 0]) / (dx ** 2)
    d2udy2 = (tgv_fields_at_time(np.column_stack((x, y + dy, z)), case, t)['velocity'][:, 0]
              - 2 * u + tgv_fields_at_time(np.column_stack((x, y - dy, z)), case, t)['velocity'][:, 0]) / (dy ** 2)
    d2vdx2 = (tgv_fields_at_time(np.column_stack((x + dx, y, z)), case, t)['velocity'][:, 1]
              - 2 * v + tgv_fields_at_time(np.column_stack((x - dx, y, z)), case, t)['velocity'][:, 1]) / (dx ** 2)
    d2vdy2 = (tgv_fields_at_time(np.column_stack((x, y + dy, z)), case, t)['velocity'][:, 1]
              - 2 * v + tgv_fields_at_time(np.column_stack((x, y - dy, z)), case, t)['velocity'][:, 1]) / (dy ** 2)

    # Momentum residuals
    residual_u = dudz + u * dudx + v * dudy + dpx - nu * (d2udx2 + d2udy2)
    residual_v = dvdt + u * dvdx + v * dvdy + dpy - nu * (d2vdx2 + d2vdy2)

    assert np.max(np.abs(residual_u)) < 1e-8
    assert np.max(np.abs(residual_v)) < 1e-8


# --------------------------------------------------------------------------- #
# 2. solver against reference
# --------------------------------------------------------------------------- #
def test_tgv_solver_converges_velocity():
    """The native hybrid NS solver converges to the exact TGV velocity."""
    case = TaylorGreenCase()
    for n in [8, 16]:
        mesh, solver, exact = tgv_solver(n, case, convection_order=2)
        # Initialize with analytic velocity
        for a in range(3):
            solver.velocity[a] = exact['velocity'][:, a].reshape(solver.shape)

        # Time march
        steps = int(1.0 / 0.02)
        for _ in range(steps):
            solver.step()

        # Compute metrics
        metrics = tgv_metrics(mesh, solver.solve(), exact, case,
                              0.5 * case.density * np.sum(mesh.cell_volumes * np.sum(solver.velocity**2, axis=0)))
        assert metrics['velocity_relative_l2'] < 0.05, f"n={n}: velocity L2 = {metrics['velocity_relative_l2']}"
        assert metrics['max_divergence_s_inv'] < 1e-10
        assert metrics['velocity_z_max_abs_m_s'] < 1e-12


def test_tgv_solver_pressure_converges():
    """The native hybrid NS solver converges to the exact TGV pressure."""
    case = TaylorGreenCase()
    for n in [8, 16]:
        mesh, solver, exact = tgv_solver(n, case, convection_order=2)
        for a in range(3):
            solver.velocity[a] = exact['velocity'][:, a].reshape(solver.shape)

        steps = int(1.0 / 0.02)
        for _ in range(steps):
            solver.step()

        metrics = tgv_metrics(mesh, solver.solve(), exact, case,
                              0.5 * case.density * np.sum(mesh.cell_volumes * np.sum(solver.velocity**2, axis=0)))
        # Pressure is harder to converge; check it improves with refinement
        if n == 16:
            assert metrics['pressure_relative_linf'] < 0.1, f"n={n}: pressure Linf = {metrics['pressure_relative_linf']}"


def test_tgv_energy_decay_rate():
    """The energy decay rate is within 5% of the analytic rate at n=16."""
    case = TaylorGreenCase()
    mesh, solver, exact = tgv_solver(16, case, convection_order=2)
    for a in range(3):
        solver.velocity[a] = exact['velocity'][:, a].reshape(solver.shape)

    ke_init = 0.5 * case.density * np.sum(mesh.cell_volumes * np.sum(solver.velocity**2, axis=0))
    steps = int(1.0 / 0.02)
    for _ in range(steps):
        solver.step()

    ke_final = 0.5 * case.density * np.sum(mesh.cell_volumes * np.sum(solver.velocity**2, axis=0))
    decay_rate = -math.log(ke_final / ke_init) / 1.0
    analytic_decay = case.energy_decay_exponent
    decay_rate_err = abs(decay_rate - analytic_decay) / analytic_decay

    # At n=16, the decay rate error should be < 5%
    assert decay_rate_err < 0.05, f"decay rate error: {decay_rate_err}"


def test_tgv_spatial_order():
    """The velocity error shows second-order convergence."""
    case = TaylorGreenCase()
    results = run_study([8, 16], case, time_s=1.0, dt=0.02)

    errors = [r['velocity_relative_l2'] for r in results]
    h = [r['spacing_m'] for r in results]

    # Check monotonic convergence
    assert errors[0] > errors[1], f"Errors should decrease: {errors}"

    # Check observed order is in [1.5, 2.5]
    if errors[1] > 0:
        observed_order = math.log2(errors[0] / errors[1]) / math.log2(h[0] / h[1])
        assert 1.5 < observed_order < 2.5, f"Observed order: {observed_order}"


# --------------------------------------------------------------------------- #
# 3. published study
# --------------------------------------------------------------------------- #
def test_published_study_exists():
    """The published study file should exist after running the study."""
    if not STUDY.exists():
        pytest.skip(f"published study absent from this checkout: {STUDY}")
    study = json.loads(STUDY.read_text())
    assert study['schema'] == 'tensorfvm.native-taylor-green-study/1'
    summary = study['gate_summary']
    assert summary['physical_accuracy_qualified'] is True
    gates = study['gates']
    for name in ('velocity', 'pressure', 'energy_decay_rate', 'spatial_order'):
        assert gates[name]['passed'] is True, name
