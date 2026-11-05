"""Verification of NativeHybridNavierStokes3D pressure gauge consistency.

The native hybrid NS solver pins the pressure null-space via a single cell
gauge.  This test proves three things that are easy to break silently:

1. The gauge value is exactly reproduced in the target cell (within linear
   solver tolerance).
2. Changing the gauge value shifts the pressure field by a constant but leaves
   velocity, flux and all derived quantities unchanged.
3. The global momentum balance closes to machine precision regardless of the
   gauge choice.
"""
import numpy as np
import pytest

from tensorfvm.native_hybrid_ns3d import NativeHybridNavierStokes3D
from tensorfvm.verification.native_hybrid_flow import channel_mesh


def _make_hydrostatic_solver(gauge_value):
    """Build a hydrostatic channel flow solver with a fixed gauge."""
    mesh = channel_mesh(2)
    g = np.array([.7, -.3, .2])
    p_exact = .4 + mesh.cell_centers @ g
    # Pin gauge to the first cell
    b = mesh.neighbor < 0
    fixed = b.copy()
    return NativeHybridNavierStokes3D(
        mesh, fixed, body_force=g, gauge_value=gauge_value,
        viscosity=1.0, backflow_velocity=np.zeros(3)
    )


def test_gauge_value_reproduced_in_target_cell():
    """The pinned cell must carry the exact gauge value."""
    for gv in (0.0, 1.5, -2.3, 100.0):
        s = _make_hydrostatic_solver(gv)
        r = s.solve(algorithm='coupled', tolerance=1e-10, max_outer_iterations=20)
        assert r['numerically_converged'], r['outer_history'][-1]
        # Gauge is applied to cell 0
        np.testing.assert_allclose(r['pressure'][0], gv, atol=1e-9)


def test_gauge_shift_leaves_velocity_unchanged():
    """Pressure is defined up to a constant; velocity must be invariant."""
    s0 = _make_hydrostatic_solver(0.0)
    r0 = s0.solve(algorithm='coupled', tolerance=1e-10, max_outer_iterations=20)
    s1 = _make_hydrostatic_solver(5.0)
    r1 = s1.solve(algorithm='coupled', tolerance=1e-10, max_outer_iterations=20)
    # Velocity, face velocity and flux must be identical
    np.testing.assert_allclose(r0['velocity'], r1['velocity'], atol=1e-10)
    np.testing.assert_allclose(r0['face_velocity'], r1['face_velocity'], atol=1e-10)
    np.testing.assert_allclose(r0['flux'], r1['flux'], atol=1e-10)
    # Pressure should differ by exactly the gauge shift
    np.testing.assert_allclose(r1['pressure'] - r0['pressure'], 5.0, atol=1e-9)


def test_momentum_balance_independent_of_gauge():
    """Integrated traction + convection must equal body force regardless of gauge."""
    mesh = channel_mesh(2)
    g = np.array([.7, -.3, .2])
    balances = []
    for gv in (-10.0, 0.0, 10.0):
        s = _make_hydrostatic_solver(gv)
        r = s.solve(algorithm='coupled', tolerance=1e-10, max_outer_iterations=20)
        assert r['numerically_converged'], r['outer_history'][-1]
        boundary_momentum = r['momentum_flux_integrated'][mesh.neighbor < 0].sum(0)
        applied_force = (mesh.cell_volumes[:, None] * g).sum(0)
        balances.append(float(np.linalg.norm(boundary_momentum - applied_force)))
    # All balances must be near zero and identical
    assert max(balances) < 1e-8
    assert max(balances) - min(balances) < 1e-12


def test_energy_balance_closes_with_arbitrary_gauge():
    """Energy identity must hold for any gauge choice."""
    for gv in (-5.0, 0.0, 7.5):
        s = _make_hydrostatic_solver(gv)
        r = s.solve(algorithm='coupled', tolerance=1e-10, max_outer_iterations=20)
        assert r['numerically_converged'], r['outer_history'][-1]
        assert r['energy_balance_error'] < 1e-8, f"gv={gv}, err={r['energy_balance_error']}"
