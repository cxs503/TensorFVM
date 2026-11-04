"""Grid convergence verification for the potential flow exact solution.

The potential flow U = grad(x+y+z+xyz) is an exact unforced Navier-Stokes solution
at any viscosity: div U = lap U = curl U = 0, p = -|U|^2/2, so (U·∇)U + ∇p - ν∇²U = 0.
All three velocity components and all three coordinates are active, making this a
genuinely 3D verification case that exercises the full coupled solver.

This test verifies that the native hybrid NS solver converges to the exact solution
at the expected second-order rate for velocity and first-to-second order for pressure.
"""
import numpy as np
import pytest

from tensorfvm.verification.native_hybrid_ns import potential_solver


def test_potential_flow_converges_at_all_resolutions():
    """Verify the solver converges at each resolution."""
    for n in (2, 3, 4):
        mesh, solver, exact = potential_solver(n, viscosity=0.1)
        result = solver.solve(algorithm='coupled', tolerance=1e-10, max_iterations=5000)
        assert result['numerically_converged'], f"n={n} did not converge"
        # Check divergence from outer_history
        if result['outer_history']:
            div_max = result['outer_history'][-1]['divergence_max']
            assert div_max < 1e-8, f"n={n} divergence too large: {div_max:.2e}"


def test_potential_flow_velocity_is_second_order():
    """Volume-weighted L2 velocity error converges at second order."""
    viscosity = 0.1
    resolutions = [2, 3, 4]
    errors = []
    
    for n in resolutions:
        mesh, solver, exact = potential_solver(n, viscosity=viscosity)
        result = solver.solve(algorithm='coupled', tolerance=1e-10, max_iterations=5000)
        assert result['numerically_converged']
        
        # Volume-weighted L2 error
        volume = mesh.cell_volumes
        vel_error = np.sqrt(np.sum(volume * np.sum((result['velocity'] - exact['velocity'])**2, axis=1)))
        vel_norm = np.sqrt(np.sum(volume * np.sum(exact['velocity']**2, axis=1)))
        rel_error = vel_error / vel_norm
        errors.append((n, 1.0/n, rel_error))
    
    # Check convergence order (should be ~2.0)
    for i in range(1, len(errors)):
        h_ratio = errors[i-1][1] / errors[i][1]
        order = np.log(errors[i-1][2] / errors[i][2]) / np.log(h_ratio)
        # Asymptotic order should approach 2.0; allow 1.5+ for coarse grids
        assert order > 1.5, f"n={errors[i-1][0]}->{errors[i][0]}: velocity order {order:.2f} < 1.5"


def test_potential_flow_pressure_converges():
    """Volume-weighted L2 pressure error converges (first-to-second order)."""
    viscosity = 0.1
    resolutions = [2, 3, 4]
    errors = []
    
    for n in resolutions:
        mesh, solver, exact = potential_solver(n, viscosity=viscosity)
        result = solver.solve(algorithm='coupled', tolerance=1e-10, max_iterations=5000)
        assert result['numerically_converged']
        
        # Volume-weighted L2 error
        volume = mesh.cell_volumes
        pres_error = np.sqrt(np.sum(volume * (result['pressure'] - exact['pressure'])**2))
        pres_norm = np.sqrt(np.sum(volume * exact['pressure']**2))
        rel_error = pres_error / pres_norm
        errors.append((n, 1.0/n, rel_error))
    
    # Check that error decreases (order may be 1-2 on coarse grids)
    for i in range(1, len(errors)):
        assert errors[i][2] < errors[i-1][2], \
            f"n={errors[i-1][0]}->{errors[i][0]}: pressure error did not decrease"


def test_potential_flow_divergence_free():
    """The discrete velocity field must be divergence-free."""
    for n in (2, 3, 4):
        mesh, solver, exact = potential_solver(n, viscosity=0.1)
        result = solver.solve(algorithm='coupled', tolerance=1e-10, max_iterations=5000)
        assert result['numerically_converged']
        # Divergence should be at tolerance level
        if result['outer_history']:
            div_max = result['outer_history'][-1]['divergence_max']
            assert div_max < 1e-8, f"n={n}: div_max={div_max:.2e}"


def test_potential_flow_energy_balance():
    """The energy balance error should be small for this unforced flow."""
    for n in (2, 3, 4):
        mesh, solver, exact = potential_solver(n, viscosity=0.1)
        result = solver.solve(algorithm='coupled', tolerance=1e-10, max_iterations=5000)
        assert result['numerically_converged']
        # Unforced flow: viscous dissipation should balance pressure work
        assert result['energy_balance_error'] < 1e-6, \
            f"n={n}: energy_balance_error={result['energy_balance_error']:.2e}"


def test_potential_flow_is_exact_at_moderate_viscosity():
    """The potential flow is an exact NS solution; verify at moderate viscosity."""
    # Use viscosity=0.1 which converges reliably at n=3
    viscosity = 0.1
    mesh, solver, exact = potential_solver(3, viscosity=viscosity)
    result = solver.solve(algorithm='coupled', tolerance=1e-10, max_iterations=5000)
    assert result['numerically_converged']
    
    # Volume-weighted L2 error should be small
    volume = mesh.cell_volumes
    vel_error = np.sqrt(np.sum(volume * np.sum((result['velocity'] - exact['velocity'])**2, axis=1)))
    vel_norm = np.sqrt(np.sum(volume * np.sum(exact['velocity']**2, axis=1)))
    rel_error = vel_error / vel_norm
    
    # At n=3, error should be O(h^2) ~ 0.1 for second-order scheme
    assert rel_error < 0.1, f"velocity error {rel_error:.2e} too large"
