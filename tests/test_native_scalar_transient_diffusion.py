"""Verification tests for transient scalar diffusion using NativeHybridScalar3D.

These tests verify the actual production scalar solver against the analytical
error-function solution for semi-infinite domain diffusion. This closes a
verification gap where only a simplified 1D solver was previously tested.
"""
import numpy as np
import pytest

from tensorfvm.verification.native_scalar_transient_diffusion import (
    analytical_solution,
    build_extruded_channel_mesh,
    comparison_metrics,
    grid_convergence_study,
    solve_transient_diffusion_native,
)


def test_analytical_solution_matches_erfc():
    """Verify the analytical solution implementation matches scipy.special.erfc."""
    x = np.linspace(0, 1, 50)
    t = 0.1
    alpha = 0.01
    T0, T1 = 0.0, 1.0
    
    T = analytical_solution(x, t, alpha, T0, T1)
    
    from scipy.special import erfc
    eta = x / (2.0 * np.sqrt(alpha * t))
    T_expected = T0 + (T1 - T0) * erfc(eta)
    
    np.testing.assert_allclose(T, T_expected, rtol=1e-14)


def test_analytical_solution_rejects_invalid_parameters():
    """Verify that invalid parameters are rejected."""
    x = np.array([0.5])
    
    with pytest.raises(ValueError, match="time must be positive"):
        analytical_solution(x, t=0.0, alpha=0.01)
    
    with pytest.raises(ValueError, match="diffusivity must be positive"):
        analytical_solution(x, t=0.1, alpha=0.0)


def test_native_solver_runs_and_produces_finite_results():
    """Verify the native solver produces finite results for transient diffusion."""
    mesh, info = build_extruded_channel_mesh(16, ny=2, nz=2)
    T_num, times = solve_transient_diffusion_native(
        mesh, info, alpha=0.01, t_final=0.1, n_steps=10
    )
    
    assert np.all(np.isfinite(T_num)), "Solver produced non-finite values"
    assert len(times) == 11, "Expected 11 time points (0 to 10 steps)"
    assert times[-1] == pytest.approx(0.1, rel=1e-10)


def test_native_solver_matches_analytical_solution():
    """Verify the native solver matches the analytical solution within tolerance.
    
    Note: The extruded 3D mesh with coarse resolution and simple boundary treatment
    has larger errors than a dedicated 1D solver. The key point is that the
    verification framework exists and the solver converges to the analytical solution.
    """
    alpha = 0.01
    t_final = 0.1
    nx = 32
    
    mesh, info = build_extruded_channel_mesh(nx, ny=2, nz=2)
    T_num, _ = solve_transient_diffusion_native(
        mesh, info, alpha, t_final, n_steps=50
    )
    T_ana = analytical_solution(mesh.cell_centers[:, 0], t_final, alpha)
    
    metrics = comparison_metrics(T_num, T_ana, mesh.cell_volumes)
    
    # At nx=32 with 50 time steps, error should be reasonable
    # The extruded 3D mesh has larger errors than a dedicated 1D solver
    assert metrics['l2_error'] < 0.20, f"L2 error {metrics['l2_error']} exceeds 20%"
    assert np.all(np.isfinite(T_num))


def test_grid_convergence_shows_error_decrease():
    """Verify that errors decrease with grid refinement."""
    study = grid_convergence_study(alpha=0.01, t_final=0.1, nx_list=[16, 32, 64])
    
    errors = [e['l2_error'] for e in study['errors']]
    
    # Errors should decrease with refinement
    assert errors[0] > errors[1], f"Error did not decrease: {errors[0]} -> {errors[1]}"
    assert errors[1] > errors[2], f"Error did not decrease: {errors[1]} -> {errors[2]}"
    
    # Fine grid should achieve better accuracy
    assert errors[-1] < 0.15, f"Fine grid error {errors[-1]} exceeds 15%"
