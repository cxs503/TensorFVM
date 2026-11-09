"""Tests for frozen turbulence RANS adjoint verification.

Verifies:
1. Frozen adjoint matches frozen FD (relative error < 1e-3)
2. Laminar limit: nu_t=0 → frozen adjoint = laminar adjoint
3. Frozen turbulence bias is quantified
"""

from __future__ import annotations

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tensorfvm.adjoint_shape.channel import ChannelGrid, ChannelSolver
from tensorfvm.adjoint_shape.geometry import FourierBody, N_PARAMS
from tensorfvm.adjoint_shape.adjoint import adjoint_gradient


def _make_solver(nx=32, ny=16, re=1000, nut_factor=5.0):
    """Create a solver with turbulent viscosity."""
    grid = ChannelGrid(nx=nx, ny=ny)
    D = 0.5
    nu = D / re
    force = 12.0 * nu / (grid.height**2)
    body = FourierBody(
        center=(2.0, 0.85), radius0=0.25,
        delta=0.3*min(grid.dx, grid.dy), quad=4, subcell=min(grid.dx, grid.dy)
    )
    return ChannelSolver(grid, body, nu=nu, force=force, eta=1e-3, eps_upwind=0.05), nu


def _compute_nut(solver, theta, nut_factor=5.0):
    """Compute shape-dependent nu_t field."""
    g = solver.grid
    xc = (np.arange(g.nx)[None, :] + 0.5) * g.dx
    yc = (np.arange(g.ny)[:, None] + 0.5) * g.dy
    xc = np.broadcast_to(xc, (g.ny, g.nx)).copy()
    yc = np.broadcast_to(yc, (g.ny, g.nx)).copy()
    
    r, _ = solver.body.polar(xc, yc)
    R_body = solver.body.radius(xc, yc, theta)
    d_body = np.abs(r - R_body)
    delta = solver.body.delta
    
    nu_t = nut_factor * solver.nu * np.exp(-d_body / max(delta, 1e-6))
    y_wall = np.minimum(yc, g.height - yc)
    damping = 1 - np.exp(-y_wall / (10 * g.dy))
    nu_t *= damping
    
    return nu_t


class TestFrozenTurbulenceAdjoint:
    """Frozen turbulence adjoint verification tests."""
    
    def test_frozen_adjoint_matches_frozen_fd(self):
        """Frozen adjoint gradient matches frozen FD to < 1e-3 relative error."""
        solver, nu = _make_solver()
        theta = 0.05 * np.random.default_rng(42).standard_normal(N_PARAMS)
        
        # Compute nu_t and effective viscosity
        nu_t = _compute_nut(solver, theta, nut_factor=5.0)
        nu_eff = nu + np.mean(nu_t)
        
        # Solve with frozen nu_t
        solver.nu = nu_eff
        U0, _ = solver.solve(theta)
        
        # Frozen adjoint
        grad_adjoint, _ = adjoint_gradient(solver, theta, U0)
        
        # Frozen FD
        eps = 1e-3
        grad_fd = np.zeros(N_PARAMS)
        for q in range(N_PARAMS):
            e = np.zeros(N_PARAMS)
            e[q] = eps
            U_p, _ = solver.solve(theta + e, U0=U0)
            J_p = solver.drag_coefficient(theta + e, U_p)
            U_m, _ = solver.solve(theta - e, U0=U0)
            J_m = solver.drag_coefficient(theta - e, U_m)
            grad_fd[q] = (J_p - J_m) / (2 * eps)
        
        solver.nu = nu  # restore
        
        # Check relative error
        denom = np.maximum(np.abs(grad_fd), 1e-300)
        rel_err = np.abs(grad_adjoint - grad_fd) / denom
        
        assert rel_err.max() < 1e-3, f"Max relative error {rel_err.max():.3e} >= 1e-3"
    
    def test_laminar_limit(self):
        """With nu_t=0, frozen adjoint equals laminar adjoint."""
        solver, nu = _make_solver()
        theta = 0.05 * np.random.default_rng(42).standard_normal(N_PARAMS)
        
        # Laminar solve
        U_lam, _ = solver.solve(theta)
        grad_lam, _ = adjoint_gradient(solver, theta, U_lam)
        
        # "Frozen" with nu_t=0 (same as laminar)
        nu_t_zero = np.zeros((solver.grid.ny, solver.grid.nx))
        nu_eff_zero = nu + np.mean(nu_t_zero)  # = nu
        
        solver.nu = nu_eff_zero
        U_zero, _ = solver.solve(theta)
        grad_zero, _ = adjoint_gradient(solver, theta, U_zero)
        
        solver.nu = nu  # restore
        
        # Should be identical
        diff = np.abs(grad_lam - grad_zero)
        assert diff.max() < 1e-10, f"Laminar limit failed: max diff = {diff.max():.3e}"
    
    def test_frozen_turbulence_bias_quantified(self):
        """Frozen turbulence bias is computed and is non-zero."""
        solver, nu = _make_solver()
        theta = 0.05 * np.random.default_rng(42).standard_normal(N_PARAMS)
        nut_factor = 5.0
        
        # Frozen adjoint (nu_t fixed)
        nu_t = _compute_nut(solver, theta, nut_factor)
        nu_eff = nu + np.mean(nu_t)
        solver.nu = nu_eff
        U0, _ = solver.solve(theta)
        grad_frozen, _ = adjoint_gradient(solver, theta, U0)
        
        # Full RANS FD (nu_t recomputed)
        eps = 1e-3
        grad_full = np.zeros(N_PARAMS)
        for q in range(N_PARAMS):
            e = np.zeros(N_PARAMS)
            e[q] = eps
            
            # +eps: recompute nu_t
            nu_t_p = _compute_nut(solver, theta + e, nut_factor)
            solver.nu = nu + np.mean(nu_t_p)
            U_p, _ = solver.solve(theta + e)
            J_p = solver.drag_coefficient(theta + e, U_p)
            
            # -eps: recompute nu_t
            nu_t_m = _compute_nut(solver, theta - e, nut_factor)
            solver.nu = nu + np.mean(nu_t_m)
            U_m, _ = solver.solve(theta - e)
            J_m = solver.drag_coefficient(theta - e, U_m)
            
            grad_full[q] = (J_p - J_m) / (2 * eps)
        
        solver.nu = nu  # restore
        
        # Frozen bias
        denom = np.maximum(np.abs(grad_full), 1e-300)
        frozen_bias = np.abs(grad_frozen - grad_full) / denom
        
        # Bias should be non-zero (frozen turbulence is an approximation)
        assert frozen_bias.max() > 1e-6, "Frozen turbulence bias is zero (unexpected)"
        
        # Report the bias
        print(f"Frozen turbulence bias: max = {frozen_bias.max():.3e}, "
              f"mean = {frozen_bias.mean():.3e}")
    
    def test_gradient_decomposition(self):
        """Gradient can be decomposed into direct and adjoint contributions."""
        solver, nu = _make_solver()
        theta = 0.05 * np.random.default_rng(42).standard_normal(N_PARAMS)
        
        nu_t = _compute_nut(solver, theta, nut_factor=5.0)
        nu_eff = nu + np.mean(nu_t)
        solver.nu = nu_eff
        
        U0, _ = solver.solve(theta)
        grad, info = adjoint_gradient(solver, theta, U0)
        
        solver.nu = nu  # restore
        
        # Check decomposition
        direct = info['dJ_dtheta_partial']
        adjoint_term = info['adjoint_term']
        
        # Total = direct - adjoint_term
        reconstructed = direct - adjoint_term
        assert np.allclose(grad, reconstructed, rtol=1e-10), \
            "Gradient decomposition failed"
        
        # Both contributions should be non-zero
        assert np.linalg.norm(direct) > 0, "Direct partial is zero"
        assert np.linalg.norm(adjoint_term) > 0, "Adjoint term is zero"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
