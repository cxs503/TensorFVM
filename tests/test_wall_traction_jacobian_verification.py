"""Finite-difference verification of the analytical wall-traction Jacobian.

Verification debt closed
------------------------
``tensorfvm.wall_functions.wall_traction_linearization`` returns an analytical
Jacobian of the Spalding wall-law traction with respect to the tangential
velocity. The derivation is non-trivial (implicit inversion of Spalding's law,
chain rule through the friction velocity, radial/tangential decomposition).

This module provides an *independent* verification: central finite differences
of the traction function itself, compared against the analytical Jacobian.
Agreement to O(eps^2) proves the analytical derivative is correct without
reading the derivation.

Scope
-----
* 2D and 3D tangential velocity vectors
* Viscous (y+ < 5), buffer (5 < y+ < 30), and log-law (y+ > 30) regimes
* Zero-speed limit (Jacobian must reduce to -mu/y * I)
* Multi-point arrays (broadcast correctness)

No claim is made for the wall model's physical accuracy; this verifies only
that the code's analytical derivative matches the code's own function.
"""
import numpy as np
import pytest

from tensorfvm.wall_functions import wall_traction, wall_traction_linearization


def finite_difference_jacobian(velocity, distance, density, viscosity, eps=1e-6):
    """Central finite-difference Jacobian of wall_traction w.r.t. velocity.
    
    Parameters
    ----------
    velocity : array_like, shape (..., d) where d in {2, 3}
    distance, density, viscosity : scalar
    eps : float
        Perturbation size for central differences.
    
    Returns
    -------
    jac_fd : ndarray, shape (..., d, d)
        d(traction)/d(velocity) by central differences.
    """
    v = np.asarray(velocity, dtype=float)
    shape = v.shape
    d = shape[-1]
    v_flat = v.reshape(-1, d)
    n = v_flat.shape[0]
    jac = np.zeros((n, d, d))
    
    for i in range(d):
        vp = v_flat.copy()
        vm = v_flat.copy()
        vp[:, i] += eps
        vm[:, i] -= eps
        tp = wall_traction(vp.reshape(shape), distance, density, viscosity)['traction_pa']
        tm = wall_traction(vm.reshape(shape), distance, density, viscosity)['traction_pa']
        tp_flat = tp.reshape(-1, d)
        tm_flat = tm.reshape(-1, d)
        jac[:, :, i] = (tp_flat - tm_flat) / (2 * eps)
    
    return jac.reshape(shape + (d,))


@pytest.mark.parametrize('speed', [0.001, 0.01, 0.1, 1.0, 10.0])
@pytest.mark.parametrize('dimension', [2, 3])
def test_analytical_jacobian_matches_finite_difference(speed, dimension):
    """Analytical Jacobian must agree with FD to ~1e-7 (central diff O(eps^2)).
    
    The Spalding law involves an implicit inversion (bisection on u_tau) and
    nonlinear functions (exp, log), so 7-digit agreement between analytical
    and finite-difference Jacobians is strong evidence of correctness.
    """
    np.random.seed(42)
    direction = np.random.randn(dimension)
    direction /= np.linalg.norm(direction)
    velocity = speed * direction
    distance = 1e-3  # 1 mm
    density = 1000.0
    viscosity = 1e-3  # water
    
    analytical = wall_traction_linearization(velocity, distance, density, viscosity)
    jac_analytical = analytical['traction_jacobian']
    jac_fd = finite_difference_jacobian(velocity, distance, density, viscosity, eps=1e-6)
    
    # Relative error normalized by the Jacobian magnitude
    scale = max(np.linalg.norm(jac_fd), 1.0)
    error = np.linalg.norm(jac_analytical - jac_fd) / scale
    
    assert error < 1e-6, f"speed={speed}, dim={dimension}: relative error {error}"


def test_zero_speed_limit_is_viscous():
    """At zero speed, Jacobian must be -mu/y * I (pure viscous, no turbulence)."""
    for dimension in (2, 3):
        velocity = np.zeros(dimension)
        distance = 1e-3
        density = 1000.0
        viscosity = 1e-3
        
        analytical = wall_traction_linearization(velocity, distance, density, viscosity)
        jac = analytical['traction_jacobian']
        
        expected = -(viscosity / distance) * np.eye(dimension)
        np.testing.assert_allclose(jac, expected, atol=1e-14)


def test_jacobian_is_symmetric_for_tangential_perturbation():
    """The Jacobian must be symmetric: d(traction_i)/d(v_j) = d(traction_j)/d(v_i).
    
    This follows from the traction being the gradient of a scalar dissipation
    potential (the friction velocity squared).
    """
    np.random.seed(123)
    velocity = np.array([0.5, 0.3, 0.1])
    distance = 1e-3
    density = 1000.0
    viscosity = 1e-3
    
    analytical = wall_traction_linearization(velocity, distance, density, viscosity)
    jac = analytical['traction_jacobian']
    
    # Check symmetry
    np.testing.assert_allclose(jac, jac.T, atol=1e-14)


def test_jacobian_broadcasts_over_multiple_points():
    """The Jacobian must correctly handle arrays of velocities."""
    np.random.seed(456)
    n_points = 5
    velocities = np.random.randn(n_points, 3) * 0.5
    distance = 1e-3
    density = 1000.0
    viscosity = 1e-3
    
    analytical = wall_traction_linearization(velocities, distance, density, viscosity)
    jac_batch = analytical['traction_jacobian']
    
    # Compare with point-by-point computation
    jac_individual = []
    for i in range(n_points):
        result = wall_traction_linearization(velocities[i], distance, density, viscosity)
        jac_individual.append(result['traction_jacobian'])
    jac_individual = np.array(jac_individual)
    
    np.testing.assert_allclose(jac_batch, jac_individual, atol=1e-14)


def test_energy_dissipation_is_nonpositive():
    """The Jacobian must satisfy v . (J @ v) <= 0 (traction resists motion).
    
    This is the discrete analogue of the continuous statement that the wall
    does negative work on the fluid (or zero at rest).
    """
    np.random.seed(789)
    for _ in range(10):
        velocity = np.random.randn(3) * np.random.uniform(0.01, 10.0)
        distance = 1e-3
        density = 1000.0
        viscosity = 1e-3
        
        analytical = wall_traction_linearization(velocity, distance, density, viscosity)
        jac = analytical['traction_jacobian']
        
        # v . (J @ v) should be non-positive
        power = velocity @ (jac @ velocity)
        assert power <= 1e-14, f"power = {power} > 0 (energy creation)"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
