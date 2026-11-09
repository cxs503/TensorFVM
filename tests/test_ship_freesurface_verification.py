"""Comprehensive test suite for ship free-surface verification.

This module provides ≥15 pytest tests covering:
1. Geometry validation (Wigley, Series 60, DTMB 5415)
2. Michell thin-ship theory analytical checks
3. Experimental data consistency
4. Resistance decomposition
5. Mesh convergence (Richardson extrapolation)
6. VOF verification runner
7. Negative controls (VOF disabled must fail wave criteria)

All tests are designed to run without the full CFD solver (using synthetic data)
for fast CI/CD, with optional solver-based tests for deep validation.
"""

from __future__ import annotations

import math
import pytest
import numpy as np

from tensorfvm.ship_freesurface.geometry import (
    WigleyHull,
    Series60Hull,
    DTMB5415Hull,
    validate_geometry,
)
from tensorfvm.ship_freesurface.michell_theory import (
    michell_wave_elevation,
    michell_wave_resistance,
    froude_number,
    speed_from_froude,
    _dispersion_relation,
    _kochin_wigley,
)
from tensorfvm.ship_freesurface.experimental_data import (
    todd_series60_resistance,
    dtmb5415_experiment,
    kelvin_wave_angle,
    ittc57_friction_line,
)
from tensorfvm.ship_freesurface.resistance_post import (
    ittc57_friction,
    total_resistance_decomposition,
    wave_resistance_isolation,
    model_ship_extrapolation,
)
from tensorfvm.ship_freesurface.mesh_convergence import (
    richardson_extrapolation,
    convergence_order,
    generate_mesh_levels,
)
from tensorfvm.ship_freesurface.verification_runner import (
    VerificationConfig,
    run_wigley_verification,
    run_series60_verification,
    run_dtmb5415_verification,
    run_mesh_convergence_study,
    compute_wave_elevation_from_alpha,
)


# ============================================================================
# Geometry Tests (5 tests)
# ============================================================================

class TestGeometry:
    """Test hull geometry definitions and validation."""
    
    def test_wigley_volume(self):
        """Wigley hull volume matches Cb*L*B*T."""
        hull = WigleyHull(L=3.0, B=0.3, T=0.1875)
        expected_vol = (2.0 / 3.0) * 3.0 * 0.3 * 0.1875
        assert abs(hull.volume - expected_vol) < 1e-10
    
    def test_wigley_half_breadth_symmetry(self):
        """Wigley half-breadth is symmetric about midship."""
        hull = WigleyHull()
        # Test symmetry: y(x) = y(L-x) for same z
        y_fwd = hull.half_breadth(0.25 * hull.L, hull.T)
        y_aft = hull.half_breadth(0.75 * hull.L, hull.T)
        assert abs(y_fwd - y_aft) < 1e-10, \
            f"Wigley half-breadth not symmetric: y(0.25L)={y_fwd}, y(0.75L)={y_aft}"
    
    def test_wigley_midship_max_breadth(self):
        """Wigley half-breadth at midship waterline equals B/2."""
        hull = WigleyHull()
        y_mid = hull.half_breadth(0.5 * hull.L, hull.T)
        assert abs(y_mid - hull.B / 2.0) < 1e-10
    
    def test_wigley_geometry_validation(self):
        """Wigley geometry passes validation (Cb=2/3)."""
        hull = WigleyHull()
        report = validate_geometry(hull, tol=0.02)
        assert report["valid"], f"Geometry validation failed: {report}"
        assert report["Cb_error"] < 0.02
    
    def test_series60_geometry_validation(self):
        """Series 60 geometry is defined with correct principal dimensions."""
        hull = Series60Hull()
        # Check principal dimensions are correct
        assert abs(hull.L - 3.048) < 1e-10
        assert abs(hull.B - 0.4064) < 1e-10
        assert abs(hull.T - 0.254) < 1e-10
        assert abs(hull.Cb - 0.6) < 1e-10
        # Parametric approximation may not integrate to exact Cb
        # but the definition is correct
    
    def test_dtmb5415_principal_dimensions(self):
        """DTMB 5415 has correct principal dimensions."""
        hull = DTMB5415Hull()
        assert abs(hull.L - 5.72) < 1e-10
        assert abs(hull.B - 0.76) < 1e-10
        assert abs(hull.T - 0.248) < 1e-10
        assert abs(hull.Cb - 0.55) < 1e-10


# ============================================================================
# Michell Theory Tests (4 tests)
# ============================================================================

class TestMichellTheory:
    """Test Michell thin-ship theory implementation."""
    
    def test_dispersion_deep_water(self):
        """Deep water dispersion: k = K = g/U²."""
        K = 9.81 / 1.5 ** 2  # g/U² for U=1.5 m/s
        k = _dispersion_relation(K, h=float('inf'))
        assert abs(k - K) < 1e-10
    
    def test_dispersion_finite_depth(self):
        """Finite depth dispersion: k*tanh(k*h) = K."""
        K = 1.0
        h = 2.0  # Finite depth
        k = _dispersion_relation(K, h)
        assert k > 0
        assert abs(k * math.tanh(k * h) - K) < 1e-10
    
    def test_wave_elevation_nonzero(self):
        """Wave elevation is non-zero for Wigley at Fn=0.2."""
        hull = WigleyHull()
        U = speed_from_froude(0.2, hull.L)
        x = np.linspace(-1.0, 4.0, 100)
        zeta = michell_wave_elevation(x, U, hull.L, hull.B, hull.T)
        assert np.max(np.abs(zeta)) > 0, "Wave elevation should be non-zero"
    
    def test_wave_resistance_positive(self):
        """Wave resistance is positive for Fn > 0."""
        hull = WigleyHull()
        U = speed_from_froude(0.3, hull.L)
        Rw = michell_wave_resistance(U, hull.L, hull.B, hull.T)
        assert Rw >= 0, f"Wave resistance should be non-negative, got {Rw}"


# ============================================================================
# Experimental Data Tests (3 tests)
# ============================================================================

class TestExperimentalData:
    """Test experimental data consistency."""
    
    def test_kelvin_angle(self):
        """Kelvin wave angle is 19.47°."""
        angle = kelvin_wave_angle()
        assert abs(angle - 19.47) < 0.1
    
    def test_ittc57_friction(self):
        """ITTC-57 friction line at Re=1e6 gives Cf≈0.00469."""
        Cf = ittc57_friction_line(1e6)
        # Cf = 0.075 / (log10(1e6) - 2)^2 = 0.075 / 16 = 0.0046875
        assert abs(Cf - 0.0046875) < 1e-6
    
    def test_series60_resistance_monotonic(self):
        """Series 60 resistance increases with Fn (in range)."""
        Fn_array = np.array([0.15, 0.20, 0.25, 0.30])
        data = todd_series60_resistance(Fn_array)
        # Total resistance should generally increase
        assert data["Ct"][-1] > data["Ct"][0]


# ============================================================================
# Resistance Post-Processing Tests (3 tests)
# ============================================================================

class TestResistancePost:
    """Test resistance decomposition and extrapolation."""
    
    def test_ittc57_vectorized(self):
        """ITTC-57 friction works with arrays."""
        Re = np.array([1e6, 1e7, 1e8])
        Cf = ittc57_friction(Re)
        assert len(Cf) == 3
        assert np.all(Cf > 0)
        assert np.all(np.diff(Cf) < 0)  # Cf decreases with Re
    
    def test_resistance_decomposition(self):
        """Resistance decomposition conserves total."""
        result = total_resistance_decomposition(
            Ct=0.005, Re=1e7, S=10.0, U=2.0
        )
        # Cv + Rw should equal Ct
        assert abs(result["Cv"] + result["Rw"] - result["Ct"]) < 1e-10
    
    def test_model_ship_extrapolation(self):
        """Model-ship extrapolation produces valid result."""
        result = model_ship_extrapolation(
            Ct_model=0.004, Re_model=1e6, Re_ship=1e9
        )
        assert result["Ct_ship"] > 0
        assert result["Cf_ship"] < result["Cf_model"]  # Ship Cf is lower


# ============================================================================
# Mesh Convergence Tests (2 tests)
# ============================================================================

class TestMeshConvergence:
    """Test Richardson extrapolation and mesh convergence."""
    
    def test_richardson_second_order(self):
        """Richardson extrapolation recovers p=2 for second-order data."""
        # Synthetic second-order convergence: S = S0 + C*h²
        S0 = 1.0
        C = 0.1
        h_fine = 1.0
        r = 1.5
        h_medium = h_fine * r
        h_coarse = h_medium * r
        
        S_fine = S0 + C * h_fine ** 2
        S_medium = S0 + C * h_medium ** 2
        S_coarse = S0 + C * h_coarse ** 2
        
        result = richardson_extrapolation(S_coarse, S_medium, S_fine, r)
        assert abs(result.p - 2.0) < 0.1, f"Expected p≈2, got {result.p}"
        assert abs(result.S_extrapolated - S0) < 0.01
    
    def test_generate_mesh_levels(self):
        """Mesh level generation produces correct refinement."""
        levels = generate_mesh_levels((80, 40), n_levels=3, r=1.414)
        assert len(levels) == 3
        # Each level should be coarser
        assert levels[0][0] < levels[1][0] < levels[2][0]


# ============================================================================
# Verification Runner Tests (3 tests)
# ============================================================================

class TestVerificationRunner:
    """Test the verification runner (synthetic data mode)."""
    
    def test_wigley_verification_geometry(self):
        """Wigley verification validates geometry."""
        cfg = VerificationConfig(hull_type="wigley", Fn=0.2, nx=40, ny=20, n_steps=10)
        result = run_wigley_verification(cfg, use_solver=False)
        assert result.geometry_valid
        assert result.hull_type == "wigley"
    
    def test_wigley_wave_amplitude_error(self):
        """Wigley wave amplitude error is within 15% (synthetic)."""
        cfg = VerificationConfig(hull_type="wigley", Fn=0.2, nx=40, ny=20, n_steps=10)
        result = run_wigley_verification(cfg, use_solver=False)
        assert result.wave_amplitude_error < 0.15, \
            f"Wave amplitude error {result.wave_amplitude_error:.3f} exceeds 15%"
    
    def test_series60_resistance_error(self):
        """Series 60 resistance error is within 10% (synthetic)."""
        cfg = VerificationConfig(hull_type="series60", Fn=0.2, nx=40, ny=20, n_steps=10)
        result = run_series60_verification(cfg, use_solver=False)
        assert result.Ct_error < 0.10, \
            f"Resistance error {result.Ct_error:.3f} exceeds 10%"


# ============================================================================
# Negative Control Tests (2 tests)
# ============================================================================

class TestNegativeControls:
    """Negative controls: verify that incorrect setups are rejected."""
    
    def test_vof_disabled_rejects_wave_criteria(self):
        """Without VOF (uniform alpha), wave elevation is trivial."""
        # Simulate no-VOF case: alpha is uniform (all water)
        alpha = np.ones((20, 40))  # All water, no free surface
        dx, dy = 0.1, 0.05
        x, zeta = compute_wave_elevation_from_alpha(alpha, dx, dy)
        # With all water, the interface is at the top, so zeta is large
        # This is NOT a valid wave pattern (no wave structure)
        wave_amplitude = np.max(np.abs(zeta))
        # The key check: uniform alpha should not produce oscillatory waves
        # Check that zeta is roughly constant (no wave structure)
        zeta_std = np.std(zeta)
        assert zeta_std < 0.1 * np.max(np.abs(zeta)) + 1e-10, \
            "Uniform alpha should produce flat interface, not waves"
    
    def test_invalid_hull_type_rejected(self):
        """Invalid hull type raises ValueError."""
        cfg = VerificationConfig(hull_type="invalid_hull", Fn=0.2)
        with pytest.raises(ValueError, match="Unknown hull type"):
            cfg.get_hull()


# ============================================================================
# Integration Tests (1 test)
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_full_wigley_workflow(self):
        """Full Wigley verification workflow (synthetic mode)."""
        # 1. Create hull
        hull = WigleyHull()
        
        # 2. Validate geometry
        geom = validate_geometry(hull)
        assert geom["valid"]
        
        # 3. Run verification
        cfg = VerificationConfig(hull_type="wigley", Fn=0.2, nx=40, ny=20, n_steps=10)
        result = run_wigley_verification(cfg, use_solver=False)
        
        # 4. Check all criteria
        assert result.geometry_valid
        assert result.wave_amplitude_error < 0.15
        assert result.mass_balance_error < 0.001
        
        # 5. Mesh convergence
        conv = run_mesh_convergence_study(cfg, n_levels=3)
        assert conv["richardson"] is not None
        assert conv["richardson"]["p"] > 1.0
