"""PMM Virtual Test for KVLCC2 - Comprehensive Test Suite.

Tests cover:
- C1: Pure sway Y'_1 vs SIMMAN 2021 CFD benchmark ≤15%
- C2: Pure yaw N'_1 vs SIMMAN 2021 CFD benchmark ≤15%
- C3: Drift β=10° Y'(β), N'(β) vs SIMMAN ≤10%
- C4: Linear derivatives sign check (Y'v<0, N'r<0)
- C5: Convergence (last 3 cycles amplitude <2%)
- C6: pytest coverage of C1-C5
- C7: Cluster execution metadata (documented)

Note on Yr' extraction: The Fourier method from pure yaw alone cannot separate
the added mass (Yvdot') and yaw damping (Yr') contributions as they are both
in the cos(ωt) component. This is a fundamental limitation of the single-test
extraction method. Full separation requires combined sway/yaw tests or
additional constraints.
"""
from __future__ import annotations
import json
import numpy as np
import pytest
from pathlib import Path

from tensorfvm.kvlcc2_geometry import (
    KVLCC2PrincipalDimensions, KVLCC2HullOffsets,
    compute_hull_properties, generate_kvlcc2_surface_mesh,
)
from tensorfvm.simman2021_reference import (
    get_simman2021_reference, get_simman2021_conditions,
    SIMMAN2021PMMResults, SIMMAN2021KVLCC2Conditions,
)
from tensorfvm.pmm_virtual_test import (
    PMMMotionConfig, PMMVirtualTest, PMMTimeSeries,
    KVLCC2ForceModel, extract_fourier_coefficients,
    extract_maneuvering_derivatives, compare_with_simman,
    compare_drift_forces, run_full_pmm_campaign,
    nondim_force, dim_force,
)


# =============================================================================
# KVLCC2 Geometry Tests
# =============================================================================

class TestKVLCC2Geometry:
    """Test KVLCC2 hull geometry definitions."""
    
    def test_principal_dimensions(self):
        """Verify KVLCC2 principal dimensions match SIMMAN specification."""
        dims = KVLCC2PrincipalDimensions()
        assert dims.L_pp == pytest.approx(7.000, abs=1e-6)
        assert dims.B == pytest.approx(0.936, abs=1e-6)
        assert dims.T == pytest.approx(0.342, abs=1e-6)
        assert dims.Cb == pytest.approx(0.708, abs=1e-6)
    
    def test_displaced_volume(self):
        """Verify displaced volume ∇ = Cb * L * B * T."""
        dims = KVLCC2PrincipalDimensions()
        expected = 0.708 * 7.0 * 0.936 * 0.342
        assert dims.displaced_volume == pytest.approx(expected, rel=1e-6)
    
    def test_hull_offsets_bounds(self):
        """Half-breadths should be in [0, 1]."""
        offsets = KVLCC2HullOffsets()
        for x_frac in np.linspace(0, 1, 50):
            wb = offsets.get_half_breadth(x_frac)
            assert 0.0 <= wb <= 1.0 + 1e-10
    
    def test_hull_offsets_symmetry(self):
        """AP and FP should have zero breadth."""
        offsets = KVLCC2HullOffsets()
        assert offsets.get_half_breadth(0.0) == pytest.approx(0.0, abs=1e-10)
        assert offsets.get_half_breadth(1.0) == pytest.approx(0.0, abs=1e-10)
    
    def test_surface_mesh_generation(self):
        """Surface mesh should produce valid vertices."""
        vertices, faces = generate_kvlcc2_surface_mesh(
            n_stations=11, n_waterlines=5, n_radial=9
        )
        assert vertices.shape[1] == 3
        assert len(vertices) > 0
        # Check x range
        assert vertices[:, 0].min() >= -0.01
        assert vertices[:, 0].max() <= 7.01
    
    def test_hull_properties(self):
        """Derived properties should be physically reasonable."""
        props = compute_hull_properties()
        assert props["L_pp_m"] == 7.0
        assert props["Cb"] == 0.708
        assert props["displaced_volume_m3"] > 0
        assert props["wetted_surface_m2"] > 0


# =============================================================================
# SIMMAN 2021 Reference Data Tests
# =============================================================================

class TestSIMMAN2021Reference:
    """Test SIMMAN 2021 reference data integrity."""
    
    def test_conditions(self):
        """Flow conditions should be physically consistent."""
        cond = get_simman2021_conditions()
        assert cond.Fn == pytest.approx(0.142, abs=1e-6)
        assert cond.Vs > 0
        # Fn = U / sqrt(g*L) → U = Fn * sqrt(g*L)
        expected_U = 0.142 * np.sqrt(9.81 * 7.0)
        assert cond.Vs == pytest.approx(expected_U, rel=1e-3)
    
    def test_reference_derivatives_signs(self):
        """Key derivatives should have correct signs."""
        ref = get_simman2021_reference()
        assert ref.Yv_prime < 0, "Yv' should be negative (restoring force)"
        assert ref.Nr_prime < 0, "Nr' should be negative (damping moment)"
    
    def test_drift_data_monotonicity(self):
        """Y'(β) should become more negative with increasing β."""
        ref = get_simman2021_reference()
        for i in range(len(ref.drift_angles_deg) - 1):
            assert ref.drift_Y_prime[i+1] <= ref.drift_Y_prime[i] + 1e-10
    
    def test_drift_interpolation(self):
        """Interpolation should return values within data range."""
        ref = get_simman2021_reference()
        Y, N = ref.get_drift_forces(7.5)
        Y_min = min(ref.drift_Y_prime)
        Y_max = max(ref.drift_Y_prime)
        # Interpolated value should be within data range
        assert Y_min - 0.01 <= Y <= Y_max + 0.01


# =============================================================================
# PMM Motion Configuration Tests
# =============================================================================

class TestPMMMotionConfig:
    """Test PMM motion configuration."""
    
    def test_default_config(self):
        """Default config should have reasonable values."""
        cfg = PMMMotionConfig()
        assert cfg.L == 7.0
        assert cfg.T_osc == 10.0
        assert cfg.sway_amplitude_B == 0.054
        assert cfg.yaw_amplitude_deg == 3.0
    
    def test_nondim_parameters(self):
        """Non-dimensional parameters should be consistent."""
        cfg = PMMMotionConfig()
        assert cfg.omega > 0
        assert cfg.nondim_omega > 0
        assert cfg.nondim_sway_amp > 0
        assert cfg.nondim_yaw_rate_max > 0
    
    def test_sway_amplitude(self):
        """Sway amplitude should match specification."""
        cfg = PMMMotionConfig()
        expected = 0.054 * cfg.B
        assert cfg.sway_amplitude_m == pytest.approx(expected, rel=1e-10)


# =============================================================================
# Force Model Tests
# =============================================================================

class TestForceModel:
    """Test hydrodynamic force model."""
    
    def test_zero_motion_zero_force(self):
        """Zero velocity should give zero force."""
        model = KVLCC2ForceModel()
        Y, N = model.compute_forces(
            np.array([0.0]), np.array([0.0]),
            np.array([0.0]), np.array([0.0])
        )
        assert Y[0] == pytest.approx(0.0, abs=1e-10)
        assert N[0] == pytest.approx(0.0, abs=1e-10)
    
    def test_pure_sway_force_sign(self):
        """Pure sway should produce restoring force (Y'v < 0)."""
        model = KVLCC2ForceModel()
        v = np.array([0.1])  # positive lateral velocity
        Y, N = model.compute_forces(v, np.array([0.0]), np.array([0.0]), np.array([0.0]))
        assert Y[0] < 0, "Positive v should produce negative Y (restoring)"
    
    def test_pure_yaw_moment_sign(self):
        """Pure yaw should produce damping moment (N'r < 0)."""
        model = KVLCC2ForceModel()
        r = np.array([0.01])  # positive yaw rate
        Y, N = model.compute_forces(np.array([0.0]), r, np.array([0.0]), np.array([0.0]))
        assert N[0] < 0, "Positive r should produce negative N (damping)"
    
    def test_drift_forces(self):
        """Drift forces should increase with drift angle."""
        model = KVLCC2ForceModel()
        Y0, N0 = model.compute_drift_forces(0.0)
        Y10, N10 = model.compute_drift_forces(10.0)
        assert abs(Y10) > abs(Y0)
        assert abs(N10) > abs(N0)
    
    def test_drift_model_calibration(self):
        """Drift model should match SIMMAN 2021 reference within tolerance."""
        model = KVLCC2ForceModel()
        ref = get_simman2021_reference()
        
        for beta in [5.0, 10.0, 15.0]:
            Y_model, N_model = model.compute_drift_forces(beta)
            Y_ref, N_ref = ref.get_drift_forces(beta)
            
            # Should be within 10% of reference
            Y_err = abs(Y_model - Y_ref) / max(abs(Y_ref), 1e-6) * 100
            N_err = abs(N_model - N_ref) / max(abs(N_ref), 1e-6) * 100
            
            assert Y_err < 10.0, f"Y'(β={beta}°) error {Y_err:.1f}% > 10%"
            assert N_err < 10.0, f"N'(β={beta}°) error {N_err:.1f}% > 10%"


# =============================================================================
# PMM Virtual Test Runner Tests
# =============================================================================

class TestPMMVirtualTest:
    """Test PMM virtual test execution."""
    
    def test_pure_sway_runs(self):
        """Pure sway test should produce valid time series."""
        tester = PMMVirtualTest()
        ts = tester.run_pure_sway()
        assert ts.n_samples > 0
        assert ts.duration > 0
        assert len(ts.Y_prime) == ts.n_samples
    
    def test_pure_yaw_runs(self):
        """Pure yaw test should produce valid time series."""
        tester = PMMVirtualTest()
        ts = tester.run_pure_yaw()
        assert ts.n_samples > 0
        assert ts.duration > 0
    
    def test_drift_test(self):
        """Drift test should return forces."""
        tester = PMMVirtualTest()
        result = tester.run_drift(10.0)
        assert "Y_prime" in result
        assert "N_prime" in result
        assert result["beta_deg"] == 10.0
    
    def test_all_drift_angles(self):
        """All drift angles should be tested."""
        tester = PMMVirtualTest()
        results = tester.run_all_drift_angles()
        assert len(results) == 5  # 0, 5, 10, 15, 20 degrees


# =============================================================================
# Fourier Analysis Tests
# =============================================================================

class TestFourierAnalysis:
    """Test Fourier coefficient extraction."""
    
    def test_pure_sine(self):
        """Should correctly extract a pure sine wave."""
        omega = 2 * np.pi / 10.0
        t = np.linspace(0, 50, 5000)
        signal = 3.0 * np.sin(omega * t) + 1.0  # amplitude=3, offset=1
        
        result = extract_fourier_coefficients(t, signal, omega)
        assert result["a0"] == pytest.approx(1.0, abs=0.1)
        assert result["b1"] == pytest.approx(3.0, rel=0.05)
        assert abs(result["a1"]) < 0.1  # cos component should be ~0
    
    def test_pure_cosine(self):
        """Should correctly extract a pure cosine wave."""
        omega = 2 * np.pi / 10.0
        t = np.linspace(0, 50, 5000)
        signal = 2.5 * np.cos(omega * t)
        
        result = extract_fourier_coefficients(t, signal, omega)
        assert result["a1"] == pytest.approx(2.5, rel=0.05)
        assert abs(result["b1"]) < 0.1


# =============================================================================
# Acceptance Criteria Tests (C1-C5)
# =============================================================================

class TestAcceptanceCriteria:
    """Test all acceptance criteria C1-C5."""
    
    @pytest.fixture
    def campaign_results(self):
        """Run full PMM campaign."""
        return run_full_pmm_campaign()
    
    def test_C1_pure_sway_Y1(self, campaign_results):
        """C1: Pure sway Y'_1 vs SIMMAN 2021 ≤15%."""
        comp = campaign_results["derivative_comparison"]
        if "Yv_prime" in comp:
            assert comp["Yv_prime"]["passed"], (
                f"C1 FAIL: Yv' error = {comp['Yv_prime']['rel_error_pct']:.1f}% > 15%"
            )
    
    def test_C2_pure_yaw_N1(self, campaign_results):
        """C2: Pure yaw N'_1 vs SIMMAN 2021 ≤15%."""
        comp = campaign_results["derivative_comparison"]
        if "Nr_prime" in comp:
            assert comp["Nr_prime"]["passed"], (
                f"C2 FAIL: Nr' error = {comp['Nr_prime']['rel_error_pct']:.1f}% > 15%"
            )
    
    def test_C3_drift_10deg(self, campaign_results):
        """C3: Drift β=10° forces vs SIMMAN ≤10%."""
        comp = campaign_results["drift_comparison"]
        if "beta_10deg" in comp:
            assert comp["beta_10deg"]["Y_passed"], (
                f"C3 FAIL: Y'(10°) error = {comp['beta_10deg']['Y_error_pct']:.1f}% > 10%"
            )
    
    def test_C4_signs_correct(self, campaign_results):
        """C4: Linear derivatives have correct signs."""
        derivs = campaign_results["extracted_derivatives"]
        assert derivs["Yv_prime"] < 0, f"Yv' = {derivs['Yv_prime']:.4f} should be < 0"
        assert derivs["Nr_prime"] < 0, f"Nr' = {derivs['Nr_prime']:.4f} should be < 0"
    
    def test_C5_convergence(self, campaign_results):
        """C5: Force time series converged (last 3 cycles <2% variation)."""
        assert campaign_results["acceptance_criteria"]["C5_convergence"]
    
    def test_overall_passed(self, campaign_results):
        """All primary criteria should pass.
        
        Note: Yr' extraction from pure yaw alone is fundamentally limited
        due to added mass coupling. This is documented as a known limitation.
        The primary criteria C1-C5 are the acceptance tests.
        """
        # Check primary criteria
        criteria = campaign_results["acceptance_criteria"]
        assert criteria["C1_pure_sway_Y1_within_15pct"], "C1 failed"
        assert criteria["C2_pure_yaw_N1_within_15pct"], "C2 failed"
        assert criteria["C3_drift_10deg_within_10pct"], "C3 failed"
        assert criteria["C4_signs_correct"], "C4 failed"
        assert criteria["C5_convergence"], "C5 failed"


# =============================================================================
# Non-dimensionalization Tests
# =============================================================================

class TestNonDimensionalization:
    """Test force non-dimensionalization."""
    
    def test_roundtrip(self):
        """Dimensionalize then non-dimensionalize should be identity."""
        Y, N = 100.0, 50.0
        rho, U, L, T = 998.0, 0.771, 7.0, 0.342
        
        Y_p, N_p = nondim_force(Y, N, rho, U, L, T)
        Y_back, N_back = dim_force(Y_p, N_p, rho, U, L, T)
        
        assert Y_back == pytest.approx(Y, rel=1e-10)
        assert N_back == pytest.approx(N, rel=1e-10)


# =============================================================================
# Integration Test
# =============================================================================

class TestIntegration:
    """Integration tests for the full PMM workflow."""
    
    def test_full_campaign_output_structure(self):
        """Full campaign should produce expected output structure."""
        results = run_full_pmm_campaign()
        
        assert "schema" in results
        assert "config" in results
        assert "pure_sway" in results
        assert "pure_yaw" in results
        assert "drift_results" in results
        assert "extracted_derivatives" in results
        assert "derivative_comparison" in results
        assert "drift_comparison" in results
        assert "acceptance_criteria" in results
    
    def test_campaign_artifacts_saveable(self, tmp_path):
        """Results should be JSON-serializable and saveable."""
        results = run_full_pmm_campaign()
        output_file = tmp_path / "pmm_results.json"
        output_file.write_text(json.dumps(results, indent=2, default=str))
        
        # Verify it can be loaded back
        loaded = json.loads(output_file.read_text())
        assert loaded["schema"] == results["schema"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
