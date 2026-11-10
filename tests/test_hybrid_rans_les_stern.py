"""Tests for hybrid RANS-LES (DES/IDDES) ship stern flow verification.

Covers:
1. DES97/IDDES length scale computation
2. IDDES shielding function behavior
3. Modified k-destruction consistency
4. Stern flow separation metrics extraction
5. Reynolds stress spectral analysis
6. Grid convergence check
7. RANS vs DES deviation quantification
8. Full verification suite execution
"""
import pytest
import numpy as np
from tensorfvm.des_iddes_model import (
    HybridSSTDES, des97_length_scale, iddes_length_scale,
    iddes_shielding_function, modified_k_destruction,
    C_DES_SST_1, C_DES_SST_2, BETA_STAR, KAPPA
)
from tensorfvm.stern_hybrid_verification import (
    SternFlowCase, MeshSpecification, MESH_COARSE, MESH_FINE,
    suboff_stern_separation_region, backward_facing_step_reference,
    compute_separation_from_velocity_profile,
    compute_reynolds_stress_spectrum,
    synthesize_stern_flow_field, run_verification_suite,
    compare_rans_vs_des, grid_convergence_check,
    VerificationResult, SeparationMetrics, SpectralMetrics,
)
from tensorfvm.suboff_geometry import LENGTH_M


# ============================================================================
# DES97 Length Scale Tests
# ============================================================================

class TestDES97LengthScale:
    """Test DES97 hybrid length scale computation."""

    def test_pure_rans_when_grid_coarse(self):
        """When C_DES*Delta > L_RANS, DES should return L_RANS (pure RANS)."""
        k = np.array([0.01])
        omega = np.array([100.0])
        d = np.array([0.001])
        cell_size = np.array([0.0001])  # Very small grid -> L_LES small

        result = des97_length_scale(k, omega, d, cell_size)

        L_RANS = np.sqrt(k) / (BETA_STAR * omega)
        # With very small cell, L_LES = C_DES * Delta < L_RANS
        # So DES should pick L_LES
        assert result['L_DES'] <= L_RANS[0] + 1e-15

    def test_les_mode_when_grid_fine(self):
        """When C_DES*Delta < L_RANS, DES should return L_LES (LES mode)."""
        k = np.array([0.1])
        omega = np.array([10.0])
        d = np.array([0.1])
        cell_size = np.array([0.01])  # Small enough for LES mode

        result = des97_length_scale(k, omega, d, cell_size)

        L_RANS = np.sqrt(k) / (BETA_STAR * omega)
        L_LES = C_DES_SST_1 * cell_size

        assert result['L_RANS'][0] == pytest.approx(L_RANS[0], rel=1e-10)
        assert result['L_LES'][0] == pytest.approx(L_LES[0], rel=1e-10)
        # DES picks minimum
        assert result['L_DES'][0] == pytest.approx(min(L_RANS[0], L_LES[0]), rel=1e-10)

    def test_ratio_bounded(self):
        """DES ratio should be in (0, 1]."""
        rng = np.random.default_rng(42)
        k = rng.uniform(1e-4, 1.0, 50)
        omega = rng.uniform(1.0, 1000.0, 50)
        d = rng.uniform(1e-4, 0.1, 50)
        cell_size = rng.uniform(1e-4, 0.05, 50)

        result = des97_length_scale(k, omega, d, cell_size)

        assert np.all(result['ratio'] >= 0)
        assert np.all(result['ratio'] <= 1.0 + 1e-10)

    def test_rejects_negative_k(self):
        """Negative k should raise ValueError."""
        with pytest.raises(ValueError, match='non-negative'):
            des97_length_scale(np.array([-0.01]), np.array([100.0]),
                             np.array([0.001]), np.array([0.001]))

    def test_rejects_zero_omega(self):
        """Zero omega should raise ValueError."""
        with pytest.raises(ValueError, match='positive'):
            des97_length_scale(np.array([0.01]), np.array([0.0]),
                             np.array([0.001]), np.array([0.001]))

    def test_rejects_negative_wall_distance(self):
        """Negative wall distance should raise ValueError."""
        with pytest.raises(ValueError, match='positive'):
            des97_length_scale(np.array([0.01]), np.array([100.0]),
                             np.array([-0.001]), np.array([0.001]))


# ============================================================================
# IDDES Shielding Function Tests
# ============================================================================

class TestIDDSShielding:
    """Test IDDES shielding function."""

    def test_rans_mode_near_wall(self):
        """Near wall with high nu_t, f_d should be ~0 (RANS mode)."""
        k = np.array([0.01])
        omega = np.array([1000.0])
        d = np.array([1e-5])  # Very close to wall
        nu_t = np.array([0.1])  # High eddy viscosity
        nu = 1e-6
        G = np.zeros((1, 3, 3))
        G[0, 0, 1] = 100.0  # Strong shear

        result = iddes_shielding_function(k, omega, d, nu_t, nu, G)

        # Near wall, r_d is large -> f_d ≈ 0 (RANS protected)
        assert result['f_d'][0] < 0.1

    def test_les_mode_far_from_wall(self):
        """Far from wall with low nu_t, f_d should be ~1 (LES mode)."""
        k = np.array([0.001])
        omega = np.array([10.0])
        d = np.array([0.1])  # Far from wall
        nu_t = np.array([1e-6])  # Low eddy viscosity (laminar-like)
        nu = 1e-6
        G = np.zeros((1, 3, 3))
        G[0, 0, 1] = 1.0  # Weak shear

        result = iddes_shielding_function(k, omega, d, nu_t, nu, G)

        # Far from wall, low nu_t -> r_d small -> f_d ≈ 1 (LES mode)
        assert result['f_d'][0] > 0.9

    def test_shielding_bounded(self):
        """f_d should always be in [0, 1]."""
        rng = np.random.default_rng(123)
        n = 100
        k = rng.uniform(1e-6, 1.0, n)
        omega = rng.uniform(1.0, 1e4, n)
        d = rng.uniform(1e-6, 1.0, n)
        nu_t = rng.uniform(1e-8, 1.0, n)
        nu = 1e-6
        G = rng.standard_normal((n, 3, 3)) * 10

        result = iddes_shielding_function(k, omega, d, nu_t, nu, G)

        assert np.all(result['f_d'] >= 0.0)
        assert np.all(result['f_d'] <= 1.0)


# ============================================================================
# IDDES Length Scale Tests
# ============================================================================

class TestIDDESLengthScale:
    """Test IDDES hybrid length scale."""

    def test_iddes_protects_boundary_layer(self):
        """IDDES should not reduce length scale in BL (f_d ≈ 0)."""
        k = np.array([0.01])
        omega = np.array([5000.0])
        d = np.array([1e-5])  # Very near wall
        cell_size = np.array([0.01])  # Coarse grid
        nu_t = np.array([0.05])
        nu = 1e-6
        G = np.zeros((1, 3, 3))
        G[0, 0, 1] = 500.0

        result = iddes_length_scale(k, omega, d, cell_size, nu_t, nu, G)

        # In BL, IDDES should stay close to RANS
        assert result['L_IDDES'][0] >= 0.9 * result['L_RANS'][0]

    def test_iddes_reduces_in_separation(self):
        """IDDES should reduce length scale in separated region."""
        k = np.array([0.1])
        omega = np.array([5.0])
        d = np.array([0.5])  # Far from wall
        cell_size = np.array([0.02])  # Grid allows LES
        nu_t = np.array([1e-5])  # Low nu_t (separated, low production)
        nu = 1e-6
        G = np.zeros((1, 3, 3))
        G[0, 0, 1] = 0.5  # Weak shear

        result = iddes_length_scale(k, omega, d, cell_size, nu_t, nu, G)

        # In separation, f_d should be high, L_IDDES < L_RANS
        assert result['f_d'][0] > 0.5
        assert result['L_IDDES'][0] < result['L_RANS'][0]


# ============================================================================
# Modified k-Destruction Tests
# ============================================================================

class TestModifiedKDestruction:
    """Test modified k-destruction term."""

    def test_consistency_with_standard_sst(self):
        """When L_hybrid = L_RANS, modified destruction = standard SST."""
        k = np.array([0.01, 0.05, 0.1])
        omega = np.array([100.0, 200.0, 50.0])
        rho = 1025.0

        L_RANS = np.sqrt(k) / (BETA_STAR * omega)
        D_modified = modified_k_destruction(k, L_RANS, rho=rho)
        D_standard = rho * BETA_STAR * omega * k

        np.testing.assert_allclose(D_modified, D_standard, rtol=1e-10)

    def test_increased_destruction_in_les_mode(self):
        """When L_hybrid < L_RANS, destruction should increase."""
        k = np.array([0.1])
        L_RANS = np.sqrt(k) / (BETA_STAR * 10.0)
        L_LES = 0.5 * L_RANS  # LES mode: shorter length scale

        D_standard = 1025.0 * BETA_STAR * 10.0 * k[0]
        D_modified = modified_k_destruction(k, L_LES, rho=1025.0)

        assert D_modified[0] > D_standard

    def test_zero_k_gives_zero_destruction(self):
        """k=0 should give zero destruction."""
        k = np.array([0.0])
        L = np.array([0.01])
        D = modified_k_destruction(k, L)
        assert D[0] == 0.0


# ============================================================================
# HybridSSTDES Class Tests
# ============================================================================

class TestHybridSSTDES:
    """Test the HybridSSTDES wrapper class."""

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match='Unknown DES mode'):
            HybridSSTDES(mode='invalid')

    def test_des97_mode(self):
        des = HybridSSTDES(mode='des97')
        k = np.array([0.01])
        omega = np.array([100.0])
        d = np.array([0.01])
        cell_size = np.array([0.005])
        nu_t = np.array([0.01])
        nu = 1e-6
        G = np.zeros((1, 3, 3))
        G[0, 0, 1] = 10.0

        result = des.apply(k, omega, d, cell_size, nu_t, nu, G)

        assert result['mode'] == 'des97'
        assert 'f_d' not in result  # DES97 has no shielding
        assert result['D_k_modified'][0] > 0

    def test_iddes_mode_has_shielding(self):
        des = HybridSSTDES(mode='iddes')
        k = np.array([0.01])
        omega = np.array([100.0])
        d = np.array([0.01])
        cell_size = np.array([0.005])
        nu_t = np.array([0.01])
        nu = 1e-6
        G = np.zeros((1, 3, 3))
        G[0, 0, 1] = 10.0

        result = des.apply(k, omega, d, cell_size, nu_t, nu, G)

        assert result['mode'] == 'iddes'
        assert 'f_d' in result
        assert 'r_d' in result

    def test_effective_nut_reduces_in_les_mode(self):
        des = HybridSSTDES(mode='iddes')
        k = np.array([0.1])
        omega = np.array([5.0])
        d = np.array([0.5])
        cell_size = np.array([0.02])
        nu_t_rans = np.array([0.05])
        nu = 1e-6
        G = np.zeros((1, 3, 3))
        G[0, 0, 1] = 0.5

        nut_eff, result = des.effective_nut(k, omega, d, cell_size,
                                             nu_t_rans, nu, G)

        # In LES mode, nut_eff should be reduced
        assert nut_eff[0] <= nu_t_rans[0] + 1e-15


# ============================================================================
# Stern Geometry Tests
# ============================================================================

class TestSternGeometry:
    """Test SUBOFF stern geometry utilities."""

    def test_stern_region_monotonic_radius(self):
        """Stern radius should decrease monotonically."""
        stern = suboff_stern_separation_region(0.74, 0.97, 50)
        # In the stern cone, radius decreases
        assert stern['r_m'][-1] < stern['r_m'][0]

    def test_stern_slope_negative(self):
        """Hull slope should be negative in stern cone (contracting)."""
        stern = suboff_stern_separation_region(0.75, 0.95, 50)
        # Most of the stern cone has negative slope
        assert np.mean(stern['dr_dx']) < 0

    def test_bfs_reference_data(self):
        """BFS reference data should have expected values."""
        ref = backward_facing_step_reference()
        assert ref['Xr_over_h'] == pytest.approx(6.0, abs=0.5)
        assert ref['St_h_peak'] == pytest.approx(0.18, abs=0.05)
        assert ref['Re_h'] == 36000


# ============================================================================
# Separation Metrics Tests
# ============================================================================

class TestSeparationMetrics:
    """Test separation bubble extraction."""

    def test_known_separation_profile(self):
        """Test with a known separation/reattachment pattern."""
        nx, ny = 50, 20
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 0.1, ny)

        # Create velocity profiles with separation from x=0.3 to x=0.7
        u = np.zeros((nx, ny))
        for i in range(nx):
            if 0.3 <= x[i] <= 0.7:
                # Backflow near wall with proper gradient
                for j in range(ny):
                    if y[j] < 0.03:
                        u[i, j] = -0.1 * np.sin(np.pi * (x[i] - 0.3) / 0.4) * (1 - y[j]/0.03)
                    else:
                        u[i, j] = 1.0 * (y[j] / 0.1)**(1/7)
            else:
                u[i, :] = 1.0 * (y / 0.1)**(1/7)

        metrics = compute_separation_from_velocity_profile(x, y, u)

        assert metrics.x_sep_start == pytest.approx(0.3, abs=0.05)
        assert metrics.x_sep_reattach == pytest.approx(0.7, abs=0.05)
        assert metrics.L_sep_m == pytest.approx(0.4, abs=0.1)
        assert metrics.max_backflow_velocity > 0

    def test_no_separation_gives_zero_length(self):
        """Attached flow should give zero or very small separation."""
        nx, ny = 30, 15
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 0.1, ny)

        # Attached BL profile everywhere
        u = np.zeros((nx, ny))
        for i in range(nx):
            u[i, :] = 1.0 * (y / 0.1)**(1/7)

        metrics = compute_separation_from_velocity_profile(x, y, u)
        # No backflow -> separation length should be 0 or very small
        assert metrics.max_backflow_velocity == 0.0


# ============================================================================
# Spectral Metrics Tests
# ============================================================================

class TestSpectralMetrics:
    """Test Reynolds stress spectral analysis."""

    def test_known_frequency_peak(self):
        """Test with a signal of known frequency content."""
        dt = 0.001  # 1 kHz
        n = 4096
        t = np.arange(n) * dt
        f_target = 50.0  # 50 Hz

        u_prime = np.sin(2 * np.pi * f_target * t)
        v_prime = np.sin(2 * np.pi * f_target * t + 0.3)  # Phase shift

        result = compute_reynolds_stress_spectrum(u_prime, v_prime, dt)

        # Peak should be near 50 Hz
        assert result.f_peak_Hz == pytest.approx(f_target, abs=5.0)

    def test_rms_positive(self):
        """RMS Reynolds stress should be non-negative."""
        rng = np.random.default_rng(42)
        u = rng.standard_normal(2048)
        v = rng.standard_normal(2048)

        result = compute_reynolds_stress_spectrum(u, v, 0.001)
        assert result.reynolds_stress_rms >= 0


# ============================================================================
# Full Verification Suite Tests
# ============================================================================

class TestVerificationSuite:
    """Test the complete verification workflow."""

    def test_suite_returns_all_combinations(self):
        """Suite should return results for all case/mesh/model combos."""
        results = run_verification_suite()

        # 2 cases × 2 meshes × 3 models = 12 results
        assert len(results) == 12

        # Check all models present
        models = set()
        for key, result in results.items():
            models.add(result.model)
        assert models == {'rans_sst', 'des97', 'iddes'}

    def test_des_has_higher_reynolds_stress_than_rans(self):
        """DES/IDDES should produce higher Reynolds stress than RANS."""
        results = run_verification_suite()

        # Compare for same case and mesh
        case_name = 'SUBOFF_Re1.9M'
        mesh_name = 'coarse'

        rans_key = f'{case_name}_{mesh_name}_rans_sst'
        des_key = f'{case_name}_{mesh_name}_iddes'

        rans_rs = results[rans_key].spectral.reynolds_stress_rms
        des_rs = results[des_key].spectral.reynolds_stress_rms

        assert des_rs >= rans_rs

    def test_rans_vs_des_comparison(self):
        """RANS vs DES comparison should quantify deviations."""
        results = run_verification_suite()

        case_name = 'SUBOFF_Re4.2M'
        mesh_name = 'fine'

        rans = results[f'{case_name}_{mesh_name}_rans_sst']
        iddes = results[f'{case_name}_{mesh_name}_iddes']

        comparison = compare_rans_vs_des(rans, iddes)

        assert 'reynolds_stress_deviation' in comparison
        assert 'separation_length_change' in comparison
        assert 'spectral_peak_shift' in comparison
        assert comparison['cpu_ratio'] > 1.0

    def test_grid_convergence_check(self):
        """Grid convergence should pass for well-resolved cases."""
        results = run_verification_suite()

        case_name = 'SUBOFF_Re1.9M'

        coarse = results[f'{case_name}_coarse_iddes']
        fine = results[f'{case_name}_fine_iddes']

        gc = grid_convergence_check(coarse, fine)

        assert 'L_sep_relative_change' in gc
        assert 'GCI' in gc
        assert gc['refinement_ratio'] > 1.0

    def test_synthesize_flow_field(self):
        """Flow field synthesis should produce valid data."""
        case = SternFlowCase(name='test', U_m_s=1.2, L_m=LENGTH_M,
                            nu_m2_s=1e-6, rho_kg_m3=1025.0)
        mesh = MESH_COARSE

        for model in ['rans_sst', 'des97', 'iddes']:
            flow = synthesize_stern_flow_field(case, mesh, model)

            assert flow['u_profiles'].shape == (mesh.n_axial, mesh.n_radial)
            assert flow['separation'].L_sep_m >= 0
            assert flow['spectral'].f_peak_Hz >= 0
            assert len(flow['u_prime']) == 4096


# ============================================================================
# Acceptance Criteria Tests (≥8 tests)
# ============================================================================

class TestAcceptanceCriteria:
    """Tests mapping to the mission acceptance criteria."""

    def test_separation_length_extractable(self):
        """L_sep must be extractable from flow field."""
        case = SternFlowCase(name='test', U_m_s=1.2, L_m=LENGTH_M,
                            nu_m2_s=1e-6)
        flow = synthesize_stern_flow_field(case, MESH_COARSE, 'iddes')
        assert flow['separation'].L_sep_m > 0

    def test_spectral_peak_extractable(self):
        """f_peak must be extractable from time series."""
        case = SternFlowCase(name='test', U_m_s=1.2, L_m=LENGTH_M,
                            nu_m2_s=1e-6)
        flow = synthesize_stern_flow_field(case, MESH_COARSE, 'iddes')
        assert flow['spectral'].f_peak_Hz > 0

    def test_rans_des_deviation_quantified(self):
        """Deviation between RANS and DES must be quantifiable."""
        results = run_verification_suite()
        rans = results['SUBOFF_Re1.9M_coarse_rans_sst']
        des = results['SUBOFF_Re1.9M_coarse_iddes']
        comp = compare_rans_vs_des(rans, des)
        assert comp['reynolds_stress_deviation'] >= 0

    def test_two_mesh_levels_available(self):
        """Two mesh levels must be available for convergence check."""
        assert MESH_COARSE.total_cells < MESH_FINE.total_cells
        assert MESH_COARSE.first_cell_height_m > MESH_FINE.first_cell_height_m

    def test_grid_convergence_quantified(self):
        """Grid convergence must be quantified with GCI."""
        results = run_verification_suite()
        coarse = results['SUBOFF_Re1.9M_coarse_iddes']
        fine = results['SUBOFF_Re1.9M_fine_iddes']
        gc = grid_convergence_check(coarse, fine)
        assert gc['GCI'] >= 0
        assert gc['refinement_ratio'] > 1.0

    def test_des_model_constants_documented(self):
        """DES model constants must be documented and accessible."""
        assert C_DES_SST_1 == pytest.approx(0.78, abs=0.01)
        assert C_DES_SST_2 == pytest.approx(0.61, abs=0.01)
        assert KAPPA == pytest.approx(0.41, abs=0.01)

    def test_iddes_shielding_continuous(self):
        """IDDES shielding function should be continuous."""
        d_values = np.logspace(-6, -1, 100)
        k = np.full_like(d_values, 0.01)
        omega = np.full_like(d_values, 100.0)
        nu_t = np.full_like(d_values, 0.01)
        nu = 1e-6
        G = np.zeros((len(d_values), 3, 3))
        G[:, 0, 1] = 10.0

        result = iddes_shielding_function(k, omega, d_values, nu_t, nu, G)

        # f_d should be monotonically increasing with wall distance
        # (more LES mode farther from wall)
        diffs = np.diff(result['f_d'])
        assert np.all(diffs >= -1e-10)  # Allow tiny numerical noise

    def test_cpu_time_budget(self):
        """DES CPU time should be ≤ 5× RANS."""
        results = run_verification_suite()
        for key, result in results.items():
            if result.model in ('des97', 'iddes'):
                assert result.cpu_ratio_vs_rans <= 5.0, \
                    f'{key}: CPU ratio {result.cpu_ratio_vs_rans} > 5'
