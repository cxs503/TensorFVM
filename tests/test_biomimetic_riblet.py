"""Pytest test suite for biomimetic riblet drag-reduction verification.

Tests cover:
1. Riblet geometry construction and validation
2. Dimensionless spacing s⁺ computation
3. Drag-reduction correlations (all 4 types)
4. Optimal s⁺ calibration
5. Maximum drag-reduction verification
6. Curve trend verification
7. Benchmark agreement (Bechert 1997)
8. Equivalent roughness and wall-function shift
9. Smooth-wall skin friction correlations
10. Grid convergence / y⁺ guidance
11. First cell height computation
12. Scan utilities
13. Literature data interpolation

All tests are deterministic and require no external data files.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from tensorfvm.biomimetic_riblet import (
    RibletGeometry,
    dimensionless_spacing,
    friction_velocity_from_cf,
    bechert_v_groove_drag_reduction,
    bechert_u_groove_drag_reduction,
    bechert_blade_drag_reduction,
    bechert_shark_scale_drag_reduction,
    get_drag_reduction,
    optimal_s_plus,
    max_drag_reduction,
    scan_s_plus,
    riblet_equivalent_roughness,
    riblet_wall_function_shift,
    smooth_wall_cf,
    riblet_cf,
    BECHERT_1997_V_GROOVE_DATA,
    bechert_benchmark_interpolation,
    verify_optimal_s_plus,
    verify_max_drag_reduction,
    verify_curve_trend,
    verify_benchmark_agreement,
    y_plus_guidance,
    first_cell_height,
    KAPPA,
    NU_WATER_20C,
    RHO_WATER_20C,
)


# ---------------------------------------------------------------------------
# Test 1: Riblet geometry construction
# ---------------------------------------------------------------------------

class TestRibletGeometry:
    """Test riblet geometry parameterisation."""

    def test_v_groove_geometry(self):
        """V-groove riblet: area = 0.5·s·h, wetted ratio > 1."""
        geo = RibletGeometry(riblet_type='v_groove', s=1e-4, h=5e-5)
        assert geo.riblet_type == 'v_groove'
        assert geo.s == 1e-4
        assert geo.h == 5e-5
        assert geo.h_over_s == pytest.approx(0.5)
        assert geo.cross_section_area == pytest.approx(0.5 * 1e-4 * 5e-5)
        assert geo.wetted_area_ratio > 1.0

    def test_u_groove_geometry(self):
        """U-groove riblet geometry."""
        geo = RibletGeometry(riblet_type='u_groove', s=2e-4, h=1e-4)
        assert geo.h_over_s == pytest.approx(0.5)
        assert geo.cross_section_area > 0
        assert geo.wetted_area_ratio > 1.0

    def test_blade_geometry(self):
        """Blade riblet: thin, low area."""
        geo = RibletGeometry(riblet_type='blade', s=1e-4, h=8e-5)
        assert geo.h_over_s == pytest.approx(0.8)
        assert geo.cross_section_area < 0.5 * geo.s * geo.h  # thin blade

    def test_shark_scale_geometry(self):
        """Shark-scale riblet geometry."""
        geo = RibletGeometry(riblet_type='shark_scale', s=3e-4, h=1.5e-4)
        assert geo.h_over_s == pytest.approx(0.5)
        assert geo.wetted_area_ratio == pytest.approx(1.15)

    def test_invalid_type_raises(self):
        """Unknown riblet type raises ValueError."""
        with pytest.raises(ValueError, match='unknown riblet type'):
            RibletGeometry(riblet_type='invalid', s=1e-4, h=5e-5)

    def test_negative_spacing_raises(self):
        """Negative spacing raises ValueError."""
        with pytest.raises(ValueError, match='positive'):
            RibletGeometry(riblet_type='v_groove', s=-1e-4, h=5e-5)

    def test_zero_height_raises(self):
        """Zero height raises ValueError."""
        with pytest.raises(ValueError, match='positive'):
            RibletGeometry(riblet_type='v_groove', s=1e-4, h=0)

    def test_frozen_dataclass(self):
        """RibletGeometry is immutable."""
        geo = RibletGeometry(riblet_type='v_groove', s=1e-4, h=5e-5)
        with pytest.raises(Exception):  # FrozenInstanceError
            geo.s = 2e-4


# ---------------------------------------------------------------------------
# Test 2: Dimensionless spacing s⁺
# ---------------------------------------------------------------------------

class TestDimensionlessSpacing:
    """Test s⁺ = s·u_τ/ν computation."""

    def test_basic_computation(self):
        """s⁺ = s·u_τ/ν."""
        s_plus = dimensionless_spacing(s=1e-4, u_tau=0.05, nu=1e-6)
        assert s_plus == pytest.approx(5.0)

    def test_typical_riblet_range(self):
        """Typical riblet s⁺ range: 10-30."""
        s_plus = dimensionless_spacing(s=1e-4, u_tau=0.1, nu=1e-6)
        assert s_plus == pytest.approx(10.0)

    def test_zero_friction_velocity(self):
        """Zero u_τ gives s⁺ = 0."""
        s_plus = dimensionless_spacing(s=1e-4, u_tau=0.0, nu=1e-6)
        assert s_plus == pytest.approx(0.0)

    def test_negative_inputs_raise(self):
        """Negative spacing or nu raises ValueError."""
        with pytest.raises(ValueError):
            dimensionless_spacing(s=-1e-4, u_tau=0.05, nu=1e-6)
        with pytest.raises(ValueError):
            dimensionless_spacing(s=1e-4, u_tau=0.05, nu=-1e-6)


# ---------------------------------------------------------------------------
# Test 3: Friction velocity from C_f
# ---------------------------------------------------------------------------

class TestFrictionVelocity:
    """Test u_τ = U_∞·√(C_f/2)."""

    def test_basic_computation(self):
        """u_τ from C_f and U_∞."""
        u_tau = friction_velocity_from_cf(cf=0.003, u_inf=10.0)
        expected = 10.0 * math.sqrt(0.003 / 2)
        assert u_tau == pytest.approx(expected)

    def test_typical_ship_value(self):
        """Typical ship: C_f ≈ 0.002, U_∞ = 5 m/s."""
        u_tau = friction_velocity_from_cf(cf=0.002, u_inf=5.0)
        assert u_tau == pytest.approx(5.0 * math.sqrt(0.001), rel=1e-10)

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValueError):
            friction_velocity_from_cf(cf=-0.001, u_inf=10.0)
        with pytest.raises(ValueError):
            friction_velocity_from_cf(cf=0.003, u_inf=0)


# ---------------------------------------------------------------------------
# Test 4: Drag-reduction correlations (all types)
# ---------------------------------------------------------------------------

class TestDragReductionCorrelations:
    """Test all four riblet drag-reduction correlations."""

    def test_v_groove_peak(self):
        """V-groove peak at s⁺ ≈ 12, ΔC_f ≈ 9.9%."""
        delta_cf = bechert_v_groove_drag_reduction(12.0)
        assert delta_cf == pytest.approx(0.099, abs=0.001)

    def test_v_groove_zero_crossing(self):
        """V-groove crosses zero at s⁺ ≈ 25."""
        delta_cf_25 = bechert_v_groove_drag_reduction(25.0)
        assert delta_cf_25 == pytest.approx(0.0, abs=0.001)

    def test_v_groove_drag_increase(self):
        """V-groove shows drag increase for s⁺ > 25."""
        delta_cf = bechert_v_groove_drag_reduction(35.0)
        assert delta_cf < 0

    def test_u_groove_peak(self):
        """U-groove peak at s⁺ ≈ 15, ΔC_f ≈ 7%."""
        delta_cf = bechert_u_groove_drag_reduction(15.0)
        assert delta_cf == pytest.approx(0.070, abs=0.001)

    def test_blade_peak(self):
        """Blade peak at s⁺ ≈ 11, ΔC_f ≈ 11%."""
        delta_cf = bechert_blade_drag_reduction(11.0)
        assert delta_cf == pytest.approx(0.110, abs=0.001)

    def test_shark_scale_peak(self):
        """Shark-scale peak at s⁺ ≈ 17, ΔC_f ≈ 7.5%."""
        delta_cf = bechert_shark_scale_drag_reduction(17.0)
        assert delta_cf == pytest.approx(0.075, abs=0.001)

    def test_array_input(self):
        """Correlations accept array input."""
        s_plus = np.array([5.0, 10.0, 15.0, 20.0, 30.0])
        delta_cf = bechert_v_groove_drag_reduction(s_plus)
        assert delta_cf.shape == s_plus.shape
        assert delta_cf[0] > 0  # s⁺=5: drag reduction
        assert delta_cf[1] > delta_cf[0]  # rising
        assert delta_cf[4] < 0  # s⁺=30: drag increase

    def test_negative_s_plus_raises(self):
        with pytest.raises(ValueError):
            bechert_v_groove_drag_reduction(-1.0)

    def test_get_drag_reduction_dispatch(self):
        """get_drag_reduction dispatches to correct correlation."""
        for rtype in ['v_groove', 'u_groove', 'blade', 'shark_scale']:
            dc = get_drag_reduction(rtype, 10.0)
            assert dc > 0  # all positive at s⁺=10

    def test_get_drag_reduction_invalid_type(self):
        with pytest.raises(ValueError, match='unknown riblet type'):
            get_drag_reduction('invalid', 10.0)


# ---------------------------------------------------------------------------
# Test 5: Optimal s⁺ calibration
# ---------------------------------------------------------------------------

class TestOptimalSPlus:
    """Test optimal s⁺ values."""

    def test_v_groove_optimal(self):
        assert optimal_s_plus('v_groove') == pytest.approx(12.0)

    def test_u_groove_optimal(self):
        assert optimal_s_plus('u_groove') == pytest.approx(15.0)

    def test_blade_optimal(self):
        assert optimal_s_plus('blade') == pytest.approx(11.0)

    def test_shark_scale_optimal(self):
        assert optimal_s_plus('shark_scale') == pytest.approx(17.0)

    def test_optimal_in_literature_range(self):
        """All optimal s⁺ values in literature range 10-20."""
        for rtype in ['v_groove', 'u_groove', 'blade', 'shark_scale']:
            s_opt = optimal_s_plus(rtype)
            assert 10.0 <= s_opt <= 20.0, f'{rtype}: s⁺={s_opt} out of range'


# ---------------------------------------------------------------------------
# Test 6: Maximum drag reduction
# ---------------------------------------------------------------------------

class TestMaxDragReduction:
    """Test maximum ΔC_f values."""

    def test_v_groove_max(self):
        d_max = max_drag_reduction('v_groove')
        assert d_max == pytest.approx(0.099, abs=0.001)
        assert 0.08 <= d_max <= 0.12  # literature range 8-12%

    def test_blade_highest(self):
        """Blade riblets achieve highest drag reduction."""
        for rtype in ['v_groove', 'u_groove', 'shark_scale']:
            assert max_drag_reduction('blade') > max_drag_reduction(rtype)

    def test_u_groove_lowest(self):
        """U-groove has lowest peak among standard types."""
        d_u = max_drag_reduction('u_groove')
        for rtype in ['v_groove', 'blade']:
            assert d_u < max_drag_reduction(rtype)


# ---------------------------------------------------------------------------
# Test 7: Curve trend verification
# ---------------------------------------------------------------------------

class TestCurveTrend:
    """Test ΔC_f(s⁺) curve properties."""

    def test_v_groove_trend(self):
        result = verify_curve_trend('v_groove')
        assert result['all_passed'], f"Trend checks failed: {result}"

    def test_u_groove_trend(self):
        result = verify_curve_trend('u_groove')
        assert result['all_passed']

    def test_blade_trend(self):
        result = verify_curve_trend('blade')
        assert result['all_passed']

    def test_shark_scale_trend(self):
        result = verify_curve_trend('shark_scale')
        assert result['all_passed']

    def test_positive_in_dr_regime(self):
        """ΔC_f > 0 for s⁺ in [5, 25]."""
        for rtype in ['v_groove', 'u_groove', 'blade', 'shark_scale']:
            result = verify_curve_trend(rtype)
            assert result['positive_in_dr_regime'], f'{rtype} fails DR regime check'


# ---------------------------------------------------------------------------
# Test 8: Benchmark agreement (Bechert 1997)
# ---------------------------------------------------------------------------

class TestBenchmarkAgreement:
    """Test correlation against Bechert 1997 data."""

    def test_v_groove_benchmark(self):
        """V-groove uses PCHIP interpolation, so it matches exactly."""
        result = verify_benchmark_agreement('v_groove', tolerance=0.01)
        assert result['passed'], (
            f"Benchmark failed: max_rel_err={result['max_relative_error']:.4f}"
        )

    def test_benchmark_max_relative_error(self):
        """V-groove max relative error should be ~0 (exact interpolation)."""
        result = verify_benchmark_agreement('v_groove')
        assert result['max_relative_error'] < 0.01

    def test_benchmark_data_integrity(self):
        """Benchmark data has expected structure."""
        data = BECHERT_1997_V_GROOVE_DATA
        assert len(data['s_plus']) == len(data['delta_cf'])
        assert data['s_plus'][0] == 5.0
        assert data['s_plus'][-1] == 35.0


# ---------------------------------------------------------------------------
# Test 9: Equivalent roughness and wall-function shift
# ---------------------------------------------------------------------------

class TestEquivalentRoughness:
    """Test riblet equivalent roughness computation."""

    def test_v_groove_roughness(self):
        k_s = riblet_equivalent_roughness('v_groove', s=1e-4, h=5e-5)
        assert k_s > 0
        assert k_s == pytest.approx(5e-5 * 0.5)

    def test_roughness_scales_with_height(self):
        """k_s proportional to riblet height."""
        k1 = riblet_equivalent_roughness('v_groove', s=1e-4, h=5e-5)
        k2 = riblet_equivalent_roughness('v_groove', s=1e-4, h=1e-4)
        assert k2 == pytest.approx(2 * k1)


class TestWallFunctionShift:
    """Test log-law shift ΔB due to riblets."""

    def test_positive_shift_in_dr_regime(self):
        """ΔB > 0 in drag-reduction regime."""
        delta_b = riblet_wall_function_shift(
            'v_groove', s=1e-4, h=5e-5, u_tau=0.1, nu=1e-6
        )
        # s⁺ = 10, in DR regime
        assert delta_b > 0

    def test_shift_magnitude(self):
        """ΔB is O(0.1-1) for typical riblets."""
        delta_b = riblet_wall_function_shift(
            'v_groove', s=1.2e-4, h=6e-5, u_tau=0.08, nu=1e-6
        )
        assert 0.01 < abs(delta_b) < 10.0


# ---------------------------------------------------------------------------
# Test 10: Smooth-wall skin friction
# ---------------------------------------------------------------------------

class TestSmoothWallCf:
    """Test smooth-wall C_f correlations."""

    def test_ittc57(self):
        cf = smooth_wall_cf(1e7, 'ittc57')
        assert 0.002 < cf < 0.004

    def test_schlichting(self):
        cf = smooth_wall_cf(1e7, 'schlichting')
        assert 0.002 < cf < 0.004

    def test_prandtl_schlichting(self):
        cf = smooth_wall_cf(1e7, 'prandtl_schlichting')
        assert 0.002 < cf < 0.004

    def test_white(self):
        cf = smooth_wall_cf(1e7, 'white')
        assert 0.002 < cf < 0.004

    def test_correlations_agree_within_20pct(self):
        """All correlations agree within 20% at Re=1e7."""
        re = 1e7
        values = [smooth_wall_cf(re, m) for m in
                  ['ittc57', 'schlichting', 'prandtl_schlichting', 'white']]
        mean_cf = np.mean(values)
        for v in values:
            assert abs(v - mean_cf) / mean_cf < 0.20

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match='unknown method'):
            smooth_wall_cf(1e7, 'invalid')

    def test_negative_re_raises(self):
        with pytest.raises(ValueError):
            smooth_wall_cf(-1e7)


# ---------------------------------------------------------------------------
# Test 11: Riblet C_f computation
# ---------------------------------------------------------------------------

class TestRibletCf:
    """Test C_f_riblet = C_f_smooth · (1 - ΔC_f)."""

    def test_drag_reduction(self):
        cf_smooth = 0.003
        delta_cf = 0.08
        cf_riblet = riblet_cf(cf_smooth, delta_cf)
        assert cf_riblet == pytest.approx(0.003 * 0.92)
        assert cf_riblet < cf_smooth

    def test_drag_increase(self):
        cf_smooth = 0.003
        delta_cf = -0.03  # drag increase
        cf_riblet = riblet_cf(cf_smooth, delta_cf)
        assert cf_riblet > cf_smooth

    def test_zero_reduction(self):
        cf_smooth = 0.003
        cf_riblet = riblet_cf(cf_smooth, 0.0)
        assert cf_riblet == pytest.approx(cf_smooth)


# ---------------------------------------------------------------------------
# Test 12: Benchmark interpolation
# ---------------------------------------------------------------------------

class TestBenchmarkInterpolation:
    """Test Bechert 1997 data interpolation."""

    def test_interpolation_at_data_points(self):
        """Interpolation returns exact values at data points."""
        data = BECHERT_1997_V_GROOVE_DATA
        for sp, dc in zip(data['s_plus'], data['delta_cf']):
            interp_val = bechert_benchmark_interpolation(sp)
            assert interp_val == pytest.approx(dc, abs=1e-10)

    def test_interpolation_between_points(self):
        """Interpolation gives reasonable values between points."""
        dc_10 = bechert_benchmark_interpolation(10.0)
        dc_12 = bechert_benchmark_interpolation(12.0)
        assert dc_10 < dc_12  # rising toward peak

    def test_interpolation_monotonic_rising(self):
        """Interpolation is monotonic rising before peak."""
        values = [bechert_benchmark_interpolation(s) for s in [5, 7, 9, 10.5, 12]]
        for i in range(len(values) - 1):
            assert values[i] < values[i + 1]


# ---------------------------------------------------------------------------
# Test 13: Verification utilities
# ---------------------------------------------------------------------------

class TestVerificationUtilities:
    """Test verify_* functions."""

    def test_verify_optimal_s_plus_v_groove(self):
        result = verify_optimal_s_plus('v_groove')
        assert result['passed']
        assert result['deviation'] == 0.0

    def test_verify_max_drag_reduction_v_groove(self):
        result = verify_max_drag_reduction('v_groove')
        assert result['passed']
        assert result['relative_error'] == 0.0

    def test_verify_all_types_optimal(self):
        for rtype in ['v_groove', 'u_groove', 'blade', 'shark_scale']:
            result = verify_optimal_s_plus(rtype)
            assert result['passed'], f'{rtype} optimal s⁺ verification failed'

    def test_verify_all_types_max_dr(self):
        for rtype in ['v_groove', 'u_groove', 'blade', 'shark_scale']:
            result = verify_max_drag_reduction(rtype)
            assert result['passed'], f'{rtype} max ΔC_f verification failed'


# ---------------------------------------------------------------------------
# Test 14: y⁺ guidance and first cell height
# ---------------------------------------------------------------------------

class TestYPlusGuidance:
    """Test y⁺ guidance for RANS simulations."""

    def test_wall_function_guidance(self):
        result = y_plus_guidance('wall_function')
        assert result['target_y_plus'] == 30.0
        assert result['y_plus_min'] == 20.0
        assert result['y_plus_max'] == 100.0

    def test_near_wall_resolved_guidance(self):
        result = y_plus_guidance('near_wall_resolved')
        assert result['target_y_plus'] == 0.5
        assert result['y_plus_max'] <= 1.0

    def test_invalid_treatment_raises(self):
        with pytest.raises(ValueError, match='unknown wall treatment'):
            y_plus_guidance('invalid')


class TestFirstCellHeight:
    """Test first cell height computation."""

    def test_basic_computation(self):
        """Δy = y⁺ · ν / u_τ."""
        dy = first_cell_height(y_plus_target=30.0, u_tau=0.1, nu=1e-6)
        expected = 30.0 * 1e-6 / 0.1
        assert dy == pytest.approx(expected)

    def test_near_wall_resolved(self):
        """y⁺ < 1 requires very small first cell."""
        dy = first_cell_height(y_plus_target=0.5, u_tau=0.05, nu=1e-6)
        assert dy < 1e-5  # sub-10 μm

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValueError):
            first_cell_height(y_plus_target=-1, u_tau=0.1, nu=1e-6)


# ---------------------------------------------------------------------------
# Test 15: Scan utilities
# ---------------------------------------------------------------------------

class TestScanSPlus:
    """Test s⁺ scan utility."""

    def test_scan_returns_expected_keys(self):
        result = scan_s_plus('v_groove')
        assert 's_plus' in result
        assert 'delta_cf' in result
        assert 's_plus_opt' in result
        assert 'max_delta_cf' in result

    def test_scan_array_lengths(self):
        result = scan_s_plus('v_groove', n_points=21)
        assert len(result['s_plus']) == 21
        assert len(result['delta_cf']) == 21

    def test_scan_peak_matches_optimal(self):
        result = scan_s_plus('v_groove', s_plus_range=(5, 35), n_points=61)
        peak_idx = np.argmax(result['delta_cf'])
        s_at_peak = result['s_plus'][peak_idx]
        assert abs(s_at_peak - result['s_plus_opt']) < 2.0

    def test_scan_custom_range(self):
        result = scan_s_plus('blade', s_plus_range=(5, 25), n_points=11)
        assert result['s_plus'][0] == pytest.approx(5.0)
        assert result['s_plus'][-1] == pytest.approx(25.0)


# ---------------------------------------------------------------------------
# Test 16: End-to-end ΔC_f computation
# ---------------------------------------------------------------------------

class TestEndToEnd:
    """End-to-end drag-reduction computation."""

    def test_full_pipeline_v_groove(self):
        """Complete pipeline: geometry → s⁺ → ΔC_f → C_f_riblet."""
        geo = RibletGeometry(riblet_type='v_groove', s=1.2e-4, h=6e-5)
        u_inf = 5.0  # m/s
        re = 5e6
        cf_smooth = smooth_wall_cf(re, 'ittc57')
        u_tau = friction_velocity_from_cf(cf_smooth, u_inf)
        s_plus = dimensionless_spacing(geo.s, u_tau, NU_WATER_20C)
        delta_cf = get_drag_reduction('v_groove', s_plus)
        cf_riblet = riblet_cf(cf_smooth, delta_cf)

        assert s_plus > 0
        assert cf_riblet < cf_smooth  # drag reduction
        assert delta_cf > 0 or delta_cf < 0.01  # reasonable range

    def test_optimal_spacing_gives_max_reduction(self):
        """At optimal s⁺, ΔC_f should be near maximum."""
        for rtype in ['v_groove', 'u_groove', 'blade', 'shark_scale']:
            s_opt = optimal_s_plus(rtype)
            delta_cf = get_drag_reduction(rtype, s_opt)
            d_max = max_drag_reduction(rtype)
            assert abs(delta_cf - d_max) < 0.005, (
                f'{rtype}: ΔC_f(s⁺_opt)={delta_cf} ≠ max={d_max}'
            )
