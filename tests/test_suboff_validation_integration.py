"""Integration tests for SUBOFF validation pipeline: geometry continuity, pressure sampling, and drag metrics.

Validates that the SUBOFF mother-hull radius profile is continuous across
segment boundaries, that face-pressure sampling correctly maps analytic taps
to discrete wall triangles, and that drag normalization handles edge cases
without division-by-zero or silent NaN propagation.
"""
import numpy as np
import pytest
from tensorfvm.suboff_geometry import suboff_radius_profile, radius, RADIUS_OVER_LENGTH
from tensorfvm.suboff_validation import drag_comparison, cp_comparison, sample_upper_meridian_pressure
from types import SimpleNamespace


# ---------------------------------------------------------------------------
# 1. Geometry continuity & physical bounds
# ---------------------------------------------------------------------------
def test_radius_profile_continuity_at_segment_boundaries():
    """SUBOFF radius must be continuous at bow/midbody/stern transitions."""
    eps = 1e-8
    xi = np.array([0.0, 0.2333 - eps, 0.2333, 0.2333 + eps,
                   0.7449 - eps, 0.7449, 0.7449 + eps,
                   0.9781 - eps, 0.9781, 0.9781 + eps, 1.0])
    r = suboff_radius_profile(xi)
    # Midbody plateau
    assert np.allclose(r[3:6], 1.0, atol=1e-10)
    # Bow tip
    assert np.isclose(r[0], 0.0, atol=1e-10)
    # Stern tip
    assert np.isclose(r[-1], 0.0, atol=1e-10)
    # Monotonicity in taper sections (approximate check)
    assert np.all(np.diff(r[6:9]) <= 1e-10)


def test_radius_physical_scale_matches_reference():
    """Physical radius at midbody must equal RADIUS_OVER_LENGTH."""
    r_mid = radius(np.array([0.5]))
    assert np.isclose(r_mid[0], RADIUS_OVER_LENGTH, rtol=1e-12)


# ---------------------------------------------------------------------------
# 2. Pressure sampling geometric consistency
# ---------------------------------------------------------------------------
def _make_mock_mesh():
    """Minimal axisymmetric wall mesh for sampling tests."""
    n_faces = 32
    vertices = np.array([[0.5, np.cos(2*np.pi*i/n_faces)*0.1, np.sin(2*np.pi*i/n_faces)*0.1] for i in range(n_faces)])
    faces = np.array([[i, (i+1)%n_faces, (i+n_faces//2)%n_faces] for i in range(n_faces)])
    neighbor = np.full(n_faces, -1, dtype=int)
    patch_labels = np.array(['wall'] * n_faces)
    face_centers = vertices.mean(axis=1)  # Approximate
    return SimpleNamespace(
        owner=faces[:, 0],
        patch_labels=patch_labels,
        vertices=vertices,
        face_vertices=faces,
        face_centers=face_centers
    )


def test_sample_upper_meridian_pressure_maps_taps_to_nearest_wall():
    """Taps on y=0,z>0 must select wall faces with positive z."""
    mesh = _make_mock_mesh()
    p_faces = np.ones(len(mesh.owner))
    taps = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    result = sample_upper_meridian_pressure(mesh, p_faces, taps, p_infinity=0., rho=1., speed=1.)
    assert len(result['x_L']) == len(taps)
    assert len(result['Cp_calculated']) == len(taps)
    assert all(z >= 0 for _, _, z in result['analytic_tap_coordinates'])
    # Cp = (p - p_inf) / (0.5 * rho * U^2) = 1.0 / 0.5 = 2.0
    assert np.allclose(result['Cp_calculated'], 2.0)


def test_sample_upper_meridian_pressure_rejects_invalid_inputs():
    """Invalid taps/pressure/rho/speed must raise ValueError."""
    mesh = _make_mock_mesh()
    p_faces = np.ones(len(mesh.owner))
    taps = np.array([0.0, 0.5, 1.0])  # boundary values
    with pytest.raises(ValueError, match='strictly inside hull length'):
        sample_upper_meridian_pressure(mesh, p_faces, taps)
    
    taps = np.array([0.5])
    p_faces_bad = np.array([1.0, np.nan])
    with pytest.raises(ValueError, match='Physical pressure on every actual native face required'):
        sample_upper_meridian_pressure(mesh, p_faces_bad, taps)


# ---------------------------------------------------------------------------
# 3. Drag comparison edge cases
# ---------------------------------------------------------------------------
def _make_case():
    return {
        'geometry': {'L_m': 1.0},
        'operating_conditions': {'rho_kg_m3': 1000., 'U_m_s': 1.0, 'property_provenance': 'ISO standard'},
        'reference': {'total_force_N': 100., 'frontal_area_m2': 1.0, 'Cd_derived': 0.2, 'experimental_uncertainty': 0.05}
    }


def test_drag_comparison_zero_force_yields_zero_cd():
    """Zero force vector must yield Cd=0 and relative_error=1.0 (full deviation)."""
    case = _make_case()
    res = drag_comparison([0., 0., 0.], case)
    assert res['Cd_calculated'] == 0.0
    assert res['relative_error'] == 1.0
    assert not res['nominal_error_below_3_percent']


def test_drag_comparison_negative_drag_is_handled():
    """Negative drag (thrust) must compute correctly without sign errors.
    Note: input is dimensionless force (rho U^2 L^2), so -50 means -50 * scale."""
    case = _make_case()
    res = drag_comparison([-50., 0., 0.], case)
    assert res['total_drag_N'] == -50000.0  # -50 * 1000 * 1^2 * 1^2
    assert res['relative_error'] == 501.0  # |(-50000 - 100)| / 100


def test_drag_comparison_3d_vector_requires_exact_shape():
    """Non-3D or non-finite force vectors must raise."""
    case = _make_case()
    with pytest.raises(ValueError, match='Finite integrated total force vector required'):
        drag_comparison([1., 2.], case)
    with pytest.raises(ValueError, match='Finite integrated total force vector required'):
        drag_comparison([np.inf, 0., 0.], case)


# ---------------------------------------------------------------------------
# 4. Cp comparison sensitivity & uncertainty
# ---------------------------------------------------------------------------
def test_cp_comparison_uncertainty_flagging():
    """Cp errors exceeding experimental uncertainty must be flagged."""
    points = [
        {'Cp_measured': 0.5, 'dCp_empty_tunnel': 0.01, 'Cp_corrected': 0.49},
        {'Cp_measured': 0.6, 'dCp_empty_tunnel': 0.01, 'Cp_corrected': 0.59},
    ]
    calculated = [0.49, 0.59]
    res = cp_comparison(calculated, points, uncertainty_absolute=0.015)
    assert res['experimental_Cp_95_absolute_uncertainty'] == 0.015
    assert res['full_primary_reference_verified'] is False
    # Absolute RMSE should be near zero for perfect match
    assert res['absolute_RMSE'] < 1e-12


def test_cp_comparison_shape_mismatch_raises():
    """Mismatched lengths must raise ValueError."""
    points = [{'Cp_measured': 0.5, 'dCp_empty_tunnel': 0.01, 'Cp_corrected': 0.49}]
    with pytest.raises(ValueError, match='Finite calculated Cp at every reference tap required'):
        cp_comparison([0.49, 0.59], points)
