"""Independent experimental SUBOFF metrics; MMS never supplies physical data."""
import numpy as np
from .suboff_geometry import radius
from .native_rans3d import _triangle_distance


def require_reynolds(actual, expected):
    if not np.isfinite(actual) or actual <= 0 or abs(actual / expected - 1) > 1e-6:
        raise ValueError('Computed Reynolds number does not match this experimental case')


def drag_comparison(force_dimensionless, case):
    """Force units rho U^2 L^2; reference uses its explicitly stated physical setup."""
    force = np.asarray(force_dimensionless, float)
    if force.shape != (3,) or not np.isfinite(force).all():
        raise ValueError('Finite integrated total force vector required')
    properties = case['operating_conditions']; geometry = case['geometry']; ref = case['reference']
    scale = properties['rho_kg_m3'] * properties['U_m_s']**2 * geometry['L_m']**2
    drag = float(force[0] * scale)
    area = ref['frontal_area_m2']
    dynamic_force = .5 * properties['rho_kg_m3'] * properties['U_m_s']**2 * area
    cd = drag / dynamic_force
    error = abs(drag - ref['total_force_N']) / abs(ref['total_force_N'])
    return dict(total_drag_N=drag, reference_drag_N=ref['total_force_N'],
        Cd_calculated=cd, Cd_reference=ref['Cd_derived'], relative_error=error,
        nominal_error_below_3_percent=bool(error < .03),
        experimental_uncertainty=ref['experimental_uncertainty'],
        normalization='total pressure plus viscous force; frontal area',
        property_provenance=properties['property_provenance'])


def sample_upper_meridian_pressure(mesh, pressure_faces, taps, *, p_infinity=0., rho=1., speed=1.):
    """Piecewise constant native face p at nearest actual triangle to each tap.

    Points lie on the analytic axisymmetric hull, on y=0,z>0. Exact triangle
    distances (including edges) select the discrete surface; geometrical offsets
    and selected face centers are retained. No fit to experimental Cp is made.
    """
    p = np.asarray(pressure_faces, float); x = np.asarray(taps, float)
    if p.shape != (len(mesh.owner),) or not np.isfinite(p).all():
        raise ValueError('Physical pressure on every actual native face required')
    if not np.isfinite(x).all() or np.any((x <= 0) | (x >= 1)):
        raise ValueError('Tap coordinates must lie strictly inside hull length')
    if not np.isfinite([p_infinity, rho, speed]).all() or rho <= 0 or speed <= 0:
        raise ValueError('Finite reference pressure and positive rho and speed required')
    wall = np.flatnonzero(mesh.patch_labels == 'wall')
    if not len(wall) or any(len(mesh.face_vertices[f]) != 3 for f in wall):
        raise ValueError('Native triangular hull wall facets required')
    triangles = mesh.vertices[np.asarray([mesh.face_vertices[f] for f in wall])]
    points = np.column_stack((x, np.zeros(len(x)), radius(x)))
    selected = []; distance = []
    for point in points:
        d = _triangle_distance(np.broadcast_to(point, (len(wall), 3)), triangles)
        index = int(np.argmin(d)); selected.append(int(wall[index])); distance.append(float(d[index]))
    selected = np.array(selected, int)
    cp = (p[selected] - p_infinity) / (.5 * rho * speed**2)
    return dict(x_L=x.tolist(), Cp_calculated=cp.tolist(), face_indices=selected.tolist(),
        analytic_tap_coordinates=points.tolist(), selected_face_centers=mesh.face_centers[selected].tolist(),
        geometry_offset_L=distance, p_infinity=float(p_infinity),
        sampling='unfiltered piecewise constant reconstructed physical pressure on nearest actual wall triangle')


def cp_comparison(calculated, candidate_points, *, uncertainty_absolute=.015):
    actual = np.asarray(calculated, float)
    listed = np.array([row['Cp_corrected'] for row in candidate_points])
    arithmetic = np.array([row['Cp_measured'] - row['dCp_empty_tunnel'] for row in candidate_points])
    if actual.shape != listed.shape or not np.isfinite(actual).all() or not np.linalg.norm(listed):
        raise ValueError('Finite calculated Cp at every reference tap required')
    error = float(np.linalg.norm(actual - listed) / np.linalg.norm(listed))
    alternate = float(np.linalg.norm(actual - arithmetic) / np.linalg.norm(arithmetic))
    return dict(candidate_Cp_listed=listed.tolist(), candidate_Cp_arithmetic=arithmetic.tolist(),
        nominal_relative_L2_error=error, correction_sensitivity_relative_L2_error=alternate,
        provisional_nominal_error_below_3_percent=bool(error < .03),
        correction_sensitivity_error_below_3_percent=bool(alternate < .03),
        absolute_RMSE=float(np.sqrt(np.mean((actual - listed)**2))),
        experimental_Cp_95_absolute_uncertainty=uncertainty_absolute,
        full_primary_reference_verified=False,
        scope='19 meridian taps; not pointwise relative percentages or an entire surface norm')
