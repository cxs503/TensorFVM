import importlib.util
from pathlib import Path

import pytest

script = Path(__file__).parents[1] / 'scripts' / 'audit_native_sections3d.py'
spec = importlib.util.spec_from_file_location('native_sections_audit', script)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_geometry_is_validated_before_any_area_claim():
    for label, record in audit.geometry_precheck().items():
        assert record['relative_volume_error'] <= 1e-12, label
        assert record['maximum_relative_closure'] <= 1e-10, label
        assert record['minimum_face_projection'] > 0., label


def test_section_areas_match_closed_form_and_polygon_structure():
    cases = {record['label']: record for record in audit.section_probe()}
    assert len(cases) == len(audit.section_cases())
    for label, record in cases.items():
        assert record['relative_error'] <= 1e-12, label
        assert record['polygons'] == record['expected_polygons'], label
        assert record['vertices'] == [record['expected_vertices']] * record['expected_polygons'], label
        assert record['orientation_positive'], label


def test_non_axis_aligned_planes_are_exact():
    # rotation onto the (1,1,1) normal keeps the analytic triangle/hexagon areas
    cases = {record['label']: record for record in audit.section_probe()}
    triangle = cases['cube_plane_111_c0.5']
    hexagon = cases['cube_plane_111_c1.5']
    assert triangle['polygons'] == 1 and triangle['vertices'] == [3]
    assert hexagon['polygons'] == 1 and hexagon['vertices'] == [6]
    assert triangle['relative_error'] <= 1e-14 and hexagon['relative_error'] <= 1e-14
    # the same plane through the corner tetrahedron gives the identical area
    assert cases['tetra_plane_111_c0.5']['computed_area'] == pytest.approx(
        triangle['computed_area'], rel=1e-14)


def test_triangulation_diagonals_cancel_on_coplanar_faces():
    cases = {record['label']: record for record in audit.section_probe()}
    coplanar = cases['cube_tri_z_bottom_coplanar']
    assert coplanar['relative_error'] <= 1e-12
    assert coplanar['vertices'] == [4]
    assert coplanar['collinear_vertices'] == [0]
    # a crossing cut keeps the exact area but carries 4 collinear ring nodes
    crossing = cases['cube_tri_z_half']
    assert crossing['relative_error'] <= 1e-12
    assert crossing['vertices'] == [8]
    assert crossing['collinear_vertices'] == [4]


def test_internal_facet_section_is_returned_once_per_owner():
    record = next(r for r in audit.section_probe() if r['label'] == 'block_x_internal_coplanar')
    assert record['polygons'] == 8
    assert record['physical_area'] == pytest.approx(4.)
    assert record['computed_area'] / record['physical_area'] == pytest.approx(2.)


def test_near_tangent_plane_snaps_onto_the_coplanar_face():
    record = next(r for r in audit.section_probe() if r['label'] == 'cube_z_near_tangent_snap')
    # within the default 1e-11 tolerance the plane must not produce a sliver
    assert record['polygons'] == 1 and record['vertices'] == [4]
    assert record['relative_error'] <= 1e-12


def test_cavalieri_integral_of_sections_equals_volume():
    for label, record in audit.integration_probe().items():
        assert record['relative_error'] <= 1e-12, label
        assert record['integral'] > 0.


def test_vertex_tangencies_are_omitted():
    for label, record in audit.tangency_probe().items():
        assert record['polygons'] == 0, label
        assert record['total_abs_area'] == 0., label


def test_invalid_input_nonplanar_facet_and_open_section_are_rejected():
    rejections = audit.rejection_probe()
    assert all(record['rejected'] for record in rejections.values())
    assert 'nonplanar' in rejections['nonplanar_facet']['message']
    assert 'nonclosed' in rejections['open_section_ring']['message']


def test_report_summary_states_all_claims_hold():
    summary = audit.build_report()['machine']['summary']
    assert summary['all_verified'] is True
    assert summary['maximum_relative_area_error'] <= 1e-12
    assert summary['maximum_relative_integral_error'] <= 1e-12
    assert summary['polygons_expected_mismatch'] == []
    assert summary['orientation_failures'] == []
    assert summary['tangency_polygons'] == 0
    assert summary['rejections_accepted'] == []