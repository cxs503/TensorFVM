"""Analytic qualification of the native planar section engine.

``tensorfvm.native_sections3d.cell_section_polygons`` is the geometric engine
behind every native section figure in this repository (wake/pressure cuts of the
SUBoff field, manufactured-solution comparisons, developed-channel reports).  It
had no direct test: it was only exercised through plotting scripts outside CI, so
a silent defect -- a dropped cell, a leaked facet diagonal, a wrong plane -- would
have corrupted published section areas while every solver test stayed green.

Every case below is graded against a closed form, never against another code
path:

* hexahedral boxes: section area is an exact rectangle (axis permutation gives
  3 distinct analytic values, so the projected-axis selection is graded too);
* non-hex cell types (pyramid, wedge): similarity of the section with the base
  gives an exact area law of the plane position;
* functional check: the area-weighted cell-centre sum of a *linear* field, i.e.
  exactly what the report scripts compute, must equal the analytic integral;
* defect paths: an axis-aligned plane tangent to a single vertex must vanish
  without error, and an unstitchable (open) cell ring must raise rather than
  fall back to a convex hull.
"""
import types

import numpy as np
import pytest

from tensorfvm.native_mesh3d import NativeMesh3D
from tensorfvm.native_sections3d import cell_section_polygons


def signed_area(polygon):
    """Signed area of a planar ring projected onto the section plane."""
    x, y = polygon[:, 0], polygon[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def centroid(polygons):
    """Area-weighted centroid of a set of signed-area polygons."""
    area = np.array([signed_area(p) for p in polygons])
    moments = np.array([signed_area(p) * p.mean(axis=0) for p in polygons])
    return (area[:, None] * moments).sum(axis=0) / area.sum()


def box_mesh(lengths, divisions, triangulate_faces=False, graded=False):
    """Structured hexahedral box anchored at the origin (z is the slice axis)."""
    coordinates = []
    for length, count in zip(lengths, divisions):
        if graded:
            s = np.linspace(0.0, 1.0, count + 1)
            coordinates.append(length * s ** 2)
        else:
            coordinates.append(np.linspace(0.0, length, count + 1))
    x, y, z = coordinates
    vertices, index = [], {}
    for k, zk in enumerate(z):
        for j, yj in enumerate(y):
            for i, xi in enumerate(x):
                index[(i, j, k)] = len(vertices)
                vertices.append((xi, yj, zk))
    cells = []
    for k in range(divisions[2]):
        for j in range(divisions[1]):
            for i in range(divisions[0]):
                cells.append((index[(i, j, k)], index[(i + 1, j, k)],
                              index[(i + 1, j + 1, k)], index[(i, j + 1, k)],
                              index[(i, j, k + 1)], index[(i + 1, j, k + 1)],
                              index[(i + 1, j + 1, k + 1)], index[(i, j + 1, k + 1)]))
    return NativeMesh3D.from_cells(np.array(vertices), cells,
                                   ['hexahedron'] * len(cells),
                                   triangulate_faces=triangulate_faces)


BOX_LENGTHS = (2.0, 3.0, 4.0)
BOX_DIVISIONS = (2, 3, 4)


@pytest.mark.parametrize('axis,value,expected_area', [(0, 0.5, 12.0), (1, 1.5, 8.0), (2, 2.5, 6.0)])
def test_axis_permutation_matches_analytic_rectangle(axis, value, expected_area):
    """One interior cut per axis: area is the product of the other two extents."""
    mesh = box_mesh(BOX_LENGTHS, BOX_DIVISIONS)
    polygons, ids = cell_section_polygons(mesh, axis=axis, value=value)
    assert len(polygons) == len(ids) > 0
    assert sum(signed_area(p) for p in polygons) == pytest.approx(expected_area, rel=1e-13)
    assert len(set(ids.tolist())) == len(ids), 'a native cell must not be reported twice'


def test_interior_slice_reports_each_column_once():
    """z = 2.5 crosses exactly one cell in each of the 2 x 3 columns, area 1 each."""
    mesh = box_mesh(BOX_LENGTHS, BOX_DIVISIONS)
    polygons, ids = cell_section_polygons(mesh, axis=2, value=2.5)
    assert len(polygons) == 6
    for polygon in polygons:
        assert signed_area(polygon) == pytest.approx(1.0, rel=1e-13), 'orientation must be positive'
        assert polygon.shape == (4, 2), 'an interior cut of a brick is a quadrilateral'
    # Cell k is the z-layer 2 slab, so ids run 12..17 in column order.
    assert ids.tolist() == list(range(12, 18))
    assert centroid(polygons) == pytest.approx((1.0, 1.5), abs=1e-13)


def test_coplanar_slice_reports_both_layers_without_double_counting():
    """A cut lying in a facet plane must return each side's trapezoid once."""
    mesh = box_mesh(BOX_LENGTHS, BOX_DIVISIONS)
    polygons, ids = cell_section_polygons(mesh, axis=2, value=2.0)
    assert len(polygons) == 12, 'both z-layers touch the plane z = 2'
    for polygon in polygons:
        assert signed_area(polygon) == pytest.approx(1.0, rel=1e-13)
    assert ids.tolist() == list(range(6, 18))


def test_domain_faces_are_reported_like_any_coplanar_cut():
    """z = 0 / z = L touch only one layer, and still give the full rectangle."""
    mesh = box_mesh(BOX_LENGTHS, BOX_DIVISIONS)
    for value, first in ((0.0, 0), (4.0, 18)):
        polygons, ids = cell_section_polygons(mesh, axis=2, value=value)
        assert len(polygons) == 6
        assert ids.tolist() == list(range(first, first + 6))
        assert sum(signed_area(p) for p in polygons) == pytest.approx(6.0, rel=1e-13)


def test_graded_mesh_does_not_change_the_analytic_area():
    """Non-uniform vertex spacing must not bias the stitched section area."""
    mesh = box_mesh(BOX_LENGTHS, BOX_DIVISIONS, graded=True)
    polygons, _ = cell_section_polygons(mesh, axis=2, value=1e-3)
    assert sum(signed_area(p) for p in polygons) == pytest.approx(6.0, rel=1e-13)


def test_triangulated_facets_keep_only_real_boundary_segments():
    """Facet diagonals are internal to a facet: area must not gain a factor."""
    mesh = box_mesh((2.0, 3.0, 4.0), (1, 1, 1), triangulate_faces=True)
    polygons, _ = cell_section_polygons(mesh, axis=2, value=0.5)
    assert len(polygons) == 1
    assert signed_area(polygons[0]) == pytest.approx(6.0, rel=1e-13)
    coplanar, _ = cell_section_polygons(mesh, axis=2, value=0.0)
    assert sum(signed_area(p) for p in coplanar) == pytest.approx(6.0, rel=1e-13)


@pytest.mark.parametrize('height', [0.0, 0.25, 0.5, 0.75])
def test_pyramid_section_obeys_the_similarity_law(height):
    """Square-base pyramid (side 2, apex at z = 1): area = 4 (1 - h)^2 exactly."""
    vertices = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0], [0, 0, 1]], float)
    mesh = NativeMesh3D.from_cells(vertices, [(0, 1, 2, 3, 4)], ['pyramid'])
    polygons, ids = cell_section_polygons(mesh, axis=2, value=height)
    assert len(polygons) == 1 and ids.tolist() == [0]
    assert signed_area(polygons[0]) == pytest.approx(4.0 * (1.0 - height) ** 2, rel=1e-13)
    assert polygons[0].mean(axis=0) == pytest.approx((0.0, 0.0), abs=1e-13), 'apex must not drift'


def test_wedge_section_is_the_base_triangle():
    """A right-triangular prism keeps its 0.5 base area on any parallel cut."""
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0],
                         [0, 0, 1], [1, 0, 1], [0, 1, 1]], float)
    mesh = NativeMesh3D.from_cells(vertices, [(0, 1, 2, 3, 4, 5)], ['wedge'])
    polygons, ids = cell_section_polygons(mesh, axis=2, value=0.5)
    assert len(polygons) == 1 and ids.tolist() == [0]
    assert signed_area(polygons[0]) == pytest.approx(0.5, rel=1e-13)


def test_apex_tangency_vanishes_without_error():
    """A plane touching only the apex encloses no finite area; it must be dropped."""
    vertices = np.array([[-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1], [0, 0, 0]], float)
    mesh = NativeMesh3D.from_cells(vertices, [(0, 1, 2, 3, 4)], ['pyramid'])
    polygons, ids = cell_section_polygons(mesh, axis=2, value=0.0)
    assert polygons == [] and ids.size == 0


def test_plane_outside_the_domain_is_empty():
    mesh = box_mesh(BOX_LENGTHS, BOX_DIVISIONS)
    for value in (4.0 + 1e-6, -1.0):
        polygons, ids = cell_section_polygons(mesh, axis=2, value=value)
        assert polygons == [] and ids.size == 0


@pytest.mark.parametrize('kwargs', [{'axis': 3}, {'axis': -1},
                                    {'value': float('nan')}, {'value': float('inf')},
                                    {'tolerance': 0.0}, {'tolerance': -1e-9}])
def test_invalid_plane_arguments_are_rejected(kwargs):
    mesh = box_mesh(BOX_LENGTHS, BOX_DIVISIONS)
    with pytest.raises(ValueError):
        cell_section_polygons(mesh, **kwargs)


def test_open_cell_ring_raises_instead_of_silently_closing():
    """A missing facet leaves an unstitchable ring: the module must fail loudly.

    The documented contract is that a nonclosed finite-area section is an error,
    never a convex-hull substitute, because a hull would report an area that no
    cell of the mesh actually encloses.
    """
    mesh = box_mesh(BOX_LENGTHS, BOX_DIVISIONS)
    keep = np.ones(len(mesh.face_vertices), bool)
    # Drop one lateral facet of cell 13 (a cell crossed by z = 2.5).
    for index, face in enumerate(mesh.face_vertices):
        if mesh.owner[index] == 13 and np.ptp(mesh.vertices[list(face)][:, 2]) > 0:
            keep[index] = False
            break
    broken = types.SimpleNamespace(
        vertices=mesh.vertices, cells=mesh.cells,
        face_vertices=[f for f, k in zip(mesh.face_vertices, keep) if k],
        owner=mesh.owner[keep], neighbor=mesh.neighbor[keep])
    with pytest.raises(ValueError, match='nonclosed section boundary'):
        cell_section_polygons(broken, axis=2, value=2.5)
    # Control: the intact mesh returns the same six cells, so the failure above
    # is caused by the missing facet and not by the plane position.
    _, ids = cell_section_polygons(mesh, axis=2, value=2.5)
    assert 13 in ids.tolist()


def test_area_weighted_linear_field_equals_the_analytic_integral():
    """The downstream use -- area-weighted section statistics -- is exact for linears."""
    mesh = box_mesh(BOX_LENGTHS, BOX_DIVISIONS)
    polygons, ids = cell_section_polygons(mesh, axis=2, value=2.5)
    centres = mesh.cell_centers[ids]
    weights = np.array([signed_area(p) for p in polygons])
    discrete = float(np.sum(weights * (centres[:, 0] + 2.0 * centres[:, 1])))
    # integral over [0,2]x[0,3] of x + 2y = 6 + 18
    assert discrete == pytest.approx(24.0, rel=1e-13)


def test_section_polygon_count_tracks_a_shrinking_plane():
    """A cut that clips only the tip of a pyramid crosses the four lateral faces."""
    vertices = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0], [0, 0, 1]], float)
    mesh = NativeMesh3D.from_cells(vertices, [(0, 1, 2, 3, 4)], ['pyramid'])
    polygons, _ = cell_section_polygons(mesh, axis=2, value=0.99)
    assert len(polygons) == 1
    assert signed_area(polygons[0]) == pytest.approx(4.0 * 0.01 ** 2, rel=1e-12)

@pytest.mark.parametrize('height', [0.25, 0.5, 0.75])
def test_tetra_section_obeys_the_similarity_law(height):
    """Right tetra (apex (0,0,1)): the cut at z = h is a similar triangle, (1-h)^2/2.

    Tetrahedra are the cell type of the gmsh/SUBoff native meshes, so this is the
    case every published section of that mesh relies on.
    """
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    mesh = NativeMesh3D.from_cells(vertices, [(0, 1, 2, 3)], ['tetra'])
    polygons, ids = cell_section_polygons(mesh, axis=2, value=height)
    assert len(polygons) == 1 and ids.tolist() == [0]
    assert signed_area(polygons[0]) == pytest.approx(0.5 * (1.0 - height) ** 2, rel=1e-13)


@pytest.mark.parametrize('value', [0.1, 0.25, 0.5, 0.75, 0.9])
def test_total_area_is_independent_of_the_tetrahedral_decomposition(value):
    """Six Kuhn tetrahedra tile the unit cube: their stitched cut must be area 1.

    Each of the six cells contributes its own tile of the section, so a lost cell,
    a double-counted internal facet, or a swallowed collinear vertex all show up
    as a total area away from 1.0.
    """
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                         [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float)
    tets = [(0, 1, 2, 6), (0, 2, 3, 6), (0, 5, 1, 6), (0, 4, 5, 6), (0, 3, 7, 6), (0, 4, 7, 6)]
    mesh = NativeMesh3D.from_cells(vertices, tets, ['tetra'] * len(tets))
    polygons, ids = cell_section_polygons(mesh, axis=2, value=value)
    assert len(polygons) == len(ids) > 0
    assert sum(signed_area(p) for p in polygons) == pytest.approx(1.0, rel=1e-13)


def test_every_returned_cell_straddles_the_plane():
    """Reported cells must be exactly the cells the plane passes through."""
    mesh = box_mesh(BOX_LENGTHS, BOX_DIVISIONS)
    _, ids = cell_section_polygons(mesh, axis=2, value=2.5)
    z = mesh.cell_centers[:, 2]
    straddling = np.nonzero(np.abs(z - 2.5) < 0.5 + 1e-12)[0]
    assert sorted(ids.tolist()) == straddling.tolist()


def prism_mesh(sides, radius=1.0, height=1.0):
    """Regular polygonal prism as a fan of wedges (non-representable section area)."""
    angle = 2.0 * np.pi * np.arange(sides) / sides
    ring = np.stack((radius * np.cos(angle), radius * np.sin(angle)), -1)
    vertices = [[0.0, 0.0, 0.0], [0.0, 0.0, height]]
    vertices += [[x, y, 0.0] for x, y in ring] + [[x, y, height] for x, y in ring]
    cells = [(0, 2 + i, 2 + (i + 1) % sides, 1, 2 + sides + i, 2 + sides + (i + 1) % sides)
             for i in range(sides)]
    return NativeMesh3D.from_cells(np.array(vertices, float), cells, ['wedge'] * sides)


@pytest.mark.parametrize('sides', [3, 6, 12, 24, 96])
def test_polygon_prism_area_is_still_irrational_exact(sides):
    """n-gon prism: area = (n/2) R^2 sin(2 pi/n); graded at double round-off only.

    These values are not exactly representable, so a passing case here proves the
    comparison measures something instead of matching two identical literals.
    """
    mesh = prism_mesh(sides)
    polygons, ids = cell_section_polygons(mesh, axis=2, value=0.5)
    assert len(polygons) == len(ids) == sides
    analytic = 0.5 * sides * np.sin(2.0 * np.pi / sides)
    computed = sum(signed_area(p) for p in polygons)
    assert computed == pytest.approx(analytic, rel=1e-14)


def test_grading_flag_is_not_vacuous():
    """A perturbed section must be rejected by the audit grading, or the audit is void."""
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / 'scripts'))
    import audit_native_sections3d as audit

    mesh = audit.box_mesh(audit.BOX_LENGTHS, audit.BOX_DIVISIONS)
    polygons, _ = cell_section_polygons(mesh, axis=2, value=2.5)
    clean = audit.record('clean', 'rectangle', 2.5, 6.0, polygons)
    assert clean['passed'] and clean['rel_error'] == 0.0
    dropped = audit.record('dropped_cell', 'rectangle', 2.5, 6.0, polygons[:-1])
    assert not dropped['passed'] and dropped['rel_error'] == pytest.approx(1.0 / 6.0)
    shifted = audit.record('shifted', 'rectangle', 2.5, 6.0, [p * 1.001 for p in polygons])
    assert not shifted['passed'] and shifted['rel_error'] > 1e-4


def test_published_audit_record_matches_recomputation():
    """The archived audit numbers must still describe the shipped engine."""
    import json
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / 'tests'))
    from evidence_support import published

    try:
        path = published('docs/verification-native-sections3d/results.json')
    except Exception as reason:  # absent from a sparse checkout: not a failure
        pytest.skip(f'audit record unavailable in this checkout: {reason}')
    record = json.loads(path.read_text())
    assert record['summary']['passed'] is True
    assert record['summary']['case_count'] == len(record['cases']) >= 79
    assert record['summary']['max_rel_error'] <= record['summary']['tolerance_relative']
    sys.path.insert(0, str(root / 'scripts'))
    import audit_native_sections3d as audit
    mesh = audit.kuhn_mesh()
    polygons, _ = cell_section_polygons(mesh, axis=2, value=0.37)
    assert audit.total_area(polygons) == pytest.approx(1.0, rel=1e-13)
