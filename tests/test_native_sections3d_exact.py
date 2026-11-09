"""Exact-section verification of ``native_sections3d.cell_section_polygons``.

Closes the verification debt that ``native_sections3d`` carried: no test in
``tests/`` referenced the module, yet every native section figure in
``docs/verification-*`` is drawn from its polygons, so a wrong polygon corrupted
those figures silently.

Each test names the quantity it pins and the anchor it is pinned to; all anchors
are closed forms of the discretisation actually handed to the routine (regular
n-gon rings, unit square), so the tolerances are machine-precision tolerances and
not "looks plausible" bands.
"""
from __future__ import annotations

import numpy as np
import pytest

import tensorfvm.verification.native_sections as vs

R_OUT, R_IN = 1.0, 0.5


def test_box_section_area_is_the_closed_form_square():
    """A plane 0 < z < a through [0,a]^3 has area a^2, split over nx*ny quads."""
    for n in (1, 2, 4, 8):
        mesh = vs.box_mesh(n, n, n)
        polygons, ids, areas = vs.section_areas(mesh, value=0.5)
        expected = n * n if n % 2 else 2 * n * n   # even n -> plane on a grid plane
        assert areas.size == expected, f"n={n}: got {areas.size} polygons, want {expected}"
        assert abs(vs.unique_section_area(mesh, value=0.5) - 1.0) < 1e-14, n
        assert np.all(areas > 0.0), "every emitted polygon must have positive area"
        assert np.allclose(areas.sum() / (expected / (n * n)), 1.0, atol=1e-14)
        assert np.unique(ids).size == ids.size, "cell ids must be unique"
        assert np.all((ids >= 0) & (ids < n * n * n))


def test_section_plane_coincident_with_an_interior_grid_plane_is_double_sided():
    """Coplanar facet edges must cancel their internal diagonals, per owner.

    With n = 2 the plane z = 0.5 *is* an interior face of the mesh: both owners of
    that face emit their own square ring (4 vertices, area 1/4). Pinned semantics:
    an 8-polygon, 2.0-area *polygon sum* with a 1.0 geometric (deduplicated) area.
    A naive "collect every facet edge once" implementation instead leaves the
    shared face with doubled edges and raises or emits a diagonal polygon, and a
    consumer that quotes ``sum(polygon_area)`` as the section area is off by 2x —
    which is why :func:`unique_section_area` exists.
    """
    mesh = vs.box_mesh(2, 2, 2)
    polygons, _, areas = vs.section_areas(mesh, value=0.5)
    assert areas.size == 8, "both owners of the coincident face must be emitted"
    assert max(len(p) for p in polygons) == 4, "rings must stay quadrilaterals"
    assert np.allclose(areas, 0.25, rtol=0.0, atol=1e-15)
    assert abs(areas.sum() - 2.0) < 1e-14
    assert abs(vs.unique_section_area(mesh, value=0.5) - 1.0) < 1e-14


def test_graded_box_section_is_still_the_exact_square():
    """The stitching is geometric, so a non-uniform grid must not change the area."""
    graded = vs.box_mesh(xs=[0.0, 0.05, 0.3, 1.0], ys=[0.0, 0.5, 1.0],
                         zs=[0.0, 0.9, 1.0])
    _, _, areas = vs.section_areas(graded, value=0.5)
    assert abs(areas.sum() - 1.0) < 1e-14


def test_boundary_coincident_plane_keeps_the_face_and_outside_plane_is_empty():
    """Plane on a boundary face -> that face (area 1); plane outside -> nothing."""
    polygons, _, on_face = vs.section_areas(vs.box_mesh(1, 1, 1), value=0.0)
    assert on_face.size == 1
    assert abs(on_face.sum() - 1.0) < 1e-14
    assert abs(vs.unique_section_area(vs.box_mesh(1, 1, 1), value=0.0) - 1.0) < 1e-14
    corners = np.array(sorted(map(tuple, np.round(polygons[0], 12))))
    assert np.allclose(corners, [[0, 0], [0, 1], [1, 0], [1, 1]])
    _, _, outside = vs.section_areas(vs.box_mesh(1, 1, 1), value=1.0001)
    assert outside.size == 0, "a plane outside the mesh must return no polygon"


def test_ring_section_matches_the_closed_form_ring_area():
    """O-grid annulus section == (n/2) sin(2pi/n)(R^2 - r^2), to machine precision.

    The grid lines *are* the vertices of the two regular n-gons, so there is no
    truncation left between the discretisation and the closed form: any gap is a
    stitching defect.
    """
    for n in (8, 16, 32, 64):
        _, _, areas = vs.section_areas(vs.annulus_mesh(n), value=0.1)
        exact = vs.concentric_ring_area(n, R_OUT, R_IN)
        assert areas.size == 2 * n, f"n={n}: one polygon per cell expected"
        assert abs(areas.sum() - exact) / exact < 1e-13, (
            f"n={n}: section {areas.sum()!r} vs closed form {exact!r}")
        assert np.all(areas > 0.0)


def test_ring_section_converges_to_the_continuum_annulus_with_sin_over_t():
    """Refinement tracks the pure quadrature deficit ``sin(t)/t``, t = 2pi/n.

    This is a *predictive* check: the analytic deficit is compared with the
    measured one, so a section that merely "converges somehow" cannot pass.
    """
    previous = None
    for n in (8, 16, 32, 64):
        _, _, areas = vs.section_areas(vs.annulus_mesh(n), value=0.1)
        measured = areas.sum() / vs.pi_annulus_area(R_OUT, R_IN) - 1.0
        analytic = vs.ring_error_ratio(n)
        assert abs(measured - analytic) < 1e-13, (
            f"n={n}: measured deficit {measured!r} vs analytic {analytic!r}")
        assert measured < 0.0, "inscribed n-gons underestimate the annulus"
        if previous is not None:
            assert abs(measured) < abs(previous), "refinement must reduce the deficit"
        previous = measured
    # quartering n -> deficit / 16 (O(h^2))
    coarse = abs(vs.ring_error_ratio(16))
    fine = abs(vs.ring_error_ratio(64))
    assert 12.0 < coarse / fine < 20.0, f"deficit ratio {coarse / fine:.2f} (want ~16)"


def test_convex_hull_negative_control_is_what_the_stitch_avoids():
    """A hull of the same ring loses the hole and overestimates by R^2/(R^2-r^2).

    This is the negative control for claim (1): if the implementation had fallen
    back to a convex hull, the measured area would be the hull area (+33.3 % at
    r = R/2) instead of the ring area.
    """
    n = 16
    _, _, areas = vs.section_areas(vs.annulus_mesh(n), value=0.1)
    exact = vs.concentric_ring_area(n, R_OUT, R_IN)
    hull = vs.hull_annulus_area(n, R_OUT)
    assert abs(hull / exact - 1.0 - 1.0 / 3.0) < 1e-3, hull / exact - 1.0
    assert abs(areas.sum() - exact) / exact < 1e-13
    assert hull / areas.sum() - 1.0 > 0.33, "measured area must not be the hull"


def test_vertex_tangency_is_dropped_and_a_genuine_face_is_kept():
    """A plane touching only one vertex has no finite area -> no polygon, no raise."""
    _, _, tangency = vs.section_areas(vs.tetra_mesh(), value=1.0)
    assert tangency.size == 0, "a single-vertex tangency must be omitted"
    _, _, face = vs.section_areas(vs.tetra_mesh(), value=0.0)
    assert face.size == 1
    assert abs(face.sum() - 0.5) < 1e-14, "z=0 face of the reference tetra is 1/2"


def test_section_study_record_is_complete_and_self_consistent():
    """The shipped study record must pass every claim it publishes."""
    record = vs.section_study(n_theta=(8, 16, 32))
    claims = record["claims"]
    assert claims["box_unique_area_max_error"] < 1e-14
    assert claims["box_polygon_count_matches_semantics"] is True
    assert claims["coincident_planes_are_double_sided"] is True
    assert claims["box_polygon_sum_excess_equals_two"][0] == 1.0
    assert claims["box_polygons_stay_quads"] == 4
    assert claims["graded_box_unique_area_error"] < 1e-14
    assert claims["boundary_face_unique_area_error"] < 1e-14
    assert claims["outside_plane_empty"] is True
    assert claims["ring_closed_form_max_rel_error"] < 1e-13
    assert claims["ring_polygon_count_ok"] is True
    assert claims["ring_deficit_tracks_analytic"] < 1e-13
    assert claims["tangency_dropped"] is True
    assert abs(claims["tangency_face_area"] - 0.5) < 1e-14
    assert claims["hull_control_rel_overestimate"] > 0.33