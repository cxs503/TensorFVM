"""Tests for :mod:`tensorfvm.radiation3d` (3-D polygon view factors).

Anchors are independent of the method under test: Hamilton & Morgan and
Hottel & Sarofim closed forms, the closed-box summation rule, and the exact 2-D
crossed-strings limit.  The degenerate-pair tests exist to keep the *measured*
failures of the ``ln r`` contour identity load-bearing -- if someone removes the
validity gate, they fail loudly instead of returning a wrong number.
"""
import math

import numpy as np
import pytest

from tensorfvm.radiation3d import (
    Polygon3D, box_enclosure, coaxial_disk_form_factor, contour_pair_is_valid,
    exchange_area, exchange_area_contour, exchange_area_matrix,
    parallel_rectangles_form_factor, perpendicular_common_edge_form_factor,
    point_form_factor, reciprocity_residual, regular_polygon_disk,
    summation_residual, view_factor_matrix,
)

SQUARE = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
CUBE_ADJ = (Polygon3D([(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)]),
            Polygon3D([(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)]))


@pytest.fixture(scope="module")
def cube():
    faces = box_enclosure((1.0, 1.0, 1.0))
    af = exchange_area_matrix(faces, order=12)
    areas = np.array([f.area for f in faces])
    return faces, af, af / areas[:, None]


def test_polygon_geometry():
    p = Polygon3D(SQUARE)
    assert p.area == pytest.approx(1.0, rel=1e-15)
    assert np.allclose(p.centroid, [0.5, 0.5, 0.0], atol=1e-15)
    assert abs(abs(float(p.normal[2])) - 1.0) < 1e-15


def test_polygon_rejects_bad_geometry():
    with pytest.raises(ValueError):
        Polygon3D([(0, 0, 0), (1, 0, 0), (0, 1, 1), (1, 1, 0)])   # non-planar
    with pytest.raises(ValueError):
        Polygon3D([(0, 0, 0), (1, 0, 0), (2, 0, 0)])               # collinear


def test_box_enclosure_faces():
    faces = box_enclosure((2.0, 1.0, 0.5))
    assert [round(f.area, 12) for f in faces] == [0.5, 0.5, 1.0, 1.0, 2.0, 2.0]
    outward = [tuple(np.round(f.normal, 12)) for f in faces]
    assert outward == [(-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, -1.0, 0.0),
                       (0.0, 1.0, 0.0), (0.0, 0.0, -1.0), (0.0, 0.0, 1.0)]


@pytest.mark.parametrize("gap", [1.0, 0.5])
def test_parallel_squares_match_closed_form(gap):
    a = Polygon3D(SQUARE)
    b = Polygon3D([(0, 0, gap), (0, 1, gap), (1, 1, gap), (1, 0, gap)])
    computed = exchange_area(a, b, order=24) / a.area
    closed = parallel_rectangles_form_factor(1.0, 1.0, gap)
    assert computed == pytest.approx(closed, rel=1e-9)


def test_perpendicular_common_edge_matches_symmetry_and_closed_form():
    f_opp = parallel_rectangles_form_factor(1.0, 1.0, 1.0)
    symmetry = (1.0 - f_opp) / 4.0
    closed = perpendicular_common_edge_form_factor(1.0, 1.0, 1.0)
    assert closed == pytest.approx(symmetry, rel=1e-12)
    p, q = CUBE_ADJ
    assert exchange_area(p, q, order=24) / p.area == pytest.approx(closed, rel=1e-10)


def test_coplanar_pair_has_zero_exchange():
    a = Polygon3D(SQUARE)
    b = Polygon3D([(1, 0, 0), (2, 0, 0), (2, 1, 0), (1, 1, 0)])
    assert abs(exchange_area(a, b, order=24)) < 1e-14


def test_cube_summation_rule_and_reciprocity(cube):
    _, af, f = cube
    assert summation_residual(f) < 1e-12
    assert reciprocity_residual(af) < 1e-12
    f_opp = parallel_rectangles_form_factor(1.0, 1.0, 1.0)
    assert f[4, 5] == pytest.approx(f_opp, rel=1e-10)
    assert f[0, 3] == pytest.approx((1.0 - f_opp) / 4.0, rel=1e-10)


def test_view_factor_matrix_matches_exchange_area(cube):
    faces, af, _ = cube
    f = view_factor_matrix(faces, order=12)
    assert np.allclose(f * np.array([x.area for x in faces])[:, None], af, rtol=0, atol=1e-15)


def test_two_sided_invariance():
    a = Polygon3D(SQUARE)
    b = Polygon3D([(0, 0, 1), (1, 0, 1), (1, 1, 2), (0, 1, 2)])
    base = exchange_area(a, b, order=16)
    assert exchange_area(a, Polygon3D(b.coords[::-1]), order=16) == pytest.approx(base, rel=1e-13)
    assert exchange_area(Polygon3D(a.coords[::-1]), b, order=16) == pytest.approx(base, rel=1e-13)


def test_contour_matches_quadrature_when_valid():
    a = Polygon3D(SQUARE)
    for b in (Polygon3D([(0, 0, 1), (0, 1, 1), (1, 1, 1), (1, 0, 1)]),
              Polygon3D([(0, 0, 2), (0, 1, 2), (0, 1, 3), (0, 0, 3)]),
              Polygon3D([(0, 0, 1), (1, 0, 1), (1, 1, 2), (0, 1, 2)])):
        assert contour_pair_is_valid(a, b)
        quad = exchange_area(a, b, order=24, method="quadrature")
        assert exchange_area_contour(a, b) == pytest.approx(quad, rel=1e-12)


def test_contour_gate_rejects_degenerate_pairs():
    """The ``ln r`` identity is invalid for touching/coplanar pairs (measured)."""
    for pair in (CUBE_ADJ,
                 (Polygon3D(SQUARE), Polygon3D([(1, 0, 0), (2, 0, 0), (2, 1, 0), (1, 1, 0)])),
                 (Polygon3D(SQUARE),
                  Polygon3D([(1.5, 0, 0), (2.5, 0, 0), (2.5, 1, 0), (1.5, 1, 0)]))):
        p, q = pair
        assert not contour_pair_is_valid(p, q)
        quad = exchange_area(p, q, order=24, method="quadrature")
        # 'auto' must fall back to quadrature, not to the broken contour path
        assert exchange_area(p, q, order=24) == pytest.approx(quad, abs=1e-15)
        # and the contour path must stay visibly wrong (gate is not cosmetic)
        cont = exchange_area_contour(p, q)
        if abs(quad) > 1e-9:
            assert abs(cont - quad) / abs(quad) > 0.25


def test_point_form_factor_contract():
    target = Polygon3D([(0, 0, 1), (0, 1, 1), (1, 1, 1), (1, 0, 1)])
    emitter = Polygon3D(SQUARE)
    with pytest.raises(ValueError):
        point_form_factor((0.5, 0.5, 0.0), target)
    value = point_form_factor((0.5, 0.5, 0.0), target, emitter.normal)
    assert 0.0 <= value <= 1.0
    # a differential element at the centre of a face sees a directly opposed unit
    # square at d=1 better than the face average does (0.2395 vs 0.1998); the
    # average is the anchored quantity, this only brackets the point value.
    average = exchange_area(emitter, target, order=24) / emitter.area
    assert average < value < 1.25 * average


def test_disk_facet_convergence():
    closed = coaxial_disk_form_factor(1.0, 1.0, 1.0)
    errs = []
    for n in (24, 48):
        d1, d2 = (regular_polygon_disk(1.0, n, (0.0, 0.0, 0.0)),
                  regular_polygon_disk(1.0, n, (0.0, 0.0, 1.0)))
        errs.append(abs(exchange_area(d1, d2, order=16) / d1.area - closed) / closed)
    assert errs[0] < 2e-2
    assert errs[1] < errs[0] / 2.0


def test_strip_limit_approaches_crossed_strings():
    """Long 3-D strips must recover the exact 2-D crossed-strings factor."""
    f2d = math.sqrt(2.0) - 1.0
    length = 50.0
    p = Polygon3D([(0, 0, 0), (length, 0, 0), (length, 1, 0), (0, 1, 0)])
    q = Polygon3D([(0, 0, 1), (0, 1, 1), (length, 1, 1), (length, 0, 1)])
    computed = exchange_area(p, q, order=16, method="quadrature") / p.area
    assert abs(computed - f2d) / f2d < 2e-2


def test_gray_enclosure_energy_closure(cube):
    from tensorfvm.radiation import GrayEnclosure
    faces, _, f = cube
    areas = [x.area for x in faces]
    temps = [400.0, 300.0, 300.0, 300.0, 300.0, 300.0]
    eps = [1.0, 0.6, 0.6, 0.6, 0.6, 0.6]
    enclosure = GrayEnclosure(f, areas, eps, temps,
                              reradiating=[True, False, False, False, False, False])
    assert enclosure.energy_residual() < 1e-12
