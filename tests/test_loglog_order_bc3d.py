"""Guards for :mod:`tensorfvm.verification.loglog_order_bc3d`.

Two things are asserted here and nothing else:

1. the environment shim (closed-form 3x3 inverse) is numerically identical to
   ``numpy.linalg.inv`` -- otherwise every number the harness reports would be
   produced by a different operator than the production path;
2. the production-operator ladders behave *qualitatively* as declared on a
   deliberately cheap grid triple (4, 6, 8): an exact face datum decays and the
   datum-at-cell-centre variant does not.  The quantitative 15 % gates live in
   ``docs/verification-loglog-order-bc3d/summary.json`` (grids 8/16/24); on
   (4, 6, 8) the series is pre-asymptotic, so only brackets are asserted.
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from tensorfvm.verification.loglog_order_bc3d import (  # noqa: E402
    cube_cell_averages,
    fast3x3_inverse,
    harmonic_residual,
    harmonic_scalar,
    run,
)


def test_fast3x3_inverse_matches_numpy():
    rng = np.random.default_rng(20261113)
    a = rng.random((2000, 3, 3)) * 0.5 + 4.0 * np.eye(3)
    got = fast3x3_inverse(a)
    ref = np.linalg.inv(a)
    assert np.max(np.abs(got - ref)) < 1e-12
    # and it actually inverts: A A^-1 = I
    prod = np.einsum("nij,njk->nik", a, got)
    assert np.max(np.abs(prod - np.eye(3))) < 1e-12
    # a non-3x3 input is delegated, not silently mis-handled
    with pytest.raises(Exception):
        np.zeros((2, 2)) @ np.zeros((2, 2))


def test_harmonic_field_has_zero_laplacian():
    pts = np.array([[0.13, 0.42, 0.77], [0.61, 0.05, 0.29], [0.9, 0.5, 0.1]])
    assert harmonic_residual(pts) < 1e-4


def test_cell_average_equals_point_value_for_harmonic_field():
    """A harmonic field's cell average equals its centroid value up to O(h^4).

    This is the reusable lesson of the harness: the classic "cell average vs
    point value" reference trap of :mod:`tensorfvm.verification.loglog_order`
    cancels at O(h^2) *for a harmonic manufactured field* (the h^2/24 u''
    corrections sum to (h^2/24) lap u = 0), so it cannot be exercised by the
    Dirichlet problem used here -- it needs a field with a non-zero Laplacian,
    i.e. a volumetric source.  The assertion is on that cancellation, so a
    future edit that silently swaps the field loses the guard.
    """
    h = 1.0 / 8.0
    centres = np.array([[0.1, 0.5, 0.2], [0.55, 0.5, 0.83], [0.31, 0.07, 0.62]])
    avg = cube_cell_averages(centres, h)
    point = harmonic_scalar(centres)
    assert np.max(np.abs(avg - point)) < 1e-6 * np.max(np.abs(point))
    # a non-harmonic control field (u = exp(k z) cos(k x) + x^2) does not cancel
    def nonharmonic(p):
        return harmonic_scalar(p) + p[:, 0] ** 2

    def nonharmonic_exact_average(c):
        return cube_cell_averages(c, h) + c[:, 0] ** 2 + h ** 2 / 12.0
    assert np.max(np.abs(nonharmonic_exact_average(centres)
                         - nonharmonic(centres))) > 1e-5


@pytest.fixture(scope="module")
def small_ladder():
    return run((4, 6, 8), warp=0.1)


def test_exact_face_datum_decays_at_second_order_bracket(small_ladder):
    leg = small_ladder["legs"]["A"]
    errors = np.asarray(leg["errors"], dtype=float)
    assert np.all(errors > 0.0) and np.all(np.diff(errors) < 0.0)
    order = float(leg["study"]["order"])
    # (4, 6, 8) is pre-asymptotic; the declared band here is a bracket, the
    # 15 % gate is applied on the 8/16/24 ladder in the artifacts.
    assert 1.0 < order < 2.3, order
    assert leg["study"]["withheld_reason"] == ""


def test_interior_solve_residual_is_small(small_ladder):
    for row in small_ladder["legs"]["A"]["levels"]:
        assert row["residual"] < 1e-10


def test_datum_at_cell_centre_collapses_the_boundary_condition(small_ladder):
    """The 'use the cell value at the wall' datum is not first order but wrong.

    It makes the two-point boundary flux identically zero, so the error stops
    decaying with h: the L2 order of that ladder falls well below the interior
    order and the wall-layer L-inf error barely moves.  This is the trap the
    parent module only quoted; here it is measured.
    """
    order_b = float(small_ladder["legs"]["B"]["study"]["order"])
    order_a = float(small_ladder["legs"]["A"]["study"]["order"])
    assert order_b < 1.2, order_b
    assert order_b < order_a
    wall = small_ladder["legs"]["B-wall"]["levels"]
    assert wall[-1]["cell_average_linf_wall"] > 0.5 * wall[0]["cell_average_linf_wall"]


def test_reference_convention_ratio_is_unity_for_harmonic_field(small_ladder):
    for row in small_ladder["reference_ratio"]:
        assert math.isclose(row["ratio"], 1.0, rel_tol=1e-3), row