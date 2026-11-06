"""Cubic face gradient vs the O(1) wall-adjacent cell divergence (pinned facts).

The strict ``xfail`` in ``tests/test_native_viscous_order3d.py``
(``test_wall_cell_divergence_must_be_second_order``) states a *wish* for the
default two-point closure.  These tests measure the alternative that is already
in the tree: :class:`tensorfvm.native_cubic_gradient3d.NativeCubicFaceGradient3D`.

Measured on this branch (manufactured field, mu = 1.3, L2 over cells):

===========  ==============  ==============  ===========
closure      wall-cell order interior order  wall L2 n=8
===========  ==============  ==============  ===========
two_point    0.003           2.10            3.342
cubic        2.51 (2.14*)     2.27 (2.05*)    0.111
===========  ==============  ==============  ===========

``*`` warped family (``warp=2``).  The wall term is what decides the verdict:
``dir tau`` is amplified by ``1/h`` in a wall cell, so the closure has to be one
order better than the interior needs.
"""
from functools import lru_cache

import numpy as np

from tensorfvm.verification import cubic_wall_truncation as cw
from tensorfvm.verification import native_order3d as study


@lru_cache(maxsize=None)
def _cubic(n, warp=0.0):
    return cw.truncation_study(n, warp)


@lru_cache(maxsize=None)
def _two_point(n, warp=0.0):
    return study.truncation_study(n, warp)


def test_affine_velocity_carries_no_face_traction_divergence():
    """A cubic reconstruction must still be consistent: linear fields are exact."""
    slope = np.array([[.7, .2, -.1], [-.3, .4, .6], [.15, -.25, .1]])
    intercept = np.array([1.2, .8, -.4])
    for n, warp in ((4, 0.0), (4, 2.0)):
        row = cw.truncation_study(n, warp, values=cw.affine_values(slope, intercept))
        scale = np.linalg.norm(intercept) + np.linalg.norm(slope)
        assert row['linf_global'] < 1e-10*scale, row['linf_global']


def test_wall_cell_divergence_is_second_order_with_the_cubic_closure():
    rows = [_cubic(n) for n in (4, 8)]
    order = cw.orders(rows, 'l2_wall')[0]
    assert order > 1.5, order


def test_interior_divergence_does_not_regress():
    rows = [_cubic(n) for n in (4, 8)]
    order = cw.orders(rows, 'l2_interior')[0]
    assert order > 1.5, order


def test_warped_family_wall_cell_divergence_is_second_order():
    """The repair must survive skew faces, not only the orthogonal family."""
    rows = [_cubic(n, 2.0) for n in (4, 8)]
    order = cw.orders(rows, 'l2_wall')[0]
    assert order > 1.5, order


def test_two_point_closure_is_the_difference_not_the_measure():
    """Negative control: same mesh, same field, same metric, worse closure."""
    for n in (4, 8):
        cubic, baseline = _cubic(n), _two_point(n)
        assert baseline['l2_wall'] > 4.0*cubic['l2_wall'], (n, cubic['l2_wall'], baseline['l2_wall'])
    assert cw.orders([_two_point(n) for n in (4, 8)], 'l2_wall')[0] < 0.3


def test_wall_face_deficit_shrinks_under_refinement():
    """The wall-only share of the error must fall, not merely the total."""
    coarse, fine = (_cubic(n) for n in (4, 8))
    assert fine['l2_wall_wall_faces_only'] < 0.6*coarse['l2_wall_wall_faces_only']


def test_relative_wall_error_is_below_the_three_percent_gate_at_n8():
    """The 3 % gate is met by the cubic closure and missed by the default one."""
    cubic, baseline = _cubic(8), _two_point(8)
    assert cubic['rel_l2_wall'] < 0.03, cubic['rel_l2_wall']
    assert baseline['l2_wall']/baseline['field_scale'] > 0.03


def test_the_fix_is_not_free_and_the_cost_is_recorded():
    """Honest accounting: the cubic stencil samples far more cells than the closure."""
    row = _cubic(8)
    assert row['samples_min'] >= 38
    assert row['face_gradient_nnz'] > 9*row['faces']