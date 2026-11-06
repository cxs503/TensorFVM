"""Negative controls for the wall/corner handling of the cubic face gradient.

Companion to ``tests/test_cubic_face_gradient_wall_cells.py``.  That file scores
the *pooled* wall layer (``l2_wall``), where cells touching two or three box
faces are a minority -- ``12(n-2)+8`` corner/edge cells against
``6n^2-12n+8`` wall cells, i.e. 27 % of the layer at ``n = 8`` -- so a defect
confined to the corners is averaged away by the face-only cells.  This module
scores the masks separately and turns each wrong way of handling the prescribed
wall data into a red test, so that no plausible-looking mean can buy a bad
closure a pass.

Measured on this branch (manufactured field of
:func:`~tensorfvm.verification.native_order3d.manufactured`, ``mu = 1.3``,
``L2`` over cells, ``rel = L2 / max|div tau|``):

=================  ============  ===========  ==========  ==========  ==========
handling           corner order  corner L2    face L2     corner rel  affine linF
=================  ============  ===========  ==========  ==========  ==========
all_wall_data      2.38          1.27e-01     1.05e-01    2.6e-03     4.2e-13
one_axis_only      -1.61         1.57e+02     1.58e+01    3.25        1.4e+03
no_dirichlet_data  -1.89         3.20e+02     2.51e+02    6.62        2.0e+03
sign_flipped       -1.89         6.41e+02     5.03e+02    13.2        4.1e+03
=================  ============  ===========  ==========  ==========  ==========

``corner`` = cell centre within one wall layer of two or three box faces (box
edges and box corners); ``face`` = exactly one.  Errors at ``n = 8``, orders
between ``n = 4`` and ``n = 8``.  ``affine linF`` is the ``Linf`` face-gradient
error on a linear velocity field at ``n = 4``: a cubic closure represents it
exactly, so anything non-zero there is closure error with no truncation term to
hide behind.

Only the first row is an acceptable handling.  In a wall cell the traction is
amplified by ``1/h``, so losing one order at a corner loses the divergence;
every wrong row misses the 3 % gate by two to four orders of magnitude.  The
handling modes live in
:func:`tensorfvm.verification.cubic_wall_truncation.wrong_handling`, shared with
``scripts/audit_cubic_face_gradient_corner_negatives.py`` so the audit and this
regression cannot drift apart.
"""
from functools import lru_cache

import numpy as np

from tensorfvm.verification import cubic_wall_truncation as cw
from tensorfvm.verification import native_order3d as study

LEVELS = (4, 8)
SLOPE = np.array([[.7, .2, -.1], [-.3, .4, .6], [.15, -.25, .1]])
INTERCEPT = np.array([1.2, .8, -.4])


@lru_cache(maxsize=None)
def _study(n, warp=0.0, mode='all_wall_data', affine=False):
    """Corner-split truncation row; the operator behind it is cached, this is not."""
    values = cw.affine_values(SLOPE, INTERCEPT) if affine else None
    return cw.corner_study(n, warp, mode=mode, values=values)


def _orders(key, mode='all_wall_data'):
    return study.orders([_study(n, mode=mode) for n in LEVELS], key)


def test_the_corner_mask_is_the_minority_of_the_wall_layer():
    """Guards the metric: corner/edge cells must stay the hidden minority.

    If the masks below did not partition the wall layer the way
    ``cw.wall_mask`` does, the negative controls would stop being
    corner-specific and the companion test's pooled mean would be the same
    measurement twice.
    """
    for n in LEVELS:
        mesh, _ = cw.cubic_operator(n)
        axes = cw.wall_axes(mesh, n)
        assert (axes >= 2).sum() == 12*(n - 2) + 8, n     # box edges + box corners
        assert (axes == 1).sum() == 6*(n - 2)**2, n       # face-only wall cells
        assert (axes == 0).sum() == (n - 2)**3, n         # interior cells
        assert np.array_equal(cw.wall_mask(mesh, n), axes >= 1), n
    fine = cw.wall_axes(cw.cubic_operator(LEVELS[-1])[0], LEVELS[-1])
    assert (fine >= 2).sum() < 0.5*(fine >= 1).sum()


def test_corner_cells_keep_second_order_when_every_wall_face_is_used():
    """The reference handling: corners converge and clear the 3 % gate."""
    corner_order = _orders('l2_corner')[0]
    assert corner_order > 1.5, corner_order
    fine = _study(LEVELS[-1])
    assert fine['rel_corner'] < 0.01, fine['rel_corner']
    assert fine['rel_face'] < 0.01, fine['rel_face']


def test_corner_cells_do_not_converge_if_the_wall_data_is_ignored():
    """Negative control: no Dirichlet data -- the corner error grows on refinement."""
    for n in LEVELS:
        good, bad = _study(n), _study(n, mode='no_dirichlet_data')
        assert bad['l2_corner'] > 100*good['l2_corner'], (n, good['l2_corner'], bad['l2_corner'])
    assert _orders('l2_corner', 'no_dirichlet_data')[0] < 0.3
    assert _study(LEVELS[-1], mode='no_dirichlet_data')['rel_corner'] > 1.0


def test_corner_cells_need_the_wall_data_of_every_face_direction():
    """Negative control: a corner cell that keeps one wall direction only.

    Dropping a wall-face source also damages the gradients of the faces that
    sample it, so the face-only cells degrade too; the corner cells are hit an
    order of magnitude harder, which is the signature pinned here.
    """
    good, bad = _study(LEVELS[-1]), _study(LEVELS[-1], mode='one_axis_only')
    assert bad['l2_corner'] > 100*good['l2_corner'], (good['l2_corner'], bad['l2_corner'])
    assert bad['rel_corner'] > 0.03
    assert bad['l2_corner']/bad['l2_face'] > 5.0*good['l2_corner']/good['l2_face']
    assert _orders('l2_corner', 'one_axis_only')[0] < 0.3


def test_the_sign_of_the_wall_source_is_not_free_at_a_corner():
    """Negative control: a sign error is visible on a field the closure knows.

    A linear velocity field is reproduced exactly, so the correct closure must
    return round-off there while the flipped source cannot; both wall masks are
    checked so the defect cannot hide on one of them.
    """
    scale = np.linalg.norm(INTERCEPT) + np.linalg.norm(SLOPE)
    for warp in (0.0, 2.0):
        good = _study(4, warp, affine=True)
        bad = _study(4, warp, mode='sign_flipped', affine=True)
        assert good['linf_corner'] < 1e-10*scale, (warp, good['linf_corner'])
        assert good['linf_face'] < 1e-10*scale, (warp, good['linf_face'])
        assert bad['linf_corner'] > 1e-2*scale, (warp, bad['linf_corner'])
        assert bad['linf_face'] > 1e-2*scale, (warp, bad['linf_face'])


def test_a_sign_error_also_breaks_the_corner_traction_ledger():
    """The same defect on the manufactured field: it must not look like noise."""
    good, bad = _study(LEVELS[-1]), _study(LEVELS[-1], mode='sign_flipped')
    assert bad['l2_corner'] > 100*good['l2_corner'], (good['l2_corner'], bad['l2_corner'])
    assert good['rel_corner'] < 0.03 and bad['rel_corner'] > 0.03
    assert _orders('l2_corner', 'sign_flipped')[0] < 0.3