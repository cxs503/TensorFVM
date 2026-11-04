"""Criterion test for ``conservative_pressure.ConservativePressureGradient``.

The mutation audit (``scripts/audit_mutation_survival.py``) recorded
``CP-WEIGHT-SWAP`` -- the projected-linear face weights applied to the wrong
owner/neighbour -- as a *surviving* mutant, because it was not turned into a
criterion-based test: the earlier attempt compared an affine field with a face
value on a skewed probe and failed **on the unmutated source**, so it was
removed rather than tuned (see ``ACCEPTED_GAPS`` in
``tests/test_mutation_coverage.py``).  The recorded remaining work was "a face
value whose exactness is independent of the weight convention".

Two convention-free criteria are used here, both consequences of the scheme's
own definition -- projected-linear interpolation places the face value on the
line joining the two cell centres, at the fraction obtained by projecting the
face-centre offset onto the face normal:

1. **Share criterion.**  On a two-cell orthogonal strip, a field that is 0 on the
   left cell and 1 on the right cell must interpolate to the *geometric* share
   of the right cell at their common face, ``(x_f - x_o)/(x_n - x_o)``.  This
   reads a physical coordinate ratio, never the source's weight convention.
2. **Exactness criterion.**  On an orthogonal graded mesh the projected fraction
   *is* the geometric fraction, so any affine field must be reproduced at every
   interior face to round-off, however strongly the cells are graded.  This is
   the second-order exactness statement behind the scheme's name.

Stretching supplies the teeth: a graded mesh has ``fraction != 1/2``, so moving
the bracket from owner to neighbour changes the answer by
``|a.d| * |1 - 2f|`` = O(1), while on a uniform mesh the two conventions are
indistinguishable by symmetry (the sweep test asserts exactly that).
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import torch

from tensorfvm.conservative_pressure import ConservativePressureGradient

DTYPE = torch.float64

#: Cell widths of the orthogonal strip, deliberately geometric (grading 4:1).
GRADED_WIDTHS = (4.0, 1.0, 1.0, 4.0)


def _graded_strip(widths, height=1.0):
    """Orthogonal strip of axis-aligned rectangles, one cell high.

    Returns ``(probe, geometry)`` where ``probe`` carries exactly the attributes
    ``_face_owner_weights``/``_interpolate`` read, and ``geometry`` the raw
    arrays the exactness check needs.  Face area vectors are owner-outward
    normals scaled by face length, matching ``body_fitted``'s ``self.S``.
    """
    nx = len(widths)
    nodes = [0.0]
    for w in widths:
        nodes.append(nodes[-1] + w)
    centers = torch.tensor(
        [[(nodes[i] + nodes[i + 1]) / 2, height / 2] for i in range(nx)], dtype=DTYPE
    )

    owner, neighbor, area_vectors, face_centers = [], [], [], []

    def add(o, n, s, c):
        owner.append(o)
        neighbor.append(n)
        area_vectors.append(s)
        face_centers.append(c)

    interior = []
    for i in range(nx - 1):
        interior.append(len(owner))
        add(i, i + 1, (height, 0.0), (nodes[i + 1], height / 2))
    add(0, -1, (-height, 0.0), (nodes[0], height / 2))
    add(nx - 1, -1, (height, 0.0), (nodes[-1], height / 2))
    for i in range(nx):
        w = nodes[i + 1] - nodes[i]
        add(i, -1, (0.0, -w), ((nodes[i] + nodes[i + 1]) / 2, 0.0))
        add(i, -1, (0.0, w), ((nodes[i] + nodes[i + 1]) / 2, height))

    o = torch.tensor(owner, dtype=torch.long)
    ni = torch.tensor(neighbor, dtype=torch.long)
    f = torch.tensor(interior, dtype=torch.long)
    S = torch.tensor(area_vectors, dtype=DTYPE)
    fc = torch.tensor(face_centers, dtype=DTYPE)
    d = fc - centers[o]
    d[f] = centers[ni[f]] - centers[o[f]]

    probe = ConservativePressureGradient()
    probe.mesh = SimpleNamespace(face_centers=fc, centers=centers)
    probe.o, probe.f = o, f
    probe.oi, probe.ni = o[f], ni[f]
    probe.count = nx
    probe.volume = torch.tensor(list(widths), dtype=DTYPE)
    probe.S, probe.d = S, d
    probe.p = torch.zeros(nx, dtype=DTYPE)
    return probe, dict(nodes=nodes, centers=centers, face_centers=fc,
                       interior=f, neighbor=ni, owner=o)


def _affine(points, a=(3.7, -2.1), b=0.6):
    return points[..., 0] * a[0] + points[..., 1] * a[1] + b


def test_face_share_of_a_step_field_is_the_geometric_share():
    """A 0/1 step across one face must interpolate to the geometric split.

    Two-cell strip: widths 4 and 1, face at x=4, centres at x=2 and x=4.5.  The
    right-cell share is (4-2)/(4.5-2) = 0.8.  The bracket swap returns 0.2, so
    the criterion separates the two conventions without naming either.
    """
    probe, geometry = _graded_strip((4.0, 1.0))
    face = int(geometry["interior"][0])
    share = (geometry["face_centers"][face, 0] - geometry["centers"][0, 0]) / (
        geometry["centers"][1, 0] - geometry["centers"][0, 0]
    )
    assert math.isclose(float(share), 0.8, abs_tol=1e-15)  # teeth: far from 1/2
    step = torch.tensor([0.0, 1.0], dtype=DTYPE)
    values = probe._interpolate(step)
    assert math.isclose(float(values[face]), float(share), abs_tol=1e-15)
    # and the owner weight is the complementary share, as its name states
    assert math.isclose(float(probe._face_owner_weights()[face]), 1 - float(share),
                        abs_tol=1e-15)


def test_affine_field_is_exact_at_interior_faces_of_a_graded_orthogonal_mesh():
    """Second-order exactness of projected-linear interpolation (kills CP-WEIGHT-SWAP).

    Orthogonal geometry makes the projection identity exact, so the interpolant
    of any affine field at a face centre must equal the analytic value to
    round-off even at a 4:1 grading.
    """
    probe, geometry = _graded_strip(GRADED_WIDTHS)
    interior = geometry["interior"]
    field = _affine(geometry["centers"])
    values = probe._interpolate(field)
    reference = _affine(geometry["face_centers"][interior])
    relative = ((values[interior] - reference).abs() / reference.abs()).max()
    assert relative < 1e-14, relative

    # kill margin of the bracket swap (= the failure mode this criterion targets)
    w = probe._face_owner_weights()[interior]
    owner_all, neighbor_all = geometry["owner"][interior], geometry["neighbor"][interior]
    swapped = w * field[neighbor_all] + (1 - w) * field[owner_all]
    margin = (swapped - reference).abs()
    # the equal-width interior face sits at the midpoint (w = 1/2) and cannot
    # separate the conventions; the graded faces carry the whole kill margin
    assert margin.max() > 0.5, margin
    assert int((margin > 0.5).sum()) >= 2, margin

    # a constant field is reproduced trivially, so the criterion above is not
    # merely asserting that the call returns something finite
    constant = torch.full((probe.count,), 2.5, dtype=DTYPE)
    assert torch.allclose(probe._interpolate(constant)[interior],
                          torch.full((len(interior),), 2.5, dtype=DTYPE), atol=0.0)


def test_grading_sweep_shows_the_criterion_has_teeth_for_every_ratio():
    """Restricting the criterion to one grading would be a snapshot; sweep it."""
    for ratio in (1.0, 1.5, 2.0, 4.0, 8.0):
        probe, geometry = _graded_strip((ratio, 1.0, 1.0, ratio))
        interior = geometry["interior"]
        field = _affine(geometry["centers"])
        error = (probe._interpolate(field)[interior]
                 - _affine(geometry["face_centers"][interior])).abs().max()
        assert error < 1e-12, (ratio, error)
        w = probe._face_owner_weights()[interior]
        oi, ni = geometry["owner"][interior], geometry["neighbor"][interior]
        margin = ((1 - 2 * w).abs() * (field[ni] - field[oi])).abs().max()
        if math.isclose(ratio, 1.0):
            # uniform mesh: the two conventions coincide, hence only grading
            # gives this criterion its teeth
            assert margin < 1e-14, margin
        else:
            assert margin > 0.2, (ratio, margin)


def test_weights_bracket_the_face_for_an_accepted_projection():
    """Guard on the production validity check, exercised on the graded strip."""
    probe, _ = _graded_strip((8.0, 1.0))
    w = probe._face_owner_weights()[probe.f]
    assert torch.all(w > 0.0) and torch.all(w < 1.0), w