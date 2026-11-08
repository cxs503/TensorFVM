"""Scope gates for ``NativeViscous3D(wall_closure='three_point')`` away from
orthogonal meshes.

``docs/native-viscous-three-point-closure/report.md`` stated the one-sided
three-point wall stencil is "exact for affine fields".  These gates re-test the
claim on meshes whose *cells* are laterally displaced (interior warp, so the
line owner-centre -> second-centre is no longer normal to the face).

Two states are pinned, because both are results:

* ``wall_second_projection=False`` (the pre-2026-11-08 behaviour) loses affine
  exactness on a warped mesh and the residual is *exactly* ``coeff2 * (t . g)``
  -- the negative result of ``docs/native-viscous-three-point-closure`` must not
  be silently reworded;
* ``wall_second_projection=True`` (the default) moves the second-cell datum onto
  the normal line with a matrix+source pair, which restores affine exactness to
  round-off and removes the ``coeff2 * (t . g)`` term
  (``docs/three-point-wall-projection``).

Measured on 2026-11-08, ``cube_mesh(n, warp=w)`` (see
``docs/native-viscous-three-point-closure/nonorthogonal-audit.json`` and
``docs/three-point-wall-projection/projection-audit.json``).
"""

import numpy as np
import pytest

from tensorfvm.native_viscous3d import NativeViscous3D
from tensorfvm.verification import native_viscous_order3d as nvw

MU = 1.3
GRADIENT = np.array([0.7, -0.3, 0.2])


def affine_field(mesh):
    return (1.2 + mesh.cell_centers @ GRADIENT,
            1.2 + mesh.face_centers @ GRADIENT,
            MU * (mesh.area_vectors @ GRADIENT))


def build(divisions=(4, 4, 4), warp=0.0, closure="three_point", projection=True):
    mesh = nvw.cube_mesh(divisions, warp=warp)
    mask = mesh.neighbor < 0
    return mesh, mask, NativeViscous3D(mesh, mask, MU, wall_closure=closure,
                                       wall_second_projection=projection)


def affine_error(divisions=(4, 4, 4), warp=0.0, closure="three_point",
                 projection=True):
    mesh, mask, op = build(divisions, warp, closure, projection)
    cell, face, exact = affine_field(mesh)
    residual = op.scalar_flux(cell, 0.0, face)[mask] - exact[mask]
    return (float(np.abs(residual).max() / np.linalg.norm(exact[mask])),
            residual, mesh, mask, op, exact)


def test_orthogonal_mesh_stays_affine_exact():
    error, *_ = affine_error(warp=0.0)
    assert error < 1e-12, error


@pytest.mark.parametrize("warp", [0.15, 0.3])
def test_two_point_closure_is_affine_exact_on_warped_meshes(warp):
    """The non-orthogonality itself is handled; two_point stays machine exact."""
    error, *_ = affine_error(warp=warp, closure="two_point")
    assert error < 1e-12, error


@pytest.mark.parametrize("warp, floor", [(0.15, 1e-3), (0.3, 2e-3)])
def test_unprojected_stencil_loses_affine_exactness_on_warped_meshes(warp, floor):
    """Negative-result gate (projection disabled): the raw one-sided stencil is
    not affine exact once the second cell centre leaves the normal line
    (measured 1.25e-3 / 2.48e-3 relative flux error vs 1e-15 orthogonal)."""
    error, *_ = affine_error(warp=warp, projection=False)
    assert error > floor, error
    assert error < 1e-2, error


@pytest.mark.parametrize("warp", [0.15, 0.3])
@pytest.mark.parametrize("divisions", [(4, 4, 4), (6, 6, 6)])
def test_projection_restores_affine_exactness_on_warped_meshes(warp, divisions):
    """The projection (matrix+source pair) makes the closure exact again: the
    second-cell datum is moved onto the normal line, so the stencil samples
    three points of one straight line instead of a kinked one."""
    error, *_ = affine_error(divisions=divisions, warp=warp)
    assert error < 1e-12, error


def test_projection_is_inert_on_an_orthogonal_mesh():
    """No geometric offset -> no correction -> identical flux to round-off."""
    mesh, mask, cell_face = build(warp=0.0, projection=True)
    _, mask_off, op_off = build(warp=0.0, projection=False)
    cell, face, _ = affine_field(mesh)
    a = cell_face.scalar_flux(cell, 0.0, face)
    b = op_off.scalar_flux(cell, 0.0, face)
    assert np.abs(a - b).max() <= 1e-13 * np.abs(a).max()


def test_warped_residual_is_carried_by_the_second_cell_offset():
    """Mechanism gate (projection disabled): the residual equals
    ``coeff2 * (t_cell . grad u)`` exactly, i.e. the defect has one cause."""
    error, residual, mesh, mask, op, exact = affine_error(warp=0.3,
                                                          projection=False)
    offset = op.wall_second_offset[mask]
    x = offset - np.einsum("ij,ij->i", offset, op.normal[mask])[:, None] * op.normal[mask]
    predicted = (op.wall_coeff[mask, 2] * (x @ GRADIENT)) * MU * op.area[mask]
    scaled = residual / (MU * op.area[mask])
    assert np.corrcoef(scaled, predicted / (MU * op.area[mask]))[0, 1] > 0.99
    # the two magnitudes agree to 1e-10 relative: the stencil residual has no
    # other source on this mesh family
    assert abs(np.abs(predicted).max() / np.abs(residual).max() - 1.0) < 1e-10


def test_second_cell_offset_vanishes_on_an_orthogonal_mesh():
    mesh, mask, op = build(warp=0.0)
    assert np.abs(op.wall_second_offset[mask]).max() < 1e-14


@pytest.mark.parametrize("warp", [0.0, 0.3])
def test_three_point_engages_on_every_dirichlet_face(warp):
    mesh, mask, op = build(warp=warp)
    assert op.wall_second_order_faces == int(mask.sum())


def test_matrix_only_gradient_estimate_cannot_supply_the_correction():
    """Negative result kept as a gate: the assembled two-point face-gradient
    rows are not self-contained on a Dirichlet face (the datum enters through
    the flux source), so a matrix-only second-cell correction is off by an
    O(1) factor and makes the residual ~9x worse (report, section 4)."""
    mesh, mask, three = build(warp=0.3)
    plain = NativeViscous3D(mesh, mask, MU, wall_closure="two_point")
    cell, _, _ = affine_field(mesh)
    offset = three.wall_offset
    directional = np.zeros(len(offset))
    for k in range(3):
        directional += offset[:, k] * (plain.face_gradient_matrices[k] @ cell)
    target = offset @ GRADIENT
    defect = np.abs(directional[mask] - target[mask]).max()
    assert defect > 5.0 * np.abs(target[mask]).max()


def test_selection_already_minimises_predicted_truncation():
    """Second falsified fix: picking the second cell by predicted truncation
    instead of face-normal alignment returns the same cell on this family, so
    the defect is a mesh-geometry property, not a selection artefact."""
    mesh = nvw.cube_mesh((4, 4, 4), warp=0.3)
    boundary = np.where(mesh.neighbor < 0)[0]
    internal = mesh.neighbor >= 0
    owner, neighbour = mesh.owner, mesh.neighbor
    fc, cc = mesh.face_centers, mesh.cell_centers
    nrm = mesh.area_vectors / np.linalg.norm(mesh.area_vectors, axis=1)[:, None]
    candidates = {}
    for face in np.where(internal)[0]:
        candidates.setdefault(owner[face], []).append((face, 1))
        candidates.setdefault(neighbour[face], []).append((face, -1))
    chosen, best = [], []
    for face in boundary:
        c0, normal = owner[face], nrm[face]
        delta0 = (fc[face] - cc[c0]) @ normal
        rows = []
        for gf, sigma in candidates.get(c0, ()):
            align = sigma * (nrm[gf] @ -normal)
            second = neighbour[gf] if owner[gf] == c0 else owner[gf]
            if second < 0 or align < 0.5:
                continue
            delta1 = (fc[face] - cc[second]) @ normal
            if delta0 <= 0 or delta1 <= delta0 * (1 + 1e-9):
                continue
            coefficient = delta0 / (delta1 * (delta1 - delta0))
            lateral = (cc[second] - cc[c0]) - ((cc[second] - cc[c0]) @ normal) * normal
            rows.append((abs(coefficient) * np.linalg.norm(lateral), align))
        if rows:
            chosen.append(max(rows, key=lambda r: r[1])[0])
            best.append(min(r[0] for r in rows))
    assert len(chosen) == len(boundary)
    assert max(chosen) <= max(best) * (1 + 1e-12)


def _wall_derivative_error(divisions, warp, projection):
    mesh, mask, op = build(divisions, warp, projection=projection)
    cell, _ = nvw.scalar_field(mesh.cell_centers, "trig3d")
    face, gradient = nvw.scalar_field(mesh.face_centers, "trig3d")
    computed = op.scalar_flux(cell, 0.0, face)[mask] / (MU * op.area[mask])
    exact = np.einsum("ij,ij->i", gradient[mask], op.normal[mask])
    return float(np.linalg.norm(computed - exact) / np.linalg.norm(exact))


def test_projection_improves_the_smooth_field_wall_derivative_and_its_order():
    """Affine exactness is cheap to win; the gate that matters is that the
    smooth-field wall derivative does not pay for it.  On the warped family the
    projection lowers the error at both grids and lifts the observed order
    (1.39 -> 1.55) back to the orthogonal-mesh level."""
    coarse = [4, 4, 4]
    fine = [6, 6, 6]
    raw = [_wall_derivative_error(d, 0.3, False) for d in (coarse, fine)]
    projected = [_wall_derivative_error(d, 0.3, True) for d in (coarse, fine)]
    assert projected[0] < raw[0]
    assert projected[1] < raw[1]
    ratio = np.log(fine[0] / coarse[0])
    assert np.log(raw[0] / raw[1]) / ratio < 1.45          # measured 1.39
    assert np.log(projected[0] / projected[1]) / ratio > 1.5  # measured 1.55
