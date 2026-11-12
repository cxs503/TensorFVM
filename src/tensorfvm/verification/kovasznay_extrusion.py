"""Extruded-2D Kovasznay (1948) case: spanwise-error attribution helpers.

Why this module exists
----------------------
``CASE-FVM-KOVASZNAY-ORDER`` (feature ledger, open) asks for an observed L2
convergence order >= 1.8 at Re = 40.  The delivered round measured 1.684 and
m16396 showed by norm decomposition that the shortfall is carried *entirely* by
the spanwise component ``w`` (exact value identically zero), which converges at
1.106.  That audit stopped at "attribution by norm" and left the question
"which term drives w?" open.

Answer (m16524): **none of the z-momentum terms** -- the drive comes from the
*symmetry-plane closure*.  The z-planes of the ``n x n x 1`` extrusion are
treated as symmetry planes: the normal component ``w`` is prescribed (w = 0),
but the tangential components are left as **independent face-momentum
unknowns** (``traction_components``).  Those unknowns are only weakly tied to
the adjacent cells, so their O(h) inconsistency feeds the deviatoric
``tau_xz = mu (du/dz + dw/dx)`` and the scheme answers with an O(h) spanwise
velocity.  Evidence:

* at the exact state the *cell* z-momentum residual is exactly ``0.0`` while the
  cell u-residual is O(h^2) -- the z-momentum operator is consistent;
* presribing the same Dirichlet data on the tangential components of the
  symmetry planes (``tangential='dirichlet'``) makes ``w`` vanish **to machine
  precision** (1.9e-17 at n = 8, 3.7e-18 at n = 16) with the in-plane error
  essentially unchanged; the delivered ``tangential='symmetry'`` setup gives
  ``max|w|`` = 1.90e-2 (n=8) / 9.41e-3 (n=16), i.e. first order.

Consequence for verification practice: a velocity L2 norm taken over an
extruded 2D case with free symmetry-plane tangential dofs mixes a genuine O(h^2)
in-plane error with an O(h) closure artefact.  Score the in-plane budget, or
impose the analytic tangential data, and say which was done.

Everything here is a *verification* driver: no solver code is modified.
"""

import numpy as np

from tensorfvm.native_hybrid_ns3d import NativeHybridNavierStokes3D
from tensorfvm.native_manufactured3d import exact_boundary_velocity_means
from tensorfvm.verification.native_hybrid_flow import channel_mesh
from tensorfvm.verification.native_hybrid_ns import kovasznay_fields

COMPONENTS = ('u', 'v', 'w')


def extrusion_solver(n, reynolds=40., *, tangential='symmetry', divisions=None,
                     convection_order=2):
    """Kovasznay solver on an extruded mesh.

    ``tangential='symmetry'``  delivered setup: tangential components of the
        z-planes are free momentum unknowns (symmetry / zero tangential
        traction).
    ``tangential='dirichlet'`` same mesh, tangential z-plane values prescribed
        to the analytic area means -- a strong-BC control that removes those
        unknowns without touching the discretisation.
    ``divisions`` overrides the mesh (e.g. ``(n, n, 4)``) for z-refinement.
    """
    if tangential not in ('symmetry', 'dirichlet'):
        raise ValueError("tangential must be 'symmetry' or 'dirichlet'")
    divisions = (n, n, 1) if divisions is None else tuple(divisions)
    mesh = channel_mesh(divisions, triangulate_faces=True)
    fields = lambda points: kovasznay_fields(points, reynolds)
    exact = fields(mesh.cell_centers)
    boundary = exact_boundary_velocity_means(mesh, fields)
    fixed = np.repeat((mesh.neighbor < 0)[:, None], 3, axis=1)
    if tangential == 'symmetry':
        z = mesh.face_centers[:, 2]
        planes = (mesh.neighbor < 0) & ((z < 1e-10) | (z > 1 - 1e-10))
        fixed[planes, :2] = False
    solver = NativeHybridNavierStokes3D(
        mesh, fixed, boundary, viscosity=1. / reynolds, body_force=0.,
        gauge_value=exact['pressure'][0], convection_order=convection_order,
        coupled_linear_method='reused_lu')
    return mesh, solver, exact


def error_budgets(mesh, solver, exact):
    """In-plane / spanwise / total volume-weighted L2 error of the cell velocity."""
    N, F = solver.N, solver.F
    state = solver.state
    cell = state[:3 * N].reshape(N, 3)
    face = state[3 * N:].reshape(F, 3)
    volumes = mesh.cell_volumes
    eu, ev, ew = (cell - exact['velocity']).T

    def l2(error):
        return float(np.sqrt(np.sum(volumes * error ** 2) / np.sum(volumes)))

    squared = np.sum(volumes * (eu ** 2 + ev ** 2 + ew ** 2))
    order = np.argsort(-np.abs(ew))[:3]
    z = mesh.face_centers[:, 2]
    planes = np.flatnonzero((mesh.neighbor < 0) & ((z < 1e-10) | (z > 1 - 1e-10)))
    internal = np.flatnonzero(mesh.neighbor >= 0)
    return dict(
        inplane_l2=l2(np.hypot(eu, ev)),
        spanwise_l2=l2(ew),
        total_l2=float(np.sqrt(squared / np.sum(volumes))),
        max_abs_w=float(np.abs(ew).max()),
        face_max_abs_w=float(np.abs(face[:, 2]).max()),
        span_share_of_squared_error=float(np.sum(volumes * ew ** 2) / max(squared, 1e-300)),
        worst_cell_centers=mesh.cell_centers[order].tolist(),
        # where the spurious spanwise velocity actually lives
        face_w_abs_max_on_internal=float(np.abs(face[internal, 2]).max(initial=0.)),
        face_w_abs_max_on_symmetry_planes=float(np.abs(face[planes, 2]).max(initial=0.)),
        symmetry_plane_count=int(len(planes)),
    )


def exact_state_residual(mesh, solver, reynolds=40.):
    """Residual of the *exact* state, split by term and by unknown block.

    A discretisation that is correct for the exact field must return an O(h^2)
    residual in every *cell* row of the z-momentum equation (the exact ``w`` is
    identically zero, so no term of that equation is O(h)).  Residuals in
    prescribed rows are reactions, not defects, and are reported separately.
    """
    fields = lambda points: kovasznay_fields(points, reynolds)
    exact_cell = fields(mesh.cell_centers)['velocity']
    exact_face = fields(mesh.face_centers)['velocity']
    state = np.concatenate((exact_cell.ravel(), exact_face.ravel()))
    prescribed = np.ravel(solver.prescribed)
    mask = prescribed != 0
    state[mask] = prescribed[mask]
    p = fields(mesh.cell_centers)['pressure'].copy()
    if solver.gauge:
        p = p - p[0] + solver.gauge_value
    convective, source = solver.convection.hybrid_system(
        solver.volume_flux(state), solver.face_mean.matrix, solver.velocity_gradient(state))
    terms = dict(viscous=solver.viscous.full_matrix @ state,
                 convection=convective @ state + source,
                 pressure=-solver.divergence.T @ p,
                 load=-solver.load)
    terms['total'] = sum(terms.values())
    blocks = {}
    for j, c in enumerate(COMPONENTS):
        blocks[f'cell-{c}'] = np.arange(j, 3 * solver.N, 3)
        blocks[f'face-{c}'] = 3 * solver.N + np.arange(j, 3 * solver.F, 3)
    cell_free = np.ravel(solver.prescribed == 0)[:3 * solver.N]
    out = {}
    for name, residual in terms.items():
        out[name] = {}
        for block, index in blocks.items():
            free = cell_free[index] if block.startswith('cell-') else None
            value = residual[index] if free is None else residual[index][free]
            out[name][block] = float(np.linalg.norm(value))
    return out


def observed_order(coarse_error, fine_error, ratio=2.):
    """Two-level observed order log_r(e_coarse/e_fine); None if not defined."""
    if not (np.isfinite(coarse_error) and np.isfinite(fine_error)):
        return None
    if coarse_error <= 0 or fine_error <= 0:
        return None
    return float(np.log(coarse_error / fine_error) / np.log(ratio))