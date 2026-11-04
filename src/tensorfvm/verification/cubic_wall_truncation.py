"""Does a cubic-exact face gradient repay the O(1) wall-adjacent cell divergence?

Motivation (verification debt closed here).  ``docs/verification/NATIVE_VISCOUS_ORDER3D.md``
records the central accuracy limitation of the native 3-D viscous operator: the
two-point Dirichlet closure makes the wall-face traction first order, so the
wall-adjacent cell divergence is ``O(1)`` (measured orders 0.003 / 0.07 under
mesh doubling) and the Dirichlet diffusion solve does not converge.  That report
also says the repair needs a higher-order reconstruction of the face gradient
"outside the scope of a verification mission", and
``tests/test_native_viscous_order3d.py`` keeps the wish as a strict ``xfail``.

Meanwhile :class:`tensorfvm.native_cubic_gradient3d.NativeCubicFaceGradient3D`
-- a compact cubic-exact, face-area-mean gradient with a separate Dirichlet
source, already merged and reachable from
:mod:`tensorfvm.native_quadratic_flow3d` -- had no test and no measured
benchmark anywhere in the repository (``grep -r NativeCubicFaceGradient3D
tests/`` was empty before this module).  The question this module answers with
numbers instead of a proposal: *is the closure that limits the operator already
in the tree, and is it good enough?*

Protocol: identical to :func:`tensorfvm.verification.native_order3d.truncation_study`
(same mesh family :func:`~tensorfvm.verification.native_order3d.hex_box`, same
manufactured field, same viscosity, same wall-cell mask, same measure
``(1/V) sum_f tau_h . S_f - div tau``), but the face flux is built from
:class:`NativeCubicFaceGradient3D` instead of :class:`NativeViscous3D`.  Nothing
in the solver is modified; this is reconstruction + measurement only.
"""
import numpy as np

from ..native_cubic_gradient3d import NativeCubicFaceGradient3D
from . import native_order3d as _order

MU = 1.3


def wall_mask(mesh, n, layers=1.5):
    """One-layer wall-adjacent cell mask (same definition as the order study)."""
    half = layers/(2.0*n)
    c = mesh.cell_centers
    return np.any((c < half) | (c > 1.0 - half), axis=1)


def face_gradient_tensor(operator, values, boundary_values):
    """Face gradient tensor ``G[f, j, i] = du_i/dx_j`` from a cubic operator.

    The operator is linear per component, so three scalar solves give the nine
    entries; the Dirichlet contribution enters through ``boundary_response``.
    """
    faces = len(operator.mesh.owner)
    tensor = np.empty((faces, 3, 3))
    for component in range(3):
        tensor[:, :, component] = operator.face_gradient(values[:, component],
                                                         boundary_values[:, component])
    return tensor


def face_traction(tensor, viscosity):
    """Symmetric deviatoric traction ``tau = mu (G + G^T - 2/3 tr(G) I)``."""
    trace = np.trace(tensor, axis1=1, axis2=2)
    return viscosity*(tensor + tensor.transpose(0, 2, 1)
                      - (2.0/3.0)*trace[:, None, None]*np.eye(3))


def cell_divergence(mesh, flux):
    """``(1/V_K) sum_f flux_f`` with the owner-outward face sign convention."""
    total = np.zeros((len(mesh.cells), 3))
    np.add.at(total, mesh.owner, flux)
    internal = mesh.neighbor >= 0
    np.add.at(total, mesh.neighbor[internal], -flux[internal])
    return total/mesh.cell_volumes[:, None]


def truncation_study(n, warp=0.0, viscosity=MU, values=None):
    """Cell divergence error with the cubic face gradient, interior vs wall cells.

    ``values`` defaults to the manufactured field of
    :mod:`~tensorfvm.verification.native_order3d`; any smooth field with an
    analytic ``div tau`` (:func:`affine_study`) can be substituted.
    """
    mesh = _order.hex_box(n, warp)
    fixed = mesh.neighbor < 0
    operator = NativeCubicFaceGradient3D(mesh, fixed)
    if values is None:
        cell_values = _order.manufactured(mesh.cell_centers)
        face_values = _order.manufactured(mesh.face_centers)
        exact = _order.divergence_of_stress(mesh.cell_centers, viscosity)
    else:
        cell_values, face_values, exact = values(mesh)
    tensor = face_gradient_tensor(operator, cell_values, face_values)
    flux = np.einsum('fij,fj->fi', face_traction(tensor, viscosity), mesh.area_vectors)
    discrete = cell_divergence(mesh, flux)
    error = np.linalg.norm(discrete - exact, axis=1)
    wall = wall_mask(mesh, n)
    # Decompose the wall-cell error by face ownership exactly as the order study
    # does, so the two studies are comparable term by term.
    deficit = flux - np.einsum('fij,fj->fi', _order.stress(mesh.face_centers, viscosity),
                               mesh.area_vectors)
    wall_only = np.zeros((len(mesh.cells), 3))
    np.add.at(wall_only, mesh.owner[fixed], deficit[fixed])
    wall_flux = np.linalg.norm(wall_only/mesh.cell_volumes[:, None], axis=1)
    scale = max(float(np.linalg.norm(exact, axis=1).max()), 1.0)  # affine control: div tau == 0
    return dict(n=n, h=1.0/n, warp=warp, cells=int(len(mesh.cells)),
                faces=int(len(mesh.owner)), wall_cells=int(wall.sum()),
                field_scale=scale,
                samples_min=int(operator.stencil_diagnostics['samples_min']),
                samples_max=int(operator.stencil_diagnostics['samples_max']),
                face_gradient_nnz=int(operator.stencil_diagnostics['face_gradient_nnz']),
                l2_global=float(np.sqrt(np.mean(error**2))),
                linf_global=float(error.max()),
                l2_interior=float(np.sqrt(np.mean(error[~wall]**2))),
                linf_interior=float(error[~wall].max()),
                l2_wall=float(np.sqrt(np.mean(error[wall]**2))),
                linf_wall=float(error[wall].max()),
                l2_wall_wall_faces_only=float(np.sqrt(np.mean(wall_flux[wall]**2))),
                rel_l2_global=float(np.sqrt(np.mean(error**2))/scale),
                rel_l2_wall=float(np.sqrt(np.mean(error[wall]**2))/scale))


def affine_values(slope, intercept):
    """Linear velocity field and its (zero) stress divergence on any mesh."""
    def cells(points):
        return intercept + points @ np.asarray(slope).T

    def values(mesh):
        return (cells(mesh.cell_centers), cells(mesh.face_centers),
                np.zeros((len(mesh.cells), 3)))
    return values


def orders(rows, key):
    """Observed order of ``key`` between successive rows (``h`` halving)."""
    return _order.orders(rows, key)