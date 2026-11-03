"""Energy-compatible (Galerkin) viscous momentum operator: dissipative by construction.

The legacy fused assembly in :class:`tensorfvm.native_viscous3d.NativeViscous3D`
builds the momentum diffusion matrix as ``D A G``: a face-gradient reconstruction
``G`` combined with a face-integration ``D A`` that is *not* its adjoint.  For
reconstructed (higher than two-point) gradients the discrete sum-by-parts
identity therefore fails, ``u^T (D A G) u`` is not a norm, and on strongly
non-orthogonal cells the operator can *create* kinetic energy
(``u^T (D A G) u < 0``).

This module assembles the same physics in energy (Galerkin) form

    K = 2 sum_c mu_c V_c sum_{i<=j} w_ij S_ij(c)^T S_ij(c),
    S = sym(grad u) (optionally traceless), w_ii = 1, w_ij = 2,

i.e. a weighted sum of squares of the cell gradients already produced by the
native gradient stencils.  ``K`` is symmetric positive semidefinite for *any*
mesh, so the associated semi-discrete viscous term is dissipative by
construction:

    -u^T K u = -2 sum_c mu_c V_c (S:S) <= 0      for every u.

For affine velocity fields the cell gradients are exact, so the discrete energy
identity ``-u^T K u = 2 sum_c mu_c V_c S:S`` holds to round-off and the
coefficient of the operator is certified (no artificial viscosity is added).
Boundary data enter through the linear term returned by
:func:`compatible_boundary_source`, which is the exact partner of ``K``:
``K u + source`` is the residual for prescribed Dirichlet/Neumann data.

Scope of this module: it exposes an alternative, provably dissipative
assembly.  It does not change any existing default path.
"""
import numpy as np
from scipy.sparse import diags, kron

from .native_viscous3d import NativeViscous3D

__all__ = [
    "compatible_momentum_operators",
    "compatible_boundary_source",
    "compatible_energy_rate",
    "compatible_momentum_matrix_in_legacy_sign",
    "reconstruction_weight",
]


def _component(matrix, component):
    """Embed an ``(n x N)`` component operator into the interleaved ``3N`` layout."""
    unit = np.zeros(3)
    unit[component] = 1.
    return kron(matrix, unit[None, :], format="csr")


def _cell_viscosity(base, dynamic_viscosity):
    mu = np.asarray(dynamic_viscosity, dtype=float)
    if mu.ndim == 0:
        return np.full(base.N, float(mu))
    if mu.shape == (base.N,):
        return mu.astype(float, copy=True)
    if mu.shape == (base.F,):
        counts = np.asarray(base.interpolation.T @ np.ones(base.F)).ravel()
        return np.asarray(base.interpolation.T @ mu).ravel() / counts
    raise ValueError("viscosity must be scalar, cell or face array")


def reconstruction_weight(mesh, dirichlet_mask):
    """Per-cell weight reproduction check helper: cell volume times multiplicity."""
    return np.asarray(mesh.cell_volumes, dtype=float)


def compatible_momentum_operators(mesh, dirichlet_mask, dynamic_viscosity=1., *,
                                  deviatoric=True):
    """Assemble the energy (Galerkin) viscous momentum operator.

    Returns a dict with ``matrix`` (``3N x 3N``, symmetric PSD).  Note the sign
    convention: ``u^T K u >= 0`` here, whereas the production slot
    ``NativeViscous3D.momentum_diffusion_matrix`` satisfies ``u^T M u <= 0``, so
    ``K`` is *not* a drop-in replacement; use
    :func:`compatible_momentum_matrix_in_legacy_sign` for that slot (verified in
    ``tests/test_native_compatible_sign_convention.py``).  Also returned are the
    per-(i,j) gradient blocks, cell viscosity, and the legacy operator used for
    its gradient stencils.
    """
    base = NativeViscous3D(mesh, dirichlet_mask, dynamic_viscosity, deviatoric=deviatoric)
    mu = _cell_viscosity(base, dynamic_viscosity)
    if not np.isfinite(mu).all() or np.any(mu <= 0):
        raise ValueError("finite positive dynamic viscosity required")
    volume = np.asarray(mesh.cell_volumes, dtype=float)
    cell_gradients = base.gradient_matrices  # [d/dx_j] as (N x N) operators
    component = [[_component(cell_gradients[j], i) for j in range(3)] for i in range(3)]
    trace = component[0][0] + component[1][1] + component[2][2]
    blocks, multiplicities = {}, {}
    for i in range(3):
        for j in range(i, 3):
            if i == j:
                block = component[i][i]
                if deviatoric:
                    block = block - trace / 3
                multiplicity = 1
            else:
                block = 0.5 * (component[i][j] + component[j][i])
                multiplicity = 2
            blocks[(i, j)] = block.tocsr()
            multiplicities[(i, j)] = multiplicity
    weight = diags(2 * mu * volume)
    matrix = sum(multiplicities[key] * (blocks[key].T @ weight @ blocks[key])
                 for key in blocks).tocsr()
    return dict(matrix=matrix, blocks=blocks, multiplicities=multiplicities,
                cell_viscosity=mu, cell_volumes=volume, deviatoric=bool(deviatoric),
                base=base, shape=(3 * base.N, 3 * base.N))


def compatible_boundary_source(operators, neumann_gradients=0., dirichlet_values=0.):
    """Exact linear partner of :func:`compatible_momentum_operators`.

    ``matrix @ u.ravel() + source`` is the energy-form residual for cell values
    ``u`` together with the supplied boundary data (same argument convention as
    ``NativeViscous3D.gradient_source``: Neumann gradients per face, Dirichlet
    values per face and velocity component).
    """
    base = operators["base"]
    values = np.broadcast_to(np.asarray(dirichlet_values, dtype=float), (base.F, 3))
    gradients = np.broadcast_to(np.asarray(neumann_gradients, dtype=float), (base.F, 3))
    if not np.isfinite(values).all() or not np.isfinite(gradients).all():
        raise ValueError("finite boundary data required")
    data = np.stack([base.gradient_source(gradients[:, i], values[:, i])
                     for i in range(3)], axis=1)  # [cell, component i, direction j]
    if not operators["deviatoric"]:
        trace = data[:, 0, 0] + data[:, 1, 1] + data[:, 2, 2]
    else:
        trace = 0.
    weight = diags(2 * operators["cell_viscosity"] * operators["cell_volumes"])
    source = np.zeros(3 * base.N)
    for (i, j), block in operators["blocks"].items():
        if i == j:
            offset = data[:, i, i]
            if operators["deviatoric"]:
                offset = offset - trace / 3
        else:
            offset = 0.5 * (data[:, i, j] + data[:, j, i])
        source += operators["multiplicities"][(i, j)] * (block.T @ (weight @ offset))
    return source


def compatible_momentum_matrix_in_legacy_sign(operators):
    """Negated compatible operator, i.e. the drop-in for a ``momentum_diffusion_matrix`` slot.

    ``NativeViscous3D.momentum_diffusion_matrix`` approximates ``+div(tau)`` and is
    negative semidefinite on well-shaped meshes (``u^T M u <= 0`` is the dissipative
    sign).  ``compatible_momentum_operators['matrix']`` is positive semidefinite
    (``-u^T K u <= 0``).  Returning ``-K`` keeps a solver's viscous update dissipative
    when the assembly is swapped, since ``u^T (-K) u <= 0`` for every ``u``.
    """
    return (-operators["matrix"]).tocsr()


def compatible_energy_rate(operators, velocity):
    """Kinetic-energy change rate ``-u^T K u`` (negative means dissipative)."""
    u = np.asarray(velocity, dtype=float)
    if u.shape != (operators["base"].N, 3) or not np.isfinite(u).all():
        raise ValueError("finite (N,3) cell velocity required")
    flat = u.ravel()
    return -float(flat @ (operators["matrix"] @ flat))