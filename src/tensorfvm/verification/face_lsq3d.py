"""Face-centred weighted least-squares reconstruction of the velocity gradient.

Motivation (see ``docs/WALL_CELL_GRADIENT_RECONSTRUCTION.md``): the assembled
3-D viscous operator (:mod:`tensorfvm.native_viscous3d`) builds the face
gradient as the interpolated *cell* gradient plus a non-orthogonal correction.
On a Dirichlet boundary face the interpolation weight on the owner cell is one
(there is no second side), so the wall face gradient is the cell-centre
gradient of the wall-adjacent cell, i.e. evaluated half a cell away from the
face.  That offset is ``O(h)``, and the cell-volume normalisation turns it into
an ``O(1)`` truncation error in the wall-adjacent cell divergence.

This module supplies the missing second side as *data*: a Taylor least-squares
fit centred on the face centre whose stencil is the union of the owner and
neighbour cell rings, plus -- on Dirichlet faces -- the face value itself.  A
quadratic fit over an ``O(h)`` stencil returns the face gradient to ``O(h^2)``,
which is what the divergence needs to become second order; a linear fit only
returns ``O(h)`` and is therefore *not* expected to fix anything.

Everything here is reconstruction only: no operator matrix is assembled and no
solver is touched, so the study can be run on its own stencils.
"""
import numpy as np
import scipy.sparse as sp

# Taylor basis columns: (constant, linear 0..2, quadratic 0..5) for degree 2.
_QUADRATIC_PAIRS = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))


def cell_adjacency(mesh):
    """Symmetric cell-cell adjacency of the internal faces, as a sparse matrix."""
    internal = mesh.neighbor >= 0
    owner = mesh.owner[internal]
    neighbour = mesh.neighbor[internal]
    n = len(mesh.cells)
    return sp.coo_matrix((np.ones(2*owner.size), (np.r_[owner, neighbour],
                                                  np.r_[neighbour, owner])),
                         shape=(n, n)).tocsr()


def cell_rings(mesh, rings=2):
    """Cells within ``rings`` graph hops of each cell (self included), as CSR."""
    adjacency = (cell_adjacency(mesh) != 0).astype(np.int8)
    n = adjacency.shape[0]
    total = adjacency.copy()
    frontier = adjacency.copy()
    for _ in range(max(0, rings - 1)):
        frontier = ((frontier @ adjacency) > 0).astype(np.int8)
        total = total + frontier
    total = total + sp.eye(n, format='csr', dtype=np.int8)
    return (total > 0).tocsr()


def taylor_design(xi, degree):
    """Design matrix of the Taylor basis for normalised offsets ``xi`` (m, 3)."""
    if degree not in (1, 2):
        raise ValueError('degree must be 1 or 2')
    columns = [np.ones(len(xi)), xi[:, 0], xi[:, 1], xi[:, 2]]
    if degree == 2:
        columns += [0.5*xi[:, a]*xi[:, b] for a, b in _QUADRATIC_PAIRS]
    return np.stack(columns, axis=1)


def reconstruct(mesh, cell_values, degree=2, rings=2, ridge=0.0,
                dirichlet=None, dirichlet_values=None, weights='inverse-square'):
    """Taylor LSQ face gradient; Dirichlet faces use the prescribed face value.

    ``cell_values`` is (N, 3); ``dirichlet`` a (F,) boolean mask and
    ``dirichlet_values`` the (F, 3) prescribed values on those faces (ignored
    elsewhere).  Faces that are neither internal nor Dirichlet are fitted from
    the owner-cell ring only, i.e. pure extrapolation.
    """
    cell_values = np.asarray(cell_values, dtype=float)
    n_cells = cell_values.shape[0]
    n_faces = len(mesh.face_centers)
    if cell_values.shape != (n_cells, 3):
        raise ValueError('cell_values must be (N, 3)')
    ring = cell_rings(mesh, rings)
    indptr, indices = ring.indptr, ring.indices
    if dirichlet is None:
        dirichlet = np.zeros(n_faces, dtype=bool)
    dirichlet = np.asarray(dirichlet, dtype=bool)
    prescribed = np.zeros((n_faces, 3))
    if dirichlet.any():
        if dirichlet_values is None:
            raise ValueError('dirichlet_values required where dirichlet is set')
        given = np.asarray(dirichlet_values, dtype=float)
        if given.shape == (n_faces, 3):
            prescribed[dirichlet] = given[dirichlet]
        elif given.shape == (int(dirichlet.sum()), 3):
            prescribed[dirichlet] = given
        else:
            raise ValueError('dirichlet_values must be (F, 3) or (sum(dirichlet), 3)')

    k = 4 if degree == 1 else 10
    gradients = np.zeros((n_faces, 3, 3))
    condition = np.zeros(n_faces)
    points = np.zeros(n_faces, dtype=int)
    rcond = 1e-12
    ring_scale = float(np.min(np.linalg.norm(
        mesh.cell_centers[mesh.owner[mesh.neighbor >= 0]]
        - mesh.cell_centers[mesh.neighbor[mesh.neighbor >= 0]], axis=1)))
    for face in range(n_faces):
        owner = mesh.owner[face]
        cells = indices[indptr[owner]:indptr[owner + 1]]
        if mesh.neighbor[face] >= 0:
            other = mesh.neighbor[face]
            cells = np.union1d(cells, indices[indptr[other]:indptr[other + 1]])
        centre = mesh.face_centers[face]
        offsets = mesh.cell_centers[cells] - centre
        sample = cell_values[cells]
        if dirichlet[face]:
            offsets = np.vstack([offsets, np.zeros((1, 3))])
            sample = np.vstack([sample, prescribed[face][None, :]])
        scale = float(np.mean(np.linalg.norm(offsets, axis=1)))
        scale = max(scale, 1e-12*ring_scale)
        xi = offsets/scale
        if weights == 'inverse-square':
            # The Dirichlet row sits at r = 0; regularising the weight by a
            # fraction of the stencil scale keeps it finite and of the same
            # order as the cell rows instead of swamping the fit.
            w = 1.0/(np.einsum('ij,ij->i', xi, xi) + 0.25)
        elif weights == 'uniform':
            w = np.ones(len(xi))
        else:
            raise ValueError('unknown weights')
        design = taylor_design(xi, degree)
        condition[face] = np.linalg.cond(design*np.sqrt(w)[:, None])
        points[face] = len(cells)
        if condition[face] > 1e12:
            raise ValueError(f'face {face} stencil is numerically singular')
        m = design*np.sqrt(w)[:, None]
        rhs = sample*np.sqrt(w)[:, None]
        if ridge:
            normal = m.T @ m + ridge*np.eye(k)
            coeffs = np.linalg.solve(normal, m.T @ rhs)
        else:
            coeffs, *_ = np.linalg.lstsq(m, rhs, rcond=rcond)
        gradients[face] = (coeffs[1:4]/scale).T
    return gradients, {'condition': condition, 'points': points}


def operator_face_gradients(operator, velocity, dirichlet_values=0.):
    """Recover the assembled face-gradient tensor of a ``NativeViscous3D``.

    ``NativeViscous3D.momentum_flux`` is ``matrix @ u + source``; the same flux is
    reproduced here from the operator's own ``face_gradient_matrices`` and
    ``face_gradient_source``, which makes individual components of the face
    gradient replaceable without touching the operator.  Returns
    ``(flux, gradients)`` with ``gradients[f, a, b] = du_a/dx_b``.
    """
    values = np.asarray(velocity, dtype=float)
    n_faces = len(operator.mesh.face_centers)
    data = np.broadcast_to(np.asarray(dirichlet_values, dtype=float),
                           (n_faces, 3)).astype(float, copy=True)
    gradients = np.zeros((n_faces, 3, 3))
    for a in range(3):
        source = operator.face_gradient_source(0., data[:, a])
        for b in range(3):
            gradients[:, a, b] = operator.face_gradient_matrices[b] @ values[:, a] + source[:, b]
    return traction_flux(operator.mesh, gradients, operator.face_viscosity), gradients


def traction_flux(mesh, gradients, viscosity):
    """``mu*(grad u + grad u^T - 2/3 tr(grad u) I) . S`` per face, shape (F, 3)."""
    mu = np.asarray(viscosity, dtype=float)
    mu = mu.reshape(-1, 1) if mu.ndim else mu
    symmetric = (gradients + gradients.transpose(0, 2, 1)
                 - 2/3*np.trace(gradients, axis1=1, axis2=2)[:, None, None]*np.eye(3))
    return mu*np.einsum('fij,fj->fi', symmetric, mesh.area_vectors)


def divergence(mesh, flux):
    """Cell-integrated divergence of a face flux, shape (N, 3) (per unit volume)."""
    divided = np.zeros((len(mesh.cells), 3))
    np.add.at(divided, mesh.owner, flux)
    internal = mesh.neighbor >= 0
    np.add.at(divided, mesh.neighbor[internal], -flux[internal])
    return divided/mesh.cell_volumes[:, None]