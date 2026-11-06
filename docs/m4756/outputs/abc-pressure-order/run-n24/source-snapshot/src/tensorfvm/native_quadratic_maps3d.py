"""Homogeneous sparse shared-face maps for NativeQuadratic3D reconstruction.

Maps act on cell-centroid point values with zero Dirichlet centroid samples.
Nonzero boundary sources belong to the caller. All construction uses sparse
coefficient maps and geometry; no dense basis probing is performed.
"""
import numpy as np
from scipy import sparse


def _selectors(op):
    rows = np.arange(op.F)
    owner = sparse.coo_matrix((np.ones(op.F), (rows, op.mesh.owner)),
                              shape=(op.F, op.N)).tocsr()
    neighbor = sparse.coo_matrix((np.ones(op.F), (rows, op.neighbor_safe)),
                                 shape=(op.F, op.N)).tocsr()
    interpolation = (sparse.diags(1-op.weight)@owner
                     + sparse.diags(op.weight)@neighbor).tocsr()
    return owner, neighbor, interpolation


def face_mean_matrix(op, prescribed_boundary_means=False):
    """CSR area-mean map, including the original Dirichlet semantics.

    False: zero centroid Dirichlet samples retain Hessian face covariance.
    True: Dirichlet face area means are prescribed zero, so their rows vanish.
    Free boundary faces retain extrapolated quadratic values in both cases.
    """
    owner, neighbor, interpolation = _selectors(op)
    result = sparse.csr_matrix((op.F, op.N))
    for selector, cells, weight in ((owner, op.mesh.owner, 1-op.weight),
                                     (neighbor, op.neighbor_safe, op.weight)):
        d = op.mesh.face_centers-op.mesh.cell_centers[cells]
        prediction = selector.copy()
        for a in range(3):
            prediction += sparse.diags(d[:, a])@selector@op.gradient_matrices[a]
            for b in range(3):
                moment = d[:, a]*d[:, b]+op.face_covariance[:, a, b]
                prediction += .5*sparse.diags(moment)@selector@op.hessian_matrices[a][b]
        result += sparse.diags(weight)@prediction
    result = sparse.diags((~op.dirichlet).astype(float))@result
    if not prescribed_boundary_means:
        for a in range(3):
            for b in range(3):
                result += .5*sparse.diags(op.dirichlet*op.face_covariance[:, a, b])@interpolation@op.hessian_matrices[a][b]
    return result.tocsr()


def face_gradient_matrices(op, compact=True, cell_gradient_matrices=None):
    """Three CSR maps matching homogeneous op.face_gradient exactly.

    An optional tuple of three (N,N) sparse maps supplies current cell gradients,
    e.g. conservative Gauss maps. Hessians remain from op's quadratic fit.
    The compact normal correction is applied on internal and Dirichlet faces;
    free pressure boundaries receive no fictitious normal derivative condition.
    """
    gradient = op.gradient_matrices if cell_gradient_matrices is None else cell_gradient_matrices
    if len(gradient) != 3 or any(not sparse.issparse(g) or g.shape != (op.N, op.N)
                                 for g in gradient):
        raise ValueError('Three finite sparse (N,N) cell gradient maps required')
    gradient = tuple(sparse.csr_matrix(g) for g in gradient)
    if any(not np.isfinite(g.data).all() for g in gradient):
        raise ValueError('Three finite sparse (N,N) cell gradient maps required')
    owner, neighbor, interpolation = _selectors(op)
    result = [sparse.csr_matrix((op.F, op.N)) for _ in range(3)]
    for selector, cells, weight in ((owner, op.mesh.owner, 1-op.weight),
                                     (neighbor, op.neighbor_safe, op.weight)):
        d = op.mesh.face_centers-op.mesh.cell_centers[cells]
        for a in range(3):
            prediction = selector@gradient[a]
            for b in range(3):
                prediction += sparse.diags(d[:, b])@selector@op.hessian_matrices[a][b]
            result[a] += sparse.diags(weight)@prediction
    if compact:
        delta = (sparse.diags(op.internal.astype(float))@(neighbor-owner)
                 - sparse.diags(op.dirichlet.astype(float))@owner)
        defect = delta.tocsr()
        for a in range(3):
            defect -= sparse.diags(op.d[:, a])@result[a]
        offset = op.mesh.face_centers-op.mesh.cell_centers[op.mesh.owner]-op.d/2
        for a in range(3):
            for b in range(3):
                defect += sparse.diags(op.d[:, a]*offset[:, b])@interpolation@op.hessian_matrices[a][b]
        active = op.internal|op.dirichlet
        for a in range(3):
            result[a] += sparse.diags(active*op.mesh.area_vectors[:, a]/op.projection)@defect
    return tuple(g.tocsr() for g in result)
