"""Cubic, compact, shared-face area-mean gradients on native 3D meshes.

Cell unknowns and prescribed boundary samples are centroid point values.
Only real adjacency selects samples; no analytic interior derivatives enter.
The line-difference correction and face second moments reproduce every cubic
polynomial, including on skew faces. This is an optional viscous operator;
it does not change pressure stabilization or advective reconstruction.
"""
import numpy as np
from scipy import sparse
from .native_face_geometry3d import polygon_face_covariance


_EXPONENTS = np.array([(a, b, c) for degree in (1, 2, 3)
                       for a in range(degree+1) for b in range(degree-a+1)
                       for c in [degree-a-b]], dtype=int)


def _basis(q):
    return np.prod(q[:, None, :] ** _EXPONENTS[None, :, :], axis=-1)


def _mean_gradient(q, covariance, whitening):
    """Exact area mean of the derivative of each cubic basis monomial."""
    result = np.zeros((len(q), 3, 19))
    for column, powers in enumerate(_EXPONENTS):
        for axis in range(3):
            if not powers[axis]:
                continue
            exponent = powers.copy(); exponent[axis] -= 1
            moment = np.prod(q ** exponent, axis=1)
            indices = np.repeat(np.arange(3), exponent)
            if len(indices) == 2:
                moment += covariance[:, indices[0], indices[1]]
            result[:, :, column] += powers[axis]*moment[:, None]*whitening[:, axis]
    return result


class NativeCubicFaceGradient3D:
    """Sparse face gradient maps, including separate Dirichlet sources."""
    def __init__(self, mesh, dirichlet_mask=None, *, min_samples=38,
                 max_depth=6, condition_limit=1e8):
        self.mesh = mesh; self.N = len(mesh.cells); self.F = len(mesh.owner)
        internal = mesh.neighbor >= 0
        fixed = np.zeros(self.F, bool) if dirichlet_mask is None else np.asarray(dirichlet_mask, bool)
        if fixed.shape != (self.F,) or np.any(fixed & internal):
            raise ValueError('Dirichlet mask must select boundary faces')
        if min_samples < 19 or max_depth < 1 or not np.isfinite(condition_limit) or condition_limit <= 1:
            raise ValueError('Cubic fit needs >=19 samples, positive depth and condition limit >1')
        adjacent = [set() for _ in range(self.N)]
        boundary_faces = [[] for _ in range(self.N)]
        incident = [[] for _ in range(self.N)]
        safe = np.where(internal, mesh.neighbor, mesh.owner)
        d = mesh.cell_centers[safe]-mesh.cell_centers[mesh.owner]
        d[~internal] = mesh.face_centers[~internal]-mesh.cell_centers[mesh.owner[~internal]]
        projection = np.einsum('fi,fi->f', mesh.area_vectors, d)
        if np.any(projection <= 0):
            raise ValueError('Positive native face projection required')
        weight = np.zeros(self.F)
        weight[internal] = np.einsum('fi,fi->f',
            (mesh.face_centers-mesh.cell_centers[mesh.owner])[internal],
            mesh.area_vectors[internal])/projection[internal]
        for face, (owner, neighbor) in enumerate(zip(mesh.owner, mesh.neighbor)):
            incident[owner].append(face)
            if neighbor >= 0:
                incident[neighbor].append(face)
                adjacent[owner].add(int(neighbor)); adjacent[neighbor].add(int(owner))
            elif fixed[face]:
                boundary_faces[owner].append(face)
        covariance = np.array([polygon_face_covariance(mesh.vertices[list(face)], center)
                               for face, center in zip(mesh.face_vertices, mesh.face_centers)])
        normal_correction = mesh.area_vectors/projection[:, None]*(internal|fixed)[:, None]
        rows, columns, values = [], [], []
        brows, bcolumns, bvalues = [], [], []
        depths, counts, conditions = [], [], []
        for cell in range(self.N):
            visited, frontier = {cell}, {cell}
            for depth in range(max_depth+1):
                if depth:
                    frontier = set().union(*(adjacent[k] for k in frontier))-visited if frontier else set()
                    visited |= frontier
                neighbors = np.array(sorted(visited-{cell}), dtype=int)
                faces = np.array(sorted({f for c in visited for f in boundary_faces[c]}), dtype=int)
                count = len(neighbors)+len(faces)
                if count < 19 or (count < min_samples and frontier and depth < max_depth):
                    continue
                offset = np.concatenate((mesh.cell_centers[neighbors], mesh.face_centers[faces]))-mesh.cell_centers[cell]
                distance = np.linalg.norm(offset, axis=1)
                if np.any(distance <= 0):
                    raise ValueError('Coincident point in cubic stencil')
                _, singular_geometry, vt = np.linalg.svd(offset, full_matrices=False)
                if singular_geometry[-1] <= 1e-13*singular_geometry[0]:
                    continue
                whitening = vt.T/(singular_geometry/np.sqrt(count))[None, :]
                design = _basis(offset@whitening)/distance[:, None]
                scale = np.linalg.norm(design, axis=0)
                inverse, _, rank, singular = np.linalg.lstsq(design/scale, np.eye(count), rcond=1e-12)
                if rank != 19 or singular[0]/singular[-1] > condition_limit:
                    continue
                response = inverse/scale[:, None]/distance[None, :]
                break
            else:
                raise ValueError(f'Cubic stencil needs full-rank well-conditioned real adjacency at cell {cell}')
            ids = np.asarray(incident[cell]); origin = mesh.cell_centers[cell]
            q = (mesh.face_centers[ids]-origin)@whitening
            cov = np.einsum('ia,fij,jb->fab', whitening, covariance[ids], whitening)
            mean = _mean_gradient(q, cov, whitening)
            start = mesh.cell_centers[mesh.owner[ids]]
            end = start+d[ids]
            line = _basis((end-origin)@whitening)-_basis((start-origin)@whitening)
            corrected = mean-normal_correction[ids, :, None]*line[:, None, :]
            blend = np.where(mesh.owner[ids] == cell, 1-weight[ids], weight[ids])
            action = np.einsum('fak,kn->fan', corrected, response)*blend[:, None, None]
            cell_action = np.concatenate((action[:, :, :len(neighbors)], -action.sum(axis=2, keepdims=True)), axis=2)
            rows.append(np.repeat(ids, len(neighbors)+1))
            columns.append(np.tile(np.r_[neighbors, cell], len(ids)))
            values.append(cell_action.transpose(1, 0, 2).reshape(3, -1))
            brows.append(np.repeat(ids, len(faces)))
            bcolumns.append(np.tile(faces, len(ids)))
            bvalues.append(action[:, :, len(neighbors):].transpose(1, 0, 2).reshape(3, -1))
            depths.append(depth); counts.append(count); conditions.append(float(singular[0]/singular[-1]))
        r, c, v = np.concatenate(rows), np.concatenate(columns), np.concatenate(values, axis=1)
        br, bc, bv = np.concatenate(brows), np.concatenate(bcolumns), np.concatenate(bvalues, axis=1)
        self.face_gradient_matrices = []
        self.boundary_response = []
        ids = np.arange(self.F)
        for axis in range(3):
            direct = sparse.coo_matrix((np.r_[-normal_correction[:, axis], normal_correction[internal, axis]],
                (np.r_[ids, ids[internal]], np.r_[mesh.owner, mesh.neighbor[internal]])), shape=(self.F, self.N)).tocsr()
            matrix = sparse.coo_matrix((v[axis], (r, c)), shape=(self.F, self.N)).tocsr()+direct
            source = sparse.coo_matrix((bv[axis], (br, bc)), shape=(self.F, self.F)).tocsr()
            source += sparse.diags(normal_correction[:, axis]*fixed)
            matrix.eliminate_zeros(); source.eliminate_zeros()
            self.face_gradient_matrices.append(matrix)
            self.boundary_response.append(source)
        self.stencil_diagnostics = dict(degree=3, min_samples=min_samples, max_depth=max_depth,
            condition_limit=condition_limit, samples_min=min(counts), samples_max=max(counts),
            normalized_condition_max=max(conditions), depth_counts={str(d): depths.count(d) for d in sorted(set(depths))},
            face_gradient_nnz=sum(m.nnz for m in self.face_gradient_matrices),
            boundary_response_nnz=sum(m.nnz for m in self.boundary_response))

    def face_gradient(self, values, boundary_values=0.):
        a = np.asarray(values, float)
        if a.ndim not in (1, 2) or len(a) != self.N or not np.isfinite(a).all():
            raise ValueError('Finite scalar/vector cell point values required')
        shape = (self.F,)+a.shape[1:]
        b = np.broadcast_to(np.asarray(boundary_values, float), shape)
        if not np.isfinite(b).all():
            raise ValueError('Finite Dirichlet centroid point data required')
        return np.stack([m@a+r@b for m, r in zip(self.face_gradient_matrices, self.boundary_response)], axis=-1)
