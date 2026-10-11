"""Polynomial-exact quadratic reconstruction on real native 3D adjacency.

Unknowns are point values at cell volume centroids, not volume averages.
Dirichlet samples are point values at face centroids. A face area mean is a
separate quantity; optional boundary_face_means explicitly prescribes it.
Free boundaries contribute no fictitious pressure/normal derivative data.
This module assembles no flow system and does not replace first-order LS.
"""
import numpy as np
from scipy import sparse
from .native_face_geometry3d import polygon_face_covariance


_PAIRS = ((0, 1), (0, 2), (1, 2))


def _basis(d):
    x, y, z = d.T
    return np.column_stack((x, y, z, x*x, y*y, z*z, x*y, x*z, y*z))


def _physical_transform(whitening):
    """Map coefficients of q=d W to coefficients of physical d."""
    transform = np.zeros((9, 9)); transform[:3, :3] = whitening
    for column in range(3, 9):
        hq = np.zeros((3, 3))
        if column < 6:
            hq[column-3, column-3] = 2
        else:
            a, b = _PAIRS[column-6]; hq[a, b] = hq[b, a] = 1
        h = whitening @ hq @ whitening.T
        transform[3:6, column] = np.diag(h)/2
        for row, (a, b) in enumerate(_PAIRS, 6):
            transform[row, column] = h[a, b]
    return transform


class NativeQuadratic3D:
    """Cell gradient/Hessian CSR maps plus boundary sources and shared faces.

    Geometry alone selects stencils. Each stencil uses SVD rotation and
    directional scaling followed by weighted, column-scaled least squares.
    Rank deficiency or excessive normalized condition raises ValueError.
    Boundary samples may extend through the visited real-neighbor cells.
    """
    def __init__(self, mesh, dirichlet_mask=None, *, min_samples=18,
                 max_depth=5, condition_limit=1e8):
        self.mesh = mesh; self.N = len(mesh.cells); self.F = len(mesh.owner)
        self.internal = mesh.neighbor >= 0; self.boundary = ~self.internal
        self.dirichlet = np.zeros(self.F, bool) if dirichlet_mask is None else np.asarray(dirichlet_mask, bool).copy()
        if self.dirichlet.shape != (self.F,) or np.any(self.dirichlet & self.internal):
            raise ValueError('Dirichlet mask must select boundary faces')
        if min_samples < 9 or max_depth < 1 or not np.isfinite(condition_limit) or condition_limit <= 1:
            raise ValueError('At least nine samples, positive depth and condition limit >1 required')
        adjacency = [set() for _ in range(self.N)]
        boundary_faces = [[] for _ in range(self.N)]
        for owner, neighbor in zip(mesh.owner[self.internal], mesh.neighbor[self.internal]):
            adjacency[owner].add(int(neighbor)); adjacency[neighbor].add(int(owner))
        for face in np.flatnonzero(self.dirichlet):
            boundary_faces[mesh.owner[face]].append(int(face))
        rows, columns, values = [], [], [[] for _ in range(9)]
        b_rows, b_columns, b_values = [], [], [[] for _ in range(9)]
        self.whitening = np.zeros((self.N, 3, 3))
        depths, sample_counts, conditions, scale_ratios = [], [], [], []
        for cell in range(self.N):
            visited, frontier = {cell}, {cell}; fitted = False
            for depth in range(max_depth+1):
                if depth:
                    frontier = set().union(*(adjacency[k] for k in frontier))-visited if frontier else set()
                    visited |= frontier
                neighbors = np.array(sorted(visited-{cell}), dtype=int)
                faces = np.array(sorted({f for c in visited for f in boundary_faces[c]}), dtype=int)
                exhausted = not frontier or depth == max_depth
                count = len(neighbors)+len(faces)
                if count < 9 or (count < min_samples and not exhausted):
                    continue
                points = np.concatenate((mesh.cell_centers[neighbors], mesh.face_centers[faces]))
                d = points-mesh.cell_centers[cell]
                distance = np.linalg.norm(d, axis=1)
                if np.any(distance <= 0):
                    raise ValueError('Coincident point in quadratic stencil')
                _, singular_geometry, vt = np.linalg.svd(d, full_matrices=False)
                if len(singular_geometry) != 3 or singular_geometry[-1] <= 1e-13*singular_geometry[0]:
                    continue
                W = vt.T/(singular_geometry/np.sqrt(count))[None, :]
                basis = _basis(d @ W); weight = 1/distance
                design = basis*weight[:, None]; column_scale = np.linalg.norm(design, axis=0)
                normalized = design/column_scale
                inverse, _, rank, singular = np.linalg.lstsq(normalized, np.eye(count), rcond=1e-12)
                if rank != 9 or singular[0]/singular[-1] > condition_limit:
                    continue
                response = _physical_transform(W) @ (inverse*weight[None, :]/column_scale[:, None])
                rows.extend([cell]*(len(neighbors)+1)); columns.extend(neighbors.tolist()+[cell])
                b_rows.extend([cell]*len(faces)); b_columns.extend(faces.tolist())
                for k in range(9):
                    values[k].extend(response[k, :len(neighbors)].tolist()+[-float(response[k].sum())])
                    b_values[k].extend(response[k, len(neighbors):].tolist())
                self.whitening[cell] = W; depths.append(depth); sample_counts.append(count)
                conditions.append(float(singular[0]/singular[-1]))
                scale_ratios.append(float(singular_geometry[0]/singular_geometry[-1]))
                fitted = True; break
            if not fitted:
                raise ValueError(f'Quadratic stencil needs full-rank well-conditioned real adjacency at cell {cell}')
        self.coefficient_matrices = [sparse.coo_matrix((data, (rows, columns)), shape=(self.N, self.N)).tocsr() for data in values]
        self.boundary_response = [sparse.coo_matrix((data, (b_rows, b_columns)), shape=(self.N, self.F)).tocsr() for data in b_values]
        self.gradient_matrices = self.coefficient_matrices[:3]
        self.gradient_boundary_response = self.boundary_response[:3]
        self.hessian_matrices = [[None]*3 for _ in range(3)]
        self.hessian_boundary_response = [[None]*3 for _ in range(3)]
        for a in range(3):
            self.hessian_matrices[a][a] = 2*self.coefficient_matrices[3+a]
            self.hessian_boundary_response[a][a] = 2*self.boundary_response[3+a]
        for k, (a, b) in enumerate(_PAIRS, 6):
            self.hessian_matrices[a][b] = self.hessian_matrices[b][a] = self.coefficient_matrices[k]
            self.hessian_boundary_response[a][b] = self.hessian_boundary_response[b][a] = self.boundary_response[k]
        self.stencil_depths = np.asarray(depths); self.stencil_sample_counts = np.asarray(sample_counts)
        self.stencil_conditions = np.asarray(conditions)
        self.stencil_diagnostics = dict(depth_counts={str(d): depths.count(d) for d in sorted(set(depths))},
            samples_min=min(sample_counts), samples_max=max(sample_counts), normalized_condition_max=max(conditions),
            directional_scale_ratio_max=max(scale_ratios), min_samples=int(min_samples), max_depth=int(max_depth),
            condition_limit=float(condition_limit), coordinate_scaling='SVD rotation and directional whitening',
            cell_coefficient_nnz=sum(a.nnz for a in self.coefficient_matrices),
            boundary_response_nnz=sum(a.nnz for a in self.boundary_response))
        self.neighbor_safe = np.where(self.internal, mesh.neighbor, mesh.owner)
        self.d = mesh.cell_centers[self.neighbor_safe]-mesh.cell_centers[mesh.owner]
        self.d[self.boundary] = mesh.face_centers[self.boundary]-mesh.cell_centers[mesh.owner[self.boundary]]
        self.projection = np.einsum('fi,fi->f', mesh.area_vectors, self.d)
        if np.any(self.projection <= 0):
            raise ValueError('Positive native face projection required')
        self.weight = np.zeros(self.F)
        self.weight[self.internal] = np.einsum('fi,fi->f',
            (mesh.face_centers-mesh.cell_centers[mesh.owner])[self.internal], mesh.area_vectors[self.internal])/self.projection[self.internal]
        if all(len(face) == 3 for face in mesh.face_vertices):
            # Preserve the previous triangle arithmetic exactly.
            vertices = mesh.vertices[np.array(mesh.face_vertices)]
            offset = vertices-mesh.face_centers[:, None]
            self.face_covariance = np.einsum('fai,faj->fij', offset, offset)/12
        else:
            self.face_covariance = np.array([polygon_face_covariance(
                mesh.vertices[list(face)], center)
                for face, center in zip(mesh.face_vertices, mesh.face_centers)])

    def _values(self, values):
        a = np.asarray(values, float)
        if a.ndim not in (1, 2) or a.shape[0] != self.N or not np.isfinite(a).all():
            raise ValueError('Finite scalar/vector cell point values required')
        return a

    def _boundary_values(self, boundary_values, components=None):
        shape = (self.F,) if components is None else (self.F, components)
        a = np.asarray(boundary_values, float)
        if components is not None and a.shape == (self.F,):
            a = a[:, None]
        a = np.broadcast_to(a, shape)
        if not np.isfinite(a).all():
            raise ValueError('Finite Dirichlet centroid point data required')
        return a

    def gradient_source(self, boundary_values=0.):
        a = np.asarray(boundary_values, float)
        a = self._boundary_values(a, None if a.ndim < 2 else a.shape[1])
        return np.stack([r@a for r in self.gradient_boundary_response], axis=-1)

    def hessian_source(self, boundary_values=0.):
        a = np.asarray(boundary_values, float)
        a = self._boundary_values(a, None if a.ndim < 2 else a.shape[1])
        return np.stack([np.stack([r@a for r in row], axis=-1) for row in self.hessian_boundary_response], axis=-2)

    def gradient(self, values, boundary_values=0.):
        a = self._values(values); b = self._boundary_values(boundary_values, None if a.ndim == 1 else a.shape[1])
        return np.stack([r@a for r in self.gradient_matrices], axis=-1)+self.gradient_source(b)

    def hessian(self, values, boundary_values=0.):
        a = self._values(values); b = self._boundary_values(boundary_values, None if a.ndim == 1 else a.shape[1])
        return np.stack([np.stack([r@a for r in row], axis=-1) for row in self.hessian_matrices], axis=-2)+self.hessian_source(b)

    def _face_geometry(self, values, boundary_values):
        scalar = np.asarray(values).ndim == 1
        a = self._values(values); a = a[:, None] if scalar else a
        b = self._boundary_values(boundary_values, a.shape[1])
        g, h = self.gradient(a, b), self.hessian(a, b)
        return scalar, a, b, g, h

    def _interpolate(self, value):
        w = self.weight.reshape((self.F,)+(1,)*(value.ndim-1))
        return (1-w)*value[self.mesh.owner]+w*value[self.neighbor_safe]

    def face_mean(self, values, boundary_values=0., *, boundary_face_means=None):
        """Polygon area means; Dir centroid point data gains Hessian covariance.

        Explicit boundary_face_means overrides means on Dirichlet faces only,
        e.g. physical constant no-slip velocity. It does not become LS samples.
        """
        scalar, a, b, g, h = self._face_geometry(values, boundary_values)
        predictions = []
        for cells in (self.mesh.owner, self.neighbor_safe):
            d = self.mesh.face_centers-self.mesh.cell_centers[cells]
            moment = d[:, :, None]*d[:, None, :]+self.face_covariance
            predictions.append(a[cells]+np.einsum('fki,fi->fk',g[cells],d)
                               +.5*np.einsum('fkij,fij->fk',h[cells],moment))
        w = self.weight[:, None]; mean = (1-w)*predictions[0]+w*predictions[1]
        hf = self._interpolate(h)
        mean[self.dirichlet] = b[self.dirichlet]+.5*np.einsum('fkij,fij->fk',hf[self.dirichlet],self.face_covariance[self.dirichlet])
        if boundary_face_means is not None:
            supplied = self._boundary_values(boundary_face_means, a.shape[1])
            mean[self.dirichlet] = supplied[self.dirichlet]
        return mean[:, 0] if scalar else mean

    def face_gradient(self, values, boundary_values=0., *, compact=True, cell_gradients=None):
        """Gradient at polygon centroid (=area mean for quadratic fields).

        cell_gradients may supply the current conservative Gauss gradient.
        The Hessian geometry correction remains from the real-cell fit.
        Compact correction uses the centroid Dir point value, never face mean.
        """
        scalar, a, b, g, h = self._face_geometry(values, boundary_values)
        if cell_gradients is not None:
            supplied = np.asarray(cell_gradients,float)
            supplied = supplied[:,None,:] if scalar else supplied
            if supplied.shape != g.shape or not np.isfinite(supplied).all():
                raise ValueError('Matching finite current cell gradients required')
            g = supplied
        predictions=[]
        for cells in (self.mesh.owner,self.neighbor_safe):
            d=self.mesh.face_centers-self.mesh.cell_centers[cells]
            predictions.append(g[cells]+np.einsum('fkij,fj->fki',h[cells],d))
        w=self.weight[:,None,None]; result=(1-w)*predictions[0]+w*predictions[1]
        if compact:
            active=self.internal|self.dirichlet; delta=a[self.neighbor_safe]-a[self.mesh.owner]
            delta[self.dirichlet]=b[self.dirichlet]-a[self.mesh.owner[self.dirichlet]]
            midpoint=self.mesh.cell_centers[self.mesh.owner]+self.d/2
            offset=self.mesh.face_centers-midpoint; hf=self._interpolate(h)
            # grad(face) dot d = delta + d^T H(face-midpoint), quadratic exact.
            defect=delta-np.einsum('fki,fi->fk',result,self.d)+np.einsum('fi,fkij,fj->fk',self.d,hf,offset)
            result+=active[:,None,None]*defect[:,:,None]*self.mesh.area_vectors[:,None,:]/self.projection[:,None,None]
        return result[:,0] if scalar else result

    def compact_normal_flux(self, values, boundary_values=0., *, cell_gradients=None):
        """One integrated grad(value) dot area vector per shared native face."""
        gradient=self.face_gradient(values,boundary_values,compact=True,cell_gradients=cell_gradients)
        return np.einsum('fi,fi->f',gradient,self.mesh.area_vectors) if gradient.ndim==2 else np.einsum('fki,fi->fk',gradient,self.mesh.area_vectors)
