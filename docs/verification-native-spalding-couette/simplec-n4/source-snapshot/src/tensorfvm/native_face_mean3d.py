"""Conservative cell mean and its dual force moments on shared faces.

R_K U = sum_f (x_f-x_K) (U_f dot S_Kf) / V_K. This preserves
constants and is dual to the existing pressure-compatible body load. It is
not an affine cell-centre interpolant on a general polyhedron.
"""
import numpy as np
from scipy import sparse


class NativeFaceMean3D:
    def __init__(self, mesh):
        self.mesh = mesh
        self.N, self.F = len(mesh.cells), len(mesh.owner)
        faces = np.arange(self.F); internal = mesh.neighbor >= 0
        cells = np.r_[mesh.owner, mesh.neighbor[internal]]
        faces = np.r_[faces, faces[internal]]
        vectors = np.vstack((mesh.area_vectors, -mesh.area_vectors[internal]))
        offsets = mesh.face_centers[faces]-mesh.cell_centers[cells]
        values = np.einsum('fi,fj->fij', offsets, vectors)/mesh.cell_volumes[cells,None,None]
        rows = np.broadcast_to((3*cells[:,None]+np.arange(3))[:,:,None], values.shape)
        cols = np.broadcast_to((3*(self.N+faces[:,None])+np.arange(3))[:,None,:], values.shape)
        self.matrix = sparse.coo_matrix((values.ravel(), (rows.ravel(),cols.ravel())),
            shape=(3*self.N,3*(self.N+self.F))).tocsr()

    def moments(self, integrated_force):
        """Outward per-cell face moments whose cell sum equals input force."""
        m = self.mesh
        force = np.asarray(integrated_force,float)/m.cell_volumes[:,None]
        if force.shape != (self.N,3) or not np.isfinite(force).all():
            raise ValueError('one finite integrated vector force per cell required')
        offset = m.face_centers-m.cell_centers[m.owner]
        owner = np.sum(force[m.owner]*offset,axis=1)[:,None]*m.area_vectors
        neighbor = np.zeros_like(owner); internal = m.neighbor >= 0
        offset = m.face_centers[internal]-m.cell_centers[m.neighbor[internal]]
        neighbor[internal] = -np.sum(force[m.neighbor[internal]]*offset,axis=1)[:,None]*m.area_vectors[internal]
        return owner, neighbor
