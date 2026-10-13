"""Nonnegative tangential gradient-jump energy for hybrid vector viscosity.

Face values constrain averages, not the reconstructed tangential variation.
On simplices this is the missing CR strain/Korn stabilization mechanism;
see Hansbo & Larson (2003), doi:10.1051/m2an:2003020, equation (26).
The polyhedral construction here is an independently audited extension.
Only INTERNAL faces contribute. No generic coercivity theorem is claimed.
"""
import numpy as np
from scipy import sparse
from .native_face_geometry3d import polygon_triangles


class NativeGradientJump3D:
    def __init__(self, mesh, local, viscosity, strength):
        self.mesh=mesh;self.N=len(mesh.cells);self.F=len(mesh.owner)
        self.internal=np.flatnonzero(mesh.neighbor>=0)
        ids=self.internal;count=len(ids);rows=[];cols=[];values=[[],[],[]]
        for cell,(faces,_,_,_,gradient,_,_) in enumerate(local):
            nodes=np.r_[cell,self.N+faces]
            rows.extend([cell]*len(nodes));cols.extend(nodes)
            for j in range(3):values[j].extend(gradient[j])
        self.gradient=[sparse.coo_matrix((v,(rows,cols)),shape=(self.N,self.N+self.F)).tocsr()
                       for v in values]
        incidence=sparse.coo_matrix((np.r_[np.ones(count),-np.ones(count)],
            (np.tile(np.arange(count),2),np.r_[mesh.owner[ids],mesh.neighbor[ids]])),
            shape=(count,self.N)).tocsr()
        jump=[incidence@g for g in self.gradient]
        covariance=[]
        for face in ids:
            triangles=polygon_triangles(mesh.vertices[list(mesh.face_vertices[face])])
            area=.5*np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)
            center=triangles.mean(axis=1);r=triangles-center[:,None,:]
            cov=np.einsum('tvi,tvj->tij',r,r)/12
            offset=center-mesh.face_centers[face]
            cov+=np.einsum('ti,tj->tij',offset,offset)
            covariance.append(np.sum(area[:,None,None]*cov,axis=0)/area.sum())
        self.covariance=np.asarray(covariance).reshape(count,3,3)
        area=np.linalg.norm(mesh.area_vectors[ids],axis=1)
        height=np.sum((mesh.cell_centers[mesh.neighbor[ids]]-mesh.cell_centers[mesh.owner[ids]])
                      *mesh.area_vectors[ids],axis=1)/area
        if np.any(height<=0):raise ValueError('positive normal center distance required for gradient jump')
        mo,mn=viscosity[mesh.owner[ids]],viscosity[mesh.neighbor[ids]]
        harmonic=2*mo*mn/(mo+mn)
        self.weight=strength*harmonic*area/height
        scalar=sparse.csr_matrix((self.N+self.F,self.N+self.F))
        for j in range(3):
            for k in range(3):
                scalar+=jump[j].T@sparse.diags(self.weight*self.covariance[:,j,k])@jump[k]
        scalar=(scalar+scalar.T)*.5
        self.matrix=sparse.kron(scalar,sparse.eye(3),format='csr')
        self.matrix.eliminate_zeros()

    def gradient_reactions(self, state):
        """Independent local energy replay, used in the actual traction ledger."""
        values=np.asarray(state).reshape(self.N+self.F,3)
        gradient=np.stack([g@values for g in self.gradient],axis=2)
        ids=self.internal;m=self.mesh
        jump=gradient[m.owner[ids]]-gradient[m.neighbor[ids]]
        response=self.weight[:,None,None]*np.einsum('fij,fjk->fik',jump,self.covariance)
        stress=np.zeros((self.N,3,3))
        np.add.at(stress,m.owner[ids],response)
        np.add.at(stress,m.neighbor[ids],-response)
        energy=float(np.sum(jump*response))
        return stress,energy
