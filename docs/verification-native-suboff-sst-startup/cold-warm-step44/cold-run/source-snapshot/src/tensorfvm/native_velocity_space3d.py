"""Exact face-normal constraint elimination; physical states remain Cartesian.

P has orthonormal columns and maps reduced increments into unconstrained
cell velocities and admissible face velocities. No normal penalty is used.
"""
import numpy as np
from scipy import sparse


class NativeVelocitySpace3D:
    def __init__(self,mesh,components,dirichlet_components,normal_faces=None):
        self.N=len(mesh.cells);self.F=len(mesh.owner);self.components=components
        self.normal_faces=np.zeros(self.F,bool) if normal_faces is None else np.asarray(normal_faces,bool).copy()
        if self.normal_faces.shape!=(self.F,) or np.any(self.normal_faces & (mesh.neighbor>=0)):
            raise ValueError('normal velocity constraints require boundary faces')
        if self.normal_faces.any() and components!=3:raise ValueError('normal velocity constraints require three components')
        if np.any(dirichlet_components[self.normal_faces]):raise ValueError('normal and Cartesian component constraints cannot overlap on a face')
        self.cartesian_fixed=components*self.N+np.flatnonzero(dirichlet_components.ravel())
        self.normal=mesh.area_vectors/np.linalg.norm(mesh.area_vectors,axis=1)[:,None]
        mask=np.vstack((np.ones((self.N,components),bool),~dirichlet_components))
        self.rotated=bool(self.normal_faces.any());bases={}
        for f in np.flatnonzero(self.normal_faces):
            normal=self.normal[f];pivot=int(np.argmax(abs(normal)))
            axes=[i for i in range(3) if i!=pivot]
            tangent=np.eye(3)[axes[0]]-normal[axes[0]]*normal
            tangent/=np.linalg.norm(tangent)
            second=np.cross(normal,tangent)
            bases[f]=(axes,np.column_stack((tangent,second)))
            mask[self.N+f,pivot]=False
        self.free=np.flatnonzero(mask.ravel());size=components*(self.N+self.F)
        columns=np.full(size,-1,int);columns[self.free]=np.arange(len(self.free))
        selected=np.r_[np.zeros(self.N,bool),self.normal_faces]
        ordinary=~selected[self.free//components]
        rows=[self.free[ordinary]];cols=[np.flatnonzero(ordinary)];data=[np.ones(np.count_nonzero(ordinary))]
        for f,(axes,basis) in bases.items():
            rows.append(np.repeat(3*(self.N+f)+np.arange(3),2))
            cols.append(np.tile(columns[3*(self.N+f)+np.asarray(axes)],3))
            data.append(basis.ravel())
        self.prolongation=sparse.coo_matrix((np.concatenate(data),(np.concatenate(rows),np.concatenate(cols))),
            shape=(size,len(self.free))).tocsr()

    def restrict(self,vector):
        return self.prolongation.T@vector if self.rotated else vector[self.free]

    def matrix(self,matrix):
        return (self.prolongation.T@matrix@self.prolongation).tocsc() if self.rotated else matrix[self.free][:,self.free].tocsc()

    def expand(self,vector):
        if self.rotated:return self.prolongation@vector
        result=np.zeros(self.prolongation.shape[0]);result[self.free]=vector;return result

    def constraint_error(self,difference):
        fixed=float(np.max(abs(difference[self.cartesian_fixed]),initial=0.))
        if not self.rotated:return fixed
        face=difference[self.components*self.N:].reshape(self.F,self.components)
        normal=float(np.max(abs(np.sum(face[self.normal_faces]*self.normal[self.normal_faces],axis=1)),initial=0.))
        return max(fixed,normal)
