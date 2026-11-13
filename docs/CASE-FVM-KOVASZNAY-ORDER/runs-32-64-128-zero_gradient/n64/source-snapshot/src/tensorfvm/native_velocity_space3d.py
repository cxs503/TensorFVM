"""Exact face-normal constraint elimination; physical states remain Cartesian.

P has orthonormal columns and maps reduced increments into unconstrained
cell velocities and admissible face velocities. No normal penalty is used.

Three kinds of linear equality are eliminated exactly (never by penalty):

``dirichlet_components``
    a face component is prescribed (an affine shift, handled by the caller);
``normal_faces``
    the face-normal component is prescribed (rotated orthonormal basis);
``zero_gradient_components``
    the face component equals the value of its owner cell -- the discrete
    strong form of the homogeneous Neumann / symmetry condition
    ``d u_c / d n = 0`` at the face centre. Tied dofs leave the unknown set and
    their *rows* (the face equilibrium equations) are dropped, so the constraint
    replaces the face equation instead of being added to the cell balance. In
    the substitution ``u = P y`` the column of a tied dof is the column of its
    owner cell dof, hence ``matrix = (A P)[free]`` (row-restricted substitution,
    not the Galerkin projection ``P^T A P``, which would double-count the face
    flux in the cell balance).
"""
import numpy as np
from scipy import sparse


class NativeVelocitySpace3D:
    def __init__(self,mesh,components,dirichlet_components,normal_faces=None,
                 zero_gradient_components=None):
        self.N=len(mesh.cells);self.F=len(mesh.owner);self.components=components
        self.owner=np.asarray(mesh.owner)
        self.normal_faces=np.zeros(self.F,bool) if normal_faces is None else np.asarray(normal_faces,bool).copy()
        if self.normal_faces.shape!=(self.F,) or np.any(self.normal_faces & (mesh.neighbor>=0)):
            raise ValueError('normal velocity constraints require boundary faces')
        if self.normal_faces.any() and components!=3:raise ValueError('normal velocity constraints require three components')
        dirichlet_components=np.asarray(dirichlet_components,bool)
        if np.any(dirichlet_components[self.normal_faces]):raise ValueError('normal and Cartesian component constraints cannot overlap on a face')
        zero_gradient=np.zeros((self.F,components),bool) if zero_gradient_components is None \
            else np.asarray(zero_gradient_components,bool)
        if zero_gradient.shape==(self.F,):
            zero_gradient=np.broadcast_to(zero_gradient[:,None],(self.F,components)).copy()
        if zero_gradient.shape!=(self.F,components):
            raise ValueError('zero-gradient mask must be shaped (F,) or (F, components)')
        if np.any(zero_gradient & (mesh.neighbor>=0)[:,None]):
            raise ValueError('zero-gradient constraints require boundary faces')
        if np.any(zero_gradient & dirichlet_components):
            raise ValueError('zero-gradient and Dirichlet components cannot overlap')
        if np.any(zero_gradient & self.normal_faces[:,None]):
            raise ValueError('zero-gradient and normal constraints cannot overlap')
        self.zero_gradient_components=zero_gradient
        self.tied_any=bool(zero_gradient.any())
        self.cartesian_fixed=components*self.N+np.flatnonzero(dirichlet_components.ravel())
        self.normal=mesh.area_vectors/np.linalg.norm(mesh.area_vectors,axis=1)[:,None]
        mask=np.vstack((np.ones((self.N,components),bool),~dirichlet_components))
        mask[self.N:][zero_gradient]=False
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
        if self.tied_any:
            rows=[self.free];cols=[np.arange(len(self.free))]
            data=[np.ones(len(self.free))]
            face_dofs=[];cell_columns=[]
            for component in range(components):
                faces=np.flatnonzero(zero_gradient[:,component])
                if not len(faces):continue
                face_dofs.append(components*(self.N+faces)+component)
                cell_columns.append(columns[components*mesh.owner[faces]+component])
            if face_dofs:
                rows.append(np.concatenate(face_dofs))
                cols.append(np.concatenate(cell_columns))
                data.append(np.ones(len(np.concatenate(face_dofs))))
            self.prolongation=sparse.coo_matrix((np.concatenate(data),
                (np.concatenate(rows),np.concatenate(cols))),shape=(size,len(self.free))).tocsr()
        else:
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
        if self.rotated:return (self.prolongation.T@matrix@self.prolongation).tocsc()
        if self.tied_any:return (matrix@self.prolongation)[self.free].tocsc()
        return matrix[self.free][:,self.free].tocsc()

    def expand(self,vector):
        if self.rotated or self.tied_any:return self.prolongation@vector
        result=np.zeros(self.prolongation.shape[0]);result[self.free]=vector;return result

    def constraint_error(self,difference):
        fixed=float(np.max(abs(difference[self.cartesian_fixed]),initial=0.))
        if self.tied_any:
            cell=difference[:self.components*self.N].reshape(self.N,self.components)
            face=difference[self.components*self.N:].reshape(self.F,self.components)
            faces,components=np.nonzero(self.zero_gradient_components)
            tied=float(np.max(abs(face[self.zero_gradient_components]
                                   -cell[self.owner[faces],components]),initial=0.))
            fixed=max(fixed,tied)
        if not self.rotated:return fixed
        face=difference[self.components*self.N:].reshape(self.F,self.components)
        normal=float(np.max(abs(np.sum(face[self.normal_faces]*self.normal[self.normal_faces],axis=1)),initial=0.))
        return max(fixed,normal)