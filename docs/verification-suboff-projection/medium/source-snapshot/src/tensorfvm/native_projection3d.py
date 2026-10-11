"""Static native-face scalar projection with mixed scalar boundary data.

This initializes a mass-conserving face field on non-periodic 3D meshes.
It is not a Navier--Stokes solver: the scalar is a velocity potential, not
physical pressure, and a projected cell velocity need not have the same
flux as the independently corrected native face field.
"""
import numpy as np
from scipy.sparse import coo_matrix,diags
from scipy.sparse.linalg import splu
from .native_operators3d import NativeOperators3D


class NativeProjection3D:
    def __init__(self,mesh,dirichlet_mask):
        self.mesh=mesh;self.op=NativeOperators3D(mesh)
        self.internal=self.op.internal;self.boundary=self.op.boundary
        self.dirichlet=np.asarray(dirichlet_mask,dtype=bool)
        if self.dirichlet.shape!=self.boundary.shape or np.any(self.dirichlet&self.internal):
            raise ValueError('Dirichlet mask must select physical boundary faces')
        if not self.dirichlet.any():raise ValueError('an anchored Dirichlet scalar boundary is required')
        self.neumann=self.boundary&~self.dirichlet
        self.N=len(mesh.cells);self.F=len(mesh.owner)
        o,n=mesh.owner,mesh.neighbor;i=self.internal
        self.normal=mesh.area_vectors/np.linalg.norm(mesh.area_vectors,axis=1)[:,None]
        d=self.op.d;distance=np.linalg.norm(d,axis=1);w=self.op.weights
        rows=d.copy();rows[self.neumann]=self.normal[self.neumann]*distance[self.neumann,None]
        matrix=self.op.integrate(w[:,None,None]*rows[:,:,None]*rows[:,None,:],signed=False)
        if np.any(np.linalg.eigvalsh(matrix)[:,0]<=1e-13*np.linalg.eigvalsh(matrix)[:,-1]):
            raise ValueError('rank-deficient mixed-boundary pressure gradient stencil')
        inverse=np.linalg.inv(matrix)
        self.owner_response=np.einsum('fij,fj->fi',inverse[o],rows)*w[:,None]
        neighbor_response=np.einsum('fij,fj->fi',inverse[n[i]],rows[i])*w[i,None]
        bd=self.dirichlet;bo=self.owner_response
        self.gradient_matrix=[]
        for k in range(3):
            row=np.r_[o[i],o[i],n[i],n[i],o[bd]]
            col=np.r_[n[i],o[i],n[i],o[i],o[bd]]
            value=np.r_[bo[i,k],-bo[i,k],neighbor_response[:,k],-neighbor_response[:,k],-bo[bd,k]]
            self.gradient_matrix.append(coo_matrix((value,(row,col)),shape=(self.N,self.N)).tocsr())
        f=np.arange(self.F)
        self.divergence_matrix=coo_matrix((np.r_[np.ones(self.F),-np.ones(i.sum())],(np.r_[o,n[i]],np.r_[f,f[i]])),shape=(self.N,self.F)).tocsr()
        s=mesh.area_vectors;projection=(s*d).sum(1)
        if np.any(projection<=0):raise ValueError('nonpositive face projection')
        self.coefficient=(s*s).sum(1)/projection
        self.tangent=s-self.coefficient[:,None]*d
        # Weighted interpolation uses the actual face-normal projected position.
        self.weight=np.zeros(self.F)
        offset=mesh.face_centers-mesh.cell_centers[o]
        self.weight[i]=(offset[i]*s[i]).sum(1)/projection[i]
        self.interpolation=coo_matrix((np.r_[1-self.weight[i],self.weight[i],np.ones(self.boundary.sum())],(np.r_[f[i],f[i],f[self.boundary]],np.r_[o[i],n[i],o[self.boundary]])),shape=(self.F,self.N)).tocsr()
        two_point=coo_matrix((np.r_[-self.coefficient[i],self.coefficient[i],-self.coefficient[bd]],(np.r_[f[i],f[i],f[bd]],np.r_[o[i],n[i],o[bd]])),shape=(self.F,self.N)).tocsr()
        flux_matrix=two_point
        active=i|bd
        for k in range(3):
            flux_matrix=flux_matrix+diags(self.tangent[:,k]*active)@self.interpolation@self.gradient_matrix[k]
        self.flux_matrix=flux_matrix.tocsr()
        # Assemble the entire nonorthogonal response. No lagged correction or
        # hidden iteration tolerance is used in this CPU reference operator.
        self.matrix=(-self.divergence_matrix@self.flux_matrix).tocsc()
        self.lu=splu(self.matrix)

    def _data(self,neumann_gradients,dirichlet_values):
        g=np.broadcast_to(np.asarray(neumann_gradients,dtype=float),(self.F,))
        p=np.broadcast_to(np.asarray(dirichlet_values,dtype=float),(self.F,))
        if not np.isfinite(g).all() or not np.isfinite(p).all():raise ValueError('finite boundary scalar data required')
        return g,p

    def gradient_source(self,neumann_gradients=0.,dirichlet_values=0.):
        g,p=self._data(neumann_gradients,dirichlet_values)
        delta=np.zeros(self.F);delta[self.dirichlet]=p[self.dirichlet]
        delta[self.neumann]=g[self.neumann]*np.linalg.norm(self.op.d[self.neumann],axis=1)
        result=np.zeros((self.N,3))
        np.add.at(result,self.mesh.owner,self.owner_response*delta[:,None])
        return result

    def gradient(self,scalar,neumann_gradients=0.,dirichlet_values=0.):
        p=np.asarray(scalar,dtype=float)
        if p.shape!=(self.N,) or not np.isfinite(p).all():raise ValueError('finite scalar cell field required')
        return np.column_stack([a@p for a in self.gradient_matrix])+self.gradient_source(neumann_gradients,dirichlet_values)

    def flux_source(self,neumann_gradients=0.,dirichlet_values=0.):
        g,p=self._data(neumann_gradients,dirichlet_values)
        gf=self.interpolation@self.gradient_source(g,p)
        result=(gf*self.tangent).sum(1)
        result[self.dirichlet]+=self.coefficient[self.dirichlet]*p[self.dirichlet]
        result[self.neumann]=g[self.neumann]*np.linalg.norm(self.mesh.area_vectors[self.neumann],axis=1)
        return result

    def scalar_flux(self,scalar,neumann_gradients=0.,dirichlet_values=0.):
        return self.flux_matrix@np.asarray(scalar)+self.flux_source(neumann_gradients,dirichlet_values)

    def scalar_face_values(self,scalar,neumann_gradients=0.,dirichlet_values=0.):
        m=self.mesh;g,p=self._data(neumann_gradients,dirichlet_values)
        gradient=self.gradient(scalar,g,p);offset=m.face_centers-m.cell_centers[m.owner]
        result=np.asarray(self.interpolation@scalar).ravel()
        displacement=offset-self.weight[:,None]*self.op.d
        result[self.internal]+=((self.interpolation@gradient)[self.internal]*displacement[self.internal]).sum(1)
        b=self.neumann;normal=self.normal[b]
        tangent=offset[b]-(offset[b]*normal).sum(1)[:,None]*normal
        result[b]=scalar[m.owner[b]]+(gradient[m.owner[b]]*tangent).sum(1)+g[b]*(offset[b]*normal).sum(1)
        result[self.dirichlet]=p[self.dirichlet]
        return result

    def project(self,tentative_flux,neumann_gradients=0.,dirichlet_values=0.):
        q=np.asarray(tentative_flux,dtype=float)
        if q.shape!=(self.F,) or not np.isfinite(q).all():raise ValueError('finite native-face tentative flux required')
        source=self.flux_source(neumann_gradients,dirichlet_values)
        rhs=-(self.divergence_matrix@(q-source))
        scalar=self.lu.solve(rhs)
        corrected=q-self.scalar_flux(scalar,neumann_gradients,dirichlet_values)
        if not np.isfinite(scalar).all() or not np.isfinite(corrected).all():raise RuntimeError('nonfinite native projection')
        return scalar,corrected
