"""Shared-face scalar diffusion and full symmetric viscous traction in 3D.

Uses linear mixed Dirichlet/normal-derivative data, fully assembled
nonorthogonal gradients and known viscosity. No flow solver or pressure BC
is supplied. Warped mixed cells must already expose planar native facets.
All vector matrices use cell/face major, component minor (interleaved) order.
"""
import numpy as np
from scipy.sparse import coo_matrix, diags, kron, eye, bmat
from .native_operators3d import NativeOperators3D


class NativeViscous3D:
    def __init__(self,mesh,dirichlet_mask,dynamic_viscosity=1.,*,deviatoric=True):
        self.mesh=mesh;self.op=NativeOperators3D(mesh)
        self.internal=self.op.internal;self.boundary=self.op.boundary
        self.N=len(mesh.cells);self.F=len(mesh.owner)
        self.dirichlet=np.asarray(dirichlet_mask,dtype=bool)
        if self.dirichlet.shape!=(self.F,) or np.any(self.dirichlet&self.internal):raise ValueError('Dirichlet mask must select native boundary faces')
        self.neumann=self.boundary&~self.dirichlet
        m=mesh;o,n=m.owner,m.neighbor;i=self.internal;bd=self.dirichlet;f=np.arange(self.F)
        s=m.area_vectors;self.area=np.linalg.norm(s,axis=1);self.normal=s/self.area[:,None]
        d=self.op.d;distance=np.linalg.norm(d,axis=1);w=self.op.weights
        rows=d.copy();rows[self.neumann]=self.normal[self.neumann]*distance[self.neumann,None]
        normal=self.op.integrate(w[:,None,None]*rows[:,:,None]*rows[:,None,:],signed=False)
        eigen=np.linalg.eigvalsh(normal)
        if np.any(eigen[:,0]<=1e-13*eigen[:,-1]):raise ValueError('rank-deficient mixed gradient stencil')
        inverse=np.linalg.inv(normal)
        self.owner_response=np.einsum('fij,fj->fi',inverse[o],rows)*w[:,None]
        nr=np.einsum('fij,fj->fi',inverse[n[i]],rows[i])*w[i,None]
        self.gradient_matrices=[]
        for k in range(3):
            self.gradient_matrices.append(coo_matrix((np.r_[self.owner_response[i,k],-self.owner_response[i,k],nr[:,k],-nr[:,k],-self.owner_response[bd,k]],(np.r_[o[i],o[i],n[i],n[i],o[bd]],np.r_[n[i],o[i],n[i],o[i],o[bd]])),shape=(self.N,self.N)).tocsr())
        self.gradient_matrix=self.gradient_matrices
        self.divergence_matrix=coo_matrix((np.r_[np.ones(self.F),-np.ones(i.sum())],(np.r_[o,n[i]],np.r_[f,f[i]])),shape=(self.N,self.F)).tocsr()
        projection=np.einsum('ij,ij->i',s,d)
        if np.any(projection<=0):raise ValueError('nonpositive face projection')
        self.weight=np.zeros(self.F);self.weight[i]=np.einsum('ij,ij->i',(m.face_centers-m.cell_centers[o])[i],s[i])/projection[i]
        self.interpolation=coo_matrix((np.r_[1-self.weight[i],self.weight[i],np.ones(self.boundary.sum())],(np.r_[f[i],f[i],f[self.boundary]],np.r_[o[i],n[i],o[self.boundary]])),shape=(self.F,self.N)).tocsr()
        self.delta_matrix=coo_matrix((np.r_[-np.ones(i.sum()),np.ones(i.sum()),-np.ones(bd.sum())],(np.r_[f[i],f[i],f[bd]],np.r_[o[i],n[i],o[bd]])),shape=(self.F,self.N)).tocsr()
        self.raw_face_gradient=[self.interpolation@g for g in self.gradient_matrices]
        contraction=sum(diags(d[:,k])@self.raw_face_gradient[k] for k in range(3))
        normal_contraction=sum(diags(self.normal[:,k])@self.raw_face_gradient[k] for k in range(3))
        active=i|bd
        self.correction_direction=s/projection[:,None]
        self.face_gradient_matrices=[(a+diags(self.correction_direction[:,k]*active)@(self.delta_matrix-contraction)-diags(self.normal[:,k]*self.neumann)@normal_contraction).tocsr() for k,a in enumerate(self.raw_face_gradient)]
        mu=np.asarray(dynamic_viscosity,dtype=float)
        if mu.ndim==0:mu=np.full(self.F,float(mu))
        elif mu.shape==(self.N,):mu=np.asarray(self.interpolation@mu)
        elif mu.shape!=(self.F,):raise ValueError('viscosity must be scalar, cell or face array')
        if not np.isfinite(mu).all() or np.any(mu<=0):raise ValueError('finite positive dynamic viscosity required')
        self.face_viscosity=mu;self.deviatoric=bool(deviatoric)
        self.scalar_flux_matrix=sum(diags(mu*s[:,k])@self.face_gradient_matrices[k] for k in range(3)).tocsr()
        self.scalar_diffusion_matrix=(self.divergence_matrix@self.scalar_flux_matrix).tocsr()
        gs=self.face_gradient_matrices
        normal_grad=sum(diags(s[:,k])@gs[k] for k in range(3))
        blocks=[]
        for a in range(3):
            row=[]
            for b in range(3):
                block=diags(mu)*(diags(s[:,b])@gs[a]-(2/3 if self.deviatoric else 0)*diags(s[:,a])@gs[b])
                if a==b:block=block+diags(mu)@normal_grad
                row.append(block)
            blocks.append(row)
        component_matrix=bmat(blocks,format='csr')
        face_permutation=np.arange(self.F*3).reshape(3,self.F).T.ravel()
        cell_permutation=np.arange(self.N*3).reshape(3,self.N).T.ravel()
        self.momentum_flux_matrix=component_matrix[face_permutation,:][:,cell_permutation].tocsr()
        self.vector_divergence_matrix=kron(self.divergence_matrix,eye(3),format='csr')
        self.momentum_diffusion_matrix=(self.vector_divergence_matrix@self.momentum_flux_matrix).tocsr()

    def _boundary_data(self,neumann_gradients,dirichlet_values):
        g=np.broadcast_to(np.asarray(neumann_gradients,dtype=float),(self.F,))
        v=np.broadcast_to(np.asarray(dirichlet_values,dtype=float),(self.F,))
        if not np.isfinite(g).all() or not np.isfinite(v).all():raise ValueError('finite boundary data required')
        return g,v

    def gradient_source(self,neumann_gradients=0.,dirichlet_values=0.):
        g,v=self._boundary_data(neumann_gradients,dirichlet_values)
        delta=np.zeros(self.F);delta[self.dirichlet]=v[self.dirichlet]
        delta[self.neumann]=g[self.neumann]*np.linalg.norm(self.op.d[self.neumann],axis=1)
        result=np.zeros((self.N,3));np.add.at(result,self.mesh.owner,self.owner_response*delta[:,None]);return result

    def face_gradient_source(self,neumann_gradients=0.,dirichlet_values=0.):
        g,v=self._boundary_data(neumann_gradients,dirichlet_values)
        raw=np.asarray(self.interpolation@self.gradient_source(g,v))
        delta=np.zeros(self.F);delta[self.dirichlet]=v[self.dirichlet]
        active=self.internal|self.dirichlet
        result=raw+self.correction_direction*(active*(delta-np.einsum('ij,ij->i',raw,self.op.d)))[:,None]
        nb=self.neumann;result[nb]+=self.normal[nb]*(g[nb]-np.einsum('ij,ij->i',raw[nb],self.normal[nb]))[:,None]
        return result

    def scalar_flux_source(self,neumann_gradients=0.,dirichlet_values=0.):
        return self.face_viscosity*np.einsum('ij,ij->i',self.face_gradient_source(neumann_gradients,dirichlet_values),self.mesh.area_vectors)

    def scalar_flux(self,values,neumann_gradients=0.,dirichlet_values=0.):
        v=np.asarray(values,dtype=float)
        if v.shape!=(self.N,) or not np.isfinite(v).all():raise ValueError('finite scalar cell values required')
        return self.scalar_flux_matrix@v+self.scalar_flux_source(neumann_gradients,dirichlet_values)

    def _vector_data(self,neumann_gradients,dirichlet_values):
        g=np.broadcast_to(np.asarray(neumann_gradients,dtype=float),(self.F,3));v=np.broadcast_to(np.asarray(dirichlet_values,dtype=float),(self.F,3))
        if not np.isfinite(g).all() or not np.isfinite(v).all():raise ValueError('finite vector boundary data required')
        return g,v

    def momentum_flux_source(self,neumann_gradients=0.,dirichlet_values=0.):
        g,v=self._vector_data(neumann_gradients,dirichlet_values)
        gradient=np.stack([self.face_gradient_source(g[:,k],v[:,k]) for k in range(3)],axis=1)
        stress=gradient+gradient.transpose(0,2,1)
        if self.deviatoric:stress-=2/3*np.trace(gradient,axis1=1,axis2=2)[:,None,None]*np.eye(3)
        return self.face_viscosity[:,None]*np.einsum('fij,fj->fi',stress,self.mesh.area_vectors)

    def momentum_flux(self,velocity,neumann_gradients=0.,dirichlet_values=0.):
        u=np.asarray(velocity,dtype=float)
        if u.shape!=(self.N,3) or not np.isfinite(u).all():raise ValueError('finite vector cell values required')
        return (self.momentum_flux_matrix@u.ravel()).reshape(self.F,3)+self.momentum_flux_source(neumann_gradients,dirichlet_values)
