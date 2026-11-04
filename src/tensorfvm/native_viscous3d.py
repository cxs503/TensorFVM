"""Shared-face scalar diffusion and full symmetric viscous traction in 3D.

Uses linear mixed Dirichlet/normal-derivative data, fully assembled
nonorthogonal gradients and known viscosity. No flow solver or pressure BC
is supplied. Warped mixed cells must already expose planar native facets.
All vector matrices use cell/face major, component minor (interleaved) order.

The face gradient is the interpolated cell gradient plus a deferred
nonorthogonal correction ``gamma*s*(du - d.grad)``.  ``correction`` selects
``gamma``: ``'over-relaxed'`` (default, ``1/(s.d)``), ``'minimum'``
(``(s.d)/(|d|^2|s|^2)``, orthogonal/minimum correction), ``'none'``, or a raw
nonnegative scalar.  ``correction_damping`` (default ``1.``) additionally
scales the correction by a face-wise factor in ``[0, 1]``.  Every choice reproduces the exact affine traction
``mu*s.grad`` because the correction vanishes for affine fields; none of them
is guaranteed to satisfy the discrete passivity inequality, see
:mod:`tensorfvm.verification.native_energy`.
Wall closure
------------
``wall_closure`` selects the Dirichlet-face normal-derivative estimate:

``'two_point'`` (default, historical behaviour)
    The correction drives the face value to the prescribed data, which makes
    the effective wall normal derivative the two-point estimate
    ``(u_D - u_cell - grad u_cell . d)/|d|``.  Its Taylor expansion misses the
    curvature term ``(delta/2) d2u/dn2``, so the wall traction is first order
    and the wall-adjacent cell divergence stays O(1) (see
    :mod:`tensorfvm.verification.native_order3d` and
    ``docs/native_viscous_dirichlet_order.md``).

``'three_point'``
    The normal part of the face gradient on every Dirichlet face is replaced
    by a one-sided three-point stencil through the face, the owner cell centre
    and a second cell centre taken along the mesh line in the face-normal
    direction (the neighbour across the internal face whose normal is best
    aligned with the inward face normal).  The tangential part of the face
    gradient is left untouched, except for the offset term described below.
    Faces without an aligned second cell (alignment < cos 60 deg, or
    degenerate spacing) keep the two-point closure; the number of upgraded
    faces is exposed as ``wall_second_order_faces``.

    Measured properties (``scripts/verify_three_point_wall_closure.py``,
    ``docs/wall-closure-three-point/report.md``): affine-exact at round-off on
    Cartesian and warped box meshes, a 2.4x smaller wall normal-derivative
    error on the quadratic benchmark, but an *observed* order of 1.2-1.5 rather
    than 2 -- the offset term is evaluated with the two-point tangential
    gradient, whose wall error is O(h) with an O(1) weight.  It also does not
    repay the O(1) wall-adjacent cell divergence, whose driver is documented in
    ``docs/native_viscous_dirichlet_order.md`` as the evaluation point of the
    whole wall-face gradient tensor, not the normal derivative.

"""
import numpy as np
from scipy.sparse import coo_matrix, diags, kron, eye, bmat
from .native_operators3d import NativeOperators3D


class NativeViscous3D:
    def __init__(self,mesh,dirichlet_mask,dynamic_viscosity=1.,*,deviatoric=True,correction='over-relaxed',correction_damping=1.,wall_closure='two_point'):
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
        dd=np.einsum('ij,ij->i',d,d);ss=np.einsum('ij,ij->i',s,s)
        self.correction=correction
        if isinstance(correction,str):
            if correction=='over-relaxed':gamma=1/projection
            elif correction=='minimum':gamma=projection/(dd*ss)
            elif correction=='none':gamma=np.zeros(self.F)
            else:raise ValueError('unknown nonorthogonal correction')
        elif np.isfinite(correction) and correction>=0:gamma=np.full(self.F,float(correction))
        else:raise ValueError('correction must be a known name or a finite nonnegative scale')
        self.correction_gamma=gamma
        self.correction_direction=gamma[:,None]*s
        damping=np.asarray(correction_damping,dtype=float)
        if damping.ndim==0:damping=np.full(self.F,float(damping))
        if damping.shape!=(self.F,) or not np.isfinite(damping).all() or np.any(damping<0) or np.any(damping>1):raise ValueError('correction damping must be a scalar or face array within [0,1]')
        self.correction_damping=damping
        damped_active=active*damping
        self.face_gradient_matrices=[(a+diags(self.correction_direction[:,k]*damped_active)@(self.delta_matrix-contraction)-diags(self.normal[:,k]*self.neumann)@normal_contraction).tocsr() for k,a in enumerate(self.raw_face_gradient)]
        self.wall_closure=wall_closure
        self.wall_second_cell=-np.ones(self.F,dtype=int)
        self.wall_coeff=np.zeros((self.F,3))
        self.wall_offset=np.zeros((self.F,3,2))
        self.wall_weight=np.zeros((self.F,3))
        self.wall_second_order_faces=0
        if wall_closure=='three_point':
            self._assemble_three_point_wall(mesh,o,n,i,bd,f)
        elif wall_closure!='two_point':raise ValueError("wall_closure must be 'two_point' or 'three_point'")
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

    def _assemble_three_point_wall(self,mesh,o,n,i,bd,f):
        """Replace the Dirichlet-face normal gradient component by a one-sided
        three-point stencil (face, owner centre, second cell centre).

        The stencil is exact for affine fields; for smooth fields the observed
        order is 1.2-1.5 because the tangential offset term below inherits the
        O(h) wall error of the two-point tangential gradient.  Faces without an
        aligned second cell keep the two-point closure.
        """
        F=self.F;N=self.N
        second=-np.ones(F,dtype=int)
        coeff=np.zeros((F,3))
        offset=np.zeros((F,3,2))
        candidates={}
        for gf in np.where(i)[0]:
            candidates.setdefault(o[gf],[]).append((gf,self.normal[gf],1))
            candidates.setdefault(n[gf],[]).append((gf,self.normal[gf],-1))
        fc=mesh.face_centers;cc=mesh.cell_centers
        for face in np.where(bd)[0]:
            c0=o[face];nf=self.normal[face]
            best_align=-2.0;best_gf=-1
            for gf,gn,sig in candidates.get(c0,()):
                # Walk *inward*, away from the wall: the second centre lies behind
                # the owner centre seen from the face, so the candidate face
                # normal (owner -> neighbour, sign corrected by ``sig``) must be
                # anti-parallel to the outward face normal ``nf``.
                align=-sig*(gn@nf)
                if align>best_align:best_align,best_gf=align,gf
            if best_gf<0 or best_align<0.5:continue
            c1=n[best_gf] if o[best_gf]==c0 else o[best_gf]
            if c1<0:continue
            # ``nf`` is outward, so the interior side is reached along ``-nf``.
            inward=-nf
            r0=cc[c0]-fc[face];r1=cc[c1]-fc[face]
            delta0=float(r0@inward);delta1=float(r1@inward)
            if delta0<=0 or delta1<=delta0*(1+1e-9):continue
            gap=delta1-delta0
            coeff[face,0]=(delta0+delta1)/(delta0*delta1)
            coeff[face,1]=-delta1/(delta0*gap)
            coeff[face,2]=delta0/(delta1*gap)
            # Tangential part of each radius vector: a triangulated boundary
            # facet puts its centroid off the normal line through the owner
            # centre, so the stencil needs ``u - e.grad_t`` to stay exact.
            offset[face,:,0]=r0-delta0*inward
            offset[face,:,1]=r1-delta1*inward
            second[face]=c1
        self.wall_second_cell=second
        self.wall_coeff=coeff
        self.wall_offset=offset
        self.wall_second_order_faces=int((second[bd]>=0).sum())
        nrm=self.normal
        fbb=f[bd][second[bd]>=0]
        if len(fbb)==0:return
        # Normal-derivative rows of the *un-upgraded* face gradients: their normal
        # component is what the one-sided stencil replaces, and what is left of a
        # row afterwards is the tangential operator used by the offset term.
        nc=sum(diags(nrm[:,a])@self.face_gradient_matrices[a] for a in range(3)).tocsr()
        tangential=[(self.face_gradient_matrices[a]-diags(nrm[:,a])@nc).tocsr() for a in range(3)]
        P=coo_matrix((np.ones(len(fbb)),(np.arange(len(fbb)),fbb)),shape=(len(fbb),F)).tocsr()
        tangential=[(P@t).tocsr() for t in tangential]
        weight=-(coeff[fbb,1][:,None]*offset[fbb,:,0]+coeff[fbb,2][:,None]*offset[fbb,:,1])
        self.wall_weight[fbb]=weight
        correction=sum(diags(weight[:,a])@tangential[a] for a in range(3)).tocsr()
        for k in range(3):
            M=self.face_gradient_matrices[k]
            selected=tangential[k]+diags(nrm[fbb,k])@correction
            selected=selected+coo_matrix((np.r_[nrm[fbb,k]*coeff[fbb,1],nrm[fbb,k]*coeff[fbb,2]],
                                          (np.r_[np.arange(len(fbb)),np.arange(len(fbb))],
                                           np.r_[o[fbb],second[fbb]])),shape=(len(fbb),N)).tocsr()
            self.face_gradient_matrices[k]=(M-P.T@(P@M)+P.T@selected).tocsr()

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
        result=raw+self.correction_direction*(self.correction_damping*active*(delta-np.einsum('ij,ij->i',raw,self.op.d)))[:,None]
        nb=self.neumann;result[nb]+=self.normal[nb]*(g[nb]-np.einsum('ij,ij->i',raw[nb],self.normal[nb]))[:,None]
        if self.wall_closure=='three_point':
            sel=self.dirichlet&(self.wall_second_cell>=0)
            n=self.normal[sel]
            base=result[sel]
            tangential=base-np.einsum('fi,fi->f',base,n)[:,None]*n
            offset_term=np.einsum('fi,fi->f',self.wall_weight[sel],tangential)
            result[sel]=tangential+n*(self.wall_coeff[sel,0]*v[sel]+offset_term)[:,None]
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
