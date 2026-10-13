"""CPU reference steady incompressible FV coupling on native shared 3D faces.

Fully coupled momentum/continuity with affine-consistent Rhie--Chow pressure
stabilization. Manufactured Dirichlet problems and laminar external-flow
pilots are supported. This module alone is not SUBOFF/RANS qualification.
Vectors use cell-major/component-minor ordering throughout.
"""
from dataclasses import dataclass
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu,spilu,gmres,LinearOperator
from .native_viscous3d import NativeViscous3D


class FreePressureReconstruction3D:
    """Extrapolate pressure on velocity walls; do not impose zero dp/dn there."""
    def __init__(self,mesh,dirichlet_mask,*,skew_correction=True):
        self.mesh=mesh;self.N=len(mesh.cells);self.F=len(mesh.owner)
        self.dirichlet=np.asarray(dirichlet_mask,dtype=bool)
        i=mesh.neighbor>=0;b=~i
        if self.dirichlet.shape!=(self.F,) or np.any(self.dirichlet&i):raise ValueError('pressure Dirichlet mask must select boundary faces')
        self.internal=i;self.boundary=b;o,n=mesh.owner,mesh.neighbor;f=np.arange(self.F)
        self.base=NativeViscous3D(mesh,b)
        d=self.base.op.d;w=self.base.op.weights;active=i|self.dirichlet
        normal=self.base.op.integrate((w*active)[:,None,None]*d[:,:,None]*d[:,None,:],signed=False)
        e=np.linalg.eigvalsh(normal)
        deficient=np.flatnonzero(e[:,0]<=1e-13*e[:,-1])
        extra_cell=[];extra_neighbor=[];extra_d=[];extra_w=[]
        adjacency=[set() for _ in range(self.N)]
        for left,right in zip(o[i],n[i]):adjacency[left].add(int(right));adjacency[right].add(int(left))
        # Boundary tetrahedra may have only one/two internal neighbors. Extend
        # those pressure least-squares stencils through real mesh adjacency;
        # never substitute a fictitious zero wall-normal pressure gradient.
        for cell in deficient:
            visited={int(cell)}|adjacency[cell];frontier=adjacency[cell].copy()
            for depth in range(2,6):
                candidates=set().union(*(adjacency[j] for j in frontier))-visited if frontier else set()
                visited.update(candidates);frontier=candidates
                for neighbor in sorted(candidates):
                    displacement=mesh.cell_centers[neighbor]-mesh.cell_centers[cell]
                    weight=1/np.dot(displacement,displacement)
                    normal[cell]+=weight*np.outer(displacement,displacement)
                    extra_cell.append(cell);extra_neighbor.append(neighbor);extra_d.append(displacement);extra_w.append(weight)
                eigen=np.linalg.eigvalsh(normal[cell])
                if eigen[0]>1e-13*eigen[-1]:break
            else:raise ValueError('pressure extrapolation needs full-rank connected neighbor stencils')
        self.extended_pressure_stencils=int(len(deficient))
        inv=np.linalg.inv(normal);ro=np.einsum('fij,fj->fi',inv[o],d)*w[:,None];rn=np.einsum('fij,fj->fi',inv[n[i]],d[i])*w[i,None]
        self.response=ro;self.G=[];bd=self.dirichlet
        for k in range(3):
            self.G.append(sparse.coo_matrix((np.r_[-ro[i,k],ro[i,k],-rn[:,k],rn[:,k],-ro[bd,k]],(np.r_[o[i],o[i],n[i],n[i],o[bd]],np.r_[o[i],n[i],o[i],n[i],o[bd]])),shape=(self.N,self.N)).tocsr())
        if extra_cell:
            ec=np.asarray(extra_cell);en=np.asarray(extra_neighbor);ed=np.asarray(extra_d);ew=np.asarray(extra_w)
            response=np.einsum('eij,ej->ei',inv[ec],ed)*ew[:,None]
            for k in range(3):
                self.G[k]+=sparse.coo_matrix((np.r_[-response[:,k],response[:,k]],(np.r_[ec,ec],np.r_[ec,en])),shape=(self.N,self.N)).tocsr()
        self.I=self.base.interpolation;self.D=self.base.divergence_matrix;self.weight=self.base.weight
        self.skew=mesh.face_centers-mesh.cell_centers[o]-self.weight[:,None]*d
        if not skew_correction:self.skew[:]=0.
        self.face_matrix=self.I.copy()
        for k in range(3):self.face_matrix+=sparse.diags(self.skew[:,k])@self.I@self.G[k]
        self.face_matrix=(sparse.diags((~bd).astype(float))@self.face_matrix).tocsr()
        s=mesh.area_vectors;projection=(s*d).sum(1);self.coefficient=(s*s).sum(1)/projection
        tangent=s-self.coefficient[:,None]*d
        delta=sparse.coo_matrix((np.r_[-np.ones(i.sum()),np.ones(i.sum()),-np.ones(bd.sum())],(np.r_[f[i],f[i],f[bd]],np.r_[o[i],n[i],o[bd]])),shape=(self.F,self.N)).tocsr()
        self.gauss_gradient_matrices=[sparse.diags(1/mesh.cell_volumes)@self.D@sparse.diags(s[:,k])@self.face_matrix for k in range(3)]
        flux=sparse.diags(self.coefficient)@delta
        interpolated=sparse.csr_matrix((self.F,self.N))
        for k in range(3):
            interpolated+=sparse.diags(s[:,k])@self.I@self.gauss_gradient_matrices[k]
            flux+=sparse.diags(tangent[:,k])@self.I@self.G[k]
        self.compact_flux=(sparse.diags(active.astype(float))@flux).tocsr()
        self.interpolated_gradient_flux=(sparse.diags(active.astype(float))@interpolated).tocsr()
        self.pressure_flux_defect=(self.interpolated_gradient_flux-self.compact_flux).tocsr()
        self.pressure_force_matrix=sparse.vstack([self.D@sparse.diags(s[:,k])@self.face_matrix for k in range(3)],format='csr')
        permutation=np.arange(3*self.N).reshape(3,self.N).T.ravel()
        self.pressure_force_matrix=self.pressure_force_matrix[permutation,:]

    def sources(self,dirichlet_values=0.):
        m=self.mesh;v=np.broadcast_to(np.asarray(dirichlet_values,dtype=float),(self.F,))
        if not np.isfinite(v).all():raise ValueError('finite pressure boundary values required')
        b=self.dirichlet;g=np.zeros((self.N,3));np.add.at(g,m.owner[b],self.response[b]*v[b,None])
        gf=self.I@g;face=(gf*self.skew).sum(1);face[b]=v[b]
        tangent=m.area_vectors-self.coefficient[:,None]*self.base.op.d
        compact=(gf*tangent).sum(1);compact[b]+=self.coefficient[b]*v[b]
        active=self.internal|b;compact[~active]=0
        force=self.base.op.integrate(face[:,None]*m.area_vectors).ravel()
        gauss_source=force.reshape(self.N,3)/m.cell_volumes[:,None]
        interp=((self.I@gauss_source)*m.area_vectors).sum(1);interp[~active]=0
        return face,interp-compact,force,g


@dataclass(frozen=True)
class NativeSteadySettings:
    viscosity: float=1.
    density: float=1.
    max_iterations: int=80
    nonlinear_tolerance: float=1e-8
    linear_tolerance: float=1e-10
    relaxation: float=.8
    convection: bool=True
    linear_method: str='direct'
    convection_order: int=2
    bounded_convection: bool=False
    lu_ordering: str='COLAMD'
    momentum_pseudo_cfl: float | None=None
    pressure_relaxation: float=.2
    face_skew_correction: bool=True
    pressure_velocity_algorithm: str='simple'


class NativeSteadySolver3D:
    def __init__(self,mesh,velocity_dirichlet,velocity_boundary,pressure_dirichlet=None,pressure_boundary=0.,*,settings=None,body_force=None,boundary_flux=None,gauge_value=0.):
        self.mesh=mesh;self.settings=settings or NativeSteadySettings();c=self.settings
        if not np.isfinite([c.viscosity,c.density,c.relaxation,c.nonlinear_tolerance,c.linear_tolerance]).all() or c.viscosity<=0 or c.density<=0 or not 0<c.relaxation<=1 or c.max_iterations<1 or c.nonlinear_tolerance<=0 or c.linear_tolerance<=0:raise ValueError('positive finite material/solver settings required')
        if c.linear_method not in ('direct','iterative','block','schur'):raise ValueError('linear_method must be direct, iterative, block or schur')
        if c.pressure_velocity_algorithm not in ('simple','simplec'):
            raise ValueError('pressure_velocity_algorithm must be simple or simplec')
        if c.lu_ordering not in ('COLAMD','MMD_ATA','MMD_AT_PLUS_A','NATURAL'):
            raise ValueError('lu_ordering must be a supported SuperLU column ordering')
        if c.momentum_pseudo_cfl is not None and (not np.isfinite(c.momentum_pseudo_cfl) or c.momentum_pseudo_cfl<=0):
            raise ValueError('momentum pseudo CFL must be positive and finite')
        self.N=len(mesh.cells);self.F=len(mesh.owner);b=mesh.neighbor<0
        self.fixed=np.asarray(velocity_dirichlet,dtype=bool)
        self.viscous=NativeViscous3D(mesh,self.fixed,c.viscosity)
        self.velocity_boundary=np.broadcast_to(np.asarray(velocity_boundary,dtype=float),(self.F,3)).copy()
        if not np.isfinite(self.velocity_boundary).all():raise ValueError('finite boundary velocity required')
        pd=np.zeros(self.F,bool) if pressure_dirichlet is None else np.asarray(pressure_dirichlet,dtype=bool)
        self.pressure=FreePressureReconstruction3D(mesh,pd,skew_correction=c.face_skew_correction)
        self.pressure_boundary=pressure_boundary
        if not pd.any() and not np.all(self.fixed[b]):raise ValueError('unanchored mixed velocity pressure problem')
        self.gauge=not pd.any();self.gauge_value=float(gauge_value)
        if not np.isfinite(self.gauge_value):raise ValueError('finite pressure gauge required')
        force=np.zeros((self.N,3)) if body_force is None else np.asarray(body_force,dtype=float)
        if force.shape!=(self.N,3) or not np.isfinite(force).all():raise ValueError('finite acceleration at all cells required')
        self.force=force;cvol=mesh.cell_volumes
        self.force_integrated=(force*c.density*cvol[:,None]).ravel()
        self.diffusion_source=self.viscous.vector_divergence_matrix@self.viscous.momentum_flux_source(dirichlet_values=self.velocity_boundary).ravel()
        self.p_face_source,self.p_defect_source,self.p_force_source,_=self.pressure.sources(pressure_boundary)
        self.face_velocity_matrices=[];self.face_velocity_source=np.zeros((self.F,3))
        off=mesh.face_centers-mesh.cell_centers[mesh.owner]-self.viscous.weight[:,None]*self.viscous.op.d
        nb=self.viscous.neumann;off[nb]-=self.viscous.normal[nb]*(off[nb]*self.viscous.normal[nb]).sum(1)[:,None]
        if not c.face_skew_correction:off[:]=0.
        for component in range(3):
            source=self.viscous.gradient_source(dirichlet_values=self.velocity_boundary[:,component])
            vf=self.viscous.interpolation.copy()
            for k in range(3):vf+=sparse.diags(off[:,k])@self.viscous.interpolation@self.viscous.gradient_matrices[k]
            vf=(sparse.diags((~self.fixed).astype(float))@vf).tocsr();self.face_velocity_matrices.append(vf)
            bs=np.sum(off*(self.viscous.interpolation@source),axis=1);bs[self.fixed]=self.velocity_boundary[self.fixed,component];self.face_velocity_source[:,component]=bs
        velocity_flux=sparse.hstack([sparse.diags(mesh.area_vectors[:,k])@self.face_velocity_matrices[k] for k in range(3)],format='csr')
        perm=np.arange(3*self.N).reshape(3,self.N).T.ravel();self.velocity_flux_matrix=velocity_flux[:,perm]
        self.velocity_flux_source=np.einsum('ij,ij->i',self.face_velocity_source,mesh.area_vectors)
        if boundary_flux is not None:
            prescribed=np.asarray(boundary_flux,dtype=float)
            if prescribed.shape!=(self.F,) or not np.isfinite(prescribed).all():raise ValueError('finite integrated boundary flux on all native faces required')
            self.velocity_flux_source[self.fixed]=prescribed[self.fixed]
        if self.gauge and abs(self.velocity_flux_source[b].sum())>1e-10*max(np.abs(self.velocity_flux_source[b]).sum(),1):raise ValueError('all-Dirichlet velocity data have unbalanced total flux')
        self.mass_velocity_matrix=(self.viscous.divergence_matrix@self.velocity_flux_matrix).tocsr()
        self.u=np.zeros((self.N,3));self.p=np.full(self.N,self.gauge_value);self.flux=np.zeros(self.F);self.history=[]
        self._diffusion=(-self.viscous.momentum_diffusion_matrix).tocsr()
        # The full nonorthogonal stress matrix need not have positive diagonal.
        # Use its positive orthogonal scalar diffusion response as the Rhie--Chow
        # mobility reference; keep the full stress in the solved momentum.
        coefficient=self.viscous.face_viscosity*np.sum(mesh.area_vectors**2,axis=1)/np.sum(mesh.area_vectors*self.viscous.op.d,axis=1)
        self._mobility_reference=(-self.viscous.divergence_matrix@sparse.diags(coefficient)@self.viscous.delta_matrix).diagonal()*10/9
        self._last_matrix=None
        self._flux_initialized=False

    def initialize_state(self,velocity,pressure,flux):
        """Restore a numerical state, including its pressure-corrected face flux."""
        values=[np.asarray(value,dtype=float) for value in (velocity,pressure,flux)]
        if any(value.shape!=shape or not np.isfinite(value).all()
               for value,shape in zip(values,[(self.N,3),(self.N,),(self.F,)])):
            raise ValueError('finite native velocity, pressure and face flux shapes required')
        self.u,self.p,self.flux=[value.copy() for value in values]
        self._flux_initialized=True

    def _convection(self,flux):
        m=self.mesh;i=m.neighbor>=0;o=m.owner;n=m.neighbor;f=np.arange(self.F)
        q=np.asarray(flux)*self.settings.density
        # Conservative first-order upwind linearization; deferred high order is
        # added separately from an actual reconstructed velocity, never reference.
        upstream=np.where(q[i]>=0,o[i],n[i]);open_faces=(~i)&~self.fixed
        outflow=open_faces&(q>=0);backflow=open_faces&(q<0)
        # A pressure outlet can temporarily become an inlet. Its incoming
        # momentum uses the explicitly supplied ambient boundary velocity,
        # exactly as scalar transport uses incoming turbulence data. Using
        # the owner value here creates a negative convection diagonal.
        matrix=sparse.coo_matrix((np.r_[np.ones(i.sum()),np.ones(outflow.sum())],(np.r_[f[i],f[outflow]],np.r_[upstream,o[outflow]])),shape=(self.F,self.N)).tocsr()
        source=np.zeros((self.F,3));prescribed=self.fixed|backflow
        source[prescribed]=self.velocity_boundary[prescribed]
        a=self.viscous.divergence_matrix@sparse.diags(q)@matrix
        if self.settings.convection_order==2:
            gradient=np.stack([np.column_stack([g@self.u[:,axis] for g in self.viscous.gradient_matrices])+self.viscous.gradient_source(dirichlet_values=self.velocity_boundary[:,axis]) for axis in range(3)],axis=1)
            offset=m.face_centers[i]-m.cell_centers[upstream]
            correction=np.einsum('fij,fj->fi',gradient[upstream],offset)
            if self.settings.bounded_convection:
                lower=self.u.copy();upper=self.u.copy()
                np.minimum.at(lower,o[i],self.u[n[i]]);np.minimum.at(lower,n[i],self.u[o[i]])
                np.maximum.at(upper,o[i],self.u[n[i]]);np.maximum.at(upper,n[i],self.u[o[i]])
                np.minimum.at(lower,o[self.fixed],self.velocity_boundary[self.fixed]);np.maximum.at(upper,o[self.fixed],self.velocity_boundary[self.fixed])
                correction=np.clip(correction,lower[upstream]-self.u[upstream],upper[upstream]-self.u[upstream])
            source[i]=correction
        elif self.settings.convection_order!=1:raise ValueError('convection_order must be 1 or 2')
        rhs=self.viscous.op.integrate(q[:,None]*source).ravel()
        return sparse.kron(a,sparse.eye(3),format='csr'),rhs

    def _system(self,flux):
        c=self.settings;m=self.mesh
        convection,source=self._convection(flux) if c.convection else (sparse.csr_matrix((3*self.N,3*self.N)),np.zeros(3*self.N))
        a=self._diffusion+convection
        diagonal=self._mobility_reference+convection.diagonal().reshape(self.N,3).mean(1)
        if np.any(diagonal<=0):raise RuntimeError('nonpositive momentum diagonal; current mesh/operator is not admissible')
        mobility=m.cell_volumes/diagonal
        alpha=np.asarray(self.viscous.interpolation@mobility)
        # Fixed-velocity boundaries keep their prescribed mass flux exactly.
        alpha[self.fixed]=0
        correction=sparse.diags(alpha)@self.pressure.pressure_flux_defect
        sp=(self.viscous.divergence_matrix@correction).tocsr()
        source_q=self.velocity_flux_source+alpha*self.p_defect_source
        rhs_u=self.force_integrated+self.diffusion_source-self.p_force_source-source
        rhs_p=-self.viscous.divergence_matrix@source_q
        block=sparse.bmat([[a,self.pressure.pressure_force_matrix],[self.mass_velocity_matrix,sp]],format='csr')
        rhs=np.r_[rhs_u,rhs_p]
        if self.gauge:
            row=3*self.N;block=block.tolil();block[row,:]=0;block[row,3*self.N]=1;block=block.tocsr();rhs[row]=self.gauge_value
        return block,rhs,a,correction,source_q

    def _linear(self,block,rhs,a):
        linear_tolerance=min(self.settings.linear_tolerance,.01*self.settings.nonlinear_tolerance)
        scale=np.maximum(np.asarray(abs(block).sum(1)).ravel(),1e-30);scaled=(sparse.diags(1/scale)@block).tocsc();target=rhs/scale
        if self.settings.linear_method=='direct':
            answer=None;counter=[]
            if getattr(self,'_cached_lu',None) is not None:
                ratio=scale/self._cached_lu_scale
                pre=LinearOperator(scaled.shape,matvec=lambda x:self._cached_lu.solve(x*ratio))
                candidate,info=gmres(scaled,target,M=pre,rtol=linear_tolerance,atol=1e-14,restart=80,maxiter=12,callback=lambda r:counter.append(r),callback_type='pr_norm')
                true=np.linalg.norm(scaled@candidate-target)/max(np.linalg.norm(target),1e-30)
                if info==0 and true<max(5*linear_tolerance,1e-9):answer=candidate
            if answer is None:
                lu=splu(scaled,permc_spec=self.settings.lu_ordering);self._cached_lu=lu;self._cached_lu_scale=scale.copy();answer=lu.solve(target)
            else:lu=self._cached_lu
            for _ in range(3):
                residual=target-scaled@answer
                if np.linalg.norm(residual)<=max(np.linalg.norm(target)*linear_tolerance,1e-14):break
                # Reused LU is only a preconditioner for the current matrix.
                answer+=lu.solve(residual*(scale/self._cached_lu_scale))
            iterations=max(1,len(counter))
        elif self.settings.linear_method in ('block','schur'):
            # Factor the velocity and orthogonal pressure response separately.
            # This is only a preconditioner: the original coupled matrix and its
            # current true residual define the solution and convergence gate.
            nv=3*self.N
            av=scaled[:nv,:nv].tocsc();gp=scaled[:nv,nv:].tocsr();bv=scaled[nv:,:nv].tocsr()
            convection_diagonal=(a-self._diffusion).diagonal().reshape(self.N,3).mean(1)
            mobility=self.mesh.cell_volumes/(self._mobility_reference+convection_diagonal)
            alpha=np.asarray(self.viscous.interpolation@mobility);alpha[self.fixed]=0
            pressure_approx=-self.viscous.divergence_matrix@sparse.diags(alpha*self.pressure.coefficient)@self.pressure.base.delta_matrix
            if self.gauge:
                pressure_approx=pressure_approx.tolil();pressure_approx[0,:]=0;pressure_approx[0,0]=1;pressure_approx=pressure_approx.tocsr()
            pressure_approx=(sparse.diags(1/scale[nv:])@pressure_approx).tocsc()
            if self.settings.linear_method=='schur':
                # Exact momentum solve plus a sparse SIMPLE Schur approximation.
                # Use the actual mass/pressure operators in this approximation;
                # triangulated thin layers can invalidate an orthogonal-only
                # pressure preconditioner even while the coupled system solves.
                response=np.repeat(1/(self._mobility_reference+convection_diagonal),3)*scale[:nv]
                pressure_approx=(scaled[nv:,nv:]-bv@sparse.diags(response)@gp).tocsc()
                ilu_v=splu(av,permc_spec=self.settings.lu_ordering);ilu_p=splu(pressure_approx,permc_spec=self.settings.lu_ordering)
            else:
                ilu_v=spilu(av,drop_tol=1e-4,fill_factor=20)
                ilu_p=spilu(pressure_approx,drop_tol=1e-4,fill_factor=20)
            def apply_preconditioner(value):
                velocity=ilu_v.solve(value[:nv]);pressure=ilu_p.solve(value[nv:]-bv@velocity)
                velocity-=ilu_v.solve(gp@pressure)
                return np.r_[velocity,pressure]
            pre=LinearOperator(scaled.shape,matvec=apply_preconditioner);counter=[]
            answer,info=gmres(scaled,target,M=pre,rtol=linear_tolerance,atol=1e-13,restart=100,maxiter=60,callback=lambda r:counter.append(r),callback_type='pr_norm')
            if info!=0:raise RuntimeError(f'current coupled block GMRES failed: {info}, {len(counter)} iterations')
            iterations=len(counter)
        else:
            ilu=spilu(scaled,drop_tol=1e-4,fill_factor=12)
            pre=LinearOperator(scaled.shape,matvec=ilu.solve);counter=[]
            answer,info=gmres(scaled,target,M=pre,rtol=linear_tolerance,atol=1e-13,restart=100,maxiter=60,callback=lambda r:counter.append(r),callback_type='pr_norm')
            if info!=0:raise RuntimeError(f'current coupled GMRES failed: {info}, {len(counter)} iterations')
            iterations=len(counter)
        residual=np.linalg.norm(scaled@answer-target)/max(np.linalg.norm(target),1e-30)
        if not np.isfinite(answer).all() or residual>max(5*linear_tolerance,1e-9):raise RuntimeError(f'current coupled true linear residual {residual}')
        return answer,residual,iterations

    def solve(self,callback=None):
        c=self.settings
        # Boundary sources are included before the first nonlinear linearization.
        if not self._flux_initialized:
            self.flux=self.velocity_flux_matrix@self.u.ravel()+self.velocity_flux_source
            self._flux_initialized=True
        converged=False
        for iteration in range(c.max_iterations):
            block,rhs,a,correction,qsource=self._system(self.flux)
            if c.momentum_pseudo_cfl is not None:
                # Consistent local pseudo time: include mass in the implicit
                # RC mobility, then defer the difference to OLD pressure.
                # At a fixed point this exactly recovers the original steady
                # face flux, unlike simply replacing its mobility with dt's.
                rate=self._mobility_reference+.5*c.density*self.viscous.op.integrate(np.abs(self.flux),signed=False)
                mass_cell=rate/c.momentum_pseudo_cfl;mass=np.repeat(mass_cell,3)
                diagonal=self._mobility_reference+(a-self._diffusion).diagonal().reshape(self.N,3).mean(1)
                alpha=np.asarray(self.viscous.interpolation@(self.mesh.cell_volumes/(diagonal+mass_cell)))
                alpha[self.fixed]=0
                transient_correction=sparse.diags(alpha)@self.pressure.pressure_flux_defect
                transient_source=qsource+(correction-transient_correction)@self.p
                # Boundary pressure source is already included in qsource;
                # it is unchanged at the fixed point and needs no lagged value.
                correction=transient_correction;qsource=np.asarray(transient_source)
                rhs[:3*self.N]+=mass*self.u.ravel()
                a=a+sparse.diags(mass)
                block=sparse.bmat([[a,self.pressure.pressure_force_matrix],
                    [self.mass_velocity_matrix,self.viscous.divergence_matrix@correction]],format='csr')
                rhs[3*self.N:]=-self.viscous.divergence_matrix@qsource
                if self.gauge:
                    row=3*self.N;block=block.tolil();block[row,:]=0;block[row,row]=1
                    block=block.tocsr();rhs[row]=self.gauge_value
            answer,linear,li=self._linear(block,rhs,a)
            newu=answer[:3*self.N].reshape(self.N,3);newp=answer[3*self.N:]
            newq=self.velocity_flux_matrix@newu.ravel()+correction@newp+qsource
            omega=c.relaxation if iteration or c.momentum_pseudo_cfl is not None else 1.
            u=(1-omega)*self.u+omega*newu;p=(1-omega)*self.p+omega*newp
            q=(1-omega)*self.flux+omega*newq
            change=float(np.linalg.norm(u-self.u)/max(np.linalg.norm(u),1e-30))
            pressure_change=float(np.linalg.norm(p-self.p)/max(np.linalg.norm(p),1.))
            flux_change=float(np.linalg.norm(q-self.flux)/max(np.linalg.norm(q),1e-30))
            self.u,self.p,self.flux=u,p,q
            current,target,_,current_correction,current_source=self._system(q)
            consistent_q=self.velocity_flux_matrix@u.ravel()+current_correction@p+current_source
            consistent_divergence=self.viscous.op.integrate(consistent_q)/self.mesh.cell_volumes
            coupling=float(np.max(np.abs(q-consistent_q)/np.linalg.norm(self.mesh.area_vectors,axis=1)))
            current_continuity=float(np.max(np.abs(consistent_divergence)))
            residual=current@np.r_[u.ravel(),p]-target
            momentum=float(np.linalg.norm(residual[:3*self.N])/max(np.linalg.norm(self.force_integrated)+np.linalg.norm(self.diffusion_source)+np.linalg.norm(self.p_force_source),1))
            divergence=self.viscous.op.integrate(q)/self.mesh.cell_volumes
            continuity=float(np.abs(divergence).max())
            row=dict(iteration=iteration+1,velocity_change=change,pressure_change=pressure_change,flux_change=flux_change,momentum_residual=momentum,native_divergence_max=continuity,current_reconstructed_divergence_max=current_continuity,face_flux_consistency_max=coupling,true_linear_relative_residual=float(linear),linear_iterations=li)
            self.history.append(row)
            if callback is not None:callback(row)
            if max(max(change,pressure_change,flux_change) if c.convection else 0.,momentum,continuity,current_continuity,coupling)<c.nonlinear_tolerance:
                converged=True;break
        self.converged=converged;return dict(converged=converged,history=self.history,velocity=self.u.copy(),pressure=self.p.copy(),flux=self.flux.copy())

    def forces(self,patch='wall'):
        wall=self.mesh.patch_labels==patch
        pface=self.pressure.face_matrix@self.p+self.p_face_source
        viscous=self.viscous.momentum_flux(self.u,dirichlet_values=self.velocity_boundary)
        # Opposite of outward fluid-domain traction: force on the solid.
        pressure_force=(pface[wall,None]*self.mesh.area_vectors[wall]).sum(0)
        viscous_force=-viscous[wall].sum(0)
        return dict(pressure_force=pressure_force,viscous_force=viscous_force,total_force=pressure_force+viscous_force,pressure_faces=pface,viscous_traction_integrated=viscous)
