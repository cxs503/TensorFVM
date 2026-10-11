"""Segregated SIMPLE on shared native 3D faces (CPU reference).

Momentum uses positive orthogonal diffusion implicitly and defers the remaining
symmetric stress. A compact pressure correction enforces conservative face flux.
Convergence includes the full current momentum and Rhie--Chow flux equations.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from .native_collocated3d import NativeSteadySolver3D


class NativeSimpleMixin:
    def _simple_momentum(self, flux):
        c=self.settings;v=self.viscous;m=self.mesh
        conv,source=self._convection(flux) if c.convection else (sparse.csr_matrix((3*self.N,3*self.N)),np.zeros(3*self.N))
        coefficient=v.face_viscosity*np.sum(m.area_vectors**2,axis=1)/np.sum(m.area_vectors*v.op.d,axis=1)
        orth=-v.divergence_matrix@sparse.diags(coefficient)@v.delta_matrix
        implicit=sparse.kron(orth,sparse.eye(3),format='csr')+conv
        diagonal=implicit.diagonal()
        if np.any(diagonal<=0):raise RuntimeError('nonpositive SIMPLE momentum diagonal')
        relaxed=diagonal/c.relaxation
        inverse=1/relaxed
        alpha=np.asarray(v.interpolation@(m.cell_volumes*inverse.reshape(self.N,3).mean(1)))
        alpha[self.fixed]=0.
        full=self._diffusion+conv
        rhs=self.force_integrated+self.diffusion_source-self.p_force_source-source
        return implicit,full,rhs,inverse,alpha,relaxed-diagonal

    def _system(self,flux):
        _,a,rhs_u,_,alpha,_=self._simple_momentum(flux)
        correction=sparse.diags(alpha)@self.pressure.pressure_flux_defect
        qsource=self.velocity_flux_source+alpha*self.p_defect_source
        block=sparse.bmat([[a,self.pressure.pressure_force_matrix],
            [self.mass_velocity_matrix,self.viscous.divergence_matrix@correction]],format='csr')
        rhs=np.r_[rhs_u,-self.viscous.divergence_matrix@qsource]
        if self.gauge:
            block=block.tolil();block[3*self.N,:]=0;block[3*self.N,3*self.N]=1
            block=block.tocsr();rhs[3*self.N]=self.gauge_value
        return block,rhs,a,correction,qsource

    def solve(self,callback=None):
        c=self.settings
        if c.momentum_pseudo_cfl is not None:raise ValueError('SIMPLE uses equation relaxation; momentum_pseudo_cfl is unsupported')
        if not 0<c.pressure_relaxation<=1:raise ValueError('pressure relaxation must lie in (0,1]')
        if not self._flux_initialized:
            self.flux=self.velocity_flux_matrix@self.u.ravel()+self.velocity_flux_source
            self._flux_initialized=True
        converged=False
        for iteration in range(c.max_iterations):
            old_u=self.u.copy();old_p=self.p.copy();old_q=self.flux.copy()
            implicit,full,rhs,inverse,alpha,extra=self._simple_momentum(self.flux)
            A=(implicit+sparse.diags(extra)).tocsc()
            target=rhs-self.pressure.pressure_force_matrix@self.p+(implicit-full)@self.u.ravel()+extra*self.u.ravel()
            predicted=splu(A,permc_spec=c.lu_ordering).solve(target)
            correction=sparse.diags(alpha)@self.pressure.pressure_flux_defect
            qstar=self.velocity_flux_matrix@predicted+correction@self.p+self.velocity_flux_source+alpha*self.p_defect_source
            response=-sparse.diags(alpha)@self.pressure.compact_flux
            P=(self.viscous.divergence_matrix@response).tolil()
            prhs=-self.viscous.divergence_matrix@qstar
            if self.gauge:P[0,:]=0;P[0,0]=1;prhs[0]=self.gauge_value-self.p[0]
            P=P.tocsc();dp=splu(P,permc_spec=c.lu_ordering).solve(prhs)
            self.u=(predicted-inverse*(self.pressure.pressure_force_matrix@dp)).reshape(self.N,3)
            self.p+=c.pressure_relaxation*dp
            if self.gauge:self.p+=self.gauge_value-self.p[0]
            self.flux=qstar+response@dp
            if not all(np.isfinite(x).all() for x in (self.u,self.p,self.flux)):raise RuntimeError('nonfinite SIMPLE state')
            block,rhs,_,corr,src=self._system(self.flux)
            consistent=self.velocity_flux_matrix@self.u.ravel()+corr@self.p+src
            residual=(block@np.r_[self.u.ravel(),self.p]-rhs)[:3*self.N]
            norm=lambda x:float(np.linalg.norm(x))
            row=dict(iteration=iteration+1,algorithm='SIMPLE',
                velocity_change=norm(self.u-old_u)/max(norm(self.u),1e-30),
                pressure_change=norm(self.p-old_p)/max(norm(self.p),1.),
                flux_change=norm(self.flux-old_q)/max(norm(self.flux),1e-30),
                momentum_residual=norm(residual)/max(norm(self.force_integrated)+norm(self.diffusion_source)+norm(self.p_force_source),1.),
                native_divergence_max=float(np.abs(self.viscous.op.integrate(self.flux)/self.mesh.cell_volumes).max()),
                current_reconstructed_divergence_max=float(np.abs(self.viscous.op.integrate(consistent)/self.mesh.cell_volumes).max()),
                face_flux_consistency_max=float(np.max(np.abs(consistent-self.flux)/np.linalg.norm(self.mesh.area_vectors,axis=1))),
                true_linear_relative_residual=max(norm(A@predicted-target)/max(norm(target),1e-30),norm(P@dp-prhs)/max(norm(prhs),1e-30)),linear_iterations=2)
            self.history.append(row)
            if callback:callback(row)
            if max(row[key] for key in ('velocity_change','pressure_change','flux_change','momentum_residual','native_divergence_max','current_reconstructed_divergence_max','face_flux_consistency_max'))<c.nonlinear_tolerance:
                converged=True;break
        self.converged=converged
        return dict(converged=converged,history=self.history,velocity=self.u.copy(),pressure=self.p.copy(),flux=self.flux.copy())


class NativeSimpleSolver3D(NativeSimpleMixin,NativeSteadySolver3D):
    pass
