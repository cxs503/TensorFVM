"""Quadratic shared-face corrections for the native steady FVM equations.

The reference sparse system preconditions the legacy corrected iteration.
An optional full sparse operator solves the complete frozen-flux equations.
Convergence replays the current corrected momentum and mass equations. No
analytic interior field or reverse-discrete manufactured force is used.
"""
import numpy as np
import time
from scipy import sparse
from scipy.sparse.linalg import splu, gmres, LinearOperator
from .native_collocated3d import NativeSteadySolver3D
from .native_quadratic3d import NativeQuadratic3D
from .native_quadratic_system3d import current_matrix


class NativeQuadraticSteadySolver3D(NativeSteadySolver3D):
    def __init__(self, *args, boundary_velocity_means=None, boundary_velocity_gradients=None, quadratic_linear_solver='matrix-free', linear_callback=None, **kwargs):
        super().__init__(*args, **kwargs)
        if quadratic_linear_solver not in ('matrix-free','assembled'):
            raise ValueError('Quadratic linear solver must be matrix-free or assembled')
        self.quadratic_linear_solver=quadratic_linear_solver
        self.linear_callback=linear_callback
        if not np.all(self.fixed[self.mesh.neighbor<0]):
            raise ValueError('Quadratic flow currently requires prescribed velocity on every boundary; mixed outlet not validated')
        if np.any(self.pressure.dirichlet):
            raise ValueError('All-velocity-Dirichlet quadratic flow requires a pressure gauge, not pressure Dirichlet faces')
        self.boundary_velocity_gradients = (None if boundary_velocity_gradients is None
            else np.broadcast_to(np.asarray(boundary_velocity_gradients,float),(self.F,3,3)).copy())
        if self.boundary_velocity_gradients is not None and not np.isfinite(self.boundary_velocity_gradients).all():
            raise ValueError('Finite prescribed boundary velocity derivatives required')
        if self.settings.bounded_convection or self.settings.convection_order != 2:
            raise ValueError('Quadratic implicit action requires unbounded order-2 reconstruction; bounded Newton linearization not implemented')
        if self.settings.linear_method != 'direct':
            raise ValueError('Quadratic flow currently requires direct sparse LU/preconditioned GMRES')
        self.quadratic_velocity = NativeQuadratic3D(self.mesh, self.fixed)
        self.quadratic_pressure = NativeQuadratic3D(self.mesh, self.pressure.dirichlet)
        self.boundary_velocity_means = (None if boundary_velocity_means is None
            else np.broadcast_to(np.asarray(boundary_velocity_means,float),(self.F,3)).copy())
        if self.boundary_velocity_means is not None and not np.isfinite(self.boundary_velocity_means).all():
            raise ValueError('Finite prescribed boundary velocity means required')

    def quadratic_pressure_faces(self):
        return self.quadratic_pressure.face_mean(self.p, self.pressure_boundary,
            boundary_face_means=self.pressure_boundary)

    def quadratic_velocity_faces(self):
        return self.quadratic_velocity.face_mean(self.u,self.velocity_boundary,
            boundary_face_means=self.boundary_velocity_means)

    def quadratic_traction(self):
        g = self.quadratic_velocity.face_gradient(self.u,self.velocity_boundary)
        nb = self.viscous.neumann
        normals = self.viscous.normal[nb]
        g[nb] -= np.einsum('fki,fi->fk',g[nb],normals)[:,:,None]*normals[:,None,:]
        tau = g+g.transpose(0,2,1)
        if self.viscous.deviatoric:
            tau -= (2/3)*np.trace(g,axis1=1,axis2=2)[:,None,None]*np.eye(3)
        return self.viscous.face_viscosity[:,None]*np.einsum('fij,fj->fi',tau,self.mesh.area_vectors)

    def quadratic_pressure_defect(self, pmean):
        m = self.mesh; op = self.quadratic_pressure
        gauss = self.viscous.op.integrate(pmean[:,None]*m.area_vectors)/m.cell_volumes[:,None]
        face = op.face_gradient(self.p,self.pressure_boundary,compact=False,cell_gradients=gauss)
        compact = op.compact_normal_flux(self.p,self.pressure_boundary,cell_gradients=gauss)
        result = np.einsum('fi,fi->f',face,m.area_vectors)-compact
        result[~(op.internal|op.dirichlet)] = 0
        return result

    def _quadratic_convection_difference(self, flux):
        """Shared high-order advected mean; no separately advanced face flux.

        Face covariance accounts for linear fluctuations of advecting and
        transported velocities. Its divergence is assembled conservatively.
        This is a second-order covariance approximation for nonlinear fields.
        """
        if not self.settings.convection:
            return np.zeros(3*self.N)
        m=self.mesh; i=m.neighbor>=0; n=m.neighbor; o=m.owner
        q=np.asarray(flux)*self.settings.density
        up=np.where(q[i]>=0,o[i],n[i]); offset=m.face_centers[i]-m.cell_centers[up]
        g=self.quadratic_velocity.gradient(self.u,self.velocity_boundary)
        h=self.quadratic_velocity.hessian(self.u,self.velocity_boundary)
        moment=offset[:,:,None]*offset[:,None,:]+self.quadratic_velocity.face_covariance[i]
        higher=np.zeros((self.F,3))
        higher[i]=np.einsum('fki,fi->fk',g[up],offset)+.5*np.einsum('fkij,fij->fk',h[up],moment)
        if self.settings.bounded_convection:
            lower=self.u.copy(); upper=self.u.copy()
            np.minimum.at(lower,o[i],self.u[n[i]]);np.minimum.at(lower,n[i],self.u[o[i]])
            np.maximum.at(upper,o[i],self.u[n[i]]);np.maximum.at(upper,n[i],self.u[o[i]])
            np.minimum.at(lower,o[self.fixed],self.velocity_boundary[self.fixed]);np.maximum.at(upper,o[self.fixed],self.velocity_boundary[self.fixed])
            higher[i]=np.clip(higher[i],lower[up]-self.u[up],upper[up]-self.u[up])
        # The existing matrix uses upstream cell values. Compare deferred RHS
        # terms, so only these reconstructed additions belong in the difference.
        if self.boundary_velocity_means is not None:
            higher[self.fixed]=self.boundary_velocity_means[self.fixed]
        else:
            higher[self.fixed]=self.quadratic_velocity_faces()[self.fixed]
        new=self.viscous.op.integrate(q[:,None]*higher).ravel()
        _,old=self._convection(flux)
        # Conservative linear-fluctuation integral; actual q remains the mean
        # advecting flux, and the added fluctuation has zero mean.
        if hasattr(self,'_frozen_covariance_divergence'):
            new+=self._frozen_covariance_divergence
            return new-old
        new+=self._covariance_divergence()
        return new-old

    def _covariance_divergence(self):
        m=self.mesh;i=m.neighbor>=0;o=m.owner;n=m.neighbor
        up=np.where(self._linear_flux[i]>=0,o[i],n[i])
        offset=m.face_centers[i]-m.cell_centers[up]
        g=self.quadratic_velocity.gradient(self.u,self.velocity_boundary)
        h=self.quadratic_velocity.hessian(self.u,self.velocity_boundary)
        gf=self.quadratic_velocity.face_gradient(self.u,self.velocity_boundary,compact=False)
        transported=gf.copy()
        transported[i]=g[up]+np.einsum('fkij,fj->fki',h[up],offset)
        cov=np.einsum('fki,fij,flj,fl->fk',transported,
            self.quadratic_velocity.face_covariance,gf,m.area_vectors)*self.settings.density
        cov[self.fixed]=0
        if self.boundary_velocity_gradients is not None:
            gb=self.boundary_velocity_gradients[self.fixed]
            cov[self.fixed]=np.einsum('fki,fij,flj,fl->fk',gb,
                self.quadratic_velocity.face_covariance[self.fixed],gb,m.area_vectors[self.fixed])*self.settings.density
        return self.viscous.op.integrate(cov).ravel()

    def _corrections(self,flux,a):
        m=self.mesh; D=self.viscous.divergence_matrix
        self._linear_flux=flux
        extra_rhs=np.zeros(4*self.N)
        pmean=self.quadratic_pressure_faces()
        pbase=self.pressure.face_matrix@self.p+self.p_face_source
        pressure_difference=self.viscous.op.integrate((pmean-pbase)[:,None]*m.area_vectors).ravel()
        stress_difference=self.viscous.op.integrate(self.quadratic_traction()-self.viscous.momentum_flux(self.u,dirichlet_values=self.velocity_boundary)).ravel()
        extra_rhs[:3*self.N]=stress_difference-pressure_difference-self._quadratic_convection_difference(flux)
        velocity_mean=self.quadratic_velocity_faces()
        delta_q=np.einsum('fi,fi->f',velocity_mean,m.area_vectors)-(self.velocity_flux_matrix@self.u.ravel()+self.velocity_flux_source)
        delta_q[self.fixed]=0
        diagonal=self._mobility_reference+(a-self._diffusion).diagonal().reshape(self.N,3).mean(1)
        alpha=np.asarray(self.viscous.interpolation@(m.cell_volumes/diagonal));alpha[self.fixed]=0
        delta_q+=alpha*(self.quadratic_pressure_defect(pmean)-(self.pressure.pressure_flux_defect@self.p+self.p_defect_source))
        extra_rhs[3*self.N:]=-D@delta_q
        if self.gauge:extra_rhs[3*self.N]=0
        return extra_rhs,delta_q

    def _system(self,flux):
        block,rhs,a,correction,qsource=super()._system(flux)
        extra,delta_q=self._corrections(flux,a)
        self._pending_flux=flux.copy();self._pending_base_qsource=qsource.copy()
        qsource=qsource+delta_q
        self._pending_qsource=qsource
        return block,rhs+extra,a,correction,qsource

    def _linear(self,block,rhs,a):
        """Dispatch the complete assembled or legacy matrix-free iteration.

        The legacy correction action freezes the baseline high-order advection
        source and the nonlinear covariance. Its baseline LU preconditions the
        correction iteration. Final acceptance replays the complete equations.
        """
        if self.quadratic_linear_solver=='assembled':return self._linear_assembled(a)
        old_u,old_p=self.u,self.p
        flux=self._pending_flux.copy()
        pending_source=self._pending_qsource
        base_source=self._pending_base_qsource.copy()
        self._linear_flux=flux
        covariance=self._covariance_divergence() if self.settings.convection else np.zeros(3*self.N)
        self._frozen_covariance_divergence=covariance
        try:
            extra,_=self._corrections(flux,a)
            base_rhs=rhs-extra
            self.u=np.zeros_like(old_u);self.p=np.zeros_like(old_p)
            zero,_=self._corrections(flux,a)
            scale=np.maximum(np.asarray(abs(block).sum(1)).ravel(),1e-30)
            target=(base_rhs+zero)/scale
            scaled=(sparse.diags(1/scale)@block).tocsc()
            if getattr(self,'_quadratic_preconditioner',None) is None:
                self._quadratic_preconditioner=splu(scaled,permc_spec=self.settings.lu_ordering)
                self._quadratic_preconditioner_scale=scale.copy()
            pre=LinearOperator(block.shape,matvec=lambda x:self._quadratic_preconditioner.solve(x*scale/self._quadratic_preconditioner_scale))
            def action(x):
                self.u=x[:3*self.N].reshape(self.N,3);self.p=x[3*self.N:]
                effect,_=self._corrections(flux,a)
                return (block@x-effect+zero)/scale
            operator=LinearOperator(block.shape,matvec=action,dtype=float)
            tolerance=min(self.settings.linear_tolerance,.01*self.settings.nonlinear_tolerance)
            counter=[]
            answer,info=gmres(operator,target,M=pre,x0=np.r_[old_u.ravel(),old_p],
                rtol=tolerance,atol=1e-14,restart=60,maxiter=30,
                callback=lambda r:counter.append(r),callback_type='pr_norm')
            true=float(np.linalg.norm(action(answer)-target)/max(np.linalg.norm(target),1e-30))
            if info!=0 or not np.isfinite(answer).all() or true>max(5*tolerance,1e-9):
                raise RuntimeError(f'Quadratic current-equation GMRES failed: info={info}, true={true}, iterations={len(counter)}')
            self.u=answer[:3*self.N].reshape(self.N,3);self.p=answer[3*self.N:]
            _,new_delta=self._corrections(flux,a)
            pending_source[:]=base_source+new_delta
            return answer,true,len(counter)
        finally:
            self.u,self.p=old_u,old_p
            del self._frozen_covariance_divergence

    def _linear_assembled(self,a):
        """Solve the complete frozen-flux operator, including all advection.

        The legacy action lags a baseline high-order convection source. Here
        the full current quadratic operator supplies both the LU and residual.
        """
        old_u,old_p=self.u,self.p
        flux=self._pending_flux.copy();pending_source=self._pending_qsource
        base_source=self._pending_base_qsource.copy();self._linear_flux=flux
        self._frozen_covariance_divergence=self._covariance_divergence() if self.settings.convection else np.zeros(3*self.N)
        try:
            started=time.perf_counter()
            matrix=current_matrix(self,flux,a)
            if self.linear_callback:self.linear_callback(dict(stage='assembled_matrix_ready',unknowns=4*self.N,nnz=matrix.nnz,assembly_seconds=time.perf_counter()-started))
            self.u=np.zeros_like(old_u);self.p=np.zeros_like(old_p)
            _,base_rhs,_,_,_=NativeSteadySolver3D._system(self,flux)
            zero,_=self._corrections(flux,a);target=base_rhs+zero
            answer,true,iterations=NativeSteadySolver3D._linear(self,matrix,target,a)
            self.u=answer[:3*self.N].reshape(self.N,3);self.p=answer[3*self.N:]
            _,delta=self._corrections(flux,a);pending_source[:]=base_source+delta
            self._last_quadratic_linear=dict(method='assembled full quadratic sparse operator',matrix_nnz=matrix.nnz,true_residual=true,iterations=iterations,elapsed_seconds=time.perf_counter()-started)
            if self.linear_callback:self.linear_callback(dict(stage='linear_solve_completed',**self._last_quadratic_linear))
            return answer,true,iterations
        finally:
            self.u,self.p=old_u,old_p
            del self._frozen_covariance_divergence

    def _assess_state(self,u,p,q):
        self.u,self.p,self.flux=u,p,q
        block,rhs,a,correction,source=self._system(q)
        consistent=self.velocity_flux_matrix@u.ravel()+correction@p+source
        residual=block@np.r_[u.ravel(),p]-rhs
        momentum=float(np.linalg.norm(residual[:3*self.N])/max(np.linalg.norm(self.force_integrated)+np.linalg.norm(self.diffusion_source)+np.linalg.norm(self.p_force_source),1))
        divergence=self.viscous.op.integrate(q)/self.mesh.cell_volumes
        current_divergence=self.viscous.op.integrate(consistent)/self.mesh.cell_volumes
        coupling=(q-consistent)/np.linalg.norm(self.mesh.area_vectors,axis=1)
        merit=float(np.sqrt(momentum**2+np.sum(divergence**2*self.mesh.cell_volumes)/self.mesh.cell_volumes.sum()+np.mean(coupling**2)))
        metrics=dict(momentum_residual=momentum,native_divergence_max=float(abs(divergence).max()),
            current_reconstructed_divergence_max=float(abs(current_divergence).max()),
            face_flux_consistency_max=float(abs(coupling).max()))
        return block,rhs,a,correction,source,merit,metrics

    def solve(self,callback=None):
        """Backtrack on the actual corrected residual; never relax the gates.

        No automatic analytic or numerical warm start is inserted here. A coarse
        nonlinear case may need an explicitly recorded baseline initialization.
        A direction with no decreasing trial is rejected and its old state kept.
        """
        c=self.settings;self.flux=self.velocity_flux_matrix@self.u.ravel()+self.velocity_flux_source
        converged=False
        for iteration in range(c.max_iterations):
            old_u,old_p,old_q=self.u.copy(),self.p.copy(),self.flux.copy()
            block,rhs,a,correction,source,merit,old_metrics=self._assess_state(old_u,old_p,old_q)
            answer,linear,li=self._linear(block,rhs,a)
            new_u=answer[:3*self.N].reshape(self.N,3);new_p=answer[3*self.N:]
            new_q=self.velocity_flux_matrix@new_u.ravel()+correction@new_p+source
            trials=[];accepted=False
            for backtrack in range(16):
                first_trial=1. if iteration==0 and not np.any(old_u) else c.relaxation
                omega=first_trial*2**(-backtrack)
                u=old_u+omega*(new_u-old_u);p=old_p+omega*(new_p-old_p);q=old_q+omega*(new_q-old_q)
                *_,trial_merit,metrics=self._assess_state(u,p,q)
                trials.append(dict(relaxation=omega,merit=trial_merit))
                trial_changes=(float(np.linalg.norm(u-old_u)/max(np.linalg.norm(u),1e-30)),
                    float(np.linalg.norm(p-old_p)/max(np.linalg.norm(p),1.)),
                    float(np.linalg.norm(q-old_q)/max(np.linalg.norm(q),1e-30)))
                trial_converged=max(*trial_changes,*metrics.values())<c.nonlinear_tolerance
                # At roundoff, a certified solution need not reduce an already
                # tiny merit. Every original residual/change gate still applies.
                if np.isfinite(trial_merit) and (trial_converged or trial_merit<=merit*(1-1e-4*omega)):
                    accepted=True;break
            if not accepted:
                self.u,self.p,self.flux=old_u,old_p,old_q
                row=dict(iteration=iteration+1,accepted=False,old_merit=merit,trials=trials,**old_metrics)
                self.history.append(row)
                if callback is not None:callback(row)
                self.converged=False
                raise RuntimeError('Quadratic nonlinear step rejected: no residual decrease in 16 trials; previous state preserved')
            changes=dict(velocity_change=float(np.linalg.norm(u-old_u)/max(np.linalg.norm(u),1e-30)),
                pressure_change=float(np.linalg.norm(p-old_p)/max(np.linalg.norm(p),1.)),
                flux_change=float(np.linalg.norm(q-old_q)/max(np.linalg.norm(q),1e-30)))
            row=dict(iteration=iteration+1,accepted=True,accepted_relaxation=omega,
                old_merit=merit,merit=trial_merit,trials=trials,
                true_linear_relative_residual=float(linear),linear_iterations=li,**changes,**metrics)
            if self.quadratic_linear_solver=='assembled':row['linear_operator']=self._last_quadratic_linear.copy()
            self.history.append(row)
            if callback is not None:callback(row)
            if max(*changes.values(),*metrics.values())<c.nonlinear_tolerance:
                converged=True;break
        self.converged=converged
        return dict(converged=converged,history=self.history,velocity=self.u.copy(),pressure=self.p.copy(),flux=self.flux.copy())

    def forces(self,patch='wall'):
        wall=self.mesh.patch_labels==patch
        pressure=self.quadratic_pressure_faces();traction=self.quadratic_traction()
        pf=(pressure[wall,None]*self.mesh.area_vectors[wall]).sum(0)
        vf=-traction[wall].sum(0)
        return dict(pressure_force=pf,viscous_force=vf,total_force=pf+vf,
            pressure_faces=pressure,viscous_traction_integrated=traction)
