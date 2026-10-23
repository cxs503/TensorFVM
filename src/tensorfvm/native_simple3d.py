"""Segregated SIMPLE/SIMPLEC on shared native 3D faces (CPU reference).

Momentum uses positive orthogonal diffusion implicitly and defers the remaining
symmetric stress. A compact pressure correction enforces conservative face flux.
Convergence includes the full current momentum and Rhie--Chow flux equations.

Accepted states obey the face-flux ledger

    flux == velocity_flux_matrix @ u + correction @ p + qsource

exactly at every sweep. Momentum under-relaxation scales the predictor only,
and ``pressure_relaxation`` is a uniform damping of the whole consistent step,
so neither parameter enters the discrete continuity operator or the converged
solution.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from .native_collocated3d import NativeSteadySolver3D
from .native_linear_diagnostics import linear_system_diagnostics


class NativeSimpleMixin:
    def _simple_momentum(self, flux):
        c=self.settings;v=self.viscous;m=self.mesh
        conv,source=self._convection(flux) if c.convection else (sparse.csr_matrix((3*self.N,3*self.N)),np.zeros(3*self.N))
        coefficient=v.face_viscosity*np.sum(m.area_vectors**2,axis=1)/np.sum(m.area_vectors*v.op.d,axis=1)
        orth=-v.divergence_matrix@sparse.diags(coefficient)@v.delta_matrix
        implicit=sparse.kron(orth,sparse.eye(3),format='csr')+conv
        if hasattr(self,"wall_implicit_correction"):implicit=implicit+self.wall_implicit_correction
        diagonal=implicit.diagonal()
        if np.any(diagonal<=0):raise RuntimeError('nonpositive SIMPLE momentum diagonal')
        relaxed=diagonal/c.relaxation
        # The velocity channel and the Rhie--Chow defect mobility share one
        # pressure response (inverse).  Because that response also defines the
        # discrete continuity operator flux = B u + diag(alpha) defect p +
        # qsource, it must not depend on the under-relaxation factors or on the
        # pressure/velocity algorithm: otherwise the converged solution would
        # drift with purely iterative parameters.  Momentum under-relaxation is
        # therefore applied to the predictor only (through `extra` below) and as
        # a uniform damping of the whole consistent step in solve(); both leave
        # the fixed point and the face-flux ledger untouched.
        inverse=1/diagonal
        self.simplec_limited_dofs=0
        if c.pressure_velocity_algorithm=='simplec':
            # aP*u' + sum(aN*uN') = -G*p'. SIMPLEC approximates same-
            # component neighbour corrections by u', giving aP+sum(aN).
            # Other velocity components are not equated to this component;
            # their implicit wall couplings remain in the momentum solve.
            entries=implicit.tocoo()
            mask=(entries.row!=entries.col)&(entries.row%3==entries.col%3)
            neighbor_sum=np.bincount(entries.row[mask],weights=entries.data[mask],minlength=3*self.N)
            denominator=diagonal+neighbor_sum
            floor=.1*diagonal
            self.simplec_limited_dofs=int(np.count_nonzero(denominator<floor))
            inverse=1/np.maximum(denominator,floor)
        # This is not a port of the whole OpenFOAM rAtU/HbyA/face-interpolation
        # discretization.
        alpha=np.asarray(v.interpolation@(m.cell_volumes*(1/diagonal).reshape(self.N,3).mean(1)))
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
        if c.linear_method not in ('direct','amg'):raise ValueError('Native SIMPLE supports direct or amg segregated linear solves')
        if c.momentum_pseudo_cfl is not None:raise ValueError('SIMPLE uses equation relaxation; momentum_pseudo_cfl is unsupported')
        if not 0<c.pressure_relaxation<=1:raise ValueError('pressure relaxation must lie in (0,1]')
        if not np.isfinite(c.pressure_increment_damping) or c.pressure_increment_damping<0:
            raise ValueError('pressure increment damping must be finite and nonnegative')
        if not self._flux_initialized:
            self.flux=self.velocity_flux_matrix@self.u.ravel()+self.velocity_flux_source
            self._flux_initialized=True
        converged=False;self.last_rejected_linear_solve=None
        for iteration in range(c.max_iterations):
            old_u=self.u.copy();old_p=self.p.copy();old_q=self.flux.copy()
            implicit,full,rhs,inverse,alpha,extra=self._simple_momentum(self.flux)
            A=(implicit+sparse.diags(extra)).tocsc()
            target=rhs-self.pressure.pressure_force_matrix@self.p+(implicit-full)@self.u.ravel()+extra*self.u.ravel()

            def solve_with_amg(matrix,vector):
                # Smoothed-aggregation AMG on the pressure-correction operator.
                # The operator is a divergence-of-response Laplacian, so AMG is
                # the natural scalable preconditioner; but thin-layer meshes
                # make its symmetrized form indefinite (a few eigenvalues below
                # zero, non-dominant rows), and V-cycles *diverge* on such an
                # operator.  Estimate the most negative eigenvalue and shift
                # the preconditioner operator into the SPD cone; the shift is
                # reported in the history and the true residual of the
                # unshifted system is still the acceptance gate.
                import pyamg
                from scipy.sparse.linalg import eigsh, gmres
                symmetrized=((matrix+matrix.T)*.5).tocsr()
                shift=0.
                try:
                    smallest=float(eigsh(symmetrized.tocsc(),k=1,which='SA',
                        return_eigenvectors=False,maxiter=60,tol=1e-3)[0])
                    if smallest<0:shift=-1.05*smallest
                except Exception:
                    shift=.5*float(np.median(np.abs(matrix.diagonal())))
                if shift>0:symmetrized=(symmetrized+sparse.eye(self.N,format='csr')*shift).tocsr()
                hierarchy=pyamg.smoothed_aggregation_solver(symmetrized,max_coarse=100)
                preconditioner=hierarchy.aspreconditioner()
                counter=[];tolerance=max(c.linear_tolerance,1e-10)
                answer,info=gmres(matrix,vector,M=preconditioner,rtol=tolerance,atol=0.,
                    restart=80,maxiter=60,callback=lambda r:counter.append(r),callback_type='pr_norm')
                residual=float(np.linalg.norm(matrix@answer-vector)/max(np.linalg.norm(vector),1e-30))
                return answer,residual,dict(pressure_amg_shift=shift,
                    pressure_amg_levels=int(len(hierarchy.levels)),
                    pressure_amg_operator_complexity=float(hierarchy.operator_complexity()),
                    pressure_amg_iterations=int(len(counter)),pressure_amg_info=int(info))

            def solve_equilibrated(matrix,vector):
                # Thin-layer meshes make the segregated systems span many row
                # scales; raw float64 residuals stall at the matvec rounding
                # floor. Mirror the coupled solver: row-equilibrate, refine,
                # and gate the true residual of the equilibrated system.
                scale=np.maximum(np.asarray(abs(matrix).sum(axis=1)).ravel(),1e-30)
                scaled=(sparse.diags(1/scale)@matrix).tocsc()
                scaled_vector=vector/scale
                factor=splu(scaled,permc_spec=c.lu_ordering)
                answer=factor.solve(scaled_vector)
                for _ in range(3):
                    shortfall=scaled_vector-scaled@answer
                    if np.linalg.norm(shortfall)<=max(np.linalg.norm(scaled_vector)*c.linear_tolerance,1e-14):break
                    answer+=factor.solve(shortfall)
                residual=float(np.linalg.norm(scaled@answer-scaled_vector)/max(np.linalg.norm(scaled_vector),1e-30))
                return answer,residual

            predicted,linear_u=solve_equilibrated(A,target)
            correction=sparse.diags(alpha)@self.pressure.pressure_flux_defect
            qsource_step=self.velocity_flux_source+alpha*self.p_defect_source
            qstar=self.velocity_flux_matrix@predicted+correction@self.p+qsource_step
            # Differentiate the actual face-flux reconstruction with respect
            # to the full SIMPLE pressure increment. The compact-only shortcut
            # assumes B diag(1/aP) G == diag(alpha) I(grad p). That identity
            # fails with variable mobility, skew reconstruction or component
            # dependent wall resistance. Keep all these contributions here.
            response=(correction-self.velocity_flux_matrix@sparse.diags(inverse)
                @self.pressure.pressure_force_matrix).tocsr()
            P=(self.viscous.divergence_matrix@response).tolil()
            prhs=-self.viscous.divergence_matrix@qstar
            if self.gauge:P[0,:]=0;P[0,0]=1;prhs[0]=self.gauge_value-self.p[0]
            P=P.tocsc()
            damping_scale=0.
            if c.pressure_increment_damping>0:
                # Anchor the pressure increment where the correction is only
                # weakly constrained (thin-layer/tail regions whose level is far
                # from the pressure-Dirichlet boundary).  The term is O(dp), so
                # it vanishes at the fixed point and changes neither the
                # converged state nor the continuity operator; it only bounds
                # the increment that a near-null mode would otherwise amplify.
                reference=float(np.median(np.abs(P.diagonal())))
                if not np.isfinite(reference) or reference<=0:
                    raise RuntimeError('nonpositive pressure correction reference scale')
                damping_scale=c.pressure_increment_damping*reference
                P=(P+sparse.eye(self.N,format='csc')*damping_scale).tocsc()
                if self.gauge:P[0,:]=0;P[0,0]=1
            # Pressure-correction solve.  The AMG preconditioner is optional
            # (tensorfvm[amg]); the direct sparse factorization remains the
            # default and the fallback, and both are gated on the true residual
            # of the unshifted system.
            pressure_extra=dict(pressure_amg_shift=0.,pressure_amg_levels=0,
                pressure_amg_operator_complexity=1.,pressure_amg_iterations=0,
                pressure_amg_info=0,pressure_linear_solver='direct')
            used_amg=False
            if c.linear_method=='amg':
                pressure_scale=np.maximum(np.asarray(abs(P).sum(axis=1)).ravel(),1e-30)
                scaled_p=(sparse.diags(1/pressure_scale)@P).tocsc()
                dp,_,amg_report=solve_with_amg(scaled_p,prhs/pressure_scale)
                # Row scaling is one-sided: diag(1/s) P dp = diag(1/s) rhs, so
                # dp itself solves the original system.  Gate on the true
                # residual of the unscaled correction operator.
                linear_p=float(np.linalg.norm(P@dp-prhs)/max(np.linalg.norm(prhs),1e-30))
                pressure_extra.update(amg_report);pressure_extra['pressure_linear_solver']='amg'
                used_amg=linear_p<=max(5*c.linear_tolerance,1e-9)
                if not used_amg:pressure_extra['pressure_linear_solver']='amg-fallback'
            if not used_amg:
                dp,linear_p=solve_equilibrated(P,prhs)
            norm=lambda x:float(np.linalg.norm(x))
            finite=np.isfinite(predicted).all() and np.isfinite(dp).all() and np.isfinite([linear_u,linear_p]).all()
            if not finite or max(linear_u,linear_p)>max(5*c.linear_tolerance,1e-9):
                self.last_rejected_linear_solve=dict(
                    momentum_relative_residual=linear_u if np.isfinite(linear_u) else None,
                    pressure_relative_residual=linear_p if np.isfinite(linear_p) else None,
                    momentum_system=linear_system_diagnostics(A,target,predicted),
                    pressure_system=linear_system_diagnostics(P,prhs,dp),
                    accepted_state_changed=False)
                raise RuntimeError(f'SIMPLE linear solve failed true residual gate: momentum={linear_u:g}, pressure={linear_p:g}; accepted state preserved')
            # The accepted triplet must stay on the face-flux ledger manifold
            # flux == velocity_flux_matrix @ u + correction @ p + qsource_step.
            # The full pressure increment dp therefore enters the velocity
            # response, the pressure state and the flux response together.
            # The legacy update p += pressure_relaxation * dp while the flux
            # took the full response left (1 - pressure_relaxation) * correction
            # @ dp off the manifold every sweep and drove thin-layer SUBOFF
            # iterations apart (docs/verification-suboff-simple-sst-wall).
            # Any damping is now a convex mix of the whole consistent step:
            # it preserves the ledger exactly and decays continuity by the
            # same factor instead of tearing the state apart.
            candidate_u=(predicted-inverse*(self.pressure.pressure_force_matrix@dp)).reshape(self.N,3)
            candidate_p=self.p+dp
            if self.gauge:candidate_p+=self.gauge_value-candidate_p[0]
            candidate_flux=qstar+response@dp
            if not all(np.isfinite(x).all() for x in (candidate_u,candidate_p,candidate_flux)):
                raise RuntimeError('nonfinite SIMPLE candidate; accepted state preserved')
            w=c.pressure_relaxation
            self.u=(1-w)*old_u+w*candidate_u
            self.p=(1-w)*old_p+w*candidate_p
            self.flux=(1-w)*old_q+w*candidate_flux
            block,rhs,_,corr,src=self._system(self.flux)
            consistent=self.velocity_flux_matrix@self.u.ravel()+corr@self.p+src
            step_identity=self.velocity_flux_matrix@self.u.ravel()+correction@self.p+qsource_step
            norms=np.linalg.norm(self.mesh.area_vectors,axis=1)
            residual=(block@np.r_[self.u.ravel(),self.p]-rhs)[:3*self.N]
            norm=lambda x:float(np.linalg.norm(x))
            row=dict(iteration=iteration+1,algorithm=c.pressure_velocity_algorithm.upper(),
                simplec_limited_dofs=self.simplec_limited_dofs,
                velocity_change=norm(self.u-old_u)/max(norm(self.u),1e-30),
                pressure_change=norm(self.p-old_p)/max(norm(self.p),1.),
                flux_change=norm(self.flux-old_q)/max(norm(self.flux),1e-30),
                momentum_residual=norm(residual)/max(norm(self.force_integrated)+norm(self.diffusion_source)+norm(self.p_force_source),1.),
                native_divergence_max=float(np.abs(self.viscous.op.integrate(self.flux)/self.mesh.cell_volumes).max()),
                current_reconstructed_divergence_max=float(np.abs(self.viscous.op.integrate(consistent)/self.mesh.cell_volumes).max()),
                face_flux_consistency_max=float(np.max(np.abs(consistent-self.flux)/norms)),
                ledger_identity_defect_max=float(np.max(np.abs(self.flux-step_identity)/norms)),
                pressure_increment_flux_response_max=float(np.max(np.abs(correction@dp)/norms)),
                # Admissibility of the pressure-correction operator.  A
                # well-posed correction behaves like a scaled Laplacian; the
                # count of non-negative-diagonal rows and the weakest
                # diagonal-to-row-sum ratio expose thin-layer/sliver cells
                # where dp is only weakly constrained.  Neither quantity is
                # used to clip anything: the state update is untouched.
                pressure_operator_negative_diagonal_count=int(np.sum(P.diagonal()<0)),
                pressure_operator_minimum_diagonal_ratio=float(np.min(
                    np.abs(P.diagonal())/np.maximum(np.asarray(abs(P).sum(axis=1)).ravel(),1e-300))),
                pressure_operator_diagonal_span=(float(P.diagonal().min()),float(P.diagonal().max())),
                maximum_increment=float(np.max(np.abs(dp))),
                pressure_increment_damping_scale=float(damping_scale),
                **pressure_extra,
                true_linear_relative_residual=max(linear_u,linear_p),linear_iterations=2)
            self.history.append(row)
            if callback:callback(row)
            if max(row[key] for key in ('velocity_change','pressure_change','flux_change','momentum_residual','native_divergence_max','current_reconstructed_divergence_max','face_flux_consistency_max'))<c.nonlinear_tolerance:
                converged=True;break
        self.converged=converged
        return dict(converged=converged,history=self.history,velocity=self.u.copy(),pressure=self.p.copy(),flux=self.flux.copy())


class NativeSimpleSolver3D(NativeSimpleMixin,NativeSteadySolver3D):
    pass
