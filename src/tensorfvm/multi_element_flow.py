"""Conservative CPU SIMPLE/SA on the shared multi-element Mesh2D backend.

Sparse current matrices and true residual gates reuse existing common modules.
No experimental coefficients enter the equations. Optional bounded linear upwind
adds equal/opposite internal-face fluxes, using neighbor extrema for limiting.
"""
import numpy as np
import torch
from .conservative_pressure import ConservativePressureGradient
from .external_high_order import LinearUpwindMomentum
from .external_cached import CachedExternalFlowSolver
from .coupled_coarse import operators,array
from scipy import sparse


class MultiElementFlowSolver(ConservativePressureGradient,LinearUpwindMomentum,CachedExternalFlowSolver):
    sa_vorticity_floor = .3
    def __init__(self,config,*,bounded=True):
        if config.mesh_type!='three-element' or config.device!='cpu' or config.time_step is not None:
            raise ValueError('Multi-element sparse verification requires steady three-element CPU mesh')
        super().__init__(config);self.bounded=bounded;self.convection_blend=1.;self.sa_inner_iterations=3
        self._sa_lu_cache={};self._sa_linear_calls={};self.anderson_depth=0
        self._in_sa_step=False;self._momentum_reference=None;self._momentum_matrix_override=None;self._momentum_face_factors=None;self._momentum_stress_source=None
        self.implicit_convection=False;self._lagged_upwind=None

    def stress_correction(self,velocity):
        """Transpose-gradient/deviatoric contribution of variable viscosity.

        SA viscosity varies in space. Its full incompressible RANS stress
        cannot be replaced by a scalar variable-coefficient Laplacian alone.
        """
        gradient=self._interpolate(self._gradient(velocity))
        viscosity=self._interpolate(self._dynamic_viscosity);viscosity[self._no_slip_faces()]=self.config.viscosity
        trace=gradient[:,0,0]+gradient[:,1,1]
        traction=viscosity[:,None]*(torch.einsum('fji,fi->fj',gradient,self.S)-(2/3)*trace[:,None]*self.S)
        traction[self.mesh.masks['outlet']]=0
        return self._sum(traction)

    def stress_correction_matrices(self):
        H,I,_,Gv,_=operators(self)
        viscosity=self._interpolate(self._dynamic_viscosity);viscosity[self._no_slip_faces()]=self.config.viscosity;viscosity[self.mesh.masks['outlet']]=0
        return [[H@(sparse.diags(array(viscosity*self.S[:,k]))@I@Gv[j]-(2/3)*sparse.diags(array(viscosity*self.S[:,j]))@I@Gv[k]) for k in (0,1)] for j in (0,1)]

    def _momentum(self,velocity,pressure,flux):
        d,a,b,rhs=super()._momentum(velocity,pressure,flux)
        stress=self.stress_correction(velocity) if self._momentum_stress_source is None else self._momentum_stress_source
        return d,a,b,rhs+stress

    def sa_stilde(self,vorticity,nu):
        chi=nu.clamp_min(0)/self.kinematic_viscosity
        fv1=chi.pow(3)/(chi.pow(3)+7.1**3)
        fv2=1-chi/(1+chi*fv1)
        return torch.maximum(vorticity+nu.clamp_min(0)*fv2/(.41**2*self.wall_distance.square()),self.sa_vorticity_floor*vorticity).clamp_min(1e-14)

    def _sa_system(self,nu_tilde):
        # OpenFOAM's documented SA guard: Stilde >= 0.3*Omega. This avoids a
        # near-zero rotation scale inside vortical coves without changing the
        # transport, diffusion, wall distance or force equations.
        diagonal,ao,an,source=super()._sa_system(nu_tilde)
        c=self.config;nu=nu_tilde.clamp_min(0);chi=nu/self.kinematic_viscosity;fv1=chi**3/(chi**3+7.1**3);fv2=1-chi/(1+chi*fv1);gradient=self._gradient(self.velocity);omega=(gradient[:,1,0]-gradient[:,0,1]).abs();d2=self.wall_distance.square();old=(omega+nu*fv2/(.41**2*d2)).clamp_min(1e-14);new=self.sa_stilde(omega,nu)
        def destruction(st):
            r=(nu/(st*.41**2*d2)).clamp(max=10);g=r+.3*(r**6-r);fw=g*((1+2.**6)/(g**6+2.**6))**(1/6)
            return c.density*self.volume*(.1355/.41**2+(1+.622)/(2/3))*fw*nu/d2
        diagonal=diagonal+destruction(new)-destruction(old)
        source=source+c.density*self.volume*.1355*(new-old)*nu
        return diagonal,ao,an,source

    def linear_upwind_correction(self,velocity,flux,gradient=None):
        if self._lagged_upwind is not None:return self._lagged_upwind.clone()
        correction=super().linear_upwind_correction(velocity,flux,gradient)
        if self._momentum_face_factors is not None:
            correction[self.f]*=self._momentum_face_factors
            return correction
        if self.bounded:
            minimum=velocity.clone();maximum=velocity.clone()
            minimum.index_reduce_(0,self.oi,velocity[self.ni],reduce='amin',include_self=True)
            minimum.index_reduce_(0,self.ni,velocity[self.oi],reduce='amin',include_self=True)
            maximum.index_reduce_(0,self.oi,velocity[self.ni],reduce='amax',include_self=True)
            maximum.index_reduce_(0,self.ni,velocity[self.oi],reduce='amax',include_self=True)
            fixed=self.mesh.boundary&~self.mesh.masks['outlet'];owners=self.o[fixed];boundary=self.boundary_velocity[fixed]
            minimum.index_reduce_(0,owners,boundary,reduce='amin',include_self=True);maximum.index_reduce_(0,owners,boundary,reduce='amax',include_self=True)
            upstream,_=self._upwind_geometry(flux);delta=correction[self.f];base=velocity[upstream]
            ratio=torch.where(delta>0,(maximum[upstream]-base)/delta.clamp_min(1e-30),(minimum[upstream]-base)/delta.clamp_max(-1e-30))
            ratio=torch.where(delta.abs()<1e-30,torch.ones_like(ratio),ratio).clamp(0,1)
            # One limiter per upstream cell/component preserves a consistent
            # linear reconstruction across all its outgoing internal faces.
            cell_ratio=torch.ones_like(velocity);cell_ratio.index_reduce_(0,upstream,ratio,reduce='amin',include_self=True)
            correction[self.f]*=cell_ratio[upstream]
        return self.convection_blend*correction

    def _turbulence_step(self,physical_old=None):
        # Turbulence and momentum have different scales/matrices: retain their
        # own preconditioners while still solving each current equation.
        cache,calls=self._lu_cache,self._linear_calls;self._lu_cache,self._linear_calls=self._sa_lu_cache,self._sa_linear_calls
        self._in_sa_step=True
        begin=len(self.linear_history)
        try:
            if self.config.turbulence_model=='laminar':return 0.,0.
            if physical_old is not None:raise ValueError('Steady multi-element SA update requires no physical old field')
            initial=self.nu_tilde.clone();c=self.config
            for _ in range(self.sa_inner_iterations):
                old=self.nu_tilde.clone();diagonal,ao,an,source=self._sa_system(old)
                relaxed=diagonal/c.turbulence_relaxation
                rhs=source+(1-c.turbulence_relaxation)*relaxed*old
                # Diagonal/RHS relaxation already damps the update. A second
                # field relaxation would multiply this step by alpha again.
                self.nu_tilde=self._linear(relaxed,ao,an,rhs,old,threshold_floor=1e-12,allow_inexact=True).clamp_min(0)
                self.turbulent_kinematic_viscosity=self._sa_eddy_viscosity(self.nu_tilde)
                d,a,b,rhs=self._sa_system(self.nu_tilde)
                residual=float((self._matvec(self.nu_tilde,d,a,b)-rhs).abs().max()/(c.density*c.inlet_velocity*c.reference_length*self.kinematic_viscosity))
                if residual<c.tolerance*.1:break
            return residual,float((self.nu_tilde-initial).abs().max()/self.kinematic_viscosity)
        finally:
            self._sa_lu_cache,self._sa_linear_calls=self._lu_cache,self._linear_calls;self._lu_cache,self._linear_calls=cache,calls
            self._in_sa_step=False
            for record in self.linear_history[begin:]:record['kind']='spalart-allmaras'

    def _matrix(self,diagonal,ao,an):
        if self._momentum_matrix_override is not None:return self._momentum_matrix_override
        return super()._matrix(diagonal,ao,an)

    def momentum_correction_matrices(self,reference,flux):
        """Implicit nonorthogonal diffusion with bounded deferred convection.

        Current unrelaxed equations remain exactly the shared equations. Only
        their Picard linear solve changes; physical boundary constants remain
        on the RHS. A separate production response verifies both components.
        """
        H,I,Gp,Gv,B=operators(self);nf=len(self.o)
        viscosity=self._interpolate(self._dynamic_viscosity);viscosity[self._no_slip_faces()]=self.config.viscosity;viscosity[self.mesh.masks['outlet']]=0
        diffusion=-H@sum((sparse.diags(array(viscosity*self.T[:,a]))@I@Gv[a] for a in (0,1)),start=sparse.csr_matrix((nf,self.count)))
        up,delta=self._upwind_geometry(flux);indices=np.flatnonzero(array(self.f))
        selector=sparse.coo_matrix((np.ones(len(indices)),(indices,array(up))),shape=(nf,self.count)).tocsr()
        reconstruction=sum((sparse.diags(np.bincount(indices,weights=array(delta)[:,a],minlength=nf))@selector@Gv[a] for a in (0,1)),start=sparse.csr_matrix((nf,self.count)))
        raw=LinearUpwindMomentum.linear_upwind_correction(self,reference,flux)[self.f];limited_full=self.linear_upwind_correction(reference,flux);limited=limited_full[self.f]
        factors=torch.where(raw.abs()>1e-30,limited/raw,torch.full_like(raw,self.convection_blend))
        matrices=[]
        for axis in (0,1):
            multiplier=np.zeros(nf);multiplier[indices]=array(factors)[:,axis]
            matrices.append(diffusion+H@sparse.diags(array(flux)*multiplier)@reconstruction if self.implicit_convection else diffusion.copy())
        probe=torch.stack([torch.sin(torch.arange(self.count,dtype=self.p.dtype)*.371),torch.cos(torch.arange(self.count,dtype=self.p.dtype)*.271)],1)
        def residual(u):
            d,a,b,rhs=self._momentum(u,self.p,flux)
            return torch.stack([self._matvec(u[:,axis],d,a,b)-rhs[:,axis] for axis in (0,1)],1)
        try:
            self._momentum_face_factors=factors
            if not self.implicit_convection:self._lagged_upwind=limited_full
            self._momentum_stress_source=self.stress_correction(reference)
            d,a,b,_=self._momentum(reference,self.p,flux);base=super()._matrix(d,a,b)
            response=array(residual(reference+probe)-residual(reference))
        finally:self._momentum_face_factors=None;self._momentum_stress_source=None;self._lagged_upwind=None
        assembled=np.stack([(base+matrices[axis])@array(probe[:,axis]) for axis in (0,1)],1)
        defect=float(np.linalg.norm(assembled-response)/np.linalg.norm(response))
        if defect>2e-11:raise RuntimeError(f'Current full momentum operator mismatch: {defect}')
        return matrices,defect

    def _linear(self,diagonal,ao,an,rhs,initial,symmetric=False,operator=None,**kwargs):
        # The pressure coefficient changes rapidly during startup on a 500C
        # domain. A fresh factor avoids hundreds of ineffective Krylov vectors
        # with the previous coefficient field as a preconditioner.
        self._linear_calls['pressure' if operator is not None else 'momentum']=0
        full_momentum=operator is None and not self._in_sa_step and self._momentum_reference is not None
        if full_momentum:
            if self._momentum_corrections is None:
                self._momentum_corrections,self._momentum_operator_defect=self.momentum_correction_matrices(self._momentum_reference,self._momentum_reference_flux)
            Q=self._momentum_corrections[self._momentum_component]
            self._momentum_matrix_override=super()._matrix(diagonal,ao,an)+Q
            if np.any(self._momentum_matrix_override.diagonal()<=0):
                self._momentum_matrix_override=None
                raise RuntimeError('Current relaxed momentum matrix has nonpositive diagonal')
            rhs=rhs+torch.from_numpy(Q@array(initial));self._momentum_component+=1
        try:answer=super()._linear(diagonal,ao,an,rhs,initial,symmetric,operator,**kwargs)
        finally:self._momentum_matrix_override=None
        if full_momentum:self.linear_history[-1]['momentum_assembled_operator_relative_defect']=self._momentum_operator_defect
        return answer

    @torch.no_grad()
    def _steady_step(self):
        self._momentum_reference=self.velocity.clone();self._momentum_reference_flux=self.mass_flux.clone();self._momentum_corrections=None;self._momentum_component=0
        try:metrics=super()._steady_step()
        finally:self._momentum_reference=None
        if self.anderson_depth:
            self._unmixed_state=tuple(q.clone() for q in (self.velocity,self.p,self.mass_flux,self.nu_tilde))
        return metrics

    def _true_metrics(self):
        c=self.config
        diagonal,ao,an,source=self._momentum(self.velocity,self.p,self.mass_flux)
        residual=torch.stack([self._matvec(self.velocity[:,a],diagonal,ao,an)-source[:,a] for a in (0,1)],1)
        force=max(c.density*c.inlet_velocity**2*c.reference_length,c.viscosity*c.inlet_velocity)
        physical,_=self._rhie_chow(self.velocity,self.p,self.volume/diagonal)
        mass_scale=c.density*c.inlet_velocity*c.height
        values=dict(momentum=float(residual.abs().sum(0).max()/force),continuity=float(self._sum(self.mass_flux).abs().max()/mass_scale),mass_imbalance=float(self.mass_flux[self.mesh.boundary].sum().abs()/mass_scale),rhie_chow_flux_defect=float(torch.linalg.vector_norm(physical-self.mass_flux)/torch.linalg.vector_norm(self.mass_flux).clamp_min(1e-14)))
        values['u_residual']=float(residual[:,0].abs().sum()/force)
        values['v_residual']=float(residual[:,1].abs().sum()/force)
        values['rhie_chow_max_normal_velocity_defect']=float(((physical-self.mass_flux).abs()/(c.density*c.inlet_velocity*self.mesh.face_lengths)).max())
        if c.turbulence_model=='spalart-allmaras':
            d,a,b,rhs=self._sa_system(self.nu_tilde);defect=self._matvec(self.nu_tilde,d,a,b)-rhs
            values['turbulence']=float(defect.abs().max()/(c.density*c.inlet_velocity*c.reference_length*self.kinematic_viscosity))
        return values

    @torch.no_grad()
    def step(self):
        previous_velocity=self.velocity.clone();previous_nu=self.nu_tilde.clone()
        metrics=super().step();c=self.config;current=self._true_metrics()
        if self.anderson_depth and hasattr(self,'_unmixed_state'):
            mixed=tuple(q.clone() for q in (self.velocity,self.p,self.mass_flux,self.nu_tilde))
            def restore(state):
                for target,value in zip((self.velocity,self.p,self.mass_flux,self.nu_tilde),state):target.copy_(value)
                self.turbulent_kinematic_viscosity=self._sa_eddy_viscosity(self.nu_tilde)
            restore(self._unmixed_state);baseline=self._true_metrics()
            keys=['momentum','turbulence','continuity','rhie_chow_flux_defect']
            accepted=all(np.isfinite(v) for v in current.values()) and sum(current.get(k,0) for k in keys)<=sum(baseline.get(k,0) for k in keys)
            if accepted:restore(mixed)
            else:current=baseline
            metrics['anderson_accepted']=float(accepted)
        metrics.update(current)
        metrics['candidate_velocity_change']=metrics['velocity_change']
        metrics['velocity_change']=float((self.velocity-previous_velocity).abs().max()/c.inlet_velocity)
        if c.turbulence_model=='spalart-allmaras':
            metrics['turbulence_change']=float((self.nu_tilde-previous_nu).abs().max()/self.kinematic_viscosity)
        metrics['convection_blend']=self.convection_blend
        self.converged=self.convection_blend==1 and max(metrics[k] for k in ('momentum','continuity','mass_imbalance','turbulence','rhie_chow_max_normal_velocity_defect') if k in metrics)<c.tolerance and metrics['rhie_chow_flux_defect']<1e-7
        return metrics
