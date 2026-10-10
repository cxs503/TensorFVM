"""Opt-in full-grid Picard correction of the same multi-element RANS equations.

The Jacobian freezes mass flux, eddy viscosity and reconstruction limiter only
within a linear solve. Nonlinear physical residuals determine whether a proposed
update is accepted. Experimental values never enter the equations.
"""
import numpy as np
import torch
from scipy import sparse
from scipy.sparse.linalg import splu
from .multi_element_flow import MultiElementFlowSolver
from .external_high_order import LinearUpwindMomentum
from .coupled_coarse import jacobian,operators,array


class CoupledMultiElementSolver(MultiElementFlowSolver):
    def __init__(self,config,**kwargs):
        super().__init__(config,**kwargs)
        self.coupled_history=[];self._frozen_multiplier=None

    def linear_upwind_correction(self,velocity,flux,gradient=None):
        if self._frozen_multiplier is None:
            return super().linear_upwind_correction(velocity,flux,gradient)
        correction=LinearUpwindMomentum.linear_upwind_correction(self,velocity,flux,gradient)
        correction[self.f]*=self._frozen_multiplier
        return correction

    def coupled_matrix(self):
        s=self
        class UpwindAdapter:
            def __getattr__(adapter,name):return getattr(s,name)
            def _momentum(adapter,u,p,flux):return super(LinearUpwindMomentum,s)._momentum(u,p,flux)
        base,_=jacobian(UpwindAdapter());H,I,Gp,Gv,B=operators(self)
        unlimited=LinearUpwindMomentum.linear_upwind_correction(self,self.velocity,self.mass_flux)[self.f]
        limited=super().linear_upwind_correction(self.velocity,self.mass_flux)[self.f]
        factor=torch.where(unlimited.abs()>1e-30,limited/unlimited,torch.full_like(unlimited,self.convection_blend))
        up,delta=self._upwind_geometry(self.mass_flux);indices=np.flatnonzero(array(self.f));nf=len(self.o)
        selector=sparse.coo_matrix((np.ones(len(indices)),(indices,array(up))),shape=(nf,self.count)).tocsr()
        reconstruction=sum((sparse.diags(np.bincount(indices,weights=array(delta)[:,a],minlength=nf))@selector@Gv[a] for a in (0,1)),start=sparse.csr_matrix((nf,self.count)))
        additions=[]
        for a in (0,1):
            face_factor=np.zeros(nf);face_factor[indices]=array(factor)[:,a]
            additions.append(H@sparse.diags(array(self.mass_flux)*face_factor)@reconstruction)
        zero=sparse.csr_matrix((self.count,self.count))
        stress=self.stress_correction_matrices()
        full=base+sparse.bmat([[additions[0]-stress[0][0],-stress[0][1],zero],[-stress[1][0],additions[1]-stress[1][1],zero],[zero,zero,zero]],format='csr')
        diagonal,_,_,_=self._momentum(self.velocity,self.p,self.mass_flux);D=self.volume/diagonal
        probe=torch.sin(torch.arange(3*self.count,dtype=self.p.dtype)*.371)
        du=torch.stack([probe[:self.count],probe[self.count:2*self.count]],1);dp=probe[2*self.count:]
        def residual(u,p):
            d,a,b,rhs=self._momentum(u,p,self.mass_flux)
            mom=torch.stack([self._matvec(u[:,axis],d,a,b)-rhs[:,axis] for axis in (0,1)],1)
            return torch.cat([mom[:,0],mom[:,1],self._sum(self._rhie_chow(u,p,D)[0])])
        try:
            self._frozen_multiplier=factor
            response=array(residual(self.velocity+du,self.p+dp)-residual(self.velocity,self.p))
        finally:self._frozen_multiplier=None
        defect=float(np.linalg.norm(full@array(probe)-response)/np.linalg.norm(response))
        if defect>2e-11:raise RuntimeError(f'Frozen-limiter coupled operator mismatch: {defect}')
        return full,defect

    def physical_flux(self):
        d,_,_,_=self._momentum(self.velocity,self.p,self.mass_flux)
        return self._rhie_chow(self.velocity,self.p,self.volume/d)[0]

    def residual_vector(self):
        d,a,b,rhs=self._momentum(self.velocity,self.p,self.mass_flux)
        mom=torch.stack([self._matvec(self.velocity[:,axis],d,a,b)-rhs[:,axis] for axis in (0,1)],1)
        return torch.cat([mom[:,0],mom[:,1],self._sum(self.mass_flux)])

    @torch.no_grad()
    def correct_coupled(self):
        # Start from the physical flux operator. This correction need not leave
        # continuity exactly zero: its true defect remains an acceptance gate.
        self.mass_flux=self.physical_flux();before=self.residual_vector()
        matrix,defect=self.coupled_matrix();rhs=-array(before)
        # Row scaling helps the mixed velocity/pressure system on stretched BLs.
        scale=np.maximum(np.asarray(abs(matrix).sum(axis=1)).ravel(),1e-30)
        lu=splu((sparse.diags(1/scale)@matrix).tocsc());q=lu.solve(rhs/scale)
        for _ in range(3):
            error=rhs-matrix@q
            if np.linalg.norm(error)<=max(np.linalg.norm(rhs)*1e-9,1e-13):break
            q+=lu.solve(error/scale)
        relative=float(np.linalg.norm(matrix@q-rhs)/max(np.linalg.norm(rhs),1e-30))
        if not np.isfinite(q).all() or relative>1e-9:raise RuntimeError(f'Coupled current equation residual failed: {relative}')
        old_u=self.velocity.clone();old_p=self.p.clone();old_flux=self.mass_flux.clone();du=torch.from_numpy(np.stack([q[:self.count],q[self.count:2*self.count]],1));dp=torch.from_numpy(q[2*self.count:])
        omega=min(.5,.25*self.config.inlet_velocity/max(float(du.abs().max()),1e-30));norm=float(before.abs().sum());accepted=False
        for _ in range(10):
            self.velocity.copy_(old_u+omega*du);self.p.copy_(old_p+omega*dp);self.mass_flux=old_flux.clone();self.mass_flux=self.physical_flux()
            candidate=self.residual_vector();new_norm=float(candidate.abs().sum())
            if torch.isfinite(candidate).all() and new_norm<norm:accepted=True;break
            omega*=.5
        if not accepted:self.velocity.copy_(old_u);self.p.copy_(old_p);self.mass_flux=old_flux
        self.coupled_history.append(dict(accepted=accepted,relaxation=omega if accepted else 0,linear_true_relative_residual=relative,assembled_operator_relative_defect=defect,residual_l1_before=norm,residual_l1_after=new_norm if accepted else norm,cells=self.count,jacobian='frozen mass flux, viscosity and bounded reconstruction; current matrix'))
        return accepted

    @torch.no_grad()
    def step(self):
        previous_velocity=self.velocity.clone()
        metrics=super().step()
        if self.convection_blend==1 and not self.converged:
            self.correct_coupled()
            # Use the shared nonlinear equations after every accepted/rejected
            # correction; never reuse the pre-correction residual as evidence.
            c=self.config
            metrics.update(self._true_metrics())
            metrics['velocity_change']=float((self.velocity-previous_velocity).abs().max()/c.inlet_velocity)
            self.converged=max(metrics[k] for k in ('momentum','continuity','mass_imbalance','turbulence','rhie_chow_max_normal_velocity_defect') if k in metrics)<c.tolerance and metrics['rhie_chow_flux_defect']<1e-7
        return metrics
