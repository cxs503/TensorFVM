"""Optional Anderson acceleration of current-matrix external-flow SIMPLE.

Reuses the existing SparseBodyFittedSolver mixing algorithm; records residuals
from actual post-mixing fields and separately gates the physical flux fixed point.
"""
import torch
from .external_cached import CachedExternalFlowSolver


class AndersonExternalFlowSolver(CachedExternalFlowSolver):
    def __init__(self,config):
        if config.time_step is not None or config.turbulence_model!='laminar':raise ValueError('External Anderson verification supports steady laminar flow')
        super().__init__(config);self.anderson_depth=8

    @torch.no_grad()
    def step(self):
        metrics=super().step();c=self.config
        d,ao,an,source=self._momentum(self.velocity,self.p,self.mass_flux)
        residual=torch.stack([self._matvec(self.velocity[:,a],d,ao,an)-source[:,a] for a in range(2)],1)
        force=max(c.density*c.inlet_velocity**2*c.residual_length,c.viscosity*c.inlet_velocity);mass=c.density*c.inlet_velocity*c.height
        metrics.update(momentum=float(residual.abs().sum(0).max())/force,continuity=float(self._sum(self.mass_flux).abs().max())/mass,mass_imbalance=abs(float(self.mass_flux[self.mesh.boundary].sum()))/mass,u_residual=float(residual[:,0].abs().sum())/force,v_residual=float(residual[:,1].abs().sum())/force)
        physical,_=self._rhie_chow(self.velocity,self.p,self.volume/d)
        metrics['rhie_chow_flux_defect']=float(torch.linalg.vector_norm(physical-self.mass_flux)/torch.linalg.vector_norm(self.mass_flux).clamp_min(1e-14))
        self.converged=max(metrics[k] for k in ('momentum','continuity','mass_imbalance'))<c.tolerance and metrics['rhie_chow_flux_defect']<1e-7
        return metrics
