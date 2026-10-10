"""Full-grid coupled Picard residual correction for laminar external-flow CPU solves.

Uses the assembled production Jacobian, including periodic mesh connectivity,
nonorthogonal diffusion, pressure LS gradients and physical Rhie--Chow flux.
No logical coarse interpolation or analytic field is used.
"""
import numpy as np
import torch
from scipy.sparse.linalg import splu
from .external_linear import ExternalFlowSolver
from .coupled_coarse import jacobian


class CoupledExternalFlowSolver(ExternalFlowSolver):
    def __init__(self,config):
        if config.device!='cpu' or config.mesh_type!='body-fitted' or config.time_step is not None or config.turbulence_model!='laminar':
            raise ValueError('Full-grid external Picard correction supports verified steady laminar CPU cylinder calculations')
        super().__init__(config);self.coupled_history=[];self._coupled_lu=None

    def _coupled_jacobian(self):
        return jacobian(self)

    def _physical_flux(self):
        diagonal,_,_,_=self._momentum(self.velocity,self.p,self.mass_flux)
        return self._rhie_chow(self.velocity,self.p,self.volume/diagonal)[0]

    @torch.no_grad()
    def step(self):
        metrics=super().step();physical=self._physical_flux()
        defect=float(torch.linalg.vector_norm(physical-self.mass_flux)/torch.linalg.vector_norm(self.mass_flux).clamp_min(1e-14))
        if not (self.converged and defect<1e-7):
            self.mass_flux=physical
            d,ao,an,source=self._momentum(self.velocity,self.p,self.mass_flux)
            mom=torch.stack([self._matvec(self.velocity[:,a],d,ao,an)-source[:,a] for a in range(2)],1);rhs=-torch.cat((mom[:,0],mom[:,1],self._sum(self.mass_flux))).numpy()
            if self._coupled_lu is None or len(self.coupled_history)%5==0:
                self._coupled_matrix,self._operator_defect=self._coupled_jacobian()
                self._coupled_lu=splu(self._coupled_matrix.tocsc())
            q=self._coupled_lu.solve(rhs);relative=float(np.linalg.norm(self._coupled_matrix@q-rhs)/max(np.linalg.norm(rhs),1e-30))
            if not np.isfinite(q).all() or relative>1e-9:raise RuntimeError('Full coupled true linear residual failed')
            du=np.stack((q[:self.count],q[self.count:2*self.count]),1);omega=min(.5,.5*self.config.inlet_velocity/max(np.max(abs(du)),1e-30))
            self.velocity+=torch.from_numpy(omega*du);self.p+=torch.from_numpy(omega*q[2*self.count:]);self.mass_flux=self._physical_flux()
            self.coupled_history.append(dict(linear_true_relative_residual=relative,assembled_operator_relative_defect=self._operator_defect,relaxation=omega,cells=self.count,backend='full-grid-scipy-superlu-cpu'))
        # Recompute every recorded final residual after all updates, including at
        # iteration limits. These values match the saved fields exactly.
        d,ao,an,source=self._momentum(self.velocity,self.p,self.mass_flux)
        residual=torch.stack([self._matvec(self.velocity[:,a],d,ao,an)-source[:,a] for a in range(2)],1)
        c=self.config;force=max(c.density*c.inlet_velocity**2*c.residual_length,c.viscosity*c.inlet_velocity);mass=c.density*c.inlet_velocity*c.height
        metrics.update(momentum=float(residual.abs().sum(0).max())/force,continuity=float(self._sum(self.mass_flux).abs().max())/mass,mass_imbalance=abs(float(self.mass_flux[self.mesh.boundary].sum()))/mass,u_residual=float(residual[:,0].abs().sum())/force,v_residual=float(residual[:,1].abs().sum())/force)
        physical=self._physical_flux();metrics['rhie_chow_flux_defect']=float(torch.linalg.vector_norm(physical-self.mass_flux)/torch.linalg.vector_norm(self.mass_flux).clamp_min(1e-14))
        self.converged=max(metrics[k] for k in ('momentum','continuity','mass_imbalance'))<c.tolerance and metrics['rhie_chow_flux_defect']<1e-7
        return metrics
