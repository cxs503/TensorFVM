"""Geometry-aware linear upwind with conservative deferred face correction.

The method follows the Taylor reconstruction documented in OpenFOAM's
linearUpwind scheme. This independently written implementation retains the
implicit upwind matrix and adds the shared internal-face correction to both
adjacent cells with opposite signs. Physical boundary corrections are zero.
Unlimited linear upwind is not a bounded/TVD scheme.
"""
import numpy as np
import torch
from scipy import sparse
from .external_anderson import AndersonExternalFlowSolver
from .external_linear import ExternalFlowSolver
from .external_coupled import CoupledExternalFlowSolver
from .coupled_coarse import jacobian, operators, array


class LinearUpwindMomentum:
    momentum_convection_scheme = 'linear-upwind'
    def _upwind_geometry(self, flux):
        owner = self.o[self.f]
        upstream = torch.where(flux[self.f]>0, owner, self.ni)
        displacement = self.mesh.face_centers[self.f]-self.mesh.centers.reshape(-1,2)[upstream]
        return upstream, displacement

    def linear_upwind_correction(self, velocity, flux, gradient=None):
        if gradient is None:
            gradient = self._gradient(velocity)
        upstream, displacement = self._upwind_geometry(flux)
        correction = torch.zeros_like(self.S)
        correction[self.f] = torch.einsum('fi,fij->fj', displacement, gradient[upstream])
        return correction

    def _momentum(self, velocity, pressure, flux):
        diagonal, ao, an, source = super()._momentum(velocity,pressure,flux)
        source = source-self._sum(flux[:,None]*self.linear_upwind_correction(velocity,flux))
        return diagonal,ao,an,source


class LinearUpwindExternalSolver(LinearUpwindMomentum,AndersonExternalFlowSolver):
    pass


class LinearUpwindCoupledSolver(LinearUpwindMomentum,CoupledExternalFlowSolver):
    def __init__(self,config):
        if config.device!='cpu' or config.mesh_type not in ('body-fitted','gmsh-cylinder') or config.time_step is not None or config.turbulence_model!='laminar':
            raise ValueError('Linear upwind coupled solver supports steady laminar CPU cylinders')
        ExternalFlowSolver.__init__(self,config)
        self.coupled_history=[];self._coupled_lu=None

    def _coupled_jacobian(self):
        # Build the original upwind Picard Jacobian through an adapter that
        # evaluates the original upwind equation during its response check.
        class UpwindAdapter:
            def __getattr__(adapter,name):
                return getattr(self,name)
            def _momentum(adapter,u,p,flux):
                return super(LinearUpwindMomentum,self)._momentum(u,p,flux)
        base,_=jacobian(UpwindAdapter())
        H,I,Gp,Gv,B=operators(self)
        upstream, displacement=self._upwind_geometry(self.mass_flux)
        indices=np.flatnonzero(array(self.f));up=array(upstream);delta=array(displacement)
        selector=sparse.coo_matrix((np.ones(len(indices)),(indices,up)),shape=(len(self.o),self.count)).tocsr()
        correction=sum((sparse.diags(np.bincount(indices,weights=delta[:,a],minlength=len(self.o)))@selector@Gv[a] for a in range(2)),start=sparse.csr_matrix((len(self.o),self.count)))
        addition=H@sparse.diags(array(self.mass_flux))@correction
        zero=sparse.csr_matrix((self.count,self.count))
        full=base+sparse.bmat([[addition,zero,zero],[zero,addition,zero],[zero,zero,zero]],format='csr')
        diagonal,_,_,_=self._momentum(self.velocity,self.p,self.mass_flux)
        D=self.volume/diagonal
        probe=torch.sin(torch.arange(3*self.count,dtype=self.p.dtype,device=self.p.device)*.371)
        du=torch.stack([probe[:self.count],probe[self.count:2*self.count]],1);dp=probe[2*self.count:]
        def residual(u,p):
            d,ao,an,rhs=self._momentum(u,p,self.mass_flux)
            mom=torch.stack([self._matvec(u[:,a],d,ao,an)-rhs[:,a] for a in range(2)],1)
            return torch.cat([mom[:,0],mom[:,1],self._sum(self._rhie_chow(u,p,D)[0])])
        response=array(residual(self.velocity+du,self.p+dp)-residual(self.velocity,self.p))
        defect=float(np.linalg.norm(full@array(probe)-response)/np.linalg.norm(response))
        if defect>2e-11:
            raise RuntimeError(f'Linear upwind coupled Jacobian mismatch: {defect}')
        return full,defect


from .conservative_pressure import ConservativePressureGradient


class ConservativeLinearUpwindCoupledSolver(ConservativePressureGradient,LinearUpwindCoupledSolver):
    pass


class ConservativeLinearUpwindSolver(ConservativePressureGradient,LinearUpwindExternalSolver):
    pass
