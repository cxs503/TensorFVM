"""Opt-in exact frozen-velocity Newton linearization of the shared SA equation.

Includes variable diffusivity, nonorthogonal diffusion, gradient-squared source
and local production/destruction derivatives. Every assembled Jacobian is
checked against an independent PyTorch automatic-differentiation response.
"""
import numpy as np
import torch
from scipy import sparse
from scipy.sparse.linalg import splu
from .coupled_coarse import operators,array


class NewtonSaMixin:
    def sa_residual(self,q):
        d,a,b,rhs=self._sa_system(q)
        return self._matvec(q,d,a,b)-rhs

    def sa_jacobian(self,q):
        H,I,Gp,G,Bp=operators(self);nf=len(self.o);n=self.count
        o=array(self.o);ni=array(self.ni);f=array(self.f);fixed=array(self.mesh.boundary&~self.mesh.masks['outlet']);ids=np.arange(nf);oi=o[f]
        B=sparse.coo_matrix((np.r_[-np.ones(f.sum()),np.ones(f.sum()),-np.ones(fixed.sum())],(np.r_[ids[f],ids[f],ids[fixed]],np.r_[oi,ni,o[fixed]])),shape=(nf,n)).tocsr()
        c=self.config;sigma=2/3;cb1=.1355;cb2=.622;kappa=.41;cw1=cb1/kappa**2+(1+cb2)/sigma
        wall=self._no_slip_faces();out=self.mesh.masks['outlet']
        gamma=self._interpolate(c.density*(self.kinematic_viscosity+q.clamp_min(0))/sigma);gamma[wall]=c.viscosity/sigma;gamma[out]=0
        grad=self._gradient(q,boundary_values=self.sa_boundary_nu_tilde)
        delta=-q[self.o];delta[self.f]=q[self.ni]-q[self.oi];delta[fixed]+=self.sa_boundary_nu_tilde[fixed]
        normal=self.k*delta+(self.T*self._interpolate(grad)).sum(-1)
        with torch.enable_grad():
            clamped=q.detach().clone().requires_grad_(True);clamp_derivative=torch.autograd.grad(clamped.clamp_min(0).sum(),clamped)[0].detach()
        gamma_derivative=sparse.diags(array((~wall)&(~out)).astype(float)*c.density/sigma)@I@sparse.diags(array(clamp_derivative))
        nonorth=sum((sparse.diags(array(self.T)[:,a])@I@G[a] for a in (0,1)),start=sparse.csr_matrix((nf,n)))
        d,a,b,rhs=self._sa_system(q);matrix=self._matrix(d,a,b)
        matrix=matrix-H@sparse.diags(array(gamma))@nonorth-H@sparse.diags(array(normal))@gamma_derivative
        matrix-=sum((sparse.diags(array(2*c.density*self.volume*cb2/sigma*grad[:,a]))@G[a] for a in (0,1)),start=sparse.csr_matrix((n,n)))
        vg=self._gradient(self.velocity);vorticity=(vg[:,1,0]-vg[:,0,1]).abs();d2=self.wall_distance.square()
        def local(z):
            nu=z.clamp_min(0);chi=nu/self.kinematic_viscosity;fv1=chi**3/(chi**3+7.1**3);fv2=1-chi/(1+chi*fv1);st=self.sa_stilde(vorticity,nu);r=(nu/(st*kappa**2*d2)).clamp(max=10);g=r+.3*(r**6-r);fw=g*((1+2.**6)/(g**6+2.**6))**(1/6)
            destruction=c.density*self.volume*cw1*fw*nu/d2
            return destruction*z-c.density*self.volume*cb1*st*nu,destruction
        with torch.enable_grad():
            z=q.detach().clone().requires_grad_(True);value,dest=local(z);derivative=torch.autograd.grad(value.sum(),z)[0]
        matrix+=sparse.diags(array(derivative-dest.detach()))
        probe=torch.sin(torch.arange(n,dtype=q.dtype)*.371)*self.kinematic_viscosity
        with torch.enable_grad():_,response=torch.autograd.functional.jvp(self.sa_residual,q,probe)
        response=array(response);defect=float(np.linalg.norm(matrix@array(probe)-response)/max(np.linalg.norm(response),1e-30))
        if defect>2e-10:raise RuntimeError(f'Exact SA Jacobian mismatch: {defect}')
        return matrix,defect

    @torch.no_grad()
    def _turbulence_step(self,physical_old=None):
        if physical_old is not None:raise ValueError('Exact SA Newton option currently supports steady equations')
        if self.config.turbulence_model=='laminar':return 0.,0.
        old=self.nu_tilde.clone();before=self.sa_residual(old);matrix,defect=self.sa_jacobian(old);rhs=-array(before);norm=float(torch.linalg.vector_norm(before));accepted=False;linear=None;used_tau=None;omega=0.;new_norm=norm
        # A pseudo-time diagonal regularizes a nearly singular production
        # Jacobian. It vanishes from the nonlinear residual and fixed point.
        for tau in [None,1.,.1,.01,.001,.0001]:
            current=matrix if tau is None else matrix+sparse.diags(array(self.config.density*self.volume)*self.config.inlet_velocity/(tau*self.config.reference_length))
            scale=np.maximum(np.asarray(abs(current).sum(axis=1)).ravel(),1e-30)
            try:
                lu=splu((sparse.diags(1/scale)@current).tocsc());q=lu.solve(rhs/scale)
                for _ in range(3):
                    error=rhs-current@q
                    if np.linalg.norm(error)<=max(np.linalg.norm(rhs)*1e-10,1e-15):break
                    q+=lu.solve(error/scale)
                trial_linear=float(np.linalg.norm(current@q-rhs)/max(np.linalg.norm(rhs),1e-30))
            except RuntimeError:continue
            if not np.isfinite(q).all() or trial_linear>1e-9:continue
            linear=trial_linear;used_tau=tau;omega=min(.8,self.config.turbulence_relaxation*2)
            for _ in range(14):
                candidate=(old+omega*torch.from_numpy(q)).clamp_min(0);after=self.sa_residual(candidate);new_norm=float(torch.linalg.vector_norm(after))
                if torch.isfinite(after).all() and new_norm<norm:accepted=True;break
                omega*=.5
            if accepted:break
        if linear is None:raise RuntimeError('SA Newton: all current regularized linear systems failed their true residual checks')
        if accepted:self.nu_tilde=candidate
        self.turbulent_kinematic_viscosity=self._sa_eddy_viscosity(self.nu_tilde)
        self.sa_newton_history=getattr(self,'sa_newton_history',[]);self.sa_newton_history.append(dict(accepted=accepted,relaxation=omega if accepted else 0,pseudo_time_in_convective_units=used_tau,assembled_operator_relative_defect=defect,linear_true_relative_residual=linear,nonlinear_l2_before=norm,nonlinear_l2_after=new_norm if accepted else norm))
        residual=self.sa_residual(self.nu_tilde);scale=self.config.density*self.config.inlet_velocity*self.config.reference_length*self.kinematic_viscosity
        return float(residual.abs().max()/scale),float((self.nu_tilde-old).abs().max()/self.kinematic_viscosity)
