"""Frozen SST-2003 transport on the energy-based hybrid scalar backend.

A step does not establish nonlinear RANS convergence or a physical benchmark.
Old cell AND shared-face states define the current gradients. No collocated
transport, face interpolation, positivity floor, or hidden clipping is used.
"""
import numpy as np
from .native_hybrid_scalar3d import NativeHybridScalar3D
from .native_sst3d import sst2003_closure, BETA_STAR, VARIANT


class HybridSSTAdmissibilityError(RuntimeError):
    """Rejected candidate and its untouched conservative scalar solve ledger."""
    def __init__(self,name,result):
        self.scalar_name=name
        self.scalar_result=result
        super().__init__(f'invalid hybrid SST {name} update; cell_min={result["cell_min"]:g}, '
                         f'face_min={result["face_min"]:g}; no clipping applied')


class NativeHybridSSTTransport3D:
    def __init__(self, mesh, *, k_dirichlet_mask=None, omega_dirichlet_mask=None, scalar_correction='none'):
        self.mesh = mesh
        if scalar_correction not in ('none','nonlinear_mmp'):raise ValueError('unknown scalar correction')
        self.scalar_correction=scalar_correction
        self.geometry = NativeHybridScalar3D(mesh,dirichlet_mask=k_dirichlet_mask)
        self.k_mask = self.geometry.diffusion.dirichlet.copy()
        self.omega_mask = (mesh.neighbor < 0) if omega_dirichlet_mask is None else np.asarray(omega_dirichlet_mask,bool).copy()
        if self.omega_mask.shape != (self.geometry.F,) or np.any(self.omega_mask & (mesh.neighbor >= 0)):
            raise ValueError('omega Dirichlet mask must select boundary faces')

    def _state(self, value, size, name, strict):
        value = self.geometry._array(value,size,name)
        invalid = np.any(value <= 0) if strict else np.any(value < 0)
        if invalid:
            raise ValueError(f'invalid {name}: no positivity clipping applied')
        return value

    def step(self, k, omega, k_faces, omega_faces, volume_flux, wall_distance, nu,
             velocity_gradient, dt, *, k_boundary, omega_boundary,
             k_inflow=None, omega_inflow=None, k_neumann=0., omega_neumann=0., rho=1.):
        """One conservative, frozen-coefficient backward Euler update.

        Negative production/cross diffusion is retained as a Patankar sink.
        New cell/face positivity is checked; invalid updates are rejected.
        Momentum coupling and nonlinear acceptance are the caller's job.
        """
        if np.ndim(rho) != 0 or not np.isfinite(rho) or rho <= 0:
            raise ValueError('positive constant density required')
        N,F = self.geometry.N,self.geometry.F
        k = self._state(k,N,'k',False); omega = self._state(omega,N,'omega',True)
        kf = self._state(k_faces,F,'face k',False); wf = self._state(omega_faces,F,'face omega',True)
        closure = sst2003_closure(k,omega,wall_distance,nu,velocity_gradient,
            self.geometry.gradient(kf),self.geometry.gradient(wf),rho=rho)
        flux = self.geometry._array(volume_flux,F,'volume flux')
        incoming = (self.mesh.neighbor < 0) & (flux < 0)
        results = {}
        for name,old,sigma,production,destruction,mask,bd,inflow,gn in [
            ('k',k,closure['sigma_k'],closure['production_limited'],BETA_STAR*omega,
             self.k_mask,k_boundary,k_inflow,k_neumann),
            ('omega',omega,closure['sigma_omega'],closure['omega_production']+closure['cross_diffusion_source'],
             closure['beta']*omega,self.omega_mask,omega_boundary,omega_inflow,omega_neumann)]:
            bd = np.broadcast_to(np.asarray(bd,float),(F,))
            self._state(bd[mask],int(mask.sum()),f'boundary {name}',name=='omega')
            if inflow is not None:
                incoming_value = np.broadcast_to(np.asarray(inflow,float),(F,))
                self._state(incoming_value[incoming],int(incoming.sum()),f'inflow {name}',name=='omega')
            negative = np.minimum(production,0.)
            if np.any((old<=0) & (negative<0)):
                raise ValueError('negative SST source cannot be linearized at zero')
            extra_sink = np.divide(-negative,old,out=np.zeros_like(old),where=old>0)
            coefficient = np.broadcast_to(np.asarray(nu,float),(N,))+sigma*closure['nu_t']
            op = NativeHybridScalar3D(self.mesh,coefficient,dirichlet_mask=mask,correction=self.scalar_correction)
            result = op.solve(flux,source=np.maximum(production,0),sink=destruction+extra_sink,
                boundary_values=bd,inflow_values=inflow,boundary_diffusive_flux=gn,old_values=old,dt=dt)
            if (np.any(result['cell_values']<0) or np.any(result['face_values']<0)
                or (name=='omega' and (np.any(result['cell_values']<=0) or np.any(result['face_values']<=0)))):
                raise HybridSSTAdmissibilityError(name,result)
            result['negative_production_sink'] = extra_sink
            result['diffusivity'] = coefficient
            results[name] = result
        return dict(k=results['k']['cell_values'],omega=results['omega']['cell_values'],
            k_faces=results['k']['face_values'],omega_faces=results['omega']['face_values'],
            closure=closure,transport=results,variant=VARIANT,
            nonlinear_rans_converged=False,physical_benchmark_passed=False)

    def steady_residual(self,k,omega,k_faces,omega_faces,volume_flux,wall_distance,nu,
                        velocity_gradient,*,k_boundary,omega_boundary,k_inflow=None,
                        omega_inflow=None,k_neumann=0.,omega_neumann=0.,rho=1.):
        """Original nonlinear SST residual at the supplied current state."""
        if np.ndim(rho)!=0 or not np.isfinite(rho) or rho<=0:
            raise ValueError('positive constant density required')
        N,F=self.geometry.N,self.geometry.F
        k=self._state(k,N,'k',False);omega=self._state(omega,N,'omega',True)
        kf=self._state(k_faces,F,'face k',False);wf=self._state(omega_faces,F,'face omega',True)
        c=sst2003_closure(k,omega,wall_distance,nu,velocity_gradient,
                        self.geometry.gradient(kf),self.geometry.gradient(wf),rho=rho)
        results={}
        for name,cell,face,mask,sigma,source,bd,inflow,gn in [
            ('k',k,kf,self.k_mask,c['sigma_k'],c['k_source'],k_boundary,k_inflow,k_neumann),
            ('omega',omega,wf,self.omega_mask,c['sigma_omega'],c['omega_source'],omega_boundary,omega_inflow,omega_neumann)]:
            op=NativeHybridScalar3D(self.mesh,np.asarray(nu)+sigma*c['nu_t'],dirichlet_mask=mask,correction=self.scalar_correction)
            results[name]=op.steady_residual(cell,face,volume_flux,source=source,boundary_values=bd,
                               inflow_values=inflow,boundary_diffusive_flux=gn)
        return dict(closure=c,**results)
