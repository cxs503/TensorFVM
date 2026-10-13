"""Experimental segregated SST/momentum coupling on native hybrid unknowns.

The momentum variable is modified pressure p*=p+2 rho k/3. Scalar nonlinear
residuals include free face equilibrium and CURRENT closure coefficients.
Numerical convergence does not qualify a physical benchmark.
"""
import numpy as np
from .native_hybrid_ns3d import NativeHybridNavierStokes3D
from .native_hybrid_sst3d import NativeHybridSSTTransport3D
from .native_sst3d import sst2003_closure


class NativeHybridRans3D:
    def __init__(self,flow,wall_distance,nu,*,k_dirichlet_mask=None,omega_dirichlet_mask=None,
                 k_boundary,omega_boundary,k_inflow=None,omega_inflow=None,k_neumann=0.,omega_neumann=0.,
                 scalar_correction='none', mmp_solver='continuation_newton'):
        if not isinstance(flow,NativeHybridNavierStokes3D):
            raise TypeError('native hybrid Navier-Stokes flow required')
        self.flow=flow;self.mesh=flow.mesh
        self.transport=NativeHybridSSTTransport3D(self.mesh,k_dirichlet_mask=k_dirichlet_mask,
                                                 omega_dirichlet_mask=omega_dirichlet_mask,scalar_correction=scalar_correction,mmp_solver=mmp_solver)
        self.nu=self.transport._state(nu,flow.N,'molecular viscosity',True)
        self.wall_distance=self.transport._state(wall_distance,flow.N,'wall distance',True)
        self.rho=flow.convection.density
        self.boundary=dict(k_boundary=k_boundary,omega_boundary=omega_boundary,
            k_inflow=k_inflow,omega_inflow=omega_inflow,k_neumann=k_neumann,omega_neumann=omega_neumann)
        self.history=[];self.converged=False;self.initialized=False

    def initialize(self,k,omega):
        """Equilibrate initial scalar faces; never seed from reference interiors."""
        from .native_hybrid_scalar3d import NativeHybridScalar3D
        values={}
        for name,old,mask in [('k',k,self.transport.k_mask),('omega',omega,self.transport.omega_mask)]:
            old=self.transport._state(old,self.flow.N,name,name=='omega')
            op=NativeHybridScalar3D(self.mesh,self.nu,dirichlet_mask=mask)
            bd=np.broadcast_to(np.asarray(self.boundary[name+'_boundary'],float),(self.flow.F,))
            gn=np.broadcast_to(np.asarray(self.boundary[name+'_neumann'],float),(self.flow.F,))
            r=op.diffusion.equilibrate_faces(old[:,None],bd[:,None],boundary_traction=gn[:,None])
            faces=self.transport._state(r['face_values'][:,0],self.flow.F,'face '+name,name=='omega')
            values[name]=old;values[name+'_faces']=faces
        for name,value in values.items():setattr(self,name,value)
        self.initialized=True;self.converged=False
        self._refresh_viscosity()

    def _closure(self):
        return sst2003_closure(self.k,self.omega,self.wall_distance,self.nu,
            self.flow.velocity_gradient(),self.transport.geometry.gradient(self.k_faces),
            self.transport.geometry.gradient(self.omega_faces),rho=self.rho)

    def _refresh_viscosity(self):
        self.closure=self._closure()
        viscosity=self.rho*(self.nu+self.closure['nu_t'])
        if not np.array_equal(viscosity,self.flow.viscous.mu):
            self.flow.update_dynamic_viscosity(viscosity)

    def current_residual(self):
        """Rebuild momentum and both original steady SST equations at one state."""
        if not self.initialized:raise RuntimeError('initialize turbulence first')
        self._refresh_viscosity()
        s=self.flow;full,source=s._current_system();free=s.viscous.free
        residual=full @ s.state+source-s.divergence.T @ s.p-s.load
        scale=max(np.linalg.norm((s.load-source-full @ s.prescribed)[free]),1.)
        t=self.transport.steady_residual(self.k,self.omega,self.k_faces,self.omega_faces,
            s.volume_flux(),self.wall_distance,self.nu,s.velocity_gradient(),rho=self.rho,**self.boundary)
        return dict(momentum=float(np.linalg.norm(residual[free])/scale),
            divergence=float(np.max(np.abs(s.divergence @ s.state/self.mesh.cell_volumes))),
            gauge=abs(float(s.p[0]-s.gauge_value)) if s.gauge else 0.,
            k=t['k']['relative_residual'],omega=t['omega']['relative_residual'],
            k_boundary=t['k']['dirichlet_error'],omega_boundary=t['omega']['dirichlet_error'],
            transport=t)

    def solve(self,*,max_iterations=100,dt=.01,relaxation=.7,tolerance=1e-7,
              momentum_method='newton',momentum_max_iterations=50,callback=None):
        if not self.initialized:raise RuntimeError('initialize turbulence first')
        if not np.isfinite(dt) or dt<=0 or not np.isfinite(relaxation) or not 0<relaxation<=1:
            raise ValueError('positive dt and relaxation in (0,1] required')
        if max_iterations<1 or not np.isfinite(tolerance) or tolerance<=0:
            raise ValueError('positive iteration limit and tolerance required')
        if momentum_method not in ('newton','simple','simplec','coupled'):
            raise ValueError('unsupported momentum method')
        self.converged=False
        self.tolerance=tolerance
        for _ in range(max_iterations):
            old={key:getattr(self,key).copy() for key in ('k','omega','k_faces','omega_faces')}
            old_u=self.flow.state.copy();old_p=self.flow.p.copy()
            self._refresh_viscosity()
            if momentum_method=='newton':
                mr=self.flow.solve_newton(max_iterations=momentum_max_iterations,tolerance=tolerance*.1)
            else:
                mr=self.flow.solve(algorithm=momentum_method,max_outer_iterations=momentum_max_iterations,tolerance=tolerance*.1)
            if not mr['numerically_converged']:
                raise RuntimeError('hybrid RANS momentum subsolve did not converge')
            step=self.transport.step(self.k,self.omega,self.k_faces,self.omega_faces,
                self.flow.volume_flux(),self.wall_distance,self.nu,self.flow.velocity_gradient(),dt,
                rho=self.rho,**self.boundary)
            for key in old:setattr(self,key,old[key]+relaxation*(step[key]-old[key]))
            current=self.current_residual()
            changes={key:float(np.linalg.norm(getattr(self,key)-old[key])/max(np.linalg.norm(getattr(self,key)),1e-30)) for key in old}
            changes['velocity']=float(np.linalg.norm(self.flow.state-old_u)/max(np.linalg.norm(self.flow.state),1.))
            changes['modified_pressure']=float(np.linalg.norm(self.flow.p-old_p)/max(np.linalg.norm(self.flow.p),1.))
            row=dict(iteration=len(self.history)+1,**{k:v for k,v in current.items() if k!='transport'},
                changes=changes,minimum_k=float(min(self.k.min(),self.k_faces.min())),
                minimum_omega=float(min(self.omega.min(),self.omega_faces.min())),
                frozen_linear_residuals={name:step['transport'][name]['linear_relative_residual'] for name in ('k','omega')})
            self.history.append(row)
            self.converged=bool(max(row[key] for key in ('momentum','divergence','gauge','k','omega','k_boundary','omega_boundary'))<tolerance
                                and max(changes.values())<tolerance)
            if callback:callback(row)
            if self.converged:break
        return self.result()

    def result(self):
        if not self.initialized:raise RuntimeError('initialize turbulence first')
        current=self.current_residual();s=self.flow;r=s.result()
        self.converged=bool(self.converged and max(current[key] for key in
            ('momentum','divergence','gauge','k','omega','k_boundary','omega_boundary'))<self.tolerance)
        physical=s.p-2/3*self.rho*self.k
        pressure_traction=-physical[self.mesh.owner,None]*self.mesh.area_vectors
        isotropic=-2/3*self.rho*self.k[self.mesh.owner,None]*self.mesh.area_vectors
        viscous=s.viscous.diagnostics(s.state)['face_flux']
        correction=r['total_traction_integrated']-(pressure_traction+isotropic+viscous)
        r.update(numerically_converged=self.converged,physical_benchmark_passed=False,
            pressure=physical.copy(),physical_pressure=physical.copy(),modified_pressure=s.p.copy(),
            k=self.k.copy(),omega=self.omega.copy(),k_faces=self.k_faces.copy(),omega_faces=self.omega_faces.copy(),
            nu_t=self.closure['nu_t'].copy(),pressure_traction_integrated=pressure_traction,
            isotropic_turbulent_traction_integrated=isotropic,deviatoric_traction_integrated=viscous,
            numerical_traction_correction=correction,current_rans_residual=current,rans_history=list(self.history))
        return r
