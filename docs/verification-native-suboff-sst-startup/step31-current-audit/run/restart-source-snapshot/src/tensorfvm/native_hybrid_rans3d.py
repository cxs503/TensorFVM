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
                 scalar_correction='none', mmp_solver='continuation_newton', scalar_work_balance_policy='absolute'):
        if not isinstance(flow,NativeHybridNavierStokes3D):
            raise TypeError('native hybrid Navier-Stokes flow required')
        self.flow=flow;self.mesh=flow.mesh
        self.transport=NativeHybridSSTTransport3D(self.mesh,k_dirichlet_mask=k_dirichlet_mask,
                                                 omega_dirichlet_mask=omega_dirichlet_mask,scalar_correction=scalar_correction,mmp_solver=mmp_solver,scalar_work_balance_policy=scalar_work_balance_policy)
        self.nu=self.transport._state(nu,flow.N,'molecular viscosity',True)
        self.wall_distance=self.transport._state(wall_distance,flow.N,'wall distance',True)
        self.rho=flow.convection.density
        self.boundary=dict(k_boundary=k_boundary,omega_boundary=omega_boundary,
            k_inflow=k_inflow,omega_inflow=omega_inflow,k_neumann=k_neumann,omega_neumann=omega_neumann)
        self.history=[];self.converged=False;self.initialized=False

    def initialize(self,k,omega,*,face_initialization='diffusion_equilibrium'):
        """Build an explicit numerical initial guess, never reference interiors.

        cell_average preserves positive cell guesses using convex shared-face
        averages and exact Dirichlet values. It does not claim face equilibrium;
        subsequent transport solves retain their original equations and gates.
        """
        if face_initialization not in ('diffusion_equilibrium','cell_average'):
            raise ValueError('unknown scalar face initialization')
        from .native_hybrid_scalar3d import NativeHybridScalar3D
        values={}
        for name,old,mask in [('k',k,self.transport.k_mask),('omega',omega,self.transport.omega_mask)]:
            old=self.transport._state(old,self.flow.N,name,name=='omega')
            op=NativeHybridScalar3D(self.mesh,self.nu,dirichlet_mask=mask)
            bd=np.broadcast_to(np.asarray(self.boundary[name+'_boundary'],float),(self.flow.F,))
            gn=np.broadcast_to(np.asarray(self.boundary[name+'_neumann'],float),(self.flow.F,))
            if face_initialization=='diffusion_equilibrium':
                r=op.diffusion.equilibrate_faces(old[:,None],bd[:,None],boundary_traction=gn[:,None])
                faces=r['face_values'][:,0]
            else:
                faces=old[self.mesh.owner].copy();internal=self.mesh.neighbor>=0
                faces[internal]=.5*(faces[internal]+old[self.mesh.neighbor[internal]])
                faces[mask]=bd[mask]
            faces=self.transport._state(faces,self.flow.F,'face '+name,name=='omega')
            values[name]=old;values[name+'_faces']=faces
        for name,value in values.items():setattr(self,name,value)
        self.initialized=True;self.converged=False
        self.initialization=dict(face_initialization=face_initialization,
            face_diffusion_equilibrium=face_initialization=='diffusion_equilibrium',
            numerical_initial_guess=True,physical_benchmark_passed=False)
        self._refresh_viscosity()

    def _closure(self):
        return sst2003_closure(self.k,self.omega,self.wall_distance,self.nu,
            self.flow.velocity_gradient(),self.transport.geometry.gradient(self.k_faces),
            self.transport.geometry.gradient(self.omega_faces),rho=self.rho)

    def restore_accepted_fields(self,fields):
        """Restore a numerical iterate without re-equilibrating shared faces.

        File provenance and mesh identity are checked by the runner. Here all
        values, shapes and prescribed conditions are checked before committing
        the state. Restoring an iterate never restores its convergence claim.
        """
        s=self.flow;values={}
        for key,shape in (('velocity',(s.N,3)),('face_velocity',(s.F,3)),
                          ('modified_pressure',(s.N,)),('k',(s.N,)),('omega',(s.N,)),
                          ('k_faces',(s.F,)),('omega_faces',(s.F,))):
            value=np.asarray(fields[key])
            if value.shape!=shape or value.dtype.kind!='f' or not np.isfinite(value).all():
                raise ValueError('invalid restart '+key)
            if key.startswith('k') and np.any(value<0):raise ValueError('negative restart '+key)
            if key.startswith('omega') and np.any(value<=0):raise ValueError('nonpositive restart '+key)
            values[key]=value.copy()
        state=np.r_[values['velocity'].ravel(),values['face_velocity'].ravel()]
        if s.viscous.velocity_space.constraint_error(state-s.prescribed)>1e-12:
            raise ValueError('restart velocity boundary mismatch')
        if s.gauge and abs(values['modified_pressure'][0]-s.gauge_value)>1e-12:
            raise ValueError('restart pressure gauge mismatch')
        for name,mask in (('k',self.transport.k_mask),('omega',self.transport.omega_mask)):
            boundary=np.broadcast_to(np.asarray(self.boundary[name+'_boundary'],float),(s.F,))
            if not np.array_equal(values[name+'_faces'][mask],boundary[mask]):
                raise ValueError('restart '+name+' boundary mismatch')
        closure=sst2003_closure(values['k'],values['omega'],self.wall_distance,self.nu,
            s.velocity_gradient(state),self.transport.geometry.gradient(values['k_faces']),
            self.transport.geometry.gradient(values['omega_faces']),rho=self.rho)
        viscosity=self.rho*(self.nu+closure['nu_t'])
        if not np.isfinite(viscosity).all() or np.any(viscosity<=0):
            raise ValueError('invalid restart viscosity')
        s.update_dynamic_viscosity(viscosity)
        s.state=state;s.p=values['modified_pressure'];s.converged=False;s.nonlinear_converged=False
        for key in ('k','omega','k_faces','omega_faces'):setattr(self,key,values[key])
        self.closure=closure;self.history=[];self.initialized=True;self.converged=False
        self.initialization=dict(face_initialization='restored_accepted_fields',
            face_diffusion_equilibrium=False,numerical_initial_guess=True,physical_benchmark_passed=False)

    def _refresh_viscosity(self):
        self.closure=self._closure()
        viscosity=self.rho*(self.nu+self.closure['nu_t'])
        if not np.array_equal(viscosity,self.flow.viscous.mu):
            self.flow.update_dynamic_viscosity(viscosity)

    def current_residual(self):
        """Rebuild momentum and both original steady SST equations at one state."""
        if not self.initialized:raise RuntimeError('initialize turbulence first')
        self._refresh_viscosity()
        s=self.flow;full,source=s._current_system();space=s.viscous.velocity_space
        residual=full @ s.state+source-s.divergence.T @ s.p-s.load
        scale=max(np.linalg.norm(space.restrict(s.load-source-full @ s.prescribed)),1.)
        t=self.transport.steady_residual(self.k,self.omega,self.k_faces,self.omega_faces,
            s.volume_flux(),self.wall_distance,self.nu,s.velocity_gradient(),rho=self.rho,**self.boundary)
        return dict(momentum=float(np.linalg.norm(space.restrict(residual))/scale),
            divergence=float(np.max(np.abs(s.divergence @ s.state/self.mesh.cell_volumes))),
            velocity_boundary=space.constraint_error(s.state-s.prescribed),
            gauge=abs(float(s.p[0]-s.gauge_value)) if s.gauge else 0.,
            k=t['k']['relative_residual'],omega=t['omega']['relative_residual'],
            k_boundary=t['k']['dirichlet_error'],omega_boundary=t['omega']['dirichlet_error'],
            transport=t)

    def solve(self,*,max_iterations=100,dt=.01,relaxation=.7,tolerance=1e-7,
              momentum_method='newton',momentum_max_iterations=50,
              momentum_coupling='converged',momentum_sweeps=1,
              momentum_relaxation=.5,callback=None):
        """Advance segregated RANS with converged momentum or explicit sweeps.

        Sweeps rebuild convection each time and retain the momentum and pressure
        linear-solve gates. Their incomplete nonlinear momentum solve is allowed
        only as an outer iterate; CURRENT coupled equations and changes must
        still converge before the RANS result can pass.
        """
        if not self.initialized:raise RuntimeError('initialize turbulence first')
        if not np.isfinite(dt) or dt<=0 or not np.isfinite(relaxation) or not 0<relaxation<=1:
            raise ValueError('positive dt and relaxation in (0,1] required')
        if max_iterations<1 or not np.isfinite(tolerance) or tolerance<=0:
            raise ValueError('positive iteration limit and tolerance required')
        if momentum_method not in ('newton','simple','simplec','coupled'):
            raise ValueError('unsupported momentum method')
        if momentum_coupling not in ('converged','sweeps'):
            raise ValueError('unknown momentum coupling')
        if momentum_coupling=='sweeps' and momentum_method not in ('simple','simplec'):
            raise ValueError('momentum sweeps require SIMPLE or SIMPLEC')
        if (not isinstance(momentum_sweeps,(int,np.integer)) or momentum_sweeps<1
                or not np.isfinite(momentum_relaxation) or not 0<momentum_relaxation<=1):
            raise ValueError('positive integer momentum sweeps and relaxation in (0,1] required')
        self.converged=False
        self.tolerance=tolerance
        for _ in range(max_iterations):
            old={key:getattr(self,key).copy() for key in ('k','omega','k_faces','omega_faces')}
            old_u=self.flow.state.copy();old_p=self.flow.p.copy()
            self._refresh_viscosity()
            momentum_start=len(self.flow.history)
            if momentum_method=='newton':
                mr=self.flow.solve_newton(max_iterations=momentum_max_iterations,tolerance=tolerance*.1)
            elif momentum_coupling=='sweeps':
                mr=self.flow.solve(algorithm=momentum_method,max_outer_iterations=momentum_sweeps,
                    max_iterations=1,relaxation=momentum_relaxation,tolerance=tolerance*.1)
            else:
                mr=self.flow.solve(algorithm=momentum_method,max_outer_iterations=momentum_max_iterations,
                    relaxation=momentum_relaxation,tolerance=tolerance*.1)
            if momentum_coupling=='converged' and not mr['numerically_converged']:
                raise RuntimeError('hybrid RANS momentum subsolve did not converge')
            momentum_records=(self.flow.newton_history if momentum_method=='newton'
                              else self.flow.history[momentum_start:])
            step=self.transport.step(self.k,self.omega,self.k_faces,self.omega_faces,
                self.flow.volume_flux(),self.wall_distance,self.nu,self.flow.velocity_gradient(),dt,
                rho=self.rho,**self.boundary)
            for key in old:setattr(self,key,old[key]+relaxation*(step[key]-old[key]))
            current=self.current_residual()
            changes={key:float(np.linalg.norm(getattr(self,key)-old[key])/max(np.linalg.norm(getattr(self,key)),1e-30)) for key in old}
            changes['velocity']=float(np.linalg.norm(self.flow.state-old_u)/max(np.linalg.norm(self.flow.state),1.))
            changes['modified_pressure']=float(np.linalg.norm(self.flow.p-old_p)/max(np.linalg.norm(self.flow.p),1.))
            row=dict(iteration=len(self.history)+1,**{k:v for k,v in current.items() if k!='transport'},
                momentum_method=momentum_method,momentum_coupling=momentum_coupling,
                momentum_subsolve_converged=bool(mr['numerically_converged']),
                momentum_iterations=sum(entry['iteration']>0 for entry in momentum_records),
                momentum_linear_residual_max=max((entry['true_linear_relative_residual'] for entry in
                    momentum_records if 'true_linear_relative_residual' in entry),default=None),
                changes=changes,minimum_k=float(min(self.k.min(),self.k_faces.min())),
                minimum_omega=float(min(self.omega.min(),self.omega_faces.min())),
                frozen_linear_residuals={name:step['transport'][name]['linear_relative_residual'] for name in ('k','omega')},
                scalar_subsolve_ledgers={name:{key:step['transport'][name][key] for key in
                    ('cell_balance_max','interface_flux_defect_max','global_balance','work_balance','work_balance_scale',
                     'relative_work_balance','absolute_work_gate_passed','work_balance_policy')} for name in ('k','omega')})
            self.history.append(row)
            self.converged=bool(max(row[key] for key in ('momentum','divergence','gauge','k','omega','k_boundary','omega_boundary','velocity_boundary'))<tolerance
                                and max(changes.values())<tolerance)
            if callback:callback(row)
            if self.converged:break
        return self.result()

    def result(self):
        if not self.initialized:raise RuntimeError('initialize turbulence first')
        current=self.current_residual();s=self.flow;r=s.result()
        self.converged=bool(self.converged and max(current[key] for key in
            ('momentum','divergence','gauge','k','omega','k_boundary','omega_boundary','velocity_boundary'))<self.tolerance)
        physical=s.p-2/3*self.rho*self.k
        pressure_traction=-physical[self.mesh.owner,None]*self.mesh.area_vectors
        isotropic=-2/3*self.rho*self.k[self.mesh.owner,None]*self.mesh.area_vectors
        viscous=s.viscous.diagnostics(s.state)['face_flux']
        correction=r['total_traction_integrated']-(pressure_traction+isotropic+viscous)
        r.update(numerically_converged=self.converged,physical_benchmark_passed=False,
            pressure=physical.copy(),physical_pressure=physical.copy(),modified_pressure=s.p.copy(),
            k=self.k.copy(),omega=self.omega.copy(),k_faces=self.k_faces.copy(),omega_faces=self.omega_faces.copy(),
            nu_t=self.closure['nu_t'].copy(),initialization=dict(self.initialization),pressure_traction_integrated=pressure_traction,
            isotropic_turbulent_traction_integrated=isotropic,deviatoric_traction_integrated=viscous,
            numerical_traction_correction=correction,current_rans_residual=current,rans_history=list(self.history))
        return r
