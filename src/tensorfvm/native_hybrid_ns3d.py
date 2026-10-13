"""Native laminar Navier--Stokes with hybrid viscosity and shared-face advection.

Picard outer iterations freeze the conservative face volume flux. Each inner
SIMPLE/SIMPLEC or coupled solve uses the shared hybrid Stokes infrastructure.
Only a residual rebuilt from CURRENT face velocity can qualify convergence.
Linear upwind reconstruction uses the current hybrid Gauss gradient; its
signed correction work is explicitly audited.
SST and wall functions are not implemented by this backend.
"""
import numpy as np
from .native_hybrid_stokes3d import NativeHybridStokes3D
from .native_hybrid_convection3d import NativeHybridConvection3D
from .native_hybrid_newton3d import solve_newton


class NativeHybridNavierStokes3D(NativeHybridStokes3D):
    def __init__(self, mesh, velocity_dirichlet, velocity_boundary=0., *,
                 density=1., backflow_velocity=None, convection_order=1, **kwargs):
        kwargs.setdefault('momentum_relaxation_basis','absolute_row_sum')
        super().__init__(mesh, velocity_dirichlet, velocity_boundary, **kwargs)
        inflow = np.broadcast_to(np.asarray(velocity_boundary, float), (self.F,3)).copy()
        open_faces = self.open_normal_boundary
        if open_faces.any():
            if backflow_velocity is None:
                raise ValueError('explicit backflow_velocity required on total traction boundaries')
            inflow[open_faces] = np.broadcast_to(np.asarray(backflow_velocity,float), (self.F,3))[open_faces]
        self.convection = NativeHybridConvection3D(mesh, inflow, density, convection_order)
        self.outer_history = []
        self.nonlinear_converged = False

    def volume_flux(self, state=None):
        state = self.state if state is None else state
        return np.sum(state[3*self.N:].reshape(self.F,3)*self.mesh.area_vectors, axis=1)

    def _current_system(self, state=None):
        convection, source = self.convection.hybrid_system(
            self.volume_flux(state),self.face_mean.matrix,self.velocity_gradient(state))
        return self.viscous.full_matrix+convection, source

    def velocity_gradient(self, state=None):
        state = self.state if state is None else state
        face = state[3*self.N:].reshape(self.F,3)
        return np.stack([self.convection.D @ (face*self.mesh.area_vectors[:,j,None])
                         /self.mesh.cell_volumes[:,None] for j in range(3)], axis=2)

    def initialize_stokes(self, *, tolerance=1e-9):
        """Compute an incompressible native Stokes initial guess, without exact fields.

        This changes the initial iterate, not the Navier--Stokes equations.
        Rebuild the viscous system so repeated use after NS is well defined.
        """
        self.A = self.viscous.matrix
        self.rhs = (self.load-self.viscous.full_matrix @ self.prescribed)[self.viscous.free]
        self._coupled_factor = None
        NativeHybridStokes3D.solve(self,algorithm='coupled',tolerance=tolerance)
        self._coupled_factor = None
        self.nonlinear_converged = False
        self.initialization_history = dict(method='native_stokes',converged=self.converged,
                                           **self.history[-1])
        if not self.converged:raise RuntimeError('Stokes initialization failed convergence check')
        return dict(self.initialization_history)

    def solve_newton(self, *, max_iterations=50, tolerance=1e-8,
                     max_backtracks=20, callback=None, allow_inexact_linear_steps=False, require_initial_correction=False):
        """Optional coupled Newton reference solve of the same native equations.

        SIMPLE/SIMPLEC retain their Picard interface. This method uses a full
        sparse Jacobian and residual line search, not a segregated method.
        """
        return solve_newton(self,max_iterations=max_iterations,tolerance=tolerance,
                            max_backtracks=max_backtracks,callback=callback,
                            allow_inexact_linear_steps=allow_inexact_linear_steps,
                            require_initial_correction=require_initial_correction)

    def solve(self, *, algorithm='simplec', max_outer_iterations=40,
              max_iterations=3000, tolerance=1e-8, relaxation=.5,
              pressure_relaxation=None, outer_relaxation=1., callback=None, momentum_callback=None):
        if not np.isfinite(outer_relaxation) or not 0 < outer_relaxation <= 1:
            raise ValueError('outer_relaxation must be in (0, 1]')
        if max_outer_iterations < 1:
            raise ValueError('positive outer iteration limit required')
        self.nonlinear_converged = False
        for outer in range(max_outer_iterations):
            old = self.state.copy(); old_p = self.p.copy()
            full, source = self._current_system()
            free = self.viscous.free
            self.A = full[free][:,free].tocsc()
            self.rhs = (self.load-source-full @ self.prescribed)[free]
            # Reuse the tested linear algorithms; retain all inner records.
            start = len(self.history)
            super().solve(algorithm=algorithm, relaxation=relaxation,
                pressure_relaxation=pressure_relaxation, max_iterations=max_iterations,
                tolerance=tolerance*.1, callback=momentum_callback)
            # Relax the nonlinear candidate, then audit the accepted CURRENT
            # equations. Prescribed values and pressure gauge are unchanged.
            self.state = old+outer_relaxation*(self.state-old)
            self.p = old_p+outer_relaxation*(self.p-old_p)
            full, source = self._current_system()
            residual = full @ self.state+source-self.divergence.T @ self.p-self.load
            scale = max(np.linalg.norm((self.load-source-full @ self.prescribed)[free]),1.)
            row = dict(iteration=outer+1, inner_iterations=len(self.history)-start,
                frozen_inner_converged=self.converged,
                current_momentum_residual=float(np.linalg.norm(residual[free])/scale),
                divergence_max=float(np.max(np.abs(self.divergence @ self.state/self.mesh.cell_volumes))),
                velocity_change=float(np.linalg.norm(self.state-old)/max(np.linalg.norm(self.state),1.)),
                pressure_change=float(np.linalg.norm(self.p-old_p)/max(np.linalg.norm(self.p),1.)))
            self.outer_history.append(row)
            self.nonlinear_converged = bool(self.converged and max(row[key] for key in (
                'current_momentum_residual','divergence_max','velocity_change','pressure_change')) < tolerance)
            if callback: callback(row)
            if self.nonlinear_converged: break
        return self.result()

    def result(self):
        result = super().result()
        full, source = self._current_system()
        mean_velocity = (self.face_mean.matrix @ self.state).reshape(self.N,3)
        transported = result['velocity'] if self.convection.order == 2 else mean_velocity
        adv = self.convection.diagnostics(transported, result['flux'], self.velocity_gradient(),
                                         base_velocity=mean_velocity)
        # Inertia now contributes face test-function reactions. Include its
        # dual moments in numerical traction, just as body-force moments are
        # subtracted. Omitting these produces a false interface force ledger.
        owner_inertia, neighbor_inertia = self.face_mean.moments(adv['integrated'])
        total_traction = result['total_traction_integrated']+owner_inertia
        internal = ~self.viscous.boundary
        neighbor_traction = result['neighbor_total_traction_integrated']+neighbor_inertia[internal]
        momentum_flux = adv['momentum_flux']-total_traction
        local_balance = self.convection.D @ momentum_flux-self.mesh.cell_volumes[:,None]*self.body_force
        reaction = full @ self.state+source-self.divergence.T @ self.p-self.load
        external = float(self.state @ self.load+self.prescribed @ reaction)
        result.update(numerically_converged=self.nonlinear_converged,
            reconstructed_velocity=mean_velocity,
            total_traction_integrated=total_traction,
            neighbor_total_traction_integrated=neighbor_traction,
            interface_traction_defect_max=float(np.max(np.abs(
                total_traction[internal]+neighbor_traction),initial=0.)),
            local_momentum_balance_max=float(np.max(np.abs(local_balance),initial=0.)),
            global_momentum_balance_max=float(np.max(np.abs(local_balance.sum(0)),initial=0.)),
            projection_correction_power=adv['projection_correction_power'],
            convection_power=adv['power'], upwind_dissipation=adv['upwind_dissipation'],
            deferred_correction_power=adv['deferred_correction_power'],
            boundary_kinetic_energy_flux=adv['boundary_kinetic_energy_flux'],
            convection_energy_identity_error=adv['energy_identity_error'],
            convection_global_balance_error=adv['global_momentum_balance_error'],
            external_work=external,
            energy_balance_error=abs(result['energy_form']+adv['power']-result['pressure_work']-external),
            momentum_flux_integrated=momentum_flux,
            total_momentum_reaction=reaction.reshape(-1,3),
            outer_history=list(self.outer_history))
        return result
