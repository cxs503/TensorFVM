"""Native laminar Navier--Stokes with hybrid viscosity and shared-face advection.

Picard outer iterations freeze the conservative face volume flux. Each inner
SIMPLE/SIMPLEC or coupled solve uses the shared hybrid Stokes infrastructure.
Only a residual rebuilt from CURRENT face velocity can qualify convergence.
Optional linear upwind reconstruction uses the current hybrid Gauss gradient.
SST and wall functions are not implemented by this backend.
"""
import numpy as np
from scipy import sparse
from .native_hybrid_stokes3d import NativeHybridStokes3D
from .native_hybrid_convection3d import NativeHybridConvection3D


class NativeHybridNavierStokes3D(NativeHybridStokes3D):
    def __init__(self, mesh, velocity_dirichlet, velocity_boundary=0., *,
                 density=1., backflow_velocity=None, convection_order=1, **kwargs):
        super().__init__(mesh, velocity_dirichlet, velocity_boundary, **kwargs)
        inflow = np.broadcast_to(np.asarray(velocity_boundary, float), (self.F,3)).copy()
        open_faces = self.viscous.traction_boundary
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

    def _current_system(self):
        cell_matrix, cell_source, _, _ = self.convection.assemble(self.volume_flux(), self.velocity_gradient())
        convection = sparse.block_diag((cell_matrix, sparse.csr_matrix((3*self.F,3*self.F))), format='csr')
        source = np.r_[cell_source, np.zeros(3*self.F)]
        full = self.viscous.full_matrix+convection
        return full, source

    def velocity_gradient(self):
        face = self.state[3*self.N:].reshape(self.F,3)
        return np.stack([self.convection.D @ (face*self.mesh.area_vectors[:,j,None])
                         /self.mesh.cell_volumes[:,None] for j in range(3)], axis=2)

    def solve(self, *, algorithm='simplec', max_outer_iterations=40,
              max_iterations=3000, tolerance=1e-8, relaxation=.5,
              pressure_relaxation=None, callback=None, momentum_callback=None):
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
        adv = self.convection.diagnostics(result['velocity'], result['flux'], self.velocity_gradient())
        reaction = full @ self.state+source-self.divergence.T @ self.p-self.load
        external = float(self.state @ self.load+self.prescribed @ reaction)
        result.update(numerically_converged=self.nonlinear_converged,
            convection_power=adv['power'], upwind_dissipation=adv['upwind_dissipation'],
            deferred_correction_power=adv['deferred_correction_power'],
            boundary_kinetic_energy_flux=adv['boundary_kinetic_energy_flux'],
            convection_energy_identity_error=adv['energy_identity_error'],
            convection_global_balance_error=adv['global_momentum_balance_error'],
            external_work=external,
            energy_balance_error=abs(result['energy_form']+adv['power']-result['pressure_work']-external),
            momentum_flux_integrated=adv['momentum_flux']-result['total_traction_integrated'],
            total_momentum_reaction=reaction.reshape(-1,3),
            outer_history=list(self.outer_history))
        return result
