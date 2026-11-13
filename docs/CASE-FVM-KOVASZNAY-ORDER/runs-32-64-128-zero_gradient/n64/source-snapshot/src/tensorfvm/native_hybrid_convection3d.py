"""Shared-face cell upwind momentum, optionally with deferred linear upwind.

Order one has a nonnegative internal upwind energy term. The unbounded order
two correction has signed work; its exact ledger is retained, not called
unconditionally dissipative or positivity preserving.
"""
import numpy as np
from scipy import sparse


class NativeHybridConvection3D:
    def __init__(self, mesh, inflow_velocity, density=1., order=1):
        self.mesh = mesh
        self.N, self.F = len(mesh.cells), len(mesh.owner)
        if not np.isfinite(density) or density <= 0:
            raise ValueError('positive finite density required')
        self.density = float(density)
        if order not in (1,2):
            raise ValueError('convection order must be 1 or 2')
        self.order = order
        self.boundary = mesh.neighbor < 0
        self.inflow = np.broadcast_to(np.asarray(inflow_velocity, float), (self.F, 3)).copy()
        if not np.isfinite(self.inflow[self.boundary]).all():
            raise ValueError('finite inflow/backflow velocity required on boundary faces')
        f = np.arange(self.F); i = ~self.boundary
        self.D = sparse.coo_matrix((np.r_[np.ones(self.F), -np.ones(i.sum())],
            (np.r_[mesh.owner, mesh.neighbor[i]], np.r_[f, f[i]])), shape=(self.N, self.F)).tocsr()

    def assemble(self, volume_flux, gradients=None):
        q = np.asarray(volume_flux, float)*self.density
        if q.shape != (self.F,) or not np.isfinite(q).all():
            raise ValueError('one finite volume flux per face required')
        i = ~self.boundary; active = i | (q >= 0)
        upstream = np.where(q >= 0, self.mesh.owner, self.mesh.neighbor)
        ids = np.flatnonzero(active)
        transport = sparse.coo_matrix((q[ids], (ids, upstream[ids])), shape=(self.F, self.N)).tocsr()
        source = np.zeros((self.F, 3)); incoming = self.boundary & (q < 0)
        source[incoming] = q[incoming, None]*self.inflow[incoming]
        if self.order == 2:
            source += self.deferred_flux(q, gradients)
        matrix = sparse.kron(self.D @ transport, sparse.eye(3), format='csr')
        return matrix, (self.D @ source).ravel(), transport, source

    def hybrid_system(self, volume_flux, mean, gradients=None):
        """Apply cell momentum to the dual of the face mean reconstruction.

        Order one transports R U and has the usual upwind energy identity.
        Order two transports U_K + G_K (x_f-x_K): the cell value is implicit,
        the gradient is deferred. The difference U_K-R_K U and the gradient
        correction have signed work, audited separately in diagnostics.
        """
        matrix, source, _, _ = self.assemble(volume_flux, gradients)
        trial = (mean if self.order == 1 else sparse.hstack(
            (sparse.eye(3*self.N),sparse.csr_matrix((3*self.N,3*self.F))),format='csr'))
        return mean.T @ matrix @ trial, mean.T @ source

    def hybrid_jacobian(self, state, mean):
        """Derivative of the actual flux, including q and Gauss gradients.

        The upwind active set is chosen at the supplied state. At zero mass
        flux this selects a one-sided derivative; line searches must rebuild
        the original nonlinear residual after any active-set change.
        """
        m = self.mesh
        state = np.asarray(state,float)
        if state.shape != (3*(self.N+self.F),) or not np.isfinite(state).all():
            raise ValueError('one finite hybrid vector state required')
        ids = np.arange(self.F)
        q_map = sparse.coo_matrix((self.density*m.area_vectors.ravel(),
            (np.repeat(ids,3),(3*(self.N+ids[:,None])+np.arange(3)).ravel())),
            shape=(self.F,len(state))).tocsr()
        q = q_map @ state
        active = ~self.boundary | (q >= 0)
        upstream = np.where(q >= 0,m.owner,m.neighbor)[active]
        pick = sparse.coo_matrix((np.ones(active.sum()),(ids[active],upstream)),
                                 shape=(self.F,self.N)).tocsr()
        if self.order == 1:
            reconstruction = sparse.kron(pick,sparse.eye(3),format='csr') @ mean
        else:
            offset = np.zeros((self.F,3))
            offset[active] = m.face_centers[active]-m.cell_centers[upstream]
            gradient_part = sparse.csr_matrix((self.F,self.F))
            for j in range(3):
                gradient = sparse.diags(1/m.cell_volumes) @ self.D @ sparse.diags(m.area_vectors[:,j])
                gradient_part += sparse.diags(offset[:,j]) @ pick @ gradient
            reconstruction = sparse.kron(sparse.hstack((pick,gradient_part)),sparse.eye(3),format='csr')
        transported = (reconstruction @ state).reshape(self.F,3)
        transported[~active] = self.inflow[~active]
        # d(q*u_adv) = q*d(u_adv) + u_adv*dq, also for incoming open boundaries.
        flux_response = sparse.coo_matrix((transported.ravel(),
            (np.arange(3*self.F),np.repeat(ids,3))),shape=(3*self.F,self.F)).tocsr()
        face_derivative = sparse.diags(np.repeat(q,3)) @ reconstruction + flux_response @ q_map
        return mean.T @ sparse.kron(self.D,sparse.eye(3),format='csr') @ face_derivative

    def deferred_flux(self, mass_flux, gradients):
        gradient = np.asarray(gradients, float)
        if gradient.shape != (self.N,3,3) or not np.isfinite(gradient).all():
            raise ValueError('finite cell velocity gradients required for linear upwind')
        active = ~self.boundary | (mass_flux >= 0)
        upstream = np.where(mass_flux >= 0, self.mesh.owner, self.mesh.neighbor)[active]
        offset = self.mesh.face_centers[active]-self.mesh.cell_centers[upstream]
        flux = np.zeros((self.F,3))
        flux[active] = mass_flux[active,None]*np.einsum('fij,fj->fi',gradient[upstream],offset)
        return flux

    def diagnostics(self, velocity, volume_flux, gradients=None, *, base_velocity=None):
        u = np.asarray(velocity, float)
        if u.shape != (self.N, 3) or not np.isfinite(u).all():
            raise ValueError('finite cell velocity required')
        base = u if base_velocity is None else np.asarray(base_velocity, float)
        if base.shape != u.shape or not np.isfinite(base).all():
            raise ValueError('finite base cell velocity required')
        matrix, source, transport, face_source = self.assemble(volume_flux, gradients)
        flux = transport @ u+face_source
        integrated = self.D @ flux
        q = self.density*np.asarray(volume_flux)
        m = self.mesh; i = ~self.boundary; incoming = self.boundary & (q < 0)
        outgoing = self.boundary & (q >= 0)
        jump = base[m.owner[i]]-base[m.neighbor[i]]
        dissipation = .5*np.sum(np.abs(q[i])*np.sum(jump**2, axis=1))
        dissipation += .5*np.sum(-q[incoming]*np.sum((base[m.owner[incoming]]-self.inflow[incoming])**2, axis=1))
        boundary_power = .5*(np.sum(q[outgoing]*np.sum(base[m.owner[outgoing]]**2, axis=1))
                            +np.sum(q[incoming]*np.sum(self.inflow[incoming]**2, axis=1)))
        divergence_power = .5*np.sum((self.D @ q)*np.sum(base*base, axis=1))
        correction_power = (float(np.sum(base*(self.D @ self.deferred_flux(q,gradients))))
                            if self.order == 2 else 0.)
        projection_power = float(np.sum(base*(self.D @ (transport @ (u-base)))))
        correction_power += projection_power
        power = float(np.sum(base*integrated))
        return dict(momentum_flux=flux, integrated=integrated, power=power,
            upwind_dissipation=float(dissipation), boundary_kinetic_energy_flux=float(boundary_power),
            divergence_power=float(divergence_power),
            deferred_correction_power=correction_power, projection_correction_power=projection_power,
            energy_identity_error=abs(power-dissipation-boundary_power-divergence_power-correction_power),
            global_momentum_balance_error=float(np.max(np.abs(integrated.sum(0)-flux[self.boundary].sum(0)))),
            assembly_error=float(np.max(np.abs((matrix @ u.ravel()+source).reshape(self.N,3)-integrated))))
