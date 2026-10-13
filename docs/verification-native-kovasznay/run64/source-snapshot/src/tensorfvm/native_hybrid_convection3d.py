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

    def diagnostics(self, velocity, volume_flux, gradients=None):
        u = np.asarray(velocity, float)
        if u.shape != (self.N, 3) or not np.isfinite(u).all():
            raise ValueError('finite cell velocity required')
        matrix, source, transport, face_source = self.assemble(volume_flux, gradients)
        flux = transport @ u+face_source
        integrated = self.D @ flux
        q = self.density*np.asarray(volume_flux)
        m = self.mesh; i = ~self.boundary; incoming = self.boundary & (q < 0)
        outgoing = self.boundary & (q >= 0)
        jump = u[m.owner[i]]-u[m.neighbor[i]]
        dissipation = .5*np.sum(np.abs(q[i])*np.sum(jump**2, axis=1))
        dissipation += .5*np.sum(-q[incoming]*np.sum((u[m.owner[incoming]]-self.inflow[incoming])**2, axis=1))
        boundary_power = .5*(np.sum(q[outgoing]*np.sum(u[m.owner[outgoing]]**2, axis=1))
                            +np.sum(q[incoming]*np.sum(self.inflow[incoming]**2, axis=1)))
        divergence_power = .5*np.sum((self.D @ q)*np.sum(u*u, axis=1))
        correction_power = (float(np.sum(u*(self.D @ self.deferred_flux(q,gradients))))
                            if self.order == 2 else 0.)
        power = float(np.sum(u*integrated))
        return dict(momentum_flux=flux, integrated=integrated, power=power,
            upwind_dissipation=float(dissipation), boundary_kinetic_energy_flux=float(boundary_power),
            divergence_power=float(divergence_power),
            deferred_correction_power=correction_power,
            energy_identity_error=abs(power-dissipation-boundary_power-divergence_power-correction_power),
            global_momentum_balance_error=float(np.max(np.abs(integrated.sum(0)-flux[self.boundary].sum(0)))),
            assembly_error=float(np.max(np.abs((matrix @ u.ravel()+source).reshape(self.N,3)-integrated))))
