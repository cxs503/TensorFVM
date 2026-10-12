"""Energy based native Stokes flow with cell/face velocity and cell pressure.

This explicitly selected backend connects the hybrid viscous operator to
incompressibility. Pressure work uses the negative transpose of the actual
face divergence. No collocated Rhie--Chow reconstruction is involved.
Dirichlet velocity and total outward traction (tau-pI)n are supported.
Convection, SST and wall functions are deliberately outside this interface.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from .native_hybrid_viscous3d import NativeHybridViscous3D


class NativeHybridStokes3D:
    def __init__(self, mesh, velocity_dirichlet, velocity_boundary=0., *,
                 viscosity=1., body_force=0., boundary_traction=0.,
                 gauge_value=0.):
        self.mesh = mesh
        self.viscous = NativeHybridViscous3D(
            mesh, viscosity, dirichlet_mask=velocity_dirichlet)
        v = self.viscous
        self.N, self.F = v.N, v.F
        self.prescribed = v._prescribed(velocity_boundary)
        force = np.broadcast_to(np.asarray(body_force, float), (self.N, 3))
        if not np.isfinite(force).all() or not np.isfinite(gauge_value):
            raise ValueError('finite body force and pressure gauge required')
        self.load = v._traction_load(boundary_traction)
        self.load[:3*self.N] += (mesh.cell_volumes[:, None]*force).ravel()
        faces = np.arange(self.F)
        internal = ~v.boundary
        rows = np.r_[mesh.owner, mesh.neighbor[internal]]
        face_ids = np.r_[faces, faces[internal]]
        vectors = np.vstack((mesh.area_vectors, -mesh.area_vectors[internal]))
        self.divergence = sparse.coo_matrix((vectors.ravel(),
            (np.repeat(rows, 3), (3*(self.N+face_ids[:, None])+np.arange(3)).ravel())),
            shape=(self.N, 3*(self.N+self.F))).tocsr()
        self.B = self.divergence[:, v.free].tocsr()
        self.A = v.matrix
        self.rhs = (self.load-v.full_matrix @ self.prescribed)[v.free]
        self.mass_rhs = -self.divergence @ self.prescribed
        self.gauge = not v.traction_boundary.any()
        self.gauge_value = float(gauge_value)
        if self.gauge and abs(self.mass_rhs.sum()) > 1e-11*max(
                np.abs(self.mass_rhs).sum(), 1.):
            raise ValueError('closed velocity boundary has unbalanced total flux')
        self.state = self.prescribed.copy()
        self.p = np.full(self.N, self.gauge_value if self.gauge else 0.)
        self.history = []
        self.converged = False

    @staticmethod
    def _linear(matrix, rhs):
        answer = splu(matrix.tocsc()).solve(rhs)
        residual = np.linalg.norm(matrix @ answer-rhs)/max(np.linalg.norm(rhs), np.finfo(float).tiny)
        if not np.isfinite(answer).all() or not np.isfinite(residual) or residual > 1e-9:
            raise RuntimeError(f'hybrid Stokes linear solve failed true residual: {residual:g}')
        return answer, float(residual)

    def _row(self, state, pressure, iteration, linear_residual, limited=0):
        unknown = state[self.viscous.free]
        momentum = self.A @ unknown-self.B.T @ pressure-self.rhs
        continuity = self.divergence @ state
        return dict(iteration=iteration, momentum_relative_residual=float(
            np.linalg.norm(momentum)/max(np.linalg.norm(self.rhs), 1.)),
            divergence_max=float(np.max(np.abs(continuity/self.mesh.cell_volumes))),
            true_linear_relative_residual=linear_residual,
            simplec_limited_dofs=limited)

    def solve(self, *, algorithm='simple', relaxation=.5, pressure_relaxation=None,
              max_iterations=3000, tolerance=1e-8, callback=None):
        """Solve native equations by SIMPLE/SIMPLEC or a coupled reference LU.

        Both iterative methods converge to the same joint cell/face equations.
        A failed linear candidate is rejected before changing accepted state.
        """
        if algorithm not in ('simple', 'simplec', 'coupled'):
            raise ValueError('algorithm must be simple, simplec or coupled')
        if pressure_relaxation is None:
            pressure_relaxation = 1. if algorithm == 'simplec' else .3
        if (not np.isfinite([relaxation, pressure_relaxation, tolerance]).all()
                or not 0 < relaxation <= 1 or not 0 < pressure_relaxation <= 1
                or tolerance <= 0 or max_iterations < 1):
            raise ValueError('positive finite tolerances and valid relaxation required')
        v = self.viscous
        size = len(v.free)
        self.converged = False
        if algorithm == 'coupled':
            matrix = sparse.bmat([[self.A, -self.B.T], [self.B, None]], format='lil')
            rhs = np.r_[self.rhs, self.mass_rhs]
            if self.gauge:
                matrix[size, :] = 0.; matrix[size, size] = 1.
                rhs[size] = self.gauge_value
            answer, residual = self._linear(matrix.tocsc(), rhs)
            state = self.prescribed.copy(); state[v.free] = answer[:size]
            pressure = answer[size:]
            row = self._row(state, pressure, 1, residual)
            self.state, self.p = state, pressure
            self.history.append(row)
            self.converged = max(row['momentum_relative_residual'], row['divergence_max']) < tolerance
            if callback: callback(row)
            return self.result()

        diagonal = self.A.diagonal()
        if np.any(diagonal <= 0):
            raise RuntimeError('nonpositive hybrid momentum diagonal')
        relaxed = diagonal/relaxation
        extra = relaxed-diagonal
        inverse = 1/relaxed
        limited = 0
        if algorithm == 'simplec':
            entries = self.A.tocoo()
            same = (entries.row != entries.col) & (entries.row % 3 == entries.col % 3)
            neighbor_sum = np.bincount(entries.row[same], weights=entries.data[same], minlength=size)
            denominator = relaxed+neighbor_sum
            limited = int(np.count_nonzero(denominator < .1*relaxed))
            inverse = 1/np.maximum(denominator, .1*relaxed)
        momentum = (self.A+sparse.diags(extra)).tocsc()
        mfactor = splu(momentum)
        response = sparse.diags(inverse) @ self.B.T
        pressure_matrix = (self.B @ response).tolil()
        if self.gauge:
            pressure_matrix[0, :] = 0.; pressure_matrix[0, 0] = 1.
        pressure_matrix = pressure_matrix.tocsc()
        pfactor = splu(pressure_matrix)
        for iteration in range(max_iterations):
            rhs = self.rhs+self.B.T @ self.p+extra*self.state[v.free]
            predicted = mfactor.solve(rhs)
            prhs = self.mass_rhs-self.B @ predicted
            if self.gauge: prhs[0] = self.gauge_value-self.p[0]
            dp = pfactor.solve(prhs)
            residual = max(np.linalg.norm(momentum @ predicted-rhs)/max(np.linalg.norm(rhs), np.finfo(float).tiny),
                           np.linalg.norm(pressure_matrix @ dp-prhs)/max(np.linalg.norm(prhs), np.finfo(float).tiny))
            if not np.isfinite(predicted).all() or not np.isfinite(dp).all() or not np.isfinite(residual) or residual > 1e-9:
                raise RuntimeError('hybrid SIMPLE linear solve failed true residual; accepted state preserved')
            state = self.prescribed.copy(); state[v.free] = predicted+response @ dp
            pressure = self.p+pressure_relaxation*dp
            if self.gauge: pressure += self.gauge_value-pressure[0]
            if not np.isfinite(state).all() or not np.isfinite(pressure).all():
                raise RuntimeError('nonfinite hybrid SIMPLE candidate; accepted state preserved')
            row = self._row(state, pressure, iteration+1, float(residual), limited)
            self.state, self.p = state, pressure
            self.history.append(row)
            if callback: callback(row)
            if max(row['momentum_relative_residual'], row['divergence_max']) < tolerance:
                self.converged = True
                break
        return self.result()

    def result(self):
        v = self.viscous
        diagnostic = v.diagnostics(self.state)
        # Pressure is cellwise constant. Its work is exactly the adjoint of
        # face divergence; it vanishes at incompressibility, including walls.
        pressure_work = float(self.p @ (self.divergence @ self.state))
        reaction = v.full_matrix @ self.state-self.divergence.T @ self.p-self.load
        # Free boundary traction is in load; Dirichlet reaction is external work.
        external_work = float(self.state @ self.load+self.prescribed @ reaction)
        balance = diagnostic['energy_form']-pressure_work-external_work
        total_traction = diagnostic['face_flux']-self.p[self.mesh.owner, None]*self.mesh.area_vectors
        internal = ~v.boundary
        neighbor_traction = (diagnostic['neighbor_face_flux'][internal]
            +self.p[self.mesh.neighbor[internal], None]*self.mesh.area_vectors[internal])
        return dict(numerically_converged=self.converged, physical_benchmark_passed=False,
            velocity=self.state[:3*self.N].reshape(self.N, 3).copy(),
            face_velocity=self.state[3*self.N:].reshape(self.F, 3).copy(),
            pressure=self.p.copy(), flux=np.sum(self.state[3*self.N:].reshape(self.F, 3)*self.mesh.area_vectors, axis=1),
            energy_form=diagnostic['energy_form'], pressure_work=pressure_work,
            external_work=external_work, energy_balance_error=abs(balance),
            total_traction_integrated=total_traction,
            interface_traction_defect_max=float(np.max(np.abs(
                total_traction[internal]+neighbor_traction), initial=0.)),
            total_momentum_reaction=reaction.reshape(-1, 3), history=list(self.history))
