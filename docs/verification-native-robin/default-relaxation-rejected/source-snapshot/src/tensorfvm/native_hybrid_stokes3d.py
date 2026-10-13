"""Energy based native Stokes flow with cell/face velocity and cell pressure.

This explicitly selected backend connects the hybrid viscous operator to
incompressibility. Pressure work uses the negative transpose of the actual
face divergence. No collocated Rhie--Chow reconstruction is involved.
Dirichlet velocity, total outward traction (tau-pI)n and dissipative Robin
conditions (tau-pI)n + C u = g are supported. For Robin faces, the
boundary_traction argument supplies g, including C u_wall for a moving wall.
Convection, SST and wall functions are deliberately outside this interface.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu, gmres, LinearOperator

from .native_hybrid_viscous3d import NativeHybridViscous3D
from .native_face_mean3d import NativeFaceMean3D
from .native_boundary_constraints3d import require_rigid_motion_constraints
from .native_block_solver3d import NativeHybridBlockSolver3D, NativeBlockSolveError


class NativeHybridStokes3D:
    def __init__(self, mesh, velocity_dirichlet, velocity_boundary=0., *,
                 viscosity=1., body_force=0., boundary_traction=0.,
                 gauge_value=0., body_force_discretization='face', coupled_linear_method='direct',
                 gradient_jump_stabilization=1., momentum_relaxation_basis='diagonal',robin_coefficient=None):
        if coupled_linear_method not in ('direct', 'reused_lu', 'block_amg'):
            raise ValueError('coupled_linear_method must be direct, reused_lu or block_amg')
        if momentum_relaxation_basis not in ('diagonal','absolute_row_sum'):
            raise ValueError('momentum_relaxation_basis must be diagonal or absolute_row_sum')
        self.momentum_relaxation_basis = momentum_relaxation_basis
        self.coupled_linear_method = coupled_linear_method
        self._coupled_factor = None
        self._block_preconditioner = None
        self.coupled_linear_history = []
        self.mesh = mesh
        self.gradient_jump_stabilization = gradient_jump_stabilization
        self.viscous = NativeHybridViscous3D(
            mesh, viscosity, dirichlet_mask=velocity_dirichlet,
            gradient_jump_stabilization=gradient_jump_stabilization,robin_coefficient=robin_coefficient)
        v = self.viscous
        self.rigid_motion_constraints = require_rigid_motion_constraints(mesh, v.dirichlet_components,v.robin_coefficient)
        self.N, self.F = v.N, v.F
        self.prescribed = v._prescribed(velocity_boundary)
        self.face_mean = NativeFaceMean3D(mesh)
        force = np.broadcast_to(np.asarray(body_force, float), (self.N, 3))
        if not np.isfinite(force).all() or not np.isfinite(gauge_value):
            raise ValueError('finite body force and pressure gauge required')
        self.load = v._traction_load(boundary_traction)
        self.body_force = force.copy()
        self.body_force_discretization = body_force_discretization
        self.owner_force_flux = np.zeros((self.F,3))
        self.neighbor_force_flux = np.zeros((self.F,3))
        if body_force_discretization == 'cell':
            self.load[:3*self.N] += (mesh.cell_volumes[:, None]*force).ravel()
        elif body_force_discretization == 'face':
            # Dual load of the divergence-conforming mean velocity
            # R_K(U)=sum_f r_Kf (U_f dot S_Kf)/V_K. Constant forces have
            # exactly the same face response as a linear pressure gradient.
            # The geometric moment identity sum S r^T=V I preserves the
            # integrated cell load. Force fluxes must also enter traction.
            self.owner_force_flux, self.neighbor_force_flux = self.face_mean.moments(
                mesh.cell_volumes[:,None]*force)
            self.load += self.face_mean.matrix.T @ (mesh.cell_volumes[:,None]*force).ravel()
        else:
            raise ValueError('body_force_discretization must be cell or face')
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
        # Pressure has a constant nullspace if every boundary NORMAL flux
        # is fixed; free tangential components on a symmetry plane do not
        # determine pressure. Component masks support axis-aligned symmetry.
        area = np.linalg.norm(mesh.area_vectors,axis=1)
        free_normal = np.abs(mesh.area_vectors)*v.traction_components
        self.open_normal_boundary = np.any(free_normal > 1e-12*area[:,None],axis=1)
        self.gauge = not self.open_normal_boundary.any()
        self.gauge_value = float(gauge_value)
        if self.gauge and abs(self.mass_rhs.sum()) > 1e-11*max(
                np.abs(self.mass_rhs).sum(), 1.):
            raise ValueError('closed velocity boundary has unbalanced total flux')
        self.state = self.prescribed.copy()
        self.p = np.full(self.N, self.gauge_value if self.gauge else 0.)
        self.history = []
        self.converged = False

    def update_dynamic_viscosity(self, viscosity):
        """Rebuild physical stress and invalidate all viscosity-dependent solves.

        Accepted velocity/pressure and prescribed loads remain the initial
        iterate. A previous convergence flag never survives a coefficient change.
        Construction is validated before changing the accepted operator.
        """
        new = NativeHybridViscous3D(self.mesh,viscosity,
            dirichlet_mask=self.viscous.dirichlet_components,
            gradient_jump_stabilization=self.gradient_jump_stabilization,robin_coefficient=self.viscous.robin_coefficient)
        self.viscous = new
        self.A = new.matrix
        self.rhs = (self.load-new.full_matrix @ self.prescribed)[new.free]
        self._coupled_factor = None
        self._block_preconditioner = None
        self.converged = False
        if hasattr(self,'nonlinear_converged'):
            self.nonlinear_converged = False

    @staticmethod
    def _linear(matrix, rhs):
        answer = splu(matrix.tocsc()).solve(rhs)
        residual = np.linalg.norm(matrix @ answer-rhs)/max(np.linalg.norm(rhs), np.finfo(float).tiny)
        if not np.isfinite(answer).all() or not np.isfinite(residual) or residual > 1e-9:
            raise RuntimeError(f'hybrid Stokes linear solve failed true residual: {residual:g}')
        return answer, float(residual)

    def _coupled_linear(self, matrix, rhs):
        """Reuse an earlier LU only as a preconditioner for the CURRENT matrix.

        Every candidate is checked against the current true residual. A stale
        preconditioner may slow convergence, but cannot qualify a wrong solve.
        Refresh by direct factorization if GMRES fails its residual check.
        """
        if self.coupled_linear_method == 'block_amg':
            if self._block_preconditioner is None:
                self._block_preconditioner = NativeHybridBlockSolver3D(self)
            try:
                answer,residual,record = self._block_preconditioner.solve(matrix,rhs)
            except NativeBlockSolveError as exc:
                self.coupled_linear_history.append(exc.record)
                raise
            self.coupled_linear_history.append(record)
            return answer,residual
        matrix = matrix.tocsc()
        iterations = 0
        reused = self.coupled_linear_method == 'reused_lu' and self._coupled_factor is not None
        fallback = False
        refinements=[]
        scale = max(np.linalg.norm(rhs), np.finfo(float).tiny)
        if reused:
            def count(_):
                nonlocal iterations
                iterations += 1
            preconditioner = LinearOperator(matrix.shape, matvec=self._coupled_factor.solve)
            answer, info = gmres(matrix, rhs, M=preconditioner, rtol=1e-11,
                atol=0., restart=40, maxiter=4, callback=count, callback_type='pr_norm')
            residual = np.linalg.norm(matrix @ answer-rhs)/scale
            fallback = bool(info != 0 or not np.isfinite(answer).all()
                            or not np.isfinite(residual) or residual > 1e-9)
        if not reused or fallback:
            factor = splu(matrix)
            answer = factor.solve(rhs)
            # A small global residual can conceal divergence in tiny cells.
            # Refine the SAME saddle equations using extended residuals and
            # the existing LU. Keep float64 states and all outer gates.
            nv=len(self.viscous.free)
            def quality(candidate):
                defect=matrix@candidate-rhs
                relative=float(np.linalg.norm(defect)/scale)
                if len(defect)!=nv+self.N:return relative
                constraint=defect[nv:]/self.mesh.cell_volumes
                if self.gauge:constraint=constraint.copy();constraint[0]=defect[nv]
                return max(relative,float(np.max(np.abs(constraint),initial=0.)))
            merit=quality(answer)
            if merit>1e-11:
                precise=matrix.astype(np.longdouble);precise_rhs=rhs.astype(np.longdouble)
                for iteration in range(3):
                    defect=precise_rhs-precise@answer.astype(np.longdouble)
                    candidate=answer+factor.solve(np.asarray(defect,float))
                    after=quality(candidate)
                    accepted=bool(np.isfinite(candidate).all() and after<merit)
                    refinements.append(dict(before=merit,after=after,accepted=accepted))
                    if not accepted:break
                    answer=candidate;merit=after
                    if merit<=1e-11:break
            residual = np.linalg.norm(matrix @ answer-rhs)/scale
            if self.coupled_linear_method == 'reused_lu':
                self._coupled_factor = factor
        if not np.isfinite(answer).all() or not np.isfinite(residual) or residual > 1e-9:
            raise RuntimeError(f'hybrid coupled solve failed true residual: {residual:g}')
        self.coupled_linear_history.append(dict(reused_preconditioner=reused,
            fallback_factorization=fallback, gmres_iterations=iterations,
            residual_refinement=refinements,
            true_relative_residual=float(residual)))
        return answer, float(residual)

    def _row(self, state, pressure, iteration, linear_residual, limited=0):
        unknown = state[self.viscous.free]
        momentum = self.A @ unknown-self.B.T @ pressure-self.rhs
        continuity = self.divergence @ state
        return dict(iteration=iteration, momentum_relative_residual=float(
            np.linalg.norm(momentum)/max(np.linalg.norm(self.rhs), 1.)),
            divergence_max=float(np.max(np.abs(continuity/self.mesh.cell_volumes))),
            true_linear_relative_residual=linear_residual,
            simplec_limited_dofs=limited, momentum_relaxation_basis=self.momentum_relaxation_basis)

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
            answer, residual = self._coupled_linear(matrix.tocsc(), rhs)
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
        # Projected inertia need not be diagonally dominant. A row-norm
        # pseudo-time diagonal stabilizes the segregated iteration without
        # altering its fixed point: the same extra term goes on both sides.
        basis = (np.asarray(abs(self.A).sum(axis=1)).ravel()
                 if self.momentum_relaxation_basis == 'absolute_row_sum' else diagonal)
        relaxed = basis/relaxation
        extra = relaxed-diagonal
        inverse = 1/relaxed
        limited = 0
        if algorithm == 'simplec':
            entries = self.A.tocoo()
            same = (entries.row != entries.col) & (v.free[entries.row] % 3 == v.free[entries.col] % 3)
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
        total_energy=diagnostic['energy_form']+diagnostic['robin_power']
        balance = total_energy-pressure_work-external_work
        total_traction = (diagnostic['face_flux']-self.p[self.mesh.owner, None]*self.mesh.area_vectors
                          -self.owner_force_flux)
        internal = ~v.boundary
        neighbor_traction = (diagnostic['neighbor_face_flux'][internal]
            +self.p[self.mesh.neighbor[internal], None]*self.mesh.area_vectors[internal]
            -self.neighbor_force_flux[internal])
        return dict(numerically_converged=self.converged, physical_benchmark_passed=False,
            velocity=self.state[:3*self.N].reshape(self.N, 3).copy(),
            face_velocity=self.state[3*self.N:].reshape(self.F, 3).copy(),
            pressure=self.p.copy(), flux=np.sum(self.state[3*self.N:].reshape(self.F, 3)*self.mesh.area_vectors, axis=1),
            energy_form=total_energy,bulk_viscous_energy=diagnostic['energy_form'],
            robin_power=diagnostic['robin_power'],robin_reaction_integrated=diagnostic['robin_reaction_integrated'],
            gradient_jump_energy=diagnostic['gradient_jump_energy'],
            pressure_work=pressure_work,
            external_work=external_work, energy_balance_error=abs(balance),
            total_traction_integrated=total_traction,
            neighbor_total_traction_integrated=neighbor_traction,
            interface_traction_defect_max=float(np.max(np.abs(
                total_traction[internal]+neighbor_traction), initial=0.)),
            total_momentum_reaction=reaction.reshape(-1, 3), history=list(self.history))
