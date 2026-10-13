"""Experimental energy-based hybrid FV diffusion on native polyhedral cells.

Cell and shared-face unknowns are independent. Face equilibrium enforces a
single conservative flux. Boundary faces may prescribe values or outward
traction (scalar diffusive flux for components=1).

The scalar construction follows the hybrid FV gradient/residual framework
of Droniou et al., https://arxiv.org/abs/0812.2097. The symmetric-deviatoric
vector extension below is a project implementation, separately verified.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu, cg


class NativeHybridViscous3D:
    def __init__(self, mesh, dynamic_viscosity=1., *, components=3,
                 deviatoric=True, stabilization=1., dirichlet_mask=None):
        if components not in (1, 3):
            raise ValueError('components must be 1 or 3')
        if not np.isfinite(stabilization) or stabilization <= 0:
            raise ValueError('positive finite stabilization required')
        self.mesh = mesh
        self.N, self.F, self.components = len(mesh.cells), len(mesh.owner), components
        mu = np.broadcast_to(np.asarray(dynamic_viscosity, float), (self.N,)).copy()
        if not np.isfinite(mu).all() or np.any(mu <= 0):
            raise ValueError('positive finite cell viscosity required')
        self.mu = mu
        self.boundary = mesh.neighbor < 0
        self.internal_faces = np.flatnonzero(~self.boundary)
        mask = self.boundary if dirichlet_mask is None else np.asarray(dirichlet_mask, bool)
        if mask.shape == (self.F,):
            mask = np.broadcast_to(mask[:,None], (self.F,components))
        if mask.shape != (self.F,components) or np.any(mask & ~self.boundary[:,None]):
            raise ValueError('Dirichlet mask must select boundary faces or their components')
        self.dirichlet_components = mask.copy()
        self.dirichlet = mask.all(axis=1)
        self.traction_components = self.boundary[:,None] & ~mask
        self.traction_boundary = self.traction_components.any(axis=1)
        self.free_faces = np.flatnonzero(~self.dirichlet)
        self.local = []
        face_ids = [[] for _ in range(self.N)]
        signs = [[] for _ in range(self.N)]
        for face, (owner, neighbor) in enumerate(zip(mesh.owner, mesh.neighbor)):
            face_ids[owner].append(face); signs[owner].append(1.)
            if neighbor >= 0:
                face_ids[neighbor].append(face); signs[neighbor].append(-1.)
        # C maps a row-major velocity gradient to symmetric viscous stress.
        constitutive = np.empty((9, 9))
        for column in range(9):
            gradient = np.eye(9)[column].reshape(3, 3)
            stress = gradient + gradient.T
            if deviatoric:
                stress -= (2/3)*np.trace(gradient)*np.eye(3)
            constitutive[:, column] = stress.ravel()
        rows, columns, values = [], [], []
        for cell in range(self.N):
            faces = np.asarray(face_ids[cell], int)
            sign = np.asarray(signs[cell])
            area_vector = sign[:, None]*mesh.area_vectors[faces]
            area = np.linalg.norm(area_vector, axis=1)
            offset = mesh.face_centers[faces]-mesh.cell_centers[cell]
            height = np.sum(offset*area_vector, axis=1)/area
            if np.any(height <= 0):
                raise ValueError('hybrid FV requires positive cell-to-face normal heights')
            jump = np.column_stack((-np.ones(len(faces)), np.eye(len(faces))))
            gradient = area_vector.T @ jump / mesh.cell_volumes[cell]
            residual = jump-offset @ gradient
            penalty = stabilization*mu[cell]*area/height
            scalar_penalty = residual.T @ (penalty[:, None]*residual)
            if components == 1:
                local = mu[cell]*mesh.cell_volumes[cell]*(gradient.T @ gradient)+scalar_penalty
            else:
                # local unknowns: cell, faces, with interleaved components.
                strain = np.zeros((9, 3*(len(faces)+1)))
                for component in range(3):
                    strain[3*component:3*component+3, component::3] = gradient
                local = mu[cell]*mesh.cell_volumes[cell]*(strain.T @ constitutive @ strain)
                local += np.kron(scalar_penalty, np.eye(3))
            local = (local+local.T)*.5
            nodes = np.r_[cell, self.N+faces]
            ids = (components*nodes[:, None]+np.arange(components)).ravel()
            rows.append(np.repeat(ids, len(ids)))
            columns.append(np.tile(ids, len(ids)))
            values.append(local.ravel())
            self.local.append((faces, sign, ids, local, gradient, residual, penalty))
        size = components*(self.N+self.F)
        self.full_matrix = sparse.coo_matrix((np.concatenate(values),
            (np.concatenate(rows), np.concatenate(columns))), shape=(size, size)).tocsr()
        self.full_matrix.eliminate_zeros()
        free_mask = np.vstack((np.ones((self.N,components),bool),~self.dirichlet_components))
        self.free = np.flatnonzero(free_mask.ravel())
        self.matrix = self.full_matrix[self.free][:, self.free].tocsc()
        self.cell_dofs = components*self.N
        self._face_factor = None

    def _prescribed(self, boundary_values):
        boundary = np.broadcast_to(np.asarray(boundary_values, float), (self.F, self.components))
        if not np.isfinite(boundary[self.dirichlet_components]).all():
            raise ValueError('finite Dirichlet boundary values required')
        state = np.zeros((self.N+self.F, self.components))
        state[self.N:][self.dirichlet_components] = boundary[self.dirichlet_components]
        return state.ravel()

    def _traction_load(self, boundary_traction):
        """Prescribed outward stress times normal, per unit face area."""
        traction = np.broadcast_to(np.asarray(boundary_traction, float), (self.F, self.components))
        if not np.isfinite(traction[self.traction_components]).all():
            raise ValueError('finite boundary traction required')
        load = np.zeros((self.N+self.F, self.components))
        integrated = np.linalg.norm(self.mesh.area_vectors,axis=1)[:,None]*traction
        load[self.N:][self.traction_components] = integrated[self.traction_components]
        return load.ravel()

    def equilibrate_faces(self, cell_values, boundary_values=0., *, boundary_traction=0., linear_method='direct'):
        """Equilibrate free faces, including prescribed traction boundaries."""
        cells = np.asarray(cell_values, float)
        if cells.shape != (self.N, self.components) or not np.isfinite(cells).all():
            raise ValueError('finite cell values with explicit component dimension required')
        state = self._prescribed(boundary_values)
        state[:self.cell_dofs] = cells.ravel()
        faces = self.free[self.cell_dofs:]
        iterations = 0
        if len(faces):
            block = self.full_matrix[faces][:, faces].tocsc()
            rhs = (self._traction_load(boundary_traction)-self.full_matrix @ state)[faces]
            if linear_method == 'direct':
                if self._face_factor is None:
                    self._face_factor = splu(block)
                answer = self._face_factor.solve(rhs)
            elif linear_method == 'cg':
                def count(value):
                    nonlocal iterations
                    iterations += 1
                answer, info = cg(block, rhs, M=sparse.diags(1/block.diagonal()),
                                  rtol=1e-11, atol=0., maxiter=10000, callback=count)
                if info != 0:
                    raise RuntimeError(f'hybrid face equilibrium CG did not converge: {info}')
            else:
                raise ValueError('hybrid face linear_method must be direct or cg')
            relative = np.linalg.norm(block @ answer-rhs)/max(np.linalg.norm(rhs), 1e-30)
            if not np.isfinite(answer).all() or relative > 1e-9:
                raise RuntimeError(f'hybrid face equilibrium failed true residual: {relative:g}')
            state[faces] = answer
        else:
            relative = 0.
        result = self.diagnostics(state)
        result.update(face_linear_relative_residual=float(relative), face_iterations=iterations)
        return result

    def solve(self, body_force=0., boundary_values=0., *, boundary_traction=0., linear_method='direct'):
        """Solve -div(tau)=force (or -div(mu grad u)=force for a scalar)."""
        force = np.broadcast_to(np.asarray(body_force, float), (self.N, self.components))
        if not np.isfinite(force).all():
            raise ValueError('finite cell body force required')
        state = self._prescribed(boundary_values)
        rhs = (self._traction_load(boundary_traction)-self.full_matrix @ state)[self.free]
        rhs[:self.cell_dofs] += (self.mesh.cell_volumes[:, None]*force).ravel()
        if linear_method == 'direct':
            answer = splu(self.matrix).solve(rhs)
        elif linear_method == 'cg':
            answer, info = cg(self.matrix, rhs, M=sparse.diags(1/self.matrix.diagonal()),
                              rtol=1e-11, atol=0., maxiter=10000)
            if info != 0:
                raise RuntimeError(f'hybrid diffusion CG did not converge: {info}')
        else:
            raise ValueError('hybrid linear_method must be direct or cg')
        relative = np.linalg.norm(self.matrix @ answer-rhs)/max(np.linalg.norm(rhs), 1e-30)
        if not np.isfinite(answer).all() or relative > 1e-9:
            raise RuntimeError(f'hybrid diffusion failed true residual: {relative:g}')
        state[self.free] = answer
        result = self.diagnostics(state)
        result['linear_relative_residual'] = float(np.linalg.norm(self.matrix @ state[self.free]-rhs)/max(np.linalg.norm(rhs), 1e-30))
        return result

    def diagnostics(self, state):
        """Replay local face fluxes; interior equilibrium must be checked."""
        state = np.asarray(state, float)
        reaction = self.full_matrix @ state
        owner_flux = np.zeros((self.F, self.components))
        neighbor_flux = np.zeros_like(owner_flux)
        energy = 0.
        for cell, (faces, sign, ids, local, gradient, residual, penalty) in enumerate(self.local):
            local_state = state[ids]
            local_reaction = (local @ local_state).reshape(-1, self.components)
            owner_flux[faces[sign > 0]] = local_reaction[1:][sign > 0]
            neighbor_flux[faces[sign < 0]] = local_reaction[1:][sign < 0]
            energy += float(local_state @ local @ local_state)
        cells = state[:self.cell_dofs].reshape(self.N, self.components)
        interface_defect = owner_flux[~self.boundary]+neighbor_flux[~self.boundary]
        boundary_work = float(np.sum(state.reshape(-1, self.components)[self.N:][self.boundary]*owner_flux[self.boundary]))
        rate = -float(cells.ravel() @ reaction[:self.cell_dofs])
        return dict(cell_values=cells.copy(), face_values=state[self.cell_dofs:].reshape(self.F, self.components).copy(),
                    face_flux=owner_flux, neighbor_face_flux=neighbor_flux,
                    energy_form=energy, kinetic_energy_rate=rate,
                    boundary_work=boundary_work,
                    energy_balance_error=abs(rate+energy-boundary_work),
                    interface_flux_defect_max=float(np.max(np.abs(interface_defect), initial=0.)),
                    cell_reaction=reaction[:self.cell_dofs].reshape(self.N, self.components))
