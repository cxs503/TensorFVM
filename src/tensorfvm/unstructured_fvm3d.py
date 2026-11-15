"""Cell-centred finite-volume operators on 3-D unstructured tetrahedral meshes.

The stencil is the owner/neighbour face structure of
:class:`tensorfvm.native_mesh3d.NativeMesh3D`; every operator below is written
against that structure only, so nothing here depends on logical (i,j,k) order.

Operators
---------
``least_squares_gradient``   inverse-distance weighted least-squares gradient
                             reconstruction from cell values plus explicit
                             boundary-face data.
``diffusion_matrix``         over-relaxed (or minimum-correction) decomposition
                             of the area-integrated diffusive flux
                             ``Gamma * grad(phi) . s``.  The implicit part keeps
                             ``|s|^2 / (s . d)`` on the diagonal; the
                             non-orthogonal remainder ``s - coef * d`` is a
                             deferred correction evaluated from the current
                             gradient.
``poisson``                  ``div(Gamma grad phi) = f`` with Dirichlet data on
                             every boundary face, iterated to the fixed point of
                             the non-orthogonal correction.
``advection``                steady convection with upwind, central or limited
                             face weighting, also as a deferred correction on the
                             (M-matrix) upwind operator.

The deferred-correction residual history is returned with every solve: a solve
that did not converge is reported as such, not silently truncated.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from .native_mesh3d import NativeMesh3D


#: Face-weighting schemes of :meth:`UnstructuredFVM3D.advection`.
SCHEMES = ("upwind", "central", "limited")

#: Flux decompositions of :class:`UnstructuredFVM3D`.  ``over-relaxed`` is the
#: standard OpenFOAM choice: it keeps the implicit coefficient largest and the
#: correction vector smallest for a skewed face, at the price of a tighter
#: correction loop.
DECOMPOSITIONS = ("over-relaxed", "orthogonal")


def discrete_l2_error(mesh, values, exact):
    """Volume-weighted L2 norm of ``values - exact(x_cell)``."""
    values = np.asarray(values, dtype=float)
    reference = exact(np.asarray(mesh.cell_centers, dtype=float))
    volume = np.asarray(mesh.cell_volumes, dtype=float)
    return float(np.sqrt(np.sum(volume * (values - reference) ** 2) / np.sum(volume)))


def observed_order(coarse, fine, ratio):
    """Order p from two errors with grid ratio ``ratio`` (fine/coarse)."""
    coarse, fine, ratio = float(coarse), float(fine), float(ratio)
    if coarse <= 0.0 or fine <= 0.0:
        raise ValueError("errors must be positive to estimate an order")
    if ratio == 1.0:
        raise ValueError("grid ratio must differ from one")
    return float(np.log(coarse / fine) / np.log(ratio))


class UnstructuredFVM3D:
    """Owner/neighbour face operators on a tetrahedral :class:`NativeMesh3D`."""

    def __init__(self, mesh, *, decomposition="over-relaxed"):
        if not isinstance(mesh, NativeMesh3D):
            raise TypeError("UnstructuredFVM3D needs a NativeMesh3D")
        if decomposition not in DECOMPOSITIONS:
            raise ValueError("unknown flux decomposition: " + str(decomposition))
        self.mesh = mesh
        self.decomposition = decomposition
        self.cells = len(mesh.cells)
        self.faces = len(mesh.owner)
        self.internal = mesh.neighbor >= 0
        self.boundary = ~self.internal
        self.n_boundary = int(self.boundary.sum())
        self.owner = np.asarray(mesh.owner, dtype=int)
        self.neighbor = np.asarray(mesh.neighbor, dtype=int)
        self.centers = np.asarray(mesh.cell_centers, dtype=float)
        self.volumes = np.asarray(mesh.cell_volumes, dtype=float)
        self.face_centers = np.asarray(mesh.face_centers, dtype=float)
        self.area_vectors = np.asarray(mesh.area_vectors, dtype=float)
        self.area = np.linalg.norm(self.area_vectors, axis=1)
        if not np.all(self.area > 0.0):
            raise ValueError("mesh has a zero-area face")

        # "other point": the neighbour centroid inside, the face centroid on the
        # boundary.  d is the vector the implicit coefficient divides by.
        d = self.face_centers - self.centers[self.owner]
        d[self.internal] = self.centers[self.neighbor[self.internal]] - self.centers[self.owner[self.internal]]
        self.d = d
        s_dot_d = np.einsum("ij,ij->i", self.area_vectors, d)
        if not np.all(s_dot_d > 0.0):
            raise ValueError("face normal must point out of the owner cell")
        if decomposition == "over-relaxed":
            self.coefficient = np.einsum("ij,ij->i", self.area_vectors, self.area_vectors) / s_dot_d
        else:
            self.coefficient = s_dot_d / np.einsum("ij,ij->i", d, d)
        self.correction_vector = self.area_vectors - self.coefficient[:, None] * d
        # Interpolation weight of the neighbour along the owner->neighbour line.
        self.weight = np.zeros(self.faces)
        weight = (np.einsum("ij,ij->i", self.face_centers[self.internal] - self.centers[self.owner[self.internal]],
                            self.area_vectors[self.internal])
                  / np.einsum("ij,ij->i", d[self.internal], self.area_vectors[self.internal]))
        self.weight[self.internal] = weight
        self.boundary_labels = [str(label) for label in mesh.patch_labels]
        self.patches = {label: np.array([name == label for name in self.boundary_labels], dtype=bool)
                        for label in sorted(set(self.boundary_labels))}

    # ------------------------------------------------------------------ gradient
    def least_squares_gradient(self, values, boundary_values=None):
        """Inverse-distance weighted least-squares gradient of cell values.

        ``boundary_values`` gives one value per boundary face (compact order of
        ``self.boundary``); a non-finite entry falls back to the owner value,
        i.e. a zero-normal-difference sample.  Interior faces contribute both
        directions, so a cell whose neighbours span three dimensions is exactly
        linearly exact.
        """
        values = np.asarray(values, dtype=float)
        if values.shape != (self.cells,):
            raise ValueError("one cell value per mesh cell required")
        if boundary_values is None:
            boundary_values = np.full(self.n_boundary, np.nan)
        boundary_values = np.asarray(boundary_values, dtype=float)
        if boundary_values.shape != (self.n_boundary,):
            raise ValueError("one boundary value per boundary face required")

        owner = self.owner[self.internal]
        neighbor = self.neighbor[self.internal]
        e_internal = self.centers[neighbor] - self.centers[owner]
        delta_internal = values[neighbor] - values[owner]
        sources = [owner, neighbor]
        offsets = [e_internal, -e_internal]
        deltas = [delta_internal, -delta_internal]
        boundary_faces = np.flatnonzero(self.boundary)
        owner_boundary = self.owner[boundary_faces]
        fallback = values[owner_boundary]
        boundary_target = np.where(np.isfinite(boundary_values), boundary_values, fallback)
        sources.append(owner_boundary)
        offsets.append(self.face_centers[boundary_faces] - self.centers[owner_boundary])
        deltas.append(boundary_target - values[owner_boundary])
        cell_of_pair = np.concatenate(sources)
        e = np.concatenate(offsets)
        delta = np.concatenate(deltas)
        distance2 = np.einsum("ij,ij->i", e, e)
        if not np.all(distance2 > 0.0):
            raise ValueError("degenerate gradient sample: zero offset")
        weight = 1.0 / distance2
        normal = np.empty((self.cells, 3, 3))
        rhs = np.empty((self.cells, 3))
        scaled = weight * delta
        for a in range(3):
            rhs[:, a] = np.bincount(cell_of_pair, weights=scaled * e[:, a], minlength=self.cells)
            for b in range(3):
                normal[:, a, b] = np.bincount(cell_of_pair, weights=weight * e[:, a] * e[:, b],
                                              minlength=self.cells)
        eigenvalues = np.linalg.eigvalsh(normal)
        if np.any(eigenvalues[:, 0] <= 1e-13 * eigenvalues[:, -1]):
            raise ValueError("rank-deficient least-squares gradient stencil")
        return np.linalg.solve(normal, rhs[..., None])[..., 0]

    def _stencils(self):
        """Cached node-sharing (ring-1) stencil plus per-cell boundary samples.

        Cells sharing a vertex with a cell form a symmetric-enough cloud around
        it for a quadratic fit, which the four face neighbours of a tetrahedron
        alone do not.  Boundary faces contribute their centroid and Dirichlet
        value as extra samples, so a boundary cell is never left with a
        one-sided cloud and no data.
        """
        cached = getattr(self, "_stencil_cache", None)
        if cached is not None:
            return cached
        cells = self.cells
        rows, cols = [], []
        for index, cell in enumerate(self.mesh.cells):
            rows.extend([index] * len(cell))
            cols.extend(int(node) for node in cell)
        incidence = sparse.coo_matrix((np.ones(len(rows)), (rows, cols)),
                                      shape=(cells, len(self.mesh.vertices))).tocsr()
        adjacency = (incidence @ incidence.T).tocsr()
        adjacency.setdiag(0)
        adjacency.eliminate_zeros()
        degree = np.diff(adjacency.indptr)
        width = int(degree.max())
        neighbor = np.zeros((cells, width), dtype=int)
        mask = np.zeros((cells, width), dtype=bool)
        for index in range(cells):
            cloud = adjacency.indices[adjacency.indptr[index]:adjacency.indptr[index + 1]]
            neighbor[index, :len(cloud)] = cloud
            mask[index, :len(cloud)] = True

        boundary_faces = np.flatnonzero(self.boundary)
        boundary_owner = self.owner[boundary_faces]
        order = np.argsort(boundary_owner, kind="stable")
        boundary_faces = boundary_faces[order]
        boundary_owner = boundary_owner[order]
        counts = np.bincount(boundary_owner, minlength=cells)
        bwidth = max(int(counts.max()) if counts.size else 0, 1)
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        position = np.arange(len(boundary_faces)) - np.repeat(starts, counts)
        face_stencil = np.zeros((cells, bwidth), dtype=int)
        face_mask = np.zeros((cells, bwidth), dtype=bool)
        face_stencil[boundary_owner, position] = boundary_faces
        face_mask[boundary_owner, position] = True
        self._stencil_cache = (neighbor, mask, face_stencil, face_mask)
        return self._stencil_cache

    def quadratic_coefficients(self, values, boundary_values=None, *, chunk=4096):
        """Gradient of a k-exact quadratic least-squares reconstruction.

        The reconstruction is exact for every quadratic polynomial and returns
        the linear coefficients, so its gradient is second-order accurate on a
        tetrahedral cloud where the plain (linear) least-squares gradient is
        only first-order.  That matters because the non-orthogonal correction
        is ``t . g`` with ``t = O(h^2)``: a first-order gradient leaves an
        O(1) divergence defect and the Poisson solution then stalls around
        first order regardless of the implicit part.
        """
        values = np.asarray(values, dtype=float)
        if values.shape != (self.cells,):
            raise ValueError("one cell value per mesh cell required")
        if boundary_values is None:
            boundary_values = np.full(self.n_boundary, np.nan)
        boundary_values = np.asarray(boundary_values, dtype=float)
        if boundary_values.shape != (self.n_boundary,):
            raise ValueError("one boundary value per boundary face required")
        neighbor, mask, face_stencil, face_mask = self._stencils()
        full_boundary = np.full(self.faces, np.nan)
        full_boundary[self.boundary] = boundary_values

        offsets_neighbor = self.centers[neighbor] - self.centers[:, None, :]
        deltas_neighbor = values[neighbor] - values[:, None]
        offsets_face = self.face_centers[face_stencil] - self.centers[:, None, :]
        face_values = full_boundary[face_stencil]
        face_values = np.where(face_mask & np.isfinite(face_values), face_values,
                               values[:, None])
        deltas_face = face_values - values[:, None]
        offsets = np.concatenate([offsets_neighbor, offsets_face], axis=1)
        deltas = np.concatenate([deltas_neighbor, deltas_face], axis=1)
        active = np.concatenate([mask, face_mask], axis=1)

        linear = np.empty((self.cells, 3))
        curvature = np.empty((self.cells, 6))
        scale = np.empty(self.cells)
        for start in range(0, self.cells, int(chunk)):
            stop = min(start + int(chunk), self.cells)
            solution, local_scale = self._quadratic_fit(offsets[start:stop],
                                                        deltas[start:stop],
                                                        active[start:stop])
            linear[start:stop] = solution[:, :3]
            curvature[start:stop] = solution[:, 3:]
            scale[start:stop] = local_scale
        return linear, curvature, scale

    def quadratic_gradient(self, values, boundary_values=None, *, chunk=4096):
        """Linear coefficients of the k-exact quadratic reconstruction."""
        linear, _, scale = self.quadratic_coefficients(values, boundary_values, chunk=chunk)
        return linear / scale[:, None]

    @staticmethod
    def _curvature_times(curvature, xi):
        """``C xi`` for the symmetric quadratic form stored as [xx,yy,zz,xy,xz,yz]."""
        cxx, cyy, czz, cxy, cxz, cyz = (curvature[:, index] for index in range(6))
        x, y, z = xi[:, 0], xi[:, 1], xi[:, 2]
        return np.stack([cxx * x + cxy * y + cxz * z,
                         cxy * x + cyy * y + cyz * z,
                         cxz * x + cyz * y + czz * z], axis=1)

    def face_gradient(self, gradients=None, boundary_values=None, values=None, *,
                      quadratic=None, method="interpolate"):
        """Gradient at face centroids.

        ``interpolate`` averages the cell gradients; ``quadratic`` evaluates the
        k-exact quadratic reconstruction of each adjacent cell at the face
        centroid instead, so no interpolation of a noisy cell gradient is
        involved in the non-orthogonal correction.
        """
        if method == "quadratic":
            if quadratic is None:
                raise ValueError("quadratic reconstruction data required")
            linear, curvature, scale = quadratic
            owner, neighbor = self.owner, self.neighbor
            result = np.empty((self.faces, 3))
            internal = self.internal
            xi_owner = (self.face_centers[internal] - self.centers[owner[internal]]) / scale[owner[internal]][:, None]
            xi_neighbor = (self.face_centers[internal] - self.centers[neighbor[internal]]) / scale[neighbor[internal]][:, None]
            evaluated_owner = (linear[owner[internal]]
                               + 2.0 * self._curvature_times(curvature[owner[internal]], xi_owner)) / scale[owner[internal]][:, None]
            evaluated_neighbor = (linear[neighbor[internal]]
                                  + 2.0 * self._curvature_times(curvature[neighbor[internal]], xi_neighbor)) / scale[neighbor[internal]][:, None]
            weight = self.weight[internal][:, None]
            result[internal] = (1.0 - weight) * evaluated_owner + weight * evaluated_neighbor
            owner_boundary = owner[self.boundary]
            xi_boundary = (self.face_centers[self.boundary] - self.centers[owner_boundary]) / scale[owner_boundary][:, None]
            result[self.boundary] = (linear[owner_boundary]
                                     + 2.0 * self._curvature_times(curvature[owner_boundary], xi_boundary)) / scale[owner_boundary][:, None]
            return result
        if method != "interpolate":
            raise ValueError("unknown face gradient method: " + str(method))
        gradients = np.asarray(gradients, dtype=float)
        if gradients.shape != (self.cells, 3):
            raise ValueError("one gradient vector per cell required")
        result = np.empty((self.faces, 3))
        owner, neighbor = self.owner[self.internal], self.neighbor[self.internal]
        weight = self.weight[self.internal][:, None]
        result[self.internal] = (1.0 - weight) * gradients[owner] + weight * gradients[neighbor]
        result[self.boundary] = gradients[self.owner[self.boundary]]
        return result

    @staticmethod
    def _quadratic_fit(offsets, deltas, active):
        """Weighted quadratic least-squares fit in stencil-scaled coordinates."""
        distance2 = np.einsum("csa,csa->cs", offsets, offsets)
        weight = np.where(active, 1.0 / np.where(distance2 > 0.0, distance2, 1.0), 0.0)
        count = np.maximum(active.sum(axis=1), 1)
        scale = np.sqrt(np.maximum(np.sum(np.where(active, distance2, 0.0), axis=1) / count,
                                   1e-300))
        local = offsets / scale[:, None, None]
        x, y, z = local[:, :, 0], local[:, :, 1], local[:, :, 2]
        basis = np.stack([x, y, z, x * x, y * y, z * z, x * y, x * z, y * z], axis=2)
        weighted = basis * weight[:, :, None]
        normal = np.einsum("csa,csb->cab", weighted, basis)
        rhs = np.einsum("csa,cs->ca", weighted, deltas)
        eigenvalues = np.linalg.eigvalsh(normal)
        if np.any(eigenvalues[:, 0] <= 1e-12 * np.maximum(eigenvalues[:, -1], 1e-300)):
            raise ValueError("rank-deficient quadratic reconstruction stencil")
        solution = np.linalg.solve(normal, rhs[..., None])[..., 0]
        return solution, scale

    def gradient(self, values, boundary_values=None, *, passes=0, method="linear"):
        """Least-squares gradient, optionally refined by Gauss-gradient passes.

        A single least-squares pass is consistent but only first-order accurate
        on a skewed stencil.  Each extra pass rebuilds second-order face values
        from the current gradient and averages them with the divergence
        theorem, which recovers second-order gradient accuracy (the correction
        term of the diffusion operator is O(h^2) in the flux, so a first-order
        gradient would cap the Poisson order at one).
        """
        if method == "quadratic":
            if passes:
                raise ValueError("Gauss passes are only defined for the linear reconstruction")
            return self.quadratic_gradient(values, boundary_values)
        if method != "linear":
            raise ValueError("unknown gradient reconstruction: " + str(method))
        values = np.asarray(values, dtype=float)
        gradients = self.least_squares_gradient(values, boundary_values)
        if passes <= 0:
            return gradients
        if boundary_values is None:
            boundary_values = np.full(self.n_boundary, np.nan)
        boundary_values = np.asarray(boundary_values, dtype=float)
        for _ in range(int(passes)):
            gradients = self._gauss_gradient(values, gradients, boundary_values)
        return gradients

    def _gauss_gradient(self, values, gradients, boundary_values):
        """Divergence-theorem gradient of second-order (skew-corrected) face values."""
        internal = self.internal
        owner, neighbor = self.owner[internal], self.neighbor[internal]
        weight = self.weight[internal]
        linear_point = self.centers[owner] + weight[:, None] * self.d[internal]
        face_gradient = ((1.0 - weight)[:, None] * gradients[owner]
                         + weight[:, None] * gradients[neighbor])
        skew = np.einsum("ij,ij->i", face_gradient, self.face_centers[internal] - linear_point)
        faces = np.empty(self.faces)
        faces[internal] = (1.0 - weight) * values[owner] + weight * values[neighbor] + skew
        boundary_faces = np.flatnonzero(self.boundary)
        owner_boundary = self.owner[boundary_faces]
        faces[boundary_faces] = np.where(np.isfinite(boundary_values), boundary_values,
                                         values[owner_boundary])
        return np.stack([self._outward_flux_balance(faces * self.area_vectors[:, axis])
                         / self.volumes for axis in range(3)], axis=1)

    # ----------------------------------------------------------------- diffusion
    def diffusion_matrix(self, conductivity=1.0):
        """Implicit (orthogonal) part of ``div(Gamma grad phi)`` as a sparse matrix.

        Returns ``(matrix, coefficient)``: ``coefficient`` is the per-face
        implicit conductance, so the Dirichlet contribution of a boundary face
        is ``coefficient_b * phi_b`` (applied by :meth:`poisson`).
        """
        conductance = self._conductivity(conductivity)
        coefficient = self.coefficient * conductance
        internal = self.internal
        owner, neighbor = self.owner[internal], self.neighbor[internal]
        rows = np.concatenate([owner, owner, neighbor, neighbor, self.owner[self.boundary]])
        cols = np.concatenate([owner, neighbor, neighbor, owner, self.owner[self.boundary]])
        data = np.concatenate([coefficient[internal], -coefficient[internal],
                               coefficient[internal], -coefficient[internal],
                               coefficient[self.boundary]])
        size = self.cells
        return sparse.coo_matrix((data, (rows, cols)), shape=(size, size)).tocsr(), coefficient

    def diffusion_correction(self, gradients=None, *, face_gradient=None):
        """Per-cell divergence of the deferred non-orthogonal flux correction."""
        if face_gradient is None:
            if gradients is None:
                raise ValueError("cell gradients or a face gradient are required")
            face_gradient = self.face_gradient(gradients)
        flux = np.einsum("ij,ij->i", self.correction_vector, face_gradient)
        return self._outward_flux_balance(flux)

    def _outward_flux_balance(self, face_flux):
        """Sum of signed outward face fluxes per cell (internal shares cancel)."""
        internal = self.internal
        balance = np.zeros(self.cells)
        np.add.at(balance, self.owner, face_flux)
        np.add.at(balance, self.neighbor[internal], -face_flux[internal])
        return balance

    def _conductivity(self, conductivity):
        if np.isscalar(conductivity):
            value = float(conductivity)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError("conductivity must be positive and finite")
            return np.full(self.faces, value)
        array = np.asarray(conductivity, dtype=float)
        if array.shape != (self.faces,):
            raise ValueError("one conductivity per face required")
        if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
            raise ValueError("conductivity must be positive and finite")
        return array

    def poisson(self, source, dirichlet, *, conductivity=1.0, gradient_passes=1,
                gradient_method="linear", face_gradient_method="interpolate",
                max_iterations=200, tolerance=1e-12, relaxation=1.0, initial=None):
        """Solve ``div(Gamma grad phi) = source`` with Dirichlet boundary data.

        ``dirichlet`` holds one value per boundary face; every boundary face
        must be specified (a missing wall value must not silently become a
        zero-normal-difference condition).  The non-orthogonal correction is
        iterated to its fixed point, so the returned field is the solution of
        the corrected (second-order) scheme, not of its first iterate.
        """
        source = np.asarray(source, dtype=float)
        if source.shape != (self.cells,):
            raise ValueError("one source value per cell required")
        if not np.all(np.isfinite(source)):
            raise ValueError("source must be finite")
        dirichlet = np.asarray(dirichlet, dtype=float)
        if dirichlet.shape != (self.n_boundary,):
            raise ValueError("one Dirichlet value per boundary face required")
        if not np.all(np.isfinite(dirichlet)):
            raise ValueError("every boundary face needs finite Dirichlet data")

        # Conserved form: sum of outward fluxes = source * volume.  The matrix
        # is assembled with the opposite (negative semi-definite) sign, i.e.
        # A phi = sum_int coef (phi_o - phi_n) + sum_b coef_b phi_o, so
        # A phi = -source V + sum_b coef_b phi_b + correction.
        matrix, coefficient = self.diffusion_matrix(conductivity)
        target = -source * self.volumes
        target = target + self._dirichlet_flux(coefficient[self.boundary] * dirichlet)
        phi = np.zeros(self.cells) if initial is None else np.asarray(initial, dtype=float).copy()
        if phi.shape != (self.cells,):
            raise ValueError("initial field has the wrong shape")
        history = []
        for iteration in range(int(max_iterations)):
            if face_gradient_method == "quadratic":
                quadratic = self.quadratic_coefficients(phi, dirichlet)
                face_gradient = self.face_gradient(quadratic=quadratic, method="quadratic")
                correction = self.diffusion_correction(face_gradient=face_gradient)
            else:
                gradients = self.gradient(phi, dirichlet, passes=gradient_passes,
                                          method=gradient_method)
                correction = self.diffusion_correction(gradients)
            rhs = target + correction
            updated = spsolve(matrix, rhs)
            if not np.all(np.isfinite(updated)):
                raise FloatingPointError("Poisson solve produced a non-finite field")
            change = np.max(np.abs(updated - phi)) if iteration else np.inf
            phi = phi + relaxation * (updated - phi)
            history.append(dict(iteration=iteration, change=float(change),
                                linear_residual=float(np.max(np.abs(matrix @ phi - rhs)))))
            if iteration and change <= tolerance * max(1.0, np.max(np.abs(phi))):
                break
        else:
            raise RuntimeError("non-orthogonal Poisson correction did not converge "
                               "in {} iterations".format(max_iterations))
        return dict(values=phi, iterations=len(history), history=history,
                    gradient_passes=int(gradient_passes), gradient_method=gradient_method,
                    face_gradient_method=face_gradient_method,
                    decomposition=self.decomposition)

    def _dirichlet_flux(self, coefficient_times_value):
        """Distribute ``coef_b * phi_b`` onto owner cells (boundary faces only)."""
        balance = np.zeros(self.cells)
        np.add.at(balance, self.owner[self.boundary], coefficient_times_value)
        return balance

    # ----------------------------------------------------------------- convection
    def advection(self, velocity, inflow_values, *, scheme="upwind", max_iterations=400,
                  tolerance=1e-9, relaxation=None, initial=None):
        """Steady convection ``div(u phi) = 0`` with the requested face weighting.

        ``velocity`` is one vector (or one per face).  ``inflow_values`` holds
        one value per boundary face; it is used wherever ``u . s < 0`` and
        ignored elsewhere (an outflow face takes its value from the owner cell).
        ``upwind`` is solved directly; ``central`` and ``limited`` are deferred
        corrections on the upwind matrix, iterated to their fixed point.
        """
        if scheme not in SCHEMES:
            raise ValueError("unknown convection scheme: " + str(scheme))
        flux = self.face_flux(velocity)
        inflow_values = np.asarray(inflow_values, dtype=float)
        if inflow_values.shape != (self.n_boundary,):
            raise ValueError("one boundary value per boundary face required")
        internal = self.internal
        owner, neighbor = self.owner[internal], self.neighbor[internal]
        face_flux = flux[internal]
        upwind = np.where(face_flux >= 0.0, owner, neighbor)
        downwind = np.where(face_flux >= 0.0, neighbor, owner)
        boundary_flux = flux[self.boundary]
        boundary_owner = self.owner[self.boundary]
        inward = boundary_flux < 0.0
        # An inflow face carries prescribed data; treating it as a wall (upwind
        # from the owner cell) is refused instead of silently created.
        if np.any(~np.isfinite(inflow_values[inward])):
            raise ValueError("inflow faces need finite boundary values")
        outflow = ~inward
        rows = np.concatenate([owner, neighbor, boundary_owner[outflow]])
        cols = np.concatenate([upwind, upwind, boundary_owner[outflow]])
        data = np.concatenate([face_flux, -face_flux, boundary_flux[outflow]])
        matrix = sparse.coo_matrix((data, (rows, cols)),
                                   shape=(self.cells, self.cells)).tocsr()
        target = np.zeros(self.cells)
        np.add.at(target, boundary_owner[inward], -boundary_flux[inward] * inflow_values[inward])

        phi = np.zeros(self.cells) if initial is None else np.asarray(initial, dtype=float).copy()
        history = []
        converged = True
        if relaxation is None:
            # Deferred higher-order corrections on a distorted tet mesh need
            # damping; the upwind solve itself is exact for its own scheme.
            relaxation = 1.0 if scheme == "upwind" else 0.7
        if scheme == "upwind":
            phi = spsolve(matrix, target)
            history.append(dict(iteration=0, change=0.0, correction_norm=0.0))
        else:
            phi = spsolve(matrix, target) if initial is None else phi
            for iteration in range(int(max_iterations)):
                boundary_data = np.full(self.n_boundary, np.nan)
                boundary_data[inward] = inflow_values[inward]
                gradients = self.gradient(phi, boundary_data, passes=1)
                face_values = self.convection_face_values(
                    phi, gradients, flux, scheme=scheme, inflow_values=inflow_values)
                correction = face_values[internal] - phi[upwind]
                rhs = target.copy()
                np.add.at(rhs, owner, -face_flux * correction)
                np.add.at(rhs, neighbor, face_flux * correction)
                updated = spsolve(matrix, rhs)
                if not np.all(np.isfinite(updated)):
                    raise FloatingPointError("advection solve produced a non-finite field")
                change = np.max(np.abs(updated - phi)) if iteration else np.inf
                phi = phi + relaxation * (updated - phi)
                history.append(dict(iteration=iteration, change=float(change),
                                    correction_norm=float(np.max(np.abs(correction)))))
                if iteration and change <= tolerance * max(1.0, np.max(np.abs(phi))):
                    break
            else:
                converged = False
        return dict(values=phi, iterations=len(history), history=history,
                    scheme=scheme, flux=flux, converged=converged,
                    relaxation=relaxation)

    def convection_face_values(self, values, gradients, flux=None, *, scheme="central",
                               inflow_values=None):
        """Face values for the requested convection scheme (all faces)."""
        if scheme not in SCHEMES:
            raise ValueError("unknown convection scheme: " + str(scheme))
        values = np.asarray(values, dtype=float)
        if flux is None:
            raise ValueError("face fluxes are required")
        if inflow_values is not None:
            inflow_values = np.asarray(inflow_values, dtype=float)
            if inflow_values.shape != (self.n_boundary,):
                raise ValueError("one boundary value per boundary face required")
        internal = self.internal
        owner, neighbor = self.owner[internal], self.neighbor[internal]
        face_flux = flux[internal]
        upwind = np.where(face_flux >= 0.0, owner, neighbor)
        downwind = np.where(face_flux >= 0.0, neighbor, owner)
        result = np.empty(self.faces)
        if scheme == "upwind":
            result[internal] = values[upwind]
        else:
            gradients = np.asarray(gradients, dtype=float)
            weight = self.weight[internal]
            linear_point = self.centers[owner] + weight[:, None] * self.d[internal]
            face_gradient = ((1.0 - weight)[:, None] * gradients[owner]
                             + weight[:, None] * gradients[neighbor])
            central = ((1.0 - weight) * values[owner] + weight * values[neighbor]
                       + np.einsum("ij,ij->i", face_gradient,
                                   self.face_centers[internal] - linear_point))
            if scheme == "central":
                result[internal] = central
            else:
                low = np.minimum(values[owner], values[neighbor])
                high = np.maximum(values[owner], values[neighbor])
                result[internal] = np.clip(central, low, high)
        boundary = self.boundary
        boundary_flux = flux[boundary]
        boundary_owner = self.owner[boundary]
        result[boundary] = values[boundary_owner]
        if inflow_values is not None:
            inflow = boundary_flux < 0.0
            result[np.flatnonzero(boundary)[inflow]] = inflow_values[inflow]
        return result

    def face_flux(self, velocity):
        """Volumetric flux ``u . s`` per face (owner outward)."""
        velocity = np.asarray(velocity, dtype=float)
        if velocity.ndim == 1:
            if velocity.shape != (3,):
                raise ValueError("velocity must be a 3-vector or one per face")
            if not np.all(np.isfinite(velocity)):
                raise ValueError("velocity must be finite")
            return self.area_vectors @ velocity
        if velocity.shape != (self.faces, 3):
            raise ValueError("velocity must be a 3-vector or one per face")
        if not np.all(np.isfinite(velocity)):
            raise ValueError("velocity must be finite")
        return np.einsum("ij,ij->i", velocity, self.area_vectors)