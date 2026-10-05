"""Polygonal O-grid and collocated conservative SIMPLE for a confined cylinder.

The outer boundary is the original rectangle. Cylinder vertices lie on the
circle, but its boundary consists of straight chord facets, not exact arcs.
All face fluxes are oriented out of their owner cell.
"""

from dataclasses import dataclass
import math

import torch


class BodyFittedMesh:
    """An O-grid with ``nx`` circumferential and ``ny`` radial cells."""

    def __init__(self, config):
        c = config
        if c.nx < 8 or c.nx % 4 or c.ny < 1:
            raise ValueError("body-fitted nx must be >= 8 and divisible by four")
        r = c.cylinder_radius
        if not r or not (r < c.cylinder_x < c.length - r
                         and r < c.cylinder_y < c.height - r):
            raise ValueError("body-fitted cylinder must lie strictly inside the channel")
        opts = dict(dtype=torch.float64, device=c.device)
        center = torch.tensor([c.cylinder_x, c.cylinder_y], **opts)
        corners = [(c.length, c.height), (0, c.height), (0, 0), (c.length, 0)]
        angles = [math.atan2(y - c.cylinder_y, x - c.cylinder_x) for x, y in corners]
        for i in range(1, 4):
            while angles[i] <= angles[i - 1]:
                angles[i] += 2 * math.pi
        angles.append(angles[0] + 2 * math.pi)
        theta = torch.tensor([
            angles[k] + (angles[k + 1] - angles[k]) * i / (c.nx // 4)
            for k in range(4) for i in range(c.nx // 4)
        ] + [angles[-1]], **opts)
        ray = torch.stack((theta.cos(), theta.sin()), -1)
        tx = torch.where(ray[:, 0] > 0, (c.length - c.cylinder_x) / ray[:, 0],
                         -c.cylinder_x / ray[:, 0])
        ty = torch.where(ray[:, 1] > 0, (c.height - c.cylinder_y) / ray[:, 1],
                         -c.cylinder_y / ray[:, 1])
        outer = center + torch.minimum(tx, ty)[:, None] * ray
        # Set corners exactly, including the periodic seam.
        for k, corner in enumerate(corners):
            outer[k * (c.nx // 4)] = torch.tensor(corner, **opts)
        outer[-1] = outer[0]
        inner = center + r * ray
        inner[-1] = inner[0]
        t = torch.linspace(0, 1, c.ny + 1, **opts)[:, None, None]
        self.vertices = inner[None] * (1 - t) + outer[None] * t
        polygon = torch.stack((self.vertices[:-1, :-1], self.vertices[1:, :-1],
                               self.vertices[1:, 1:], self.vertices[:-1, 1:]), -2)
        nxt = polygon.roll(-1, -2)
        cross = polygon[..., 0] * nxt[..., 1] - nxt[..., 0] * polygon[..., 1]
        self.volumes = cross.sum(-1) / 2
        self.centers = ((polygon + nxt) * cross[..., None]).sum(-2) / (
            6 * self.volumes[..., None])
        if not torch.all(self.volumes > 0):
            raise ValueError("invalid body-fitted cell geometry")
        owners, neighbors, labels, endpoints = [], [], [], []
        known = {}
        for j in range(c.ny):
            for i in range(c.nx):
                ids = [(j, i), (j + 1, i), (j + 1, (i + 1) % c.nx),
                       (j, (i + 1) % c.nx)]
                for a, b in zip(ids, ids[1:] + ids[:1]):
                    key = tuple(sorted((a, b)))
                    if key in known:
                        neighbors[known[key]] = j * c.nx + i
                        continue
                    known[key] = len(owners)
                    owners.append(j * c.nx + i)
                    neighbors.append(-1)
                    endpoints.append(torch.stack((self.vertices[a], self.vertices[b])))
                    if a[0] == b[0] == 0:
                        label = "cylinder"
                    elif a[0] == b[0] == c.ny:
                        midpoint = (self.vertices[a] + self.vertices[b]) / 2
                        if abs(float(midpoint[0])) < 1e-12 * c.length:
                            label = "inlet"
                        elif abs(float(midpoint[0]) - c.length) < 1e-12 * c.length:
                            label = "outlet"
                        else:
                            label = "wall"
                    else:
                        label = "interior"
                    labels.append(label)
        self.owner = torch.tensor(owners, dtype=torch.long, device=c.device)
        self.neighbor = torch.tensor(neighbors, dtype=torch.long, device=c.device)
        self.boundary_labels = tuple(labels)
        self.face_vertices = torch.stack(endpoints)
        self.face_centers = self.face_vertices.mean(1)
        edge = self.face_vertices[:, 1] - self.face_vertices[:, 0]
        self.face_area_vectors = torch.stack((edge[:, 1], -edge[:, 0]), -1)
        self.face_lengths = torch.linalg.vector_norm(edge, dim=-1)
        self.face_normals = self.face_area_vectors / self.face_lengths[:, None]
        self.interior = self.neighbor >= 0
        self.boundary = ~self.interior
        self.masks = {name: torch.tensor([s == name for s in labels],
                                       dtype=torch.bool, device=c.device)
                      for name in ("inlet", "outlet", "wall", "cylinder")}


@dataclass
class BodyFittedResult:
    config: object
    u: torch.Tensor
    v: torch.Tensor
    p: torch.Tensor
    fluid: torch.Tensor
    x: torch.Tensor
    y: torch.Tensor
    history: list
    converged: bool
    mesh: BodyFittedMesh
    mass_flux: torch.Tensor
    boundary_velocity: torch.Tensor

    def cell_center_velocity(self):
        return self.u, self.v


class BodyFittedSolver:
    """Steady SIMPLE with upwind convection and corrected central diffusion.

    Least-squares gradients provide deferred nonorthogonal diffusion. Rhie--Chow
    fluxes and the correction equation use the same relaxed momentum diagonal.
    Pressure corrections have a Dirichlet outlet reference; their corrected
    outlet flux is retained, never overwritten by velocity extrapolation.
    Pseudo-time inertia stabilizes iterations but is absent from steady residuals.
    """

    def __init__(self, config):
        self.config = config
        self.mesh = m = BodyFittedMesh(config)
        self.o, self.n = m.owner, m.neighbor
        self.f = m.interior
        self.oi, self.ni = self.o[self.f], self.n[self.f]
        self.count = config.nx * config.ny
        opts = dict(dtype=torch.float64, device=config.device)
        self.velocity = torch.zeros((self.count, 2), **opts)
        self.velocity[:, 0] = config.inlet_velocity
        self.p = torch.zeros(self.count, **opts)
        self.u = self.velocity[:, 0].reshape(config.ny, config.nx)
        self.v = self.velocity[:, 1].reshape(config.ny, config.nx)
        self.x, self.y = m.centers[..., 0], m.centers[..., 1]
        self.fluid = torch.ones((config.ny, config.nx), dtype=torch.bool,
                               device=config.device)
        self.volume = m.volumes.flatten()
        self.S = m.face_area_vectors
        self.d = m.face_centers - m.centers.reshape(-1, 2)[self.o]
        self.d[self.f] = (m.centers.reshape(-1, 2)[self.ni]
                         - m.centers.reshape(-1, 2)[self.oi])
        self.k = (self.S * self.S).sum(-1) / (self.S * self.d).sum(-1)
        if not torch.all(self.k > 0):
            raise ValueError("mesh has nonpositive face-to-cell projected distance")
        self.T = self.S - self.k[:, None] * self.d
        self.boundary_velocity = torch.zeros_like(self.S)
        self.boundary_velocity[m.masks["inlet"], 0] = config.inlet_velocity
        face_velocity = self.velocity[self.o].clone()
        fixed = m.boundary & ~m.masks["outlet"]
        face_velocity[fixed] = self.boundary_velocity[fixed]
        self.mass_flux = config.density * (face_velocity * self.S).sum(-1)
        self.history = []
        self.converged = False
        self._gradient_weights = {}

    def _sum(self, face):
        result = torch.zeros((self.count,) + face.shape[1:], dtype=face.dtype,
                             device=face.device)
        result.index_add_(0, self.o, face)
        result.index_add_(0, self.ni, -face[self.f])
        return result

    def _gradient(self, field, pressure=False):
        scalar = field.ndim == 1
        q = field[:, None] if scalar else field
        delta = torch.zeros((len(self.o), q.shape[1]), dtype=q.dtype, device=q.device)
        delta[self.f] = q[self.ni] - q[self.oi]
        active = self.f.clone()
        if pressure:
            active |= self.mesh.masks["outlet"]
            delta[self.mesh.masks["outlet"]] = -q[self.o[self.mesh.masks["outlet"]]]
        else:
            fixed = self.mesh.boundary & ~self.mesh.masks["outlet"]
            active |= fixed
            delta[fixed] = self.boundary_velocity[fixed] - q[self.o[fixed]]
        if pressure not in self._gradient_weights:
            weight = active.to(q.dtype) / (self.d * self.d).sum(-1)
            matrix_face = weight[:, None, None] * self.d[:, :, None] * self.d[:, None, :]
            matrix = torch.zeros((self.count, 2, 2), dtype=q.dtype, device=q.device)
            matrix.index_add_(0, self.o, matrix_face)
            matrix.index_add_(0, self.ni, matrix_face[self.f])
            inverse = torch.linalg.inv(matrix)
            wd = weight[:, None] * self.d
            go = torch.einsum("fij,fj->fi", inverse[self.o], wd)
            gn = torch.einsum("fij,fj->fi", inverse[self.ni], wd[self.f])
            self._gradient_weights[pressure] = go, gn
        go, gn = self._gradient_weights[pressure]
        gradient = torch.zeros((self.count, 2, q.shape[1]), dtype=q.dtype, device=q.device)
        gradient.index_add_(0, self.o, go[:, :, None] * delta[:, None, :])
        gradient.index_add_(0, self.ni, gn[:, :, None] * delta[self.f, None, :])
        return gradient[..., 0] if scalar else gradient

    def _interpolate(self, field):
        result = field[self.o].clone()
        result[self.f] = (field[self.oi] + field[self.ni]) / 2
        return result

    def _momentum(self, velocity, pressure, flux):
        c, m = self.config, self.mesh
        diffusion = c.viscosity * self.k
        diffusion[m.masks["outlet"]] = 0
        diagonal = torch.zeros_like(self.p)
        diagonal.index_add_(0, self.o, diffusion + flux.clamp_min(0))
        diagonal.index_add_(0, self.ni,
                            diffusion[self.f] + (-flux[self.f]).clamp_min(0))
        ao = diffusion[self.f] + (-flux[self.f]).clamp_min(0)
        an = diffusion[self.f] + flux[self.f].clamp_min(0)
        fixed = m.boundary & ~m.masks["outlet"]
        source = self._sum(torch.where(fixed[:, None],
                           (diffusion + (-flux).clamp_min(0))[:, None]
                           * self.boundary_velocity, 0))
        gradient = self._interpolate(self._gradient(velocity))
        correction = c.viscosity * torch.einsum("fi,fij->fj", self.T, gradient)
        correction[m.masks["outlet"]] = 0
        source += self._sum(correction) - self.volume[:, None] * self._gradient(
            pressure, pressure=True)
        # Extrapolated backflow uses the owner velocity (zero-gradient outlet).
        diagonal.index_add_(0, self.o[m.masks["outlet"]],
                            flux[m.masks["outlet"]].clamp_max(0))
        return diagonal, ao, an, source

    def _matvec(self, x, diagonal, ao, an):
        y = diagonal * x
        y.index_add_(0, self.oi, -ao * x[self.ni])
        y.index_add_(0, self.ni, -an * x[self.oi])
        return y

    def _linear(self, diagonal, ao, an, rhs, initial, symmetric=False, operator=None):
        """Diagonal-preconditioned CG or BiCGSTAB, with a true residual check."""
        x = initial.clone()
        mv = operator if operator is not None else lambda z: self._matvec(z, diagonal, ao, an)
        residual = rhs - mv(x)
        threshold = max(1e-12, float(torch.linalg.vector_norm(rhs)) * 1e-10)
        if float(torch.linalg.vector_norm(residual)) <= threshold:
            return x
        if symmetric:
            z = residual / diagonal
            direction = z.clone()
            rz = torch.dot(residual, z)
            for _ in range(max(100, self.count * 3)):
                ad = mv(direction)
                alpha = rz / torch.dot(direction, ad)
                x += alpha * direction
                residual -= alpha * ad
                if float(torch.linalg.vector_norm(residual)) <= threshold:
                    break
                z = residual / diagonal
                next_rz = torch.dot(residual, z)
                direction = z + (next_rz / rz) * direction
                rz = next_rz
        else:
            shadow = residual.clone()
            direction = torch.zeros_like(x)
            v = direction.clone()
            rho_old = alpha = omega = 1.0
            for _ in range(max(100, self.count * 2)):
                rho = torch.dot(shadow, residual)
                if abs(float(rho)) < 1e-30:
                    break
                beta = (rho / rho_old) * (alpha / omega)
                direction = residual + beta * (direction - omega * v)
                phat = direction / diagonal
                v = mv(phat)
                alpha = rho / torch.dot(shadow, v)
                s = residual - alpha * v
                x += alpha * phat
                if float(torch.linalg.vector_norm(s)) <= threshold:
                    break
                shat = s / diagonal
                t = mv(shat)
                omega = torch.dot(t, s) / torch.dot(t, t)
                x += omega * shat
                residual = s - omega * t
                if float(torch.linalg.vector_norm(residual)) <= threshold:
                    break
                if abs(float(omega)) < 1e-30:
                    break
                rho_old = rho
        if not torch.isfinite(x).all():
            raise RuntimeError("nonfinite body-fitted linear solve")
        return x

    def _rhie_chow(self, velocity, pressure, D, steady_D=None, old_velocity=None):
        gp = self._gradient(pressure, pressure=True)
        df = self._interpolate(D)
        face_velocity = self._interpolate(velocity + D[:, None] * gp)
        dp = -pressure[self.o]
        dp[self.f] = pressure[self.ni] - pressure[self.oi]
        normal_pressure = self.k * dp + (self.T * self._interpolate(gp)).sum(-1)
        flux = self.config.density * ((face_velocity * self.S).sum(-1)
                                      - df * normal_pressure)
        if steady_D is not None:
            # Retain pseudo-time face-flux memory. At a fixed point its damping
            # cancels, leaving exactly the inertia-free Rhie--Chow operator.
            steady_df = self._interpolate(steady_D)
            beta = (1 - df / steady_df).clamp(0, 1)
            base = self.config.density * (self._interpolate(velocity) * self.S).sum(-1)
            old_base = self.config.density * (
                self._interpolate(old_velocity) * self.S).sum(-1)
            defect = self.config.density * (
                (self._interpolate(steady_D[:, None] * gp) * self.S).sum(-1)
                - steady_df * normal_pressure)
            flux = base + (1 - beta) * defect + beta * (self.mass_flux - old_base)
        fixed = self.mesh.boundary & ~self.mesh.masks["outlet"]
        flux[fixed] = self.config.density * (
            self.boundary_velocity[fixed] * self.S[fixed]).sum(-1)
        coefficient = self.config.density * df * self.k
        coefficient[fixed] = 0
        return flux, coefficient

    def _correct(self, predicted_flux, coefficient, D):
        diagonal = self._sum(coefficient * 0)
        diagonal.index_add_(0, self.o, coefficient)
        diagonal.index_add_(0, self.ni, coefficient[self.f])
        rhs = -self._sum(predicted_flux)
        def nonorthogonal(q):
            gradient = self._interpolate(self._gradient(q, pressure=True))
            return coefficient / self.k * (self.T * gradient).sum(-1)

        def operator(q):
            return (self._matvec(q, diagonal, coefficient[self.f], coefficient[self.f])
                    - self._sum(nonorthogonal(q)))

        correction = self._linear(diagonal, coefficient[self.f], coefficient[self.f],
                                  rhs, torch.zeros_like(self.p), operator=operator)
        delta = correction[self.o].clone()
        delta[self.f] -= correction[self.ni]
        self.mass_flux = predicted_flux + coefficient * delta - nonorthogonal(correction)
        self.velocity -= D[:, None] * self._gradient(correction, pressure=True)
        self.p += self.config.pressure_relaxation * correction

    def step(self):
        """Advance one SIMPLE iteration and return conservative steady metrics."""
        c = self.config
        inlet_mass = c.density * c.inlet_velocity * c.height
        force_scale = max(c.density * c.inlet_velocity ** 2 * c.height,
                          c.viscosity * c.inlet_velocity)
        old = self.velocity.clone()
        diagonal, ao, an, source = self._momentum(old, self.p, self.mass_flux)
        transient = (c.density * self.volume / c.pseudo_time_step
                     if c.pseudo_time_step is not None else torch.zeros_like(diagonal))
        relaxed = (diagonal + transient) / c.velocity_relaxation
        rhs = source + (transient + (1 - c.velocity_relaxation)
                        * relaxed)[:, None] * old
        for component in range(2):
            self.velocity[:, component] = self._linear(
                relaxed, ao, an, rhs[:, component], old[:, component])
        D = self.volume / relaxed
        steady_D = self.volume * c.velocity_relaxation / diagonal
        flux, coefficient = self._rhie_chow(self.velocity, self.p, D, steady_D, old)
        self._correct(flux, coefficient, D)
        if not (torch.isfinite(self.velocity).all() and torch.isfinite(self.p).all()
                and torch.isfinite(self.mass_flux).all()):
            raise RuntimeError("nonfinite body-fitted SIMPLE iterate")
        diagonal, ao, an, source = self._momentum(
            self.velocity, self.p, self.mass_flux)
        residual = torch.stack([self._matvec(self.velocity[:, k], diagonal, ao, an)
                                - source[:, k] for k in range(2)], -1)
        continuity = float(self._sum(self.mass_flux).abs().max()) / inlet_mass
        momentum = float(residual.abs().sum(0).max()) / force_scale
        imbalance = abs(float(self.mass_flux[self.mesh.boundary].sum())) / inlet_mass
        metrics = dict(iteration=len(self.history) + 1, continuity=continuity,
                       momentum=momentum, mass_imbalance=imbalance,
                       velocity_change=float((self.velocity - old).abs().max())
                       / c.inlet_velocity,
                       u_residual=float(residual[:, 0].abs().sum()) / force_scale,
                       v_residual=float(residual[:, 1].abs().sum()) / force_scale)
        self.history.append(metrics)
        self.converged = max(continuity, momentum, imbalance) < c.tolerance
        return metrics

    def solve(self):
        c = self.config
        for _ in range(c.max_iterations):
            self.step()
            if self.converged:
                break
        shape = (c.ny, c.nx)
        return BodyFittedResult(c, self.velocity[:, 0].reshape(shape).clone(),
                                self.velocity[:, 1].reshape(shape).clone(),
                                self.p.reshape(shape).clone(),
                                torch.ones(shape, dtype=torch.bool, device=c.device),
                                self.mesh.centers[..., 0].clone(),
                                self.mesh.centers[..., 1].clone(),
                                [item.copy() for item in self.history], self.converged,
                                self.mesh, self.mass_flux.clone(),
                                self.boundary_velocity.clone())
