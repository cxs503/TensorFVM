"""Structured body-fitted meshes and collocated conservative SIMPLE solvers.

The O-grid wraps a polygonal cylinder; the C-grid wraps a NACA airfoil and joins
the wake seam internally. All face fluxes are oriented out of their owner cell.
"""

from dataclasses import dataclass
import math

import torch


def _naca4_profile(config):
    """Return a closed polygonal NACA four-digit profile in physical coordinates."""
    code = config.airfoil_code
    camber, position, thickness = int(code[0]) / 100, int(code[1]) / 10, int(code[2:]) / 100
    if camber and not position:
        raise ValueError("a cambered NACA four-digit airfoil requires a nonzero camber position")
    count = max(128, config.nx * 4)
    chord = config.airfoil_chord
    upper, lower = [], []
    for i in range(count + 1):
        x = (1 - math.cos(math.pi * i / count)) / 2
        yt = 5 * thickness * (
            0.2969 * math.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2
            + 0.2843 * x ** 3 - 0.1036 * x ** 4
        )
        yc, slope = _naca4_camberline(camber, position, x)
        theta = math.atan(slope)
        upper.append((x * chord - yt * math.sin(theta),
                      yc * chord + yt * math.cos(theta)))
        lower.append((x * chord + yt * math.sin(theta),
                      yc * chord - yt * math.cos(theta)))
    profile = list(reversed(upper)) + lower[1:]
    return [(config.airfoil_x + x, config.airfoil_y + y) for x, y in profile]


def _naca4_camberline(camber, position, x):
    if not camber:
        return 0.0, 0.0
    if x < position:
        return (camber / position ** 2 * (2 * position * x - x ** 2),
                2 * camber / position ** 2 * (position - x))
    return (
        camber / (1 - position) ** 2 * (
            (1 - 2 * position) + 2 * position * x - x ** 2
        ),
        2 * camber / (1 - position) ** 2 * (position - x),
    )


def _cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


class BodyFittedMesh:
    """An O-grid for a cylinder or C-grid for a NACA airfoil."""

    def __init__(self, config):
        c = config
        if c.nx < 8 or c.nx % 4 or c.ny < 1:
            raise ValueError("body-fitted nx must be >= 8 and divisible by four")
        opts = dict(dtype=torch.float64, device=c.device)
        self.body_label = "cylinder"
        if c.mesh_type == "c-grid":
            profile = _naca4_profile(c)
            if (any(x <= 0 or x >= c.length or y <= 0 or y >= c.height
                    for x, y in profile)
                    or not (0 < c.airfoil_x + 0.25 * c.airfoil_chord < c.length
                            and 0 < c.airfoil_y < c.height)):
                raise ValueError("NACA airfoil must lie strictly inside the C-grid domain")
            self.body_label = "airfoil"
            camber = int(c.airfoil_code[0]) / 100
            position = int(c.airfoil_code[1]) / 10
            quarter_camber, _ = _naca4_camberline(camber, position, 0.25)
            center_xy = (c.airfoil_x + 0.25 * c.airfoil_chord,
                         c.airfoil_y + quarter_camber * c.airfoil_chord)
            corner_xy = [(c.length, c.height), (0, c.height), (0, 0), (c.length, 0)]
            corner_angles = sorted(
                math.atan2(y - center_xy[1], x - center_xy[0]) % (2 * math.pi)
                for x, y in corner_xy
            )
            breaks = [0.0, *corner_angles, 2 * math.pi]
            spans = [b - a for a, b in zip(breaks, breaks[1:])]
            counts = [max(1, int(c.nx * span / (2 * math.pi))) for span in spans]
            while sum(counts) < c.nx:
                index = max(range(len(spans)),
                            key=lambda i: c.nx * spans[i] / (2 * math.pi) - counts[i])
                counts[index] += 1
            while sum(counts) > c.nx:
                candidates = [i for i, count in enumerate(counts) if count > 1]
                index = min(candidates, key=lambda i: (
                    c.nx * spans[i] / (2 * math.pi) - counts[i]
                ))
                counts[index] -= 1
            angles = [0.0]
            for (a, b), count in zip(zip(breaks, breaks[1:]), counts):
                angles.extend(a + (b - a) * i / count for i in range(1, count + 1))
            theta = torch.tensor(angles, **opts)
            center = torch.tensor(center_xy, **opts)
            corner_indices = [
                min(range(c.nx + 1), key=lambda i: abs(angles[i] - angle))
                for angle in corner_angles
            ]
        else:
            r = c.cylinder_radius
            if not r or not (r < c.cylinder_x < c.length - r
                             and r < c.cylinder_y < c.height - r):
                raise ValueError("body-fitted cylinder must lie strictly inside the channel")
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
        tx, ty = torch.full_like(theta, math.inf), torch.full_like(theta, math.inf)
        positive_x, negative_x = ray[:, 0] > 1e-14, ray[:, 0] < -1e-14
        positive_y, negative_y = ray[:, 1] > 1e-14, ray[:, 1] < -1e-14
        tx[positive_x] = (c.length - center[0]) / ray[positive_x, 0]
        tx[negative_x] = -center[0] / ray[negative_x, 0]
        ty[positive_y] = (c.height - center[1]) / ray[positive_y, 1]
        ty[negative_y] = -center[1] / ray[negative_y, 1]
        outer = center + torch.minimum(tx, ty)[:, None] * ray
        if c.mesh_type == "c-grid":
            # Intersect each ray from inside the airfoil with its polygonal surface.
            inner_points = []
            for angle in angles:
                direction = (math.cos(angle), math.sin(angle))
                distances = []
                for a, b in zip(profile, profile[1:] + profile[:1]):
                    edge = (b[0] - a[0], b[1] - a[1])
                    denominator = _cross(direction, edge)
                    if abs(denominator) < 1e-14:
                        continue
                    offset = (a[0] - center_xy[0], a[1] - center_xy[1])
                    distance = _cross(offset, edge) / denominator
                    fraction = _cross(offset, direction) / denominator
                    if distance > 0 and -1e-12 <= fraction <= 1 + 1e-12:
                        distances.append(distance)
                if not distances:
                    raise ValueError("could not intersect a ray with the NACA airfoil")
                distance = min(distances)
                inner_points.append((center_xy[0] + distance * direction[0],
                                     center_xy[1] + distance * direction[1]))
            inner = torch.tensor(inner_points, **opts)
            corners = [(c.length, c.height), (0, c.height), (0, 0), (c.length, 0)]
            for index, corner in zip(corner_indices, corners):
                outer[index] = torch.tensor(corner, **opts)
            inner[-1], outer[-1] = inner[0], outer[0]
        else:
            # Set corners exactly, including the periodic seam.
            for k, corner in enumerate(corners):
                outer[k * (c.nx // 4)] = torch.tensor(corner, **opts)
            inner = center + c.cylinder_radius * ray
            inner[-1] = inner[0]
        outer[-1] = outer[0]
        eta = torch.linspace(0, 1, c.ny + 1, **opts)
        if c.mesh_type == "c-grid":
            stretch = 2.5
            t = torch.expm1(stretch * eta) / math.expm1(stretch)
        elif c.mesh_type == "body-fitted" and c.body_fitted_stretching:
            stretch = c.body_fitted_stretching
            t = torch.expm1(stretch * eta) / math.expm1(stretch)
        else:
            t = eta
        t = t[:, None, None]
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
                        label = self.body_label
                    elif a[0] == b[0] == c.ny:
                        midpoint = (self.vertices[a] + self.vertices[b]) / 2
                        if abs(float(midpoint[0])) < 1e-12 * c.length:
                            label = "inlet"
                        elif abs(float(midpoint[0]) - c.length) < 1e-12 * c.length:
                            label = "outlet"
                        elif c.mesh_type == "c-grid" or c.outer_boundary == "far-field":
                            label = "far-field"
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
                      for name in ("inlet", "outlet", "wall", "far-field",
                                   "cylinder", "airfoil")}


class CGridMesh(BodyFittedMesh):
    """A C-type mesh around a NACA four-digit airfoil with a joined wake seam."""

    def __init__(self, config):
        if config.mesh_type != "c-grid":
            raise ValueError("CGridMesh requires mesh_type='c-grid'")
        super().__init__(config)


class FlatPlateMesh:
    """Wall-resolved structured mesh for a zero-pressure-gradient flat plate.

    The bottom edge is a no-slip plate, the upper edge is a far-field boundary,
    and the wall-normal coordinates are exponentially clustered at the plate.
    Its face data deliberately matches :class:`BodyFittedMesh` so the same
    collocated SIMPLE and RANS implementation can be used without an immersed
    boundary or a stair-step wall.
    """

    def __init__(self, config):
        if config.mesh_type != "flat-plate":
            raise ValueError("FlatPlateMesh requires mesh_type='flat-plate'")
        c = config
        opts = dict(dtype=torch.float64, device=c.device)
        x = torch.linspace(0, c.length, c.nx + 1, **opts)
        eta = torch.linspace(0, 1, c.ny + 1, **opts)
        if c.flat_plate_stretching:
            t = torch.expm1(c.flat_plate_stretching * eta) / math.expm1(
                c.flat_plate_stretching
            )
        else:
            t = eta
        y = c.height * t
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        self.vertices = torch.stack((grid_x, grid_y), -1)
        polygon = torch.stack((self.vertices[:-1, :-1], self.vertices[:-1, 1:],
                               self.vertices[1:, 1:], self.vertices[1:, :-1]), -2)
        nxt = polygon.roll(-1, -2)
        cross = polygon[..., 0] * nxt[..., 1] - nxt[..., 0] * polygon[..., 1]
        self.volumes = cross.sum(-1) / 2
        self.centers = ((polygon + nxt) * cross[..., None]).sum(-2) / (
            6 * self.volumes[..., None]
        )
        if not torch.all(self.volumes > 0):
            raise ValueError("invalid flat-plate cell geometry")
        owners, neighbors, labels, endpoints = [], [], [], []
        known = {}
        for j in range(c.ny):
            for i in range(c.nx):
                ids = [(j, i), (j, i + 1), (j + 1, i + 1), (j + 1, i)]
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
                        label = "wall"
                    elif a[0] == b[0] == c.ny:
                        label = "far-field"
                    elif a[1] == b[1] == 0:
                        label = "inlet"
                    elif a[1] == b[1] == c.nx:
                        label = "outlet"
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
                      for name in ("inlet", "outlet", "wall", "far-field",
                                   "cylinder", "airfoil")}


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
    mesh: object
    mass_flux: torch.Tensor
    boundary_velocity: torch.Tensor
    aerodynamic_coefficients: dict[str, float] | None = None
    surface_forces: dict[str, dict[str, float]] | None = None
    nu_tilde: torch.Tensor | None = None
    turbulent_kinematic_viscosity: torch.Tensor | None = None
    force_history: list[dict[str, float | int]] | None = None

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
        self.mesh = m = (
            CGridMesh(config) if config.mesh_type == "c-grid"
            else FlatPlateMesh(config) if config.mesh_type == "flat-plate"
            else BodyFittedMesh(config)
        )
        self.o, self.n = m.owner, m.neighbor
        self.f = m.interior
        self.oi, self.ni = self.o[self.f], self.n[self.f]
        self.count = config.nx * config.ny
        opts = dict(dtype=torch.float64, device=config.device)
        alpha = math.radians(config.angle_of_attack)
        freestream = torch.tensor(
            [config.inlet_velocity * math.cos(alpha),
             config.inlet_velocity * math.sin(alpha)], **opts
        )
        self.velocity = torch.zeros((self.count, 2), **opts)
        self.velocity[:] = freestream
        if config.initial_perturbation:
            centre = m.centers.reshape(-1, 2)
            # Deterministic antisymmetric seed breaks the mathematically exact
            # symmetric URANS state without introducing random-number/device
            # dependence into a shedding benchmark.
            pattern = (torch.sin(math.pi * centre[:, 0] / config.length)
                       * torch.sin(2 * math.pi * centre[:, 1] / config.height))
            self.velocity[:, 1] += config.initial_perturbation * pattern
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
        inflow = m.masks["inlet"] | m.masks["far-field"]
        self.boundary_velocity[inflow] = freestream
        if config.inlet_profile == "parabolic":
            inlet = m.masks["inlet"]
            endpoints = m.face_vertices[inlet, :, 1]
            y0, y1 = endpoints[:, 0], endpoints[:, 1]
            # Exact face average of 6 U_mean (y/H) (1-y/H).  Keeping the
            # integral exact makes the prescribed total inlet flux rho*U*H
            # independent of the number and placement of inlet faces.
            self.boundary_velocity[inlet, 0] = (
                6 * config.inlet_velocity
                * ((y1.square() - y0.square()) / (2 * config.height)
                   - (y1.pow(3) - y0.pow(3)) / (3 * config.height ** 2))
                / (y1 - y0)
            )
            self.boundary_velocity[inlet, 1] = 0
        face_velocity = self.velocity[self.o].clone()
        fixed = m.boundary & ~m.masks["outlet"]
        face_velocity[fixed] = self.boundary_velocity[fixed]
        self.mass_flux = config.density * (face_velocity * self.S).sum(-1)
        self.history = []
        self.force_history = []
        self.time = 0.0
        self.converged = False
        self._gradient_weights = {}
        self.kinematic_viscosity = config.viscosity / config.density
        self.nu_tilde = torch.zeros(self.count, **opts)
        self.turbulent_kinematic_viscosity = torch.zeros_like(self.nu_tilde)
        self.sa_boundary_nu_tilde = torch.zeros(len(self.o), **opts)
        self.wall_distance = None
        if config.turbulence_model == "spalart-allmaras":
            self.nu_tilde.fill_(config.sa_freestream_ratio * self.kinematic_viscosity)
            self.turbulent_kinematic_viscosity = self._sa_eddy_viscosity(self.nu_tilde)
            self.sa_boundary_nu_tilde[inflow] = (
                config.sa_freestream_ratio * self.kinematic_viscosity
            )
            self.wall_distance = self._wall_distance()

    def _sum(self, face):
        result = torch.zeros((self.count,) + face.shape[1:], dtype=face.dtype,
                             device=face.device)
        result.index_add_(0, self.o, face)
        result.index_add_(0, self.ni, -face[self.f])
        return result

    def _wall_distance(self):
        """Return the exact minimum distance from each centre to a no-slip face."""
        wall = (self.mesh.masks["wall"] | self.mesh.masks["cylinder"]
                | self.mesh.masks["airfoil"])
        if not bool(wall.any()):
            raise ValueError("Spalart-Allmaras requires at least one no-slip wall")
        endpoints = self.mesh.face_vertices[wall]
        start, edge = endpoints[:, 0], endpoints[:, 1] - endpoints[:, 0]
        length_squared = edge.square().sum(-1).clamp_min(torch.finfo(self.p.dtype).tiny)
        centers = self.mesh.centers.reshape(-1, 2)
        distance = torch.empty(self.count, dtype=self.p.dtype, device=self.p.device)
        # Chunking bounds memory on the DFG-sized grids while retaining exact
        # point-to-segment distances at corners and on curved polygon faces.
        for begin in range(0, self.count, 2048):
            point = centers[begin:begin + 2048, None, :]
            fraction = (((point - start[None]) * edge[None]).sum(-1)
                        / length_squared[None]).clamp(0, 1)
            closest = start[None] + fraction[..., None] * edge[None]
            distance[begin:begin + 2048] = torch.linalg.vector_norm(
                point - closest, dim=-1
            ).min(-1).values
        return distance.clamp_min(torch.finfo(self.p.dtype).eps)

    def _sa_eddy_viscosity(self, nu_tilde):
        """SA conversion from working variable to kinematic eddy viscosity."""
        chi = nu_tilde.clamp_min(0) / self.kinematic_viscosity
        fv1 = chi.pow(3) / (chi.pow(3) + 7.1 ** 3)
        return nu_tilde.clamp_min(0) * fv1

    def _gradient(self, field, pressure=False, boundary_values=None):
        """Least-squares gradient with outlet pressure or supplied Dirichlet data."""
        scalar = field.ndim == 1
        q = field[:, None] if scalar else field
        delta = torch.zeros((len(self.o), q.shape[1]), dtype=q.dtype, device=q.device)
        delta[self.f] = q[self.ni] - q[self.oi]
        active = self.f.clone()
        if pressure:
            active |= self.mesh.masks["outlet"]
            delta[self.mesh.masks["outlet"]] = -q[self.o[self.mesh.masks["outlet"]]]
            cache_key = "pressure"
        else:
            fixed = self.mesh.boundary & ~self.mesh.masks["outlet"]
            active |= fixed
            values = self.boundary_velocity if boundary_values is None else boundary_values
            values = values[:, None] if values.ndim == 1 else values
            delta[fixed] = values[fixed] - q[self.o[fixed]]
            # Turbulence and velocity share this same fixed-face topology.
            cache_key = "non-outlet-dirichlet"
        if cache_key not in self._gradient_weights:
            weight = active.to(q.dtype) / (self.d * self.d).sum(-1)
            matrix_face = weight[:, None, None] * self.d[:, :, None] * self.d[:, None, :]
            matrix = torch.zeros((self.count, 2, 2), dtype=q.dtype, device=q.device)
            matrix.index_add_(0, self.o, matrix_face)
            matrix.index_add_(0, self.ni, matrix_face[self.f])
            inverse = torch.linalg.inv(matrix)
            wd = weight[:, None] * self.d
            go = torch.einsum("fij,fj->fi", inverse[self.o], wd)
            gn = torch.einsum("fij,fj->fi", inverse[self.ni], wd[self.f])
            self._gradient_weights[cache_key] = go, gn
        go, gn = self._gradient_weights[cache_key]
        gradient = torch.zeros((self.count, 2, q.shape[1]), dtype=q.dtype, device=q.device)
        gradient.index_add_(0, self.o, go[:, :, None] * delta[:, None, :])
        gradient.index_add_(0, self.ni, gn[:, :, None] * delta[self.f, None, :])
        return gradient[..., 0] if scalar else gradient

    def _interpolate(self, field):
        result = field[self.o].clone()
        result[self.f] = (field[self.oi] + field[self.ni]) / 2
        return result

    @property
    def _dynamic_viscosity(self):
        return (self.config.viscosity
                + self.config.density * self.turbulent_kinematic_viscosity)

    def _momentum(self, velocity, pressure, flux):
        c, m = self.config, self.mesh
        face_viscosity = self._interpolate(self._dynamic_viscosity)
        no_slip = m.masks["wall"] | m.masks["cylinder"] | m.masks["airfoil"]
        face_viscosity[no_slip] = c.viscosity
        diffusion = face_viscosity * self.k
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
        correction = face_viscosity[:, None] * torch.einsum("fi,fij->fj", self.T, gradient)
        correction[m.masks["outlet"]] = 0
        source += self._sum(correction) - self.volume[:, None] * self._gradient(
            pressure, pressure=True)
        # Extrapolated backflow uses the owner velocity (zero-gradient outlet).
        diagonal.index_add_(0, self.o[m.masks["outlet"]],
                            flux[m.masks["outlet"]].clamp_max(0))
        return diagonal, ao, an, source

    def _sa_system(self, nu_tilde):
        """Assemble a fully turbulent, wall-resolved Spalart--Allmaras step.

        The working variable is kinematic ``nu_tilde``.  Diffusion and
        convection are implicit; production, the nonlinear destruction term,
        and the nonorthogonal correction are lagged once per SIMPLE iteration.
        This is the original fully turbulent SA model (trip terms disabled).
        """
        if self.config.turbulence_model != "spalart-allmaras":
            raise RuntimeError("SA transport requested for a laminar configuration")
        c, m = self.config, self.mesh
        sigma, cb1, cb2, kappa = 2 / 3, 0.1355, 0.622, 0.41
        cw2, cw3 = 0.3, 2.0
        cw1 = cb1 / kappa ** 2 + (1 + cb2) / sigma
        no_slip = m.masks["wall"] | m.masks["cylinder"] | m.masks["airfoil"]
        gamma = c.density * (self.kinematic_viscosity + nu_tilde.clamp_min(0)) / sigma
        face_gamma = self._interpolate(gamma)
        face_gamma[no_slip] = c.viscosity / sigma
        diffusion = face_gamma * self.k
        diffusion[m.masks["outlet"]] = 0
        flux = self.mass_flux
        diagonal = torch.zeros_like(self.p)
        diagonal.index_add_(0, self.o, diffusion + flux.clamp_min(0))
        diagonal.index_add_(0, self.ni,
                            diffusion[self.f] + (-flux[self.f]).clamp_min(0))
        ao = diffusion[self.f] + (-flux[self.f]).clamp_min(0)
        an = diffusion[self.f] + flux[self.f].clamp_min(0)
        fixed = m.boundary & ~m.masks["outlet"]
        source = self._sum(torch.where(
            fixed,
            (diffusion + (-flux).clamp_min(0)) * self.sa_boundary_nu_tilde,
            torch.zeros_like(diffusion),
        ))
        gradient = self._gradient(nu_tilde, boundary_values=self.sa_boundary_nu_tilde)
        correction = face_gamma * (self.T * self._interpolate(gradient)).sum(-1)
        correction[m.masks["outlet"]] = 0
        source += self._sum(correction)
        velocity_gradient = self._gradient(self.velocity)
        vorticity = (velocity_gradient[:, 1, 0] - velocity_gradient[:, 0, 1]).abs()
        chi = nu_tilde.clamp_min(0) / self.kinematic_viscosity
        fv1 = chi.pow(3) / (chi.pow(3) + 7.1 ** 3)
        fv2 = 1 - chi / (1 + chi * fv1)
        distance_squared = self.wall_distance.square()
        s_tilde = (vorticity + nu_tilde.clamp_min(0) * fv2
                   / (kappa ** 2 * distance_squared)).clamp_min(1e-14)
        ratio = (nu_tilde.clamp_min(0)
                 / (s_tilde * kappa ** 2 * distance_squared)).clamp(max=10)
        g = ratio + cw2 * (ratio.pow(6) - ratio)
        fw = g * ((1 + cw3 ** 6) / (g.pow(6) + cw3 ** 6)).pow(1 / 6)
        production = c.density * self.volume * cb1 * s_tilde * nu_tilde.clamp_min(0)
        cross_diffusion = (c.density * self.volume * cb2 / sigma
                           * gradient.square().sum(-1))
        destruction = (c.density * self.volume * cw1 * fw * nu_tilde.clamp_min(0)
                       / distance_squared)
        source += production + cross_diffusion
        diagonal += destruction
        # A zero-gradient outlet may receive backflow, exactly as momentum does.
        diagonal.index_add_(0, self.o[m.masks["outlet"]],
                            flux[m.masks["outlet"]].clamp_max(0))
        return diagonal, ao, an, source

    def _turbulence_step(self, physical_old=None):
        """Advance SA once; ``physical_old`` activates implicit Euler URANS."""
        if self.config.turbulence_model == "laminar":
            return 0.0, 0.0
        old = self.nu_tilde.clone()
        diagonal, ao, an, source = self._sa_system(old)
        inertia = torch.zeros_like(diagonal)
        if physical_old is not None:
            inertia = self.config.density * self.volume / self.config.time_step
        alpha = self.config.turbulence_relaxation
        relaxed = (diagonal + inertia) / alpha
        rhs = source + (1 - alpha) * relaxed * old
        if physical_old is not None:
            rhs += inertia * physical_old
        # Strong wall-normal stretching can stagnate BiCGSTAB before its very
        # small inner target.  The outer SA residual below remains mandatory,
        # so an inexact transport update is retried on every SIMPLE iteration.
        solved = self._linear(relaxed, ao, an, rhs, old, threshold_floor=1e-12,
                              allow_inexact=True)
        self.nu_tilde = (alpha * solved + (1 - alpha) * old).clamp_min(0)
        self.turbulent_kinematic_viscosity = self._sa_eddy_viscosity(self.nu_tilde)
        diagonal, ao, an, source = self._sa_system(self.nu_tilde)
        residual = self._matvec(self.nu_tilde, diagonal, ao, an) - source
        if physical_old is not None:
            residual += inertia * (self.nu_tilde - physical_old)
        scale = max(self.config.density * self.config.inlet_velocity
                    * self.config.reference_length * self.kinematic_viscosity,
                    torch.finfo(self.p.dtype).tiny)
        return float(residual.abs().max() / scale), float((self.nu_tilde - old).abs().max()
                                                           / self.kinematic_viscosity)

    def _matvec(self, x, diagonal, ao, an):
        y = diagonal * x
        y.index_add_(0, self.oi, -ao * x[self.ni])
        y.index_add_(0, self.ni, -an * x[self.oi])
        return y

    def _linear(self, diagonal, ao, an, rhs, initial, symmetric=False, operator=None,
                threshold_floor=1e-10, relative_tolerance=1e-9, allow_inexact=False):
        """Diagonal-preconditioned CG or BiCGSTAB, with a true residual check."""
        x = initial.clone()
        mv = operator if operator is not None else lambda z: self._matvec(z, diagonal, ao, an)
        residual = rhs - mv(x)
        # Solving each segregated equation close to machine precision is both
        # unnecessary and, on refined highly stretched meshes, can trigger a
        # false breakdown after the outer SIMPLE residual is already orders of
        # magnitude larger.  This remains substantially tighter than the
        # supported outer tolerances while avoiding round-off stagnation.
        threshold = max(threshold_floor,
                        float(torch.linalg.vector_norm(rhs)) * relative_tolerance)
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
        true_residual = torch.linalg.vector_norm(rhs - mv(x))
        if not torch.isfinite(true_residual):
            raise RuntimeError("nonfinite body-fitted linear residual")
        if float(true_residual) > 10 * threshold:
            if not allow_inexact:
                raise RuntimeError(
                    f"body-fitted linear solve did not converge: residual "
                    f"{float(true_residual):.3g}, tolerance {10 * threshold:.3g}")
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
            # Retain pseudo-time/under-relaxation face-flux memory. At a fixed
            # point its damping cancels, leaving the unrelaxed, inertia-free
            # Rhie--Chow operator.
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

    def _correct(self, predicted_flux, coefficient, D, steady_D=None):
        diagonal = self._sum(coefficient * 0)
        diagonal.index_add_(0, self.o, coefficient)
        diagonal.index_add_(0, self.ni, coefficient[self.f])
        rhs = -self._sum(predicted_flux)
        # Pure under-relaxation has constant damping; its interpolation
        # response cancels identically, avoiding two extra gathers per matvec.
        nonuniform_damping = steady_D is not None and self.config.pseudo_time_step is not None
        if nonuniform_damping:
            beta = (1 - self._interpolate(D) / self._interpolate(steady_D)).clamp(0, 1)

        def additional_flux(q):
            gradient = self._gradient(q, pressure=True)
            face_gradient = self._interpolate(gradient)
            flux = -coefficient / self.k * (self.T * face_gradient).sum(-1)
            if nonuniform_damping:
                # Exact sensitivity of the damped predictor, including the
                # collocated velocity correction. With nonuniform damping,
                # interpolation and multiplication by D do not commute.
                response = ((1 - beta)[:, None]
                            * self._interpolate(steady_D[:, None] * gradient)
                            - self._interpolate(D[:, None] * gradient))
                extra = self.config.density * (response * self.S).sum(-1)
                flux += extra.masked_fill(coefficient == 0, 0)
            return flux

        def operator(q):
            return (self._matvec(q, diagonal, coefficient[self.f], coefficient[self.f])
                    + self._sum(additional_flux(q)))

        flat_plate = self.config.mesh_type == "flat-plate"
        correction = self._linear(
            diagonal, coefficient[self.f], coefficient[self.f], rhs,
            torch.zeros_like(self.p), operator=operator,
            # The high-Re wall-resolved plate matrix is far more ill-conditioned
            # than the existing O/C grids.  Its outer continuity tolerance is
            # still two orders looser than this accepted inner residual.
            threshold_floor=1e-8 if flat_plate else 1e-10,
            relative_tolerance=1e-8 if flat_plate else 1e-9,
        )
        delta = correction[self.o].clone()
        delta[self.f] -= correction[self.ni]
        self.mass_flux = predicted_flux + coefficient * delta + additional_flux(correction)
        self.velocity -= D[:, None] * self._gradient(correction, pressure=True)
        self.p += self.config.pressure_relaxation * correction

    def _surface_force(self, mask):
        """Integrate pressure and molecular wall shear over selected no-slip faces."""
        if not bool(mask.any()):
            return torch.zeros(2, dtype=self.p.dtype, device=self.p.device)
        owners = self.o[mask]
        velocity_gradient = self._gradient(self.velocity)
        # At a no-slip wall the SA eddy viscosity is zero.  Reconstructing the
        # molecular wall gradient at the actual face avoids using a cell-centre
        # eddy viscosity as a wall-function surrogate.
        wall_d = self.d[mask]
        wall_gradient = velocity_gradient[owners].clone()
        wall_error = (-self.velocity[owners]
                      - torch.einsum("fi,fij->fj", wall_d, wall_gradient))
        wall_gradient += (wall_d[:, :, None] * wall_error[:, None, :]
                          / wall_d.square().sum(-1)[:, None, None])
        stress = self.config.viscosity * (
            wall_gradient + wall_gradient.transpose(-1, -2)
        )
        traction = torch.einsum("fij,fj->fi", stress, self.S[mask])
        return (self.p[owners, None] * self.S[mask] - traction).sum(0)

    def _aerodynamic_coefficients(self):
        if self.config.mesh_type not in ("c-grid", "body-fitted"):
            return None
        c = self.config
        body = "airfoil" if c.mesh_type == "c-grid" else "cylinder"
        mask = self.mesh.masks[body]
        force = self._surface_force(mask)
        reference_length = (c.airfoil_chord if body == "airfoil"
                            else 2 * c.cylinder_radius)
        scale = 0.5 * c.density * c.inlet_velocity ** 2 * reference_length
        # Report aerodynamic axes, not fixed global x/y components.  They are
        # identical for the cylinder benchmark (alpha=0), while this rotation
        # is essential for a lifting airfoil at nonzero incidence.
        alpha = math.radians(c.angle_of_attack if body == "airfoil" else 0.0)
        drag = force[0] * math.cos(alpha) + force[1] * math.sin(alpha)
        lift = -force[0] * math.sin(alpha) + force[1] * math.cos(alpha)
        return {"drag": float(drag / scale), "lift": float(lift / scale)}

    @torch.no_grad()
    def _steady_step(self):
        """Advance one SIMPLE iteration and return conservative steady metrics."""
        c = self.config
        inlet_mass = c.density * c.inlet_velocity * c.height
        residual_length = (c.airfoil_chord if c.mesh_type == "c-grid" else
                           c.length if c.mesh_type == "flat-plate" else c.height)
        force_scale = max(c.density * c.inlet_velocity ** 2 * residual_length,
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
        steady_D = self.volume / diagonal
        flux, coefficient = self._rhie_chow(self.velocity, self.p, D, steady_D, old)
        self._correct(flux, coefficient, D, steady_D)
        turbulence, turbulence_change = self._turbulence_step()
        if not (torch.isfinite(self.velocity).all() and torch.isfinite(self.p).all()
                and torch.isfinite(self.mass_flux).all()
                and torch.isfinite(self.nu_tilde).all()):
            raise RuntimeError("nonfinite body-fitted SIMPLE or turbulence iterate")
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
        if c.turbulence_model == "spalart-allmaras":
            metrics["turbulence"] = turbulence
            metrics["turbulence_change"] = turbulence_change
        self.history.append(metrics)
        criteria = (continuity, momentum, imbalance)
        if c.turbulence_model == "spalart-allmaras":
            criteria += (turbulence,)
        self.converged = max(criteria) < c.tolerance
        return metrics

    def _transient_step(self):
        """Advance one physical implicit-Euler URANS step with SIMPLE subiterations."""
        c = self.config
        physical_velocity = self.velocity.clone()
        physical_nu = self.nu_tilde.clone()
        inlet_mass = c.density * c.inlet_velocity * c.height
        residual_length = (c.airfoil_chord if c.mesh_type == "c-grid" else
                           c.length if c.mesh_type == "flat-plate" else c.height)
        force_scale = max(c.density * c.inlet_velocity ** 2 * residual_length,
                          c.viscosity * c.inlet_velocity)
        inertia = c.density * self.volume / c.time_step
        for inner in range(c.inner_iterations):
            old = self.velocity.clone()
            diagonal, ao, an, source = self._momentum(old, self.p, self.mass_flux)
            relaxed = (diagonal + inertia) / c.velocity_relaxation
            rhs = (source + inertia[:, None] * physical_velocity
                   + (1 - c.velocity_relaxation) * relaxed[:, None] * old)
            for component in range(2):
                self.velocity[:, component] = self._linear(
                    relaxed, ao, an, rhs[:, component], old[:, component],
                    allow_inexact=True)
            D = self.volume / relaxed
            flux, coefficient = self._rhie_chow(self.velocity, self.p, D)
            self._correct(flux, coefficient, D)
            turbulence, turbulence_change = self._turbulence_step(physical_nu)
            if not (torch.isfinite(self.velocity).all() and torch.isfinite(self.p).all()
                    and torch.isfinite(self.mass_flux).all()
                    and torch.isfinite(self.nu_tilde).all()):
                raise RuntimeError("nonfinite transient URANS iterate")
            diagonal, ao, an, source = self._momentum(
                self.velocity, self.p, self.mass_flux)
            residual = torch.stack([
                self._matvec(self.velocity[:, k], diagonal, ao, an) - source[:, k]
                + inertia * (self.velocity[:, k] - physical_velocity[:, k])
                for k in range(2)
            ], -1)
            continuity = float(self._sum(self.mass_flux).abs().max()) / inlet_mass
            momentum = float(residual.abs().sum(0).max()) / force_scale
            imbalance = abs(float(self.mass_flux[self.mesh.boundary].sum())) / inlet_mass
            criteria = (continuity, momentum, imbalance)
            if c.turbulence_model == "spalart-allmaras":
                criteria += (turbulence,)
            if max(criteria) < c.tolerance:
                break
        self.time += c.time_step
        metrics = dict(iteration=len(self.history) + 1, time=self.time,
                       inner_iterations=inner + 1, continuity=continuity,
                       momentum=momentum, mass_imbalance=imbalance,
                       velocity_change=float((self.velocity - physical_velocity).abs().max())
                       / c.inlet_velocity,
                       u_residual=float(residual[:, 0].abs().sum()) / force_scale,
                       v_residual=float(residual[:, 1].abs().sum()) / force_scale)
        if c.turbulence_model == "spalart-allmaras":
            metrics["turbulence"] = turbulence
            metrics["turbulence_change"] = turbulence_change
        self.history.append(metrics)
        self.converged = max(criteria) < c.tolerance
        coefficients = self._aerodynamic_coefficients()
        if coefficients is not None:
            self.force_history.append({"step": len(self.history), "time": self.time,
                                       **coefficients})
        return metrics

    @torch.no_grad()
    def step(self):
        """Advance one steady SIMPLE iteration or one physical URANS time step."""
        if self.config.time_step is not None:
            return self._transient_step()
        return self._steady_step()

    def solve(self):
        c = self.config
        for _ in range(max(0, c.max_iterations - len(self.history))):
            if c.time_step is None and self.converged:
                break
            self.step()
            if c.time_step is None and self.converged:
                break
        shape = (c.ny, c.nx)
        surface_name = ("plate" if c.mesh_type == "flat-plate" else
                        "airfoil" if c.mesh_type == "c-grid" else "cylinder")
        surface_mask = self.mesh.masks["wall" if surface_name == "plate" else surface_name]
        surface_force = self._surface_force(surface_mask)
        surface_forces = {surface_name: {"x": float(surface_force[0]),
                                         "y": float(surface_force[1])}}
        return BodyFittedResult(c, self.velocity[:, 0].reshape(shape).clone(),
                                self.velocity[:, 1].reshape(shape).clone(),
                                self.p.reshape(shape).clone(),
                                torch.ones(shape, dtype=torch.bool, device=c.device),
                                self.mesh.centers[..., 0].clone(),
                                self.mesh.centers[..., 1].clone(),
                                [item.copy() for item in self.history], self.converged,
                                self.mesh, self.mass_flux.clone(),
                                self.boundary_velocity.clone(),
                                self._aerodynamic_coefficients(), surface_forces,
                                self.nu_tilde.reshape(shape).clone()
                                if c.turbulence_model == "spalart-allmaras" else None,
                                self.turbulent_kinematic_viscosity.reshape(shape).clone()
                                if c.turbulence_model == "spalart-allmaras" else None,
                                [dict(item) for item in self.force_history]
                                if self.force_history else None)
