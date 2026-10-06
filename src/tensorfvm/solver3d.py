"""Three-dimensional transient cylinder-flow prototype with shared runtime APIs.

The first 3-D kernel is a structured, collocated fractional-step solver.  It is
purposefully isolated from the mature 2-D SIMPLE implementation: a 3-D
body-fitted/DES solver must not be claimed before its mesh and halo operators
exist.  This module provides the executable 3-D external-cylinder baseline and
a stable API for replacing its immersed Cartesian discretisation.
"""

from dataclasses import dataclass
import math

import torch

from .mesh3d import CartesianCylinderMesh3D
from .runtime import DistributedRuntime


@dataclass
class Cylinder3DConfig:
    nx: int = 64
    ny: int = 40
    nz: int = 16
    length: float = 20.0
    height: float = 12.0
    span: float = 3.0
    cylinder_x: float = 5.0
    cylinder_y: float = 6.0
    cylinder_radius: float = 0.5
    inlet_velocity: float = 1.0
    reynolds: float = 3900.0
    density: float = 1.0
    time_step: float = 0.01
    max_steps: int = 100
    pressure_iterations: int = 200
    smagorinsky_constant: float = 0.1
    device: str = "cpu"

    def __post_init__(self):
        for name in ("nx", "ny", "nz", "max_steps", "pressure_iterations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if min(self.nx, self.ny, self.nz) < 4:
            raise ValueError("nx, ny, and nz must each be at least 4")
        for name in ("length", "height", "span", "cylinder_x", "cylinder_y",
                     "cylinder_radius", "inlet_velocity", "reynolds", "density",
                     "time_step"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.smagorinsky_constant) or self.smagorinsky_constant < 0:
            raise ValueError("smagorinsky_constant must be finite and nonnegative")
        dx, dy = self.length / self.nx, self.height / self.ny
        if not (self.cylinder_radius + dx < self.cylinder_x < self.length - self.cylinder_radius - dx
                and self.cylinder_radius + dy < self.cylinder_y < self.height - self.cylinder_radius - dy):
            raise ValueError("cylinder must lie inside the 3-D domain with grid clearance")
        try:
            target = torch.device(self.device)
            torch.empty(0, dtype=torch.float64, device=target)
        except (RuntimeError, ValueError, TypeError, AssertionError) as exc:
            raise ValueError(f"unavailable float64 device: {self.device}") from exc

    @property
    def diameter(self) -> float:
        return 2 * self.cylinder_radius

    @property
    def kinematic_viscosity(self) -> float:
        return self.inlet_velocity * self.diameter / self.reynolds


@dataclass
class Cylinder3DResult:
    config: Cylinder3DConfig
    velocity: torch.Tensor
    pressure: torch.Tensor
    fluid: torch.Tensor
    x: torch.Tensor
    y: torch.Tensor
    z: torch.Tensor
    history: list[dict[str, float | int]]
    force_history: list[dict[str, float | int]]
    runtime: DistributedRuntime


class Cylinder3DSolver:
    """3-D incompressible projection solver with periodic spanwise direction.

    x is streamwise, y is cross-stream, and z is periodic.  A Smagorinsky SGS
    viscosity is available for high-Re exploratory LES.  The solver currently
    runs one global structured domain per rank; ``DistributedRuntime`` supplies
    partition metadata but true z-slab halo exchange is intentionally rejected
    until the common halo operator is integrated.
    """

    def __init__(self, config: Cylinder3DConfig, runtime: DistributedRuntime | None = None):
        self.config = config
        self.runtime = runtime or DistributedRuntime.discover(config.device)
        if self.runtime.world_size != 1:
            raise NotImplementedError(
                "3-D z-slab decomposition is planned but halo exchange is not implemented; "
                "run this kernel with one rank"
            )
        self.mesh = CartesianCylinderMesh3D.build(config, self.runtime.device)
        self.velocity = torch.zeros((config.nz, config.ny, config.nx, 3),
                                    dtype=torch.float64, device=self.runtime.device)
        self.velocity[..., 0] = config.inlet_velocity
        self.pressure = torch.zeros(self.mesh.fluid.shape, dtype=torch.float64,
                                    device=self.runtime.device)
        self.history: list[dict[str, float | int]] = []
        self.force_history: list[dict[str, float | int]] = []
        self.time = 0.0
        self._apply_velocity_boundaries(self.velocity)

    def _x_neighbors(self, field):
        return (torch.cat((field[..., 1:], field[..., -1:]), dim=-1),
                torch.cat((field[..., :1], field[..., :-1]), dim=-1))

    def _y_neighbors(self, field):
        return (torch.cat((field[:, 1:], field[:, -1:]), dim=1),
                torch.cat((field[:, :1], field[:, :-1]), dim=1))

    def _derivative(self, field, axis: int, spacing: float):
        if axis == 2:
            forward, backward = self._x_neighbors(field)
        elif axis == 1:
            forward, backward = self._y_neighbors(field)
        else:
            forward, backward = torch.roll(field, -1, 0), torch.roll(field, 1, 0)
        return (forward - backward) / (2 * spacing)

    def _laplacian(self, field):
        m = self.mesh
        east, west = self._x_neighbors(field)
        north, south = self._y_neighbors(field)
        top, bottom = torch.roll(field, -1, 0), torch.roll(field, 1, 0)
        return ((east - 2 * field + west) / m.dx ** 2
                + (north - 2 * field + south) / m.dy ** 2
                + (top - 2 * field + bottom) / m.dz ** 2)

    def _apply_velocity_boundaries(self, velocity):
        c, fluid = self.config, self.mesh.fluid
        velocity.masked_fill_(~fluid[..., None], 0)
        # Uniform inlet and far field; outlet is zero-gradient.  z is periodic.
        velocity[:, :, 0] = 0
        velocity[:, :, 0, 0] = c.inlet_velocity
        velocity[:, 0] = 0
        velocity[:, 0, :, 0] = c.inlet_velocity
        velocity[:, -1] = 0
        velocity[:, -1, :, 0] = c.inlet_velocity
        velocity[:, :, -1] = velocity[:, :, -2]
        velocity.masked_fill_(~fluid[..., None], 0)

    def _apply_pressure_boundaries(self, pressure):
        fluid = self.mesh.fluid
        pressure[:, :, -1] = 0  # gauge reference at outlet
        pressure[:, :, 0] = pressure[:, :, 1]
        pressure[:, 0] = pressure[:, 1]
        pressure[:, -1] = pressure[:, -2]
        pressure.masked_fill_(~fluid, 0)

    def _divergence(self, velocity):
        return (self._derivative(velocity[..., 0], 2, self.mesh.dx)
                + self._derivative(velocity[..., 1], 1, self.mesh.dy)
                + self._derivative(velocity[..., 2], 0, self.mesh.dz)).masked_fill(
                    ~self.mesh.fluid, 0
                )

    def _velocity_gradient(self, velocity):
        return torch.stack([
            torch.stack([self._derivative(velocity[..., component], axis, spacing)
                         for axis, spacing in ((2, self.mesh.dx), (1, self.mesh.dy),
                                               (0, self.mesh.dz))], -1)
            for component in range(3)
        ], -2)

    def _eddy_viscosity(self, velocity):
        if self.config.smagorinsky_constant == 0:
            return torch.zeros_like(self.pressure)
        gradient = self._velocity_gradient(velocity)
        strain = 0.5 * (gradient + gradient.transpose(-1, -2))
        magnitude = torch.sqrt(2 * strain.square().sum(dim=(-1, -2)))
        delta = self.mesh.cell_volume ** (1 / 3)
        return ((self.config.smagorinsky_constant * delta) ** 2 * magnitude).masked_fill(
            ~self.mesh.fluid, 0
        )

    def _pressure_projection(self, rhs):
        m = self.mesh
        pressure = self.pressure.clone()
        denominator = 2 / m.dx ** 2 + 2 / m.dy ** 2 + 2 / m.dz ** 2
        for _ in range(self.config.pressure_iterations):
            east, west = self._x_neighbors(pressure)
            north, south = self._y_neighbors(pressure)
            top, bottom = torch.roll(pressure, -1, 0), torch.roll(pressure, 1, 0)
            update = ((east + west) / m.dx ** 2 + (north + south) / m.dy ** 2
                      + (top + bottom) / m.dz ** 2 - rhs) / denominator
            pressure = torch.where(m.fluid, update, torch.zeros_like(update))
            self._apply_pressure_boundaries(pressure)
        self.pressure = pressure

    def _surface_force(self):
        """Approximate pressure force on stair-step cylinder faces."""
        p, fluid = self.pressure, self.mesh.fluid
        solid = ~fluid
        east_solid = torch.cat((solid[..., 1:], torch.zeros_like(solid[..., :1])), -1)
        west_solid = torch.cat((torch.zeros_like(solid[..., :1]), solid[..., :-1]), -1)
        north_solid = torch.cat((solid[:, 1:], torch.zeros_like(solid[:, :1])), 1)
        south_solid = torch.cat((torch.zeros_like(solid[:, :1]), solid[:, :-1]), 1)
        fx = ((p * (fluid & east_solid)).sum() - (p * (fluid & west_solid)).sum())
        fy = ((p * (fluid & north_solid)).sum() - (p * (fluid & south_solid)).sum())
        fx *= self.mesh.dy * self.mesh.dz
        fy *= self.mesh.dx * self.mesh.dz
        scale = 0.5 * self.config.density * self.config.inlet_velocity ** 2 \
            * self.config.diameter * self.config.span
        return float(fx / scale), float(fy / scale)

    @torch.no_grad()
    def step(self):
        c, m = self.config, self.mesh
        old = self.velocity.clone()
        self._apply_velocity_boundaries(old)
        gradient = self._velocity_gradient(old)
        convection = torch.einsum("...j,...ij->...i", old, gradient)
        viscosity = c.kinematic_viscosity + self._eddy_viscosity(old)
        laplacian = torch.stack([self._laplacian(old[..., component])
                                 for component in range(3)], -1)
        tentative = old + c.time_step * (-convection + viscosity[..., None] * laplacian)
        self._apply_velocity_boundaries(tentative)
        self._pressure_projection(c.density / c.time_step * self._divergence(tentative))
        pressure_gradient = torch.stack((
            self._derivative(self.pressure, 2, m.dx),
            self._derivative(self.pressure, 1, m.dy),
            self._derivative(self.pressure, 0, m.dz),
        ), -1)
        self.velocity = tentative - c.time_step / c.density * pressure_gradient
        self._apply_velocity_boundaries(self.velocity)
        self.time += c.time_step
        divergence = self._divergence(self.velocity)
        velocity_change = (self.velocity - old).abs().max()
        cfl = (c.time_step * (self.velocity[..., 0].abs() / m.dx
                               + self.velocity[..., 1].abs() / m.dy
                               + self.velocity[..., 2].abs() / m.dz)).max()
        drag, lift = self._surface_force()
        metric = {
            "step": len(self.history) + 1,
            "time": self.time,
            "continuity": float(divergence.abs().max()) * c.diameter / c.inlet_velocity,
            "velocity_change": float(velocity_change) / c.inlet_velocity,
            "cfl": float(cfl),
            "max_eddy_viscosity": float(self._eddy_viscosity(self.velocity).max()),
        }
        if not all(math.isfinite(value) for value in metric.values()):
            raise RuntimeError("nonfinite 3-D projection iterate")
        self.history.append(metric)
        self.force_history.append({"step": metric["step"], "time": self.time,
                                   "drag": drag, "lift": lift})
        return metric

    def solve(self):
        while len(self.history) < self.config.max_steps:
            self.step()
        x, y, z = self.mesh.centers()
        return Cylinder3DResult(self.config, self.velocity.clone(), self.pressure.clone(),
                                self.mesh.fluid.clone(), x.clone(), y.clone(), z.clone(),
                                [dict(item) for item in self.history],
                                [dict(item) for item in self.force_history], self.runtime)
