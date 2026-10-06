"""Steady, confined-cylinder flow using staggered finite volumes and SIMPLE.

The Cartesian cylinder is a stair-step solid.  The inlet is uniform, the channel
walls and cylinder are no-slip, and the outlet has zero gauge pressure.  This is
a *confined channel* problem, not the unbounded-cylinder drag benchmark.
"""

from dataclasses import dataclass
import math

import torch


@dataclass
class SolverConfig:
    nx: int = 80
    ny: int = 40
    length: float = 4.0
    height: float = 2.0
    cylinder_x: float = 1.0
    cylinder_y: float = 1.0
    cylinder_radius: float | None = 0.2
    inlet_velocity: float = 1.0
    reynolds: float = 20.0
    density: float = 1.0
    velocity_relaxation: float = 0.7
    pressure_relaxation: float = 0.3
    max_iterations: int = 500
    tolerance: float = 1e-5
    device: str = "cpu"
    pseudo_time_step: float | None = None
    mesh_type: str = "cartesian"
    inlet_profile: str = "uniform"
    turbulence_model: str = "laminar"
    turbulence_relaxation: float = 0.5
    sa_freestream_ratio: float = 3.0
    flat_plate_stretching: float = 4.0
    body_fitted_stretching: float = 0.0
    outer_boundary: str = "channel"
    time_step: float | None = None
    inner_iterations: int = 1
    initial_perturbation: float = 0.0
    airfoil_code: str = "0012"
    airfoil_chord: float = 1.0
    airfoil_x: float = 1.0
    airfoil_y: float = 1.0
    angle_of_attack: float = 0.0

    def __post_init__(self):
        for name in ("nx", "ny", "max_iterations", "inner_iterations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.nx < 4 or self.ny < 4 or self.max_iterations < 1:
            raise ValueError("nx and ny must be >= 4; max_iterations must be positive")
        if self.inner_iterations < 1:
            raise ValueError("inner_iterations must be positive")
        if self.mesh_type not in ("cartesian", "body-fitted", "c-grid", "flat-plate"):
            raise ValueError("mesh_type must be cartesian, body-fitted, c-grid, or flat-plate")
        if self.inlet_profile not in ("uniform", "parabolic"):
            raise ValueError("inlet_profile must be uniform or parabolic")
        if self.inlet_profile == "parabolic" and self.mesh_type != "body-fitted":
            raise ValueError("parabolic inlet profile is only supported on the cylinder O-grid")
        if self.turbulence_model not in ("laminar", "spalart-allmaras"):
            raise ValueError("turbulence_model must be laminar or spalart-allmaras")
        if (self.turbulence_model != "laminar"
                and self.mesh_type not in ("body-fitted", "c-grid", "flat-plate")):
            raise ValueError("Spalart-Allmaras requires a body-fitted, c-grid, or flat-plate mesh")
        if self.outer_boundary not in ("channel", "far-field"):
            raise ValueError("outer_boundary must be channel or far-field")
        if self.outer_boundary == "far-field" and self.mesh_type != "body-fitted":
            raise ValueError("far-field outer_boundary is currently supported only by the cylinder O-grid")
        if self.mesh_type == "body-fitted" and (self.nx < 8 or self.nx % 4):
            raise ValueError("body-fitted nx must be >= 8 and divisible by 4")
        if self.mesh_type == "c-grid" and (self.nx < 16 or self.nx % 4):
            raise ValueError("c-grid nx must be >= 16 and divisible by 4")
        for name in ("length", "height", "inlet_velocity", "reynolds",
                     "density", "tolerance"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("velocity_relaxation", "pressure_relaxation", "turbulence_relaxation"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        if not math.isfinite(self.sa_freestream_ratio) or self.sa_freestream_ratio < 0:
            raise ValueError("sa_freestream_ratio must be finite and nonnegative")
        if (not math.isfinite(self.flat_plate_stretching)
                or self.flat_plate_stretching < 0):
            raise ValueError("flat_plate_stretching must be finite and nonnegative")
        if (not math.isfinite(self.body_fitted_stretching)
                or self.body_fitted_stretching < 0):
            raise ValueError("body_fitted_stretching must be finite and nonnegative")
        if self.pseudo_time_step is not None:
            if not math.isfinite(self.pseudo_time_step) or self.pseudo_time_step <= 0:
                raise ValueError("pseudo_time_step must be finite and positive")
        if not math.isfinite(self.initial_perturbation) or self.initial_perturbation < 0:
            raise ValueError("initial_perturbation must be finite and nonnegative")
        if self.time_step is not None:
            if not math.isfinite(self.time_step) or self.time_step <= 0:
                raise ValueError("time_step must be finite and positive")
            if self.pseudo_time_step is not None:
                raise ValueError("time_step and pseudo_time_step cannot be used together")
            if self.mesh_type not in ("body-fitted", "c-grid", "flat-plate"):
                raise ValueError("time_step requires a collocated fitted mesh")
        elif self.initial_perturbation:
            raise ValueError("initial_perturbation requires time_step")
        if not math.isfinite(self.cylinder_x) or not math.isfinite(self.cylinder_y):
            raise ValueError("cylinder coordinates must be finite")
        radius = self.cylinder_radius
        if radius is not None and (not math.isfinite(radius) or radius < 0):
            raise ValueError("cylinder_radius must be finite and nonnegative, or None")
        if self.mesh_type == "body-fitted" and not radius:
            raise ValueError("body-fitted mesh requires a positive cylinder_radius")
        if self.mesh_type == "flat-plate" and radius not in (None, 0):
            raise ValueError("flat-plate mesh does not use cylinder_radius; set it to None or 0")
        if radius and self.mesh_type != "c-grid":
            dx, dy = self.length / self.nx, self.height / self.ny
            if self.mesh_type == "body-fitted":
                dx = dy = 0
            if not (radius + dx < self.cylinder_x < self.length - radius - dx
                    and radius + dy < self.cylinder_y < self.height - radius - dy):
                raise ValueError("cylinder must lie inside the domain with grid clearance")
            represented = self.mesh_type == "body-fitted" or any(
                ((i + 0.5) * dx - self.cylinder_x) ** 2
                + ((j + 0.5) * dy - self.cylinder_y) ** 2 <= radius ** 2
                for j in range(self.ny) for i in range(self.nx)
            )
            if not represented:
                raise ValueError("cylinder is not represented on this grid; refine the mesh")
        if (not isinstance(self.airfoil_code, str)
                or len(self.airfoil_code) != 4
                or not self.airfoil_code.isascii()
                or not self.airfoil_code.isdigit()
                or int(self.airfoil_code[2:]) == 0):
            raise ValueError("airfoil_code must be a four-digit NACA code with nonzero thickness")
        for name in ("airfoil_chord",):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("airfoil_x", "airfoil_y", "angle_of_attack"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if not -180 <= self.angle_of_attack <= 180:
            raise ValueError("angle_of_attack must be between -180 and 180 degrees")
        if (not math.isfinite(self.viscosity) or self.viscosity <= 0
                or self.length / self.nx == 0 or self.height / self.ny == 0):
            raise ValueError("parameters must produce finite positive viscosity and cell sizes")
        try:
            target = torch.device(self.device)
            torch.empty(0, dtype=torch.float64, device=target)
        except (RuntimeError, ValueError, TypeError, AssertionError) as exc:
            raise ValueError(f"unavailable float64 device: {self.device}") from exc

    @property
    def reference_length(self) -> float:
        """Characteristic length used for Reynolds number and force scaling."""
        if self.mesh_type == "c-grid":
            return self.airfoil_chord
        if self.mesh_type == "flat-plate":
            return self.length
        return 2 * self.cylinder_radius if self.cylinder_radius else self.height

    @property
    def viscosity(self) -> float:
        """Dynamic viscosity derived from the configured reference length."""
        return self.density * self.inlet_velocity * self.reference_length / self.reynolds


@dataclass
class SolverResult:
    config: SolverConfig
    u: torch.Tensor
    v: torch.Tensor
    p: torch.Tensor
    fluid: torch.Tensor
    x: torch.Tensor
    y: torch.Tensor
    history: list[dict[str, float | int]]
    converged: bool

    def cell_center_velocity(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return centered velocities, with solid cells set to zero."""
        return ((0.5 * (self.u[:, :-1] + self.u[:, 1:])).masked_fill(~self.fluid, 0),
                (0.5 * (self.v[:-1, :] + self.v[1:, :])).masked_fill(~self.fluid, 0))


def _neighbors(a):
    """East, west, north, south values, zero beyond the array."""
    east, west, north, south = (torch.zeros_like(a) for _ in range(4))
    east[:, :-1], west[:, 1:] = a[:, 1:], a[:, :-1]
    north[:-1, :], south[1:, :] = a[1:, :], a[:-1, :]
    return east, west, north, south


class SimpleSolver:
    """Conservative MAC finite-volume SIMPLE iteration in torch float64.

    Momentum uses implicit first-order upwinding and central diffusion.  Normal
    solid faces are blocked; tangential wall diffusion uses a half-cell distance
    (a zero-velocity ghost), rather than merely masking the normal velocity.
    Pressure correction uses the *under-relaxed momentum diagonal*, including
    the half-width outlet momentum volume.  Outlet corrections are retained.
    Optional pseudo-time inertia damps iterations but is excluded from the
    reported steady momentum residual and convergence criterion.
    """

    def __new__(cls, config: SolverConfig):
        if cls is SimpleSolver and config.mesh_type in ("body-fitted", "c-grid", "flat-plate"):
            from .body_fitted import BodyFittedSolver

            return BodyFittedSolver(config)
        return super().__new__(cls)

    def __init__(self, config: SolverConfig):
        self.config = config
        c = config
        self.dx, self.dy = c.length / c.nx, c.height / c.ny
        opts = {"dtype": torch.float64, "device": c.device}
        self.x = (torch.arange(c.nx, **opts) + 0.5) * self.dx
        self.y = (torch.arange(c.ny, **opts) + 0.5) * self.dy
        self.fluid = torch.ones((c.ny, c.nx), dtype=torch.bool, device=c.device)
        if c.cylinder_radius:
            self.fluid = ((self.x[None, :] - c.cylinder_x) ** 2
                          + (self.y[:, None] - c.cylinder_y) ** 2
                          > c.cylinder_radius ** 2)
        self.u = torch.zeros((c.ny, c.nx + 1), **opts)
        self.v = torch.zeros((c.ny + 1, c.nx), **opts)
        self.p = torch.zeros((c.ny, c.nx), **opts)
        self._u_open = torch.zeros_like(self.u, dtype=torch.bool)
        self._u_open[:, 0] = self.fluid[:, 0]
        self._u_open[:, -1] = self.fluid[:, -1]
        self._u_open[:, 1:-1] = self.fluid[:, :-1] & self.fluid[:, 1:]
        self._v_open = torch.zeros_like(self.v, dtype=torch.bool)
        self._v_open[1:-1] = self.fluid[:-1] & self.fluid[1:]
        self._u_active = self._u_open.clone()
        self._u_active[:, 0] = False
        self.u[self._u_open] = c.inlet_velocity
        self.history = []
        self.converged = False

    def _boundaries(self):
        self.u.masked_fill_(~self._u_open, 0)
        self.u[:, 0] = self.config.inlet_velocity
        self.v.masked_fill_(~self._v_open, 0)
        self.p.masked_fill_(~self.fluid, 0)

    def _momentum(self, component):
        """Return unrelaxed diagonal, neighbor coefficients and pressure force."""
        c, dx, dy = self.config, self.dx, self.dy
        mu, rho = c.viscosity, c.density
        if component == "u":
            field, active, opened = self.u, self._u_active, self._u_open
            e, w, _, _ = _neighbors(field)
            fe, fw = rho * dy * (field + e) / 2, rho * dy * (field + w) / 2
            fe[:, -1] = rho * dy * field[:, -1]
            vn = torch.zeros_like(field)
            vs = torch.zeros_like(field)
            vn[:, 1:-1] = 0.5 * (self.v[1:, :-1] + self.v[1:, 1:])
            vs[:, 1:-1] = 0.5 * (self.v[:-1, :-1] + self.v[:-1, 1:])
            vn[:, -1], vs[:, -1] = self.v[1:, -1], self.v[:-1, -1]
            width = torch.full_like(field, dx)
            width[:, -1] = dx / 2
            fn, fs = rho * width * vn, rho * width * vs
            de = torch.full_like(field, mu * dy / dx)
            dw = de.clone()
            de[:, -1] = 0
            dn, ds = mu * width / dy, mu * width / dy
            pressure = torch.zeros_like(field)
            pressure[:, 1:-1] = dy * (self.p[:, :-1] - self.p[:, 1:])
            pressure[:, -1] = dy * self.p[:, -1]
            tangent = (False, False, True, True)
        else:
            field, active, opened = self.v, self._v_open, self._v_open
            _, _, n, s = _neighbors(field)
            fn, fs = rho * dx * (field + n) / 2, rho * dx * (field + s) / 2
            ue, uw = torch.zeros_like(field), torch.zeros_like(field)
            ue[1:-1] = 0.5 * (self.u[:-1, 1:] + self.u[1:, 1:])
            uw[1:-1] = 0.5 * (self.u[:-1, :-1] + self.u[1:, :-1])
            fe, fw = rho * dy * ue, rho * dy * uw
            de, dw = (torch.full_like(field, mu * dy / dx) for _ in range(2))
            dn, ds = (torch.full_like(field, mu * dx / dy) for _ in range(2))
            width = torch.full_like(field, dy)
            pressure = torch.zeros_like(field)
            pressure[1:-1] = dx * (self.p[:-1] - self.p[1:])
            tangent = (True, True, False, False)
        neighbor_open = _neighbors(opened)
        fluxes, diffusion = [fe, fw, fn, fs], [de, dw, dn, ds]
        # Tangential cylinder/wall boundaries have zero convective transport.
        # v has zero-gradient outflow, but prescribed zero transverse inflow.
        v_outlet = torch.zeros_like(opened)
        if component == "v":
            v_outlet[:, -1] = active[:, -1]
        for k in range(4):
            if tangent[k]:
                wall = ~neighbor_open[k]
                if component == "v" and k == 0:
                    wall = wall & ~v_outlet
                diffusion[k] = diffusion[k] * torch.where(wall, 2.0, 1.0)
                fluxes[k] = fluxes[k].masked_fill(wall, 0)
        if component == "v":
            diffusion[0] = diffusion[0].masked_fill(v_outlet, 0)
        fe, fw, fn, fs = fluxes
        diagonal = sum(diffusion) + fe.clamp_min(0) + (-fw).clamp_min(0) \
            + fn.clamp_min(0) + (-fs).clamp_min(0)
        coeffs = [diffusion[0] + (-fe).clamp_min(0),
                  diffusion[1] + fw.clamp_min(0),
                  diffusion[2] + (-fn).clamp_min(0),
                  diffusion[3] + fs.clamp_min(0)]
        # Dirichlet zero neighbors contribute only to the diagonal.
        coeffs = [a.masked_fill(~mask, 0) for a, mask in zip(coeffs, neighbor_open)]
        if component == "v":
            # The outlet's extrapolated transverse velocity is the local unknown.
            diagonal -= (-fe).clamp_min(0).masked_fill(~v_outlet, 0)
        diagonal = torch.where(active, diagonal, torch.ones_like(diagonal))
        return diagonal, coeffs, pressure, active, width

    @staticmethod
    def _neighbor_sum(field, coeffs):
        return sum(a * value for a, value in zip(coeffs, _neighbors(field)))

    def _predict(self, component):
        field = self.u if component == "u" else self.v
        diagonal, coeffs, pressure, active, width = self._momentum(component)
        old = field.clone()
        if self.config.pseudo_time_step is not None:
            area = width * (self.dy if component == "u" else self.dx)
            inertia = self.config.density * area / self.config.pseudo_time_step
            diagonal = diagonal + inertia
            pressure = pressure + inertia * old
        alpha = self.config.velocity_relaxation
        relaxed = diagonal / alpha
        rhs = pressure + (1 - alpha) / alpha * diagonal * old
        inner_tolerance = min(1e-7, self.config.tolerance * 0.01)
        for _ in range(250):
            update = (rhs + self._neighbor_sum(field, coeffs)) / relaxed
            error = torch.max(torch.abs(update[active] - field[active]))
            field[active] = 0.9 * update[active] + 0.1 * field[active]
            if error.item() / self.config.inlet_velocity < inner_tolerance:
                break
        distance_area = self.dy if component == "u" else self.dx
        return torch.where(active, distance_area / relaxed, 0)

    def _divergence(self):
        return ((self.u[:, 1:] - self.u[:, :-1]) / self.dx
                + (self.v[1:] - self.v[:-1]) / self.dy).masked_fill(~self.fluid, 0)

    def _pressure_correction(self, du, dv):
        """Solve the SPD pressure-correction equation with diagonal-preconditioned CG."""
        rho = self.config.density
        cu, cv = rho * self.dy * du, rho * self.dx * dv
        ce, cw, cn, cs = (torch.zeros_like(self.p) for _ in range(4))
        ce[:, :-1], cw[:, 1:] = cu[:, 1:-1], cu[:, 1:-1]
        ce[:, -1] = cu[:, -1]
        cn[:-1], cs[1:] = cv[1:-1], cv[1:-1]
        diagonal = ce + cw + cn + cs
        diagonal = torch.where(self.fluid, diagonal, torch.ones_like(diagonal))
        coeffs = [ce.clone(), cw, cn, cs]
        coeffs[0][:, -1] = 0  # p'=0 at the outlet, not a neighboring unknown.

        def matvec(q):
            return diagonal * q - self._neighbor_sum(q, coeffs)

        b = -rho * self.dx * self.dy * self._divergence()
        q = torch.zeros_like(b)
        r = b.clone()
        norm = torch.linalg.vector_norm(b).item()
        if norm == 0:
            return q
        z = r / diagonal
        direction, rz = z.clone(), torch.sum(r * z)
        target = max(norm * 1e-10, 1e-14 * rho * self.config.inlet_velocity
                     * min(self.dx, self.dy))
        for _ in range(max(100, self.config.nx * self.config.ny)):
            ad = matvec(direction)
            denominator = torch.sum(direction * ad)
            if denominator.item() <= 0 or not torch.isfinite(denominator):
                raise RuntimeError("pressure-correction CG lost positive definiteness")
            step = rz / denominator
            q += step * direction
            r -= step * ad
            if torch.linalg.vector_norm(r).item() <= target:
                return q
            z = r / diagonal
            new_rz = torch.sum(r * z)
            direction = z + (new_rz / rz) * direction
            rz = new_rz
        raise RuntimeError("pressure-correction CG did not converge")

    def _steady_momentum_residual(self):
        residual = 0.0
        for component, field in (("u", self.u), ("v", self.v)):
            diagonal, coeffs, pressure, active, _ = self._momentum(component)
            defect = (diagonal * field - self._neighbor_sum(field, coeffs) - pressure)
            residual = max(residual, torch.max(torch.abs(defect[active] / diagonal[active])).item())
        return residual / self.config.inlet_velocity

    @torch.no_grad()
    def step(self) -> dict[str, float | int]:
        """Perform one SIMPLE iteration and return dimensionless steady residuals."""
        self._boundaries()
        old_u, old_v = self.u.clone(), self.v.clone()
        du, dv = self._predict("u"), self._predict("v")
        correction = self._pressure_correction(du, dv)
        self.u[:, 1:-1] += du[:, 1:-1] * (correction[:, :-1] - correction[:, 1:])
        self.u[:, -1] += du[:, -1] * correction[:, -1]
        self.v[1:-1] += dv[1:-1] * (correction[:-1] - correction[1:])
        self.p += self.config.pressure_relaxation * correction
        self._boundaries()
        inlet = torch.sum(self.u[:, 0]).item() * self.dy
        outlet = torch.sum(self.u[:, -1]).item() * self.dy
        item = {
            "iteration": len(self.history) + 1,
            "continuity": torch.max(torch.abs(self._divergence())).item()
                          * self.config.height / self.config.inlet_velocity,
            "momentum": self._steady_momentum_residual(),
            "velocity_change": max(torch.max(torch.abs(self.u - old_u)).item(),
                                   torch.max(torch.abs(self.v - old_v)).item())
                               / self.config.inlet_velocity,
            "mass_imbalance": abs(outlet - inlet) / abs(inlet),
        }
        if (not all(math.isfinite(value) for value in item.values())
                or not torch.isfinite(self.p).all().item()):
            raise RuntimeError("non-finite SIMPLE iterate; reduce relaxation or use pseudo_time_step")
        self.history.append(item)
        self.converged = (item["continuity"] < self.config.tolerance
                          and item["momentum"] < self.config.tolerance
                          and item["mass_imbalance"] < self.config.tolerance)
        return item

    def solve(self) -> SolverResult:
        """Iterate to steady residual tolerance or the configured iteration limit."""
        while len(self.history) < self.config.max_iterations and not self.converged:
            self.step()
        return SolverResult(self.config, self.u.clone(), self.v.clone(), self.p.clone(),
                            self.fluid.clone(), self.x.clone(), self.y.clone(),
                            [dict(item) for item in self.history], self.converged)
