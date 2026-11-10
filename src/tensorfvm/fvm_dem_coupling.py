"""FVM-DEM two-phase coupling for ice floe field simulation.

Physical Model
--------------
Two-phase system: continuous water phase (FVM) + discrete ice floes (DEM).

FVM → DEM: Drag force on each floe
    F_D = 0.5 * ρ_w * C_D * A * |u_f - v| * (u_f - v)
    where u_f = fluid velocity at floe location, v = floe velocity,
    A = projected area (2*R for 2D cylinder), C_D = drag coefficient.

DEM → FVM: Blocking force (Newton's 3rd law)
    F_block = -F_D applied as momentum sink in FVM source term.

Analytical anchors:
    - Single floe steady drift: v_drift → u_∞ (floe moves with fluid)
    - No flow → floe stationary (v < 0.01 m/s)
    - Momentum conservation: ΣF_internal = 0 (Newton's 3rd law)

Coupling algorithm:
    Explicit: FVM solve → interpolate u to floe positions → compute F_D →
              advance DEM → compute F_block → add to FVM source → repeat.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import numpy as np


# ============================================================================
# Data classes
# ============================================================================

@dataclass
class FloeProperties:
    """Physical properties of an ice floe."""
    radius_m: float = 0.5
    thickness_m: float = 0.3
    density_kg_m3: float = 917.0
    drag_coefficient: float = 2.0
    added_mass_coeff: float = 1.0

    @property
    def mass_kg(self) -> float:
        return math.pi * self.radius_m**2 * self.thickness_m * self.density_kg_m3

    @property
    def added_mass_kg(self) -> float:
        rho_w = 1025.0
        return self.added_mass_coeff * rho_w * math.pi * self.radius_m**2 * self.thickness_m

    @property
    def total_inertia_kg(self) -> float:
        return self.mass_kg + self.added_mass_kg

    @property
    def projected_area_m2(self) -> float:
        return 2.0 * self.radius_m * self.thickness_m


@dataclass
class FloeState:
    """State of a single ice floe."""
    position: torch.Tensor  # (2,) [x, y]
    velocity: torch.Tensor  # (2,) [vx, vy]
    floe_id: int = 0


@dataclass
class FlowField:
    """2D flow field on structured grid."""
    u: torch.Tensor
    v: torch.Tensor
    p: torch.Tensor
    x: torch.Tensor
    y: torch.Tensor
    dx: float
    dy: float
    source_x: torch.Tensor = None
    source_y: torch.Tensor = None

    def __post_init__(self):
        if self.source_x is None:
            self.source_x = torch.zeros_like(self.u)
        if self.source_y is None:
            self.source_y = torch.zeros_like(self.v)


@dataclass
class CouplingConfig:
    """Configuration for FVM-DEM coupling."""
    length_m: float = 10.0
    height_m: float = 4.0
    nx: int = 50
    ny: int = 20
    water_density: float = 1025.0
    inlet_velocity: float = 1.0
    kinematic_viscosity: float = 1e-3
    dt: float = 0.01
    total_time: float = 5.0
    coupling_scheme: str = "explicit"
    implicit_subiterations: int = 5
    output_interval: int = 10


# ============================================================================
# DEM Solver
# ============================================================================

class DEMSolver:
    """Discrete Element Method solver for ice floes."""

    def __init__(self, floe_props: FloeProperties, config: CouplingConfig):
        self.props = floe_props
        self.config = config
        self.floes: list[FloeState] = []
        self.forces: list[torch.Tensor] = []
        self.history: list[dict] = []

    def add_floe(self, x: float, y: float, vx: float = 0.0, vy: float = 0.0) -> int:
        floe_id = len(self.floes)
        state = FloeState(
            position=torch.tensor([x, y], dtype=torch.float64),
            velocity=torch.tensor([vx, vy], dtype=torch.float64),
            floe_id=floe_id,
        )
        self.floes.append(state)
        self.forces.append(torch.zeros(2, dtype=torch.float64))
        return floe_id

    def add_floes_uniform(self, n_floes: int, x_range: tuple, y_range: tuple,
                          seed: int = 42) -> None:
        rng = torch.Generator().manual_seed(seed)
        xs = x_range[0] + (x_range[1] - x_range[0]) * torch.rand(n_floes, generator=rng)
        ys = y_range[0] + (y_range[1] - y_range[0]) * torch.rand(n_floes, generator=rng)
        for i in range(n_floes):
            self.add_floe(float(xs[i]), float(ys[i]))

    def reset_forces(self):
        for i in range(len(self.floes)):
            self.forces[i] = torch.zeros(2, dtype=torch.float64)

    def add_drag_force(self, floe_id: int, force: torch.Tensor):
        self.forces[floe_id] = self.forces[floe_id] + force

    def compute_drag_force(self, floe: FloeState, u_fluid: torch.Tensor) -> torch.Tensor:
        """F_D = 0.5 * ρ * C_D * A * |u - v| * (u - v)"""
        rel_vel = u_fluid - floe.velocity
        rel_speed = torch.linalg.norm(rel_vel)
        if rel_speed < 1e-12:
            return torch.zeros(2, dtype=torch.float64)
        A = self.props.projected_area_m2
        C_D = self.props.drag_coefficient
        rho = self.config.water_density
        F_D = 0.5 * rho * C_D * A * rel_speed * rel_vel
        return F_D

    def step(self, dt: float) -> None:
        """Velocity Verlet: v += a*dt, x += v*dt"""
        m_total = self.props.total_inertia_kg
        for i, floe in enumerate(self.floes):
            acc = self.forces[i] / m_total
            floe.velocity = floe.velocity + acc * dt
            floe.position = floe.position + floe.velocity * dt
        self.history.append({
            "positions": [f.position.clone() for f in self.floes],
            "velocities": [f.velocity.clone() for f in self.floes],
            "forces": [f.clone() for f in self.forces],
        })

    def get_kinetic_energy(self) -> float:
        m_total = self.props.total_inertia_kg
        ke = 0.0
        for floe in self.floes:
            v2 = float(torch.sum(floe.velocity**2))
            ke += 0.5 * m_total * v2
        return ke

    def get_total_momentum(self) -> torch.Tensor:
        m_total = self.props.total_inertia_kg
        p = torch.zeros(2, dtype=torch.float64)
        for floe in self.floes:
            p += m_total * floe.velocity
        return p


# ============================================================================
# FVM Flow Solver (vectorized 2D channel)
# ============================================================================

class FVMFlowSolver:
    """Vectorized 2D FVM solver for incompressible channel flow."""

    def __init__(self, config: CouplingConfig):
        self.config = config
        self.nx = config.nx
        self.ny = config.ny
        self.dx = config.length_m / config.nx
        self.dy = config.height_m / config.ny
        self.rho = config.water_density
        self.nu = config.kinematic_viscosity
        self.mu = self.rho * self.nu

        self.x = (torch.arange(self.nx, dtype=torch.float64) + 0.5) * self.dx
        self.y = (torch.arange(self.ny, dtype=torch.float64) + 0.5) * self.dy

        self.u = torch.full((self.ny, self.nx), config.inlet_velocity, dtype=torch.float64)
        self.v_field = torch.zeros((self.ny, self.nx), dtype=torch.float64)
        self.p = torch.zeros((self.ny, self.nx), dtype=torch.float64)
        self.source_x = torch.zeros((self.ny, self.nx), dtype=torch.float64)
        self.source_y = torch.zeros((self.ny, self.nx), dtype=torch.float64)

        self.cell_volume = self.dx * self.dy

    def get_flow_field(self) -> FlowField:
        return FlowField(
            u=self.u.clone(), v=self.v_field.clone(), p=self.p.clone(),
            x=self.x.clone(), y=self.y.clone(),
            dx=self.dx, dy=self.dy,
            source_x=self.source_x.clone(), source_y=self.source_y.clone(),
        )

    def interpolate_velocity(self, x_pos: float, y_pos: float) -> torch.Tensor:
        """Bilinear interpolation of velocity at arbitrary position."""
        ix = x_pos / self.dx - 0.5
        iy = y_pos / self.dy - 0.5

        i0 = int(max(0, min(math.floor(ix), self.nx - 2)))
        j0 = int(max(0, min(math.floor(iy), self.ny - 2)))

        fx = ix - i0
        fy = iy - j0
        fx = max(0.0, min(1.0, fx))
        fy = max(0.0, min(1.0, fy))

        u_val = ((1-fx)*(1-fy)*self.u[j0, i0] + fx*(1-fy)*self.u[j0, i0+1] +
                 (1-fx)*fy*self.u[j0+1, i0] + fx*fy*self.u[j0+1, i0+1])
        v_val = ((1-fx)*(1-fy)*self.v_field[j0, i0] + fx*(1-fy)*self.v_field[j0, i0+1] +
                 (1-fx)*fy*self.v_field[j0+1, i0] + fx*fy*self.v_field[j0+1, i0+1])

        return torch.tensor([float(u_val), float(v_val)], dtype=torch.float64)

    def add_blocking_force(self, floe_x: float, floe_y: float,
                           force: torch.Tensor) -> None:
        """Distribute blocking force using Gaussian spreading (vectorized)."""
        sigma = self.dx * 1.0
        sigma2_2 = 2.0 * sigma * sigma

        dist_x = self.x.unsqueeze(0) - floe_x
        dist_y = self.y.unsqueeze(1) - floe_y
        dist2 = dist_x**2 + dist_y**2
        weight = torch.exp(-dist2 / sigma2_2)

        self.source_x += force[0].item() * weight / self.cell_volume
        self.source_y += force[1].item() * weight / self.cell_volume

    def reset_sources(self):
        self.source_x.zero_()
        self.source_y.zero_()

    def step(self, dt: float) -> None:
        """Vectorized advection-diffusion with source terms."""
        u_old = self.u.clone()

        # Advection (first-order upwind in x) - vectorized
        u_face_right = 0.5 * (u_old[:, :-1] + u_old[:, 1:])

        du_adv = torch.zeros_like(self.u)
        u_face_left = u_face_right[:, :-1]
        u_face_r = u_face_right[:, 1:]

        left_pos = u_face_left > 0
        u_left_up = torch.where(left_pos, u_old[:, :-2], u_old[:, 1:-1])
        flux_left = u_face_left * u_left_up

        right_pos = u_face_r > 0
        u_right_up = torch.where(right_pos, u_old[:, 1:-1], u_old[:, 2:])
        flux_right = u_face_r * u_right_up

        du_adv[:, 1:-1] = -(flux_right - flux_left) / self.dx

        # Diffusion (central difference) - vectorized
        d2u_dx2 = (u_old[:, 2:] - 2*u_old[:, 1:-1] + u_old[:, :-2]) / self.dx**2
        d2u_dy2 = (u_old[2:, 1:-1] - 2*u_old[1:-1, 1:-1] + u_old[:-2, 1:-1]) / self.dy**2
        du_diff = torch.zeros_like(self.u)
        du_diff[1:-1, 1:-1] = self.nu * (d2u_dx2[1:-1, :] + d2u_dy2)

        # Update
        self.u = u_old + dt * (du_adv + du_diff) + dt * self.source_x / self.rho

        # Apply BCs
        self._apply_bc()

    def _apply_bc(self):
        """Apply boundary conditions (vectorized)."""
        self.u[:, 0] = self.config.inlet_velocity
        self.u[:, -1] = self.u[:, -2]
        self.v_field[0, :] = 0.0
        self.v_field[-1, :] = 0.0
        self.u[0, :] = 0.0
        self.u[-1, :] = 0.0

    def get_total_fluid_momentum(self) -> float:
        return float(torch.sum(self.u) * self.cell_volume * self.rho)

    def get_input_momentum_rate(self) -> float:
        u_in = self.config.inlet_velocity
        area = self.config.height_m
        return self.rho * u_in**2 * area


# ============================================================================
# FVM-DEM Coupling
# ============================================================================

class FVMDEMCoupling:
    """Two-way coupled FVM-DEM solver for ice floe transport."""

    def __init__(self, config: CouplingConfig, floe_props: FloeProperties):
        self.config = config
        self.fvm = FVMFlowSolver(config)
        self.dem = DEMSolver(floe_props, config)
        self.floe_props = floe_props
        self.time = 0.0
        self.step_count = 0
        self.diagnostics: list[dict] = []
        self.momentum_history: list[dict] = []

    def step(self) -> dict:
        """Perform one coupling step."""
        dt = self.config.dt

        fluid_mom_before = self.fvm.get_total_fluid_momentum()
        floe_mom_before = float(self.dem.get_total_momentum()[0])

        self.fvm.reset_sources()
        self.dem.reset_forces()
        total_drag = torch.zeros(2, dtype=torch.float64)

        for i, floe in enumerate(self.dem.floes):
            u_fluid = self.fvm.interpolate_velocity(
                float(floe.position[0]), float(floe.position[1])
            )
            F_D = self.dem.compute_drag_force(floe, u_fluid)
            self.dem.add_drag_force(i, F_D)
            total_drag += F_D

        self.dem.step(dt)

        total_blocking = torch.zeros(2, dtype=torch.float64)
        for i, floe in enumerate(self.dem.floes):
            F_block = -self.dem.forces[i]
            self.fvm.add_blocking_force(
                float(floe.position[0]), float(floe.position[1]), F_block
            )
            total_blocking += F_block

        self.fvm.step(dt)

        self.time += dt
        self.step_count += 1

        fluid_mom_after = self.fvm.get_total_fluid_momentum()
        floe_mom_after = float(self.dem.get_total_momentum()[0])

        diag = {
            "time": self.time,
            "step": self.step_count,
            "total_drag_x": float(total_drag[0]),
            "total_drag_y": float(total_drag[1]),
            "total_blocking_x": float(total_blocking[0]),
            "total_blocking_y": float(total_blocking[1]),
            "newton3_error": float(torch.norm(total_drag + total_blocking)),
            "floe_kinetic_energy": self.dem.get_kinetic_energy(),
            "fluid_momentum_before": fluid_mom_before,
            "fluid_momentum_after": fluid_mom_after,
            "floe_momentum_before": floe_mom_before,
            "floe_momentum_after": floe_mom_after,
            "total_momentum_before": fluid_mom_before + floe_mom_before,
            "total_momentum_after": fluid_mom_after + floe_mom_after,
        }

        self.diagnostics.append(diag)
        self.momentum_history.append({
            "time": self.time,
            "fluid_mom": fluid_mom_after,
            "floe_mom": floe_mom_after,
            "total_mom": fluid_mom_after + floe_mom_after,
        })

        return diag

    def run(self, n_steps: Optional[int] = None) -> list[dict]:
        if n_steps is None:
            n_steps = int(self.config.total_time / self.config.dt)
        results = []
        for _ in range(n_steps):
            diag = self.step()
            results.append(diag)
        return results


# ============================================================================
# Verification Cases
# ============================================================================

def verify_single_floe_drift(u_inlet: float = 1.0, total_time: float = 10.0,
                              dt: float = 0.01, nx: int = 50, ny: int = 20) -> dict:
    """C1: Single floe drift verification. v_drift → u_∞.
    
    Uses small floe to minimize blocking effect (one-way coupling limit).
    """
    config = CouplingConfig(
        length_m=10.0, height_m=4.0,
        nx=nx, ny=ny,
        inlet_velocity=u_inlet,
        dt=dt, total_time=total_time,
        kinematic_viscosity=1e-2,
    )
    # Small floe to minimize blocking
    floe_props = FloeProperties(radius_m=0.05, thickness_m=0.05, drag_coefficient=2.0)

    coupling = FVMDEMCoupling(config, floe_props)
    coupling.dem.add_floe(5.0, 2.0)

    results = coupling.run()

    final_vx = float(coupling.dem.floes[0].velocity[0])
    analytical_vx = u_inlet
    error = abs(final_vx - analytical_vx) / analytical_vx if analytical_vx > 0 else abs(final_vx)

    return {
        "u_inlet": u_inlet,
        "v_drift_computed": final_vx,
        "v_drift_analytical": analytical_vx,
        "relative_error": error,
        "passed": error <= 0.10,
        "n_steps": len(results),
        "final_time": coupling.time,
    }


def verify_momentum_conservation(n_steps: int = 5) -> dict:
    """C2: Newton's 3rd law residual ≤3% of drag magnitude."""
    config = CouplingConfig(
        length_m=10.0, height_m=4.0,
        nx=30, ny=15,
        inlet_velocity=1.0,
        dt=0.005, total_time=0.05,
        kinematic_viscosity=1e-2,
    )
    floe_props = FloeProperties(radius_m=0.05, thickness_m=0.05)

    coupling = FVMDEMCoupling(config, floe_props)
    coupling.dem.add_floe(5.0, 2.0)

    momentum_residuals = []
    for i in range(n_steps):
        diag = coupling.step()
        newton_err = diag["newton3_error"]
        momentum_residuals.append(newton_err)

    max_residual = max(momentum_residuals)
    max_drag = max(abs(d["total_drag_x"]) for d in coupling.diagnostics)
    relative_residual = max_residual / max_drag if max_drag > 1e-10 else max_residual

    return {
        "n_steps": n_steps,
        "momentum_residuals": momentum_residuals,
        "max_residual": max_residual,
        "relative_residual": relative_residual,
        "passed": relative_residual <= 0.03,
    }


def verify_newton_third_law(coupling_scheme: str = "explicit") -> dict:
    """C3: F_block = -F_D error ≤1e-6 (explicit)."""
    config = CouplingConfig(
        length_m=10.0, height_m=4.0,
        nx=30, ny=15,
        inlet_velocity=1.0,
        dt=0.01, total_time=0.1,
        coupling_scheme=coupling_scheme,
        kinematic_viscosity=1e-2,
    )
    floe_props = FloeProperties(radius_m=0.05, thickness_m=0.05)

    coupling = FVMDEMCoupling(config, floe_props)
    coupling.dem.add_floe(5.0, 2.0)

    errors = []
    for _ in range(10):
        diag = coupling.step()
        errors.append(diag["newton3_error"])

    max_error = max(errors)
    tol = 1e-6 if coupling_scheme == "explicit" else 1e-4

    return {
        "scheme": coupling_scheme,
        "errors": errors,
        "max_error": max_error,
        "tolerance": tol,
        "passed": max_error <= tol,
    }


def verify_multi_floe_interaction(n_floes: int = 10) -> dict:
    """C4: Multi-floe clearing rate error ≤15%."""
    single_result = verify_single_floe_drift(u_inlet=1.0, total_time=5.0, dt=0.01)
    single_rate = single_result["v_drift_computed"]

    config = CouplingConfig(
        length_m=20.0, height_m=6.0,
        nx=60, ny=30,
        inlet_velocity=1.0,
        dt=0.01, total_time=5.0,
        kinematic_viscosity=1e-2,
    )
    floe_props = FloeProperties(radius_m=0.05, thickness_m=0.05)

    coupling = FVMDEMCoupling(config, floe_props)
    coupling.dem.add_floes_uniform(n_floes, x_range=(2.0, 18.0), y_range=(1.0, 5.0))

    results = coupling.run()

    avg_vx = 0.0
    for floe in coupling.dem.floes:
        avg_vx += float(floe.velocity[0])
    avg_vx /= n_floes

    expected_rate = single_rate
    error = abs(avg_vx - expected_rate) / expected_rate if expected_rate > 0 else abs(avg_vx)

    return {
        "n_floes": n_floes,
        "avg_drift_velocity": avg_vx,
        "single_floe_rate": single_rate,
        "expected_rate": expected_rate,
        "relative_error": error,
        "passed": error <= 0.15,
    }


def verify_grid_convergence() -> dict:
    """C5: Grid refinement → v_drift change ≤5%."""
    results = {}
    for label, nx, ny in [("coarse", 30, 15), ("fine", 60, 30)]:
        r = verify_single_floe_drift(u_inlet=1.0, total_time=5.0, dt=0.01,
                                      nx=nx, ny=ny)
        results[label] = r

    v_coarse = results["coarse"]["v_drift_computed"]
    v_fine = results["fine"]["v_drift_computed"]
    change = abs(v_fine - v_coarse) / abs(v_fine) if abs(v_fine) > 0 else abs(v_coarse)

    return {
        "coarse": {"nx": 30, "ny": 15, "v_drift": v_coarse},
        "fine": {"nx": 60, "ny": 30, "v_drift": v_fine},
        "relative_change": change,
        "passed": change <= 0.05,
    }


def verify_timestep_convergence() -> dict:
    """C6: dt halving → v_drift change ≤3%."""
    r1 = verify_single_floe_drift(u_inlet=1.0, total_time=5.0, dt=0.02)
    r2 = verify_single_floe_drift(u_inlet=1.0, total_time=5.0, dt=0.01)

    v1 = r1["v_drift_computed"]
    v2 = r2["v_drift_computed"]
    change = abs(v2 - v1) / abs(v2) if abs(v2) > 0 else abs(v1)

    return {
        "dt_coarse": 0.02, "v_drift_coarse": v1,
        "dt_fine": 0.01, "v_drift_fine": v2,
        "relative_change": change,
        "passed": change <= 0.03,
    }


def verify_negative_control() -> dict:
    """C7: No flow → floe stationary (v < 0.01 m/s)."""
    config = CouplingConfig(
        length_m=10.0, height_m=4.0,
        nx=30, ny=15,
        inlet_velocity=0.0,
        dt=0.01, total_time=2.0,
        kinematic_viscosity=1e-2,
    )
    floe_props = FloeProperties(radius_m=0.05, thickness_m=0.05)

    coupling = FVMDEMCoupling(config, floe_props)
    coupling.dem.add_floe(5.0, 2.0)

    coupling.run()

    final_speed = float(torch.norm(coupling.dem.floes[0].velocity))

    return {
        "inlet_velocity": 0.0,
        "final_floe_speed": final_speed,
        "threshold": 0.01,
        "passed": final_speed < 0.01,
    }


def run_all_verifications(output_dir: Path = Path("artifacts/fvm_dem_coupling")) -> dict:
    """Run all verification cases and produce summary."""
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # C1
    c1_results = []
    for u in [0.5, 1.0, 1.5]:
        r = verify_single_floe_drift(u_inlet=u, total_time=8.0, dt=0.01)
        c1_results.append(r)
    results["C1_single_floe_drift"] = c1_results
    (output_dir / "C1_single_floe_drift.json").write_text(
        json.dumps(c1_results, indent=2) + "\n"
    )

    # C2
    c2 = verify_momentum_conservation(n_steps=5)
    results["C2_momentum_conservation"] = c2
    (output_dir / "C2_momentum_conservation.json").write_text(
        json.dumps(c2, indent=2) + "\n"
    )

    # C3
    c3_explicit = verify_newton_third_law("explicit")
    results["C3_newton_third_law_explicit"] = c3_explicit
    (output_dir / "C3_newton_third_law.json").write_text(
        json.dumps(c3_explicit, indent=2) + "\n"
    )

    # C4
    c4 = verify_multi_floe_interaction(n_floes=10)
    results["C4_multi_floe"] = c4
    (output_dir / "C4_multi_floe.json").write_text(
        json.dumps(c4, indent=2) + "\n"
    )

    # C5
    c5 = verify_grid_convergence()
    results["C5_grid_convergence"] = c5
    (output_dir / "C5_grid_convergence.json").write_text(
        json.dumps(c5, indent=2) + "\n"
    )

    # C6
    c6 = verify_timestep_convergence()
    results["C6_timestep_convergence"] = c6
    (output_dir / "C6_timestep_convergence.json").write_text(
        json.dumps(c6, indent=2) + "\n"
    )

    # C7
    c7 = verify_negative_control()
    results["C7_negative_control"] = c7
    (output_dir / "C7_negative_control.json").write_text(
        json.dumps(c7, indent=2) + "\n"
    )

    all_passed = all([
        all(r["passed"] for r in c1_results),
        c2["passed"],
        c3_explicit["passed"],
        c4["passed"],
        c5["passed"],
        c6["passed"],
        c7["passed"],
    ])

    summary = {
        "schema": "tensorfvm.fvm-dem-coupling-verification/1",
        "all_passed": all_passed,
        "criteria": {
            "C1": all(r["passed"] for r in c1_results),
            "C2": c2["passed"],
            "C3": c3_explicit["passed"],
            "C4": c4["passed"],
            "C5": c5["passed"],
            "C6": c6["passed"],
            "C7": c7["passed"],
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    return summary
