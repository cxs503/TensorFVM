"""Numerical wave tank solver for TensorFVM.

Integrates:
- Relaxation-zone wave generation (Stokes-I/II target waves)
- Damping-zone wave absorption
- Two-phase VOF solver with gravity
- Wave probe instrumentation
- 6DOF floating body motion
- Reflection coefficient analysis (Goda two-point method)
- Added resistance computation (Gerritsma-Meijers energy method)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from .two_phase_solver import TwoPhaseSolver, GridConfig, TwoPhaseConfig
from .wave_theory import (
    WaveParams, stokes1_elevation, stokes2_elevation,
    stokes1_velocity, stokes2_velocity,
    relaxation_weight, damping_weight, dispersion_relation,
)
from .wave_probes import WaveProbeArray, goda_reflection_coefficient, invert_wavenumber
from .hull_geometry import WigleyHull, verify_wigley_geometry
from .six_dof import SixDOFConfig, SixDOFIntegrator

__all__ = [
    "WaveTankConfig",
    "WaveTank",
    "compute_added_resistance_gerritsma_meijers",
]


@dataclass
class WaveTankConfig:
    """Configuration for the numerical wave tank."""
    nx: int = 200
    ny: int = 60
    length: float = 10.0
    height: float = 3.0
    water_depth: float = 1.0
    wave_height: float = 0.04
    wave_period: float = 1.2
    wave_order: int = 2
    relaxation_length: float = 1.0
    damping_length: float = 2.0
    damping_gamma: float = 1.0
    gravity: float = 9.81
    rho_liquid: float = 1000.0
    rho_gas: float = 1.0
    mu_liquid: float = 1.0e-3
    mu_gas: float = 1.0e-5
    vof_scheme: str = "upwind"
    device: str = "cpu"
    probe_positions: list[float] | None = None
    hull: WigleyHull | None = None
    six_dof_config: SixDOFConfig | None = None

    def __post_init__(self):
        if self.water_depth <= 0:
            raise ValueError("water_depth must be positive")
        if self.wave_height <= 0:
            raise ValueError("wave_height must be positive")
        if self.wave_period <= 0:
            raise ValueError("wave_period must be positive")
        if self.height <= self.water_depth + self.wave_height:
            raise ValueError("tank height must exceed water depth + wave height")
        if self.damping_length < 0:
            raise ValueError("damping_length must be non-negative")


class WaveTank:
    """Numerical wave tank with relaxation-zone forcing and damping-zone absorption."""

    def __init__(self, config: WaveTankConfig):
        self.config = config
        c = config

        self.dx = c.length / c.nx
        self.dy = c.height / c.ny
        self.wp = WaveParams(
            H=c.wave_height, T=c.wave_period, h=c.water_depth,
            g=c.gravity, order=c.wave_order
        )
        self.wavelength = self.wp.wavelength

        self.x_rel_start = 0.0
        self.x_rel_end = c.relaxation_length
        self.x_damp_start = c.length - c.damping_length
        self.x_damp_end = c.length

        grid_cfg = GridConfig(
            nx=c.nx, ny=c.ny, length=c.length, height=c.height,
            device=c.device, time_step=None, inlet_velocity=0.0,
        )
        tp_cfg = TwoPhaseConfig(
            enabled=True,
            rho_liquid=c.rho_liquid, rho_gas=c.rho_gas,
            mu_liquid=c.mu_liquid, mu_gas=c.mu_gas,
            gravity=c.gravity,
            top_atmosphere=True, bottom_wall=True, side_outlet=False,
            vof_scheme=c.vof_scheme,
            pressure_form="reduced",
            domain_type="closed",
        )
        self.solver = TwoPhaseSolver(grid_cfg, tp_cfg)

        self._init_vof()

        # Cell-centre positions
        self.x_cell = (torch.arange(c.nx, dtype=torch.float64, device=c.device) + 0.5) * self.dx
        self.y_cell = (torch.arange(c.ny, dtype=torch.float64, device=c.device) + 0.5) * self.dy

        # Relaxation and damping weights (cell-centred, shape (nx,))
        self.relax_weight = relaxation_weight(
            self.x_cell, self.x_rel_start, self.x_rel_end
        )
        self.damp_weight = damping_weight(
            self.x_cell, self.x_damp_start, self.x_damp_end, gamma=c.damping_gamma
        )

        if c.probe_positions is None:
            working_start = self.x_rel_end + self.wavelength
            working_end = self.x_damp_start - self.wavelength
            if working_end > working_start:
                spacing = (working_end - working_start) / 4
                c.probe_positions = [
                    working_start + spacing,
                    working_start + 2 * spacing,
                    working_start + 3 * spacing,
                ]
            else:
                c.probe_positions = [c.length * 0.3, c.length * 0.5, c.length * 0.7]

        self.probes = WaveProbeArray(
            x_positions=c.probe_positions,
            h=c.water_depth, dx=self.dx, dy=self.dy, ny=c.ny,
            device=c.device,
        )

        self.six_dof: SixDOFIntegrator | None = None
        if c.six_dof_config is not None:
            self.six_dof = SixDOFIntegrator(
                c.six_dof_config, rho=c.rho_liquid, g=c.gravity
            )

        self.hull_mask: torch.Tensor | None = None
        if c.hull is not None:
            self._init_hull_mask(c.hull)

        self.time = 0.0
        self.step_count = 0

    def _init_vof(self):
        c = self.config
        for j in range(c.ny):
            z_cell = (j + 0.5) * self.dy - c.water_depth
            if z_cell < -self.dy / 2:
                self.solver.alpha[j, :] = 1.0
            elif z_cell > self.dy / 2:
                self.solver.alpha[j, :] = 0.0
            else:
                frac = (c.water_depth - j * self.dy) / self.dy
                self.solver.alpha[j, :] = max(0.0, min(1.0, frac))
        self.solver._update_properties()

    def _init_hull_mask(self, hull: WigleyHull):
        c = self.config
        x_hull_center = c.length / 2.0
        z_swl = c.water_depth

        x = (torch.arange(c.nx, dtype=torch.float64, device=c.device) + 0.5) * self.dx - x_hull_center
        z = (torch.arange(c.ny, dtype=torch.float64, device=c.device) + 0.5) * self.dy - z_swl

        X, Z = torch.meshgrid(x, z, indexing='ij')
        half_y_wl = hull.halfbreadth_vec(X.T, torch.zeros_like(Z.T))
        self.hull_mask = (torch.abs(X.T) < half_y_wl) & (Z.T <= 0)
        self.solver.fluid = ~self.hull_mask

    def _target_elevation(self, t: float) -> torch.Tensor:
        if self.wp.order == 1:
            return stokes1_elevation(self.x_cell, t, self.wp)
        else:
            return stokes2_elevation(self.x_cell, t, self.wp)

    def _apply_relaxation(self, t: float):
        """Apply relaxation-zone wave generation."""
        c = self.config
        R = self.relax_weight  # (nx,)

        # Target elevation → target alpha profile
        eta_target = self._target_elevation(t)  # (nx,)
        alpha_target = torch.zeros((c.ny, c.nx), dtype=torch.float64, device=c.device)
        for j in range(c.ny):
            z_cell = (j + 0.5) * self.dy - c.water_depth
            alpha_target[j, :] = (z_cell < eta_target).float()

        # Apply relaxation to alpha
        R_2d = R[None, :].expand(c.ny, c.nx)
        self.solver.alpha = (1.0 - R_2d) * self.solver.alpha + R_2d * alpha_target

        # For velocity, we apply relaxation only to the u component at the inlet boundary
        # and to cells in the relaxation zone. We use cell-centred relaxation weight
        # and interpolate to faces.
        # u faces: (ny, nx+1). Interior faces i=1..nx-1 use average of adjacent cells.
        # Boundary face i=0 uses R[0]=0 (no relaxation at inlet).
        # Boundary face i=nx uses R[-1].
        R_u = torch.zeros((c.ny, c.nx + 1), dtype=torch.float64, device=c.device)
        # Interior faces: average of adjacent cell weights
        R_u[:, 1:-1] = 0.5 * (R[None, :-1] + R[None, 1:])
        # Left boundary: R[0] = 0
        R_u[:, 0] = 0.0
        # Right boundary: R[-1]
        R_u[:, -1] = R[-1]

        # Target velocity at cell centres
        z_cell = self.y_cell - c.water_depth  # (ny,)
        X = self.x_cell[None, :].expand(c.ny, c.nx)
        Z = z_cell[:, None].expand(c.ny, c.nx)

        if self.wp.order == 1:
            u_target, w_target = stokes1_velocity(X, Z, t, self.wp)
        else:
            u_target, w_target = stokes2_velocity(X, Z, t, self.wp)

        # Interpolate u_target to faces (average of adjacent cells)
        u_target_face = torch.zeros((c.ny, c.nx + 1), dtype=torch.float64, device=c.device)
        u_target_face[:, 1:-1] = 0.5 * (u_target[:, :-1] + u_target[:, 1:])
        u_target_face[:, 0] = u_target[:, 0]  # left boundary
        u_target_face[:, -1] = u_target[:, -1]  # right boundary

        # Interpolate w_target to v faces
        w_target_face = torch.zeros((c.ny + 1, c.nx), dtype=torch.float64, device=c.device)
        w_target_face[1:-1, :] = 0.5 * (w_target[:-1, :] + w_target[1:, :])
        w_target_face[0, :] = w_target[0, :]
        w_target_face[-1, :] = w_target[-1, :]

        # Apply relaxation to velocity
        self.solver.u = (1.0 - R_u) * self.solver.u + R_u * u_target_face

        # For v, use the same R (relaxation is primarily in x-direction)
        R_v = R_u.T[:c.nx, :].T  # This doesn't work, let's use cell-centred R
        R_v = R[None, :].expand(c.ny + 1, c.nx)
        self.solver.v = (1.0 - R_v) * self.solver.v + R_v * w_target_face

    def _apply_damping(self):
        """Apply damping-zone momentum sink."""
        c = self.config
        D = self.damp_weight  # (nx,)

        # Damping on u faces
        D_u = torch.zeros((c.ny, c.nx + 1), dtype=torch.float64, device=c.device)
        D_u[:, 1:-1] = 0.5 * (D[None, :-1] + D[None, 1:])
        D_u[:, 0] = 0.0
        D_u[:, -1] = D[-1]

        dt = self.solver._current_dt if self.solver._current_dt else 0.01
        self.solver.u = self.solver.u * torch.exp(-D_u * dt)

        # Damping on v faces
        D_v = D[None, :].expand(c.ny + 1, c.nx)
        self.solver.v = self.solver.v * torch.exp(-D_v * dt)

    def step(self, dt: float):
        """Advance the wave tank by one time step."""
        self.solver._current_dt = dt

        self._apply_relaxation(self.time)
        self._apply_damping()

        self.solver.step(dt)

        self.time += dt
        self.step_count += 1

        self.probes.measure_all(self.solver.alpha, self.time)

    def probe(self) -> list[float]:
        return [p.history[-1] if p.history else 0.0 for p in self.probes.probes]

    def reflection_coefficient(self, probe1: int = 0, probe2: int = 1) -> float:
        t1, eta1 = self.probes.get_history(probe1)
        t2, eta2 = self.probes.get_history(probe2)

        if len(eta1) < 4 or len(eta2) < 4:
            return 0.0

        eta1_t = torch.tensor(eta1, dtype=torch.float64, device=self.config.device)
        eta2_t = torch.tensor(eta2, dtype=torch.float64, device=self.config.device)

        dx_probe = abs(self.config.probe_positions[probe2] -
                       self.config.probe_positions[probe1])

        return goda_reflection_coefficient(
            eta1_t, eta2_t, dx_probe, self.wp.omega,
            self.config.water_depth, self.config.gravity
        )

    def wave_number_error(self) -> tuple[float, float]:
        if len(self.config.probe_positions) < 3:
            return 0.0, self.wp.k

        t1, eta1 = self.probes.get_history(0)
        t2, eta2 = self.probes.get_history(1)
        t3, eta3 = self.probes.get_history(2)

        if len(eta1) < 4:
            return 0.0, self.wp.k

        eta1_t = torch.tensor(eta1, dtype=torch.float64, device=self.config.device)
        eta2_t = torch.tensor(eta2, dtype=torch.float64, device=self.config.device)
        eta3_t = torch.tensor(eta3, dtype=torch.float64, device=self.config.device)

        dx12 = abs(self.config.probe_positions[1] - self.config.probe_positions[0])
        dx23 = abs(self.config.probe_positions[2] - self.config.probe_positions[1])

        return invert_wavenumber(
            eta1_t, eta2_t, eta3_t, dx12, dx23,
            self.wp.omega, self.config.water_depth, self.config.gravity
        )


def compute_added_resistance_gerritsma_meijers(
    wave_amplitude: float,
    omega: float,
    k: float,
    h: float,
    rho: float,
    g: float,
    B: float,
    L: float,
    Cb: float,
    speed: float = 0.0,
) -> float:
    """Gerritsma-Meijers energy method for added resistance in waves."""
    E = 0.5 * rho * g * wave_amplitude ** 2

    kh = k * h
    tanh_kh = math.tanh(kh)
    cg = 0.5 * (omega / k) * (1 + 2 * kh / math.sinh(2 * kh))

    P = E * cg * B * Cb

    c = omega / k
    delta_R = P / (c + 1e-12)

    if speed > 0:
        Fn = speed / math.sqrt(g * L)
        delta_R *= (1 + 2 * Fn)

    return delta_R
