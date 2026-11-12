"""Six-degree-of-freedom (6DOF) rigid-body motion for floating bodies.

Implements the equations of motion for a floating body in waves,
with configurable constraints (e.g., surge fixed, heave+pitch free).
Uses a symplectic Euler integrator for time stepping.

Coordinate system
-----------------
- x: longitudinal (positive forward, surge)
- y: transverse (positive to port, sway)
- z: vertical (positive upward, heave)
- Roll about x, pitch about y, yaw about z

For head-sea (beam sea) problems the typical constraint is:
- Surge: fixed (or with prescribed speed)
- Sway, roll, yaw: constrained (symmetry)
- Heave, pitch: free

Equations of motion
-------------------
    M · ẍ = F_hydrostatic + F_hydrodynamic + F_mooring

where M is the mass matrix (including added mass if specified).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

__all__ = [
    "SixDOFState",
    "SixDOFConfig",
    "SixDOFIntegrator",
    "hydrostatic_restoring_heave",
    "hydrostatic_restoring_pitch",
]


@dataclass
class SixDOFState:
    """6DOF rigid-body state.

    Attributes
    ----------
    x, y, z : float
        Position (m).
    roll, pitch, yaw : float
        Orientation (rad).
    u, v, w : float
        Linear velocity (m/s).
    p, q, r : float
        Angular velocity (rad/s).
    """
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    u: float = 0.0
    v: float = 0.0
    w: float = 0.0
    p: float = 0.0
    q: float = 0.0
    r: float = 0.0

    def to_tensor(self, device: str = "cpu") -> torch.Tensor:
        """Return state as 12-element tensor."""
        return torch.tensor([
            self.x, self.y, self.z,
            self.roll, self.pitch, self.yaw,
            self.u, self.v, self.w,
            self.p, self.q, self.r
        ], dtype=torch.float64, device=device)

    @classmethod
    def from_tensor(cls, t: torch.Tensor) -> "SixDOFState":
        """Create state from 12-element tensor."""
        return cls(
            x=float(t[0]), y=float(t[1]), z=float(t[2]),
            roll=float(t[3]), pitch=float(t[4]), yaw=float(t[5]),
            u=float(t[6]), v=float(t[7]), w=float(t[8]),
            p=float(t[9]), q=float(t[10]), r=float(t[11])
        )


@dataclass
class SixDOFConfig:
    """6DOF configuration for a floating body.

    Parameters
    ----------
    mass : float
        Body mass (kg).
    Ixx, Iyy, Izz : float
        Moments of inertia (kg·m²).
    x_cg, y_cg, z_cg : float
        Centre of gravity (m, in body frame).
    x_cb, y_cb, z_cb : float
        Centre of buoyancy at equilibrium (m, in body frame).
    waterplane_area : float
        Waterplane area at equilibrium (m²).
    waterplane_pitch_moi : float
        Waterplane second moment about transverse axis (m⁴).
    free_dof : list[str]
        List of free degrees of freedom. Valid: 'surge', 'sway', 'heave',
        'roll', 'pitch', 'yaw'. Default: ['heave', 'pitch'].
    added_mass : dict[str, float] | None
        Added mass coefficients (diagonal only). Keys: 'surge', 'sway',
        'heave', 'roll', 'pitch', 'yaw'. Values in kg or kg·m².
    damping : dict[str, float] | None
        Linear damping coefficients. Same keys as added_mass.
    """
    mass: float = 1000.0
    Ixx: float = 500.0
    Iyy: float = 500.0
    Izz: float = 500.0
    x_cg: float = 0.0
    y_cg: float = 0.0
    z_cg: float = 0.0
    x_cb: float = 0.0
    y_cb: float = 0.0
    z_cb: float = 0.0
    waterplane_area: float = 1.0
    waterplane_pitch_moi: float = 1.0
    free_dof: list[str] = field(default_factory=lambda: ["heave", "pitch"])
    added_mass: dict[str, float] | None = None
    damping: dict[str, float] | None = None

    def __post_init__(self):
        valid_dofs = {"surge", "sway", "heave", "roll", "pitch", "yaw"}
        for dof in self.free_dof:
            if dof not in valid_dofs:
                raise ValueError(f"Invalid DOF: {dof}. Must be one of {valid_dofs}")


def hydrostatic_restoring_heave(z: float, rho: float, g: float,
                                 waterplane_area: float) -> float:
    """Hydrostatic restoring force in heave.

    F = −ρ·g·A_wp · z

    Parameters
    ----------
    z : float
        Heave displacement from equilibrium (m, positive up).
    rho : float
        Water density (kg/m³).
    g : float
        Gravity (m/s²).
    waterplane_area : float
        Waterplane area (m²).

    Returns
    -------
    F : float
        Restoring force (N, positive up).
    """
    return -rho * g * waterplane_area * z


def hydrostatic_restoring_pitch(pitch: float, rho: float, g: float,
                                 waterplane_pitch_moi: float) -> float:
    """Hydrostatic restoring moment in pitch.

    M = −ρ·g·I_wp · θ

    Parameters
    ----------
    pitch : float
        Pitch angle from equilibrium (rad, bow-up positive).
    rho : float
        Water density (kg/m³).
    g : float
        Gravity (m/s²).
    waterplane_pitch_moi : float
        Waterplane second moment about transverse axis (m⁴).

    Returns
    -------
    M : float
        Restoring moment (N·m).
    """
    return -rho * g * waterplane_pitch_moi * pitch


class SixDOFIntegrator:
    """Symplectic Euler integrator for 6DOF rigid-body motion.

    Parameters
    ----------
    config : SixDOFConfig
        Body configuration.
    rho : float
        Water density (kg/m³).
    g : float
        Gravity (m/s²).
    """

    #: Placeholder damping ratio used when ``config.damping`` does not cover a
    #: requested DOF.  Kept for backward compatibility with the historical
    #: hard-coded 5 %; it is not a physical model and should be overridden by
    #: configuring real damping coefficients.
    DEFAULT_DAMPING_RATIO = 0.05

    def __init__(self, config: SixDOFConfig, rho: float = 1000.0,
                 g: float = 9.81):
        self.config = config
        self.rho = rho
        self.g = g
        self.state = SixDOFState()
        self.history: list[dict] = []
        self.time: float = 0.0

        # Effective mass (mass + added mass)
        self.effective_mass = {
            "surge": config.mass + (config.added_mass or {}).get("surge", 0.0),
            "sway": config.mass + (config.added_mass or {}).get("sway", 0.0),
            "heave": config.mass + (config.added_mass or {}).get("heave", 0.0),
            "roll": config.Ixx + (config.added_mass or {}).get("roll", 0.0),
            "pitch": config.Iyy + (config.added_mass or {}).get("pitch", 0.0),
            "yaw": config.Izz + (config.added_mass or {}).get("yaw", 0.0),
        }

    def compute_forces(self, F_ext: dict[str, float] | None = None) -> dict[str, float]:
        """Compute total forces and moments on the body.

        Parameters
        ----------
        F_ext : dict, optional
            External forces/moments from fluid (hydrodynamic + wave excitation).
            Keys: 'surge', 'sway', 'heave', 'roll', 'pitch', 'yaw'.

        Returns
        -------
        forces : dict
            Total forces/moments for each DOF.
        """
        forces = {dof: 0.0 for dof in ["surge", "sway", "heave", "roll", "pitch", "yaw"]}

        # Hydrostatic restoring
        if "heave" in self.config.free_dof:
            forces["heave"] += hydrostatic_restoring_heave(
                self.state.z, self.rho, self.g, self.config.waterplane_area
            )
        if "pitch" in self.config.free_dof:
            forces["pitch"] += hydrostatic_restoring_pitch(
                self.state.pitch, self.rho, self.g, self.config.waterplane_pitch_moi
            )

        # Linear damping
        if self.config.damping:
            vel_map = {
                "surge": self.state.u, "sway": self.state.v, "heave": self.state.w,
                "roll": self.state.p, "pitch": self.state.q, "yaw": self.state.r
            }
            for dof in self.config.free_dof:
                if dof in self.config.damping:
                    forces[dof] -= self.config.damping[dof] * vel_map.get(dof, 0.0)

        # External forces (from fluid solver)
        if F_ext:
            for dof in self.config.free_dof:
                forces[dof] += F_ext.get(dof, 0.0)

        return forces

    def step(self, dt: float, F_ext: dict[str, float] | None = None):
        """Advance the 6DOF state by one time step.

        Parameters
        ----------
        dt : float
            Time step (s).
        F_ext : dict, optional
            External forces/moments from fluid solver.
        """
        forces = self.compute_forces(F_ext)

        # Symplectic Euler: update velocity first, then position
        dof_map = {
            "surge": ("u", "x"), "sway": ("v", "y"), "heave": ("w", "z"),
            "roll": ("p", "roll"), "pitch": ("q", "pitch"), "yaw": ("r", "yaw")
        }

        for dof in self.config.free_dof:
            vel_attr, pos_attr = dof_map[dof]
            m_eff = self.effective_mass[dof]
            if m_eff > 0:
                accel = forces[dof] / m_eff
                # Update velocity
                setattr(self.state, vel_attr, getattr(self.state, vel_attr) + accel * dt)
                # Update position
                setattr(self.state, pos_attr,
                        getattr(self.state, pos_attr) + getattr(self.state, vel_attr) * dt)

        # Record history.  Time is accumulated from the actual dt values rather
        # than inferred from the history length: ``len(history) * dt`` silently
        # reports wrong times whenever the caller changes dt during a run
        # (e.g. adaptive sub-stepping inside the fluid solver).
        self.time += dt
        self.history.append({
            "time": self.time,
            "z": self.state.z,
            "pitch": self.state.pitch,
            "w": self.state.w,
            "q": self.state.q,
        })

    def natural_frequency_heave(self) -> float:
        """Natural frequency in heave (rad/s).

        ω_n = √(ρ·g·A_wp / m_eff)
        """
        m_eff = self.effective_mass["heave"]
        if m_eff <= 0:
            return 0.0
        return math.sqrt(self.rho * self.g * self.config.waterplane_area / m_eff)

    def natural_frequency_pitch(self) -> float:
        """Natural frequency in pitch (rad/s).

        ω_n = √(ρ·g·I_wp / I_eff)
        """
        i_eff = self.effective_mass["pitch"]
        if i_eff <= 0:
            return 0.0
        return math.sqrt(self.rho * self.g * self.config.waterplane_pitch_moi / i_eff)

    def restoring_stiffness(self, dof: str = "heave") -> float:
        """Hydrostatic restoring stiffness ``k`` for ``dof`` (N/m or N·m/rad).

        ``k = rho * g * A_wp`` for heave and ``k = rho * g * I_wp`` for pitch,
        i.e. exactly the coefficient used by :func:`hydrostatic_restoring_heave`
        and :func:`hydrostatic_restoring_pitch`.
        """
        if dof == "heave":
            return self.rho * self.g * self.config.waterplane_area
        if dof == "pitch":
            return self.rho * self.g * self.config.waterplane_pitch_moi
        return 0.0

    def damping_ratio(self, dof: str = "heave") -> float:
        """Damping ratio ``zeta`` implied by the configured linear damping.

        ``zeta = c / (2 * sqrt(k * m_eff))`` with ``c`` taken from
        ``config.damping[dof]``.  When no damping coefficient is configured for
        ``dof`` (the historical default) ``DEFAULT_DAMPING_RATIO = 0.05`` is
        returned so that callers relying on the old behaviour are unaffected;
        that value is a placeholder, not a physical model.
        """
        c = (self.config.damping or {}).get(dof, 0.0)
        if c <= 0.0:
            return self.DEFAULT_DAMPING_RATIO
        m_eff = self.effective_mass.get(dof, 0.0)
        k = self.restoring_stiffness(dof)
        if m_eff <= 0.0 or k <= 0.0:
            return self.DEFAULT_DAMPING_RATIO
        return c / (2.0 * math.sqrt(k * m_eff))

    def rao(self, omega: float, dof: str = "heave",
            zeta: float | None = None) -> float:
        """Magnitude of the linear SDOF transfer function.

        ``RAO = 1 / sqrt((1 - r^2)^2 + (2 zeta r)^2)`` with ``r = omega/omega_n``.

        Parameters
        ----------
        omega : float
            Wave frequency (rad/s).
        dof : str
            Degree of freedom ('heave' or 'pitch').
        zeta : float, optional
            Damping ratio.  When omitted it is derived from
            ``config.damping[dof]`` via :meth:`damping_ratio` (which falls back
            to ``DEFAULT_DAMPING_RATIO = 0.05`` if no damping is configured).

        Returns
        -------
        rao : float
            RAO magnitude (m/m for heave, rad/m for pitch).  ``1 / (2 zeta)`` at
            resonance, so a finite value is returned for any ``zeta > 0``.
        """
        if dof == "heave":
            omega_n = self.natural_frequency_heave()
        elif dof == "pitch":
            omega_n = self.natural_frequency_pitch()
        else:
            return 0.0

        if omega_n <= 0:
            return 0.0

        ratio = omega / omega_n
        if zeta is None:
            zeta = self.damping_ratio(dof)
        denom = math.sqrt((1 - ratio ** 2) ** 2 + (2 * zeta * ratio) ** 2)
        return 1.0 / denom
