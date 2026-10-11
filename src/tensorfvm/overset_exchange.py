"""SI macroscopic exchange contract for future FVM / isothermal LBM coupling.

This module exchanges point values through existing overset connectivity. It
does not exchange populations, viscous stress, conservative fluxes or solve a
coupled pressure equation. LBM pressure is explicitly referenced to its density
reference: p - p_ref = rho_scale * cs_lattice^2 * (dx/dt)^2
                         * (rho_lattice - rho_ref/rho_scale).
Only single phase isothermal LBM is represented; nonideal multiphase EOS is not.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class MacroscopicState:
    """Cell point values: density kg/m³, velocity m/s, pressure Pa, time s."""
    density: np.ndarray
    velocity: np.ndarray
    pressure: np.ndarray
    time: float

    def __post_init__(self):
        rho, u, p = (np.array(x, dtype=float, copy=True) for x in
                     (self.density, self.velocity, self.pressure))
        if (rho.ndim != 1 or p.shape != rho.shape or u.ndim != 2
                or u.shape[0] != len(rho) or u.shape[1] not in (2, 3)):
            raise ValueError('Expected density/pressure (N,) and velocity (N,2|3)')
        if (not all(np.isfinite(x).all() for x in (rho, u, p))
                or np.any(rho <= 0) or not np.isfinite(self.time)):
            raise ValueError('State must be finite with positive density')
        for name, value in zip(('density', 'velocity', 'pressure'), (rho, u, p)):
            value.setflags(write=False)
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class LatticeUnits:
    """Explicit SI/lattice scales and density policy; no implicit gauge.

    ``incompressible_reference`` keeps the FVM material density fixed and uses
    LBM density perturbations to encode pressure. ``isothermal_eos`` requires
    physical density and pressure to satisfy the configured EOS. max_mach is an
    input admissibility limit, not an accuracy certificate.
    """
    dx: float
    dt: float
    density_scale: float
    reference_density: float
    reference_pressure: float
    cs2: float
    density_policy: str
    max_mach: float = 0.1
    consistency_rtol: float = 1e-9

    def __post_init__(self):
        positive = (self.dx, self.dt, self.density_scale,
                    self.reference_density, self.cs2, self.max_mach)
        if not all(np.isfinite(x) and x > 0 for x in positive):
            raise ValueError('Unit scales, reference density and limits must be positive')
        if not np.isfinite(self.reference_pressure):
            raise ValueError('Reference pressure must be finite')
        if not np.isfinite(self.consistency_rtol) or self.consistency_rtol < 0:
            raise ValueError('Density consistency tolerance must be nonnegative')
        if self.density_policy not in ('incompressible_reference', 'isothermal_eos'):
            raise ValueError('Choose an explicit supported density policy')

    @property
    def pressure_scale(self):
        return self.density_scale * (self.dx / self.dt)**2

    def lattice_viscosity(self, kinematic_viscosity):
        """Convert SI kinematic viscosity to lattice units, independent of EOS.

        Collision relaxation is deliberately not set here: BGK, MRT, central
        moment and turbulence implementations have different configuration.
        """
        nu = np.asarray(kinematic_viscosity, dtype=float)
        if not np.isfinite(nu).all() or np.any(nu < 0):
            raise ValueError('Kinematic viscosity must be finite and nonnegative')
        return nu * self.dt / self.dx**2

    def encode(self, state):
        """Return lattice density and velocity; never reconstruct populations."""
        rho = (self.reference_density / self.density_scale
               + (state.pressure - self.reference_pressure)
               / (self.cs2 * self.pressure_scale))
        expected = (np.full_like(state.density, self.reference_density)
                    if self.density_policy == 'incompressible_reference'
                    else rho * self.density_scale)
        if not np.allclose(state.density, expected, rtol=self.consistency_rtol, atol=0):
            raise ValueError('Physical density conflicts with configured pressure/EOS policy')
        u = state.velocity * self.dt / self.dx
        self._check_lattice(rho, u)
        return rho, u

    def _check_lattice(self, rho, u):
        if (rho.ndim != 1 or u.ndim != 2 or u.shape[0] != len(rho)
                or u.shape[1] not in (2, 3) or not np.isfinite(rho).all()
                or not np.isfinite(u).all() or np.any(rho <= 0)):
            raise ValueError('Invalid lattice density or velocity')
        if np.any(np.linalg.norm(u, axis=1) / np.sqrt(self.cs2) > self.max_mach):
            raise ValueError('Lattice velocity exceeds configured Mach limit')

    def decode(self, density, velocity, *, time):
        rho, u = np.asarray(density, float), np.asarray(velocity, float)
        self._check_lattice(rho, u)
        p = (self.reference_pressure + self.cs2 * self.pressure_scale
             * (rho - self.reference_density / self.density_scale))
        physical_rho = (np.full_like(rho, self.reference_density)
                        if self.density_policy == 'incompressible_reference'
                        else rho * self.density_scale)
        return MacroscopicState(physical_rho, u * self.dx / self.dt, p, time)


def interpolate_time(earlier, later, time):
    """Linear SI point-state interpolation for subcycling; reject extrapolation."""
    if (not np.isfinite(time) or later.time <= earlier.time
            or not earlier.time <= time <= later.time):
        raise ValueError('Requested time must be bracketed by increasing timestamps')
    if earlier.velocity.shape != later.velocity.shape:
        raise ValueError('Temporal states must have identical cell/component shapes')
    weight = (time - earlier.time) / (later.time - earlier.time)
    return MacroscopicState(*[(1-weight)*getattr(earlier, name)
                              + weight*getattr(later, name)
                              for name in ('density', 'velocity', 'pressure')], time)


def exchange_macroscopic(connectivity, states):
    """Simultaneously interpolate SI point states at a common physical time.

    Conversion belongs at solver boundaries: components may have different
    dx/dt scales. No volume, momentum or energy conservation is claimed.
    """
    if len(states) != len(connectivity.grids) or not states:
        raise ValueError('One state per overset grid required')
    if any(s.time != states[0].time for s in states):
        raise ValueError('Synchronize physical time before spatial exchange')
    transferred = [connectivity.interpolate(tuple(getattr(s, name) for s in states))
                   for name in ('density', 'velocity', 'pressure')]
    return tuple(MacroscopicState(r, u, p, states[0].time)
                 for r, u, p in zip(*transferred))
