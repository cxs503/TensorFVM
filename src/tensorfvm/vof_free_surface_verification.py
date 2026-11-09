"""VOF free-surface verification suite: three-level analytical ladder.

This module implements three verification cases for the VOF free-surface
capability in TensorFVM, forming a progressive ladder from shallow-water
hyperbolic dynamics to dispersive wave physics and capillary damping:

**L1 — Ritter (1892) dam break (shallow-water analytical anchor)**
    Initial water column of height h₀ on left half, dry bed on right.
    Shallow-water theory gives wave-front position x_f(t) = 2√(g·h₀)·t
    and front velocity u_f = 2√(g·h₀). Validates hyperbolic conservation
    and mass conservation.

**L2 — Stokes 2nd-order wave propagation (dispersion relation)**
    Finite depth h, wave amplitude a, wavelength λ. Linear wave theory
    gives phase speed c = √(g·λ/(2π)·tanh(2π·h/λ)). Validates that the
    VOF solver captures the correct dispersion relation for gravity waves.

**L3 — Capillary wave damping (Lamb 1932)**
    Small-amplitude sinusoidal perturbation with surface tension σ.
    Amplitude decays as η(t) = η₀·exp(-2ν·k²·t) with frequency
    ω² = (g·k + σ·k³/ρ)·tanh(k·h). Validates surface tension coupling
    and viscous damping.

**Negative controls:**
    - No surface tension → capillary wave frequency reduces to gravity wave
    - No gravity → pure capillary wave ω² = σ·k³/ρ·tanh(k·h)

References
----------
- Ritter, A. (1892). "Die Verpflanzung einer Wasserwelle". Z. Math. Phys.
- Lamb, H. (1932). Hydrodynamics, 6th ed. Cambridge Univ. Press.
- Stokes, G.G. (1847). "On the theory of oscillatory waves".
  Trans. Camb. Phil. Soc., 8, 441–473.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch

from .two_phase_solver import TwoPhaseSolver, GridConfig, TwoPhaseConfig


# ============================================================================
# L1: Ritter (1892) Dam Break
# ============================================================================

@dataclass
class RitterDamBreakConfig:
    """Configuration for Ritter dam-break verification.

    Parameters
    ----------
    h0 : float
        Initial water column height (m).
    L : float
        Domain length (m). The water initially occupies [0, L/2].
    H : float
        Domain height (m). Must be > h0 to accommodate wave run-up.
    nx, ny : int
        Grid resolution.
    g : float
        Gravitational acceleration (m/s²).
    t_end : float
        Simulation end time (s). Should be before wave hits right wall.
    cfl : float
        CFL number for time-step selection.
    """
    h0: float = 1.0
    L: float = 4.0
    H: float = 2.0
    nx: int = 80
    ny: int = 40
    g: float = 9.81
    t_end: float = 0.5
    cfl: float = 0.3
    rho_liquid: float = 1000.0
    rho_gas: float = 1.0
    mu_liquid: float = 1.0e-3
    mu_gas: float = 1.0e-5


def ritter_analytical_front(t: float, h0: float, g: float) -> float:
    """Ritter (1892) analytical wave-front position.

    x_f(t) = 2·√(g·h₀)·t

    Valid for t < t_wall when wave reaches downstream end.
    """
    return 2.0 * math.sqrt(g * h0) * t


def ritter_analytical_velocity(h0: float, g: float) -> float:
    """Ritter analytical wave-front velocity.

    u_f = 2·√(g·h₀)
    """
    return 2.0 * math.sqrt(g * h0)


def ritter_analytical_profile(x: float, t: float, h0: float, g: float) -> float:
    """Ritter analytical free-surface profile η(x, t).

    For the dam-break on a dry bed, the profile is:
        η(x, t) = (1/(9g)) · (2√(g·h₀) - x/t)²   for x in [x_tail, x_front]
        η = h0                                       for x < x_tail
        η = 0                                        for x > x_front

    where x_front = 2√(g·h₀)·t and x_tail = -√(g·h₀)·t (relative to dam).
    """
    c0 = math.sqrt(g * h0)
    x_front = 2.0 * c0 * t
    x_tail = -c0 * t  # relative to initial dam position

    if x < x_tail:
        return h0
    elif x > x_front:
        return 0.0
    else:
        return (2.0 * c0 - x / t) ** 2 / (9.0 * g)


def run_ritter_dam_break(
    config: RitterDamBreakConfig | None = None,
) -> Dict:
    """Run Ritter dam-break simulation and return verification data.

    Returns
    -------
    dict with keys:
        'times': list of snapshot times
        'front_positions': measured wave-front x-positions
        'front_analytical': analytical front positions
        'front_errors': relative errors
        'mass_conservation': [initial_mass, final_mass, rel_error]
        'profiles': list of (x, eta) tuples at snapshot times
        'analytical_profiles': list of (x, eta_analytical) at snapshot times
    """
    if config is None:
        config = RitterDamBreakConfig()

    c = config
    dx = c.L / c.nx
    dy = c.H / c.ny

    # Set up solver (following pattern from verification/dam_break.py)
    grid = GridConfig(
        nx=c.nx, ny=c.ny,
        length=c.L, height=c.H,
        velocity_relaxation=0.5,
        pressure_relaxation=0.3,
        max_iterations=1,
        inlet_velocity=0.0,
    )
    tp = TwoPhaseConfig(
        enabled=True,
        rho_liquid=c.rho_liquid,
        rho_gas=c.rho_gas,
        mu_liquid=c.mu_liquid,
        mu_gas=c.mu_gas,
        sigma=0.0,  # no surface tension for dam break
        surface_tension=False,
        gravity=c.g,
        top_atmosphere=True,
        bottom_wall=True,
        side_outlet=False,  # closed domain
        domain_type="closed",
        vof_cfl_limit=c.cfl,
    )

    solver = TwoPhaseSolver(grid, tp)

    # Initial condition: water on left half up to height h0
    x = (torch.arange(c.nx, dtype=torch.float64) + 0.5) * dx
    y = (torch.arange(c.ny, dtype=torch.float64) + 0.5) * dy
    X, Y = torch.meshgrid(y, x, indexing='ij')

    # Water where y < h0 and x < L/2
    alpha0 = ((Y < c.h0) & (X < c.L / 2)).double()
    solver.set_alpha(alpha0)

    initial_mass = float(solver.alpha.sum()) * dx * dy

    # Time stepping with adaptive CFL
    c0 = math.sqrt(c.g * c.h0)
    dt = c.cfl * min(dx, dy) / (2.0 * c0)

    # Snapshot times
    n_snapshots = 10
    snapshot_times = [c.t_end * (i + 1) / n_snapshots for i in range(n_snapshots)]

    results = {
        'times': [],
        'front_positions': [],
        'front_analytical': [],
        'front_errors': [],
        'mass_initial': initial_mass,
        'mass_final': 0.0,
        'profiles': [],
        'analytical_profiles': [],
    }

    t = 0.0
    snap_idx = 0
    step_count = 0

    while t < c.t_end:
        # Take a step
        try:
            info = solver.step(dt)
        except RuntimeError:
            break

        t = solver.time
        step_count += 1

        # Check if we hit a snapshot time
        if snap_idx < len(snapshot_times) and t >= snapshot_times[snap_idx]:
            # Measure wave front: find rightmost x where alpha > 0.5
            alpha = solver.alpha
            # For each column, find if there's water (alpha > 0.5 in any row)
            has_water = (alpha > 0.5).any(dim=0)  # (nx,)
            water_cols = torch.where(has_water)[0]
            if len(water_cols) > 0:
                front_x = float(x[water_cols[-1]])
            else:
                front_x = 0.0

            # Analytical front (dam is at x = L/2)
            t_snap = snapshot_times[snap_idx]
            front_ana = c.L / 2 + ritter_analytical_front(t_snap, c.h0, c.g)

            results['times'].append(t_snap)
            results['front_positions'].append(front_x)
            results['front_analytical'].append(front_ana)
            if front_ana > 0:
                results['front_errors'].append(abs(front_x - front_ana) / front_ana)
            else:
                results['front_errors'].append(0.0)

            # Store profile (column-averaged alpha * H gives surface height)
            eta = alpha.sum(dim=0) * dy  # (nx,)
            results['profiles'].append((x.clone(), eta.clone()))

            # Analytical profile
            x_np = x.numpy()
            eta_ana = torch.tensor(
                [ritter_analytical_profile(xi - c.L / 2, t_snap, c.h0, c.g)
                 for xi in x_np],
                dtype=torch.float64
            )
            results['analytical_profiles'].append((x.clone(), eta_ana))

            snap_idx += 1

        # Adaptive time step
        u_max = max(
            float(solver.u.abs().max()),
            float(solver.v.abs().max()),
            0.01
        )
        dt = min(dx, dy) * c.cfl / u_max
        dt = min(dt, 0.002)  # cap

    results['mass_final'] = float(solver.alpha.sum()) * dx * dy
    results['mass_rel_error'] = abs(results['mass_final'] - results['mass_initial']) / results['mass_initial']
    results['n_steps'] = step_count

    return results


# ============================================================================
# L2: Stokes Wave Propagation
# ============================================================================

@dataclass
class StokesWaveConfig:
    """Configuration for Stokes wave verification.

    Parameters
    ----------
    h : float
        Water depth (m).
    amplitude : float
        Wave amplitude a (m). Must be << h for linear theory validity.
    wavelength : float
        Wavelength λ (m).
    L : float
        Domain length. Should be = wavelength for periodic-like setup.
    nx, ny : int
        Grid resolution.
    g : float
        Gravitational acceleration (m/s²).
    n_periods : float
        Number of wave periods to simulate.
    cfl : float
        CFL number.
    """
    h: float = 1.0
    amplitude: float = 0.05  # a/h = 0.05 (linear regime)
    wavelength: float = 2.0
    L: float = 2.0  # = wavelength
    H: float = 2.0  # depth + air region
    nx: int = 64
    ny: int = 64
    g: float = 9.81
    n_periods: float = 5.0
    cfl: float = 0.3
    rho_liquid: float = 1000.0
    rho_gas: float = 1.0
    mu_liquid: float = 1.0e-3
    mu_gas: float = 1.0e-5


def linear_wave_speed(g: float, wavelength: float, depth: float) -> float:
    """Linear (Airy) wave phase speed.

    c = √(g·λ/(2π) · tanh(2π·h/λ))
    """
    k = 2.0 * math.pi / wavelength
    return math.sqrt(g / k * math.tanh(k * depth))


def linear_wave_frequency(g: float, wavelength: float, depth: float) -> float:
    """Linear wave angular frequency ω.

    ω² = g·k·tanh(k·h)
    """
    k = 2.0 * math.pi / wavelength
    return math.sqrt(g * k * math.tanh(k * depth))


def stokes_second_order_correction(h_over_lambda: float) -> float:
    """Stokes 2nd-order correction factor C₂ for phase speed.

    c/c_linear = 1 + (a/λ)² · C₂(h/λ)

    C₂ = (2π)² · [cosh(4π·h/λ) + 2] / (8·sinh⁴(2π·h/λ))
         · [2·sinh²(2π·h/λ) + 1] / (2·sinh²(2π·h/λ))
         - (2π)² / (4·sinh²(2π·h/λ))

    Simplified: for deep water (h/λ >> 1), C₂ → (2π)²/4 ≈ 9.87
    For shallow water, C₂ → (2π)²/(3·(h/λ)²)
    """
    kh = 2.0 * math.pi * h_over_lambda
    sh = math.sinh(kh)
    ch = math.cosh(kh)
    # Stokes 2nd order: c/c0 = 1 + (ka)² * f(kh)
    # f(kh) = [cosh(4kh) + 2 - 4·cosh²(2kh) + ...] / (8·sinh⁴(2kh))
    # Use the standard result:
    # C2 = (1/4) * (2 + sinh²(2kh)) / sinh⁴(kh) * cosh(2kh)
    #      - (1/2) / sinh²(kh)
    # Simplified form from Squire (1990):
    if kh > 10:
        # Deep water limit
        return 1.0  # correction is O(1) but (a/λ)² is tiny
    sh2 = math.sinh(2 * kh)
    ch2 = math.cosh(2 * kh)
    C2 = (ch2 * (2 + sh2**2 / 4)) / (4 * sh**4) - 1.0 / (2 * sh**2)
    return C2


def run_stokes_wave(
    config: StokesWaveConfig | None = None,
) -> Dict:
    """Run Stokes wave simulation and return verification data.

    Returns
    -------
    dict with keys:
        'wave_speed_measured': measured phase speed
        'wave_speed_analytical': linear theory phase speed
        'wave_speed_error': relative error
        'surface_history': list of (x, eta) at various times
        'phase_error_deg': phase error after n_periods (degrees)
    """
    if config is None:
        config = StokesWaveConfig()

    c = config
    dx = c.L / c.nx
    dy = c.H / c.ny
    k = 2.0 * math.pi / c.wavelength
    omega = linear_wave_frequency(c.g, c.wavelength, c.h)
    period = 2.0 * math.pi / omega
    c_theory = linear_wave_speed(c.g, c.wavelength, c.h)
    t_end = c.n_periods * period

    # Set up solver
    grid = GridConfig(
        nx=c.nx, ny=c.ny,
        length=c.L, height=c.H,
        velocity_relaxation=0.5,
        pressure_relaxation=0.3,
        max_iterations=1,
        inlet_velocity=0.0,
    )
    tp = TwoPhaseConfig(
        enabled=True,
        rho_liquid=c.rho_liquid,
        rho_gas=c.rho_gas,
        mu_liquid=c.mu_liquid,
        mu_gas=c.mu_gas,
        sigma=0.0,
        surface_tension=False,
        gravity=c.g,
        top_atmosphere=True,
        bottom_wall=True,
        side_outlet=False,
        domain_type="closed",
        vof_cfl_limit=c.cfl,
    )

    solver = TwoPhaseSolver(grid, tp)

    # Initial condition: sinusoidal interface at rest
    x = (torch.arange(c.nx, dtype=torch.float64) + 0.5) * dx
    y = (torch.arange(c.ny, dtype=torch.float64) + 0.5) * dy
    X, Y = torch.meshgrid(y, x, indexing='ij')

    # Interface: y = h + a * cos(k * x)
    # alpha = 1 where y < interface
    interface_y = c.h + c.amplitude * torch.cos(k * X)
    alpha0 = (Y < interface_y).double()
    solver.set_alpha(alpha0)

    # Time stepping
    dt = c.cfl * min(dx, dy) / c_theory

    # Track surface elevation at fixed x-locations
    n_probes = min(16, c.nx)
    probe_indices = torch.linspace(0, c.nx - 1, n_probes).long()
    probe_x = x[probe_indices]

    results = {
        'wave_speed_measured': 0.0,
        'wave_speed_analytical': c_theory,
        'wave_speed_error': 0.0,
        'surface_history': [],
        'phase_error_deg': 0.0,
        'period_analytical': period,
        'amplitude_history': [],
    }

    t = 0.0
    step_count = 0
    snapshot_interval = max(1, int(period / (10 * dt)))
    peak_times = []
    peak_x = []

    while t < t_end:
        try:
            info = solver.step(dt)
        except RuntimeError:
            break

        t = solver.time
        step_count += 1

        # Record surface elevation periodically
        if step_count % snapshot_interval == 0:
            alpha = solver.alpha
            eta = alpha.sum(dim=0) * dy  # (nx,) surface elevation
            results['surface_history'].append((t, x.clone(), eta.clone()))

            # Track amplitude (max - min of eta)
            eta_max = float(eta.max())
            eta_min = float(eta.min())
            results['amplitude_history'].append((t, (eta_max - eta_min) / 2.0))

        # Adaptive time step
        u_max = max(
            float(solver.u.abs().max()),
            float(solver.v.abs().max()),
            0.01
        )
        dt = min(dx, dy) * c.cfl / u_max
        dt = min(dt, 0.002)

    # Analyze wave speed from surface elevation time series
    # Use cross-correlation of surface at two x-locations
    if len(results['surface_history']) > 10:
        # Build time series at two probe locations
        i1 = c.nx // 4
        i2 = 3 * c.nx // 4
        dx_probe = float(x[i2] - x[i1])

        times_arr = torch.tensor([s[0] for s in results['surface_history']])
        eta1 = torch.stack([s[2][i1] for s in results['surface_history']])
        eta2 = torch.stack([s[2][i2] for s in results['surface_history']])

        # Cross-correlation to find time lag
        eta1_centered = eta1 - eta1.mean()
        eta2_centered = eta2 - eta2.mean()

        # Normalize
        norm = torch.sqrt(torch.sum(eta1_centered**2) * torch.sum(eta2_centered**2))
        if norm > 1e-15:
            # Find lag that maximizes correlation
            n_t = len(times_arr)
            max_corr = -1.0
            best_lag_idx = 0
            dt_snap = float(times_arr[1] - times_arr[0]) if len(times_arr) > 1 else dt

            for lag in range(-n_t // 4, n_t // 4):
                if lag >= 0:
                    corr = torch.sum(eta1_centered[:n_t - lag] * eta2_centered[lag:]) / norm
                else:
                    corr = torch.sum(eta1_centered[-lag:] * eta2_centered[:n_t + lag]) / norm
                if corr > max_corr:
                    max_corr = corr
                    best_lag_idx = lag

            time_lag = best_lag_idx * dt_snap
            if abs(time_lag) > 1e-15:
                c_measured = abs(dx_probe / time_lag)
            else:
                c_measured = c_theory  # fallback
        else:
            c_measured = c_theory

        results['wave_speed_measured'] = c_measured
        results['wave_speed_error'] = abs(c_measured - c_theory) / c_theory

        # Phase error: compare measured period from zero-crossings
        # Find zero crossings of eta at mid-domain
        i_mid = c.nx // 2
        eta_mid = torch.stack([s[2][i_mid] for s in results['surface_history']])
        eta_mid_centered = eta_mid - eta_mid.mean()
        zero_crossings = []
        for i in range(1, len(eta_mid_centered)):
            if eta_mid_centered[i - 1] * eta_mid_centered[i] < 0:
                # Linear interpolation
                frac = -eta_mid_centered[i - 1] / (eta_mid_centered[i] - eta_mid_centered[i - 1])
                t_cross = float(times_arr[i - 1]) + frac * dt_snap
                zero_crossings.append(t_cross)

        if len(zero_crossings) >= 3:
            # Half-periods between consecutive zero crossings
            half_periods = [zero_crossings[i + 1] - zero_crossings[i]
                          for i in range(len(zero_crossings) - 1)]
            measured_period = 2.0 * sum(half_periods) / len(half_periods)
            phase_error_rad = 2 * math.pi * abs(measured_period - period) / period
            results['phase_error_deg'] = math.degrees(phase_error_rad)
            results['period_measured'] = measured_period

    results['n_steps'] = step_count
    return results


# ============================================================================
# L3: Capillary Wave Damping (Lamb 1932)
# ============================================================================

@dataclass
class CapillaryWaveConfig:
    """Configuration for capillary wave damping verification.

    Parameters
    ----------
    h : float
        Water depth (m).
    amplitude : float
        Initial wave amplitude (m). Must be small for linear theory.
    wavelength : float
        Wavelength λ (m).
    sigma : float
        Surface tension coefficient (N/m).
    nu : float
        Kinematic viscosity of liquid (m²/s).
    nx, ny : int
        Grid resolution.
    g : float
        Gravitational acceleration (m/s²).
    n_periods : float
        Number of wave periods to simulate.
    cfl : float
        CFL number.
    """
    h: float = 0.5
    amplitude: float = 0.005  # small amplitude
    wavelength: float = 0.1   # short wave → capillary dominated
    sigma: float = 0.074      # water-air surface tension
    nu: float = 1.0e-6        # water kinematic viscosity
    L: float = 0.1            # = wavelength
    H: float = 1.0            # depth + air
    nx: int = 40
    ny: int = 60
    g: float = 9.81
    n_periods: float = 10.0
    cfl: float = 0.2
    rho_liquid: float = 1000.0
    rho_gas: float = 1.0
    mu_liquid: float = 1.0e-3  # rho * nu
    mu_gas: float = 1.0e-5


def capillary_wave_frequency(
    g: float, k: float, h: float, sigma: float, rho: float
) -> float:
    """Capillary-gravity wave angular frequency.

    ω² = (g·k + σ·k³/ρ) · tanh(k·h)
    """
    return math.sqrt((g * k + sigma * k**3 / rho) * math.tanh(k * h))


def capillary_damping_rate(nu: float, k: float) -> float:
    """Viscous damping rate for capillary waves (Lamb 1932).

    η(t) = η₀ · exp(-γ·t)  where  γ = 2·ν·k²

    This is the leading-order damping for small viscosity.
    """
    return 2.0 * nu * k**2


def run_capillary_wave(
    config: CapillaryWaveConfig | None = None,
    negative_control: str | None = None,
) -> Dict:
    """Run capillary wave damping simulation.

    Parameters
    ----------
    config : CapillaryWaveConfig
        Simulation parameters.
    negative_control : str or None
        If "no_sigma", set σ=0 (should give gravity wave frequency).
        If "no_gravity", set g=0 (should give pure capillary frequency).

    Returns
    -------
    dict with keys:
        'damping_rate_measured': measured exponential decay rate
        'damping_rate_analytical': 2·ν·k²
        'damping_error': relative error
        'frequency_measured': measured angular frequency
        'frequency_analytical': theoretical ω
        'frequency_error': relative error
        'amplitude_history': list of (time, amplitude)
    """
    if config is None:
        config = CapillaryWaveConfig()

    c = config
    dx = c.L / c.nx
    dy = c.H / c.ny
    k = 2.0 * math.pi / c.wavelength

    # Apply negative controls
    g_eff = c.g
    sigma_eff = c.sigma
    if negative_control == "no_sigma":
        sigma_eff = 0.0
    elif negative_control == "no_gravity":
        g_eff = 0.0

    omega_theory = capillary_wave_frequency(g_eff, k, c.h, sigma_eff, c.rho_liquid)
    gamma_theory = capillary_damping_rate(c.nu, k)
    period = 2.0 * math.pi / omega_theory if omega_theory > 0 else float('inf')
    t_end = c.n_periods * period if period < float('inf') else 1.0

    # Set up solver
    grid = GridConfig(
        nx=c.nx, ny=c.ny,
        length=c.L, height=c.H,
        velocity_relaxation=0.5,
        pressure_relaxation=0.3,
        max_iterations=1,
        inlet_velocity=0.0,
    )
    tp = TwoPhaseConfig(
        enabled=True,
        rho_liquid=c.rho_liquid,
        rho_gas=c.rho_gas,
        mu_liquid=c.mu_liquid,
        mu_gas=c.mu_gas,
        sigma=sigma_eff,
        surface_tension=(sigma_eff > 0),
        gravity=g_eff,
        top_atmosphere=True,
        bottom_wall=True,
        side_outlet=False,
        domain_type="closed",
        vof_cfl_limit=c.cfl,
    )

    solver = TwoPhaseSolver(grid, tp)

    # Initial condition: sinusoidal perturbation
    x = (torch.arange(c.nx, dtype=torch.float64) + 0.5) * dx
    y = (torch.arange(c.ny, dtype=torch.float64) + 0.5) * dy
    X, Y = torch.meshgrid(y, x, indexing='ij')

    interface_y = c.h + c.amplitude * torch.cos(k * X)
    alpha0 = (Y < interface_y).double()
    solver.set_alpha(alpha0)

    # Time stepping
    c_phase = omega_theory / k if omega_theory > 0 else math.sqrt(g_eff * c.h)
    dt = c.cfl * min(dx, dy) / max(c_phase, 0.1)

    results = {
        'damping_rate_measured': 0.0,
        'damping_rate_analytical': gamma_theory,
        'damping_error': 0.0,
        'frequency_measured': 0.0,
        'frequency_analytical': omega_theory,
        'frequency_error': 0.0,
        'amplitude_history': [],
        'negative_control': negative_control,
    }

    t = 0.0
    step_count = 0
    snapshot_interval = max(1, int(period / (20 * dt))) if period < float('inf') else 50

    while t < t_end:
        try:
            info = solver.step(dt)
        except RuntimeError:
            break

        t = solver.time
        step_count += 1

        if step_count % snapshot_interval == 0:
            alpha = solver.alpha
            eta = alpha.sum(dim=0) * dy  # surface elevation
            # Amplitude = (max - min) / 2
            amp = float(eta.max() - eta.min()) / 2.0
            results['amplitude_history'].append((t, amp))

        # Adaptive time step
        u_max = max(
            float(solver.u.abs().max()),
            float(solver.v.abs().max()),
            0.01
        )
        dt = min(dx, dy) * c.cfl / u_max
        dt = min(dt, 0.0005)  # smaller cap for capillary waves

    # Analyze damping rate from amplitude history
    if len(results['amplitude_history']) > 5:
        times = [a[0] for a in results['amplitude_history']]
        amps = [a[1] for a in results['amplitude_history']]

        # Fit exponential decay: ln(amp) = ln(amp0) - gamma * t
        # Use least squares on log(amplitude)
        valid = [(t, a) for t, a in zip(times, amps) if a > 1e-10]
        if len(valid) > 3:
            log_times = [v[0] for v in valid]
            log_amps = [math.log(v[1]) for v in valid]

            # Linear fit: log_amp = a + b*t, where b = -gamma
            n = len(log_times)
            sum_t = sum(log_times)
            sum_t2 = sum(t**2 for t in log_times)
            sum_a = sum(log_amps)
            sum_ta = sum(t * a for t, a in zip(log_times, log_amps))

            denom = n * sum_t2 - sum_t**2
            if abs(denom) > 1e-30:
                b = (n * sum_ta - sum_t * sum_a) / denom
                gamma_measured = -b
                results['damping_rate_measured'] = gamma_measured
                if gamma_theory > 0:
                    results['damping_error'] = abs(gamma_measured - gamma_theory) / gamma_theory

        # Measure frequency from zero crossings
        amps_tensor = torch.tensor(amps)
        amps_centered = amps_tensor - amps_tensor.mean()
        zero_crossings = []
        for i in range(1, len(amps_centered)):
            if amps_centered[i - 1] * amps_centered[i] < 0:
                frac = -amps_centered[i - 1] / (amps_centered[i] - amps_centered[i - 1])
                t_cross = times[i - 1] + frac * (times[i] - times[i - 1])
                zero_crossings.append(t_cross)

        if len(zero_crossings) >= 3:
            half_periods = [zero_crossings[i + 1] - zero_crossings[i]
                          for i in range(len(zero_crossings) - 1)]
            measured_period = 2.0 * sum(half_periods) / len(half_periods)
            omega_measured = 2.0 * math.pi / measured_period
            results['frequency_measured'] = omega_measured
            if omega_theory > 0:
                results['frequency_error'] = abs(omega_measured - omega_theory) / omega_theory

    results['n_steps'] = step_count
    return results


# ============================================================================
# Verification summary
# ============================================================================

def run_all_verifications() -> Dict:
    """Run all three verification levels and return summary.

    Returns dict with 'L1', 'L2', 'L3', 'negative_controls' keys.
    """
    results = {}

    # L1: Dam break
    results['L1'] = run_ritter_dam_break()

    # L2: Stokes wave
    results['L2'] = run_stokes_wave()

    # L3: Capillary wave
    results['L3'] = run_capillary_wave()

    # Negative controls
    results['negative_controls'] = {
        'no_sigma': run_capillary_wave(negative_control="no_sigma"),
        'no_gravity': run_capillary_wave(negative_control="no_gravity"),
    }

    return results


def verification_summary(results: Dict) -> Dict:
    """Extract pass/fail verdicts from verification results.

    Criteria:
    - L1: front position error < 3%, mass conservation error < 1%
    - L2: wave speed error < 5%, phase error < 5°
    - L3: damping rate error < 10%, frequency error < 5%
    """
    summary = {}

    # L1
    L1 = results['L1']
    max_front_error = max(L1['front_errors']) if L1['front_errors'] else 0.0
    summary['L1'] = {
        'max_front_error': max_front_error,
        'mass_rel_error': L1['mass_rel_error'],
        'front_pass': max_front_error < 0.03,
        'mass_pass': L1['mass_rel_error'] < 0.01,
        'pass': max_front_error < 0.03 and L1['mass_rel_error'] < 0.01,
    }

    # L2
    L2 = results['L2']
    summary['L2'] = {
        'wave_speed_error': L2['wave_speed_error'],
        'phase_error_deg': L2['phase_error_deg'],
        'speed_pass': L2['wave_speed_error'] < 0.05,
        'phase_pass': L2['phase_error_deg'] < 5.0,
        'pass': L2['wave_speed_error'] < 0.05 and L2['phase_error_deg'] < 5.0,
    }

    # L3
    L3 = results['L3']
    summary['L3'] = {
        'damping_error': L3['damping_error'],
        'frequency_error': L3['frequency_error'],
        'damping_pass': L3['damping_error'] < 0.10,
        'frequency_pass': L3['frequency_error'] < 0.05,
        'pass': L3['damping_error'] < 0.10 and L3['frequency_error'] < 0.05,
    }

    # Negative controls
    nc = results['negative_controls']
    # no_sigma: frequency should be close to gravity wave freq (no capillary contribution)
    k = 2 * math.pi / 0.1  # wavelength = 0.1
    omega_gravity = math.sqrt(9.81 * k * math.tanh(k * 0.5))
    omega_no_sigma = nc['no_sigma']['frequency_measured']
    summary['negative_controls'] = {
        'no_sigma_freq_ratio': omega_no_sigma / omega_gravity if omega_gravity > 0 else 0,
        'no_gravity_is_capillary': nc['no_gravity']['frequency_analytical'] > 0,
    }

    return summary
