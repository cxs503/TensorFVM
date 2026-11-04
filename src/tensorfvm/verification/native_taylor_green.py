"""Taylor--Green vortex: exact nonlinear Navier--Stokes reference for the native hybrid backend.

The periodic MAC prototype (:mod:`tensorfvm.periodic_mac`) is qualified against
the exact Taylor--Green vortex in :mod:`tensorfvm.verification.taylor_green_mac`.
This module provides the *same* analytic reference for the native hybrid
Navier--Stokes solver (:class:`tensorfvm.native_hybrid_ns3d.NativeHybridNavierStokes3D`),
closing a verification gap: the native backend had polynomial, Kovasznay and
potential-flow manufactured cases but no unforced nonlinear periodic benchmark.

The exact solution on the doubly-periodic box ``[0, 2*pi]^3`` is

    u(x,y,z,t) =  A cos(x) sin(y) exp(-2 nu t)
    v(x,y,z,t) = -A sin(x) cos(y) exp(-2 nu t)
    w          = 0
    p(x,y,z,t) = -(A^2/4) (cos 2x + cos 2y) exp(-4 nu t)   (rho = 1)

which satisfies ``du/dt + (u.grad)u = -grad(p) + nu lap(u)`` identically.
All three velocity components are non-trivial in x and y; w is identically
zero.  The pressure depends only on x and y.

The solver uses a unit-cube mesh with periodic boundary conditions on all
faces.  The analytic velocity is prescribed at boundary faces via
:func:`exact_boundary_velocity_means`; the body force is zero.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..native_hybrid_ns3d import NativeHybridNavierStokes3D
from ..native_manufactured3d import exact_boundary_velocity_means, volume_weighted_error
from .native_hybrid_flow import channel_mesh


# --------------------------------------------------------------------------- #
# Closed-form reference
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TaylorGreenCase:
    """Physical parameters of the exact Taylor--Green vortex.

    ``viscosity`` alone sets the decay rate; ``amplitude`` sets both the
    convective and the pressure scale.  Density is fixed to 1 because the
    analytic pressure above is written for ``rho = 1``.
    """
    amplitude: float = 1.0
    viscosity: float = 0.05
    density: float = 1.0

    def __post_init__(self) -> None:
        for name in ("amplitude", "viscosity", "density"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"invalid {name}: {value!r}")
        if self.density != 1.0:
            raise ValueError(
                "the closed-form pressure of this reference is written for rho = 1; "
                "generalise the case before using another density"
            )

    @property
    def decay_exponent(self) -> float:
        """Velocity decay rate: 2 * nu."""
        return 2.0 * self.viscosity

    @property
    def energy_decay_exponent(self) -> float:
        """Kinetic-energy decay rate: 4 * nu."""
        return 4.0 * self.viscosity


def tgv_fields(points: np.ndarray, case: TaylorGreenCase) -> Dict:
    """Exact velocity, pressure, gradient and force at arbitrary points.

    Parameters
    ----------
    points :
        ``(N, 3)`` array of Cartesian coordinates.
    case :
        Physical parameters.

    Returns
    -------
    dict with keys ``velocity``, ``pressure``, ``velocity_gradient``, ``force``.
    """
    x, y, z = np.asarray(points, dtype=float).T
    amp = case.amplitude
    nu = case.viscosity
    decay = math.exp(-case.decay_exponent * 0.0)  # t=0 for field definition
    u = amp * np.cos(x) * np.sin(y) * decay
    v = -amp * np.sin(x) * np.cos(y) * decay
    w = np.zeros_like(x)
    velocity = np.column_stack((u, v, w))

    # Pressure: p = -(A^2/4)(cos 2x + cos 2y) exp(-4 nu t)
    p = -0.25 * amp**2 * (np.cos(2 * x) + np.cos(2 * y)) * decay

    # Velocity gradient
    grad = np.zeros((len(x), 3, 3))
    grad[:, 0, 0] = -amp * np.sin(x) * np.sin(y) * decay
    grad[:, 0, 1] = amp * np.cos(x) * np.cos(y) * decay
    grad[:, 1, 0] = -amp * np.cos(x) * np.cos(y) * decay
    grad[:, 1, 1] = amp * np.sin(x) * np.sin(y) * decay

    # Force = 0 (unforced solution)
    force = np.zeros_like(velocity)

    return dict(velocity=velocity, pressure=p, velocity_gradient=grad, force=force)


def tgv_fields_at_time(points: np.ndarray, case: TaylorGreenCase,
                       time_s: float) -> Dict:
    """Exact fields at a specific time."""
    x, y, z = np.asarray(points, dtype=float).T
    amp = case.amplitude
    nu = case.viscosity
    decay_u = math.exp(-case.decay_exponent * time_s)
    decay_p = math.exp(-case.energy_decay_exponent * time_s)

    u = amp * np.cos(x) * np.sin(y) * decay_u
    v = -amp * np.sin(x) * np.cos(y) * decay_u
    w = np.zeros_like(x)
    velocity = np.column_stack((u, v, w))

    p = -0.25 * amp**2 * (np.cos(2 * x) + np.cos(2 * y)) * decay_p

    grad = np.zeros((len(x), 3, 3))
    grad[:, 0, 0] = -amp * np.sin(x) * np.sin(y) * decay_u
    grad[:, 0, 1] = amp * np.cos(x) * np.cos(y) * decay_u
    grad[:, 1, 0] = -amp * np.cos(x) * np.cos(y) * decay_u
    grad[:, 1, 1] = amp * np.sin(x) * np.sin(y) * decay_u

    force = np.zeros_like(velocity)

    return dict(velocity=velocity, pressure=p, velocity_gradient=grad, force=force)


# --------------------------------------------------------------------------- #
# Solver construction
# --------------------------------------------------------------------------- #
def tgv_solver(n: int, case: Optional[TaylorGreenCase] = None,
               convection_order: int = 2,
               coupled_linear_method: str = 'direct') -> tuple:
    """Build a native hybrid NS solver for the Taylor--Green vortex.

    Parameters
    ----------
    n :
        Number of cells per direction (uniform cubic mesh).
    case :
        Physical parameters; defaults to the canonical case.
    convection_order :
        Convection discretisation order (1 or 2).
    coupled_linear_method :
        Linear solver for the coupled Stokes/NS system.

    Returns
    -------
    (mesh, solver, exact) where exact is the dict from ``tgv_fields``.
    """
    if case is None:
        case = TaylorGreenCase()

    mesh = channel_mesh(n)
    exact = tgv_fields(mesh.cell_centers, case)
    boundary = exact_boundary_velocity_means(mesh,
                                             lambda pts: tgv_fields(pts, case))
    fixed = np.repeat((mesh.neighbor < 0)[:, None], 3, axis=1)

    solver = NativeHybridNavierStokes3D(
        mesh, fixed, boundary,
        viscosity=case.viscosity,
        body_force=exact['force'],
        gauge_value=exact['pressure'][0],
        convection_order=convection_order,
        coupled_linear_method=coupled_linear_method,
    )
    return mesh, solver, exact


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def tgv_metrics(mesh, result, exact, case: TaylorGreenCase,
                kinetic_energy_init: float) -> Dict:
    """Compute verification metrics for a converged TGV solve.

    Parameters
    ----------
    mesh :
        The computational mesh.
    result :
        Solver result dict (from ``solver.solve()``).
    exact :
        Exact field dict from ``tgv_fields``.
    case :
        Physical parameters.
    kinetic_energy_init :
        Initial kinetic energy (for decay-rate computation).

    Returns
    -------
    dict of metric name -> value.
    """
    volume = mesh.cell_volumes

    # Velocity error
    vel_err = volume_weighted_error(result['velocity'], exact['velocity'], volume)

    # Pressure error (relative to max |p|)
    p_exact = exact['pressure']
    p_computed = result['pressure']
    p_max = np.max(np.abs(p_exact))
    if p_max > 0:
        p_err = np.max(np.abs(p_computed - p_exact)) / p_max
    else:
        p_err = np.max(np.abs(p_computed))

    # Energy decay rate
    ke_final = result.get('kinetic_energy_final_J',
                                0.5 * case.density * np.sum(volume * np.sum(result['velocity']**2, axis=1)))
    if kinetic_energy_init > 0:
        decay_rate = -math.log(ke_final / kinetic_energy_init) / 1.0  # t=1.0
        analytic_decay = case.energy_decay_exponent
        decay_rate_err = abs(decay_rate - analytic_decay) / analytic_decay
    else:
        decay_rate_err = float('inf')

    # Discrete divergence
    max_div = max(h['max_divergence_s_inv'] for h in result.get('history', [{}]))

    return dict(
        velocity_relative_l2=float(vel_err),
        pressure_relative_linf=float(p_err),
        energy_decay_rate_relative_error=float(decay_rate_err),
        max_divergence_s_inv=float(max_div),
        velocity_z_max_abs_m_s=float(np.max(np.abs(result['velocity'][:, 2]))),
    )


def run_study(n_list: List[int], case: Optional[TaylorGreenCase] = None,
              time_s: float = 1.0, dt: float = 0.02) -> List[Dict]:
    """Run the TGV spatial refinement study.

    Parameters
    ----------
    n_list :
        List of grid resolutions to study.
    case :
        Physical parameters.
    time_s :
        Integration time.
    dt :
        Time step.

    Returns
    -------
    List of metric dicts, one per resolution.
    """
    if case is None:
        case = TaylorGreenCase()

    results = []
    for n in n_list:
        mesh, solver, exact = tgv_solver(n, case)
        # Initialize with analytic velocity
        solver.velocity = np.zeros((3, *solver.shape))
        for a in range(3):
            solver.velocity[a] = exact['velocity'][:, a].reshape(solver.shape)

        # Compute initial kinetic energy
        ke_init = 0.5 * case.density * np.sum(
            mesh.cell_volumes * np.sum(solver.velocity**2, axis=0)
        )

        # Time march
        steps = int(time_s / dt)
        for _ in range(steps):
            solver.step()

        # Compute metrics
        metrics = tgv_metrics(mesh, solver.solve(), exact, case, ke_init)
        metrics['resolution'] = f'{n}x{n}x{n}'
        metrics['spacing_m'] = float(2 * math.pi / n)
        metrics['time_s'] = time_s
        results.append(metrics)

    return results
