"""Taylor--Green vortex decay: an *exact nonlinear* Navier--Stokes reference.

The periodic MAC prototype (:mod:`tensorfvm.periodic_mac`) is qualified in
``docs/periodic-mac/study.json`` only against manufactured solutions, whose
source terms are chosen *after* the discretisation.  That proves the code solves
the discrete equations it claims to solve, but it cannot expose a physical
modelling error: any consistent-but-wrong momentum flux would be absorbed into
the manufactured forcing.  The study therefore records
``physical_accuracy_qualified: false``.

This module closes that gap with the single-mode Taylor--Green vortex, which is
an exact solution of the *unforced, incompressible, nonlinear* Navier--Stokes
equations in a fully periodic box::

    u(x,y,t) =  A cos(x) sin(y) exp(-2 nu t)
    v(x,y,t) = -A sin(x) cos(y) exp(-2 nu t)
    w        = 0
    p(x,y,t) = -(A^2/4) (cos 2x + cos 2y) exp(-4 nu t)          (rho = 1)

Nothing here is fitted: the reference decays at the rate set by viscosity alone,
and the pressure field is fixed by the nonlinear advection balance.  Two
properties make it unusually sharp as a verification case for a *staggered*
kernel:

* **Exactly discrete-divergence-free.**  Sampling ``u`` on x-faces and ``v`` on
  y-faces gives a backward-difference divergence
  ``A sin x sin y (2 sin(dy/2)/dy - 2 sin(dx/2)/dx)``, which is *identically
  zero* when ``dx == dy`` -- to machine precision, not to truncation order.  The
  initial condition therefore needs no projection, so a nonzero initial
  divergence indicts the sampling, not the solver.
* **Kinetic energy is a clean scalar functional.**  ``E(t) = rho Lz A^2 pi^2
  exp(-4 nu t)``; the logarithmic decay rate ``-d ln E / dt = 4 nu`` is a
  single number the scheme can get wrong while still being "stable".

Conventions follow the solver: ``velocity[a][k,j,i]`` is the ``+a`` face of cell
``(k,j,i)``, cell centres sit at ``(i+1/2) dx`` and the face stored in ``[i]``
lies at ``(i+1) dx``.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from ..periodic_mac import PeriodicMACConfig, PeriodicMACSolver

__all__ = [
    "TaylorGreenCase",
    "exact_fields",
    "initialise",
    "relative_l2",
    "energy_decay_rate_error",
    "run_to",
    "compare",
]


@dataclass(frozen=True)
class TaylorGreenCase:
    """Physical parameters of the exact vortex.

    ``viscosity`` alone sets the decay; ``amplitude`` sets both the convective
    and the pressure scale.  Density is fixed to 1 because the analytic pressure
    above is written for ``rho = 1``.
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
        """``2 nu``: amplitude decay rate of the velocity field."""
        return 2 * self.viscosity

    @property
    def energy_decay_exponent(self) -> float:
        """``4 nu``: decay rate of the kinetic energy (amplitude squared)."""
        return 4 * self.viscosity


def _axis_lengths(solver: PeriodicMACSolver, axis: int) -> tuple[int, float]:
    """Map a *physical* axis (0=x, 1=y, 2=z) to (count, spacing).

    The solver stores fields as ``(z, y, x)`` but declares ``spacing`` in
    ``(x, y, z)`` order, so the two indices run opposite ways.
    """
    return solver.shape[2 - axis], solver.spacing[axis]


def _centres(solver: PeriodicMACSolver, axis: int) -> torch.Tensor:
    n, d = _axis_lengths(solver, axis)
    return (torch.arange(n, dtype=torch.float64, device=solver.device) + 0.5) * d


def _faces(solver: PeriodicMACSolver, axis: int) -> torch.Tensor:
    n, d = _axis_lengths(solver, axis)
    return (torch.arange(n, dtype=torch.float64, device=solver.device) + 1.0) * d


def exact_fields(solver: PeriodicMACSolver, case: TaylorGreenCase, time: float) -> dict:
    """Exact face velocities and cell pressure of the vortex at ``time``.

    Returns the fields laid out exactly like ``solver.velocity`` / ``solver.pressure``
    so that numeric and analytic arrays can be compared element by element.
    """
    xc, yc, zc = _centres(solver, 0), _centres(solver, 1), _centres(solver, 2)
    xf, yf, zf = _faces(solver, 0), _faces(solver, 1), _faces(solver, 2)
    amplitude = case.amplitude
    velocity_decay = math.exp(-case.decay_exponent * time)
    pressure_decay = math.exp(-case.energy_decay_exponent * time)

    shape = solver.shape
    velocity = torch.zeros((3, *shape), dtype=torch.float64, device=solver.device)
    # u lives on x-faces: x from xf, y/z from centres.
    _, yu, xu = torch.meshgrid(zc, yc, xf, indexing="ij")
    velocity[0] = amplitude * torch.cos(xu) * torch.sin(yu) * velocity_decay
    # v lives on y-faces.
    _, yv, xv = torch.meshgrid(zc, yf, xc, indexing="ij")
    velocity[1] = -amplitude * torch.sin(xv) * torch.cos(yv) * velocity_decay
    # w is identically zero for this z-invariant solution (z-faces left at zero).
    _, yp, xp = torch.meshgrid(zc, yc, xc, indexing="ij")
    pressure = -amplitude**2 / 4 * (torch.cos(2 * xp) + torch.cos(2 * yp)) * pressure_decay
    energy = (
        case.density * solver.config.lengths_m[2] * amplitude**2 * math.pi**2 * pressure_decay
    )
    return {
        "time_s": time,
        "velocity": velocity,
        "pressure": pressure,
        "kinetic_energy_J": energy,
    }


def initialise(solver: PeriodicMACSolver, case: TaylorGreenCase = TaylorGreenCase(),
               time: float = 0.0) -> dict:
    """Load the exact solution into ``solver`` and report the discrete witnesses.

    No projection is applied: the analytic samples are expected to be divergence
    free *discretely* when ``dx == dy``, and that identity is the first gate.
    """
    fields = exact_fields(solver, case, time)
    solver.velocity = fields["velocity"].clone()
    solver.pressure = fields["pressure"].clone()
    solver.time = time
    solver.history = []
    dx, dy = solver.spacing[0], solver.spacing[1]
    sampled_divergence = float(solver.divergence(solver.velocity).abs().max())
    return {
        "time_s": time,
        "sampled_max_divergence_s_inv": sampled_divergence,
        "spacing_equal_xy": bool(abs(dx - dy) <= 1e-15 * max(dx, dy)),
        "kinetic_energy_J": float(solver.kinetic_energy(solver.velocity)),
        "exact_kinetic_energy_J": fields["kinetic_energy_J"],
        "energy_sampling_relative_error": abs(
            float(solver.kinetic_energy(solver.velocity)) - fields["kinetic_energy_J"]
        ) / fields["kinetic_energy_J"],
        "max_velocity_m_s": float(solver.velocity.abs().max()),
    }


def relative_l2(numeric: torch.Tensor, reference: torch.Tensor) -> float:
    """``||numeric - reference||_2 / ||reference||_2`` with the reference norm.

    Raises for an identically zero reference: a relative error against nothing is
    not a weak statement, it is a meaningless one.
    """
    numerator = float(torch.linalg.vector_norm(numeric - reference))
    denominator = float(torch.linalg.vector_norm(reference))
    if denominator == 0.0:
        raise ValueError(
            "reference field is identically zero; compare an absolute norm instead"
        )
    return numerator / denominator


def energy_decay_rate_error(solver: PeriodicMACSolver, case: TaylorGreenCase,
                            initial_energy_j: float) -> dict:
    """Compare the measured kinetic-energy decay with ``exp(-4 nu t)``.

    Using the ratio ``E(t)/E(0)`` cancels the O(h^2) sampling offset of the
    discrete energy functional, leaving the *rate* -- the physical content --
    as the thing being tested.
    """
    energy = float(solver.kinetic_energy(solver.velocity))
    ratio = energy / initial_energy_j
    measured = -math.log(ratio) / solver.time if solver.time > 0 else float("nan")
    expected = case.energy_decay_exponent
    return {
        "time_s": solver.time,
        "energy_J": energy,
        "energy_ratio": ratio,
        "exact_energy_ratio": math.exp(-expected * solver.time),
        "relative_error": abs(ratio - math.exp(-expected * solver.time))
        / math.exp(-expected * solver.time),
        "measured_decay_rate_1_s": measured,
        "exact_decay_rate_1_s": expected,
        "decay_rate_relative_error": abs(measured - expected) / expected
        if solver.time > 0
        else float("nan"),
    }


def run_to(solver: PeriodicMACSolver, final_time: float, max_steps: int = 20000) -> int:
    """Advance to ``final_time`` with whole steps of the configured ``dt``."""
    dt = solver.config.time_step_s
    steps = int(round((final_time - solver.time) / dt))
    if steps < 0 or abs(solver.time + steps * dt - final_time) > 1e-12 * max(1.0, final_time):
        raise ValueError("final_time must be an integer multiple of the solver time step")
    for _ in range(steps):
        solver.step()
    return steps


def compare(solver: PeriodicMACSolver, case: TaylorGreenCase, initial_energy_j: float) -> dict:
    """Full analytic comparison of the current state against the exact solution."""
    exact = exact_fields(solver, case, solver.time)
    pressure_error = relative_l2(solver.pressure - solver.pressure.mean(),
                                 exact["pressure"] - exact["pressure"].mean())
    velocity_error = relative_l2(solver.velocity, exact["velocity"])
    # u, v are compared relatively; w is identically zero in this reference, so
    # the only honest statement about it is an absolute norm.
    per_component = {
        "velocity_x_relative_l2": relative_l2(solver.velocity[0], exact["velocity"][0]),
        "velocity_y_relative_l2": relative_l2(solver.velocity[1], exact["velocity"][1]),
        "velocity_z_max_abs_m_s": float(solver.velocity[2].abs().max()),
    }
    return {
        "grid": {"nx": solver.config.nx, "ny": solver.config.ny, "nz": solver.config.nz},
        "dx_m": solver.spacing[0],
        "dy_m": solver.spacing[1],
        "time_step_s": solver.config.time_step_s,
        "time_s": solver.time,
        "velocity_relative_l2": velocity_error,
        "pressure_relative_l2": pressure_error,
        "max_divergence_s_inv": float(solver.divergence(solver.velocity).abs().max()),
        "energy": energy_decay_rate_error(solver, case, initial_energy_j),
        **per_component,
    }