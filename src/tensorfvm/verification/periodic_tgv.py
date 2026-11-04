"""Physical-accuracy qualification of the periodic MAC kernel against the exact
two-dimensional Taylor-Green vortex.

Why this module exists
----------------------
``tensorfvm.periodic_mac`` is a single-rank, staggered, dual control-volume
momentum/energy prototype.  Its published evidence (``docs/periodic-mac/study.json``)
qualifies *refinement order* only: the Taylor-Green case there rescales the
initial face velocity by ``exp(-2 nu t)`` and compares that with the final
field.  The pressure field, the kinetic energy and the dissipation rate are
never compared with the closed-form solution, which is why that study honestly
reports ``physical_accuracy_qualified = False``.

This module supplies the missing physical reference: the unforced, unsteady,
genuinely nonlinear Navier-Stokes solution

    u(x, y, t) =   sin(x) cos(y) exp(-2 nu t)
    v(x, y, t) = - cos(x) sin(y) exp(-2 nu t)
    w          = 0
    p(x, y, t) = rho / 4 * (cos 2x + cos 2y) exp(-4 nu t)

which satisfies ``du/dt + (u.grad)u = -grad(p)/rho + nu lap(u)`` identically on
the doubly periodic box ``[0, 2 pi]^2`` extended uniformly in ``z``.  Its exact
budgets are

    E(t)   = rho V / 4 exp(-4 nu t),        total kinetic energy
    eps(t) = -dE/dt = 4 nu E(t),            viscous dissipation rate
    omega_z(x, y, t) = 2 sin(x) sin(y) exp(-2 nu t),

and the convective power ``int u . ((u.grad) u) dV`` vanishes exactly, so a
consistent dual control-volume discretisation must report zero work for its
convection operator while the viscous power must equal ``-eps``.

Every reference below is evaluated *at the discretisation's own sample points*
(staggered faces for velocity, cell centres for pressure and vorticity), so the
reported errors isolate scheme error from the O(h^2) representation error of a
smooth field.  The continuum (analytic) energy is reported separately so the
representation floor stays visible instead of being hidden inside the gate.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import torch

from ..periodic_mac import PeriodicMACConfig, PeriodicMACSolver

__all__ = [
    'TaylorGreenReference',
    'exact_face_velocity',
    'exact_pressure',
    'exact_cell_vorticity',
    'analytic_kinetic_energy',
    'analytic_dissipation_rate',
    'cell_vorticity',
    'relative_l2',
    'initialise',
    'run_case',
    'run_study',
]


# --------------------------------------------------------------------------- #
# Closed-form reference
# --------------------------------------------------------------------------- #
class TaylorGreenReference:
    """Closed-form 2-D Taylor-Green reference bound to one solver layout."""

    def __init__(self, solver: PeriodicMACSolver):
        self.solver = solver
        self.density = float(solver.config.density_kg_m3)
        self.viscosity = float(solver.config.viscosity_pa_s)
        # ``solver.volume`` is one *cell* volume; the energy reference needs the box.
        self.volume = float(solver.volume) * math.prod(solver.shape)

    # -- fields ------------------------------------------------------------ #
    def face_velocity(self, time_s: float) -> torch.Tensor:
        """Exact velocity sampled on the staggered faces of every component."""
        decay = math.exp(-2.0 * self.viscosity * time_s)
        out = torch.zeros_like(self.solver.velocity)
        for axis in (0, 1):
            x, y, _ = self.solver.face_coordinates(axis)
            shape = (torch.sin(x) * torch.cos(y) if axis == 0
                     else -torch.cos(x) * torch.sin(y))
            out[axis] = decay * shape
        return out

    def pressure(self, time_s: float) -> torch.Tensor:
        """Exact pressure at cell centres (zero mean, matching the projection)."""
        x, y, _ = self.solver.centers()
        return (0.25 * self.density * (torch.cos(2 * x) + torch.cos(2 * y))
                * math.exp(-4.0 * self.viscosity * time_s))

    def cell_vorticity(self, time_s: float) -> torch.Tensor:
        """Exact ``omega_z`` at cell centres."""
        x, y, _ = self.solver.centers()
        return (2.0 * torch.sin(x) * torch.sin(y)
                * math.exp(-2.0 * self.viscosity * time_s))

    # -- scalars ----------------------------------------------------------- #
    def analytic_kinetic_energy(self, time_s: float) -> float:
        """``rho V / 4 exp(-4 nu t)`` for the exact field."""
        return (0.25 * self.density * self.volume
                * math.exp(-4.0 * self.viscosity * time_s))

    def discrete_kinetic_energy(self, time_s: float) -> float:
        """Kinetic energy of the *sampled* exact field: the fair discrete target."""
        return float(self.solver.kinetic_energy(self.face_velocity(time_s)))

    def analytic_dissipation_rate(self, time_s: float) -> float:
        """``4 nu E(t)`` for the exact field."""
        return 4.0 * self.viscosity * self.analytic_kinetic_energy(time_s)

    def discrete_dissipation_rate(self, time_s: float) -> float:
        """``4 nu E(t)`` built on the sampled exact field."""
        return 4.0 * self.viscosity * self.discrete_kinetic_energy(time_s)


def exact_face_velocity(solver, time_s):
    return TaylorGreenReference(solver).face_velocity(time_s)


def exact_pressure(solver, time_s):
    return TaylorGreenReference(solver).pressure(time_s)


def exact_cell_vorticity(solver, time_s):
    return TaylorGreenReference(solver).cell_vorticity(time_s)


def analytic_kinetic_energy(solver, time_s):
    return TaylorGreenReference(solver).analytic_kinetic_energy(time_s)


def analytic_dissipation_rate(solver, time_s):
    return TaylorGreenReference(solver).analytic_dissipation_rate(time_s)


def cell_vorticity(solver, velocity):
    """Staggered curl ``omega_z`` at cell centres of the solver grid."""
    u, v = velocity[0], velocity[1]
    return ((v - solver.previous(v, 0)) / solver.spacing[0]
            - (u - solver.previous(u, 1)) / solver.spacing[1])


def relative_l2(numeric, exact) -> float:
    return float(torch.linalg.vector_norm(numeric - exact)
                 / torch.linalg.vector_norm(exact))


def initialise(solver) -> torch.Tensor:
    """Set the exact t=0 field as the initial condition."""
    solver.velocity = TaylorGreenReference(solver).face_velocity(0.0)
    return solver.velocity.clone()


# --------------------------------------------------------------------------- #
# Case driver
# --------------------------------------------------------------------------- #
@dataclass
class CaseResult:
    name: str
    n: int
    time_step_s: float
    steps: int
    viscosity_pa_s: float
    time_end_s: float
    wall_time_s: float = 0.0
    # field errors at the final time (relative L2 vs exact at own sample points)
    velocity_error: float = math.nan
    pressure_error: float = math.nan
    vorticity_error: float = math.nan
    # energy / dissipation closure
    energy_error: float = math.nan          # vs sampled exact field
    energy_error_continuum: float = math.nan  # vs rho V / 4 exp(-4 nu t)
    initial_energy_error: float = math.nan  # representation floor at t = 0
    dissipation_error: float = math.nan     # finite-difference dE/dt vs 4 nu E
    viscous_power_error: float = math.nan   # solver viscous work vs -4 nu E
    max_abs_convective_power_W: float = math.nan
    max_abs_divergence_s_inv: float = math.nan
    power_scale_W: float = math.nan
    nonlinear_iterations_max: int = 0
    nonlinear_all_converged: bool = True
    trace: list = field(default_factory=list)
    fields: dict = field(default_factory=dict)

    def summary(self) -> dict:
        out = {k: v for k, v in self.__dict__.items() if k not in ('trace', 'fields')}
        return out


def _make_solver(n, viscosity_pa_s, time_step_s, length_m, density_kg_m3):
    config = PeriodicMACConfig(nx=n, ny=n, nz=4,
                               lengths_m=(length_m, length_m, length_m),
                               density_kg_m3=density_kg_m3,
                               viscosity_pa_s=viscosity_pa_s,
                               time_step_s=time_step_s)
    return PeriodicMACSolver(config)


def run_case(n: int, *, viscosity_pa_s: float = .05, time_end_s: float = 1.0,
             steps_per_unit_time: int = 100, length_m: float = 2 * math.pi,
             density_kg_m3: float = 1.0, name: str | None = None) -> CaseResult:
    """Advance the exact Taylor-Green vortex and score it against closed form.

    ``steps_per_unit_time`` fixes ``dt``; refining the grid while scaling it with
    ``n`` keeps ``dt = O(h)`` so that spatial and temporal truncation errors are
    both second order and a single observed order is meaningful.
    """
    time_step_s = 1.0 / float(steps_per_unit_time)
    steps = int(round(time_end_s * steps_per_unit_time))
    solver = _make_solver(n, viscosity_pa_s, time_step_s, length_m, density_kg_m3)
    ref = TaylorGreenReference(solver)
    solver.velocity = ref.face_velocity(0.0)
    result = CaseResult(name=name or f'tgv-n{n}', n=n, time_step_s=time_step_s,
                        steps=steps, viscosity_pa_s=viscosity_pa_s,
                        time_end_s=time_end_s)
    result.power_scale_W = ref.analytic_dissipation_rate(0.0)
    result.initial_energy_error = float(abs(
        solver.kinetic_energy(solver.velocity) - ref.analytic_kinetic_energy(0.0)
    ) / ref.analytic_kinetic_energy(0.0))

    started = time.perf_counter()
    for step in range(steps):
        record = solver.step()
        t_mid = record['time_s'] - 0.5 * time_step_s
        energy_numeric = record['kinetic_after_J']
        result.trace.append(dict(
            step=record['step'], time_s=record['time_s'],
            energy_numeric_J=energy_numeric,
            energy_exact_discrete_J=ref.discrete_kinetic_energy(record['time_s']),
            energy_exact_continuum_J=ref.analytic_kinetic_energy(record['time_s']),
            dissipation_numeric_W=-(record['kinetic_after_J'] - record['kinetic_before_J']) / time_step_s,
            dissipation_exact_W=ref.analytic_dissipation_rate(t_mid),
            viscous_power_numeric_W=record['viscous_power_W'],
            convective_power_numeric_W=record['convective_power_W'],
        ))
        result.nonlinear_iterations_max = max(result.nonlinear_iterations_max,
                                             record['nonlinear_iterations'])
        result.nonlinear_all_converged = (result.nonlinear_all_converged
                                          and record['nonlinear_converged'])
    result.wall_time_s = time.perf_counter() - started

    exact_velocity = ref.face_velocity(solver.time)
    exact_pressure = ref.pressure(solver.time)
    exact_vorticity = ref.cell_vorticity(solver.time)
    numeric_vorticity = cell_vorticity(solver, solver.velocity)
    result.velocity_error = relative_l2(solver.velocity, exact_velocity)
    result.pressure_error = relative_l2(solver.pressure, exact_pressure)
    result.vorticity_error = relative_l2(numeric_vorticity, exact_vorticity)

    energy_numeric = float(solver.kinetic_energy(solver.velocity))
    result.energy_error = abs(energy_numeric - ref.discrete_kinetic_energy(solver.time)
                              ) / ref.discrete_kinetic_energy(solver.time)
    result.energy_error_continuum = abs(
        energy_numeric - ref.analytic_kinetic_energy(solver.time)
    ) / ref.analytic_kinetic_energy(solver.time)

    last = result.trace[-1]
    result.dissipation_error = float(abs(last['dissipation_numeric_W']
                                         - last['dissipation_exact_W']) / last['dissipation_exact_W'])
    result.viscous_power_error = float(abs(-last['viscous_power_numeric_W']
                                           - last['dissipation_exact_W']) / last['dissipation_exact_W'])
    result.max_abs_convective_power_W = float(max(abs(x['convective_power_numeric_W'])
                                                  for x in result.trace))
    result.max_abs_divergence_s_inv = float(solver.divergence(solver.velocity).abs().max())

    x, y, _ = solver.centers()
    result.fields = dict(
        centres_x=x.tolist(), centres_y=y.tolist(),
        pressure_numeric=solver.pressure.tolist(), pressure_exact=exact_pressure.tolist(),
        vorticity_numeric=numeric_vorticity.tolist(), vorticity_exact=exact_vorticity.tolist(),
        velocity_error=relative_l2(solver.velocity, exact_velocity),
    )
    return result


def _order(coarse: float, fine: float, ratio: float) -> float:
    """Observed order from two grids whose spacing and time step both scale by ``ratio``."""
    return math.log(coarse / fine) / math.log(ratio)


def run_study(grids=(16, 32, 64), *, viscosity_pa_s: float = .05,
              time_end_s: float = 1.0, base_steps_per_unit_time: int = 100,
              reference_n: int = 32,
              limit_viscosity_pa_s: float = .002,
              limit_steps_per_unit_time: int = 200,
              limit_time_end_s: float = 1.0,
              tolerance: float = .03) -> dict:
    """Run the refinement matrix plus a low-viscosity honest limit case.

    ``limit_viscosity_pa_s`` probes the regime where centred convection aliases
    faster than viscosity can damp it.  That case is *expected to fail* the 3 %
    gate; recording the failure is what keeps the qualification discriminating
    instead of vacuous.
    """
    cases = []
    for n in grids:
        steps_per_unit_time = int(round(base_steps_per_unit_time * n / reference_n))
        case = run_case(n, viscosity_pa_s=viscosity_pa_s, time_end_s=time_end_s,
                        steps_per_unit_time=steps_per_unit_time)
        cases.append(case)

    def passed(value):
        return bool(math.isfinite(value) and value < tolerance)

    refinement = []
    for case in cases:
        refinement.append(dict(
            name=case.name, n=case.n, time_step_s=case.time_step_s, steps=case.steps,
            velocity_error=case.velocity_error, pressure_error=case.pressure_error,
            vorticity_error=case.vorticity_error, energy_error=case.energy_error,
            energy_error_continuum=case.energy_error_continuum,
            initial_energy_error=case.initial_energy_error,
            dissipation_error=case.dissipation_error,
            viscous_power_error=case.viscous_power_error,
            max_abs_convective_power_W=case.max_abs_convective_power_W,
            max_abs_divergence_s_inv=case.max_abs_divergence_s_inv,
            power_scale_W=case.power_scale_W,
            nonlinear_iterations_max=case.nonlinear_iterations_max,
            nonlinear_all_converged=case.nonlinear_all_converged,
            wall_time_s=case.wall_time_s,
            velocity_3_percent_passed=passed(case.velocity_error),
            pressure_3_percent_passed=passed(case.pressure_error),
            vorticity_3_percent_passed=passed(case.vorticity_error),
            energy_3_percent_passed=passed(case.energy_error),
            dissipation_3_percent_passed=passed(case.dissipation_error),
            viscous_power_3_percent_passed=passed(case.viscous_power_error),
        ))

    order = {}
    if len(cases) > 1:
        order = dict(
            velocity=[_order(a.velocity_error, b.velocity_error, b.n / a.n)
                      for a, b in zip(cases, cases[1:])],
            pressure=[_order(a.pressure_error, b.pressure_error, b.n / a.n)
                      for a, b in zip(cases, cases[1:])],
            vorticity=[_order(a.vorticity_error, b.vorticity_error, b.n / a.n)
                       for a, b in zip(cases, cases[1:])],
        )
    order_passed = bool(order) and all(1.8 < o < 2.2 for key in ('velocity', 'pressure')
                                       for o in order[key])

    limit_failure = None
    try:
        limit = run_case(min(grids), name='limit-high-re', viscosity_pa_s=limit_viscosity_pa_s,
                         time_end_s=limit_time_end_s,
                         steps_per_unit_time=limit_steps_per_unit_time)
    except RuntimeError as exc:  # nonlinear midpoint solve rejected the step
        limit_failure = f'{type(exc).__name__}: {exc}'
        limit = CaseResult(name='limit-high-re', n=min(grids),
                           time_step_s=1.0 / limit_steps_per_unit_time, steps=0,
                           viscosity_pa_s=limit_viscosity_pa_s,
                           time_end_s=limit_time_end_s)
    # The negative control must fail on at least one physical quantity to prove the gate
    # is discriminating. Velocity alone is nearly exact on the staggered grid, so we
    # check pressure, vorticity, and energy as well.
    limit_all_passed = (passed(limit.velocity_error) and passed(limit.pressure_error)
                        and passed(limit.vorticity_error) and passed(limit.energy_error)
                        and limit_failure is None)
    limit_passed = limit_all_passed

    finest = cases[-1]
    budget_passed = bool(finest.max_abs_convective_power_W < 1e-10 * finest.power_scale_W)
    convergence_passed = all(c.nonlinear_all_converged for c in cases)
    physical_qualified = bool(
        passed(finest.velocity_error) and passed(finest.pressure_error)
        and passed(finest.vorticity_error) and passed(finest.energy_error)
        and passed(finest.dissipation_error) and passed(finest.viscous_power_error)
        and order_passed and budget_passed and convergence_passed
    )
    return dict(
        schema='tensorfvm.periodic-tgv-study/1',
        case_definition=dict(
            flow='2-D Taylor-Green vortex, doubly periodic, extended uniformly in z',
            box='[0, 2 pi]^2 x [0, 2 pi]',
            velocity='u=sin x cos y exp(-2 nu t), v=-cos x sin y exp(-2 nu t)',
            pressure='p = rho/4 (cos 2x + cos 2y) exp(-4 nu t)',
            kinetic_energy='E = rho V / 4 exp(-4 nu t)',
            dissipation='eps = 4 nu E',
            vorticity='omega_z = 2 sin x sin y exp(-2 nu t)',
        ),
        viscosity_pa_s=viscosity_pa_s, time_end_s=time_end_s,
        grids=list(grids), base_steps_per_unit_time=base_steps_per_unit_time,
        refinement=refinement,
        observed_order=order, observed_order_passed=order_passed,
        tolerance=tolerance,
        convective_budget_passed=budget_passed,
        nonlinear_convergence_passed=convergence_passed,
        physical_accuracy_qualified=physical_qualified,
        limit_case=dict(
            name=limit.name, n=limit.n, viscosity_pa_s=limit.viscosity_pa_s,
            reynolds_number=float(2 * math.pi / limit.viscosity_pa_s),
            reynolds_definition='U L / nu with U = 1 m/s and L = 2 pi m (box side)',
            time_step_s=limit.time_step_s, steps=limit.steps,
            velocity_error=limit.velocity_error, pressure_error=limit.pressure_error,
            vorticity_error=limit.vorticity_error,
            energy_error=limit.energy_error,
            dissipation_error=limit.dissipation_error,
            velocity_3_percent_passed=passed(limit.velocity_error),
            pressure_3_percent_passed=passed(limit.pressure_error),
            vorticity_3_percent_passed=passed(limit.vorticity_error),
            energy_3_percent_passed=passed(limit.energy_error),
            all_quantities_passed=limit_passed,
            expected_failure_observed=not limit_passed,
            failure_reason=limit_failure,
        ),
        negative_control_qualified=not limit_passed,
        distributed_supported=PeriodicMACSolver.distributed_supported,
        traces={c.name: c.trace for c in cases},
        fields={cases[-1].name: cases[-1].fields},
    )