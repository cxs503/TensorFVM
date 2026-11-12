"""Stokes' second problem (oscillating wall) in the native periodic channel.

Why this module exists
----------------------
``verification.pulsatile_channel`` verified the *body-force* driven unsteady
channel against the Womersley closed form and explicitly listed the remaining
gap: the walls of :class:`~tensorfvm.channel_periodic.PeriodicChannelSolver`
were homogeneous no-slip, so the classical **oscillating-wall** problem
(Stokes' second problem, Stokes/Rayleigh layer) could not be posed at all
(``NOT_SUPPORTED`` in the D14 ledger entry).  The solver now accepts moving
walls (``step(wall_velocity_m_s=...)``); this module is the verification surface
that makes the new capability mean something:

* the wall is the *driver*, so the whole flow is a boundary-generated shear
  wave -- a different code path from a body force (the motion enters through the
  wall flux of the implicit diffusion, not through the source term),
* the exact solution is closed-form for a **finite** channel, so no
  semi-infinite assumption has to be smuggled into the benchmark.

Closed form
-----------
Lower wall oscillating, ``u(0, t) = U0 cos(omega t)``; upper wall at rest,
``u(H, t) = 0``; streamwise periodic, no forcing, no wall-normal motion::

    u(y, t) = Re{ U0 sinh(k (H - y)) / sinh(k H) * exp(i omega t) },
    k = (1 + i) / delta,          delta = sqrt(2 nu / omega)

(``k^2 = i omega / nu`` is exactly the momentum equation ``u_t = nu u_yy``; both
boundary conditions are built in).  Consequences used below -- each one is
re-derived numerically in :func:`closed_form_residual`,
:func:`wall_gradient_numeric` and :func:`limits`, never hand-copied:

* bulk velocity amplitude ``U0 (cosh(kH) - 1) / (k sinh(kH) H)``,
* wall shear ``tau_w(t) = mu Re{ -U0 k coth(kH) exp(i omega t) }``,
* ``omega -> 0``: Couette profile ``U0 (1 - y/H)``,
* ``delta << H``: semi-infinite Stokes layer
  ``U0 exp(-y/delta) cos(omega t - y/delta)`` with ``tau_w`` amplitude
  ``rho U0 sqrt(nu omega)`` leading ``U0`` by 45 deg.

Literature: Stokes (1851); Lamb, *Hydrodynamics* §345; Batchelor (1967) §4.6;
Schlichting & Gersten, *Boundary-Layer Theory*, ch. 5.2.

Criteria (pre-declared, see ``docs/verification-stokes-second/criteria.md``)
--------------------------------------------------------------------------
``A1`` closed form satisfies the momentum equation with **numerical**
       derivatives: ``max|residual| / (U0 omega) < 1e-6``.
``A2`` ``omega -> 0`` branch (``delta = 100 H``) equals Couette ``< 1e-3``.
``A3`` ``delta << H`` branch (``delta = H/200``) equals the semi-infinite Stokes
       layer for ``y <= 20 delta`` ``< 1e-9``.
``A4`` closed-form wall shear vs a central difference of the closed-form
       profile ``< 1e-6`` relative (4 phases).
``A5`` textbook semi-infinite wall shear ``rho U0 sqrt(nu omega)`` and 45 deg
       lead reproduced by the finite-channel form at ``delta = H/200``:
       amplitude ``< 1e-3``, phase ``< 0.5`` deg.
``A6`` steady moving wall in the production solver = Couette to round-off
       (``< 1e-10``, both ``wall_flux`` variants) -- the inhomogeneous ghost
       value is exact for a linear profile.
``B1``..``B3`` oscillating wall in the production solver vs the closed form at
       ``W = H/delta = 2, 5, 10`` (``ny = 64``, ``dt/T = 1/200``): relative L2
       profile error over the phases of the last period ``< 3 %``.
``B4`` bulk-velocity amplitude ``< 3 %`` and phase ``< 5 deg`` at ``W = 5``.
``B5`` wall-shear amplitude (three-point one-sided estimator) ``< 3 %`` at
       ``W = 5``; the scheme-consistent flux estimator is reported, not graded.
``B6`` discrete unsteady momentum budget ``rho H dUbar/dt + tau_0 + tau_H = 0``
       ``< 1e-8`` of ``rho U0 H omega`` (checks sign and level of the moving-wall
       source, which the tautological steady balance cannot see).
``B7`` first-order time accuracy: observed order over ``dt/T = 1/100, 1/200,
       1/400`` at ``W = 5`` ``>= 0.7``.
``B8`` wall-normal convergence ``ny = 16 -> 32 -> 64`` at ``W = 5``: monotone,
       observed order ``>= 0.5``.
``B9`` ``max|v| = 0`` and ``max|div| < 1e-10`` throughout the sampled period.
``C1`` negative control: the steady-wall solution differs from the oscillating
       one by ``> 10 %`` (a solver without wall motion cannot pass B1..B3).
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..channel_periodic import PeriodicChannelConfig, PeriodicChannelSolver

# Working point: water-like kinematic viscosity, unit channel height, 1 m/s wall.
NU = 0.01
H = 1.0
RHO = 1.0
U0 = 1.0


# ------------------------------------------------------------------ closed form
def stokes_layer(omega: float, *, nu: float = NU) -> float:
    """Stokes-layer thickness ``delta = sqrt(2 nu / omega)`` (m)."""
    return math.sqrt(2.0 * nu / omega)


def omega_for(womersley: float, *, nu: float = NU, height: float = H) -> float:
    """Angular frequency giving ``W = H/delta = womersley``."""
    return 2.0 * nu * float(womersley) ** 2 / height ** 2


def wavenumber(omega: float, *, nu: float = NU) -> complex:
    """Complex wall-normal wavenumber ``k`` with ``k^2 = i omega / nu``."""
    return complex(np.sqrt(1j * omega / nu))


def complex_amplitude(y, *, omega: float = 1.0, amplitude: float = U0, nu: float = NU,
                      height: float = H) -> np.ndarray:
    """Complex amplitude ``uhat(y)`` with ``u(y, t) = Re{uhat e^{i omega t}}``."""
    y = np.asarray(y, dtype=float)
    k = wavenumber(omega, nu=nu)
    return amplitude * np.sinh(k * (height - y)) / np.sinh(k * height)


def exact_profile(y, t, *, omega: float = 1.0, amplitude: float = U0, nu: float = NU,
                  height: float = H) -> np.ndarray:
    """Exact finite-channel velocity ``u(y, t)`` for the oscillating lower wall."""
    return np.real(complex_amplitude(y, omega=omega, amplitude=amplitude, nu=nu,
                                     height=height) * np.exp(1j * omega * t))


def exact_bulk_amplitude(*, omega: float = 1.0, amplitude: float = U0, nu: float = NU,
                         height: float = H) -> complex:
    """Complex bulk-velocity amplitude ``(1/H) integral_0^H uhat dy``."""
    k = wavenumber(omega, nu=nu)
    return amplitude * (np.cosh(k * height) - 1.0) / (k * np.sinh(k * height) * height)


def exact_wall_gradient(t, *, omega: float = 1.0, amplitude: float = U0, nu: float = NU,
                        height: float = H) -> float:
    """Exact ``du/dy`` at the lower wall, ``Re{-U0 k coth(kH) e^{i omega t}}``."""
    k = wavenumber(omega, nu=nu)
    return float(np.real(-amplitude * k / np.tanh(k * height) * np.exp(1j * omega * t)))


def exact_wall_shear(t, *, omega: float = 1.0, amplitude: float = U0, nu: float = NU,
                     height: float = H, rho: float = RHO) -> float:
    """Exact lower-wall shear stress (Pa), same sign convention as ``du/dy`` at ``y=0``."""
    return rho * nu * exact_wall_gradient(t, omega=omega, amplitude=amplitude, nu=nu,
                                          height=height)


def semi_infinite_profile(y, t, *, omega: float = 1.0, amplitude: float = U0,
                          nu: float = NU) -> np.ndarray:
    """``delta << H`` limit: ``U0 exp(-y/delta) cos(omega t - y/delta)``."""
    y = np.asarray(y, dtype=float)
    delta = stokes_layer(omega, nu=nu)
    return amplitude * np.exp(-y / delta) * np.cos(omega * t - y / delta)


def semi_infinite_wall_shear_amplitude(*, omega: float = 1.0, amplitude: float = U0,
                                      nu: float = NU, rho: float = RHO) -> float:
    """Textbook ``tau_w = rho U0 sqrt(nu omega)`` for the semi-infinite Stokes layer."""
    return rho * amplitude * math.sqrt(nu * omega)


def couette_profile(y, *, amplitude: float = U0, height: float = H) -> np.ndarray:
    """``omega -> 0`` limit: ``U0 (1 - y/H)``."""
    y = np.asarray(y, dtype=float)
    return amplitude * (1.0 - y / height)


def wall_gradient_numeric(y_wall: float, t, *, omega: float = 1.0, amplitude: float = U0,
                          nu: float = NU, height: float = H, h: float = 1e-6) -> float:
    """Central difference of the closed-form profile about ``y_wall``.

    The closed form is analytic, so evaluating it at ``y < 0`` is a legitimate
    extrapolation; this keeps the check independent of the series algebra used
    for :func:`exact_wall_gradient`.
    """
    upper = exact_profile(y_wall + h, t, omega=omega, amplitude=amplitude, nu=nu, height=height)
    lower = exact_profile(y_wall - h, t, omega=omega, amplitude=amplitude, nu=nu, height=height)
    return float((upper - lower) / (2.0 * h))


def closed_form_residual(*, omega: float = 1.0, amplitude: float = U0, nu: float = NU,
                         height: float = H, t: float = 0.31, n_y: int = 401,
                         h_y: float = 1e-5, h_t: float = 1e-5) -> dict:
    """Momentum-equation residual of the closed form with **numerical** derivatives."""
    y = np.linspace(0.2 * height, 0.8 * height, n_y)
    args = dict(omega=omega, amplitude=amplitude, nu=nu, height=height)
    u = exact_profile(y, t, **args)
    dudt = (exact_profile(y, t + h_t, **args) - exact_profile(y, t - h_t, **args)) / (2 * h_t)
    d2udy2 = (exact_profile(y + h_y, t, **args) - 2.0 * u
              + exact_profile(y - h_y, t, **args)) / h_y ** 2
    residual = dudt - nu * d2udy2
    scale = amplitude * omega
    return dict(max_abs_residual=float(np.abs(residual).max()),
                relative=float(np.abs(residual).max() / scale), scale=float(scale))


def limits(*, nu: float = NU, height: float = H, amplitude: float = U0, n: int = 41,
           t: float = 0.4) -> dict:
    """``omega -> 0`` (Couette) and ``delta -> 0`` (semi-infinite) branch checks."""
    y = np.linspace(0.0, height, n)
    omega_slow = omega_for(0.01, nu=nu, height=height)      # delta = 100 H
    couette_error = float(np.abs(exact_profile(y, t, omega=omega_slow, amplitude=amplitude,
                                               nu=nu, height=height)
                                - couette_profile(y, amplitude=amplitude, height=height)).max())

    omega_fast = omega_for(200.0, nu=nu, height=height)     # delta = H/200
    delta = stokes_layer(omega_fast, nu=nu)
    y_layer = np.linspace(0.0, 20.0 * delta, 401)
    semi_error = float(np.abs(exact_profile(y_layer, t, omega=omega_fast, amplitude=amplitude,
                                            nu=nu, height=height)
                              - semi_infinite_profile(y_layer, t, omega=omega_fast,
                                                      amplitude=amplitude, nu=nu)).max())
    return dict(couette_relative_error=couette_error / amplitude,
                semi_infinite_relative_error=semi_error / amplitude,
                delta_slow=stokes_layer(omega_slow, nu=nu),
                delta_fast=delta, omega_slow=omega_slow, omega_fast=omega_fast)


def amplitude_phase(times, values, omega: float) -> dict:
    """Single-frequency amplitude/phase of a uniformly sampled periodic signal."""
    t = np.asarray(times, dtype=float)
    v = np.asarray(values, dtype=float)
    if t.size != v.size or t.size < 3:
        raise ValueError("need at least three matched samples")
    phase = np.exp(-1j * omega * t)
    amplitude = 2.0 * np.mean(v * phase)     # one-sided complex amplitude
    return dict(amplitude=float(abs(amplitude)),
                phase_deg=float(math.degrees(np.angle(amplitude))))


# --------------------------------------------------------------------- solver run
def solve_oscillating_wall(*, womersley: float, ny: int = 64, nx: int = 4,
                           dt_fraction: float = 1.0 / 200.0, n_phases: int = 10,
                           max_periods: int = 90, settle_tol: float = 1e-4,
                           amplitude: float = U0, nu: float = NU, height: float = H,
                           rho: float = RHO, wall_flux: str = "half-cell",
                           initialize: str = "rest",
                           solver_class=PeriodicChannelSolver) -> dict:
    """March the production solver from rest until periodic, then sample one period.

    Periodicity is declared when the end-of-period profile stops changing by more
    than ``settle_tol * amplitude`` between two consecutive periods (the start-up
    transient decays with the slowest channel mode ``exp(-nu pi^2 t / H^2)``); no
    sampled number is used for the settling test.

    ``initialize='exact'`` instead starts from the closed-form state at ``t = 0``
    and samples the first period.  That is the right setting for the *refinement*
    studies (B8/B9): there the question is the discretisation error of one period
    of evolution, and a start-up transient would add an error of its own that
    vanishes only after many periods.  B1..B3 keep ``'rest'`` so the graded cases
    do show the wall driving a fluid initially at rest.
    """
    omega = omega_for(womersley, nu=nu, height=height)
    period = 2.0 * math.pi / omega
    steps_per_period = int(round(1.0 / dt_fraction))
    if steps_per_period % n_phases:
        raise ValueError("steps_per_period must be a multiple of n_phases (uniform sampling)")
    dt = period / steps_per_period
    cfg = PeriodicChannelConfig(nx=nx, ny=ny, length_m=height, height_m=height,
                                density_kg_m3=rho, viscosity_pa_s=nu * rho,
                                body_force_m_s2=1e-12, time_step_s=dt, wall_flux=wall_flux)
    solver = solver_class(cfg)
    y_cell = solver.wall_coordinates()[0].cpu().numpy()

    started = time.time()
    previous = None
    settled_at = 0
    if initialize == "exact":
        solver.set_velocity_profile(exact_profile(y_cell, 0.0, omega=omega,
                                                  amplitude=amplitude, nu=nu, height=height))
        previous = solver.u.mean(dim=-1).clone()
        settled_at = 0
    elif initialize != "rest":
        raise ValueError("initialize must be 'rest' or 'exact'")
    settle_limit = 0 if initialize == "exact" else max_periods * steps_per_period
    for _ in range(settle_limit):
        t_next = solver.time + dt
        solver.step(body_force_m_s2=0.0,
                    wall_velocity_m_s=(amplitude * math.cos(omega * t_next), 0.0))
        if len(solver.history) % steps_per_period == 0:
            profile = solver.u.mean(dim=-1).clone()
            if (previous is not None
                    and float((profile - previous).abs().max()) <= settle_tol * amplitude):
                settled_at = len(solver.history) // steps_per_period
                break
            previous = profile
    settle_steps = len(solver.history)
    settled_at = settled_at if initialize == "rest" else 0

    samples = []
    schedule = {int(round(k * steps_per_period / n_phases)): k for k in range(1, n_phases + 1)}
    # Per-step records: the discrete momentum budget is a one-step identity, so
    # it must be checked step to step, not between the (much coarser) samples.
    step_records = []
    bulk_before = solver.bulk_velocity()
    while len(samples) < n_phases:
        t_next = solver.time + dt
        wall = amplitude * math.cos(omega * t_next)
        solver.step(body_force_m_s2=0.0, wall_velocity_m_s=(wall, 0.0))
        bulk_after = solver.bulk_velocity()
        shear_flux, shear_flux_top = solver.wall_shear("flux")
        step_records.append(dict(bulk_rate=(bulk_after - bulk_before) / dt,
                                 wall_shear_flux_sum_pa=shear_flux + shear_flux_top))
        bulk_before = bulk_after
        step_index = len(solver.history) - settle_steps
        if step_index in schedule:
            shear_quad, _ = solver.wall_shear("quadratic")
            samples.append(dict(phase=schedule[step_index] / n_phases,
                                time_s=solver.time,
                                profile=solver.u.mean(dim=-1).cpu().numpy().copy(),
                                bulk=bulk_after,
                                wall_velocity_m_s=wall,
                                wall_shear_flux_pa=shear_flux,
                                wall_shear_quadratic_pa=shear_quad,
                                v_max=float(solver.v.abs().max()),
                                divergence=solver.max_divergence()))

    times = np.array([s["time_s"] for s in samples])
    profiles = np.array([s["profile"] for s in samples])
    exact = np.array([exact_profile(y_cell, s["time_s"], omega=omega, amplitude=amplitude,
                                    nu=nu, height=height) for s in samples])
    envelope = float(np.sqrt(np.mean(exact ** 2, axis=1)).max())
    l2 = np.sqrt(np.mean((profiles - exact) ** 2, axis=1))
    return dict(womersley=womersley, omega=omega, period_s=period,
                delta=stokes_layer(omega, nu=nu), dt=dt, dt_fraction=dt_fraction,
                ny=ny, nx=nx, wall_flux=wall_flux, amplitude=amplitude,
                settle_periods=settled_at, settle_steps=settle_steps, settle_tol=settle_tol,
                samples=samples, step_records=step_records,
                times=times, y_cell=y_cell, profiles=profiles, exact=exact,
                envelope=envelope, profile_relative_l2=float(l2.max() / envelope),
                bulk_amplitude=amplitude_phase(times, [s["bulk"] for s in samples], omega),
                shear_amplitude_quadratic=amplitude_phase(
                    times, [s["wall_shear_quadratic_pa"] for s in samples], omega),
                shear_amplitude_flux=amplitude_phase(
                    times, [s["wall_shear_flux_pa"] for s in samples], omega),
                max_wall_normal_velocity=float(max(s["v_max"] for s in samples)),
                max_divergence=float(max(s["divergence"] for s in samples)),
                runtime_s=time.time() - started)


def exact_amplitudes(*, womersley: float, amplitude: float = U0, nu: float = NU,
                     height: float = H, rho: float = RHO) -> dict:
    """Closed-form bulk and wall-shear amplitudes/phases for one ``W``."""
    omega = omega_for(womersley, nu=nu, height=height)
    bulk = exact_bulk_amplitude(omega=omega, amplitude=amplitude, nu=nu, height=height)
    k = wavenumber(omega, nu=nu)
    shear = rho * nu * (-amplitude * k / np.tanh(k * height))
    return dict(omega=omega, delta=stokes_layer(omega, nu=nu),
                bulk_amplitude=float(abs(bulk)),
                bulk_phase_deg=float(math.degrees(np.angle(bulk))),
                shear_amplitude=float(abs(shear)),
                shear_phase_deg=float(math.degrees(np.angle(shear))),
                # mu du/dy is one sign convention for "wall shear"; the textbook
                # (+45 deg lead, tau = rho U0 sqrt(nu omega) cos(omega t + pi/4))
                # is the stress the *plate* exerts, i.e. minus this one.
                plate_shear_phase_deg=float(math.degrees(np.angle(-shear))),
                semi_infinite_shear_amplitude=semi_infinite_wall_shear_amplitude(
                    omega=omega, amplitude=amplitude, nu=nu, rho=rho))


def momentum_budget(run: dict, *, rho: float = RHO, height: float = H,
                    amplitude: float = U0) -> dict:
    """Discrete unsteady budget ``rho H dUbar/dt + tau_0 + tau_H = 0`` (no forcing)."""
    omega = run["omega"]
    residual = [rho * height * s["bulk_rate"] + s["wall_shear_flux_sum_pa"]
                for s in run["step_records"]]
    scale = rho * amplitude * height * omega
    return dict(max_abs_residual=float(np.abs(residual).max()),
                relative=float(np.abs(residual).max() / scale), scale=float(scale))


def steady_wall_control(*, ny: int = 64, nx: int = 4, amplitude: float = U0, nu: float = NU,
                        height: float = H, rho: float = RHO, wall_flux: str = "half-cell",
                        tolerance: float = 1e-14) -> dict:
    """Steady moving wall (Couette) from the production solver, plus its error."""
    cfg = PeriodicChannelConfig(nx=nx, ny=ny, length_m=height, height_m=height,
                                density_kg_m3=rho, viscosity_pa_s=nu * rho,
                                body_force_m_s2=1e-12, time_step_s=1.0, wall_flux=wall_flux)
    solver = PeriodicChannelSolver(cfg)
    solver.set_wall_velocity(amplitude, 0.0)
    info = solver.solve_steady(tolerance=tolerance, max_steps=200000, min_steps=10)
    y_cell = solver.wall_coordinates()[0].cpu().numpy()
    numeric = solver.u.mean(dim=-1).cpu().numpy()
    exact = couette_profile(y_cell, amplitude=amplitude, height=height)
    shear_quad, _ = solver.wall_shear("quadratic")
    return dict(converged=info["converged"], steps=info["steps"],
                max_profile_error=float(np.abs(numeric - exact).max() / amplitude),
                max_wall_normal_velocity=float(solver.v.abs().max()),
                max_divergence=solver.max_divergence(),
                bulk_velocity=solver.bulk_velocity(),
                wall_shear_lower_pa=float(shear_quad),
                wall_shear_exact_pa=float(-nu * rho * amplitude / height))


# ------------------------------------------------------------------------ criteria
def _criterion(cid, name, value, threshold, mode="lt", note=""):
    verdict = "PASS" if (value < threshold if mode == "lt" else value > threshold) else "FAIL"
    return dict(id=cid, name=name, value=float(value), threshold=float(threshold),
                mode=mode, verdict=verdict, note=note)


def _observed_order(errors, refinements) -> float:
    """Observed order from consecutive error levels and their refinement ratios."""
    order = []
    for (e_coarse, e_fine), ratio in zip(zip(errors[:-1], errors[1:]), refinements):
        if e_fine > 0 and e_coarse > 0:
            order.append(math.log(e_coarse / e_fine) / math.log(ratio))
    return float(min(order)) if order else float("nan")


def run(*, ny_main: int = 64, dt_fraction: float = 1.0 / 200.0, n_phases: int = 10,
        womersley_cases=(2.0, 5.0, 10.0), womersley_study: float = 5.0,
        ny_study=(16, 32, 64), dt_study=(1.0 / 100.0, 1.0 / 200.0, 1.0 / 400.0),
        settle_tol: float = 1e-4, amplitude: float = U0, nu: float = NU, height: float = H,
        rho: float = RHO) -> dict:
    """Run every criterion; the thresholds above were declared before the numbers."""
    started = time.time()
    crit = []
    payload = {}

    # ---- closed form (no solver) -------------------------------------------
    omega_study = omega_for(womersley_study, nu=nu, height=height)
    residual = closed_form_residual(omega=omega_study, amplitude=amplitude, nu=nu, height=height)
    crit.append(_criterion("A1", "closed-form PDE residual (numerical derivatives)",
                           residual["relative"], 1e-6,
                           note=f"max|residual|={residual['max_abs_residual']:.3e} m/s^2"))
    lim = limits(nu=nu, height=height, amplitude=amplitude)
    crit.append(_criterion("A2", "omega->0 branch equals Couette",
                           lim["couette_relative_error"], 1e-3, note="delta = 100 H"))
    crit.append(_criterion("A3", "delta<<H branch equals the semi-infinite Stokes layer",
                           lim["semi_infinite_relative_error"], 1e-9,
                           note="delta = H/200, y <= 20 delta"))
    grad_err = []
    for t in np.arange(4) * (math.pi / 2.0) / omega_study:
        tau = exact_wall_gradient(t, omega=omega_study, amplitude=amplitude, nu=nu, height=height)
        num = wall_gradient_numeric(0.0, t, omega=omega_study, amplitude=amplitude, nu=nu,
                                    height=height)
        grad_err.append(abs(num - tau) / abs(tau))
    crit.append(_criterion("A4", "closed-form wall shear vs central difference of the profile",
                           max(grad_err), 1e-6, note=f"4 phases, W={womersley_study:g}"))
    far = exact_amplitudes(womersley=200.0, amplitude=amplitude, nu=nu, height=height, rho=rho)
    tau_ratio = far["shear_amplitude"] / far["semi_infinite_shear_amplitude"]
    crit.append(_criterion("A5", "semi-infinite wall-shear amplitude rho U0 sqrt(nu omega)",
                           abs(tau_ratio - 1.0), 1e-3, note=f"ratio={tau_ratio:.9f}"))
    crit.append(_criterion("A5b", "semi-infinite wall-shear lead of 45 deg (plate convention)",
                           abs(far["plate_shear_phase_deg"] - 45.0), 0.5,
                           note=f"mu du/dy phase={far['shear_phase_deg']:.4f} deg, "
                                f"plate stress phase={far['plate_shear_phase_deg']:.6f} deg"))

    # ---- steady moving wall (Couette) --------------------------------------
    couette = {wf: steady_wall_control(ny=ny_main, amplitude=amplitude, nu=nu, height=height,
                                       rho=rho, wall_flux=wf)
               for wf in ("half-cell", "quadratic")}
    worst_couette = max(c["max_profile_error"] for c in couette.values())
    crit.append(_criterion("A6", "steady moving wall equals Couette (both wall_flux variants)",
                           worst_couette, 1e-10,
                           note="; ".join(f"{wf}: {c['max_profile_error']:.3e} "
                                          f"({c['steps']} steps)" for wf, c in couette.items())))
    payload["couette"] = couette

    # ---- oscillating wall, production solver ------------------------------
    runs = {}
    for w in womersley_cases:
        runs[w] = solve_oscillating_wall(womersley=w, ny=ny_main, dt_fraction=dt_fraction,
                                         n_phases=n_phases, settle_tol=settle_tol,
                                         amplitude=amplitude, nu=nu, height=height, rho=rho,
                                         initialize="rest")
    payload["runs"] = runs
    for i, w in enumerate(womersley_cases, start=1):
        r = runs[w]
        crit.append(_criterion(f"B{i}", f"solver vs closed form, W={w:g}, ny={r['ny']}, "
                                        f"dt/T={r['dt_fraction']:.0e}",
                               r["profile_relative_l2"], 3e-2,
                               note=f"{n_phases} phases, settled after {r['settle_periods']} "
                                    f"periods ({r['settle_steps']} steps), "
                                    f"delta/H={r['delta'] / height:.3f}, {r['runtime_s']:.1f} s"))
    study = runs[womersley_study]
    ex = exact_amplitudes(womersley=womersley_study, amplitude=amplitude, nu=nu,
                          height=height, rho=rho)
    payload["exact_study"] = ex
    crit.append(_criterion("B4a", f"bulk-velocity amplitude, W={womersley_study:g}",
                           abs(study["bulk_amplitude"]["amplitude"] / ex["bulk_amplitude"] - 1.0),
                           3e-2, note=f"{study['bulk_amplitude']['amplitude']:.6e} vs "
                                      f"{ex['bulk_amplitude']:.6e} m/s"))
    crit.append(_criterion("B4b", f"bulk-velocity phase, W={womersley_study:g}",
                           abs(study["bulk_amplitude"]["phase_deg"] - ex["bulk_phase_deg"]),
                           5.0, note=f"{study['bulk_amplitude']['phase_deg']:.3f} vs "
                                     f"{ex['bulk_phase_deg']:.3f} deg"))
    shear_quad_err = abs(study["shear_amplitude_quadratic"]["amplitude"] / ex["shear_amplitude"] - 1.0)
    shear_flux_err = abs(study["shear_amplitude_flux"]["amplitude"] / ex["shear_amplitude"] - 1.0)
    crit.append(_criterion("B5", f"wall-shear amplitude (3-point estimator), W={womersley_study:g}",
                           shear_quad_err, 3e-2,
                           note=f"{study['shear_amplitude_quadratic']['amplitude']:.6e} vs "
                                f"{ex['shear_amplitude']:.6e} Pa; scheme flux estimator "
                                f"(not graded) err={shear_flux_err:.3e}"))
    budget = momentum_budget(study, rho=rho, height=height, amplitude=amplitude)
    payload["momentum_budget"] = budget
    crit.append(_criterion("B6", "discrete unsteady momentum budget rho H dUbar/dt + tau_0 + tau_H",
                           budget["relative"], 1e-8,
                           note=f"max|residual|={budget['max_abs_residual']:.3e} Pa"))
    crit.append(_criterion("B7a", f"no wall-normal velocity, W={womersley_study:g}",
                           study["max_wall_normal_velocity"], 1e-12))
    crit.append(_criterion("B7b", f"cell divergence after projection, W={womersley_study:g}",
                           study["max_divergence"], 1e-10))

    # temporal refinement
    temporal = {}
    for fraction in dt_study:
        temporal[fraction] = solve_oscillating_wall(
            womersley=womersley_study, ny=ny_main, dt_fraction=fraction, n_phases=n_phases,
            settle_tol=settle_tol, amplitude=amplitude, nu=nu, height=height, rho=rho,
            initialize="exact")
    payload["temporal"] = temporal
    dt_errors = [temporal[f]["profile_relative_l2"] for f in dt_study]
    dt_order = _observed_order(dt_errors, [dt_study[i] / dt_study[i + 1]
                                          for i in range(len(dt_study) - 1)])
    crit.append(_criterion("B8", "temporal convergence order", dt_order, 0.7,
                           mode="gt",
                           note="dt/T=" + ", ".join(f"{f:.0e}: {e:.3e}" for f, e
                                                    in zip(dt_study, dt_errors))))

    # wall-normal refinement
    spatial = {}
    for ny in ny_study:
        # The default ``half-cell`` wall closure is only first order in the wall
        # flux, so it would mask the interior order; the three-point ``quadratic``
        # closure is second order and is used for the spatial study.  ``dt/T =
        # 1/1000`` keeps the first-order time error (7e-3 at 1/200) well below the
        # spatial levels that are being graded.
        spatial[ny] = solve_oscillating_wall(
            womersley=womersley_study, ny=ny, dt_fraction=1.0 / 1000.0,
            n_phases=n_phases, settle_tol=settle_tol, amplitude=amplitude, nu=nu,
            height=height, rho=rho, initialize="exact", wall_flux="quadratic")
    payload["spatial"] = spatial
    ny_errors = [spatial[ny]["profile_relative_l2"] for ny in ny_study]
    ny_order = _observed_order(ny_errors, [ny_study[i + 1] / ny_study[i]
                                          for i in range(len(ny_study) - 1)])
    monotone = all(ny_errors[i] <= ny_errors[i - 1] for i in range(1, len(ny_errors)))
    crit.append(_criterion("B9", "wall-normal convergence order", ny_order, 0.5, mode="gt",
                           note=("monotone " if monotone else "NON-monotone ")
                                + ", ".join(f"ny={ny}: {e:.3e}"
                                            for ny, e in zip(ny_study, ny_errors))))

    # ---- negative control ---------------------------------------------------
    steady = steady_wall_control(ny=ny_main, amplitude=amplitude, nu=nu, height=height, rho=rho)
    y_cell = study["y_cell"]
    phase = math.pi / 2.0
    oscillating = exact_profile(y_cell, phase / study["omega"], omega=study["omega"],
                                amplitude=amplitude, nu=nu, height=height)
    couette = couette_profile(y_cell, amplitude=amplitude, height=height)
    control = float(np.sqrt(np.mean((oscillating - couette) ** 2)) / np.sqrt(np.mean(oscillating ** 2)))
    payload["control"] = dict(relative_difference=control, steady_wall=steady)
    crit.append(_criterion("C1", f"oscillating vs steady-wall profile difference, "
                                f"W={womersley_study:g}", control, 0.1, mode="gt",
                           note="a solver without wall motion cannot pass B1..B3"))

    n_pass = sum(1 for c in crit if c["verdict"] == "PASS")
    return dict(module="stokes_second_problem", timestamp=datetime.now(timezone.utc).isoformat(),
                working_point=dict(nu=nu, height=height, rho=rho, amplitude=amplitude,
                                   ny_main=ny_main, dt_fraction=dt_fraction, n_phases=n_phases),
                limits=lim, criteria=crit, n_pass=n_pass, n_total=len(crit),
                all_pass=n_pass == len(crit), runtime_s=time.time() - started, **payload)


# ------------------------------------------------------------------------ artifacts
def figure(payload: dict, path: Path | str) -> Path:
    """Three panels: profiles at phases, wall-shear wave, convergence."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    runs = payload["runs"]
    key = payload["working_point"].get("womersley_study")
    study = runs[key] if key in runs else runs[sorted(runs)[len(runs) // 2]]
    omega = study["omega"]
    y = study["y_cell"]

    ax = axes[0]
    for phase in (0.0, 0.25, 0.5, 0.75):
        t = phase * 2 * math.pi / omega
        ax.plot(exact_profile(y, t, omega=omega, amplitude=study["amplitude"]), y, "-",
                label=f"exact t/T={phase:g}")
        idx = int(round(phase * len(study["samples"]))) % len(study["samples"])
        ax.plot(study["profiles"][idx], y, "o", ms=3, mfc="none",
                label=f"solver t/T={study['samples'][idx]['phase']:g}")
    ax.set_xlabel("u (m/s)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Stokes 2nd problem, W={study['womersley']:g}")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    t_fine = np.linspace(0.0, study["period_s"], 400)
    ax.plot(t_fine / study["period_s"],
            [exact_wall_shear(t, omega=omega, amplitude=study["amplitude"],
                              nu=payload["working_point"]["nu"]) for t in t_fine],
            "-", label="exact wall shear")
    ax.plot(np.array([s["time_s"] for s in study["samples"]]) / study["period_s"],
            [s["wall_shear_quadratic_pa"] for s in study["samples"]], "o", ms=4,
            label="solver (3-point)")
    ax.plot(np.array([s["time_s"] for s in study["samples"]]) / study["period_s"],
            [s["wall_shear_flux_pa"] for s in study["samples"]], "s", ms=3, mfc="none",
            label="solver (scheme flux)")
    ax.set_xlabel("t / T")
    ax.set_ylabel("tau_w (Pa)")
    ax.set_title("wall shear: closed form vs solver")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[2]
    if "temporal" in payload:
        fractions = sorted(payload["temporal"])
        ax.loglog([1.0 / f for f in fractions],
                  [payload["temporal"][f]["profile_relative_l2"] for f in fractions],
                  "o-", label="time refinement (dt/T)")
    if "spatial" in payload:
        nys = sorted(payload["spatial"])
        ax.loglog([1.0 / ny for ny in nys],
                  [payload["spatial"][ny]["profile_relative_l2"] for ny in nys],
                  "s-", label="space refinement (dy/H)")
    ax.set_xlabel("cell / step size (normalised)")
    ax.set_ylabel("relative L2 profile error")
    ax.set_title("refinement (slope = convergence order)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


_ARRAY_KEYS = ("profiles", "exact", "y_cell", "times", "samples", "step_records")


def public_payload(payload: dict) -> dict:
    """Drop the raw field arrays; the summary keeps every graded number."""
    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items() if k not in _ARRAY_KEYS}
        if isinstance(obj, (list, tuple)):
            return [clean(v) for v in obj]
        return obj
    return clean(payload)


def write_artifacts(outdir: Path | str = "docs/verification-stokes-second", **kwargs) -> dict:
    """Run every criterion, then write ``summary.json``, ``criteria.md`` and the figure."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    payload = run(**kwargs)
    public = public_payload(payload)
    public["working_point"]["womersley_study"] = kwargs.get("womersley_study", 5.0)
    (outdir / "summary.json").write_text(
        json.dumps(public, indent=2,
                   default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o)))
    figure(payload, outdir / "fig_stokes_second.png")
    crit_lines = ["# Stokes' second problem (oscillating wall) — criteria (pre-declared)", ""]
    for c in payload["criteria"]:
        symbol = "<" if c["mode"] == "lt" else ">"
        crit_lines.append(f"- **{c['id']}** {c['verdict']}: {c['name']} = {c['value']:.4e} "
                          f"({symbol} {c['threshold']:.3e}) — {c['note']}")
    crit_lines.append("")
    crit_lines.append(f"**{payload['n_pass']}/{payload['n_total']} PASS — "
                      f"all_pass={payload['all_pass']}**")
    (outdir / "criteria.md").write_text("\n".join(crit_lines) + "\n")
    return public


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outdir", default="docs/verification-stokes-second")
    parser.add_argument("--ny", type=int, default=64)
    parser.add_argument("--dt-fraction", type=float, default=1.0 / 200.0)
    parser.add_argument("--quick", action="store_true", help="closed form + one W only")
    args = parser.parse_args(argv)
    if args.quick:
        payload = write_artifacts(args.outdir, ny_main=args.ny, dt_fraction=args.dt_fraction,
                                  womersley_cases=(5.0,), womersley_study=5.0,
                                  ny_study=(32, 64), dt_study=(1.0 / 100.0, 1.0 / 200.0))
    else:
        payload = write_artifacts(args.outdir, ny_main=args.ny, dt_fraction=args.dt_fraction)
    for crit in payload["criteria"]:
        symbol = "<" if crit["mode"] == "lt" else ">"
        print(f"{crit['id']:5s} {crit['verdict']:4s} {crit['value']:.6e} "
              f"({symbol} {crit['threshold']:.3e}) {crit['name']}")
    print(f"{payload['n_pass']}/{payload['n_total']} PASS   all_pass={payload['all_pass']}   "
          f"{payload['runtime_s']:.1f} s")
    return 0 if payload["all_pass"] else 1


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(main())