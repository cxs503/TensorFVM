"""Verification surface: pulsatile (Womersley) plane-channel flow.

Capability under test
---------------------
:class:`tensorfvm.channel_periodic.PeriodicChannelSolver` -- the staggered MAC
solver (x-periodic, wall bounded in y, semi-implicit: explicit convection,
implicit diffusion, exact discrete projection).  Its **steady** Poiseuille
behaviour is already graded against the analytic parabola; what was never
graded is its **unsteady** behaviour: time accuracy, the phase relationship
between forcing and velocity, and the wall shear under oscillatory forcing.
That is the gap this module closes.

Physics
-------
Plane channel ``0 <= y <= H``, walls at rest, streamwise body force
``G(t) = G0 cos(omega t)`` (``G = -1/rho dp/dx``), constant ``nu``::

    du/dt = G0 cos(omega t) + nu d2u/dy2,   u(0,t) = u(H,t) = 0.

The exact periodic solution (Womersley 1955, plane-channel form; the same
closed form as :data:`tensorfvm.verification.lbm`-side pulsatile benchmarks)
follows from ``u = Re{uhat(y) e^{i omega t}}``::

    uhat(y) = G0/(i omega) [ 1 - cosh(m (y - H/2)) / cosh(m H/2) ],
    m = sqrt(i omega / nu) = (1 + i) / delta,   delta = sqrt(2 nu / omega),

with ``delta`` the Stokes-layer thickness.  Consequences that the verification
uses as *independent* analytic anchors:

* ``omega -> 0``: ``uhat -> (G0 / (2 nu)) (H^2/4 - (y - H/2)^2)`` (steady
  Poiseuille; centre speed ``G0 H^2 / (8 nu)``, bulk ``G0 H^2 / (12 nu)``);
* bulk velocity ``uhat_b = G0/(i omega) [1 - tanh(m H/2) / (m H/2)]``;
* wall gradient ``duhat/dy(0) = G0/(i omega) * m tanh(m H/2)`` -- the wall
  shear **leads** the forcing by up to 45 deg as ``omega`` grows;
* Stokes-layer decay: the deficit from the outer value obeys
  ``|uhat(y) - uhat(H/2)| = (G0/omega) exp(-y/delta)`` for ``delta << y << H/2``.

Only the closed form above is used as the reference; nothing is fitted to the
solver output.

Criteria (pre-declared, thresholds fixed before the run)
-------------------------------------------------------
``A1`` closed-form PDE residual: substitute the closed form into the momentum
       equation with **numerical** derivatives -> ``max|residual| / G0 < 1e-6``
       (guards against an algebraic slip in the derivation; independent of it).
``A2`` steady limit: ``omega -> 0`` closed form equals the analytic parabola,
       centre speed and bulk velocity ``< 1e-6`` relative (the round-off floor of
       the ``1 - cosh/cosh`` cancellation at ``omega H^2/nu = 1e-6`` is ~1e-10).
``A3`` bulk and wall-gradient formulas agree with ``1e-6``-accurate numerical
       integration/differentiation of the profile at four phases.
``A4`` Stokes-layer decay law: fitted slope of ``ln|uhat(y) - uhat(H/2)|`` vs
       ``-1/delta`` within 2 % (W = 40, 80; closed form only, no solver).
``A5`` steady-limit solver behaviour unchanged: with the constant body force the
       solver reproduces ``G0 H^2/(12 nu)`` bulk velocity ``< 1e-3`` relative at
       ny = 64 (measured 4.9e-4, O(dy^2); regression guard for the new per-step
       forcing hook).
``B1``..``B3`` solver vs closed form at ``W = 2, 5, 10`` (ny = 64, dt/T = 1/300):
       L2 profile error over the phases of the last period, normalised by the
       envelope ``max_t ||u_exact(t)||`` (a per-phase norm would blow up at the
       zero crossing), ``< 3 %``.
``B4`` bulk-velocity amplitude error ``< 3 %`` and phase error ``< 5 deg``
       (``W = 5``).
``B5`` wall-shear amplitude error ``< 3 %`` with the solver's three-point
       one-sided estimator (``W = 5``, ny = 128); the scheme-consistent
       (first-order) wall flux is reported, not graded.
``B6`` temporal convergence at ``W = 5``: error must fall with ``dt`` and the
       fitted order must be ``>= 0.7`` (backward Euler is first order).
``B7`` wall-normal convergence at ``W = 5``: error must fall from ny = 32 -> 64.
``B8`` projection does not pollute the one-dimensional solution:
       ``max|v| = 0`` and ``max|div| < 1e-10 1/s``.
``C1`` discrimination (negative control): at ``W = 5`` the oscillatory profile
       differs from the steady parabola by ``> 10 %`` somewhere in the period,
       so B1..B3 cannot pass by accident with a steady solver.

Run::

    PYTHONPATH=src python -m tensorfvm.verification.pulsatile_channel

Artifacts land in ``docs/verification-pulsatile-channel/``.

References
----------
Womersley, J. R. (1955) *Method for the calculation of velocity, rate of flow
and viscous drag in arteries when the pressure gradient is known*, J. Physiol.
127, 553.  Stokes-layer/oscillatory-channel form also in Batchelor (1967) §4.6.
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..channel_periodic import PeriodicChannelConfig, PeriodicChannelSolver

# Default working point: water-like kinematic viscosity, unit channel height.
NU = 0.01
H = 1.0
RHO = 1.0
G0 = 0.012          # m/s^2, gives bulk ~ 0.1 m/s at W = 2 (laminar, small dU/dy)


# --------------------------------------------------------------------- closed form
def womersley_number(omega: float, *, nu: float = NU, height: float = H) -> float:
    """``W = H / delta`` with the Stokes thickness ``delta = sqrt(2 nu / omega)``."""
    return height / math.sqrt(2.0 * nu / omega)


def stokes_layer(omega: float, *, nu: float = NU) -> float:
    """Stokes-layer thickness ``delta = sqrt(2 nu / omega)`` (m)."""
    return math.sqrt(2.0 * nu / omega)


def womersley_profile(y, t, *, omega: float, g0: float = G0, nu: float = NU,
                      height: float = H) -> np.ndarray:
    """Exact periodic velocity ``u(y, t)`` for ``G(t) = g0 cos(omega t)``."""
    y = np.asarray(y, dtype=float)
    m = np.sqrt(1j * omega / nu)
    uhat = g0 / (1j * omega) * (1.0 - np.cosh(m * (y - height / 2.0))
                                / np.cosh(m * height / 2.0))
    return np.real(uhat * np.exp(1j * omega * t))


def womersley_bulk(t, *, omega: float, g0: float = G0, nu: float = NU,
                   height: float = H) -> float:
    """Exact bulk velocity ``(1/H) integral u dy``."""
    m = np.sqrt(1j * omega / nu)
    uhat_b = g0 / (1j * omega) * (1.0 - np.tanh(m * height / 2.0)
                                  / (m * height / 2.0))
    return float(np.real(uhat_b * np.exp(1j * omega * t)))


def womersley_wall_gradient(t, *, omega: float, g0: float = G0, nu: float = NU,
                            height: float = H) -> float:
    """Exact ``du/dy`` at the lower wall, ``Re{m tanh(m H/2) G0/(i omega) e^{i omega t}}``."""
    m = np.sqrt(1j * omega / nu)
    dhat = g0 / (1j * omega) * m * np.tanh(m * height / 2.0)
    return float(np.real(dhat * np.exp(1j * omega * t)))


def steady_poiseuille(y, *, g0: float = G0, nu: float = NU, height: float = H) -> np.ndarray:
    """Steady limit of the closed form: ``G0/(2 nu) (H^2/4 - (y - H/2)^2)``."""
    y = np.asarray(y, dtype=float)
    return g0 / (2.0 * nu) * (height ** 2 / 4.0 - (y - height / 2.0) ** 2)


def closed_form_residual(*, omega: float = 1.0, g0: float = G0, nu: float = NU,
                         height: float = H, n_y: int = 401, t: float = 0.7,
                         h_y: float = 1e-4, h_t: float = 1e-4) -> dict:
    """Momentum-equation residual of the closed form with numerical derivatives.

    Independent of the derivation: ``du/dt`` and ``d2u/dy2`` are finite
    differences of :func:`womersley_profile`, so an algebra slip in ``uhat``
    shows up as a residual of order one instead of cancelling.
    """
    y = np.linspace(0.4 * height, 0.6 * height, n_y)
    dudt = (womersley_profile(y, t + h_t, omega=omega, g0=g0, nu=nu, height=height)
            - womersley_profile(y, t - h_t, omega=omega, g0=g0, nu=nu, height=height)) / (2 * h_t)
    d2udy2 = (womersley_profile(y + h_y, t, omega=omega, g0=g0, nu=nu, height=height)
              - 2 * womersley_profile(y, t, omega=omega, g0=g0, nu=nu, height=height)
              + womersley_profile(y - h_y, t, omega=omega, g0=g0, nu=nu, height=height)) / h_y ** 2
    residual = dudt - g0 * math.cos(omega * t) - nu * d2udy2
    return dict(max_abs_residual=float(np.abs(residual).max()),
                relative=float(np.abs(residual).max() / g0))


def womersley_complex(y, *, omega: float, g0: float = G0, nu: float = NU,
                      height: float = H) -> np.ndarray:
    """Complex amplitude ``uhat(y)`` of the closed form (``u = Re{uhat e^{i omega t}}``)."""
    y = np.asarray(y, dtype=float)
    m = np.sqrt(1j * omega / nu)
    return g0 / (1j * omega) * (1.0 - np.cosh(m * (y - height / 2.0))
                                / np.cosh(m * height / 2.0))


def near_wall_decay_slope(*, omega: float, g0: float = G0, nu: float = NU,
                          height: float = H, lo: float = 2.0,
                          hi: float = 3.0) -> dict:
    """Stokes-layer law: the deficit from the outer value decays as ``exp(-y/delta)``.

    ``uhat(y) - uhat(H/2) = -(g0/(i omega)) cosh(m y)/cosh(m H/2)``, whose magnitude
    is ``(g0/omega) exp(-y/delta)`` once ``y >> delta`` and ``H/2 - y >> delta``.
    Fitting ``ln|deficit|`` against the wall distance ``y`` over
    ``[lo*delta, H/2 - hi*delta]`` must therefore recover the slope ``-1/delta``:
    that is an independent check of the closed form (and of ``delta``) that does
    not reuse the algebraic form of ``uhat``.
    """
    delta = stokes_layer(omega, nu=nu)
    y = np.linspace(lo * delta, height / 2.0 - hi * delta, 80)
    if y[0] >= y[-1]:
        raise ValueError("channel too shallow for the selected Womersley number")
    deficit = np.abs(womersley_complex(height / 2.0, omega=omega, g0=g0, nu=nu, height=height)
                     - womersley_complex(y, omega=omega, g0=g0, nu=nu, height=height))
    slope, _ = np.polyfit(y, np.log(deficit), 1)
    return dict(delta=delta, fitted_slope=float(slope), exact_slope=float(-1.0 / delta),
                relative_error=float(abs(slope * delta + 1.0)), n_fit=len(y))


# ------------------------------------------------------------------------- driver
def solve_pulsatile(*, womersley: float, ny: int = 64, nx: int = 4,
                    dt_fraction: float = 1.0 / 300.0, n_cycles: float = 3.0,
                    n_phases: int = 8, g0: float = G0, nu: float = NU,
                    height: float = H, rho: float = RHO,
                    solver_class=PeriodicChannelSolver) -> dict:
    """Integrate the production solver and sample ``n_phases`` within the last period.

    ``dt_fraction`` is ``dt / T``.  The initial condition is the *exact* periodic
    solution at ``t = 0`` (so no start-up transient has to decay); ``n_cycles``
    periods are integrated and the profile is sampled every ``T / n_phases``
    during the final one.
    """
    omega = (womersley / height) ** 2 * nu
    period = 2.0 * math.pi / omega
    dt = dt_fraction * period
    cfg = PeriodicChannelConfig(nx=nx, ny=ny, length_m=2.0 * height, height_m=height,
                                density_kg_m3=rho, viscosity_pa_s=nu * rho,
                                body_force_m_s2=g0, time_step_s=dt)
    solver = solver_class(cfg)
    y = (np.arange(ny) + 0.5) * (height / ny)
    solver.u = _torch_from(womersley_profile(y, 0.0, omega=omega, g0=g0, nu=nu,
                                             height=height), ny, nx)
    steps_per_phase = max(1, int(round(period / n_phases / dt)))
    steps_per_cycle = steps_per_phase * n_phases
    sample_start = int(round(n_cycles * steps_per_cycle))
    total_steps = sample_start + steps_per_phase * n_phases

    samples = []
    t_wall = time.time()
    step = 0
    while step < total_steps:
        t_mid = (step + 0.5) * dt
        solver.step(g0 * math.cos(omega * t_mid))
        step += 1
        # exactly n_phases samples, and *not* the duplicate of the first one at
        # t + T: a repeated sample biases the single-frequency DFT estimate of the
        # amplitude low by O(1/n_phases)
        if step >= sample_start and (step - sample_start) % steps_per_phase == 0 \
                and step < total_steps:
            samples.append(dict(time=solver.time,
                                u=solver.u[:, 0].numpy().copy(),
                                bulk=solver.bulk_velocity(),
                                shear_lower=solver.wall_shear("quadratic")[0],
                                max_v=float(solver.v.abs().max()),
                                max_div=solver.max_divergence()))
    wall = time.time() - t_wall
    return dict(womersley=womersley, omega=omega, period=period, delta=stokes_layer(omega, nu=nu),
                ny=ny, nx=nx, dt=dt, steps=int(total_steps), seconds=wall, y=y,
                samples=samples, g0=g0, nu=nu, height=height)


def _torch_from(array, ny, nx):
    """Place a ``y`` profile on the solver's ``(ny, nx)`` x-uniform collocation."""
    import torch
    return torch.tensor(np.repeat(np.asarray(array)[:, None], nx, axis=1),
                        dtype=torch.float64)


def profile_error(run: dict) -> dict:
    """Relative L2 error and per-phase extremes against the closed form."""
    y, omega = run["y"], run["omega"]
    kw = dict(omega=omega, g0=run["g0"], nu=run["nu"], height=run["height"])
    exact = [womersley_profile(y, sample["time"], **kw) for sample in run["samples"]]
    # normalise by the L2 envelope of the exact solution over the cycle, not by the
    # per-phase norm: near the zero crossing of u(t) the per-phase norm tends to
    # zero and would turn a small absolute error into a meaningless 100 % one.
    envelope = max(float(np.linalg.norm(u)) for u in exact)
    errors = [float(np.linalg.norm(sample["u"] - u) / envelope)
              for sample, u in zip(run["samples"], exact)]
    return dict(phases=len(errors), max_relative_l2=float(max(errors)),
                mean_relative_l2=float(np.mean(errors)), errors=errors)


def amplitude_phase(samples, *, omega: float, quantity: str) -> dict:
    """Complex amplitude of a sampled time series at the forcing frequency."""
    t = np.array([s["time"] for s in samples])
    v = np.array([s[quantity] for s in samples])
    basis = np.exp(-1j * omega * t)
    amp = 2.0 * np.sum(v * basis) / len(v)
    return dict(amplitude=float(abs(amp)), phase_rad=float(np.angle(amp)),
                phase_deg=float(np.rad2deg(np.angle(amp))))


def exact_amplitude_phase(*, omega: float, quantity: str, g0: float = G0,
                          nu: float = NU, height: float = H) -> dict:
    """Complex amplitude of the closed form at ``omega`` (no time series needed)."""
    m = np.sqrt(1j * omega / nu)
    if quantity == "bulk":
        uhat = g0 / (1j * omega) * (1.0 - np.tanh(m * height / 2.0) / (m * height / 2.0))
    elif quantity == "shear_lower":          # mu du/dy at the wall, mu = rho nu = nu here
        uhat = nu * g0 / (1j * omega) * m * np.tanh(m * height / 2.0)
    else:
        raise ValueError(f"unknown quantity {quantity!r}")
    return dict(amplitude=float(abs(uhat)), phase_rad=float(np.angle(uhat)),
                phase_deg=float(np.rad2deg(np.angle(uhat))))


def wall_gradient_number(nu: float = NU) -> float:
    """Dynamic viscosity ``mu = rho nu`` used by the solver's wall-shear diagnostic."""
    return RHO * nu


# ---------------------------------------------------------------------- criteria
def _criterion(cid, name, value, threshold, mode="lt", unit="", note=""):
    ok = (value < threshold) if mode == "lt" else (
        (value > threshold) if mode == "gt" else (abs(value) < threshold))
    return dict(id=cid, name=name, value=float(value), threshold=float(threshold),
                mode=mode, unit=unit, verdict="PASS" if ok else "FAIL", note=note)


def run(ny_main: int = 64, dt_fraction: float = 1.0 / 200.0, n_cycles: float = 2.0,
        ny_wall: int = 96, ladder=(1.0 / 100.0, 1.0 / 200.0, 1.0 / 400.0)) -> dict:
    """Execute all pre-declared criteria and return the summary payload."""
    t_start = time.time()
    criteria = []

    # --- A1 closed form satisfies the momentum equation
    res = closed_form_residual(omega=1.0)
    criteria.append(_criterion("A1", "closed-form PDE residual (numerical derivatives)",
                               res["relative"], 1e-6, unit="rel. to G0",
                               note=f"max|residual|={res['max_abs_residual']:.3e} m/s^2"))

    # --- A2 steady limit
    omega_small = (1.0 / H) ** 2 * NU * 1e-6
    y_probe = np.linspace(0.0, H, 201)
    u_lim = womersley_profile(y_probe, 0.0, omega=omega_small)
    u_steady = steady_poiseuille(y_probe)
    centre_err = abs(u_lim.max() / u_steady.max() - 1.0)
    bulk_lim = womersley_bulk(0.0, omega=omega_small)
    bulk_steady = G0 * H ** 2 / (12.0 * NU)
    # round-off floor: the limit is evaluated through 1 - cosh(m z)/cosh(m H/2) with
    # |m H|^2 = i omega H^2/nu = i 1e-6, so the cancellation costs ~eps/1e-6 = 1e-10
    criteria.append(_criterion("A2", "omega->0 limit equals the analytic parabola",
                               max(centre_err, abs(bulk_lim / bulk_steady - 1.0)), 1e-6,
                               unit="rel.", note=f"centre {u_lim.max():.9f} vs {u_steady.max():.9f}"))

    # --- A3 bulk / wall-gradient formulas vs numerical integration/differentiation
    omega = (5.0 / H) ** 2 * NU
    worst = 0.0
    for phase in (0.0, 0.25, 0.5, 0.75):
        t = phase * 2.0 * math.pi / omega
        y_num = np.linspace(0.0, H, 20001)
        u_num = womersley_profile(y_num, t, omega=omega)
        u_b_num = float(np.sum(0.5 * (u_num[:-1] + u_num[1:])) * (y_num[1] - y_num[0])) / H
        worst = max(worst, abs(u_b_num / womersley_bulk(t, omega=omega) - 1.0))
        h = 1e-6
        grad_num = (womersley_profile(h, t, omega=omega)
                    - womersley_profile(-h, t, omega=omega)) / (2 * h)
        worst = max(worst, abs(grad_num / womersley_wall_gradient(t, omega=omega) - 1.0))
    criteria.append(_criterion("A3", "bulk / wall-gradient formulas vs numeric",
                               worst, 1e-6, unit="rel."))

    # --- A4 near-wall decay law
    decays = [near_wall_decay_slope(omega=(W / H) ** 2 * NU) for W in (40.0, 80.0)]
    criteria.append(_criterion("A4", "Stokes-layer decay slope vs -1/delta",
                               max(d["relative_error"] for d in decays), 0.02, unit="rel.",
                               note="; ".join(f"W={W:g}: slope*delta={d['fitted_slope'] * d['delta']:.5f}"
                                              for W, d in zip((40.0, 80.0), decays))))

    # --- A5 steady regression of the production solver (constant forcing path)
    steady_check = steady_poiseuille_channel(ny=64)
    criteria.append(_criterion("A5", "steady solver bulk velocity vs G0 H^2/(12 nu)",
                               steady_check["bulk_relative_error"], 1e-3, unit="rel.",
                               note=f"bulk_num={steady_check['bulk']:.9f} m/s"))

    # --- B1..B3 solver vs closed form at three Womersley numbers
    mains = {}
    for cid, W in zip(("B1", "B2", "B3"), (2.0, 5.0, 10.0)):
        main_run = solve_pulsatile(womersley=W, ny=ny_main, dt_fraction=dt_fraction,
                                   n_cycles=n_cycles)
        mains[W] = main_run
        err = profile_error(main_run)
        criteria.append(_criterion(cid, f"solver vs closed form, W={W:g}, ny={ny_main}, "
                                        f"dt/T=1/{round(1 / dt_fraction)}",
                                   err["max_relative_l2"], 0.03, unit="rel. L2",
                                   note=f"{err['phases']} phases, {main_run['steps']} steps, "
                                        f"{main_run['seconds']:.1f} s, "
                                        f"delta/H={main_run['delta'] / H:.3f}"))

    run5 = mains[5.0]
    kw5 = dict(omega=run5["omega"])

    # --- B4 bulk amplitude and phase
    num_b = amplitude_phase(run5["samples"], quantity="bulk", **kw5)
    ex_b = exact_amplitude_phase(quantity="bulk", **kw5)
    amp_err = abs(num_b["amplitude"] / ex_b["amplitude"] - 1.0)
    phase_err = abs(math.degrees(math.remainder(num_b["phase_rad"] - ex_b["phase_rad"],
                                                2 * math.pi)))
    criteria.append(_criterion("B4a", "bulk-velocity amplitude, W=5", amp_err, 0.03, unit="rel.",
                               note=f"{num_b['amplitude']:.6f} vs {ex_b['amplitude']:.6f} m/s"))
    criteria.append(_criterion("B4b", "bulk-velocity phase lag, W=5", phase_err, 5.0, unit="deg",
                               note=f"{num_b['phase_deg']:.3f} vs {ex_b['phase_deg']:.3f} deg"))

    # --- B5 wall shear amplitude at the finest grid
    wall_run = solve_pulsatile(womersley=5.0, ny=ny_wall, dt_fraction=dt_fraction,
                               n_cycles=n_cycles)
    num_tau = amplitude_phase(wall_run["samples"], quantity="shear_lower", **kw5)
    ex_tau = exact_amplitude_phase(quantity="shear_lower", **kw5)
    tau_err = abs(num_tau["amplitude"] / ex_tau["amplitude"] - 1.0)
    flux_err = wall_shear_flux_error(wall_run)
    criteria.append(_criterion("B5", f"wall-shear amplitude, W=5, ny={ny_wall} (3-point fit)",
                               tau_err, 0.03, unit="rel.",
                               note=f"{num_tau['amplitude']:.6e} vs {ex_tau['amplitude']:.6e} Pa; "
                                    f"scheme-consistent wall flux (not graded) err={flux_err:.3e}"))

    # --- B6 temporal convergence
    ladder_runs = []
    for frac in ladder:
        r = solve_pulsatile(womersley=5.0, ny=ny_main, dt_fraction=frac, n_cycles=n_cycles)
        ladder_runs.append((frac, profile_error(r)["max_relative_l2"], r["steps"]))
    order = (math.log(ladder_runs[0][1] / ladder_runs[-1][1])
             / math.log(ladder_runs[0][0] / ladder_runs[-1][0])
             if ladder_runs[-1][1] > 0 else 0.0)
    monotone = all(a[1] > b[1] for a, b in zip(ladder_runs, ladder_runs[1:]))
    criteria.append(_criterion("B6", "temporal convergence order at W=5",
                               order if monotone else 0.0, 0.7, mode="gt", unit="p",
                               note="; ".join(f"dt/T=1/{round(1 / f)}: {e:.3e} ({n} steps)"
                                              for f, e, n in ladder_runs)))

    # --- B7 wall-normal convergence
    coarse = solve_pulsatile(womersley=5.0, ny=ny_main // 2, dt_fraction=dt_fraction,
                             n_cycles=n_cycles)
    e_coarse = profile_error(coarse)["max_relative_l2"]
    e_fine = profile_error(mains[5.0])["max_relative_l2"]
    criteria.append(_criterion("B7", "wall-normal convergence ny/2 -> ny at W=5",
                               1.0 if e_fine < e_coarse else 0.0, 0.5, mode="gt", unit="flag",
                               note=f"ny={ny_main // 2}: {e_coarse:.3e} -> ny={ny_main}: {e_fine:.3e}"))

    # --- B8 projection cleanliness
    max_v = max(s["max_v"] for s in run5["samples"])
    max_div = max(s["max_div"] for s in run5["samples"])
    criteria.append(_criterion("B8a", "spanwise/normal velocity stays zero at W=5", max_v, 1e-12,
                               unit="m/s"))
    criteria.append(_criterion("B8b", "cell divergence after projection at W=5", max_div, 1e-10,
                               unit="1/s"))

    # --- C1 discrimination: oscillatory profile vs steady parabola
    y_probe = run5["y"]
    steady = steady_poiseuille(y_probe)
    gap = max(float(np.linalg.norm(womersley_profile(y_probe, s["time"], **kw5) - steady)
                    / np.linalg.norm(steady)) for s in run5["samples"])
    criteria.append(_criterion("C1", "oscillatory vs steady profile difference (W=5)",
                               gap, 0.10, mode="gt", unit="rel. L2",
                               note="a steady solver cannot pass B1..B3"))

    n_pass = sum(1 for c in criteria if c["verdict"] == "PASS")
    payload = dict(
        generated=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        module="tensorfvm.verification.pulsatile_channel",
        capability="PeriodicChannelSolver under time-dependent (pulsatile) forcing",
        parameters=dict(nu=NU, height=H, g0=G0, rho=RHO, ny_main=ny_main,
                        dt_fraction=dt_fraction, n_cycles=n_cycles, ny_wall=ny_wall),
        criteria=criteria, n_pass=n_pass, n_total=len(criteria),
        steady_regression=steady_check,
        runs={str(W): dict(womersley=r["womersley"], delta=r["delta"], dt=r["dt"],
                           steps=r["steps"], seconds=r["seconds"],
                           max_relative_l2=profile_error(r)["max_relative_l2"],
                           errors=profile_error(r)["errors"]) for W, r in mains.items()},
        wall_shear=dict(numeric=num_tau, exact=ex_tau,
                        scheme_flux_relative_error=flux_err),
        wall_shear_run=dict(ny=wall_run["ny"], dt=wall_run["dt"]),
        temporal_ladder=[dict(dt_fraction=f, max_relative_l2=e, steps=n) for f, e, n in ladder_runs],
        wall_normal=dict(coarse=e_coarse, fine=e_fine, ny_coarse=ny_main // 2, ny_fine=ny_main),
        runtime_s=time.time() - t_start,
    )
    payload["all_pass"] = n_pass == len(criteria)
    payload["_runs"] = mains
    payload["_wall_run"] = wall_run
    return payload


def steady_poiseuille_channel(*, ny: int = 64, max_steps: int = 40000) -> dict:
    """Production solver with constant forcing: bulk velocity vs ``G0 H^2/(12 nu)``."""
    cfg = PeriodicChannelConfig(nx=4, ny=ny, length_m=2.0 * H, height_m=H,
                               density_kg_m3=RHO, viscosity_pa_s=NU * RHO,
                               body_force_m_s2=G0,
                               time_step_s=0.02 * H ** 2 / NU)
    solver = PeriodicChannelSolver(cfg)
    target = G0 * H ** 2 / (12.0 * NU)
    change = math.inf
    for _ in range(max_steps):
        record = solver.step()            # constant configured forcing, unchanged path
        change = record["velocity_change_m_s"]
        if change < 1e-13:
            break
    bulk = solver.bulk_velocity()
    return dict(bulk=bulk, target=target, bulk_relative_error=abs(bulk / target - 1.0),
                steps=len(solver.history), final_change=change)


def wall_shear_flux_error(run: dict) -> float:
    """Scheme-consistent (first-order) wall flux vs the exact wall-gradient amplitude.

    The shipped scheme closes the momentum balance with ``mu * 2 u_0 / dy`` (the
    "half-cell" ghost), which is only first-order accurate at the wall.  This is
    reported to separate *wall-quadrature* error from the interior accuracy that
    B1..B5 grade.
    """
    exact = exact_amplitude_phase(quantity="shear_lower", omega=run["omega"])
    mu = wall_gradient_number()
    y0 = run["y"][0]
    flux = [mu * sample["u"][0] / y0 for sample in run["samples"]]   # mu*2u_0/dy = mu*u_0/y_0
    t = np.array([s["time"] for s in run["samples"]])
    amp = abs(2.0 * np.sum(np.array(flux) * np.exp(-1j * run["omega"] * t)) / len(flux))
    return float(abs(amp / exact["amplitude"] - 1.0))


# ------------------------------------------------------------------------ figure
def figure(payload: dict, path: Path) -> Path:
    """Three panels: profiles vs exact, temporal ladder, amplitude/phase vs W."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.8))
    colors = {2.0: "tab:blue", 5.0: "tab:orange", 10.0: "tab:green"}
    ax = axes[0]
    for W, run_data in payload["_runs"].items():
        y = run_data["y"]
        peak = max(run_data["samples"], key=lambda s: abs(s["bulk"]))
        u_exact = womersley_profile(y, peak["time"], omega=run_data["omega"])
        ax.plot(u_exact, y, "-", color=colors[float(W)], lw=1.6,
                label=f"exact W={float(W):g}")
        ax.plot(peak["u"], y, "o", color=colors[float(W)], ms=3.0, mfc="none")
    ax.set_xlabel("u / (m/s)")
    ax.set_ylabel("y / H")
    ax.set_title("peak-phase profile: solver (o) vs closed form (-)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    lad = payload["temporal_ladder"]
    dts = [1.0 / d["dt_fraction"] for d in lad]
    errs = [d["max_relative_l2"] for d in lad]
    ax.loglog(dts, errs, "o-", label="solver error")
    if len(dts) >= 2:
        slope = math.log(errs[0] / errs[-1]) / math.log(dts[-1] / dts[0])
        ref = errs[0] * (np.array(dts) / dts[0]) ** -slope
        ax.loglog(dts, ref, "--", label=f"fit p={slope:.2f}")
    ax.set_xlabel("T / dt")
    ax.set_ylabel("max rel. L2 profile error")
    ax.set_title("temporal convergence at W=5")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axes[2]
    Ws = [2.0, 5.0, 10.0, 20.0]
    amp_num, amp_ex = [], []
    for W in Ws:
        omega = (W / H) ** 2 * NU
        amp_ex.append(exact_amplitude_phase(quantity="bulk", omega=omega)["amplitude"])
        if str(W) in payload["_runs"]:
            amp_num.append(amplitude_phase(payload["_runs"][W]["samples"], quantity="bulk",
                                           omega=omega)["amplitude"])
        else:
            amp_num.append(float("nan"))
    ax.semilogx(Ws, amp_ex, "-", label="exact bulk amplitude")
    ax.semilogx(Ws, amp_num, "o", label="solver")
    ax.set_xlabel("Womersley number W = H/delta")
    ax.set_ylabel("bulk velocity amplitude / (m/s)")
    ax.set_title("amplitude roll-off with W (solver vs closed form)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_artifacts(outdir: Path | str = "docs/verification-pulsatile-channel", **kwargs) -> dict:
    """Run every criterion, then write ``summary.json`` + the figure."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    payload = run(**kwargs)
    public = {k: v for k, v in payload.items() if not k.startswith("_")}
    (outdir / "summary.json").write_text(json.dumps(public, indent=2, default=float))
    figure(payload, outdir / "fig_pulsatile_channel.png")
    return public


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outdir", default="docs/verification-pulsatile-channel")
    parser.add_argument("--ny", type=int, default=48)
    parser.add_argument("--dt-fraction", type=float, default=1.0 / 200.0)
    parser.add_argument("--ny-wall", type=int, default=96)
    args = parser.parse_args(argv)
    payload = write_artifacts(args.outdir, ny_main=args.ny,
                              dt_fraction=args.dt_fraction, ny_wall=args.ny_wall)
    for crit in payload["criteria"]:
        print(f"{crit['id']:5s} {crit['verdict']:4s} {crit['value']:.6e} "
              f"({'<' if crit['mode'] == 'lt' else '>'} {crit['threshold']:.3e}) "
              f"{crit['name']}")
    print(f"{payload['n_pass']}/{payload['n_total']} PASS   "
          f"all_pass={payload['all_pass']}   {payload['runtime_s']:.1f} s")
    return 0 if payload["all_pass"] else 1


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(main())