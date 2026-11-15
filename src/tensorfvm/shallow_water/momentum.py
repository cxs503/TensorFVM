"""Momentum identity and velocity-field error budget for the dam-break anchor.

Motivation
----------
The case-pool entry ``CASE-FVM-DAMBREAK-STOKER`` grades six quantities
(G1-G6): fan/bore position, plateau ratio, depth L1, the ``x_f(t)`` series and
the absence of spurious oscillations.  **Every one of them is a functional of
the depth field.**  Neither the global momentum balance nor the velocity field
is graded, so a solver (or a new discretisation) can reproduce the whole gate
list while getting ``u`` wrong.  This module supplies the two missing
functionals and, for the dry bed, an *explained* error budget:

1. :func:`exact_momentum` -- the global momentum of the exact solution has a
   one-line closed form.  Both states are at rest and the waves never reach the
   boundaries, so the only momentum sources are the hydrostatic pressure fluxes
   at the two end faces::

       dM/dt = g/2 (h_L^2 - h_R^2)   =>   M(t) = M(0) + g/2 (h_L^2 - h_R^2) t

   with ``(h_L, h_R)`` the *undisturbed* boundary depths.  ``M(0) = 0`` for the
   anchor (both sides at rest).  This is exact for the PDE, not a fit: it is
   certified against a fine quadrature of the exact ``h u`` profile by
   :func:`momentum_quadrature`.

2. :func:`velocity_budget` -- decomposes ``int |u - u_exact| dx`` into the four
   regions that carry it:

   * ``missing_tip``: cells the exact solution has wet but the solver keeps dry
     (``h_num == 0``).  On the dry bed the exact velocity at the wash limit is
     ``u -> 2 c0`` (fastest water, zero depth), so this sliver carries O(c0)
     velocity and **cannot be neglected in a velocity norm** even though its
     mass is nil;
   * ``overshoot``: cells the solver wets where the exact solution is dry;
   * ``depth_inherited``: inside the fan, the exact similarity solution obeys
     ``u = 2 c0 - 2 sqrt(g h)`` (:func:`similarity_velocity`, certified to
     machine precision), so a depth error ``dh`` produces a velocity error
     ``O(sqrt(dh))`` -- the square-root law;
   * ``scheme``: the residual mismatch between the numerical ``u`` and the
     value implied by the numerical depth through that same identity.

   ``missing_tip + overshoot + depth_inherited + scheme`` bounds the measured
   error (they coincide on disjoint cells; inside the fan the triangle
   inequality is slack only by cancellation).

Both functionals are declared *before* the run (module constants) and the
verdict is computed from the artifacts, not typed.

Scope / assumptions
-------------------
* Flat bed, no friction, no Coriolis.  ``swe1d`` also supports Manning friction
  and a constant slope, but then the momentum identity above is **not** the
  right reference (friction is a momentum sink) -- the guard in
  :func:`momentum_closure` reports the active source terms so a caller cannot
  silently compare apples to oranges.
* The identity requires both end faces to stay undisturbed for the whole run;
  :func:`undisturbed_guard` checks that explicitly.
* :func:`similarity_velocity` is the Ritter (dry-bed) fan identity.  It also
  holds inside the *left rarefaction* of the Stoker wet-bed solution, but not
  across the bore, so the decomposition is only applied to ``kind == "dry"``.
"""

from __future__ import annotations

import numpy as np

from .dam_break import CASES, RITTER_DRY, STOKER_WET, DamBreakCase
from .exact import gravity_wave_speed, ritter_head_position, stoker_constant_state
from .swe1d import ShallowWater1D

__all__ = [
    "TOLERANCE",
    "TIP_FRACTION",
    "UNDISTURBED_MARGIN",
    "boundary_depths",
    "exact_momentum_rate",
    "exact_momentum",
    "momentum_quadrature",
    "similarity_velocity",
    "similarity_identity_residual",
    "undisturbed_guard",
    "momentum_closure",
    "velocity_budget",
    "velocity_resolution_study",
    "defect_injection_control",
]

#: gate tolerance of the case-pool anchor (``<5 %``), declared before any run
TOLERANCE = 0.05
#: tip band used by the region split: ``0 < h_exact <= TIP_FRACTION * h0``
TIP_FRACTION = 0.05
#: the waves must stay this far (m) inside both end faces for the identity
UNDISTURBED_MARGIN = 0.5


# --------------------------------------------------------------------------- #
# exact side
# --------------------------------------------------------------------------- #
def boundary_depths(case: DamBreakCase) -> tuple[float, float]:
    """Undisturbed depths at the two end faces ``(h_left, h_right)``."""
    return (float(case.h0), float(case.h1))


def exact_momentum_rate(case: DamBreakCase) -> float:
    """``dM/dt = g/2 (h_L^2 - h_R^2)`` (m^3/s^2), exact for the flat-bed PDE."""
    hl, hr = boundary_depths(case)
    return 0.5 * case.gravity * (hl * hl - hr * hr)


def exact_momentum(case: DamBreakCase, t: float | None = None) -> float:
    """Global momentum ``int h u dx`` of the exact solution at time ``t``.

    ``M(t) = M(0) + rate * t``.  ``M(0) = 0`` because the anchor initial
    condition is at rest on both sides (the solver's initial state is checked in
    :func:`momentum_closure`).
    """
    t = case.t_end if t is None else float(t)
    return exact_momentum_rate(case) * t


def momentum_quadrature(case: DamBreakCase, t: float | None = None,
                        n_quad: int = 200_000) -> float:
    """Midpoint quadrature of the exact ``h u`` profile over the domain.

    This certifies :func:`exact_momentum` without re-using its derivation.  The
    bore in the wet case is a genuine discontinuity, so the quadrature error is
    O(dx) there; ``n_quad`` is chosen fine enough that the residual is orders of
    magnitude below any gate tolerance.
    """
    t = case.t_end if t is None else float(t)
    x = np.linspace(case.x_min, case.x_max, int(n_quad), endpoint=False)
    x = x + 0.5 * (x[1] - x[0]) if n_quad > 1 else x
    h, u = case.exact(x, t)
    dx = (case.x_max - case.x_min) / float(n_quad)
    return float(np.sum(np.asarray(h) * np.asarray(u)) * dx)


def similarity_velocity(h, h0: float, gravity: float = 9.81):
    """Ritter fan identity ``u = 2 c0 - 2 sqrt(g h)`` (m/s).

    Derivation: in the fan ``h = (2 c0 - x/t)^2 / (9 g)`` and
    ``u = (2/3)(x/t + c0)``; eliminating ``x/t`` gives ``u = 2 c0 - 2 sqrt(g h)``.
    The map is *not* smooth at ``h = 0``: ``du/dh -> -inf``, which is exactly why
    a depth error turns into a square-root velocity error at the wash limit.
    """
    c0 = gravity_wave_speed(h0, gravity)
    return 2.0 * c0 - 2.0 * np.sqrt(gravity * np.maximum(np.asarray(h, dtype=float), 0.0))


def similarity_identity_residual(case: DamBreakCase = RITTER_DRY,
                                 n: int = 4001) -> float:
    """Max ``|u_exact - similarity_velocity(h_exact)|`` over the exact fan."""
    x = np.linspace(case.x_min, case.x_max, n)
    h, u = case.exact(x, case.t_end)
    h = np.asarray(h, dtype=float)
    u = np.asarray(u, dtype=float)
    wet = h > 0.0
    if not wet.any():
        return 0.0
    ref = similarity_velocity(h[wet], case.h0, case.gravity)
    return float(np.max(np.abs(u[wet] - ref)))


def undisturbed_guard(case: DamBreakCase, t: float | None = None,
                      margin: float = UNDISTURBED_MARGIN) -> dict:
    """Check that no wave reaches (or comes within ``margin`` of) a boundary.

    Returns the wave positions and a boolean ``undisturbed``; a failed guard
    means both the momentum identity and the "undisturbed far field" metrics of
    the anchor do not apply at that time.
    """
    t = case.t_end if t is None else float(t)
    c0 = gravity_wave_speed(case.h0, case.gravity)
    if case.kind == "dry":
        head = ritter_head_position(t, case.h0, case.gravity, case.x_dam)
        speed = 2.0 * c0
        front = case.x_dam + speed * t
        left, right = float(head), float(front)
    else:
        sol = stoker_constant_state(case.h0, case.h1, case.gravity)
        left = float(case.x_dam + sol["head_speed"] * t)
        right = float(case.x_dam + sol["shock_speed"] * t)
    clear_left = left - case.x_min
    clear_right = case.x_max - right
    return {
        "case": case.key,
        "t": t,
        "left_front": left,
        "right_front": right,
        "clearance_left": clear_left,
        "clearance_right": clear_right,
        "margin": margin,
        "undisturbed": bool(clear_left >= margin and clear_right >= margin),
    }


# --------------------------------------------------------------------------- #
# numerical side
# --------------------------------------------------------------------------- #
def momentum_closure(case: DamBreakCase, n: int = 1600, cfl: float = 0.4,
                     order: int = 2, t_star: float | None = None,
                     n_quad: int = 200_000) -> dict:
    """Run one case and close the global momentum balance against the exact value.

    Returns the numerical ``M``, the exact ``M`` (closed form) and its
    independent quadrature, plus the guard/source-terms bookkeeping so the
    comparison is only made where it is meaningful.
    """
    t_star_end = case.t_star_end if t_star is None else float(t_star)
    from .exact import t_star_to_time

    t_end = t_star_to_time(t_star_end, case.h0, case.gravity)
    solver = ShallowWater1D(n, case.x_min, case.x_max, gravity=case.gravity,
                            cfl=cfl, order=order)
    initial = np.where(solver.x < case.x_dam, case.h0, case.h1)
    solver.set_state(initial)
    momentum0 = solver.momentum()
    solver.advance(t_end)
    momentum1 = solver.momentum()
    exact = exact_momentum(case, solver.time) + momentum0
    quad = momentum_quadrature(case, solver.time, n_quad=n_quad) + momentum0
    guard = undisturbed_guard(case, solver.time)
    denom = max(abs(exact), 1e-300)
    return {
        "case": case.key,
        "n": n,
        "dx": solver.dx,
        "cfl": cfl,
        "order": order,
        "t": solver.time,
        "steps": solver.steps,
        "momentum_initial": momentum0,
        "momentum_numeric": momentum1,
        "momentum_exact_closed_form": exact,
        "momentum_exact_quadrature": quad,
        "rate_closed_form": exact_momentum_rate(case),
        "closure_error": abs(momentum1 - exact) / denom,
        "quadrature_vs_closed_form": abs(quad - exact) / denom,
        "guard": guard,
        "sources": {"manning_n": solver.manning_n, "slope": solver.slope},
        "identity_applies": bool(guard["undisturbed"]
                                 and solver.manning_n == 0.0
                                 and solver.slope == 0.0),
    }


def velocity_budget(case: DamBreakCase, x, h, u, tip_fraction: float = TIP_FRACTION,
                    h0: float | None = None) -> dict:
    """Split ``int |u - u_exact| dx`` (relative L1) into four additive regions.

    ``h``/``u`` are the numerical cell values, ``x`` the cell centres; the exact
    profile is evaluated at the same points.  All terms are normalised by
    ``int |u_exact| dx`` so the numbers are dimensionless relative errors.
    """
    h0 = case.h0 if h0 is None else float(h0)
    x = np.asarray(x, dtype=float)
    h = np.asarray(h, dtype=float)
    u = np.asarray(u, dtype=float)
    h_ex, u_ex = case.exact(x)
    h_ex = np.asarray(h_ex, dtype=float)
    u_ex = np.asarray(u_ex, dtype=float)
    dx = float(x[1] - x[0]) if x.size > 1 else 1.0

    num_wet = h > 0.0
    ex_wet = h_ex > 0.0
    denom = float(np.sum(np.abs(u_ex)) * dx)
    if denom <= 0.0:
        raise ValueError("the exact velocity field is identically zero")

    missing = ex_wet & ~num_wet
    overshoot = num_wet & ~ex_wet
    inner = num_wet & ex_wet
    band = ex_wet & (h_ex <= tip_fraction * h0)

    u_pred = similarity_velocity(h, h0, case.gravity)
    term_missing = float(np.sum(np.abs(u_ex[missing])) * dx) / denom
    term_overshoot = float(np.sum(np.abs(u[overshoot])) * dx) / denom
    term_depth = float(np.sum(np.abs(u_pred[inner] - u_ex[inner])) * dx) / denom
    term_scheme = float(np.sum(np.abs(u[inner] - u_pred[inner])) * dx) / denom

    measured = float(np.sum(np.abs(u - u_ex)) * dx) / denom
    bound = term_missing + term_overshoot + term_depth + term_scheme
    # mass weighted companion norm (what momentum/energy fluxes see)
    measured_mass = float(np.sum(np.abs(u - u_ex) * h_ex) * dx) / max(
        float(np.sum(np.abs(u_ex) * h_ex) * dx), 1e-300)
    return {
        "case": case.key,
        "dx": dx,
        "tip_fraction": tip_fraction,
        "u_l1": measured,
        "u_l1_mass": measured_mass,
        "hu_l1": float(np.sum(np.abs(h * u - h_ex * u_ex)) * dx) / max(
            float(np.sum(np.abs(h_ex * u_ex)) * dx), 1e-300),
        "depth_l1": float(np.sum(np.abs(h - h_ex)) * dx) / max(
            float(np.sum(h_ex) * dx), 1e-300),
        "budget": {
            "missing_tip": term_missing,
            "overshoot": term_overshoot,
            "depth_inherited": term_depth,
            "scheme": term_scheme,
            "sum_bound": bound,
            "measured": measured,
            "cancellation": bound - measured,
        },
        "share_missing_tip": term_missing / bound if bound > 0 else 0.0,
        "share_band": float(np.sum(np.abs(u - u_ex)[band]) * dx) / max(
            float(np.sum(np.abs(u - u_ex)) * dx), 1e-300),
        "cells": {
            "missing_tip": int(missing.sum()),
            "overshoot": int(overshoot.sum()),
            "fan": int(inner.sum()),
            "band": int(band.sum()),
        },
        "gap_length": float(missing.sum() * dx),
        "u_max_numeric": float(np.max(np.abs(u))),
        "u_max_exact": float(np.max(np.abs(u_ex))),
        "tolerance": TOLERANCE,
        "pass_5pct": measured <= TOLERANCE,
        "pass_3pct": measured <= 0.03,
    }


def velocity_resolution_study(case: DamBreakCase, resolutions=(800, 1600, 3200, 6400),
                              cfl: float = 0.4, order: int = 2) -> dict:
    """Refine the grid and report the observed order of both velocity norms."""
    rows: list[dict] = []
    for n in resolutions:
        solver = ShallowWater1D(int(n), case.x_min, case.x_max,
                                gravity=case.gravity, cfl=cfl, order=order)
        solver.set_state(np.where(solver.x < case.x_dam, case.h0, case.h1))
        solver.advance(case.t_end)
        h, u = solver.primitive()
        entry = velocity_budget(case, solver.x, h, u)
        entry["n"] = int(n)
        entry["steps"] = solver.steps
        entry["clipped_cells"] = solver.clipped_cells
        entry["depth_l1"] = float(
            np.sum(np.abs(h - np.asarray(case.exact(solver.x)[0]))) * solver.dx) / max(
            float(np.sum(np.asarray(case.exact(solver.x)[0])) * solver.dx), 1e-300)
        rows.append(entry)

    def orders(key: str) -> list[float]:
        out = []
        for a, b in zip(rows, rows[1:]):
            ea, eb = a[key], b[key]
            if ea <= 0.0 or eb <= 0.0:
                out.append(0.0)
            else:
                out.append(float(np.log(ea / eb) / np.log(b["n"] / a["n"])))
        return out

    return {
        "case": case.key,
        "cfl": cfl,
        "order": order,
        "rows": rows,
        "observed_order_u_l1": orders("u_l1"),
        "observed_order_u_l1_mass": orders("u_l1_mass"),
        "observed_order_depth_l1": orders("depth_l1"),
    }


def defect_injection_control(case: DamBreakCase, n: int = 800,
                             epsilons=(1e-3, 1e-2, 1e-1),
                             window: tuple[float, float] | None = None) -> dict:
    """Show both detectors are non-empty, with a *predicted* response.

    A known relative defect is injected into the converged profile in a window
    that excludes the wetting tip (default: the last metre of the fan upstream
    of the dam, ``[x_dam - 1, x_dam]``, where the baseline error is small), so
    the response cannot hide behind the existing bias:

    * **momentum channel** -- scale ``hu`` by ``(1 + eps)`` in the window; the
      closure error the identity must report is analytically
      ``eps * int_W |h u| dx / |M_exact|`` (the solver's own closure is
      1e-12, so any ``eps >= 1e-5`` is detectable).  ``gain`` = measured /
      predicted must be 1.
    * **velocity channel** -- add ``+/- eps * c0`` to ``u`` in the same window
      and report the change of the L1 error for *both* signs; a defect of
      either sign is detected, so the control reports ``max`` over the signs
      (a single sign can cancel the baseline bias -- that is reported, not
      hidden).
    """
    from .exact import t_star_to_time

    solver = ShallowWater1D(int(n), case.x_min, case.x_max,
                            gravity=case.gravity, cfl=0.4, order=2)
    initial = np.where(solver.x < case.x_dam, case.h0, case.h1)
    solver.set_state(initial)
    momentum0 = solver.momentum()
    solver.advance(case.t_end)
    h, u = solver.primitive()
    x = solver.x
    dx = solver.dx
    if window is None:
        lo, hi = case.x_dam - 1.0, case.x_dam
    else:
        lo, hi = window
    mask = (x >= lo) & (x <= hi)
    if not mask.any():
        raise ValueError("injection window selects no cell")

    base = velocity_budget(case, x, h, u)
    m_exact = exact_momentum(case, solver.time) + momentum0
    hu = h * u
    window_energy = float(np.sum(np.abs(hu[mask])) * dx)
    c0 = gravity_wave_speed(case.h0, case.gravity)

    rows = []
    for eps in epsilons:
        # momentum channel: analytic prediction, no re-run needed
        m_bad = float(np.sum(np.where(mask, hu * (1.0 + eps), hu)) * dx)
        predicted = eps * window_energy / max(abs(m_exact), 1e-300)
        measured = abs(m_bad - m_exact) / max(abs(m_exact), 1e-300)
        momentum_gain = measured / predicted if predicted > 0 else float("inf")
        # velocity channel: two-sided
        deltas = {}
        for sign in (+1.0, -1.0):
            u_bad = u.copy()
            u_bad[mask] = u_bad[mask] + sign * eps * c0
            got = velocity_budget(case, x, h, u_bad)
            deltas[f"{sign:+.0f}"] = got["u_l1"] - base["u_l1"]
        rows.append({
            "epsilon": float(eps),
            "momentum_predicted": predicted,
            "momentum_measured": measured,
            "momentum_gain": momentum_gain,
            "velocity_delta_plus": deltas["+1"],
            "velocity_delta_minus": deltas["-1"],
            "velocity_max_abs_delta": max(abs(v) for v in deltas.values()),
            "velocity_predicted": eps * c0 * float(mask.sum()) * dx / max(
                float(np.sum(np.abs(case.exact(x)[1])) * dx), 1e-300),
        })
    return {
        "case": case.key,
        "n": int(n),
        "window": [float(lo), float(hi)],
        "window_cells": int(mask.sum()),
        "baseline_u_l1": base["u_l1"],
        "baseline_momentum_closure": abs(solver.momentum() - m_exact) / max(abs(m_exact), 1e-300),
        "rows": rows,
    }


def load_cases(keys=("ritter-dry", "stoker-wet")) -> list[DamBreakCase]:
    """Convenience accessor for the anchor's two cases."""
    return [CASES[k] for k in keys]


# keep the two anchor cases importable for driver code
ANCHOR_CASES = (RITTER_DRY, STOKER_WET)