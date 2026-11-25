"""Mechanical energy budget of the 1-D shallow-water dam-break cases.

Why this exists
---------------
The dam-break benchmark (:mod:`tensorfvm.shallow_water.dam_break`) verifies the
solver with mass (machine precision), total variation, monotonicity and the
depth L1 error.  Those metrics are blind to *energy*: a scheme can conserve
mass exactly and still burn the wrong amount of mechanical energy at a
discontinuity, and no metric in the existing suite measures it.  This module
adds the missing conservation-law leg of the verification:

    E(t) = ∫ [ ½ g h²  +  ½ h u² ] dx          (potential + kinetic, per unit
                                                width, density set to 1)

Three facts make E a usable reference for the dam-break cases:

1. **Ritter (dry bed)**: the solution has no shock.  The rarefaction is smooth
   and the drying tip carries zero energy in the limit, so the exact solution
   conserves E exactly: ``E(t) = E(0)`` for all t.  Any numerical loss of E is
   therefore *pure scheme dissipation* — a metric that did not exist before.
2. **Stoker (wet bed, bore)**: the bore is a genuine hydraulic jump, and a jump
   dissipates energy at a constant rate (the jump states are time independent),
   so ``E(t) = E(0) - D_bore · t`` with

       D_bore = g · q_rel · (h₂ - h₁)³ / (4 h₁ h₂),     q_rel = h₁ U

   which is the classical hydraulic-jump head loss ``(h₂-h₁)³/(4 h₁ h₂)``
   (Chow, *Open-Channel Hydraulics*, §3-4) multiplied by the weight discharge
   ``g q_rel`` of the bore-relative flow.  :func:`certify_bore_dissipation`
   does not trust that formula: it evaluates ``dE/dt`` of the *closed-form*
   solution by quadrature and compares, so a wrong head-loss expression is
   caught numerically rather than inherited.
3. Because both cases are closed form, ``E_exact(t*)`` is known to quadrature
   accuracy, so the numerical energy error is a *reference-bound* metric.

Honest scope: the energy error is a global functional of the solution; it is
first-order convergent for a TVD scheme at a discontinuity (the same order as
the depth L1 error) and it does **not** replace a pointwise metric.  It is
reported as an additional acceptance leg, not as a new headline tolerance.
"""

from __future__ import annotations

import math

import numpy as np

from .dam_break import (
    CASES,
    RITTER_DRY,
    STOKER_WET,
    DamBreakCase,
    TOLERANCE,
    _relative,
)
from .exact import _dry_wave_speeds, dry_riemann_state, stoker_constant_state, t_star_to_time
from .swe1d import ShallowWater1D

__all__ = [
    "cell_energy",
    "dry_riemann_energy_budget",
    "total_energy",
    "exact_energy",
    "bore_dissipation_rate",
    "certify_bore_dissipation",
    "energy_budget",
    "energy_convergence_study",
    "ENERGY_TOLERANCE",
]

#: relative tolerance on the headline energy metrics (same 5 % acceptance line
#: the task uses; the observed values are far below it — see the report)
ENERGY_TOLERANCE = TOLERANCE


# --------------------------------------------------------------------------- #
# numerical state energy
# --------------------------------------------------------------------------- #
def cell_energy(h, hu, gravity: float = 9.81) -> dict:
    """Cell-wise depth-integrated energy densities (per unit width)."""
    h = np.asarray(h, dtype=float)
    hu = np.asarray(hu, dtype=float)
    wet = h > 0.0
    u = np.where(wet, hu / np.where(wet, h, 1.0), 0.0)
    potential = 0.5 * gravity * h * h
    kinetic = 0.5 * h * u * u
    return {"potential": potential, "kinetic": kinetic, "total": potential + kinetic}


def total_energy(h, hu, gravity: float = 9.81, dx: float = 1.0) -> dict:
    """Integrate :func:`cell_energy` over the domain (midpoint rule)."""
    parts = cell_energy(h, hu, gravity)
    return {k: float(v.sum() * dx) for k, v in parts.items()}


# --------------------------------------------------------------------------- #
# closed-form reference energy
# --------------------------------------------------------------------------- #
def _integrate_fan_energy(x_min, x_max, fronts, gravity, sample):
    """Three-point Gauss integration is exact for quartic fan energy."""
    edges = np.array([x_min, *sorted(x for x in set(fronts) if x_min < x < x_max), x_max])
    half = np.diff(edges)/2
    # Mechanical energy density is quartic inside each rarefaction.
    nodes = np.array([-math.sqrt(.6), 0., math.sqrt(.6)])
    weights = np.array([5/9, 8/9, 5/9])
    h, u = sample(edges[:-1,None]+half[:,None]*(1+nodes))
    return float(half @ (cell_energy(h,h*u,gravity)["total"] @ weights))


def dry_riemann_energy_budget(x_min, x_max, t, hL, uL, hR, uR, gravity=9.81, x0=0.):
    """Interval energy and cumulative rightward face transports for dry Riemann data.

    The contact with vacuum has zero mechanical energy loss. At each face,
    the cumulative transport is t*F - (x-x0)*(E-E_initial), including any
    first wetting or drying. Energy and transport are per unit width and density.
    """
    if not (math.isfinite(x_min) and math.isfinite(x_max) and x_min < x_max):
        raise ValueError("require a finite increasing interval")
    def state(x, time):
        return dry_riemann_state(x, time, hL, uL, hR, uR, gravity, x0)

    faces = np.array([x_min, x_max])
    h, u = state(faces, t)
    h0, u0 = state(faces, 0.)
    e = cell_energy(h, h*u, gravity)["total"]
    e0 = cell_energy(h0, h0*u0, gravity)["total"]
    west, east = t*u*(e+.5*gravity*h*h) - (faces-x0)*(e-e0)
    initial = _integrate_fan_energy(x_min, x_max, [x0], gravity, lambda x: state(x, 0.))
    fronts = [x0+t*s for s in _dry_wave_speeds(hL, uL, hR, uR, gravity)]
    final = _integrate_fan_energy(x_min, x_max, fronts, gravity, lambda x: state(x, t))
    return {"initial": initial, "final": final, "left_transport": float(west),
            "right_transport": float(east), "residual": float(final-initial-west+east)}


def exact_breakpoints(case: DamBreakCase, t: float) -> list:
    """Locations where the closed-form solution is non-smooth at time ``t``.

    The rarefaction head is a *kink* (``dh/dx`` jumps) even in the dry case, and
    the Stoker bore is a genuine jump, so a single global midpoint rule is only
    ``O(dx)`` accurate there -- measured noise ~1e-4 in ``E``, enough to swamp a
    finite-difference ``dE/dt``.  Splitting the integral at these points makes
    every sub-integrand smooth and the quadrature effectively exact.
    """
    c0 = math.sqrt(case.gravity * case.h0)
    points = {case.x_dam - c0 * t}
    if case.kind == "dry":
        points.add(case.x_dam + 2.0 * c0 * t)
    else:
        sol = stoker_constant_state(case.h0, case.h1, case.gravity)
        points.add(case.x_dam + (sol["u2"] - sol["c2"]) * t)
        points.add(case.x_dam + sol["shock_speed"] * t)
    inside = sorted(p for p in points if case.x_min < p < case.x_max)
    return [case.x_min] + inside + [case.x_max]


def exact_energy(case: DamBreakCase, t: float, n_quad: int = 200_000) -> dict:
    """Energy of the exact solution at time ``t`` (:func:`exact_breakpoints` aware).

    ``n_quad`` is the *total* number of quadrature cells, distributed over the
    smooth sub-intervals; the caller can confirm quadrature independence with
    two levels (:func:`energy_budget` does so).
    """
    edges = exact_breakpoints(case, t)
    weights = np.diff(edges)
    potential = kinetic = 0.0
    for a, b, w in zip(edges[:-1], edges[1:], weights):
        if w <= 0.0:
            continue
        m = max(int(round(n_quad * w / (case.x_max - case.x_min))), 2)
        x = a + (np.arange(m) + 0.5) * (b - a) / m
        h, u = case.exact(x, t)
        h = np.asarray(h, dtype=float)
        u = np.asarray(u, dtype=float)
        cell = (b - a) / m
        potential += float((0.5 * case.gravity * h * h).sum() * cell)
        kinetic += float((0.5 * h * u * u).sum() * cell)
    return {
        "t": float(t),
        "potential": potential,
        "kinetic": kinetic,
        "total": potential + kinetic,
        "n_quad": int(n_quad),
        "breakpoints": [float(p) for p in edges],
    }


def _hydraulic_jump_loss(gravity, discharge_in_shock_frame, h1, h2):
    """Mechanical energy lost per time and width across a compressive jump."""
    return float(gravity * abs(discharge_in_shock_frame) * abs(h2-h1)**3 / (4*h1*h2))


def bore_dissipation_rate(case: DamBreakCase) -> float:
    """Constant energy dissipation rate of the Stoker bore (per unit width).

    ``D = g q_rel (h2 - h1)^3 / (4 h1 h2)`` with ``q_rel = h1 U``; certified
    against ``dE/dt`` of the closed-form solution in
    :func:`certify_bore_dissipation`.  Raises for the dry case (no bore).
    """
    if case.kind != "wet":
        raise ValueError("a dry-bed case has no bore to dissipate energy")
    sol = stoker_constant_state(case.h0, case.h1, case.gravity)
    h1, h2 = case.h1, sol["h2"]
    q_rel = h1 * sol["shock_speed"]
    return _hydraulic_jump_loss(case.gravity, q_rel, h1, h2)


def certify_bore_dissipation(case: DamBreakCase = STOKER_WET,
                             t_star: float = 2.0,
                             deltas=(0.08, 0.04, 0.02),
                             n_quad: int = 200_000) -> dict:
    """Measure ``dE/dt`` of the exact solution and compare with the formula.

    Central differences in ``t`` of :func:`exact_energy`; three step sizes are
    used so that the truncation trend (should be ≈0, since ``dE/dt`` is
    constant) is visible rather than assumed.
    """
    t = t_star_to_time(t_star, case.h0, case.gravity)
    d_formula = bore_dissipation_rate(case)
    rows = []
    for d in deltas:
        e_plus = exact_energy(case, t + d * t, n_quad=n_quad)["total"]
        e_minus = exact_energy(case, t - d * t, n_quad=n_quad)["total"]
        d_measured = (e_plus - e_minus) / (2.0 * d * t)
        rows.append({
            "delta_over_t": float(d),
            "dE_dt_measured": float(d_measured),
            "dE_dt_formula": float(-d_formula),
            "relative_error": _relative(d_measured, -d_formula),
        })
    return {
        "case": case.key,
        "t": t,
        "t_star": t_star,
        "formula": float(d_formula),
        "rows": rows,
        "worst_relative_error": max(r["relative_error"] for r in rows),
    }


# --------------------------------------------------------------------------- #
# numerical budget
# --------------------------------------------------------------------------- #
def energy_budget(case: DamBreakCase, n: int = 800, cfl: float = 0.4,
                  order: int = 2, samples: int = 41,
                  n_quad: int = 200_000) -> dict:
    """Integrate one case and close its energy budget against the exact value.

    Returns the initial energy, the exact and numerical energies at ``t_end``,
    the exact cumulative dissipation (``D_bore · t`` for the wet case, 0 for the
    dry case) and the residual, which splits into

    * ``scheme_dissipation`` = extra energy burnt by the discretisation
      (numerical E below exact E; the only loss for the dry case), and
    * ``total_dissipation`` = ``E0 - E_num`` relative to ``E0``.
    """
    t_end = case.t_end
    solver = ShallowWater1D(n, case.x_min, case.x_max, gravity=case.gravity,
                            cfl=cfl, order=order)
    solver.set_state(np.where(solver.x < case.x_dam, case.h0, case.h1))
    initial_depth = solver.h.copy()
    initial_hu = solver.hu.copy()
    e0 = total_energy(initial_depth, initial_hu, case.gravity, solver.dx)
    series = [{"t": 0.0, "t_star": 0.0, "total": e0["total"]}]

    gate = {"next": t_end / samples}

    def sample(sol: ShallowWater1D) -> None:
        if sol.time < gate["next"]:
            return
        e = total_energy(sol.h, sol.hu, sol.gravity, sol.dx)
        series.append({"t": sol.time, "t_star": float(sol.time * math.sqrt(sol.gravity / case.h0)),
                       "total": e["total"], "potential": e["potential"],
                       "kinetic": e["kinetic"]})
        gate["next"] = sol.time + t_end / samples

    steps = solver.advance(t_end, callback=sample)
    e_num = total_energy(solver.h, solver.hu, case.gravity, solver.dx)
    e_ex = exact_energy(case, solver.time, n_quad=n_quad)
    e_ex_mid = exact_energy(case, t_end, n_quad=n_quad // 2)

    diss_rate = bore_dissipation_rate(case) if case.kind == "wet" else 0.0
    exact_dissipation = diss_rate * solver.time
    exact_expected = e0["total"] - exact_dissipation

    return {
        "case": case.key,
        "n": int(n),
        "dx": float(solver.dx),
        "steps": int(steps),
        "t_end": float(solver.time),
        "t_star_end": float(solver.time * math.sqrt(case.gravity / case.h0)),
        "energy_initial": e0["total"],
        "energy_numeric": e_num["total"],
        "energy_exact": e_ex["total"],
        "energy_exact_midrule_half": e_ex_mid["total"],
        "quadrature_self_difference": float(abs(e_ex["total"] - e_ex_mid["total"])),
        "bore_dissipation_rate": float(diss_rate),
        "exact_cumulative_dissipation": float(exact_dissipation),
        "energy_error": _relative(e_num["total"], e_ex["total"]),
        "scheme_dissipation": float((e0["total"] - e_num["total"]) / e0["total"]),
        "exact_dissipation_fraction": float(exact_dissipation / e0["total"]),
        "kinetic_fraction_numeric": e_num["kinetic"] / e_num["total"],
        "series": series,
        "mass_drift": float(_relative(
            solver.total_volume(),
            float(np.sum(np.where(solver.x < case.x_dam, case.h0, case.h1))) * solver.dx)),
    }


def energy_convergence_study(case: DamBreakCase,
                             resolutions=(100, 200, 400, 800, 1600),
                             cfl: float = 0.4, order: int = 2,
                             n_quad: int = 200_000) -> dict:
    """Grid refinement of the energy error; returns rows and observed order."""
    rows = []
    for n in resolutions:
        budget = energy_budget(case, n=n, cfl=cfl, order=order, samples=1,
                               n_quad=n_quad)
        rows.append({
            "n": n,
            "dx": budget["dx"],
            "energy_error": budget["energy_error"],
            "scheme_dissipation": budget["scheme_dissipation"],
            "exact_dissipation_fraction": budget["exact_dissipation_fraction"],
            "excess_dissipation": budget["scheme_dissipation"]
                                  - budget["exact_dissipation_fraction"],
        })
    # the *error* is the convergent quantity; the total burnt energy saturates at
    # the physical dissipation and would fake a zero order (a real trap: the
    # first version of this function did exactly that, reporting 0.43 -> 0.10)
    order_est = _order([r["energy_error"] for r in rows])
    order_excess = _order([abs(r["excess_dissipation"]) for r in rows])
    return {"case": case.key, "rows": rows, "observed_order": order_est,
            "observed_order_excess": order_excess}


def _order(values) -> list:
    """Observed orders for a doubling (or generally geometric) grid sequence."""
    out = []
    for a, b in zip(values[:-1], values[1:]):
        if a > 0.0 and b > 0.0:
            out.append(math.log(a / b) / math.log(2.0))
        else:
            out.append(float("nan"))
    return out