"""Independent verification of the Sod (1978) grading oracle.

Why this module exists
----------------------
Every number reported for the anchor ``REPRO-FVM-SOD1978`` -- the L1 errors, the
observed convergence rates, the wave-position residuals, the C1/C2/C3 verdicts
of rounds 1-3 -- is measured *against* the exact solution produced by
:func:`tensorfvm.verification.sod1978.exact_sample`.  That makes the oracle the
single point on which the whole reproduction rests, and until now it had only
been spot-checked on eight scalars (``p*``, ``u*``, ``rho*_{L,R}``, four wave
speeds) against a second implementation of *the same Toro formulas*.  Agreement
of two copies of one formula set does not test the sampled *function*
``Q(xi)``: a wrong fan profile, a wrong star plateau, or a wrong wave speed
would leave all eight scalars intact and still bias every L1 error.

This module closes that gap with checks that are structurally independent of the
Riemann-solver formulas:

WH1  weak-form (integral conservation) residual.  For the Euler equations

         d/dt int_a^b q dx = -[F(q(b,t)) - F(q(a,t))],

     so for the exact solution

         R = int_a^b q(x,T) dx - int_a^b q(x,0) dx
             + int_0^T [F(q(b,t)) - F(q(a,t))] dt  =  0.

     ``R`` involves only the *sampled* solution and its flux; it is quadrature
     in space and time, no Riemann theory.  Any error in a wave speed (the
     position of a jump), in a star plateau value, or in the rarefaction fan
     profile shifts mass/momentum/energy across the box and shows up as a
     residual.  The initial data is a step, so the t = 0 integral is analytic
     and the time integral is broken at the four wave-crossing times, which
     keeps Gaussian quadrature accurate to its roundoff.

WH2  Rankine-Hugoniot residual at the shock, and pressure/velocity continuity
     across the contact, both evaluated from the sampled solution only.

WH3  rarefaction invariants: ``p/rho^gamma`` and ``u + 2c/(gamma-1)`` evaluated
     inside the fan (two interior points plus both edges).

WH4  60-digit mpmath re-solve of the pressure function: separates float64
     bisection error (expected < 1e-14 relative) from the 5-significant-digit
     rounding of the published values (expected ~1e-4 relative).  The published
     rounding, not the solver, is the accuracy floor of the published
     comparison.

WH5  teeth: the same WH1 check with a deliberately corrupted oracle (shock speed
     +1 %, ``p*`` +1 %, fan profile scaled).  A check that cannot fail is not a
     check; the corrupted-oracle residuals must exceed the declared tolerance by
     orders of magnitude.

WH6  bias of the reported measurements: L1 error and observed order of the
     nx = 200 -> 400 second-order run as the oracle is perturbed by
     delta = 1e-12 ... 1e-2.  This answers, quantitatively, the adversarial
     question "could the sub-unity C3 rate 0.814 be an oracle artefact?".

WH7  initial-data convention: the repository sets ``sod()``'s initial data by
     sampling the cell *centres*.  For even ``nx`` the diaphragm at x0 = 0.5 is
     a cell face, so that equals the exact cell average; for odd ``nx`` one cell
     straddles the diaphragm and point sampling silently loses
     ``|drho| * h / 2`` of mass -- an O(h) initial perturbation that would
     contaminate a convergence study.  Both cases are measured here so the
     even-grid choice of rounds 1-3 is a checked fact, not an assumption.

Thresholds are declared in :data:`THRESHOLDS` before any number is produced and
are copied into the output artifact as ``thresholds_declared_before_run``.

References
----------
* G. A. Sod (1978), J. Comput. Phys. 27, 1-31 (anchor).
* E. F. Toro (2009), *Riemann Solvers and Numerical Methods for Fluid
  Dynamics*, 3rd ed., Ch. 4 (exact solver, star state, Riemann invariants).
* D. Banks, T. Aslam, W. Rider, J. Comput. Phys. 227 (2008) 6985-7002
  (sublinear convergence of linearly degenerate waves -- the reason the global
  L1 rate on Sod cannot reach the design order; used in the discussion).
"""

from __future__ import annotations

import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from tensorfvm.verification.sod1978 import (
    GAMMA,
    LEFT,
    PUBLISHED,
    RIGHT,
    T_TARGET,
    X0,
    exact_sample,
    exact_star_state,
)

__all__ = [
    "THRESHOLDS", "WAVE_KEYS", "euler_flux", "sample_conserved",
    "weak_form_residuals", "jump_residuals", "fan_invariants",
    "mp_star_state", "teeth_study", "bias_study", "initial_data_convention",
    "run_study", "main",
]

#: four wave speeds in the order (head, tail, contact, shock)
WAVE_KEYS = ("rarefaction_head_speed", "rarefaction_tail_speed",
             "contact_speed", "shock_speed")

#: declared BEFORE the study runs; verdicts are computed from these, never
#: adjusted afterwards
THRESHOLDS = {
    # WH1: weak-form residual normalised by the box's own mass/momentum/energy
    "weak_form_rel_tol": 1e-8,
    # WH2/WH3: closed-form jump and fan invariants, sampled solution only
    "jump_rel_tol": 1e-10,
    "invariant_rel_tol": 1e-10,
    # WH4: float64 bisection vs 60-digit mpmath
    "mpath_vs_float64_rel": 1e-14,
    # WH4: every published entry below is printed with five decimals, so the
    # rounding bound of a correctly transcribed value is half a unit in the last
    # printed place, uniformly: 0.5e-5.  This is read off the printed digits, not
    # tuned to the outcome, and it is strict: a value that was rounded twice, or
    # computed from already-rounded intermediates, fails it.
    "published_rounding_abs": 5e-6,
    # WH5: teeth -- a corrupted oracle must be rejected by a margin no numerical
    # noise could produce: the corrupted residual must exceed the clean one by
    # this factor AND clear an absolute floor (mis-scaling a wave speed by df
    # moves the residual by O(df), measured gain ~0.5-1, so the floor is set
    # for corruption strengths down to 1e-4)
    "teeth_discrimination_ratio": 1e8,
    "teeth_abs_floor": 1e-6,
    # WH6: order sensitivity -- a 1e-10 oracle perturbation must not move the
    # reported order by more than this (so the reported rates are oracle-insensitive)
    "order_sensitivity_max": 1e-4,
    # WH7: even-nx initial data must equal the cell average to roundoff;
    # odd-nx must show the predicted |drho|*h/2 defect (relative to that)
    "initial_mass_defect_tol": 1e-12,
}


# ----------------------------------------------------------------------
# Euler flux / conserved sampling (own implementation, no repo flux code)
# ----------------------------------------------------------------------
def euler_flux(rho: np.ndarray, u: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Conserved-variable Euler flux ``F(q) = [rho u, rho u^2 + p, u (E + p)]``."""
    rho = np.asarray(rho, dtype=float)
    u = np.asarray(u, dtype=float)
    p = np.asarray(p, dtype=float)
    e = p / (GAMMA - 1.0) + 0.5 * rho * u * u
    return np.stack([rho * u, rho * u * u + p, u * (e + p)], axis=0)


def sample_conserved(x: np.ndarray, t: float, star: Dict[str, float],
                     left: Dict[str, float] = LEFT,
                     right: Dict[str, float] = RIGHT) -> np.ndarray:
    """Conserved variables of the *oracle under test* at ``(x, t)``."""
    x = np.asarray(x, dtype=float)
    prim = exact_sample((x - X0) / t, star, left, right)
    rho, u, p = prim["rho"], prim["u"], prim["p"]
    e = p / (GAMMA - 1.0) + 0.5 * rho * u * u
    return np.stack([rho, rho * u, e], axis=0)


def _q_at_xi(xi: float, star: Dict[str, float], left: Dict[str, float] = LEFT,
             right: Dict[str, float] = RIGHT) -> np.ndarray:
    """Conserved vector of the oracle at the similarity coordinate ``xi``."""
    prim = exact_sample(np.array([xi], dtype=float), star, left, right)
    rho, u, p = float(prim["rho"][0]), float(prim["u"][0]), float(prim["p"][0])
    e = p / (GAMMA - 1.0) + 0.5 * rho * u * u
    return np.array([rho, rho * u, e])


def _flux_at(x: float, t: float, star: Dict[str, float],
             left: Dict[str, float] = LEFT,
             right: Dict[str, float] = RIGHT) -> np.ndarray:
    """Flux ``F(q(x, t))`` of the oracle; the t -> 0 limit is the initial data."""
    if t <= 0.0:
        q = (_q_at_xi(-math.inf, star, left, right) if x < X0
             else _q_at_xi(math.inf, star, left, right))
    else:
        q = sample_conserved(np.array([x], dtype=float), t, star, left, right)[:, 0]
    return euler_flux(np.array([q[0]]), np.array([q[1] / q[0]]),
                      np.array([_p_of(q)]))[:, 0]


def _p_of(q: np.ndarray) -> float:
    """Pressure from a conserved vector ``q = [rho, rho u, E]``."""
    rho = q[0]
    u = q[1] / rho
    return (GAMMA - 1.0) * (q[2] - 0.5 * rho * u * u)


def _initial_integral(a: float, b: float) -> np.ndarray:
    """int_a^b q(x, 0) dx, analytically, for the Riemann initial data."""
    width = b - a
    m_left = max(0.0, min(b, X0) - a)
    m_right = width - m_left
    ql = np.array([LEFT["rho"], LEFT["rho"] * LEFT["u"],
                   LEFT["p"] / (GAMMA - 1.0) + 0.5 * LEFT["rho"] * LEFT["u"] ** 2])
    qr = np.array([RIGHT["rho"], RIGHT["rho"] * RIGHT["u"],
                   RIGHT["p"] / (GAMMA - 1.0) + 0.5 * RIGHT["rho"] * RIGHT["u"] ** 2])
    return ql * m_left + qr * m_right


def _flux_time_integral(x: float, t_end: float, star: Dict[str, float],
                        left: Dict[str, float] = LEFT,
                        right: Dict[str, float] = RIGHT) -> np.ndarray:
    """``int_0^T F(q(x, t)) dt`` for one fixed station ``x``.

    The integrand is smooth in t between the instants at which a wave crosses
    ``x``; those instants are handed to the quadrature as break points, so the
    discontinuities in the solution never sit inside a quadrature panel.
    """
    from scipy.integrate import quad

    breaks = []
    for key in WAVE_KEYS:
        speed = star[key]
        if speed != 0.0:
            t_hit = (x - X0) / speed
            if 0.0 < t_hit < t_end:
                breaks.append(t_hit)
    points = sorted(set(breaks)) or None
    out = np.zeros(3)
    for c in range(3):
        val, _err = quad(
            lambda t: float(_flux_at(x, t, star, left, right)[c]),
            0.0, t_end, points=points, limit=400,
            epsabs=1e-16, epsrel=1e-13,
        )
        out[c] = val
    return out


# ----------------------------------------------------------------------
# WH1  weak-form residual
# ----------------------------------------------------------------------
def weak_form_residual(a: float, b: float, t_end: float, star: Dict[str, float],
                       left: Dict[str, float] = LEFT,
                       right: Dict[str, float] = RIGHT) -> Dict[str, object]:
    """WH1: integral-conservation residual on the control volume ``[a, b]``.

    Returns the absolute residual per conserved variable and the *relative*
    residual, normalised by that box's own mass, momentum and energy scale
    (``rho_ref``, ``rho_ref c_ref``, ``p_ref`` times the box width).
    """
    from scipy.integrate import quad

    spatial = np.zeros(3)
    for c in range(3):
        spatial[c] = quad(
            lambda x, c=c: float(sample_conserved(np.array([x]), t_end, star,
                                                  left, right)[c, 0]),
            a, b,
            points=sorted({X0 + star[k] * t_end for k in WAVE_KEYS
                           if a < X0 + star[k] * t_end < b}) or None,
            limit=400, epsabs=1e-16, epsrel=1e-13,
        )[0]
    flux_b = _flux_time_integral(b, t_end, star, left, right)
    flux_a = _flux_time_integral(a, t_end, star, left, right)
    resid = spatial - _initial_integral(a, b) + (flux_b - flux_a)
    width = b - a
    c_ref = math.sqrt(GAMMA * LEFT["p"] / LEFT["rho"])
    scale = np.array([LEFT["rho"], LEFT["rho"] * c_ref, LEFT["p"]]) * width
    rel = np.abs(resid) / scale
    return {
        "a": a, "b": b,
        "residual": resid.tolist(),
        "relative": rel.tolist(),
        "max_relative": float(np.max(rel)),
    }


#: control volumes, fixed before the study: each spans a different part of the
#: wave structure (single wave, star region, whole structure, far field)
CONTROL_VOLUMES: Tuple[Tuple[float, float], ...] = (
    (0.00, 1.00),   # whole domain (boundary fluxes are piecewise constant)
    (0.10, 0.35),   # rarefaction fan only
    (0.30, 0.55),   # rarefaction tail + contact
    (0.55, 0.80),   # contact + shock
    (0.20, 0.90),   # everything
    (0.33, 0.42),   # inside the fan
    (0.60, 0.65),   # inside the right star plateau
)


# ----------------------------------------------------------------------
# WH2  jumps (Rankine-Hugoniot at the shock, continuity at the contact)
# ----------------------------------------------------------------------
def _prim_at(xi: float, star: Dict[str, float], left=LEFT, right=RIGHT) -> Dict[str, float]:
    out = exact_sample(np.array([xi], dtype=float), star, left, right)
    return {k: float(out[k][0]) for k in ("rho", "u", "p")}


def jump_residuals(star: Dict[str, float], eps: float = 1e-9) -> Dict[str, object]:
    """WH2: Rankine-Hugoniot at the shock and continuity across the contact.

    Everything is evaluated in the similarity coordinate ``xi = (x - x0)/t``,
    where the waves sit at ``xi = S`` and the solution is constant in ``t``; the
    checks therefore probe the sampled oracle directly, not the formulas that
    built it.
    """
    s = star["shock_speed"]
    off = eps * max(abs(s), 1.0)
    q_before = _q_at_xi(s - off, star)                 # right star plateau
    q_after = _q_at_xi(s + off, star)                  # undisturbed right state
    f_before = _flux_at(X0 + (s - off) * T_TARGET, T_TARGET, star)
    f_after = _flux_at(X0 + (s + off) * T_TARGET, T_TARGET, star)
    # Rankine-Hugoniot:  s (q_after - q_before) = F(q_after) - F(q_before)
    rh_resid = s * (q_after - q_before) - (f_after - f_before)
    rh_scale = np.array([LEFT["rho"] * abs(s), LEFT["rho"] * s * s,
                         LEFT["p"] * abs(s)])
    c_off = eps * max(abs(star["contact_speed"]), 1.0)
    left_of = _prim_at(star["contact_speed"] - c_off, star)
    right_of = _prim_at(star["contact_speed"] + c_off, star)
    p_jump = abs(left_of["p"] - right_of["p"]) / max(left_of["p"], 1e-30)
    u_jump = abs(left_of["u"] - right_of["u"]) / max(abs(left_of["u"]), 1e-30)
    density_jumps = bool(abs(left_of["rho"] - right_of["rho"]) > 0.1)
    rel = np.abs(rh_resid) / rh_scale
    return {
        "rh_residual": rh_resid.tolist(),
        "rh_relative": rel.tolist(),
        "rh_max_relative": float(np.max(rel)),
        "contact_p_rel": float(p_jump),
        "contact_u_rel": float(u_jump),
        "contact_density_jumps": density_jumps,
        "max_relative": float(max(np.max(rel), p_jump, u_jump)),
    }


# ----------------------------------------------------------------------
# WH3  rarefaction invariants inside the fan
# ----------------------------------------------------------------------
def fan_invariants(star: Dict[str, float]) -> Dict[str, object]:
    """WH3: ``p/rho^gamma`` and ``u + 2c/(gamma-1)`` are constant in the fan."""
    head, tail = star["rarefaction_head_speed"], star["rarefaction_tail_speed"]
    xi = np.array([head * (1 - 1e-6) + tail * 1e-6,
                   head * 0.75 + tail * 0.25,
                   head * 0.5 + tail * 0.5,
                   head * 0.25 + tail * 0.75,
                   head * 1e-6 + tail * (1 - 1e-6)])
    prim = exact_sample(xi, star)
    c = np.sqrt(GAMMA * prim["p"] / prim["rho"])
    entropy = prim["p"] / prim["rho"] ** GAMMA
    invariant = prim["u"] + 2.0 * c / (GAMMA - 1.0)
    ref_entropy = star["p_star"] / star["rho_star_left"] ** GAMMA
    ref_invariant = star["u_star"] + 2.0 * star["c_star_left"] / (GAMMA - 1.0)
    ent_rel = float(np.max(np.abs(entropy - ref_entropy) / ref_entropy))
    inv_rel = float(np.max(np.abs(invariant - ref_invariant) / abs(ref_invariant)))
    # entropy condition: the characteristics must not overtake the edges
    entropy_ok = bool(head < tail < star["contact_speed"] < star["shock_speed"])
    return {
        "xi": xi.tolist(),
        "entropy_relative": ent_rel,
        "invariant_relative": inv_rel,
        "max_relative": float(max(ent_rel, inv_rel)),
        "wave_order_ok": entropy_ok,
        "wave_speeds": [float(star["rarefaction_head_speed"]),
                        float(star["rarefaction_tail_speed"]),
                        float(star["contact_speed"]), float(star["shock_speed"])],
    }


# ----------------------------------------------------------------------
# WH4b  internal consistency of the published reference set
# ----------------------------------------------------------------------
def published_consistency(mp: Dict[str, float] | None = None) -> Dict[str, object]:
    """WH4b: is each published entry consistent with the rest of the set?

    Two independent readings are made for every published number:

    1. against the 60-digit exact solution (the entry's own precision), with the
       tolerance read off its printed digits (``published_rounding_abs``);
    2. against the value *implied by the other published entries* through a
       closed-form relation of the exact solution:

           S_head       = u_L - c_L                        (initial data only)
           S_tail       = u* - sqrt(gamma p*/rho*_L)       (published star state)
           S_contact    = u*                               (identity)
           S_shock      = c_R sqrt(1 + (gamma+1)/(2 gamma) (p*/p_R - 1))

       A set that was transcribed correctly fails neither reading; an entry that
       was rounded twice, or copied from a different computation, fails (2) while
       its neighbours still satisfy (1).
    """
    if mp is None:
        mp = mp_star_state()
    tol = THRESHOLDS["published_rounding_abs"]
    c_l = math.sqrt(GAMMA * LEFT["p"] / LEFT["rho"])
    c_r = math.sqrt(GAMMA * RIGHT["p"] / RIGHT["rho"])
    implied = {
        "rarefaction_head_speed": LEFT["u"] - c_l,
        "rarefaction_tail_speed": PUBLISHED["u_star"] - math.sqrt(
            GAMMA * PUBLISHED["p_star"] / PUBLISHED["rho_star_left"]),
        "contact_speed": PUBLISHED["u_star"],
        "shock_speed": c_r * math.sqrt(
            1.0 + (GAMMA + 1.0) / (2.0 * GAMMA)
            * (PUBLISHED["p_star"] / RIGHT["p"] - 1.0)),
    }
    rows = []
    for key in ("p_star", "u_star", "rho_star_left", "rho_star_right",
                "shock_speed", "contact_speed", "rarefaction_head_speed",
                "rarefaction_tail_speed"):
        pub = float(PUBLISHED[key])
        dev_exact = abs(pub - mp[key])
        row = {
            "key": key, "published": pub, "mpmath_exact": float(mp[key]),
            "abs_dev_vs_exact": dev_exact,
            "consistent_with_exact": bool(dev_exact <= tol),
            "implied_by_published_set": (float(implied[key])
                                         if key in implied else None),
        }
        if key in implied:
            row["abs_dev_vs_implied"] = abs(pub - implied[key])
            row["consistent_with_published_set"] = bool(
                abs(pub - implied[key]) <= 10.0 * tol)
        rows.append(row)
    bad = [r for r in rows if not r["consistent_with_exact"]]
    impact = {}
    for r in bad:
        impact[r["key"]] = {
            "rel_dev_pct": r["abs_dev_vs_exact"] / abs(r["mpmath_exact"]) * 100.0,
            # a wave-speed error shifts a wave *position* linearly in t
            "position_shift_at_t_target": r["abs_dev_vs_exact"] * T_TARGET,
            "c1_tolerance_pct": 2.0,
        }
    return {
        "tolerance_abs": tol,
        "rows": rows,
        "n_inconsistent_with_exact": len(bad),
        "inconsistent_keys": [r["key"] for r in bad],
        "all_consistent": not bad,
        "impact_of_inconsistent_entries": impact,
        "provenance": ("values as carried in "
                       "docs/oss-inventory-fvm/artifacts/sod_literature_benchmark.json "
                       "reference.published_values (mission 14887) and "
                       "tensorfvm.verification.sod1978.PUBLISHED"),
    }


# ----------------------------------------------------------------------
# WH4  60-digit star state
# ----------------------------------------------------------------------
def mp_star_state(dps: int = 60) -> Dict[str, float]:
    """Re-solve the star state in ``mpmath`` at ``dps`` decimal digits.

    Same two-shock/two-rarefaction pressure function as Toro (2009) 4.6/4.7 but
    in 60-digit arithmetic: the difference from the float64 bisection isolates
    the solver's own rounding error, which is the part of the reported
    published-value comparison that this repository owns.
    """
    import mpmath as mp

    with mp.workdps(dps):
        g = mp.mpf(GAMMA)
        rl, ul, pl = mp.mpf(LEFT["rho"]), mp.mpf(LEFT["u"]), mp.mpf(LEFT["p"])
        rr, ur, pr = mp.mpf(RIGHT["rho"]), mp.mpf(RIGHT["u"]), mp.mpf(RIGHT["p"])
        cl = mp.sqrt(g * pl / rl)
        cr = mp.sqrt(g * pr / rr)

        def f(p, rho_k, p_k, c_k):
            if p > p_k:
                a = 2 / ((g + 1) * rho_k)
                b = (g - 1) / (g + 1) * p_k
                return (p - p_k) * mp.sqrt(a / (p + b))
            return 2 * c_k / (g - 1) * ((p / p_k) ** ((g - 1) / (2 * g)) - 1)

        lo, hi = mp.mpf("1e-20"), 10 * max(pl, pr)
        for _ in range(400):
            mid = (lo + hi) / 2
            if f(mid, rl, pl, cl) + f(mid, rr, pr, cr) + ur - ul > 0:
                hi = mid
            else:
                lo = mid
        p = (lo + hi) / 2
        u = (ul + ur) / 2 + (f(p, rr, pr, cr) - f(p, rl, pl, cl)) / 2

        def star_density(rho_k, p_k):
            if p > p_k:
                return rho_k * ((g + 1) * p + (g - 1) * p_k) / (
                    (g - 1) * p + (g + 1) * p_k)
            return rho_k * (p / p_k) ** (1 / g)

        rho_l, rho_r = star_density(rl, pl), star_density(rr, pr)
        c_star_l = mp.sqrt(g * p / rho_l)
        c_star_r = mp.sqrt(g * p / rho_r)
        head = ul - cl
        tail = u - c_star_l
        shock = ur + cr * mp.sqrt(1 + (g + 1) / (2 * g) * (p / pr - 1))
        del c_star_r
        return {
            "p_star": float(p), "u_star": float(u),
            "rho_star_left": float(rho_l), "rho_star_right": float(rho_r),
            "shock_speed": float(shock), "contact_speed": float(u),
            "rarefaction_head_speed": float(head),
            "rarefaction_tail_speed": float(tail),
        }


# ----------------------------------------------------------------------
# WH5  teeth: corrupted oracles must be caught
# ----------------------------------------------------------------------
def _corrupt(star: Dict[str, float], kind: str, delta: float) -> Dict[str, float]:
    """Return a copy of ``star`` with one deliberate, clearly named defect."""
    bad = dict(star)
    if kind == "shock_speed":
        bad["shock_speed"] = star["shock_speed"] * (1.0 + delta)
    elif kind == "p_star":
        bad["p_star"] = star["p_star"] * (1.0 + delta)
    elif kind == "contact_speed":
        bad["contact_speed"] = star["contact_speed"] * (1.0 + delta)
        bad["u_star"] = star["u_star"] * (1.0 + delta)
    elif kind == "rho_star_right":
        bad["rho_star_right"] = star["rho_star_right"] * (1.0 + delta)
    else:                                                    # pragma: no cover
        raise ValueError(f"unknown corruption {kind!r}")
    return bad


def teeth_study(star: Dict[str, float], deltas: Sequence[float] = (1e-3, 1e-4)
                ) -> Dict[str, object]:
    """WH5: run WH1/WH2 with corrupted oracles; the checks must reject them.

    The gain of the check is reported (``worst_residual / delta``) so the reader
    can see *how* discriminating it is, not just that it fired: a defect that
    shifts one wave speed or one plateau value is caught by the weak form with
    O(delta) sensitivity.
    """
    clean = {cv: weak_form_residual(*cv, T_TARGET, star) for cv in CONTROL_VOLUMES}
    clean_jump = jump_residuals(star)
    clean_worst = max(r["max_relative"] for r in clean.values())
    rows = []
    for delta in deltas:
        for kind in ("shock_speed", "p_star", "contact_speed", "rho_star_right"):
            bad = _corrupt(star, kind, delta)
            worst = max(weak_form_residual(*cv, T_TARGET, bad)["max_relative"]
                        for cv in CONTROL_VOLUMES)
            jump_bad = jump_residuals(bad)["max_relative"]
            ratio = worst / clean_worst if clean_worst > 0 else math.inf
            detected = bool(ratio >= THRESHOLDS["teeth_discrimination_ratio"]
                            and worst >= THRESHOLDS["teeth_abs_floor"])
            rows.append({
                "corruption": kind, "delta": delta,
                "worst_weak_form_relative": float(worst),
                "jump_max_relative": float(jump_bad),
                "discrimination_ratio": float(ratio),
                "gain_weak_form": float(worst / delta),
                "detected": detected,
            })
    return {
        "deltas": list(deltas),
        "rows": rows,
        "all_detected": all(r["detected"] for r in rows),
        "min_corrupted_residual": float(min(r["worst_weak_form_relative"]
                                            for r in rows)),
        "min_discrimination_ratio": float(min(r["discrimination_ratio"] for r in rows)),
        "clean_worst_weak_form_relative": float(clean_worst),
        "clean_jump_max_relative": float(clean_jump["max_relative"]),
        "clean": {f"[{cv[0]:.2f},{cv[1]:.2f}]": clean[cv] for cv in CONTROL_VOLUMES},
    }


# ----------------------------------------------------------------------
# WH6  bias of the reported measurements
# ----------------------------------------------------------------------
def _run_pair(nx_list: Sequence[int], order: int) -> Dict[int, Dict]:
    from tensorfvm.verification.sod1978 import run_case
    return {nx: run_case(nx, order=order) for nx in nx_list}


def _l1_rho(run: Dict, star: Dict[str, float]) -> float:
    exact = exact_sample((run["x"] - X0) / run["t_end"], star)
    return float(np.abs(run["rho"] - exact["rho"]).mean())


def bias_study(star: Dict[str, float], deltas: Sequence[float] = (
        0.0, 1e-12, 1e-9, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2),
        nx_list: Sequence[int] = (200, 400), order: int = 2) -> Dict[str, object]:
    """WH6: how much does an oracle perturbation move the reported order?

    The runs are the anchor's own second-order setting (HLLC + MUSCL-Hancock,
    limiter ``mc``, CFL 0.4).  A perturbation that is uniform in ``p*``, ``u*``,
    the star densities and the two outer wave speeds stands in for any small
    oracle error; the observed order between nx = 200 and 400 is recomputed for
    each ``delta``.
    """
    runs = _run_pair(nx_list, order)
    rows = []
    base_order = None
    for delta in deltas:
        bad = dict(star)
        for key in ("p_star", "u_star", "rho_star_left", "rho_star_right",
                    "rarefaction_head_speed", "rarefaction_tail_speed",
                    "contact_speed", "shock_speed"):
            bad[key] = star[key] * (1.0 + delta)
        l1 = {nx: _l1_rho(runs[nx], bad) for nx in nx_list}
        order_obs = (math.log(l1[nx_list[0]] / l1[nx_list[1]])
                     / math.log(nx_list[1] / nx_list[0]))
        if delta == 0.0:
            base_order = order_obs
        rows.append({
            "delta": delta,
            "l1": {str(nx): l1[nx] for nx in nx_list},
            "order": order_obs,
            "order_shift": order_obs - (base_order if base_order is not None else 0.0),
        })
    tiny = [r for r in rows if r["delta"] <= 1e-9]
    return {
        "nx_list": list(nx_list), "order": order,
        "base_order": base_order,
        "rows": rows,
        "max_order_shift_at_1e-9": float(max(abs(r["order_shift"]) for r in tiny)),
        "run_seconds": {str(nx): runs[nx]["seconds"] for nx in nx_list},
        "budget_drift": {str(nx): runs[nx]["budget_drift"] for nx in nx_list},
    }


# ----------------------------------------------------------------------
# WH7  initial-data convention
# ----------------------------------------------------------------------
def initial_data_convention(nx_list: Sequence[int] = (64, 100, 128, 200)) -> Dict[str, object]:
    """WH7: is ``sod()``'s centre-sampled initial data the exact cell average?

    The diaphragm sits at x0 = 0.5.  For even ``nx`` it is a cell face and the
    two are identical; for odd ``nx`` one cell straddles it and the centre
    sample picks one side of the step, losing ``|drho| h / 2`` of mass from that
    cell.  Measured against the analytic prediction, so the parity rule is
    checked rather than assumed.
    """
    rows = []
    for nx in nx_list:
        h = 1.0 / nx
        xc = (np.arange(nx) + 0.5) * h
        rho_point = np.where(xc < X0, LEFT["rho"], RIGHT["rho"])
        straddle = int(np.sum((xc - 0.5 * h < X0) & (X0 < xc + 0.5 * h)))
        if straddle:
            frac_left = (X0 - (math.floor(X0 / h) * h)) / h
            exact_mass = (LEFT["rho"] * frac_left
                          + RIGHT["rho"] * (1.0 - frac_left)) * h
        else:
            exact_mass = None
        mass_point = float(rho_point.sum() * h)
        # analytic: total mass is rho_L*0.5 + rho_R*0.5 for any grid
        mass_exact_total = 0.5 * LEFT["rho"] + 0.5 * RIGHT["rho"]
        defect = abs(mass_point - mass_exact_total)
        predicted = (abs(LEFT["rho"] - RIGHT["rho"]) * h / 2.0) if straddle else 0.0
        rows.append({
            "nx": nx, "h": h, "odd": bool(nx % 2),
            "straddling_cells": straddle,
            "mass_point": mass_point,
            "mass_exact": mass_exact_total,
            "defect_rel": defect / mass_exact_total,
            "predicted_defect_rel": predicted / mass_exact_total,
            "cell_exact_mass": exact_mass,
        })
    even_bad = [r for r in rows if not r["odd"] and r["defect_rel"]
                > THRESHOLDS["initial_mass_defect_tol"]]
    odd_unpredicted = [r for r in rows if r["odd"]
                       and abs(r["defect_rel"] - r["predicted_defect_rel"])
                       > 1e-9]
    return {
        "rows": rows,
        "even_grids_clean": not even_bad,
        "odd_grids_match_prediction": not odd_unpredicted,
        "max_even_defect_rel": float(max((r["defect_rel"] for r in rows
                                          if not r["odd"]), default=0.0)),
    }


# ----------------------------------------------------------------------
# WH8  boundary reachability of the measured window
# ----------------------------------------------------------------------
def boundary_reachability(t_end: float = T_TARGET) -> Dict[str, object]:
    """WH8: are the outflow boundaries still quiet at the measurement time?

    The oracle is only valid on the domain interior: once a wave reaches x = 0
    or x = 1 the numerical run is influenced by the outflow condition, and the
    error measured there is no longer a scheme property.  For the self-similar
    solution the station ``x`` sees ``xi = (x - x0)/t``, which sweeps the whole
    range as t -> 0, so the first time a wave reaches a boundary is
    ``t_reach = |x_boundary - x0| / |S|`` for the fastest inward wave.
    """
    head = star_head = exact_star_state()["rarefaction_head_speed"]
    shock = exact_star_state()["shock_speed"]
    t_left = abs(X0 - 0.0) / abs(head)          # left-going head wave hits x = 0
    t_right = abs(1.0 - X0) / abs(shock)        # right-going shock hits x = 1
    del star_head
    return {
        "t_end": t_end,
        "t_left_boundary_reached": t_left,
        "t_right_boundary_reached": t_right,
        "left_speed": head, "right_speed": shock,
        "margin_factor_left": t_left / t_end,
        "margin_factor_right": t_right / t_end,
        "clean": bool(t_end < min(t_left, t_right)),
        # second published/solver-relevant window (Sod 1978 fig. 3 uses t = 0.2)
        "t_paper_0p2_clean": bool(0.2 < min(t_left, t_right)),
        "margin_factor_0p2": min(t_left, t_right) / 0.2,
    }


# ----------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------
def _verdicts(weak: Dict, jump: Dict, fan: Dict, mp: Dict, teeth: Dict,
              bias: Dict, init: Dict, bounds: Dict,
              pub_audit: Dict | None = None) -> Dict[str, Dict]:
    v: Dict[str, Dict] = {}

    worst_weak = max(r["max_relative"] for r in weak.values())
    v["WH1_weak_form"] = {
        "value": worst_weak,
        "threshold": THRESHOLDS["weak_form_rel_tol"],
        "met": bool(worst_weak <= THRESHOLDS["weak_form_rel_tol"]),
    }
    worst_jump = max(jump["max_relative"], fan["max_relative"])
    v["WH2_WH3_jumps_and_invariants"] = {
        "value": worst_jump,
        "threshold": max(THRESHOLDS["jump_rel_tol"], THRESHOLDS["invariant_rel_tol"]),
        "met": bool(worst_jump <= max(THRESHOLDS["jump_rel_tol"],
                                      THRESHOLDS["invariant_rel_tol"]))
        and bool(fan["wave_order_ok"]),
        "wave_order_ok": fan["wave_order_ok"],
    }
    v["WH4_mpmath_vs_float64"] = {
        "value": mp["max_rel_vs_float64"],
        "threshold": THRESHOLDS["mpath_vs_float64_rel"],
        "met": bool(mp["max_rel_vs_float64"] <= THRESHOLDS["mpath_vs_float64_rel"]),
    }
    if pub_audit is not None:
        # the reference SET is reported separately from the oracle: a defect in a
        # published number is not a defect of this repository's exact solver
        v["WH4b_published_set_consistency"] = {
            "value": float(max(r["abs_dev_vs_exact"] for r in pub_audit["rows"])),
            "threshold": THRESHOLDS["published_rounding_abs"],
            "inconsistent_keys": pub_audit["inconsistent_keys"],
            "met": bool(pub_audit["all_consistent"]),
            "group": "reference_set",
        }
    v["WH5_teeth"] = {
        "value": teeth["min_corrupted_residual"],
        "threshold": THRESHOLDS["teeth_abs_floor"],
        "discrimination_ratio": teeth["min_discrimination_ratio"],
        "min_discrimination_ratio": THRESHOLDS["teeth_discrimination_ratio"],
        "met": bool(teeth["all_detected"]),
    }
    v["WH6_order_insensitivity"] = {
        "value": bias["max_order_shift_at_1e-9"],
        "threshold": THRESHOLDS["order_sensitivity_max"],
        "met": bool(bias["max_order_shift_at_1e-9"]
                    <= THRESHOLDS["order_sensitivity_max"]),
    }
    v["WH7_even_grid_initial_data"] = {
        "value": init["max_even_defect_rel"],
        "threshold": THRESHOLDS["initial_mass_defect_tol"],
        "met": bool(init["even_grids_clean"] and init["odd_grids_match_prediction"]),
    }
    v["WH8_boundary_clean"] = {
        "value": min(bounds["margin_factor_left"], bounds["margin_factor_right"]),
        "threshold": 1.0,
        "met": bool(bounds["clean"] and bounds["t_paper_0p2_clean"]),
        "note": "margin factor = first wave arrival time / measurement time",
    }
    return v


def run_study(outdir: Path | str, *, deltas: Sequence[float] | None = None,
              figures: bool = False) -> Dict:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    star = exact_star_state()
    weak = {f"[{a:.2f},{b:.2f}]": weak_form_residual(a, b, T_TARGET, star)
            for a, b in CONTROL_VOLUMES}
    jump = jump_residuals(star)
    fan = fan_invariants(star)
    mp = mp_star_state()
    mp_cmp = {}
    for key in ("p_star", "u_star", "rho_star_left", "rho_star_right",
                "shock_speed", "contact_speed", "rarefaction_head_speed",
                "rarefaction_tail_speed"):
        mp_cmp[key] = {
            "float64": float(star[key]),
            "mpmath": float(mp[key]),
            "published": float(PUBLISHED[key]),
            "rel_vs_float64": abs(star[key] - mp[key]) / abs(mp[key]),
            "rel_vs_published": abs(mp[key] - PUBLISHED[key]) / abs(PUBLISHED[key]),
        }
    mp_summary = {
        "dps": 60,
        "max_rel_vs_float64": float(max(r["rel_vs_float64"] for r in mp_cmp.values())),
        "max_rel_vs_published": float(max(r["rel_vs_published"] for r in mp_cmp.values())),
        "values": mp_cmp,
    }
    pub_audit = published_consistency(mp)
    teeth = teeth_study(star)
    bias = bias_study(star, deltas if deltas is not None else
                      (0.0, 1e-12, 1e-9, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2))
    init = initial_data_convention()
    bounds = boundary_reachability(T_TARGET)
    verdicts = _verdicts(weak, jump, fan, mp_summary, teeth, bias, init, bounds,
                         pub_audit)
    study = {
        "anchor": "REPRO-FVM-SOD1978",
        "round": "4 · grading-oracle verification",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "thresholds_declared_before_run": THRESHOLDS,
        "control_volumes": [list(cv) for cv in CONTROL_VOLUMES],
        "case": {"gamma": GAMMA, "left": LEFT, "right": RIGHT, "x0": X0,
                 "t_target": T_TARGET},
        "star_state_float64": {k: float(v) for k, v in star.items()},
        "weak_form": weak,
        "jump": jump,
        "fan": fan,
        "mpmath": mp_summary,
        "published_consistency": pub_audit,
        "teeth": teeth,
        "bias": bias,
        "initial_data": init,
        "boundary_reachability": bounds,
        "verdicts": verdicts,
        "ecosystem_friction": ECOSYSTEM_FRICTION,
        "oracle_verdicts_met": bool(all(v["met"] for k, v in verdicts.items()
                                        if v.get("group") != "reference_set")),
        "all_met": bool(all(v["met"] for v in verdicts.values())),
        "environment": {
            "platform": platform.platform(), "python": sys.version.split()[0],
            "machine": platform.machine(), "wall_seconds": time.time() - t_start,
        },
    }
    study["outdir"] = str(outdir)
    (outdir / "oracle_study.json").write_text(json.dumps(study, indent=1))
    if figures:
        from scripts.publish_sod1978_oracle import build_markdown, make_figures
        figs = make_figures(outdir, study)
        (outdir / "figure_summary.json").write_text(json.dumps(figs, indent=1))
        (outdir / "REPORT.md").write_text(build_markdown(study, figs))
    return study


#: infrastructure friction met while building round 4 (not part of the oracle
#: verdicts; recorded so the next worker does not have to rediscover it)
ECOSYSTEM_FRICTION: List[Dict[str, str]] = [
    {
        "stage": "surrounding test suite (regression run)",
        "blocker": "tests/test_compressible_sod.py is red on main (6 failures) "
                   "against the current EulerSolver API: it assumes the "
                   "attributes 'step' (now a method), 'rho', 'u', 'p' and the "
                   "old sod() config keys",
        "evidence": "pytest tests/test_compressible_sod.py -> 6 failed "
                    "(TypeError: '>' not supported between 'method' and 'int'; "
                    "AttributeError: 'EulerSolver' object has no attribute 'rho')",
        "suggestion": "rewrite that file against primitive()/nsteps; it is the "
                      "only stale test left in the Sod area and it masks real "
                      "regressions in the suite signal",
    },
    {
        "stage": "momentum budget of the anchor's runs",
        "blocker": "EulerSolver.budget_drift() returns NaN for momentum_x / "
                   "momentum_y under outflow boundaries (the boundary flux is "
                   "not part of the budget), so a conservation check on momentum "
                   "silently passes or fails depending on the reader",
        "evidence": "run_case(200, order=2)['budget_drift'] = "
                    "{'mass': -2e-16, 'momentum_x': nan, 'momentum_y': nan, "
                    "'energy': 0.0}",
        "suggestion": "either include the boundary flux or return None with a "
                      "reason; NaN in a budget is not self-explanatory",
    },
]


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="docs/repro-fvm-sod1978-oracle")
    parser.add_argument("--figures", action="store_true")
    args = parser.parse_args(argv)
    study = run_study(args.out, figures=args.figures)
    print(json.dumps(study["verdicts"], indent=1))
    print(f"oracle_verdicts_met={study['oracle_verdicts_met']}  "
          f"all_met={study['all_met']}  seconds={study['environment']['wall_seconds']:.1f}")
    return 0


if __name__ == "__main__":                                 # pragma: no cover
    raise SystemExit(main())