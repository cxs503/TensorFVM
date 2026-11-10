"""Planing-hull hydrodynamics after Savitsky (1964): dynamic lift, spray drag,
trim / mean-wetted-length equilibrium and pitch (porpoising) stability.

Why this module exists
----------------------
A high-speed planing craft is not a displacement ship: below the hump speed the
hull is buoyancy supported, above it the hull runs on a *wetted planform* whose
length is set by the running trim and by the requirement that the hydrodynamic
resultant pass through the centre of gravity.  TensorFVM had no
CFD-independent reference surface for that regime, so a planing-hull CFD run
(e.g. VOF/RANS on a prismatic hull) had nothing to be validated against except
displacement-ship regressions.  This module supplies the analytic reference
surface:

* :func:`lift_coefficient_flat_plate`, :func:`deadrise_correction`,
  :func:`lift_coefficient` -- Savitsky's empirical dynamic-lift law
  ``C_L0 = tau^1.1 (0.0120 lam^0.5 + 0.0055 lam^2.5 / C_V^2)`` and the deadrise
  decrement ``C_Lb = C_L0 - 0.0065 beta C_L0^0.6``, with ``tau``/``beta`` in
  **degrees** and ``C_L = Delta / (0.5 rho V^2 b^2)`` normalised on the **beam
  squared** (not on wetted area);
* :func:`center_of_pressure_over_beam` -- ``l_cp/b = lam (0.75 - 1/(5.21 C_V^2 /
  lam^2 + 2.39))`` and its exact inverse
  :func:`mean_wetted_length_from_center_of_pressure`.  This closes the pitching
  moment equation, because ``l_cp/b`` depends on ``(lam, C_V)`` only -- not on
  ``tau``;
* :func:`spray_drag_coefficient` -- spray (whisker-spray) drag increment
  ``dC_D = 0.003 (b/L)^0.5 C_V^2`` on the same ``0.5 rho V^2 b^2`` scale;
* :func:`wetted_area`, :func:`frictional_resistance` -- ITTC-57 friction on the
  wetted planform ``A_wet = lam b^2`` with ``Re = V lam b / nu``;
* :func:`total_resistance` -- ``R_T = W tan(tau) + (R_f + R_spray) cos(tau)``,
  the horizontal component of the normal force plus the tangential drag
  projected onto the hull bottom;
* :func:`equilibrium` -- Savitsky's two-equation running condition: the moment
  equation ``l_cp/b = LCG/b`` fixes ``lam`` at a given ``C_V``; the lift equation
  ``C_L(lam, C_V, tau, beta) = W / (0.5 rho V^2 b^2)`` then fixes ``tau``.
  Returns a :class:`PlaningState` with the resistance breakdown and an explicit
  in/out-of-envelope flag;
* :func:`trim_curve`, :func:`trim_slope` -- the trim-speed curve and its slope,
  the classical *operational* porpoising indicator (a positive
  ``d tau / d C_V`` means the hull must trim up with speed, which a static lift
  curve cannot sustain without oscillating);
* :func:`quasi_static_stiffness` -- the 2-DOF (heave, pitch) stiffness matrix
  from a perturbation of the running condition with the kinematic closure
  ``lam = h / (b sin tau)`` (``h`` = transom immersion); static stability needs
  ``J_zz > 0``, ``J_tt > 0`` and ``det J > 0``;
* :func:`stability_diagnostics`, :func:`porpoising_map` -- the sign criteria
  ``dC_L/dlam``, ``dM*/dlam`` and the stiffness determinant evaluated on a
  ``(C_V, beta)`` grid, i.e. the machine-checkable porpoising boundary.

Conventions and limits (stated, not hidden)
-------------------------------------------
* Angles are **degrees** in the public API (Savitsky's correlations are
  degree-based); conversions are explicit in the code.
* The centre of gravity is taken **on the hull bottom line** (``VCG = 0``): the
  normal force and the tangential drag then share no pitch arm, which is the
  assumption of the classic closed-form method.  A non-zero VCG adds an
  ``~ (VCG/b)(D/N)`` term to the moment equation and is *not* modelled here.
* Thrust is taken parallel to the hull bottom (no propeller normal force, no
  thrust-line inclination); ``R_T`` is the hydrodynamic drag the thrust must
  balance, not the shaft thrust.
* Savitsky's laws are fits to planing flat-plate / V-bottom data.  Declared
  validity envelope: ``C_V`` in [0.6, 13], ``lam`` in [0.4, 5], ``beta <= 30``
  deg, ``tau`` in [2, 15] deg.  Outside it the functions still evaluate (so that
  limits stay testable) but :func:`in_envelope` / ``PlaningState.in_envelope``
  report the violation.  In particular the fit has a **non-vanishing V -> 0
  residual** (the ``lam^2.5/C_V^2`` term times ``0.5 rho V^2 b^2`` is
  speed-independent, i.e. it degenerates into a buoyant-equivalent load); below
  ``C_V ~ 1`` the craft is not planing -- see :func:`hydrostatic_residual_force`.
* Porpoising in reality is a *dynamic* heave-pitch instability: it needs the
  rate-dependent (added mass / damping) terms.  The criteria implemented here
  are quasi-static and therefore **necessary, not sufficient**; the validation
  report measures that limitation instead of papering over it.

References
----------
Savitsky, D. (1964), *Hydrodynamic design of planing hulls*, Marine Technology
1(1), 71-95 (lift, mean wetted length, centre of pressure, spray drag).
Savitsky, D. & Brown, P.W. (1976), *Procedures for hydrodynamic evaluation of
planing hulls in smooth and rough water*, Marine Technology 13(4) (running
attitude solution, resistance build-up, porpoising discussion).
ITTC-57 friction line ``0.075/(log10 Re - 2)^2`` (1978 Performance Prediction
Method).  Constants follow the platform single source: ``g = 9.80665 m/s^2``,
sea water ``rho = 1025 kg/m^3``, ``nu = 1.19e-6 m^2/s``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

#: Standard gravity [m/s^2] (platform single source).
G = 9.80665
#: Sea-water mass density [kg/m^3] (platform single source, 15 degC).
RHO_SEA_WATER = 1025.0
#: Sea-water kinematic viscosity [m^2/s] (15 degC).
NU_SEA_WATER = 1.19e-6

#: Declared Savitsky validity envelope.
C_V_MIN, C_V_MAX = 0.6, 13.0
LAMBDA_MIN, LAMBDA_MAX = 0.4, 5.0
BETA_MAX_DEG = 30.0
TAU_MIN_DEG, TAU_MAX_DEG = 2.0, 15.0


# --------------------------------------------------------------------------- #
# kinematics
# --------------------------------------------------------------------------- #
def speed_coefficient(speed: float, beam: float, g: float = G) -> float:
    """Savitsky speed coefficient ``C_V = V / sqrt(g b)`` (beam based)."""
    if beam <= 0.0:
        raise ValueError("beam must be positive")
    if speed < 0.0:
        raise ValueError("speed must be non-negative")
    return speed / math.sqrt(g * beam)


def speed_from_coefficient(C_V: float, beam: float, g: float = G) -> float:
    """Inverse of :func:`speed_coefficient`."""
    if beam <= 0.0:
        raise ValueError("beam must be positive")
    return C_V * math.sqrt(g * beam)


def dynamic_pressure_scale(C_V: float, beam: float, rho: float = RHO_SEA_WATER,
                           g: float = G) -> float:
    """``0.5 rho V^2 b^2`` written through ``C_V``: ``0.5 rho g b^3 C_V^2``."""
    return 0.5 * rho * g * beam ** 3 * C_V ** 2


def wetted_area(lam: float, beam: float) -> float:
    """Wetted planform area ``A_wet ~ lam b^2`` (Savitsky wetted-length model)."""
    if lam < 0.0:
        raise ValueError("lam must be non-negative")
    return lam * beam ** 2


def mean_wetted_length(lam: float, beam: float) -> float:
    """Mean wetted length ``L_m = lam b``."""
    return lam * beam


def transom_immersion(lam: float, tau_deg: float, beam: float) -> float:
    """Transom immersion ``h = lam b sin(tau)`` implied by the running condition.

    The mean wetted length of a flat planing surface at trim ``tau`` is set by
    the transom immersion; this closure makes the 2-DOF perturbation of
    :func:`quasi_static_stiffness` well posed.
    """
    return lam * beam * math.sin(math.radians(tau_deg))


def chine_wetted_lengths(lam: float, tau_deg: float, beta_deg: float, beam: float
                         ) -> Tuple[float, float]:
    """Keel / chine wetted lengths ``(L_K, L_C)`` from the mean wetted length.

    ``L_K = L_C + (b/2) tan(beta)/tan(tau)`` is the geometric relation between
    the keel and chine waterlines of a deadrise section (half-beam ``b/2``),
    combined with ``lam = (L_K + L_C)/(2 b)``.
    """
    if tau_deg <= 0.0:
        raise ValueError("tau must be positive")
    drop = 0.5 * beam * math.tan(math.radians(beta_deg)) / math.tan(math.radians(tau_deg))
    return lam * beam + 0.5 * drop, lam * beam - 0.5 * drop


# --------------------------------------------------------------------------- #
# dynamic lift (Savitsky 1964)
# --------------------------------------------------------------------------- #
def lift_coefficient_flat_plate(C_V: float, lam: float, tau_deg: float) -> float:
    """Flat-plate (``beta = 0``) dynamic-lift coefficient.

    ``C_L0 = tau^1.1 (0.0120 lam^0.5 + 0.0055 lam^2.5 / C_V^2)`` with ``tau`` in
    degrees and ``C_L0 = Delta / (0.5 rho V^2 b^2)`` (beam-squared normalised).
    """
    if C_V <= 0.0:
        raise ValueError("C_V must be positive")
    if lam < 0.0:
        raise ValueError("lam must be non-negative")
    if tau_deg < 0.0:
        raise ValueError("tau must be non-negative")
    return tau_deg ** 1.1 * (0.0120 * math.sqrt(lam) + 0.0055 * lam ** 2.5 / C_V ** 2)


def deadrise_correction(C_L0: float, beta_deg: float) -> float:
    """Savitsky's deadrise decrement ``0.0065 beta C_L0^0.6`` (beta in degrees)."""
    if beta_deg < 0.0:
        raise ValueError("beta must be non-negative")
    if C_L0 < 0.0:
        raise ValueError("C_L0 must be non-negative")
    return 0.0065 * beta_deg * C_L0 ** 0.6


def lift_coefficient(C_V: float, lam: float, tau_deg: float, beta_deg: float = 0.0) -> float:
    """Deadrise-corrected dynamic-lift coefficient ``C_Lb``.

    ``C_Lb = C_L0 - 0.0065 beta C_L0^0.6``.  For ``beta = 0`` this **is**
    ``C_L0`` (negative control: the flat-plate law must be recovered exactly).
    """
    cl0 = lift_coefficient_flat_plate(C_V, lam, tau_deg)
    if beta_deg <= 0.0:
        return cl0
    return cl0 - deadrise_correction(cl0, beta_deg)


def deadrise_breakdown_lift(beta_deg: float) -> float:
    """``C_L0`` below which the deadrise decrement drives ``C_Lb`` negative.

    ``C_L0 - 0.0065 beta C_L0^0.6 = 0`` at ``C_L0 = (0.0065 beta)^2.5``.  For
    ``beta = 15 deg`` that is ``C_L0 = 0.0030``: the corrected law is only
    trustworthy while the flat-plate lift stays well above ~3e-3 (small trim /
    high speed), and callers should treat a negative ``C_Lb`` as
    out-of-envelope instead of believing it.
    """
    if beta_deg < 0.0:
        raise ValueError("beta must be non-negative")
    return (0.0065 * beta_deg) ** 2.5


def lift_coefficient_required(weight: float, C_V: float, beam: float,
                              rho: float = RHO_SEA_WATER, g: float = G) -> float:
    """Lift coefficient demanded by the weight: ``Delta / (0.5 rho g b^3 C_V^2)``."""
    if weight <= 0.0:
        raise ValueError("weight must be positive")
    return weight / dynamic_pressure_scale(C_V, beam, rho=rho, g=g)


def lift_force(C_V: float, beam: float, lam: float, tau_deg: float,
               beta_deg: float = 0.0, rho: float = RHO_SEA_WATER, g: float = G) -> float:
    """Dimensional dynamic lift ``N = 0.5 rho V^2 b^2 C_Lb`` (vertical force)."""
    return dynamic_pressure_scale(C_V, beam, rho=rho, g=g) * lift_coefficient(
        C_V, lam, tau_deg, beta_deg)


def hydrostatic_residual_force(C_V: float, beam: float, lam: float, tau_deg: float,
                              rho: float = RHO_SEA_WATER, g: float = G) -> float:
    """Speed-independent part of :func:`lift_force` (the ``lam^2.5`` term).

    ``0.5 rho V^2 b^2 * tau^1.1 * 0.0055 lam^2.5 / C_V^2 = 0.5 rho g b^3 tau^1.1
    0.0055 lam^2.5``: it does **not** vanish as ``V -> 0``.  This is the fit's
    buoyant-equivalent remnant, and the reason the planing law is unusable below
    ``C_V ~ 1`` (there the craft is hydrostatically supported and ``lam`` is not
    a planing wetted length any more).
    """
    return 0.5 * rho * g * beam ** 3 * tau_deg ** 1.1 * 0.0055 * lam ** 2.5


# --------------------------------------------------------------------------- #
# centre of pressure and its inverse
# --------------------------------------------------------------------------- #
def center_of_pressure_over_beam(C_V: float, lam: float) -> float:
    """Centre of pressure aft of the transom, in beams.

    ``l_cp/b = lam (0.75 - 1/(5.21 C_V^2/lam^2 + 2.39))``; monotonically
    increasing in ``lam`` with ``l_cp/b -> 0.75 lam`` as ``C_V -> inf``.
    """
    if C_V <= 0.0:
        raise ValueError("C_V must be positive")
    if lam <= 0.0:
        raise ValueError("lam must be positive")
    return lam * (0.75 - 1.0 / (5.21 * C_V ** 2 / lam ** 2 + 2.39))


def mean_wetted_length_from_center_of_pressure(C_V: float, lcp_over_beam: float,
                                               tol: float = 1e-12) -> float:
    """Invert :func:`center_of_pressure_over_beam` (bisection, monotone map)."""
    if C_V <= 0.0:
        raise ValueError("C_V must be positive")
    if lcp_over_beam <= 0.0:
        raise ValueError("l_cp/b must be positive")
    lo, hi = 1e-9, max(1.0, 4.0 * lcp_over_beam) + 10.0
    while center_of_pressure_over_beam(C_V, hi) < lcp_over_beam:
        hi *= 2.0
        if hi > 1e9:
            raise ValueError("l_cp/b out of invertible range")
    for _ in range(400):
        mid = 0.5 * (lo + hi)
        if center_of_pressure_over_beam(C_V, mid) < lcp_over_beam:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol * max(1.0, hi):
            break
    return 0.5 * (lo + hi)


def pitch_moment_coefficient(C_V: float, lam: float, tau_deg: float, beta_deg: float,
                            lcg_over_beam: float) -> float:
    """Dimensionless pitch moment about the CG, positive = bow-down.

    ``M* = C_Lb (l_cp/b - LCG/b)``: the lift resultant acting aft of the CG
    pitches the bow down.  ``M* = 0`` is the running-condition (moment) equation,
    which for ``beta``-independent ``l_cp/b(lam, C_V)`` reduces to
    ``l_cp/b = LCG/b``.
    """
    return lift_coefficient(C_V, lam, tau_deg, beta_deg) * (
        center_of_pressure_over_beam(C_V, lam) - lcg_over_beam)


# --------------------------------------------------------------------------- #
# spray drag and friction
# --------------------------------------------------------------------------- #
def spray_drag_coefficient(C_V: float, beam: float, hull_length: float) -> float:
    """Spray drag increment ``dC_D = 0.003 (b/L)^0.5 C_V^2``.

    Normalised like the lift, i.e. ``R_spray = dC_D * 0.5 rho V^2 b^2``.  This is
    the whipped-spray ("whisker spray") increment of the planing bottom: a small
    but strongly speed-dependent contribution that is missing from a
    lift-plus-friction build-up.

    **Applicability limit (measured).** Because ``dC_D`` already carries
    ``C_V^2``, the *force* ``dC_D * 0.5 rho V^2 b^2`` grows as ``C_V^4``.  On the
    reference hull of the validation report it is 27 % of the resistance at
    ``C_V = 2`` and 640 % of the weight at ``C_V = 7`` -- i.e. the increment as
    tabulated here is a low-``C_V`` correction
    (``C_V <=~ 4``) and must not be used outside that band.  See
    ``docs/planing-savitsky-verification/REPORT.md`` (limitation L3).
    """
    if beam <= 0.0 or hull_length <= 0.0:
        raise ValueError("beam and hull_length must be positive")
    if beam / hull_length > 1.0:
        raise ValueError("beam cannot exceed hull length")
    return 0.003 * math.sqrt(beam / hull_length) * C_V ** 2


def spray_drag_force(C_V: float, beam: float, hull_length: float,
                     rho: float = RHO_SEA_WATER, g: float = G) -> float:
    """Dimensional spray drag ``R_spray = dC_D * 0.5 rho V^2 b^2``."""
    return spray_drag_coefficient(C_V, beam, hull_length) * dynamic_pressure_scale(
        C_V, beam, rho=rho, g=g)


def ittc57_friction_coefficient(Re: float) -> float:
    """ITTC-57 line ``C_F = 0.075/(log10 Re - 2)^2``."""
    if Re <= 0.0:
        raise ValueError("Re must be positive")
    denom = math.log10(Re) - 2.0
    if denom <= 0.0:
        raise ValueError("ITTC-57 undefined for Re <= 100")
    return 0.075 / denom ** 2


def frictional_resistance(C_V: float, lam: float, beam: float, rho: float = RHO_SEA_WATER,
                          nu: float = NU_SEA_WATER, g: float = G,
                          roughness_delta_cf: float = 0.0
                          ) -> Tuple[float, float, float]:
    """Tangential friction on the wetted planform; returns ``(R_f, Re, C_F)``.

    ``Re = V L_m / nu`` with ``L_m = lam b``, ``C_F`` from ITTC-57 plus an
    optional roughness/correlation allowance, ``R_f = C_F * 0.5 rho V^2 A_wet``.
    """
    speed = speed_from_coefficient(C_V, beam, g=g)
    if speed == 0.0:
        return 0.0, 0.0, 0.0
    re = speed * mean_wetted_length(lam, beam) / nu
    cf = ittc57_friction_coefficient(re) + roughness_delta_cf
    return cf * 0.5 * rho * speed ** 2 * wetted_area(lam, beam), re, cf


def total_resistance(weight: float, tau_deg: float, friction_force: float,
                     spray_force: float) -> float:
    """Total hydrodynamic drag ``R_T = W tan(tau) + (R_f + R_spray) cos(tau)``.

    The first term is the horizontal component of the normal force
    (``N = W/cos tau``), the second the tangential drag projected onto the hull
    bottom.  Thrust is assumed parallel to the bottom, so ``R_T`` is the drag the
    thrust must balance.
    """
    if weight <= 0.0:
        raise ValueError("weight must be positive")
    tau = math.radians(tau_deg)
    return weight * math.tan(tau) + (friction_force + spray_force) * math.cos(tau)


# --------------------------------------------------------------------------- #
# validity envelope
# --------------------------------------------------------------------------- #
def envelope_violations(C_V: float, lam: float, tau_deg: float, beta_deg: float
                        ) -> List[str]:
    """List the declared-envelope violations of a running condition."""
    bad: List[str] = []
    if not (C_V_MIN <= C_V <= C_V_MAX):
        bad.append("C_V")
    if not (LAMBDA_MIN <= lam <= LAMBDA_MAX):
        bad.append("lam")
    if not (TAU_MIN_DEG <= tau_deg <= TAU_MAX_DEG):
        bad.append("tau")
    if beta_deg > BETA_MAX_DEG:
        bad.append("beta")
    return bad


def in_envelope(C_V: float, lam: float, tau_deg: float, beta_deg: float) -> bool:
    """``True`` when the running condition sits inside the declared envelope."""
    return not envelope_violations(C_V, lam, tau_deg, beta_deg)


# --------------------------------------------------------------------------- #
# running condition (equilibrium)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PlaningState:
    """Solution of Savitsky's two running-condition equations."""

    C_V: float
    speed: float
    beam: float
    weight: float
    beta_deg: float
    lcg_over_beam: float
    lam: float
    tau_deg: float
    C_L: float
    C_L_required: float
    lcp_over_beam: float
    transom_immersion: float
    wetted_length: float
    friction_force: float
    spray_force: float
    resistance: float
    reynolds: float
    friction_coefficient: float
    lift_to_drag: float
    violations: Tuple[str, ...] = ()

    @property
    def in_envelope(self) -> bool:
        return not self.violations

    def as_dict(self) -> Dict[str, float]:
        d = {k: v for k, v in self.__dict__.items()}
        d["violations"] = list(self.violations)
        d["in_envelope"] = self.in_envelope
        return d


def _bisect(f, target: float, lo: float, hi: float, tol: float = 1e-12,
            max_iter: int = 400) -> float:
    """Bisection on a monotonically increasing function ``f``."""
    f_lo, f_hi = f(lo), f(hi)
    if f_lo > target or f_hi < target:
        raise ValueError("target outside the bracketed range")
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if f(mid) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo <= tol * max(1.0, hi):
            break
    return 0.5 * (lo + hi)


def trim_for_lift(C_V: float, lam: float, beta_deg: float, C_L_target: float,
                  tau_lo: float = 0.05, tau_hi: float = 45.0) -> float:
    """Solve the lift equation for the running trim ``tau`` [deg]."""
    def f(tau: float) -> float:
        return lift_coefficient(C_V, lam, tau, beta_deg)
    return _bisect(f, C_L_target, tau_lo, tau_hi)


def wetted_length_for_lcg(C_V: float, lcg_over_beam: float) -> float:
    """Solve the moment equation ``l_cp/b(lam) = LCG/b`` for ``lam``."""
    return mean_wetted_length_from_center_of_pressure(C_V, lcg_over_beam)


def running_condition(C_V: float, beam: float, weight: float, beta_deg: float,
                      lcg_over_beam: float, hull_length: float,
                      rho: float = RHO_SEA_WATER, nu: float = NU_SEA_WATER,
                      g: float = G, roughness_delta_cf: float = 0.0) -> PlaningState:
    """Solve Savitsky's running condition for a fixed speed (see module docs)."""
    lam = wetted_length_for_lcg(C_V, lcg_over_beam)
    C_L_req = lift_coefficient_required(weight, C_V, beam, rho=rho, g=g)
    tau = trim_for_lift(C_V, lam, beta_deg, C_L_req)
    C_L = lift_coefficient(C_V, lam, tau, beta_deg)
    r_f, re, cf = frictional_resistance(C_V, lam, beam, rho=rho, nu=nu, g=g,
                                        roughness_delta_cf=roughness_delta_cf)
    r_spray = spray_drag_force(C_V, beam, hull_length, rho=rho, g=g)
    resistance = total_resistance(weight, tau, r_f, r_spray)
    violations = tuple(envelope_violations(C_V, lam, tau, beta_deg))
    return PlaningState(
        C_V=C_V,
        speed=speed_from_coefficient(C_V, beam, g=g),
        beam=beam,
        weight=weight,
        beta_deg=beta_deg,
        lcg_over_beam=lcg_over_beam,
        lam=lam,
        tau_deg=tau,
        C_L=C_L,
        C_L_required=C_L_req,
        lcp_over_beam=center_of_pressure_over_beam(C_V, lam),
        transom_immersion=transom_immersion(lam, tau, beam),
        wetted_length=mean_wetted_length(lam, beam),
        friction_force=r_f,
        spray_force=r_spray,
        resistance=resistance,
        reynolds=re,
        friction_coefficient=cf,
        lift_to_drag=(weight / resistance) if resistance > 0 else float("inf"),
        violations=violations,
    )


def equilibrium(C_V: float, beam: float, weight: float, beta_deg: float,
                lcg_over_beam: float, hull_length: float, **kwargs) -> PlaningState:
    """Alias of :func:`running_condition` (Savitsky's equilibrium condition)."""
    return running_condition(C_V, beam, weight, beta_deg, lcg_over_beam, hull_length,
                             **kwargs)


def trim_curve(C_V_values: Sequence[float], beam: float, weight: float, beta_deg: float,
               lcg_over_beam: float, hull_length: float,
               **kwargs) -> List[PlaningState]:
    """Equilibrium running condition over a speed sweep."""
    return [running_condition(cv, beam, weight, beta_deg, lcg_over_beam, hull_length,
                              **kwargs) for cv in C_V_values]


def trim_slope(C_V: float, beam: float, weight: float, beta_deg: float,
               lcg_over_beam: float, hull_length: float, h: float = 1e-4,
               **kwargs) -> float:
    """``d tau / d C_V`` along the equilibrium curve (central difference).

    Positive slope = the hull must trim up as it goes faster.  That is the
    classical operational porpoising indicator (Savitsky & Brown 1976): a static
    lift curve cannot hold a rising trim in equilibrium, and the surplus is fed
    into the heave-pitch oscillation.
    """
    plus = running_condition(C_V + h, beam, weight, beta_deg, lcg_over_beam,
                             hull_length, **kwargs)
    minus = running_condition(C_V - h, beam, weight, beta_deg, lcg_over_beam,
                              hull_length, **kwargs)
    return (plus.tau_deg - minus.tau_deg) / (2.0 * h)


# --------------------------------------------------------------------------- #
# quasi-static heave/pitch stiffness and porpoising criteria
# --------------------------------------------------------------------------- #
def quasi_static_stiffness(C_V: float, beam: float, weight: float, beta_deg: float,
                           lcg_over_beam: float, hull_length: float,
                           delta: float = 1e-5) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """2-DOF stiffness matrix ``J[i][j]`` of the running condition.

    Perturbation model: downward heave ``z`` and nose-up pitch ``theta`` are
    applied about the equilibrium, with the kinematic closure
    ``lam = (h_e + z) / (b sin(tau_e + theta))`` (``h_e`` from
    :func:`transom_immersion`) and ``tau = tau_e + theta``.  The generalised
    forces are the dimensionless lift ``N* = C_Lb`` and the pitch moment
    ``M* = C_Lb (l_cp/b - LCG/b)`` (positive = bow-down).

    ``J_zz = dN*/dz``, ``J_zt = dN*/dtheta``, ``J_tz = dM*/dz``,
    ``J_tt = dM*/dtheta``; static stability requires a positive-definite matrix,
    i.e. ``J_zz > 0``, ``J_tt > 0`` and ``det J > 0``.
    """
    q = dynamic_pressure_scale(C_V, beam)
    C_L_req = weight / q
    lam_e = wetted_length_for_lcg(C_V, lcg_over_beam)
    tau_e = trim_for_lift(C_V, lam_e, beta_deg, C_L_req)
    h_e = transom_immersion(lam_e, tau_e, beam)

    def forces(z: float, theta_deg: float):
        tau = tau_e + theta_deg
        if tau <= 0.0:
            raise ValueError("perturbation drives tau non-positive")
        lam = (h_e + z) / (beam * math.sin(math.radians(tau)))
        cl = lift_coefficient(C_V, lam, tau, beta_deg)
        moment = cl * (center_of_pressure_over_beam(C_V, lam) - lcg_over_beam)
        return cl, moment

    dz = delta * beam
    dth = delta * max(1.0, tau_e)
    n_zp, m_zp = forces(+dz, 0.0)
    n_zm, m_zm = forces(-dz, 0.0)
    n_tp, m_tp = forces(0.0, +dth)
    n_tm, m_tm = forces(0.0, -dth)
    j_zz = (n_zp - n_zm) / (2.0 * dz) / beam   # dimensionless (dz measured in m, /b)
    j_zt = (n_tp - n_tm) / (2.0 * dth)
    j_tz = (m_zp - m_zm) / (2.0 * dz) / beam
    j_tt = (m_tp - m_tm) / (2.0 * dth)
    return (j_zz, j_zt), (j_tz, j_tt)


def lift_gradient_over_wetted_length(C_V: float, lam: float, tau_deg: float,
                                     beta_deg: float, delta: float = 1e-5) -> float:
    """``dC_L/dlam`` at fixed speed and trim (heave restoring indicator)."""
    d = delta * max(lam, 1.0)
    return (lift_coefficient(C_V, lam + d, tau_deg, beta_deg)
            - lift_coefficient(C_V, lam - d, tau_deg, beta_deg)) / (2.0 * d)


def moment_gradient_over_wetted_length(C_V: float, lam: float, tau_deg: float,
                                       beta_deg: float, lcg_over_beam: float,
                                       delta: float = 1e-5) -> float:
    """``dM*/dlam`` at fixed speed and trim (pitch restoring indicator)."""
    d = delta * max(lam, 1.0)
    return (pitch_moment_coefficient(C_V, lam + d, tau_deg, beta_deg, lcg_over_beam)
            - pitch_moment_coefficient(C_V, lam - d, tau_deg, beta_deg, lcg_over_beam)) / (2.0 * d)


@dataclass(frozen=True)
class StabilityDiagnostics:
    """Porpoising-related diagnostics at one equilibrium point."""

    C_V: float
    beta_deg: float
    lcg_over_beam: float
    lam: float
    tau_deg: float
    dC_L_dlam: float
    dM_dlam: float
    trim_slope: float
    j_zz: float
    j_zt: float
    j_tz: float
    j_tt: float
    stiffness_det: float
    heave_stable: bool
    pitch_stable: bool
    coupled_stable: bool
    trim_slope_stable: bool
    dCL_dlam_positive: bool
    quasi_static_unstable: bool
    porpoising_prone: bool
    violations: Tuple[str, ...] = ()

    def as_dict(self) -> Dict[str, object]:
        d = {k: v for k, v in self.__dict__.items()}
        d["violations"] = list(self.violations)
        return d


def stability_diagnostics(C_V: float, beam: float, weight: float, beta_deg: float,
                          lcg_over_beam: float, hull_length: float,
                          **kwargs) -> StabilityDiagnostics:
    """Evaluate all porpoising indicators at one running condition.

    Classification used downstream (documented, quasi-static, necessary-only):

    * ``heave_stable``: ``dC_L/dlam > 0`` (sinking adds lift) *and*
      ``J_zz > 0``;
    * ``pitch_stable``: ``J_tt > 0`` (a bow-up perturbation produces a bow-down
      moment);
    * ``coupled_stable``: ``det J > 0`` (the two DOF do not conspire);
    * ``trim_slope_stable``: ``d tau / dC_V < 0``;
    * ``quasi_static_unstable`` = ``not coupled_stable`` (diagnostic only);
    * ``porpoising_prone`` = ``not trim_slope_stable``.

    Only the trim-slope criterion discriminates.  Measured on the reference hull
    (``docs/planing-savitsky-verification/REPORT.md``) the rate-free stiffness
    matrix is indefinite -- ``J_tt < 0``, ``det J < 0`` -- for *every* planing
    condition, because pitching the bow up shortens the wetted length and walks
    the centre of pressure forward of the CG faster than any lift gain can
    compensate.  A rate-free 2-DOF model therefore has no restoring pitch
    stiffness and cannot call any running condition stable; the flag is kept and
    reported as a diagnostic, never as the decision criterion.
    """
    state = running_condition(C_V, beam, weight, beta_deg, lcg_over_beam, hull_length,
                              **kwargs)
    j = quasi_static_stiffness(C_V, beam, weight, beta_deg, lcg_over_beam, hull_length)
    det = j[0][0] * j[1][1] - j[0][1] * j[1][0]
    dcl = lift_gradient_over_wetted_length(C_V, state.lam, state.tau_deg, beta_deg)
    dmd = moment_gradient_over_wetted_length(C_V, state.lam, state.tau_deg, beta_deg,
                                             lcg_over_beam)
    slope = trim_slope(C_V, beam, weight, beta_deg, lcg_over_beam, hull_length, **kwargs)
    heave_stable = bool(dcl > 0.0 and j[0][0] > 0.0)
    pitch_stable = bool(j[1][1] > 0.0)
    coupled_stable = bool(heave_stable and pitch_stable and det > 0.0)
    trim_stable = bool(slope < 0.0)
    return StabilityDiagnostics(
        C_V=C_V, beta_deg=beta_deg, lcg_over_beam=lcg_over_beam,
        lam=state.lam, tau_deg=state.tau_deg,
        dC_L_dlam=dcl, dM_dlam=dmd, trim_slope=slope,
        j_zz=j[0][0], j_zt=j[0][1], j_tz=j[1][0], j_tt=j[1][1],
        stiffness_det=det,
        heave_stable=heave_stable, pitch_stable=pitch_stable,
        coupled_stable=coupled_stable, trim_slope_stable=trim_stable,
        dCL_dlam_positive=bool(dcl > 0.0),
        quasi_static_unstable=bool(not coupled_stable),
        porpoising_prone=bool(not trim_stable),
        violations=state.violations,
    )


def porpoising_map(C_V_values: Sequence[float], beta_values: Sequence[float],
                   beam: float, weight: float, lcg_over_beam: float, hull_length: float,
                   **kwargs) -> List[StabilityDiagnostics]:
    """Porpoising classification over a ``(C_V, beta)`` grid."""
    return [stability_diagnostics(cv, beam, weight, beta, lcg_over_beam, hull_length,
                                  **kwargs)
            for beta in beta_values for cv in C_V_values]


def porpoising_boundary(C_V_values: Sequence[float], beta_deg: float, beam: float,
                        weight: float, lcg_over_beam: float, hull_length: float,
                        **kwargs) -> Dict[str, object]:
    """Unstable speed interval of a sweep, per criterion.

    Returns the ``C_V`` interval on which each criterion calls the running
    condition porpoising-prone (``d tau/dC_V > 0`` for the trim-slope criterion,
    ``not coupled_stable`` for the rate-free stiffness criterion), so the two can
    be compared instead of averaged into one number.  ``None`` = no unstable
    point inside the swept range.
    """
    rows = [stability_diagnostics(cv, beam, weight, beta_deg, lcg_over_beam,
                                  hull_length, **kwargs) for cv in C_V_values]

    def interval(pred):
        flagged = [row.C_V for row in rows if pred(row)]
        return [min(flagged), max(flagged)] if flagged else None

    return {
        "beta_deg": beta_deg,
        "C_V_swept": [C_V_values[0], C_V_values[-1]],
        "unstable_interval_trim_slope": interval(lambda r: not r.trim_slope_stable),
        "unstable_interval_stiffness": interval(lambda r: r.quasi_static_unstable),
        "n_trim_slope_unstable": sum(1 for r in rows if not r.trim_slope_stable),
        "n_stiffness_unstable": sum(1 for r in rows if r.quasi_static_unstable),
        "max_trim_slope": max(r.trim_slope for r in rows),
        "peak_trim_slope_C_V": max(rows, key=lambda r: r.trim_slope).C_V,
    }
