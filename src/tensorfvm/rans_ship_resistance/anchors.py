"""Analytic anchors for the RANS ship-resistance verification surface.

Three independent families of anchors are used, all classical and all closed
form (no solver output enters them, so they can grade the solver):

1. **Blasius (1908) similarity solution** of the laminar flat-plate boundary
   layer.  ``f''' + f f''/2 = 0`` with ``f(0)=f'(0)=0``, ``f'(inf)=1``.  The
   wall derivative ``f''(0) = 0.3320573362...`` sets the skin friction
   ``C_f = 0.664 Re_x^{-1/2}``, the thickness ``delta_99 = 5.0 x/sqrt(Re_x)``
   (``f' = 0.99``) and the momentum thickness ``theta = 0.664 x/sqrt(Re_x)``.
   The solution here is obtained by shooting with Newton iteration on
   ``f''(0)`` and is checked against those three tabulated values.

2. **1/7 power-law turbulent boundary layer** (von Karman / Prandtl):
   ``u/U = (y/delta)^{1/7}`` implies ``C_f = 0.0592 Re_x^{-1/5}`` locally (the
   form the Schlichting text quotes for ``5e5 < Re_x < 1e7``) and
   ``C_F = 0.072 Re^{-1/5}`` on the mean.  It is the anchor that criterion C2
   names; it is *not* interchangeable with the logarithmic line, see L2 below.

3. **Logarithmic friction lines**: Prandtl-Schlichting
   ``C_F = 0.455/(log10 Re)^{2.58}`` (the line criterion C2/C3 quote),
   ITTC-57 ``0.075/(log10 Re - 2)^2`` and Schoenherr, all three taken from
   :mod:`tensorfvm.resistance_chain`, which the platform has already
   cross-verified against its own independent implementation
   (``docs/resistance-chain-form-factor``).  The spread between the three lines
   at a given Re is the irreducible uncertainty of the reference itself.

Provenance discipline
---------------------
``SIMMAN2008_KVLCC2`` records the workshop's KVLCC2 reference case.  The
*geometric and kinematic* entries (particulars, Froude number, Reynolds number,
model scale) are structural facts quoted identically across the KVLCC2
literature.  The *resistance coefficient* entry carries
``provenance="withheld"``: no primary workshop table exists in this workspace and
no network route to one was available, so the number is a literature-recalled
central value with a stated band, not a machine-verified datum.  Every consumer
of that record must propagate the flag -- see
:func:`tensorfvm.rans_ship_resistance.speed_sweep.criterion_c3`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from ..resistance_chain import ittc57_cf, prandtl_schlichting_cf, schoenherr_cf

__all__ = [
    "BLASIUS_FPP", "blasius_solution", "blasius_cf", "blasius_delta", "blasius_theta",
    "one_seventh_profile", "one_seventh_local_cf", "one_seventh_mean_cf",
    "schlichting_mean_cf", "FRICTION_LINES", "SIMMAN2008_KVLCC2",
    "SimmanKvlcc2Reference",
]

#: ``f''(0)`` of the Blasius similarity solution (Howarth 1938 / tabulated).
BLASIUS_FPP = 0.33205733621519630


def blasius_solution(eta_max: float = 12.0, n: int = 601,
                     tol: float = 1e-13) -> Dict[str, np.ndarray]:
    """Solve the Blasius similarity ODE by shooting on ``f''(0)``.

    Returns a dict with ``eta``, ``f``, ``fp`` (:math:`u/U`), ``fpp`` and the
    converged ``fpp0``.  Newton iteration on the missing initial slope, with the
    ODE integrated by a fourth-order Runge-Kutta scheme on a uniform
    :math:`\\eta` grid.  The tolerance is applied to ``f'(eta_max) - 1``.
    """
    if eta_max <= 0 or n < 3:
        raise ValueError("eta_max must be positive and n >= 3")
    eta = np.linspace(0.0, eta_max, n)
    h = eta[1] - eta[0]

    def integrate(fpp0: float):
        f = np.zeros(n)
        fp = np.zeros(n)
        fpp = np.zeros(n)
        fp[0] = 0.0
        fpp[0] = fpp0
        # state vector y = (f, f', f''); f''' = -f f''/2
        def rhs(state):
            return np.array([state[1], state[2], -0.5 * state[0] * state[2]])
        state = np.array([0.0, 0.0, fpp0])
        for i in range(n - 1):
            k1 = rhs(state)
            k2 = rhs(state + 0.5 * h * k1)
            k3 = rhs(state + 0.5 * h * k2)
            k4 = rhs(state + h * k3)
            f[i], fp[i], fpp[i] = state
            state = state + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        f[-1], fp[-1], fpp[-1] = state
        return f, fp, fpp

    guess = BLASIUS_FPP
    for _ in range(80):
        f, fp, fpp = integrate(guess)
        error = fp[-1] - 1.0
        if abs(error) < tol:
            break
        # d f'(eta_max) / d fpp0 from a one-sided difference
        perturbed = integrate(guess * (1.0 + 1e-6))[1][-1] - 1.0
        slope = (perturbed - error) / (guess * 1e-6)
        guess -= error / slope
    else:  # pragma: no cover - the Newton iteration converges in <10 steps
        raise RuntimeError("Blasius shooting failed to converge")
    f, fp, fpp = integrate(guess)
    return {"eta": eta, "f": f, "fp": fp, "fpp": fpp, "fpp0": float(guess)}


def blasius_eta_at_fraction(fraction: float = 0.99, eta_max: float = 12.0) -> float:
    """``eta`` where ``f'(eta) = fraction`` (``eta_99 = 4.910`` for 0.99)."""
    sol = blasius_solution(eta_max=eta_max)
    return float(np.interp(fraction, sol["fp"], sol["eta"]))


def blasius_cf(re_x):
    """Local laminar skin friction ``C_f = 0.664 Re_x^{-1/2}``."""
    re_x = np.asarray(re_x, dtype=float)
    if np.any(re_x <= 0):
        raise ValueError("Re_x must be positive")
    return 0.664 / np.sqrt(re_x)


def blasius_delta(re_x):
    """``delta_99 = 5.0 x / sqrt(Re_x)`` i.e. ``eta_99 = 4.91`` over ``sqrt(nu x/U)``."""
    re_x = np.asarray(re_x, dtype=float)
    return blasius_eta_at_fraction(0.99) / np.sqrt(re_x)


def blasius_theta(re_x):
    """Momentum thickness ``theta/x = 0.664 Re_x^{-1/2}``."""
    re_x = np.asarray(re_x, dtype=float)
    return 0.664 / np.sqrt(re_x)


def one_seventh_profile(y_over_delta):
    """``u/U = (y/delta)^{1/7}``."""
    y = np.asarray(y_over_delta, dtype=float)
    if np.any(y < 0):
        raise ValueError("y/delta must be non-negative")
    return np.clip(y, 0.0, 1.0) ** (1.0 / 7.0)


def one_seventh_local_cf(re_x):
    """Local turbulent skin friction implied by the 1/7 profile, ``0.0592 Re_x^{-1/5}``."""
    re_x = np.asarray(re_x, dtype=float)
    if np.any(re_x <= 0):
        raise ValueError("Re_x must be positive")
    return 0.0592 * re_x ** (-0.2)


def one_seventh_mean_cf(re_y):
    """Mean skin friction of the same profile, ``0.072 Re^{-1/5}``."""
    re_y = np.asarray(re_y, dtype=float)
    if np.any(re_y <= 0):
        raise ValueError("Re must be positive")
    return 0.072 * re_y ** (-0.2)


def schlichting_mean_cf(re_y):
    """Prandtl-Schlichting mean line ``0.455/(log10 Re)^{2.58}`` (criterion C2)."""
    re_y = np.asarray(re_y, dtype=float)
    if np.any(re_y <= 0):
        raise ValueError("Re must be positive")
    return 0.455 / np.log10(re_y) ** 2.58


#: The three logarithmic friction lines, reused from the platform-verified
#: implementation in :mod:`tensorfvm.resistance_chain`.
FRICTION_LINES = {
    "prandtl_schlichting": prandtl_schlichting_cf,
    "ittc57": ittc57_cf,
    "schoenherr": schoenherr_cf,
}


@dataclass(frozen=True)
class SimmanKvlcc2Reference:
    """The SIMMAN 2008 KVLCC2 reference case as recorded in this workspace.

    ``c_t`` is a *literature-recalled central value*; ``c_t_band`` is the spread
    the same workshop reports between facilities.  ``provenance`` is ``withheld``
    because no primary table is machine-verifiable here: the number may be used
    for a consistency statement but must never be reported as a verified
    experimental anchor.
    """
    name: str = "KVLCC2 (SIMMAN 2008)"
    l_pp_full: float = 320.0
    b_full: float = 58.0
    t_full: float = 20.8
    cb: float = 0.8098
    cm: float = 0.9982
    cwp: float = 0.870
    lcb_pct: float = -1.688
    s_full: float = 27578.0           # [m^2] bare-hull wetted surface, full scale
    model_length: float = 7.0         # [m] L_pp at model scale
    fn_design: float = 0.142
    re_design: float = 7.23e6
    c_t: float = 4.00e-3              # literature-recalled central value at Fn=0.142
    c_t_band: tuple = (3.90e-3, 4.10e-3)
    provenance: str = "withheld"
    notes: tuple = field(default_factory=lambda: (
        "geometric/kinematic entries are structural KVLCC2 facts quoted across the literature",
        "c_t is a recalled central value (no primary SIMMAN 2008 table in this workspace, no network route)",
        "consumers must propagate provenance='withheld' into any verdict",
    ))

    @property
    def scale_ratio(self) -> float:
        """Model scale ratio ``lambda = L_full / L_model``."""
        return self.l_pp_full / self.model_length

    @property
    def s_model(self) -> float:
        """Model wetted surface ``S_full / lambda^2``."""
        return self.s_full / self.scale_ratio ** 2

    def speed_for_froude(self, fn: float) -> float:
        """Model speed [m/s] at ``Fn`` based on the model length."""
        return fn * math.sqrt(9.81 * self.model_length)

    def froude_for_ship_speed(self, v_kn: float) -> float:
        """Froude number of the full-scale ship at ``v_kn``."""
        v = v_kn * 1852.0 / 3600.0
        return v / math.sqrt(9.81 * self.l_pp_full)

    def model_speed_for_ship_speed(self, v_kn: float) -> float:
        """Froude-scaled model speed [m/s]."""
        return self.speed_for_froude(self.froude_for_ship_speed(v_kn))

    def c_t_relative_deviation(self, c_t: float) -> float:
        """Signed relative deviation of ``c_t`` from the recalled central value."""
        return c_t / self.c_t - 1.0


#: The single shared reference record.
SIMMAN2008_KVLCC2 = SimmanKvlcc2Reference()