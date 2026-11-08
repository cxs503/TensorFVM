"""Closure algebra for k-omega SST (Menter 2003) and Spalart-Allmaras (1994).

Both closures are written for a thin-shear-layer state: the only resolved strain
is the wall-normal derivative of the streamwise velocity, ``S = |du/dy|``.  The
functions here are pure algebra on profiles, which keeps them testable in
isolation (log-layer equilibrium, eddy-viscosity ratio, Spalart's
``kappa``-consistency) and keeps the marching solver in
:mod:`tensorfvm.rans_ship_resistance.boundary_layer` free of model constants.

Constants are those of the original publications:

* SST-2003: ``beta* = 0.09``, ``beta1 = 0.075``, ``beta2 = 0.0828``,
  ``gamma1 = 5/9``, ``gamma2 = 0.44``, ``sigma_k1 = 0.85``, ``sigma_k2 = 1.0``,
  ``sigma_w1 = 0.5``, ``sigma_w2 = 0.856``, ``a1 = 0.31``, ``kappa = 0.41``.
  The production limiter is the standard ``10 beta* k omega``.
* SA (Spalart & Allmaras 1994, without the trip term): ``sigma = 2/3``,
  ``cb1 = 0.1355``, ``cb2 = 0.622``, ``kappa = 0.41``, ``cw1 = cb1/kappa^2 +
  (1+cb2)/sigma``, ``cw2 = 0.3``, ``cw3 = 2``, ``cv1 = 7.1``.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

__all__ = [
    "SST_CONSTANTS", "SA_CONSTANTS", "sst_blending", "sst_eddy_viscosity",
    "sst_production", "sa_eddy_viscosity", "sa_production", "sa_destruction",
    "sa_source_terms",
]

SST_CONSTANTS: Dict[str, float] = {
    "beta_star": 0.09, "beta1": 0.075, "beta2": 0.0828,
    "gamma1": 5.0 / 9.0, "gamma2": 0.44,
    "sigma_k1": 0.85, "sigma_k2": 1.0, "sigma_omega1": 0.5, "sigma_omega2": 0.856,
    "a1": 0.31, "kappa": 0.41,
    # numerical floors (not model constants)
    "cd_floor": 1e-10, "production_limit": 10.0,
}

SA_CONSTANTS: Dict[str, float] = {
    "sigma": 2.0 / 3.0, "cb1": 0.1355, "cb2": 0.622, "kappa": 0.41,
    "cw2": 0.3, "cw3": 2.0, "cv1": 7.1,
}
SA_CONSTANTS["cw1"] = (SA_CONSTANTS["cb1"] / SA_CONSTANTS["kappa"] ** 2
                       + (1.0 + SA_CONSTANTS["cb2"]) / SA_CONSTANTS["sigma"])


def _check(profile, name):
    arr = np.asarray(profile, dtype=float)
    if not np.isfinite(arr).all():
        raise ValueError("%s must be finite" % name)
    return arr


def _blend(f1_value, first, second):
    return f1_value * first + (1.0 - f1_value) * second


def sst_blending(k, omega, dk_dy, domega_dy, d, nu,
                 constants: Dict[str, float] = SST_CONSTANTS) -> Dict[str, np.ndarray]:
    """Menter's ``F1`` / ``F2`` blending functions for SST-2003.

    ``d`` is the wall distance, ``nu`` the molecular viscosity (per unit mass),
    ``dk_dy``/``domega_dy`` the wall-normal derivatives of ``k`` and ``omega``
    (the cross-diffusion term needs them).
    """
    c = constants
    k = np.maximum(_check(k, "k"), 0.0)
    omega = np.maximum(_check(omega, "omega"), 1e-30)
    d = np.maximum(_check(d, "d"), 1e-30)
    sqrt_k = np.sqrt(k)
    cd = np.maximum(2.0 * c["sigma_omega2"] / omega * _check(dk_dy, "dk_dy")
                    * _check(domega_dy, "domega_dy"), c["cd_floor"])
    arg1 = np.minimum(
        np.maximum(sqrt_k / (c["beta_star"] * omega * d), 500.0 * nu / (d * d * omega)),
        4.0 * c["sigma_omega2"] * k / (cd * d * d),
    )
    f1 = np.tanh(arg1 ** 4)
    arg2 = np.maximum(2.0 * sqrt_k / (c["beta_star"] * omega * d),
                      500.0 * nu / (d * d * omega))
    f2 = np.tanh(arg2 ** 2)
    return {"f1": f1, "f2": f2, "cd": cd}


def sst_eddy_viscosity(k, omega, strain, f2, constants: Dict[str, float] = SST_CONSTANTS):
    """``nu_t = a1 k / max(a1 omega, S F2)`` with the shear-stress limiter."""
    c = constants
    k = np.maximum(_check(k, "k"), 0.0)
    omega = np.maximum(_check(omega, "omega"), 1e-30)
    strain = np.abs(_check(strain, "strain"))
    f2 = _check(f2, "f2")
    return c["a1"] * k / np.maximum(c["a1"] * omega, strain * f2)


def sst_production(nu_t, strain, k, omega, constants: Dict[str, float] = SST_CONSTANTS):
    """Limited production rate ``P_k = min(nu_t S^2, 10 beta* k omega)`` per unit mass."""
    c = constants
    unlimited = nu_t * np.asarray(strain, dtype=float) ** 2
    limited = c["production_limit"] * c["beta_star"] * np.maximum(k, 0.0) * np.maximum(omega, 1e-30)
    return np.minimum(unlimited, limited)


def sst_coefficients(f1, constants: Dict[str, float] = SST_CONSTANTS) -> Dict[str, np.ndarray]:
    """Blended ``beta``, ``gamma``, ``sigma_k``, ``sigma_omega``."""
    c = constants
    f1 = _check(f1, "f1")
    return {
        "beta": _blend(f1, c["beta1"], c["beta2"]),
        "gamma": _blend(f1, c["gamma1"], c["gamma2"]),
        "sigma_k": _blend(f1, c["sigma_k1"], c["sigma_k2"]),
        "sigma_omega": _blend(f1, c["sigma_omega1"], c["sigma_omega2"]),
    }


def sa_eddy_viscosity(nu_tilde, nu, constants: Dict[str, float] = SA_CONSTANTS):
    """``nu_t = nu_tilde fv1``, ``fv1 = chi^3/(chi^3 + cv1^3)``, ``chi = nu_tilde/nu``."""
    c = constants
    nu_tilde = _check(nu_tilde, "nu_tilde")
    if nu <= 0:
        raise ValueError("nu must be positive")
    chi = nu_tilde / nu
    fv1 = chi ** 3 / (chi ** 3 + c["cv1"] ** 3)
    return np.maximum(nu_tilde * fv1, 0.0)


def sa_source_terms(nu_tilde, strain, d, nu,
                    constants: Dict[str, float] = SA_CONSTANTS) -> Dict[str, np.ndarray]:
    """SA production and destruction coefficients per unit mass.

    Returns ``production`` (:math:`c_{b1}\\tilde S\\tilde\\nu`, linear in
    :math:`\\tilde\\nu`), ``destruction_coefficient`` (:math:`c_{w1}f_w\\tilde\\nu/d^2`,
    multiplied by :math:`\\tilde\\nu` at use time) and the intermediate
    quantities ``S_tilde``, ``fv2``, ``fw``, ``r``.
    """
    c = constants
    nu_tilde = _check(nu_tilde, "nu_tilde")
    strain = np.abs(_check(strain, "strain"))
    d = np.maximum(_check(d, "d"), 1e-30)
    if nu <= 0:
        raise ValueError("nu must be positive")
    chi = nu_tilde / nu
    fv1 = chi ** 3 / (chi ** 3 + c["cv1"] ** 3)
    fv2 = 1.0 - chi / (1.0 + chi * fv1)
    s_tilde = strain + nu_tilde / (c["kappa"] ** 2 * d * d) * fv2
    s_tilde_floor = np.maximum(strain * 0.3, 1e-12)  # Spalart's positivity guard
    s_tilde = np.maximum(s_tilde, s_tilde_floor)
    r = np.minimum(nu_tilde / (s_tilde * c["kappa"] ** 2 * d * d), 10.0)
    g = r + c["cw2"] * (r ** 6 - r)
    fw = g * ((1.0 + c["cw3"] ** 6) / (g ** 6 + c["cw3"] ** 6)) ** (1.0 / 6.0)
    production = c["cb1"] * s_tilde
    destruction_coefficient = c["cw1"] * fw / (d * d)
    return {
        "production": production * nu_tilde,
        "destruction_coefficient": destruction_coefficient,
        "S_tilde": s_tilde, "fv2": fv2, "fw": fw, "r": r, "fv1": fv1,
    }


def sa_production(nu_tilde, strain, d, nu, constants: Dict[str, float] = SA_CONSTANTS):
    """SA production term alone [m^2/s^2]."""
    return sa_source_terms(nu_tilde, strain, d, nu, constants)["production"]


def sa_destruction(nu_tilde, strain, d, nu, constants: Dict[str, float] = SA_CONSTANTS):
    """SA destruction term alone [m^2/s^2]."""
    terms = sa_source_terms(nu_tilde, strain, d, nu, constants)
    return terms["destruction_coefficient"] * np.asarray(nu_tilde, dtype=float)