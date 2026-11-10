"""Thermal convection verification face of TensorFVM.

Three modules, each answering one question the campaign asks of the platform:

* :mod:`tensorfvm.thermal.stability` -- *does the layer convect?*  Linear
  stability of the conduction state: the marginal Rayleigh number ``Ra_c`` and
  the critical wavenumber, against the closed-form (stress-free, 657.51) and
  textbook (rigid, 1707.76) anchors.
* :mod:`tensorfvm.thermal.rayleigh_benard` -- *how much heat does it carry?*
  Nonlinear Boussinesq layer between a hot lower and a cold upper plate, plate
  Nusselt numbers, energy-conservation residual, vertical flux profile.
* :mod:`tensorfvm.thermal.scaling` -- *how does that transport scale?*  The
  ``Nu(Ra)`` power-law fit with an honest uncertainty, and the acceptance ledger
  that turns every number of this face into a pass/fail criterion.

Reference values live in :mod:`tensorfvm.thermal.reference`; the driver that
produces the published artifacts is :mod:`tensorfvm.thermal.study`.
"""

from .rayleigh_benard import RayleighBenardCase, RayleighBenardConfig
from .reference import (
    NU_EXPONENT_CLASSICAL,
    PRANDTL_AIR,
    RA_CRITICAL_FREE,
    RA_CRITICAL_RIGID,
    WAVENUMBER_CRITICAL_FREE,
    WAVENUMBER_CRITICAL_RIGID,
)
from .scaling import evaluate_criteria, fit_power_law
from .stability import StabilityConfig, critical_rayleigh, growth_rate, spectrum

__all__ = [
    "RayleighBenardCase",
    "RayleighBenardConfig",
    "StabilityConfig",
    "critical_rayleigh",
    "growth_rate",
    "spectrum",
    "fit_power_law",
    "evaluate_criteria",
    "RA_CRITICAL_FREE",
    "RA_CRITICAL_RIGID",
    "WAVENUMBER_CRITICAL_FREE",
    "WAVENUMBER_CRITICAL_RIGID",
    "NU_EXPONENT_CLASSICAL",
    "PRANDTL_AIR",
]