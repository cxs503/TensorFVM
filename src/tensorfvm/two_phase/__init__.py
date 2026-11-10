"""Two-phase flow verification chain: drift-flux, PLIC-VOF, cavitation, Stefan.

The package bundles four independently verifiable building blocks plus the
acceptance chain that ties them together:

* :mod:`tensorfvm.two_phase.drift_flux` -- 1-D drift-flux model for vertical
  gas-liquid pipe flow (Wallis 1969 closed-form void fraction + conservative FVM);
* :mod:`tensorfvm.two_phase.vof` -- 2-D PLIC geometric VOF interface transport;
* :mod:`tensorfvm.two_phase.cavitation_transport` -- Rayleigh-Plesset bubble
  dynamics, the Plesset linear reference and the cavitation void-transport
  equation;
* :mod:`tensorfvm.two_phase.stefan` -- one-phase Stefan phase change (enthalpy
  method vs the Carslaw & Jaeger similarity solution);
* :mod:`tensorfvm.two_phase.chain` -- the four-segment acceptance chain with a
  machine-checked criterion table;
* :mod:`tensorfvm.two_phase.provenance` -- the SQLite ledger that makes every
  reported number traceable (and prevents hand-copied numbers).

Evidence: ``docs/two_phase_chain/`` (report JSON, SQLite
ledger, figures) and ``docs/reports/P254W1-N01_two_phase_chain.md``.
"""

from __future__ import annotations

from .cavitation_transport import (
    BubbleConfig,
    CavitationTransportConfig,
    free_oscillation,
    integrate_bubble,
    linear_frequency,
    measured_frequency,
    plesset_solution,
    rayleigh_growth,
    solve_void_transport,
)
from .drift_flux import (
    DriftFluxConfig,
    compare_profiles,
    homogeneous_void_fraction,
    solve_drift_flux,
    wallis_drift_velocity,
    wallis_profile,
    wallis_void_fraction,
)
from .provenance import ProvenanceDB, sha256_file
from .stefan import (
    StefanConfig,
    conduction_temperature,
    neumann_front,
    neumann_temperature,
    solve_stefan_1d,
    stefan_lambda,
    stefan_lambda_two_phase,
)
from .vof import PLICVOF, dam_break_run, partial_volume, static_interface_drift

__all__ = [
    "BubbleConfig",
    "CavitationTransportConfig",
    "DriftFluxConfig",
    "PLICVOF",
    "ProvenanceDB",
    "StefanConfig",
    "compare_profiles",
    "conduction_temperature",
    "dam_break_run",
    "free_oscillation",
    "homogeneous_void_fraction",
    "integrate_bubble",
    "linear_frequency",
    "measured_frequency",
    "neumann_front",
    "neumann_temperature",
    "partial_volume",
    "plesset_solution",
    "rayleigh_growth",
    "sha256_file",
    "solve_drift_flux",
    "solve_stefan_1d",
    "solve_void_transport",
    "static_interface_drift",
    "stefan_lambda",
    "stefan_lambda_two_phase",
    "wallis_drift_velocity",
    "wallis_profile",
    "wallis_void_fraction",
]
