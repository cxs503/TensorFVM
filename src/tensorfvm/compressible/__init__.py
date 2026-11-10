"""Compressible multi-material finite-volume surface (two-material Euler).

Entry points
------------
``EulerSolver``
    MUSCL-Hancock + HLLC solver on a uniform Cartesian grid, with an optional
    ghost-fluid interface treatment at material faces.
``cases``
    Sod, air/He material contact, Richtmyer--Meshkov (air/SF6), shock--bubble
    (Mach 1.22 He cylinder) and co-rotating compressible vortex pair.
``verification.run_all``
    Runs the verification legs and writes the JSON artifact plus figures.

The capability is *new* to TensorFVM: before this module the repository had no
compressible solver at all (every solver was incompressible/projection based),
so the whole compressible path -- EOS, Riemann solvers, reconstruction,
interface treatment, conservation ledger -- is introduced here and verified
against exact Riemann solutions and analytic instability theory.
"""

from __future__ import annotations

from .eos import StiffenedGasMixture
from .riemann import ExactRiemann, hllc_flux, riemann_star
from .solver import EulerSolver
from . import cases, diagnostics, verification, vortex

__all__ = [
    "StiffenedGasMixture",
    "ExactRiemann",
    "EulerSolver",
    "hllc_flux",
    "riemann_star",
    "cases",
    "diagnostics",
    "verification",
    "vortex",
]
