"""Laminar internal-flow verification cases on the staggered MAC solver.

The three cases in this package are deliberately independent of the LBM
verification chain that lives in :mod:`tensorfvm.verification`:

* :mod:`tensorfvm.internal_flow.backward_step` -- backward-facing step at
  ``Re_H = 800`` (reattachment length, wall friction distribution, mesh
  convergence),
* :mod:`tensorfvm.internal_flow.sudden_expansion` -- planar sudden expansion
  with area ratio 4 at ``Re_D = 100`` (reattachment length, pressure-loss
  coefficient vs Borda-Carnot),
* :mod:`tensorfvm.internal_flow.natural_convection` -- differentially heated
  square cavity at ``Ra = 1e4 ... 1e6``, ``Pr = 0.71`` (mean Nusselt number vs
  the de Vahl Davis benchmarks).

Shared physics/verification values live in
:mod:`tensorfvm.internal_flow.reference`; the driver that produces the
artifacts of the published report is :mod:`tensorfvm.internal_flow.study`.
"""

from .mac import MACConfig, StaggeredMAC, vanleer_face

__all__ = ["MACConfig", "StaggeredMAC", "vanleer_face"]