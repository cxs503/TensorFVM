"""TensorFVM kernel core: the five-interface adapter layer (strangler pattern).

m23215's architecture audit (``docs/m23215/KERNEL_ARCHITECTURE.md``) found that
the repository has no shared mesh / flux / boundary-condition / linear-solver /
time-integration layer: one ``Protocol`` (``mesh_api.Mesh2D``, 2-D only), 39
modules each defining their own ``class.step()``, five linear backends with no
protocol, and two orphaned shared pieces (``overset_flux``, ``pmm_solver``)
with zero ``src`` references.

This package is the strangler: every adapter here *delegates* to an existing
module and never rewrites its numerics.  Old modules are not modified; new
callers may migrate to the contracts below one family at a time.

Subpackages
-----------
``core.mesh``
    ``MeshData`` -- a 2-D/3-D finite-volume mesh contract seeded from
    ``mesh_api.Mesh2D`` -- plus adapters for the existing mesh classes.
``core.linear``
    ``LinearSolver`` protocol, backend registry and fallback chain over the
    five existing linear backends.
``core.flux``
    ``FluxFunction`` protocol and the registry of the four flux families
    (VOF PLIC / CICSAM / THINC, compressible slope limiting) plus the
    previously orphaned ``overset_flux`` transfer.
``core.bc``
    Boundary-condition type registry with adapters for the existing
    boundary modules.
``core.time``
    ``Stepper`` protocol with pilot adapters for ``solver.SimpleSolver`` and
    ``two_phase_solver.TwoPhaseSolver``.

Contract and adaptation matrix: ``docs/m23813/CORE_SKELETON.md``.
"""

from . import bc, flux, linear, mesh, time  # noqa: F401

__all__ = ["mesh", "linear", "flux", "bc", "time"]
