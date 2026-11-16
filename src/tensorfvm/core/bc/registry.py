"""BC type registry with adapters for the existing boundary modules.

Adapted (of the nine boundary modules m23215 counted):

* ``free_surface_bc.FreeSurfaceBC``            -> ``free_surface_open``
* ``native_boundary_pressure3d.NativeBoundaryPressureTrace3D``
                                                -> ``pressure_trace``
* ``native_boundary_constraints3d.require_rigid_motion_constraints``
                                                -> ``rigid_motion_constraint``

The remaining six (``external_open_boundary``, ``developed_boundary_audit``,
``external_boundary_audit``, ``integral_boundary_layer`` and the two audit
helpers) are solvers/audits/integral models rather than reusable BC operators;
they are listed in ``docs/m23813/CORE_SKELETON.md`` with the reason they are
not registered as BC types.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ...registry import capability

__all__ = [
    "BCEntry",
    "FreeSurfaceBCAdapter",
    "PressureTraceAdapter",
    "RigidMotionConstraintAdapter",
    "register_bc",
    "get_bc",
    "list_bc_types",
    "BC_TYPES",
]


class FreeSurfaceBCAdapter:
    """``free_surface_open``: open-top / wall / outlet BC manager (2-D MAC)."""

    kind = "free_surface_open"

    def __init__(self, bc):
        self.bc = bc

    def apply(self, p: Any, u: Any, v: Any):
        """Delegate to the manager: pressure mask + in-place velocity BCs."""
        p_new = self.bc.apply_pressure(p)
        u_new, v_new = self.bc.apply_velocity_bc(u, v)
        return p_new, u_new, v_new


class PressureTraceAdapter:
    """``pressure_trace``: least-squares pressure trace on 3-D boundary faces."""

    kind = "pressure_trace"

    def __init__(self, trace):
        self.trace = trace

    def apply(self, cell_pressure: Any) -> dict:
        return self.trace.reconstruct(cell_pressure)


class RigidMotionConstraintAdapter:
    """``rigid_motion_constraint``: six-mode rigid-motion rank check (3-D)."""

    kind = "rigid_motion_constraint"

    def __init__(self, mesh):
        self.mesh = mesh

    def apply(self, dirichlet_components: Any, robin_coefficient: Any = None,
              normal_velocity_faces: Any = None) -> dict:
        from ...native_boundary_constraints3d import require_rigid_motion_constraints
        return require_rigid_motion_constraints(
            self.mesh, dirichlet_components,
            robin_coefficient=robin_coefficient,
            normal_velocity_faces=normal_velocity_faces)


@dataclass(frozen=True)
class BCEntry:
    """One registered BC type: a factory producing a ``BoundaryCondition``."""

    kind: str
    factory: Callable[..., Any]
    doc: str = ""


BC_TYPES: dict[str, BCEntry] = {}


def register_bc(kind: str, factory: Callable[..., Any], *, doc: str = "") -> None:
    """Register a BC type (idempotent per process)."""
    if kind in BC_TYPES:
        raise ValueError(f"BC type {kind!r} is already registered")
    BC_TYPES[kind] = BCEntry(kind, factory, doc)


@capability('fvm.core.bc_registry',
            kind='boundary',
            maturity='developing',
            test_path='tests/test_core_bc.py',
            wired=True,
            tested=True,
            verified=False,
            evidence={
                'grade': 'MECHANICS',
                'note': ('BC type registry; adapters delegate to '
                         'free_surface_bc / native_boundary_pressure3d / '
                         'native_boundary_constraints3d without rewriting them'),
                'docs': 1,
            })
def get_bc(kind: str, *args: Any, **kwargs: Any):
    """Instantiate a registered BC type."""
    entry = BC_TYPES.get(kind)
    if entry is None:
        raise KeyError(f"unknown BC type {kind!r}; known: {sorted(BC_TYPES)}")
    return entry.factory(*args, **kwargs)


def list_bc_types() -> list[str]:
    """Names of all registered BC types, sorted."""
    return sorted(BC_TYPES)


register_bc("free_surface_open", FreeSurfaceBCAdapter,
            doc="open-top / no-slip-wall / side-outlet manager (2-D MAC grid)")
register_bc("pressure_trace", PressureTraceAdapter,
            doc="internal-cell least-squares pressure trace on 3-D boundary faces")
register_bc("rigid_motion_constraint", RigidMotionConstraintAdapter,
            doc="six-mode rigid-motion constraint audit for native 3-D solves")
