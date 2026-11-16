"""Boundary-condition type registry and adapters."""

from .protocol import BoundaryCondition
from .registry import (
    BC_TYPES,
    BCEntry,
    FreeSurfaceBCAdapter,
    PressureTraceAdapter,
    RigidMotionConstraintAdapter,
    get_bc,
    list_bc_types,
    register_bc,
)

__all__ = [
    "BoundaryCondition",
    "BCEntry",
    "FreeSurfaceBCAdapter",
    "PressureTraceAdapter",
    "RigidMotionConstraintAdapter",
    "register_bc",
    "get_bc",
    "list_bc_types",
    "BC_TYPES",
]
