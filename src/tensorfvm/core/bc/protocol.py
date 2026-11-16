"""Boundary-condition type registry.

m23215 counted nine boundary-related modules, each writing its boundary
treatment inline, with no shared BC abstraction.  This registry gives the BC
types a name and a uniform ``apply`` entry point; the adapters delegate to the
existing modules and add no numerics of their own.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

__all__ = ["BoundaryCondition"]


@runtime_checkable
class BoundaryCondition(Protocol):
    """Uniform identity/apply surface for a registered BC type.

    Attributes
    ----------
    kind
        Registry name of the BC type (e.g. ``'free_surface_open'``).
    """

    kind: str

    def apply(self, *args: Any, **kwargs: Any) -> Any:
        """Apply the boundary condition (delegates to the source module)."""
        ...
