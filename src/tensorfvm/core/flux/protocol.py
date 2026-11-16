"""``FluxFunction`` protocol for face-flux / interface-transfer operators.

The four flux families in the repository (VOF PLIC, VOF CICSAM, VOF THINC and
compressible slope limiting) plus the overset transfer each have their own
native argument lists, so the protocol fixes only the *family identity* and a
single ``flux`` entry point; the adapters delegate to the existing functions
without re-implementing any numerics.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

__all__ = ["FluxFunction"]


@runtime_checkable
class FluxFunction(Protocol):
    """Uniform entry point for a registered flux family.

    Attributes
    ----------
    family
        Registry name of the family (e.g. ``'vof_plic'``).
    """

    family: str

    def flux(self, *args: Any, **kwargs: Any) -> Any:
        """Backend-native flux evaluation (delegates to the source module)."""
        ...
