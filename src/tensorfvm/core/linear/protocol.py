"""``LinearSolver`` protocol for the five existing linear backends.

The backends are heterogeneous (solver-embedded GMRES, a CPU direct solve, an
AMG-preconditioned block GMRES, a standalone overset Poisson driver, and a
diagnostics routine), so the protocol fixes only the *identity and
availability* surface; ``solve`` keeps each backend's native argument list.
The registry (:mod:`tensorfvm.core.linear.registry`) adds the uniform
registration and fallback-chain machinery on top.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

__all__ = ["LinearSolver", "LINEAR_KINDS"]

#: Backend roles.  ``iterative``/``direct`` solve a linear system;
#: ``preconditioner`` accelerates an outer iteration; ``diagnostics`` analyses
#: a supplied solution of the original system.
LINEAR_KINDS: tuple[str, ...] = ("iterative", "direct", "preconditioner", "diagnostics")


@runtime_checkable
class LinearSolver(Protocol):
    """Uniform identity/availability surface for a linear backend.

    Attributes
    ----------
    name
        Registry name of the backend (e.g. ``'sparse_simple'``).
    kind
        One of :data:`LINEAR_KINDS`.
    """

    name: str
    kind: str

    def available(self) -> bool:
        """Whether the backend can run in this environment (deps present)."""
        ...

    def solve(self, *args: Any, **kwargs: Any) -> Any:
        """Backend-native solve/analysis entry point (delegates, never rewrites)."""
        ...
