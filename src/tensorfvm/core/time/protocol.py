"""``Stepper`` protocol for time-advancing solvers.

m23215 counted 39 modules defining their own ``class.step()`` with no shared
time-integration layer.  This protocol is the pilot: only
``solver.SimpleSolver`` and ``two_phase_solver.TwoPhaseSolver`` are hooked up
here; the other 37 step definitions are left untouched (strangler pattern).
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable

__all__ = ["Stepper"]


@runtime_checkable
class Stepper(Protocol):
    """Uniform surface for a time-advancing solver.

    Attributes
    ----------
    name
        Registry name of the stepper (e.g. ``'simple_2d'``).
    """

    name: str

    def step(self, *args: Any, **kwargs: Any) -> Mapping[str, Any]:
        """Advance one step and return the solver's metric dict (delegated)."""
        ...

    @property
    def converged(self) -> bool:
        """The solver's own convergence flag."""
        ...
