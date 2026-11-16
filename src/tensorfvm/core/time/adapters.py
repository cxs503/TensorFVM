"""Pilot stepper adapters: ``solver.SimpleSolver`` and
``two_phase_solver.TwoPhaseSolver``.

Only these two of the 39 ``class.step()`` definitions are hooked up (the
strangler pilot); the other 37 are untouched.  The adapters add no time
integration of their own -- ``step`` is a straight delegation and the driver
:func:`run_stepper` only loops and collects the solvers' own metric dicts.
"""

from __future__ import annotations

from typing import Any, Mapping

from ...registry import capability

__all__ = [
    "SimpleSolverStepper",
    "TwoPhaseSolverStepper",
    "run_stepper",
    "register_stepper",
    "get_stepper",
    "list_steppers",
    "STEPPERS",
]


class SimpleSolverStepper:
    """Pilot adapter for ``tensorfvm.solver.SimpleSolver`` (2-D SIMPLE)."""

    name = "simple_2d"

    def __init__(self, solver):
        self.solver = solver

    def step(self) -> dict:
        return self.solver.step()

    @property
    def converged(self) -> bool:
        return self.solver.converged

    @property
    def history(self) -> list:
        return self.solver.history


class TwoPhaseSolverStepper:
    """Pilot adapter for ``tensorfvm.two_phase_solver.TwoPhaseSolver``."""

    name = "two_phase_2d"

    def __init__(self, solver):
        self.solver = solver

    def step(self, dt: float | None = None) -> dict:
        return self.solver.step(dt)

    @property
    def converged(self) -> bool:
        return self.solver.converged

    @property
    def history(self) -> list:
        return self.solver.history


STEPPERS: dict[str, type] = {}


def register_stepper(name: str, stepper_cls: type, *, doc: str = "") -> None:
    """Register a stepper adapter class under ``name``."""
    if name in STEPPERS:
        raise ValueError(f"stepper {name!r} is already registered")
    STEPPERS[name] = stepper_cls


def list_steppers() -> list[str]:
    """Names of all registered steppers, sorted."""
    return sorted(STEPPERS)


@capability('fvm.core.time_stepper',
            kind='time_integration',
            maturity='developing',
            test_path='tests/test_core_time.py',
            wired=True,
            tested=True,
            verified=False,
            evidence={
                'grade': 'MECHANICS',
                'note': ('Stepper protocol pilot over solver.SimpleSolver and '
                         'two_phase_solver.TwoPhaseSolver; the other 37 step '
                         'definitions are intentionally untouched'),
                'docs': 1,
            })
def get_stepper(name: str, solver: Any):
    """Instantiate the registered stepper adapter around an existing solver."""
    cls = STEPPERS.get(name)
    if cls is None:
        raise KeyError(f"unknown stepper {name!r}; known: {sorted(STEPPERS)}")
    return cls(solver)


def run_stepper(stepper, max_steps: int, *, dt: float | None = None,
                stop_on_convergence: bool = True) -> list[dict]:
    """Drive a stepper for up to ``max_steps`` steps.

    Returns the list of metric dicts returned by the solver's own ``step``.
    ``dt`` is only forwarded to steppers that accept it (the two-phase pilot);
    steady steppers are called with no argument.
    """
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    items: list[dict] = []
    for _ in range(max_steps):
        items.append(stepper.step(dt) if dt is not None else stepper.step())
        if stop_on_convergence and stepper.converged:
            break
    return items


register_stepper("simple_2d", SimpleSolverStepper,
                doc="2-D SIMPLE steady iteration (solver.py)")
register_stepper("two_phase_2d", TwoPhaseSolverStepper,
                doc="2-D single/two-phase projection (two_phase_solver.py)")
