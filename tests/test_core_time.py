"""Core time protocol: pilot steppers for solver.py and two_phase_solver.py."""
import pytest

from tensorfvm.core.time import (
    Stepper,
    get_stepper,
    list_steppers,
    run_stepper,
)


def test_two_pilots_registered():
    assert set(list_steppers()) == {"simple_2d", "two_phase_2d"}


def test_unknown_stepper_raises():
    with pytest.raises(KeyError):
        get_stepper("no_such_stepper", object())


def test_simple_stepper_delegates():
    from tensorfvm.solver import SimpleSolver, SolverConfig
    solver = SimpleSolver(SolverConfig(nx=8, ny=4, cylinder_radius=None, max_iterations=2))
    stepper = get_stepper("simple_2d", solver)
    assert isinstance(stepper, Stepper)
    items = run_stepper(stepper, 2)
    assert len(items) == 2
    assert len(solver.history) == 2
    assert "continuity" in items[0] and "momentum" in items[0]


def test_two_phase_stepper_delegates():
    from tensorfvm.two_phase_solver import GridConfig, TwoPhaseSolver
    solver = TwoPhaseSolver(GridConfig(nx=8, ny=8))
    stepper = get_stepper("two_phase_2d", solver)
    assert isinstance(stepper, Stepper)
    items = run_stepper(stepper, 2, stop_on_convergence=False)
    assert len(items) == 2
    assert "continuity" in items[0]


def test_run_stepper_requires_positive_steps():
    from tensorfvm.solver import SimpleSolver, SolverConfig
    stepper = get_stepper("simple_2d", SimpleSolver(SolverConfig(nx=8, ny=4, cylinder_radius=None)))
    with pytest.raises(ValueError):
        run_stepper(stepper, 0)
