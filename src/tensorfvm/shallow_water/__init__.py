"""Shallow-water (depth-averaged) capability: exact dam-break solutions and a
1-D finite-volume solver, with the Ritter/Stoker benchmark harness.

The module is deliberately small and self-contained so that the dam-break
benchmark measures the physics agreement of a classical scheme (HLL + MUSCL +
SSP-RK3) rather than a bespoke discretisation.  It is the shallow-water
companion of the incompressible/VOF solvers in this package: those solve the
full Navier-Stokes equations, whereas the Ritter and Stoker dam-break
references are solutions of the hydrostatic shallow-water system.
"""

from .dam_break import (
    CASES,
    RITTER_DRY,
    STOKER_WET,
    TOLERANCE,
    DamBreakCase,
    convergence_study,
    exactness_checks,
    figure_profiles_and_convergence,
    front_definition_study,
    run_case,
    sensitivity_study,
    write_artifacts,
)
from .exact import (
    check_stoker_solution,
    ritter_front_position,
    ritter_state,
    stoker_constant_state,
    stoker_state,
    t_star_to_time,
    time_to_t_star,
)
from .bed import ShallowWaterBed1D
from .swe2d import ShallowWater2D
from .thacker import ThackerBasin, check_thacker_solution
from .swe1d import ShallowWater1D

__all__ = [
    "CASES",
    "RITTER_DRY",
    "STOKER_WET",
    "TOLERANCE",
    "DamBreakCase",
    "ShallowWater1D",
    "ShallowWaterBed1D",
    "check_stoker_solution",
    "check_thacker_solution",
    "ShallowWater2D",
    "ThackerBasin",
    "convergence_study",
    "exactness_checks",
    "figure_profiles_and_convergence",
    "front_definition_study",
    "ritter_front_position",
    "ritter_state",
    "run_case",
    "sensitivity_study",
    "stoker_constant_state",
    "stoker_state",
    "t_star_to_time",
    "time_to_t_star",
    "write_artifacts",
]
