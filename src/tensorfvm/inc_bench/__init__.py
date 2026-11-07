"""Independent incompressible-flow verification benchmarks for the FVM kernel.

Modules
-------
``solver``     steady masked MAC/SIMPLE kernels (forced and buoyant)
``reference``  literature and analytic reference values with provenance
``cases``      the three benchmark cases, their metrics and acceptance criteria
``report``     machine-checkable JSON summaries
"""

from .solver import SolverOptions, SteadyFlow, cell_velocity, streamfunction

__all__ = ["SolverOptions", "SteadyFlow", "cell_velocity", "streamfunction"]