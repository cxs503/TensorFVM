"""Discrete-adjoint shape sensitivity for the incompressible MAC solver.

Module map
----------
``geometry``   star-shaped Fourier body ``R(phi; theta)`` (8 parameters) and its
               compact-support penalisation indicator with an analytic
               ``d alpha / d theta``.
``channel``    steady incompressible MAC solver on a streamwise-periodic channel
               driven by a constant body force; returns the residual **and** its
               analytic Jacobian for the penalty-parametrised shape.
``adjoint``    hand-transposed discrete adjoint (``A^T lambda = dJ/dU``) giving
               ``dC_D/dtheta``, plus the Taylor-remainder and central-difference
               verification drivers.
``optimize``   L-BFGS-B drag-reduction loop on the adjoint gradient.
``study``      full verification/optimisation driver that writes ``result.json``
               and the report figures.

Nothing in the sensitivity path is differenced numerically: the shape enters only
through the Brinkman indicator, whose ``theta`` derivative is closed form, and the
Jacobian is assembled analytically (verified against dense finite differences in
``tests/test_adjoint_shape_sensitivity.py``).
"""

from __future__ import annotations

__version__ = "0.1.0"

from .adjoint import (  # noqa: F401
    adjoint_gradient,
    central_difference_gradient,
    gradient_grid_convergence,
    taylor_remainder,
)
from .channel import ChannelGrid, ChannelSolver, default_solver, poiseuille  # noqa: F401
from .geometry import MODES, N_PARAMS, PARAM_NAMES, FourierBody  # noqa: F401
from .optimize import drag_reduction, evaluate_shape  # noqa: F401

__all__ = [
    "ChannelGrid",
    "ChannelSolver",
    "FourierBody",
    "MODES",
    "N_PARAMS",
    "PARAM_NAMES",
    "__version__",
    "adjoint_gradient",
    "central_difference_gradient",
    "default_solver",
    "drag_reduction",
    "evaluate_shape",
    "gradient_grid_convergence",
    "poiseuille",
    "taylor_remainder",
]