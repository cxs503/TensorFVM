"""Drag-reduction loop driven by the discrete-adjoint gradient.

L-BFGS-B (bound constrained, so the star-shaped parametrisation stays valid) with
the gradient supplied by :func:`tensorfvm.adjoint_shape.adjoint.adjoint_gradient`.
The shape parameters live on the *geometry*, not on the grid, so the optimised
shape can be re-evaluated on a refined mesh without any interpolation -- that is
what makes the mesh-independence check of the closed loop meaningful.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .adjoint import adjoint_gradient
from .geometry import N_PARAMS, PARAM_NAMES

#: Box constraints on the Fourier amplitudes (keeps ``R(phi) > 0``).
DEFAULT_BOUND = 0.12


def drag_reduction(
    solver,
    theta0=None,
    maxiter: int = 20,
    bound: float = DEFAULT_BOUND,
    theta=None,
):
    """Run L-BFGS-B on ``J = C_D`` with discrete-adjoint gradients.

    Returns a dict with the objective history, the gradient history and the
    final shape; every iterate is recorded from a fresh Newton solve so the
    reported trajectory contains no warm-start artefacts.
    """
    if theta0 is None:
        theta0 = np.zeros(N_PARAMS)
    theta0 = np.asarray(theta0, dtype=float)
    history = {"J": [], "theta": [], "grad_norm": [], "adjoint_residual": []}
    counter = {"n": 0}

    def fun(x):
        counter["n"] += 1
        grad, info = adjoint_gradient(solver, x)
        history["J"].append(info["J"])
        history["theta"].append(np.array(x, dtype=float))
        history["grad_norm"].append(float(np.linalg.norm(grad)))
        history["adjoint_residual"].append(info["adjoint_residual"])
        return info["J"], grad

    res = minimize(
        fun,
        theta0,
        jac=True,
        method="L-BFGS-B",
        bounds=[(-bound, bound)] * theta0.size,
        options={"maxiter": maxiter, "maxfun": 2 * maxiter + 5, "ftol": 0.0, "gtol": 0.0},
    )
    j0 = history["J"][0]
    jf = history["J"][-1]
    return {
        "theta0": theta0,
        "theta_opt": np.asarray(res.x, dtype=float),
        "J0": float(j0),
        "J_final": float(jf),
        "reduction": float((j0 - jf) / j0),
        "history": {k: list(v) for k, v in history.items()},
        "n_objective_calls": counter["n"],
        "n_iterations": int(res.nit),
        "message": str(res.message),
        "param_names": list(PARAM_NAMES),
    }


def evaluate_shape(solver, theta):
    """``(J, U, info)`` for a shape on an arbitrary grid (mesh-independence check)."""
    U, info = solver.solve(theta)
    j = solver.drag_coefficient(theta, U)
    return float(j), U, info


__all__ = ["DEFAULT_BOUND", "drag_reduction", "evaluate_shape"]