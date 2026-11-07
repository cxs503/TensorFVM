"""Hand-transposed discrete adjoint for the penalised channel solver.

Given the discrete residual ``R(U, theta) = 0`` and the objective ``J(U, theta)``::

    dJ/dtheta = dJ/dtheta|_U  -  lambda^T dR/dtheta|_U,     A^T lambda = (dJ/dU)^T

with ``A = dR/dU`` the *analytic* Jacobian returned by
:meth:`tensorfvm.adjoint_shape.channel.ChannelSolver.assemble`.  Because the shape
enters only through the Brinkman indicator, ``dR/dtheta`` is local::

    dR_u[j, i]/dtheta_q = dalpha_u[j, i]/dtheta_q * u[j, i] / eta      (same for v)

so the whole sensitivity costs one transposed linear solve per shape, no matter
how many parameters -- and no finite difference touches the gradient path.

Also collects the two verification drivers used by the study:
:func:`taylor_remainder` (slope test of the second-order remainder) and
:func:`central_difference_gradient` (per-parameter central differences with an
epsilon ladder).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse.linalg as spla

from .geometry import N_PARAMS


def adjoint_state(solver, theta, U=None):
    """Return ``lambda`` (the transposed solve) and the objective partials."""
    if U is None:
        U, _ = solver.solve(theta)
    _, A = solver.assemble(theta, U)
    j, dJ_dU, dJ_dtheta = solver.objective(theta, U)
    lam = spla.spsolve(A.T.tocsr(), dJ_dU)
    return lam, {"J": float(j), "dJ_dU": dJ_dU, "dJ_dtheta": dJ_dtheta, "U": U}


def adjoint_gradient(solver, theta, U=None, jacobian=None):
    """Total derivative ``dJ/dtheta`` by the discrete adjoint.

    Returns ``(grad, info)``; ``info`` carries the residual of the transposed
    solve so the caller can audit that the linear system was solved, not fudged.
    """
    theta = np.asarray(theta, dtype=float)
    if U is None:
        U, solve_info = solver.solve(theta)
    else:
        solve_info = None
    _, A = solver.assemble(theta, U)
    j, dJ_dU, dJ_dtheta = solver.objective(theta, U)
    lam = spla.spsolve(A.T.tocsr(), dJ_dU)
    resid = float(np.linalg.norm(A.T @ lam - dJ_dU) / max(np.linalg.norm(dJ_dU), 1e-300))

    g = solver.grid
    eta = solver.eta
    alpha_u, dalpha_u, alpha_v, dalpha_v = solver.geometry(theta)
    Uu = U[: g.nu].reshape(g.ny, g.nx)
    Uv = U[g.nu : g.nu + g.nv].reshape(g.ny - 1, g.nx)
    lam_u = lam[: g.nu].reshape(g.ny, g.nx)
    lam_v = lam[g.nu : g.nu + g.nv].reshape(g.ny - 1, g.nx)

    inner = (
        np.einsum("qji,ji->q", dalpha_u, lam_u * Uu)
        + np.einsum("qji,ji->q", dalpha_v, lam_v * Uv)
    ) / eta
    grad = dJ_dtheta - inner
    return grad, {
        "J": float(j),
        "adjoint_residual": resid,
        "dJ_dtheta_partial": dJ_dtheta.copy(),
        "adjoint_term": inner.copy(),
        "lambda_norm": float(np.linalg.norm(lam)),
        "U": U,
        "solve": solve_info,
        "n_params": N_PARAMS,
    }


def objective_only(solver, theta, U0=None):
    """``J(theta)`` from a fresh Newton solve (used by the finite differences)."""
    U, _ = solver.solve(theta, U0=U0)
    return solver.drag_coefficient(theta, U), U


def taylor_remainder(solver, theta, direction=None, epsilons=None, seed: int = 0):
    """Taylor remainder ``|J(theta + eps d) - J(theta) - eps g.d|`` over an eps ladder."""
    theta = np.asarray(theta, dtype=float)
    if direction is None:
        rng = np.random.default_rng(seed)
        direction = rng.standard_normal(N_PARAMS)
        direction /= np.linalg.norm(direction)
    direction = np.asarray(direction, dtype=float)
    grad, info = adjoint_gradient(solver, theta)
    j0 = info["J"]
    U0 = info["U"]
    if epsilons is None:
        epsilons = 10.0 ** np.arange(-6.0, -1.0 + 1e-9, 0.5)
    rows = []
    for eps in epsilons:
        jp, _ = objective_only(solver, theta + eps * direction, U0=U0)
        jm, _ = objective_only(solver, theta - eps * direction, U0=U0)
        predicted = eps * float(np.dot(grad, direction))
        rows.append(
            {
                "epsilon": float(eps),
                "J_plus": float(jp),
                "J_minus": float(jm),
                "J0": float(j0),
                "first_order_prediction": float(predicted),
                # one-sided remainder, as in the acceptance criterion: the even
                # (quadratic) part survives, so the slope is 2 for a smooth J
                "remainder": float(abs(jp - j0 - predicted)),
                "remainder_symmetric": float(abs(0.5 * (jp - jm) - predicted)),
                "directional_derivative_fd": float((jp - jm) / (2.0 * eps)),
                "directional_derivative_adjoint": float(np.dot(grad, direction)),
            }
        )
    eps = np.array([r["epsilon"] for r in rows])
    rem = np.array([r["remainder"] for r in rows])
    keep = rem > 0.0
    slope = float("nan")
    if keep.sum() >= 3:
        slope = float(np.polyfit(np.log(eps[keep]), np.log(rem[keep]), 1)[0])
    return {"direction": direction, "rows": rows, "slope": slope, "gradient": grad}


def central_difference_gradient(solver, theta, epsilons=(1e-4, 1e-3, 1e-2)):
    """Per-parameter central differences on an epsilon ladder.

    Returns a dict with the adjoint gradient, the FD gradients for each epsilon and
    the per-parameter relative errors.
    """
    theta = np.asarray(theta, dtype=float)
    grad, info = adjoint_gradient(solver, theta)
    U0 = info["U"]
    out = {}
    for eps in epsilons:
        fd = np.zeros(N_PARAMS)
        for q in range(N_PARAMS):
            e = np.zeros(N_PARAMS)
            e[q] = eps
            jp, _ = objective_only(solver, theta + e, U0=U0)
            jm, _ = objective_only(solver, theta - e, U0=U0)
            fd[q] = (jp - jm) / (2.0 * eps)
        denom = np.maximum(np.abs(fd), 1e-300)
        rel = np.abs(grad - fd) / denom
        out[float(eps)] = {
            "fd": fd,
            "rel_error": rel,
            "max_rel_error": float(rel.max()),
            "mean_rel_error": float(rel.mean()),
        }
    return {"adjoint": grad, "by_epsilon": out, "J0": info["J"]}


def gradient_grid_convergence(solvers, theta):
    """Adjoint gradients on a grid ladder and their pairwise relative changes."""
    grads = []
    for s in solvers:
        g, info = adjoint_gradient(s, theta)
        grads.append(g)
    pairs = []
    for a, b in zip(grads[:-1], grads[1:]):
        denom = max(float(np.linalg.norm(b)), 1e-300)
        pairs.append(float(np.linalg.norm(b - a) / denom))
    return {"gradients": grads, "relative_change": pairs}


__all__ = [
    "adjoint_gradient",
    "adjoint_state",
    "central_difference_gradient",
    "gradient_grid_convergence",
    "objective_only",
    "taylor_remainder",
]