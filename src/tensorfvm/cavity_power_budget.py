"""Steady-state power / drag budget for the 2-D lid-driven cavity.

Why this module exists
----------------------
The acceptance gate of ``CASE-FVM-CAVITY-RE1000`` scores the *shape* of the two
centreline profiles and the primary-vortex centre against Ghia et al. (1982).
Those metrics say nothing about whether the merged solution satisfies the two
**exact** global identities that any steady incompressible solution of this
configuration must satisfy, whatever the reference table says:

* **Input power equals dissipation** (unit square, side ``L``, lid speed ``U``,
  density ``rho``, dynamic viscosity ``mu = rho nu``)::

      P_in = U * F_lid,   F_lid = mu * int_0^L (du/dy)|_{y=L} dx
      eps  = mu * int_A |grad u|^2 dA
      P_in = eps                        (steady state, no normal flow at walls)

* **Stokes' theorem / total vorticity** (lid moves in +x, other walls at rest,
  flow therefore clockwise and ``omega < 0``)::

      int_A omega dA = -U L

Both are *reference-free*: they need no literature table, only the solution
field.  The relative residual of the first identity is the fraction of the
mechanical input power that the discretisation does not hand back as resolved
dissipation, i.e. an upper bound on the numerical dissipation of the scheme at
that grid.  The second identity measures how much net vorticity the wall
treatment fails to represent.

They are computed here for two independent discretisations:

* ``budget_staggered`` - the production MAC finite-volume solver
  (:class:`tensorfvm.lid_cavity.LidCavitySolver`), whose fields are staggered
  (``u`` on x-faces, ``v`` on y-faces);
* ``budget_spectral`` - the Chebyshev collocation streamfunction-vorticity
  reference (:class:`tensorfvm.cavity_spectral.SpectralCavity`).

Running both is the point: the spectral solution is accurate enough that its
budget residual is a lower bound on what the audit *operator* can resolve, so
the FVM residual can be attributed to the FVM discretisation rather than to the
measurement.

Conventions
-----------
Coordinates are normalised by ``L``, velocities by ``U`` (so ``U = 1``,
``L = 1``), ``rho = 1`` and ``mu = nu = 1/Re``.  ``omega = dv/dx - du/dy``.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "staggered_cell_velocity",
    "staggered_gradients",
    "budget_staggered",
    "budget_spectral",
    "budget_from_artifact",
]


# --------------------------------------------------------------------------- #
# staggered (MAC) fields -> cell-centred gradients
# --------------------------------------------------------------------------- #
def staggered_cell_velocity(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cell-centred velocity from staggered faces.

    ``u`` has shape ``(n + 1, n)`` with ``u[i, j]`` at ``x = i h``,
    ``y = (j + 1/2) h``; ``v`` has shape ``(n, n + 1)`` with ``v[i, j]`` at
    ``x = (i + 1/2) h``, ``y = j h``.  Arithmetic face averages, exactly as in
    :meth:`LidCavitySolver.cell_velocity`.
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    if u.ndim != 2 or v.ndim != 2:
        raise ValueError("u and v must be 2-D staggered arrays")
    n = u.shape[1]
    if v.shape[0] != n or u.shape != (n + 1, n) or v.shape != (n, n + 1):
        raise ValueError(f"inconsistent staggered shapes {u.shape} / {v.shape}")
    uc = 0.5 * (u[:-1, :] + u[1:, :])
    vc = 0.5 * (v[:, :-1] + v[:, 1:])
    return uc, vc


def staggered_gradients(u: np.ndarray, v: np.ndarray, h: float,
                        lid_velocity: float = 1.0) -> dict[str, np.ndarray]:
    """Cell-centred velocity gradients of a MAC field (second order).

    Interior stencils are centred differences of the cell-centred velocities;
    at the walls the ghost values carry the boundary condition (no slip on the
    three fixed walls, ``u = lid_velocity`` on ``y = L``), which reproduces a
    linear field *exactly* (see ``tests/test_cavity_power_budget.py``).
    """
    uc, vc = staggered_cell_velocity(u, v)
    n = uc.shape[0]
    if n < 3:
        raise ValueError("n >= 3 required")

    def _dd_axis(a: np.ndarray, axis: int, ghost_lo: float, ghost_hi: float) -> np.ndarray:
        """Centred difference along ``axis`` with prescribed ghost-cell values."""
        out = np.empty_like(a)
        d = (slice(None),) * axis
        lo = list(a.shape)
        if axis == 0:
            out[1:-1, :] = (a[2:, :] - a[:-2, :]) / (2.0 * h)
            out[0, :] = (a[1, :] - (2.0 * ghost_lo - a[0, :])) / (2.0 * h)
            out[-1, :] = ((2.0 * ghost_hi - a[-1, :]) - a[-2, :]) / (2.0 * h)
        else:
            out[:, 1:-1] = (a[:, 2:] - a[:, :-2]) / (2.0 * h)
            out[:, 0] = (a[:, 1] - (2.0 * ghost_lo - a[:, 0])) / (2.0 * h)
            out[:, -1] = ((2.0 * ghost_hi - a[:, -1]) - a[:, -2]) / (2.0 * h)
        del d, lo
        return out

    # u: tangential on the y-walls (0 at y=0, lid_velocity at y=L), normal on x-walls (0)
    # v: normal on the y-walls (0) and tangential on the x-walls (0)
    du_dy = _dd_axis(uc, 1, 0.0, float(lid_velocity))
    dv_dy = _dd_axis(vc, 1, 0.0, 0.0)
    du_dx = _dd_axis(uc, 0, 0.0, 0.0)
    dv_dx = _dd_axis(vc, 0, 0.0, 0.0)
    return {"du_dx": du_dx, "du_dy": du_dy, "dv_dx": dv_dx, "dv_dy": dv_dy}


def _trapezoid(values: np.ndarray, h: float) -> float:
    """Trapezoidal integral of samples on an evenly spaced grid (both ends)."""
    values = np.asarray(values, dtype=float).ravel()
    return float(h * (0.5 * values[0] + values[1:-1].sum() + 0.5 * values[-1]))


def budget_staggered(u, v, h: float, nu: float, lid_velocity: float = 1.0,
                     rho: float = 1.0) -> dict:
    """Power / drag / vorticity budget of a staggered MAC field.

    ``F_lid`` uses the *same* one-sided wall gradient the solver applies through
    its mirror ghost, ``(du/dy)|_wall = 2 (U - u_adjacent) / h``; ``eps`` is the
    resolved dissipation from centred differences of the cell-centred field.  The
    two are therefore measured by different stencils, which is deliberate: the
    identity ``P_in = eps`` is exact for the continuum solution, so a non-zero
    residual is a property of the discrete field, not of a shared stencil.
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    n = u.shape[1]
    mu = rho * nu
    uc, vc = staggered_cell_velocity(u, v)
    g = staggered_gradients(u, v, h, lid_velocity=lid_velocity)

    eps = mu * float((g["du_dx"] ** 2 + g["du_dy"] ** 2
                      + g["dv_dx"] ** 2 + g["dv_dy"] ** 2).sum()) * h * h
    # lid drag: ghost-slope shear at y = L, integrated over the lid
    wall_slope = 2.0 * (lid_velocity - u[:, -1]) / h
    force = mu * _trapezoid(wall_slope, h)
    power = force * lid_velocity
    omega = g["dv_dx"] - g["du_dy"]
    kinetic = 0.5 * rho * float((uc ** 2 + vc ** 2).sum()) * h * h
    return {
        "n": int(n),
        "h": float(h),
        "nu": float(nu),
        "lid_force": float(force),
        "power_in": float(power),
        "dissipation": float(eps),
        "power_residual": float(power - eps),
        "power_residual_rel": float((power - eps) / power) if power else float("nan"),
        "numerical_dissipation_share": float(max(power - eps, 0.0) / power) if power else float("nan"),
        "lid_force_coefficient": float(force / (0.5 * rho * lid_velocity ** 2)),
        "total_vorticity": float(omega.sum() * h * h),
        "total_vorticity_rel": float(omega.sum() * h * h / (-lid_velocity * 1.0)),
        "kinetic_energy": kinetic,
        "max_wall_slope": float(np.abs(wall_slope).max()),
    }


# --------------------------------------------------------------------------- #
# spectral (Chebyshev collocation) fields
# --------------------------------------------------------------------------- #
def budget_spectral(psi: np.ndarray, omega: np.ndarray, nu: float,
                    lid_velocity: float = 1.0, rho: float = 1.0,
                    order: int | None = None) -> dict:
    """Same budget for a Chebyshev solution on Chebyshev-Gauss-Lobatto nodes.

    ``psi``/``omega`` have shape ``(N+1, N+1)`` indexed ``[iy, ix]`` on the unit
    square (node 0 at ``x = 0``, node ``N`` at ``x = 1``).  Integration uses the
    Clenshaw-Curtis weights, which are exact for the polynomial degree involved,
    so the budget residual here is a property of the *solution*, not of the
    quadrature.

    ``u = psi_y``, ``v = -psi_x`` hence ``|grad u|^2 = psi_xx^2 + psi_yy^2
    + 2 psi_xy^2``.  On the lid ``psi`` is constant along the wall, so
    ``omega_wall = -psi_yy`` and ``F_lid = -mu int omega dx``.
    """
    from .cavity_spectral import clenshaw_curtis_weights, differentiation_matrices

    psi = np.asarray(psi, dtype=float)
    omega = np.asarray(omega, dtype=float)
    if psi.shape != omega.shape or psi.ndim != 2 or psi.shape[0] != psi.shape[1]:
        raise ValueError("psi and omega must be equal square 2-D arrays")
    n = psi.shape[0] - 1 if order is None else int(order)
    if n != psi.shape[0] - 1:
        raise ValueError("order does not match the field size")
    mu = rho * nu
    xi, D1, D2 = differentiation_matrices(n)
    # x = (1 - xi)/2, y = (1 - xi)/2  =>  d/dx = d/dy = -2 d/dxi
    Ax = -2.0 * D1
    A2 = 4.0 * D2
    psi_x = psi @ Ax.T
    psi_y = Ax @ psi
    psi_xx = psi @ A2.T
    psi_yy = A2 @ psi
    psi_xy = Ax @ psi @ Ax.T
    u, v = psi_y, -psi_x
    del psi_x, psi_y

    w = clenshaw_curtis_weights(xi) / 2.0          # [-1, 1] -> [0, 1]
    quad2 = lambda f: float(w @ f @ w)             # noqa: E731

    eps = mu * quad2(psi_xx ** 2 + psi_yy ** 2 + 2.0 * psi_xy ** 2)
    force = -mu * float(w @ omega[-1, :])          # lid = last node row (y = 1)
    power = force * lid_velocity
    kinetic = 0.5 * rho * quad2(u ** 2 + v ** 2)
    total_vorticity = quad2(omega)
    return {
        "n": int(n),
        "nu": float(nu),
        "lid_force": float(force),
        "power_in": float(power),
        "dissipation": float(eps),
        "power_residual": float(power - eps),
        "power_residual_rel": float((power - eps) / power) if power else float("nan"),
        "numerical_dissipation_share": float(max(power - eps, 0.0) / power) if power else float("nan"),
        "lid_force_coefficient": float(force / (0.5 * rho * lid_velocity ** 2)),
        "total_vorticity": float(total_vorticity),
        "total_vorticity_rel": float(total_vorticity / (-lid_velocity * 1.0)),
        "kinetic_energy": float(kinetic),
    }


def budget_from_artifact(artifact: dict, nu: float | None = None,
                         lid_velocity: float | None = None) -> dict:
    """Convenience wrapper: budget straight from a saved resolution artefact."""
    u = np.asarray(artifact["u_field"] if "u_field" in artifact else artifact["u"])
    v = np.asarray(artifact["v_field"] if "v_field" in artifact else artifact["v"])
    h = float(artifact["h"])
    nu = float(artifact.get("viscosity", 1.0 / artifact["reynolds"]) if nu is None else nu)
    lid = float(artifact.get("lid_velocity", 1.0) if lid_velocity is None else lid_velocity)
    return budget_staggered(u, v, h, nu, lid_velocity=lid)