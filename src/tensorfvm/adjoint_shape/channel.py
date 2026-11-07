"""Steady incompressible MAC solver on a streamwise-periodic channel.

Why periodic
------------
A pressure-outlet patch (Dirichlet ``p`` replacing a continuity row) leaves a
spurious near-null mode of the saddle system: the outflow velocity is no longer
tied to mass conservation, the Jacobian becomes singular to working precision and
Newton stalls at a few percent of the initial residual.  Instead the channel is
**periodic in x** and driven by a constant body force ``f``.  Then

* every face is an interior face (no outflow defect at all),
* the only null direction is the constant pressure, pinned by one row,
* and the body-free case has an *exact* discrete solution -- plane Poiseuille
  ``u(y) = f y (H - y) / (2 nu)`` -- which the solver must reproduce to machine
  precision (used as the analytic benchmark).

Discretisation
--------------
Uniform staggered (MAC) grid, ``u`` on vertical faces (``ny x nx``), ``v`` on
horizontal faces (``(ny-1) x nx``, walls eliminated), ``p`` at cell centres
(``ny x nx``).  Diffusion is central, walls use a half cell distance for the
wall-normal ``u`` gradient, and convection is a *smooth upwind* flux::

    a     = 0.5 (u_L + u_R)                 face normal velocity
    |a|_e = a tanh(a / eps)                 C-infinity |a|
    F     = 0.5 a (u_L + u_R) - 0.5 |a|_e (u_R - u_L)

i.e. classical first-order upwind where ``|a| >> eps`` but differentiable
everywhere, including ``a = 0`` (walls, symmetry plane).  A piecewise upwind flux
is only ``C^0`` there and the discrete adjoint then disagrees with a central
difference exactly on the symmetry plane.

Body
----
The body is not meshed: a Brinkman volume penalisation ``+ (alpha/eta) u`` is
added to both momentum components, ``alpha`` being the smoothed indicator of
:class:`tensorfvm.adjoint_shape.geometry.FourierBody`.  ``alpha = alpha(x; theta)``
on a *fixed* grid, so the unknown vector has a shape-independent dimension and
``dR/dtheta`` is a closed form -- exactly what a discrete adjoint needs.

Unknown / residual layout
-------------------------
``U = [u (ny nx) | v ((ny-1) nx) | p (ny nx)]``; residual rows use the same
blocking so ``dR/dU`` is square.  :meth:`ChannelSolver.assemble` returns the
residual *and* its analytic Jacobian; nothing is differenced numerically.

Objective
---------
``J = C_D = F_x / (0.5 rho U_m^2 D)``, ``D = 2 r0``, ``U_m`` the bulk velocity
(determined by the flow rate), and ``F_x = sum (alpha/eta) u dx dy`` the
penalisation reaction on the body.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .geometry import N_PARAMS, FourierBody


@dataclass(frozen=True)
class ChannelGrid:
    """Uniform MAC grid of a plane channel, periodic in ``x``."""

    nx: int
    ny: int
    length: float = 6.0
    height: float = 2.0

    @property
    def dx(self) -> float:
        return self.length / self.nx

    @property
    def dy(self) -> float:
        return self.height / self.ny

    @property
    def nu(self) -> int:
        return self.ny * self.nx

    @property
    def nv(self) -> int:
        return (self.ny - 1) * self.nx

    @property
    def nc(self) -> int:
        return self.ny * self.nx

    @property
    def size(self) -> int:
        return self.nu + self.nv + self.nc

    def iu(self, j: int, i: int) -> int:
        return j * self.nx + (i % self.nx)

    def iv(self, j: int, i: int) -> int:
        return self.nu + (j - 1) * self.nx + (i % self.nx)

    def ip(self, j: int, i: int) -> int:
        return self.nu + self.nv + j * self.nx + (i % self.nx)

    def y_u(self, j: int) -> float:
        return (j + 0.5) * self.dy


def poiseuille(y, height: float, force: float, nu: float):
    """Exact plane-Poiseuille profile of the driven channel."""
    y = np.asarray(y, dtype=float)
    return force * y * (height - y) / (2.0 * nu)


class ChannelSolver:
    """Steady incompressible periodic channel flow with a penalised body."""

    def __init__(
        self,
        grid: ChannelGrid,
        body: FourierBody | None,
        nu: float = 0.05,
        force: float = 0.15,
        eta: float = 1.0e-3,
        rho: float = 1.0,
        eps_upwind: float = 0.05,
    ) -> None:
        self.grid = grid
        self.body = body
        self.nu = float(nu)
        self.force = float(force)
        self.eta = float(eta)
        self.rho = float(rho)
        self.eps = float(eps_upwind)
        self._geometry = None

    # -- geometry on the two velocity lattices ----------------------------
    def geometry(self, theta):
        """Smoothed indicator (and its theta-derivative) on ``u`` and ``v`` faces."""
        theta = np.asarray(theta, dtype=float)
        key = theta.tobytes()
        if self._geometry is not None and self._geometry[0] == key:
            return self._geometry[1]
        g = self.grid
        if self.body is None:
            zero_u = np.zeros((g.ny, g.nx))
            zero_v = np.zeros((g.ny - 1, g.nx))
            data = (zero_u, np.zeros((N_PARAMS,) + zero_u.shape), zero_v,
                    np.zeros((N_PARAMS,) + zero_v.shape))
        else:
            x_u, y_u = np.meshgrid(
                np.arange(g.nx, dtype=float) * g.dx,
                (np.arange(g.ny, dtype=float) + 0.5) * g.dy,
            )
            alpha_u, dalpha_u = self.body.indicator(x_u, y_u, theta)
            x_v, y_v = np.meshgrid(
                (np.arange(g.nx, dtype=float) + 0.5) * g.dx,
                np.arange(1, g.ny, dtype=float) * g.dy,
            )
            alpha_v, dalpha_v = self.body.indicator(x_v, y_v, theta)
            data = (alpha_u, dalpha_u, alpha_v, dalpha_v)
        self._geometry = (key, data)
        return data

    # -- smooth upwind flux ----------------------------------------------
    def _flux(self, a: float, ul: float, ur: float):
        """Smooth upwind flux and partials ``(F, dF/duL, dF/duR, dF/da)``."""
        s = np.tanh(a / self.eps)
        absa = a * s
        flux = 0.5 * a * (ul + ur) - 0.5 * absa * (ur - ul)
        d_ul = 0.5 * a + 0.5 * absa
        d_ur = 0.5 * a - 0.5 * absa
        dabs = s + (a / self.eps) * (1.0 - s * s)
        d_a = 0.5 * (ul + ur) - 0.5 * dabs * (ur - ul)
        return flux, d_ul, d_ur, d_a

    # -- residual + Jacobian ----------------------------------------------
    def assemble(self, theta, U, jac: bool = True):
        """Return ``(R, A)`` with ``A = dR/dU`` (``A = None`` when ``jac=False``)."""
        g = self.grid
        nx, ny, dx, dy = g.nx, g.ny, g.dx, g.dy
        nu, eta = self.nu, self.eta
        force = self.force
        alpha_u, _, alpha_v, _ = self.geometry(theta)

        Uu = U[: g.nu].reshape(ny, nx)
        Uv = U[g.nu : g.nu + g.nv].reshape(ny - 1, nx)
        Up = U[g.nu + g.nv :].reshape(ny, nx)

        R = np.zeros(g.size)
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []

        def add(r: int, c: int, v: float) -> None:
            rows.append(r)
            cols.append(c)
            vals.append(v)

        def v_at(j: int, i: int) -> float:
            if j <= 0 or j >= ny:
                return 0.0
            return float(Uv[j - 1, i % nx])

        # ------------------------------------------------ u momentum -----
        for j in range(ny):
            for i in range(nx):
                r = g.iu(j, i)
                # east / west convective faces (periodic wrap)
                a = 0.5 * (Uu[j, i] + Uu[j, (i + 1) % nx])
                f, dul, dur, da = self._flux(a, Uu[j, i], Uu[j, (i + 1) % nx])
                R[r] += f / dx
                if jac:
                    add(r, g.iu(j, i), (dul + 0.5 * da) / dx)
                    add(r, g.iu(j, i + 1), (dur + 0.5 * da) / dx)
                a = 0.5 * (Uu[j, i - 1] + Uu[j, i])
                f, dul, dur, da = self._flux(a, Uu[j, i - 1], Uu[j, i])
                R[r] -= f / dx
                if jac:
                    add(r, g.iu(j, i - 1), -(dul + 0.5 * da) / dx)
                    add(r, g.iu(j, i), -(dur + 0.5 * da) / dx)

                # transverse faces (walls carry a = 0 and contribute nothing)
                if j + 1 <= ny - 1:
                    a = 0.5 * (v_at(j + 1, i - 1) + v_at(j + 1, i))
                    f, dul, dur, da = self._flux(a, Uu[j, i], Uu[j + 1, i])
                    R[r] += f / dy
                    if jac:
                        add(r, g.iu(j, i), dul / dy)
                        add(r, g.iu(j + 1, i), dur / dy)
                        add(r, g.iv(j + 1, i - 1), 0.5 * da / dy)
                        add(r, g.iv(j + 1, i), 0.5 * da / dy)
                if j >= 1:
                    a = 0.5 * (v_at(j, i - 1) + v_at(j, i))
                    f, dul, dur, da = self._flux(a, Uu[j - 1, i], Uu[j, i])
                    R[r] -= f / dy
                    if jac:
                        add(r, g.iu(j - 1, i), -dul / dy)
                        add(r, g.iu(j, i), -dur / dy)
                        add(r, g.iv(j, i - 1), -0.5 * da / dy)
                        add(r, g.iv(j, i), -0.5 * da / dy)

                # diffusion (central, half cell to the walls for u)
                R[r] -= nu * (Uu[j, (i + 1) % nx] - 2.0 * Uu[j, i]
                              + Uu[j, i - 1]) / (dx * dx)
                if jac:
                    add(r, g.iu(j, i + 1), -nu / (dx * dx))
                    add(r, g.iu(j, i - 1), -nu / (dx * dx))
                    add(r, g.iu(j, i), 2.0 * nu / (dx * dx))
                if j < ny - 1:
                    R[r] -= nu * (Uu[j + 1, i] - Uu[j, i]) / (dy * dy)
                    if jac:
                        add(r, g.iu(j + 1, i), -nu / (dy * dy))
                        add(r, g.iu(j, i), nu / (dy * dy))
                else:
                    R[r] += 2.0 * nu * Uu[j, i] / (dy * dy)
                    if jac:
                        add(r, g.iu(j, i), 2.0 * nu / (dy * dy))
                if j > 0:
                    R[r] -= nu * (Uu[j - 1, i] - Uu[j, i]) / (dy * dy)
                    if jac:
                        add(r, g.iu(j - 1, i), -nu / (dy * dy))
                        add(r, g.iu(j, i), nu / (dy * dy))
                else:
                    R[r] += 2.0 * nu * Uu[j, i] / (dy * dy)
                    if jac:
                        add(r, g.iu(j, i), 2.0 * nu / (dy * dy))

                # pressure gradient: face i separates cells i - 1 and i
                R[r] += (Up[j, i] - Up[j, i - 1]) / dx
                if jac:
                    add(r, g.ip(j, i), 1.0 / dx)
                    add(r, g.ip(j, i - 1), -1.0 / dx)

                # Brinkman penalisation and driving force
                w = alpha_u[j, i] / eta
                R[r] += w * Uu[j, i] - force
                if jac:
                    add(r, g.iu(j, i), w)

        # ------------------------------------------------ v momentum -----
        for j in range(1, ny):
            for i in range(nx):
                r = g.iv(j, i)
                # north face (j + 1 == ny is the top wall: uR fixed at 0)
                if j + 1 <= ny - 1:
                    a = 0.5 * (v_at(j, i) + v_at(j + 1, i))
                    f, dul, dur, da = self._flux(a, v_at(j, i), v_at(j + 1, i))
                    R[r] += f / dy
                    if jac:
                        add(r, g.iv(j, i), (dul + 0.5 * da) / dy)
                        add(r, g.iv(j + 1, i), (dur + 0.5 * da) / dy)
                else:
                    a = 0.5 * v_at(j, i)
                    f, dul, dur, da = self._flux(a, v_at(j, i), 0.0)
                    R[r] += f / dy
                    if jac:
                        add(r, g.iv(j, i), (dul + 0.5 * da) / dy)
                # south face (j - 1 == 0 is the bottom wall: uL fixed at 0)
                if j - 1 >= 1:
                    a = 0.5 * (v_at(j - 1, i) + v_at(j, i))
                    f, dul, dur, da = self._flux(a, v_at(j - 1, i), v_at(j, i))
                    R[r] -= f / dy
                    if jac:
                        add(r, g.iv(j - 1, i), -(dul + 0.5 * da) / dy)
                        add(r, g.iv(j, i), -(dur + 0.5 * da) / dy)
                else:
                    a = 0.5 * v_at(j, i)
                    f, dul, dur, da = self._flux(a, 0.0, v_at(j, i))
                    R[r] -= f / dy
                    if jac:
                        add(r, g.iv(j, i), -(dur + 0.5 * da) / dy)

                # east / west convective faces (periodic wrap)
                a = 0.5 * (Uu[j - 1, (i + 1) % nx] + Uu[j, (i + 1) % nx])
                f, dul, dur, da = self._flux(a, v_at(j, i), v_at(j, i + 1))
                R[r] += f / dx
                if jac:
                    add(r, g.iv(j, i), dul / dx)
                    add(r, g.iv(j, i + 1), dur / dx)
                    add(r, g.iu(j - 1, i + 1), 0.5 * da / dx)
                    add(r, g.iu(j, i + 1), 0.5 * da / dx)
                a = 0.5 * (Uu[j - 1, i] + Uu[j, i])
                f, dul, dur, da = self._flux(a, v_at(j, i - 1), v_at(j, i))
                R[r] -= f / dx
                if jac:
                    add(r, g.iv(j, i - 1), -dul / dx)
                    add(r, g.iv(j, i), -dur / dx)
                    add(r, g.iu(j - 1, i), -0.5 * da / dx)
                    add(r, g.iu(j, i), -0.5 * da / dx)

                # diffusion
                if j + 1 <= ny - 1:
                    R[r] -= nu * (Uv[j, i] - Uv[j - 1, i]) / (dy * dy)
                    if jac:
                        add(r, g.iv(j + 1, i), -nu / (dy * dy))
                        add(r, g.iv(j, i), nu / (dy * dy))
                else:
                    R[r] += nu * Uv[j - 1, i] / (dy * dy)
                    if jac:
                        add(r, g.iv(j, i), nu / (dy * dy))
                if j - 1 >= 1:
                    R[r] -= nu * (Uv[j - 2, i] - Uv[j - 1, i]) / (dy * dy)
                    if jac:
                        add(r, g.iv(j - 1, i), -nu / (dy * dy))
                        add(r, g.iv(j, i), nu / (dy * dy))
                else:
                    R[r] += nu * Uv[j - 1, i] / (dy * dy)
                    if jac:
                        add(r, g.iv(j, i), nu / (dy * dy))
                R[r] -= nu * (Uv[j - 1, (i + 1) % nx] - 2.0 * Uv[j - 1, i]
                              + Uv[j - 1, i - 1]) / (dx * dx)
                if jac:
                    add(r, g.iv(j, i + 1), -nu / (dx * dx))
                    add(r, g.iv(j, i - 1), -nu / (dx * dx))
                    add(r, g.iv(j, i), 2.0 * nu / (dx * dx))

                # pressure
                R[r] += (Up[j, i] - Up[j - 1, i]) / dy
                if jac:
                    add(r, g.ip(j, i), 1.0 / dy)
                    add(r, g.ip(j - 1, i), -1.0 / dy)

                # Brinkman penalisation
                w = alpha_v[j - 1, i] / eta
                R[r] += w * Uv[j - 1, i]
                if jac:
                    add(r, g.iv(j, i), w)

        # ------------------------------- continuity / pressure level -----
        for j in range(ny):
            for i in range(nx):
                r = g.ip(j, i)
                if j == 0 and i == 0:
                    R[r] += Up[j, i]  # pin the constant-pressure null mode
                    if jac:
                        add(r, r, 1.0)
                    continue
                R[r] += (Uu[j, (i + 1) % nx] - Uu[j, i]) / dx
                R[r] += (v_at(j + 1, i) - v_at(j, i)) / dy
                if jac:
                    add(r, g.iu(j, i + 1), 1.0 / dx)
                    add(r, g.iu(j, i), -1.0 / dx)
                    if j + 1 <= ny - 1:
                        add(r, g.iv(j + 1, i), 1.0 / dy)
                    if j >= 1:
                        add(r, g.iv(j, i), -1.0 / dy)

        if not jac:
            return R, None
        A = sp.coo_matrix(
            (np.asarray(vals), (np.asarray(rows), np.asarray(cols))),
            shape=(g.size, g.size),
        ).tocsr()
        return R, A

    # -- forward solve ----------------------------------------------------
    def initial_guess(self) -> np.ndarray:
        """Poiseuille plug: exact solution of the body-free problem."""
        g = self.grid
        U = np.zeros(g.size)
        u = U[: g.nu].reshape(g.ny, g.nx)
        for j in range(g.ny):
            u[j, :] = poiseuille(g.y_u(j), g.height, self.force, self.nu)
        return U

    def solve(self, theta, U0=None, tol: float = 1e-11, maxit: int = 40):
        """Newton solve of ``R(U, theta) = 0``; returns ``(U, info)``."""
        U = self.initial_guess() if U0 is None else np.array(U0, dtype=float)
        R, A = self.assemble(theta, U)
        norm0 = max(float(np.linalg.norm(R)), 1e-300)
        history = [norm0]
        converged = False
        it = 0
        for it in range(1, maxit + 1):
            norm = float(np.linalg.norm(R))
            if norm <= tol * norm0:
                converged = True
                break
            du = spla.spsolve(A.tocsc(), -R)
            lam = 1.0
            for _ in range(12):
                trial = U + lam * du
                nt = float(np.linalg.norm(self.assemble(theta, trial, jac=False)[0]))
                if nt < (1.0 - 1e-4 * lam) * norm:
                    break
                lam *= 0.5
            U = U + lam * du
            R, A = self.assemble(theta, U)
            history.append(float(np.linalg.norm(R)))
        info = {
            "iterations": it,
            "residual_norm": float(np.linalg.norm(R)),
            "residual_norm_initial": norm0,
            "relative_residual": float(np.linalg.norm(R)) / norm0,
            "history": history,
            "converged": bool(converged),
            "jacobian": A,
        }
        return U, info

    def solve_reference(self, theta, U0=None, atol: float = 1e-14, maxit: int = 500):
        """Independent linear-algebra route to the *same* discrete solution.

        Same residual / Jacobian assembly, but the final Newton correction comes
        from a matrix-free GMRES with an ILU preconditioner instead of a sparse
        direct factorisation.  Used for the path-equivalence check on ``C_D``.
        """
        U, info = self.solve(theta, U0=U0, tol=1e-11)
        R, A = self.assemble(theta, U)
        ilu = spla.spilu(A.tocsc(), drop_tol=1e-9, fill_factor=60)
        M = spla.LinearOperator(A.shape, matvec=ilu.solve)
        du, flag = spla.gmres(A, -R, rtol=0.0, atol=atol, maxiter=maxit, M=M, restart=120)
        U = U + du
        R2, _ = self.assemble(theta, U, jac=False)
        return U, {
            "gmres_flag": int(flag),
            "residual_norm": float(np.linalg.norm(R2)),
            "newton": info,
        }

    # -- objective --------------------------------------------------------
    def bulk_velocity(self, U) -> float:
        """Bulk velocity ``U_m = Q / H`` from the flux through a periodic plane."""
        g = self.grid
        Uu = U[: g.nu].reshape(g.ny, g.nx)
        return float(np.sum(Uu[:, 0]) * g.dy / g.height)

    def bulk_velocity_gradient(self) -> np.ndarray:
        g = self.grid
        d = np.zeros(g.size)
        d[: g.nu] = (g.dy / g.height) * np.tile(
            (np.arange(g.nx) == 0).astype(float), g.ny
        )
        return d

    def drag(self, theta, U):
        """Return ``(F_x, dF/dU, dF/dtheta)`` for the penalisation reaction."""
        g = self.grid
        alpha_u, dalpha_u, _, _ = self.geometry(theta)
        Uu = U[: g.nu].reshape(g.ny, g.nx)
        cell = g.dx * g.dy
        w = alpha_u / self.eta
        F = cell * float(np.sum(w * Uu))
        dF_dU = np.zeros(g.size)
        dF_dU[: g.nu] = (cell * w).ravel()
        dF_dtheta = (cell / self.eta) * np.einsum("qji,ji->q", dalpha_u, Uu)
        return F, dF_dU, dF_dtheta

    def objective(self, theta, U):
        """``J = C_D`` and its partials ``(J, dJ/dU, dJ/dtheta)``."""
        u_m = self.bulk_velocity(U)
        d_drag = 2.0 * self.body.radius0 if self.body is not None else 1.0
        a = 0.5 * self.rho * u_m * u_m * d_drag
        F, dF_dU, dF_dtheta = self.drag(theta, U)
        da_dU = self.rho * u_m * d_drag * self.bulk_velocity_gradient()
        j = F / a
        dJ_dU = dF_dU / a - F / (a * a) * da_dU
        dJ_dtheta = dF_dtheta / a
        return j, dJ_dU, dJ_dtheta

    def drag_coefficient(self, theta, U) -> float:
        return float(self.objective(theta, U)[0])

    # -- physics cross-check ---------------------------------------------
    def momentum_balance(self, theta, U) -> dict:
        """Global x-momentum budget: driving force vs drag plus wall friction."""
        g = self.grid
        Uu = U[: g.nu].reshape(g.ny, g.nx)
        f_drive = self.force * g.length * g.height
        tau = self.rho * self.nu * (
            float(np.sum(Uu[0, :])) + float(np.sum(Uu[g.ny - 1, :]))
        ) / (0.5 * g.dy)
        f_wall_drag = tau * g.dx
        f_body = self.drag(theta, U)[0]
        return {
            "F_drive": f_drive,
            "F_body_penalisation": f_body,
            "F_wall_friction_drag": f_wall_drag,
            "residual": f_drive - f_body - f_wall_drag,
            "ratio": (f_body + f_wall_drag) / f_drive,
        }


def default_solver(
    nx: int = 64,
    ny: int = 32,
    re: float = 20.0,
    eta: float = 1.0e-3,
    delta_cells: float = 1.0,
    u_mean: float = 1.0,
    with_body: bool = True,
    eps_upwind: float = 0.05,
):
    """Convenience factory: centred body, ``nu = u_mean D / Re``, ``delta ~ h``."""
    grid = ChannelGrid(nx=nx, ny=ny)
    nu = u_mean * 1.0 / re
    force = 12.0 * nu * u_mean / (grid.height**2)
    body = None
    if with_body:
        body = FourierBody(
            center=(2.0, 1.0), radius0=0.5, delta=delta_cells * min(grid.dx, grid.dy)
        )
    return ChannelSolver(grid, body, nu=nu, force=force, eta=eta, eps_upwind=eps_upwind)


__all__ = [
    "ChannelGrid",
    "ChannelSolver",
    "N_PARAMS",
    "default_solver",
    "poiseuille",
]