"""Cross-flow cylinder: spectral azimuthal derivatives + sparse radial solves.

Why a second cylinder solver
----------------------------
:class:`tensorfvm.cht.polar.PolarVorticitySolver` discretises *everything* with
finite differences.  At ``Re = 40`` the azimuthal cell Peclet number on a
72-point ring is ``u_theta r dtheta/nu ~ 1.7``, so first-order upwind adds a
numerical viscosity of the same order as the molecular one.  Measured effect:
``Cd = 1.93`` against the benchmark ``1.57`` (+23 %) and ``Nu = 3.91`` against
Churchill-Bernstein ``3.36`` (+16 %); switching the azimuthal transport to a
limited second-order reconstruction over-corrects (``Cd = 0.69``) and the
explicit scheme loses stability as soon as the ring is refined.  The legacy
solver therefore cannot support a 5 % heat-transfer claim, which is what this
module fixes:

* **azimuthal derivatives** are spectral (FFT): no numerical diffusion at all;
* **linear solves** (Poisson for psi, implicit diffusion of omega and T) use one
  precomputed sparse LU each, so the cost per time step is small;
* **radial advection** is upwind (first order) but the wall-clustered spacing
  keeps its numerical viscosity far below ``nu``;
* the no-slip wall enters as ``omega_w = -2a`` from ``psi = a s^2 + b s^3``
  (second order) and the wall heat flux from a quadratic one-sided fit.

Formulation (unit diameter, unit free stream, ``Re = U D / nu``)::

    d omega/dt + u_r d omega/d r + (u_theta/r) d omega/d theta = nu lap(omega)
    lap(psi) = -omega,   psi = 0 at r = 1/2,   psi = U r sin(theta) at r = R
    dT/dt + u.grad T = alpha lap(T),  T = T_wall(theta) at r = 1/2, T = 0 at r = R
"""

from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

__all__ = ["FourierCylinderFlow", "FourierCylinderEnergy"]


def _geometric_nodes(r_wall: float, r_out: float, n: int, first: float) -> np.ndarray:
    """``n+1`` nodes from ``r_wall`` to ``r_out`` with first spacing ``first``."""
    length = r_out - r_wall
    if first * n >= length:
        return np.linspace(r_wall, r_out, n + 1)
    lo, hi = 1.0 + 1e-12, 2.0
    for _ in range(300):
        q = 0.5 * (lo + hi)
        total = first * (q ** n - 1.0) / (q - 1.0) if abs(q - 1.0) > 1e-12 else first * n
        if total < length:
            lo = q
        else:
            hi = q
    q = 0.5 * (lo + hi)
    steps = first * q ** np.arange(n)
    nodes = np.concatenate([[r_wall], r_wall + np.cumsum(steps)])
    nodes[-1] = r_out
    return nodes


class _Annulus:
    """Radial grid, metrics and the sparse laplacian on the interior nodes."""

    def __init__(self, r_inner: float, r_out: float, n_cells: int, first: float,
                 n_theta: int):
        self.r = _geometric_nodes(r_inner, r_out, n_cells, first)
        self.n_node = self.r.size
        self.n_theta = n_theta
        self.dtheta = 2.0 * math.pi / n_theta
        self.theta = np.arange(n_theta) * self.dtheta
        self.r_face = np.concatenate([[self.r[0]], 0.5 * (self.r[:-1] + self.r[1:]),
                                      [self.r[-1]]])
        self.dr = np.diff(self.r)
        self.drcv = np.diff(self.r_face)
        self.ni = self.n_node - 2
        i = np.arange(1, self.n_node - 1)
        d_west = self.dr[i - 1]
        d_east = self.dr[i]
        denom = self.r[i] * self.drcv[i]
        self.coef_w = self.r_face[i] / d_west / denom
        self.coef_e = self.r_face[i + 1] / d_east / denom
        self.diag_r = -(self.coef_e + self.coef_w)
        self.coef_t = 1.0 / (self.r[i] ** 2 * self.dtheta ** 2)
        self._build_laplacian()

    def _build_laplacian(self):
        ni, nt = self.ni, self.n_theta
        idx = np.arange(ni * nt).reshape(ni, nt)
        rows, cols, vals = [], [], []
        diag = self.diag_r - 2.0 * self.coef_t
        for k in range(ni - 1):
            a, b = idx[k, :], idx[k + 1, :]
            rows += [a, b]
            cols += [b, a]
            vals += [np.full(nt, self.coef_e[k]), np.full(nt, self.coef_w[k + 1])]
        for j in range(nt):
            jp, jm = (j + 1) % nt, (j - 1) % nt
            a = idx[:, j]
            rows += [a, idx[:, jp], idx[:, jm]]
            cols += [a, a, a]
            vals += [np.zeros(ni), self.coef_t, self.coef_t]
        rows.append(np.arange(ni * nt))
        cols.append(np.arange(ni * nt))
        vals.append(np.repeat(diag, nt))
        self.L = sp.coo_matrix((np.concatenate(vals),
                                (np.concatenate(rows), np.concatenate(cols))),
                               shape=(ni * nt, ni * nt)).tocsc()
        self.idx = idx
        self._poisson_lu = spla.splu(self.L)
        self._diff_lu = {}

    def implicit_lu(self, scale: float):
        """LU of ``I - scale * L`` (cached per ``scale``)."""
        key = round(float(scale), 14)
        if key not in self._diff_lu:
            self._diff_lu[key] = spla.splu((sp.identity(self.L.shape[0], format="csc")
                                           - key * self.L).tocsc())
        return self._diff_lu[key]

    # ------------------------------------------------------------- operators
    def rfft(self, field):
        return np.fft.rfft(field, axis=1)

    def irfft(self, modes):
        return np.fft.irfft(modes, n=self.n_theta, axis=1)

    def theta_modes(self):
        return np.arange(self.n_theta // 2 + 1)

    def dealias(self, field):
        """Two-thirds rule: drop the top third of the azimuthal spectrum.

        The azimuthal derivative is spectral, so a product of two fields puts
        energy into modes the grid cannot represent; without de-aliasing that
        energy folds back into low modes and the march diverges.
        """
        modes = self.rfft(field)
        kmax = max(2, self.n_theta // 3)
        modes[:, kmax:] = 0.0
        return self.irfft(modes)

    def spectral_d_theta(self, field):
        modes = self.rfft(field) * (1j * self.theta_modes())[None, :]
        return self.irfft(modes)

    def radial_derivative_upwind(self, field, ur):
        out = np.zeros_like(field)
        d_west = (field[1:-1, :] - field[:-2, :]) / self.dr[:-1, None]
        d_east = (field[2:, :] - field[1:-1, :]) / self.dr[1:, None]
        out[1:-1, :] = np.where(ur[1:-1, :] > 0.0, d_west, d_east)
        return out

    def radial_derivative(self, field):
        out = np.zeros_like(field)
        out[1:-1, :] = (field[2:, :] - field[:-2, :]) / (self.r[2:] - self.r[:-2])[:, None]
        return out


class FourierCylinderFlow:
    """Steady cross-flow past a circular cylinder (spectral theta, FV r)."""

    def __init__(self, *, re: float, n_r: int = 120, n_theta: int = 96,
                 r_out: float = 25.0, wall_cell: float = 4.0e-3, u_inf: float = 1.0):
        if re <= 0:
            raise ValueError("Reynolds number must be positive")
        self.re = float(re)
        self.nu = 1.0 / self.re
        self.u_inf = float(u_inf)
        self.grid = _Annulus(0.5, r_out, n_r, wall_cell, n_theta)
        self.psi = np.zeros((self.grid.n_node, n_theta))
        self.omega = np.zeros((self.grid.n_node, n_theta))
        self.history: list[float] = []

    # -------------------------------------------------------------- operators
    def solve_poisson(self, omega_full):
        """``lap(psi) = -omega`` (wall ``psi = 0``, far field ``U r sin theta``)."""
        g = self.grid
        outer = self.u_inf * g.r[-1] * np.sin(g.theta)
        # ``L`` is the +laplacian, so the right-hand side is -omega; the known
        # outer Dirichlet value leaves the interior system through coef_e[-1].
        rhs = -omega_full[1:-1, :]
        rhs[-1, :] -= g.coef_e[-1] * outer
        psi = np.zeros((g.n_node, g.n_theta))
        psi[1:-1, :] = g._poisson_lu.solve(rhs.ravel()).reshape(g.ni, g.n_theta)
        psi[-1, :] = outer
        return psi

    def wall_vorticity(self, psi):
        """``omega_w = -2a`` from ``psi = a s^2 + b s^3`` (second order)."""
        r = self.grid.r
        s1 = r[1] - r[0]
        s2 = r[2] - r[0]
        a1 = psi[1, :] / (s1 * s1)
        a2 = psi[2, :] / (s2 * s2)
        b = (a2 - a1) / (s2 - s1)
        return -2.0 * (a1 - b * s1)

    def velocities(self, psi):
        g = self.grid
        dpsi_theta = g.spectral_d_theta(psi)
        psi_r = g.radial_derivative(psi)
        ur = np.zeros_like(psi)
        ur[1:-1, :] = dpsi_theta[1:-1, :] / g.r[1:-1, None]
        ut = -psi_r
        return ur, ut

    # ----------------------------------------------------------------- march
    def march(self, *, t_end: float, dt: float, tol_steady: float = 1e-7,
              max_steps: int = 400000, check_from: int = 200, t_ramp: float = 10.0,
              callback=None):
        g = self.grid
        lu = g.implicit_lu(dt * self.nu)
        omega = self.omega.copy()
        psi = self.solve_poisson(omega)
        omega[0, :] = g.dealias(np.atleast_2d(self.wall_vorticity(psi)))[0]
        history: list[float] = []
        drag_history = []
        steady = False
        step = 0
        n_steps = min(int(round(t_end / dt)), max_steps)
        prev = omega[0, :].copy()
        for step in range(n_steps):
            ur, ut = self.velocities(psi)
            omega_r = g.radial_derivative_upwind(omega, ur)
            omega_theta = g.spectral_d_theta(omega)
            adv = np.zeros_like(omega)
            adv[1:-1, :] = -(ur[1:-1, :] * omega_r[1:-1, :]
                             + ut[1:-1, :] / g.r[1:-1, None] * omega_theta[1:-1, :])
            adv = g.dealias(adv)
            rhs = omega[1:-1, :] + dt * adv[1:-1, :] + dt * self.nu * np.broadcast_to(
                (g.coef_w[0] * omega[0, :])[None, :], (g.ni, g.n_theta)) * \
                np.eye(g.ni)[:, 0:1]
            omega_new = np.zeros_like(omega)
            omega_new[0, :] = omega[0, :]
            omega_new[1:-1, :] = lu.solve(rhs.ravel()).reshape(g.ni, g.n_theta)
            omega = omega_new
            psi = self.solve_poisson(omega)
            # The wall vorticity is a second radial derivative of psi divided by
            # the first-cell spacing squared, so it amplifies the high azimuthal
            # modes of any solver noise; the same two-thirds rule as the advection
            # keeps that loop from diverging.
            omega[0, :] = g.dealias(np.atleast_2d(self.wall_vorticity(psi)))[0]
            change = float(np.max(np.abs(omega[0, :] - prev)))
            history.append(change)
            prev = omega[0, :].copy()
            if (step % 200) == 0:
                drag_history.append((float(step * dt),) + self.drag_lift(omega[0, :]))
                if callback is not None:
                    callback(step, omega, psi)
            if step * dt >= t_ramp and step > check_from and len(history) > 200:
                if max(history[-150:]) < tol_steady:
                    steady = True
                    break
        self.omega = omega
        self.psi = psi
        self.history = history
        return {"omega": omega, "psi": psi, "steps": step + 1, "dt": dt,
                "time": (step + 1) * dt, "steady": steady,
                "wall_vorticity": omega[0, :].copy(),
                "wall_vorticity_history": history, "drag_history": drag_history}

    # ------------------------------------------------------------ diagnostics
    def drag_lift(self, wall_vorticity=None):
        w = self.omega[0, :] if wall_vorticity is None else wall_vorticity
        dth = self.grid.dtheta
        cd = -(2.0 / self.re) * float(np.sum(w * np.sin(self.grid.theta))) * dth
        cl = -(2.0 / self.re) * float(np.sum(w * np.cos(self.grid.theta))) * dth
        return cd, cl

    def wall_shear_max(self, wall_vorticity=None):
        w = self.omega[0, :] if wall_vorticity is None else wall_vorticity
        return float(np.max(np.abs(w)))


class FourierCylinderEnergy:
    """Steady thermal field for a prescribed wall temperature distribution."""

    def __init__(self, flow: FourierCylinderFlow, *, pr: float = 0.7):
        self.flow = flow
        self.pr = float(pr)
        self.alpha = flow.nu / self.pr
        self.T = np.zeros((flow.grid.n_node, flow.grid.n_theta))

    def solve(self, t_wall, *, t_outer: float = 0.0, t_end: float = 60.0,
              dt: float | None = None, tol: float = 1e-10, max_steps: int = 400000,
              check_from: int = 100):
        g = self.flow.grid
        ur, ut = self.flow.velocities(self.flow.psi)
        if dt is None:
            dt = min(2.0e-3, 0.4 * float(np.min(g.dr))
                     / max(float(np.max(np.abs(ur))), float(np.max(np.abs(ut))), 1e-30))
        t_wall = np.asarray(t_wall, dtype=float)
        if t_wall.size == 1:
            t_wall = np.full(g.n_theta, float(t_wall))
        lu = g.implicit_lu(dt * self.alpha)
        wall_pad = np.broadcast_to((g.coef_w[0] * t_wall)[None, :], (g.ni, g.n_theta))
        eye0 = np.eye(g.ni)[:, 0:1]
        T = np.full((g.n_node, g.n_theta), float(t_outer))
        T[0, :] = t_wall
        history = []
        step = 0
        n_steps = min(int(round(t_end / dt)), max_steps)
        for step in range(n_steps):
            T_r = g.radial_derivative_upwind(T, ur)
            T_theta = g.spectral_d_theta(T)
            adv = np.zeros_like(T)
            adv[1:-1, :] = -(ur[1:-1, :] * T_r[1:-1, :]
                             + ut[1:-1, :] / g.r[1:-1, None] * T_theta[1:-1, :])
            adv = g.dealias(adv)
            rhs = (T[1:-1, :] + dt * adv[1:-1, :]
                   + dt * self.alpha * wall_pad * eye0
                   + dt * self.alpha * g.coef_e[-1] * float(t_outer)
                   * np.eye(g.ni)[:, -1:])
            T_new = np.full_like(T, float(t_outer))
            T_new[0, :] = t_wall
            T_new[1:-1, :] = lu.solve(rhs.ravel()).reshape(g.ni, g.n_theta)
            delta = float(np.max(np.abs(T_new - T)))
            T = T_new
            history.append(delta)
            if step > check_from and delta < tol * max(1.0, float(np.max(np.abs(T)))):
                break
        self.T = T
        self.history = history
        self.steps = step + 1
        self.dt = dt
        q_wall, nu_local = self.wall_flux(t_wall, t_outer)
        return {"T": T, "q_wall": q_wall, "nu_local": nu_local,
                "nusselt_mean": float(np.mean(nu_local)), "steps": self.steps,
                "dt": dt, "history_tail": history[-5:],
                "converged_delta": history[-1] if history else float("nan")}

    def wall_flux(self, t_wall, t_outer: float = 0.0):
        g = self.flow.grid
        T = self.T
        s1 = g.r[1] - g.r[0]
        s2 = g.r[2] - g.r[0]
        c0 = -(s1 + s2) / (s1 * s2)
        c1 = s2 / (s1 * (s2 - s1))
        c2 = -s1 / (s2 * (s2 - s1))
        slope = c0 * T[0, :] + c1 * T[1, :] + c2 * T[2, :]
        q_wall = -self.alpha * slope                      # positive into the fluid
        t_wall = np.asarray(t_wall, dtype=float)
        if t_wall.size == 1:
            t_wall = np.full(g.n_theta, float(t_wall))
        ref = float(np.mean(t_wall)) - float(t_outer)
        nu_local = q_wall / (self.alpha * max(ref, 1e-30))
        return q_wall, nu_local