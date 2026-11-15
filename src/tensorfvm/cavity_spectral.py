"""Chebyshev-collocation reference solution of the lid-driven cavity (steady).

Why this module exists
----------------------
``docs/CASE-FVM-CAVITY-RE1000/XCHECK.md`` §7 records the standing limitation of
the cavity verification: the production solver (2nd-order finite volume,
pseudo-time marching) and the independent check code used there (2nd-order
explicit Euler + QUICK) share the *same truncation structure*, so their
agreement cannot separate

* "the extrapolated FVM limit is the PDE solution and the residual ~1.8 % offset
  against Ghia et al. (1982) comes from the reference's own 129x129
  discretisation error", from
* "both 2nd-order codes carry a common family bias".

Deciding between the two needs a *higher-order* reference.  This module provides
one, built independently of :mod:`tensorfvm.lid_cavity`:

======================  ==========================================  ==============================
aspect                  production path (``lid_cavity``)           this module
======================  ==========================================  ==============================
formulation             primitive (u, v, p), projection             streamfunction-vorticity
discretisation          2nd-order cell-centred finite volume        Chebyshev collocation
time treatment          pseudo-time march to steady state           steady Newton iteration
vorticity at the wall   Thom (2nd order) on the FV stencil          spectral ghost-slope closure
gradients               face-based Green-Gauss                      global spectral matrices
======================  ==========================================  ==============================

Nothing is imported from the production solver, so the two codes share only the
governing equations.

Discretisation
--------------
Chebyshev-Gauss-Lobatto nodes in both directions on the unit square, moving lid
(``u = U_lid``) at ``y = 1``.  Steady flow in ``psi`` (streamfunction) and
``omega`` (vorticity), with ``u = psi_y`` and ``v = -psi_x``::

    psi_xx + psi_yy + omega            = 0                (interior)
    psi_y omega_x - psi_x omega_y      = (1/Re) L(omega)  (interior)
    psi                                = 0                (whole boundary)
    omega                              = -psi_yy |wall    (y = 0, y = 1)
    omega                              = -psi_xx |wall    (x = 0, x = 1)

Wall closure.  Along a wall ``psi`` is constant, so the tangential curvature term
of ``omega = -(psi_xx + psi_yy)`` vanishes and only ``psi_nn`` is needed.  That
second derivative is *not* taken from a plain interpolant of the nodal ``psi``:
on the moving wall the interpolant must also honour the no-slip slope
``psi_y = U_lid``.  The functional used here is the unique degree-``N``
polynomial satisfying the ``N-1`` interior node values plus the two wall
conditions ``psi(wall) = 0`` and ``psi_n(wall) = U_t``; the resulting weight
vector on the data ``[psi(interior), psi(wall), psi_n(wall)]`` is what enters the
residual.  This is the spectral analogue of the classical ghost-node elimination
``omega_0 = -(2 psi_1 - 2 h U) / h**2`` -- omitting the slope term makes the
system admit the trivial solution ``psi = omega = 0`` for any lid speed (the
defect this module was debugged against; see
``tests/test_cavity_spectral_ref.py::test_lid_drives_the_flow``).

Unknown vector, ``2*N**2 + 2`` entries::

    [ omega(all (N+1)^2 nodes) | psi(interior (N-1)^2 nodes) ]

with the same number of equations ((N-1)^2 Poisson + (N-1)^2 transport rows and
4*N wall rows).  The Newton Jacobian is dense (spectral matrices are dense), so
it is LU-factorised directly; the cost therefore grows like ``N**3`` and the
module stays at moderate ``N`` (<= 160).

Usage
-----
>>> s = SpectralCavity(24)                       # 25 x 25 nodes
>>> sol = s.solve(reynolds=100.0, continuation=(10.0, 50.0), verbose=False)
>>> round(sol.metrics["u_min"], 5)               # Ghia (1982) table I: -0.21090
-0.2109...

All engineering decisions (order, tolerances, continuation path) are keyword
arguments; nothing about the case is hard-coded beyond the benchmark geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import scipy.sparse as sp

__all__ = [
    "chebyshev_nodes",
    "differentiation_matrices",
    "barycentric_matrix",
    "clenshaw_curtis_weights",
    "wall_second_derivative_functional",
    "SpectralCavity",
    "SpectralSolution",
]


# ---------------------------------------------------------------------------
# 1-D Chebyshev toolbox
# ---------------------------------------------------------------------------
def chebyshev_nodes(n: int) -> np.ndarray:
    """Chebyshev-Gauss-Lobatto nodes ``xi_j = cos(pi j / n)`` on ``[-1, 1]``."""
    if n < 1:
        raise ValueError("n must be >= 1")
    return np.cos(np.pi * np.arange(n + 1) / n)


def differentiation_matrices(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chebyshev collocation derivative matrices ``(xi, D1, D2)`` on ``[-1, 1]``.

    ``D1`` follows Trefethen (2000, *Spectral Methods in MATLAB*, ch. 6); ``D2``
    is formed as ``D1 @ D1``, which is the exact collocation second-derivative
    operator and is accurate enough for ``n <= 160`` in double precision.
    """
    xi = chebyshev_nodes(n)
    c = np.hstack([2.0, np.ones(n - 1), 2.0]) * (-1.0) ** np.arange(n + 1)
    xmat = np.tile(xi, (n + 1, 1)).T
    dx = xmat - xmat.T
    D1 = np.outer(c, 1.0 / c) / (dx + np.eye(n + 1))
    D1 = D1 - np.diag(D1.sum(axis=1))
    return xi, D1, D1 @ D1


def _barycentric_weights(xi: np.ndarray) -> np.ndarray:
    n = xi.size - 1
    c = np.hstack([0.5, np.ones(n - 1), 0.5])
    return (-1.0) ** np.arange(n + 1) * c


def barycentric_matrix(targets: Iterable[float], xi: np.ndarray) -> np.ndarray:
    """Dense Lagrange-basis matrix ``L`` with ``f(t_k) ~= L[k] @ f(xi)``.

    ``targets`` are given in the reference coordinate ``xi in [-1, 1]``.  Exact
    node hits are handled by row replacement (no 0/0).
    """
    xi = np.asarray(xi, dtype=float)
    w = _barycentric_weights(xi)
    t = np.atleast_1d(np.asarray(targets, dtype=float))
    diff = t[:, None] - xi[None, :]
    L = np.zeros((t.size, xi.size))
    with np.errstate(divide="ignore", invalid="ignore"):
        wd = w[None, :] / diff
    hit = np.abs(diff) < 1e-14 * max(1.0, float(np.max(np.abs(xi))))
    rows_hit = hit.any(axis=1)
    if rows_hit.any():
        idx = np.where(rows_hit)[0]
        cols = hit[idx].argmax(axis=1)
        L[idx, cols] = 1.0
    free = ~rows_hit
    if free.any():
        L[free] = wd[free] / wd[free].sum(axis=1, keepdims=True)
    return L


def clenshaw_curtis_weights(xi: np.ndarray) -> np.ndarray:
    """Quadrature weights for CGL nodes on ``[-1, 1]`` (exact to degree ``2n``).

    Obtained by integrating the Lagrange basis with Gauss-Legendre quadrature,
    which avoids transcribing the classical closed-form formula.
    """
    n = xi.size - 1
    gx, gw = np.polynomial.legendre.leggauss(n + 2)
    L = barycentric_matrix(gx, xi)
    return gw @ L


def wall_second_derivative_functional(
    n: int, interior_X: np.ndarray, wall_sign: int
) -> tuple[np.ndarray, float, float]:
    """Weight vector for ``psi_nn`` on a wall, honouring the no-slip slope.

    The interpolant is the unique polynomial of degree ``<= n`` in ``X in [0, 1]``
    that takes the ``n-1`` interior node values and satisfies ``psi(X_wall) = 0``
    and ``dpsi/dX(X_wall) = U_t``.  Then ``psi_nn(X_wall)`` is a fixed linear
    functional of the data vector ``[psi(interior nodes, ascending X),
    psi(X_wall), dpsi/dX(X_wall)]``.

    Parameters
    ----------
    n :
        Polynomial order of the collocation.
    interior_X :
        The ``n-1`` interior node coordinates, ascending.
    wall_sign :
        ``+1`` for the wall at ``X = 1`` (top / right), ``-1`` for ``X = 0``.

    Returns
    -------
    ``(w_interior, w_wall, w_slope)`` so that

    ``psi_nn(wall) = w_interior @ psi(interior) + w_wall * psi(wall)
                     + w_slope * dpsi_n(wall)``.
    """
    interior_X = np.asarray(interior_X, dtype=float)
    if interior_X.size != n - 1:
        raise ValueError("expected n-1 interior nodes")
    k = np.arange(n + 1, dtype=float)
    s = float(wall_sign)
    Y = 2.0 * interior_X - 1.0                       # ascending, inside (-1, 1)
    # Chebyshev basis T_k(Y) sampled at the interior nodes
    B = np.cos(np.outer(np.arccos(np.clip(Y, -1.0, 1.0)), k))
    # rows for the two wall conditions (data order: interior ..., wall, slope)
    B = np.vstack([
        B,
        s ** k,                                       # T_k(+-1)
        2.0 * s ** (k + 1) * k ** 2,                  # d/dX T_k at the wall
    ])
    # d2/dX2 T_k at the wall:  d/dX = 2 d/dY,  T_k''(+-1) = (+-1)^k k^2 (k^2-1)/3
    q = 4.0 * s ** k * k ** 2 * (k ** 2 - 1.0) / 3.0
    w = np.linalg.solve(B.T, q)                       # w @ data = psi_nn(wall)
    return w[: n - 1].copy(), float(w[n - 1]), float(w[n])


# ---------------------------------------------------------------------------
# Steady solver
# ---------------------------------------------------------------------------
@dataclass
class SpectralSolution:
    """Container for one converged spectral cavity solution."""

    reynolds: float
    n: int
    xi: np.ndarray
    psi: np.ndarray                 # (n+1, n+1), index [iy, ix]
    omega: np.ndarray               # (n+1, n+1)
    u: np.ndarray                   # psi_y
    v: np.ndarray                   # -psi_x
    lid_velocity: float = 1.0
    metrics: dict = field(default_factory=dict)
    newton: dict = field(default_factory=dict)

    def centreline_u(self, y_targets) -> np.ndarray:
        """``u(y)`` on the vertical centreline ``x = 0.5``."""
        y = np.atleast_1d(np.asarray(y_targets, dtype=float))
        L = barycentric_matrix(1.0 - 2.0 * y, self.xi)
        return np.array([row @ self.u[:, self.n // 2] for row in L])

    def centreline_v(self, x_targets) -> np.ndarray:
        """``v(x)`` on the horizontal centreline ``y = 0.5``."""
        x = np.atleast_1d(np.asarray(x_targets, dtype=float))
        L = barycentric_matrix(1.0 - 2.0 * x, self.xi)
        return np.array([row @ self.v[self.n // 2, :] for row in L])


class SpectralCavity:
    """Chebyshev collocation + Newton solver for the steady lid-driven cavity.

    Parameters
    ----------
    n :
        Polynomial order per direction (``n + 1`` nodes per side).  Must be even
        so that ``x = 0.5`` / ``y = 0.5`` are grid lines.
    """

    def __init__(self, n: int, *, corner_rule: str = "y") -> None:
        if n % 2:
            raise ValueError("n must be even so that the centreline is a grid line")
        if n > 160:
            raise ValueError("n > 160 exceeds the dense-LU cost/memory budget")
        if n < 6:
            raise ValueError("n < 6 is not a usable cavity discretisation")
        self.n = int(n)
        self.n1 = n + 1
        self.xi, self.D1xi, self.D2xi = differentiation_matrices(self.n)
        # map xi -> X in [0, 1]:  X = (1 - xi) / 2  =>  d/dX = -2 d/dxi
        self.D1 = -2.0 * self.D1xi
        self.D2 = 4.0 * self.D2xi
        self.X = 0.5 * (1.0 - self.xi)
        self._quad_w = clenshaw_curtis_weights(self.xi)   # on [-1, 1]

        n1 = self.n1
        I = sp.identity(n1, format="csr")
        D1s = sp.csr_matrix(self.D1)
        D2s = sp.csr_matrix(self.D2)
        # operators on the flattened (n1, n1) field, C order: index = iy*n1 + ix
        self._Ax = sp.kron(I, D1s, format="csr")      # d/dX
        self._Ay = sp.kron(D1s, I, format="csr")      # d/dY
        self._L = (sp.kron(I, D2s, format="csr")
                   + sp.kron(D2s, I, format="csr"))   # Laplacian

        interior_X = self.X[1 : self.n]
        self._w_top, self._w_top_wall, self._w_top_slope = (
            wall_second_derivative_functional(self.n, interior_X, +1))
        self._w_bot, self._w_bot_wall, self._w_bot_slope = (
            wall_second_derivative_functional(self.n, interior_X, -1))
        self._w_rgt, self._w_rgt_wall, self._w_rgt_slope = (
            wall_second_derivative_functional(self.n, interior_X, +1))
        self._w_lft, self._w_lft_wall, self._w_lft_slope = (
            wall_second_derivative_functional(self.n, interior_X, -1))

        iy, ix = np.meshgrid(np.arange(n1), np.arange(n1), indexing="ij")
        flat = (iy * n1 + ix).ravel()
        interior_mask = ((iy > 0) & (iy < self.n)
                         & (ix > 0) & (ix < self.n)).ravel()
        self._interior = flat[interior_mask]
        self._wall = np.setdiff1d(flat, self._interior)
        wy, wx = self._wall // n1, self._wall % n1

        m = self._interior.size
        col_of = -np.ones(n1 * n1, dtype=int)         # flat -> psi-unknown index
        col_of[self._interior] = np.arange(m)

        def _rows_y(row_idx, jx, at_top):
            w = self._w_top if at_top else self._w_bot
            row = np.zeros(m)
            for kk, iy_i in enumerate(range(1, self.n)):
                row[col_of[iy_i * n1 + jx]] = w[kk]
            return row, (self._w_top_slope if at_top else self._w_bot_slope)

        def _rows_x(row_idx, iy_i, at_right):
            w = self._w_rgt if at_right else self._w_lft
            row = np.zeros(m)
            for kk, jx_i in enumerate(range(1, self.n)):
                row[col_of[iy_i * n1 + jx_i]] = w[kk]
            return row, (self._w_rgt_slope if at_right else self._w_lft_slope)

        nw_wall = self._wall.size
        Ky = np.zeros((nw_wall, m))
        Kx = np.zeros((nw_wall, m))
        cy = np.zeros(nw_wall)
        cx = np.zeros(nw_wall)
        on_y, on_x = wy % self.n == 0, wx % self.n == 0
        for r, node in enumerate(self._wall):
            if on_y[r]:
                row, slope = _rows_y(r, wx[r], wy[r] == self.n)
                Ky[r] = row
                # only the moving lid carries a non-zero tangential velocity
                cy[r] = slope if wy[r] == self.n else 0.0
            if on_x[r]:
                row, slope = _rows_x(r, wy[r], wx[r] == self.n)
                Kx[r] = row
                cx[r] = 0.0                                # v = 0 on the side walls

        K = np.zeros((nw_wall, m))
        cvec = np.zeros(nw_wall)
        y_only, x_only = on_y & ~on_x, on_x & ~on_y
        corner = on_x & on_y
        K[y_only], cvec[y_only] = Ky[y_only], cy[y_only]
        K[x_only], cvec[x_only] = Kx[x_only], cx[x_only]
        if corner_rule == "y":
            K[corner], cvec[corner] = Ky[corner], cy[corner]
        elif corner_rule == "x":
            K[corner], cvec[corner] = Kx[corner], cx[corner]
        elif corner_rule == "mean":
            K[corner] = 0.5 * (Ky[corner] + Kx[corner])
            cvec[corner] = 0.5 * (cy[corner] + cx[corner])
        else:
            raise ValueError("corner_rule must be 'y', 'x' or 'mean'")
        self.corner_rule = corner_rule
        self._K = sp.csr_matrix(K)
        self._wall_const_unit_lid = cvec
        self._wall_kind_y = on_y

    # -- residual / Jacobian ----------------------------------------------
    def _fields(self, state: np.ndarray, lid_velocity: float):
        """Expand the unknowns into full ``(psi, omega)`` fields.

        The unknowns are ``[omega(interior) | psi(interior)]``; the wall
        vorticity is a slave of ``psi`` through the no-slip closure
        ``omega_wall = -(psi_nn(wall))``.
        """
        nw = self.n1 * self.n1
        m = self._interior.size
        omega = np.zeros(nw)
        omega[self._interior] = state[:m]
        omega[self._wall] = -(self._K @ state[m:] + lid_velocity * self._wall_const_unit_lid)
        psi = np.zeros(nw)
        psi[self._interior] = state[m:]
        return psi.reshape(self.n1, self.n1), omega.reshape(self.n1, self.n1)

    def residual(self, state: np.ndarray, reynolds: float,
                 lid_velocity: float = 1.0) -> np.ndarray:
        """Steady residual ``[Poisson(interior) | transport(interior)]``."""
        psi, omega = self._fields(state, lid_velocity)
        psif, omf = psi.ravel(), omega.ravel()
        inter = self._interior
        psi_x, psi_y = self._Ax @ psif, self._Ay @ psif
        om_x, om_y = self._Ax @ omf, self._Ay @ omf
        r_poisson = (self._L @ psif + omf)[inter]
        r_transport = (psi_y * om_x - psi_x * om_y - (self._L @ omf) / reynolds)[inter]
        return np.concatenate([r_poisson, r_transport])

    def jacobian(self, state: np.ndarray, reynolds: float,
                 lid_velocity: float = 1.0) -> np.ndarray:
        """Dense Newton Jacobian of :meth:`residual` (size ``2m x 2m``)."""
        psi, omega = self._fields(state, lid_velocity)
        psif, omf = psi.ravel(), omega.ravel()
        inter, wall = self._interior, self._wall
        m = inter.size

        psi_x, psi_y = self._Ax @ psif, self._Ay @ psif
        om_x, om_y = self._Ax @ omf, self._Ay @ omf

        J = np.zeros((2 * m, 2 * m))
        # Poisson rows: r = L psi + omega  ->  d/domega_int = I, d/dpsi_int = L_int
        J[:m, :m] = np.eye(m)
        J[:m, m:] = self._L[inter][:, inter].toarray()

        # transport rows
        A = (sp.diags(psi_y) @ self._Ax - sp.diags(psi_x) @ self._Ay
             - self._L / reynolds)[inter]          # d/domega (m x nw)
        B = (sp.diags(om_x) @ self._Ay - sp.diags(om_y) @ self._Ax)[inter]  # d/dpsi (m x nw)
        J[m:, :m] = A[:, inter].toarray()
        # d/dpsi_int goes through psi directly and through the slave omega_wall
        J[m:, m:] = B[:, inter].toarray() - A[:, wall] @ self._K
        return J

    # -- driver -----------------------------------------------------------
    def solve(
        self,
        reynolds: float,
        *,
        lid_velocity: float = 1.0,
        psi: np.ndarray | None = None,
        omega: np.ndarray | None = None,
        tol: float = 1e-10,
        max_iter: int = 60,
        continuation: Iterable[float] = (),
        verbose: bool = False,
    ) -> SpectralSolution:
        """Newton solve at ``reynolds``, optionally through a continuation path.

        ``continuation`` is a sequence of intermediate Reynolds numbers solved in
        order, each warm-started from the previous one; that is what makes the
        Newton iteration robust up to Re = 1000.  The returned solution is always
        the one at ``reynolds``.
        """
        n1 = self.n1
        psi = np.zeros((n1, n1)) if psi is None else np.array(psi, dtype=float, copy=True)
        omega = np.zeros((n1, n1)) if omega is None else np.array(omega, dtype=float, copy=True)

        history: list[dict] = []
        for re_step in list(continuation) + [float(reynolds)]:
            psi, omega = self._newton(psi, omega, float(re_step), lid_velocity,
                                      tol, max_iter, history, verbose)
        return self._assemble(float(reynolds), lid_velocity, psi, omega, history)

    def _newton(self, psi, omega, re_val, lid_velocity, tol, max_iter, history, verbose):
        n1 = self.n1
        inter, wall = self._interior, self._wall
        state = np.concatenate([omega.ravel()[inter], psi.ravel()[inter]])
        res = self.residual(state, re_val, lid_velocity)
        norm = float(np.linalg.norm(res))
        scale = max(1.0, float(np.linalg.norm(state)),
                    lid_velocity * float(np.max(np.abs(self._wall_const_unit_lid))))
        converged = norm <= tol * scale
        for it in range(1, max_iter + 1):
            if converged:
                break
            J = self.jacobian(state, re_val, lid_velocity)
            # The spectral vorticity closure puts entries ~N**4 in the psi block
            # while the Poisson rows are O(1); without equilibration the dense LU
            # resolves only ~8 significant digits.  Row/column equilibration is
            # exact (a diagonal change of variables) and drops the condition
            # number by ~4 orders of magnitude.
            try:
                row_s = 1.0 / np.maximum(np.abs(J).max(axis=1), 1e-300)
                Js = J * row_s[:, None]
                col_s = 1.0 / np.maximum(np.abs(Js).max(axis=0), 1e-300)
                Js *= col_s[None, :]
                # NOTE: scipy.linalg.lu_factor returns a factorisation with a
                # spurious exact-zero pivot on this matrix (NaN step) while
                # LAPACK dgesv as reached through numpy solves it to 1e-15;
                # numpy's wrapper is therefore used here.
                step = np.linalg.solve(Js, -res * row_s) * col_s
            except (np.linalg.LinAlgError, ValueError) as exc:  # pragma: no cover
                raise RuntimeError(f"singular Newton Jacobian at Re={re_val}") from exc
            alpha, accepted = 1.0, False
            for _ in range(30):
                candidate = state + alpha * step
                cand_res = self.residual(candidate, re_val, lid_velocity)
                cand_norm = float(np.linalg.norm(cand_res))
                if np.isfinite(cand_norm) and cand_norm < norm:
                    accepted = True
                    break
                alpha *= 0.5
            if not accepted:  # pragma: no cover
                raise RuntimeError(
                    f"Newton line search failed at Re={re_val} (|R|={norm:.3e})")
            state, res, norm = candidate, cand_res, cand_norm
            converged = norm <= tol * scale
            history.append({"reynolds": re_val, "iteration": it,
                            "residual_norm": norm, "damping": alpha})
            if verbose:
                print(f"  Re={re_val:7.1f} it={it:3d} |R|={norm:.3e} alpha={alpha:.3f}")
        if not converged:
            raise RuntimeError(
                f"Newton did not converge at Re={re_val}: |R|={norm:.3e} "
                f"after {max_iter} iterations")
        psi_out = np.zeros(n1 * n1)
        psi_out[inter] = state[inter.size:]
        omega_out = np.zeros(n1 * n1)
        omega_out[inter] = state[:inter.size]
        omega_out[wall] = -(self._K @ state[inter.size:]
                            + lid_velocity * self._wall_const_unit_lid)
        return psi_out.reshape(n1, n1), omega_out.reshape(n1, n1)

    def _assemble(self, reynolds, lid_velocity, psi, omega, history) -> SpectralSolution:
        n1 = self.n1
        u = (self._Ay @ psi.ravel()).reshape(n1, n1)
        v = -(self._Ax @ psi.ravel()).reshape(n1, n1)
        sol = SpectralSolution(reynolds=reynolds, n=self.n, xi=self.xi,
                               psi=psi, omega=omega, u=u, v=v,
                               lid_velocity=lid_velocity,
                               newton={"iterations": len(history),
                                       "history": history})
        sol.metrics = self.metrics(sol)
        return sol

    def remap(self, psi: np.ndarray, omega: np.ndarray, source_xi: np.ndarray):
        """Interpolate ``(psi, omega)`` from another CGL grid onto this one."""
        Lt = barycentric_matrix(self.xi, np.asarray(source_xi, dtype=float))
        psi_new = Lt @ np.asarray(psi, dtype=float) @ Lt.T
        omega_new = Lt @ np.asarray(omega, dtype=float) @ Lt.T
        psi_new[0, :] = psi_new[-1, :] = psi_new[:, 0] = psi_new[:, -1] = 0.0
        return psi_new, omega_new

    # -- diagnostics -------------------------------------------------------
    def integrate(self, field: np.ndarray) -> float:
        """Quadrature integral of ``field`` over the unit square."""
        wx = self._quad_w / 2.0            # map [-1, 1] -> [0, 1]
        return float(wx @ field @ wx)

    def metrics(self, sol: SpectralSolution) -> dict:
        """Grid-independent quantities plus the diagnostics that qualify them."""
        n, n1 = self.n, self.n1
        X = self.X
        yfine = np.linspace(0.0, 1.0, 4001)
        ufine = sol.centreline_u(yfine)
        vfine = sol.centreline_v(yfine)
        kmin = int(np.argmin(ufine))
        kmax = int(np.argmax(vfine))

        # streamfunction minimum and primary-vortex centre (refined by 1-D
        # spectral interpolation of the local patch)
        psi = sol.psi
        sub = np.unravel_index(int(np.argmin(psi[1:n, 1:n])), (n - 1, n - 1))
        iy0, ix0 = sub[0] + 1, sub[1] + 1
        ypatch = np.linspace(X[iy0 - 1], X[min(iy0 + 1, n)], 41)
        xpatch = np.linspace(X[ix0 - 1], X[min(ix0 + 1, n)], 41)
        Ly = barycentric_matrix(1.0 - 2.0 * ypatch, self.xi)
        Lx = barycentric_matrix(1.0 - 2.0 * xpatch, self.xi)
        col = Ly @ psi[:, ix0]
        jy = int(np.argmin(col))
        y_c = float(ypatch[jy])
        row = Lx @ psi[iy0, :]
        jx = int(np.argmin(row))
        x_c = float(xpatch[jx])

        # wall slip defect: no-slip demands psi_y = U_lid on the lid and
        # psi_n = 0 on the other walls; report the achieved value.
        psi_x = (self._Ax @ psi.ravel()).reshape(n1, n1)
        psi_y = (self._Ay @ psi.ravel()).reshape(n1, n1)
        # the lid corners carry the usual lid-driven-cavity singularity, so the
        # slip is measured where the spectral series is resolved (|x - 1/2| < 0.4)
        inner = slice(int(np.argmax(self.X >= 0.1)), int(np.argmax(self.X > 0.9)))
        lid_slip = float(np.max(np.abs(psi_y[-1, inner] - sol.lid_velocity)))
        other_slip = float(max(
            np.max(np.abs(psi_y[0, inner])),
            np.max(np.abs(np.concatenate([psi_x[inner, 0], psi_x[inner, -1]]))),
        ))
        speed2 = sol.u ** 2 + sol.v ** 2
        return {
            "u_min": float(ufine[kmin]),
            "y_at_u_min": float(yfine[kmin]),
            "v_max": float(vfine[kmax]),
            "x_at_v_max": float(yfine[kmax]),
            "psi_min": float(col[jy]),
            "vortex_centre": [x_c, y_c],
            "kinetic_energy": 0.5 * self.integrate(speed2),
            "lid_slip_defect": lid_slip,
            "wall_slip_defect": other_slip,
            "poisson_residual_max": float(np.max(np.abs(
                (self._L @ psi.ravel() + sol.omega.ravel()).reshape(n1, n1)[1:n, 1:n]))),
            "lid_velocity_max": float(np.max(sol.u[-1, :])),
        }