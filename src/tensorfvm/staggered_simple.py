"""Steady incompressible Navier--Stokes on a staggered Cartesian grid.

Shared numerical kernel for the incompressible verification baselines of
:mod:`tensorfvm` (backward-facing step, planar sudden expansion, differentially
heated cavity).  It is deliberately independent of the LBM reference chain:
nothing here reuses lattice-Boltzmann data, and the discretisation is a textbook
finite-volume SIMPLE scheme, so the two code paths can fail independently.

Why NumPy and not torch
-----------------------
The kernels here run on 2-D grids of 10^4--10^5 cells and are called thousands
of times inside the SIMPLE outer loop, i.e. they are *dispatch* bound, not
bandwidth bound.  Measured on the loong64 development host, a single 192k-element
torch elementwise add costs ~0.7 ms (14 ms with the default 128-thread pool)
against ~0.13 ms for the same NumPy operation, and small 2k-element operations
cost 14 us vs 2 us.  The torch version of this module was 5--100x slower for
exactly the same discretisation, so the kernel is written in NumPy
(``float64``); the surrounding package and its other solvers stay on torch.

Discretisation
--------------
* MAC (staggered) grid: ``u`` on x-faces ``(nx+1, ny)``, ``v`` on y-faces
  ``(nx, ny+1)``, scalars (``p``, ``T``) at cell centres ``(nx, ny)``.
* Geometry is a **blocked-cell** mask: solid cells are removed from the
  pressure/transport stencils and every face touching them carries zero mass
  flux and zero velocity (stair-step walls).  A face whose neighbour is a wall
  uses the half-cell distance ``h/2`` (twice the conductance), so the wall shear
  is a one-sided gradient at the right distance instead of being smeared over a
  full cell.
* Convection is implicit first-order upwind (this is what keeps the matrix
  diagonally dominant) plus a **deferred correction** with a van Leer
  (harmonic, TVD) limited reconstruction, so the converged solution is
  second-order accurate in smooth regions while staying bounded at the
  re-entrant corners of the step geometry.
* Pressure--velocity coupling is SIMPLE (Patankar): momentum with the current
  pressure gradient as a source, then a pressure-correction Poisson equation
  whose solution also corrects the face fluxes.
* The pressure-correction operator is solved by a geometric multigrid V-cycle
  (red-black Gauss--Seidel smoother, face-conductance averaging, bilinear
  prolongation, direct solve on the coarsest level) rebuilt from the current
  conductances on every cycle, so the same class also serves the
  variable-coefficient SIMPLE operator.
* Optional Boussinesq coupling: steady energy equation for ``T`` plus the source
  ``rho*g*beta*(T - T_ref)`` in the y-momentum equation.

Sign convention: the discrete operator assembled everywhere in this module is
``A phi|_c = sum_faces k_f (phi_nb - phi_c) = rhs``; the Gauss--Seidel update is
therefore ``phi_c = (sum_faces k_f phi_nb - rhs) / sum_faces k_f``.
"""

from __future__ import annotations

import numpy as np

DTYPE = np.float64
_TINY = 1e-30


# --------------------------------------------------------------------------- #
# array helpers
# --------------------------------------------------------------------------- #
def _shift(f: np.ndarray, di: int, dj: int) -> np.ndarray:
    """``g[i, j] = f[i + di, j + dj]`` with zero outside the array."""
    out = np.zeros_like(f)
    nx, ny = f.shape
    si = slice(0, nx - di) if di >= 0 else slice(-di, nx)
    sj = slice(0, ny - dj) if dj >= 0 else slice(-dj, ny)
    ti = slice(di, nx) if di >= 0 else slice(0, nx + di)
    tj = slice(dj, ny) if dj >= 0 else slice(0, ny + dj)
    out[si, sj] = f[ti, tj]
    return out


def _shift_clamped(f: np.ndarray, di: int, dj: int) -> np.ndarray:
    """Like :func:`_shift` but edge values are replicated."""
    nx, ny = f.shape
    ii = np.clip(np.arange(nx) + di, 0, nx - 1)
    jj = np.clip(np.arange(ny) + dj, 0, ny - 1)
    return f[ii][:, jj]


def _rbgs(phi, rhs, e, w, n, s, diag, update, sweeps, pinned=None):
    """Red-black Gauss--Seidel sweeps for ``sum_nb k_nb phi_nb - diag phi = rhs``.

    The update is ``phi_c = (sum_nb k_nb phi_nb - rhs) / diag``, consistent with
    :meth:`MaskedPoisson._apply`.
    """
    if sweeps <= 0:
        return phi
    nx, ny = phi.shape
    ii = np.arange(nx)[:, None]
    jj = np.arange(ny)[None, :]
    parity = (ii + jj) % 2
    colours = (parity == 0, parity == 1)
    safe = np.where(diag > 0, diag, 1.0)
    for _ in range(sweeps):
        for col in colours:
            num = (e * _shift(phi, 1, 0) + w * _shift(phi, -1, 0)
                   + n * _shift(phi, 0, 1) + s * _shift(phi, 0, -1))
            new = (num - rhs) / safe
            mask = col & update
            phi[mask] = new[mask]
            if pinned is not None:
                phi[pinned] = 0.0
    return phi


class MaskedPoisson:
    """Geometric multigrid for ``A phi|_c = sum_f k_f (phi_nb - phi_c)``.

    ``k`` are x-face ``(nx+1, ny)`` and y-face ``(nx, ny+1)`` conductances.
    ``k = 0`` on a face means zero normal gradient (Neumann); a Dirichlet
    boundary is a *nonzero* face conductance whose outside value is zero, so it
    contributes to the diagonal only.
    """

    def __init__(self, fluid: np.ndarray):
        levels = [fluid]
        f = fluid
        while (f.shape[0] % 2 == 0 and f.shape[1] % 2 == 0
               and min(f.shape) > 4):
            # OR-coarsening: a coarse cell is active if any child is, which keeps
            # the coarse operator free of isolated cells (AND-coarsening creates
            # them at re-entrant corners and the correction then stagnates)
            f = (f[0::2, 0::2] | f[1::2, 0::2] | f[0::2, 1::2] | f[1::2, 1::2])
            levels.append(f)
        self.levels = levels
        self.nlevels = len(levels)
        self._coarse_solver = None

    # -- operator pieces ---------------------------------------------------
    @staticmethod
    def _operator(fluid, ku, kv):
        diag = ku[1:, :] + ku[:-1, :] + kv[:, 1:] + kv[:, :-1]
        e = ku[1:, :] * _shift(fluid, 1, 0)
        w = ku[:-1, :] * _shift(fluid, -1, 0)
        n = kv[:, 1:] * _shift(fluid, 0, 1)
        s = kv[:, :-1] * _shift(fluid, 0, -1)
        return e, w, n, s, diag

    @staticmethod
    def _apply(e, w, n, s, diag, phi):
        return (e * _shift(phi, 1, 0) + w * _shift(phi, -1, 0)
                + n * _shift(phi, 0, 1) + s * _shift(phi, 0, -1) - diag * phi)

    @staticmethod
    def _coarsen_vector(v):
        return 0.25 * (v[0::2, 0::2] + v[1::2, 0::2] + v[0::2, 1::2] + v[1::2, 1::2])

    @staticmethod
    def _coarsen_faces(ku, kv):
        kuc = 0.5 * (ku[0::2, 0::2] + ku[0::2, 1::2])
        kvc = 0.5 * (kv[0::2, 0::2] + kv[1::2, 0::2])
        return kuc, kvc

    @staticmethod
    def _prolong(c: np.ndarray, fine_shape) -> np.ndarray:
        """Bilinear coarse-to-fine transfer (edge values replicated)."""
        cm = np.empty_like(c)
        cm[1:] = c[:-1]
        cm[0] = c[0]
        cp = np.empty_like(c)
        cp[:-1] = c[1:]
        cp[-1] = c[-1]
        cs = np.empty_like(c)
        cs[:, 1:] = c[:, :-1]
        cs[:, 0] = c[:, 0]
        cn = np.empty_like(c)
        cn[:, :-1] = c[:, 1:]
        cn[:, -1] = c[:, -1]
        diag = {}
        for name, base in (("m", cm), ("p", cp)):
            for nm, shift in (("s", 1), ("n", -1)):
                d = np.empty_like(c)
                if shift > 0:
                    d[:, 1:] = base[:, :-1]
                    d[:, 0] = base[:, 0]
                else:
                    d[:, :-1] = base[:, 1:]
                    d[:, -1] = base[:, -1]
                diag[name + nm] = d
        out = np.empty(fine_shape, dtype=c.dtype)
        out[0::2, 0::2] = (9.0 * c + 3.0 * cm + 3.0 * cs + diag["ms"]) / 16.0
        out[0::2, 1::2] = (9.0 * c + 3.0 * cm + 3.0 * cn + diag["mn"]) / 16.0
        out[1::2, 0::2] = (9.0 * c + 3.0 * cp + 3.0 * cs + diag["ps"]) / 16.0
        out[1::2, 1::2] = (9.0 * c + 3.0 * cp + 3.0 * cn + diag["pn"]) / 16.0
        return out

    def _coarse_direct(self, level, ku, kv, rhs, pinned):
        """Dense direct solve on the coarsest level (a few hundred unknowns).

        Written with NumPy only: the kernel must run where SciPy is unavailable
        (the loong64 compute nodes ship NumPy but no SciPy), and a dense solve of
        the coarse system is both simpler and faster than a sparse factorisation
        at this size.
        """
        fluid = self.levels[level]
        e, w, n, s, diag = self._operator(fluid, ku, kv)
        nx, ny = fluid.shape
        cells = np.argwhere(fluid)
        m = len(cells)
        out = np.zeros((nx, ny), dtype=rhs.dtype)
        if m == 0:
            return out
        index = -np.ones((nx, ny), dtype=int)
        index[fluid] = np.arange(m)
        A = np.zeros((m, m), dtype=rhs.dtype)
        diag_c = diag[fluid]
        active = diag_c > 0
        A[np.arange(m)[active], np.arange(m)[active]] = -diag_c[active]
        A[np.arange(m)[~active], np.arange(m)[~active]] = 1.0
        for arr, di, dj in ((e, 1, 0), (w, -1, 0), (n, 0, 1), (s, 0, -1)):
            kvals = np.where(active, arr[fluid], 0.0)
            ii = cells[:, 0] + di
            jj = cells[:, 1] + dj
            ok = (ii >= 0) & (ii < nx) & (jj >= 0) & (jj < ny)
            nb = np.full(m, -1, dtype=int)
            nb[ok] = index[ii[ok], jj[ok]]
            has = nb >= 0
            A[np.arange(m)[has], nb[has]] += kvals[has]
        bm = np.where(active, rhs[fluid], 0.0)
        if pinned is not None:
            pi = min(int(pinned[0]), nx - 1)
            pj = min(int(pinned[1]), ny - 1)
            if fluid[pi, pj]:
                row = index[pi, pj]
                A[row, :] = 0.0
                A[:, row] = 0.0
                A[row, row] = 1.0
                bm[row] = 0.0
        try:
            sol = np.linalg.solve(A, bm)
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(A, bm, rcond=None)[0]
        sol = np.where(np.isfinite(sol), sol, 0.0)
        out[fluid] = sol
        return out

    def solve(self, ku, kv, rhs, phi=None, *, cycles=8, tol=1e-6,
              pre=1, post=1, pinned=None):
        fluid = self.levels[0]
        if phi is None:
            phi = np.zeros_like(rhs)
        scale = float(np.abs(rhs).max())
        if scale <= 0.0:
            return phi
        e, w, n, s, diag = self._operator(fluid, ku, kv)
        for _ in range(cycles):
            phi = self._vcycle(0, ku, kv, rhs, phi, pre, post, pinned)
            res = rhs - self._apply(e, w, n, s, diag, phi)
            res = np.where(fluid, res, 0.0)
            if float(np.abs(res).max()) <= tol * scale:
                break
        return phi

    def _vcycle(self, level, ku, kv, rhs, phi, pre, post, pinned):
        fluid = self.levels[level]
        e, w, n, s, diag = self._operator(fluid, ku, kv)
        local_pin = None
        if pinned is not None:
            local_pin = (min(pinned[0] >> level, fluid.shape[0] - 1),
                         min(pinned[1] >> level, fluid.shape[1] - 1))
        if level == self.nlevels - 1:
            out = self._coarse_direct(level, ku, kv, rhs, local_pin)
            return np.where(fluid, out, 0.0)
        phi = _rbgs(phi, rhs, e, w, n, s, diag, fluid, pre, pinned=local_pin)
        res = rhs - self._apply(e, w, n, s, diag, phi)
        kuc, kvc = self._coarsen_faces(ku, kv)
        rc = self._coarsen_vector(res)
        rc = np.where(self.levels[level + 1], rc, 0.0)
        ec = self._vcycle(level + 1, kuc, kvc, rc, np.zeros_like(rc),
                          pre, post, pinned)
        phi = phi + self._prolong(ec, phi.shape)
        phi = _rbgs(phi, rhs, e, w, n, s, diag, fluid, post, pinned=local_pin)
        return phi



class SparseDirectPoisson:
    """Sparse direct pressure-correction solver (optional SciPy fast path).

    Solves exactly the same masked 5-point operator as :class:`MaskedPoisson`
    (``sum_faces k_f (phi_nb - phi_c) = rhs``) but factorises it with SuperLU
    instead of running V-cycles, which removes the multigrid coarse-grid dense
    solve from the inner SIMPLE loop (measured: 96 % of the iteration time at
    4.4k cells).

    The conductances change every outer iteration, so the factorisation is
    **reused** while they vary by less than ``stale_tol`` (and for at most
    ``refresh`` calls); a stale factorisation is an approximate inverse, and the
    returned field is polished with ``refine`` iterative-refinement steps so it
    still solves the *current* operator to round-off.  Because the SIMPLE fixed
    point only requires the *correction* to drive the continuity defect to zero,
    neither the reuse policy nor ``refine`` changes the converged solution.

    SciPy is an optional dependency for this kernel: the loong64 compute nodes
    ship NumPy only, and :class:`MaskedPoisson` remains the default there.
    """

    def __init__(self, fluid: np.ndarray, *, refresh: int = 30,
                 stale_tol: float = 1.0e-2, refine: int = 1,
                 min_age: int = 4) -> None:
        from scipy.sparse import csc_matrix  # noqa: F401  (availability probe)

        if fluid.ndim != 2:
            raise ValueError("fluid must be 2-D")
        self.fluid = np.ascontiguousarray(fluid, dtype=bool)
        self.nx, self.ny = self.fluid.shape
        self.cells = np.argwhere(self.fluid)
        self.index = -np.ones(self.fluid.shape, dtype=np.int64)
        self.index[self.fluid] = np.arange(len(self.cells))
        self.refresh = int(refresh)
        self.stale_tol = float(stale_tol)
        self.refine = int(refine)
        self.min_age = int(min_age)
        self.nlevels = 1
        self.factorizations = 0
        self.solves = 0
        self._lu = None
        self._ku = None
        self._kv = None
        self._age = 0

    # ---------------------------------------------------------------- assembly
    def _assemble(self, ku, kv, pinned):
        from scipy.sparse import csc_matrix
        fluid, index, cells = self.fluid, self.index, self.cells
        e, w, n, s, diag = MaskedPoisson._operator(fluid, ku, kv)
        m = len(cells)
        diag_c = diag[fluid]
        active = diag_c > 0
        rows = [np.arange(m)]
        cols = [np.arange(m)]
        vals = [np.where(active, -diag_c, 1.0)]
        nx, ny = self.nx, self.ny
        for arr, di, dj in ((e, 1, 0), (w, -1, 0), (n, 0, 1), (s, 0, -1)):
            kvals = np.where(active, arr[fluid], 0.0)
            ii = cells[:, 0] + di
            jj = cells[:, 1] + dj
            ok = (ii >= 0) & (ii < nx) & (jj >= 0) & (jj < ny)
            nb = np.full(m, -1, dtype=np.int64)
            nb[ok] = index[ii[ok], jj[ok]]
            has = nb >= 0
            rows.append(np.arange(m)[has])
            cols.append(nb[has])
            vals.append(kvals[has])
        A = csc_matrix((np.concatenate(vals),
                        (np.concatenate(rows), np.concatenate(cols))),
                       shape=(m, m))
        if pinned is not None:
            pi = min(int(pinned[0]), nx - 1)
            pj = min(int(pinned[1]), ny - 1)
            if fluid[pi, pj]:
                row = int(index[pi, pj])
                A = A.tolil()
                A[row, :] = 0.0
                A[:, row] = 0.0
                A[row, row] = 1.0
                A = A.tocsc()
        return A, active

    def _factorise(self, ku, kv, pinned):
        from scipy.sparse.linalg import splu

        A, _ = self._assemble(ku, kv, pinned)
        #: symmetric-structure ordering keeps the fill of the 2-D Laplacian
        #: factorisation 1.6x smaller than the COLAMD default (measured 3.6M vs
        #: 5.7M nonzeros at 76,800 unknowns) and removes the pivoting search.
        self._lu = splu(A, permc_spec="MMD_AT_PLUS_A",
                        options=dict(SymmetricMode=True))
        self._ku, self._kv = ku.copy(), kv.copy()
        self._age = 0
        self.factorizations += 1

    def _reusable(self, ku, kv) -> bool:
        """Reuse the factorisation while the conductances drift slowly.

        Both criteria are checked: the age bound alone would keep a factorisation
        computed during the start-up transient for ``refresh`` iterations, and the
        coefficient bound alone would re-factorise every iteration while the field
        is still developing (measured: every one of the first 3000 iterations).
        A stale factorisation is only ever used as an approximate inverse -- the
        ``refine`` steps in :meth:`solve` re-solve the *current* operator.
        """
        if self._lu is None or self._age >= self.refresh:
            return False
        if self._age < self.min_age:
            return True
        scale_u = max(float(np.abs(self._ku).max()), _TINY)
        scale_v = max(float(np.abs(self._kv).max()), _TINY)
        du = float(np.abs(ku - self._ku).max()) / scale_u
        dv = float(np.abs(kv - self._kv).max()) / scale_v
        return max(du, dv) <= self.stale_tol

    # ------------------------------------------------------------------- solve
    def solve(self, ku, kv, rhs, phi=None, *, cycles=8, tol=1e-6,
              pre=1, post=1, pinned=None):
        """Return ``phi`` with ``sum_faces k_f (phi_nb - phi_c) = rhs`` on fluid cells."""
        del pre, post  # accepted for interface compatibility with MaskedPoisson
        fluid = self.fluid
        scale = float(np.abs(rhs).max())
        if scale <= 0.0 or not fluid.any():
            return np.zeros_like(rhs) if phi is None else phi
        if not self._reusable(ku, kv):
            self._factorise(ku, kv, pinned)
        b = np.where(MaskedPoisson._operator(fluid, ku, kv)[4][fluid] > 0,
                     rhs[fluid], 0.0)
        if pinned is not None:
            pi = min(int(pinned[0]), self.nx - 1)
            pj = min(int(pinned[1]), self.ny - 1)
            if fluid[pi, pj]:
                b[int(self.index[pi, pj])] = 0.0
        sol = self._lu.solve(b)
        self.solves += 1
        self._age += 1
        e, w, n, s, diag = MaskedPoisson._operator(fluid, ku, kv)
        for _ in range(self.refine):
            field = np.zeros(self.nx * self.ny, dtype=DTYPE)
            field[self._flat_fluid()] = sol
            field = field.reshape(self.nx, self.ny)
            res = rhs - MaskedPoisson._apply(e, w, n, s, diag, field)
            res = np.where(fluid, res, 0.0)
            if float(np.abs(res).max()) <= tol * scale:
                break
            corr = self._lu.solve(res[fluid])
            if not np.all(np.isfinite(corr)):
                break
            sol = sol + corr
        out = np.zeros((self.nx, self.ny), dtype=DTYPE)
        out[fluid] = np.where(np.isfinite(sol), sol, 0.0)
        return out

    def _flat_fluid(self):
        return self.fluid.ravel()


def _vanleer_face(f: np.ndarray, vel_face: np.ndarray, axis: int) -> np.ndarray:
    """TVD (van Leer harmonic) limited upwind face value of ``f``.

    ``axis = 0``: face normal along x, ``f`` shaped like x-faces; ``axis = 1``:
    face normal along y.  Only the sign of ``vel_face`` is used.
    """
    if axis == 0:
        up = np.where(vel_face > 0, f, _shift(f, 1, 0))
        far = np.where(vel_face > 0, _shift(f, -1, 0), _shift(f, 2, 0))
        near = np.where(vel_face > 0, _shift(f, 1, 0), _shift(f, -1, 0))
    else:
        up = np.where(vel_face > 0, f, _shift(f, 0, 1))
        far = np.where(vel_face > 0, _shift(f, 0, -1), _shift(f, 0, 2))
        near = np.where(vel_face > 0, _shift(f, 0, 1), _shift(f, 0, -1))
    d_up = up - far
    d_down = near - up
    same = d_up * d_down > 0
    ratio = np.where(same, d_down / np.where(same, np.abs(d_up), 1.0), 0.0)
    r = np.clip(ratio, 0.0, None)
    lim = np.where(same, 2.0 * r / np.maximum(1.0 + r, _TINY), 0.0)
    return up + 0.5 * np.clip(lim, 0.0, 1.0) * d_down


class StaggeredSimple:
    """Steady incompressible flow with SIMPLE on a blocked-cell MAC grid.

    Boundary conventions (all three verification cases fit them):

    * west boundary: ``inlet_profile`` given -> inflow with prescribed ``u``
      (``v = 0``); otherwise a no-slip wall.
    * east boundary: ``east_outlet`` -> zero-gradient velocity and ``p = 0``
      reference; otherwise a no-slip wall.
    * south/north boundaries: no-slip walls.
    * interior block: any cell with ``fluid == False``.
    """

    def __init__(self, nx: int, ny: int, hx: float, hy: float, nu: float,
                 fluid: np.ndarray | None = None, *,
                 inlet_profile: np.ndarray | None = None,
                 east_outlet: bool = True,
                 density: float = 1.0,
                 alpha_u: float = 0.7, alpha_p: float = 0.3, alpha_T: float = 0.8,
                 buoyancy: float = 0.0, t_ref: float = 0.0, prandtl: float = 0.71,
                 limited: bool = True, energy: bool = False,
                 dc_blend: float = 1.0, dc_relax: float = 1.0,
                 pseudo_cfl: float = 0.5, pseudo_uref: float | None = None,
                 poisson_solver: str = "multigrid"):
        if nx < 4 or ny < 4:
            raise ValueError("need at least 4x4 cells")
        self.nx, self.ny = int(nx), int(ny)
        self.hx, self.hy = float(hx), float(hy)
        self.nu = float(nu)
        self.rho = float(density)
        self.alpha_u, self.alpha_p = float(alpha_u), float(alpha_p)
        self.alpha_T = float(alpha_T)
        self.buoyancy, self.t_ref = float(buoyancy), float(t_ref)
        self.prandtl = float(prandtl)
        self.limited = bool(limited)
        self.dc_blend = float(dc_blend)
        self.dc_relax = float(dc_relax)
        self.pseudo_cfl = float(pseudo_cfl)
        self.pseudo_uref = None if pseudo_uref is None else float(pseudo_uref)
        self.energy = bool(energy) or self.buoyancy != 0.0
        self.east_outlet = bool(east_outlet)
        if fluid is None:
            fluid = np.ones((nx, ny), dtype=bool)
        self.fluid = np.ascontiguousarray(fluid, dtype=bool)
        fl = self.fluid

        self.u = np.zeros((nx + 1, ny), dtype=DTYPE)
        self.v = np.zeros((nx, ny + 1), dtype=DTYPE)
        self.p = np.zeros((nx, ny), dtype=DTYPE)
        self.T = np.zeros((nx, ny), dtype=DTYPE)

        # ---------------- static face topology ----------------------------
        kx = np.full((nx + 1, ny), 3, dtype=np.int8)
        kx[1:nx] = np.where(fl[:-1] & fl[1:], 0, 3)
        if inlet_profile is not None:
            kx[0] = np.where(fl[0], 1, 3)
        if self.east_outlet:
            kx[nx] = np.where(fl[nx - 1], 2, 3)
        ky = np.full((nx, ny + 1), 3, dtype=np.int8)
        ky[:, 1:ny] = np.where(fl[:, :-1] & fl[:, 1:], 0, 3)
        self.kx, self.ky = kx, ky
        self.u_free = kx == 0
        self.v_free = ky == 0
        self.u_open = (kx != 3)
        self.v_open = (ky != 3)
        if inlet_profile is not None:
            self.inlet_values = np.array(inlet_profile, dtype=DTYPE).reshape(ny)
        else:
            self.inlet_values = np.zeros(ny, dtype=DTYPE)

        self._dc_u = None
        self._dc_v = None
        self._dc_T = None
        self.dc_frozen = False
        if poisson_solver not in ("multigrid", "sparse"):
            raise ValueError("poisson_solver must be 'multigrid' or 'sparse'")
        self.poisson_solver = poisson_solver
        if poisson_solver == "sparse":
            self.poisson = SparseDirectPoisson(fl)
        else:
            self.poisson = MaskedPoisson(fl)
        self.pinned = None if self.east_outlet else (0, 0)
        self.iterations = 0
        self._outflow_initialised = False
        self.history: list[dict] = []

    # ------------------------------------------------------------------ #
    def _pseudo_inertia(self):
        """Pseudo-transient inertia ``rho*V/dt_p`` (see :meth:`iterate`).

        A pure steady SIMPLE loses diagonal dominance where the mass fluxes
        cancel (re-entrant corners, stagnation points): ``a_P -> 0`` there makes
        the velocity-correction scale ``d = A/a_P`` unbounded and the step
        diverges.  Adding the pseudo-transient term bounds
        ``d <= dt_p/h`` and keeps the scheme positive; the steady solution is
        unchanged because the matching source ``a_P0*phi_old`` is added to the
        right-hand side.
        """
        if self.pseudo_cfl <= 0.0:
            return 0.0
        if self.pseudo_uref is not None:
            # Fixed reference velocity: the instantaneous max|u| oscillates (the
            # global mode of the recirculation), and using it as the time-step
            # scale feeds that oscillation back into every coefficient of the
            # momentum equation, which shows up as a persistent dU ~ 1e-2 limit
            # cycle instead of a decaying change.
            uref = max(self.pseudo_uref, 1e-8)
        else:
            uref = max(float(np.abs(self.u).max()), float(np.abs(self.v).max()), 1e-8)
        dt = self.pseudo_cfl * min(self.hx, self.hy) / uref
        return self.rho * self.hx * self.hy / dt

    def apply_bc(self):
        kx, nx, ny = self.kx, self.nx, self.ny
        self.u[0] = np.where(kx[0] == 1, self.inlet_values, 0.0)
        if not self._outflow_initialised:
            # the outflow face is a mass-conserving unknown after the first
            # application: it starts as a zero-gradient extrapolation but is
            # afterwards only adjusted by the pressure correction, otherwise the
            # global mass balance of the domain could never be satisfied.
            self.u[nx] = np.where(kx[nx] == 2, self.u[nx - 1], 0.0)
            self._outflow_initialised = True
        else:
            self.u[nx] = np.where(kx[nx] == 2, self.u[nx], 0.0)
        np.copyto(self.u, 0.0, where=~self.u_open)
        np.copyto(self.v, 0.0, where=~self.v_open)

    # ------------------------------------------------------------------ #
    # cross-grid interpolations
    # ------------------------------------------------------------------ #
    def _v_on_u(self, v):
        """Mean v over the two y-faces spanning a u-cv face -> ``(nx+1, ny)``."""
        vp = v[:, 1:]
        avg = 0.5 * (vp + _shift(vp, -1, 0))
        out = np.zeros((self.nx + 1, self.ny), dtype=DTYPE)
        out[1:] = avg
        return out

    def _u_on_v(self, u):
        """u on the west/east faces of a v-cv -> two ``(nx, ny+1)`` arrays."""
        ua = 0.5 * (u + _shift(u, 0, -1))
        west = np.zeros((self.nx, self.ny + 1), dtype=DTYPE)
        east = np.zeros((self.nx, self.ny + 1), dtype=DTYPE)
        west[:, 1:] = ua[: self.nx, :]
        east[:, 1:] = ua[1:, :]
        return west, east

    @staticmethod
    def _mult(free, is_wall):
        """Diffusion multiplier: 2 at a wall (half-cell), 1 elsewhere."""
        return np.where(free, 1.0, np.where(is_wall, 2.0, 1.0))

    # ------------------------------------------------------------------ #
    def _relax_dc(self, target: np.ndarray, previous: np.ndarray | None
                  ) -> np.ndarray:
        """Relaxed update of a deferred-correction (van Leer) source.

        The limited high-order flux is a *defect*: it is evaluated from the
        current field and added to the momentum/energy source.  Recomputing it
        every outer iteration (``dc_relax = 1``) is the textbook form, but it is
        only a *contraction* while the cell Peclet number stays below ~3; above
        that the explicit defect overshoots and the coupled SIMPLE iteration
        diverges (measured here at ``Pe_cell = 10`` for the sudden expansion and
        ``Pe_cell = 40`` for the backward-facing step, both blowing up within
        500 iterations).  With ``0 < dc_relax < 1`` the source is advanced by a
        fraction of the target per outer iteration, which keeps the same fixed
        point -- at convergence ``dc = dc_target(field)`` exactly -- while making
        the defect iteration a contraction at any Peclet number.
        """
        if previous is None or self.dc_relax >= 1.0:
            return target
        return previous + self.dc_relax * (target - previous)

    # ------------------------------------------------------------------ #
    def _u_coefficients(self):
        nu, hx, hy, rho = self.nu, self.hx, self.hy, self.rho
        u = self.u
        uE, uW, uN, uS = _shift(u, 1, 0), _shift(u, -1, 0), _shift(u, 0, 1), _shift(u, 0, -1)
        freeE, freeW = _shift(self.u_free, 1, 0), _shift(self.u_free, -1, 0)
        freeN, freeS = _shift(self.u_free, 0, 1), _shift(self.u_free, 0, -1)
        d_ew, d_ns = nu * hy / hx, nu * hx / hy
        # u lives on x-faces but on y-cell-centres, so along x the momentum-cv
        # boundaries fall *inside* cells (distance hx to the neighbour face, no
        # wall can sit there), while along y they coincide with cell faces: a
        # non-free y-neighbour is a wall at distance hy/2 -> twice the conductance.
        cE = d_ew * self._mult(freeE, False)
        cW = d_ew * self._mult(freeW, False)
        cN = d_ns * self._mult(freeN, True)
        cS = d_ns * self._mult(freeS, True)
        Ue = 0.5 * (u + uE)
        Uw = 0.5 * (uW + u)
        Vn = self._v_on_u(self.v)
        Vs = self._v_on_u(_shift(self.v, 0, -1))
        # The east/west faces of a u-cv lie *inside* cells i and i-1, so their
        # openness is the cell fluidity (not the neighbour-face freedom): at the
        # outlet column the outflow term must survive, otherwise aP loses
        # diagonal dominance and the last column diverges.
        fm_e = np.zeros((self.nx + 1, self.ny), dtype=DTYPE)
        fm_w = np.zeros((self.nx + 1, self.ny), dtype=DTYPE)
        fm_e[: self.nx] = self.fluid
        fm_w[1:] = self.fluid
        Fe = np.where(fm_e > 0.5, rho * Ue * hy, 0.0)
        Fw = np.where(fm_w > 0.5, rho * Uw * hy, 0.0)
        Fn = np.where(freeN, rho * Vn * hx, 0.0)
        Fs = np.where(freeS, rho * Vs * hx, 0.0)
        aE = cE + np.clip(-Fe, 0.0, None)
        aW = cW + np.clip(Fw, 0.0, None)
        aN = cN + np.clip(-Fn, 0.0, None)
        aS = cS + np.clip(Fs, 0.0, None)
        aP = aE + aW + aN + aS + (Fe - Fw) + (Fn - Fs)
        aP0 = self._pseudo_inertia()
        aP = np.where(aP + aP0 > 1e-12, aP + aP0, 1.0)
        pe = np.zeros((self.nx + 1, self.ny), dtype=DTYPE)
        pw = np.zeros((self.nx + 1, self.ny), dtype=DTYPE)
        pe[: self.nx] = self.p
        pw[1:] = self.p
        b = -(pe - pw) * hy
        if self.limited:
            if self.dc_frozen and self._dc_u is not None:
                b = b - self._dc_u
            else:
                uf_e = _vanleer_face(u, Ue, 0)
                uf_w = _vanleer_face(u, Uw, 0)
                uf_n = _vanleer_face(u, Vn, 1)
                uf_s = _vanleer_face(u, Vs, 1)
                dc = self.dc_blend * (
                    rho * hy * (Ue * (uf_e - np.where(Ue > 0, u, uE))
                                - Uw * (uf_w - np.where(Uw > 0, uW, u)))
                    + rho * hx * (Vn * (uf_n - np.where(Vn > 0, u, uN))
                                  - Vs * (uf_s - np.where(Vs > 0, uS, u))))
                self._dc_u = self._relax_dc(dc, self._dc_u)
                b = b - self._dc_u
        return aE, aW, aN, aS, aP, b

    def _v_coefficients(self):
        nu, hx, hy, rho = self.nu, self.hx, self.hy, self.rho
        v = self.v
        vE, vW, vN, vS = _shift(v, 1, 0), _shift(v, -1, 0), _shift(v, 0, 1), _shift(v, 0, -1)
        freeE, freeW = _shift(self.v_free, 1, 0), _shift(self.v_free, -1, 0)
        freeN, freeS = _shift(self.v_free, 0, 1), _shift(self.v_free, 0, -1)
        d_ew, d_ns = nu * hy / hx, nu * hx / hy
        cE = d_ew * self._mult(freeE, True)
        cW = d_ew * self._mult(freeW, True)
        cN = d_ns * self._mult(freeN, False)
        cS = d_ns * self._mult(freeS, False)
        west, east = self._u_on_v(self.u)
        Vn = 0.5 * (v + vN)
        Vs = 0.5 * (vS + v)
        gm_n = np.zeros((self.nx, self.ny + 1), dtype=DTYPE)
        gm_s = np.zeros((self.nx, self.ny + 1), dtype=DTYPE)
        gm_n[:, : self.ny] = self.fluid
        gm_s[:, 1:] = self.fluid
        Fe = np.where(freeE, rho * east * hy, 0.0)
        Fw = np.where(freeW, rho * west * hy, 0.0)
        Fn = np.where(gm_n > 0.5, rho * Vn * hx, 0.0)
        Fs = np.where(gm_s > 0.5, rho * Vs * hx, 0.0)
        aE = cE + np.clip(-Fe, 0.0, None)
        aW = cW + np.clip(Fw, 0.0, None)
        aN = cN + np.clip(-Fn, 0.0, None)
        aS = cS + np.clip(Fs, 0.0, None)
        aP = aE + aW + aN + aS + (Fe - Fw) + (Fn - Fs)
        aP0 = self._pseudo_inertia()
        aP = np.where(aP + aP0 > 1e-12, aP + aP0, 1.0)
        pn = np.zeros((self.nx, self.ny + 1), dtype=DTYPE)
        ps = np.zeros((self.nx, self.ny + 1), dtype=DTYPE)
        pn[:, : self.ny] = self.p
        ps[:, 1:] = self.p
        b = -(pn - ps) * hx
        if self.buoyancy != 0.0:
            tcell = 0.5 * (self.T + _shift(self.T, 0, -1))
            tface = np.zeros((self.nx, self.ny + 1), dtype=DTYPE)
            tface[:, 1:] = tcell
            b = b + self.rho * self.buoyancy * (tface - self.t_ref) * hx * hy
        if self.limited:
            if self.dc_frozen and self._dc_v is not None:
                b = b - self._dc_v
            else:
                vf_n = _vanleer_face(v, Vn, 1)
                vf_s = _vanleer_face(v, Vs, 1)
                dc = self.dc_blend * (
                    rho * hx * (Vn * (vf_n - np.where(Vn > 0, v, vN))
                                - Vs * (vf_s - np.where(Vs > 0, vS, v)))
                    + rho * hy * (east * (_vanleer_face(v, east, 0) - np.where(east > 0, v, vE))
                                  - west * (_vanleer_face(v, west, 0) - np.where(west > 0, vW, v))))
                self._dc_v = self._relax_dc(dc, self._dc_v)
                b = b - self._dc_v
        return aE, aW, aN, aS, aP, b

    # ------------------------------------------------------------------ #
    def _solve_energy(self):
        fl, hx, hy, rho = self.fluid, self.hx, self.hy, self.rho
        alpha = self.nu / self.prandtl
        T = self.T
        TE, TW, TN, TS = _shift(T, 1, 0), _shift(T, -1, 0), _shift(T, 0, 1), _shift(T, 0, -1)
        Ue = 0.5 * (self.u[:-1] + self.u[1:])
        Uw = 0.5 * (_shift(self.u, -1, 0)[:-1] + _shift(self.u, -1, 0)[1:])
        Vn = 0.5 * (self.v[:, :-1] + self.v[:, 1:])
        Vs = 0.5 * (_shift(self.v, 0, -1)[:, :-1] + _shift(self.v, 0, -1)[:, 1:])
        d_ew, d_ns = alpha * hy / hx, alpha * hx / hy
        cE = d_ew * np.where(_shift(fl, 1, 0), 1.0, 2.0)
        cW = d_ew * np.where(_shift(fl, -1, 0), 1.0, 2.0)
        cN = d_ns * np.where(_shift(fl, 0, 1), 1.0, 2.0)
        cS = d_ns * np.where(_shift(fl, 0, -1), 1.0, 2.0)
        Fe, Fw, Fn, Fs = rho * Ue * hy, rho * Uw * hy, rho * Vn * hx, rho * Vs * hx
        aE = cE + np.clip(-Fe, 0.0, None)
        aW = cW + np.clip(Fw, 0.0, None)
        aN = cN + np.clip(-Fn, 0.0, None)
        aS = cS + np.clip(Fs, 0.0, None)
        aP = aE + aW + aN + aS + (Fe - Fw) + (Fn - Fs)
        aP = np.where(aP > 1e-12, aP, 1.0)
        b = np.zeros_like(T)
        if self.limited:
            if self.dc_frozen and self._dc_T is not None:
                b = b - self._dc_T
            else:
                dc = self.dc_blend * (
                    rho * hy * (Ue * (_vanleer_face(T, Ue, 0) - np.where(Ue > 0, T, TE))
                                - Uw * (_vanleer_face(T, Uw, 0) - np.where(Uw > 0, TW, T)))
                    + rho * hx * (Vn * (_vanleer_face(T, Vn, 1) - np.where(Vn > 0, T, TN))
                                  - Vs * (_vanleer_face(T, Vs, 1) - np.where(Vs > 0, TS, T))))
                self._dc_T = self._relax_dc(dc, self._dc_T)
                b = b - self._dc_T
        bc = getattr(self, "t_bc", None)
        if bc is not None:
            for side, arr, add, dface in (("west", aW, 0, d_ew), ("east", aE, -1, d_ew),
                                          ("south", aS, 0, d_ns), ("north", aN, -1, d_ns)):
                kind, value = bc.get(side, ("adiabatic", 0.0))
                if kind == "fixed":
                    if side in ("west", "east"):
                        b[add, :] += 2.0 * dface * self.rho * value
                    else:
                        b[:, add] += 2.0 * dface * self.rho * value
                elif kind == "adiabatic":
                    if side in ("west", "east"):
                        arr[add, :] = 0.0
                    else:
                        arr[:, add] = 0.0
            aP = aE + aW + aN + aS + (Fe - Fw) + (Fn - Fs)
            aP = np.where(aP > 1e-12, aP, 1.0)
        br = b + (1.0 - self.alpha_T) / self.alpha_T * aP * T
        Tn = _rbgs(T.copy(), -br, aE, aW, aN, aS, aP / self.alpha_T, fl, 4)
        self.T = np.where(fl, Tn, T)

    # ------------------------------------------------------------------ #
    def divergence(self):
        return ((self.u[1:] - self.u[:-1]) * self.hy
                + (self.v[:, 1:] - self.v[:, :-1]) * self.hx)

    def mass_flux(self):
        """Total mass flux through the inlet (for normalising residuals)."""
        return float(np.abs(self.u[0] * self.hy * (self.kx[0] == 1)).sum())

    def iterate(self, n: int = 1, p_cycles: int = 6, p_tol: float = 1e-4,
                momentum_sweeps: int = 6):
        for _ in range(n):
            self.apply_bc()
            aE, aW, aN, aS, aP, b = self._u_coefficients()
            diag_u = aP / self.alpha_u
            rhs_u = -(b + (1.0 - self.alpha_u) / self.alpha_u * aP * self.u
                      + self._pseudo_inertia() * self.u)
            u_new = _rbgs(self.u.copy(), rhs_u, aE, aW, aN, aS, diag_u,
                          self.u_free, momentum_sweeps)
            self.u = np.where(self.u_free, u_new, self.u)
            self.apply_bc()

            vE, vW, vN, vS, vP, bv = self._v_coefficients()
            diag_v = vP / self.alpha_u
            rhs_v = -(bv + (1.0 - self.alpha_u) / self.alpha_u * vP * self.v
                      + self._pseudo_inertia() * self.v)
            v_new = _rbgs(self.v.copy(), rhs_v, vE, vW, vN, vS, diag_v,
                          self.v_free, momentum_sweeps)
            self.v = np.where(self.v_free, v_new, self.v)
            self.apply_bc()

            if self.energy:
                self._solve_energy()

            d_u = np.where(self.u_free, self.hy / np.maximum(diag_u, 1e-30), 0.0)
            east = self.kx[self.nx] == 2
            d_u[self.nx] = np.where(east, d_u[self.nx - 1], d_u[self.nx])
            d_v = np.where(self.v_free, self.hx / np.maximum(diag_v, 1e-30), 0.0)
            ku = self.rho * self.hy * d_u
            kv = self.rho * self.hx * d_v
            div = self.divergence()
            rhs = np.where(self.fluid, div, 0.0)
            pp = self.poisson.solve(ku, kv, rhs, phi=None, cycles=p_cycles,
                                    tol=p_tol, pinned=self.pinned)
            pp = np.where(self.fluid, pp, 0.0)
            if self.pinned is not None:
                pp[self.pinned] = 0.0
            nx, ny = self.nx, self.ny
            pw = np.zeros((nx + 1, ny), dtype=DTYPE)
            pe = np.zeros((nx + 1, ny), dtype=DTYPE)
            pw[1:] = pp
            pe[:nx] = pp
            corr_u = np.where(self.u_free | (self.kx == 2), d_u * (pw - pe), 0.0)
            ps = np.zeros((nx, ny + 1), dtype=DTYPE)
            pn = np.zeros((nx, ny + 1), dtype=DTYPE)
            ps[:, 1:] = pp
            pn[:, :ny] = pp
            corr_v = np.where(self.v_free, d_v * (ps - pn), 0.0)
            self.u = self.u + self.alpha_p * corr_u
            self.v = self.v + self.alpha_p * corr_v
            self.p = self.p + self.alpha_p * pp
            self.apply_bc()
            self.iterations += 1
        return self.state()

    def refresh_deferred(self, blend: float = 1.0):
        """Freeze the van Leer deferred-correction sources from the current field.

        The correction is a *defect correction*: solving with the correction
        frozen at the first-order solution removes the O(h) upwind error without
        the feedback that makes a fully explicit correction diverge above
        ``Pe_cell ~ 3``.
        """
        self.dc_blend = float(blend)
        self.dc_frozen = False
        self._dc_u = self._dc_v = self._dc_T = None
        self._u_coefficients()
        self._v_coefficients()
        if self.energy:
            self._solve_energy()
        self.dc_frozen = True
        return None

    def state(self):
        # normalise the continuity residual by the inlet flux when there is one,
        # otherwise by the dynamic scale rho*U*L (closed cavities have no inflow)
        flux = self.mass_flux()
        if flux <= 0.0:
            uref = max(float(np.abs(self.u).max()), float(np.abs(self.v).max()), 1e-12)
            flux = self.rho * uref * max(self.hx, self.hy)
        cont = float(np.abs(self.divergence()).max()) / max(flux, 1e-30)
        return {"iterations": self.iterations, "continuity": cont,
                "max_velocity": float(max(np.abs(self.u).max(), np.abs(self.v).max()))}

    def run(self, max_iter: int = 6000, tol: float = 1e-7, check: int = 5,
            verbose: bool = False, record_every: int = 200,
            max_stall: int = 400, **kw):
        """Iterate at most ``max_iter`` steps.

        ``max_stall`` bounds the number of consecutive check blocks without a
        decrease of ``max|du|`` before the run gives up.  The default (400 blocks)
        is fine for a quick probe, but a slowly marching recirculation is *not*
        stalled -- its change per block oscillates around a decaying mean -- so the
        verification driver passes a very large value and relies on the iteration
        budget instead.
        """
        prev_u, prev_v = self.u.copy(), self.v.copy()
        stall, best = 0, float("inf")
        state = self.state()
        change = float("inf")
        for k in range(max(1, max_iter // check)):
            state = self.iterate(check, **kw)
            change = float(max(np.abs(self.u - prev_u).max(), np.abs(self.v - prev_v).max()))
            prev_u, prev_v = self.u.copy(), self.v.copy()
            if record_every and (k % record_every == 0):
                self.history.append(dict(iteration=self.iterations,
                                         continuity=state["continuity"], change=change))
            if verbose and k % 50 == 0:
                print(f"  it={self.iterations:6d} cont={state['continuity']:.3e} "
                      f"dU={change:.3e}", flush=True)
            if state["continuity"] < tol and change < tol:
                break
            if change < best * 0.999:
                best, stall = change, 0
            else:
                stall += 1
            if stall > max_stall:
                break
        return {"iterations": self.iterations, "continuity": state["continuity"],
                "change": change, "history": self.history}