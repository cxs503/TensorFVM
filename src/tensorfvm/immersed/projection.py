"""Masked Poisson projection operator plus its fast solvers.

The projection step of a fractional-step scheme is the dimensionless 5-point
problem

    A phi = b,     (A phi)_c = sum_faces open(c, face) * (phi_c - phi_nb) + dirc_c * phi_c

which is exactly ``-h**2 * laplacian`` restricted to fluid cells.  Faces of solid
cells are *closed*: the face normal velocity is hard zero, so the correction has
a homogeneous Neumann condition there and the face contributes nothing.  The
outlet plane is closed by ``phi = 0`` outside the domain (``dirc = 1``), which
makes ``A`` SPD and pins the ``Q_out = Q_in`` constraint.  The inlet plane is
Neumann: its velocity is prescribed and must not be corrected.

Solvers
-------
``MaskedPoisson.solve`` runs geometric-multigrid V-cycles (mask coarsened by 2x2
blocking, damped-Jacobi smoothing, piecewise-constant prolongation) and polishes
with masked-Jacobi preconditioned conjugate gradients.  MG supplies the
mesh-independent part of the convergence, CG guarantees monotone convergence and
a tight final residual, which the mass ledger depends on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def _east(a: torch.Tensor) -> torch.Tensor:
    return torch.cat([a[:, 1:], a[:, -1:]], dim=1)


def _west(a: torch.Tensor) -> torch.Tensor:
    return torch.cat([a[:, :1], a[:, :-1]], dim=1)


def _north(a: torch.Tensor) -> torch.Tensor:
    return torch.cat([a[1:, :], a[-1:, :]], dim=0)


def _south(a: torch.Tensor) -> torch.Tensor:
    return torch.cat([a[:1, :], a[:-1, :]], dim=0)


@dataclass(frozen=True)
class LevelOperator:
    """Discretisation of ``A`` on one grid level."""

    mask: torch.Tensor
    aE: torch.Tensor
    aW: torch.Tensor
    aN: torch.Tensor
    aS: torch.Tensor
    dirc: torch.Tensor
    diag: torch.Tensor

    def apply(self, phi: torch.Tensor) -> torch.Tensor:
        out = self.diag * phi
        out = out - self.aE * _east(phi) - self.aW * _west(phi)
        out = out - self.aN * _north(phi) - self.aS * _south(phi)
        return torch.where(self.mask, out, torch.zeros_like(out))


def build_level(mask: torch.Tensor, *, outlet_dirichlet: bool = True) -> LevelOperator:
    """Build the masked operator for one level from a cell-centred fluid mask."""
    mask = mask.bool()
    ny, nx = mask.shape
    zero = torch.zeros_like(mask, dtype=torch.float64)
    one = torch.ones_like(zero)
    mf = mask.to(torch.float64)

    east_nb = torch.cat([mf[:, 1:], zero[:, -1:]], dim=1)
    west_nb = torch.cat([zero[:, :1], mf[:, :-1]], dim=1)
    north_nb = torch.cat([mf[1:, :], zero[-1:, :]], dim=0)
    south_nb = torch.cat([zero[:1, :], mf[:-1, :]], dim=0)

    aE = mf * east_nb
    aW = mf * west_nb
    aN = mf * north_nb
    aS = mf * south_nb

    dirc = zero.clone()
    if outlet_dirichlet and nx > 0:
        dirc[:, -1] = mf[:, -1]

    diag = aE + aW + aN + aS + dirc
    diag = torch.where(diag > 0.0, diag, one)
    return LevelOperator(mask=mask, aE=aE, aW=aW, aN=aN, aS=aS,
                         dirc=dirc, diag=diag)


def _coarsen_mask(mask: torch.Tensor) -> torch.Tensor:
    ny, nx = mask.shape
    if ny % 2 or nx % 2:
        mask = torch.nn.functional.pad(
            mask.to(torch.float64)[None, None], (0, nx % 2, 0, ny % 2)).bool()[0, 0]
        ny, nx = mask.shape
    blocks = mask.view(ny // 2, 2, nx // 2, 2)
    return blocks.any(dim=3).any(dim=1)


def _restrict(residual: torch.Tensor, coarse_shape: tuple[int, int]) -> torch.Tensor:
    """Full-weighting restriction (adjoint of the bilinear prolongation below).

    Cell-centred transfers: the coarse cell ``(I, J)`` owns the fine cells
    ``(2I+a, 2J+b)``, and the bilinear weights are ``3/4`` toward the owning
    coarse cell and ``1/4`` toward its neighbour.  Using the exact transpose pair
    keeps the V-cycle symmetric, hence usable as a CG preconditioner.
    """
    ny, nx = residual.shape
    ny_c, nx_c = coarse_shape
    out = torch.zeros(ny_c, nx_c, dtype=residual.dtype)
    for a in (0, 1):
        for b in (0, 1):
            sub = residual[a:2 * ny_c:2, b:2 * nx_c:2]
            tx = 0.75 * sub + 0.25 * (_east(sub) if a == 0 else _west(sub))
            ty = 0.75 * tx + 0.25 * (_north(tx) if b == 0 else _south(tx))
            out = out + ty
    return out


def _prolong(error: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    """Cell-centred bilinear prolongation."""
    ny, nx = shape
    ny_c, nx_c = error.shape
    out = torch.zeros(ny, nx, dtype=error.dtype)
    for a in (0, 1):
        px = 0.75 * error + 0.25 * (_west(error) if a == 0 else _east(error))
        for b in (0, 1):
            py = 0.75 * px + 0.25 * (_south(px) if b == 0 else _north(px))
            out[a:2 * ny_c:2, b:2 * nx_c:2] = py
    return out


def _smooth(op: LevelOperator, b: torch.Tensor, x: torch.Tensor, sweeps: int,
            omega: float = 0.7) -> torch.Tensor:
    for _ in range(sweeps):
        r = b - op.apply(x)
        x = torch.where(op.mask, x + omega * r / op.diag, torch.zeros_like(x))
    return x


def _coarse_solve(op: LevelOperator, b: torch.Tensor, x: torch.Tensor,
                  maxiter: int = 300, tol: float = 1e-3) -> torch.Tensor:
    """Jacobi-preconditioned CG on the coarsest level (cheap: few thousand cells)."""
    r = b - op.apply(x)
    z = torch.where(op.mask, r / op.diag, torch.zeros_like(r))
    p = z.clone()
    rz = _dot(r, z, op.mask)
    bnorm = math.sqrt(max(float(_dot(b, b, op.mask)), 1e-300))
    for _ in range(maxiter):
        ap = op.apply(p)
        denom = _dot(p, ap, op.mask)
        if float(denom) <= 0.0:
            break
        alpha = rz / denom
        x = x + alpha * p
        r = r - alpha * ap
        if math.sqrt(max(float(_dot(r, r, op.mask)), 0.0)) <= tol * bnorm:
            break
        z = torch.where(op.mask, r / op.diag, torch.zeros_like(r))
        rz_new = _dot(r, z, op.mask)
        if float(rz) == 0.0:
            break
        p = z + (rz_new / rz) * p
        rz = rz_new
    return x


def _vcycle(levels: list[LevelOperator], rhs: list[torch.Tensor], level: int,
            x: torch.Tensor, nu1: int = 2, nu2: int = 2,
            omega: float = 0.7) -> torch.Tensor:
    op = levels[level]
    x = _smooth(op, rhs[level], x, nu1, omega)
    if level == len(levels) - 1:
        return _coarse_solve(op, rhs[level], x)
    r = rhs[level] - op.apply(x)
    coarse_shape = levels[level + 1].mask.shape
    rc = _restrict(torch.where(op.mask, r, torch.zeros_like(r)), coarse_shape)
    rhs[level + 1] = torch.where(levels[level + 1].mask, rc, torch.zeros_like(rc))
    ec = _vcycle(levels, rhs, level + 1, torch.zeros_like(rc), nu1, nu2, omega)
    x = x + torch.where(op.mask, _prolong(ec, op.mask.shape), torch.zeros_like(x))
    return _smooth(op, rhs[level], x, nu2, omega)


def _dot(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (a * b * mask).sum()


@dataclass
class SolveReport:
    iterations: int = 0
    relative_residual: float = 0.0
    mg_cycles: int = 0
    polished: bool = False


class MaskedPoisson:
    """Masked Poisson solver for one fixed mask (hierarchy built once)."""

    def __init__(self, mask: torch.Tensor, *, outlet_dirichlet: bool = True,
                 max_levels: int = 8, min_size: int = 4, nu1: int = 2, nu2: int = 2,
                 omega: float = 0.7, use_mg_preconditioner: bool = False,
                 deep: bool = False, direct: bool = True):
        self.mask = mask.bool()
        self.outlet_dirichlet = outlet_dirichlet
        self.nu1, self.nu2, self.omega = nu1, nu2, omega
        self.use_mg_preconditioner = use_mg_preconditioner
        self._lu = None
        self._matrix = None
        if not deep:
            max_levels = 2
        self.levels = [build_level(self.mask, outlet_dirichlet=outlet_dirichlet)]
        while (len(self.levels) < max_levels
               and min(self.levels[-1].mask.shape) >= 2 * min_size):
            coarse = _coarsen_mask(self.levels[-1].mask)
            if not bool(coarse.any()):
                break
            self.levels.append(build_level(coarse, outlet_dirichlet=outlet_dirichlet))
            if coarse.shape == self.levels[-2].mask.shape:
                break
        self.last = SolveReport()
        if direct:
            self.factorise()

    # ---------------------------------------------------------------- helpers
    @property
    def neumann_everywhere(self) -> bool:
        return not bool(self.levels[0].dirc.any())

    def apply(self, phi: torch.Tensor) -> torch.Tensor:
        return self.levels[0].apply(phi)

    def residual(self, phi: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return b - self.apply(phi)

    def norm(self, a: torch.Tensor) -> float:
        return math.sqrt(max(float(_dot(a, a, self.levels[0].mask)), 0.0))

    # ------------------------------------------------------------------ solve
    # ------------------------------------------------------- sparse direct path
    def factorise(self) -> None:
        """Factorise ``A`` once with SuperLU; every later solve is a back-substitution.

        The mask and the grid never change during a run, so the projection matrix
        is fixed and the factorisation is paid a single time.  This gives an
        *exact* masked projection (divergence at machine precision) without a
        Krylov iteration count that grows with the grid size.
        """
        import numpy as np
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla

        mask = self.mask.numpy()
        ny, nx = mask.shape
        idx = -np.ones((ny, nx), dtype=np.int64)
        fluid = np.flatnonzero(mask.ravel())
        idx.ravel()[fluid] = np.arange(fluid.size)
        op = self.levels[0]
        aE = op.aE.numpy().ravel()
        aW = op.aW.numpy().ravel()
        aN = op.aN.numpy().ravel()
        aS = op.aS.numpy().ravel()
        diag = op.diag.numpy().ravel()
        unknown = np.arange(fluid.size)
        rows = [unknown]
        cols = [unknown]
        vals = [diag[fluid]]
        idxf = idx.ravel()
        for scale, off in ((aE, 1), (aW, -1), (aN, nx), (aS, -nx)):
            src = np.arange(ny * nx)
            tgt = src + off
            ok = (scale > 0.0)
            if off in (1, -1):
                valid = (tgt >= 0) & (tgt < ny * nx) & (((tgt // nx) == (src // nx)))
            else:
                valid = (tgt >= 0) & (tgt < ny * nx)
            ok = ok & valid
            ok &= mask.ravel()
            r = idxf[src[ok]]
            c = idxf[tgt[ok]]
            keep = (r >= 0) & (c >= 0)
            rows.append(r[keep])
            cols.append(c[keep])
            vals.append(-scale[ok][keep])
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        vals = np.concatenate(vals)
        self._matrix = sp.csr_matrix((vals, (rows, cols)), shape=(fluid.size, fluid.size))
        # MMD_AT_PLUS_A: for this SPD 5-point operator it halves the fill and the
        # back-substitution cost compared with the COLAMD default.
        self._lu = spla.splu(self._matrix.tocsc(), permc_spec="MMD_AT_PLUS_A",
                             diag_pivot_thresh=0.0)
        self._fluid_index = fluid
        self._index_grid = idx

    def _precondition(self, r: torch.Tensor) -> torch.Tensor:
        """One symmetric V-cycle used as a CG preconditioner."""
        mask = self.levels[0].mask
        rhs = [torch.where(mask, r, torch.zeros_like(r))]
        for lv in self.levels[1:]:
            rhs.append(torch.zeros_like(lv.mask, dtype=torch.float64))
        out = _vcycle(self.levels, rhs, 0, torch.zeros_like(r), self.nu1, self.nu2,
                      self.omega)
        return torch.where(mask, out, torch.zeros_like(out))

    def solve(self, b: torch.Tensor, *, tol: float = 1e-10, maxiter: int = 4000,
              x0: torch.Tensor | None = None, cycles: int = 0,
              polish: bool = True) -> torch.Tensor:
        """Solve ``A phi = b`` on the mask.

        ``cycles`` V-cycles are run first (they remove the smooth part of the
        error with a mesh-independent rate); the result then seeds a
        preconditioned CG so the final residual is guaranteed monotone and tight.
        """
        mask = self.levels[0].mask
        b = torch.where(mask, b, torch.zeros_like(b))
        x = torch.zeros_like(b) if x0 is None else torch.where(
            mask, x0, torch.zeros_like(b))
        if self._lu is not None:
            rhs = b.numpy().ravel()[self._fluid_index]
            phi = self._lu.solve(rhs)
            out = torch.zeros(b.numel(), dtype=torch.float64)
            out[self._fluid_index] = torch.from_numpy(phi.copy())
            phi_grid = out.view(b.shape)
            nrm = self.norm(b)
            self.last = SolveReport(1, self.norm(self.residual(phi_grid, b)) / nrm
                                    if nrm > 0 else 0.0, 0, True)
            return phi_grid
        bnorm = self.norm(b)
        if bnorm == 0.0:
            self.last = SolveReport(0, 0.0, 0, True)
            return x
        target = tol * bnorm
        report = SolveReport()
        for _ in range(cycles):
            if self.norm(self.residual(x, b)) <= target:
                break
            x = x + self._precondition(self.residual(x, b))
            report.mg_cycles += 1
        r = self.residual(x, b)
        if polish and self.norm(r) > target:
            z = self._precondition(r) if self.use_mg_preconditioner else torch.where(
                mask, r / self.levels[0].diag, torch.zeros_like(r))
            p = z.clone()
            rz = _dot(r, z, mask)
            for it in range(maxiter):
                if self.norm(r) <= target:
                    break
                ap = self.apply(p)
                denom = _dot(p, ap, mask)
                if float(denom) <= 0.0:
                    break
                alpha = rz / denom
                x = x + alpha * p
                r = r - alpha * ap
                z = self._precondition(r) if self.use_mg_preconditioner else torch.where(
                    mask, r / self.levels[0].diag, torch.zeros_like(r))
                rz_new = _dot(r, z, mask)
                if float(rz) == 0.0:
                    break
                p = z + (rz_new / rz) * p
                rz = rz_new
                report.iterations = it + 1
            report.polished = True
        report.relative_residual = self.norm(self.residual(x, b)) / bnorm
        self.last = report
        return x
def masked_pcg(mask: torch.Tensor, b: torch.Tensor, *, tol: float = 1e-10,
               maxiter: int = 400, outlet_dirichlet: bool = True,
               x0: torch.Tensor | None = None) -> tuple[torch.Tensor, SolveReport]:
    """Convenience wrapper: solve ``A phi = b`` on ``mask`` and return the report."""
    solver = MaskedPoisson(mask, outlet_dirichlet=outlet_dirichlet)
    phi = solver.solve(b, tol=tol, maxiter=maxiter, x0=x0)
    return phi, solver.last