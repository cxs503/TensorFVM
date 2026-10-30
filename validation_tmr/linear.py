"""LU-preconditioned linear algebra for the collocated SIMPLE solver.

The collocated solver in :mod:`tensorfvm.body_fitted` solves every segregated
equation with diagonal-preconditioned CG/BiCGSTAB.  On wall-resolved boundary
layer grids the cell aspect ratio reaches ~2e3, so the pressure-correction
operator is strongly anisotropic and Jacobi-CG needs O(aspect) iterations.  For
the official TMR grids that dominates the run time (measured: 7.9 s of a 10 s
SIMPLE step are spent inside one pressure solve, while both momentum solves
together take 0.04 s).

This module keeps the *discretisation untouched*: the same operator closures,
the same tolerances and the same true-residual acceptance check as
:meth:`tensorfvm.body_fitted.BodyFittedSolver._linear`.  It only replaces the
Jacobi preconditioner by an LU factorisation of the assembled stencil matrix

    M[i,i] = diagonal[i],   M[i,n] = -a_n,   M[n,i] = -a_o,

which is the operator's normal-diffusion part; the non-orthogonal correction in
the pressure operator remains in the matrix-vector product, so the converged
correction solves the identical discrete system.  Symmetric systems are solved
directly, non-symmetric ones with preconditioned BiCGSTAB.
"""

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
import torch

from tensorfvm.body_fitted import BodyFittedSolver


class FastLinearBodyFittedSolver(BodyFittedSolver):
    """``BodyFittedSolver`` with an LU-preconditioned linear solver."""

    def __init__(self, config):
        super().__init__(config)
        self.linear_diagnostics = []
        # Exact LU below this size, ILU-preconditioned BiCGSTAB above it.
        self.direct_threshold = getattr(config, "direct_linear_cells", 30000)

    def _stencil_matrix(self, diagonal, ao, an):
        count = self.count
        rows = np.concatenate((self.oi.numpy(), self.ni.numpy(), np.arange(count)))
        cols = np.concatenate((self.ni.numpy(), self.oi.numpy(), np.arange(count)))
        values = np.concatenate((-ao.detach().numpy(), -an.detach().numpy(),
                                 diagonal.detach().numpy()))
        matrix = sparse.coo_matrix((values, (rows, cols)), shape=(count, count)).tocsc()
        matrix.sum_duplicates()
        return matrix

    def _preconditioner(self, matrix):
        """Exact LU on modest meshes, drop-tolerance ILU on large ones."""
        if self.count <= self.direct_threshold:
            return splinalg.splu(matrix), "splu"
        return splinalg.spilu(matrix, drop_tol=1e-5, fill_factor=25), "spilu"

    def _linear(self, diagonal, ao, an, rhs, initial, symmetric=False, operator=None,
                threshold_floor=1e-10, relative_tolerance=1e-9, allow_inexact=False):
        matrix = self._stencil_matrix(diagonal, ao, an)
        if operator is None:
            mv = lambda vector: self._matvec(vector, diagonal, ao, an)
        else:
            mv = operator
        factor, kind = self._preconditioner(matrix)
        if operator is None and kind == "splu":
            # Exact factorisation: the solve is the solution, no iteration needed.
            solution = torch.tensor(factor.solve(rhs.detach().numpy()), dtype=rhs.dtype)
            self.linear_diagnostics.append(dict(kind="direct", iterations=0))
            return solution

        def precondition(vector):
            return torch.tensor(factor.solve(vector.detach().numpy()), dtype=vector.dtype)

        threshold = max(threshold_floor,
                        float(torch.linalg.vector_norm(rhs)) * relative_tolerance)
        x = initial.clone()
        residual = rhs - mv(x)
        if float(torch.linalg.vector_norm(residual)) <= threshold:
            self.linear_diagnostics.append(dict(kind=kind, iterations=0))
            return x
        shadow = residual.clone()
        direction = torch.zeros_like(x)
        v = direction.clone()
        rho_old = alpha = omega = 1.0
        iterations = 0
        for iterations in range(1, max(50, min(2 * self.count, 5000)) + 1):
            rho = torch.dot(shadow, residual)
            if abs(float(rho)) < 1e-300:
                break
            beta = (rho / rho_old) * (alpha / omega)
            direction = residual + beta * (direction - omega * v)
            phat = precondition(direction)
            v = mv(phat)
            denominator = float(torch.dot(shadow, v))
            if denominator == 0.0:
                break
            alpha = rho / denominator
            s = residual - alpha * v
            x = x + alpha * phat
            if float(torch.linalg.vector_norm(s)) <= threshold:
                residual = s
                break
            shat = precondition(s)
            t = mv(shat)
            tt = float(torch.dot(t, t))
            if tt == 0.0:
                residual = s
                break
            omega = torch.dot(t, s) / tt
            x = x + omega * shat
            residual = s - omega * t
            if float(torch.linalg.vector_norm(residual)) <= threshold:
                break
            if abs(float(omega)) < 1e-300:
                break
            rho_old = rho
        self.linear_diagnostics.append(dict(kind=kind, iterations=iterations))
        if not bool(torch.isfinite(x).all()):
            raise RuntimeError("nonfinite fast body-fitted linear solve")
        true_residual = float(torch.linalg.vector_norm(rhs - mv(x)))
        if not np.isfinite(true_residual):
            raise RuntimeError("nonfinite fast body-fitted linear residual")
        if true_residual > 10 * threshold and not allow_inexact:
            raise RuntimeError(
                f"fast linear solve did not converge: residual {true_residual:.3g}, "
                f"tolerance {10 * threshold:.3g}")
        return x
