"""Adapters for the five existing linear backends (delegation only).

* ``sparse_simple``            -- ``SparseBodyFittedSolver._linear`` (restart
  GMRES with line + coarse-grid preconditioning, scipy CSR on the torch device)
* ``external_linear``          -- ``ExternalFlowSolver._linear`` (scipy
  SuperLU direct on CPU, GMRES on CUDA)
* ``native_block_solver3d``    -- ``NativeHybridBlockSolver3D.solve`` (AMG
  block-preconditioned GMRES; needs the optional ``tensorfvm[amg]`` extra)
* ``overset_poisson``          -- ``solve_overset_poisson`` (standalone FV
  Poisson driver on an overset connectivity)
* ``native_linear_diagnostics`` -- ``linear_system_diagnostics`` (backward
  error of a supplied solution of the original system)

The solver-embedded backends expose ``solve(A, b)``: ``A`` is decomposed into
the ``(diagonal, ao, an)`` pattern the solver's own ``_matrix`` assembles, the
round-trip is verified, and the actual solve is the solver's existing code.
"""

from __future__ import annotations

import numpy as np
import torch

from .registry import register_backend

__all__ = [
    "SparseSimpleBackend",
    "ExternalLinearBackend",
    "NativeBlockSolver3DBackend",
    "OversetPoissonBackend",
    "LinearDiagnosticsBackend",
]


class _SolverLinearBackend:
    """Adapter for a solver-embedded backend (``_linear`` entry point)."""

    name = "solver_linear"
    kind = "iterative"
    doc = ""

    def __init__(self, solver):
        self.solver = solver

    def available(self) -> bool:
        return hasattr(self.solver, "_linear")

    def solve(self, A, b, *, initial=None) -> np.ndarray:
        """Solve ``A x = b`` with the solver's own linear machinery."""
        from scipy import sparse

        s = self.solver
        A = sparse.csr_matrix(A, dtype=float)
        b = np.asarray(b, dtype=float).reshape(-1)
        n = int(s.count)
        if A.shape != (n, n) or b.shape != (n,):
            raise ValueError(f"expected a ({n}, {n}) matrix and a ({n},) rhs")
        oi = s.oi.detach().cpu().numpy()
        ni = s.ni.detach().cpu().numpy()
        diagonal = np.array(A.diagonal())
        ao = -A[oi, ni].toarray().ravel()
        an = -A[ni, oi].toarray().ravel()
        opts = dict(device=s.p.device, dtype=s.p.dtype)
        diagonal_t = torch.as_tensor(diagonal, **opts)
        ao_t = torch.as_tensor(ao, **opts)
        an_t = torch.as_tensor(an, **opts)
        rebuilt = s._matrix(diagonal_t, ao_t, an_t)
        diff = A - rebuilt
        scale = max(1.0, float(abs(A).max()))
        if diff.nnz and float(abs(diff).max()) > 1e-12 * scale:
            raise ValueError(
                "A does not match the backend's (diagonal, ao, an) assembly pattern")
        rhs = torch.as_tensor(b, **opts)
        if initial is None:
            x0 = torch.zeros(n, **opts)
        else:
            x0 = torch.as_tensor(np.asarray(initial, dtype=float), **opts)
        x = s._linear(diagonal_t, ao_t, an_t, rhs, x0)
        return x.detach().cpu().numpy()


class SparseSimpleBackend(_SolverLinearBackend):
    """``fvm.linear.sparse_body_fitted``: GMRES over the assembled CSR system."""

    name = "sparse_simple"
    kind = "iterative"
    doc = ("Restarted left-preconditioned GMRES with line + coarse-grid "
           "preconditioning; scipy CSR assembly, torch CPU/CUDA solve.")


class ExternalLinearBackend(_SolverLinearBackend):
    """``fvm.linear.external``: SuperLU direct on CPU, GMRES on CUDA."""

    name = "external_linear"
    kind = "direct"
    doc = ("scipy SuperLU direct solve on CPU (the solver's GMRES path is "
           "retained for CUDA rhs).")


class NativeBlockSolver3DBackend:
    """``fvm.linear.block_amg3d``: AMG block-preconditioned GMRES (pyamg)."""

    name = "native_block_solver3d"
    kind = "iterative"
    doc = ("Component AMG cycles + pressure mass approximation preconditioning "
           "for the native hybrid 3-D velocity/pressure system.")

    def __init__(self, solver):
        self._solver = solver
        self._backend = None

    def available(self) -> bool:
        try:
            import pyamg  # noqa: F401
        except ImportError:
            return False
        return True

    def solve(self, matrix, rhs, *, allow_inexact=False):
        if self._backend is None:
            from ...native_block_solver3d import NativeHybridBlockSolver3D
            self._backend = NativeHybridBlockSolver3D(self._solver)
        return self._backend.solve(matrix, rhs, allow_inexact=allow_inexact)


class OversetPoissonBackend:
    """``fvm.linear.overset_poisson``: standalone overset FV Poisson driver."""

    name = "overset_poisson"
    kind = "iterative"
    doc = ("Solves the actual FV equations with simultaneous fringe "
           "interpolation rows on an overset connectivity.")

    def available(self) -> bool:
        return True

    def solve(self, connectivity, *, radius=0.3):
        from ...overset_poisson import solve_overset_poisson
        return solve_overset_poisson(connectivity, radius=radius)


class LinearDiagnosticsBackend:
    """``fvm.linear.diagnostics``: original-system backward-error analysis."""

    name = "native_linear_diagnostics"
    kind = "diagnostics"
    doc = ("Normwise and componentwise backward error of a supplied solution; "
           "never alters state or tolerances.")

    def available(self) -> bool:
        return True

    def solve(self, matrix, rhs, solution, *, estimate_condition=False) -> dict:
        from ...native_linear_diagnostics import linear_system_diagnostics
        return linear_system_diagnostics(matrix, rhs, solution,
                                         estimate_condition=estimate_condition)


register_backend("sparse_simple", kind="iterative", factory=SparseSimpleBackend,
                doc=SparseSimpleBackend.doc)
register_backend("external_linear", kind="direct", factory=ExternalLinearBackend,
                doc=ExternalLinearBackend.doc)
register_backend("native_block_solver3d", kind="iterative",
                factory=NativeBlockSolver3DBackend, doc=NativeBlockSolver3DBackend.doc)
register_backend("overset_poisson", kind="iterative", factory=OversetPoissonBackend,
                doc=OversetPoissonBackend.doc)
register_backend("native_linear_diagnostics", kind="diagnostics",
                 factory=LinearDiagnosticsBackend, doc=LinearDiagnosticsBackend.doc)
