"""Core linear registry: protocol, five backends, fallback chain."""
import importlib.util

import numpy as np
import pytest
import torch
from scipy import sparse

import tensorfvm.curved_channel  # noqa: F401  (registers 'curved-channel')

from tensorfvm.core.linear import (
    LinearSolver,
    available_backends,
    fallback_chain,
    get_backend,
    list_backends,
    solve_with_fallback,
)

FIVE = {"sparse_simple", "external_linear", "native_block_solver3d",
        "overset_poisson", "native_linear_diagnostics"}


def test_all_five_backends_registered():
    assert set(list_backends()) == FIVE


def test_standalone_backends_conform_to_protocol():
    for name in ("overset_poisson", "native_linear_diagnostics"):
        assert isinstance(get_backend(name), LinearSolver)


def test_solver_bound_backends_conform_to_protocol():
    from tensorfvm.solver import SolverConfig
    from tensorfvm.sparse_simple import SparseBodyFittedSolver
    solver = SparseBodyFittedSolver(SolverConfig(
        mesh_type="curved-channel", nx=16, ny=8, length=2.0, height=1.0,
        cylinder_radius=None, inlet_profile="parabolic", reynolds=1.0,
        tolerance=1e-6, max_iterations=1))
    for name in ("sparse_simple", "external_linear"):
        assert isinstance(get_backend(name, solver), LinearSolver)


def test_sparse_simple_backend_solves_assembled_system():
    from tensorfvm.solver import SolverConfig
    from tensorfvm.sparse_simple import SparseBodyFittedSolver
    solver = SparseBodyFittedSolver(SolverConfig(
        mesh_type="curved-channel", nx=16, ny=8, length=2.0, height=1.0,
        cylinder_radius=None, inlet_profile="parabolic", reynolds=1.0,
        tolerance=1e-6, max_iterations=1))
    n = solver.count
    rng = np.random.default_rng(7)
    diagonal = rng.random(n) + 1.0
    ao = rng.random(len(solver.oi)) * 0.1
    an = rng.random(len(solver.ni)) * 0.1
    A = solver._matrix(torch.as_tensor(diagonal), torch.as_tensor(ao),
                       torch.as_tensor(an))
    b = torch.as_tensor(rng.random(n))
    backend = get_backend("sparse_simple", solver)
    x = backend.solve(A, b)          # backend returns a numpy array
    b_np = b.detach().cpu().numpy()
    assert np.linalg.norm(A.toarray() @ x - b_np) / np.linalg.norm(b_np) < 1e-8


def test_sparse_simple_backend_rejects_foreign_pattern():
    from tensorfvm.solver import SolverConfig
    from tensorfvm.sparse_simple import SparseBodyFittedSolver
    solver = SparseBodyFittedSolver(SolverConfig(
        mesh_type="curved-channel", nx=16, ny=8, length=2.0, height=1.0,
        cylinder_radius=None, inlet_profile="parabolic", reynolds=1.0,
        tolerance=1e-6, max_iterations=1))
    n = solver.count
    A = np.eye(n)
    A[0, n - 1] = 1.0  # off-diagonal at a non-face position
    with pytest.raises(ValueError, match="assembly pattern"):
        backend = get_backend("sparse_simple", solver)
        backend.solve(A, np.ones(n))


def test_diagnostics_backend_delegates():
    A = sparse.csr_matrix(np.array([[4.0, 1.0], [1.0, 3.0]]))
    b = np.array([1.0, 2.0])
    x = np.linalg.solve(A.toarray(), b)
    result = get_backend("native_linear_diagnostics").solve(A, b, x)
    assert result["relative_rhs_residual"] < 1e-12
    assert result["finite_solution"]


def test_fallback_chain_picks_first_available():
    # without a solver instance the AMG backend cannot be constructed, so the
    # chain degrades to the standalone overset driver
    backend = fallback_chain(["native_block_solver3d", "overset_poisson"])
    assert backend.name == "overset_poisson"


def test_fallback_chain_unknown_name_raises():
    with pytest.raises(KeyError):
        fallback_chain(["no_such_backend"])


def test_fallback_chain_exhausted_raises():
    if importlib.util.find_spec("pyamg") is None:
        with pytest.raises(LookupError):
            fallback_chain(["native_block_solver3d"])


def test_solve_with_fallback_returns_backend_and_result():
    A = sparse.csr_matrix(np.array([[4.0, 1.0], [1.0, 3.0]]))
    b = np.array([1.0, 2.0])
    x = np.linalg.solve(A.toarray(), b)
    backend = fallback_chain(["native_linear_diagnostics"])
    backend, result = solve_with_fallback(backend, A, b, x)
    assert backend.name == "native_linear_diagnostics"
    assert result["relative_rhs_residual"] < 1e-12


def test_available_backends_reports_environment():
    names = available_backends()
    assert "overset_poisson" in names
    assert "native_linear_diagnostics" in names
    if importlib.util.find_spec("pyamg") is None:
        assert "native_block_solver3d" not in names
