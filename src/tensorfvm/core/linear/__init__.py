"""Linear-solver protocol, backend registry and fallback chain."""

from . import backends  # noqa: F401  (registers the five backends)
from .backends import (
    ExternalLinearBackend,
    LinearDiagnosticsBackend,
    NativeBlockSolver3DBackend,
    OversetPoissonBackend,
    SparseSimpleBackend,
)
from .protocol import LINEAR_KINDS, LinearSolver
from .registry import (
    BackendEntry,
    available_backends,
    fallback_chain,
    get_backend,
    list_backends,
    register_backend,
    solve_with_fallback,
)

__all__ = [
    "LinearSolver",
    "LINEAR_KINDS",
    "BackendEntry",
    "register_backend",
    "get_backend",
    "list_backends",
    "available_backends",
    "fallback_chain",
    "solve_with_fallback",
    "SparseSimpleBackend",
    "ExternalLinearBackend",
    "NativeBlockSolver3DBackend",
    "OversetPoissonBackend",
    "LinearDiagnosticsBackend",
]
