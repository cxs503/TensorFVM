"""Backend registry and fallback chain for the linear backends.

Registration is explicit (the five adapters in
:mod:`tensorfvm.core.linear.backends` call :func:`register_backend` at import
time).  The fallback chain instantiates the first *available* backend in the
given priority order -- availability is the backend's own call (e.g. the AMG
block backend reports unavailable when the optional ``pyamg`` extra is
missing), so a chain degrades honestly instead of failing at import time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ...registry import capability
from .protocol import LINEAR_KINDS, LinearSolver

__all__ = [
    "BackendEntry",
    "register_backend",
    "get_backend",
    "list_backends",
    "available_backends",
    "fallback_chain",
    "solve_with_fallback",
]


@dataclass(frozen=True)
class BackendEntry:
    """One registered backend: identity, role and construction factory."""

    name: str
    kind: str
    factory: Callable[..., LinearSolver]
    doc: str = ""


_BACKENDS: dict[str, BackendEntry] = {}
_CACHE: dict[str, LinearSolver] = {}


def register_backend(name: str, *, kind: str, factory: Callable[..., LinearSolver],
                    doc: str = "") -> None:
    """Register a linear backend under ``name`` (idempotent per process)."""
    if name in _BACKENDS:
        raise ValueError(f"linear backend {name!r} is already registered")
    if kind not in LINEAR_KINDS:
        raise ValueError(f"kind {kind!r} not one of {LINEAR_KINDS}")
    _BACKENDS[name] = BackendEntry(name, kind, factory, doc)


def get_backend(name: str, *args: Any, **kwargs: Any) -> LinearSolver:
    """Instantiate backend ``name`` (zero-argument builds are cached)."""
    entry = _BACKENDS.get(name)
    if entry is None:
        raise KeyError(f"unknown linear backend {name!r}; known: {sorted(_BACKENDS)}")
    if not args and not kwargs and name in _CACHE:
        return _CACHE[name]
    backend = entry.factory(*args, **kwargs)
    if not hasattr(backend, "solve") or not hasattr(backend, "available"):
        raise TypeError(f"factory for {name!r} did not produce a LinearSolver")
    if not args and not kwargs:
        _CACHE[name] = backend
    return backend


def list_backends() -> list[str]:
    """Names of all registered backends, sorted."""
    return sorted(_BACKENDS)


def available_backends(*args: Any, **kwargs: Any) -> list[str]:
    """Names of the backends that report ``available()`` in this environment."""
    out = []
    for name in sorted(_BACKENDS):
        try:
            if get_backend(name, *args, **kwargs).available():
                out.append(name)
        except Exception:
            continue
    return out


@capability('fvm.core.linear_registry',
            kind='linear_solver',
            maturity='developing',
            test_path='tests/test_core_linear.py',
            wired=True,
            tested=True,
            verified=False,
            evidence={
                'grade': 'MECHANICS',
                'note': ('protocol + registry + fallback chain over the five '
                         'existing backends (sparse_simple, external_linear, '
                         'native_block_solver3d, overset_poisson, '
                         'native_linear_diagnostics); adapters delegate only'),
                'docs': 1,
            })
def fallback_chain(names, *args: Any, **kwargs: Any) -> LinearSolver:
    """Instantiate the first *available* backend in priority order.

    Raises ``KeyError`` for an unknown name and ``LookupError`` when no
    backend in the chain is available.
    """
    tried: list[str] = []
    for name in names:
        if name not in _BACKENDS:
            raise KeyError(f"unknown linear backend {name!r}; known: {sorted(_BACKENDS)}")
        try:
            backend = get_backend(name, *args, **kwargs)
        except Exception:
            # A backend that cannot be constructed in this context (e.g. a
            # solver-bound backend without its solver) is skipped, not fatal.
            tried.append(name)
            continue
        if backend.available():
            return backend
        tried.append(name)
    raise LookupError(f"no available linear backend in {list(names)} (tried {tried})")


def solve_with_fallback(backend: LinearSolver, *args: Any, **kwargs: Any) -> tuple[LinearSolver, Any]:
    """Run ``solve`` on a backend selected by :func:`fallback_chain`.

    Returns ``(backend, result)`` so the caller can report which backend ran.
    """
    if not backend.available():
        raise LookupError(f"backend {backend.name!r} is not available in this environment")
    return backend, backend.solve(*args, **kwargs)
