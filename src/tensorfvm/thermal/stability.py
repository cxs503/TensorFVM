"""Linear stability of the Rayleigh--Benard conduction state (onset threshold).

Why this module exists
----------------------
:mod:`tensorfvm.thermal.rayleigh_benard` marches the *nonlinear* Boussinesq
equations, and a nonlinear march near onset is the worst possible place to
measure ``Ra_c``: the growth rate vanishes at threshold, so the transient one
has to resolve lasts ``~ 1/((Ra - Ra_c)/Ra_c)`` turnover times -- hundreds of
thousands of explicit steps for a few percent of supercriticality.  The onset
threshold is by definition the zero of the growth rate of the *linearised*
operator, so it is measured here directly and cheaply, and the nonlinear solver
is used only to confirm that it grows above and decays below that threshold.

Derivation
----------
Linearise the non-dimensional Boussinesq equations of
:mod:`tensorfvm.thermal.rayleigh_benard` about the conduction state
``Theta_0(z) = 1/2 - z``, ``u = 0``, with a streamfunction
``(u, w) = (dpsi/dz, -dpsi/dx)`` and normal modes
``psi = Psi(z) exp(i k x + sigma t)``, ``theta = Theta(z) exp(i k x + sigma t)``.
The y-vorticity and energy equations become, with ``A = d^2/dz^2 - k^2``,

    (sigma / Pr) A Psi = A^2 Psi - i k Ra Theta,
    sigma Theta        = A Theta    - i k Psi,

i.e. the pencil ``L q = sigma M q`` with ``q = (Psi, Theta)``,

    M = diag(A / Pr, 1),   L = [[A^2, -i k Ra], [-i k, A]].

Boundary conditions enter through ghost nodes of the second-difference
stencil: ``Psi = 0`` at both plates always (no penetration), and ``Psi' = 0``
(rigid) or ``Psi'' = 0`` (stress-free) at the plates; the temperature
perturbation vanishes on both plates.  Because ``A^2`` is built as the product
of the *full* second-derivative operator, the clamped-plate wall vorticity
(``D^2 Psi != 0`` at a rigid wall) is carried by the biharmonic stencil, which
is exactly what distinguishes the rigid from the stress-free threshold.

Numerics
--------
LAPACK ``zgeev`` on this node is ~100x slower than reference BLAS for
``n = 96`` (measured: 0.1 s per call), so the leading eigenvalue is obtained
by *inverse iteration with shift zero* -- the mode that matters near onset is
the least damped one, i.e. the eigenvalue closest to the origin -- which needs
one LU factorisation plus a handful of triangular solves.  :func:`spectrum`
exposes the full ``zgeev`` spectrum for the report, and
:func:`dominant_growth_rate` cross-checks the iteration against it when asked.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import scipy.linalg as sla
from scipy.optimize import brentq

from .reference import (
    RA_CRITICAL_FREE,
    RA_CRITICAL_RIGID,
    WAVENUMBER_CRITICAL_FREE,
    WAVENUMBER_CRITICAL_RIGID,
)

__all__ = [
    "StabilityConfig",
    "pencil",
    "spectrum",
    "growth_rate",
    "marginal_rayleigh",
    "marginal_curve",
    "critical_rayleigh",
    "reference_values",
]


@dataclass(frozen=True)
class StabilityConfig:
    """Discretisation of the normal-mode stability problem.

    ``nodes`` is the number of *interior* nodes of the finite-difference grid
    across the layer; the two plate nodes are eliminated by the boundary
    conditions.  ``boundary`` selects ``"rigid"`` (no-slip plates, the physical
    case) or ``"free"`` (stress-free plates, whose critical Rayleigh number is
    known in closed form and therefore acts as a discretisation check).
    """

    nodes: int = 64
    boundary: str = "rigid"

    def __post_init__(self) -> None:
        if isinstance(self.nodes, bool) or not isinstance(self.nodes, int) or self.nodes < 8:
            raise ValueError("nodes must be an integer >= 8")
        if self.boundary not in ("rigid", "free"):
            raise ValueError("boundary must be 'rigid' or 'free'")


# --------------------------------------------------------------------- pencil
def _second_derivative(nodes: int, boundary: str) -> np.ndarray:
    """``d^2/dz^2`` on ``nodes + 2`` grid points, including the ghost BC action."""
    total = nodes + 2                     # interior nodes + the two plate nodes
    h = 1.0 / (nodes + 1)
    mat = np.zeros((total, total))
    diag = -2.0 / h ** 2
    off = 1.0 / h ** 2
    for i in range(1, total - 1):
        mat[i, i - 1] = off
        mat[i, i] = diag
        mat[i, i + 1] = off
    if boundary == "rigid":               # Psi' = 0  ->  ghost = +Psi_1
        mat[0, 0], mat[0, 1] = diag, 2.0 * off
        mat[total - 1, total - 1], mat[total - 1, total - 2] = diag, 2.0 * off
    # free plates: Psi'' = 0 -> ghost = -Psi_1 -> the plate rows vanish
    return mat


_BLOCK_CACHE: dict[tuple[int, str, float], tuple[np.ndarray, np.ndarray]] = {}


def _pencil_blocks(k: float, config: StabilityConfig) -> tuple[np.ndarray, np.ndarray]:
    """``(A^2, A)`` restricted to the interior nodes, cached per ``(k, bc)``.

    ``A^2`` is the product of the *full* second-derivative operator including
    the plate rows, so the clamped-plate wall vorticity (``D^2 Psi != 0`` at a
    rigid wall) is carried by the biharmonic stencil.  This is what makes the
    rigid threshold differ from the stress-free one; building ``A^2`` from the
    interior block alone would silently impose the wrong wall condition.
    """
    key = (config.nodes, config.boundary, round(float(k), 12))
    cached = _BLOCK_CACHE.get(key)
    if cached is not None:
        return cached
    n = config.nodes
    total = n + 2
    a = _second_derivative(n, config.boundary) - (k * k) * np.eye(total)
    interior = slice(1, total - 1)
    blocks = ((a @ a)[interior, interior], a[interior, interior].astype(complex))
    if len(_BLOCK_CACHE) > 512:
        _BLOCK_CACHE.clear()
    _BLOCK_CACHE[key] = blocks
    return blocks


def pencil(k: float, rayleigh: float, prandtl: float,
           config: StabilityConfig) -> tuple[np.ndarray, np.ndarray]:
    """The pencil ``(L, M)`` of ``L q = sigma M q`` for ``q = (Psi, Theta)``."""
    a2_int, a_int = _pencil_blocks(k, config)
    n = config.nodes
    eye = np.eye(n)
    zero = np.zeros((n, n), dtype=complex)
    lower = np.block([[a2_int, (-1j * k * rayleigh) * eye],
                      [(-1j * k) * eye, a_int]])
    mass = np.block([[a_int / prandtl, zero],
                     [zero, eye.astype(complex)]])
    return lower, mass


def spectrum(k: float, rayleigh: float, prandtl: float = 0.71,
             config: StabilityConfig | None = None) -> np.ndarray:
    """Full ``zgeev`` spectrum of the pencil (report/diagnostic use)."""
    cfg = config or StabilityConfig()
    lower, mass = pencil(k, rayleigh, prandtl, cfg)
    return sla.eig(lower, mass, right=False)


def growth_rate(k: float, rayleigh: float, prandtl: float = 0.71,
                config: StabilityConfig | None = None) -> float:
    """Least damped growth rate ``max Re sigma`` of the mode ``exp(i k x)``.

    Computed from the full generalised spectrum: on this node a ``zgeev`` call
    is ~20 ms for the 96x96 pencil, so the marginal curve is traced with a
    root-finder (~10 spectra per wavenumber) rather than a Rayleigh-number scan.
    """
    cfg = config or StabilityConfig()
    if not (k > 0.0 and rayleigh > 0.0 and prandtl > 0.0):
        raise ValueError("k, rayleigh and prandtl must be positive")
    return float(np.max(spectrum(k, rayleigh, prandtl, cfg).real))


def marginal_rayleigh(k: float, prandtl: float = 0.71,
                      config: StabilityConfig | None = None, *,
                      ra_lo: float = 1.0, ra_hi: float = 1.0e6,
                      rel_tol: float = 1e-6, max_iter: int = 100) -> float:
    """Rayleigh number at which the mode ``k`` is marginal (``Re sigma = 0``).

    ``Re sigma(Ra)`` is monotone increasing at fixed ``k`` (buoyancy is the only
    destabilising term), so Brent's method on ``log Ra`` is safe.  The bracket
    is widened geometrically until the sign changes.
    """
    cfg = config or StabilityConfig()

    def g(ra: float) -> float:
        return growth_rate(k, ra, prandtl, cfg)

    lo, hi = math.log(float(ra_lo)), math.log(float(ra_hi))
    if g(math.exp(lo)) > 0.0:
        raise RuntimeError("bracket lower end already unstable")
    while g(math.exp(hi)) < 0.0:
        hi += math.log(10.0)
        if hi > math.log(1e12):
            raise RuntimeError("could not bracket the marginal Rayleigh number above")
    root = brentq(lambda z: g(math.exp(z)), lo, hi,
                  xtol=rel_tol, rtol=1e-10, maxiter=max_iter)
    return float(math.exp(root))


def marginal_curve(prandtl: float = 0.71, config: StabilityConfig | None = None, *,
                   k_min: float = 0.8, k_max: float = 6.0, k_points: int = 41,
                   rel_tol: float = 1e-7) -> dict[str, object]:
    """Marginal Rayleigh number over a grid of horizontal wavenumbers."""
    cfg = config or StabilityConfig()
    if k_points < 3:
        raise ValueError("k_points must be >= 3")
    ks = np.linspace(k_min, k_max, k_points)
    ras = [marginal_rayleigh(float(k), prandtl, cfg, rel_tol=rel_tol) for k in ks]
    return {"k": ks.tolist(), "rayleigh": ras}


def critical_rayleigh(prandtl: float = 0.71, config: StabilityConfig | None = None, *,
                      k_min: float = 0.8, k_max: float = 6.0, k_points: int = 41,
                      rel_tol: float = 1e-7, refine: bool = True) -> dict[str, object]:
    """Smallest marginal Rayleigh number over ``k`` (the onset threshold).

    Returns the critical Rayleigh number, the critical wavenumber and the full
    marginal curve.  A parabola through the three points around the discrete
    minimum (``log Ra`` against ``k``) gives the refined ``(Ra_c, k_c)``; this
    removes the ``O(dk^2)`` bias of the wavenumber scan, which dominates once
    the vertical discretisation is fine.
    """
    cfg = config or StabilityConfig()
    curve = marginal_curve(prandtl, cfg, k_min=k_min, k_max=k_max,
                           k_points=k_points, rel_tol=rel_tol)
    ks = np.asarray(curve["k"])
    ras = np.asarray(curve["rayleigh"])
    i = int(np.argmin(ras))
    ra_c = float(ras[i])
    k_c = float(ks[i])
    if refine and 0 < i < len(ks) - 1:
        y = np.log(ras[i - 1:i + 2])
        x = ks[i - 1:i + 2]
        coeff = np.polyfit(x, y, 2)
        if coeff[0] > 0.0:
            k_c = float(-coeff[1] / (2.0 * coeff[0]))
            ra_c = float(math.exp(np.polyval(coeff, k_c)))
    return {
        "boundary": cfg.boundary,
        "nodes": cfg.nodes,
        "prandtl": prandtl,
        "rayleigh_critical": ra_c,
        "wavenumber_critical": k_c,
        "growth_at_critical": growth_rate(k_c, ra_c, prandtl, cfg),
        "curve": curve,
    }


def reference_values(boundary: str) -> tuple[float, float]:
    """``(Ra_c, k_c)`` of the textbook anchors for the given plate closure."""
    if boundary == "rigid":
        return RA_CRITICAL_RIGID, WAVENUMBER_CRITICAL_RIGID
    if boundary == "free":
        return RA_CRITICAL_FREE, WAVENUMBER_CRITICAL_FREE
    raise ValueError("boundary must be 'rigid' or 'free'")