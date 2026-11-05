"""Grid-convergence analysis on an arbitrary number of refinement levels (N >= 3).

:mod:`tensorfvm.verification.richardson` implements the three-grid procedure of
Roache (1994) / Celik et al. (2008): a single triple of spacings fixes the
observed order by exact interpolation, so

* a fourth (or fifth) grid level, however expensive it was to compute, is
  simply discarded, and
* the reported observed order has **no uncertainty at all** -- with three
  points and three unknowns the fit is an interpolation, and a residual-based
  error estimate is undefined (zero degrees of freedom).

This module fits the same truncation model

    f(h) = L + C h^p            (one leading error term)

to *all* available levels by least squares in ``(L, C)`` for each candidate
``p``, and returns the order together with a linearised standard error that is
only defined when ``N > 3``.  With ``N == 3`` the fit is an interpolation and
the result is (up to solver tolerance) identical to
:func:`tensorfvm.verification.richardson.richardson_gci`, which is verified in
``tests/test_multi_grid_gci.py`` -- the new path therefore *adds* capability
without moving any already-published number.

Reported quantities are deliberately kept in the language of the three-grid
routine so that numbers stay comparable across studies of this repository:

* ``order``: least-squares order over all levels (``nan`` when the series has
  no monotone one-term decay),
* ``order_stderr``: linearised standard error of ``order`` (``nan`` for N == 3,
  where it is not identifiable -- this is an honest withholding, not a zero),
* ``extrapolated``: the fitted limit ``L``,
* ``extrapolated_triple``: the classical three-finest-grid Richardson value
  with the order used, so a reader can see how much the extra levels moved it,
* ``gci``/``band``: same convention as ``richardson_gci`` (band centred on the
  extrapolated value, ``Fs`` = 1.25 for a resolved order, 3.0 for the
  fall-back), i.e. the uncertainty is normalised by the fine-pair difference
  and ``r21**p - 1``.

The known trap of this estimator is documented and measured in
``docs/multi-grid-gci/report.md``: adding coarse levels biases ``p`` low when a
second truncation term is active on those levels, because the least-squares
objective weights every level equally.  ``multi_grid_order`` therefore also
returns the per-level residual so a user can see the misfit instead of only its
consequence.
"""
from dataclasses import dataclass
import math

import numpy as np

from .richardson import MIN_RATIO, ORDER_BOUNDS, SAFETY_FALLBACK, SAFETY_OBSERVED

#: Number of log-spaced trial orders used to bracket the minimum of the
#: least-squares objective before the golden-section refinement.
SCAN_POINTS = 220

#: Golden-section refinement iterations (40^(1/600) bracket -> ~1e-14 relative).
REFINE_ITERATIONS = 90


@dataclass(frozen=True)
class MultiGridStudy:
    """Result of an N-level (N >= 3) grid-convergence study."""

    spacings: tuple
    values: tuple
    order: float
    order_stderr: float
    order_dof: int
    order_used: float
    order_ratio: float
    convergence: str
    monotone: bool
    asymptotic: bool
    extrapolated: float
    extrapolated_triple: float
    coefficient: float
    gci: float
    safety_factor: float
    band: tuple
    rms_residual: float
    residual_relative: float
    theory_order: float
    refinement_ratio: float
    levels: int
    #: Which fall-back rule produced the band: ``'none'`` for a fitted study,
    #: ``'non-monotone'`` when the error does not decay monotonically, and
    #: ``'unfittable'`` when the order fit itself failed.  Carried on the study
    #: because :func:`~tensorfvm.verification.richardson.reference_agreement`
    #: must echo the qualifier next to any coverage verdict; without the field
    #: the multi-grid path raised ``AttributeError`` (CI red on test (3.11)).
    fallback_reason: str = "none"


def _check_levels(spacings, values):
    h = np.asarray(spacings, dtype=float)
    v = np.asarray(values, dtype=float)
    if h.shape != v.shape:
        raise ValueError("spacings and values must have the same length")
    if h.ndim != 1 or h.size < 3:
        raise ValueError("at least three grid levels are required")
    if not (np.isfinite(h).all() and np.isfinite(v).all()):
        raise ValueError("non-finite spacing or metric value")
    if not (h > 0.).all():
        raise ValueError("spacings must be positive")
    if not (np.diff(h) > 0.).all():
        raise ValueError("spacings must be strictly increasing fine -> coarse")
    ratios = h[1:]/h[:-1]
    if (ratios < MIN_RATIO).any():
        raise ValueError(f"refinement ratios must exceed {MIN_RATIO}")
    return h, v


def _linear_fit(h, v, p):
    """Least squares ``v ~ L + C h^p`` for a fixed order ``p``.

    Returns ``(L, C, residuals, sum_squared)``.  ``p`` near zero makes the two
    basis columns collinear, in which case the fit is reported as infeasible
    (infinite sum of squares) instead of returning a meaningless ``L``.
    """
    hp = np.power(h, p)
    s0 = float(hp.size)
    s1 = float(hp.sum())
    s2 = float((hp*hp).sum())
    t0 = float(v.sum())
    t1 = float((v*hp).sum())
    det = s0*s2 - s1*s1
    scale = max(abs(s0*s2), 1e-300)
    if not det > 1e-12*scale:
        return float("nan"), float("nan"), np.full(h.shape, np.nan), math.inf
    L = (s2*t0 - s1*t1)/det
    C = (s0*t1 - s1*t0)/det
    res = v - (L + C*hp)
    return float(L), float(C), res, float(res @ res)


def _minimise_order(h, v):
    """Bracketed scan + golden-section minimisation of the fit residual over p."""
    lo, hi = math.log(ORDER_BOUNDS[0]), math.log(ORDER_BOUNDS[1])
    grid = np.linspace(lo, hi, SCAN_POINTS)
    best_i, best = 0, math.inf
    for i, lp in enumerate(grid):
        _, _, _, ss = _linear_fit(h, v, math.exp(lp))
        if ss < best:
            best, best_i = ss, i
    a = grid[max(best_i-1, 0)]
    b = grid[min(best_i+1, grid.size-1)]
    if not (b > a):
        return math.exp(grid[best_i])
    phi = (math.sqrt(5.)-1.)/2.
    c, d = b - phi*(b-a), a + phi*(b-a)
    fc = _linear_fit(h, v, math.exp(c))[3]
    fd = _linear_fit(h, v, math.exp(d))[3]
    for _ in range(REFINE_ITERATIONS):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi*(b-a)
            fc = _linear_fit(h, v, math.exp(c))[3]
        else:
            a, c, fc = c, d, fd
            d = a + phi*(b-a)
            fd = _linear_fit(h, v, math.exp(d))[3]
    return math.exp(.5*(a+b))


def _order_stderr(h, v, p, L, C, ss, dof):
    """Linearised standard error of the fitted order (``nan`` when ``dof <= 0``)."""
    if dof <= 0:
        return float("nan")
    hp = np.power(h, p)
    jac = np.column_stack([np.ones_like(h), hp, C*hp*np.log(h)])
    try:
        normal = jac.T @ jac
        cov = np.linalg.inv(normal)
    except np.linalg.LinAlgError:  # pragma: no cover - singular design
        return float("nan")
    var = ss/max(dof, 1)
    return float(math.sqrt(max(cov[2, 2]*var, 0.)))


def _is_monotone(v):
    d = np.diff(v)
    if not (d != 0.).all():
        return False
    return bool((d > 0.).all() or (d < 0.).all())


def multi_grid_order(spacings, values, *, theory_order=None, order_tolerance=.2,
                     safety_factor=None):
    """Fit ``f(h) = L + C h^p`` over N >= 3 levels and report a GCI band.

    ``spacings``/``values`` are ordered fine -> coarse.  A series without a
    monotone error decay, or one whose fitted order leaves the bracket
    ``ORDER_BOUNDS``, is reported as ``convergence='inconsistent'`` and falls
    back to ``theory_order`` (first order if unknown) with the conservative
    two-grid safety factor 3.0 -- exactly the behaviour of the three-grid
    routine, so a broken ladder still cannot report a confident band.
    """
    h, v = _check_levels(spacings, values)
    n = h.size
    if safety_factor is not None and not (math.isfinite(safety_factor) and safety_factor >= 1.):
        raise ValueError("safety factor must be finite and >= 1")
    if v[0] == 0.:
        raise ValueError("relative GCI needs a non-zero fine-grid metric")
    if theory_order is not None and not (math.isfinite(theory_order) and theory_order > 0.):
        raise ValueError("theory order must be finite and positive")

    r21 = float(h[1]/h[0])
    monotone = _is_monotone(v)
    p = float("nan")
    L = C = float("nan")
    stderr = float("nan")
    ss = math.inf
    res = np.full(h.shape, np.nan)
    if monotone:
        p = _minimise_order(h, v)
        L, C, res, ss = _linear_fit(h, v, p)
    fallback = not (monotone and math.isfinite(p) and math.isfinite(L))
    if fallback:
        convergence = "inconsistent"
        p_used = float(1. if theory_order is None else theory_order)
        eps = abs((v[1]-v[0])/v[0])
        correction = (v[1]-v[0])/(r21**p_used - 1.)
        L = float(v[0]-correction)
        C = float("nan")
        p = float("nan")
        stderr = float("nan")
        res = np.full(h.shape, np.nan)
        ss = math.inf
    else:
        convergence = "monotone"
        p_used = p
        stderr = _order_stderr(h, v, p, L, C, ss, n-3)

    eps = abs((v[1]-v[0])/v[0])
    gci = (safety_factor or (SAFETY_FALLBACK if fallback else SAFETY_OBSERVED))/(r21**p_used - 1.)*eps
    half = gci*abs(L)
    triple = float(v[0]-(v[1]-v[0])/(r21**p_used - 1.))
    dof = 0 if fallback else n-3
    if theory_order is None:
        asymptotic = bool(not fallback)
        order_ratio = float("nan")
    else:
        order_ratio = float(p_used/theory_order)
        asymptotic = bool(not fallback and abs(p_used-theory_order) <= order_tolerance)
    rms = float("nan") if fallback else math.sqrt(ss/n)
    return MultiGridStudy(
        spacings=tuple(float(x) for x in h), values=tuple(float(x) for x in v),
        order=float(p), order_stderr=stderr, order_dof=int(dof),
        order_used=float(p_used), order_ratio=order_ratio,
        convergence=convergence, monotone=bool(monotone),
        asymptotic=asymptotic, extrapolated=float(L),
        extrapolated_triple=triple, coefficient=float(C),
        gci=float(gci),
        safety_factor=float(safety_factor or (SAFETY_FALLBACK if fallback else SAFETY_OBSERVED)),
        band=(float(L-half), float(L+half)),
        rms_residual=rms,
        residual_relative=float("nan") if fallback else abs(rms/L) if L else float("nan"),
        theory_order=float("nan") if theory_order is None else float(theory_order),
        refinement_ratio=r21, levels=int(n),
        fallback_reason="none" if not fallback else
        ("non-monotone" if not monotone else "unfittable"))


def per_level_residuals(study):
    """Signed residuals ``f_i - (L + C h_i^p)`` of a converged study (fine -> coarse)."""
    if study.convergence != "monotone":
        raise ValueError("no fitted model for an inconsistent study")
    h = np.asarray(study.spacings)
    v = np.asarray(study.values)
    return [float(x) for x in v - (study.extrapolated + study.coefficient*np.power(h, study.order))]