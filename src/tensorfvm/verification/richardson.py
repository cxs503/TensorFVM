"""Generalised Richardson extrapolation and Roache grid-convergence index (GCI).

Every ``observed order`` claim in this repository used to come from
:func:`tensorfvm.verification.core.observed_orders`, which only divides two
differences of a metric and therefore (a) requires a uniform refinement ratio,
(b) cannot tell a monotone refinement series from an inconsistent one, and
(c) never produced an uncertainty band around the extrapolated value.  This
module is the reusable implementation of the full three-grid procedure
(Roache 1994, *Verification and Validation in Computational Science and
Engineering*; Celik et al. 2008, *J. Fluids Eng.* 130:078001):

``f(h) = f_exact + C h^p + O(h^q),  q > p``

* :func:`observed_order` solves for ``p`` from three grids with **non-uniform**
  refinement ratios,
* :func:`richardson_gci` returns the extrapolated value, the relative GCI,
  an asymptotic-range verdict against a theory order, and a fall-back with the
  two-grid safety factor when the three-grid sequence is inconsistent,
* :func:`reference_agreement` compares the extrapolated value with an
  independent reference and reports whether the GCI band covers it.

The functions are pure NumPy and deliberately do not touch a solver, so the
verification tool itself can be verified against synthetic data of known order
(``p``, ``f_exact``) before it is used to qualify a discretisation.
"""
from dataclasses import asdict, dataclass
import math

import numpy as np

#: Roache safety factors: 3-grid studies that use the observed order are
#: reported with 1.25, 2-grid (or theory-order fall-back) studies with 3.0.
SAFETY_OBSERVED = 1.25
SAFETY_FALLBACK = 3.0

#: Minimum refinement ratio for a refinement series to carry order information.
MIN_RATIO = 1.01

#: Bracket used when solving for the observed order; h^p is monotone in p.
ORDER_BOUNDS = (1e-8, 40.0)


@dataclass(frozen=True)
class RefinementStudy:
    """Result of a three-grid Richardson/GCI analysis (dict-serialisable)."""

    spacings: tuple
    values: tuple
    order: float
    order_used: float
    order_ratio: float
    convergence: str
    asymptotic: bool
    extrapolated: float
    gci: float
    safety_factor: float
    band: tuple
    theory_order: float
    refinement_ratio: float

    def as_dict(self):
        return asdict(self)


def _check(spacings, values):
    h = np.asarray(spacings, dtype=float)
    v = np.asarray(values, dtype=float)
    if h.shape != (3,) or v.shape != (3,):
        raise ValueError("exactly three grids (fine, medium, coarse) are required")
    if not (np.isfinite(h).all() and np.isfinite(v).all()):
        raise ValueError("non-finite spacing or metric value")
    if not (h[0] < h[1] < h[2]):
        raise ValueError("spacings must be strictly increasing fine -> coarse")
    if h[1]/h[0] < MIN_RATIO or h[2]/h[1] < MIN_RATIO:
        raise ValueError(f"refinement ratios must exceed {MIN_RATIO}")
    return h, v


def _ratio_function(p, r21, r32):
    """Model ratio ``(f3-f2)/(f2-f1)`` for a pure ``C h^p`` error term."""
    return math.expm1(p*math.log(r32))/(-math.expm1(-p*math.log(r21)))


def observed_order(spacings, values):
    """Solve ``(f3-f2)/(f2-f1) = (r32^p-1)/(1-r21^-p)`` for ``p > 0``.

    Uniform ratios reduce to the textbook ``p = ln|(f3-f2)/(f2-f1)|/ln r``.
    Returns ``nan`` when the series is inconsistent with a monotone ``C h^p``
    error (``(f3-f2)/(f2-f1) <= 0``, i.e. the metric does not decay
    monotonically towards a limit from one side).
    """
    h, v = _check(spacings, values)
    d1, d2 = v[1]-v[0], v[2]-v[1]
    if d1 == 0.:
        raise ValueError("fine and medium metrics are identical: no order signal")
    q = d2/d1
    if not q > 0.:
        return float("nan")
    r21, r32 = h[1]/h[0], h[2]/h[1]
    lo, hi = ORDER_BOUNDS
    if not _ratio_function(lo, r21, r32) <= q <= _ratio_function(hi, r21, r32):
        return float("nan")
    for _ in range(200):
        mid = .5*(lo+hi)
        if _ratio_function(mid, r21, r32) < q:
            lo = mid
        else:
            hi = mid
    return .5*(lo+hi)


def richardson_gci(fine, medium, coarse, spacings, *, theory_order=None,
                   order_tolerance=.2, safety_factor=None):
    """Three-grid Richardson extrapolation with a Roache GCI band.

    ``theory_order`` is the formal order of the scheme; when given, the study is
    only declared inside the asymptotic range if ``|p_observed-p_theory| <=
    order_tolerance``.  An inconsistent series (no monotone error decay) falls
    back to ``theory_order`` (or first order if unknown) with the conservative
    two-grid safety factor 3.0, so that a broken refinement study can never
    report a confident band.
    """
    h, v = _check(spacings, (fine, medium, coarse))
    if safety_factor is not None and not (math.isfinite(safety_factor) and safety_factor >= 1.):
        raise ValueError("safety factor must be finite and >= 1")
    r21, r32 = h[1]/h[0], h[2]/h[1]
    p = observed_order(h, v)
    fallback = p != p  # nan: no monotone C h^p error decay in this series
    p_used = float(1. if theory_order is None else theory_order) if fallback else p
    convergence = "inconsistent" if fallback else "monotone"
    if v[0] == 0.:
        raise ValueError("relative GCI needs a non-zero fine-grid metric")
    eps = abs((v[1]-v[0])/v[0])
    gci = (safety_factor or (SAFETY_FALLBACK if fallback else SAFETY_OBSERVED))*eps/(r21**p_used-1.)
    correction = (v[1]-v[0])/(r21**p_used-1.)
    extrapolated = float(v[0]-correction)
    if theory_order is None:
        asymptotic = not fallback
        order_ratio = float("nan")
    else:
        if not (math.isfinite(theory_order) and theory_order > 0.):
            raise ValueError("theory order must be finite and positive")
        order_ratio = float(p_used/theory_order)
        asymptotic = bool(not fallback and abs(p_used-theory_order) <= order_tolerance)
    half = gci*abs(extrapolated)
    return RefinementStudy(
        spacings=tuple(float(x) for x in h), values=tuple(float(x) for x in v),
        order=float(p), order_used=float(p_used), order_ratio=order_ratio,
        convergence=convergence,
        asymptotic=asymptotic, extrapolated=extrapolated,
        gci=float(gci),
        safety_factor=float(safety_factor or (SAFETY_FALLBACK if fallback else SAFETY_OBSERVED)),
        band=(float(extrapolated-half), float(extrapolated+half)),
        theory_order=float("nan") if theory_order is None else float(theory_order),
        refinement_ratio=float(r21))


def power_law_series(exact, primary, order, spacings, *, secondary=None,
                     secondary_order=None):
    """Synthetic metrics ``exact + primary*h^order (+ secondary*h^secondary_order)``.

    Used to give the estimator a closed-form target: with ``secondary=None`` the
    observed order and the extrapolated value must be recovered to round-off.
    """
    h = np.asarray(spacings, dtype=float)
    if not (h.ndim == 1 and h.size >= 2 and np.isfinite(h).all() and (h > 0).all()):
        raise ValueError("positive finite spacings required")
    values = exact + primary*h**order
    if secondary is not None:
        if secondary_order is None:
            raise ValueError("a secondary term needs its order")
        values = values + secondary*h**secondary_order
    return [float(x) for x in values]


def reference_agreement(study, reference):
    """Compare a study with an independent reference value.

    Returns the relative error of the extrapolated value, the relative GCI, the
    coverage verdict (does the band contain the reference, which is the only
    property a GCI can honestly claim) and the efficiency index
    ``actual_error/GCI`` (Roache: values far below one mean the band is
    conservative, values above one mean the band was not a bound).
    """
    if not math.isfinite(reference):
        raise ValueError("finite reference required")
    if reference == 0.:
        raise ValueError("zero reference needs an explicitly scaled metric")
    error = abs(study.extrapolated-reference)/abs(reference)
    lo, hi = study.band
    covered = bool(lo <= reference <= hi)
    efficiency = float("nan") if study.gci == 0. else error/study.gci
    return dict(reference=float(reference), extrapolated=study.extrapolated,
                relative_error=float(error), gci=study.gci,
                efficiency_index=efficiency, band_covered=covered,
                band=study.band)