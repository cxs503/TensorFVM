"""Attribution arithmetic for a code-vs-table benchmark comparison.

Why this module exists
----------------------
``CASE-FVM-CAVITY-RE1000`` compares three numerical solutions (production MAC
FVM, a cross-check psi-omega code and a Chebyshev spectral solver) with the
Ghia, Ghia & Shin (1982) tables.  All three sit roughly 1.5 % away from the
tables on the extremum quantities (``u_min`` on ``x = L/2``, ``v_max`` on
``y = L/2``).  ``docs/CASE-FVM-CAVITY-RE1000/COMPREHENSIVE_REPORT.md`` §3.2
concludes from the mutual agreement of the codes that this residual "belongs to
Ghia's own 129^2 discretisation".

That is an *attribution* claim, and an attribution claim is only as good as the
arithmetic behind it.  The three tests implemented here are the ones that
attribution needs and that a bare "the codes agree" argument skips:

1. :func:`contraction_ratios` / :func:`is_asymptotic` - is the refinement
   sequence actually in its asymptotic range?  Richardson-style extrapolation
   (and any conclusion drawn from two extrapolations agreeing) is meaningless
   outside it.  The diagnostic works on the *increments*, which are observable,
   instead of on a fitted limit, which is not.
2. :func:`gci` - Roache/Celik grid-convergence index: the discretisation
   uncertainty band of the *finest* solution.  If the tabulated reference value
   lies inside that band, the residual is not significant and there is nothing
   to attribute; reporting a *cause* for a non-significant residual is the error
   this function is designed to block.
3. :func:`extremum_floor` - the reference-side floor: how well can a *tabulated*
   profile pin its own extremum?  Computed from the table alone (three
   interpolants, no solver), this bounds what any solver may legitimately blame
   on the table.

:func:`attribution_verdict` combines the three into a verdict that is allowed to
answer "unresolved", and does so on the repository's own stored artefacts (see
``tests/test_cavity_attribution.py``).

Everything is a pure function of numbers handed in by the caller; nothing here
knows about the cavity, so the module is reusable for any refinement study
against a tabulated benchmark.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence

import numpy as np

__all__ = [
    "contraction_ratios",
    "is_asymptotic",
    "gci",
    "extremum_floor",
    "finest_grid_disagreement",
    "attribution_verdict",
]


# --------------------------------------------------------------------------- #
# 1. Is the refinement sequence in its asymptotic range?
# --------------------------------------------------------------------------- #
def contraction_ratios(values: Sequence[float]) -> list[float]:
    """Ratios of successive increments ``(v[i+1]-v[i]) / (v[i]-v[i-1])``.

    A sequence converging as ``v_h = v_exact + C h^p`` produces a *constant*
    contraction ratio ``(h_{i+1}/h_i)^p``; drift in the ratio means the
    asymptotic range has not been reached.  Working with increments avoids
    needing the unknown limit, so the test cannot be circular.
    """
    v = [float(x) for x in values]
    if len(v) < 3:
        raise ValueError("need at least three refinement levels")
    out: list[float] = []
    for i in range(1, len(v) - 1):
        d_prev = v[i] - v[i - 1]
        d_next = v[i + 1] - v[i]
        if d_prev == 0.0:
            raise ValueError(f"zero increment at level {i}; sequence is flat")
        out.append(d_next / d_prev)
    return out


def is_asymptotic(values: Sequence[float], *, max_spread: float = 0.5) -> dict:
    """Decide whether ``values`` looks like a single geometric contraction.

    ``max_spread`` bounds the allowed relative spread of the contraction ratios:
    the estimator is

    .. math::  \\mathrm{spread} = \\frac{\\max\\rho - \\min\\rho}{\\max\\rho}

    A sequence with ``spread <= max_spread`` is treated as asymptotic.  The
    default (0.5, i.e. the ratios agree within a factor of two) is deliberately
    loose - the point is to *reject* sequences whose ratios differ by an order of
    magnitude, not to certify high precision.
    """
    rho = contraction_ratios(values)
    lo, hi = min(rho), max(rho)
    spread = 0.0 if hi == 0.0 else (hi - lo) / abs(hi)
    #: With only three levels there is a single contraction ratio, so ``spread``
    #: is identically zero and the test cannot fail.  ``informative`` records
    #: whether the sequence actually constrains the asymptotic range; a verdict
    #: that leans on ``asymptotic`` alone while ``informative`` is False is
    #: leaning on nothing.
    informative = len(rho) >= 2
    return {
        "ratios": rho,
        "n_ratios": len(rho),
        "min_ratio": lo,
        "max_ratio": hi,
        "spread": spread,
        "max_spread": max_spread,
        "informative": bool(informative),
        "asymptotic": bool(
            informative and spread <= max_spread and all(abs(r) < 1.0 for r in rho)
        ),
    }


# --------------------------------------------------------------------------- #
# 2. Discretisation uncertainty of the finest solution
# --------------------------------------------------------------------------- #
def gci(
    coarse: float,
    medium: float,
    fine: float,
    *,
    refinement: float = 2.0,
    safety: float = 1.25,
) -> dict:
    """Roache/Celik grid-convergence index for a three-level, uniform-refinement ladder.

    Parameters
    ----------
    coarse, medium, fine :
        The same functional on successively refined grids (coarse first).
    refinement :
        Grid refinement factor ``r`` between consecutive levels.
    safety :
        Factor of safety ``Fs`` (1.25 for a three-grid study).

    Returns
    -------
    dict with the observed order ``p``, the relative difference
    ``epsilon = (medium-fine)/fine``, ``gci_fine`` (relative, i.e. a fraction),
    the resulting absolute band ``[fine - band, fine + band]`` and, when a
    reference value is supplied to :func:`attribution_verdict`, the relative
    position of the reference inside the band (0 = at the centre, 1 = on the
    edge).
    """
    c, m, f = float(coarse), float(medium), float(fine)
    d_cm = m - c
    d_mf = f - m
    if d_mf == 0.0:
        raise ValueError("medium and fine levels are identical; no order estimate")
    ratio = d_cm / d_mf
    if ratio <= 0.0:
        raise ValueError(
            "increments change sign between levels: the sequence is oscillatory "
            "and no asymptotic order exists"
        )
    p = math.log(ratio) / math.log(refinement)
    eps = d_mf / f
    gci_fine = safety * abs(eps) / (refinement ** p - 1.0)
    band = gci_fine * abs(f)
    return {
        "order": p,
        "epsilon": eps,
        "increment_ratio": ratio,
        "gci_fine": gci_fine,
        "band": band,
        "interval": [f - band, f + band],
        "refinement": refinement,
        "safety": safety,
    }


# --------------------------------------------------------------------------- #
# 3. How well can a tabulated profile pin its own extremum?
# --------------------------------------------------------------------------- #
def _monotone_cubic(x: np.ndarray, y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Fritsch-Carlson monotone cubic Hermite (shape preserving, no new extrema)."""
    h = np.diff(x)
    d = np.diff(y) / h
    n = x.size
    m = np.zeros(n)
    for i in range(1, n - 1):
        if d[i - 1] * d[i] <= 0.0:
            m[i] = 0.0
        else:
            w1 = 2.0 * h[i] + h[i - 1]
            w2 = h[i] + 2.0 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i])
    m[0], m[-1] = d[0], d[-1]
    out = np.empty_like(X)
    for k, xx in enumerate(X):
        i = int(np.clip(np.searchsorted(x, xx) - 1, 0, n - 2))
        t = (xx - x[i]) / h[i]
        h00 = 2 * t**3 - 3 * t**2 + 1
        h10 = t**3 - 2 * t**2 + t
        h01 = -2 * t**3 + 3 * t**2
        h11 = t**3 - t**2
        out[k] = h00 * y[i] + h10 * h[i] * m[i] + h01 * y[i + 1] + h11 * h[i] * m[i + 1]
    return out


def extremum_floor(
    samples: Iterable[tuple[float, float]],
    *,
    mode: str = "min",
    n_eval: int = 40001,
) -> dict:
    """Reference-side floor: the spread of the extremum implied by the tabulated samples.

    A table lists ``f`` at a finite set of coordinates.  The quantity a solver is
    graded against (``min f``, ``max f``) is a property of the *continuous*
    profile, so the table only pins it down to the accuracy of the interpolant
    used to read it.  Three interpolants of increasing stiffness are applied to
    the same samples:

    * monotone cubic (Fritsch-Carlson) - shape preserving, cannot create an
      extremum between nodes, therefore a **lower** bound on the floor;
    * natural cubic spline - smooth, can shift the extremum slightly;
    * local parabola through the bracketing triple of the tabulated extremum -
      the most aggressive reading, an **upper** bound.

    The returned ``floor_lower``/``floor_upper`` are relative to the tabulated
    extremum.  Any solver-versus-table discrepancy below ``floor_upper`` cannot
    be attributed to the solver.  Note that when the extremum coincides with a
    tabulated node the monotone interpolant returns the node value exactly, so
    ``floor_lower`` collapses to 0 - that is a property of the interpolant, not
    evidence that the table is exact; use ``floor_upper`` for attribution.

    Parameters
    ----------
    samples :
        ``(coordinate, value)`` pairs; order is irrelevant (they are sorted).
    mode :
        ``"min"`` or ``"max"``.
    """
    if mode not in ("min", "max"):
        raise ValueError("mode must be 'min' or 'max'")
    p = np.array(sorted(((float(a), float(b)) for a, b in samples), key=lambda t: t[0]))
    if p.shape[0] < 4:
        raise ValueError("need at least four samples")
    x, y = p[:, 0], p[:, 1]
    if np.any(np.diff(x) <= 0.0):
        raise ValueError("duplicate coordinates in samples")
    i = int(np.argmin(y) if mode == "min" else np.argmax(y))
    tabulated = float(y[i])
    if tabulated == 0.0:
        raise ValueError("tabulated extremum is zero; use an absolute floor")
    X = np.linspace(float(x[0]), float(x[-1]), n_eval)
    pick = np.min if mode == "min" else np.max

    v_mono = float(pick(_monotone_cubic(x, y, X)))
    # natural cubic spline (numpy only, tridiagonal system)
    h = np.diff(x)
    n = x.size
    A = np.zeros((n, n))
    rhs = np.zeros(n)
    A[0, 0] = A[-1, -1] = 1.0
    for k in range(1, n - 1):
        A[k, k - 1] = h[k - 1]
        A[k, k] = 2.0 * (h[k - 1] + h[k])
        A[k, k + 1] = h[k]
        rhs[k] = 3.0 * ((y[k + 1] - y[k]) / h[k] - (y[k] - y[k - 1]) / h[k - 1])
    c = np.linalg.solve(A, rhs)
    b = (np.diff(y) / h) - h * (2.0 * c[:-1] + c[1:]) / 3.0
    d = (c[1:] - c[:-1]) / (3.0 * h)
    idx = np.clip(np.searchsorted(x, X) - 1, 0, n - 2)
    t = X - x[idx]
    spline_vals = y[idx] + b[idx] * t + c[idx] * t**2 + d[idx] * t**3
    v_spline = float(pick(spline_vals))

    j = [max(i - 1, 0), i, min(i + 1, n - 1)]
    if len({*j}) < 3:
        v_quad, x_quad = tabulated, float(x[i])
    else:
        co = np.polyfit(x[j], y[j], 2)
        x_quad = float(-co[1] / (2.0 * co[0]))
        v_quad = float(np.polyval(co, x_quad))

    floor = lambda v: abs((v - tabulated) / tabulated)
    return {
        "mode": mode,
        "tabulated": tabulated,
        "x_tabulated": float(x[i]),
        "interpolants": {"monotone": v_mono, "spline": v_spline, "local_parabola": v_quad},
        "x_parabola_vertex": x_quad,
        "bracket_gap": float(x[j[2]] - x[j[0]]),
        "floor_lower": floor(v_mono),
        "floor_mid": floor(v_spline),
        "floor_upper": max(floor(v_mono), floor(v_spline), floor(v_quad)),
    }


# --------------------------------------------------------------------------- #
# 4. Raw (non-extrapolated) agreement between two codes
# --------------------------------------------------------------------------- #
def finest_grid_disagreement(a: float, b: float) -> float:
    """Relative difference of two *measured* (not extrapolated) fine-grid values."""
    a, b = float(a), float(b)
    scale = 0.5 * (abs(a) + abs(b))
    if scale == 0.0:
        raise ValueError("both values are zero")
    return abs(a - b) / scale


# --------------------------------------------------------------------------- #
# 5. The verdict
# --------------------------------------------------------------------------- #
def attribution_verdict(
    *,
    label: str,
    coarse: float,
    medium: float,
    fine: float,
    reference: float,
    ladder: Mapping[str, Sequence[float]] | None = None,
    reference_floor: float | None = None,
    refinement: float = 2.0,
) -> dict:
    """Combine the three tests into one of three verdicts.

    ``within_own_uncertainty``
        The tabulated reference lies inside the finest solution's GCI band.  The
        residual is not significant; nothing may be attributed.  *This is the
        only verdict that permits the report to say "consistent with the table".*
    ``reference_attributable``
        The residual is significant **and** ``reference_floor`` (from
        :func:`extremum_floor`) is large enough to account for it.
    ``unresolved``
        The residual is significant but the reference-side floor is too small to
        explain it.  Neither party may be blamed on the present evidence.

    Parameters
    ----------
    label :
        Name of the functional, echoed back.
    coarse, medium, fine :
        The solver's own ladder for this functional.
    reference :
        Tabulated value being aimed at.
    ladder :
        Optional ``{name: [coarse, medium, fine]}`` for other codes measuring the
        same functional.  When given, the *finest-grid* (not extrapolated)
        disagreement between ``fine`` and every other code is reported - that is
        the number an extrapolation must beat, and the number that is usually
        quoted in extrapolated form by mistake.
    reference_floor :
        Upper bound of the reference-side floor (relative), from
        :func:`extremum_floor`.  ``None`` means "not measured": the verdict can
        then only be ``within_own_uncertainty`` or ``unresolved``.
    """
    g = gci(coarse, medium, fine, refinement=refinement)
    lo, hi = g["interval"]
    residual = abs((float(fine) - float(reference)) / float(reference))
    rr = (abs(float(reference) - float(fine))) / g["band"] if g["band"] else float("inf")
    inside = bool(lo <= float(reference) <= hi)

    out = {
        "label": label,
        "finest": float(fine),
        "reference": float(reference),
        "residual_relative": residual,
        "gci": g,
        "reference_inside_gci_band": inside,
        "reference_position_in_band": rr,
    }

    asym = is_asymptotic([coarse, medium, fine])
    out["asymptotic"] = asym
    #: The GCI band is only a *bound* if the ladder is in its asymptotic range.
    #: With three levels that assumption is untested (see ``is_asymptotic``);
    #: the flag is propagated so a caller cannot silently treat the band as
    #: verified.
    out["gci_assumption_tested"] = asym["informative"]

    if ladder:
        out["finest_grid_disagreement"] = {
            name: finest_grid_disagreement(float(fine), float(seq[-1]))
            for name, seq in ladder.items()
        }

    if inside:
        verdict = "within_own_uncertainty"
        reason = (
            f"reference {reference:+.6g} lies inside the GCI band "
            f"[{lo:+.6g}, {hi:+.6g}] of the finest solution {fine:+.6g} "
            f"(position {rr:.2f} of the half-width, p={g['order']:.2f}); "
            "the residual is not significant and must not be attributed"
        )
    elif reference_floor is not None and reference_floor >= residual:
        verdict = "reference_attributable"
        reason = (
            f"reference-side floor {reference_floor:.3%} accounts for the "
            f"residual {residual:.3%}"
        )
    else:
        verdict = "unresolved"
        why = (
            "no reference-side floor supplied"
            if reference_floor is None
            else f"reference-side floor {reference_floor:.3%} < residual {residual:.3%}"
        )
        reason = (
            f"residual {residual:.3%} is significant (reference outside the GCI "
            f"band, position {rr:.2f}) but {why}; neither the solver nor the "
            "table may be blamed on this evidence"
        )
    if not asym["informative"]:
        reason += (
            " [caveat: only three refinement levels, so the asymptotic-range "
            "assumption behind the GCI band is untested - add a fourth level]"
        )
    out["verdict"] = verdict
    out["reason"] = reason
    return out