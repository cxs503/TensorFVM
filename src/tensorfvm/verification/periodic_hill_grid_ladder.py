"""Grid/time-step convergence attribution for ``CASE-FVM-PERIODIC-HILLS``.

Why this module exists (round 22)
---------------------------------
Rounds 17--21 of the periodic-hill anchor established two things:

* the closure-free (``laminar``, :math:`\\nu_t \\equiv 0`) 2-D solution at 96x48
  gives ``x_r/H = 8.26`` against the ERCOFTAC reference ``5.41`` (+53 %), and the
  analogous SST run gives ``7.6--7.7`` (+41 %);
* switching the closure only shortens the bubble by 0.6--0.7 ``H`` out of the
  2.85 ``H`` that would be needed, i.e. **77--80 % of the error cannot be
  explained by that class of closure change** (round 21, table 2).

The remaining candidate that had never been measured is *discretisation*: every
published number of this anchor lives on one mesh (96x48, ``dx = 0.094 H``,
``dy = 0.063 H``) at one time step.  This module supplies the estimators needed
to attribute the error to discretisation or to rule it out, so that the answer
does not depend on who reads the numbers:

* :func:`observed_order` -- grid-triplet order estimate, with an explicit
  ``None`` when the triplet is not in the asymptotic range (the usual way order
  estimates are faked is to report a number for a non-monotone triplet);
* :func:`richardson_extrapolation` / :func:`gci_fine` -- extrapolated value and
  its grid-convergence index (Roache), cell-centred form;
* :func:`attribute` -- the **pre-registered** decision rule (see its docstring);
  the rule is fixed before the ladder is read, so that a number cannot be
  re-interpreted after the fact;
* :func:`temporal_sensitivity` -- the same test in time at fixed mesh, and
  :func:`combined_span` -- how much of a stated error either refinement axis can
  account for.

Nothing here knows about a particular mesh or run: callers pass plain numbers.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "refinement_ratio",
    "observed_order",
    "richardson_extrapolation",
    "gci_fine",
    "temporal_sensitivity",
    "combined_span",
    "attribute",
    "ORDER_UNDEFINED",
    "DISCRETISATION_ATTRIBUTABLE",
    "ASYMPTOTE_OUTSIDE_BAND",
    "INCONCLUSIVE",
]

ORDER_UNDEFINED = "order_undefined"
DISCRETISATION_ATTRIBUTABLE = "discretisation_attributable"
ASYMPTOTE_OUTSIDE_BAND = "asymptote_outside_band"
INCONCLUSIVE = "inconclusive"

#: differences whose ratio falls outside ``1/|_DIFF_RATIO_LIMIT .. _DIFF_RATIO_LIMIT|``
#: are not treated as an asymptotic triplet.  A ratio of 2**p is expected for
#: order ``p`` under uniform refinement (``p = 1..4`` => 2 .. 16); the limit is
#: deliberately loose (5 decades) because coarse grids overshoot wildly, while a
#: ratio of 1 (identical differences) or a sign change (negative ratio) is not an
#: asymptotic sequence at all and must not yield an "order".
_DIFF_RATIO_LIMIT = 1.0e5


def refinement_ratio(h_coarse, h_fine):
    """``h_coarse / h_fine`` (>1).  Raises for a non-refined pair."""
    hc = float(h_coarse)
    hf = float(h_fine)
    if not (math.isfinite(hc) and math.isfinite(hf)) or hc <= 0.0 or hf <= 0.0:
        raise ValueError("refinement levels must be positive and finite")
    r = hc / hf
    if r <= 1.0:
        raise ValueError(f"not a refinement: h_coarse/h_fine = {r:g}")
    return r


def observed_order(values, ratio):
    """Observed order of accuracy ``p`` from a three-level triplet.

    Parameters
    ----------
    values : sequence of 3 floats
        Quantity at **fine, medium, coarse** level, in that order.
    ratio : float
        Uniform refinement ratio ``h_{l+1}/h_l`` (>1); the triplet must be
        uniformly refined for a single ``p`` to have a meaning.

    Returns
    -------
    dict with ``p`` (``None`` when undefined), ``status`` and the two
    differences ``d_medium`` = ``f_medium - f_fine``, ``d_coarse`` =
    ``f_coarse - f_medium``.

    ``p is None`` (``status == ORDER_UNDEFINED``) whenever an order estimate is
    not defensible:

    * fewer than three levels, or a non-finite value;
    * ``d_medium == 0`` or ``d_coarse == 0`` (the quantity is already at its
      asymptotic value *or* is grid-insensitive -- neither licenses an order);
    * the differences have opposite signs (non-monotone sequence) or their ratio
      is outside ``1e-5 .. 1e5``.
    """
    vals = np.asarray(values, dtype=float).ravel()
    if vals.size != 3:
        raise ValueError("observed_order needs exactly 3 levels")
    r = float(ratio)
    if r <= 1.0:
        raise ValueError("ratio must be > 1")
    base = {"p": None, "status": ORDER_UNDEFINED, "ratio": r,
            "d_medium": None, "d_coarse": None, "values": [float(v) for v in vals]}
    if not np.all(np.isfinite(vals)):
        base["reason"] = "non-finite level"
        return base
    d_m = float(vals[1] - vals[0])
    d_c = float(vals[2] - vals[1])
    base["d_medium"] = d_m
    base["d_coarse"] = d_c
    if d_m == 0.0 or d_c == 0.0:
        base["reason"] = "zero difference: quantity is grid-independent or already converged"
        return base
    q = d_c / d_m
    base["difference_ratio"] = q
    if q <= 0.0:
        base["reason"] = "non-monotone triplet (differences change sign)"
        return base
    if not (1.0 / _DIFF_RATIO_LIMIT <= q <= _DIFF_RATIO_LIMIT):
        base["reason"] = "difference ratio outside the asymptotic range"
        return base
    base["p"] = float(math.log(abs(q)) / math.log(r))
    base["status"] = "order_estimated"
    return base


def richardson_extrapolation(f_fine, f_next, ratio, p):
    """Richardson extrapolation of a uniform refinement series.

    ``h -> 0`` value estimated from the two finest levels, assuming order ``p``.
    Returns ``float`` (may be non-finite if ``p`` is degenerate; ``p <= 0`` is
    rejected because it has no extrapolation meaning).
    """
    a = float(f_fine)
    b = float(f_next)
    r = float(ratio)
    pp = float(p)
    if not (r > 1.0):
        raise ValueError("ratio must be > 1")
    if not math.isfinite(pp) or pp <= 0.0:
        raise ValueError("order p must be positive and finite")
    return a + (a - b) / (r ** pp - 1.0)


def gci_fine(f_fine, f_next, ratio, p, safety=1.25):
    """Grid-convergence index of the **fine** level (Roache, cell-centred).

    ``GCI = Fs |(f_fine - f_next)/f_fine| / (r^p - 1)`` -- the estimated relative
    error of the fine-grid value.  Returns ``nan`` for a zero reference value.
    """
    a = float(f_fine)
    b = float(f_next)
    r = float(ratio)
    pp = float(p)
    fs = float(safety)
    if not (r > 1.0):
        raise ValueError("ratio must be > 1")
    if not math.isfinite(pp) or pp <= 0.0:
        raise ValueError("order p must be positive and finite")
    if fs <= 0.0:
        raise ValueError("safety factor must be positive")
    if a == 0.0:
        return float("nan")
    return fs * abs((a - b) / a) / (r ** pp - 1.0)


def temporal_sensitivity(x_r_coarse_dt, x_r_fine_dt):
    """Relative change of a quantity under time-step halving (fixed mesh).

    Returns ``{"absolute", "relative", "fine_minus_coarse", "resolved": bool}``
    where ``resolved`` is ``True`` only when both values are finite; a diverged
    coarse run therefore yields ``relative = inf`` and ``resolved = False``
    instead of a silently dropped case.
    """
    a = float(x_r_coarse_dt)
    b = float(x_r_fine_dt)
    out = {"coarse_dt": a, "fine_dt": b, "absolute": None,
           "relative": None, "fine_minus_coarse": None, "resolved": False}
    if not (math.isfinite(a) and math.isfinite(b)):
        out["relative"] = float("inf")
        return out
    if a == 0.0 or b == 0.0:
        return out
    out["resolved"] = True
    out["fine_minus_coarse"] = b - a
    out["absolute"] = abs(b - a)
    out["relative"] = abs(b - a) / abs(a)
    return out


def combined_span(values):
    """``max - min`` of the finite entries, or ``nan``.

    Used for "how much of an error of magnitude ``E`` can *any* refinement level
    of this ladder explain": the honest bound is the spread of the ladder, not
    the size of the last step.
    """
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)],
                     dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(arr.max() - arr.min())


def attribute(ladder, reference, tolerance=0.10):
    """Pre-registered attribution of a reference mismatch to discretisation.

    Parameters
    ----------
    ladder : sequence of ``(h, x_r)`` pairs, **coarsest first**, uniformly
        refined (constant ``h_{i}/h_{i+1}``).
    reference : float
        Published reference value of the quantity.
    tolerance : float
        The case's own acceptance band, relative (0.10 for this anchor; the
        platform paper-level band 0.03 can be passed instead).

    Decision rule (**fixed before the ladder is read**, round 22):

    1. ``n_levels < 2`` -> ``INCONCLUSIVE``.
    2. With >= 3 levels: estimate ``p`` from the two finest ratios.  If ``p`` is
       undefined -> ``ORDER_UNDEFINED`` (no verdict; a non-monotone ladder cannot
       support either attribution).
    3. ``p`` defined: ``x_inf`` = Richardson extrapolation of the two finest.
       * ``|x_inf - ref| <= tolerance * ref`` -> ``DISCRETISATION_ATTRIBUTABLE``
         (the sequence is heading into the acceptance band).
       * ``|x_inf - x_finest| <= tolerance * |x_finest|`` **and**
         ``|x_inf - ref| > tolerance * ref`` -> ``ASYMPTOTE_OUTSIDE_BAND``
         (further refinement cannot change the verdict: the residual error is
         not discretisation of this family).
       * otherwise -> ``INCONCLUSIVE`` (the sequence is still moving; the ladder
         must be extended before any attribution).
    4. With exactly 2 levels only step 3's movement test is applied, using the
       change between the two levels as the estimate of remaining movement; the
       verdict can never be ``DISCRETISATION_ATTRIBUTABLE`` from two levels.

    Returns a dict with ``verdict``, ``reference``, ``tolerance``,
    ``x_finest``, ``x_coarse``, ``x_extrapolated``, ``order``, ``gci_fine``,
    ``movement_rel``, ``error_finest_rel``, ``error_extrapolated_rel`` and a
    ``rule`` string naming the branch that fired.
    """
    pairs = list(ladder)
    ref = float(reference)
    tol = float(tolerance)
    if tol <= 0.0:
        raise ValueError("tolerance must be positive")
    out = {"reference": ref, "tolerance": tol, "n_levels": len(pairs),
           "verdict": INCONCLUSIVE, "rule": "", "x_finest": None, "x_coarse": None,
           "x_extrapolated": None, "order": None, "gci_fine": None,
           "movement_rel": None, "error_finest_rel": None,
           "error_extrapolated_rel": None, "ladder": [[float(h), float(v)]
                                                      for h, v in pairs]}
    if len(pairs) < 2:
        out["rule"] = "fewer than two levels"
        return out
    hs = [float(h) for h, _ in pairs]
    vs = [float(v) for _, v in pairs]
    if not np.all(np.isfinite(vs)):
        out["rule"] = "non-finite level"
        return out
    out["x_coarse"] = vs[0]
    out["x_finest"] = vs[-1]
    out["error_finest_rel"] = abs(vs[-1] - ref) / abs(ref) if ref else float("nan")
    ratios = [refinement_ratio(hs[i], hs[i + 1]) for i in range(len(hs) - 1)]
    r_last = ratios[-1]
    if max(ratios) - min(ratios) > 1.0e-6 * max(ratios):
        out["rule"] = "non-uniform refinement ratios"
        return out
    if len(vs) >= 3:
        order = observed_order([vs[-1], vs[-2], vs[-3]], r_last)
        out["order"] = order["p"]
        if order["p"] is None:
            out["verdict"] = ORDER_UNDEFINED
            out["rule"] = f"order undefined: {order.get('reason', '')}"
            return out
        p = order["p"]
        x_inf = richardson_extrapolation(vs[-1], vs[-2], r_last, p)
        out["x_extrapolated"] = x_inf
        out["gci_fine"] = gci_fine(vs[-1], vs[-2], r_last, p)
        out["movement_rel"] = (abs(x_inf - vs[-1]) / abs(vs[-1])
                               if vs[-1] else float("nan"))
        if ref:
            out["error_extrapolated_rel"] = abs(x_inf - ref) / abs(ref)
        if out["error_extrapolated_rel"] is not None and \
                out["error_extrapolated_rel"] <= tol:
            out["verdict"] = DISCRETISATION_ATTRIBUTABLE
            out["rule"] = ("extrapolated value enters the tolerance band: "
                           "the mismatch is discretisation of this mesh family")
            return out
        if (out["movement_rel"] is not None and out["movement_rel"] <= tol
                and out["error_extrapolated_rel"] is not None
                and out["error_extrapolated_rel"] > tol):
            out["verdict"] = ASYMPTOTE_OUTSIDE_BAND
            out["rule"] = ("extrapolated value is within tolerance of the finest "
                           "level yet still outside the reference band: further "
                           "refinement of this family cannot change the verdict")
            return out
        out["rule"] = "sequence still moving: extend the ladder"
        return out
    # exactly two levels
    movement = abs(vs[-1] - vs[0]) / abs(vs[-1]) if vs[-1] else float("nan")
    out["movement_rel"] = movement
    if np.isfinite(movement) and movement <= tol and \
            out["error_finest_rel"] is not None and out["error_finest_rel"] > tol:
        out["verdict"] = ASYMPTOTE_OUTSIDE_BAND
        out["rule"] = ("two levels differ by less than the tolerance while both "
                       "sit outside the band")
        return out
    out["rule"] = "two levels only: no extrapolation, movement not negligible"
    return out