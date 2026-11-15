"""Independent, non-wall-shear measurements on a periodic-hill mean field.

Why this module exists (round 21 of ``CASE-PERIODIC-HILLS``)
-----------------------------------------------------------
Every published ``x_r/H`` in this repository is read off the **wall shear**
profile: ``tau_b(x)`` is post-processed from the near-wall tangential velocity
gradient and its sign changes are paired into recirculation bubbles
(:mod:`tensorfvm.verification.wall_shear_bubbles`).  Round 17 certified that
*operator* -- but not the quantity it is applied to.  If the wall-shear
post-processing itself is biased (metric terms on the terrain-following mesh,
face-to-cell averaging, ...), no amount of care inside the crossing operator
will notice.

The mean velocity field is a genuinely independent data path: it is the raw
time-averaged primitive variable, and a reversed-flow region is visible in it
without differentiating anything.  :func:`backflow_extent` reads the length of
the near-wall reversed-flow region straight from ``mean_u`` on the lowest cell
row, so the two numbers can be compared:

    wall-shear crossing of the mean profile   vs.   extent of mean_u < 0

They are two different definitions AND two different data paths.  Agreement
does not prove the physics is right -- both read the same discretised field --
but disagreement of tens of percent proves that at least one of them is not
measuring "the recirculation length".

Second point, also from round 21: for an *unsteady* run there are two
legitimate definitions of ``x_r/H`` --

* crossing of the **time-mean** wall shear (what rounds 17--20 report), and
* the **ensemble mean of instantaneous** reattachment points.

For a statistically stationary run these coincide.  :func:`definition_spread`
quantifies how far apart they are so that a published number can never be
quoted without saying which of the two it is.
"""

from __future__ import annotations

import numpy as np

__all__ = ["backflow_extent", "backflow_extent_rows", "definition_spread",
           "bubble_object_report", "BUBBLE_SINGLE", "BUBBLE_OBJECT_CONSISTENT",
           "BUBBLE_OBJECT_MISMATCH", "DEFINITION_MEAN_FIELD", "DEFINITION_ENSEMBLE"]

DEFINITION_MEAN_FIELD = "crossing_of_time_mean_wall_shear"
DEFINITION_ENSEMBLE = "ensemble_mean_of_instantaneous_reattachment"


def _validate(u_row, x_over_h):
    u = np.asarray(u_row, dtype=float).ravel()
    x = np.asarray(x_over_h, dtype=float).ravel()
    if u.shape != x.shape:
        raise ValueError("u_row and x_over_h must have the same length")
    if u.size < 3:
        raise ValueError("need at least 3 streamwise samples")
    return u, x


def backflow_extent(u_row, x_over_h, *, threshold=0.0):
    """Downstream end of the reversed-flow region on one cell row.

    ``threshold`` is the value of ``u`` counted as *not* reversed (default 0).

    Returns a dict with ``x_end`` (abscissa of the last reversed cell, ``None``
    when the row has no reversed cell), ``n_cells``, ``contiguous_from_first``
    (whether the reversed cells form one contiguous run starting at the first
    reversed cell -- a detached patch is reported, never silently merged into
    one bubble), and ``seam_wrapped`` (the run crosses the periodic seam).
    """
    u, x = _validate(u_row, x_over_h)
    neg = u < threshold
    if not neg.any():
        return {"x_end": None, "n_cells": 0, "contiguous_from_first": True,
                "seam_wrapped": False}
    idx = np.where(neg)[0]
    runs = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
    main = max(runs, key=len)
    seam = bool(neg[0] and neg[-1])
    return {"x_end": float(x[main[-1]]), "n_cells": int(neg.sum()),
            "contiguous_from_first": bool(len(runs) == 1),
            "seam_wrapped": seam,
            "n_runs": int(len(runs)),
            "longest_run_cells": int(len(main))}


def backflow_extent_rows(mean_u, x_over_h, rows=(0, 1, 2, 3)):
    """``backflow_extent`` for a list of cell rows (row 0 = wall-adjacent)."""
    m = np.atleast_2d(np.asarray(mean_u, dtype=float))
    return {int(r): backflow_extent(m[r], x_over_h) for r in rows if r < m.shape[0]}


def definition_spread(x_r_mean_field, x_r_ensemble):
    """Compare the two legitimate definitions of ``x_r/H``.

    Returns ``{"x_r": ..., "delta_abs": ..., "delta_rel": ..., "flag": ...}``
    where ``flag`` is ``"definition_sensitive"`` as soon as the two definitions
    disagree by more than 10 % (the acceptance tolerance of the case itself), so
    quoting one without the other is a demonstrable error for that run.
    """
    a = None if x_r_mean_field is None else float(x_r_mean_field)
    b = None if x_r_ensemble is None else float(x_r_ensemble)
    if a is None or b is None:
        return {"x_r": {"mean_field": a, "ensemble": b}, "delta_abs": None,
                "delta_rel": None, "flag": "not_measured"}
    d = b - a
    rel = abs(d) / abs(a) if a else float("inf")
    return {"x_r": {"mean_field": a, "ensemble": b}, "delta_abs": d,
            "delta_rel": rel,
            "flag": "definition_sensitive" if rel > 0.10 else "definition_insensitive"}

# ---------------------------------------------------------------------------
# round 22: which *bubble* does a published x_r/H refer to?
# ---------------------------------------------------------------------------
BUBBLE_SINGLE = "single_bubble"
BUBBLE_OBJECT_CONSISTENT = "object_consistent"
BUBBLE_OBJECT_MISMATCH = "object_mismatch"


def bubble_object_report(tau, x_over_h, *, window=None, primary="largest_area"):
    """Report which reversed-shear bubble the selected ``x_r/H`` belongs to.

    ``x_r/H`` is only meaningful together with *which* bubble it is measured on.
    The reference of the periodic-hill case (``x_r/H = 5.41``, Breuer et al.
    2009) is the reattachment of the **crest bubble** -- the one that separates
    just downstream of the hill crest.  :func:`wall_shear_bubbles.
    separation_reattachment` defaults to ``primary="largest_area"``, which is the
    right rule when a solution has one bubble but silently switches to a
    *different* bubble when it has two.

    This function runs the certified operator under both rules and reports
    whether they agree, so that a cross-object comparison cannot be published as
    an error of the quantity.

    Returns ``{"n_bubbles", "n_crossings", "selected", "crest",
    "selected_is_crest", "x_r_over_H": {"largest_area", "first"}, "flag",
    "reason"}`` where ``flag`` is :data:`BUBBLE_SINGLE` (nothing to disambiguate),
    :data:`BUBBLE_OBJECT_CONSISTENT` (both rules pick the crest bubble) or
    :data:`BUBBLE_OBJECT_MISMATCH` (the default rule picks another bubble, so the
    number is *not* the same object as the reference).
    """
    from .wall_shear_bubbles import separation_reattachment

    largest = separation_reattachment(tau, x_over_h, window=window, primary=primary)
    first = separation_reattachment(tau, x_over_h, window=window, primary="first")

    def _entry(res):
        idx = res.get("primary_index")
        bubbles = res.get("bubbles") or []
        b = bubbles[idx] if idx is not None and 0 <= idx < len(bubbles) else None
        return {"index": idx,
                "separation": res.get("x_s_over_H"),
                "reattachment": res.get("x_r_over_H"),
                "length": None if b is None else float(b.get("length", float("nan"))),
                "reversed_area": None if b is None else float(b.get("reversed_area",
                                                                    float("nan")))}

    n = len(largest.get("bubbles") or [])
    out = {
        "n_bubbles": int(n),
        "n_crossings": int(largest.get("n_crossings") or 0),
        "selected": _entry(largest),
        "crest": _entry(first),
        "x_r_over_H": {"largest_area": largest.get("x_r_over_H"),
                       "first": first.get("x_r_over_H")},
        "selected_is_crest": bool(largest.get("primary_index") is not None
                                  and largest.get("primary_index") ==
                                  first.get("primary_index")),
    }
    if n <= 1:
        out["flag"] = BUBBLE_SINGLE
        out["reason"] = "one reversed-shear bubble: the two rules cannot disagree"
    elif out["selected_is_crest"]:
        out["flag"] = BUBBLE_OBJECT_CONSISTENT
        out["reason"] = "the largest bubble is also the crest bubble"
    else:
        out["flag"] = BUBBLE_OBJECT_MISMATCH
        out["reason"] = (
            "the largest-area bubble is NOT the crest bubble: its reattachment "
            f"{out['x_r_over_H']['largest_area']} is a different physical object "
            f"from the crest bubble's {out['x_r_over_H']['first']}")
    return out
