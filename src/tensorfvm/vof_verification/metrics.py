"""Error metrics and Grid Convergence Index (GCI) for the VOF verification suite.

Definitions
-----------
``relative_l1``
    ``sum |alpha_num - alpha_exact| dV / sum alpha_exact dV`` -- the standard
    interface-capturing error norm (Rider & Kothe 1995).  It is normalised by
    the *liquid* volume, so a value of 0.05 means "5 % of the liquid was placed
    in the wrong cell".  This is the quantity the acceptance criteria C2/C3
    (deviation < 5 %) are stated in.

``symmetric_difference``
    ``|{alpha > 0.5} XOR {alpha_exact > 0.5}| / |{alpha_exact > 0.5}|`` -- a
    purely geometric measure of how much *area* the reconstructed interface
    misplaces.  Complementary to ``relative_l1``: a diffuse but correctly
    centred interface can have a large L1 error and a small symmetric
    difference (and vice versa).

``mass_drift_relative``
    ``|V(T) - V(0)| / V(0)`` -- global conservation (C5).

``gci``
    Roache's Grid Convergence Index for three systematically refined grids
    (refinement ratio ``r``), with observed order

    .. math:: p = \\frac{\\ln(e_1/e_2)}{\\ln r}

    from the two finest grids (``e`` = error against the exact solution, so the
    "solutions" converge to zero error) and

    .. math:: \\mathrm{GCI}_{fine} = \\frac{F_s\\,|e_3/e_2|}{r^p - 1},

    ``F_s = 1.25`` (Roache's recommendation for a two-grid-pair estimate).
    The asymptotic ratio ``GCI_12 / (r^p GCI_23)`` is reported; a value near one
    means the triple is inside the asymptotic range, so ``p`` is trustworthy.
    An honest ``p <= 0`` or a wild ratio is reported, never clipped.
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "field_metrics",
    "observed_order",
    "gci",
    "triple_summary",
]


def field_metrics(alpha: torch.Tensor, exact: torch.Tensor,
                  cell_volume: float) -> dict:
    """L1/relative-L1/Linf/symmetric-difference indicators of one field."""
    diff = (alpha - exact).abs()
    liquid = float(exact.sum().item()) * cell_volume
    l1 = float(diff.sum().item()) * cell_volume
    num_mask = alpha > 0.5
    ref_mask = exact > 0.5
    sym = float((num_mask ^ ref_mask).sum().item()) * cell_volume
    ref_area = float(ref_mask.sum().item()) * cell_volume
    return {
        "l1": l1,
        "relative_l1": l1 / liquid if liquid > 0.0 else float("inf"),
        "linf": float(diff.max().item()),
        "symmetric_difference": sym / ref_area if ref_area > 0.0 else float("inf"),
        "liquid_volume": liquid,
        "bounded": [float(alpha.min().item()), float(alpha.max().item())],
    }


def observed_order(e_coarse: float, e_fine: float, ratio: float = 2.0):
    """Pairwise observed order ``p = ln(e_c/e_f)/ln(r)`` (``None`` if undefined)."""
    if e_coarse <= 0.0 or e_fine <= 0.0 or ratio <= 1.0:
        return None
    return math.log(e_coarse / e_fine) / math.log(ratio)


def gci(errors, refinement_ratio: float = 2.0, safety: float = 1.25) -> dict:
    """Grid Convergence Index of an exact-reference error triple.

    ``errors`` are ordered coarse -> medium -> fine and are errors against the
    exact solution, so the sequence converges to zero.  Reported:

    * ``p_fine``  = ``ln(e2/e3)/ln r`` -- observed order on the finest pair
      (this is the order the acceptance criterion C6 is judged on);
    * ``p_coarse`` = ``ln(e1/e2)/ln r`` -- second estimate, and
    * ``p_successive`` = ``ln((e1-e2)/(e2-e3))/ln r`` -- Roache's classical
      successive-difference form (identical in the asymptotic range);
    * ``GCI_fine`` = ``Fs e3 / (r^p - 1)`` -- the discretisation-uncertainty
      band of the finest grid's error, ``Fs = 1.25``;
    * ``status`` = ``"asymptotic"`` only when the two pairwise orders agree
      within 20 % and ``p_fine > 0``; otherwise the triple is flagged, and a
      non-monotone / zero-error triple reports ``p = None`` rather than a
      clipped number.
    """
    if len(errors) != 3:
        raise ValueError("gci needs exactly three grid levels")
    e1, e2, e3 = (float(e) for e in errors)
    r = float(refinement_ratio)
    p_coarse = observed_order(e1, e2, r)
    p_fine = observed_order(e2, e3, r)
    d12, d23 = e1 - e2, e2 - e3
    p_succ = (math.log(d12 / d23) / math.log(r)
              if d12 > 0.0 and d23 > 0.0 else None)
    out = {
        "errors": [e1, e2, e3],
        "refinement_ratio": r,
        "safety_factor": safety,
        "p_coarse": p_coarse,
        "p_fine": p_fine,
        "p_successive": p_succ,
        "observed_order": p_fine,
        "gci_fine": None,
        "status": "order not defined (non-monotone or zero error; honest)",
    }
    if p_fine is None or not math.isfinite(p_fine) or p_fine <= 0.0:
        out["observed_order"] = None
        return out
    denom = r ** p_fine - 1.0
    if denom > 0.0:
        out["gci_fine"] = safety * e3 / denom
        out["gci_relative_to_fine_error"] = safety / denom
    if d12 <= 0.0:
        out["status"] = ("non-monotone on the coarse pair: p_fine is "
                         "indicative only (honest, not clipped)")
        return out
    consistent = (p_coarse is not None and p_succ is not None
                  and abs(p_coarse - p_fine) <= 0.2 * p_fine)
    out["status"] = ("asymptotic" if consistent
                     else "outside asymptotic range (p_fine indicative only)")
    return out


def triple_summary(levels: list[dict], key: str = "relative_l1",
                   refinement_ratio: float = 2.0, safety: float = 1.25) -> dict:
    """GCI analysis of one metric across three grid levels (coarse -> fine)."""
    errors = [lv[key] for lv in levels]
    cells = [lv["cells"] for lv in levels]
    out = gci(errors, refinement_ratio, safety)
    out.update({"metric": key, "cells": cells,
                "pair_orders": [observed_order(errors[0], errors[1],
                                               refinement_ratio),
                                observed_order(errors[1], errors[2],
                                               refinement_ratio)]})
    return out