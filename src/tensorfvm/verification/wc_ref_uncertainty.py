"""Reference-side uncertainty budget for a cross-scheme grade (Woodward--Colella).

The problem this module exists for
----------------------------------
A grade is a *difference*: ``|q_graded - q_reference|``.  When the reference is itself a
numerical solution, that difference contains the reference's own discretisation error, so a
measured ``L1 = 0.89 %`` says nothing about the graded solution unless the reference's
uncertainty ``U_ref`` is smaller.  Round-3 of ``XVAL-FVM-M18737-ROUND2`` certified
``MUSCL/HLLC@6400`` at ``0.8931 % < 1 %`` against ``WENO5@12800``, whose self-certification
is ``0.5437 %`` -- the same order as the number being certified.

This module turns that observation into machinery:

* :func:`u_ref_from_pair` -- Richardson estimate of the *fine* member's error from a nested
  (ratio 2) self-certification difference, ``U_ref = e / (2**p - 1)``.
* :func:`selfcert_ladder` / :func:`u_ref_of_reference` -- ``U_ref`` of any reference rung,
  measured against the next finer member of the same family (never against the graded
  solution, which would be circular).
* :func:`grade_with_budget` -- measured L1 carrying its ``U_ref``, plus the honest verdict:
  a certificate needs ``L1 + U_ref < bar`` (triangle inequality); otherwise the number is an
  *upper bound*, which is a weaker but still useful claim.
* :func:`nx_ref_for_bar` -- the reference resolution the certificate would demand,
  extrapolating the observed self-certification order.

Nothing here runs a solver: it is pure post-processing of stored profiles, so the verdict
can be recomputed without recomputation.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from . import wc_refine as wcr

#: certificate bar for the graded solution (%)
BAR_PERCENT = 1.0


def u_ref_from_pair(e_selfcert_percent: float, order: float) -> float:
    """Reference uncertainty of the *fine* member of a nested ratio-2 pair.

    ``e = |q_coarse - q_fine|``.  With ``q - q_exact ~ C h^p`` and ``h_fine = h_coarse/2``,

        ``e ~ C h_coarse^p - C h_fine^p = C h_fine^p (2^p - 1)``

    so the fine solution's own error is ``e / (2^p - 1)``.  This is the quantity that has to
    be added to a measured grade before the grade can be called a certificate.
    """
    factor = 2.0 ** float(order) - 1.0
    if not math.isfinite(factor) or factor <= 0.0:
        raise ValueError(f"order {order!r} gives a non-positive Richardson factor")
    if e_selfcert_percent < 0.0:
        raise ValueError(f"self-certification difference must be >= 0, got {e_selfcert_percent}")
    return float(e_selfcert_percent) / factor


def selfcert_ladder(data_dir: Path | str, family: str = "weno5") -> List[Dict]:
    """Self-certification of every consecutive ratio-2 pair of the reference family.

    Each row is one pair ``coarse -> fine`` and carries the fine member's uncertainty
    ``U_ref``.  ``order`` is the *observed* log--log order of the pair measured against the
    next finer pair; the finest row therefore has no measured order, and its ``U_ref`` is
    flagged ``order_source="extrapolated"`` (constant-order assumption) instead of being
    presented as measured.  A row whose order cannot even be extrapolated is left ``None``.
    """
    dirs = Path(data_dir)
    nx = [n for n in wcr.WENO5_RUNGS if wcr.try_load_profile(dirs, n, family) is not None]
    rows: List[Dict] = []
    for coarse, fine in zip(nx, nx[1:]):
        if fine % coarse:
            continue
        r = wcr.reference_selfcert(dirs, coarse, fine, family)
        rows.append({"coarse_nx": coarse, "fine_nx": fine,
                     "l1_percent": r["l1_percent"], "peak_percent": r["peak_percent"],
                     "mass_percent": r["mass_percent"],
                     "order": None, "order_source": "none", "u_ref_percent": None})
    for lo, hi in zip(rows, rows[1:]):
        p = math.log(lo["l1_percent"] / hi["l1_percent"], 2)
        lo["order"] = p
        lo["order_source"] = "measured_pair"
        lo["u_ref_percent"] = u_ref_from_pair(lo["l1_percent"], p)
    if len(rows) >= 2 and rows[-1]["u_ref_percent"] is None:
        # the finest pair has no successor to measure its order against: extrapolate the
        # coarser pair's order, and say so -- this is an assumption, not a measurement
        p = rows[-2]["order"]
        if p is not None and p > 0:
            rows[-1]["order"] = p
            rows[-1]["order_source"] = "extrapolated"
            rows[-1]["u_ref_percent"] = u_ref_from_pair(rows[-1]["l1_percent"], p)
    return rows


def u_ref_of_reference(data_dir: Path | str, ref_nx: int, family: str = "weno5",
                       ladder: Optional[Sequence[Dict]] = None) -> Dict:
    """Uncertainty of ``ref_nx`` measured against the next finer member of its own family.

    Grading a solution against a reference whose own error was never estimated is circular,
    so a rung with neither a finer partner nor a coarser one to extrapolate from returns
    ``measurable=False`` with ``u_ref_percent=None``: an uncertified reference must not
    silently pass as one.
    """
    rows = list(ladder) if ladder is not None else selfcert_ladder(data_dir, family)
    for row in rows:
        if row["fine_nx"] == ref_nx:
            return {"ref_nx": ref_nx, "family": family, "measurable": True,
                    "l1_percent": row["l1_percent"], "order": row.get("order"),
                    "order_source": row.get("order_source"),
                    "u_ref_percent": row.get("u_ref_percent"),
                    "self_certified": bool(row["l1_percent"] < BAR_PERCENT)}
    present = [n for n in wcr.WENO5_RUNGS
               if wcr.try_load_profile(data_dir, n, family) is not None]
    return {"ref_nx": ref_nx, "family": family, "measurable": False,
            "l1_percent": None, "order": None, "order_source": "none",
            "u_ref_percent": None, "self_certified": False,
            "note": ("no paired member of the same family present; uncertainty unknown"
                     f" (reference rungs available: {present})")}


def grade_with_budget(measured_l1_percent: float, u_ref_percent: Optional[float],
                      bar_percent: float = BAR_PERCENT) -> Dict:
    """Attach the reference uncertainty to a measured grade and adjudicate honestly.

    ``verdict`` is one of

    * ``"certified"`` -- ``measured + U_ref < bar``: the grade survives even if the whole
      reference error points the wrong way (triangle inequality).
    * ``"upper_bound_only"`` -- ``measured < bar`` but ``measured + U_ref >= bar``: the
      number is an honest upper bound inside the bar, not a certificate.
    * ``"fail"`` -- ``measured >= bar``.
    * ``"unresolved"`` -- ``U_ref`` unknown (unpaired reference): no claim either way.
    """
    m = float(measured_l1_percent)
    if u_ref_percent is None:
        return {"measured_percent": m, "u_ref_percent": None,
                "worst_case_percent": None, "bar_percent": bar_percent,
                "verdict": "unresolved", "margin_percent": None}
    u = float(u_ref_percent)
    worst = m + u
    if worst < bar_percent:
        verdict = "certified"
    elif m < bar_percent:
        verdict = "upper_bound_only"
    else:
        verdict = "fail"
    return {"measured_percent": m, "u_ref_percent": u, "worst_case_percent": worst,
            "bar_percent": bar_percent, "verdict": verdict,
            "margin_percent": bar_percent - worst}


def nx_ref_for_bar(measured_l1_percent: float, u_ref_percent: float, ref_nx: int,
                   order: float, bar_percent: float = BAR_PERCENT) -> Dict:
    """Reference resolution whose uncertainty would leave room for a certificate.

    ``U_ref`` of a reference at ``nx`` scales as ``nx**-p`` (the self-certification order p),
    so ``U(nx*) = bar - measured`` gives ``nx* = ref_nx * (U_ref / (bar - measured))**(1/p)``.
    ``satisfiable=False`` when the measured grade alone already exceeds the bar.
    """
    allowance = bar_percent - float(measured_l1_percent)
    if allowance <= 0.0:
        return {"satisfiable": False, "nx_ref_needed": None,
                "allowance_percent": allowance,
                "note": "graded error alone exceeds the bar; no reference resolution helps"}
    if u_ref_percent is None or order is None or order <= 0.0:
        return {"satisfiable": False, "nx_ref_needed": None,
                "allowance_percent": allowance,
                "note": "reference uncertainty or order unknown"}
    if u_ref_percent <= allowance:
        return {"satisfiable": True, "nx_ref_needed": float(ref_nx),
                "allowance_percent": allowance,
                "note": "current reference resolution already sufficient"}
    need = float(ref_nx) * (float(u_ref_percent) / allowance) ** (1.0 / float(order))
    return {"satisfiable": True, "nx_ref_needed": need, "allowance_percent": allowance,
            "note": "extrapolated from the observed self-certification order"}


def grade_across_references(data_dir: Path | str, graded_nx: int,
                            ref_rungs: Iterable[int] = (6400, 12800),
                            family: str = "weno5") -> List[Dict]:
    """Re-grade one graded rung against several references, each carrying its ``U_ref``.

    This is the discriminating experiment: if swapping the reference for a finer one moves
    the measured L1 by more than the uncertainty claimed for the old one, the old number was
    reference-limited and the certificate built on it was not honest.
    """
    dirs = Path(data_dir)
    prof = wcr.try_load_profile(dirs, graded_nx, "hllc")
    if prof is None:
        raise wcr.DataMissing(f"graded profile nx={graded_nx} missing under {dirs}")
    ladder = selfcert_ladder(dirs, family)
    rows: List[Dict] = []
    for ref_nx in ref_rungs:
        ref = wcr.try_load_profile(dirs, ref_nx, family)
        if ref is None:
            rows.append({"ref_nx": ref_nx, "measured": False})
            continue
        agg = wcr.cell_average_l1(prof["rho"], ref["rho"])
        ur = u_ref_of_reference(dirs, ref_nx, family, ladder=ladder)
        row = {"ref_nx": ref_nx, "measured": True,
               "l1_percent": agg["l1_percent"], "peak_percent": agg["peak_percent"],
               "mass_percent": agg["mass_percent"],
               **grade_with_budget(agg["l1_percent"], ur["u_ref_percent"])}
        row["reference_u_ref_percent"] = ur["u_ref_percent"]
        row["reference_self_certified"] = ur["self_certified"]
        rows.append(row)
    measured = [r for r in rows if r.get("measured")]
    for lo, hi in zip(measured, measured[1:]):
        lo["delta_l1_to_next_percent"] = hi["l1_percent"] - lo["l1_percent"]
        lo["within_claimed_u_ref"] = bool(
            lo["reference_u_ref_percent"] is not None
            and abs(lo["delta_l1_to_next_percent"]) <= lo["reference_u_ref_percent"])
    return rows


def budget_report(data_dir: Path | str, graded_nx: int = 6400,
                  ref_rungs: Sequence[int] = (6400, 12800),
                  family: str = "weno5") -> Dict:
    """Full round-4 budget: self-certification ladder, re-grades, promise-vs-evidence."""
    rows = grade_across_references(data_dir, graded_nx, ref_rungs, family)
    measured = [r for r in rows if r.get("measured")]
    finest = measured[-1] if measured else None
    ladder = selfcert_ladder(data_dir, family)
    out: Dict = {"graded_nx": graded_nx, "family": family,
                 "selfcert_ladder": ladder, "regrades": rows,
                 "bar_percent": BAR_PERCENT}
    if finest is not None:
        out["finest_reference"] = {
            "ref_nx": finest["ref_nx"],
            "measured_l1_percent": finest["l1_percent"],
            "u_ref_percent": finest["reference_u_ref_percent"],
            "verdict": finest["verdict"],
            "worst_case_percent": finest["worst_case_percent"]}
        ur = u_ref_of_reference(data_dir, finest["ref_nx"], family, ladder=ladder)
        need = nx_ref_for_bar(finest["l1_percent"], ur["u_ref_percent"],
                              finest["ref_nx"], ur["order"])
        out["reference_resolution_needed"] = need
    return out