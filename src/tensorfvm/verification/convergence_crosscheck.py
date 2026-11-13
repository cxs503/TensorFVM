"""Three-grid GCI cross-check for the convergence diagnostics.

Why this module exists
----------------------
The convergence diagnosis in :mod:`tensorfvm.verification.sod1978_convergence`
estimates an observed order in two ways: adjacent-grid successive rates and a
log-log least-squares slope over the finest grids ("fitted exponent").  Both are
*estimators*: they are only an order of accuracy when the error sequence is
monotonically decreasing and the grids are inside the asymptotic range.
Outside that range the least-squares slope of a non-monotone sequence is simply
a number -- on the committed Sod artifact it comes out negative (shock-position
"order" -1.25) while the finest adjacent pair gives +1.84.  Nothing in the
artifact flagged that, and the anchor's mechanism verdict was read off the
unguarded exponent.

The cross-validation against the mature open-source / standards practice
(Roache 1994; Celik, Ghia, Roache, Freitas, Coleman & Raad, *ASME J. Fluids
Eng.* **130** (2008) 078001; ITTC 7.5-03-01-01 for the marine domain) shows
that the practitioners' procedure differs in three load-bearing details:

1. it uses *differences* of the functional, not only the error magnitude;
2. it classifies the sequence (monotonic / oscillatory / divergent) before
   attributing any order, and declines to report one otherwise;
3. it reports an uncertainty band (GCI) and an explicit asymptotic-range test
   (their eq. (17), ``GCI_21/(r^p GCI_32) ~ 1``).

This module ports exactly those three details so the v2 diagnosis can be read
with the same safety net, without giving up its extra measures (restricted
windows, cell-average L1, wave-position fingerprints), which the GCI procedure
does not have.

Cross-validation evidence (independent implementation, ref vs got tables,
difference list and the porting trade-off) lives in
``docs/xval/convergence-diagnosis-v2/``; the guard added here is exercised by
``tests/test_convergence_diagnosis_xval.py``.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence

FS_3GRID = 1.25          # Celik et al. (2008): safety factor for three grids
ASYMPTOTIC_BAND = 0.15   # |GCI_21/(r^p GCI_32) - 1| tolerance (declared a priori)


def classify(values: Sequence[float]) -> str:
    """Classify a three-grid sequence, coarse -> fine (Celik et al. table 3).

    ``R = eps_32/eps_21`` with ``eps_ij`` the functional differences: a
    converging sequence has ``0 < R < 1``; ``R < 0`` is oscillatory and
    ``R > 1`` divergent.  A zero difference makes the order undefined and is
    reported as divergent (the published procedure does not attribute an order).
    """
    if len(values) != 3:
        raise ValueError("exactly three grids required")
    eps21 = values[1] - values[0]
    eps32 = values[2] - values[1]
    if eps21 == 0.0 or eps32 == 0.0:
        return "divergent"
    ratio = eps32 / eps21
    if ratio < 0.0:
        return "oscillatory"
    return "monotonic" if ratio <= 1.0 else "divergent"


def triplet_order(values: Sequence[float], r: float) -> float:
    """Celik et al. (2008) eq. (3): ``p = |ln|eps32/eps21|| / ln r``.

    ``nan`` when the sequence is not monotonically convergent.
    """
    if classify(values) != "monotonic":
        return float("nan")
    eps21 = abs(values[1] - values[0])
    eps32 = abs(values[2] - values[1])
    if eps21 <= 0.0 or eps32 <= 0.0:
        return float("nan")
    return abs(math.log(eps32 / eps21)) / math.log(r)


def richardson_extrapolated(values: Sequence[float], r: float, p: float) -> float:
    """Richardson extrapolation of a three-grid triplet (Celik eqs. 5-6)."""
    if not (math.isfinite(p) and p > 0.0):
        return float("nan")
    rp = r ** p
    return (rp * values[2] - values[1]) / (rp - 1.0)


def gci(values: Sequence[float], r: float, p: float, *, which: str = "fine",
        fs: float = FS_3GRID) -> float:
    """Grid Convergence Index (Roache 1994; Celik et al. eq. 7).

    Relative form with ``Fs`` (1.25 for three grids); absolute when the fine
    value is zero.
    """
    if not (math.isfinite(p) and p > 0.0) or r <= 1.0:
        return float("nan")
    if which == "fine":
        eps, ref = abs(values[2] - values[1]), values[2]
    elif which == "medium":
        eps, ref = abs(values[1] - values[0]), values[1]
    else:
        raise ValueError("which must be 'fine' or 'medium'")
    e_rel = eps / abs(ref) if ref != 0.0 else eps
    return fs * e_rel / (r ** p - 1.0)


def asymptotic_ratio(values: Sequence[float], r: float, p: float,
                     fs: float = FS_3GRID) -> float:
    """Celik et al. (2008) eq. (17): ``GCI_medium/(r^p GCI_fine)``; ~1 in range.

    Published caveat that the cross-validation turned up: ``p`` is estimated
    from the same three grids, so with a single triplet and a single refinement
    ratio this ratio is an identity -- it validates the arithmetic, not the
    asymptotic range.  Use :func:`triplet_consistency` (>= 4 grids) for the
    non-vacuous test.
    """
    g_fine = gci(values, r, p, which="fine", fs=fs)
    g_med = gci(values, r, p, which="medium", fs=fs)
    if not (math.isfinite(g_fine) and math.isfinite(g_med)) or g_fine == 0.0:
        return float("nan")
    return g_med / (r ** p * g_fine)


def in_asymptotic_range(values: Sequence[float], r: float, p: float,
                        band: float = ASYMPTOTIC_BAND) -> bool:
    ratio = asymptotic_ratio(values, r, p)
    return bool(math.isfinite(ratio) and abs(ratio - 1.0) <= band)


def crosscheck_series(values: Sequence[float], h: Sequence[float], *,
                      fs: float = FS_3GRID,
                      band: float = ASYMPTOTIC_BAND) -> Dict[str, object]:
    """Reference diagnosis of the finest three grids of ``values``.

    ``values``/``h`` are ordered coarse -> fine.  The realised refinement ratios
    are taken from ``h`` (and their non-uniformity reported) instead of being
    assumed; ``observed_order`` is ``nan`` unless the sequence is monotone.
    """
    if len(values) < 3:
        raise ValueError("at least three grids required")
    vals = [float(v) for v in values[-3:]]
    hs = [float(v) for v in h[-3:]]
    r21, r32 = hs[1] / hs[2], hs[0] / hs[1]
    r = r21
    cls = classify(vals)
    p = triplet_order(vals, r)
    ext = richardson_extrapolated(vals, r, p)
    out: Dict[str, object] = {
        "phi_coarse_medium_fine": vals,
        "h_coarse_medium_fine": hs,
        "r21": r21, "r32": r32,
        "r_uniformity": abs(r21 - r32) / max(r21, r32),
        "convergence_class": cls,
        "observed_order": p,
        "extrapolated": ext,
        "gci_fine": gci(vals, r, p, which="fine", fs=fs),
        "gci_medium": gci(vals, r, p, which="medium", fs=fs),
        "asymptotic_ratio": asymptotic_ratio(vals, r, p, fs=fs),
        "asymptotic_band": band,
        "in_asymptotic_range": in_asymptotic_range(vals, r, p, band=band),
        "fs": fs,
    }
    out["estimated_relative_error_fine"] = (
        abs(ext - vals[2]) / abs(ext) if math.isfinite(ext) and ext != 0.0
        else float("nan"))
    out["limit_form"] = limit_form(vals, hs, p)
    out["gci_absolute_fine"] = gci_absolute(vals, r, p, which="fine", fs=fs)
    if len(values) >= 4:
        out["triplet_consistency"] = triplet_consistency(values, h, fs=fs, band=band)
    else:
        out["triplet_consistency"] = None
    return out


def guarded_loglog_exponent(values: Sequence[float], h: Sequence[float]) -> Dict[str, object]:
    """Log-log least-squares exponent **with** the assumption it needs.

    The v2 estimator (``fitted_exponent``) is a slope through ``(log h, log e)``;
    it is an order of accuracy only if ``e`` decreases monotonically with ``h``.
    ``valid`` records that test so downstream verdicts cannot read a slope that
    means nothing (the committed Sod artifact's -1.25 shock-position
    "exponent" is the failure mode).
    """
    vals = [float(v) for v in values]
    hs = [float(v) for v in h]
    if len(vals) < 2:
        return {"exponent": float("nan"), "valid": False, "reason": "too few grids"}
    if any(not (math.isfinite(v) and v > 0.0) for v in vals):
        return {"exponent": float("nan"), "valid": False,
                "reason": "non-positive or non-finite error"}
    decreasing = all(b < a for a, b in zip(vals, vals[1:]))
    increasing = all(b > a for a, b in zip(vals, vals[1:]))
    lx = [math.log(v) for v in hs]
    ly = [math.log(v) for v in vals]
    n = len(lx)
    mx, my = sum(lx) / n, sum(ly) / n
    sxx = sum((v - mx) ** 2 for v in lx)
    sxy = sum((v - mx) * (w - my) for v, w in zip(lx, ly))
    slope = sxy / sxx if sxx > 0.0 else float("nan")
    valid = bool((decreasing or increasing) and math.isfinite(slope) and slope > 0.0)
    reason = ""
    if not (decreasing or increasing):
        reason = "error sequence not monotone -> slope is not an order of accuracy"
    elif not (math.isfinite(slope) and slope > 0.0):
        reason = "non-positive slope"
    return {"exponent": slope, "valid": valid, "reason": reason,
            "monotone_decreasing": decreasing}


def successive_rate(values: Sequence[float], h: Sequence[float]) -> float:
    """Rate of the finest adjacent pair, ``ln(e1/e2)/ln(h1/h2)``."""
    if len(values) < 2:
        return float("nan")
    e1, e2 = float(values[-2]), float(values[-1])
    h1, h2 = float(h[-2]), float(h[-1])
    if e1 <= 0.0 or e2 <= 0.0 or h1 <= 0.0 or h2 <= 0.0:
        return float("nan")
    return math.log(e1 / e2) / math.log(h1 / h2)


def effective_exponent(values: Sequence[float], h: Sequence[float]) -> Dict[str, object]:
    """Exponent to use for a verdict, plus where it came from.

    Prefers the least-squares exponent while it is defensible (monotone,
    positive); otherwise falls back to the finest-pair rate and says so.  A
    caller can therefore never read a meaningless slope silently.
    """
    guard = guarded_loglog_exponent(values, h)
    if guard["valid"]:
        return {"value": float(guard["exponent"]), "source": "loglog_lsq",
                "valid": True, "reason": guard["reason"],
                "class": classify([float(v) for v in values[-3:]])
                if len(values) >= 3 else "unknown"}
    return {"value": successive_rate(values, h), "source": "successive_finest",
            "valid": False, "reason": guard["reason"],
            "class": classify([float(v) for v in values[-3:]])
            if len(values) >= 3 else "unknown"}


def crosscheck_table(rows: Sequence[Dict], keys: Sequence[str]) -> Dict[str, Dict]:
    """Cross-check every key of a v2 rate table (``rows``: coarse -> fine)."""
    out: Dict[str, Dict] = {}
    for key in keys:
        pairs = [(1.0 / float(r["nx"]), float(r[key])) for r in rows
                 if key in r and isinstance(r[key], (int, float))
                 and math.isfinite(float(r[key]))]
        if len(pairs) < 3:
            out[key] = {"crosscheck": None,
                        "effective_exponent": None,
                        "reason": f"only {len(pairs)} finite grids"}
            continue
        hs = [p[0] for p in pairs]
        vals = [p[1] for p in pairs]
        out[key] = {
            "crosscheck": crosscheck_series(vals, hs),
            "effective_exponent": effective_exponent(vals, hs),
            "loglog_guard": guarded_loglog_exponent(vals, hs),
        }
    return out

def gci_absolute(values: Sequence[float], r: float, p: float, *,
                 which: str = "fine", fs: float = FS_3GRID) -> float:
    """Absolute GCI ``Fs |eps|/(r^p - 1)`` -- the form for zero-limit functionals.

    An error norm converges to zero, so the relative form (a percentage of the
    fine value) and the eq. (17) ratio both degenerate; the absolute band is
    what carries information there.
    """
    if not (math.isfinite(p) and p > 0.0) or r <= 1.0:
        return float("nan")
    if which == "fine":
        eps = abs(values[2] - values[1])
    elif which == "medium":
        eps = abs(values[1] - values[0])
    else:
        raise ValueError("which must be 'fine' or 'medium'")
    return fs * eps / (r ** p - 1.0)


def limit_form(values: Sequence[float], h: Sequence[float], p: float) -> str:
    """``zero_limit`` / ``nonzero_limit`` / ``undefined`` for the fine triplet.

    Decides which GCI form is meaningful (see :func:`gci_absolute`).
    """
    vals = [float(v) for v in values[-3:]]
    if len(vals) < 3 or not (math.isfinite(p) and p > 0.0):
        return "undefined"
    r = float(h[-2]) / float(h[-1])
    ext = richardson_extrapolated(vals, r, p)
    if not math.isfinite(ext) or vals[-1] == 0.0:
        return "undefined"
    return "zero_limit" if abs(ext) <= 0.10 * abs(vals[-1]) else "nonzero_limit"


def triplet_consistency(values: Sequence[float], h: Sequence[float], *,
                        fs: float = FS_3GRID,
                        band: float = ASYMPTOTIC_BAND) -> Dict[str, object]:
    """Overlapping-triplet consistency: the non-vacuous asymptotic-range test.

    With four monotonically refined grids two independent triplets exist; a
    sequence inside the asymptotic range yields the same observed order from
    both, and their relative difference measures how far outside it we are.
    ``consistent`` is the predeclared 15 % statement about that difference.
    """
    if len(values) < 4:
        raise ValueError("four grids required for an overlapping-triplet test")
    vals = [float(v) for v in values[-4:]]
    hs = [float(v) for v in h[-4:]]
    p_coarse = triplet_order(vals[0:3], hs[1] / hs[2])
    p_fine = triplet_order(vals[1:4], hs[2] / hs[3])
    rel = (abs(p_fine - p_coarse) / abs(p_coarse)
           if math.isfinite(p_coarse) and math.isfinite(p_fine) and p_coarse != 0.0
           else float("nan"))
    return {"p_triplet_coarse": p_coarse, "p_triplet_fine": p_fine,
            "rel_diff": rel,
            "consistent": bool(math.isfinite(rel) and rel <= band),
            "class_coarse": classify(vals[0:3]), "class_fine": classify(vals[1:4]),
            "band": band}
