"""Independent reference implementation of the standard three-grid GCI procedure.

XVAL counterpart of TensorFVM's ``convergence-diagnosis-v2``
(:mod:`tensorfvm.verification.sod1978_convergence`).  This module is written
from the published procedure *only* and deliberately imports **nothing** from
``tensorfvm`` (audit convention: the reference face must not share code with the
face under cross-validation, otherwise agreement proves nothing).

Sources (formulas, not numbers -- every number in the artifact is recomputed):

* P. J. Roache, "Perspective: a method for uniform reporting of grid refinement
  studies", *ASME J. Fluids Eng.* **116** (1994) 405-413 -- GCI definition.
* I. B. Celik, U. Ghia, P. J. Roache, C. J. Freitas, H. Coleman & P. E. Raad,
  "Procedure for estimation and reporting of uncertainty due to discretization
  in CFD applications", *ASME J. Fluids Eng.* **130** (2008) 078001 -- the
  three-grid procedure, the observed-order formula, Richardson extrapolation,
  the safety factor Fs = 1.25 and the asymptotic-range check eq. (17).

The point of the cross-validation is that the two procedures answer the same
question ("what is the observed order / the extrapolated value / the
uncertainty band of this functional?") with different estimators:

    reference (this file)   : three grids, one refinement ratio, *differences*
                              of the functional, sign-aware, plus a
                              convergence class and an uncertainty band.
    v2 (repo under XVAL)    : all adjacent pairs, log-log least squares over
                              the finest grids of the *error magnitude*,
                              no class, no band.

They coincide only inside the asymptotic range; outside it they diverge, and
the divergence is the useful measurement -- the reference has an explicit test
for the assumption while v2 does not.
"""

from __future__ import annotations

import math
from typing import Dict, Sequence

FS_3GRID = 1.25           # Celik et al. (2008): safety factor for 3 grids
ASYMPTOTIC_BAND = 0.15    # |GCI21/(r^p GCI32) - 1| tolerance, declared a priori


def convergence_class(phi: Sequence[float]) -> str:
    """Classify a three-grid sequence per Celik et al. (2008) table 3.

    ``phi`` is ordered coarse -> medium -> fine.  Returns one of
    ``"monotonic"`` (0 < eps21/eps32 < 1), ``"oscillatory"`` (ratio < 0) or
    ``"divergent"`` (ratio > 1, or a zero difference that makes the order
    undefined).
    """
    if len(phi) != 3:
        raise ValueError("three grids required")
    eps21 = phi[1] - phi[0]
    eps32 = phi[2] - phi[1]
    if eps32 == 0.0 or eps21 == 0.0:
        return "divergent"
    # R = eps32/eps21 (fine pair over coarse pair): a converging sequence has
    # |eps32| < |eps21|, i.e. 0 < R < 1 (Celik table 3 as in Roache (1994),
    # the same convention as ``tensorfvm.ship_freesurface.mesh_convergence``).
    ratio = eps32 / eps21
    if ratio < 0.0:
        return "oscillatory"
    if ratio > 1.0:
        return "divergent"
    return "monotonic"


def observed_order(phi: Sequence[float], r: float) -> float:
    """Celik et al. (2008) eq. (3): ``p = |ln|eps32/eps21|| / ln r``.

    Returns ``nan`` when the sequence is not monotonically convergent: the
    published procedure attributes no order to oscillatory/divergent data.
    """
    if convergence_class(phi) != "monotonic":
        return float("nan")
    eps21 = abs(phi[1] - phi[0])
    eps32 = abs(phi[2] - phi[1])
    if eps21 <= 0.0 or eps32 <= 0.0:
        return float("nan")
    return abs(math.log(eps32 / eps21)) / math.log(r)


def richardson_extrapolated(phi: Sequence[float], r: float, p: float) -> float:
    """Celik et al. (2008) eqs. (5)-(6), generalised to ``r = r21 = r32``.

    ``phi_ext = (r^p phi_fine - phi_medium) / (r^p - 1)``; returns ``nan`` if
    ``p`` is not a positive finite number (no extrapolation is defined).
    """
    if not (math.isfinite(p) and p > 0.0):
        return float("nan")
    rp = r ** p
    return (rp * phi[2] - phi[1]) / (rp - 1.0)


def gci(phi: Sequence[float], r: float, p: float, *,
        which: str = "fine", fs: float = FS_3GRID) -> float:
    """Grid Convergence Index on the ``which`` grid (Celik eq. (7)).

    ``GCI = Fs * |eps| / ((r^p - 1) * |phi|)`` where ``eps`` is the difference
    across the pair that brackets ``which`` and the relative error is taken
    with respect to the fine solution (Celik eq. (4)/(8) form), falling back to
    the absolute error when ``phi_fine`` is zero.
    """
    if not (math.isfinite(p) and p > 0.0):
        return float("nan")
    if which == "fine":
        eps = abs(phi[2] - phi[1])
        ref = phi[2]
    elif which == "medium":
        eps = abs(phi[1] - phi[0])
        ref = phi[1]
    else:  # pragma: no cover - defensive
        raise ValueError("which must be 'fine' or 'medium'")
    e_rel = eps / abs(ref) if abs(ref) > 0.0 else eps
    return fs * e_rel / (r ** p - 1.0)


def asymptotic_ratio(phi: Sequence[float], r: float, p: float,
                     fs: float = FS_3GRID) -> float:
    """Celik et al. (2008) eq. (17): ``GCI_medium / (r^p GCI_fine)``.

    Note the published caveat, which matters here: ``p`` is estimated from the
    *same* three grids, so with one triplet and one refinement ratio this check
    is an identity -- it confirms the arithmetic, not the asymptotic range.  The
    non-vacuous test needs at least four grids (overlapping triplets):
    :func:`triplet_consistency`.
    """
    g_fine = gci(phi, r, p, which="fine", fs=fs)
    g_med = gci(phi, r, p, which="medium", fs=fs)
    if not (math.isfinite(g_fine) and math.isfinite(g_med)) or g_fine == 0.0:
        return float("nan")
    return g_med / (r ** p * g_fine)


def in_asymptotic_range(phi: Sequence[float], r: float, p: float,
                        band: float = ASYMPTOTIC_BAND) -> bool:
    """``True`` iff eq. (17) reproduces one within the declared ``band``."""
    ratio = asymptotic_ratio(phi, r, p)
    return bool(math.isfinite(ratio) and abs(ratio - 1.0) <= band)


def loglog_exponent(pairs: Sequence[tuple[float, float]]) -> Dict[str, object]:
    """Uncertainty-free log-log least squares, i.e. the v2 estimator.

    ``pairs`` are ``(h, error)``.  Reported here so the cross-validation can
    quantify how far the v2 estimator is from the reference on identical data.
    Returns ``exponent``/``valid``/``reason``; ``valid`` is ``False`` when the
    sequence is not monotonically decreasing (a least-squares slope of a
    non-monotone error sequence is not an order of accuracy).
    """
    h = [p[0] for p in pairs]
    e = [p[1] for p in pairs]
    if len(pairs) < 2:
        return {"exponent": float("nan"), "valid": False, "reason": "too few grids"}
    monotone = all(b < a for a, b in zip(e, e[1:])) or all(b > a for a, b in zip(e, e[1:]))
    n = len(pairs)
    lx = [math.log(hi) for hi in h]
    ly = [math.log(ei) if ei > 0 else float("nan") for ei in e]
    if any(not math.isfinite(v) for v in ly):
        return {"exponent": float("nan"), "valid": False, "reason": "non-positive error"}
    mx = sum(lx) / n
    my = sum(ly) / n
    sxx = sum((v - mx) ** 2 for v in lx)
    sxy = sum((v - mx) * (w - my) for v, w in zip(lx, ly))
    slope = sxy / sxx if sxx > 0 else float("nan")
    return {"exponent": slope, "valid": bool(monotone and slope > 0),
            "reason": "" if monotone else "error sequence not monotone"}


def crosscheck_series(values: Sequence[float], h: Sequence[float], *,
                      fs: float = FS_3GRID,
                      band: float = ASYMPTOTIC_BAND) -> Dict[str, object]:
    """Full reference diagnosis of the finest three grids of ``values``.

    ``values`` are ordered coarse -> fine and assumed uniformly refined
    (``r = h[-2]/h[-1] = h[-3]/h[-2]``); the realised ratio is taken from
    ``h`` and its deviation from uniformity is reported, not silently assumed.
    """
    if len(values) < 3:
        raise ValueError("at least three grids required")
    phi = [float(v) for v in values[-3:]]
    hh = [float(v) for v in h[-3:]]
    r21 = hh[1] / hh[2]
    r32 = hh[0] / hh[1]
    r = r21
    cls = convergence_class(phi)
    p = observed_order(phi, r)
    ext = richardson_extrapolated(phi, r, p)
    out: Dict[str, object] = {
        "phi_coarse_medium_fine": phi,
        "h_coarse_medium_fine": hh,
        "r21": r21, "r32": r32, "r_uniformity": abs(r21 - r32) / max(r21, r32),
        "convergence_class": cls,
        "observed_order": p,
        "extrapolated": ext,
        "gci_fine": gci(phi, r, p, which="fine", fs=fs),
        "gci_medium": gci(phi, r, p, which="medium", fs=fs),
        "asymptotic_ratio": asymptotic_ratio(phi, r, p, fs=fs),
        "asymptotic_band": band,
        "in_asymptotic_range": in_asymptotic_range(phi, r, p, band=band),
        "fs": fs,
    }
    if math.isfinite(ext) and abs(ext) > 0.0:
        out["estimated_relative_error_fine"] = abs(ext - phi[2]) / abs(ext)
    else:
        out["estimated_relative_error_fine"] = float("nan")
    out["loglog_v2_estimator"] = loglog_exponent(list(zip(hh, [abs(v) for v in phi])))
    out["limit_form"] = limit_form(values, h, p)
    out["gci_absolute_fine"] = gci_absolute(values, r, p, which="fine", fs=fs)
    out["triplet_consistency"] = (triplet_consistency(values, h, fs=fs, band=band)
                                  if len(values) >= 4 else None)
    return out


if __name__ == "__main__":  # pragma: no cover - smoke check
    # analytic self-test: phi = 1 + C h^2 on r = 2 must give p = 2 exactly
    hs = [0.04, 0.02, 0.01]
    phi = [1.0 + 0.5 * h ** 2 for h in hs]
    res = crosscheck_series(phi, hs)
    print("class", res["convergence_class"], "p", res["observed_order"],
          "ext", res["extrapolated"], "asym", res["asymptotic_ratio"])

def gci_absolute(phi: Sequence[float], r: float, p: float, *,
                 which: str = "fine", fs: float = FS_3GRID) -> float:
    """Absolute (non-relative) GCI: ``Fs |eps| / (r^p - 1)``.

    This is the form to use when the functional converges to zero (an error
    norm, for instance): dividing by the fine value would turn the band into a
    percentage of a vanishing quantity and the eq. (17) ratio into ``r^-p``
    for a perfectly asymptotic sequence.
    """
    if not (math.isfinite(p) and p > 0.0) or r <= 1.0:
        return float("nan")
    if which == "fine":
        eps = abs(phi[2] - phi[1])
    elif which == "medium":
        eps = abs(phi[1] - phi[0])
    else:  # pragma: no cover - defensive
        raise ValueError("which must be 'fine' or 'medium'")
    return fs * eps / (r ** p - 1.0)


def limit_form(values: Sequence[float], h: Sequence[float], p: float) -> str:
    """Whether the functional converges to zero (only absolute GCI is meaningful).

    ``"zero_limit"`` when the extrapolated limit is under 10 % of the fine value
    in magnitude, ``"nonzero_limit"`` otherwise, ``"undefined"`` when no order
    is available.  The distinction decides which GCI form is reported: the
    relative form of Roache/Celik is a percentage of the fine value and becomes
    meaningless as that value vanishes.
    """
    p3 = [float(v) for v in values[-3:]]
    if len(p3) < 3 or not (math.isfinite(p) and p > 0.0):
        return "undefined"
    r = float(h[-2]) / float(h[-1])
    ext = richardson_extrapolated(p3, r, p)
    if not math.isfinite(ext) or abs(p3[-1]) == 0.0:
        return "undefined"
    return "zero_limit" if abs(ext) <= 0.10 * abs(p3[-1]) else "nonzero_limit"


def triplet_consistency(values: Sequence[float], h: Sequence[float], *,
                        fs: float = FS_3GRID, band: float = ASYMPTOTIC_BAND) -> Dict[str, object]:
    """Overlapping-triplet consistency, the non-vacuous asymptotic-range test.

    With four grids ``phi1..phi4`` two independent triplets exist; a sequence in
    the asymptotic range gives the same observed order from both.  Their
    relative difference is therefore a real test of the assumption behind both
    estimators (the single-triplet eq. (17) ratio is an identity).
    """
    if len(values) < 4:
        raise ValueError("four grids required for an overlapping-triplet test")
    vals = [float(v) for v in values[-4:]]
    hs = [float(v) for v in h[-4:]]
    p_coarse = observed_order(vals[0:3], hs[1] / hs[2])
    p_fine = observed_order(vals[1:4], hs[2] / hs[3])
    rel = (abs(p_fine - p_coarse) / abs(p_coarse)
           if math.isfinite(p_coarse) and math.isfinite(p_fine) and p_coarse != 0.0
           else float("nan"))
    return {
        "p_triplet_coarse": p_coarse,
        "p_triplet_fine": p_fine,
        "rel_diff": rel,
        "consistent": bool(math.isfinite(rel) and rel <= band),
        "class_coarse": convergence_class(vals[0:3]),
        "class_fine": convergence_class(vals[1:4]),
        "band": band,
        "phi_coarse_to_fine": vals,
        "h_coarse_to_fine": hs,
    }
