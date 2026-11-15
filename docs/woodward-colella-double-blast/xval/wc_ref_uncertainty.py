"""Reference-side uncertainty budget for the certified WENO5 reference (W&C double blast).

Why this script exists
----------------------
``wc_certified_ref.py`` closes the round-2 anchor with three gates whose first
gate is a **two-level** self-convergence check: WENO5@6400 vs WENO5@12800 must
agree to <1 %.  That gate can only ever produce a *lower* bound on the
reference's own error -- with two grids there is neither an observed order nor
an error bar, so the strongest statement available from it is the triangle
bound

    E_own(MUSCL) <= E_measured(MUSCL, WENO5@12800) + d(WENO5@6400, WENO5@12800)

which, plugged with the round-2 numbers (0.893 % + 0.544 %), is 1.44 % > 1 %.
In other words: **the "<1 %" verdict cannot be established from a two-level
reference check at all** -- it is a point estimate, not a certificate.

This script removes that ambiguity with the data already in the repository: the
WENO5 family has a three-level ladder (3200 / 6400 / 12800), which yields an
observed convergence order ``p_ref`` and hence a Richardson/GCI error bar
``eps_ref`` for the 12800 reference itself, plus a Richardson-extrapolated
reference *field* ``rho_inf`` (the best available stand-in for the exact
solution).  With that, three independent statements can be made and compared:

1. **Cross-family**: MUSCL/HLLC vs WENO5@12800 (the round-2 gate number).
2. **Reference-corrected**: MUSCL/HLLC vs ``rho_inf`` -- the part of the residual
   that is *ours*, with the reference-side error removed by extrapolation.
3. **Intra-family**: MUSCL/HLLC's own Richardson estimate
   ``d(MUSCL@nx/2, MUSCL@nx)/(2^p_m - 1)`` -- no reference involved at all.

If (2) and (3) agree inside ``eps_ref``, the "<1 %" verdict is a genuine
measurement rather than an artefact of the reference; if they do not, the
verdict has to be downgraded to the honest interval.  Nothing here calls a
solver: everything is numpy on the committed profiles, so the whole budget
re-runs in seconds and every number in the report is reproducible.

Usage::

    PYTHONPATH=src python docs/woodward-colella-double-blast/xval/wc_ref_uncertainty.py
    PYTHONPATH=src python .../wc_ref_uncertainty.py --json out.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                          # docs/woodward-colella-double-blast
DATA = ROOT / "ladder" / "data"
ARTIFACTS = ROOT / "artifacts"

#: the three-level WENO5 ladder (grid doubling between levels, so the observed
#: order is ``log2(d_coarse / d_fine)`` with no cell-size arithmetic)
WENO5_LADDER = (3200, 6400, 12800)
#: graded TensorFVM family (MUSCL-Hancock + MC + HLLC + SSP-RK, CFL 0.4)
MUSCL_LADDER = (800, 1600, 3200, 6400)
#: the <1 % bar of the round-2 closure
CERTIFIED_PERCENT = 1.0
#: GCI-style safety factor (ASME V&V-20 recommends 1.25 for three grids)
SAFETY_FACTOR = 1.25
#: plausible band for an observed order; outside it the estimate is flagged
ORDER_BAND = (0.2, 6.0)


def _certified_mod():
    """Import the sibling ``wc_certified_ref`` module (for ``_load``/metrics)."""
    spec = importlib.util.spec_from_file_location("wc_certified_ref", HERE / "wc_certified_ref.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("wc_certified_ref", mod)
    spec.loader.exec_module(mod)
    return mod


WCR = _certified_mod()
WINDOW = WCR.WINDOW


# --------------------------------------------------------------------------- #
# distances: always integrate on the FINER grid so that the fine profile is
# never smeared by interpolation (the direction matters at shocks -- see
# ``interp_sensitivity`` below).
# --------------------------------------------------------------------------- #
def _load(nx: int, family: str) -> dict | None:
    """Load a ladder rung, or ``None`` when the rung has not been computed."""
    try:
        p = WCR._load(f"nx{nx}_{family}")
    except FileNotFoundError:
        return None
    p["nx"] = nx
    p["family"] = family
    return p


def l1_percent(fine: dict, coarse: dict) -> float:
    """Relative L1 distance (percent) on the comparison window, integrated on
    the *fine* grid with the coarse profile linearly interpolated onto it."""
    mask = (fine["x"] >= WINDOW[0]) & (fine["x"] <= WINDOW[1])
    x = fine["x"][mask]
    a = fine["rho"][mask]
    b = np.interp(x, coarse["x"], coarse["rho"])
    den = float(np.trapezoid(np.abs(a), x))
    if den <= 0.0:
        return float("nan")
    return 100.0 * float(np.trapezoid(np.abs(a - b), x)) / den


def peak_percent(fine: dict, coarse: dict) -> float:
    """Relative peak-density difference (percent) on the window."""
    mask = (fine["x"] >= WINDOW[0]) & (fine["x"] <= WINDOW[1])
    x = fine["x"][mask]
    a = float(np.max(fine["rho"][mask]))
    b = float(np.max(np.interp(x, coarse["x"], coarse["rho"])))
    return 100.0 * abs(a - b) / b if b else float("nan")


def peak_on_window(profile: dict) -> float:
    """Maximum density inside the comparison window (no interpolation)."""
    m = (profile["x"] >= WINDOW[0]) & (profile["x"] <= WINDOW[1])
    return float(np.max(profile["rho"][m]))


def interp_sensitivity(fine: dict, coarse: dict) -> float:
    """How much the L1 distance moves when integrated on the coarse grid instead.

    A cheap, honest floor on the interpolation error of the whole budget: if the
    two directions disagree by a sizeable fraction of the distance itself, the
    number is interpolation-limited and must be quoted as such.
    """
    d_fine = l1_percent(fine, coarse)
    # integrate on the coarse grid instead: fine profile sampled at coarse nodes
    mask = (coarse["x"] >= WINDOW[0]) & (coarse["x"] <= WINDOW[1])
    x = coarse["x"][mask]
    a = coarse["rho"][mask]
    b = np.interp(x, fine["x"], fine["rho"])
    den = float(np.trapezoid(np.abs(b), x))
    d_coarse = 100.0 * float(np.trapezoid(np.abs(a - b), x)) / den if den > 0 else float("nan")
    return abs(d_fine - d_coarse)


def observed_order(d_coarse: float, d_fine: float, ratio: float = 2.0) -> tuple[float, str]:
    """Observed order from two successive-grid distances (``ratio`` = refinement).

    ``d`` is proportional to ``h**p``, so ``p = log(dc/df)/log(ratio)``.  A
    non-physical value (non-positive, or outside :data:`ORDER_BAND`) is replaced
    by the first-order fallback and flagged -- a pure L1-on-shocks measurement
    can land anywhere, and silently quoting 7.3 would be dishonest.
    """
    if not (d_coarse > 0.0 and d_fine > 0.0 and math.isfinite(d_coarse) and math.isfinite(d_fine)):
        return 1.0, "non-positive distance -> first-order fallback"
    p = math.log(d_coarse / d_fine) / math.log(ratio)
    if not (ORDER_BAND[0] <= p <= ORDER_BAND[1]):
        return 1.0, f"observed order {p:.3f} outside {ORDER_BAND} -> first-order fallback"
    return p, "none"


def richardson_error(d_fine: float, order: float, ratio: float = 2.0) -> float:
    """Discretisation-error estimate of the *fine* member of a pair (percent)."""
    den = ratio ** order - 1.0
    return d_fine / den if den > 0 else float("nan")


def extrapolate_field(fine: dict, coarse: dict, order: float) -> dict:
    """Richardson-extrapolated field on the fine grid: ``rho_inf = rho_f + (rho_f - rho_c)/(r-1)``."""
    x = fine["x"]
    rho_c = np.interp(x, coarse["x"], coarse["rho"])
    den = 2.0 ** order - 1.0
    rho_inf = fine["rho"] + (fine["rho"] - rho_c) / den
    return {"x": x, "rho": rho_inf}


def _distance_percent(fine: dict, ref: dict) -> float:
    """L1 distance between two profiles already living on comparable grids."""
    return l1_percent({**fine}, {"x": np.asarray(ref["x"], float),
                                 "rho": np.asarray(ref["rho"], float)})


# --------------------------------------------------------------------------- #
# the budget
# --------------------------------------------------------------------------- #
def _published_diagnostic(fine: dict, rho_inf: dict) -> dict:
    """Distance of the finest WENO5 field and of its extrapolation to the
    digitised published curve.

    This is *not* used as a validation anchor: the published curve is a PDF
    digitisation of a 3200-cell reference and carries a coordinate/axis bias
    (see ``reference/axis_calibration.json``), so its residual does not shrink
    with our refinement.  Recording the two numbers makes that explicit instead
    of letting the reader assume the published curve is the truth.
    """
    from tensorfvm.compressible import wc_reference as wcr

    try:
        pub = wcr.load_reference(calibrated=True)
    except Exception as exc:                      # pragma: no cover - data guard
        return {"status": "UNAVAILABLE", "reason": str(exc)}
    pubp = {"x": np.asarray(pub["x"], float), "rho": np.asarray(pub["rho"], float)}
    return {
        "finest_weno5_vs_published_percent": _distance_percent(fine, pubp),
        "extrapolated_vs_published_percent": _distance_percent(rho_inf, pubp),
        "note": ("published curve is a digitisation of a 3200-cell reference with a "
                 "known additive y-calibration; its residual does not vanish with our "
                 "refinement, so it bounds nothing"),
    }


def reference_uncertainty() -> dict:
    """Observed order + error bar of the WENO5@12800 reference (three levels)."""
    rungs = [(nx, _load(nx, "weno5")) for nx in WENO5_LADDER]
    present = [(nx, r) for nx, r in rungs if r is not None]
    out: dict = {
        "ladder_requested": list(WENO5_LADDER),
        "ladder_available": [nx for nx, _ in present],
        "successive": [],
        "missing": [nx for nx, r in rungs if r is None],
    }
    if len(present) < 3:
        out["status"] = "INSUFFICIENT_LADDER"
        out["note"] = ("three WENO5 levels are required for an observed order; "
                       "the two-level gate yields no error bar")
        return out

    for (nx_c, rc), (nx_f, rf) in zip(present[:-1], present[1:]):
        out["successive"].append({
            "coarse_nx": nx_c, "fine_nx": nx_f,
            "l1_percent": l1_percent(rf, rc),
            "peak_percent": peak_percent(rf, rc),
            "interp_sensitivity_percent": interp_sensitivity(rf, rc),
        })

    finest_pair = out["successive"][-1]
    prev_pair = out["successive"][-2]
    p_ref, reason = observed_order(prev_pair["l1_percent"], finest_pair["l1_percent"])
    eps_ref = richardson_error(finest_pair["l1_percent"], p_ref)
    eps_ref_gci = SAFETY_FACTOR * eps_ref
    rho_inf = extrapolate_field(present[-1][1], present[-2][1], p_ref)

    # conservation sanity of the extrapolated reference (smooth functional)
    mask = (rho_inf["x"] >= WINDOW[0]) & (rho_inf["x"] <= WINDOW[1])
    mass_inf = float(np.trapezoid(rho_inf["rho"][mask], rho_inf["x"][mask]))
    fine_last = present[-1][1]
    mass_fine = float(np.trapezoid(fine_last["rho"][mask], fine_last["x"][mask]))
    mass_delta_percent = 100.0 * abs(mass_inf - mass_fine) / mass_fine

    # the error bar is metric-dependent: the L1 *field* distance is dominated by
    # shock smearing (order ~1), while the peak value converges much faster, so
    # the peak claim gets a much tighter bar.  Quoting one bar for both would
    # overstate the peak uncertainty and understate the field one.
    p_peak, peak_reason = observed_order(prev_pair["peak_percent"], finest_pair["peak_percent"])
    eps_peak = richardson_error(finest_pair["peak_percent"], p_peak)

    # naive two-level triangle bound (what the existing gate 1 can support)
    two_level_bound = finest_pair["l1_percent"]

    out.update({
        "status": "OK",
        "observed_order": p_ref,
        "order_reason": reason,
        "reference_nx": present[-1][0],
        "l1_percent_two_level": finest_pair["l1_percent"],
        "eps_ref_percent": eps_ref,
        "peak": {
            "observed_order": p_peak,
            "order_reason": peak_reason,
            "l1_percent_two_level": finest_pair["peak_percent"],
            "eps_ref_percent": eps_peak,
        },
        "published_reference_diagnostic": _published_diagnostic(present[-1][1], rho_inf),
        "eps_ref_gci_percent": eps_ref_gci,
        "two_level_triangle_bound_percent": two_level_bound,
        "eps_ref_source": "Richardson on WENO5 successive-grid L1 distances (three levels)",
        "mass_extrapolated_vs_fine_percent": mass_delta_percent,
        "rho_inf": rho_inf,
        "ref_fine": fine_last,
        "cases_available": [nx for nx, _ in present],
    })
    return out


def grade(ref_budget: dict) -> dict:
    """Re-grade every MUSCL/HLLC rung against the reference *and* the
    extrapolated reference; then reconcile the two independent error routes."""
    if ref_budget.get("status") != "OK":
        return {"status": "SKIPPED", "reason": "reference budget unavailable"}
    ref = {"x": ref_budget["ref_fine"]["x"], "rho": ref_budget["ref_fine"]["rho"]}
    rho_inf = ref_budget["rho_inf"]
    eps = ref_budget["eps_ref_percent"]
    peak_ref = peak_on_window(ref)

    graded = {}
    for nx in MUSCL_LADDER:
        p = _load(nx, "hllc")
        if p is None:
            continue
        d_ref = _distance_percent(p, ref)
        d_inf = _distance_percent(p, rho_inf)
        graded[nx] = {
            "l1_vs_reference_percent": d_ref,
            "peak_vs_reference_percent": 100.0 * abs(peak_on_window(p) - peak_ref) / peak_ref,
            "l1_vs_extrapolated_percent": d_inf,
            "own_error_interval_percent": [max(0.0, d_ref - eps), d_ref + eps],
            "band_contains_extrapolated": bool(max(0.0, d_ref - eps) <= d_inf <= d_ref + eps),
        }

    # intra-family Richardson: no reference involved at all
    intra = {}
    avail = sorted(graded)
    for a, b in zip(avail[:-1], avail[1:]):
        if b != 2 * a:
            continue
        intra[f"{a}->{b}"] = {"l1_percent": l1_percent(_load(b, "hllc"), _load(a, "hllc"))}
    intra_order = {}
    keys = list(intra)
    for k0, k1 in zip(keys[:-1], keys[1:]):
        p, reason = observed_order(intra[k0]["l1_percent"], intra[k1]["l1_percent"])
        intra_order[k1] = {"observed_order": p, "reason": reason,
                           "own_error_estimate_percent":
                               richardson_error(intra[k1]["l1_percent"], p)}

    # symmetric cross-family check: extrapolate BOTH families to h -> 0 and
    # compare the two continuum stand-ins (the difference should collapse far
    # below either family's own error if both extrapolations are trustworthy)
    cross = {}
    if len(avail) >= 3 and avail[-1] == 2 * avail[-2] and avail[-2] == 2 * avail[-3]:
        a, b, c = avail[-3], avail[-2], avail[-1]
        pm, pm_reason = observed_order(intra[f"{a}->{b}"]["l1_percent"],
                                       intra[f"{b}->{c}"]["l1_percent"])
        m_inf = extrapolate_field(_load(c, "hllc"), _load(b, "hllc"), pm)
        cross = {
            "graded_nx": c,
            "muscl_observed_order": pm,
            "order_reason": pm_reason,
            "muscl_own_error_estimate_percent":
                richardson_error(intra[f"{b}->{c}"]["l1_percent"], pm),
            # compare on the reference grid (the finer of the two)
            "continuum_vs_continuum_l1_percent": l1_percent(rho_inf, m_inf),
            "note": ("both families Richardson-extrapolated to h->0; the residual "
                     "is the part neither extrapolation removes"),
        }

    finest_nx = max(graded) if graded else None
    verdict = None
    if finest_nx is not None:
        f = graded[finest_nx]
        est = f["l1_vs_extrapolated_percent"]
        upper = est + eps
        key = f"{finest_nx // 2}->{finest_nx}"
        intra_est = intra_order.get(key, {}).get("own_error_estimate_percent")
        routes_agree = (abs(intra_est - est) if intra_est is not None else None)
        # order sensitivity of the intra-family route: p_obs is an L1-on-shocks
        # measurement and tends to 1 in the asymptotic shock regime, so quote
        # both ends instead of trusting the fitted value alone.
        d_pair = intra.get(key, {}).get("l1_percent")
        intra_band = None
        if d_pair is not None:
            p_obs = intra_order.get(key, {}).get("observed_order", 1.0)
            intra_band = sorted([richardson_error(d_pair, p_obs),
                                 richardson_error(d_pair, 1.0)])
        verdict = {
            "graded_nx": finest_nx,
            "route_1_cross_family_percent": est,
            "route_2_intra_family_percent": intra_est,
            "routes_agree_percent": routes_agree,
            "route_2_order_band_percent": intra_band,
            "reference_uncertainty_percent": eps,
            "cross_family_interval_percent": [max(0.0, est - eps), est + eps],
            "honest_upper_bound_percent": upper,
            "two_level_only_bound_percent":
                f["l1_vs_reference_percent"] + ref_budget["l1_percent_two_level"],
            "certified_under_1pct_point_estimate": bool(est < CERTIFIED_PERCENT),
            "certified_under_1pct_provable": bool(upper < CERTIFIED_PERCENT),
            "margin_percent": 100.0 * (CERTIFIED_PERCENT - est) / CERTIFIED_PERCENT,
            "statement": (
                "MUSCL/HLLC @ nx=%d: cross-family (vs Richardson-extrapolated WENO5) "
                "%.3f %%; intra-family Richardson %.3f %%; agreement %.3f %%; "
                "reference side +/-%.3f %% -> honest upper bound %.3f %% -- "
                % (finest_nx, est, intra_est if intra_est is not None else float("nan"),
                   routes_agree if routes_agree is not None else float("nan"), eps, upper))
                + ("<1 %% is provable." if upper < CERTIFIED_PERCENT else
                   "<1 %% holds as a point estimate only, NOT provable from these grids."),
        }
    return {
        "status": "OK",
        "graded": graded,
        "intra_family_successive": intra,
        "intra_family_order": intra_order,
        "cross_family_extrapolated": cross,
        "verdict": verdict,
        "certified_bar_percent": CERTIFIED_PERCENT,
        "safety_factor": SAFETY_FACTOR,
    }


def run() -> dict:
    ref_budget = reference_uncertainty()
    out = {
        "reference": {k: v for k, v in ref_budget.items()
                      if k not in ("rho_inf", "ref_fine")},
        "grading": grade(ref_budget),
        "window": list(WINDOW),
        "certified_bar_percent": CERTIFIED_PERCENT,
        "reference_xval_module": "docs/woodward-colella-double-blast/xval/wc_certified_ref.py",
    }
    out["_fields"] = {"rho_inf": ref_budget.get("rho_inf"),
                      "ref_fine": ref_budget.get("ref_fine")}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--profiles", type=Path, default=None,
                    help="optional npz dump of the extrapolated reference field")
    args = ap.parse_args()
    res = run()
    fields = res.pop("_fields")
    print(json.dumps(res, indent=2))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n")
    if args.profiles and fields.get("rho_inf") is not None:
        args.profiles.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.profiles, x=fields["rho_inf"]["x"],
                            rho=fields["rho_inf"]["rho"])


if __name__ == "__main__":
    main()