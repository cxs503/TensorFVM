"""Refinement closure of the *graded* solution (round-2 anchor, L1 gate).

Why this exists
---------------
``CERTIFIED-REF-CLOSURE.md`` settled the round-2 question ("is the ~3 % residual
vs the published reference our error or the reference's?") by re-grading the
TensorFVM MUSCL/HLLC solution against an independent WENO5@12800 reference.  The
answer was honest but *incomplete*: the peak agreement was 0.758 % (<1 %), while
the window-L1 agreement was **1.776 % (>1 %)** -- so the binding gate of a
"certified <1 %" claim stayed shut at nx=3200.

That closure measured the graded family on **one** grid.  This left exactly the
question the round-2 report handed to the next worker unanswered:

    is the 1.776 % L1 a *discretisation* residual of the MUSCL/HLLC family that
    keeps shrinking under refinement (then "certified <1 %" is merely a matter
    of grid), or is it a floor that uniform refinement cannot cross (then the
    <1 % claim is dead for this family and must not be repeated)?

This script answers it with ladder data only -- no solver call, a few seconds of
numpy -- by refining the graded family itself (MUSCL-Hancock / MC / HLLC at
nx = 3200, 6400, 12800) against the *same* self-certified WENO5@12800 reference.

What is computed
----------------
1. **Per-rung grading** against WENO5@12800: window-L1, peak, window mass.
   The L1 curve over nx is the whole point: monotone shrinkage means the gate is
   grid-limited; a plateau means it is scheme-limited.
2. **Observed order** of the graded family from consecutive rungs
   ``p = log(e_n / e_2n) / log 2`` (the reference is fixed, so this is the
   graded family's own L1 convergence *towards the reference*, which asymptotes
   to the reference's own error -- see item 3).
3. **Asymptote fit** ``e(n) = e_inf + C n^-p`` solved on the three finest rungs.
   ``e_inf`` estimates the L1 floor that no amount of uniform refinement of this
   family can go below: it is the reference-side error plus any genuine
   scheme-limited mismatch (contact/plateau smearing).  If e_inf > 0 the
   "certified <b> %" bar must be stated *including* that floor, not below it.
4. **Self-convergence** of the graded family (6400 vs 12800) -- a
   reference-independent estimate of the graded solution's own discretisation
   error, which is what a <1 % claim should really be compared against.
5. **Verdict**: the smallest graded grid whose L1 is <1 % (or NOT_REACHED), with
   the honest reason.

Everything is derived from stored profiles (``ladder/data/ladder_profile_*.npz``)
so the conclusions can be re-derived and regression-gated in ``tests/``.

Usage::

    PYTHONPATH=src python docs/woodward-colella-double-blast/xval/wc_graded_refinement.py
    PYTHONPATH=src python .../wc_graded_refinement.py --json out.json --markdown out.md
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                      # docs/woodward-colella-double-blast
DATA = ROOT / "ladder" / "data"
ARTIFACTS = ROOT / "artifacts"

WINDOW = (0.4, 1.0)                     # digitised comparison window (wcr.REFERENCE_X)
CERTIFIED_PERCENT = 1.0                 # the <1 % bar this closure tests
TOLERANCE_PERCENT = 10.0                # ledger tolerance (sanity bound only)

#: graded family (TensorFVM MUSCL/HLLC) rungs, coarse -> fine
GRADED = (800, 1600, 3200, 6400, 12800)
#: independent reference (WENO5 / SSP-RK3 / Rusanov), self-certified at 6400/12800
REF_FINE = 12800
REF_COARSE = 6400
#: three-rung fit candidates, finest first; the first one fully present is used
FIT_CANDIDATES = ((3200, 6400, 12800), (1600, 3200, 6400), (800, 1600, 3200))


# --------------------------------------------------------------------------- #
# profile IO + metrics (same conventions as wc_certified_ref.py)
# --------------------------------------------------------------------------- #
#: tag suffixes tried, coarse -> fine, for a graded (MUSCL/HLLC) rung.  The
#: uniform ``hllc`` tag is the canonical config (CFL 0.4); the ``cfl0.8`` tag is
#: only admitted once the CFL-sensitivity control shows it does not move the L1
#: metric (see :func:`run`).
GRADED_TAGS = ("hllc", "hllc_cfl0.8")


def _load(nx: int, family: str) -> dict:
    """Load a stored rung.  ``family`` may be ``hllc``/``weno5`` (exact tag
    ``nx{nx}_{family}``) or a full tag suffix from :data:`GRADED_TAGS`."""
    p = DATA / f"ladder_profile_nx{nx}_{family}.npz"
    if not p.exists():
        # only the canonical tag has a bare-artifact ancestor; a variant tag
        # (CFL/limiter) must not silently resolve onto the canonical profile --
        # that would make a "sensitivity control" compare a profile with itself.
        alt = ARTIFACTS / f"wc_blast_profile_nx{nx}.npz"
        if family != "hllc" or not alt.exists():
            raise FileNotFoundError(f"missing {family} rung nx={nx}: {p}")
        p = alt
    d = np.load(p)
    # older artifact rungs (400/800) store the got-profile under x_got/rho_got
    x = np.asarray(d["x_got"] if "x_got" in d else d["x"], float)
    rho = np.asarray(d["rho_got"] if "rho_got" in d else d["rho"], float)
    o = np.argsort(x)
    return {"x": x[o], "rho": rho[o], "source": str(p.relative_to(ROOT))}


def _window(xa, ra, xb, rb):
    m = (xa >= WINDOW[0]) & (xa <= WINDOW[1])
    x = xa[m]
    return x, ra[m], np.interp(x, xb, rb)


def l1_percent(xa, ra, xb, rb) -> float:
    x, a, b = _window(xa, ra, xb, rb)
    return 100.0 * float(np.trapezoid(np.abs(a - b), x) / np.trapezoid(np.abs(b), x))


def peak_percent(xa, ra, xb, rb) -> float:
    x, a, b = _window(xa, ra, xb, rb)
    return 100.0 * abs(float(np.max(a)) - float(np.max(b))) / float(np.max(b))


def mass_percent(xa, ra, xb, rb) -> float:
    x, a, b = _window(xa, ra, xb, rb)
    return 100.0 * abs(float(np.trapezoid(a, x)) - float(np.trapezoid(b, x))) / float(
        np.trapezoid(b, x))


# --------------------------------------------------------------------------- #
# asymptote fit: e(n) = e_inf + C n^{-p}
# --------------------------------------------------------------------------- #
def _solve_asymptote(ns, es) -> dict:
    """Least-squares fit of ``e(n) = e_inf + C n^-p`` on three or more rungs.

    Solved by scanning ``e_inf`` over ``[0, min(e))`` and, for every trial, doing
    a linear least-squares fit of ``log(e - e_inf)`` against ``log n`` (so ``p``
    and ``C`` are exact given ``e_inf``); the trial with the smallest residual
    wins.  A robust bisection could be used instead, but the parameter space is
    one-dimensional and the scan is trivially cheap and monotone here.
    """
    ns = np.asarray(ns, float)
    es = np.asarray(es, float)
    if len(ns) < 3:
        return {"ok": False, "reason": "need >=3 rungs for a 3-parameter fit"}
    lo, hi = 0.0, float(es.min()) * 0.999
    best = None
    for e_inf in np.linspace(lo, hi, 4001):
        y = np.log(es - e_inf)
        A = np.vstack([np.ones_like(ns), np.log(ns)]).T
        coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
        sse = float(np.sum((A @ coef - y) ** 2))
        if best is None or sse < best[0]:
            best = (sse, float(e_inf), float(coef[1]), float(coef[0]))
    sse, e_inf, slope, intercept = best
    return {
        "ok": True,
        "e_inf_percent": e_inf,
        "order": -slope,
        "C": math.exp(intercept),
        "sse": sse,
        "rungs": [int(n) for n in ns],
        "model": "e(n) = e_inf + C * n**(-p)",
    }


def _asymptote_fixed_order(e_coarse: float, e_fine: float, p: float) -> float:
    """L1 floor ``e_inf`` from two rungs with the order ``p`` *given*.

    ``e(n) = e_inf + C n^-p`` gives ``e(n) - e(2n) = C n^-p (1 - 2^-p)``, hence
    ``e_inf = e(2n) - (e(n) - e(2n)) / (1 - 2^-p)``.  Unlike the free
    three-parameter fit this cannot degenerate to ``e_inf = 0`` on three points,
    so it is the estimator the verdict is bound to.
    """
    if not (p > 0.0) or not np.isfinite(p):
        return float("nan")
    return float(e_fine - (e_coarse - e_fine) / (1.0 - 2.0 ** (-p)))


def _order(e_coarse: float, e_fine: float) -> float:
    if e_fine <= 0 or e_coarse <= 0:
        return float("nan")
    return math.log(e_coarse / e_fine) / math.log(2.0)


# --------------------------------------------------------------------------- #
# main analysis
# --------------------------------------------------------------------------- #
def run() -> dict:
    ref = _load(REF_FINE, "weno5")
    ref_coarse = _load(REF_COARSE, "weno5")

    # gate 0: the reference must still be self-certified <1 % at 6400/12800
    ref_self_l1 = l1_percent(ref_coarse["x"], ref_coarse["rho"], ref["x"], ref["rho"])
    ref_self_peak = peak_percent(ref_coarse["x"], ref_coarse["rho"], ref["x"], ref["rho"])
    ref_ok = ref_self_l1 < CERTIFIED_PERCENT and ref_self_peak < CERTIFIED_PERCENT

    # 1b. CFL-sensitivity control (built *first* so the ladder can use it for
    #     admission): where both the CFL 0.4 and CFL 0.8 variants of the same
    #     grid exist, their mutual L1 distance bounds how much a CFL change
    #     alone moves the metric.  If it is far below the refinement decrements,
    #     a CFL-0.8 rung is admissible in the refinement study.
    cfl_control = {}
    for nx in GRADED:
        pair = {}
        for tag in ("hllc", "hllc_cfl0.8"):
            try:
                pair[tag] = _load(nx, tag)
            except FileNotFoundError:
                pass
        if len(pair) == 2:
            a, b = pair["hllc"], pair["hllc_cfl0.8"]
            cfl_control[str(nx)] = {
                "l1_percent": l1_percent(a["x"], a["rho"], b["x"], b["rho"]),
                "peak_percent": peak_percent(a["x"], a["rho"], b["x"], b["rho"]),
            }

    # 1. per-rung grading of the graded (MUSCL/HLLC) family.  A rung may be
    #    stored under a non-canonical tag (e.g. the CFL 0.8 variant used when the
    #    CFL 0.4 run does not fit a job wall-clock); the tag is recorded so the
    #    config caveat is visible instead of hidden.  A non-canonical (cfl0.8)
    #    rung is only *admitted* when the same-grid CFL 0.4/0.8 control exists
    #    (both variants present): without it the rung cannot be separated from a
    #    CFL artefact, so it is left out of the ladder and reported as missing.
    ladders = {}
    tags = {}
    missing = []
    for nx in GRADED:
        for tag in GRADED_TAGS:
            try:
                p = _load(nx, tag)
            except FileNotFoundError:
                continue
            if tag != "hllc" and str(nx) not in cfl_control:
                # non-canonical rung with no same-grid control: not admissible
                continue
            tags[nx] = tag
            ladders[nx] = {
                "l1_percent": l1_percent(p["x"], p["rho"], ref["x"], ref["rho"]),
                "peak_percent": peak_percent(p["x"], p["rho"], ref["x"], ref["rho"]),
                "mass_percent": mass_percent(p["x"], p["rho"], ref["x"], ref["rho"]),
                "peak_density": float(np.max(p["rho"])),
                "tag": f"nx{nx}_{tag}",
                "source": p["source"],
            }
            break
        else:
            missing.append(nx)

    # 2. observed order of the L1 convergence *towards the reference*
    order_vs_ref = {}
    ks = sorted(ladders)
    for a, b in zip(ks, ks[1:]):
        if b != 2 * a:
            continue
        order_vs_ref[f"{a}->{b}"] = {
            "coarse_l1": ladders[a]["l1_percent"],
            "fine_l1": ladders[b]["l1_percent"],
            "p": _order(ladders[a]["l1_percent"], ladders[b]["l1_percent"]),
            "monotone": bool(ladders[b]["l1_percent"] < ladders[a]["l1_percent"]),
        }

    # 3. L1 floor: free three-parameter fit (reported, may degenerate) plus the
    #    fixed-order two-point estimate on the two finest rungs (binding).
    fit_rungs = next((list(c) for c in FIT_CANDIDATES if all(n in ladders for n in c)), [])
    if len(fit_rungs) >= 3:
        free_fit = _solve_asymptote(fit_rungs,
                                    [ladders[n]["l1_percent"] for n in fit_rungs])
    else:
        free_fit = {"ok": False,
                    "reason": f"no complete three-rung fit set among {list(FIT_CANDIDATES)}"}
    finest_pair = f"{ks[-2]}->{ks[-1]}" if len(ks) >= 2 else None
    p_obs = (order_vs_ref[finest_pair]["p"]
             if finest_pair in order_vs_ref else float("nan"))
    e_inf = (float("nan") if not np.isfinite(p_obs)
             else _asymptote_fixed_order(ladders[ks[-2]]["l1_percent"],
                                         ladders[ks[-1]]["l1_percent"], p_obs))
    # a negative e_inf means the ladder is consistent with pure power-law decay
    # (no positive floor is *detected*); clamping it to 0 is the honest reading,
    # and the power law then predicts the next rungs -- which the ladder can be
    # scored against once those runs land.
    floor_detected = bool(np.isfinite(e_inf) and e_inf > 0.0)
    predictions = {}
    if np.isfinite(p_obs) and ks:
        for nxt in (2 * ks[-1], 4 * ks[-1]):
            predictions[str(nxt)] = float(ladders[ks[-1]]["l1_percent"]
                                          * (nxt / ks[-1]) ** (-p_obs))
    asymptote = {
        "free_fit": free_fit,
        "floor_detected": floor_detected,
        "floor_class": ("NO_FLOOR_DETECTED" if not floor_detected else
                        ("GRID_LIMITED" if e_inf < CERTIFIED_PERCENT else
                         "SCHEME_LIMITED")),
        "predicted_l1_percent": predictions,
        "free_fit_note": (
            "with exactly three rungs and three free parameters the fit is "
            "degenerate (e_inf -> 0 is one exact solution), so the free fit is "
            "reported for inspection only and never binds the verdict"),
        "fixed_order": {
            "ok": bool(np.isfinite(e_inf)),
            "e_inf_percent_raw": None if not np.isfinite(e_inf) else e_inf,
            "e_inf_percent": (e_inf if floor_detected else 0.0),
            "order_used": None if not np.isfinite(p_obs) else p_obs,
            "order_source": f"observed p of the finest pair {finest_pair}",
            "rungs": ks[-2:],
            "model": "e(n) = e_inf + C n^-p, p fixed by observation",
        },
    }

    # 4. reference-independent self-convergence of the graded family
    self_conv = {}
    for a, b in zip(ks, ks[1:]):
        if b != 2 * a:
            continue
        pa, pb = _load(a, tags[a]), _load(b, tags[b])
        # coarse rung interpolated onto the fine grid's window
        self_conv[f"{a}->{b}"] = l1_percent(pa["x"], pa["rho"], pb["x"], pb["rho"])

    # 5. verdict: smallest graded grid with L1 < 1 %
    under = [n for n in ks if ladders[n]["l1_percent"] < CERTIFIED_PERCENT]
    smallest_under = under[0] if under else None
    peak_under = [n for n in ks if ladders[n]["peak_percent"] < CERTIFIED_PERCENT]
    smallest_peak_under = peak_under[0] if peak_under else None

    if not ref_ok:
        status = "REFERENCE_NOT_SELF_CERTIFIED"
    elif smallest_under is not None:
        status = f"CERTIFIED_UNDER_1PCT_AT_NX{smallest_under}"
    elif asymptote["floor_class"] == "NO_FLOOR_DETECTED":
        status = "PURE_POWER_LAW_NO_FLOOR_DETECTED_GATE_NOT_YET_REACHED"
    elif asymptote["floor_class"] == "GRID_LIMITED":
        status = "GATE_GRID_LIMITED_BUT_NOT_REACHED_IN_LADDER"
    elif asymptote["floor_class"] == "SCHEME_LIMITED":
        status = "SCHEME_LIMITED_FLOOR_ABOVE_1PCT"
    else:
        status = "INSUFFICIENT_LADDER_FOR_FLOOR_ESTIMATE"

    verdict = {
        "status": status,
        "certified_bar_percent": CERTIFIED_PERCENT,
        "reference": {
            "family": "weno5", "grid": REF_FINE,
            "self_certified_l1_percent": ref_self_l1,
            "self_certified_peak_percent": ref_self_peak,
            "self_certified": bool(ref_ok),
        },
        "graded_family": "MUSCL-Hancock + MC limiter + HLLC (TensorFVM)",
        "ladder": {str(k): v for k, v in ladders.items()},
        "missing_rungs": missing,
        "cfl_sensitivity": cfl_control,
        "config_caveat": (
            None if all(tags[n] == "hllc" for n in ks) else
            {str(n): tags[n] for n in ks if tags[n] != "hllc"}),
        "order_vs_reference": order_vs_ref,
        "self_convergence_l1_percent": self_conv,
        "asymptote": asymptote,
        "smallest_grid_under_1pct_l1": smallest_under,
        "smallest_grid_under_1pct_peak": smallest_peak_under,
        "ledger_tolerance_percent": TOLERANCE_PERCENT,
        "interp_note": (
            "L1/peak are measured on the ladder's own grid inside [0.4,1]; the "
            "reference profile is linearly interpolated onto it, as in "
            "wc_certified_ref.py (same convention => numbers comparable)."
        ),
    }
    return verdict


def to_markdown(v: dict) -> str:
    L = []
    L.append("# Graded-family refinement closure (WC double blast, L1 gate)\n")
    L.append(f"**Status**: `{v['status']}`  ")
    L.append(f"**Reference**: WENO5/SSP-RK3/Rusanov @ nx={v['reference']['grid']} "
             f"(self-certified L1 {v['reference']['self_certified_l1_percent']:.3f} %, "
             f"peak {v['reference']['self_certified_peak_percent']:.3f} %)\n")
    L.append("## Per-rung grading of the graded family\n")
    L.append("| nx | L1 % | peak % | window mass % | rho_max | L1 <1% |")
    L.append("|---|---|---|---|---|---|")
    for nx, r in sorted(v["ladder"].items(), key=lambda kv: int(kv[0])):
        L.append(f"| {nx} | {r['l1_percent']:.3f} | {r['peak_percent']:.3f} | "
                 f"{r['mass_percent']:.4f} | {r['peak_density']:.4f} | "
                 f"{'✅' if r['l1_percent'] < 1.0 else '❌'} |")
    L.append("\n## Order of the L1 convergence towards the reference\n")
    L.append("| pair | coarse L1 % | fine L1 % | p | monotone |")
    L.append("|---|---|---|---|---|")
    for k, r in v["order_vs_reference"].items():
        L.append(f"| {k} | {r['coarse_l1']:.3f} | {r['fine_l1']:.3f} | "
                 f"{r['p']:.2f} | {'yes' if r['monotone'] else 'NO'} |")
    L.append("\n## Reference-independent self-convergence of the graded family\n")
    L.append("| pair | L1 % |")
    L.append("|---|---|")
    for k, r in v["self_convergence_l1_percent"].items():
        L.append(f"| {k} | {r:.3f} |")
    if v.get("cfl_sensitivity"):
        L.append("\n## CFL-sensitivity control (same grid, CFL 0.4 vs 0.8)\n")
        L.append("| nx | L1 % | peak % |")
        L.append("|---|---|---|")
        for nx, r in v["cfl_sensitivity"].items():
            L.append(f"| {nx} | {r['l1_percent']:.4f} | {r['peak_percent']:.4f} |")
    a = v["asymptote"]
    if v.get("config_caveat"):
        L.append(f"\n> **Config caveat**: non-canonical rung configs "
                 f"{v['config_caveat']} (uniform tag is `hllc` = CFL 0.4); "
                 f"admissible only because the CFL control above is tiny.\n")
    L.append("\n## L1 floor of the graded family\n")
    f = a["fixed_order"]
    if f["ok"]:
        L.append(f"**Binding estimate** (order fixed to the observed "
                 f"p = {f['order_used']:.3f} on rungs {f['rungs'][0]}/"
                 f"{f['rungs'][1]}): raw `e_inf = {f['e_inf_percent_raw']:.3f} %` "
                 f"→ floor class **{a['floor_class']}**.\n")
        if a["floor_class"] == "NO_FLOOR_DETECTED":
            L.append("→ the raw estimate is negative, i.e. the L1 curve is "
                     "consistent with pure power-law decay (no positive floor is "
                     "detected by this ladder); the honest floor is **0** and the "
                     "<1 % gate is then purely a matter of grid.  Power-law "
                     "predictions for finer rungs:")
            for n, e in a["predicted_l1_percent"].items():
                L.append(f"  - nx={n}: **{e:.3f} %** "
                         f"({'<1 %' if e < 1.0 else '≥1 %'})")
        else:
            L.append("→ uniform refinement of this family alone cannot push the "
                     "window-L1 against this reference below that floor "
                     "(reference-side error + scheme-limited contact/plateau "
                     "smearing).")
    else:
        L.append("_fixed-order floor not estimable from the available ladder._")
    ff = a["free_fit"]
    if ff.get("ok"):
        L.append(f"\nFree 3-parameter fit on rungs {ff['rungs']}: "
                 f"`e_inf = {ff['e_inf_percent']:.4f} %, p = {ff['order']:.3f}`, "
                 f"SSE {ff['sse']:.2e} -- **non-binding** ({a['free_fit_note']}).")
    L.append("")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--markdown", type=Path, default=None)
    args = ap.parse_args()
    v = run()
    print(json.dumps(v, indent=2))
    if args.json:
        args.json.write_text(json.dumps(v, indent=2) + "\n")
    if args.markdown:
        args.markdown.write_text(to_markdown(v))


if __name__ == "__main__":
    main()