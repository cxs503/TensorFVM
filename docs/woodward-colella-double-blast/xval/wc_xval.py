"""Cross-validation of the Woodward-Colella resolution ladder (round 2).

Why this exists
---------------
``CASE-WOODWARD-COLELLA-calibration-bias.md`` closed the previous round with a
verdict built on grids ``nx = 200 ... 1600``: after correcting the +0.022017
vertical digitisation bias of the published reference curve, *all* key metrics
were reported below 3 % on every grid ``nx >= 800``.  That verdict was published
while the ``nx = 3200`` rung was still queued; the rung has since returned
(``ladder_profile_nx3200_hllc.npz``) and it does **not** confirm the monotone
picture: the peak-density and L1 errors grow again between 1600 and 3200.

So the open question of this round is not "is the case right or wrong" but
"**which part of the residual is our discretisation error and which part belongs
to the reference**".  Comparing two numbers (our solution, the published curve)
cannot answer that: every comparison metric mixes

* the solver error of our scheme at resolution ``dx``,
* the solver error of the *published* solution (itself a 3200-cell RPJ run that
  was digitised through a PDF vector path), and
* the interpolative error of that digitisation.

This script separates them with measurements that need **no new physics**, only
the stored profiles:

1. ``self_convergence`` -- grid-to-grid L1 between successive rungs
   (``nx`` vs ``2 nx``).  This is a *reference-free* estimate of our own
   discretisation error, so the discrepancy with the reference can be compared
   against it.  With three successive differences we also get an observed order
   and a Richardson-style estimate of our remaining error at the finest grid.
2. ``error_budget`` -- ``total^2 = own^2 + reference_side^2`` (independent-error
   quadrature, valid because the two sources have no common mechanism): the
   measured L1 against the calibrated reference minus our own error leaves the
   part that must come from the reference side.
3. ``shift_scan`` -- the L1 of the finest profile against the reference as a
   function of a rigid translation of our profile.  A residual that collapses
   under a shift of a few reference grid spacings is a *phase* discrepancy
   (wave speeds / digitisation axis), not an amplitude discrepancy.
4. ``resolution`` -- total variation, maximum |d rho/dx| and 10-90 % shock rise
   width of the reference curve versus our profiles.  Numerically diffusive
   solutions have lower TV and wider shocks; the reference was digitised from a
   printed curve whose path vertices were merged by ghostscript, so its own
   smoothing shows up here as a quantifiable floor.
5. ``peak_fit`` -- ``rho_max(nx) = rho_inf + C nx^-p`` least-squares fit over the
   ladder, i.e. the continuum peak *we* converge to, compared with the peak the
   reference claims.

Everything is computed from files that are already in the repo, so the script is
a few seconds of numpy and can be re-run inside a unit test (see
``tests/test_compressible_wc_xval.py``).

Usage::

    PYTHONPATH=src python docs/woodward-colella-double-blast/xval/wc_xval.py
    PYTHONPATH=src python .../wc_xval.py --json out.json --figdir artifacts
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tensorfvm.compressible import wc_reference as wcr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                      # docs/woodward-colella-double-blast
DATA = ROOT / "ladder" / "data"
ARTIFACTS = ROOT / "artifacts"
REFDIR = ROOT / "reference"
CALIBRATED = REFDIR / "reference_density_calibrated.csv"

WINDOW = wcr.REFERENCE_X                # (0.4, 1.0) — the digitised window


# --------------------------------------------------------------------------- #
# stored profiles
# --------------------------------------------------------------------------- #
def stored_profiles() -> dict[str, dict]:
    """Every stored density profile, keyed by ``nxNNN_<family>``."""
    out: dict[str, dict] = {}
    for nx in (400, 800):
        p = ARTIFACTS / f"wc_blast_profile_nx{nx}.npz"
        if p.exists():
            d = np.load(p)
            out[f"nx{nx}_hllc"] = {"nx": nx, "family": "fvm",
                                   "x": np.asarray(d["x_got"], float),
                                   "rho": np.asarray(d["rho_got"], float),
                                   "source": str(p.relative_to(ROOT))}
    for p in sorted(DATA.glob("ladder_profile_*.npz")):
        d = np.load(p)
        tag = p.stem.replace("ladder_profile_", "")
        nx = int(tag.split("_")[0].lstrip("nx"))
        family = ("weno5" if "weno5" in tag
                  else "fvm_minmod" if "limiterminmod" in tag
                  else "fvm")
        out[tag] = {"nx": nx, "family": family,
                    "x": np.asarray(d["x"], float),
                    "rho": np.asarray(d["rho"], float),
                    "source": str(p.relative_to(ROOT))}
    for v in out.values():
        order = np.argsort(v["x"])
        v["x"], v["rho"] = v["x"][order], v["rho"][order]
    return out


def _on_common(xa, ra, xb, rb) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Both profiles restricted to the comparison window on ``xa``'s grid."""
    x = xa[(xa >= WINDOW[0]) & (xa <= WINDOW[1])]
    return x, ra[(xa >= WINDOW[0]) & (xa <= WINDOW[1])], np.interp(x, xb, rb)


def l1_between(xa, ra, xb, rb) -> float:
    """Relative L1 distance between two profiles on their common window."""
    x, a, b = _on_common(xa, ra, xb, rb)
    den = float(np.trapezoid(np.abs(b), x))
    return float(np.trapezoid(np.abs(a - b), x) / den)


def total_variation(x, rho) -> float:
    s = (x >= WINDOW[0]) & (x <= WINDOW[1])
    return float(np.sum(np.abs(np.diff(rho[s]))))


def max_gradient(x, rho) -> float:
    s = (x >= WINDOW[0]) & (x <= WINDOW[1])
    return float(np.max(np.abs(np.gradient(rho[s], x[s]))))


def rise_width(x, rho, x_centre: float, frac=(0.1, 0.9)) -> float | None:
    """10-90 % rise distance of the steepest jump nearest ``x_centre``."""
    s = (x >= WINDOW[0]) & (x <= WINDOW[1])
    xs, rs = x[s], rho[s]
    g = np.abs(np.gradient(rs, xs))
    i = int(np.argmin(np.abs(xs - x_centre))) if x_centre is not None else int(np.argmax(g))
    # nearest local maximum of |grad|
    lo, hi = max(i - int(0.02 / (xs[1] - xs[0])), 1), min(
        i + int(0.02 / (xs[1] - xs[0])) + 1, g.size - 1)
    j = lo + int(np.argmax(g[lo:hi]))
    # walk left/right to the 10/90 levels of the jump straddling j
    level_hi, level_lo = rs[j], rs[j]
    k = j
    while k > 0 and g[k] > 0.1 * g[j] and abs(xs[k] - xs[j]) < 0.05:
        k -= 1
        level_lo = min(level_lo, rs[k])
    k2 = j
    while k2 < g.size - 1 and g[k2] > 0.1 * g[j] and abs(xs[k2] - xs[j]) < 0.05:
        k2 += 1
        level_hi = max(level_hi, rs[k2])
    if level_hi - level_lo <= 0:
        return None
    l10 = level_lo + frac[0] * (level_hi - level_lo)
    l90 = level_lo + frac[1] * (level_hi - level_lo)
    # x where the monotone branch crosses the levels (scan the window)
    def crossing(level):
        d = rs - level
        idx = np.where(np.sign(d[:-1]) != np.sign(d[1:]))[0]
        if idx.size == 0:
            return None
        return float(xs[idx[int(np.argmin(np.abs(xs[idx] - xs[j])))]])

    x10, x90 = crossing(l10), crossing(l90)
    if x10 is None or x90 is None:
        return None
    return abs(x90 - x10)


# --------------------------------------------------------------------------- #
# analyses
# --------------------------------------------------------------------------- #
def self_convergence(profiles: dict[str, dict]) -> dict:
    """Reference-free grid-to-grid differences for each family."""
    out: dict[str, dict] = {}
    for family in sorted({v["family"] for v in profiles.values()}):
        rungs = sorted((v for v in profiles.values() if v["family"] == family),
                       key=lambda v: v["nx"])
        if len(rungs) < 2:
            continue
        diffs = []
        for coarse, fine in zip(rungs, rungs[1:]):
            if fine["nx"] != 2 * coarse["nx"]:
                continue
            diffs.append({
                "coarse_nx": coarse["nx"], "fine_nx": fine["nx"],
                "l1_window_percent": 100 * l1_between(coarse["x"], coarse["rho"],
                                                      fine["x"], fine["rho"]),
                "l1_full_percent": 100 * l1_between_pair_full(coarse, fine),
                "peak_delta_percent": 100 * abs(fine["rho"].max() - coarse["rho"].max())
                / fine["rho"].max(),
            })
        orders = []
        for a, b in zip(diffs, diffs[1:]):
            if b["l1_window_percent"] > 0:
                orders.append(float(np.log2(a["l1_window_percent"]
                                            / b["l1_window_percent"])))
        fine = rungs[-1]
        own = diffs[-1]["l1_window_percent"] if diffs else None
        p = orders[-1] if orders else None
        out[family] = {
            "rungs": [v["nx"] for v in rungs],
            "differences": diffs,
            "observed_orders": orders,
            "finest_nx": fine["nx"],
            "finest_self_l1_percent": own,
            "finest_error_estimate_percent": (
                None if (own is None or p is None or p <= 0)
                else own / (2 ** p - 1)),
            "peak_sequence": [round(float(v["rho"].max()), 4) for v in rungs],
        }
    return out


def l1_between_pair_full(a: dict, b: dict) -> float:
    """Relative L1 over the whole domain [0, 1] (no window restriction)."""
    x = a["x"]
    aa, bb = a["rho"], np.interp(x, b["x"], b["rho"])
    den = float(np.trapezoid(np.abs(bb), x))
    return float(np.trapezoid(np.abs(aa - bb), x) / den)


def error_budget(total_percent: float, own_percent: float | None) -> dict:
    """Split a measured reference error into own/reference-side contributions."""
    if own_percent is None:
        return {"total_percent": total_percent, "own_percent": None,
                "reference_side_percent": None}
    rest = total_percent ** 2 - own_percent ** 2
    return {"total_percent": total_percent,
            "own_percent": own_percent,
            "reference_side_percent": float(np.sqrt(rest)) if rest > 0 else 0.0,
            "own_fraction_percent": 100 * own_percent / total_percent if total_percent else None}


def shift_scan(xr, rr, xm, rm, span=0.008, n=161) -> dict:
    """L1 of our profile vs the reference under a rigid translation of ours."""
    best = (np.inf, 0.0)
    scans = []
    for d in np.linspace(-span, span, n):
        val = l1_between(xr, rr, xm + d, rm)
        scans.append([float(d), 100 * val])
        if val < best[0]:
            best = (val, float(d))
    base = 100 * l1_between(xr, rr, xm, rm)
    return {"base_percent": base,
            "best_percent": 100 * best[0],
            "best_shift": best[1],
            "reduction_percent": base - 100 * best[0],
            "scan": scans}


def peak_fit(rungs: list[dict]) -> dict:
    """Least-squares ``rho_max = rho_inf + C nx^-p`` over the stored rungs."""
    nx = np.array([v["nx"] for v in rungs], float)
    rmax = np.array([float(v["rho"].max()) for v in rungs], float)
    best = None
    for p in np.linspace(0.2, 3.0, 561):
        A = np.vstack([np.ones_like(nx), nx ** (-p)]).T
        coef, res, *_ = np.linalg.lstsq(A, rmax, rcond=None)
        ss = float(np.sum((A @ coef - rmax) ** 2))
        if best is None or ss < best[0]:
            best = (ss, p, coef[0], coef[1])
    ss, p, rho_inf, C = best
    fitted = rho_inf + C * nx ** (-p)
    return {"nx": nx.tolist(), "peak": rmax.tolist(),
            "order": float(p), "rho_inf": float(rho_inf), "C": float(C),
            "rms_fit": float(np.sqrt(ss / nx.size)),
            "fitted": fitted.tolist()}


def reference_geometry(ref: dict) -> dict:
    """Empirical smoothing of the digitised reference curve."""
    x, r = ref["x"], ref["rho"]
    return {"n_points": int(x.size),
            "spacing": float(np.median(np.diff(x))),
            "peak": float(r.max()), "peak_x": float(x[int(np.argmax(r))]),
            "total_variation": total_variation(x, r),
            "max_gradient": max_gradient(x, r),
            "rise_width_peak": rise_width(x, r, float(x[int(np.argmax(r))]))}



def box_smooth(x, rho, width: float) -> np.ndarray:
    """Top-hat average of ``rho`` over ``width`` in x (edges handled by padding)."""
    dx = float(np.median(np.diff(x)))
    k = max(1, int(round(width / dx)))
    if k <= 1:
        return rho.copy()
    kernel = np.ones(k) / k
    pad = k // 2
    padded = np.pad(rho, (pad, k - 1 - pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def smoothing_floor(xr, rr, xm, rm) -> dict:
    """L1 vs the reference as our (sharp) profile is smoothed to coarser widths.

    If the residual is limited by the *reference* being smoother than our
    solution, then a top-hat filter of the right width removes part of it and
    the curve ``L1(width)`` has an interior minimum.  The width of that minimum
    is an estimate of the reference curve's own smoothing length; the value at
    the minimum is the residual that survives at matched resolution (i.e. the
    amplitude/scheme part, not the sharpness part).
    """
    widths = np.concatenate([[0.0], np.logspace(np.log10(2e-4), np.log10(2e-2), 60)])
    rows = []
    for w in widths:
        sm = rm if w == 0 else box_smooth(xm, rm, float(w))
        rows.append([float(w), 100 * l1_between(xr, rr, xm, sm)])
    arr = np.asarray(rows)
    i = int(np.argmin(arr[:, 1]))
    base = float(arr[0, 1])
    return {"widths": arr[:, 0].tolist(),
            "l1_percent": arr[:, 1].tolist(),
            "base_percent": base,
            "best_width": float(arr[i, 0]),
            "best_percent": float(arr[i, 1]),
            "reduction_percent": base - float(arr[i, 1]),
            "sharpness_fraction_percent": (100 * (base - float(arr[i, 1])) / base
                                           if base else None)}


def band_contributions(xr, rr, xm, rm, bands=((0.40, 0.55), (0.55, 0.70),
                                              (0.70, 0.76), (0.76, 0.80),
                                              (0.80, 0.90), (0.90, 1.00))) -> dict:
    """Where the remaining L1 error sits (fraction of the normaliser)."""
    s = (xm >= WINDOW[0]) & (xm <= WINDOW[1])
    x, r = xm[s], rm[s]
    ref = np.interp(x, xr, rr)
    d = np.abs(r - ref)
    den = float(np.trapezoid(np.interp(xr, xr, rr), x)) if False else float(
        np.trapezoid(ref, x))
    out = {}
    for lo, hi in bands:
        m = (x >= lo) & (x <= hi)
        out[f"{lo:.2f}-{hi:.2f}"] = {
            "l1_contribution_percent": 100 * float(np.trapezoid(d[m], x[m])) / den,
            "max_abs_diff": float(d[m].max()),
            "x_of_max": float(x[m][int(d[m].argmax())]),
        }
    return out



def peak_extrapolation(rungs: list[dict], nfit: int = 3) -> dict:
    """Triple-grid extrapolation of the peak density on the finest ``nfit`` rungs.

    Uses the classic three-level Richardson estimate
    ``p = log2((f2-f1)/(f3-f2))`` on ``f1 > f2 > f3`` (from coarse to fine) and
    reports the continuum value plus the estimated error of the finest grid.
    The coarse rungs (nx = 200, 400) are excluded because their peak is
    dissipation-limited, not in the asymptotic range -- including them makes the
    fitted limit meaningless (it moves by >4 %).
    """
    rungs = sorted(rungs, key=lambda v: v["nx"])[-nfit:]
    nx = [v["nx"] for v in rungs]
    f = [float(v["rho"].max()) for v in rungs]
    if len(f) != 3:
        return {"nx": nx, "peak": f, "note": "need three rungs"}
    d1, d2 = f[1] - f[0], f[2] - f[1]
    out = {"nx": nx, "peak": f, "d_coarse": d1, "d_fine": d2}
    if d1 > 0 and d2 > 0 and d1 > d2:
        p = float(np.log2(d1 / d2))
        f_inf = f[2] + d2 / (2 ** p - 1)
        out.update({"order": p, "peak_infinity": float(f_inf),
                    "finest_error_estimate": float(d2 / (2 ** p - 1)),
                    "asymptotic": bool(0.5 <= p <= 4.0)})
    else:
        out.update({"order": None, "peak_infinity": None,
                    "finest_error_estimate": None, "asymptotic": False})
    return out


def face_phases(xr, rr, xm, rm, half_width: float = 0.012) -> list[dict]:
    """Local optimal x-shift of each steepest face (phase error, not amplitude).

    For every one of the four steepest density faces the profile is shifted in a
    window of +/- ``half_width`` around the face and the shift minimising the
    local L1 is reported next to the shift the reference itself reports.  A face
    whose optimal shift is ≈0 is aligned; a face with a shift of several
    reference spacings is a *position* (wave-speed) discrepancy.
    """
    fr = wcr.faces(xr, rr)
    out = []
    dx_ref = float(np.median(np.diff(xr)))
    for xf in fr:
        lo, hi = xf - half_width, xf + half_width
        s = (xm >= lo) & (xm <= hi)
        xs, rs = xm[s], rm[s]
        if xs.size < 8:
            continue
        xr_w = xr[(xr >= lo) & (xr <= hi)]
        rr_w = rr[(xr >= lo) & (xr <= hi)]
        best = (np.inf, 0.0)
        for d in np.linspace(-half_width / 2, half_width / 2, 241):
            val = float(np.trapezoid(np.abs(np.interp(xr_w, xs + d, rs) - rr_w), xr_w))
            if val < best[0]:
                best = (val, float(d))
        base = float(np.trapezoid(np.abs(np.interp(xr_w, xs, rs) - rr_w), xr_w))
        out.append({"face_x": float(xf),
                    "local_l1_base": base, "local_l1_best": best[0],
                    "optimal_shift": best[1],
                    "optimal_shift_in_ref_spacings": best[1] / dx_ref,
                    "local_reduction_percent": (100 * (base - best[0]) / base
                                                if base else None)})
    return out


def run() -> dict:
    raw = wcr.load_reference()
    cal = wcr.load_reference(CALIBRATED)
    profiles = stored_profiles()
    selfconv = self_convergence(profiles)

    rows = {}
    for tag, p in sorted(profiles.items(), key=lambda kv: (kv[1]["family"], kv[1]["nx"])):
        m_raw = wcr.metrics_from_profiles(raw["x"], raw["rho"], p["x"], p["rho"])
        m_cal = wcr.metrics_from_profiles(cal["x"], cal["rho"], p["x"], p["rho"])
        rows[tag] = {
            "nx": p["nx"], "family": p["family"], "source": p["source"],
            "peak_percent": m_raw["peak_density"]["rel_error_percent"],
            "l1_percent": m_raw["l1_relative"]["percent"],
            "mass_percent": m_raw["window_mass"]["rel_error_percent"],
            "peak_percent_calibrated": m_cal["peak_density"]["rel_error_percent"],
            "l1_percent_calibrated": m_cal["l1_relative"]["percent"],
            "mass_percent_calibrated": m_cal["window_mass"]["rel_error_percent"],
            "peak_position_error": m_cal["peak_position"]["abs_error"],
            "verdict_calibrated": m_cal["verdict"],
            "total_variation": total_variation(p["x"], p["rho"]),
            "max_gradient": max_gradient(p["x"], p["rho"]),
            "peak": float(p["rho"].max()),
        }

    finest_fvm = max((v for v in profiles.values() if v["family"] == "fvm"),
                     key=lambda v: v["nx"])
    finest_tag = f"nx{finest_fvm['nx']}_hllc"

    budgets = {}
    for metric, key in (("l1", "l1_percent_calibrated"), ("peak", "peak_percent_calibrated")):
        own = (selfconv["fvm"]["finest_self_l1_percent"] if metric == "l1"
               else selfconv["fvm"]["differences"][-1]["peak_delta_percent"])
        budgets[metric] = error_budget(rows[finest_tag][key], own)

    analysis = {
        "window": list(WINDOW),
        "ledger_tolerance_percent": wcr.TOLERANCE_PERCENT,
        "reference": {
            "raw": reference_geometry(raw),
            "calibrated": reference_geometry(cal),
            "source": raw["path"],
        },
        "profiles": rows,
        "self_convergence": selfconv,
        "error_budget_finest": budgets,
        "shift_scan_finest": shift_scan(cal["x"], cal["rho"],
                                        finest_fvm["x"], finest_fvm["rho"]),
        "peak_fit_fvm": peak_fit(sorted((v for v in profiles.values()
                                         if v["family"] == "fvm"),
                                        key=lambda v: v["nx"])),
        "peak_fit_weno5": (peak_fit(sorted((v for v in profiles.values()
                                           if v["family"] == "weno5"),
                                          key=lambda v: v["nx"]))
                           if sum(v["family"] == "weno5" for v in profiles.values()) >= 3
                           else None),
        "finest_tag": finest_tag,
        "smoothing_floor_finest": smoothing_floor(cal["x"], cal["rho"],
                                                  finest_fvm["x"], finest_fvm["rho"]),
        "band_contributions_finest": band_contributions(cal["x"], cal["rho"],
                                                        finest_fvm["x"], finest_fvm["rho"]),
        "peak_extrapolation_fvm": peak_extrapolation(
            [v for v in profiles.values() if v["family"] == "fvm"]),
        "face_phases_finest": face_phases(cal["x"], cal["rho"],
                                          finest_fvm["x"], finest_fvm["rho"]),
    }

    # cross-code agreement at the coarsest common grid (scheme independence)
    by_nx: dict[int, dict[str, dict]] = {}
    for v in profiles.values():
        by_nx.setdefault(v["nx"], {})[v["family"]] = v
    xcode = {}
    for nx, fam in sorted(by_nx.items()):
        if "fvm" in fam and "weno5" in fam:
            a, b = fam["fvm"], fam["weno5"]
            xcode[nx] = {
                "l1_between_percent": 100 * l1_between(a["x"], a["rho"], b["x"], b["rho"]),
                "peak_fvm": float(a["rho"].max()), "peak_weno5": float(b["rho"].max()),
                "peak_delta_percent": 100 * abs(a["rho"].max() - b["rho"].max())
                / b["rho"].max(),
            }
    analysis["cross_code"] = xcode
    return analysis


def report(a: dict) -> None:
    print(f"window {a['window']}  tolerance {a['ledger_tolerance_percent']} %")
    print("\nprofile            nx  fam    peak%   L1%   peak_cal%  L1_cal%  mass_cal%  TV")
    for tag, r in a["profiles"].items():
        print(f"{tag:<18} {r['nx']:>4} {r['family']:<6} {r['peak_percent']:6.2f} "
              f"{r['l1_percent']:6.2f} {r['peak_percent_calibrated']:8.2f} "
              f"{r['l1_percent_calibrated']:7.2f} {r['mass_percent_calibrated']:8.2f} "
              f"{r['total_variation']:7.3f}")
    sc = a["self_convergence"].get("fvm")
    if sc:
        print(f"\nfvm rungs {sc['rungs']}  orders {['%.2f' % o for o in sc['observed_orders']]}")
        for d in sc["differences"]:
            print(f"  {d['coarse_nx']:>4}->{d['fine_nx']:<4} window L1 "
                  f"{d['l1_window_percent']:6.3f} %   full L1 {d['l1_full_percent']:6.3f} %"
                  f"   peak Δ {d['peak_delta_percent']:6.3f} %")
        print(f"  finest self-L1 {sc['finest_self_l1_percent']:.3f} %  "
              f"-> own error estimate {sc['finest_error_estimate_percent']:.3f} %")
    for k, b in a["error_budget_finest"].items():
        print(f"\nerror budget [{k}]: total {b['total_percent']:.3f} %  "
              f"own {b['own_percent']:.3f} %  reference-side {b['reference_side_percent']:.3f} %")
    ss = a["shift_scan_finest"]
    print(f"\nshift scan: base {ss['base_percent']:.3f} %  best {ss['best_percent']:.3f} % "
          f"at δ={ss['best_shift']:+.4f} (Δ {ss['reduction_percent']:.3f} pp)")
    pf = a["peak_fit_fvm"]
    print(f"peak fit (fvm): rho_inf {pf['rho_inf']:.4f}  order {pf['order']:.2f}  "
          f"rms {pf['rms_fit']:.4f}   (reference raw {a['reference']['raw']['peak']:.4f}, "
          f"calibrated {a['reference']['calibrated']['peak']:.4f})")
    pe = a["peak_extrapolation_fvm"]
    print(f"peak extrapolation nx={pe['nx']}: peak_inf {pe.get('peak_infinity')} "
          f"order {pe.get('order')} asymptotic {pe.get('asymptotic')}")
    print("face phases (finest):")
    for f in a["face_phases_finest"]:
        print(f"   face {f['face_x']:.4f}  shift {f['optimal_shift']:+.4f} "
              f"({f['optimal_shift_in_ref_spacings']:+.1f} ref spacings)  "
              f"local L1 {100 * f['local_l1_base']:.4f} -> {100 * f['local_l1_best']:.4f}")
    for nx, c in a["cross_code"].items():
        print(f"cross-code nx={nx}: L1 between codes {c['l1_between_percent']:.3f} %  "
              f"peak Δ {c['peak_delta_percent']:.3f} %")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, default=HERE / "xval_round2.json")
    args = ap.parse_args()
    a = run()
    report(a)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(a, indent=2) + "\n")
    print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()