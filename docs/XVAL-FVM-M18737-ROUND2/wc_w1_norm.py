"""Transport-norm (W1 / negative-norm s=-1) certification for CASE-WOODWARD-COLELLA.

Why this exists
---------------
``R4-REFERENCE-UNCERTAINTY-CLOSURE.md`` closed round-4 with an honest verdict:
the graded MUSCL/HLLC solution sits 0.89 % (L1, nx=6400 vs WENO5@12800) from the
independent reference, but the *reference's own* uncertainty in the same norm is
``U_ref = 0.7181 %``, so only ``upper_bound_only`` could be claimed -- pushing the
pointwise ``<1 %`` bound below 1 % would need ``nx_ref ~ 1.33e5`` (see R4 P5).

That is an expensive question only because the *measure* is expensive.  The
pointwise L1 of a shock-dominated profile is dominated by the shock-smearing
width, which shrinks only like ``h**p`` with ``p ~ 0.81``.  The classical fix
(Banks & Hittinger style negative-norm error estimation; in 1D equivalent to the
Wasserstein-1 distance between the normalised cumulative mass distributions) is
to *integrate before measuring*: the accumulated difference ``D(x) = int e``
cancels the in-band sawtooth of a smeared shock and gains roughly one order of
convergence, so the same stored ladder can certify far more cheaply.

This script is the predeclared test of that route.  Criteria P1-P9 are frozen in
``PREDECLARATION-w1-norm.md`` **before** any of the numbers below were computed.

What it does
------------
* No solver is run: every quantity is re-integrated from the stored profiles
  ``ladder/data/ladder_profile_*.npz`` (MUSCL/HLLC 800/1600/3200/6400,
  WENO5 1600/3200/6400/12800).  Therefore any difference from the round-4
  verdict comes from the *metric*, not from the solutions.
* Metrics (window ``[0.4, 1]``, integrated on the finer grid, coarse profile
  linearly interpolated -- the same direction as the round-2/4 ``l1_percent``,
  so the fine shock is never smeared by interpolation):

  ``d_L1  = 100 * int |e| / int |rho_f|``                    (percent, round-2 metric)
  ``d_W1  = 100 * int |D| / (mass_f * L)``,  ``D = int_0.4^x e``   (percent of span)

* Invariant checked on every pair: ``d_W1 <= d_L1`` (Cauchy-style bound
  ``int|D| <= L int|e|``).  A violation is an implementation bug, not physics.
* Honest bound: ``d_W1(graded) + eps_ref_W1`` -- the W1 analogue of the
  round-4 triangle-inequality gate.

Usage::

    PYTHONPATH=src python docs/XVAL-FVM-M18737-ROUND2/wc_w1_norm.py --json /tmp/w1.json
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
ROOT = HERE.parent                                   # docs/
DATA = ROOT / "woodward-colella-double-blast" / "ladder" / "data"
ARTIFACTS = ROOT / "woodward-colella-double-blast" / "artifacts"

XV = ROOT / "woodward-colella-double-blast" / "xval"

WINDOW = (0.4, 1.0)
VARS = ("rho", "p", "u")

MUSCL_LADDER = (800, 1600, 3200, 6400)
WENO_LADDER = (1600, 3200, 6400, 12800)
GRADED_FINEST = 6400
REF_FINEST = 12800

# --------------------------------------------------------------------------- #
# FROZEN CRITERIA -- see PREDECLARATION-w1-norm.md (written before computing).
# Do not tune: a failed criterion is reported as FAIL, not re-thresholded.
# --------------------------------------------------------------------------- #
FROZEN = {
    "P1_order_band": (1.3, 2.6),          # observed order of d_W1 on the ref family
    "P1_order_gain_over_L1": 0.5,         # p_W1 >= p_L1 + this
    "P2_eps_ref_max": 0.20,               # percent of span, @ WENO5 12800
    "P3_graded_w1_max": 0.30,             # percent of span, MUSCL 6400 vs WENO5 12800
    "P4_honest_bound_max": 1.0,           # certified <1 % in the transport norm
    "P5_l1_floor": 0.5,                   # pointwise L1 must stay above this (no free lunch)
    "P7_interp_sensitivity_frac": 0.25,   # |fine-dir - coarse-dir| / d_W1
    "P9_var_spread_factor": 5.0,          # max d_W1 across rho/p/u / min d_W1
}


def _xval_mod(name: str):
    """Import a sibling xval tool by path (no package install needed)."""
    spec = importlib.util.spec_from_file_location(name, XV / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, mod)
    spec.loader.exec_module(mod)
    return mod


WCR = _xval_mod("wc_certified_ref")      # provides ``_load`` + WINDOW
WRU = _xval_mod("wc_ref_uncertainty")    # provides ``observed_order``/``richardson_error``
assert tuple(WCR.WINDOW) == WINDOW, (WCR.WINDOW, WINDOW)


# --------------------------------------------------------------------------- #
# profile I/O
# --------------------------------------------------------------------------- #
def load(nx: int, family: str) -> dict | None:
    """Load a ladder rung (``None`` when it has not been computed)."""
    path = DATA / f"ladder_profile_nx{nx}_{family}.npz"
    if not path.exists():
        return None
    d = np.load(path)
    out = {k: np.asarray(d[k], float) for k in d.files}
    out["nx"] = nx
    out["family"] = family
    return out


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def _pair_on_fine(fine: dict, coarse: dict, var: str, window=None):
    """Window mask + fine-node samples + coarse profile interpolated onto them."""
    lo, hi = WINDOW if window is None else window
    m = (fine["x"] >= lo) & (fine["x"] <= hi)
    x = fine["x"][m]
    a = fine[var][m]
    b = np.interp(x, coarse["x"], coarse[var])
    return x, a, b


def d_l1_percent(fine: dict, coarse: dict, var: str = "rho", window=None) -> float:
    """Pointwise relative L1 (percent) -- identical to the round-2/4 metric."""
    x, a, b = _pair_on_fine(fine, coarse, var, window)
    den = float(np.trapezoid(np.abs(a), x))
    return 100.0 * float(np.trapezoid(np.abs(a - b), x)) / den if den > 0 else float("nan")


def d_w1_percent(fine: dict, coarse: dict, var: str = "rho", window=None) -> float:
    """Transport (W1 / negative-norm s=-1) distance, percent of the window span.

    ``D(x) = int_{0.4}^{x} (rho_c - rho_f)`` is the accumulated difference; the
    metric is ``100 * int|D| / (mass_f * L)``, i.e. the W1 distance between the
    two normalised cumulative mass distributions expressed as a fraction of the
    window length.  Physically: how far the profile has been *transported* wrong.
    """
    x, a, b = _pair_on_fine(fine, coarse, var, window)
    span = float(x[-1] - x[0])
    mass = float(np.trapezoid(np.abs(a), x))
    if mass <= 0 or span <= 0:
        return float("nan")
    e = a - b
    # cumulative trapezoid of the difference, starting at 0 on the window edge
    dx = np.diff(x)
    seg = 0.5 * (e[1:] + e[:-1]) * dx
    D = np.concatenate(([0.0], np.cumsum(seg)))
    return 100.0 * float(np.trapezoid(np.abs(D), x)) / (mass * span)


def d_w1_percent_coarse_dir(fine: dict, coarse: dict, var: str = "rho") -> float:
    """Same metric but integrated on the *coarse* grid (direction sensitivity)."""
    m = (coarse["x"] >= WINDOW[0]) & (coarse["x"] <= WINDOW[1])
    x = coarse["x"][m]
    a = coarse[var][m]
    b = np.interp(x, fine["x"], fine[var])
    span = float(x[-1] - x[0])
    mass = float(np.trapezoid(np.abs(b), x))
    if mass <= 0 or span <= 0:
        return float("nan")
    e = a - b
    dx = np.diff(x)
    seg = 0.5 * (e[1:] + e[:-1]) * dx
    D = np.concatenate(([0.0], np.cumsum(seg)))
    return 100.0 * float(np.trapezoid(np.abs(D), x)) / (mass * span)


def cumulative_diff_field(fine: dict, coarse: dict, var: str = "rho") -> dict:
    """The accumulated difference ``D(x)`` (for the figure), in mass units."""
    x, a, b = _pair_on_fine(fine, coarse, var)
    e = a - b
    dx = np.diff(x)
    D = np.concatenate(([0.0], np.cumsum(0.5 * (e[1:] + e[:-1]) * dx)))
    return {"x": x, "D": D, "e": e}


def window_mass(prof: dict, var: str = "rho") -> float:
    """Conserved window mass ``int_{0.4}^{1} |q| dx`` (mass units, not percent)."""
    m = (prof["x"] >= WINDOW[0]) & (prof["x"] <= WINDOW[1])
    return float(np.trapezoid(np.abs(prof[var][m]), prof["x"][m]))


# --------------------------------------------------------------------------- #
# the budget
# --------------------------------------------------------------------------- #
def ref_family(profiles: dict) -> dict:
    """WENO5 self-certification ladder in both norms."""
    rungs = [n for n in WENO_LADDER if profiles.get(("weno5", n)) is not None]
    pairs = []
    for lo, hi in zip(rungs, rungs[1:]):
        f, c = profiles[("weno5", hi)], profiles[("weno5", lo)]
        pairs.append(
            {
                "coarse_nx": lo,
                "fine_nx": hi,
                "d_l1": d_l1_percent(f, c),
                "d_w1": d_w1_percent(f, c),
                "d_w1_coarse_dir": d_w1_percent_coarse_dir(f, c),
            }
        )
    out = {"rungs": rungs, "pairs": pairs}
    if len(pairs) >= 2:
        p_w1, src_w1 = WRU.observed_order(pairs[-2]["d_w1"], pairs[-1]["d_w1"])
        p_l1, src_l1 = WRU.observed_order(pairs[-2]["d_l1"], pairs[-1]["d_l1"])
        out.update(
            {
                "p_w1": p_w1,
                "p_w1_source": src_w1,
                "p_l1": p_l1,
                "p_l1_source": src_l1,
                "eps_ref_w1": WRU.richardson_error(pairs[-1]["d_w1"], p_w1),
                "eps_ref_l1": WRU.richardson_error(pairs[-1]["d_l1"], p_l1),
            }
        )
    elif pairs:                                   # one pair only: no order measurable
        out.update(
            {
                "p_w1": 1.0,
                "p_w1_source": "single pair -> first-order fallback",
                "eps_ref_w1": WRU.richardson_error(pairs[-1]["d_w1"], 1.0),
                "eps_ref_l1": WRU.richardson_error(pairs[-1]["d_l1"], 1.0),
            }
        )
    return out


def graded_family(profiles: dict, ref: dict) -> dict:
    """MUSCL/HLLC rungs re-graded against the finest WENO5 reference, both norms."""
    rungs = [n for n in MUSCL_LADDER if profiles.get(("hllc", n)) is not None]
    rows = []
    for nx in rungs:
        g = profiles[("hllc", nx)]
        rows.append(
            {
                "nx": nx,
                "d_l1": d_l1_percent(ref, g),
                "d_w1": d_w1_percent(ref, g),
                "d_w1_coarse_dir": d_w1_percent_coarse_dir(ref, g),
                "w1_in_coarse_cells": None,       # filled below
            }
        )
        dx_c = (WINDOW[1] - WINDOW[0]) / nx
        rows[-1]["w1_in_coarse_cells"] = (
            rows[-1]["d_w1"] / 100.0 * (WINDOW[1] - WINDOW[0]) / dx_c
        )
    out = {"rungs": rungs, "rows": rows, "ref_nx": ref["nx"]}
    if len(rows) >= 2:
        p_w1, src = WRU.observed_order(rows[-2]["d_w1"], rows[-1]["d_w1"])
        p_l1, _ = WRU.observed_order(rows[-2]["d_l1"], rows[-1]["d_l1"])
        out.update(
            {
                "p_w1": p_w1,
                "p_w1_source": src,
                "p_l1": p_l1,
                "eps_own_w1": WRU.richardson_error(rows[-1]["d_w1"], p_w1),
                "eps_own_l1": WRU.richardson_error(rows[-1]["d_l1"], p_l1),
            }
        )
    return out


def adjudicate(ref: dict, graded: dict, profiles: dict) -> dict:
    """Apply the frozen criteria P1-P9."""
    rows = {r["nx"]: r for r in graded["rows"]}
    finest = max(rows)
    res: dict = {}

    p_w1 = ref.get("p_w1", float("nan"))
    p_l1 = ref.get("p_l1", float("nan"))
    lo, hi = FROZEN["P1_order_band"]
    res["P1"] = {
        "criterion": f"p_W1 in [{lo}, {hi}] and p_W1 >= p_L1 + {FROZEN['P1_order_gain_over_L1']}",
        "p_w1": p_w1,
        "p_l1": p_l1,
        "pass": bool(lo <= p_w1 <= hi and p_w1 >= p_l1 + FROZEN["P1_order_gain_over_L1"]),
    }

    eps = ref.get("eps_ref_w1", float("nan"))
    res["P2"] = {
        "criterion": f"eps_ref_W1 <= {FROZEN['P2_eps_ref_max']} %",
        "eps_ref_w1": eps,
        "eps_ref_l1": ref.get("eps_ref_l1", float("nan")),
        "pass": bool(math.isfinite(eps) and eps <= FROZEN["P2_eps_ref_max"]),
    }

    d = rows[finest]["d_w1"]
    res["P3"] = {
        "criterion": f"d_W1(nx={finest}) <= {FROZEN['P3_graded_w1_max']} %",
        "d_w1": d,
        "nx": finest,
        "pass": bool(math.isfinite(d) and d <= FROZEN["P3_graded_w1_max"]),
    }

    bound = d + eps if math.isfinite(d) and math.isfinite(eps) else float("nan")
    res["P4"] = {
        "criterion": f"d_W1 + eps_ref_W1 < {FROZEN['P4_honest_bound_max']} % (certified <1 % in transport norm)",
        "honest_bound": bound,
        "pass": bool(math.isfinite(bound) and bound < FROZEN["P4_honest_bound_max"]),
    }

    l1_fine = rows[finest]["d_l1"]
    l1_32 = rows.get(3200, {}).get("d_l1", float("nan"))
    res["P5"] = {
        "criterion": f"pointwise L1 stays > {FROZEN['P5_l1_floor']} % (no free lunch)",
        "d_l1_3200": l1_32,
        "d_l1_finest": l1_fine,
        "pass": bool(
            math.isfinite(l1_fine)
            and math.isfinite(l1_32)
            and l1_fine > FROZEN["P5_l1_floor"]
            and l1_32 > FROZEN["P5_l1_floor"]
        ),
    }

    violations = []
    all_pairs = [("ref", p["coarse_nx"], p["fine_nx"], p["d_l1"], p["d_w1"]) for p in ref["pairs"]]
    all_pairs += [("graded", r["nx"], graded["ref_nx"], r["d_l1"], r["d_w1"]) for r in graded["rows"]]
    for var in VARS:
        for lo_n, hi_n in zip(WENO_LADDER, WENO_LADDER[1:]):
            f, c = profiles.get(("weno5", hi_n)), profiles.get(("weno5", lo_n))
            if f and c:
                a1 = d_l1_percent(f, c, var)
                a2 = d_w1_percent(f, c, var)
                all_pairs.append(("ref", lo_n, hi_n, a1, a2))
        for nx in MUSCL_LADDER:
            g = profiles.get(("hllc", nx))
            if g:
                a1 = d_l1_percent(profiles[("weno5", REF_FINEST)], g, var)
                a2 = d_w1_percent(profiles[("weno5", REF_FINEST)], g, var)
                all_pairs.append(("graded", nx, REF_FINEST, a1, a2))
    for tag, lo_n, hi_n, l1v, w1v in all_pairs:
        if math.isfinite(l1v) and math.isfinite(w1v) and w1v > l1v * (1 + 1e-9):
            violations.append({"family": tag, "coarse": lo_n, "fine": hi_n, "d_l1": l1v, "d_w1": w1v})
    res["P6"] = {
        "criterion": "d_W1 <= d_L1 on every pair and every variable",
        "n_pairs": len(all_pairs),
        "violations": violations,
        "pass": not violations,
    }

    sens, detail = [], []
    for tag, rows_ in (("reference", ref["pairs"]), ("graded", graded["rows"])):
        for row in rows_:
            if row["d_w1"] > 0:
                frac = abs(row["d_w1"] - row["d_w1_coarse_dir"]) / row["d_w1"]
                sens.append(frac)
                label = (f"{row['coarse_nx']}v{row['fine_nx']}"
                         if tag == "reference" else f"hllc{row['nx']}v{graded['ref_nx']}")
                detail.append({"family": tag, "pair": label, "d_w1": row["d_w1"],
                               "d_w1_coarse_dir": row["d_w1_coarse_dir"], "sensitivity": frac})
    worst = max(sens) if sens else float("nan")
    res["P7"] = {
        "criterion": f"interp-direction sensitivity <= {FROZEN['P7_interp_sensitivity_frac']} of d_W1",
        "worst_fraction": worst,
        "n_pairs": len(sens),
        "n_violating": sum(1 for v in sens if v > FROZEN["P7_interp_sensitivity_frac"]),
        "detail": detail,
        "pass": bool(math.isfinite(worst) and worst <= FROZEN["P7_interp_sensitivity_frac"]),
    }

    seq = [rows[n]["d_w1"] for n in graded["rungs"]]
    mono = all(b <= a for a, b in zip(seq, seq[1:]))
    res["P8"] = {
        "criterion": "d_W1 decreases monotonically with refinement (fixed reference)",
        "sequence": seq,
        "pass": bool(mono),
    }

    per_var = {}
    for var in VARS:
        g = profiles[("hllc", finest)]
        per_var[var] = d_w1_percent(profiles[("weno5", REF_FINEST)], g, var)
    vals = [v for v in per_var.values() if math.isfinite(v) and v > 0]
    spread = (max(vals) / min(vals)) if len(vals) == len(VARS) else float("nan")
    res["P9"] = {
        "criterion": f"rho/p/u d_W1 within a factor {FROZEN['P9_var_spread_factor']}",
        "per_var": per_var,
        "spread": spread,
        "pass": bool(math.isfinite(spread) and spread <= FROZEN["P9_var_spread_factor"]),
    }
    return res


def build(verbose: bool = True) -> dict:
    profiles = {}
    for fam, ladder in (("hllc", MUSCL_LADDER), ("weno5", WENO_LADDER)):
        for nx in ladder:
            p = load(nx, fam)
            if p is not None:
                profiles[(fam, nx)] = p
    if ("weno5", REF_FINEST) not in profiles:
        raise FileNotFoundError(f"missing reference rung WENO5@{REF_FINEST}")

    ref_prof = profiles[("weno5", REF_FINEST)]
    m = (ref_prof["x"] >= WINDOW[0]) & (ref_prof["x"] <= WINDOW[1])
    mass_ref = float(np.trapezoid(np.abs(ref_prof["rho"][m]), ref_prof["x"][m]))

    ref = ref_family(profiles)
    graded = graded_family(profiles, ref_prof)
    verdict = adjudicate(ref, graded, profiles)

    # post-hoc: what the metric sees, and where its residual lives
    probes = sensitivity_probes(ref_prof, floor_percent=verdict["P4"]["honest_bound"])
    loc = {f"hllc{nx}_vs_weno5{REF_FINEST}": localize(ref_prof, profiles[("hllc", nx)])
           for nx in graded["rungs"]}
    for p_ in ref["pairs"]:
        key = f"weno5{p_['coarse_nx']}_vs_weno5{p_['fine_nx']}"
        loc[key] = localize(profiles[("weno5", p_["fine_nx"])],
                            profiles[("weno5", p_["coarse_nx"])])
    n_detect = sum(1 for v in probes.values() if v.get("detects"))
    same_grid = {}
    for nx in MUSCL_LADDER:
        g, w = profiles.get(("hllc", nx)), profiles.get(("weno5", nx))
        if g is not None and w is not None:
            same_grid[f"nx{nx}"] = {"d_l1": d_l1_percent(w, g), "d_w1": d_w1_percent(w, g)}

    mass = {
        f"{fam}@{nx}": {
            "mass": window_mass(profiles[(fam, nx)]),
            "drift_percent": 100.0
            * (window_mass(profiles[(fam, nx)]) - mass_ref)
            / mass_ref,
        }
        for (fam, nx) in sorted(profiles)
    }

    out = {
        "case": "CASE-WOODWARD-COLELLA",
        "anchor": "XVAL-FVM-M18737-ROUND2",
        "metric": {
            "window": list(WINDOW),
            "d_l1": "100*int|e|/int|rho_f|  (percent, round-2 metric)",
            "d_w1": "100*int|D|/(mass_f*L), D=int_{0.4}^{x} e  (percent of span)",
            "truncation": "integrated on the finer grid, coarse profile interpolated",
        },
        "reference_family": ref,
        "graded_family": graded,
        "mass": mass,
        "verdict": verdict,
        "probes_post_hoc": probes,
        "localization": loc,
        "probes_n_detecting": n_detect,
        "same_grid_cross_family": same_grid,
        "frozen": FROZEN,
    }
    out["all_pass"] = all(v["pass"] for v in verdict.values())
    if verbose:
        print(json.dumps({"reference_family": ref, "graded_family": graded,
                          "verdict": verdict, "probes_post_hoc": probes,
                          "localization": loc, "all_pass": out["all_pass"]},
                         indent=2, default=float))
    return out


# --------------------------------------------------------------------------- #
# POST-HOC metric characterisation (NOT predeclared criteria)
#
# P1/P7 had already failed when these probes were written, so they must not be
# read as pass/fail gates on the anchor.  They answer a different question:
# *what does d_W1 actually see, and what does it go blind to?*  Without this, a
# d_W1 = 0.01 % cannot be told apart from a metric that measures nothing.
# --------------------------------------------------------------------------- #
def _perturbed(profile: dict, fn) -> dict:
    return {"x": profile["x"].copy(), "rho": fn(profile["x"], profile["rho"]),
            "p": profile["p"].copy(), "u": profile["u"].copy(),
            "nx": profile["nx"], "family": profile.get("family", "probe")}


def sensitivity_probes(ref_prof: dict, floor_percent: float | None = None) -> dict:
    """Inject *known* errors into the reference and read d_W1 back.

    ========================== =============================== ==============
    injected error             analytic d_W1 (percent)          expected
    ========================== =============================== ==============
    rigid shift, 1 cell @3200  100*dx/L                        ~1 (detects)
    global amplitude x1.01     0 (CDF is scale invariant)      ~0 (blind)
    1 % of window mass moved   100*(0.01*(1-x0))/L             ~1 (detects)
    ========================== =============================== ==============
    """
    x, rho = ref_prof["x"], ref_prof["rho"]
    span = WINDOW[1] - WINDOW[0]
    m = (x >= WINDOW[0]) & (x <= WINDOW[1])
    mass = float(np.trapezoid(np.abs(rho[m]), x[m]))
    out = {}

    dx3200 = 1.0 / 3200.0
    shifted = _perturbed(ref_prof, lambda xx, rr: np.interp(xx - dx3200, xx, rr))

    def _shift_analytic(xx, rr, delta):
        """d_W1 of a rigid shift delta on a *truncated* window.

        ``D(x) = F(x)-F(lo) - [F(x-delta)-F(lo-delta)] = delta*(rho(x)-rho(lo)) + O(delta^2)``
        because the integral restarts at ``lo``: the identity ``W1 = delta`` only
        holds for a CDF normalised at -inf, not on a truncated window.  This is
        the exact closed form (to O(delta)) the implementation must reproduce.
        """
        mm = (xx >= WINDOW[0]) & (xx <= WINDOW[1])
        xs, rs = xx[mm], np.abs(rr[mm])
        mass_x = float(np.trapezoid(rs, xs))
        integ = float(np.trapezoid(np.abs(rs - rs[0]), xs))
        return 100.0 * delta * integ / (mass_x * (xs[-1] - xs[0]))

    probe = d_w1_percent(ref_prof, shifted, "rho")
    expect = _shift_analytic(ref_prof["x"], ref_prof["rho"], dx3200)
    out["shift_1cell_nx3200"] = {
        "measured": probe, "analytic": expect,
        "ratio": probe / expect if expect else float("nan"),
        "detects": bool(0.95 <= probe / expect <= 1.05),
    }

    # Because the normalisation mass_f is taken from the *reference* (a common
    # denominator, not each profile's own mass), the metric is NOT blind to a
    # global amplitude error: D = -eps*int rho = -eps*mass*Fhat, so
    # d = 100*eps*int(Fhat)dx/L exactly.  (Normalising each profile by its own
    # mass would make it blind -- that choice is deliberately avoided.)
    eps_amp = 0.01
    scaled = _perturbed(ref_prof, lambda xx, rr: (1.0 + eps_amp) * rr)
    probe = d_w1_percent(ref_prof, scaled, "rho")
    xm = ref_prof["x"][m]
    rho_m = np.abs(ref_prof["rho"][m])
    Fhat = np.concatenate(([0.0], np.cumsum(0.5 * (rho_m[1:] + rho_m[:-1]) * np.diff(xm)))) / mass
    expect = 100.0 * eps_amp * float(np.trapezoid(Fhat, xm)) / span
    out["amplitude_1pct"] = {
        "measured": probe, "analytic": expect,
        "ratio": probe / expect if expect else float("nan"),
        "detects": bool(0.9 <= probe / expect <= 1.1),
        "note": "common-denominator normalisation keeps a global amplitude error visible",
    }

    # (1b) exact implementation check: a synthetic profile that is *constant at
    # both window edges* plus a bump well inside the window can be shifted by
    # delta with no mass crossing the window boundary, so W1 = delta exactly.
    # This validates the quadrature of d_w1 itself, independently of any solver.
    def _bumpy(xx):
        return 1.0 + np.exp(-0.5 * ((xx - 0.7) / 0.03) ** 2)

    base = _perturbed(ref_prof, lambda xx, rr: _bumpy(xx))
    base_shift = _perturbed(ref_prof, lambda xx, rr: _bumpy(xx - dx3200))
    probe = d_w1_percent(base, base_shift, "rho")
    expect = _shift_analytic(base["x"], base["rho"], dx3200)
    out["bump_shift_1cell_exact"] = {
        "measured": probe, "analytic": expect,
        "ratio": probe / expect if expect else float("nan"),
        "detects": bool(0.95 <= probe / expect <= 1.05),
    }
    out["shift_1cell_nx3200"]["note"] = (
        "real shock profile: analytic response is delta*int|rho-rho(lo)|/(mass*L), "
        "not delta (truncated-window identity)"
    )

    x0, sigma = 0.7, 0.01
    amp = 0.01 * mass / (sigma * math.sqrt(2.0 * math.pi))
    bumped = _perturbed(ref_prof, lambda xx, rr: rr + amp * np.exp(-0.5 * ((xx - x0) / sigma) ** 2))
    probe = d_w1_percent(ref_prof, bumped, "rho")
    expect = 100.0 * (0.01 * (WINDOW[1] - x0)) / span
    out["mass_1pct_misplaced"] = {
        "measured": probe, "analytic": expect,
        "ratio": probe / expect if expect else float("nan"),
        "detects": bool(0.8 <= probe / expect <= 1.2),
    }
    # (4) negative control: a *known different* solution (the digitised published
    # curve, which carries its own calibration bias) must land far above our own
    # noise floor.  If it did not, d_W1 would be unable to tell right from wrong.
    try:
        from tensorfvm.compressible import wc_reference as wcr

        pub = wcr.load_reference(calibrated=True)
        pub_prof = {"x": np.asarray(pub["x"], float), "rho": np.asarray(pub["rho"], float),
                    "p": np.asarray(pub["p"], float) if "p" in pub else None,
                    "nx": int(np.size(pub["x"])), "family": "published"}
        probe = d_w1_percent(ref_prof, pub_prof, "rho")
        out["published_curve_control"] = {
            "measured": probe,
            "floor_percent": floor_percent,
            "ratio_to_floor": probe / floor_percent if floor_percent else float("nan"),
            "detects": bool(math.isfinite(probe) and floor_percent and probe > 3.0 * floor_percent),
            "note": ("digitised RPJ@3200 curve: an independent, known-different solution; "
                     "must land far above our own d_W1 floor"),
        }
    except Exception as exc:                      # pragma: no cover - data guard
        out["published_curve_control"] = {"error": f"{type(exc).__name__}: {exc}", "detects": False}
    return out


def localize(fine: dict, coarse: dict, var: str = "rho") -> dict:
    """Where does the accumulated difference live?

    ``w_eff = int|D| / max|D|`` is the effective width of the residual in units
    of the *coarse* cell: a few cells means a shock-band artefact, a large value
    means a broad (smooth-region) mismatch.  ``net_mass = int e`` over the
    window is the part a conservative transport metric cannot see.
    """
    f = cumulative_diff_field(fine, coarse, var)
    D, x = f["D"], f["x"]
    if D.size == 0 or not np.any(D):
        return {"w_eff_cells": float("nan"), "x_peak": float("nan"),
                "net_mass": 0.0, "max_abs": 0.0}
    amax = float(np.max(np.abs(D)))
    integ = float(np.trapezoid(np.abs(D), x))
    dx_c = (WINDOW[1] - WINDOW[0]) / coarse["nx"]
    w_eff = integ / amax if amax > 0 else float("nan")
    span = float(x[-1] - x[0])
    return {
        "w_eff_length": w_eff,
        "w_eff_frac_span": w_eff / span if math.isfinite(w_eff) else float("nan"),
        "w_eff_coarse_cells": w_eff / dx_c if math.isfinite(w_eff) else float("nan"),
        "x_peak": float(x[int(np.argmax(np.abs(D)))]),
        "net_mass": float(D[-1]),
        "rel_net_mass": abs(float(D[-1])) / window_mass(coarse, var) if window_mass(coarse, var) else float("nan"),
        "max_abs": amax,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, default=None, help="write the budget here")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    out = build(verbose=not args.quiet)
    ARTF = ARTIFACTS
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2, default=float) + "\n")
    else:
        ARTF.mkdir(parents=True, exist_ok=True)
        (ARTF / "wc_w1_norm.json").write_text(json.dumps(out, indent=2, default=float) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
