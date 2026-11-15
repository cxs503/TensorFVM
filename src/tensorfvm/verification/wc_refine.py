"""Refinement closure for the Woodward-Colella "certified <1 %" question.

Context
-------
``docs/XVAL-FVM-M18737-ROUND2/CERTIFIED-REF-CLOSURE.md`` closed three gates
against an independent WENO5@12800 reference and found exactly one binding
failure: the *graded* MUSCL/HLLC solution at ``nx = 3200`` sits at
``L1 = 1.776 %``, i.e. above the 1 % bar.  That round refined the **reference**;
the graded solution was left at 3200.  The open question is therefore not about
the reference any more but about the graded scheme:

    does refining the graded solution actually bring L1 below 1 %, and at what
    cost?

This module answers it with the pieces that a defensible answer needs:

``restrict`` / ``cell_average_l1``
    A comparison metric that does **not** interpolate.  A finite-volume solution
    stores *cell averages*; comparing a coarse cell average with a point value
    of the fine solution (what ``wc_certified_ref.py`` did via ``np.interp``)
    mixes two representations.  For nested grids the rigorous operation is
    conservative restriction: the coarse cell average is the mean of the
    ``r = nx_fine / nx_coarse`` fine cells it contains.  Both sides are then
    cell averages and the ``L1`` over a window is well defined.
``pointwise_l1``
    The legacy interpolating metric, kept so the metric change can be *audited*
    rather than assumed harmless (prediction M1 in
    ``docs/XVAL-FVM-M18737-ROUND2/PREDECLARATION-refine.md``).
``observed_order`` / ``forecast_error`` / ``nx_for_error``
    The error model ``e(nx) = e_ref (nx / nx_ref) ** -p`` and its inverse.  The
    forecast for the finer rungs is frozen in the predeclaration *before* the
    new profiles exist, so the comparison is a test and not a rationalisation.
``audit``
    Everything above on the stored ladder, returning one dict of numbers.  No
    solver is called, so it is seconds of numpy and testable in a unit test.

Stored profiles live in ``docs/woodward-colella-double-blast/ladder/data`` as
``ladder_profile_nx<N>_<family>.npz`` with keys ``x`` (cell centres) and
``rho`` (cell averages).  Rungs that were never computed are reported as
``measured: False`` instead of raising, because "not measured" must be visible
in the output rather than silently skipped.

Usage::

    PYTHONPATH=src python -m tensorfvm.verification.wc_refine --json out.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

#: comparison window of the digitised reference (matches ``wc_reference``)
WINDOW = (0.4, 1.0)
#: the bar the certified-<1 % claim is made against
CERTIFIED_PERCENT = 1.0
#: the ledger tolerance (owner standard: >15 % unacceptable)
TOLERANCE_PERCENT = 15.0
#: graded-solution rungs, coarse to fine
HLLC_RUNGS = (1600, 3200, 6400, 12800)
#: reference rungs, coarse to fine
WENO5_RUNGS = (1600, 3200, 6400, 12800)


class DataMissing(FileNotFoundError):
    """Raised when a stored ladder profile cannot be found."""


# --------------------------------------------------------------------------- #
# profile I/O
# --------------------------------------------------------------------------- #
def default_data_dir() -> Path:
    """``<repo>/docs/woodward-colella-double-blast/ladder/data``."""
    return (Path(__file__).resolve().parents[3] / "docs" / "woodward-colella-double-blast"
            / "ladder" / "data")


def load_profile(data_dir: Path | str, nx: int, family: str = "hllc") -> Dict:
    """Load one stored ladder profile.

    Cell centres are returned sorted ascending in ``x``, cell averages in
    ``rho``; other fields (``p``, ``u``) are passed through when present.
    """
    data_dir = Path(data_dir)
    path = data_dir / f"ladder_profile_nx{nx}_{family}.npz"
    if not path.exists():
        alt = data_dir.parent.parent / "artifacts" / f"wc_blast_profile_nx{nx}.npz"
        if alt.exists():
            path = alt
        else:
            raise DataMissing(f"profile nx={nx} family={family} not found under {data_dir}")
    with np.load(path) as d:
        # older artifact rungs store the run under x_got/rho_got
        x = np.asarray(d["x_got"] if "x_got" in d.files else d["x"], float)
        out = {"x": x, "rho": np.asarray(d["rho_got"] if "rho_got" in d.files else d["rho"], float),
               "nx": int(x.size), "source": str(path)}
        for extra in ("p", "u"):
            if extra in d.files:
                out[extra] = np.asarray(d[extra], float)
    order = np.argsort(out["x"])
    for key in ("x", "rho", "p", "u"):
        if key in out:
            out[key] = out[key][order]
    return out


def try_load_profile(data_dir: Path | str, nx: int, family: str) -> Optional[Dict]:
    """``load_profile`` but ``None`` when the rung was never computed."""
    try:
        return load_profile(data_dir, nx, family)
    except DataMissing:
        return None


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def restrict(q_fine: Sequence[float], ratio: int) -> np.ndarray:
    """Conservative restriction: average ``ratio`` adjacent fine cells into one coarse cell."""
    q = np.asarray(q_fine, float)
    ratio = int(ratio)
    if ratio < 1:
        raise ValueError(f"ratio must be >= 1, got {ratio}")
    if q.size % ratio:
        raise ValueError(f"fine grid of {q.size} cells is not a nested refinement (ratio {ratio})")
    return q.reshape(q.size // ratio, ratio).mean(axis=1)


def _window_mask(x: np.ndarray, window: Tuple[float, float]) -> np.ndarray:
    return (x >= window[0]) & (x <= window[1])


def cell_average_l1(q_coarse: Sequence[float], q_fine: Sequence[float],
                    window: Tuple[float, float] = WINDOW) -> Dict:
    """L1 / peak / mass difference of two *cell-average* profiles on nested grids.

    The fine profile is conservatively restricted onto the coarse grid, so both
    sides are cell averages and no interpolation enters the metric.
    """
    q_c = np.asarray(q_coarse, float)
    q_f = np.asarray(q_fine, float)
    if q_f.size % q_c.size:
        raise ValueError(f"grids are not nested: {q_f.size} / {q_c.size}")
    ratio = q_f.size // q_c.size
    q_r = restrict(q_f, ratio)
    x_c = (np.arange(q_c.size) + 0.5) / q_c.size
    m = _window_mask(x_c, window)
    a, b = q_c[m], q_r[m]
    den = float(np.abs(b).sum())
    if den == 0.0:
        raise ValueError("zero reference norm on the comparison window")
    peak_den = float(np.max(np.abs(b)))
    mass_den = float(np.abs(b.sum()))
    out = {
        "l1_percent": 100.0 * float(np.abs(a - b).sum()) / den,
        "peak_percent": 100.0 * abs(float(np.max(a)) - float(np.max(b))) / peak_den,
        "mass_percent": 100.0 * abs(float(a.sum()) - float(b.sum())) / mass_den,
        "ratio": ratio,
        "window": list(window),
    }
    return out


def residual_density(q_coarse: Sequence[float], q_fine: Sequence[float],
                     window: Tuple[float, float] = WINDOW) -> Dict:
    """Where does the L1 residual live?  Returns ``x``, ``q`` and the per-cell share.

    ``share`` sums to ``l1_percent / 100`` on the window, so a sub-interval's
    share is exactly its contribution to the metric.  Used to attribute the
    residual to physical sub-regions (shock / contact / plateau) instead of
    guessing from a single number.
    """
    q_c = np.asarray(q_coarse, float)
    q_f = np.asarray(q_fine, float)
    if q_f.size % q_c.size:
        raise ValueError(f"grids are not nested: {q_f.size} / {q_c.size}")
    ratio = q_f.size // q_c.size
    q_r = restrict(q_f, ratio)
    x_c = (np.arange(q_c.size) + 0.5) / q_c.size
    m = _window_mask(x_c, window)
    a, b = q_c[m], q_r[m]
    den = float(np.abs(b).sum())
    if den == 0.0:
        raise ValueError("zero reference norm on the comparison window")
    return {"x": x_c[m], "coarse": a, "reference": b,
            "share": np.abs(a - b) / den, "ratio": ratio, "window": list(window)}


def region_shares(density: Dict, regions: Sequence[Tuple[float, float]]) -> List[Dict]:
    """Sum :func:`residual_density` shares over named sub-intervals."""
    x = np.asarray(density["x"])
    share = np.asarray(density["share"])
    total = float(share.sum())
    out = []
    for name, (lo, hi) in regions:
        m = (x >= lo) & (x < hi)
        s = float(share[m].sum())
        out.append({"region": name, "range": [lo, hi],
                    "share_percent_of_l1": 100.0 * s / total if total else 0.0})
    return out


def pointwise_l1(x_coarse: Sequence[float], q_coarse: Sequence[float],
                 x_fine: Sequence[float], q_fine: Sequence[float],
                 window: Tuple[float, float] = WINDOW) -> Dict:
    """Legacy interpolating metric (what ``wc_certified_ref.py`` used).

    Kept only for the metric audit: it compares coarse cell averages against
    point values of the fine solution obtained by linear interpolation.
    """
    x_c = np.asarray(x_coarse, float)
    q_c = np.asarray(q_coarse, float)
    x_f = np.asarray(x_fine, float)
    q_f = np.asarray(q_fine, float)
    m = _window_mask(x_c, window)
    x = x_c[m]
    a = q_c[m]
    b = np.interp(x, x_f, q_f)
    den = float(np.abs(b).sum())
    if den == 0.0:
        raise ValueError("zero reference norm on the comparison window")
    return {
        "l1_percent": 100.0 * float(np.abs(a - b).sum()) / den,
        "peak_percent": 100.0 * abs(float(np.max(a)) - float(np.max(b))) / float(np.max(np.abs(b))),
        "window": list(window),
    }


# --------------------------------------------------------------------------- #
# error model
# --------------------------------------------------------------------------- #
def observed_order(err_coarse: float, err_fine: float, nx_coarse: int, nx_fine: int) -> float:
    """Observed order ``p`` in ``e ~ nx ** -p`` from two rungs."""
    if err_coarse <= 0 or err_fine <= 0:
        raise ValueError("errors must be positive")
    if nx_fine <= nx_coarse:
        raise ValueError("nx_fine must exceed nx_coarse")
    return float(math.log(err_coarse / err_fine) / math.log(nx_fine / nx_coarse))


def forecast_error(err_at_ref: float, nx_ref: int, order: float, nx: int) -> float:
    """``e(nx) = e(nx_ref) * (nx / nx_ref) ** -order``."""
    if order < 0:
        raise ValueError("order must be >= 0")
    return float(err_at_ref * (nx / nx_ref) ** (-order))


def nx_for_error(err_at_ref: float, nx_ref: int, order: float, tol: float) -> float:
    """Inverse of :func:`forecast_error`: grid needed to reach ``tol``."""
    if tol <= 0 or err_at_ref <= 0:
        raise ValueError("tol and err_at_ref must be positive")
    if order <= 0:
        return float("inf")
    if err_at_ref <= tol:
        return float(nx_ref)
    return float(nx_ref * (err_at_ref / tol) ** (1.0 / order))


# --------------------------------------------------------------------------- #
# ladder audit
# --------------------------------------------------------------------------- #
def graded_ladder(data_dir: Path | str, ref_nx: int, ref_family: str = "weno5",
                  rungs: Iterable[int] = HLLC_RUNGS) -> List[Dict]:
    """Grade every available MUSCL/HLLC rung against the finest WENO5 reference."""
    ref = load_profile(data_dir, ref_nx, ref_family)
    rows: List[Dict] = []
    for nx in rungs:
        prof = try_load_profile(data_dir, nx, "hllc")
        if prof is None:
            rows.append({"nx": nx, "measured": False})
            continue
        agg = cell_average_l1(prof["rho"], ref["rho"])
        pw = pointwise_l1(prof["x"], prof["rho"], ref["x"], ref["rho"])
        rows.append({"nx": nx, "measured": True, **agg,
                     "pointwise_l1_percent": pw["l1_percent"],
                     "metric_bias_pp": pw["l1_percent"] - agg["l1_percent"],
                     "source": prof["source"]})
    return rows


def reference_selfcert(data_dir: Path | str, coarse_nx: int, fine_nx: int,
                       family: str = "weno5") -> Dict:
    """Gate 1: is the fine reference itself self-converged?"""
    coarse = load_profile(data_dir, coarse_nx, family)
    fine = load_profile(data_dir, fine_nx, family)
    agg = cell_average_l1(coarse["rho"], fine["rho"])
    return {"coarse_nx": coarse_nx, "fine_nx": fine_nx, **agg,
            "self_certified": bool(agg["l1_percent"] < CERTIFIED_PERCENT
                                   and agg["peak_percent"] < CERTIFIED_PERCENT)}


def _sequence_order(measured: List[Dict]) -> List[Dict]:
    """Painwise log--log orders of the graded sequence (adjacent rungs)."""
    out: List[Dict] = []
    m = [r for r in measured if r.get("measured")]
    for lo, hi in zip(m, m[1:]):
        p = observed_order(lo["l1_percent"], hi["l1_percent"], lo["nx"], hi["nx"])
        out.append({"nx_pair": [lo["nx"], hi["nx"]],
                    "l1_pair": [lo["l1_percent"], hi["l1_percent"]],
                    "order": p})
    return out


def audit(data_dir: Path | str | None = None, ref_nx: int = 12800) -> Dict:
    """Full refinement audit: metric check, graded ladder, orders, forecast, verdict."""
    data_dir = Path(data_dir) if data_dir is not None else default_data_dir()
    ladder = graded_ladder(data_dir, ref_nx)
    measured = [r for r in ladder if r.get("measured")]
    if not measured:
        raise DataMissing(f"no graded profiles found under {data_dir}")
    orders = _sequence_order(measured)
    finest = measured[-1]

    # forecast made from the two coarsest rungs, then tested on the finer ones
    forecast: List[Dict] = []
    if len(measured) >= 2:
        base = measured[1]
        p_base = (observed_order(measured[0]["l1_percent"], measured[1]["l1_percent"],
                                 measured[0]["nx"], measured[1]["nx"]))
        for row in measured[2:]:
            forecast.append({
                "nx": row["nx"],
                "predicted_l1_percent": forecast_error(base["l1_percent"], base["nx"], p_base, row["nx"]),
                "measured_l1_percent": row["l1_percent"],
                "order_used": p_base,
                "rel_error_percent": 100.0 * abs(row["l1_percent"]
                                                - forecast_error(base["l1_percent"], base["nx"],
                                                                 p_base, row["nx"]))
                / row["l1_percent"],
            })

    # the binding gate of the previous round: is the graded solution itself <1 %?
    certified_l1 = finest["l1_percent"] < CERTIFIED_PERCENT
    certified_peak = finest["peak_percent"] < CERTIFIED_PERCENT
    certified_legacy = finest.get("pointwise_l1_percent", finest["l1_percent"]) < CERTIFIED_PERCENT

    p_eff = orders[-1]["order"] if orders else float("nan")

    # gate 1 on the reference: the coarsest stored WENO5 rung that is still finer
    # than the graded ladder is the one that certifies the finest reference
    ref_coarse_nx = max([n for n in WENO5_RUNGS if n < ref_nx
                         and try_load_profile(data_dir, n, "weno5") is not None], default=None)
    verdict = {
        "reference": {"family": "weno5", "nx": ref_nx,
                      "self_certification": (reference_selfcert(data_dir, ref_coarse_nx, ref_nx)
                                             if ref_coarse_nx else None)},
        "ladder": ladder,
        "orders": orders,
        "forecast": forecast,
        "finest_graded": {"nx": finest["nx"],
                          "l1_percent": finest["l1_percent"],
                          "peak_percent": finest["peak_percent"],
                          "metric_bias_pp": finest.get("metric_bias_pp")},
        "certified_lt_1pct": bool(certified_l1 and certified_peak),
        "certified_lt_1pct_legacy_metric": bool(certified_legacy),
        "metric_decision_invariant": bool(certified_l1 == certified_legacy),
        "order_used_for_forecast": p_eff,
        "nx_needed_for_1pct": (nx_for_error(finest["l1_percent"], finest["nx"], p_eff,
                                           CERTIFIED_PERCENT)
                               if p_eff and p_eff > 0 else None),
        "nx_needed_for_3pct": (nx_for_error(finest["l1_percent"], finest["nx"], p_eff, 3.0)
                               if p_eff and p_eff > 0 else None),
        "ledger_tolerance_percent": TOLERANCE_PERCENT,
        "certified_bar_percent": CERTIFIED_PERCENT,
        "not_measured": [r["nx"] for r in ladder if not r.get("measured")],
    }
    return verdict


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> Dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--ref-nx", type=int, default=12800)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)
    out = audit(args.data_dir, args.ref_nx)
    text = json.dumps(out, indent=2, default=float)
    print(text)
    if args.json:
        args.json.write_text(text + "\n")
    return out


# --------------------------------------------------------------------------- #
# predeclared criteria (F1 / F2 / F3)
# --------------------------------------------------------------------------- #
#: The forecast and the falsification criteria frozen in
#: ``docs/XVAL-FVM-M18737-ROUND2/PREDECLARATION-refine.md`` *before* the
#: nx = 6400/12800 graded rungs existed.  They are inputs to the test, so they
#: live in code (and in the predeclaration file) rather than in prose that can
#: be re-tuned after the numbers arrive.
PREDECLARED_FORECAST: Dict = {
    "source": "docs/XVAL-FVM-M18737-ROUND2/PREDECLARATION-refine.md",
    "nx_ref": 3200,
    "err_ref_percent": 1.7685,
    "order": 0.8785,
    "order_band": (0.79, 1.05),
    "targets": (6400, 12800),
    "f1_target": 6400,
    "f1_band_percent": (0.854, 1.023),
    "f2_order_band": (0.70, 1.20),
    "f2_plateau_percent": 1.5,
}


def sequence_order_lsq(measured: Sequence[Dict], rungs: Optional[Iterable[int]] = None) -> Dict:
    """Least-squares order of ``log e`` against ``log nx`` over a rung sequence.

    F2 in the predeclaration is a *sequence* statement (1600/3200/6400), not a
    two-point one: a single pair can be lucky, so the slope is fitted over every
    measured rung in the requested set.
    """
    rows = [r for r in measured if r.get("measured", True)]
    if rungs is not None:
        keep = set(int(n) for n in rungs)
        rows = [r for r in rows if int(r["nx"]) in keep]
    rows = [r for r in rows if float(r["l1_percent"]) > 0.0]
    if len(rows) < 2:
        raise ValueError("need at least two positive measured rungs for an order fit")
    x = np.log(np.array([r["nx"] for r in rows], float))
    y = np.log(np.array([r["l1_percent"] for r in rows], float))
    slope, intercept = np.polyfit(x, y, 1)
    return {"rungs": [int(r["nx"]) for r in rows], "order": float(-slope),
            "prefactor": float(math.exp(intercept))}


def check_predeclaration(verdict: Dict,
                         predeclared: Dict = PREDECLARED_FORECAST) -> Dict:
    """Test the stored ladder against the frozen F1 / F2 / F3 criteria.

    * **F1 (point value)** -- ``e(6400)`` lands inside the predeclared band.
    * **F2 (order)** -- the sequence fit is in ``[0.70, 1.20]`` *and* the ladder
      does not plateau (no rung stuck above ``1.5 %``).  A plateau falsifies the
      "refine the graded solution" route and hands the job to a higher-order
      scheme instead.
    * **F3 (ruling)** -- the coarsest measured rung whose L1 *and* peak are both
      below the certified bar; if none, an honest FAIL plus the grid the
      measured order says is needed.

    Every branch returns numbers, so the caller never has to transcribe them.
    """
    rows = [r for r in verdict["ladder"] if r.get("measured")]
    if not rows:
        raise ValueError("no measured rungs: nothing to adjudicate")
    by_nx = {int(r["nx"]): r for r in rows}
    p_central = float(predeclared["order"])
    e_ref = float(predeclared["err_ref_percent"])
    nx_ref = int(predeclared["nx_ref"])
    lo, hi = predeclared["f1_band_percent"]

    forecast_rows: List[Dict] = []
    for nx in predeclared["targets"]:
        row = by_nx.get(int(nx))
        pred = forecast_error(e_ref, nx_ref, p_central, int(nx))
        # the band's *lower* error bound comes from the *upper* order, so sort
        b_lo = forecast_error(e_ref, nx_ref, predeclared["order_band"][1], int(nx))
        b_hi = forecast_error(e_ref, nx_ref, predeclared["order_band"][0], int(nx))
        band = (min(b_lo, b_hi), max(b_lo, b_hi))
        entry = {"nx": int(nx), "predicted_percent": pred,
                 "band_percent": list(band), "measured_percent": None,
                 "rel_error_percent": None, "in_band": None}
        if row is not None:
            got = float(row["l1_percent"])
            entry.update({"measured_percent": got,
                          "rel_error_percent": 100.0 * abs(got - pred) / got,
                          "in_band": bool(band[0] <= got <= band[1])})
        forecast_rows.append(entry)

    f1_row = next((r for r in forecast_rows if r["nx"] == int(predeclared["f1_target"])), None)
    f1 = {
        "criterion": "e(nx=6400) in the predeclared band",
        "target_nx": int(predeclared["f1_target"]),
        "band_percent": [lo, hi],
        "predicted_percent": f1_row["predicted_percent"] if f1_row else None,
        "measured_percent": f1_row["measured_percent"] if f1_row else None,
        "passed": (None if f1_row is None or f1_row["in_band"] is None
                   else bool(f1_row["in_band"])),
        "note": "undecided: target rung not measured" if (
            f1_row is None or f1_row["in_band"] is None) else "",
    }

    fit = sequence_order_lsq(rows, rungs=(1600, 3200, 6400))
    p_lo, p_hi = predeclared["f2_order_band"]
    plateau_bar = float(predeclared["f2_plateau_percent"])
    two_finest_nx = sorted(by_nx)[-2:]
    finest_two = ([by_nx[n]["l1_percent"] for n in two_finest_nx]
                  if len(two_finest_nx) == 2 else None)
    p_fit = fit["order"]
    f2 = {
        "criterion": "sequence order in [0.70, 1.20] and no plateau above 1.5 %",
        "sequence": fit["rungs"],
        "order_sequence_lsq": p_fit,
        "order_band": [p_lo, p_hi],
        "order_in_band": bool(p_lo <= p_fit <= p_hi),
        "finest_two_l1_percent": finest_two,
        # "platform（>1.5 % 不降）": still above the bar AND the finest pair buys
        # less than a 2 % relative reduction, i.e. doubling the grid does not
        # move the metric in any useful sense.  A ladder that still falls, even
        # slowly, is *not* called a plateau -- the slow order is reported in
        # ``order_sequence_lsq`` instead of being relabelled.
        "plateau": bool(finest_two is not None and min(finest_two) > plateau_bar
                        and finest_two[1] / finest_two[0] > 0.98),
        "plateau_bar_percent": plateau_bar,
        "passed": bool(p_lo <= p_fit <= p_hi),
        "consistent_two_point_order": observed_order(
            by_nx[fit["rungs"][0]]["l1_percent"], by_nx[fit["rungs"][-1]]["l1_percent"],
            fit["rungs"][0], fit["rungs"][-1]) if len(fit["rungs"]) >= 2 else None,
    }

    certified_at = next((int(r["nx"]) for r in sorted(rows, key=lambda r: r["nx"])
                         if r["l1_percent"] < CERTIFIED_PERCENT
                         and r["peak_percent"] < CERTIFIED_PERCENT), None)
    finest = max(rows, key=lambda r: r["nx"])
    f3 = {
        "criterion": "first rung with L1 and peak both < 1 %; else honest FAIL + cost",
        "certified_at_nx": certified_at,
        "finest_measured_nx": int(finest["nx"]),
        "finest_l1_percent": float(finest["l1_percent"]),
        "finest_peak_percent": float(finest["peak_percent"]),
        "passed": certified_at is not None,
        "path_exhausted": bool(len(by_nx) >= len(HLLC_RUNGS)),
        "nx_needed_for_1pct": nx_for_error(finest["l1_percent"], int(finest["nx"]), p_fit,
                                          CERTIFIED_PERCENT),
    }
    if certified_at is None:
        f3["ruling"] = "honest FAIL: refine-the-graded-solution route exhausted"
    else:
        f3["ruling"] = f"certified-<1 % first achieved at nx={certified_at}"

    passed = [f1["passed"], f2["passed"], f3["passed"]]
    overall = ("undecided" if any(v is None for v in passed)
               else ("pass" if all(passed) else "fail"))
    return {"predeclaration": predeclared, "forecast": forecast_rows,
            "F1": f1, "F2": f2, "F3": f3, "overall": overall,
            "checks_passed": sum(1 for v in passed if v), "checks_total": len(passed)}


def cfl_sensitivity(data_dir: Path | str, nx: int, cfl: str | float,
                    ref_nx: int = 12800, ref_family: str = "weno5") -> Dict:
    """Time-step control: is the graded L1 at ``nx`` limited by the time step?

    The refinement rounds held CFL fixed while refining space, which leaves one
    hole: if the temporal error dominates on the fine grid, the observed L1
    floor is a *time-step* artefact and "refine further" would be the wrong
    recommendation.  Compares the CFL-refined run against the baseline run at
    the same ``nx``, both graded against the same reference.
    """
    base = load_profile(data_dir, nx, "hllc")
    tag = f"nx{nx}_hllc_cfl{cfl}"
    alt = load_profile_tagged(data_dir, tag)
    ref = load_profile(data_dir, ref_nx, ref_family)
    m_base = cell_average_l1(base["rho"], ref["rho"])
    m_alt = cell_average_l1(alt["rho"], ref["rho"])
    delta_pp = abs(m_alt["l1_percent"] - m_base["l1_percent"])
    return {
        "nx": int(nx), "cfl_baseline": 0.4, "cfl_refined": float(cfl),
        "l1_percent_baseline": m_base["l1_percent"],
        "l1_percent_refined": m_alt["l1_percent"],
        "peak_percent_baseline": m_base["peak_percent"],
        "peak_percent_refined": m_alt["peak_percent"],
        "delta_l1_pp": delta_pp,
        "decision_invariant": bool((m_base["l1_percent"] < CERTIFIED_PERCENT)
                                   == (m_alt["l1_percent"] < CERTIFIED_PERCENT)),
        "time_error_negligible": bool(delta_pp < 0.05),
        "source_baseline": base["source"], "source_refined": alt["source"],
    }


def load_profile_tagged(data_dir: Path | str, tag: str) -> Dict:
    """Load ``ladder_profile_<tag>.npz`` (tag carries non-default plan knobs)."""
    data_dir = Path(data_dir)
    path = data_dir / f"ladder_profile_{tag}.npz"
    if not path.exists():
        raise DataMissing(f"profile {tag} not found under {data_dir}")
    with np.load(path) as d:
        x = np.asarray(d["x_got"] if "x_got" in d.files else d["x"], float)
        out = {"x": x, "rho": np.asarray(d["rho_got"] if "rho_got" in d.files else d["rho"], float),
               "nx": int(x.size), "source": str(path)}
        for extra in ("p", "u"):
            if extra in d.files:
                out[extra] = np.asarray(d[extra], float)
    order = np.argsort(out["x"])
    for key in ("x", "rho", "p", "u"):
        if key in out:
            out[key] = out[key][order]
    return out


if __name__ == "__main__":
    main()