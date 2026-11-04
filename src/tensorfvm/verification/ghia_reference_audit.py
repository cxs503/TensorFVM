"""Consistency audit of the transcribed Ghia et al. (1982) centreline tables.

The m3003 lid-cavity ladder left one entry red: at Re = 400 the v-centreline
residual *grows* under grid refinement (0.115 / 0.142 / 0.149 for n = 32 / 64 /
128) while u and both primary extrema converge.  The ladder report called it a
wall-adjacent extraction defect and left it unrefined.

This module decides the attribution from the committed three-grid evidence
without any new CFD.  For every published point two causes of a residual are
separated:

* **discretisation / extraction** - ``|delta|`` must shrink with ``h``;
* **reference defect** - ``|delta|`` saturates at a grid-independent value, because
  the numerical profiles keep converging to one smooth limit that is not the
  tabulated number.

Per point it reports

``ratio_prev_over_finest``    ``|delta(n/2)| / |delta(n)|``; ~4 is 2nd order,
                              ~1 means the residual stopped responding to the grid;
``resolution_uncertainty``    ``|value(n) - value(n/2)|``, i.e. the sample's own
                              uncertainty (interpolation + discretisation);
``residual_over_uncertainty`` how far outside its own uncertainty the residual is.

It also runs a *control* smoothness test on the tabulated profile itself (divided
differences vs the curvature present in the discrete field).  That test is
reported but **not** used for the verdict: genuine published near-extrema
(x = 0.2344 in every Reynolds) produce the same large slope jumps, so it has no
discriminating power on its own.

The verdict is consumed by :func:`tensorfvm.lid_cavity_reference.suspect_entries`
through the hand-curated :data:`SUSPECT_ENTRIES` table; this module is the
machine evidence behind that curation.

Usage from a script: ``audit_bundle(json.load(open(summary)))``.
"""

from __future__ import annotations

import numpy as np

from ..lid_cavity_reference import TABLE_PROVENANCE, table

ACCEPTANCE_BAND = 0.03  # ``ghia_acceptance`` pointwise band (units of the lid speed)


def grid_profile(field: list[list[float]], n_half: int,
                  which: str = "v") -> tuple[np.ndarray, np.ndarray]:
    """Return ``(coord, value)`` on the requested centreline of a cached field.

    Cached layout is the staggered one: ``u`` has shape ``(n+1, n)`` (axis 0 =
    x face, axis 1 = y cell) and ``v`` has shape ``(n, n+1)`` (axis 0 = x cell,
    axis 1 = y face).  ``u`` therefore lives on the *row* ``x = L/2`` and ``v``
    on the *column* ``y = L/2``; mixing the two axes is exactly the kind of
    extraction error this audit has to rule out.
    """
    arr = np.asarray(field, dtype=float)
    n = n_half * 2
    if which == "u":
        values = arr[n_half, :]
    else:
        values = arr[:, n_half]
    coord = (np.arange(len(values)) + 0.5) / n
    return coord, values


def hermite_uniform(x: np.ndarray, f: np.ndarray, targets) -> np.ndarray:
    """Catmull-Rom cubic, bit-identical to ``tensorfvm.lid_cavity.hermite_interpolate``."""
    slopes = np.empty_like(f)
    slopes[1:-1] = (f[2:] - f[:-2]) / (x[2:] - x[:-2])
    slopes[0] = (f[1] - f[0]) / (x[1] - x[0])
    slopes[-1] = (f[-1] - f[-2]) / (x[-1] - x[-2])
    t = np.asarray(list(targets), dtype=float)
    idx = np.clip(np.searchsorted(x, t) - 1, 0, len(x) - 2)
    x0, x1 = x[idx], x[idx + 1]
    f0, f1 = f[idx], f[idx + 1]
    m0, m1 = slopes[idx], slopes[idx + 1]
    s = (t - x0) / (x1 - x0)
    h00 = 2 * s**3 - 3 * s**2 + 1
    h10 = s**3 - 2 * s**2 + s
    h01 = -2 * s**3 + 3 * s**2
    h11 = s**3 - s**2
    return h00 * f0 + h10 * (x1 - x0) * m0 + h01 * f1 + h11 * (x1 - x0) * m1


def _field_slope_bound(x: np.ndarray, f: np.ndarray, x0: float, window: int = 8) -> float:
    """Max ``|dv/dx|`` within ``window`` cells of ``x0``, from the discrete column."""
    centre = int(np.argmin(np.abs(x - x0)))
    lo, hi = max(1, centre - window), min(len(x) - 1, centre + window)
    slopes = np.abs(np.diff(f[lo - 1 : hi + 1]) / np.diff(x[lo - 1 : hi + 1]))
    return float(slopes.max())


def audit_bundle(bundle: dict) -> dict:
    cases = {(float(c["reynolds"]), int(c["n"])): c for c in bundle["cases"]}
    reynolds_values = sorted({re for re, _ in cases})
    grids = {re: sorted(n for r, n in cases if r == re) for re in reynolds_values}

    report: dict = {
        "title": "Ghia (1982) reference-table consistency audit",
        "bundle": "docs/verification-lid-cavity-ladder/summary.json",
        "provenance": dict(TABLE_PROVENANCE),
        "acceptance_band": ACCEPTANCE_BAND,
        "reynolds": {},
    }

    for re in reynolds_values:
        ns = grids[re]
        block: dict = {"grids": ns, "which": {}}
        for which, key in (("u", "u"), ("v", "v")):
            points = table(re, which)
            targets = [p[0] for p in points]
            ref = np.asarray([p[1] for p in points], dtype=float)
            # interpolated profiles per grid
            sampled = {}
            for n in ns:
                field = cases[(re, n)]["field"][key]
                x, f = grid_profile(field, n // 2, which)
                sampled[n] = (hermite_uniform(x, f, targets), x, f)
            finest = ns[-1]
            prev = ns[-2] if len(ns) > 1 else None
            rows = []
            for j, x0 in enumerate(targets):
                row = {
                    "x": x0,
                    "reference": float(ref[j]),
                    "computed": {str(n): float(sampled[n][0][j]) for n in ns},
                    "delta": {str(n): float(sampled[n][0][j] - ref[j]) for n in ns},
                }
                d_f = row["delta"][str(finest)]
                if prev is None:  # single-grid Reynolds: no refinement information
                    row["ratio_prev_over_finest"] = None
                    row["resolution_uncertainty"] = None
                    row["residual_over_uncertainty"] = None
                else:
                    d_p = row["delta"][str(prev)]
                    row["ratio_prev_over_finest"] = (
                        abs(d_p) / abs(d_f) if d_f != 0.0 else float("inf")
                    )
                    # resolution uncertainty of the finest sample: |value_n - value_(n/2)|
                    row["resolution_uncertainty"] = abs(row["computed"][str(finest)]
                                                        - row["computed"][str(prev)])
                    row["residual_over_uncertainty"] = (
                        abs(d_f) / row["resolution_uncertainty"]
                        if row["resolution_uncertainty"] > 0.0 else float("inf")
                    )
                rows.append(row)
            block["which"][which] = {
                "points": rows,
                "max_abs_error": {str(n): float(np.abs(sampled[n][0] - ref).max()) for n in ns},
                "rms_error": {str(n): float(np.sqrt(((sampled[n][0] - ref) ** 2).mean()))
                              for n in ns},
            }
        block["shape"] = shape_checks(cases, re, grids[re])
        report["reynolds"][str(re)] = block

    report["verdict"] = classify(report)
    return report


def curvature_bound(x: np.ndarray, f: np.ndarray, x0: float, window: int = 8) -> float:
    """Max ``|d2f/dx2|`` of the discrete column within ``window`` cells of ``x0``."""
    centre = int(np.argmin(np.abs(x - x0)))
    lo = max(1, centre - window)
    hi = min(len(x) - 1, centre + window)
    d2 = (f[lo + 1 : hi + 1] - 2.0 * f[lo:hi] + f[lo - 1 : hi - 1]) / (x[1] - x[0]) ** 2
    return float(np.abs(d2).max())


def shape_checks(cases: dict, re: float, ns: list[int]) -> dict:
    """Smoothness test on the *tabulated* profile itself.

    The centreline table is a sampling of a smooth solution, so the first divided
    difference may vary between neighbouring intervals but not by an order of
    magnitude: a jump ``j`` over a spacing ``dx`` costs a curvature
    ``|f''| ~ j / dx``.  The same curvature is measured on the finest discrete
    column, which is a faithful sample of the same smooth profile.  A table whose
    implied curvature exceeds the field's own by a large factor at one point, while
    every other interval behaves, is not a sampling of this flow.
    """
    which = "v"
    points = table(re, which)
    targets = [p[0] for p in points]
    ref = np.asarray([p[1] for p in points], dtype=float)
    n = ns[-1]
    x, f = grid_profile(cases[(re, n)]["field"][which], n // 2, which)
    d1 = np.diff(ref) / np.diff(targets)          # intervals i -> i+1, dx negative
    out: dict = {"grid": n, "rows": []}
    for j in range(1, len(targets) - 1):
        left, right = d1[j - 1], d1[j]
        dx = abs(targets[j + 1] - targets[j - 1]) / 2.0
        if min(abs(left), abs(right)) == 0.0:
            continue
        # a genuine interior extremum flips the sign of the slope; that is physical
        flipping = left * right < 0.0
        jump = max(abs(left), abs(right)) / min(abs(left), abs(right))
        implied = abs(right - left) / dx
        bound = curvature_bound(x, f, targets[j])
        out["rows"].append({
            "x": targets[j],
            "reference": float(ref[j]),
            "divided_difference_before": float(left),
            "divided_difference_after": float(right),
            "slope_jump_ratio": float(jump),
            "slope_sign_flip": bool(flipping),
            "implied_curvature": float(implied),
            "max_field_curvature_nearby": float(bound),
            "curvature_excess_factor": float(implied / bound) if bound else float("inf"),
        })
    return out


def classify(report: dict) -> dict:
    """Classify every published point: inside the band, under-resolved, or a defect.

    A point is *consistent* when its finest-grid residual is inside the acceptance
    band (0.03 of the lid speed) - the tables themselves are digitised off a plot,
    so sub-band residuals are not failures.  A point outside the band is only a
    reference defect when its residual *saturates*: ``r = |delta(128)| /
    |delta(64)|`` stays near one, i.e. the numerical profile keeps converging to a
    limit that is not the tabulated value.  If instead ``r > 2`` the residual is
    still shrinking and the honest verdict is "under-resolved", not "table wrong".
    """
    findings = []
    counts: dict = {}
    for re_key, block in report["reynolds"].items():
        re = float(re_key)
        finest = block["grids"][-1]
        for which, wblock in block["which"].items():
            for row in wblock["points"]:
                delta = abs(row["delta"][str(finest)])
                ratio = row["ratio_prev_over_finest"]
                inside = delta <= ACCEPTANCE_BAND
                if inside:
                    verdict = "consistent"
                elif ratio is None:
                    verdict = "unverified_single_grid"
                elif ratio > 2.0:
                    verdict = "under_resolved"
                else:
                    verdict = "reference_defect"
                row["verdict"] = verdict
                row["inside_band"] = inside
                row["convergence"] = (
                    "no_refinement_evidence" if ratio is None
                    else ("saturated" if ratio <= 1.5 else "shrinking"))
                counts[verdict] = counts.get(verdict, 0) + 1
                if verdict != "consistent":
                    shape = next((r for r in block["shape"]["rows"]
                                  if r["x"] == row["x"]), None)
                    findings.append({
                        "reynolds": re,
                        "which": which,
                        "x": row["x"],
                        "reference": row["reference"],
                        "computed_finest": row["computed"][str(finest)],
                        "delta_finest": row["delta"][str(finest)],
                        "ratio_prev_over_finest": ratio,
                        "resolution_uncertainty": row["resolution_uncertainty"],
                        "excess_factor": row["residual_over_uncertainty"],
                        "implied_curvature": None if shape is None else
                        shape["implied_curvature"],
                        "max_field_curvature": None if shape is None else
                        shape["max_field_curvature_nearby"],
                        "curvature_excess_factor": None if shape is None else
                        shape["curvature_excess_factor"],
                        "slope_jump_ratio": None if shape is None else
                        shape["slope_jump_ratio"],
                        "slope_sign_flip": None if shape is None else
                        shape["slope_sign_flip"],
                        "verdict": verdict,
                    })
    return {
        "shape_test_used_for_verdict": False,
        "shape_test_note": (
            "reported for completeness only: the genuine published near-extremum at "
            "x=0.2344 produces a larger slope jump than the flagged entry, so the "
            "control cannot separate a digitisation error from a thin feature"
        ),
        "counts": counts,
        "n_findings": len(findings),
        "findings": sorted(findings, key=lambda f: -abs(f["delta_finest"])),
        "reference_defects": [f for f in findings if f["verdict"] == "reference_defect"],
    }


