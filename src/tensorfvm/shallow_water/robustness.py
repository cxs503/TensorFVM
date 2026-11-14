"""Robustness envelope of the ``CASE-FVM-DAMBREAK-STOKER`` evidence.

The anchor is graded on one declared geometry: dam at ``x = 0`` in ``[-4, 6]``,
outflow boundaries 2 m / 4.1 m behind the two waves, depth ratio
``h1/h0 = 0.5``, ``t* = 2``.  Every published error (``0.017 %`` dry front,
``0.021 %`` wet bore at ``n = 1600``) therefore holds **on that geometry**.
This module asks the adversarial question the graded run never asks: *how much
of the answer is an accident of that geometry?*

Four perturbation axes are swept.  Each one perturbs the *declaration*, never
the metric: the graded estimators of
:func:`tensorfvm.shallow_water.dam_break.run_case` are used unchanged.

``translation``
    The whole geometry -- ``x_min``, ``x_max`` and ``x_dam`` -- is translated by
    ``dx/4`` and ``dx/2``, so the initial discontinuity keeps its distance to
    every cell centre but no longer sits on a "nice" coordinate.  This is the
    verification-practice failure mode where a published number is an accident
    of the discontinuity landing on a cell face.  In this solver the initial
    data is *sampled at cell centres* (:func:`run_case`), so the discrete
    problem is translation invariant up to the boundary placement; the axis
    measures that claim instead of assuming it.

``boundary``
    The downstream outflow is moved to 8 cells behind the exact wave position
    (from 2 m / 4.1 m in the baseline).  A transmissive boundary that is not
    transparent shows up here as a change of the headline error.

``domain``
    The domain length is doubled and ``n`` doubled with it, i.e. the **same**
    ``dx`` and the same resolution, with the extra length on both sides.  This
    isolates boundary contamination from grid resolution.

``similitude``
    All lengths and ``h0, h1`` are doubled at fixed ``n`` and fixed ``t*``.
    The shallow-water problem is invariant under ``x -> 2x``, ``h -> 2h``,
    ``t -> sqrt(2) t``, so a dimensionally consistent discretisation has to
    reproduce the *non-dimensional* answers; any drift is a scale-dependent
    constant in the code (an absolute depth floor, an absolute tolerance inside
    a limiter, a hard-coded length).  The comparison is made on the
    non-dimensional position ratio ``x_measured / x_exact`` and on the
    already-relative ``L1`` depth error; the error-relative drift is reported as
    a diagnostic only, because ``error = |x - x_exact|`` is a small difference
    of large numbers and amplifies round-off by ``~1/error``.

``control``
    ``tight_dx_rounding`` repeats the short-domain variant the naive way -- the
    cell count is rounded instead of the domain length, so ``dx`` is off by a
    fraction of a percent.  It is **not gated**: it is the control that shows
    how much of a "boundary effect" is really a resolution effect, because the
    same perturbation done with ``dx`` held exactly gives an identical answer.

``depth_ratio``
    The wet-bed case is graded at ``h1/h0 = 0.5`` in the baseline (0.25 in
    ``tests/test_shallow_water_dam_break.py``).  Ratios 0.2 and 0.8 are added:
    they roughly halve / double the bore strength, which is where an HLL
    wave-speed estimate or the ``(h1 + h2)/2`` level crossing is most likely to
    break, and where the "no spurious oscillation" clause of the acceptance has
    the least margin.

Thresholds are declared *before* any run (``*_MAX_PP``, ``SIMILARITY_REL_TOL``
below) and applied to the graded headline error of
:func:`tensorfvm.shallow_water.dam_break.run_case` -- the same number the anchor
report publishes.

References
----------
Ritter (1892) *Z. Ver. Deutsch. Ing.* **36**, 947; Stoker (1957) *Water Waves*
ch. 10.  Reference values come from :mod:`tensorfvm.shallow_water.exact`.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

from .dam_break import (
    RITTER_DRY,
    STOKER_WET,
    DamBreakCase,
    run_case,
)
from .exact import stoker_constant_state

__all__ = [
    "ALIGNMENT_MAX_PP",
    "BOUNDARY_MAX_PP",
    "DOMAIN_MAX_PP",
    "SIMILARITY_REL_TOL",
    "DEPTH_RATIO_MAX_ERROR",
    "EXTREMUM_AMPLITUDE_FLOOR",
    "VARIANTS",
    "headline_error",
    "variant_cases",
    "run_sweep",
    "sweep_verdict",
]

#: allowed change of the graded headline error (percentage points) under a
#: translation of the geometry.  0.5 pp = 10 % of the anchor tolerance: an
#: alignment accident that eats more than a tenth of the acceptance budget
#: would make the published number geometry-specific.
ALIGNMENT_MAX_PP = 0.5

#: allowed change (pp) when the downstream outflow is 8 cells behind the wave.
BOUNDARY_MAX_PP = 0.5

#: allowed change (pp) at identical ``dx`` on a domain twice as long.
DOMAIN_MAX_PP = 0.1

#: allowed relative drift of the non-dimensional answers under the similitude
#: transformation.  A dimensionally consistent scheme reproduces
#: ``x_measured / x_exact`` and the relative ``L1`` error to round-off; the
#: loose-looking value accommodates the step-count round-off of the CFL
#: controller (``dt`` is recomputed every step).
SIMILARITY_REL_TOL = 1e-9

#: anchor acceptance, applied unchanged to the added depth ratios.
DEPTH_RATIO_MAX_ERROR = 0.05

#: the amplitude floor of the repo-wide "spurious oscillation" count
#: (``dam_break`` counts interior extrema whose two slopes exceed this), in
#: units of ``h0``.
EXTREMUM_AMPLITUDE_FLOOR = 1e-4

#: name of the graded headline error metric of each case kind.
_HEADLINE = {"dry": "front_fan_error", "wet": "shock_error"}

VARIANTS = (
    "baseline",
    "translated_quarter",
    "translated_half",
    "tight_downstream",
    "tight_dx_rounding",
    "wide_domain",
    "similarity_x2",
    "depth_ratio_0.2",
    "depth_ratio_0.8",
)


def headline_error(kind: str, metrics: dict) -> float:
    """Graded headline error of one run (dry: fan front, wet: bore position)."""
    return float(metrics[_HEADLINE[kind]])


def _exact_wave_speed(case: DamBreakCase) -> float:
    """Speed of the leading wave of ``case`` (front speed or bore speed)."""
    if case.kind == "dry":
        return 2.0 * math.sqrt(case.gravity * case.h0)
    return stoker_constant_state(case.h0, case.h1, case.gravity)["shock_speed"]


def _translate(case: DamBreakCase, n: int, fraction: float) -> DamBreakCase:
    """Translate dam **and** domain by ``fraction * dx`` (grid stays uniform)."""
    dx = (case.x_max - case.x_min) / float(n)
    shift = fraction * dx
    return dataclasses.replace(case,
                               x_dam=case.x_dam + shift,
                               x_min=case.x_min + shift,
                               x_max=case.x_max + shift,
                               key=f"{case.key}-translated{fraction:g}dx")


def _extremum_amplitude(h, h0: float,
                        floor: float = EXTREMUM_AMPLITUDE_FLOOR) -> float:
    """Largest interior extremum prominence, in units of ``h0``.

    Same detector as :func:`tensorfvm.shallow_water.dam_break._local_extrema`
    (both neighbour slopes must exceed ``floor``), but it returns the amplitude
    instead of the count, so a run that trips the monotonicity clause can be
    reported by *how much* it trips it instead of only that it did.
    """
    h = np.asarray(h, dtype=float)
    if h.size < 3:
        return 0.0
    left = h[1:-1] - h[:-2]
    right = h[2:] - h[1:-1]
    peak = (left > 0) & (right < 0) & (np.abs(left) > floor * h0) & (np.abs(right) > floor * h0)
    dip = (left < 0) & (right > 0) & (np.abs(left) > floor * h0) & (np.abs(right) > floor * h0)
    sel = peak | dip
    if not np.any(sel):
        return 0.0
    prominence = np.minimum(np.abs(left), np.abs(right))[sel]
    return float(np.max(prominence) / h0)


def variant_cases(n: int) -> list[dict]:
    """All robustness variants at the baseline grid resolution ``n``.

    Each entry is ``{"variant", "case", "n", "ref_n"}``.  Variants that change
    the domain length get their own cell count so that **``dx`` is preserved**,
    so a boundary-placed run differs from the baseline only in *where* the
    boundary is, not in how finely the wave is resolved.  ``ref_n`` is then the
    baseline resolution with the same ``dx``, i.e. the run the variant must be
    compared against; it is ``None`` when the variant is compared in
    non-dimensional terms instead (``similarity_x2``, ``depth_ratio_*``).
    """
    out: list[dict] = []

    for base in (RITTER_DRY, STOKER_WET):
        dx = (base.x_max - base.x_min) / float(n)
        wave = base.x_dam + _exact_wave_speed(base) * base.t_end

        out.append({"variant": "baseline", "case": base, "n": n,
                    "ref_n": None})
        for fraction, name in ((0.25, "translated_quarter"),
                               (0.5, "translated_half")):
            out.append({"variant": name,
                        "case": _translate(base, n, fraction),
                        "n": n, "ref_n": n})
        # place the outflow at least 8 cells behind the wave *and* keep dx
        # exactly: the domain length is quantised to whole cells instead of
        # rounding ``n`` (a 0.2 % dx mismatch is enough to move a first-order
        # error by tenths of a percentage point and would be misread as
        # boundary contamination).
        n_tight = int(math.ceil((wave + 8.0 * dx - base.x_min) / dx))
        tight = dataclasses.replace(
            base, x_max=base.x_min + n_tight * dx,
            key=f"{base.key}-tight-downstream")
        out.append({"variant": "tight_downstream", "case": tight,
                    "n": n_tight, "ref_n": n})
        rounded = dataclasses.replace(base, x_max=wave + 8.0 * dx,
                                      key=f"{base.key}-tight-dx-rounding")
        n_round = int(round((rounded.x_max - rounded.x_min) / dx))
        out.append({"variant": "tight_dx_rounding", "case": rounded,
                    "n": n_round, "ref_n": n})
        # same quantisation trick for the widened domain (n doubled exactly)
        n_wide = 2 * n
        wide = dataclasses.replace(base, x_min=base.x_min - 0.5 * n * dx,
                                   x_max=base.x_max + 0.5 * n * dx,
                                   key=f"{base.key}-wide-domain")
        out.append({"variant": "wide_domain", "case": wide,
                    "n": n_wide, "ref_n": n})
        scaled = dataclasses.replace(base, h0=2.0 * base.h0, h1=2.0 * base.h1,
                                     x_dam=2.0 * base.x_dam,
                                     x_min=2.0 * base.x_min,
                                     x_max=2.0 * base.x_max,
                                     key=f"{base.key}-similarity-x2")
        out.append({"variant": "similarity_x2", "case": scaled, "n": n,
                    "ref_n": None})

    for ratio in (0.2, 0.8):
        out.append({
            "variant": f"depth_ratio_{ratio:g}",
            "case": dataclasses.replace(STOKER_WET, h1=ratio * STOKER_WET.h0,
                                        key=f"stoker-wet-ratio-{ratio:g}"),
            "n": n, "ref_n": None,
        })
    return out


def _graded(case: DamBreakCase, run: dict) -> dict:
    """The graded numbers of one run, in a JSON-safe form."""
    metrics = run["metrics"]
    kind = case.kind
    position = metrics["front_fan"] if kind == "dry" else metrics["shock_position"]
    exact = metrics["front_exact"] if kind == "dry" else metrics["shock_exact"]
    return {
        "headline": headline_error(kind, metrics),
        "headline_metric": _HEADLINE[kind],
        "headline_position": float(position),
        "exact_position": float(exact),
        "position_ratio": float(position) / float(exact),
        "depth_l1": float(metrics["depth_l1"]),
        "local_extrema": int(metrics["local_extrema"]),
        "extrema_amplitude_over_h0": _extremum_amplitude(run["profile"]["h_numeric"],
                                                        case.h0),
        "tv_ratio": float(metrics["tv_ratio"]),
        "mass_drift": float(metrics["mass_drift"]),
        "h_min": float(metrics["h_min"]),
        "series_max_error": float(metrics["front_series_max_error"]),
        "all_pass": bool(run["verdict"]["all_pass"]),
        "dx": (case.x_max - case.x_min) / float(run["discretisation"]["n"]),
        "wall_time_s": float(run["discretisation"]["wall_time_s"]),
    }


def run_sweep(ns=(400, 800, 1600)) -> dict:
    """Run every variant at every resolution and return a JSON-safe report."""
    runs = []
    for n in ns:
        for spec in variant_cases(n):
            case = spec["case"]
            run = run_case(case, n=spec["n"])
            runs.append({
                "variant": spec["variant"],
                "base": "ritter-dry" if case.kind == "dry" else "stoker-wet",
                "case": case.summary(),
                "n": spec["n"],
                "sweep_n": n,
                "ref_n": spec["ref_n"],
                "graded": _graded(case, run),
            })
    return {"ns": list(ns), "tolerance": DEPTH_RATIO_MAX_ERROR, "runs": runs}


def sweep_verdict(sweep: dict) -> dict:
    """Apply the predeclared thresholds to a :func:`run_sweep` result."""
    index = {(r["base"], r["variant"], r["sweep_n"]): r for r in sweep["runs"]}
    ns = sweep["ns"]
    checks: dict[str, dict] = {}
    limits = {"translated_quarter": ALIGNMENT_MAX_PP,
              "translated_half": ALIGNMENT_MAX_PP,
              "tight_downstream": BOUNDARY_MAX_PP,
              "wide_domain": DOMAIN_MAX_PP}

    for base in ("ritter-dry", "stoker-wet"):
        for variant, limit in limits.items():
            rows = []
            for n in ns:
                probe = index.get((base, variant, n))
                if probe is None:
                    continue
                ref = index.get((base, "baseline", probe["ref_n"]))
                if ref is None:
                    continue
                rows.append({
                    "n": probe["n"],
                    "reference_n": ref["n"],
                    "dx_rel": abs(probe["graded"]["dx"] / ref["graded"]["dx"] - 1.0),
                    "delta_pp": 100.0 * (probe["graded"]["headline"]
                                         - ref["graded"]["headline"]),
                    "variant_error": probe["graded"]["headline"],
                    "baseline_error": ref["graded"]["headline"],
                })
            if not rows:
                continue
            worst = max(rows, key=lambda d: abs(d["delta_pp"]))
            checks[f"{base}/{variant}"] = {
                "limit_pp": limit,
                "worst_delta_pp": worst["delta_pp"],
                "worst_n": worst["n"],
                "reference_n": worst["reference_n"],
                "max_dx_rel": max(r["dx_rel"] for r in rows),
                "pass": abs(worst["delta_pp"]) <= limit,
                "rows": rows,
            }

        # control: the same short-domain perturbation with dx rounded instead
        # of quantised.  Not gated -- it exists to separate a resolution effect
        # from a boundary effect, and its dx mismatch must be non-zero or the
        # control is inactive.
        rows = []
        for n in ns:
            probe = index.get((base, "tight_dx_rounding", n))
            ref = index.get((base, "baseline", probe["ref_n"])) if probe else None
            if probe is None or ref is None:
                continue
            rows.append({
                "n": probe["n"],
                "reference_n": ref["n"],
                "dx_rel": abs(probe["graded"]["dx"] / ref["graded"]["dx"] - 1.0),
                "delta_pp": 100.0 * (probe["graded"]["headline"]
                                     - ref["graded"]["headline"]),
            })
        if rows:
            checks[f"{base}/control_tight_dx_rounding"] = {
                "gated": False,
                "max_dx_rel": max(r["dx_rel"] for r in rows),
                "max_abs_delta_pp": max(abs(r["delta_pp"]) for r in rows),
                # informational: on a domain where 8 dx divides the length
                # exactly the control is inactive (nothing to diagnose)
                "active": max(r["dx_rel"] for r in rows) > 1e-4,
                "pass": True,
                "rows": rows,
            }

        # similitude: the non-dimensional answers must be reproduced
        rows = []
        for n in ns:
            probe = index.get((base, "similarity_x2", n))
            ref = index.get((base, "baseline", n))
            if probe is None or ref is None:
                continue
            row = {"n": n}
            for key in ("position_ratio", "depth_l1"):
                a, b = probe["graded"][key], ref["graded"][key]
                row[key] = {"baseline": b, "scaled": a,
                            "rel": abs(a - b) / max(abs(b), 1e-300)}
            # diagnostic only: error-relative drift (ill-conditioned, ~1/error)
            a, b = probe["graded"]["headline"], ref["graded"]["headline"]
            row["headline_rel_diagnostic"] = abs(a - b) / max(abs(b), 1e-300)
            row["rel_max"] = max(row["position_ratio"]["rel"],
                                 row["depth_l1"]["rel"])
            rows.append(row)
        if rows:
            worst = max(rows, key=lambda r: r["rel_max"])
            checks[f"{base}/similarity_x2"] = {
                "limit_rel": SIMILARITY_REL_TOL,
                "worst_rel": worst["rel_max"],
                "worst_n": worst["n"],
                "pass": worst["rel_max"] <= SIMILARITY_REL_TOL,
                "rows": rows,
            }

        # added depth ratios: anchor tolerance unchanged, plus the monotonicity
        # clause reported by how far the offending extremum rises above the floor
        if base == "stoker-wet":
            rows = []
            for ratio in ("depth_ratio_0.2", "depth_ratio_0.8"):
                for n in ns:
                    probe = index.get((base, ratio, n))
                    if probe is None:
                        continue
                    rows.append({
                        "ratio": ratio, "n": probe["n"],
                        "headline": probe["graded"]["headline"],
                        "local_extrema": probe["graded"]["local_extrema"],
                        "extrema_amplitude_over_h0":
                            probe["graded"]["extrema_amplitude_over_h0"],
                        "all_pass": probe["graded"]["all_pass"],
                    })
            if rows:
                worst = max(rows, key=lambda r: r["headline"])
                checks["stoker-wet/depth_ratio"] = {
                    "limit": DEPTH_RATIO_MAX_ERROR,
                    "worst_error": worst["headline"],
                    "worst_variant": worst["ratio"],
                    "worst_n": worst["n"],
                    "max_extrema_amplitude": max(
                        r["extrema_amplitude_over_h0"] for r in rows),
                    "pass": worst["headline"] <= DEPTH_RATIO_MAX_ERROR,
                    "rows": rows,
                }

    return {"checks": checks,
            "all_pass": all(c["pass"] for c in checks.values())}
