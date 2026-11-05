"""Consistency of the *published* Ghia tables with this solver's convergence sequence.

Why this module exists
----------------------
``docs/verification-lid-cavity-ladder/summary.json`` scores three Reynolds numbers
on a 32/64/128 ladder against the tables of Ghia, Ghia & Shin (1982).  One point of
the Re = 400 ``v`` centreline stays outside the acceptance band while everything
around it converges, and ``docs/GHIA_RE400_V_ATTRIBUTION.md`` attributes it to the
tabulated entry rather than to the solver.  That attribution was established by a
one-off script; nothing in the package *re-derives* it, and nothing checks that the
register of disputed entries (:data:`tensorfvm.lid_cavity_reference.DISPUTES`) is
exactly the set of points the committed evidence cannot explain.

This module turns both into a repeatable audit:

``pointwise``
    for every published ordinate of every table, the residual ``computed - Ghia``
    on each available grid, the residual ratio between successive grids, the
    observed order in ``h``, the Richardson limit and the remaining resolution
    uncertainty.  A point is classified as

    * ``consistent`` -- residual inside the acceptance band (0.03 U).  The tables
      are digitised off a plot, so sub-band residuals are not failures;
    * ``under_resolved`` -- outside the band but the residual is still *shrinking*
      (ratio > 2 between successive grids): more refinement is the answer;
    * ``reference_defect`` -- outside the band and *saturated* (ratio <= 1.5): the
      discrete solution converges to a limit that is not the tabulated value;
    * ``unverified_single_grid`` -- outside the band on the only grid available.

``register``
    the classification against :data:`DISPUTES`; agreement in both directions is
    required, so neither a new unexplained point nor a stale register entry can
    survive the audit.

``acceptance``
    the committed gate verdict re-derived from each case record (strict and with
    ``allow_registered_disputes=True``), including which points were excluded, so
    the documented exclusion cannot drift away from the evidence.

``negative_control``
    a synthetic corruption injected at an *unregistered* ordinate; the audit has
    to see it, otherwise "only one defect" would be a statement about the audit's
    blindness rather than about the tables.

Scope, stated honestly: the audit reads the committed ladder bundle.  It does not
re-run the solver and it cannot adjudicate a disputed entry (that needs the printed
Table II or an independent DNS).  Where the bundle has fewer than two grids for a
case -- Re = 1000 is committed on n = 64 only -- the classification is limited to
the band test and says so (``unverified_single_grid`` / no order estimate).
"""
from __future__ import annotations

import copy
import datetime as _dt
import hashlib
import json
import math
import os

from tensorfvm.lid_cavity import ghia_acceptance
from tensorfvm.lid_cavity_reference import (
    DISPUTES,
    REFERENCE,
    disputes_by_case,
    self_check,
    smoothness_audit,
    table,
)

__all__ = [
    "ACCEPTANCE_BAND",
    "DEFAULT_BUNDLE",
    "SHRINKING_RATIO",
    "available_grids",
    "case_record",
    "load_bundle",
    "pointwise",
    "profile_error_reproduction",
    "register_crosscheck",
    "acceptance_audit",
    "negative_control",
    "inject_defect",
    "audit",
    "summary_lines",
]

#: Pointwise acceptance band of :func:`tensorfvm.lid_cavity.ghia_acceptance`,
#: in units of the lid speed.
ACCEPTANCE_BAND = 0.03

#: Residual ratio (coarse grid over next finer grid) above which the residual is
#: still shrinking, i.e. the discrepancy is a resolution artefact.
SHRINKING_RATIO = 2.0

#: Residual ratio below which the residual is treated as saturated (grid
#: independent) for the purpose of calling a tabulated entry a defect.
SATURATED_RATIO = 1.5

DEFAULT_BUNDLE = os.path.join("docs", "verification-lid-cavity-ladder", "summary.json")


# --------------------------------------------------------------------------- #
# bundle access
# --------------------------------------------------------------------------- #
def load_bundle(path: str = DEFAULT_BUNDLE) -> dict:
    """Load the committed ladder bundle, or exit with a readable reason."""
    if not os.path.exists(path):
        raise SystemExit(f"ladder bundle not found: {path}")
    with open(path) as handle:
        bundle = json.load(handle)
    if not bundle.get("cases"):
        raise SystemExit(f"ladder bundle has no cases: {path}")
    return bundle


def _digest(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.md5(handle.read()).hexdigest()


def case_record(bundle: dict, reynolds: float, n: int) -> dict:
    """Return the single case record for ``(reynolds, n)``."""
    matches = [c for c in bundle["cases"]
               if float(c["reynolds"]) == float(reynolds) and int(c["n"]) == int(n)]
    if not matches:
        raise KeyError(f"no case for Re={reynolds} n={n}")
    return matches[0]


def available_grids(bundle: dict) -> dict[float, list[int]]:
    """``{Re: sorted grid sizes present in the bundle}``."""
    grids: dict[float, list[int]] = {}
    for record in bundle["cases"]:
        grids.setdefault(float(record["reynolds"]), []).append(int(record["n"]))
    return {re: sorted(ns) for re, ns in grids.items()}


def _ordinal(value: float) -> float:
    return round(float(value), 4)


# --------------------------------------------------------------------------- #
# pointwise convergence of the residuals
# --------------------------------------------------------------------------- #
def _order(coarse: float, mid: float, fine: float) -> float | None:
    """Observed order in ``h`` from the increments of a 2:1 refinement sequence."""
    step_coarse = abs(mid - coarse)
    step_fine = abs(fine - mid)
    if step_fine <= 0.0 or step_coarse <= 0.0:
        return None
    return math.log2(step_coarse / step_fine)


def _richardson(mid: float, fine: float, order: float | None) -> float | None:
    """Richardson limit of the sequence; ``None`` when the order is not clean."""
    if order is None or order <= 0.5:
        return None
    return fine + (fine - mid) / (2.0 ** order - 1.0)


def _classify(residual: float, ratio: float | None) -> tuple[str, str]:
    """``(verdict, convergence)`` for one published ordinate."""
    inside = residual <= ACCEPTANCE_BAND
    if ratio is None:
        convergence = "no_refinement_evidence"
        verdict = "consistent" if inside else "unverified_single_grid"
        return verdict, convergence
    convergence = "saturated" if ratio <= SATURATED_RATIO else "shrinking"
    if inside:
        return "consistent", convergence
    return ("reference_defect" if ratio <= SATURATED_RATIO else "under_resolved"), convergence


def pointwise(bundle: dict) -> dict:
    """Per-ordinate residual history and classification for every published table."""
    grids = available_grids(bundle)
    out: dict = {"reynolds": {}, "band": ACCEPTANCE_BAND}
    for reynolds in sorted(grids):
        ns = grids[reynolds]
        block: dict = {"grids": ns, "which": {}}
        for which in ("u", "v"):
            points = table(reynolds, which)
            rows = []
            for coord, published in sorted(points):
                x = _ordinal(coord)
                computed = {n: case_record(bundle, reynolds, n)["u" if which == "u" else "v"]
                            ["profiles"]["computed"][
                                [ _ordinal(c) for c in
                                  case_record(bundle, reynolds, n)[which]["profiles"]["coord"]
                                ].index(x)]
                            for n in ns}
                residual = {n: abs(computed[n] - published) for n in ns}
                finest = ns[-1]
                ratio = None
                if len(ns) >= 2:
                    previous = residual[ns[-2]]
                    ratio = previous / residual[finest] if residual[finest] > 0.0 else float("inf")
                order = None
                limit = None
                uncertainty = None
                if len(ns) >= 3:
                    order = _order(computed[ns[0]], computed[ns[1]], computed[ns[2]])
                    limit = _richardson(computed[ns[1]], computed[ns[2]], order)
                    if order is not None and order > 0.5:
                        uncertainty = abs(computed[ns[2]] - computed[ns[1]]) / (2.0 ** order - 1.0)
                elif len(ns) == 2:
                    uncertainty = abs(computed[ns[1]] - computed[ns[0]])
                verdict, convergence = _classify(residual[finest], ratio)
                rows.append({
                    "x": x,
                    "published": float(published),
                    "computed": {str(n): computed[n] for n in ns},
                    "residual": {str(n): residual[n] for n in ns},
                    "delta_signed_finest": computed[finest] - float(published),
                    "ratio_previous_over_finest": ratio,
                    "observed_order": order,
                    "richardson_limit": limit,
                    "limit_residual": None if limit is None else abs(limit - float(published)),
                    "resolution_uncertainty": uncertainty,
                    "residual_over_uncertainty":
                        None if not uncertainty else residual[finest] / uncertainty,
                    "inside_band": residual[finest] <= ACCEPTANCE_BAND,
                    "convergence": convergence,
                    "verdict": verdict,
                })
            block["which"][which] = {
                "reference_points": len(points),
                "rows": rows,
                "max_residual": {str(n): max(r["residual"][str(n)] for r in rows) for n in ns},
                "max_residual_consistent": max(
                    (r["residual"][str(ns[-1])] for r in rows if r["verdict"] == "consistent"),
                    default=0.0),
            }
        out["reynolds"][str(reynolds)] = block
    return out


def _flatten(points: dict) -> list[dict]:
    rows = []
    for re_key, block in points["reynolds"].items():
        for which, wblock in block["which"].items():
            for row in wblock["rows"]:
                rows.append(dict(row, reynolds=float(re_key), which=which))
    return rows


def _largest_consistent(rows: list[dict]) -> dict | None:
    """The consistent ordinate with the largest residual (context for the band)."""
    candidates = [r for r in rows if r["verdict"] == "consistent"]
    if not candidates:
        return None
    best = max(candidates, key=lambda r: r["residual"][max(r["residual"], key=int)])
    return {"reynolds": best["reynolds"], "which": best["which"], "x": best["x"],
            "residual": best["residual"][max(best["residual"], key=int)],
            "grids": sorted(int(n) for n in best["residual"])}


def _verdict_block(points: dict) -> dict:
    rows = _flatten(points)
    counts: dict = {}
    for row in rows:
        counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1
    defects = [row for row in rows if row["verdict"] == "reference_defect"]
    under = [row for row in rows if row["verdict"] == "under_resolved"]
    single = [row for row in rows if row["verdict"] == "unverified_single_grid"]
    return {
        "published_points": len(rows),
        "counts": counts,
        "reference_defects": sorted(
            ({"reynolds": r["reynolds"], "which": r["which"], "x": r["x"],
              "published": r["published"],
              "computed_finest": r["computed"][max(r["computed"], key=int)],
              "residual_finest": r["residual"][max(r["residual"], key=int)],
              "ratio_previous_over_finest": r["ratio_previous_over_finest"],
              "observed_order": r["observed_order"],
              "richardson_limit": r["richardson_limit"],
              "limit_residual": r["limit_residual"],
              "residual_over_uncertainty": r["residual_over_uncertainty"]}
             for r in defects), key=lambda d: (d["reynolds"], d["which"], d["x"])),
        "under_resolved": sorted(
            ({"reynolds": r["reynolds"], "which": r["which"], "x": r["x"],
              "residual_finest": r["residual"][max(r["residual"], key=int)],
              "ratio_previous_over_finest": r["ratio_previous_over_finest"]}
             for r in under), key=lambda d: (d["reynolds"], d["which"], d["x"])),
        "single_grid_outside_band": [
            {"reynolds": r["reynolds"], "which": r["which"], "x": r["x"],
             "residual_finest": r["residual"][max(r["residual"], key=int)]}
            for r in single],
        "largest_consistent_residual": max(
            (r["residual"][max(r["residual"], key=int)]
             for r in rows if r["verdict"] == "consistent"), default=0.0),
        "largest_consistent": _largest_consistent(rows),
    }


# --------------------------------------------------------------------------- #
# cross-checks
# --------------------------------------------------------------------------- #
def profile_error_reproduction(bundle: dict) -> list[dict]:
    """Do the committed ``max_abs_error`` numbers follow from the stored profiles?"""
    checks = []
    for record in sorted(bundle["cases"], key=lambda c: (float(c["reynolds"]), int(c["n"]))):
        for which in ("u", "v"):
            block = record[which]
            deltas = [abs(float(d)) for d in block["profiles"]["delta"]]
            recomputed = max(deltas)
            checks.append({
                "reynolds": float(record["reynolds"]),
                "n": int(record["n"]),
                "which": which,
                "committed_max_abs_error": float(block["max_abs_error"]),
                "recomputed_max_abs_error": recomputed,
                "agrees": abs(recomputed - float(block["max_abs_error"])) <= 1e-9,
            })
    return checks


def register_crosscheck(verdict: dict) -> dict:
    """Compare the detected defects with :data:`DISPUTES`, in both directions."""
    declared = {(_ordinal(k[0]), k[1], _ordinal(k[2])) for k in DISPUTES}
    detected = {(_ordinal(d["reynolds"]), d["which"], _ordinal(d["x"]))
                for d in verdict["reference_defects"]}
    grouped = disputes_by_case()
    return {
        "declared": [{"reynolds": re, "which": which, "x": x,
                      "published": grouped[(re, which)][x]["published"]}
                     for re, which, x in sorted(declared)],
        "detected": [{"reynolds": re, "which": which, "x": x}
                     for re, which, x in sorted(detected)],
        "missing_from_register": [
            {"reynolds": re, "which": which, "x": x}
            for re, which, x in sorted(detected - declared)],
        "missing_from_detection": [
            {"reynolds": re, "which": which, "x": x}
            for re, which, x in sorted(declared - detected)],
        "agreement": declared == detected,
        "records": {f"{re:g}/{which}/{x:.4f}": copy.deepcopy(grouped[(re, which)][x])
                    for re, which, x in sorted(declared)},
    }


def acceptance_audit(bundle: dict) -> list[dict]:
    """Re-derive every committed gate verdict, strictly and with the register."""
    out = []
    for record in sorted(bundle["cases"], key=lambda c: (float(c["reynolds"]), int(c["n"]))):
        result = {"reynolds": float(record["reynolds"]),
                  "u": record["u"], "v": record["v"],
                  "vortex_centre": record["vortex_centre"]}
        strict = ghia_acceptance(result)
        lenient = ghia_acceptance(result, allow_registered_disputes=True)
        committed = record.get("acceptance") or {}
        out.append({
            "reynolds": float(record["reynolds"]),
            "n": int(record["n"]),
            "strict_pass": bool(strict["pass"]),
            "lenient_pass": bool(lenient["pass"]),
            "failing_criteria": sorted(name for name, item in strict["criteria"].items()
                                       if not item["pass"]),
            "excluded_points": lenient["registered_disputes"],
            "max_abs_error": strict["max_abs_error"],
            "committed_pass": None if not committed else bool(committed.get("pass")),
            "reproduces_committed":
                None if not committed else bool(committed.get("pass")) == bool(strict["pass"]),
        })
    return out


def key_quantities(bundle: dict) -> list[dict]:
    """Per-case engineering quantities and their relative errors against Ghia.

    These are the numbers a reader of the benchmark actually quotes: the deepest
    reverse-flow ``u``, the strongest ``v``, the primary-vortex centre and the
    pointwise envelope.  Relative errors are reported against the published value
    with the platform band (15 %) so the acceptance statement does not depend on
    the units of the profile residuals.
    """
    from tensorfvm.lid_cavity_reference import VORTEX_CENTRE

    out = []
    for record in sorted(bundle["cases"], key=lambda c: (float(c["reynolds"]), int(c["n"]))):
        re = float(record["reynolds"])
        ext_u = record["u"]["extremum"]
        ext_v = record["v"]["extremum"]
        centre = record["vortex_centre"]["computed"]
        ref_centre = VORTEX_CENTRE.get(re)
        u_ref, u_got = ext_u["min_reference"][1], ext_u["min_computed"][1]
        v_ref, v_got = ext_v["max_reference"][1], ext_v["max_computed"][1]
        out.append({
            "reynolds": re,
            "n": int(record["n"]),
            "u_min": {"reference": u_ref, "computed": u_got,
                      "relative_error": abs(u_got - u_ref) / abs(u_ref),
                      "pass_15pct": abs(u_got - u_ref) / abs(u_ref) <= 0.15},
            "v_max": {"reference": v_ref, "computed": v_got,
                      "relative_error": abs(v_got - v_ref) / abs(v_ref),
                      "pass_15pct": abs(v_got - v_ref) / abs(v_ref) <= 0.15},
            "vortex_centre": {
                "reference": None if ref_centre is None else list(ref_centre),
                "computed": [float(centre[0]), float(centre[1])],
                "distance": None if ref_centre is None else math.hypot(
                    centre[0] - ref_centre[0], centre[1] - ref_centre[1]),
            },
            "max_abs_error_u": float(record["u"]["max_abs_error"]),
            "max_abs_error_v": float(record["v"]["max_abs_error"]),
            "max_divergence": float(record.get("max_divergence", float("nan"))),
            "converged_residual": float(record.get("residual", float("nan"))),
            "steps": int(record.get("steps", 0)),
            "wall_seconds": float((record.get("solver") or {}).get("wall_seconds", float("nan"))),
        })
    return out


# --------------------------------------------------------------------------- #
# negative control
# --------------------------------------------------------------------------- #
def inject_defect(bundle: dict, reynolds: float, which: str, x: float,
                  magnitude: float) -> dict:
    """Copy of ``bundle`` with ``magnitude`` added to every grid at one ordinate.

    A constant offset across the ladder leaves the residual *saturated*, which is
    the signature the audit claims to detect; a resolution artefact would instead
    shrink with ``h`` (that control is built by :func:`negative_control` too).
    """
    poisoned = copy.deepcopy(bundle)
    target = _ordinal(x)
    touched = 0
    for record in poisoned["cases"]:
        if float(record["reynolds"]) != float(reynolds):
            continue
        profile = record[which]["profiles"]
        coords = [_ordinal(c) for c in profile["coord"]]
        if target not in coords:
            raise KeyError(f"{x} is not a published ordinate of Re={reynolds} {which}")
        index = coords.index(target)
        profile["computed"][index] += magnitude
        profile["delta"][index] = profile["computed"][index] - profile["reference"][index]
        touched += 1
    if not touched:
        raise KeyError(f"no case for Re={reynolds}")
    return poisoned


def negative_control(bundle: dict, *, reynolds: float = 100.0, which: str = "u",
                     x: float = 0.5000, magnitude: float = 0.05) -> dict:
    """Inject a documented corruption and report whether the audit sees it."""
    clean = _verdict_block(pointwise(bundle))
    poisoned_bundle = inject_defect(bundle, reynolds, which, x, magnitude)
    poisoned = _verdict_block(pointwise(poisoned_bundle))
    clean_keys = {(_ordinal(d["reynolds"]), d["which"], _ordinal(d["x"]))
                  for d in clean["reference_defects"]}
    poisoned_keys = {(_ordinal(d["reynolds"]), d["which"], _ordinal(d["x"]))
                     for d in poisoned["reference_defects"]}
    injected = (_ordinal(reynolds), which, _ordinal(x))
    return {
        "injected": {"reynolds": reynolds, "which": which, "x": _ordinal(x),
                     "magnitude": magnitude},
        "detected": injected in poisoned_keys,
        "new_defects": [{"reynolds": re, "which": w, "x": xx}
                        for re, w, xx in sorted(poisoned_keys - clean_keys)],
        "no_other_new_defects": (poisoned_keys - clean_keys) == {injected},
        "control_passes": injected not in clean_keys,
    }


# --------------------------------------------------------------------------- #
# the audit
# --------------------------------------------------------------------------- #
def audit(path: str = DEFAULT_BUNDLE, *, negative_control_case: bool = True) -> dict:
    """Run the whole consistency audit over the committed ladder bundle."""
    bundle = load_bundle(path)
    grids = available_grids(bundle)
    points = pointwise(bundle)
    verdict = _verdict_block(points)
    meta = bundle.get("meta", {})
    expected = {float(re): sorted(int(n) for n in meta.get("grids", []))
                for re in meta.get("reynolds", [])} or {}
    missing = [{"reynolds": re, "n": n}
               for re, ns in expected.items() for n in ns if n not in grids.get(re, [])]
    table_check = self_check()
    report = {
        "meta": {
            "code": "tensorfvm.verification.ghia_reference_audit.audit",
            "bundle": path,
            "bundle_md5": _digest(path),
            "generated_utc": _dt.datetime.now(_dt.timezone.utc)
                                     .replace(microsecond=0).isoformat(),
            "bundle_timestamp": meta.get("timestamp"),
            "bundle_provenance": meta.get("provenance"),
            "reference": REFERENCE,
            "reynolds": sorted(grids),
            "grids_present": {f"{re:g}": ns for re, ns in sorted(grids.items())},
            "grids_missing": missing,
            "bundle_is_partial": bool(missing),
            "acceptance_band": ACCEPTANCE_BAND,
            "shrinking_ratio": SHRINKING_RATIO,
            "saturated_ratio": SATURATED_RATIO,
        },
        "reference_table_self_check": table_check,
        "profile_error_reproduction": profile_error_reproduction(bundle),
        "key_quantities": key_quantities(bundle),
        "shape_control": dict(
            smoothness_audit(),
            used_for_verdict=False,
            why_not=("the genuine published near-extremum at Re = 400, x = 0.2344 "
                     "produces a larger secant-slope jump than the flagged entry, so "
                     "the control cannot separate a digitisation error from a thin "
                     "feature; it is reported as context only and every verdict above "
                     "comes from the residual under refinement")),
        "pointwise": points,
        "verdict": verdict,
        "register": register_crosscheck(verdict),
        "acceptance": acceptance_audit(bundle),
    }
    report["profile_errors_reproduced"] = all(
        item["agrees"] for item in report["profile_error_reproduction"])
    report["gate_verdicts_reproduced"] = all(
        item["reproduces_committed"] for item in report["acceptance"]
        if item["reproduces_committed"] is not None)
    report["negative_control"] = (negative_control(bundle)
                                  if negative_control_case else None)
    report["ok"] = bool(
        table_check["ok"]
        and report["profile_errors_reproduced"]
        and report["gate_verdicts_reproduced"]
        and report["register"]["agreement"]
        and not report["shape_control"]["unexplained"]
        and (report["negative_control"] is None
             or (report["negative_control"]["detected"]
                 and report["negative_control"]["no_other_new_defects"]))
        and not verdict["under_resolved"]
    )
    return report


def summary_lines(report: dict) -> list[str]:
    """Short human-readable digest, used by the CLI and by the report generator."""
    verdict = report["verdict"]
    lines = [
        f"bundle            : {report['meta']['bundle']} "
        f"(md5 {report['meta']['bundle_md5']})",
        f"grids present     : {report['meta']['grids_present']}",
        f"grids missing     : {report['meta']['grids_missing'] or 'none'}",
        f"published points  : {verdict['published_points']}",
        f"verdict counts    : {verdict['counts']}",
        f"register agreement: {report['register']['agreement']}",
        f"gate reproduction : {report['gate_verdicts_reproduced']}",
        f"negative control  : "
        f"{None if report['negative_control'] is None else report['negative_control']['detected']}",
        f"audit ok          : {report['ok']}",
    ]
    return lines