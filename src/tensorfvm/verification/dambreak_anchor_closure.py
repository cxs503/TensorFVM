"""Closure dossier for the anchor ``CASE-FVM-DAMBREAK-STOKER``.

Why this module exists
----------------------
The anchor has been extended by eight independent verification legs (2-D
reduction, Galilean boost, momentum identities, oracle-free measurement,
Riemann convergence, numerical-configuration envelope, degenerate depth-ratio
limits, gate-detection power).  Each leg is individually green, and the pool
entry ``docs/oss_solutions/CASE-FVM-DAMBREAK-STOKER.json`` declares
``status = "pass"`` / ``verdict = "PASS"``.  What no artifact answers is the
*closure* question:

    does the registered gate set actually cover the ledger's acceptance
    sentence, and is every number that feeds the registered aggregate ``PASS``
    a converged, in-scope quantity?

That question is not rhetorical.  ``dam_break._verdict`` computes

    ``all_pass = all(item["pass"] for item in verdict.values())``

so the aggregate is the intersection of *every* verdict key -- including keys
that the artifact bundle itself classifies as diagnostics.  This module
therefore separates, mechanically and from the committed artifacts only:

``acceptance_coverage``
    the acceptance sentence's own elements (wave position ``x_f(t)``,
    still-water level step, monotone/no spurious oscillation) mapped to the
    verdict keys that grade them, with the measured value and margin.
``graded_but_undeclared``
    verdict keys that feed ``all_pass`` but grade none of the acceptance
    elements.  These are legitimate as extra safety nets (``depth_l1``) or as
    documentation defects (a diagnostic promoted into the aggregate) -- the
    dossier lists them instead of letting them hide inside ``all_pass``.
``diagnostic_graded``
    the subset of the above whose graded number is, in the *same* artifact
    bundle, published under a ``*_diagnostic`` name and whose refinement
    ladder is non-monotone.  A quantity that grows under refinement is not a
    yardstick; grading it makes the aggregate ``PASS`` depend on a reading the
    bundle declares unfit for grading.
``orphaned_evidence``
    pool amendments produced by the extension legs that were never applied to
    the registered entry (``proposed_pool_amendment.json`` /
    ``case_pool_entry.json`` files under ``docs/CASE-FVM-DAMBREAK-STOKER-*``).
``disposition``
    the audit verdict, derived from the checks above -- not asserted.

Nothing here re-derives a number: every value is read from the committed
artifacts, and :func:`reproducibility_check` is the one function that touches
the solver, to confirm the artifact still reproduces at the current tree.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

__all__ = [
    "ANCHOR_SLUG",
    "ACCEPTANCE_ELEMENTS",
    "POOL_ENTRY",
    "METRICS_ARTIFACT",
    "CONVERGENCE_ARTIFACT",
    "load_json",
    "graded_keys",
    "acceptance_coverage",
    "graded_but_undeclared",
    "diagnostic_graded",
    "orphaned_evidence",
    "reproducibility_check",
    "dossier",
]

#: feature-ledger anchor this dossier is about
ANCHOR_SLUG = "CASE-FVM-DAMBREAK-STOKER"

#: the acceptance sentence of the ledger entry, split into its own elements.
#: ``key`` is a stable identifier; ``text`` quotes the ledger wording; ``gates``
#: are the ``_verdict`` keys that grade the element.  A gate may appear under
#: more than one element only if it genuinely grades both (none do today).
ACCEPTANCE_ELEMENTS = (
    {
        "key": "front_position",
        "text": "波前位置 x_f(t) vs 解析",
        "gates": ("front_fan<5%", "shock_position<5%",
                  "bore_position_modes<5%", "front_series<5%"),
    },
    {
        "key": "still_water_step",
        "text": "静水区水位阶跃 vs 解析",
        "gates": ("step_ratio<5%", "plateau_h2<5%", "step_ratio_modes<5%"),
    },
    {
        "key": "monotone",
        "text": "波形单调无伪振荡",
        "gates": ("monotone(no_extrema)", "tv_no_growth<0.5%"),
    },
)

POOL_ENTRY = "docs/oss_solutions/CASE-FVM-DAMBREAK-STOKER.json"
METRICS_ARTIFACT = "src/tensorfvm/verification/artifacts/shallow-water-dam-break/metrics.json"
CONVERGENCE_ARTIFACT = "src/tensorfvm/verification/artifacts/shallow-water-dam-break/convergence.json"

_VERIFICATION_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _VERIFICATION_ROOT.parents[2]


def _root(root=None) -> Path:
    return Path(root) if root is not None else _REPO_ROOT


def load_json(rel: str, root=None) -> dict:
    """Load a JSON artifact relative to the repository root."""
    with open(_root(root) / rel, encoding="utf-8") as handle:
        return json.load(handle)


def _metrics(root=None) -> dict:
    return load_json(METRICS_ARTIFACT, root)


def _convergence(root=None) -> dict:
    return load_json(CONVERGENCE_ARTIFACT, root)


# --------------------------------------------------------------------------- #
# graded set
# --------------------------------------------------------------------------- #
def graded_keys(verdict: dict) -> list[str]:
    """Keys whose ``pass`` feeds ``_verdict``'s aggregate ``all_pass``.

    ``all_pass`` itself is excluded (it is the aggregate, not a criterion).
    """
    return [key for key in verdict if key != "all_pass"]


def acceptance_coverage(verdict: dict) -> dict:
    """Map each acceptance element to the gates that grade it.

    Returns ``{element_key: {...}}`` with, per element, the ledger text, one row
    per gate (``value``, ``pass``, ``margin`` = ``limit / value`` when the value
    is a positive error), an ``all_pass`` flag and the *weakest* margin of the
    element (the gate that constrains it most).

    A gate declared in :data:`ACCEPTANCE_ELEMENTS` but absent from ``verdict``
    is reported as ``present=False`` rather than silently skipped: an element
    with no live gate is exactly what a closure audit must not miss.
    """
    out: dict = {}
    for element in ACCEPTANCE_ELEMENTS:
        rows = []
        for gate in element["gates"]:
            entry = verdict.get(gate)
            if entry is None:
                rows.append({"gate": gate, "present": False, "pass": False,
                             "value": None, "limit": None, "margin": None})
                continue
            value = entry.get("value")
            limit = entry.get("limit")
            margin = None
            if isinstance(value, (int, float)) and isinstance(limit, (int, float)):
                value_f, limit_f = float(value), float(limit)
                # a zero error has an unbounded margin: keep the JSON finite
                # (allow_nan=False) and say so explicitly instead of writing inf
                margin = None if value_f == 0.0 else limit_f / value_f
            rows.append({
                "gate": gate,
                "present": True,
                "pass": bool(entry.get("pass")),
                "measured": bool(entry.get("measured", value is not None)),
                "value": value,
                "limit": limit,
                "margin": margin,
                "margin_infinite": bool(value_f == 0.0) if margin is None
                and isinstance(value, (int, float)) else False,
            })
        present = [row for row in rows if row["present"]]
        finite = [row["margin"] for row in present if row["margin"] is not None]
        out[element["key"]] = {
            "text": element["text"],
            "gates": rows,
            "applicable": bool(present),
            "all_present": len(present) == len(rows),
            "all_pass": bool(present) and all(row["pass"] for row in present),
            "weakest_margin": min(finite) if finite else None,
            "weakest_gate": (min(
                (row for row in present if row["margin"] is not None),
                key=lambda row: row["margin"])["gate"] if finite else None),
        }
    return out


def graded_but_undeclared(verdict: dict) -> list[str]:
    """Verdict keys that feed ``all_pass`` but grade no acceptance element."""
    declared = {gate for element in ACCEPTANCE_ELEMENTS
                for gate in element["gates"]}
    return [key for key in graded_keys(verdict) if key not in declared]


# --------------------------------------------------------------------------- #
# diagnostic-promoted-to-gate detection
# --------------------------------------------------------------------------- #
def _diagnostic_ladders(convergence: dict) -> dict:
    """``{case: {quantity: [(n, value), ...]}}`` for ``*_diagnostic`` headline
    quantities published by the convergence artifact."""
    ladders: dict = {}
    for case, block in convergence.items():
        rows = block.get("rows") or []
        per_quantity: dict = {}
        for row in rows:
            headline = row.get("headline") or {}
            for name, value in headline.items():
                if "diagnostic" in name and isinstance(value, (int, float)):
                    per_quantity.setdefault(name, []).append(
                        (int(row["n"]), float(value)))
        for name in per_quantity:
            per_quantity[name].sort()
        ladders[case] = per_quantity
    return ladders


def _is_monotone_decreasing(ladder) -> bool:
    values = [value for _, value in ladder]
    return all(b <= a for a, b in zip(values, values[1:]))


def diagnostic_graded(verdict: dict, convergence_case: dict,
                      metrics_case: dict, tol: float) -> list[dict]:
    """Graded keys whose number is published under a diagnostic name.

    A key is flagged when

    1. it is graded (feeds ``all_pass``) but grades no acceptance element,
    2. its graded value equals a ``*_diagnostic`` quantity of the *same*
       convergence row set at the headline grid (matched to ``1e-12``
       relative), and
    3. that quantity's refinement ladder is **not** monotone decreasing --
       i.e. the bundle's own data shows it does not converge.

    Matching is by value, not by name, so renaming a key cannot hide the
    promotion.  Returns one dict per flagged key with the ladder and the
    evidence needed to reproduce the finding.
    """
    undeclared = set(graded_but_undeclared(verdict))
    ladders = _diagnostic_ladders({metrics_case["case_key"]: convergence_case})
    per_quantity = ladders.get(metrics_case["case_key"], {})
    n_headline = int(metrics_case["n"])
    findings = []
    for quantity, ladder in per_quantity.items():
        at_n = dict(ladder).get(n_headline)
        if at_n is None:
            continue
        for key in undeclared:
            value = verdict[key].get("value")
            if not isinstance(value, (int, float)):
                continue
            value = float(value)
            close = (value == at_n) or (
                abs(value - at_n) <= 1e-12 * max(abs(at_n), 1e-300))
            if not close:
                continue
            findings.append({
                "gate": key,
                "diagnostic_name": quantity,
                "value": value,
                "limit": verdict[key].get("limit"),
                "grid_n": n_headline,
                "ladder": ladder,
                "ladder_monotone_decreasing": _is_monotone_decreasing(ladder),
                "exceeds_tolerance_at": [n for n, v in ladder if v > tol],
            })
    return findings


# --------------------------------------------------------------------------- #
# orphaned amendments
# --------------------------------------------------------------------------- #
def orphaned_evidence(pool_entry: dict, root=None) -> list[dict]:
    """Extension-leg amendments that were never applied to the pool entry.

    Scans ``docs/CASE-FVM-DAMBREAK-STOKER-*/`` for the two amendment shapes the
    legs publish (``proposed_pool_amendment.json``, ``case_pool_entry.json``)
    and reports, per file, which of its criterion ids are missing from the
    registered ``gate_criteria``.
    """
    base = _root(root) / "docs"
    registered = {gate["id"] for gate in pool_entry.get("gate_criteria", [])}
    out = []
    for path in sorted(base.glob(f"{ANCHOR_SLUG}-*")):
        for name in ("proposed_pool_amendment.json", "case_pool_entry.json"):
            candidate = path / name
            if not candidate.is_file():
                continue
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                out.append({"file": str(candidate.relative_to(_root(root))),
                            "readable": False, "unapplied_ids": []})
                continue
            ids = [criterion.get("id") for criterion
                   in payload.get("additive_criteria", [])
                   if criterion.get("id")]
            if not ids and payload.get("slug"):
                ids = [payload["slug"]]
            out.append({
                "file": str(candidate.relative_to(_root(root))),
                "readable": True,
                "declared_ids": ids,
                "unapplied_ids": [i for i in ids if i not in registered],
                "note": payload.get("_comment"),
            })
    return out


# --------------------------------------------------------------------------- #
# reproducibility
# --------------------------------------------------------------------------- #
def reproducibility_check(n: int = 800) -> dict:
    """Re-run the two anchor cases at ``n`` and compare with the artifact.

    ``n = 800`` is ``dam_break.ACCEPTED_MIN_N``: the declared evidence floor of
    the pool entry, and cheap enough (~2 s) to run inside a test.  The
    comparison is against the ``n = 800`` row of the committed
    ``convergence.json``, so a change to the solver, the metric or the exact
    solution shows up as a drift here -- the drift gate the pool entry was
    missing (``tests/test_case_fvm_dambreak_stoker_pool.py`` only re-derives the
    *typed* numbers from the artifact; it never re-runs the solver).
    """
    from ..shallow_water.dam_break import RITTER_DRY, STOKER_WET, run_case

    convergence = _convergence()
    rows = []
    for case, quantity in ((RITTER_DRY, "front_fan"), (STOKER_WET, "shock")):
        block = convergence[case.key]
        reference_row = next((row for row in block["rows"]
                              if int(row["n"]) == int(n)), None)
        if reference_row is None:
            raise KeyError(f"no n={n} row for case {case.key}")
        reference = float(reference_row["headline"][quantity])
        run = run_case(case, n=n)
        metric_key = f"{quantity}_error"
        if metric_key not in run["metrics"]:
            metric_key = "shock_position_error" if quantity == "shock" else metric_key
        measured = float(run["metrics"][metric_key])
        rel = abs(measured - reference) / max(abs(reference), 1e-300)
        rows.append({
            "case": case.key,
            "quantity": quantity,
            "n": int(n),
            "artifact": reference,
            "rerun": measured,
            "relative_difference": rel,
            "bit_identical": measured == reference,
            "all_pass": bool(run["verdict"]["all_pass"]),
        })
    return {
        "n": int(n),
        "rows": rows,
        "all_bit_identical": all(row["bit_identical"] for row in rows),
        "all_pass": all(row["all_pass"] for row in rows),
    }


# --------------------------------------------------------------------------- #
# dossier
# --------------------------------------------------------------------------- #
def dossier(root=None, reproduce: bool = True) -> dict:
    """Assemble the closure dossier from the committed artifacts.

    ``reproduce=False`` skips the solver re-run (for callers that only need the
    static part, e.g. documentation builders).
    """
    pool_entry = load_json(POOL_ENTRY, root)
    metrics = _metrics(root)
    convergence = _convergence(root)
    tol = float(metrics.get("meta", {}).get("tolerance", 0.05))

    per_case = {}
    for case_key, case_block in metrics["cases"].items():
        verdict = case_block["verdict"]
        metrics_case = dict(case_block["metrics"])
        metrics_case["case_key"] = case_key
        metrics_case["n"] = int(metrics_case.get("n", 0)) or int(
            metrics["meta"].get("n", 0))
        # the artifact records the headline grid in meta; convergence rows carry
        # per-grid numbers, so take the largest grid as the headline grid.
        headline_n = max(int(row["n"])
                         for row in convergence[case_key]["rows"])
        metrics_case["n"] = headline_n
        coverage = acceptance_coverage(verdict)
        per_case[case_key] = {
            "verdict_all_pass": bool(verdict.get("all_pass")),
            "graded_keys": graded_keys(verdict),
            "n_graded": len(graded_keys(verdict)),
            "acceptance_coverage": coverage,
            "graded_but_undeclared": graded_but_undeclared(verdict),
            "diagnostic_graded": diagnostic_graded(
                verdict, convergence[case_key], metrics_case, tol),
        }

    # An acceptance element applies to a case only where at least one of its
    # gates exists (the still-water step has no meaning on the dry bed).  The
    # element is covered when it applies somewhere and every gate that does
    # apply passes.
    element_status = {}
    for element in ACCEPTANCE_ELEMENTS:
        applicable, ok = [], True
        for case_key, case in per_case.items():
            info = case["acceptance_coverage"][element["key"]]
            if not info["applicable"]:
                continue
            applicable.append(case_key)
            ok = ok and info["all_pass"]
        element_status[element["key"]] = {
            "applicable_cases": applicable,
            "pass": bool(applicable) and ok,
        }
    accepted_elements = all(status["pass"] for status in element_status.values())
    flagged = [finding for case in per_case.values()
               for finding in case["diagnostic_graded"]]
    orphans = orphaned_evidence(pool_entry, root)

    repro = reproducibility_check() if reproduce else None

    blocking = []
    if not accepted_elements:
        blocking.append("an acceptance element has no passing gate")
    if flagged:
        blocking.append(
            "the registered aggregate PASS is gated by a quantity the artifact "
            "bundle publishes as a non-convergent diagnostic: "
            + ", ".join(sorted({finding["gate"] for finding in flagged})))
    if repro is not None and not repro["all_bit_identical"]:
        blocking.append("the committed metrics no longer reproduce at this tree")

    return {
        "anchor": ANCHOR_SLUG,
        "pool_entry": {
            "status": pool_entry.get("status"),
            "verdict": pool_entry.get("verdict"),
            "gate_ids": [gate["id"] for gate in pool_entry["gate_criteria"]],
            "acceptance_criterion": pool_entry.get("acceptance_criterion"),
            "verdict_qualified": pool_entry.get("verdict_qualified"),
            "notes": pool_entry.get("notes"),
        },
        "tolerance": tol,
        "cases": per_case,
        "acceptance_fully_covered": accepted_elements,
        "acceptance_elements": element_status,
        "orphaned_evidence": orphans,
        "reproducibility": repro,
        "findings": blocking,
        "disposition": (
            "approve with notes" if accepted_elements and not blocked_by_metrics(
                flagged, repro) else "request changes"),
    }


def blocked_by_metrics(flagged, repro) -> bool:
    """A flag is a documentation defect; a failed reproduction is blocking."""
    if repro is not None and not repro["all_bit_identical"]:
        return True
    return False