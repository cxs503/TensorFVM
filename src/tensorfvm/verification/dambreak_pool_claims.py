"""Machine binding for the numbers the ``CASE-FVM-DAMBREAK-STOKER`` pool entry quotes.

Why this exists
---------------
``docs/oss_solutions/CASE-FVM-DAMBREAK-STOKER.json`` quotes numbers that come
from **two independent artifact bundles**:

* ``src/tensorfvm/verification/artifacts/shallow-water-dam-break/`` --
  ``metrics.json`` (the ``n = 1600`` verdict, which carries value *and* limit for
  every graded key) and ``convergence.json`` (the ``front_wetted_diagnostic``
  ladder);
* ``docs/case-fvm-dambreak-stoker/artifacts/reproduction.json`` -- the
  probe-based grid ladder that ``verdict_qualified`` summarises
  ("PASS at n >= 800 ... at n = 200 the fan-head probe fails").

The first bundle was already bound to the entry by
``tests/test_case_fvm_dambreak_stoker_pool.py``.  The second was quoted in prose
and watched by **no** gate at all (``grep`` for the path under ``tests/``,
``scripts/`` and ``src/`` returns nothing outside this module), and the *margin*
sentences of both ``verdict_qualified`` and ``graded_set_note`` were typed by
hand.  Re-deriving them (this module, 2026-11-16) showed they did not hold:

* ``verdict_qualified`` claimed the headline six pass "with 100x+ margin";
  the true minimum is 5.92x (``stoker-wet front_series<5%``), and
  ``ritter-dry depth_l1`` is only 64.8x, so the claim failed on two keys;
* ``graded_set_note`` claimed ``front_wetted<5%`` is "the only graded key with a
  margin under 12x"; four graded keys are under 12x;
* the same note put the acceptance sentence's weakest margin at 11.8x; the true
  weakest finite margin of an acceptance element is 4.98x
  (``stoker-wet tv_no_growth<0.5%``) -- 11.8x was the ``ritter-dry``
  ``front_series`` margin alone.

The acceptance state of the case is unaffected (every graded key passes, worst
error 0.844 %, no artifact number is changed); what was wrong was the *margin
rhetoric*, i.e. an overstated claim.  This module makes that class of error
unrepeatable:

1. :func:`probe_ladder` / :func:`gate_margins` re-derive the two structured
   blocks the entry now carries, so every number in them is injected from the
   artifacts and never typed by hand;
2. :func:`claim_problems` re-derives every number still written in prose
   (percentages, ``Nx`` margins, ``n >= K`` grid claims, graded-set membership)
   and returns the ones that do not reproduce;
3. :func:`graded_margins` / :func:`acceptance_weakest` expose the margin table
   used by the figures and the report.

No solver is run here -- this module binds prose to committed evidence.  Drift
between evidence and solver is the job of
``tests/test_dambreak_stoker_repro_verification.py`` (headline bundle) and
``tests/test_dambreak_probe_ladder_reproduction.py`` (probe bundle).
"""

from __future__ import annotations

import json
import pathlib
import re

from . import dambreak_anchor_closure as closure

__all__ = [
    "POOL_ENTRY",
    "METRICS_ARTIFACT",
    "CONVERGENCE_ARTIFACT",
    "PROBE_BUNDLE",
    "PROBE",
    "PROSE_FIELDS",
    "load_json",
    "graded_margins",
    "acceptance_weakest",
    "smallest_finite",
    "keys_under",
    "probe_ladder",
    "gate_margins",
    "percentage_claims",
    "factor_claims",
    "grid_claims",
    "claim_problems",
]

#: the pool entry this module binds
POOL_ENTRY = "docs/oss_solutions/CASE-FVM-DAMBREAK-STOKER.json"
#: headline evidence (``n = 1600`` verdict with value *and* limit per key)
METRICS_ARTIFACT = closure.METRICS_ARTIFACT
#: the ``front_wetted_diagnostic`` ladder
CONVERGENCE_ARTIFACT = closure.CONVERGENCE_ARTIFACT
#: the probe-based grid ladder behind ``verdict_qualified``
PROBE_BUNDLE = "docs/case-fvm-dambreak-stoker/artifacts/reproduction.json"
#: the probe name ``verdict_qualified`` quotes as "the fan-head probe"
PROBE = "fan_head"

#: prose fields whose numbers are bound to evidence (``gate_criteria`` and
#: ``quantitative_targets`` are bound separately, by the pool binding test)
PROSE_FIELDS = ("verdict_qualified", "graded_set_note")

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]

# "5.58%", "10.5 %"
_PERCENT_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*%")
# "1.73x", "5.92x", "12 x" -- a bound token ("below 2x") is handled separately
_FACTOR_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*[x\u00d7](?![0-9a-zA-Z])")
# "n>=800", "n>= 800", "n = 200"
_GRID_GE_RE = re.compile(r"n\s*>=\s*(\d+)")
_GRID_EQ_RE = re.compile(r"(?<!>)"r"n\s*=\s*(\d+)")


def load_json(rel: str, root=None) -> dict:
    """Load a repository-relative JSON file."""
    base = _REPO_ROOT if root is None else pathlib.Path(root)
    with open(base / rel, encoding="utf-8") as handle:
        return json.load(handle)


# --------------------------------------------------------------------------- #
# margin table
# --------------------------------------------------------------------------- #
def graded_margins(metrics: dict) -> dict:
    """``{case: {gate: {value, limit, margin, margin_infinite, pass}}}``.

    ``margin`` is ``limit / value`` -- how many times the measured error still
    fits under its own gate.  A zero error has an unbounded margin; it is
    reported as ``None`` with ``margin_infinite = True`` so the payload stays
    finite JSON.
    """
    table = {}
    for case, block in metrics["cases"].items():
        rows = {}
        for gate, item in block["verdict"].items():
            if gate == "all_pass":
                continue
            value = float(item["value"])
            limit = float(item["limit"])
            infinite = value == 0.0
            rows[gate] = {
                "value": value,
                "limit": limit,
                "margin": None if infinite else limit / value,
                "margin_infinite": infinite,
                "pass": bool(item["pass"]),
            }
        table[case] = rows
    return table


def keys_under(margins: dict, bound: float) -> list[tuple[str, str, float]]:
    """Graded ``(case, gate, margin)`` rows whose finite margin is ``< bound``."""
    rows = [(case, gate, row["margin"])
            for case, rows_ in margins.items()
            for gate, row in rows_.items()
            if row["margin"] is not None and row["margin"] < bound]
    return sorted(rows, key=lambda row: row[2])


def smallest_finite(margins: dict) -> tuple[str, str, float]:
    """The graded key with the smallest finite margin (``(case, gate, margin)``)."""
    rows = keys_under(margins, float("inf"))
    if not rows:
        raise ValueError("no finite margin in the graded set")
    return rows[0]


def acceptance_weakest(metrics: dict) -> dict:
    """Weakest margin per acceptance element, per case, and overall.

    Reuses :data:`tensorfvm.verification.dambreak_anchor_closure.ACCEPTANCE_ELEMENTS`
    so the element -> gate mapping lives in exactly one place.
    """
    per_case = {}
    for case, block in metrics["cases"].items():
        coverage = closure.acceptance_coverage(block["verdict"])
        per_case[case] = {
            element: {
                "gate": info["weakest_margin_gate"] if "weakest_margin_gate" in info
                        else _weakest_gate(info),
                "margin": info["weakest_margin"],
            }
            for element, info in coverage.items()
            if info["weakest_margin"] is not None
        }
    overall = None
    for case, elements in per_case.items():
        for element, info in elements.items():
            if overall is None or info["margin"] < overall["margin"]:
                overall = {"case": case, "element": element,
                           "gate": info["gate"], "margin": info["margin"]}
    return {"per_case": per_case, "overall": overall}


def _weakest_gate(info: dict) -> str | None:
    """Gate carrying the weakest finite margin of one acceptance element."""
    finite = [row for row in info["gates"] if row["margin"] is not None]
    if not finite:
        return None
    return min(finite, key=lambda row: row["margin"])["gate"]


# --------------------------------------------------------------------------- #
# probe ladder (the bundle ``verdict_qualified`` summarises)
# --------------------------------------------------------------------------- #
def probe_ladder(bundle: dict, probe: str = PROBE) -> dict:
    """The probe's relative error on every (case, resolution) of the bundle."""
    tolerance = float(bundle["tolerance"])
    relative_error: dict[str, dict[str, float]] = {}
    failing: dict[str, list[int]] = {}
    first_pass: dict[str, int | None] = {}
    for case, runs in bundle["runs"].items():
        errors = {}
        for n in sorted(int(key) for key in runs):
            errors[n] = float(runs[str(n)]["verdict"]["checks"][probe])
        relative_error[case] = {str(n): err for n, err in errors.items()}
        failing[case] = [n for n, err in errors.items() if err > tolerance]
        first_pass[case] = next(
            (n for n, err in errors.items() if err <= tolerance), None)
    return {
        "artifact": PROBE_BUNDLE,
        "probe": probe,
        "tolerance": tolerance,
        "accepted_n": int(bundle["accepted_n"]),
        "resolutions": [int(n) for n in bundle["resolutions"]],
        "relative_error": relative_error,
        "failing_rungs": failing,
        "first_passing_rung_n": first_pass,
    }


# --------------------------------------------------------------------------- #
# gate margin block (injected into the entry)
# --------------------------------------------------------------------------- #
#: the "headline six" as the entry's ``verdict_qualified`` names them, mapped to
#: the verdict keys that grade them (``monotone`` -> the extrema count)
HEADLINE_GATES = {
    "ritter-dry": ("front_fan<5%", "depth_l1<5%", "front_series<5%",
                   "monotone(no_extrema)"),
    "stoker-wet": ("shock_position<5%", "step_ratio<5%", "depth_l1<5%",
                   "front_series<5%", "monotone(no_extrema)"),
}

#: bounds under which the corrected ``graded_set_note`` counts keys.  A bound is
#: a *reading* of the table, not a gate: it exists so "only one key is below X"
#: becomes a machine-checkable statement instead of a hand-typed boast.
MARGIN_BOUNDS = (2.0, 12.0)


def gate_margins(metrics: dict) -> dict:
    """The derived ``gate_margins`` block of the pool entry."""
    margins = graded_margins(metrics)
    grid = {case: block["discretisation"]["n"]
            for case, block in metrics["cases"].items()}
    if len(set(grid.values())) != 1:
        raise ValueError(f"cases are not on one grid: {grid}")
    under = {f"under_{int(b)}x": [
        {"case": case, "gate": gate, "margin": margin}
        for case, gate, margin in keys_under(margins, b)]
        for b in MARGIN_BOUNDS}
    case, gate, margin = smallest_finite(margins)
    headline = {}
    for case_key, gates in HEADLINE_GATES.items():
        finite = [(gate_name, margins[case_key][gate_name]["margin"])
                  for gate_name in gates
                  if margins[case_key][gate_name]["margin"] is not None]
        weakest_gate, weakest = min(finite, key=lambda row: row[1])
        headline[case_key] = {
            "gates": list(gates),
            "weakest_gate": weakest_gate,
            "weakest_margin": weakest,
        }
    return {
        "source": METRICS_ARTIFACT,
        "grid_n": next(iter(grid.values())),
        "definition": ("margin = limit / value at the pool grid; a zero-valued "
                       "error has an unbounded margin (null + margin_infinite)"),
        "graded_set_size": {case_key: len(rows) for case_key, rows in margins.items()},
        "graded": margins,
        **under,
        "smallest_finite": {"case": case, "gate": gate, "margin": margin},
        "headline_weakest": headline,
        "acceptance_weakest": acceptance_weakest(metrics),
    }


# --------------------------------------------------------------------------- #
# prose claim re-derivation
# --------------------------------------------------------------------------- #
def percentage_claims(text: str) -> list[tuple[str, int]]:
    """``(token, decimals)`` for every ``p %`` written in ``text``."""
    return [(match.group(1), len(match.group(1).split(".")[1])
             if "." in match.group(1) else 0)
            for match in _PERCENT_RE.finditer(text)]


def factor_claims(text: str) -> list[tuple[str, int]]:
    """``(token, decimals)`` for every ``Nx`` written in ``text``.

    Bound tokens ("below 2x") are *included*: they are checked against the
    count statement they belong to, not against the margin table.
    """
    return [(match.group(1), len(match.group(1).split(".")[1])
             if "." in match.group(1) else 0)
            for match in _FACTOR_RE.finditer(text)]


def grid_claims(text: str) -> dict:
    """The ``n >= K`` / ``n = K`` claims of ``text``."""
    return {
        "at_least": [int(m.group(1)) for m in _GRID_GE_RE.finditer(text)],
        "equals": [int(m.group(1)) for m in _GRID_EQ_RE.finditer(text)],
    }


def _matches(value: float, token: str, decimals: int, scale: float = 1.0) -> bool:
    """``round(scale * value, decimals)`` renders as ``token``?"""
    return abs(round(scale * value, decimals) - float(token)) < 1e-12


def _explainable_percent(token: str, decimals: int, block: dict, metrics: dict,
                         convergence: dict, bundle: dict) -> bool:
    """Is a ``p %`` claim one of the numbers the artifacts actually publish?

    ``block`` is the *derived* probe ladder, never the committed one: a stale
    committed block must not be able to explain (i.e. excuse) a stale sentence.
    """
    values = []
    # probe ladder (verdict_qualified's coarse-grid sentence)
    values.extend(err for case in block["relative_error"].values()
                  for err in case.values())
    values.append(float(bundle["tolerance"]))
    values.append(block["tolerance"])
    # the diagnostic ladder (graded_set_note's non-convergence sentence)
    for case, series in convergence.items():
        headline = (series.get("rows") or [{}])[0].get("headline") or {}
        for name in headline:
            if name.endswith("_diagnostic"):
                values.extend(float(row["headline"][name])
                              for row in series["rows"]
                              if name in (row.get("headline") or {}))
    # graded values and their own limits (e.g. tv_no_growth's 0.5 % limit)
    for case, rows in metrics["cases"].items():
        for gate, item in rows["verdict"].items():
            if gate == "all_pass":
                continue
            values.extend([float(item["value"]), float(item["limit"])])
    return any(_matches(value, token, decimals, scale=100.0) for value in values)


def _explainable_factor(token: str, decimals: int, block: dict) -> bool:
    """Is an ``Nx`` claim the rendered margin of some graded key?"""
    values = [row["margin"] for case in (block.get("graded") or {}).values()
              for row in case.values() if row.get("margin") is not None]
    values.extend(row["margin"] for key in block
                  if key.startswith("under_")
                  for row in block[key])
    if block.get("smallest_finite"):
        values.append(block["smallest_finite"]["margin"])
    for weakest in (block.get("headline_weakest") or {}).values():
        if weakest.get("weakest_margin") is not None:
            values.append(weakest["weakest_margin"])
    overall = (block.get("acceptance_weakest") or {}).get("overall") or {}
    if overall.get("margin") is not None:
        values.append(overall["margin"])
    return any(_matches(value, token, decimals) for value in values)


_GRADED_SET_RE = re.compile(r"(ritter-dry|stoker-wet)\s*=\s*([^;(]+?)\((\d+)\)")
_UNDER_CLAIM_RE = re.compile(r"(?:below|under)\s+(\d+(?:\.\d+)?)\s*[x\u00d7]")


def _bare_gate(gate: str) -> str:
    """``front_wetted<5%`` -> ``front_wetted``, ``monotone(no_extrema)`` -> ``monotone``."""
    return gate.split("<")[0].split("(")[0]


def _graded_set_claim_problems(note: str, metrics: dict) -> list[str]:
    """The note's per-case graded-set enumeration must be the real graded set."""
    problems = []
    actual = {case: {_bare_gate(gate) for gate in block["verdict"]
                     if gate != "all_pass"}
              for case, block in metrics["cases"].items()}
    seen = set()
    for match in _GRADED_SET_RE.finditer(note):
        case = match.group(1)
        seen.add(case)
        listed = {name.strip() for name in match.group(2).split(",") if name.strip()}
        if listed != actual[case]:
            problems.append(
                f"graded_set_note: {case} lists {sorted(listed)} but the artifact "
                f"grades {sorted(actual[case])}")
        if int(match.group(3)) != len(actual[case]):
            problems.append(
                f"graded_set_note: {case} is counted {match.group(3)}x but the "
                f"artifact grades {len(actual[case])} keys")
    if seen != set(actual):
        problems.append(f"graded_set_note: cases {sorted(seen)} != {sorted(actual)}")
    return problems


def _only_claims(window: str) -> list[str]:
    """``sole`` / ``not_sole`` readings of every "only" in a sentence window.

    ``not the only`` is the negation of the same reading, and it is a claim in
    its own right (at least two keys must be below the bound), so the checker
    keeps the two apart instead of matching a bare ``only``.
    """
    claims = []
    for match in re.finditer(r"\bonly\b", window):
        prefix = window[max(0, match.start() - 12):match.start()]
        claims.append("not_sole" if re.search(r"\bnot\b", prefix) else "sole")
    return claims


def _bound_claim_problems(note: str, margins: dict) -> list[str]:
    """Every "below Nx" statement must agree with the margin table.

    Two readings are supported, because both are what a reader would check:

    * "only ... below Nx" -> exactly one graded key is below that bound;
    * "not the only ... below Nx" -> at least two are;
    * "all K keys ... below Nx" -> ``K`` equals the number below the bound.

    The window of a claim stops at the previous bound claim, so the "only" of a
    "below 2x" clause is never mis-read as a claim about a later "below 12x".
    """
    problems = []
    window_start = 0
    for match in _UNDER_CLAIM_RE.finditer(note):
        window = note[window_start:match.start()]
        window_start = match.end()
        cut = max(window.rfind(". "), window.rfind(";"), window.rfind("\u2014"))
        if cut >= 0:
            window = window[cut + 1:]
        bound = float(match.group(1))
        below = keys_under(margins, bound)
        for claim in _only_claims(window):
            if claim == "sole" and len(below) != 1:
                problems.append(
                    f"graded_set_note: 'only ... below {match.group(1)}x' but "
                    f"{len(below)} graded keys are below it: "
                    f"{[(c, g) for c, g, _ in below]}")
            elif claim == "not_sole" and len(below) < 2:
                problems.append(
                    f"graded_set_note: 'not the only ... below {match.group(1)}x' "
                    f"but the table has {len(below)} key(s) below it")
        for count in re.finditer(r"\ball (\d+) keys\b", window):
            if int(count.group(1)) != len(below):
                problems.append(
                    f"graded_set_note: claims all {count.group(1)} keys below "
                    f"{match.group(1)}x but the table has {len(below)}")
    return problems


_WEAKEST_CLAIM_RE = re.compile(
    r"(?:weakest|smallest|minimum)[^.;]{0,80}?(\d+(?:\.\d+)?)\s*[x\u00d7]")


def _weakest_claim_problems(note: str, block: dict) -> list[str]:
    """"weakest / smallest / minimum ... Nx" must quote the real minimum.

    The scope is read off the sentence: a claim near the word *acceptance* is
    pinned to ``gate_margins.acceptance_weakest.overall``; a claim near
    *headline* is pinned to the weakest headline margin; anything else is
    pinned to the smallest finite margin of the graded set.  This is the check
    that catches a *mis-scoped* margin (a real number quoted for the wrong set),
    which plain value-explainability cannot see.
    """
    problems = []
    for match in _WEAKEST_CLAIM_RE.finditer(note):
        token = match.group(1)
        decimals = len(token.split(".")[1]) if "." in token else 0
        prefix = note[max(0, match.start() - 240):match.start()]
        if "acceptance" in prefix:
            overall = (block.get("acceptance_weakest") or {}).get("overall") or {}
            allowed = [overall["margin"]] if overall.get("margin") is not None else []
            scope = "the acceptance elements"
        elif "headline" in prefix:
            allowed = [row["weakest_margin"]
                       for row in (block.get("headline_weakest") or {}).values()
                       if row.get("weakest_margin") is not None]
            scope = "the headline six"
        else:
            smallest = (block.get("smallest_finite") or {}).get("margin")
            allowed = [smallest] if smallest is not None else []
            scope = "the graded set"
        if allowed and not any(_matches(value, token, decimals) for value in allowed):
            problems.append(
                f"graded_set_note: 'weakest ... {token}x' does not reproduce: the "
                f"weakest margin of {scope} is "
                f"{min(allowed):.4g}x")
    return problems


def _grid_claim_problems(text: str, block: dict, label: str) -> list[str]:
    """``n >= K`` must be the bundle's accepted grid, ``n = K`` a measured rung."""
    problems = []
    claims = grid_claims(text)
    for rung in claims["at_least"]:
        if rung != block["accepted_n"]:
            problems.append(
                f"{label}: 'n>={rung}' but the bundle accepts at n={block['accepted_n']}")
        for case, first in block["first_passing_rung_n"].items():
            if first is None or first > rung:
                problems.append(
                    f"{label}: 'n>={rung}' but {case} only passes from n={first}")
    for rung in claims["equals"]:
        if rung not in block["resolutions"]:
            problems.append(
                f"{label}: 'n={rung}' is not a measured rung "
                f"{block['resolutions']}")
    return problems


def claim_problems(entry: dict, metrics: dict | None = None,
                   bundle: dict | None = None,
                   convergence: dict | None = None) -> list[str]:
    """Every way the entry's quoted numbers fail to re-derive from its evidence.

    Returns a list of human-readable problems; an empty list means the entry is
    fully bound (its structured blocks are the derived payload and every number
    left in prose reproduces from an artifact).
    """
    metrics = load_json(METRICS_ARTIFACT) if metrics is None else metrics
    bundle = load_json(PROBE_BUNDLE) if bundle is None else bundle
    convergence = (load_json(CONVERGENCE_ARTIFACT) if convergence is None
                   else convergence)
    problems = []

    derived_ladder = probe_ladder(bundle)
    if entry.get("probe_ladder") != derived_ladder:
        problems.append("probe_ladder: committed block != derived from "
                        f"{PROBE_BUNDLE}")

    derived_margins = gate_margins(metrics)
    if entry.get("gate_margins") != derived_margins:
        problems.append("gate_margins: committed block != derived from "
                        f"{METRICS_ARTIFACT}")

    margins = graded_margins(metrics)
    for field in PROSE_FIELDS:
        text = entry.get(field) or ""
        for token, decimals in percentage_claims(text):
            if not _explainable_percent(token, decimals, derived_ladder, metrics,
                                        convergence, bundle):
                problems.append(
                    f"{field}: percentage {token}% does not reproduce from any "
                    "artifact value")
        bounds = [m.span(1) for m in _UNDER_CLAIM_RE.finditer(text)]
        for match in _FACTOR_RE.finditer(text):
            if any(start <= match.start(1) < end for start, end in bounds):
                continue  # a bound, checked by _bound_claim_problems
            token = match.group(1)
            decimals = len(token.split(".")[1]) if "." in token else 0
            if not _explainable_factor(token, decimals, derived_margins):
                problems.append(
                    f"{field}: margin {token}x is not a margin of any graded key")

    problems.extend(_grid_claim_problems(entry.get("verdict_qualified") or "",
                                         derived_ladder, "verdict_qualified"))
    problems.extend(_graded_set_claim_problems(entry.get("graded_set_note") or "",
                                               metrics))
    problems.extend(_bound_claim_problems(entry.get("graded_set_note") or "",
                                          margins))
    problems.extend(_weakest_claim_problems(entry.get("graded_set_note") or "",
                                            derived_margins))
    return problems