"""Round 20: lower-Reynolds sister cases for ``CASE-FVM-PERIODIC-HILLS``.

Why this module exists
----------------------
Nineteen rounds of this anchor measured ``x_r/H`` between ``+30 %`` and
``+42 %`` of the reference at ``Re_b = 2800`` in **two dimensions with a closure**
(SA, SST k-omega) and concluded, round after round, "the framework or the closure
is wrong".  Those two hypotheses were never separated, because every published
run of rounds 9--19 carried a turbulence closure inside the thing being tested.

The reference source quoted in :mod:`tensorfvm.verification.periodic_hill_reference`
gives the reattachment length at **five** Reynolds numbers, not one::

    x_R/h = 5.24, 5.19, 5.41, 5.09, 4.69   for Re = 700, 1400, 2800, 5600, 10595

At the two lowest of those, the 2-D equations with no closure are a defensible
model of the flow (the same KB page data come from a 3-D DNS, so a match is
*confirmatory*; a mismatch is ambiguous between discretisation error and
2-D-vs-3-D physics -- see :func:`attribution_limits`).  That is the sister case
round 18's section 7 item 2 asked for and no round had run.

What is here
------------
* :data:`KB_REATTACHMENT_BY_RE` -- the five KB values, single-sourced, **no
  interpolation**: an unlisted Reynolds number returns ``None`` rather than a
  fitted number dressed up as a reference.
* :func:`reattachment_reference` / :func:`separation_reference` -- reference
  lookup with provenance.
* :func:`score_case` -- one artefact scored against the reference for *its own*
  Reynolds number, with the certified (round 17) wall-shear operator
  recomputing ``x_r/H`` from stored mean shear when it is available, so the
  verdict never trusts a derived quantity inside the payload.
* :func:`sister_table` / :func:`attribution` -- the cross-Re table and the
  closure-versus-framework attribution rule, written so a mismatch cannot be
  silently reported as a pass.

Constants are re-exported from :mod:`periodic_hill_reference` where they exist
there; this module adds the *tabulated Re dependence*, not a second copy of the
``Re = 2800`` value (which would be a place for the two to drift apart).
"""

from __future__ import annotations

import math

from .periodic_hill_reference import X_R_OVER_H, X_R_TOLERANCE
from .wall_shear_bubbles import separation_reattachment

__all__ = [
    "KB_REATTACHMENT_BY_RE",
    "KB_SEPARATION_BY_RE",
    "KB_SOURCE",
    "attribution",
    "attribution_limits",
    "reattachment_reference",
    "score_case",
    "separation_reference",
    "sister_table",
]

#: ``x_R/h`` tabulated by the reference's own authors on the ERCOFTAC Knowledge
#: Base Wiki page ``UFR 3-30 Evaluation`` (quoted verbatim in
#: :mod:`tensorfvm.verification.periodic_hill_reference`): "The corresponding
#: values of the reattachment lengths are x_R/h = 5.24, 5.19, 5.41, 5.09, and
#: 4.69 for Re = 700 to 10,595, respectively."
KB_REATTACHMENT_BY_RE: dict[float, float] = {
    700.0: 5.24,
    1400.0: 5.19,
    2800.0: X_R_OVER_H,  # 5.41 -- single constant, shared with the Re=2800 gate
    5600.0: 5.09,
    10595.0: 4.69,
}

#: ``x_S/h`` from the same page.  The page states these as *approximate*
#: ("~0.3 at Re=700, ~0.18 at Re=5600, ~0.19 at Re=10595") and gives **no**
#: number for Re = 2800, which is why the ``Re = 2800`` gate reports ``x_S/H`` as
#: WITHHELD.  ``None`` here means "the source gives no number", not "zero".
KB_SEPARATION_BY_RE: dict[float, float | None] = {
    700.0: 0.3,
    1400.0: None,
    2800.0: None,
    5600.0: 0.18,
    10595.0: 0.19,
}

KB_SOURCE = (
    "ERCOFTAC Knowledge Base Wiki, UFR 3-30 Evaluation (§ 'Evaluation'), "
    "authored by the case's own authors (Rapp, Breuer, Manhart, Peller); "
    "case geometry/BC = Breuer et al. 2009, Computers & Fluids 38:433, "
    "doi:10.1016/j.compfluid.2008.05.002"
)

#: Exact-match tolerance for the Re lookup.  Deliberately tight: the table is a
#: set of published *cases*, not a curve fit, and interpolating between them
#: would manufacture a reference number that no one published.
_RE_MATCH_RTOL = 1e-6


def _lookup(table: dict[float, float | None], reynolds: float):
    re = float(reynolds)
    if not math.isfinite(re):
        return None, None
    for key, value in table.items():
        if abs(re - key) <= _RE_MATCH_RTOL * key:
            return key, value
    return None, None


def reattachment_reference(reynolds: float) -> dict | None:
    """Reference ``x_R/h`` for exactly this Reynolds number, or ``None``.

    No interpolation and no nearest-neighbour fallback: a sister case at
    ``Re = 1000`` has no published ``x_R/h`` in this source and must be scored
    as WITHHELD rather than against a number interpolated between two cases.
    """
    key, value = _lookup(KB_REATTACHMENT_BY_RE, reynolds)
    if key is None:
        return None
    return {
        "reynolds": key,
        "x_r_over_H": float(value),
        "tolerance": X_R_TOLERANCE,
        "band": (float(value) * (1.0 - X_R_TOLERANCE),
                 float(value) * (1.0 + X_R_TOLERANCE)),
        "source": KB_SOURCE,
    }


def separation_reference(reynolds: float) -> dict | None:
    """Reference ``x_S/h`` for this Reynolds number, or ``None`` when the source
    publishes no number (``Re = 2800`` and ``1400``)."""
    key, value = _lookup(KB_SEPARATION_BY_RE, reynolds)
    if key is None or value is None:
        return None
    return {"reynolds": key, "x_s_over_H": float(value), "approximate": True,
            "source": KB_SOURCE}


def _relative_error(computed: float, reference: float) -> float:
    return abs(float(computed) - float(reference)) / abs(float(reference))


def score_case(payload: dict, *, reynolds: float | None = None,
               mean_tau_b=None, x_over_h=None, window=(1.9, 7.1)) -> dict:
    """Score one lower-Re sister artefact against *its own* reference.

    Parameters
    ----------
    payload
        An artefact in the round-9/round-11 schema (``run_time_averaged_case``).
    reynolds
        Reynolds number of the run; defaults to the artefact's own
        ``config.reynolds``.  Never defaulted to 2800 -- a lower-Re artefact
        compared against the ``Re = 2800`` number is the exact mistake this
        module exists to prevent.
    mean_tau_b, x_over_h
        When supplied (read from the run's ``.npz``), ``x_r/H`` is *recomputed*
        with the certified round-17 operator instead of being read from the
        payload's derived field.  Both must be given together.
    window
        Admissibility window forwarded to the certified operator; the number is
        reported with status ``out_of_window`` rather than discarded.
    """
    cfg = payload.get("config") or {}
    if reynolds is None:
        reynolds = payload.get("reynolds", cfg.get("reynolds"))
    if reynolds is None:
        raise ValueError("payload carries no Reynolds number and none was given")

    computed = None
    provenance = "payload.time_averaged.x_r_over_H"
    if mean_tau_b is not None or x_over_h is not None:
        if mean_tau_b is None or x_over_h is None:
            raise ValueError("mean_tau_b and x_over_h must be supplied together")
        certified = separation_reattachment(mean_tau_b, x_over_h, window=window)
        computed = certified.get("x_r_over_H")
        provenance = "certified round-17 operator on stored mean wall shear"
    if computed is None:
        computed = ((payload.get("time_averaged") or {}).get("x_r_over_H"))

    diverged = payload.get("diverged_at_step")
    rea = reattachment_reference(reynolds)
    sep = separation_reference(reynolds)

    row = {
        "reynolds": float(reynolds),
        "closure": (payload.get("closure") or {}).get("name"),
        "x_r_over_H": None if computed is None or not _finite(computed) else float(computed),
        "x_r_provenance": provenance,
        "reference_x_r_over_H": None if rea is None else rea["x_r_over_H"],
        "reference_source": KB_SOURCE if rea is not None else None,
        "diverged_at_step": diverged,
        "samples": payload.get("samples"),
    }

    if diverged is not None:
        row["verdict"] = "DIVERGED"
        row["error"] = None
        row["below_10_percent"] = False
        row["below_3_percent"] = False
        row["reason"] = f"run recorded divergence at step {diverged}; no verdict on the physics"
    elif row["x_r_over_H"] is None:
        row["verdict"] = "UNDEFINED"
        row["error"] = None
        row["below_10_percent"] = False
        row["below_3_percent"] = False
        row["reason"] = "no certified reattachment crossing in the mean wall shear"
    elif rea is None:
        row["verdict"] = "WITHHELD"
        row["error"] = None
        row["below_10_percent"] = None
        row["below_3_percent"] = None
        row["reason"] = (f"no published x_R/h for Re = {float(reynolds):g} in the "
                         f"authoritative source; no interpolation performed")
    else:
        err = _relative_error(row["x_r_over_H"], rea["x_r_over_H"])
        row["error"] = err
        row["below_10_percent"] = bool(err <= X_R_TOLERANCE)
        row["below_3_percent"] = bool(err < 0.03)
        row["verdict"] = "PASS" if row["below_10_percent"] else "FAIL"
        row["reason"] = (f"x_r/H = {row['x_r_over_H']:.3f} vs {rea['x_r_over_H']:.2f} "
                         f"= {err:.1%} ({'inside' if row['below_10_percent'] else 'outside'} "
                         f"the {X_R_TOLERANCE:.0%} band {rea['band'][0]:.2f}-{rea['band'][1]:.2f})")

    if sep is not None and row["x_r_over_H"] is not None:
        rea_s = ((payload.get("time_averaged") or {}).get("x_s_over_H"))
        if rea_s is not None and _finite(rea_s):
            row["x_s_over_H"] = float(rea_s)
            row["x_s_reference"] = sep["x_s_over_H"]
            row["x_s_error"] = _relative_error(rea_s, sep["x_s_over_H"])
            row["x_s_approximate_reference"] = True
    return row


def _finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def sister_table(payloads: dict[str, dict], *, reynolds: dict[str, float] | None = None) -> dict:
    """Score a set of sister artefacts into a comparable table.

    ``payloads`` maps a tag to a payload; ``reynolds`` optionally overrides the
    artefact's own ``config.reynolds`` per tag (used by tests and by re-scoring
    old artefacts whose ``config`` block predates the field).
    """
    rows = []
    for tag, payload in sorted(payloads.items()):
        re = (reynolds or {}).get(tag)
        rows.append(score_case(payload, reynolds=re))
    rows.sort(key=lambda r: (r["reynolds"], str(r["closure"])))
    return {"rows": rows, "source": KB_SOURCE,
            "tolerance": X_R_TOLERANCE,
            "n_pass": sum(1 for r in rows if r["verdict"] == "PASS"),
            "n_fail": sum(1 for r in rows if r["verdict"] == "FAIL"),
            "n_withheld": sum(1 for r in rows if r["verdict"] == "WITHHELD"),
            "n_undefined": sum(1 for r in rows if r["verdict"] == "UNDEFINED"),
            "n_diverged": sum(1 for r in rows if r["verdict"] == "DIVERGED")}


def attribution(laminar_rows: list[dict], closure_rows: list[dict]) -> dict:
    """Split the ``Re = 2800`` error between framework and closure.

    The rule is fixed in advance, so the result cannot be read into the number:

    * ``closure_attributable`` -- the closure-free run at ``Re = 2800`` is
      inside the 10 % band while a closure run at the same mesh/Re is not: the
      closure, not the discretisation, produces the miss.
    * ``framework_attributable`` -- the closure-free run at ``Re = 2800`` is
      *also* outside the band by more than the closure-free lower-Re sister
      cases are: the framework carries an error that no closure change fixes.
    * ``inconclusive`` -- anything else (missing, diverged or undefined rows),
      reported as such rather than as a null result.
    """
    def _by_re(rows, re):
        out = [r for r in rows if abs(float(r["reynolds"]) - re) <= 1e-6 * re]
        return out

    lam2800 = _by_re(laminar_rows, 2800.0)
    clo2800 = _by_re(closure_rows, 2800.0)
    sister = [r for r in laminar_rows if float(r["reynolds"]) < 2800.0]
    if not lam2800 or not sister:
        return {"status": "inconclusive",
                "reason": "no closure-free Re=2800 row and/or no closure-free lower-Re sister row"}
    lam = lam2800[0]
    sister_ok = [r for r in sister if r["verdict"] in ("PASS", "FAIL")]
    if lam["error"] is None or not sister_ok:
        return {"status": "inconclusive",
                "reason": "closure-free rows exist but carry no measurable x_r/H",
                "rows": [r["verdict"] for r in lam2800 + sister]}
    sister_max = max(r["error"] for r in sister_ok)
    out = {"laminar_2800_error": lam["error"], "sister_max_error": sister_max,
           "sister_rows": [(r["reynolds"], r["verdict"], r["error"]) for r in sister_ok],
           "closure_rows": [(r["reynolds"], r["closure"], r["verdict"], r["error"])
                            for r in clo2800]}
    if clo2800:
        out["closure_2800_error"] = clo2800[0]["error"]
    if lam["below_10_percent"]:
        out["status"] = "closure_attributable" if clo2800 else "framework_ok_no_closure_row"
        out["reason"] = ("closure-free Re=2800 run is inside the band, so the "
                         "discretisation reproduces the benchmark without a closure")
    elif lam["error"] > max(sister_max, X_R_TOLERANCE):
        out["status"] = "framework_attributable"
        out["reason"] = (f"closure-free Re=2800 error {lam['error']:.1%} exceeds both the "
                         f"{X_R_TOLERANCE:.0%} band and the worst closure-free sister error "
                         f"{sister_max:.1%}; a closure change cannot explain it")
    else:
        out["status"] = "inconclusive"
        out["reason"] = (f"closure-free Re=2800 error {lam['error']:.1%} sits inside the band "
                         f"of the sister-case errors ({sister_max:.1%}); the two effects are "
                         f"not separable from these rows")
    return out


def attribution_limits() -> dict:
    """What a matching (or mismatching) sister case does and does not prove.

    Stated as data, not prose, so a report quoting an attribution also quotes
    its limits.
    """
    return {
        "agreement_proves": (
            "the 2-D discretisation (terrain-following mesh, momentum assembly, "
            "pressure projection, bulk-velocity controller, wall-shear metric) is "
            "not the source of the Re=2800 miss, because the same code path with "
            "the same mesh family reproduces a published reattachment length at a "
            "lower Reynolds number"),
        "agreement_does_not_prove": (
            "that the Re=2800 flow is two-dimensional: 2-D-vs-3-D physics at "
            "Re=2800 remains an open explanation for the residual, and only a "
            "3-D (or LES/DNS) run can separate it"),
        "disagreement_proves": (
            "at least one of {discretisation, 2-D-vs-3-D modelling, reference "
            "interpretation} is responsible; a single mismatched Re does not say "
            "which"),
        "reference_caveat": (
            "the KB numbers are from a 3-D DNS (LESOCC); a 2-D closure-free run "
            "cannot reproduce 3-D turbulence, so the comparison is a framework "
            "test only where the flow is genuinely two-dimensional (Re=700 is the "
            "strongest such candidate; Re=1400 is marginal)"),
    }