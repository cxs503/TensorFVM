"""Case-pool registration of ``CASE-FVM-DAMBREAK-STOKER`` (Ritter/Stoker).

Why this module exists
----------------------
The case had verification evidence (``docs/case-fvm-dambreak-stoker/``) and a
graded driver (``scripts/run_shallow_water_dam_break.py``) but it was **not in
the case pool**: ``docs/verification-catalog.md`` -- the index the catalog gate
(``tensorfvm.verification.catalog_gate``) audits -- mentioned no shallow-water
dam break at all, and there was no ``case_pool_entry.json`` like the
compressible ``CASE-WOODWARD-COLELLA`` registration.  A case that cannot be
found in the pool cannot be reused, and nothing stopped its graded numbers from
drifting away from the artifact.

This module derives the registration **from the committed artifact only**
(``src/tensorfvm/verification/artifacts/shallow-water-dam-break/``): the entry
JSON and the catalog section are rendered here, ``tests/test_case_pool_dambreak_stoker.py``
regenerates and diffs them, so a hand edit or a stale artifact is a test
failure rather than a silent inconsistency.

Ledger contract mapped to graded quantities (tolerance 5 %, ``t* = 2``):

======================  ===============================================  =============
ledger wording           graded quantity                                 verdict key
======================  ===============================================  =============
波前位置 ``x_f(t)``      dry-bed fan front at ``t* = 2``                 ``front_fan<5%``
波前位置 ``x_f(t)``      dry-bed front series (max over samples)         ``front_series<5%``
静水区水位阶跃           wet-bed still-water step ``h2 / h1``            ``step_ratio<5%``
                        wet-bed plateau depth ``h2``                    ``plateau_h2<5%``
波前位置 ``x_f(t)``      wet-bed bore, oracle-free measurement           ``bore_position_modes<5%``
静水区水位阶跃           wet-bed step, oracle-free measurement           ``step_ratio_modes<5%``
波形单调无伪振荡         interior local extrema of ``h(x)``              ``monotone(no_extrema)``
                        total-variation growth                          ``tv_no_growth<0.5%``
======================  ===============================================  =============

``depth_l1`` and the legacy wetted-edge front stay in the entry as secondary
(diagnostic) criteria; they are quoted with their measured margins, not hidden.
"""

from __future__ import annotations

import json
import math
import pathlib
from typing import Any, Dict, List, Optional

__all__ = [
    "SLUG",
    "TOLERANCE",
    "T_STAR",
    "ARTIFACT_DIR",
    "ENTRY_PATH",
    "CATALOG_SECTION_ID",
    "load_artifacts",
    "build_entry",
    "render_entry_json",
    "render_catalog_section",
    "write_entry",
]

SLUG = "CASE-FVM-DAMBREAK-STOKER"
TOLERANCE = 0.05
T_STAR = 2.0
CATALOG_SECTION_ID = "19.1"
CATALOG_SECTION_ID = "19.1"
#: report that substantiates the *scheme* order on smooth data (operator audit)
ORDER_EVIDENCE_REPORT = "docs/case-fvm-dambreak-order-audit/REPORT.md"
ENTRY_PATH = pathlib.Path("docs") / "case-fvm-dambreak-stoker" / "case_pool_entry.json"
ARTIFACT_DIR = (
    pathlib.Path(__file__).resolve().parent
    / "artifacts"
    / "shallow-water-dam-break"
)

#: ledger-facing (primary) verdict keys, with the quantity they grade
PRIMARY_CRITERIA: Dict[str, Dict[str, str]] = {
    "ritter-dry": {
        "front_fan<5%": "wave-front position x_f(t*) on the dry bed vs 2 t sqrt(g h0)",
        "front_series<5%": "wave-front series x_f(t) over the run (max graded error)",
        "monotone(no_extrema)": "waveform monotone: interior local extrema of h(x) above 1e-4 h0",
    },
    "stoker-wet": {
        "shock_position<5%": "bore position x_s(t*) on the wet bed vs the Stoker solution",
        "front_series<5%": "bore series x_s(t) over the run (max graded error)",
        "step_ratio<5%": "still-water level step h2/h1 across the bore",
        "plateau_h2<5%": "still-water plateau depth h2 behind the bore",
        "monotone(no_extrema)": "waveform monotone: interior local extrema of h(x) above 1e-4 h0",
        "bore_position_modes<5%": (
            "bore position x_s(t*) measured from the depth distribution only "
            "(no analytic window/level) vs the Stoker solution"),
        "step_ratio_modes<5%": (
            "still-water level step h2/h1 measured from the depth distribution only "
            "(no analytic window/level) vs the Stoker solution"),
    },
}

#: how each graded verdict is compared, for the JSON criteria table
_COMPARATORS = {
    "monotone(no_extrema)": ("flag", "eq", 0.0),
    "tv_no_growth<0.5%": ("%", "lt", 0.5),
}


def load_artifacts(artifact_dir: Optional[pathlib.Path] = None) -> Dict[str, Any]:
    """Load every committed artifact of the case (all numbers come from here)."""

    root = pathlib.Path(artifact_dir) if artifact_dir is not None else ARTIFACT_DIR
    out: Dict[str, Any] = {}
    for name in (
        "metrics",
        "convergence",
        "sensitivity",
        "exactness",
        "front_definitions",
        "bore_subcell",
    ):
        path = root / f"{name}.json"
        out[name] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    return out


def _rel(path: pathlib.Path) -> str:
    """Path relative to the repository root, POSIX separators."""

    root = pathlib.Path(__file__).resolve().parents[3]
    try:
        return pathlib.Path(path).resolve().relative_to(root).as_posix()
    except ValueError:  # pragma: no cover - artifacts outside the checkout
        return str(path)


def _criteria(case_key: str, verdict: Dict[str, Any]) -> List[Dict[str, Any]]:
    primary = PRIMARY_CRITERIA[case_key]
    out: List[Dict[str, Any]] = []
    for key in verdict:
        if not isinstance(verdict[key], dict):  # ``all_pass`` roll-up flag
            continue
        value = float(verdict[key]["value"])
        limit = float(verdict[key]["limit"])
        unit, comparator, threshold = _COMPARATORS.get(key, ("%", "lt", 100.0 * limit))
        if unit == "%":
            value = 100.0 * value
        is_primary = key in primary
        item: Dict[str, Any] = {
            "criterion": key,
            "quantity": primary.get(
                key,
                {
                    "depth_l1<5%": "depth profile L1 error vs the exact solution",
                    "tv_no_growth<0.5%": "total-variation growth vs the exact solution",
                    "front_wetted<5%": "legacy wetted-edge front (diagnostic only; see docs/case-fvm-dambreak-stoker/AUDIT.md F1)",
                    "plateau_h2<5%": "still-water plateau depth h2 behind the bore",
                }.get(key, key),
            ),
            "value": value,
            "unit": unit,
            "comparator": comparator,
            "threshold": threshold,
            "primary": is_primary,
            "pass": bool(verdict[key]["pass"]),
            "graded_by": "tensorfvm.shallow_water.dam_break._verdict",
        }
        if "samples" in verdict[key]:
            item["series_samples"] = int(verdict[key]["samples"])
        if "samples_graded" in verdict[key]:
            item["series_samples_graded"] = int(verdict[key]["samples_graded"])
        out.append(item)
    return out


def _loglog_slope(points: List[Any]) -> Optional[float]:
    """Least-squares slope of ``log(y)`` against ``log(x)`` (``None`` if undefined).

    Used to *recompute* the observed order of accuracy from the artifact's own
    refinement rows, so the registration carries a number it can be caught on
    rather than a copied digit.
    """

    pts = [(float(x), float(y)) for x, y in points if x and y and x > 0.0 and y > 0.0]
    if len(pts) < 2:
        return None
    n = float(len(pts))
    mx = sum(math.log(x) for x, _ in pts) / n
    my = sum(math.log(y) for _, y in pts) / n
    den = sum((math.log(x) - mx) ** 2 for x, _ in pts)
    if den <= 0.0:  # pragma: no cover - degenerate ladder
        return None
    num = sum((math.log(x) - mx) * (math.log(y) - my) for x, y in pts)
    return num / den


def _order_claim(case_key: str, disc: Dict[str, Any], conv: Dict[str, Any]) -> Dict[str, Any]:
    """``solver.order`` as claimed, next to the order the artifact actually measures.

    The entry used to expose ``order = 2`` alone.  ``2`` is the order of the
    *scheme* (MUSCL/minmod + HLL + SSP-RK3) on smooth data; at the anchored dam
    break the solution is discontinuous, so every committed refinement ladder
    observes ~1.  A reader re-using the pool entry sees only the nominal order
    and cannot tell the two apart, which is precisely the "advertised order vs
    measured accuracy" confusion the order audit
    (``ORDER_EVIDENCE_REPORT``) exists to settle.  Both numbers are derived
    here, from the artifact rows, so the caveat cannot drift away from them.
    """

    rows = conv.get("rows") or []
    observed_l1 = _loglog_slope(
        [(r.get("dx"), r.get("depth_l1")) for r in rows if r.get("depth_l1")]
    )
    # the graded headline quantity differs per case: fan front on the dry bed,
    # bore position on the wet bed -- pick whichever the artifact records.
    headline_key = None
    for candidate in ("front_fan", "shock"):
        if rows and all(candidate in (r.get("headline") or {}) for r in rows):
            headline_key = candidate
            break
    observed_headline = (
        _loglog_slope(
            [(r.get("dx"), (r.get("headline") or {}).get(headline_key)) for r in rows]
        )
        if headline_key
        else None
    )
    nominal = int(disc["order"])
    below = observed_l1 is not None and observed_l1 < nominal - 0.1
    return {
        "nominal": nominal,
        "nominal_kind": "scheme order on smooth data (MUSCL minmod + HLL + SSP-RK3)",
        "grids": [int(r["n"]) for r in rows],
        "observed_depth_l1": observed_l1,
        "observed_depth_l1_declared": conv.get("observed_order_l1"),
        "observed_headline_quantity": headline_key,
        "observed_headline": observed_headline,
        "observed_headline_declared": conv.get("observed_order_headline"),
        "observed_below_nominal": below,
        "evidence_report": ORDER_EVIDENCE_REPORT,
        "scope": (
            "the anchored solution contains a jump (dry front / bore), so the observed "
            "order is owned by the data regularity, not by the operator; do not read "
            "'order = 2' as the accuracy of this case under the 5 % gate"
        ),
    }


def build_entry(artifacts: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Assemble the ``case_pool_entry.json`` payload from the artifact."""

    art = artifacts if artifacts is not None else load_artifacts()
    metrics = art["metrics"]
    convergence = art["convergence"] or {}
    sensitivity = art["sensitivity"] or {}
    exactness = art["exactness"] or {}

    cases: List[Dict[str, Any]] = []
    for key, blob in metrics["cases"].items():
        disc = blob["discretisation"]
        case = blob["case"]
        cases.append(
            {
                "key": key,
                "kind": case["kind"],
                "h0": case["h0"],
                "h1": case["h1"],
                "gravity": case["gravity"],
                "x_min": case.get("x_min"),
                "x_max": case.get("x_max"),
                "x_dam": case.get("x_dam"),
                "dry_depth": disc.get("dry_depth"),
                "t_star": disc["t_star_end"],
                "t_end": disc["t_end"],
                "grid": int(disc["n"]),
                "dx": disc["dx"],
                "cfl": disc["cfl"],
                "order": disc["order"],
                "order_claim": _order_claim(key, disc, convergence.get(key) or {}),
                "steps": int(disc["steps"]),
                "volume_drift": float(disc["volume_final"]) / float(disc["volume_initial"]) - 1.0,
                "reference": case["reference"],
                "criteria": _criteria(key, blob["verdict"]),
                "series": {
                    name: blob["metrics"]["front_series"][name]
                    for name in (
                        "min_cells",
                        "samples",
                        "samples_graded",
                        "max_error",
                        "max_error_t_star",
                        "unresolved_samples",
                        "unresolved_max_error",
                        "legacy_max_error",
                        "legacy_max_error_t_star",
                    )
                },
                "pass": bool(blob["verdict"]["all_pass"])
                and bool(all(c["pass"] for c in _criteria(key, blob["verdict"]))),
            }
        )

    coarse_rows: List[Dict[str, Any]] = []
    for case_key in ("ritter-dry", "stoker-wet"):
        blob = convergence.get(case_key) or {}
        rows = {int(r["n"]): r for r in blob.get("rows", [])}
        if not coarse_rows:
            coarse_rows = [
                {"n": n, "gate_percent": 100.0 * TOLERANCE, "cases": {}} for n in sorted(rows)
            ]
        for row in coarse_rows:
            src = rows.get(row["n"])
            if src is None:
                continue
            row["cases"][case_key] = {
                "front_series_error_percent": 100.0 * float(src["front_series"]),
                "depth_l1_percent": 100.0 * float(src["depth_l1"]),
                "local_extrema": int(src["local_extrema"]),
                "verdict_failing": list(src["verdict_failing"]),
                "pass": bool(src["verdict_all_pass"]),
            }

    honest_fail = [
        {
            "n": row["n"],
            "case": case_key,
            "failing_criteria": data["verdict_failing"],
            "detail": (
                f"front series error {data['front_series_error_percent']:.3f} % "
                f"/ local extrema {data['local_extrema']} at n = {row['n']}"
            ),
        }
        for row in coarse_rows
        for case_key, data in row["cases"].items()
        if not data["pass"]
    ]
    passing_grids = sorted(
        {row["n"] for row in coarse_rows if all(d["pass"] for d in row["cases"].values())}
    )
    pool_grid = max(c["grid"] for c in cases)
    accepted_min = passing_grids[0] if passing_grids else pool_grid

    reference_check: Dict[str, Any] = {}
    if exactness:
        ritter = exactness.get("ritter", {})
        stoker = exactness.get("stoker", {})
        state = exactness.get("stoker_state", {})
        reference_check = {
            "ritter": {
                "h_at_dam_over_h0": ritter.get("h_at_dam_over_h0"),
                "u_at_dam_over_c0": ritter.get("u_at_dam_over_c0"),
                "front_over_c0t": ritter.get("front_over_c0t"),
                "fan_mass_relative_residual": ritter.get("fan_mass_relative"),
            },
            "stoker": {
                "h2": state.get("h2"),
                "u2": state.get("u2"),
                "shock_speed": state.get("shock_speed"),
                "step_ratio": state.get("step_ratio"),
                "riemann_invariant_relative_residual": stoker.get(
                    "riemann_invariant_relative"
                ),
                "mass_jump_relative_residual": stoker.get("mass_jump_relative"),
                "momentum_jump_relative_residual": stoker.get("momentum_jump_relative"),
            },
        }

    return {
        "slug": SLUG,
        "title": "算例池·浅水溃坝 Ritter/Stoker 精确解",
        "ledger_contract": {
            "reference": (
                "Ritter, A. (1892) Z. Ver. Deutsch. Ing. 36:947-954 (dry bed); "
                "Stoker, J.J. (1957) Water Waves, ch. 10 (wet bed / bore)"
            ),
            "tolerance_percent": 100.0 * TOLERANCE,
            "t_star": T_STAR,
            "graded_quantities": [
                "wave-front position x_f(t) and x_f(t*)",
                "still-water level step h2/h1",
                "waveform monotonicity (no spurious oscillation)",
            ],
        },
        "solver": {
            "module": "tensorfvm.shallow_water.dam_break",
            "discretisation": (
                "1-D finite volume, MUSCL (minmod) reconstruction, HLL Riemann solver, "
                "SSP-RK3, CFL 0.4, reflective walls"
            ),
            "order": 2,
            "order_evidence": ORDER_EVIDENCE_REPORT,
            "order_scope": (
                "nominal scheme order; the anchored case is discontinuous and measures ~1 "
                "(see each case's order_claim) -- operator order is substantiated on smooth "
                "data in the evidence report"
            ),
            "cfl": 0.4,
            "driver": "scripts/run_shallow_water_dam_break.py",
        },
        "pool_grid": pool_grid,
        "accepted_grid_min": accepted_min,
        "artifact": {
            name: _rel(ARTIFACT_DIR / f"{name}.json")
            for name in ("metrics", "convergence", "sensitivity", "exactness", "front_definitions", "bore_subcell")
            if (ARTIFACT_DIR / f"{name}.json").exists()
        },
        "reports": [
            "docs/case-fvm-dambreak-stoker/REPORT.md",
            "docs/case-fvm-dambreak-stoker/AUDIT.md",
            "docs/case-fvm-dambreak-stoker-2d/REPORT.md",
            "docs/case-fvm-dambreak-stoker-robustness/REPORT.md",
            ORDER_EVIDENCE_REPORT,
        ],
        "tests": [
            "tests/test_case_pool_dambreak_stoker.py",
            "tests/test_dambreak_stoker_series.py",
            "tests/test_shallow_water_dam_break.py",
        ],
        "reference_check": reference_check,
        "cases": cases,
        "grid_study": {
            "gate_percent": 100.0 * TOLERANCE,
            "rows": coarse_rows,
            "passing_grids": passing_grids,
        },
        "honest_fail": honest_fail,
        "sensitivity": {
            "reference_point": sensitivity.get("reference_point"),
            "worst_headline_error_percent": 100.0 * float(sensitivity["worst_headline_error"])
            if sensitivity.get("worst_headline_error") is not None
            else None,
            "variants": len(sensitivity.get("rows", [])),
        },
        "verdict": {
            "all_primary_pass_at_pool_grid": bool(
                all(c["pass"] for case in cases for c in case["criteria"] if c["primary"])
            ),
            "all_criteria_pass_at_pool_grid": bool(all(case["pass"] for case in cases)),
            "tolerance_percent": 100.0 * TOLERANCE,
            "accepted_grid_min": accepted_min,
            "pool_grid": pool_grid,
            "honest_fail_count": len(honest_fail),
        },
    }


def render_entry_json(entry: Optional[Dict[str, Any]] = None) -> str:
    """Deterministic JSON text of the pool entry (sorted keys, UTF-8, ``\\n``)."""

    payload = entry if entry is not None else build_entry()
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def render_catalog_section(entry: Optional[Dict[str, Any]] = None) -> str:
    """Catalog markdown for ``docs/verification-catalog.md`` (numbers from ``entry``)."""

    data = entry if entry is not None else build_entry()
    art = data["artifact"]
    lines: List[str] = [
        f"### {CATALOG_SECTION_ID} Shallow-Water Dam Break (Ritter 1892 / Stoker 1957)",
        "",
        f"**Ledger slug**: `{data['slug']}` -- 算例池·浅水溃坝 Ritter/Stoker 精确解",
        "",
        "**Reference**: Ritter, A. (1892) *Die Fortpflanzung der Wasserwellen*, "
        "Z. Ver. Deutsch. Ing. 36:947-954 (dry bed); Stoker, J.J. (1957) *Water Waves*, "
        "ch. 10 (wet bed: rarefaction + constant state + bore).",
        "",
        "**Physics**: inviscid depth-averaged shallow-water equations (hydrostatic, no "
        "friction), initial depth jump `h0/h1` at `x = 0`, dam removed at `t = 0`. "
        "Dry bed -> Ritter rarefaction `h = (2c0 - x/t)^2 / 9g`, front `x_f = 2 t sqrt(g h0)`; "
        "wet bed -> Stoker solution with the bore state from the Rankine-Hugoniot jumps.",
        "",
        f"**Implementation**: `src/tensorfvm/shallow_water/swe1d.py`, "
        f"`src/tensorfvm/shallow_water/dam_break.py`, `src/tensorfvm/shallow_water/exact.py`",
        f"**Driver**: `scripts/run_shallow_water_dam_break.py`",
        "**Tests**: " + ", ".join(f"`{t}`" for t in data["tests"]),
        "**Report**: " + ", ".join(f"`{r}`" for r in data["reports"]),
        "**Artifact**: " + ", ".join(f"`{p}`" for p in art.values()),
        f"**Case-pool entry**: `{ENTRY_PATH.as_posix()}`",
        "",
        "**Parameters**:",
        "",
        "| case | h0 [m] | h1 [m] | g [m/s^2] | t* | grid (pool) | dx [m] |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for case in data["cases"]:
        lines.append(
            "| `{key}` | {h0:.3f} | {h1:.3f} | {g:.2f} | {t:.1f} | {n} | {dx:.5f} |".format(
                key=case["key"],
                h0=case["h0"],
                h1=case["h1"],
                g=case["gravity"],
                t=case["t_star"],
                n=case["grid"],
                dx=case["dx"],
            )
        )

    claims = [
        (case["key"], case["order_claim"])
        for case in data["cases"]
        if (case.get("order_claim") or {}).get("observed_depth_l1") is not None
    ]
    if claims:
        lines += [
            "",
            "**Order claim** -- `order = 2` is the *scheme* order, not this case's accuracy: "
            + "; ".join(
                "`{k}` observed depth-L1 order **{l1:.3f}** ({hq} {hh:.3f}) on `n = {g}`".format(
                    k=key,
                    l1=claim["observed_depth_l1"],
                    hq=claim.get("observed_headline_quantity") or "headline",
                    hh=claim["observed_headline"],
                    g="/".join(str(n) for n in claim["grids"]),
                )
                for key, claim in claims
            )
            + ". The anchored ladder contains a jump (dry front / bore), so the observed order is "
            "owned by the data regularity, not by the operator -- the operator itself is "
            "substantiated second order on smooth data in the evidence report "
            f"`{ORDER_EVIDENCE_REPORT}`. Both numbers are recomputed from the convergence "
            "artifact, so the caveat cannot drift away from the ladder it describes.",
        ]

    for case in data["cases"]:
        lines += [
            "",
            f"**{case['key']}** (reference: {case['reference']})",
            "",
            "| criterion | quantity | value | unit | gate | class | verdict |",
            "|---|---|---:|---|---:|---|---|",
        ]
        for crit in case["criteria"]:
            lines.append(
                "| `{name}` | {q} | {val:.4f} | {unit} | {gate:.2f} | {cls} | {v} |".format(
                    name=crit["criterion"],
                    q=crit["quantity"].replace("|", "/"),
                    val=crit["value"],
                    unit=crit["unit"],
                    gate=crit["threshold"],
                    cls="primary" if crit["primary"] else "diagnostic",
                    v="PASS" if crit["pass"] else "FAIL",
                )
            )
        series = case["series"]
        lines.append("")
        lines.append(
            "front series: `{graded}` graded samples of `{total}` (resolution floor "
            "`{floor:.0f}` cells), max error `{err:.4f} %` at `t* = {t:.4f}`; legacy "
            "estimator would have quoted `{legacy:.3f} %`.".format(
                graded=series["samples_graded"],
                total=series["samples"],
                floor=series["min_cells"],
                err=100.0 * series["max_error"],
                t=series["max_error_t_star"],
                legacy=100.0 * series["legacy_max_error"],
            )
        )

    lines += [
        "",
        "**Grid study** (same graded quantities, `n` = 100 ... 1600):",
        "",
        "| n | case | x_f/x_s series error | local extrema | depth L1 | verdict |",
        "|---:|---|---:|---:|---:|---|",
    ]
    for row in data["grid_study"]["rows"]:
        for case_key in ("ritter-dry", "stoker-wet"):
            cell = row["cases"].get(case_key)
            if cell is None:
                continue
            lines.append(
                "| {n} | `{k}` | {ser:.3f} % | {ext} | {l1:.3f} % | {v} |".format(
                    n=row["n"],
                    k=case_key,
                    ser=cell["front_series_error_percent"],
                    ext=cell["local_extrema"],
                    l1=cell["depth_l1_percent"],
                    v="PASS" if cell["pass"] else "**honest FAIL** (" + ", ".join(cell["verdict_failing"]) + ")",
                )
            )

    ref = data["reference_check"]
    verdict = data["verdict"]
    if ref:
        lines += [
            "",
            "**Reference self-consistency** (the analytic solution is itself audited, so a "
            "wrong closed form cannot pass): Ritter `h/h0 = {hr:.4f}` and `u/c0 = {ur:.4f}` at "
            "the dam, `x_f/(c0 t) = {fr:.4f}`, fan-mass residual `{fm:.1e}`; Stoker "
            "`h2 = {h2:.4f}`, `u2 = {u2:.4f}`, bore speed `{s:.4f}`, step `h2/h1 = {sr:.4f}`, "
            "Rankine-Hugoniot residuals mass `{m:.1e}` / momentum `{p:.1e}`.".format(
                hr=ref["ritter"]["h_at_dam_over_h0"],
                ur=ref["ritter"]["u_at_dam_c0"] if "u_at_dam_c0" in ref["ritter"] else ref["ritter"]["u_at_dam_over_c0"],
                fr=ref["ritter"]["front_over_c0t"],
                fm=ref["ritter"]["fan_mass_relative_residual"],
                h2=ref["stoker"]["h2"],
                u2=ref["stoker"]["u2"],
                s=ref["stoker"]["shock_speed"],
                sr=ref["stoker"]["step_ratio"],
                m=ref["stoker"]["mass_jump_relative_residual"],
                p=ref["stoker"]["momentum_jump_relative_residual"],
            ),
        ]

    lines += [
        "",
        f"**Verification Status**: ✅ PASS at `n >= {verdict['accepted_grid_min']}` "
        f"({verdict['tolerance_percent']:.0f} % gate, `t* = 2`, pool grid "
        f"`n = {verdict['pool_grid']}`). Honest FAIL kept on the record: "
        + (
            "; ".join(f"`{f['case']}` n = {f['n']} ({', '.join(f['failing_criteria'])})" for f in data["honest_fail"])
            if data["honest_fail"]
            else "none"
        )
        + ". "
        + _failing_cell_sentence(data["honest_fail"])
        + " Sensitivity sweep "
        f"({data['sensitivity']['variants']} variants, worst headline error "
        f"{data['sensitivity']['worst_headline_error_percent']:.3f} %) and the "
        "sub-cell bore position are in the artifact bundle.",
        "",
    ]
    return "\n".join(lines)


def _failing_cell_sentence(honest_fail: List[Dict[str, Any]]) -> str:
    """Which coarse-grid cells fail, split by criterion class -- derived, never typed.

    The previous wording hard-coded both the cells and one error value ("n = 200/400
    miss the front *series* (6.61 %)"), which survived a change of the series
    instrument and then contradicted the table it sits under.  Every number and every
    cell name here is read from the entry's own ``honest_fail`` list.
    """

    def _label(f):
        return f"`{f['case']}` n = {f['n']} ({', '.join(f['failing_criteria'])})"

    def _is_series(name):
        return "series" in name

    diagnostic = [
        f for f in honest_fail
        if f["failing_criteria"] and all("front_wetted" in c for c in f["failing_criteria"])
    ]
    graded = [f for f in honest_fail if any(_is_series(c) for c in f["failing_criteria"])]
    parts = []
    parts.append(
        "Cells failing only a *diagnostic* criterion (legacy wetted-edge front, which "
        "the entry does not grade): " + ("; ".join(_label(f) for f in diagnostic) if diagnostic else "none") + "."
    )
    parts.append(
        "Cells failing the graded front/bore *series* criterion: "
        + ("; ".join(_label(f) for f in graded) if graded else "none")
        + " -- recorded, not averaged away."
    )
    return " ".join(parts)


def write_entry(path: Optional[pathlib.Path] = None) -> pathlib.Path:
    """Write the pool entry next to its report (idempotent)."""

    target = pathlib.Path(path) if path is not None else pathlib.Path(ENTRY_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_entry_json(), encoding="utf-8")
    return target


def main(argv: Optional[List[str]] = None) -> int:  # pragma: no cover - CLI wrapper
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ENTRY_PATH))
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if the committed entry differs from the artifact-derived one",
    )
    args = parser.parse_args(argv)
    target = pathlib.Path(args.out)
    rendered = render_entry_json()
    if args.check:
        current = target.read_text(encoding="utf-8") if target.exists() else ""
        if current != rendered:
            print(f"stale case-pool entry: {target}")
            return 1
        print(f"case-pool entry up to date: {target}")
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(rendered, encoding="utf-8")
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())