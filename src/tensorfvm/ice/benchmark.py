"""Benchmark of the ice-domain guardrail: reference vs measured, numerically.

The subject of this benchmark is the **guard**, not the resistance formula.
Each row below states a reference (what a correct guard must do, derived from a
transcribed specification table or from the purpose of the guard) and the value
the implementation actually produces, so the error is arithmetic rather than a
claim.

Rows
----
G1  decision conformance against an independent oracle over a boundary grid
    (spec table transcribed below, oracle rule re-implemented from the
    documented semantics; neither the oracle nor the grid calls the
    implementation's decision code)
G2  ``nan`` containment rate (purpose: 100 % rejected)
G3  ``+-inf`` containment rate and clamped-value finiteness
G4  silent-extrapolation closure on the load path (an out-of-domain input must
    not reach the Lindqvist arithmetic)
G5  in-domain pass-through (guarding an admissible input must not change the
    load: relative change 0 %)
G6  legacy anchor magnitudes: the two failure inputs recorded by the anchor
    (``h = 50 m`` -> 2450 MN-class, ``sigma_c = 100 MPa`` -> 143.9 MN) against
    the in-repo reproduction.  The historical platform build that produced the
    quoted numbers is not in this tree, so this row is reported as WITHHELD
    with the measured deviation, never as a pass.

Run
---
    PYTHONPATH=src python -m tensorfvm.ice --out docs/ICE-STRUCT-GUARD-REPAIR/artifacts
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .guardrails import (
    clamp_ice_parameters,
    get_parameter_report,
    validate_ice_parameters,
)
from .load import (
    IceCondition,
    IceShipGeometry,
    guarded_level_ice_resistance,
    level_ice_resistance_kN,
)
from .parameters import ALL_ICE_PARAMETERS
from ..registry import capability

REPRO_CMD = "PYTHONPATH=src python -m tensorfvm.ice --out docs/ICE-STRUCT-GUARD-REPAIR/artifacts"

# --------------------------------------------------------------------------- #
# Independent oracle: the specification table, transcribed by hand.
# Values are the documented validity domain (docs/ICE-STRUCT-GUARD-REPAIR/
# SPEC.md) in the order (hard_lo, hard_hi, soft_lo, soft_hi, typical_lo,
# typical_hi).  This table is deliberately a second copy: the G1 row measures
# spec<->code drift, so sharing the constants would make the check vacuous.
# --------------------------------------------------------------------------- #
SPEC_BOUNDS: dict[str, tuple[float, float, float, float, float, float]] = {
    "ice_thickness": (0.01, 10.0, 0.05, 5.0, 0.3, 2.5),
    "ice_strength": (0.1, 30.0, 0.3, 15.0, 1.0, 5.0),
    "ice_friction": (0.0, 1.0, 0.01, 0.4, 0.05, 0.20),
    "ice_speed": (0.0, 15.0, 0.1, 8.0, 0.5, 5.0),
    "ice_density": (800.0, 920.0, 850.0, 918.0, 880.0, 910.0),
    "ice_elastic_modulus": (0.1, 20.0, 1.0, 12.0, 3.0, 8.0),
    "ice_poisson_ratio": (0.0, 0.5, 0.25, 0.40, 0.30, 0.35),
    "ice_temperature": (-60.0, 0.0, -45.0, -1.0, -30.0, -5.0),
    "ship_length": (10.0, 500.0, 30.0, 400.0, 50.0, 350.0),
    "ship_beam": (2.0, 80.0, 8.0, 60.0, 15.0, 50.0),
    "ship_draft": (0.3, 20.0, 2.0, 16.0, 5.0, 14.0),
    "ship_block_coefficient": (0.30, 0.90, 0.45, 0.88, 0.55, 0.85),
}

#: Nominal in-domain operating points (published trial vessels: Riska et al.
#: 1997 for MV Arctic; Kuehnlein 1997 for Polarstern; Spencer & Jones 2001 for
#: Terry Fox).  Used only as admissible inputs, not as accuracy references.
NOMINAL_CASES: dict[str, dict[str, float]] = {
    "MV_Arctic": {
        "ice_thickness": 1.0, "ice_strength": 1.5, "ice_friction": 0.15,
        "ice_speed": 3.0, "ice_density": 900.0, "ice_elastic_modulus": 5.0,
        "ice_poisson_ratio": 0.33, "ice_temperature": -10.0,
        "ship_length": 222.5, "ship_beam": 32.2, "ship_draft": 13.5,
        "ship_block_coefficient": 0.82,
    },
    "Polarstern": {
        "ice_thickness": 0.8, "ice_strength": 1.2, "ice_friction": 0.12,
        "ice_speed": 2.5, "ice_density": 900.0, "ice_elastic_modulus": 5.0,
        "ice_poisson_ratio": 0.33, "ice_temperature": -12.0,
        "ship_length": 118.0, "ship_beam": 25.0, "ship_draft": 10.5,
        "ship_block_coefficient": 0.68,
    },
    "Terry_Fox": {
        "ice_thickness": 0.6, "ice_strength": 1.0, "ice_friction": 0.10,
        "ice_speed": 2.0, "ice_density": 900.0, "ice_elastic_modulus": 5.0,
        "ice_poisson_ratio": 0.33, "ice_temperature": -8.0,
        "ship_length": 75.0, "ship_beam": 18.0, "ship_draft": 7.5,
        "ship_block_coefficient": 0.60,
    },
}

#: The two out-of-domain inputs recorded by the anchor, with the magnitudes the
#: platform reported for them at the time of the incident.
LEGACY_ANCHOR_CASES = {
    "h_50m": {
        "overrides": {"ice_thickness": 50.0},
        "ref": {"resistance_MN": 2450.0,
                "source": "JOUR-ICE-STRUCT-GUARD anchor record (platform build of 2026-11-09)"},
    },
    "sigma_c_100MPa": {
        "overrides": {"ice_strength": 100.0},
        "ref": {"resistance_MN": 143.9,
                "source": "JOUR-ICE-STRUCT-GUARD anchor record (platform build of 2026-11-09)"},
    },
}

TOLERANCE = 0.03          # the platform-wide 3 % acceptance band
LOAD_TOLERANCE = 1e-12    # pass-through must be bit-comparable, not merely 3 %


# --------------------------------------------------------------------------- #
# Independent oracle rule
# --------------------------------------------------------------------------- #
def oracle_decision(name: str, value: float,
                    allow_extrapolation: bool = False) -> str:
    """Label a value ``pass`` / ``warn`` / ``fail`` from the spec table only.

    Rule (docs/ICE-STRUCT-GUARD-REPAIR/SPEC.md 2.1):
      hard violation            -> fail
      soft violation, allow=True-> warn
      soft violation, allow=False-> fail
      otherwise                 -> pass
      non-finite               -> fail
    """
    if name not in SPEC_BOUNDS:
        return "fail"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "fail"
    v = float(value)
    if not math.isfinite(v):
        return "fail"
    hard_lo, hard_hi, soft_lo, soft_hi, _, _ = SPEC_BOUNDS[name]
    if v < hard_lo or v > hard_hi:
        return "fail"
    if v < soft_lo or v > soft_hi:
        return "warn" if allow_extrapolation else "fail"
    return "pass"


def _boundary_grid() -> list[tuple[str, float]]:
    """Boundary-dense probe set: both sides of every bound plus the interior."""
    pts: list[tuple[str, float]] = []
    for name, (hlo, hhi, slo, shi, tlo, thi) in SPEC_BOUNDS.items():
        span = hhi - hlo
        eps = max(span * 1e-6, 1e-12)
        probes = [
            hlo - eps, hlo, hlo + eps, slo - eps, slo, slo + eps,
            tlo, (tlo + thi) / 2.0, thi, shi - eps, shi, shi + eps,
            hhi - eps, hhi, hhi + eps,
        ]
        pts.extend((name, float(v)) for v in probes)
    return pts


def _impl_label(name: str, value: float, allow_extrapolation: bool = False) -> str:
    res = validate_ice_parameters({name: value},
                                 allow_extrapolation=allow_extrapolation)
    return res.status.value


# --------------------------------------------------------------------------- #
# Benchmark rows
# --------------------------------------------------------------------------- #
@dataclass
class Row:
    metric: str
    ref: Any
    got: Any
    unit: str
    error: Optional[float]
    tolerance: Optional[float]
    passed: Optional[bool]
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric, "ref": self.ref, "got": self.got,
            "unit": self.unit, "error": self.error, "tolerance": self.tolerance,
            "passed": self.passed, "note": self.note,
        }


def row_g1_decision_grid() -> tuple[Row, list[dict[str, Any]]]:
    grid = _boundary_grid()
    detail = []
    mismatch = 0
    for name, value in grid:
        ref = oracle_decision(name, value)
        got = _impl_label(name, value)
        if ref != got:
            mismatch += 1
        detail.append({"param": name, "value": value, "ref": ref, "got": got,
                       "agree": ref == got})
    err = mismatch / len(grid)
    return Row(
        metric="G1 decision conformance vs independent spec oracle",
        ref=f"oracle over {len(grid)} boundary probes",
        got=f"{len(grid) - mismatch}/{len(grid)} agree",
        unit="fraction", error=err, tolerance=TOLERANCE,
        passed=err <= TOLERANCE,
        note="2 x 15 probes per parameter (both sides of hard/soft bounds, "
             "typical band, interior)",
    ), detail


def row_g2_nan() -> Row:
    names = list(SPEC_BOUNDS)
    rejected = 0
    leaked: list[str] = []
    for name in names:
        res = validate_ice_parameters({name: float("nan")})
        if (not res.is_valid) and (name not in res.validated_params):
            rejected += 1
        else:
            leaked.append(name)
        clamped = clamp_ice_parameters({name: float("nan")})
        if name in clamped.clamped_params and not math.isfinite(
                clamped.clamped_params[name]):
            leaked.append(f"clamp:{name}")
    rate = rejected / len(names)
    return Row(
        metric="G2 nan containment (rejected and never echoed back)",
        ref=1.0, got=rate, unit="fraction", error=1.0 - rate,
        tolerance=0.0, passed=rate == 1.0,
        note=("leaked: " + ",".join(leaked)) if leaked else "no NaN leaked",
    )


def row_g3_inf() -> Row:
    names = list(SPEC_BOUNDS)
    ok = 0
    problems: list[str] = []
    for name in names:
        for value in (float("inf"), float("-inf")):
            res = validate_ice_parameters({name: value})
            good = (not res.is_valid) and name not in res.validated_params
            clamped = clamp_ice_parameters({name: value})
            cval = clamped.clamped_params.get(name)
            good = good and cval is not None and math.isfinite(cval)
            if good:
                ok += 1
            else:
                problems.append(f"{name}@{value:+.0f}")
    total = 2 * len(names)
    rate = ok / total
    return Row(
        metric="G3 +-inf containment (rejected; clamp returns a finite bound)",
        ref=1.0, got=rate, unit="fraction", error=1.0 - rate,
        tolerance=0.0, passed=rate == 1.0,
        note=("problems: " + ",".join(problems)) if problems else "all contained",
    )


def _out_of_domain_probes() -> list[tuple[str, float]]:
    probes = []
    for name, (hlo, hhi, *_rest) in SPEC_BOUNDS.items():
        span = hhi - hlo
        probes.append((name, hhi + span))       # one span beyond the hard bound
        probes.append((name, hlo - span))
    return probes


@capability('fvm.ice.load_path_closure',
            kind='ice',
            maturity='developing',
            test_path='tests/test_ice_guard_pipeline.py',
            wired=True,
            tested=True,
            verified=False,
            evidence={'grade': 'MECHANICS', 'docs': 0})
def row_g4_load_path_closure() -> tuple[Row, list[dict[str, Any]]]:
    base = dict(NOMINAL_CASES["MV_Arctic"])
    detail = []
    blocked = 0
    probes = _out_of_domain_probes()
    for name, value in probes:
        params = dict(base)
        params[name] = value
        res = guarded_level_ice_resistance(params, mode="reject")
        stopped = (res.resistance_kN is None)
        if stopped:
            blocked += 1
        detail.append({
            "param": name, "injected": value,
            "guarded_resistance_kN": res.resistance_kN,
            "unguarded_resistance_kN": res.unguarded_resistance_kN,
            "blocked": stopped,
            "errors": res.guard.errors[:2],
        })
    rate = blocked / len(probes)
    return Row(
        metric="G4 out-of-domain inputs blocked before the load arithmetic",
        ref=1.0, got=rate, unit="fraction", error=1.0 - rate,
        tolerance=0.0, passed=rate == 1.0,
        note=f"{len(probes)} injections over the 12-parameter set, mode=reject",
    ), detail


def row_g5_passthrough() -> Row:
    worst = 0.0
    detail = {}
    for case, params in NOMINAL_CASES.items():
        guarded = guarded_level_ice_resistance(params, mode="reject")
        ice = IceCondition(params["ice_thickness"], params["ice_strength"],
                           params["ice_friction"], params["ice_speed"])
        ship = IceShipGeometry(params["ship_length"], params["ship_beam"],
                               params["ship_draft"], params["ship_block_coefficient"])
        raw = level_ice_resistance_kN(ship, ice)
        rel = (abs(guarded.resistance_kN - raw) / raw
               if guarded.resistance_kN is not None else float("inf"))
        worst = max(worst, rel)
        detail[case] = {"raw_kN": raw, "guarded_kN": guarded.resistance_kN,
                        "rel_diff": rel}
    return Row(
        metric="G5 in-domain pass-through (guard must not perturb the load)",
        ref="guarded == unguarded", got=f"max rel diff {worst:.3e}",
        unit="relative", error=worst, tolerance=LOAD_TOLERANCE,
        passed=worst <= LOAD_TOLERANCE,
        note=json.dumps(detail, indent=1),
    )


def row_g6_legacy() -> tuple[Row, dict[str, Any]]:
    detail: dict[str, Any] = {}
    max_err = 0.0
    for case, spec in LEGACY_ANCHOR_CASES.items():
        params = dict(NOMINAL_CASES["MV_Arctic"])
        params.update(spec["overrides"])
        res = guarded_level_ice_resistance(params, mode="reject")
        got_mn = (res.unguarded_resistance_kN / 1000.0
                  if res.unguarded_resistance_kN is not None else None)
        ref_mn = spec["ref"]["resistance_MN"]
        err = abs(got_mn - ref_mn) / ref_mn if got_mn else None
        if err is not None:
            max_err = max(max_err, err)
        detail[case] = {
            "ref_MN": ref_mn, "ref_source": spec["ref"]["source"],
            "got_MN_in_repo_reproduction": got_mn,
            "rel_error": err,
            "guarded_resistance_kN": res.resistance_kN,
            "blocked": res.resistance_kN is None,
            "guard_errors": res.guard.errors,
        }
    return Row(
        metric="G6 legacy anchor magnitudes (out-of-domain load reproduction)",
        ref="2450 MN (h=50 m) / 143.9 MN (sigma_c=100 MPa) as recorded 2026-11-09",
        got=f"max rel deviation {max_err:.3f}",
        unit="relative", error=max_err, tolerance=TOLERANCE,
        passed=None,   # WITHHELD: the historical build is not in this tree
        note="WITHHELD (not a pass): the quoted magnitudes come from a platform "
             "build that is not part of this repository; the in-repo "
             "reproduction of the same Lindqvist formula differs, so provenance "
             "cannot be closed here. What *is* measured is that both inputs are "
             "rejected and that the unguarded magnitudes are 10^3 MN-class — "
             "i.e. the failure mode is real.",
    ), detail


def run_benchmark() -> dict[str, Any]:
    """Execute every row and return the machine-readable benchmark bundle."""
    g1, g1_detail = row_g1_decision_grid()
    g2 = row_g2_nan()
    g3 = row_g3_inf()
    g4, g4_detail = row_g4_load_path_closure()
    g5 = row_g5_passthrough()
    g6, g6_detail = row_g6_legacy()

    rows = [g1, g2, g3, g4, g5, g6]
    graded = [r for r in rows if r.passed is not None]
    all_pass = all(r.passed for r in graded)

    h_sweep = []
    for h in [0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 20.0, 50.0]:
        params = dict(NOMINAL_CASES["MV_Arctic"])
        params["ice_thickness"] = h
        res = guarded_level_ice_resistance(params, mode="reject")
        h_sweep.append({
            "ice_thickness_m": h,
            "unguarded_kN": res.unguarded_resistance_kN,
            "guarded_kN": res.resistance_kN,
            "in_domain": res.resistance_kN is not None,
        })

    return {
        "schema": "tensorfvm.ice-guard-benchmark/2",
        "subject": "ice-domain guardrail (JOUR-ICE-STRUCT-GUARD repair, m22342)",
        "tolerance": TOLERANCE,
        "repro_cmd": REPRO_CMD,
        "rows": [r.as_dict() for r in rows],
        "all_graded_rows_pass": all_pass,
        "detail": {
            "g1_decision_grid": g1_detail,
            "g4_load_path": g4_detail,
            "g6_legacy": g6_detail,
            "thickness_sweep": h_sweep,
        },
        "parameter_table": {
            name: {"hard": [b.hard_min, b.hard_max],
                   "soft": [b.soft_min, b.soft_max],
                   "typical": [b.typical_min, b.typical_max],
                   "unit": b.unit,
                   "reference": b.reference}
            for name, b in
            ((n, p.bounds) for n, p in ALL_ICE_PARAMETERS.items())
        },
    }


# --------------------------------------------------------------------------- #
# Artifacts
# --------------------------------------------------------------------------- #
def _write_figure(bundle: dict[str, Any], path: Path) -> bool:
    """Three-panel figure; returns False when matplotlib is unavailable."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:       # pragma: no cover - optional dependency
        return False

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))
    names = list(SPEC_BOUNDS)
    # Panel A: normalised three-tier bands, spec (line) vs implementation (band)
    ax = axes[0]
    for i, name in enumerate(names):
        hlo, hhi, slo, shi, tlo, thi = SPEC_BOUNDS[name]
        span = hhi - hlo
        norm = lambda v: (v - hlo) / span  # noqa: E731
        ax.barh(i, norm(shi) - norm(slo), left=norm(slo), height=0.5,
                color="#cfe3ff", edgecolor="#5b8fd6")
        ax.plot([norm(tlo), norm(thi)], [i, i], color="#1f3f70", lw=4)
        ax.plot([0, 1], [i, i], color="#9aa7b4", lw=0.8, alpha=0.6)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("position inside hard bounds (0 = hard_min, 1 = hard_max)")
    ax.set_title("A. Validity domain per parameter\n(band = calibration, bar = typical)",
                 fontsize=10)

    # Panel B: containment rates, reference vs measured
    ax = axes[1]
    rows = {r["metric"].split()[0]: r for r in bundle["rows"]}
    labels = ["G2 nan", "G3 inf", "G4 load-path"]
    measured = [rows["G2"]["got"], rows["G3"]["got"], rows["G4"]["got"]]
    ref = [rows["G2"]["ref"], rows["G3"]["ref"], rows["G4"]["ref"]]
    x = range(len(labels))
    ax.bar([i - 0.2 for i in x], ref, width=0.4, label="reference (100 %)",
           color="#7fb069")
    ax.bar([i + 0.2 for i in x], measured, width=0.4, label="measured",
           color="#2f6690")
    for i, m in enumerate(measured):
        ax.text(i + 0.2, m + 0.02, f"{m * 100:.1f}%", ha="center", fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("containment / closure rate")
    ax.set_title("B. Non-finite and out-of-domain containment\n"
                 "(m22209 delivery: nan rate was 0 %)", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")

    # Panel C: unguarded thickness sweep with the validity domain shaded
    ax = axes[2]
    sweep = bundle["detail"]["thickness_sweep"]
    hs = [p["ice_thickness_m"] for p in sweep]
    raw = [p["unguarded_kN"] / 1000.0 for p in sweep]   # MN
    ax.plot(hs, raw, "o-", color="#b03a2e", label="unguarded R(h) [MN]")
    ax.axvspan(SPEC_BOUNDS["ice_thickness"][0], SPEC_BOUNDS["ice_thickness"][4],
               color="#d9ead3", alpha=0.7, label="calibration domain")
    ax.axvline(SPEC_BOUNDS["ice_thickness"][1], color="#444", ls="--", lw=1,
               label="hard bound (10 m)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("level-ice thickness h [m]")
    ax.set_ylabel("Lindqvist resistance [MN]")
    ax.set_title("C. Why the guard matters: load vs h\n"
                 "(h = 50 m -> 10^3 MN, silently)", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3, which="both")

    fig.suptitle("Ice-domain guardrail benchmark — JOUR-ICE-STRUCT-GUARD repair "
                 "(m22342)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def write_artifacts(out_dir: str | Path = "docs/ICE-STRUCT-GUARD-REPAIR/artifacts") -> dict[str, Any]:
    """Run the benchmark and persist the evidence bundle under ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    bundle = run_benchmark()
    (out / "benchmark.json").write_text(json.dumps(bundle, indent=2) + "\n")
    (out / "decision_grid.json").write_text(
        json.dumps(bundle["detail"]["g1_decision_grid"], indent=2) + "\n")
    (out / "load_path_closure.json").write_text(
        json.dumps(bundle["detail"]["g4_load_path"], indent=2) + "\n")
    (out / "legacy_anchor.json").write_text(
        json.dumps(bundle["detail"]["g6_legacy"], indent=2) + "\n")

    report = get_parameter_report(dict(NOMINAL_CASES["MV_Arctic"]))
    (out / "parameter_report.txt").write_text(report + "\n")

    fig_ok = _write_figure(bundle, out / "fig_ice_guard_benchmark.png")

    summary = {
        "schema": "tensorfvm.ice-guard-summary/2",
        "all_graded_rows_pass": bundle["all_graded_rows_pass"],
        "rows": [{"metric": r["metric"], "ref": r["ref"], "got": r["got"],
                  "error": r["error"], "tolerance": r["tolerance"],
                  "passed": r["passed"]} for r in bundle["rows"]],
        "repro_cmd": REPRO_CMD,
        "artifacts": sorted(p.name for p in out.iterdir() if p.is_file()),
        "figure_written": fig_ok,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


__all__ = [
    "LEGACY_ANCHOR_CASES",
    "NOMINAL_CASES",
    "REPRO_CMD",
    "SPEC_BOUNDS",
    "oracle_decision",
    "run_benchmark",
    "write_artifacts",
]
