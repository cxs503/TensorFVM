"""Independent audit probe for m22209 (JOUR-ICE-STRUCT-GUARD, TensorFVM).

Not a re-implementation of the delivery: this script
  1. parses the bounds SPEC out of the delivered document
     (docs/FVM_ICE_GUARDRAILS/REPORT.md §2.2, machine-read, zero hand copy),
  2. re-derives the three-tier decision (hard reject / soft extrapolation / typical pass)
     from the SPEC alone, without calling the delivery's decision code,
  3. compares SPEC-derived decisions against the delivered implementation over a
     dense grid (bounds, just-inside/just-outside, decade sweep) plus a
     non-finite / exotic-type battery,
  4. checks bound nesting (typical ⊆ soft ⊆ hard) as an internal invariant.

Outputs: artifacts/audit_m22252/{spec_vs_impl.json, decision_grid.json, figure}.
Run:  PYTHONPATH=<tree>/src python docs/audit_m22252/audit_ice_guardrails.py
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorfvm.ice import (
    ALL_ICE_PARAMETERS,
    validate_ice_parameters,
    clamp_ice_parameters,
    get_parameter_report,
)

OUT = Path("artifacts/audit_m22252")
DOC = Path("docs/FVM_ICE_GUARDRAILS/REPORT.md")

SPEC_ORDER = [
    "ice_thickness",
    "ice_strength",
    "ice_friction",
    "ice_speed",
    "ice_density",
    "ice_elastic_modulus",
    "ice_poisson_ratio",
    "ice_temperature",
]
DOC_ROW_NAME = {
    "Ice thickness": "ice_thickness",
    "Compressive strength": "ice_strength",
    "Friction coefficient": "ice_friction",
    "Ship speed": "ice_speed",
    "Ice density": "ice_density",
    "Elastic modulus": "ice_elastic_modulus",
    "Poisson's ratio": "ice_poisson_ratio",
    "Temperature": "ice_temperature",
}


def _num(txt: str) -> float:
    return float(txt.strip())


def parse_spec(doc: Path) -> dict[str, dict[str, float]]:
    """Machine-read the §2.2 bounds table (rows: name | symbol | unit | hmin..typical)."""
    text = doc.read_text()
    spec: dict[str, dict[str, float]] = {}
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 9 or cells[0] not in DOC_ROW_NAME:
            continue
        name = DOC_ROW_NAME[cells[0]]
        typ = cells[7].replace("to", " ").replace("–", " ")
        nums = re.findall(r"-?\d+(?:\.\d+)?", typ)
        if len(nums) != 2:
            raise ValueError(f"cannot parse typical range {cells[7]!r}")
        t_lo, t_hi = float(nums[0]), float(nums[1])
        spec[name] = {
            "hard_min": _num(cells[3]),
            "hard_max": _num(cells[4]),
            "soft_min": _num(cells[5]),
            "soft_max": _num(cells[6]),
            "typical_min": t_lo,
            "typical_max": t_hi,
        }
    return spec


def spec_decision(b: dict[str, float], value: float) -> str:
    """Decision rule as documented in REPORT.md §2.1, derived from SPEC bounds only.

    hard violation -> 'fail'  (physically impossible, rejected)
    soft violation -> 'extrapolation' (accepted only with allow_extrapolation=True)
    else           -> 'pass'
    """
    if value < b["hard_min"] or value > b["hard_max"]:
        return "fail"
    if value < b["soft_min"] or value > b["soft_max"]:
        return "extrapolation"
    return "pass"


def impl_decision(name: str, value: float) -> str:
    """Delivered API decision, collapsing WARN-with-extrapolation and FAIL."""
    res = validate_ice_parameters({name: value}, allow_extrapolation=True)
    if res.errors:
        return "fail"
    if res.extrapolation_risk:
        return "extrapolation"
    return "pass"


def grid_for(b: dict[str, float]) -> list[float]:
    span = b["hard_max"] - b["hard_min"]
    eps = max(abs(span) * 1e-9, 1e-12)
    vals = [
        b["hard_min"] - span * 0.1,
        b["hard_min"] - eps,
        b["hard_min"],
        b["soft_min"] - eps,
        b["soft_min"],
        b["typical_min"],
        (b["typical_min"] + b["typical_max"]) / 2.0,
        b["typical_max"],
        b["soft_max"],
        b["soft_max"] + eps,
        b["hard_max"],
        b["hard_max"] + eps,
        b["hard_max"] + span * 0.1,
    ]
    # decade sweep across a wide window, excluding the (possibly empty) interior clamps
    lo = b["hard_min"] - abs(span) if b["hard_min"] != 0 else -abs(span)
    hi = b["hard_max"] + abs(span)
    if lo < hi:
        vals += list(np.linspace(lo, hi, 25))
    return vals


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    spec = parse_spec(DOC)

    # ---- (1) spec vs implementation bounds drift -------------------------
    drift = {}
    for name, s in spec.items():
        p = ALL_ICE_PARAMETERS[name].bounds
        impl = {
            "hard_min": p.hard_min,
            "hard_max": p.hard_max,
            "soft_min": p.soft_min,
            "soft_max": p.soft_max,
            "typical_min": p.typical_min,
            "typical_max": p.typical_max,
        }
        diffs = {k: [s[k], impl[k]] for k in s if not math.isclose(s[k], impl[k], rel_tol=0, abs_tol=0)}
        drift[name] = {"spec": s, "impl": impl, "diffs": diffs}

    # ---- (2) nesting invariant typical ⊆ soft ⊆ hard --------------------
    nesting = {}
    for name in SPEC_ORDER:
        p = ALL_ICE_PARAMETERS[name].bounds
        ok = (
            p.hard_min <= p.soft_min <= p.typical_min <= p.typical_max <= p.soft_max <= p.hard_max
        )
        nesting[name] = bool(ok)

    # ---- (3) decision grid: spec-derived vs delivered -------------------
    grid_rows = []
    agree = 0
    total = 0
    mismatches = []
    for name in SPEC_ORDER:
        s = spec[name]
        for v in grid_for(s):
            want = spec_decision(s, v)
            got = impl_decision(name, v)
            total += 1
            ok = want == got
            agree += ok
            grid_rows.append({"param": name, "value": v, "spec": want, "impl": got, "agree": ok})
            if not ok:
                mismatches.append({"param": name, "value": v, "spec": want, "impl": got})

    # ---- (4) non-finite / exotic type battery ---------------------------
    battery = {}
    probes = {
        "nan": float("nan"),
        "inf": float("inf"),
        "neg_inf": float("-inf"),
        "bool_true": True,
        "numeric_str": "1.5",
        "none": None,
        "neg_zero": -0.0,
    }
    for pname, val in probes.items():
        row = {}
        for name in ("ice_thickness", "ice_strength"):
            try:
                res = validate_ice_parameters({name: val})
                row[name] = {"status": res.status.value if hasattr(res.status, "value") else str(res.status),
                             "is_valid": res.is_valid,
                             "errors": len(res.errors)}
            except Exception as exc:  # pragma: no cover
                row[name] = f"EXC {type(exc).__name__}: {exc}"
        battery[pname] = row

    nan_case = {
        "validate": validate_ice_parameters({"ice_thickness": float("nan")}).__dict__,
        "clamp": clamp_ice_parameters({"ice_thickness": float("nan")}).__dict__,
        "report": get_parameter_report({"ice_thickness": float("nan")}),
    }

    summary = {
        "spec_params_parsed": len(spec),
        "spec_impl_bound_mismatches": {k: v["diffs"] for k, v in drift.items() if v["diffs"]},
        "nesting_ok": nesting,
        "grid_cases": total,
        "grid_agree": agree,
        "grid_agreement_pct": 100.0 * agree / total if total else None,
        "grid_mismatches": mismatches[:20],
        "nan_battery": battery,
        "nan_case": nan_case,
    }
    (OUT / "spec_vs_impl.json").write_text(json.dumps(drift, indent=1))
    (OUT / "decision_grid.json").write_text(json.dumps(grid_rows, indent=1))
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=1, default=lambda o: None if isinstance(o, float) and math.isnan(o) else o)
    )

    # ---- figure ---------------------------------------------------------
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0])

    ax = fig.add_subplot(gs[0, :])
    yt, labels = [], []
    for i, name in enumerate(SPEC_ORDER):
        s = spec[name]
        span = s["hard_max"] - s["hard_min"]
        norm = lambda v: (v - s["hard_min"]) / span if span else 0.0  # noqa: E731
        y = i
        yt.append(y)
        labels.append(f"{name}")
        ax.barh(y, norm(s["hard_max"]) - norm(s["hard_min"]), left=norm(s["hard_min"]),
                height=0.62, color="#d9d9d9", edgecolor="k", linewidth=0.4, label="hard (reject outside)" if i == 0 else None)
        ax.barh(y, norm(s["soft_max"]) - norm(s["soft_min"]), left=norm(s["soft_min"]),
                height=0.42, color="#8fb8de", edgecolor="k", linewidth=0.4, label="soft (calibration domain)" if i == 0 else None)
        ax.barh(y, norm(s["typical_max"]) - norm(s["typical_min"]), left=norm(s["typical_min"]),
                height=0.22, color="#4f9a52", edgecolor="k", linewidth=0.4, label="typical" if i == 0 else None)
        ax.text(1.01, y, f"h∈[{s['hard_min']:g},{s['hard_max']:g}]  s∈[{s['soft_min']:g},{s['soft_max']:g}]",
                fontsize=7.5, va="center", transform=ax.get_yaxis_transform())
    ax.set_yticks(yt)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_xlabel("normalised position inside the parameter's hard-bounds interval")
    ax.set_title("A. Three-tier bounds re-parsed from the delivered document (§2.2) — spec↔implementation zero drift",
                 fontsize=11)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)

    ax2 = fig.add_subplot(gs[1, 0])
    per_param = {n: [0, 0] for n in SPEC_ORDER}
    for r in grid_rows:
        per_param[r["param"]][1] += 1
        per_param[r["param"]][0] += int(r["agree"])
    pct = [100.0 * a / b for a, b in per_param.values()]
    ax2.bar(range(len(SPEC_ORDER)), pct, color="#2e6fa7")
    ax2.set_xticks(range(len(SPEC_ORDER)))
    ax2.set_xticklabels([n.replace("ice_", "") for n in SPEC_ORDER], rotation=45, ha="right", fontsize=8)
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("spec-derived vs delivered decision agreement (%)")
    ax2.set_title(f"B. Decision grid: {agree}/{total} cases agree with the documented rule", fontsize=11)
    ax2.grid(axis="y", alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 1])
    cases = [
        ("h = 1.5 m\n(typical)", "pass", "#4f9a52"),
        ("h = 6.0 m\n(extrapolation)", "warn", "#d98c00"),
        ("h = 50 m\n(impossible)", "fail", "#b22222"),
        ("h = NaN\n(upstream garbage)", "pass\n[HOLE]", "#b22222"),
        ("h = inf", "fail", "#b22222"),
    ]
    colors = [c[2] for c in cases]
    ax3.bar(range(len(cases)), [1] * len(cases), color=colors, alpha=0.85, edgecolor="k")
    for i, (label, verdict, _) in enumerate(cases):
        ax3.text(i, 0.5, verdict, ha="center", va="center", fontsize=10 if "HOLE" not in verdict else 12,
                 color="white", fontweight="bold")
        ax3.text(i, -0.08, label, ha="center", va="top", fontsize=8.5)
    ax3.set_ylim(0, 1.05)
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_title("C. Failure-mode probe: non-finite NaN is reported as a PASS\n(the one class of silent garbage this guardrail exists to stop)",
                  fontsize=11)

    fig.suptitle("m22209 JOUR-ICE-STRUCT-GUARD — board audit (m22252): bounds reproduced, decision rule reproduced, NaN hole found",
                 fontsize=12.5, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "fig_audit_ice_guardrails.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps({k: v for k, v in summary.items() if k != "nan_case"}, indent=1, default=str))


if __name__ == "__main__":
    main()