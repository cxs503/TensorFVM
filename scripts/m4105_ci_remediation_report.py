#!/usr/bin/env python3
"""Numbers for the m4105 CI remediation note (no hand-copied digits).

Reads the committed ladder bundle (restored to the published Ghia table) and
emits ``docs/m4105-ci-remediation/metrics_table.md`` plus ``docs/m4105-ci-remediation/metrics.json``:

* per-row u/v max|err| of the solver against the *published* reference,
* the Re = 400 v-centreline debt (unchanged, still registered),
* the three consistency-gate invariants that the round-1 reference edit broke.

Run: ``python3 scripts/m4105_ci_remediation_report.py``.
"""

from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from tensorfvm import lid_cavity_reference as ref  # noqa: E402

BUNDLE = os.path.join(ROOT, "docs", "verification-lid-cavity-ladder", "summary.json")
OUTDIR = os.path.join(ROOT, "docs", "m4105-ci-remediation")
REJECTED_VALUE = -0.38760
KEY = (400.0, "v", 0.9063)


def rows() -> list[dict]:
    with open(BUNDLE) as handle:
        payload = json.load(handle)
    out = []
    for case in payload["cases"]:
        criteria = case["acceptance"]["criteria"]
        out.append(
            {
                "re": case["reynolds"],
                "n": case["n"],
                "steps": case["solver"]["steps"],
                "converged": case["solver"]["converged"],
                "residual_rate": case["solver"]["residual_rate"],
                "max_divergence": case["max_divergence"],
                "u_max_abs_err": criteria["u_max_abs_error"]["value"],
                "v_max_abs_err": criteria["v_max_abs_error"]["value"],
                "pass": case["acceptance"]["pass"],
            }
        )
    return sorted(out, key=lambda r: (r["re"], r["n"]))


def gate_numbers() -> dict:
    with open(BUNDLE) as handle:
        payload = json.load(handle)
    cases = {(c["reynolds"], c["n"]): c for c in payload["cases"]}
    profile = cases[(400.0, 128)]["v"]["profiles"]
    idx = profile["coord"].index(0.9063)
    computed = profile["computed"][idx]
    published = dict(ref.V_CENTRELINE[400.0])[0.9063]
    bound = ref.DISPUTES[KEY]["metric_interpolation_error_bound_n128"]
    gaps = {}
    for y in (0.9453, 0.9063, 0.8594, 0.8047):
        j = profile["coord"].index(y)
        gaps[y] = abs(profile["computed"][j] - dict(ref.V_CENTRELINE[400.0])[y])
    neighbour_max = max(gaps[y] for y in gaps if y != 0.9063)
    audit_published = ref.smoothness_audit()
    original = ref.V_CENTRELINE[400.0]
    ref.V_CENTRELINE[400.0] = tuple(
        (y, REJECTED_VALUE if abs(y - 0.9063) < 1e-9 else v) for y, v in original
    )
    try:
        audit_patched = ref.smoothness_audit()
    finally:
        ref.V_CENTRELINE[400.0] = original
    return {
        "published": published,
        "computed_n128": computed,
        "richardson_limit": ref.DISPUTES[KEY]["richardson_limit"],
        "observed_order": ref.DISPUTES[KEY]["observed_order"],
        "gap_n128": gaps[0.9063],
        "gap_rel_n128": gaps[0.9063] / abs(published),
        "neighbour_max_gap_n128": neighbour_max,
        "gap_over_neighbour": gaps[0.9063] / neighbour_max,
        "metric_bound_n128": bound,
        "gap_over_bound": gaps[0.9063] / bound,
        "audit_published_ok": audit_published["ok"],
        "audit_published_flagged": [i["key"] for i in audit_published["flagged"]],
        "audit_patched_ok": audit_patched["ok"],
        "audit_patched_flagged": [i["key"] for i in audit_patched["flagged"]],
        "audit_patched_registered_flagged": audit_patched["registered_flagged"],
    }


def figure(data: dict, path: str) -> None:
    """u/v max|err| vs n per Reynolds number, with the 3 % acceptance band."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6), sharey=True)
    for ax, which, label in (
        (axes[0], "u", "u-centreline (x = 0.5)"),
        (axes[1], "v", "v-centreline (y = 0.5)"),
    ):
        key = f"{which}_max_abs_err"
        for re in sorted({r["re"] for r in data["rows"]}):
            pts = [(r["n"], r[key]) for r in data["rows"] if r["re"] == re]
            if not pts:
                continue
            ax.plot(
                [p[0] for p in pts], [p[1] for p in pts], "o-", label=f"Re = {re:g}"
            )
        ax.axhline(0.03, ls="--", color="tab:red", lw=1.2, label="3 % acceptance band")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("grid n")
        ax.set_title(label)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("max |computed - Ghia| (as published)")
    fig.suptitle(
        "TensorFVM Ghia ladder, published reference restored (m4105): "
        "u converges, the Re = 400 v row stays outside the band",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> int:
    data = {"rows": rows(), "gate": gate_numbers()}
    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, "metrics.json"), "w") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)

    lines = [
        "| Re | n | steps | conv. | resid. rate | max div | u max err | v max err | 3% gate |",
        "|---:|---:|---:|:---:|---:|---:|---:|---:|:---:|",
    ]
    for r in data["rows"]:
        flag = "PASS" if r["pass"] else "FAIL (v)"
        ok3 = "yes" if r["u_max_abs_err"] < 0.03 and r["v_max_abs_err"] < 0.03 else "no"
        lines.append(
            f"| {r['re']:g} | {r['n']} | {r['steps']} | {'yes' if r['converged'] else 'no'} "
            f"| {r['residual_rate']:.2e} | {r['max_divergence']:.1e} "
            f"| {r['u_max_abs_err']:.4f} | {r['v_max_abs_err']:.4f} | u:{ok3} {flag} |"
        )
    figure(data, os.path.join(OUTDIR, "error_vs_grid.png"))
    table = "\n".join(lines)
    with open(os.path.join(OUTDIR, "metrics_table.md"), "w") as handle:
        handle.write(table + "\n")
    print(table)
    gate = data["gate"]
    print()
    print(
        f"Re=400 v(0.9063): published {gate['published']:.5f}, "
        f"n=128 {gate['computed_n128']:.5f}, Richardson {gate['richardson_limit']:.5f}"
    )
    print(
        f"gap {gate['gap_n128']:.5f} ({gate['gap_rel_n128'] * 100:.1f}%) = "
        f"{gate['gap_over_bound']:.0f}x the metric bound and "
        f"{gate['gap_over_neighbour']:.0f}x the worst neighbour gap"
    )
    print(
        f"audit(published) ok={gate['audit_published_ok']} "
        f"flagged={gate['audit_published_flagged']}"
    )
    print(
        f"audit(patched)   ok={gate['audit_patched_ok']} "
        f"flagged={gate['audit_patched_flagged']} "
        f"registered_flagged={gate['audit_patched_registered_flagged']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
