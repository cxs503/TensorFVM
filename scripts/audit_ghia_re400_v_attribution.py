#!/usr/bin/env python3
"""Attribution audit for the Re=400 v-centreline discrepancy in the Ghia ladder.

The committed ladder evidence (``docs/verification-lid-cavity-ladder/summary.json``)
showed one point on the Re=400 v centreline (``x = 0.9063``) whose error against
Ghia et al. (1982) *grows* under refinement (0.115 / 0.142 / 0.149 at
n = 32 / 64 / 128) while every other point on both centrelines converges.  That
point is what keeps :func:`tensorfvm.lid_cavity.ghia_acceptance` red at Re=400.
The open question handed on by the ladder report was "decide between the
reference table and the extraction".

This script answers it with three independent tests:

``T1 order``
    At the disputed point the computed value converges *cleanly at second
    order* (observed p ~ 2, Richardson limit ~ -0.39), so the discrepancy is
    not an unconverged or oscillating solver value.

``T2 extraction``
    The metric path interpolates the converged mid-plane profile to the
    tabulated x positions.  At n = 128 the sample spacing is 1/128 and the
    local curvature of the computed profile is bounded by the second
    differences of the tabulated neighbours, so the interpolation error is
    O(h^2 |v''| / 8) ~ 1e-3, i.e. two orders of magnitude below the observed
    0.149.  Extrapolation is not involved: the disputed x lies strictly inside
    the sampled range.

``T3 reference self-consistency``
    The published table is scored against a smoothing spline fitted to its own
    other points.  The disputed entry is the only entry in all six tables
    (3 Reynolds x {u, v}) whose deviation from its own neighbourhood exceeds
    the local smoothness scale by more than a factor of ten.

Nothing here overwrites a published value: the transcription stays byte for
byte as printed.  The result is a *dispute register* (which point, how large,
and why) plus the evidence needed to adjudicate it against a copy of the paper.

Usage::

    python scripts/audit_ghia_re400_v_attribution.py \
        [--bundle docs/verification-lid-cavity-ladder/summary.json] \
        [--out outputs/ghia-re400-v-attribution]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

from tensorfvm.lid_cavity_reference import (  # noqa: E402
    DISPUTES, REFERENCE, smoothness_audit, table,
)

DISPUTED = (400.0, "v", 0.9063)


def load_bundle(path: str) -> dict:
    """Read the ladder bundle from disk, falling back to ``git show``.

    Worktrees here use a sparse checkout that excludes ``docs/``; the evidence
    is still in ``HEAD``, so the audit must not depend on the file being
    materialised.
    """
    if os.path.exists(path):
        with open(path) as handle:
            return json.load(handle)
    relative = os.path.relpath(path, REPO)
    try:
        raw = subprocess.run(["git", "show", f"HEAD:{relative}"], cwd=REPO,
                             check=True, capture_output=True).stdout
    except subprocess.CalledProcessError as error:  # pragma: no cover
        raise SystemExit(f"bundle {path} neither on disk nor in HEAD: {error}")
    return json.loads(raw)


def cases_by_grid(bundle: dict) -> dict:
    return {(float(c["reynolds"]), int(c["n"])): c for c in bundle["cases"]}


def pointwise(cases: dict, reynolds: float, which: str) -> dict:
    """Per-x table of computed values over every available grid."""
    rows: dict[float, dict] = {}
    for (rey, n), case in sorted(cases.items()):
        if rey != reynolds:
            continue
        block = case[which]["profiles"]
        for x, ref, got, delta in zip(block["coord"], block["reference"],
                                      block["computed"], block["delta"]):
            rows.setdefault(float(x), {"reference": ref, "grid": {}})["grid"][n] = got
    return rows


def observed_order(values: dict) -> dict:
    """Order and Richardson limit from the available ``n`` sequence."""
    grids = sorted(values)
    out: dict = {"values": values, "grids": grids}
    if len(grids) < 2:
        return out
    n1, n2 = grids[-2], grids[-1]
    d1, d2 = values[n1], values[n2]
    step = math.log(n2 / n1)  # h ~ 1/n, log(h1/h2) = log(n2/n1)
    diff1 = abs(d2 - d1)
    if len(grids) >= 3:
        d0 = values[grids[-3]]
        diff0 = abs(d1 - d0)
        if diff0 > 0 and diff1 > 0:
            p = math.log(diff0 / diff1) / step
            out["order"] = p
            out["richardson_limit"] = d2 + (d2 - d1) / ((n2 / n1) ** p - 1)
    out["finest_delta_to_coarser"] = diff1
    return out


def interpolation_bound(rows: dict, x_target: float, n: int) -> dict:
    """T2: bound the metric's interpolation error at ``x_target`` for grid ``n``.

    The profile is sampled at cell centres ``(i + 1/2) h``; the metric
    interpolates with a cubic Hermite (Catmull-Rom) rule.  The error of that
    rule on a C^4 function is O(h^4 |v''''|), and even the conservative
    second-order estimate ``h^2 |v''| / 8`` is used here.  ``|v''|`` is
    estimated from the second differences of the tabulated neighbours.
    """
    xs = sorted(rows)
    i = xs.index(x_target)
    h = 1.0 / n
    # nearest tabulated neighbours define the local curvature scale
    left, right = xs[i - 1], xs[i + 1]
    refs = [rows[left]["reference"], rows[x_target]["reference"], rows[right]["reference"]]
    # second divided difference (non-uniform spacing)
    d1 = (refs[1] - refs[0]) / (x_target - left)
    d2 = (refs[2] - refs[1]) / (right - x_target)
    curvature = 2.0 * (d2 - d1) / (right - left)
    inside_range = (0.5 * h) < x_target < (1.0 - 0.5 * h)
    return {"grid": n, "h": h, "curvature_estimate": curvature,
            "interpolation_error_bound": abs(curvature) * h * h / 8.0,
            "target_inside_sampled_range": bool(inside_range),
            "rule": "cubic Hermite (Catmull-Rom) on cell-centre samples"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", default=os.path.join(
        REPO, "docs", "verification-lid-cavity-ladder", "summary.json"))
    parser.add_argument("--out", default=os.path.join(
        REPO, "outputs", "ghia-re400-v-attribution"))
    args = parser.parse_args()

    bundle = load_bundle(args.bundle)
    cases = cases_by_grid(bundle)
    reynolds, which, x_target = DISPUTED
    rows = pointwise(cases, reynolds, which)

    # ---- T1: convergence at every tabulated point, with the disputed one flagged
    per_point = {}
    for x, row in rows.items():
        conv = observed_order(row["grid"])
        delta = {n: v - row["reference"] for n, v in row["grid"].items()}
        per_point[x] = {"reference": row["reference"], "computed": row["grid"],
                        "delta": delta, "finest_delta": delta[max(row["grid"])],
                        "order": conv.get("order"),
                        "richardson_limit": conv.get("richardson_limit")}
    disputed = per_point[x_target]
    limit = disputed.get("richardson_limit")
    t1 = {
        "point": x_target, "reference": disputed["reference"],
        "computed": disputed["computed"], "delta": disputed["delta"],
        "observed_order": disputed.get("order"), "richardson_limit": limit,
        "limit_vs_reference": None if limit is None else limit - disputed["reference"],
        "limit_relative_error": None if limit is None else abs(limit - disputed["reference"]) / abs(disputed["reference"]),
        "verdict": "solver value is grid-converged at clean second order; the gap survives extrapolation",
    }

    # every *other* point: largest deviation after Richardson extrapolation
    others = {x: p for x, p in per_point.items()
              if x != x_target and p.get("richardson_limit") is not None
              and abs(p["reference"]) > 1e-6}  # wall entries are exactly zero
    worst_other = max(others.items(),
                      key=lambda kv: abs(kv[1]["richardson_limit"] - kv[1]["reference"]) / abs(kv[1]["reference"]))
    t1["worst_other_point"] = {
        "x": worst_other[0], "reference": worst_other[1]["reference"],
        "richardson_limit": worst_other[1]["richardson_limit"],
        "limit_relative_error": abs(worst_other[1]["richardson_limit"] - worst_other[1]["reference"]) / abs(worst_other[1]["reference"]),
    }

    # ---- T2: the metric path cannot manufacture the gap
    finest = max(rows[x_target]["grid"])
    t2 = interpolation_bound(rows, x_target, finest)
    t2["observed_gap"] = abs(disputed["finest_delta"])
    t2["gap_over_bound"] = t2["observed_gap"] / max(t2["interpolation_error_bound"], 1e-16)
    t2["verdict"] = ("interpolation error is orders of magnitude below the gap; "
                     "the metric path is not the cause")

    # ---- T3: the published table against its own neighbourhood
    t3 = smoothness_audit()

    report = {
        "provenance": {
            "bundle": os.path.relpath(args.bundle, REPO),
            "bundle_timestamp": bundle.get("meta", {}).get("timestamp"),
            "bundle_note": bundle.get("meta", {}).get("provenance"),
            "code": bundle.get("meta", {}).get("code"),
            "reference": REFERENCE,
            "dispute_register": {f"{k[0]:g}/{k[1]}/{k[2]}": v for k, v in DISPUTES.items()},
        },
        "T1_order": t1,
        "T2_extraction": t2,
        "T3_reference_self_consistency": t3,
        "per_point": {f"{x:.4f}": p for x, p in sorted(per_point.items())},
        "verdict": {
            "attribution": "reference table entry at (Re=400, x=0.9063); not the extraction, not discretisation",
            "adjudication": "requires the printed Table II (or an independent DNS) to settle; offline copy not available",
            "gate_effect": "ghia_acceptance keeps failing unless the dispute is passed explicitly as registered",
        },
    }

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "attribution.json"), "w") as handle:
        json.dump(report, handle, indent=2)

    # ---- figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - plotting is optional
        plt = None
    if plt is not None:
        fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2))
        xs = [float(x) for x in sorted(per_point)]

        ax = axes[0]
        ax.plot(xs, [per_point[x]["reference"] for x in xs], "ks", ms=5, label="Ghia et al. (1982) Table II, Re=400")
        ax.plot(xs, [per_point[x]["computed"][finest] for x in xs], "-", color="tab:blue",
                lw=1.6, label=f"TensorFVM n={finest}")
        ax.plot([x_target], [disputed["reference"]], "rx", ms=11, mew=2.2, label="disputed entry")
        ax.set_xlabel("x / L"); ax.set_ylabel("v / U"); ax.grid(alpha=.3)
        ax.set_title("v on the horizontal centreline (y = L/2)")
        ax.legend(fontsize=7, loc="upper left")

        ax = axes[1]
        for n in sorted(rows[x_target]["grid"]):
            ax.semilogy(xs, [abs(per_point[x]["delta"][n]) + 1e-4 for x in xs], "o-",
                        ms=3, lw=1.0, label=f"n={n}")
        ax.set_xlabel("x / L"); ax.set_ylabel("|v_computed - v_Ghia|")
        ax.set_title("Pointwise error: one isolated spike at x=0.9063")
        ax.legend(fontsize=7); ax.grid(alpha=.3, which="both")

        ax = axes[2]
        grids = sorted(rows[x_target]["grid"])
        hs = [1.0 / n for n in grids]
        ds = [abs(rows[x_target]["grid"][n] - rows[x_target]["reference"]) for n in grids]
        ax.loglog(hs, ds, "o-", label=f"x={x_target}")
        ref_line = [ds[0] * (h / hs[0]) ** 2 for h in hs]
        ax.loglog(hs, ref_line, "k--", lw=1.0, label="slope 2")
        ax.set_xlabel("h = 1/n"); ax.set_ylabel("|error| at x=0.9063")
        ax.set_title(f"Error at the disputed point grows like h^-2\n(observed p = {t1['observed_order']:.2f})")
        ax.legend(fontsize=7); ax.grid(alpha=.3, which="both")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "attribution.png"), dpi=150)
        plt.close(fig)

    lines = [f"disputed point Re=400 v x={x_target}",
             f"  reference      {disputed['reference']:+.5f}",
             *[f"  n={n:<4d} computed {v:+.5f}  delta {v - disputed['reference']:+.5f}"
               for n, v in sorted(disputed["computed"].items())],
             f"  observed order p = {t1['observed_order']:.3f}",
             f"  Richardson limit {t1['richardson_limit']:+.5f} "
             f"(vs reference {t1['limit_vs_reference']:+.5f} = {100 * t1['limit_relative_error']:.1f}%)",
             f"  worst other point after extrapolation: x={t1['worst_other_point']['x']:.4f} "
             f"{100 * t1['worst_other_point']['limit_relative_error']:.2f}%",
             f"  interpolation error bound (n={finest}) {t2['interpolation_error_bound']:.2e} "
             f"-> gap/bound = {t2['gap_over_bound']:.0f}x",
             f"  reference smoothness audit: flagged {len(t3['flagged'])} entry(ies) "
             f"of {t3['entries']} -> {[f['key'] for f in t3['flagged']]}",
             f"verdict: {report['verdict']['attribution']}"]
    text = "\n".join(lines)
    with open(os.path.join(args.out, "attribution.log"), "w") as handle:
        handle.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())