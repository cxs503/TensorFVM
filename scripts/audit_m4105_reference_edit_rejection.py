#!/usr/bin/env python3
"""Negative control for the m3240/m3884 "Ghia transcription correction" (m4105).

Round 1 of PR #126 resolved a merge conflict by adopting
``V_CENTRELINE[400.0][0.9063] = -0.38760`` -- the solver's Richardson limit --
in place of the published ``-0.23827``.  That edit contradicts the policy stated
in :mod:`tensorfvm.lid_cavity_reference` ("a dispute is *not* a licence to edit
the transcription") and it makes ``tests/test_ghia_reference_consistency.py``
fail.  This audit quantifies both sides so the decision is reproducible instead
of argued:

1. **positive control** -- with the published table the smoothness audit flags
   exactly one corner in all six tables, the registered dispute;
2. **negative control** -- with the rejected edit applied *in memory only* the
   registered gap collapses (0.1493 -> 1.8e-5) and the register's fingerprint
   disappears from the audit (``flagged`` empties, ``registered_flagged``
   empties), so the gate fails: the metric would be scoring the solver against
   its own Richardson limit, i.e. verification becomes circular;
3. **convergence** -- the solver values at n = 32/64/128 extrapolate to the
   registered Richardson limit, so the *dispute* is real and stays registered.

Writes ``docs/m4105-ci-remediation/reference_edit_audit.json`` and a two-panel figure.
Run: ``python3 scripts/audit_m4105_reference_edit_rejection.py``.
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

REJECTED_VALUE = -0.38760  # the round-1 edit, kept here only as a counterfactual
KEY = (400.0, "v", 0.9063)


def load_cases() -> dict[tuple[float, int], dict]:
    with open(BUNDLE) as handle:
        payload = json.load(handle)
    return {(case["reynolds"], case["n"]): case for case in payload["cases"]}


def published_audit() -> dict:
    return ref.smoothness_audit()


def patched_audit() -> dict:
    """Run the same audit with the rejected edit applied to an in-memory copy."""

    original = ref.V_CENTRELINE[400.0]
    patched = tuple(
        (y, REJECTED_VALUE if abs(y - 0.9063) < 1e-9 else v) for y, v in original
    )
    ref.V_CENTRELINE[400.0] = patched
    try:
        return ref.smoothness_audit()
    finally:
        ref.V_CENTRELINE[400.0] = original


def gap_at_dispute(cases: dict[tuple[float, int], dict]) -> dict:
    """Metric-path interpolation error bound and the gap, from committed evidence."""
    published = dict(ref.V_CENTRELINE[400.0])
    target = published[0.9063]
    rows = {}
    for n in (32, 64, 128):
        case = cases.get((400.0, n))
        if case is None:
            continue
        profile = case["v"]["profiles"]
        index = profile["coord"].index(0.9063)
        computed = profile["computed"][index]
        rows[n] = {
            "computed": computed,
            "gap": abs(computed - target),
            "rel_gap": abs(computed - target) / abs(target),
        }
    # interpolation error bound quoted in the register (n = 128 metric path)
    bound = ref.DISPUTES[KEY]["metric_interpolation_error_bound_n128"]
    fine = rows.get(128, {})
    return {
        "published": target,
        "rejected_proposal": REJECTED_VALUE,
        "computed": rows,
        "metric_interpolation_error_bound_n128": bound,
        "gap_over_bound_n128": (fine.get("gap", float("nan")) / bound)
        if bound
        else None,
        "richardson_limit": ref.DISPUTES[KEY]["richardson_limit"],
        "observed_order": ref.DISPUTES[KEY]["observed_order"],
        "proposal_gap_to_published": abs(REJECTED_VALUE - target),
        "proposal_gap_to_limit": abs(
            REJECTED_VALUE - ref.DISPUTES[KEY]["richardson_limit"]
        ),
    }


def neighbour_gaps(cases: dict[tuple[float, int], dict]) -> dict:
    """Slope-error proxy at the disputed ordinate and its neighbours (n = 128)."""
    case = cases[(400.0, 128)]
    profile = case["v"]["profiles"]
    coord, computed = profile["coord"], profile["computed"]
    table = dict(ref.V_CENTRELINE[400.0])
    out = {}
    for y in (0.9453, 0.9063, 0.8594, 0.8047):
        if y not in table:
            continue
        index = coord.index(y)
        out[y] = abs(computed[index] - table[y])
    return out


def figure(report: dict, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = load_cases()
    case = cases[(400.0, 64)]
    profile = case["v"]["profiles"]
    table = ref.V_CENTRELINE[400.0]
    coords = [y for y, _ in table]

    fig, (left, right) = plt.subplots(1, 2, figsize=(12.4, 5.0))

    left.plot(
        profile["computed"],
        profile["coord"],
        "k-",
        lw=1.8,
        label="TensorFVM n=64 (committed ladder)",
    )
    left.plot(
        [v for _, v in table],
        coords,
        "o",
        ms=4.0,
        mfc="none",
        color="tab:blue",
        label="Ghia (1982) Table II, as published",
    )
    left.plot(
        [REJECTED_VALUE],
        [0.9063],
        "x",
        ms=9,
        mew=2,
        color="tab:red",
        label="round-1 edit -0.38760 (rejected)",
    )
    left.set_xlabel("v (wall-normal velocity), y = 0.5")
    left.set_ylabel("x")
    left.set_title("Re=400 v-centreline: published point vs rejected edit")
    left.grid(alpha=0.3)
    left.legend(loc="lower right", fontsize=8)

    computed = report["dispute"]["computed"]
    ns = sorted(int(n) for n in computed)
    deltas = [
        computed[n]["computed"] if n in computed else computed[str(n)]["computed"]
        for n in ns
    ]
    inv_h2 = [1.0 / (n * n) for n in ns]
    right.plot(inv_h2, deltas, "o-", color="tab:green", label="solver v(x=0.9063)")
    for n, d in zip(ns, deltas):
        right.annotate(
            f"n={n}",
            (1.0 / (n * n), d),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )
    right.axhline(
        report["dispute"]["published"],
        ls="--",
        color="tab:blue",
        label=f"published {report['dispute']['published']:.5f}",
    )
    right.axhline(
        report["dispute"]["richardson_limit"],
        ls=":",
        color="tab:purple",
        label=f"Richardson limit {report['dispute']['richardson_limit']:.5f}",
    )
    right.axhline(
        REJECTED_VALUE,
        ls="-.",
        color="tab:red",
        lw=1.0,
        label=f"rejected edit {REJECTED_VALUE:.5f}",
    )
    right.set_xlabel(r"$1/n^2$")
    right.set_ylabel("v at x = 0.9063")
    right.set_title("Second-order convergence to the registered limit")
    right.grid(alpha=0.3)
    right.legend(fontsize=8)

    fig.suptitle(
        "m4105: the Re=400 v-centreline dispute is real; the published number is not editable",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> int:
    cases = load_cases()
    report = {
        "published_table_audit": published_audit(),
        "patched_table_audit": patched_audit(),
        "dispute": gap_at_dispute(cases),
        "neighbour_gaps_n128": neighbour_gaps(cases),
        "bundle": os.path.relpath(BUNDLE, ROOT),
    }
    os.makedirs(OUTDIR, exist_ok=True)
    json_path = os.path.join(OUTDIR, "reference_edit_audit.json")
    with open(json_path, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    png_path = os.path.join(OUTDIR, "reference_edit_audit.png")
    figure(report, png_path)

    published = report["published_table_audit"]
    patched = report["patched_table_audit"]
    dispute = report["dispute"]
    print(
        f"published table: ok={published['ok']} flagged="
        f"{[item['key'] for item in published['flagged']]}"
    )
    print(
        f"patched table  : ok={patched['ok']} flagged="
        f"{[item['key'] for item in patched['flagged']]}"
    )
    print(
        f"published {dispute['published']:.5f}  limit {dispute['richardson_limit']:.5f}  "
        f"n128 computed {dispute['computed'][128]['computed']:.5f}"
    )
    print(f"gap/bound at n=128: {dispute['gap_over_bound_n128']:.1f}")
    print(f"neighbour gaps n=128: {report['neighbour_gaps_n128']}")
    print(f"wrote {json_path} and {png_path}")
    # the audit must show the published table clean and the edit not clean
    clean = published["ok"] and any(
        i["key"] == "400/v/0.9063" for i in published["flagged"]
    )
    edit_not_clean = not patched["ok"]
    return 0 if (clean and edit_not_clean) else 1


if __name__ == "__main__":
    raise SystemExit(main())
