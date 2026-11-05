"""Traceability audit for the numbers quoted in ``docs/**/report.md``.

For every verification report this script answers three questions and archives
the answers as machine-readable evidence:

1. **Locally traced** -- is each quoted measured number reproduced by a numeric
   artifact committed *next to* the report (``docs/<case>/**`` JSON/``.dat``/
   ``.csv``)?  That is the only provenance tight enough to gate on, so it is also
   written to ``claims.json`` and re-verified by
   ``tests/test_report_traceability.py``.
2. **Tree traced** -- failing (1), does the number appear anywhere in the
   committed ``docs`` evidence tree?  Weaker provenance (a collision with an
   unrelated array is possible), but it distinguishes "archived elsewhere" from
   "nowhere".
3. **Reproduce pointer** -- does every ``scripts/*.py`` quoted by the report
   still exist?  A dead pointer makes the claim unreproducible even when the
   number is right.

Usage::

    python3 scripts/audit_report_traceability.py            # write artifacts
    python3 scripts/audit_report_traceability.py --check    # audit only
"""

from __future__ import annotations

import argparse
import bisect
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorfvm.verification.doc_claims import (  # noqa: E402
    audit_traceability,
    build_registry,
    evidence_artifacts,
    printed_band,
    scan_report,
    summarize,
)

OUT = ROOT / "docs" / "verification-report-traceability"
NUMERIC = (".json", ".dat", ".csv", ".txt")
EXCLUDE = {"config.json", "environment.json", "claims.json", "traceability.json"}


def evidence_pool(root):
    pool = []
    for path in (root / "docs").rglob("*"):
        if (
            path.is_file()
            and path.suffix.lower() in NUMERIC
            and path.name not in EXCLUDE
        ):
            from tensorfvm.verification.doc_claims import artifact_numbers

            pool.extend(artifact_numbers(path).values())
    return sorted(set(pool))


def traced_elsewhere(token, pool):
    band, value = printed_band(token), float(token)
    index = bisect.bisect_left(pool, value - band)
    return index < len(pool) and pool[index] <= value + band


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="print the summary only")
    args = parser.parse_args()

    rows = audit_traceability(str(ROOT / "docs"))
    summary = summarize(rows)
    pool = evidence_pool(ROOT)
    elsewhere = 0
    for report in sorted((ROOT / "docs").rglob("report.md")):
        if not evidence_artifacts(report.parent):
            scan = scan_report(report, [])
        else:
            scan = scan_report(report)
        elsewhere += sum(
            1 for token in scan["untraced"] if traced_elsewhere(token, pool)
        )
    summary["tree_traced"] = summary["traced"] + elsewhere
    summary["nowhere"] = summary["claimable"] - summary["tree_traced"]
    summary["evidence_numbers_in_tree"] = len(pool)
    summary["tree_trace_ratio"] = (
        summary["tree_traced"] / summary["claimable"] if summary["claimable"] else 0.0
    )
    summary["dead_script_pointers"] = [
        dict(doc=r["doc"], scripts=r["dead_scripts"]) for r in rows if r["dead_scripts"]
    ]

    print(
        json.dumps(
            {k: v for k, v in summary.items() if k != "dead_script_pointers"}, indent=1
        )
    )
    for entry in summary["dead_script_pointers"]:
        print("dead reproduce pointer:", entry["doc"], entry["scripts"])
    if args.check:
        return 0

    registry = build_registry(str(ROOT / "docs"))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "traceability.json").write_text(
        json.dumps(dict(summary=summary, reports=rows), indent=1)
    )
    (OUT / "claims.json").write_text(json.dumps(registry, indent=1))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return 0

    top = sorted(rows, key=lambda r: -r["claimable"])[:12]
    labels = [pathlib.Path(r["doc"]).parent.name[:32] for r in top][::-1]
    local = [r["traced"] for r in top][::-1]
    loose = [r["claimable"] - r["traced"] for r in top][::-1]
    fig, axis = plt.subplots(figsize=(8.2, 5.4), dpi=140)
    axis.barh(labels, local, color="#1b6ca8", label="locally traced (gated)")
    axis.barh(
        labels, loose, left=local, color="#d9d9d9", label="no provenance in report dir"
    )
    axis.set_xlabel("claimable measured numbers quoted by report.md")
    axis.set_title(
        "TensorFVM report traceability: %d/%d (%.1f%%) tree-traced, %d locally gated"
        % (
            summary["tree_traced"],
            summary["claimable"],
            100 * summary["tree_trace_ratio"],
            summary["traced"],
        )
    )
    axis.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "traceability.png")
    print(
        "wrote",
        OUT / "traceability.png",
        OUT / "traceability.json",
        OUT / "claims.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
