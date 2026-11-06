"""Audit archival staleness of verification evidence and refresh its ledger.

Writes/refreshes ``docs/verification-archived-source-staleness/``:

* ``staleness.json`` -- per-case classification of every recorded source binding,
  plus the summary and the ratchet ledger (``known_stale_cases``).
* ``report.md`` -- the same numbers, generated here (never hand-copied).
* ``staleness-bars.png`` -- provenance census by manifest key and status.

Usage::

    PYTHONPATH=src python3 scripts/audit_archived_source_staleness.py
    PYTHONPATH=src python3 scripts/audit_archived_source_staleness.py --check
    PYTHONPATH=src python3 scripts/audit_archived_source_staleness.py --update-ledger

``--update-ledger`` accepts the current debt: the ratchet in
``tests/test_source_staleness.py`` then only fires on *new* stale cases, so an
acceptance must be a deliberate, reviewable edit rather than silence.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorfvm.verification.source_staleness import (  # noqa: E402
    audit,
    source_entries,
)

OUT = ROOT / "docs" / "verification-archived-source-staleness"
LEDGER = OUT / "staleness.json"


def key_census(root, cases):
    """``{(key, status): n}`` over every recorded binding, for the figure."""
    census = collections.Counter()
    for case in cases:
        manifest = json.loads((root / case["doc"] / "manifest.json").read_text())
        keys = [k for k, _r, _d in source_entries(manifest)]
        # ``classify_case`` iterates keys in SOURCE_KEYS order, so the per-entry
        # statuses line up with this list.
        for key, entry in zip(keys, case["entries"]):
            census[(key, entry["status"])] += 1
    return census


def figure(summary, census, path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # optional dependency: the audit still runs
        return False
    keys = sorted({k for k, _s in census})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = {"fresh": "#3b7d3b", "changed": "#c58a00", "missing": "#b3261e"}
    left = [0] * len(keys)
    for status in ("fresh", "changed", "missing"):
        values = [census.get((k, status), 0) for k in keys]
        axes[0].barh(keys, values, left=left, color=colors[status], label=status)
        left = [a + b for a, b in zip(left, values)]
    axes[0].set_xlabel("recorded SHA-256 bindings")
    axes[0].set_title("Archived source bindings by manifest key")
    axes[0].legend(fontsize=8)
    axes[0].tick_params(labelsize=7)
    labels = ["fresh\ncases", "stale\ncases", "stale cases\ncited by ledger"]
    values = [
        summary["cases"] - summary["stale_cases"],
        summary["stale_cases"],
        summary["headline_stale_cases"],
    ]
    bars = axes[1].bar(labels, values, color=["#3b7d3b", "#c58a00", "#b3261e"])
    axes[1].bar_label(bars)
    axes[1].set_ylabel("verification cases")
    axes[1].set_title(
        "%d cases: %d source bindings, %d stale (%d headline)"
        % (
            summary["cases"],
            summary["entries"],
            summary["changed"] + summary["missing"],
            summary["headline_stale_cases"],
        )
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def table(report):
    rows = [
        "| case | bindings | fresh | stale | missing | cited by ledger | changed or missing paths |",
        "|---|---:|---:|---:|---:|:---:|---|",
    ]
    for case in sorted(report["cases"], key=lambda c: (-c["counts"]["changed"] - c["counts"]["missing"], c["case"])):
        if not case["stale"]:
            continue
        paths = ", ".join(
            "`%s`%s" % (e["path"], "" if e["status"] == "changed" else " (MISSING)")
            for e in case["entries"]
            if e["status"] != "fresh"
        )
        rows.append(
            "| `%s` | %d | %d | %d | %d | %s | %s |"
            % (
                case["case"],
                len(case["entries"]),
                case["counts"]["fresh"],
                case["counts"]["changed"],
                case["counts"]["missing"],
                "yes" if case["cited_by_ledger"] else "no",
                paths,
            )
        )
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="print the summary only")
    parser.add_argument(
        "--update-ledger",
        action="store_true",
        help="accept the current stale cases as known debt",
    )
    args = parser.parse_args()

    report = audit(ROOT)
    summary = report["summary"]
    if args.check:
        print(json.dumps(summary, indent=1))
        return 0

    census = key_census(ROOT, report["cases"])
    OUT.mkdir(parents=True, exist_ok=True)
    known = []
    if LEDGER.is_file() and not args.update_ledger:
        known = json.loads(LEDGER.read_text()).get("known_stale_cases", [])
    if args.update_ledger or not known:
        known = sorted(c["case"] for c in report["cases"] if c["stale"])
    payload = dict(
        summary=summary,
        known_stale_cases=known,
        census={"%s/%s" % k: v for k, v in sorted(census.items())},
        cases=report["cases"],
    )
    LEDGER.write_text(json.dumps(payload, indent=1) + "\n")
    made = figure(summary, census, OUT / "staleness-bars.png")
    (OUT / "report.md").write_text(
        "# Archived source staleness audit\n\n"
        "Generated by `scripts/audit_archived_source_staleness.py`; "
        "the tables below are printed from `staleness.json`, not typed by hand.\n\n"
        "## Summary\n\n"
        "| cases | bindings | fresh | changed | missing | stale cases | stale and cited by `docs/benchmarks-index.md` |\n"
        "|---:|---:|---:|---:|---:|---:|---:|\n"
        "| %d | %d | %d | %d | %d | %d | %d |\n\n"
        "## Stale cases\n\n%s\n\n"
        "## Figure\n\n![staleness](staleness-bars.png)\n"
        % (
            summary["cases"],
            summary["entries"],
            summary["fresh"],
            summary["changed"],
            summary["missing"],
            summary["stale_cases"],
            summary["headline_stale_cases"],
            table(report),
        )
    )
    print(json.dumps(summary, indent=1))
    print("figure:", made)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())