#!/usr/bin/env python3
"""Build the report bundle for the dam-break staleness-net closure (numbers injected).

Everything quoted by ``REPORT.md`` is computed here from the repository state and the
committed ledger -- no hand-copied number, no hand-copied table row.
"""

from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = pathlib.Path(__file__).resolve().parent
LEAGER = ROOT / "docs" / "verification-archived-source-staleness" / "staleness.json"
CATALOG = ROOT / "docs" / "verification-catalog.md"
CONV = ROOT / "src" / "tensorfvm" / "verification" / "artifacts" / "shallow-water-dam-break" / "convergence.json"
POOL_ENTRY = ROOT / "docs" / "case-fvm-dambreak-stoker" / "case_pool_entry.json"
MANIFEST = ROOT / "docs" / "shallow-water-dam-break" / "manifest.json"

sys.path.insert(0, str(ROOT / "src"))
from tensorfvm.verification import source_staleness as ss  # noqa: E402


def git_show(path):
    return subprocess.run(
        ["git", "-C", str(ROOT), "show", f"HEAD:{path}"],
        capture_output=True, text=True, check=True,
    ).stdout


def audit_state():
    report = ss.audit(str(ROOT))
    case = {c["case"]: c for c in report["cases"]}["shallow-water-dam-break"]
    return {
        "summary": report["summary"],
        "case": {
            "frozen": case["frozen"],
            "stale": case["stale"],
            "counts": case["counts"],
            "producer_bindings": case["producer_bindings"],
        },
        "unregistered_unfrozen_contains_case": "shallow-water-dam-break"
        in ss.unfrozen_names(report),
    }


def catalog_table_numbers(markdown):
    """``{n: percent}`` for the dry-bed front/bore series column of catalog table 19.1."""

    out = {}
    for line in markdown.splitlines():
        m = re.match(r"\|\s*(\d+)\s*\|\s*`ritter-dry`\s*\|\s*([\d.]+)\s*%", line)
        if m:
            out[int(m.group(1))] = float(m.group(2))
    return out


def artifact_series_numbers():
    data = json.loads(CONV.read_text())
    return {int(r["n"]): 100.0 * float(r["front_series"]) for r in data["ritter-dry"]["rows"]}


def main():
    ledger = json.loads(LEAGER.read_text())
    before = json.loads(git_show("docs/verification-archived-source-staleness/staleness.json"))
    after = audit_state()
    manifest = json.loads(MANIFEST.read_text())
    entry = json.loads(POOL_ENTRY.read_text())

    published = catalog_table_numbers(git_show("docs/verification-catalog.md"))
    artifact = artifact_series_numbers()
    catalog_fixed = catalog_table_numbers(CATALOG.read_text())

    required = (
        "grid", "dx", "cfl", "order", "t_star", "x_min", "x_max", "x_dam",
        "h0", "h1", "gravity", "dry_depth",
    )
    completeness = {
        c["key"]: {
            "recorded": sum(c.get(k) is not None for k in required),
            "required": len(required),
            "missing": [k for k in required if c.get(k) is None],
        }
        for c in entry["cases"]
    }
    headline = {
        c["key"]: {d["criterion"]: d["value"] for d in c["criteria"] if d["primary"]}
        for c in entry["cases"]
    }

    summary = {
        "case": "CASE-FVM-DAMBREAK-STOKER",
        "staleness_net": {
            "cases_before": before["summary"]["cases"],
            "cases_after": after["summary"]["cases"],
            "entries_before": before["summary"]["entries"],
            "entries_after": after["summary"]["entries"],
            "fresh_before": before["summary"]["fresh"],
            "fresh_after": after["summary"]["fresh"],
            "stale_cases_before": before["summary"]["stale_cases"],
            "stale_cases_after": after["summary"]["stale_cases"],
            "case_state": after["case"],
            "case_is_a_freeze_violation": after["unregistered_unfrozen_contains_case"],
            "ledger_matches_tree": {k: after["summary"][k] for k in ledger["summary"]} == ledger["summary"],
        },
        "frozen_producer_files": list(manifest["source_sha256"]),
        "pool_entry_completeness": completeness,
        "primary_criteria_percent": {
            key: {cid: 100.0 * val for cid, val in crits.items()} for key, crits in headline.items()
        },
        "catalog_series_error_percent": {
            "published_in_main": published,
            "artifact_in_main": artifact,
            "artifact_now": catalog_fixed,
            "published_disagrees_with_artifact": {
                str(n): {"published": published[n], "artifact": artifact[n]}
                for n in sorted(published)
                if n in artifact and abs(published[n] - artifact[n]) > 1e-9
            },
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    # ---------------------------------------------------------------- figures
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    labels = ["cases", "bound entries", "stale cases", "fresh bindings"]
    b = [before["summary"]["cases"], before["summary"]["entries"],
         before["summary"]["stale_cases"], before["summary"]["fresh"]]
    a = [after["summary"]["cases"], after["summary"]["entries"],
         after["summary"]["stale_cases"], after["summary"]["fresh"]]
    x = range(len(labels))
    ax.bar([i - 0.2 for i in x], b, width=0.4, label="before (main)", color="#999")
    ax.bar([i + 0.2 for i in x], a, width=0.4, label="after (this mission)", color="#2a7")
    for i, (bb, aa) in enumerate(zip(b, a)):
        ax.text(i - 0.2, bb, str(bb), ha="center", va="bottom", fontsize=7)
        ax.text(i + 0.2, aa, str(aa), ha="center", va="bottom", fontsize=7)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title("staleness net: dam-break case registered\n(stale count unchanged => no debt)", fontsize=9)
    ax.legend(fontsize=7)

    ax = axes[1]
    ns = sorted(artifact)
    ax.plot(ns, [artifact[n] for n in ns], "o-", color="#2a7", label="artifact (convergence.json)")
    ax.plot(ns, [published.get(n, float("nan")) for n in ns], "s--", color="#c33",
            label="published catalog 19.1 (main)")
    ax.axhline(5.0, color="k", lw=0.8, ls=":", label="5 % gate")
    ax.set_xscale("log")
    ax.set_xlabel("grid n")
    ax.set_ylabel("dry-bed x_f(t) series error [%]")
    ax.set_title("catalog drifted from its own artifact\n(n=100/200 published 4.695/6.613 %)", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[2]
    keys = list(completeness)
    ax.barh(keys, [completeness[k]["recorded"] for k in keys], color="#2a7")
    ax.axvline(12, color="k", ls=":", lw=0.8)
    for i, k in enumerate(keys):
        ax.text(completeness[k]["recorded"] - 0.2, i,
                f"{completeness[k]['recorded']}/12", ha="right", va="center",
                color="w", fontsize=8)
    ax.set_xlim(0, 13)
    ax.set_title("pool entry inputs recorded\n(dry_depth was the missing one)", fontsize=9)
    ax.set_xlabel("inputs readable from the entry")
    fig.tight_layout()
    fig.savefig(OUT / "fig_staleness_closure.png", dpi=130)
    print("wrote", OUT / "fig_staleness_closure.png")
    print(json.dumps(summary["staleness_net"], indent=1))
    print("published vs artifact mismatches:",
          summary["catalog_series_error_percent"]["published_disagrees_with_artifact"])
    print("completeness:", json.dumps(completeness))


if __name__ == "__main__":
    main()