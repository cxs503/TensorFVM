#!/usr/bin/env python3
"""Audit the union produced by merging PR #82 (branch ``agent/m3240``) into ``main``.

The merge of PR #82 was dirty because both sides touched the same verification
surface from opposite directions:

* ``agent/m3240`` rewrote the published Ghia Re = 400 v-centreline entry at
  x = 0.9063 (-0.23827 -> -0.38760) and turned the red ladder gate green by
  re-scoring the committed bundle against the rewritten value;
* ``main`` kept the published value and instead *registered* the discrepancy as a
  dispute (``lid_cavity_reference.DISPUTES``), added a smoothness audit that can
  only be satisfied by a registered dispute, and kept the ladder defect visible as
  a strict xfail.

A textual merge would have produced a file set that is internally inconsistent in
at least three ways (a rewritten oracle, a gate that silently flips to green, a
docstring announcing a fix that is not in force).  This script makes the union
checkable instead of reviewable-by-eye, in the spirit of the platform's
"mechanical overlay loses content" ban:

1. it re-derives the three-way file classification (only-m3240 / only-main / both)
   from git, so no file added by either side can disappear unnoticed;
2. it asserts the *semantic* invariants of the union: the published table is
   byte-identical to ``main``'s, every file delivered by ``agent/m3240`` is either
   present or explicitly adjudicated in ``docs/merge-082/README.md``;
3. it archives the numeric claim of the rejected proposal as a machine-readable
   artefact and plots the dispute (computed ladder vs published vs proposed) so
   the next adjudicator can overturn the decision with data, not prose;
4. it folds the pytest JUnit reports of the merge tree and of the ``main``
   baseline into one comparison table (regression check, not self-report).

Usage::

    python3 scripts/audit_merge_082_union.py --json docs/merge-082/union_audit.json \
        --figure docs/merge-082/re400_v_dispute.png \
        --junit-merge /tmp/junit_merge.xml --junit-base /tmp/junit_base.xml

The script is read-only with respect to git: it inspects refs, never writes them.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: The two sides of the merge under audit.
MAIN_REF = "origin/main"
BRANCH_REF = "origin/agent/m3240"
MERGE_BASE_HINT = "auto"  # resolved with git merge-base

#: Files delivered by the cured branch that the union deliberately does *not* take
#: verbatim, with the exact reason.  Anything delivered but neither present nor
#: listed here fails the audit (that is the "silent loss" detector).
ADJUDICATED = {
    "src/tensorfvm/lid_cavity_reference.py": (
        "union: main's published table + DISPUTES registry kept byte-for-byte; the "
        "branch's proposed value -0.38760 is recorded inside DISPUTES['correction_proposed'] "
        "and in the docstring dispute history instead of replacing the oracle"
    ),
    "tests/test_lid_cavity_re1000_evidence.py": (
        "union: main's strict xfail kept; the branch's claim (<3 % under a rewritten "
        "reference) is preserved as two regression tests over the archived numbers, so "
        "the claim stays testable without becoming an acceptance basis"
    ),
    "docs/verification-lid-cavity-ladder/summary.json": (
        "main's bundle (scored against the published table) is authoritative; the "
        "branch's re-scored bundle is archived as "
        "docs/verification-lid-cavity-correction/corrected_scoring.json"
    ),
    "docs/verification-lid-cavity-correction/report.md": (
        "kept verbatim with an adjudication banner prepended (status SUPERSEDED, "
        "proposal rejected, evidence retained)"
    ),
}


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout


def changed_files(base: str, ref: str) -> set[str]:
    out = git("diff", "--name-only", base, ref)
    return {line.strip() for line in out.splitlines() if line.strip()}


def classify(base: str) -> dict[str, object]:
    branch = changed_files(base, BRANCH_REF)
    main = changed_files(base, MAIN_REF)
    return {
        "merge_base": base,
        "main_ref": MAIN_REF,
        "main_sha": git("rev-parse", MAIN_REF).strip(),
        "branch_ref": BRANCH_REF,
        "branch_sha": git("rev-parse", BRANCH_REF).strip(),
        "only_branch": sorted(branch - main),
        "only_main": sorted(main - branch),
        "both": sorted(branch & main),
    }


def present_in_union(path: str) -> bool:
    return (ROOT / path).exists()


def audit_union_files(plan: dict[str, object]) -> dict[str, object]:
    """Every file the branch delivered must survive the merge, verbatim or adjudicated."""
    delivered = plan["only_branch"] + plan["both"]
    missing = [p for p in delivered if not present_in_union(p) and p not in ADJUDICATED]
    adjudicated_present = [p for p in delivered if p in ADJUDICATED and present_in_union(p)]
    verbatim = [p for p in delivered if present_in_union(p) and p not in ADJUDICATED]
    return {
        "delivered_by_branch": len(delivered),
        "verbatim": verbatim,
        "adjudicated": sorted(adjudicated_present),
        "silently_lost": missing,
        "ok": not missing,
    }


def audit_oracle() -> dict[str, object]:
    """The one invariant that matters: the published table is main's, unedited."""
    sys.path.insert(0, str(ROOT / "src"))
    from tensorfvm.lid_cavity_reference import DISPUTES, V_CENTRELINE, table

    published = dict(V_CENTRELINE[400.0])[0.9063]
    main_table = git("show", f"{MAIN_REF}:src/tensorfvm/lid_cavity_reference.py")
    branch_table = git("show", f"{BRANCH_REF}:src/tensorfvm/lid_cavity_reference.py")
    union_table = (ROOT / "src/tensorfvm/lid_cavity_reference.py").read_text()
    record = DISPUTES[(400.0, "v", 0.9063)]
    return {
        "published_value_in_union": published,
        "published_value_on_main": -0.23827,
        "published_value_on_branch": -0.38760,
        "published_value_unchanged_vs_main": published == -0.23827,
        "table_rows_identical_to_main": _table_block(union_table)
        == _table_block(main_table),
        "branch_did_edit_table": _table_block(branch_table) != _table_block(main_table),
        "proposal_recorded": record.get("correction_proposed") == -0.38760,
        "dispute_id": record.get("dispute_id"),
        "ok": (published == -0.23827
               and _table_block(union_table) == _table_block(main_table)
               and record.get("correction_proposed") == -0.38760),
    }


def _table_block(source: str) -> str:
    """The transcribed numbers only (both data tables), independent of prose."""
    lines = []
    inside = False
    for line in source.splitlines():
        if line.startswith("U_CENTRELINE") or line.startswith("V_CENTRELINE"):
            inside = True
        if inside:
            if line.startswith("#") or line.startswith("def "):
                inside = False
                continue
            stripped = line.split("#")[0].rstrip()
            if stripped:
                lines.append(stripped)
    return "\n".join(lines)


def disputed_profile() -> dict[str, object]:
    """Computed / published / proposed values at the disputed entry, from committed data."""
    bundle = json.loads(
        (ROOT / "docs/verification-lid-cavity-ladder/summary.json").read_text()
    )
    rows = {}
    for case in bundle["cases"]:
        if float(case["reynolds"]) != 400.0:
            continue
        profiles = case["v"]["profiles"]
        index = profiles["coord"].index(0.9063)
        rows[int(case["n"])] = {
            "computed": profiles["computed"][index],
            "reference": profiles["reference"][index],
            "max_abs_error": case["v"]["max_abs_error"],
        }
    archive = json.loads(
        (ROOT / "docs/verification-lid-cavity-correction/corrected_scoring.json").read_text()
    )
    archived = {int(r["n"]): r["v_max_abs_error"] for r in archive["rows"]
                if float(r["reynolds"]) == 400.0}
    return {"grids": rows, "error_under_proposed_value": archived,
            "published": -0.23827, "proposed": -0.38760}


def junit_summary(path: str | None) -> dict[str, object] | None:
    if not path:
        return None
    file = Path(path)
    if not file.exists():
        return {"path": str(file), "missing": True}
    tree = ET.parse(file)
    cases = {}
    for case in tree.iter("testcase"):
        name = f"{case.get('classname')}::{case.get('name')}"
        status = "passed"
        for tag in ("failure", "error", "skipped"):
            if case.find(tag) is not None:
                status = {"failure": "failed", "error": "error", "skipped": "skipped"}[tag]
                break
        cases[name] = status
    counts = {s: sum(1 for v in cases.values() if v == s)
              for s in ("passed", "failed", "error", "skipped")}
    return {"path": str(file), "counts": counts, "cases": cases}


def regression(merge: dict | None, base: dict | None) -> dict[str, object]:
    if not merge or not base or base.get("missing") or merge.get("missing"):
        return {"ok": None, "reason": "junit report(s) unavailable"}
    base_cases, merge_cases = base["cases"], merge["cases"]
    new_tests = sorted(set(merge_cases) - set(base_cases))
    lost = sorted(set(base_cases) - set(merge_cases))
    regressed = sorted(n for n in set(base_cases) & set(merge_cases)
                       if base_cases[n] == "passed" and merge_cases[n] != "passed")
    return {
        "base_counts": base["counts"],
        "merge_counts": merge["counts"],
        "new_tests": new_tests,
        "missing_tests": lost,
        "regressed_tests": regressed,
        "ok": not regressed and not lost and merge["counts"]["failed"] == 0
              and merge["counts"]["error"] == 0,
    }


def figure(path: str, profile: dict[str, object]) -> str | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - plotting is optional
        return f"skipped: {exc}"
    grids = profile["grids"]
    ns = sorted(grids)
    col = "C0"
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(ns, [grids[n]["computed"] for n in ns], "o-", color=col,
                 label="TensorFVM (ladder bundle)")
    axes[0].axhline(profile["published"], color="C3", ls="--",
                    label="Ghia published (in force)")
    axes[0].axhline(profile["proposed"], color="C2", ls=":",
                    label="PR #82 proposal (rejected)")
    axes[0].set_xlabel("grid n"); axes[0].set_ylabel("v at x = 0.9063, y = 0.5")
    axes[0].set_title("Disputed entry: solver converges away from the print")
    axes[0].set_xscale("log", base=2); axes[0].legend(fontsize=8); axes[0].grid(alpha=.3)

    axes[1].plot(ns, [grids[n]["max_abs_error"] for n in ns], "s-", color="C3",
                 label="scored vs published (gate in force)")
    axes[1].plot(sorted(profile["error_under_proposed_value"]),
                 [profile["error_under_proposed_value"][n]
                  for n in sorted(profile["error_under_proposed_value"])],
                 "^--", color="C2", label="scored vs proposal (archived)")
    axes[1].axhline(0.03, color="k", lw=.8, ls=":", label="3 % acceptance band")
    axes[1].set_xlabel("grid n"); axes[1].set_ylabel("v-centreline max |error|")
    axes[1].set_title("Re = 400 v-centreline error: red vs laundered green")
    axes[1].set_xscale("log", base=2); axes[1].legend(fontsize=8); axes[1].grid(alpha=.3)
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return str(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", default="docs/merge-082/union_audit.json")
    parser.add_argument("--figure", default="docs/merge-082/re400_v_dispute.png")
    parser.add_argument("--junit-merge")
    parser.add_argument("--junit-base")
    parser.add_argument("--strict", action="store_true",
                        help="exit non-zero when any invariant fails")
    args = parser.parse_args(argv)

    base = git("merge-base", MAIN_REF, BRANCH_REF).strip()
    plan = classify(base)
    union = audit_union_files(plan)
    oracle = audit_oracle()
    profile = disputed_profile()
    merge_junit = junit_summary(args.junit_merge)
    base_junit = junit_summary(args.junit_base)
    reg = regression(merge_junit, base_junit)
    fig = figure(args.figure, profile)

    report = {
        "merge": plan,
        "union_files": union,
        "oracle": oracle,
        "disputed_profile": profile,
        "conflicts_resolved_by_hand": sorted(ADJUDICATED),
        "tests": {"merge": merge_junit, "baseline": base_junit, "regression": reg},
        "figure": fig,
        "ok": bool(union["ok"] and oracle["ok"] and (reg["ok"] is not False)),
    }
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n")

    print(f"union files ok      : {union['ok']} ({len(union['verbatim'])} verbatim, "
          f"{len(union['adjudicated'])} adjudicated, {len(union['silently_lost'])} lost)")
    print(f"oracle untouched    : {oracle['ok']} (published {oracle['published_value_in_union']})")
    if reg["ok"] is not None:
        print(f"no regression       : {reg['ok']} "
              f"base={reg['base_counts']} merge={reg['merge_counts']} "
              f"new={len(reg['new_tests'])}")
    print(f"artefacts           : {out} {fig}")
    return 0 if report["ok"] or not args.strict else 1


if __name__ == "__main__":
    raise SystemExit(main())