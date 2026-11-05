#!/usr/bin/env python3
"""No-silent-loss audit for the PR #82 merge (agent/m3240 -> main).

The merge of ``agent/m3240`` (PR #82) into ``main`` was *not* a text conflict in
every file: ``docs/verification-lid-cavity-ladder/summary.json`` was regenerated on
one side only, so git's auto-merge would have overwritten it silently.  Mechanical
overlays have a history of dropping content in this repo, so the merge is audited
mechanically instead of trusted.

Method -- for every file touched by either parent, compare the *merged* tree with
both parents using a line multiset (whitespace-only lines ignored, because git
resolves them with the surrounding block):

    preserved(side)  <=>  multiset(merged) - multiset(base)
                          >= multiset(side) - multiset(base)      (with multiplicity)

Every line a side added relative to the merge base must still be present, as many
times as it was added.  Lines that are genuinely absent are printed as ``dropped``
and must each be covered by an explicit, machine-checked *adjudication* declared in
``ADJUDICATIONS`` below -- and that adjudication's ``must_contain`` needles must
actually be present in the merged file, so "documented" cannot be used as a euphemism
for "lost".  A side whose content was moved to another path is covered by
``RELOCATIONS``, which demands byte equality (sha256) between source and new path.

Usage:
    python3 scripts/m3948_merge_loss_audit.py [--out docs/m3948/merge-loss-audit.json]
                                              [--base SHA] [--main SHA] [--pr SHA]
                                              [--merged REV]

Exit code 0 iff no unexplained drop and every relocation is byte-identical.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import subprocess
import sys

# Merge base of PR #82 and main, and the two tips that were merged.
DEFAULT_BASE = "0e9bd51dfac1ec543dac9f64079855ba9368298c"  # Merge #49 (m2766)
DEFAULT_MAIN = "db97955686af7832a7a56992e419aaf3c90d07ad"  # Merge #78 (m3149)
DEFAULT_PR = "99f8e6eab3a4ad0f7594b6d18730e4177e3ad54f"    # PR #82 head
DEFAULT_MERGED = "HEAD"

# Files whose PR-side content was deliberately re-housed.  sha256 must match.
RELOCATIONS = [
    {
        "from": "docs/verification-lid-cavity-ladder/summary.json",
        "to": "docs/verification-lid-cavity-correction/summary.json",
        "why": (
            "the shared ladder bundle is what the Re=1000 evidence tests score, and "
            "it must score against the printed Ghia table, so it is restored to the "
            "published-transcription scoring; PR #82's corrected-scoring "
            "regeneration is kept byte-for-byte next to the report that was computed "
            "from it"
        ),
    },
]

# Adjudications: blocks of lines one side added that are *not* expected verbatim in
# the merge because the merge states the same fact in a checkable way.  ``match`` is
# a substring test on the dropped line; ``must_contain`` needles are searched in the
# merged file and all must be present, otherwise the adjudication does not apply.
ADJUDICATIONS = [
    {
        "path": "src/tensorfvm/lid_cavity_reference.py",
        "side": "pr",
        "match": "-0.38760),  # CORRECTED",
        "why": (
            "PR #82 rewrote the published Ghia entry v(Re=400, x=0.9063) "
            "-0.23827 -> -0.38760 inside V_CENTRELINE.  main (unit m3099) adopted "
            "the opposite invariant in the meantime: a published transcription is "
            "never overwritten, the mismatch is registered with evidence.  The "
            "proposal survives as a first-class opt-in register entry "
            "(CORRECTIONS / corrected_table / applied_corrections) instead of an "
            "edit of the reference table, so the correction is reproducible and the "
            "printed table is still the thing being scored."
        ),
        "must_contain": [
            "-0.38760",
            "CORRECTIONS: dict",
            "def corrected_table",
            "def applied_corrections",
            '"applied": False',
            "proposed_correction",
        ],
    },
    {
        "path": "src/tensorfvm/lid_cavity_reference.py",
        "side": "pr",
        "match": "",
        "matches": [
            "Correction history",
            "The original value -0.23827",
            "This fixes the verification debt",
            "the corrected value is -0.38760",
            "profile and the solver results",
            "refinement (0.115/0.142/0.149",
            "Fixed transcription error in V_CENTRELINE",
        ],
        "why": (
            "PR #82's '+ Correction history' docstring block asserted the "
            "transcription had been fixed.  Kept as a history/proposal-register "
            "block that states what actually holds (proposal registered, never "
            "written into the table), while preserving the proposed value, its "
            "grid-convergence basis and the remaining debt numbers."
        ),
        "must_contain": [
            "Correction history / proposal register",
            "-0.23827 -> -0.38760",
            "not* written into :data:`V_CENTRELINE`",
            ":data:`CORRECTIONS` / :func:`corrected_table`",
            "0.115/0.142/0.149",
        ],
    },
    {
        "path": "src/tensorfvm/lid_cavity_reference.py",
        "side": "pr",
        "match": "------------------",
        "why": (
            "underline of the docstring heading above; the merged heading is "
            "'Correction history / proposal register' with its own underline."
        ),
        "must_contain": ["Correction history / proposal register"],
    },
    {
        "path": "src/tensorfvm/lid_cavity_reference.py",
        "side": "pr",
        "match": "raise KeyError",
        "why": (
            "cosmetic rewording of the same KeyError; main's variant (which prints "
            "the Re= value and the supported list) is a superset of PR #82's text."
        ),
        "must_contain": ['raise KeyError(f"no Ghia table for Re='],
    },
    {
        "path": "tests/test_lid_cavity_re1000_evidence.py",
        "side": "pr",
        "match": "def test_re400_v_centreline_discrepancy_resolved",
        "why": (
            "PR #82 called its claim test "
            "'test_re400_v_centreline_discrepancy_resolved'; the merge keeps the "
            "claim and renames it '..._resolved_by_the_proposed_correction' so the "
            "name states that it is scored against the proposal, not the print."
        ),
        "must_contain": [
            "def test_re400_v_centreline_discrepancy_resolved_by_the_proposed_correction",
            "def test_re400_v_centreline_discrepancy_is_still_present_and_unexplained",
        ],
    },
    {
        "path": "tests/test_lid_cavity_re1000_evidence.py",
        "side": "pr",
        "match": "",  # remaining PR-side test-body lines, see `matches`
        "matches": [
            "max_abs_error",
            "corrected reference",
            "transcription error in the Ghia reference data",
            "With the corrected reference",
            "verifies the fix",
            "consistent with the other Reynolds",
            "has been resolved by correcting",
            "exceeds 3% after reference correction",
            "caused by a transcription",
            "reference data (was -0.23827",
            "key not in cases",
            "key = (400.0, n)",
            "case = cases[key]",
        ],
        "why": (
            "PR #82's test body asserted case['v']['max_abs_error'] from the "
            "committed ladder bundle -- which only passes if that shared bundle is "
            "scored against the proposed table.  The merge re-derives the same "
            "pointwise error from the committed published-scoring bundle, so the "
            "test no longer depends on which bundle is committed, and it is "
            "stronger: it pins the published value, the only-differing-x set, the "
            "10x shrink and the n=32 > 3% control."
        ),
        "must_contain": [
            "def test_re400_v_centreline_discrepancy_resolved_by_the_proposed_correction",
            'table(400.0, "v", corrected=True)',
            "assert error < 0.03",
            'case["v"]["max_abs_error"] / 10.0',
            "assert error > 0.03",
        ],
    },
]


def git(*args: str) -> str:
    return subprocess.run(
        ("git",) + args, check=True, capture_output=True, text=True
    ).stdout


def blob(rev: str, path: str) -> bytes | None:
    proc = subprocess.run(("git", "show", f"{rev}:{path}"), capture_output=True)
    return proc.stdout if proc.returncode == 0 else None


def lines(data: bytes | None) -> list[str]:
    """Content lines only: whitespace-only lines carry no information here."""
    if data is None:
        return []
    return [ln for ln in data.decode("utf-8", "surrogateescape").splitlines() if ln.strip()]


def extra(side: list[str], base: list[str]) -> collections.Counter:
    return collections.Counter(side) - collections.Counter(base)


def missing(side_extra: collections.Counter, merged_extra: collections.Counter):
    out = []
    for line, count in side_extra.items():
        have = merged_extra.get(line, 0)
        if have < count:
            out.append({"line": line, "needed": count, "present": have})
    return sorted(out, key=lambda item: item["line"])


def changed_paths(base: str, other: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for row in git("diff", "--name-status", base, other).splitlines():
        parts = row.split("\t")
        if len(parts) >= 2:
            out[parts[-1]] = parts[0][:1]  # A/M/D/R/T
    return out


def covering_rules(path: str, side: str, line: str) -> list[int]:
    """Indices of adjudications that claim this dropped line."""
    hits = []
    for index, rule in enumerate(ADJUDICATIONS):
        if rule["path"] != path or rule["side"] != side:
            continue
        needles = rule.get("matches") or [rule["match"]]
        if any(needle in line for needle in needles if needle):
            hits.append(index)
    return hits


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--main", default=DEFAULT_MAIN)
    ap.add_argument("--pr", default=DEFAULT_PR)
    ap.add_argument("--merged", default=DEFAULT_MERGED)
    ap.add_argument("--out", default="docs/m3948/merge-loss-audit.json")
    args = ap.parse_args(argv)

    base_rev, main_rev, pr_rev = args.base, args.main, args.pr
    merged = git("rev-parse", args.merged).strip()

    report: dict[str, object] = {
        "base": base_rev,
        "main": main_rev,
        "pr": pr_rev,
        "merged": merged,
        "method": (
            "line multiset (whitespace-only lines ignored): every line a parent "
            "added relative to the merge base must still be present in the merged "
            "tree with at least the same multiplicity; residual drops must be "
            "covered by an ADJUDICATIONS rule whose must_contain needles are found "
            "in the merged file; relocated files must be byte-identical."
        ),
        "files": [],
        "relocations": [],
        "both_sided_files": [],
    }
    failures = 0

    verified_relocations = {}
    for rule in RELOCATIONS:
        src = blob(pr_rev, rule["from"])
        dst = blob(merged, rule["to"])
        ok = src is not None and dst is not None and src == dst
        verified_relocations[rule["from"]] = ok
        report["relocations"].append(
            {
                "from": rule["from"],
                "to": rule["to"],
                "why": rule["why"],
                "sha256_from": hashlib.sha256(src or b"").hexdigest(),
                "sha256_to": hashlib.sha256(dst or b"").hexdigest(),
                "identical": ok,
            }
        )
        if not ok:
            failures += 1

    on_main = changed_paths(base_rev, main_rev)
    on_pr = changed_paths(base_rev, pr_rev)
    rule_credit: collections.Counter = collections.Counter()

    for path in sorted(set(on_main) | set(on_pr)):
        status_main, status_pr = on_main.get(path), on_pr.get(path)
        rec: dict[str, object] = {
            "path": path,
            "main_status": status_main,
            "pr_status": status_pr,
        }
        merged_blob = blob(merged, path)
        rec["present_in_merged"] = merged_blob is not None
        if merged_blob is None:
            rec["verdict"] = "MISSING_IN_MERGE"
            failures += 1
            report["files"].append(rec)
            continue

        base_lines = lines(blob(base_rev, path))
        merged_extra = extra(lines(merged_blob), base_lines)
        merged_text = merged_blob.decode("utf-8", "surrogateescape")
        rec["sides"] = []

        for side_name, rev, status in (("main", main_rev, status_main), ("pr", pr_rev, status_pr)):
            if status is None:
                continue
            drops = missing(extra(lines(blob(rev, path)), base_lines), merged_extra)
            unexplained = []
            applied = set()
            for drop in drops:
                covered = False
                for index in covering_rules(path, side_name, drop["line"]):
                    rule = ADJUDICATIONS[index]
                    if all(needle in merged_text for needle in rule["must_contain"]):
                        applied.add(index)
                        rule_credit[index] += 1
                        covered = True
                        break
                if not covered:
                    unexplained.append(drop)

            entry = {
                "side": side_name,
                "status": status,
                "added_lines": sum(extra(lines(blob(rev, path)), base_lines).values()),
                "dropped_or_rewritten": len(drops),
                "unexplained": unexplained,
                "adjudications_applied": [
                    {"why": ADJUDICATIONS[i]["why"], "lines": sum(
                        1 for d in drops
                        if i in covering_rules(path, side_name, d["line"]))}
                    for i in sorted(applied)
                ],
            }
            if verified_relocations.get(path) and side_name == "pr":
                # byte-identical copy verified at the new path: nothing was lost,
                # the file moved (documented in RELOCATIONS)
                entry["verdict"] = "preserved_by_relocation"
                entry["relocated_to"] = next(
                    r["to"] for r in RELOCATIONS if r["from"] == path)
                entry["unexplained"] = []
            elif unexplained:
                entry["verdict"] = "CONTENT_LOST"
                failures += 1
            else:
                entry["verdict"] = "preserved"
            rec["sides"].append(entry)

        if len(rec["sides"]) == 2:
            report["both_sided_files"].append(path)
        report["files"].append(rec)

    report["adjudication_usage"] = {
        f"{ADJUDICATIONS[i]['path']} [{ADJUDICATIONS[i]['side']}]: "
        f"{ADJUDICATIONS[i]['match'] or 'body-block'}": count
        for i, count in sorted(rule_credit.items())
    }
    unused = [i for i in range(len(ADJUDICATIONS)) if i not in rule_credit]
    report["adjudications_unused"] = [
        f"{ADJUDICATIONS[i]['path']} [{ADJUDICATIONS[i]['side']}]: "
        f"{ADJUDICATIONS[i]['match'] or 'body-block'}" for i in unused
    ]
    report["summary"] = {
        "files_audited": len(report["files"]),
        "both_sided_files": report["both_sided_files"],
        "relocations": len(RELOCATIONS),
        "adjudications_used": len(rule_credit),
        "adjudications_declared": len(ADJUDICATIONS),
        "failures": failures,
        "ok": failures == 0,
    }

    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write(text)

    for rec in report["files"]:
        for side in rec["sides"]:
            mark = "OK " if side["verdict"] != "CONTENT_LOST" else "BAD"
            print(f"{mark} {rec['path']} [{side['side']}] +{side['added_lines']} "
                  f"rewritten={side['dropped_or_rewritten']} "
                  f"unexplained={len(side['unexplained'])} -> {side['verdict']}")
    for rel in report["relocations"]:
        print(f"{'OK ' if rel['identical'] else 'BAD'} relocation {rel['from']} -> {rel['to']}")
    if unused:
        print(f"WARN unused adjudications: {report['adjudications_unused']}")
    print(f"\nfailures={failures} ok={report['summary']['ok']} -> {args.out}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())