"""Is archived verification evidence still produced by the source that is in the tree?

Every case under ``docs/<case>/manifest.json`` records SHA-256 digests of the
files that produced its numbers (``source_sha256`` family).  ``audit_external_evidence.py``
checks that a digest *matches the archived copy* it is bound to.  Nothing checked
the other direction: whether the *current tree* still is that source.  A module
that is edited after its verification was archived leaves a published claim
resting on code nobody runs any more -- the archive keeps looking fresh because
the digest still matches the frozen snapshot.

This module classifies every recorded source reference of every case:

``fresh``
    The resolved file still has the recorded digest: the archived evidence is
    backed by the source in the tree.
``changed``
    Digest mismatch: the producer source has been edited since archiving.
``missing``
    No candidate location holds the file, so the binding cannot be checked at
    all (worse than stale: the input is neither archived nor in the tree).

Resolution order is *snapshot first*, then the repository root, then the case
directory.  A case that froze its source under ``source_snapshot_root`` is
compared against the frozen copy; if the frozen copy does not exist the recorded
path is interpreted as repository-relative, which is how the archiving missions
of this repository wrote non-source inputs (``docs/*.json``, mesh files, data
READMEs).  Every entry carries the provenance it resolved through, so a reader
can tell which rule applied.
"""

from __future__ import annotations

import hashlib
import pathlib
import re

#: Manifest keys whose values are ``{relative path: sha256}`` producer bindings.
SOURCE_KEYS = (
    "source_sha256",
    "acceptance_source_sha256",
    "report_source_sha256",
    "integration_source_sha256",
    "plot_sources",
)

STATUSES = ("fresh", "changed", "missing")

#: The repository's public claim ledger; a stale case cited there is headline debt.
HEADLINE = pathlib.Path("docs/benchmarks-index.md")


def sha256(path):
    """Hex SHA-256 of a file's bytes."""
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def source_entries(manifest):
    """``[(key, relative_path, digest)]`` for every source binding in *manifest*."""
    entries = []
    for key in SOURCE_KEYS:
        block = manifest.get(key)
        if not block:
            continue
        items = block.items() if isinstance(block, dict) else block
        for item in items:
            # Some keys (``plot_sources``) mix bare names with ``{path: sha}``
            # pairs; only digests are auditable bindings.
            if not (isinstance(item, (list, tuple)) and len(item) == 2):
                continue
            relative, digest = item
            if isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest):
                entries.append((key, str(relative), digest))
    return entries


def resolve(case_dir, relative, snapshot_root, root):
    """Locate the file a binding refers to; return ``(path or None, provenance)``."""
    case_dir, root = pathlib.Path(case_dir), pathlib.Path(root)
    candidates = []
    if snapshot_root:
        candidates.append((case_dir / snapshot_root / relative, "snapshot"))
    candidates.append((root / relative, "root"))
    candidates.append((case_dir / relative, "case"))
    for path, provenance in candidates:
        if path.is_file():
            return path, provenance
    return None, "missing"


def classify_entry(case_dir, relative, digest, snapshot_root, root):
    """Classify one ``{path: sha256}`` binding of one archived case."""
    path, provenance = resolve(case_dir, relative, snapshot_root, root)
    if path is None:
        status = "missing"
    else:
        status = "fresh" if sha256(path) == digest else "changed"
    return dict(
        path=relative,
        digest=digest,
        provenance=provenance,
        status=status,
        resolved=None if path is None else pathlib.Path(path).as_posix(),
    )


def classify_case(case_dir, manifest=None, root="."):
    """Classify every source binding of one archived case."""
    import json

    case_dir = pathlib.Path(case_dir)
    if manifest is None:
        manifest = json.loads((case_dir / "manifest.json").read_text())
    snapshot_root = manifest.get("source_snapshot_root")
    entries = [
        classify_entry(case_dir, relative, digest, snapshot_root, root)
        for _key, relative, digest in source_entries(manifest)
    ]
    counts = {status: sum(e["status"] == status for e in entries) for status in STATUSES}
    return dict(
        case=case_dir.name,
        doc=case_dir.as_posix(),
        snapshot_root=snapshot_root,
        entries=entries,
        counts=counts,
        stale=counts["changed"] + counts["missing"] > 0,
    )


def audit(root="."):
    """Classify all archived cases that carry at least one source binding."""
    root = pathlib.Path(root)
    cases = []
    for path in sorted(root.glob("docs/*/manifest.json")):
        case = classify_case(path.parent, root=root)
        if case["entries"]:
            cases.append(case)
    index = root / HEADLINE
    headline = index.read_text() if index.is_file() else ""
    for case in cases:
        # Citing the case directory is how the ledger points at its evidence.
        case["cited_by_ledger"] = pathlib.PurePosixPath(case["doc"]).name in headline
    summary = dict(
        cases=len(cases),
        entries=sum(len(c["entries"]) for c in cases),
        fresh=sum(c["counts"]["fresh"] for c in cases),
        changed=sum(c["counts"]["changed"] for c in cases),
        missing=sum(c["counts"]["missing"] for c in cases),
        stale_cases=sum(c["stale"] for c in cases),
        headline_stale_cases=sum(c["stale"] and c["cited_by_ledger"] for c in cases),
    )
    return dict(summary=summary, cases=cases)


def stale_cases(report):
    """Case names with a changed or missing binding."""
    return [c["case"] for c in report["cases"] if c["stale"]]


def unregistered_stale(report, ledger_cases):
    """Stale cases that the committed ledger does not already record.

    This is the ratchet: the known debt is listed once, and a *new* stale case
    (or a case that gains a stale binding) turns the gate red.
    """
    return sorted(set(stale_cases(report)) - set(ledger_cases))