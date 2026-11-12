"""Machine gate for ``docs/verification-catalog.md`` (verification of the verifier's index).

Why this module exists
----------------------
``docs/verification-catalog.md`` is the repository's public index of what is
verified and what is *not*.  It is hand-written prose, so nothing checked it:
a renamed test file, a moved module or a capability that got verified after the
catalog was written all leave the index *silently* wrong.  That is not a
cosmetic problem -- the index is quoted as evidence ("capability X is
unverified"), and a stale entry is an unsupported claim in the very document
that defines which claims are supported.

This gate turns the index into executable statements:

``pointer``
    Every repository-relative path quoted in a catalog entry (``src/...``,
    ``tests/...``, ``docs/...``) must exist.  A dead pointer means the entry's
    provenance is gone.

``evidence``
    Every entry marked as passing must cite at least one existing
    ``tests/test_*.py`` file.  A PASS with no live test file is unsupported.

``gap``
    Every item of the *Current Gaps* **and** the *Planned Additions* lists must
    carry a status annotation.  Three statuses are understood:

    * ``[[gap-status: open]]`` -- not delivered (an explicit reason in the item
      text is expected but not machine-checked);
    * ``[[gap-status: closed-by <path>[, <path>...]]]`` -- delivered and
      verified; every cited path must exist;
    * ``[[gap-status: partially-closed-by <path>[, <path>...]]]`` -- delivered
      with a *bounded* honest FAIL / WITHHELD subset (e.g. one Rayleigh number
      inside the 3 % band, another still outside).  Every cited path must exist.

    The third status exists because the binary choice would push an honest
    partial result into either a false "open" (hides delivered work) or a false
    "closed-by" (overstates it).  That is exactly the failure mode the gate is
    meant to prevent, so the gate has to be able to say "partly".

    *Planned Additions* is audited with the same rules: an item that lists a
    capability as *planned* while the capability has already been delivered is
    a stale claim of the same kind, and (§10.2 was un-audited until m16836) was
    invisible to this gate.

The parser is deliberately dumb (regex over headings and backticks) so the
gate has no second source of truth: the catalog stays the single place where
the index is written down.

CLI::

    PYTHONPATH=src python -m tensorfvm.verification.catalog_gate \
        --catalog docs/verification-catalog.md --json /tmp/catalog_gate.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "CatalogEntry",
    "GapItem",
    "CatalogAudit",
    "parse_catalog",
    "audit_catalog",
    "render_report",
]

_POINTER_RE = re.compile(r"`([^`\s]+\.(?:py|json|md|npz|csv))`")
_ENTRY_RE = re.compile(r"^###\s+(\d+\.\d+)\s+(.+)$")
_GAP_RE = re.compile(r"^\d+\.\s+\*\*(.+?)\*\*\s*:?\s*(.*)$")
_GAP_STATUS_RE = re.compile(r"\[\[gap-status:\s*([^\]]+?)\]\]")
_CLOSED_BY_RE = re.compile(r"(?:partially-)?closed-by\s+(.+)$")
_TEST_NAME_RE = re.compile(r"^tests/test_[A-Za-z0-9_]+\.py$")
_PASS_MARK = "\u2705"  # ✅
#: statuses whose annotation must cite existing paths
_CITED_STATUSES = ("closed-by", "partially-closed-by")


@dataclass
class CatalogEntry:
    ident: str
    title: str
    body: str
    pointers: List[str] = field(default_factory=list)
    passing: bool = False

    @property
    def test_pointers(self) -> List[str]:
        return [p for p in self.pointers if _TEST_NAME_RE.match(p)]


@dataclass
class GapItem:
    text: str
    annotation: Optional[str]
    section: str = "current"

    @property
    def status(self) -> str:
        if not self.annotation:
            return "unannotated"
        ann = self.annotation.strip()
        low = ann.lower()
        # order matters: "partially-closed-by" must not be shadowed by "closed-by"
        if low.startswith("partially-closed-by"):
            return "partially-closed-by"
        if low.startswith("closed-by"):
            return "closed-by"
        if low.startswith("open"):
            return "open"
        return "unknown"


@dataclass
class CatalogAudit:
    catalog: str
    entries: List[CatalogEntry]
    gaps: List[GapItem]
    dangling_pointers: List[Dict[str, str]]
    unsupported_entries: List[str]
    unannotated_gaps: List[str]
    dangling_gap_closures: List[Dict[str, str]]
    planned: List[GapItem] = field(default_factory=list)
    unannotated_planned: List[str] = field(default_factory=list)
    dangling_planned_closures: List[Dict[str, str]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not (
            self.dangling_pointers
            or self.unsupported_entries
            or self.unannotated_gaps
            or self.dangling_gap_closures
            or self.unannotated_planned
            or self.dangling_planned_closures
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "catalog": self.catalog,
            "verdict": "PASS" if self.passed else "FAIL",
            "counts": {
                "entries": len(self.entries),
                "entries_passing": sum(1 for e in self.entries if e.passing),
                "pointers": sum(len(e.pointers) for e in self.entries),
                "gaps": len(self.gaps),
                "gaps_open": sum(1 for g in self.gaps if g.status == "open"),
                "gaps_closed": sum(1 for g in self.gaps if g.status == "closed-by"),
                "gaps_partial": sum(
                    1 for g in self.gaps if g.status == "partially-closed-by"
                ),
                "planned": len(self.planned),
                "planned_open": sum(1 for g in self.planned if g.status == "open"),
                "planned_closed": sum(
                    1 for g in self.planned if g.status == "closed-by"
                ),
                "planned_partial": sum(
                    1 for g in self.planned if g.status == "partially-closed-by"
                ),
            },
            "dangling_pointers": self.dangling_pointers,
            "entries_without_live_test": self.unsupported_entries,
            "gaps_without_status": self.unannotated_gaps,
            "dangling_gap_closures": self.dangling_gap_closures,
            "planned_without_status": self.unannotated_planned,
            "dangling_planned_closures": self.dangling_planned_closures,
        }


def _split_sections(text: str) -> List[str]:
    return re.split(r"\n###\s+", text)


def _gap_block(text: str) -> str:
    match = re.search(r"##\s*10\.\s*Verification Gaps.*?\n(.*?)(?:\n##\s|\Z)", text, re.S)
    return match.group(1) if match else ""


def _split_gap_subsections(block: str) -> Tuple[str, str]:
    """Split §10 into (*Current Gaps*, *Planned Additions*) blocks (the latter may be empty)."""

    parts = re.split(r"\n###\s+10\.2[^\n]*", block)
    current = re.split(r"\n###\s+10\.2", block)[0]
    planned = parts[1] if len(parts) > 1 else ""
    return current, planned


def _parse_gap_items(block: str, *, section: str) -> List[GapItem]:
    items: List[GapItem] = []
    for line in block.splitlines():
        match = _GAP_RE.match(line.strip())
        if not match:
            continue
        status = _GAP_STATUS_RE.search(line)
        items.append(
            GapItem(
                text=match.group(1).strip(),
                annotation=status.group(1) if status else None,
                section=section,
            )
        )
    return items


def parse_catalog(text: str, *, root: str = ".") -> tuple[List[CatalogEntry], List[GapItem]]:
    """Parse entries (``### N.M Title``) and gap items (``1. **Name**: ...``).

    Both §10.1 (*Current Gaps*) and §10.2 (*Planned Additions*) items are
    returned; ``GapItem.section`` tells them apart.
    """

    entries: List[CatalogEntry] = []
    for chunk in _split_sections(text)[1:]:
        head, _, rest = chunk.partition("\n")
        match = _ENTRY_RE.match("### " + head.strip())
        if match:
            ident, title = match.group(1), match.group(2).strip()
        else:  # non-numbered headings (Summary, Gaps, ...) are not entries
            continue
        pointers = []
        for ptr in _POINTER_RE.findall(rest):
            norm = os.path.normpath(ptr)
            if norm not in pointers:
                pointers.append(norm)
        entries.append(
            CatalogEntry(
                ident=ident,
                title=title,
                body=rest,
                pointers=pointers,
                passing=_PASS_MARK in rest,
            )
        )

    block = _gap_block(text)
    current, planned = _split_gap_subsections(block)
    gaps = _parse_gap_items(current, section="current")
    gaps += _parse_gap_items(planned, section="planned")
    return entries, gaps


def _exists(root: str, rel: str) -> bool:
    return os.path.exists(os.path.join(root, rel))


def _cited_targets(annotation: Optional[str]) -> List[str]:
    closure = _CLOSED_BY_RE.search(annotation or "")
    if not closure:
        return []
    return re.findall(r"[A-Za-z0-9_./-]+\.(?:py|json|md)", closure.group(1))


def _audit_items(
    items: Sequence[GapItem], root: str
) -> Tuple[List[str], List[Dict[str, str]]]:
    """Return (items without a usable status, closures citing dead paths)."""

    unannotated: List[str] = []
    dangling: List[Dict[str, str]] = []
    for item in items:
        status = item.status
        if status in ("unannotated", "unknown"):
            unannotated.append(item.text)
            continue
        if status in _CITED_STATUSES:
            targets = _cited_targets(item.annotation)
            if not targets:
                unannotated.append(item.text)
                continue
            for target in targets:
                if not _exists(root, target):
                    dangling.append(
                        {"gap": item.text, "pointer": target, "status": status}
                    )
    return unannotated, dangling


def audit_catalog(catalog_path: str = "docs/verification-catalog.md", *, root: str = ".") -> CatalogAudit:
    with open(os.path.join(root, catalog_path), "r", encoding="utf-8") as handle:
        text = handle.read()
    entries, all_items = parse_catalog(text, root=root)
    gaps = [item for item in all_items if item.section == "current"]
    planned = [item for item in all_items if item.section == "planned"]

    dangling: List[Dict[str, str]] = []
    unsupported: List[str] = []
    for entry in entries:
        for ptr in entry.pointers:
            if not _exists(root, ptr):
                dangling.append({"entry": f"{entry.ident} {entry.title}", "pointer": ptr})
        if entry.passing and not any(_exists(root, p) for p in entry.test_pointers):
            unsupported.append(f"{entry.ident} {entry.title}")

    unannotated, dangling_closures = _audit_items(gaps, root)
    unannotated_planned, dangling_planned = _audit_items(planned, root)

    return CatalogAudit(
        catalog=catalog_path,
        entries=entries,
        gaps=gaps,
        dangling_pointers=dangling,
        unsupported_entries=unsupported,
        unannotated_gaps=unannotated,
        dangling_gap_closures=dangling_closures,
        planned=planned,
        unannotated_planned=unannotated_planned,
        dangling_planned_closures=dangling_planned,
    )


def render_report(audit: CatalogAudit) -> str:
    data = audit.to_dict()
    counts = data["counts"]
    lines = [
        "# Verification catalog gate",
        "",
        f"catalog: `{audit.catalog}`",
        f"verdict: **{data['verdict']}**",
        "",
        "| check | count |",
        "|---|---:|",
        f"| entries | {counts['entries']} |",
        f"| entries marked PASS | {counts['entries_passing']} |",
        f"| quoted pointers | {counts['pointers']} |",
        f"| gaps declared (§10.1) | {counts['gaps']} |",
        f"| gaps open | {counts['gaps_open']} |",
        f"| gaps closed-by | {counts['gaps_closed']} |",
        f"| gaps partially-closed-by | {counts['gaps_partial']} |",
        f"| planned items declared (§10.2) | {counts['planned']} |",
        f"| planned open | {counts['planned_open']} |",
        f"| planned closed-by | {counts['planned_closed']} |",
        f"| planned partially-closed-by | {counts['planned_partial']} |",
        f"| dangling pointers | {len(audit.dangling_pointers)} |",
        f"| PASS entries without live test | {len(audit.unsupported_entries)} |",
        f"| gaps without status | {len(audit.unannotated_gaps)} |",
        f"| dangling gap closures | {len(audit.dangling_gap_closures)} |",
        f"| planned items without status | {len(audit.unannotated_planned)} |",
        f"| dangling planned closures | {len(audit.dangling_planned_closures)} |",
    ]
    for label, rows in (
        ("dangling pointers", audit.dangling_pointers),
        ("dangling gap closures", audit.dangling_gap_closures),
        ("dangling planned closures", audit.dangling_planned_closures),
    ):
        if rows:
            lines += ["", f"## {label}", ""]
            for row in rows:
                lines.append("- " + ", ".join(f"{k}=`{v}`" for k, v in row.items()))
    for label, rows in (
        ("PASS entries without a live test file", audit.unsupported_entries),
        ("gaps without a status annotation", audit.unannotated_gaps),
        ("planned items without a status annotation", audit.unannotated_planned),
    ):
        if rows:
            lines += ["", f"## {label}", ""] + [f"- {r}" for r in rows]
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="docs/verification-catalog.md")
    parser.add_argument("--root", default=".")
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)

    audit = audit_catalog(args.catalog, root=args.root)
    report = render_report(audit)
    print(report)
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(audit.to_dict(), handle, indent=1)
    return 0 if audit.passed else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())