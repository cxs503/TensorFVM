"""Bind numbers printed in ``docs/**/report.md`` to archived evidence.

Every verification report in this repository quotes measured numbers; the
evidence behind them is either a committed JSON artifact next to the report or a
``scripts/*.py`` re-run quoted in the report.  Until now nothing checked that
either pointer stayed alive: editing a report, regenerating an artifact, or
deleting/renaming a reproduce script all break a published claim *silently*.

This module makes three statements executable:

*claim*
    A numeric token of a report whose value is reproduced by a leaf of a sibling
    JSON artifact within the *printed precision band* (half a unit in the last
    printed decimal place).  The leaf pointer is the claim's provenance.

*traceability*
    The share of claimable tokens that have such provenance.  Round gates
    (``1e-10``) and structural integers (``48x16x4``) are excluded: they carry
    no measured content.

*reproduce pointer*
    A ``scripts/....py`` path quoted by the report.  It must exist.

A claim is violated when the report text moves out of the band **or** its
recorded provenance leaf moves out of the band, so a retyped report and a
regenerated artifact fail equally loudly.
"""

from __future__ import annotations

import json
import math
import pathlib
import re

MIN_SIGNIFICANT = 3
MAX_POINTERS = 4
CLAIM_FILE = "claims.json"
TOKEN = re.compile(r"-?\d+\.\d+(?:[eE][-+]?\d+)?")
SCRIPT = re.compile(r"scripts/[A-Za-z0-9_./-]+\.py")
NON_EVIDENCE = {"config.json", "environment.json", CLAIM_FILE}
TEXT_EVIDENCE = (".dat", ".csv", ".txt")


def _text_numbers(path):
    """``{line<L>:<index>: float}`` for whitespace/comma separated numeric dumps."""
    leaves = {}
    for line_no, line in enumerate(pathlib.Path(path).read_text(errors="ignore").splitlines()):
        for index, match in enumerate(TOKEN.findall(line)):
            leaves[f"line{line_no}:{index}"] = float(match)
    return leaves


def artifact_numbers(path):
    """Flatten an evidence file to ``{pointer: float}`` (JSON leaves or numeric dumps)."""
    path = pathlib.Path(path)
    if path.suffix.lower() in TEXT_EVIDENCE:
        return _text_numbers(path)
    try:
        payload = json.loads(path.read_text(errors="ignore"))
    except (OSError, ValueError):
        return {}
    leaves = {}

    def walk(node, pointer):
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{pointer}/{key}")
        elif isinstance(node, (int, float)) and not isinstance(node, bool):
            if math.isfinite(node):
                leaves[pointer] = float(node)
        elif isinstance(node, str):
            for index, match in enumerate(TOKEN.findall(node)):
                value = float(match)
                if math.isfinite(value):
                    leaves[f"{pointer}#{index}"] = value

    walk(payload, "")
    return leaves


def printed_band(token):
    """Half a unit in the last printed decimal place of *token*."""
    mantissa, _, exponent = token.lower().partition("e")
    decimals = len(mantissa.split(".")[1]) if "." in mantissa else 0
    exponent = int(exponent) if exponent else 0
    return 0.5 * 10.0 ** (exponent - decimals)


def significant_digits(token):
    digits = "".join(c for c in token.lower().split("e")[0] if c.isdigit())
    return len(digits.lstrip("0").rstrip("0") or "0")


def is_claimable(token):
    """Keep measured numbers, drop round gates and structural integers."""
    if significant_digits(token) < MIN_SIGNIFICANT:
        return False
    return abs(float(token.lower().split("e")[0])) != 1.0


def evidence_artifacts(directory):
    """Committed evidence files under a report directory (JSON plus numeric dumps)."""
    found = []
    for p in sorted(pathlib.Path(directory).rglob("*")):
        if not p.is_file() or p.name in NON_EVIDENCE:
            continue
        if p.suffix.lower() == ".json" or p.suffix.lower() in TEXT_EVIDENCE:
            found.append(p)
    return found


def scan_report(report, artifacts=None):
    """Classify every claimable token of one report.

    Returns ``dict`` with ``claimable``, ``claimable_tokens`` (untraced ones),
    ``claims`` (token + provenance pointers) and ``scripts`` (quoted reproduce
    pointers with an ``exists`` flag).
    """
    report = pathlib.Path(report)
    if artifacts is None:
        artifacts = evidence_artifacts(report.parent)
    values = []
    for artifact in artifacts:
        for pointer, value in artifact_numbers(artifact).items():
            values.append((artifact.as_posix(), pointer, value))
    text = report.read_text(errors="ignore")
    claims, untraced, seen = [], [], set()
    for token in TOKEN.findall(text):
        if token in seen or not is_claimable(token):
            continue
        seen.add(token)
        band = printed_band(token)
        printed = float(token)
        hits = [v for v in values if abs(v[2] - printed) <= band * (1 + 1e-9)]
        if not hits:
            untraced.append(token)
            continue
        claims.append(dict(doc=report.as_posix(), token=token, band=band,
                           pointers=[dict(artifact=a, pointer=p, value=v) for a, p, v in hits[:MAX_POINTERS]],
                           n_pointers=len(hits)))
    scripts = []
    for quoted in dict.fromkeys(SCRIPT.findall(text)):
        scripts.append(dict(path=quoted, exists=pathlib.Path(quoted).exists()))
    return dict(doc=report.as_posix(), artifacts=[a.as_posix() for a in artifacts],
                claimable=len(claims) + len(untraced), untraced=untraced,
                claims=claims, scripts=scripts)


def build_registry(root="docs"):
    """Registry of bound claims over every report directory that has artifacts."""
    reports = sorted(pathlib.Path(root).rglob("report.md"))
    claims, hooked = [], 0
    for report in reports:
        artifacts = evidence_artifacts(report.parent)
        if not artifacts:
            continue
        scan = scan_report(report, artifacts)
        if scan["claims"]:
            hooked += 1
            claims.extend(scan["claims"])
    return dict(schema=1, reports_scanned=len(reports), reports_hooked=hooked,
                n_claims=len(claims), claims=claims)


def verify_registry(registry):
    """Re-verify every bound claim; return ``(checked, violations)``."""
    cache, violations, checked = {}, [], 0
    for claim in registry["claims"]:
        doc = pathlib.Path(claim["doc"])
        if not doc.exists():
            violations.append(dict(kind="doc_missing", claim=claim))
            continue
        band, printed = claim["band"], float(claim["token"])
        tokens = [float(t) for t in TOKEN.findall(doc.read_text(errors="ignore"))]
        if not any(abs(t - printed) <= band * (1 + 1e-9) for t in tokens):
            violations.append(dict(kind="doc_drifted", claim=claim))
            continue
        alive = False
        for item in claim["pointers"]:
            artifact = pathlib.Path(item["artifact"])
            if not artifact.exists():
                continue
            if artifact not in cache:
                cache[artifact] = artifact_numbers(artifact)
            value = cache[artifact].get(item["pointer"])
            if value is not None and abs(value - printed) <= band * (1 + 1e-9):
                alive = True
                break
        if not alive:
            violations.append(dict(kind="artifact_drifted", claim=claim))
            continue
        checked += 1
    return checked, violations


def audit_traceability(root="docs"):
    """Per-report traceability plus dead reproduce pointers."""
    rows = []
    for report in sorted(pathlib.Path(root).rglob("report.md")):
        artifacts = evidence_artifacts(report.parent)
        scan = scan_report(report, artifacts)
        if not scan["claimable"] and not scan["scripts"]:
            continue
        rows.append(dict(doc=scan["doc"], artifacts=len(artifacts), claimable=scan["claimable"],
                         traced=len(scan["claims"]), untraced=len(scan["untraced"]),
                         dead_scripts=[s["path"] for s in scan["scripts"] if not s["exists"]]))
    return rows


def summarize(rows):
    claimable = sum(r["claimable"] for r in rows)
    traced = sum(r["traced"] for r in rows)
    return dict(reports=len(rows), claimable=claimable, traced=traced,
                untraced=claimable - traced,
                trace_ratio=(traced / claimable) if claimable else 0.0,
                reports_with_artifacts=sum(1 for r in rows if r["artifacts"]),
                reports_fully_untraced=sum(1 for r in rows if r["traced"] == 0 and r["claimable"]),
                dead_scripts=sum(len(r["dead_scripts"]) for r in rows))