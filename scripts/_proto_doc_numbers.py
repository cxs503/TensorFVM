#!/usr/bin/env python3
"""Prototype: extract high-precision numeric claims from docs/*.md and see how many
are sourced from a JSON artifact in the same doc directory (or the docs root)."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

# A "claim" = a number printed with an exponent or >= 4 significant digits, or a
# percent with >= 3 significant digits.  Bare small integers / years are skipped.
NUM = re.compile(r"[-+]?\d+\.\d+(?:[eE][-+]?\d+)?|\d+(?:\.\d+)?[eE][-+]?\d+")


def numbers_in_json(path: Path) -> list[float]:
    out: list[float] = []
    try:
        data = json.loads(path.read_text())
    except Exception:
        return out

    def walk(node):
        if isinstance(node, bool):
            return
        if isinstance(node, (int, float)):
            out.append(float(node))
        elif isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
        elif isinstance(node, str):
            for m in NUM.findall(node):
                try:
                    out.append(float(m))
                except ValueError:
                    pass

    walk(data)
    return out


def main() -> int:
    total = 0
    for md in sorted(DOCS.rglob("*.md")):
        if "source-snapshot" in str(md):
            continue
        jsons = list(md.parent.glob("*.json")) + list(DOCS.glob("*.json"))
        pool: list[float] = []
        for j in jsons:
            pool.extend(numbers_in_json(j))
        if not pool:
            continue
        text = md.read_text(errors="ignore")
        claims = NUM.findall(text)
        big = []
        for c in claims:
            try:
                v = float(c)
            except ValueError:
                continue
            digits = len(c.replace("-", "").replace("+", "").split("e")[0].replace(".", "").lstrip("0"))
            if ("e" in c.lower()) or digits >= 5:
                big.append(v)
        total += len(big)
        print(f"{md.relative_to(ROOT)}  json={len(jsons)} pool={len(pool)} bigclaims={len(big)}")
    print("TOTAL high-precision claims in dirs that have JSON:", total)
    return 0


if __name__ == "__main__":
    sys.exit(main())