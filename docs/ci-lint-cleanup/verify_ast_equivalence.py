"""Prove the cleanup is semantics-preserving where it is meant to be.

For every ``.py`` file under ``src/``, ``tests/`` and ``scripts/`` this compares
the ``ast.dump`` of the pristine ``main`` revision against the cleaned working
tree (line numbers excluded, so pure formatting does not register).  Files that
still differ are the ones carrying an intentional semantic delta: dead-code
removal (F841), unused-import removal (F401), lambda-to-def (E731), ambiguous
name renames (E741), ``noqa`` placement (E402) and split imports (E401).

Usage:
    git archive main src tests scripts | tar -x -C /tmp/tfvm_head
    python docs/ci-lint-cleanup/verify_ast_equivalence.py /tmp/tfvm_head \
        > docs/ci-lint-cleanup/ast_equivalence.json
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

ROOTS = ("src", "tests", "scripts")


def digest(path: Path) -> str | None:
    try:
        return ast.dump(ast.parse(path.read_text()))
    except SyntaxError:  # pragma: no cover - a syntax error is itself a finding
        return None


def main() -> None:
    baseline = Path(sys.argv[1])
    same, changed, missing = [], [], []
    for root in ROOTS:
        for path in sorted(Path(root).rglob("*.py")):
            other = baseline / path
            if not other.exists():
                missing.append(str(path))
                continue
            if digest(path) == digest(other):
                same.append(str(path))
            else:
                changed.append(str(path))
    result = {
        "ast_identical": len(same),
        "ast_changed": len(changed),
        "missing_in_baseline": missing,
        "changed_files": changed,
    }
    print(json.dumps(result, indent=2))
    if missing:
        raise SystemExit(f"files absent from baseline: {missing}")


if __name__ == "__main__":
    main()
