#!/usr/bin/env python3
"""Collect the raw m21018 run evidence into this audit directory.

Source: the m21018 mission worktree /root/tp-agent/work/m21018 (the branch that
was merged as PR #719 / commit 69311e24).  The mission left ``runs/`` and
``artifacts/`` *untracked*, so nothing of the raw force history is in git --
this script copies the compact referee payloads verbatim, keeps the first
``HEAD_ROWS`` of every force history (truncation is recorded), and writes a
sha256 manifest of the full originals so a later reader can re-verify.

Usage: python3 docs/audit/board-audit-m21018/collect_evidence.py
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

SRC = Path("/root/tp-agent/work/m21018")
DST = Path(__file__).resolve().parent / "artifacts"
GRIDS = (12, 20, 32)
HEAD_ROWS = 40
KEEP = ("result.json", "benchmark.json", "errors.json", "report.md")
#: the cpd20 record is the one that carries the reported benchmark numbers -> keep the
#: full force history (the referee needs the raw signal, not a head)
FULL_HISTORY = (20,)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    DST.mkdir(parents=True, exist_ok=True)
    manifest = {"source_worktree": str(SRC), "head_rows_kept": HEAD_ROWS, "files": {}}
    for cpd in GRIDS:
        run = SRC / "runs" / "case-cylinder-re100" / f"cpd{cpd}"
        out = DST / f"cpd{cpd}"
        out.mkdir(parents=True, exist_ok=True)
        for name in KEEP:
            src = run / name
            if not src.is_file():
                continue
            shutil.copy2(src, out / name)
            manifest["files"][f"cpd{cpd}/{name}"] = {
                "sha256": sha256(src), "bytes": src.stat().st_size, "copy": "verbatim"}
        hist = run / "force_history.csv"
        if hist.is_file() and cpd in FULL_HISTORY:
            shutil.copy2(hist, out / "force_history.csv")
            manifest["files"][f"cpd{cpd}/force_history.csv"] = {
                "sha256": sha256(hist), "bytes": hist.stat().st_size, "copy": "verbatim"}
        if hist.is_file():
            rows = hist.read_text().splitlines()
            (out / "force_history_head.csv").write_text("\n".join(rows[:HEAD_ROWS]) + "\n")
            manifest["files"][f"cpd{cpd}/force_history_head.csv"] = {
                "sha256_original_full": sha256(hist),
                "original_rows": len(rows) - 1,
                "copy": f"first {HEAD_ROWS} data rows only (header + {HEAD_ROWS})",
                "original_bytes": hist.stat().st_size,
            }
    for rel in ("docs/case-cylinder-re100/README.md", "docs/case-cylinder-re100/ladder.json",
                "artifacts/case-cylinder-re100/reference/reference.json"):
        src = SRC / rel
        if src.is_file():
            manifest["files"][rel] = {"sha256": sha256(src), "bytes": src.stat().st_size,
                                      "copy": "verbatim (in repo docs/)"}
    (DST / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: len(v) for k, v in manifest.items() if k == "files"}))
    print("wrote", DST / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())