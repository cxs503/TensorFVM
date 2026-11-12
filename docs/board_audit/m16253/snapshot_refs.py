#!/usr/bin/env python3
"""Freeze the audited delivery (m16219) into this audit worktree.

Why a snapshot: the delivery commit 4f87f691 exists only on ``agent/m16219``
(the ecosystem mirror ``/root/tensor-ecosystem/TensorFVM`` is at 0cd53fa5 and
does not contain the case at all), and the delivery worktree is recyclable.
Evidence pointers in the verdict must stay resolvable after both events, so the
files the verdict cites are copied here byte-for-byte with a sha256 + git
provenance manifest.

Run: python3 docs/board_audit/m16253/snapshot_refs.py
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REFS = HERE / "refs"
DELIVERY = Path("/root/tp-agent/work/m16219")
COMMIT = "4f87f691"

FILES = [
    "src/tensorfvm/case_blasius_plate.py",
    "scripts/run_case_blasius_plate.py",
    "scripts/render_case_blasius_plate_report.py",
    "src/tensorfvm/verification/cases.py",
    "src/tensorfvm/verification/__main__.py",
    "tests/test_case_blasius_plate.py",
    "docs/case-blasius-plate/REPORT.md",
    "docs/case-blasius-plate/metrics.csv",
    "docs/case-blasius-plate/artifacts/march.json",
    "docs/case-blasius-plate/artifacts/march-sensitivity.json",
    "docs/case-blasius-plate/artifacts/summary.json",
    "docs/case-blasius-plate/artifacts/fvm-baseline.trace.jsonl",
    "docs/case-blasius-plate/figures/fig1_blasius_reference.png",
    "docs/case-blasius-plate/figures/fig2_ref_vs_got.png",
    "docs/case-blasius-plate/figures/fig3_march_sensitivity.png",
    "docs/case-blasius-plate/figures/fig4_fvm_convergence.png",
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    if REFS.exists():
        shutil.rmtree(REFS)
    entries = []
    for rel in FILES:
        src = DELIVERY / rel
        if not src.is_file():
            entries.append({"path": rel, "present": False})
            continue
        dst = REFS / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        entries.append({"path": rel, "present": True, "bytes": dst.stat().st_size,
                        "sha256": sha(dst)})

    head = subprocess.run(["git", "-C", str(DELIVERY), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    committed = subprocess.run(["git", "-C", str(DELIVERY), "ls-tree", "-r",
                                "--name-only", COMMIT], capture_output=True,
                               text=True).stdout.split()
    manifest = {
        "source_worktree": str(DELIVERY),
        "delivery_commit": COMMIT,
        "delivery_head": head,
        "committed_files": len(committed),
        "case_files_in_commit": sorted(p for p in committed
                                       if "blasius" in p or p.endswith("metrics.csv")),
        "eco_mirror_head": subprocess.run(
            ["git", "-C", "/root/tensor-ecosystem/TensorFVM", "rev-parse", "HEAD"],
            capture_output=True, text=True).stdout.strip(),
        "eco_mirror_has_case": (Path("/root/tensor-ecosystem/TensorFVM")
                                / "src/tensorfvm/case_blasius_plate.py").is_file(),
        "files": entries,
    }
    (REFS / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    missing = [e["path"] for e in entries if not e["present"]]
    print(f"snapshot: {len(entries) - len(missing)}/{len(entries)} files")
    if missing:
        print("MISSING:", missing)
    print("eco_mirror_head:", manifest["eco_mirror_head"],
          "has_case:", manifest["eco_mirror_has_case"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())