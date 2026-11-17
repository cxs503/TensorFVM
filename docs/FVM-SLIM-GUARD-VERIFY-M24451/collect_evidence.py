#!/usr/bin/env python3
"""m24451 FVM-SLIM-GUARD: collect fresh, dated verification evidence.

Runs the full guard suite (staged / ci / live / report) on the REAL repo,
the e2e injection probe, and the guard test suite; records every raw
returncode + JSON payload into evidence.json.  Stdlib only.

Usage:  python3 docs/FVM-SLIM-GUARD-VERIFY-M24451/collect_evidence.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GUARD = REPO_ROOT / "scripts" / "guard" / "repo_guard.py"
OUT = Path(__file__).resolve().parent / "evidence.json"


def run(cmd: list[str]) -> dict:
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    return {
        "cmd": " ".join(cmd),
        "rc": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def main() -> int:
    ev: dict = {
        "mission": "m24451",
        "anchor": "FVM-SLIM-GUARD",
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "head": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip(),
    }

    # 1) full guard suite on the real repo
    for sub in ("staged", "ci", "live", "report"):
        r = run([sys.executable, str(GUARD), sub])
        entry = {
            "rc": r["rc"],
            "stdout": r["stdout"].strip(),
            "stderr": r["stderr"].strip(),
        }
        if sub in ("live", "report"):
            entry["stdout_json"] = json.loads(r["stdout"])
        ev[f"guard_{sub}"] = entry

    # 2) e2e injection probe (real hook wiring, throwaway repo)
    inj = run([sys.executable, str(REPO_ROOT / "scripts" / "guard"
                                    / "injection_test.py"),
               "--out", str(OUT.parent)])
    try:
        inj_payload = json.loads(
            (OUT.parent / "injection-test.json").read_text())
    except Exception:
        inj_payload = None
    ev["injection"] = {"rc": inj["rc"], "payload": inj_payload,
                       "stderr_tail": inj["stderr"].strip().splitlines()[-3:]}

    # 3) guard test suite
    tst = run([sys.executable, "-m", "pytest", "tests/test_repo_guard.py",
               "-q", "--no-header", "-p", "no:cacheprovider"])
    ev["pytest"] = {
        "rc": tst["rc"],
        "tail": [l for l in tst["stdout"].strip().splitlines() if l][-3:],
    }

    # 4) hook wiring on this worktree
    ev["hook_wiring"] = {
        "core_hooksPath": subprocess.run(
            ["git", "config", "core.hooksPath"], cwd=REPO_ROOT,
            capture_output=True, text=True,
        ).stdout.strip(),
        "hook_tracked": bool(subprocess.run(
            ["git", "ls-files", ".githooks/pre-commit"], cwd=REPO_ROOT,
            capture_output=True, text=True,
        ).stdout.strip()),
    }

    OUT.write_text(json.dumps(ev, indent=2, ensure_ascii=False) + "\n")
    print(f"evidence -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
