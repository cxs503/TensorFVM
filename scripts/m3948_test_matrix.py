#!/usr/bin/env python3
"""Run the merge-relevant test suites on one tree and record the counts as JSON.

The box this ran on is shared and heavily loaded, and the harness kills a shell
call after ~120 s, so the matrix is filled one suite at a time and accumulated in
a JSON file instead of being transcribed by hand:

    python3 scripts/m3948_test_matrix.py --tree . --label merged \
        --suites ref,ladder-surface --out docs/m3948/tests.json

Suites are named in ``SUITES``.  A suite that does not finish inside
``--timeout`` is recorded as ``timeout`` (never as a pass), so the report can be
honest about what was and was not observed.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

SUITES = {
    "ref": [  # the two files PR #82 conflicted on / the bundle they read
        "tests/test_ghia_reference_consistency.py",
        "tests/test_lid_cavity_re1000_evidence.py",
    ],
    "ladder-surface": [  # everything that scores against the ladder bundle
        "tests/test_lid_cavity_extraction.py",
        "tests/test_wall_functions.py",
        "tests/test_gci_band_coverage.py",
        "tests/test_wall_cell_gradient_attribution.py",
    ],
    "verification": [
        "tests/test_analytic_reference_residuals.py",
        "tests/test_porous_forchheimer_reference.py",
    ],
    "lid-cavity": [  # live smoke run of the reference module
        "tests/test_lid_cavity.py",
    ],
}

TAIL = re.compile(
    r"^(?P<passed>\d+) passed(?:, (?P<failed>\d+) failed)?(?:, (?P<skipped>\d+) skipped)?"
    r"(?:, (?P<xfailed>\d+) xfailed)?(?:, (?P<xpassed>\d+) xpassed)?"
    r"(?:, (?P<warnings>\d+) warnings?)?"
)


def run_suite(tree: str, files: list[str], timeout: float) -> dict:
    env = dict(os.environ, PYTHONPATH=os.path.join(os.path.abspath(tree), "src"))
    started = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", *files, "-q", "-p", "no:cacheprovider"],
            cwd=tree, env=env, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "timeout",
            "seconds": round(time.time() - started, 1),
            "files": files,
            "tail": (exc.stdout or "")[-200:] if isinstance(exc.stdout, str) else "",
        }
    out = proc.stdout + proc.stderr
    counts = {"passed": 0, "failed": 0, "skipped": 0, "xfailed": 0, "xpassed": 0}
    for line in reversed(out.splitlines()):
        match = TAIL.match(line.strip())
        if match:
            for key, value in match.groupdict().items():
                if key in counts and value:
                    counts[key] = int(value)
            break
    else:
        return {"status": "unparsed", "seconds": round(time.time() - started, 1),
                "files": files, "tail": out[-200:]}
    return {
        "status": "ok" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "seconds": round(time.time() - started, 1),
        "files": files,
        **counts,
        "summary_line": [ln for ln in out.splitlines() if TAIL.match(ln.strip())][-1:],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tree", default=".")
    ap.add_argument("--label", required=True)
    ap.add_argument("--suites", required=True, help="comma separated, e.g. ref,verification")
    ap.add_argument("--out", default="docs/m3948/tests.json")
    ap.add_argument("--timeout", type=float, default=100.0)
    ap.add_argument("--rev", default="")
    args = ap.parse_args(argv)

    data = {"trees": {}}
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as handle:
            data = json.load(handle)
    record = data["trees"].setdefault(args.label, {"tree": os.path.abspath(args.tree),
                                                  "rev": args.rev, "suites": {}})
    record["rev"] = args.rev or record.get("rev", "")

    for name in args.suites.split(","):
        name = name.strip()
        if name not in SUITES:
            raise SystemExit(f"unknown suite {name!r}; known: {sorted(SUITES)}")
        result = run_suite(args.tree, SUITES[name], args.timeout)
        record["suites"][name] = result
        print(f"[{args.label}] {name}: {result['status']} "
              f"passed={result.get('passed', '-')} failed={result.get('failed', '-')} "
              f"skipped={result.get('skipped', '-')} xfailed={result.get('xfailed', '-')} "
              f"in {result['seconds']}s")

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())