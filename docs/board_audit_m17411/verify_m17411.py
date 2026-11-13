#!/usr/bin/env python3
"""m17411 (XVAL-FVM-THREE-GRID-LOGLOG-SLOPE-CELLCENTRE-FV-OR) board-audit re-verification.

Independent re-run of the audited mission's acceptance triple:
  V1 tests:  pytest tests/test_xval_three_grid_loglog_slope.py  (expect 17 passed)
  V2 xval:   run_xval.py 17/17 PASS + xval_results.json byte-identical to committed
  V3 bench:  tensor_benchmark_cases anchor row count (expect 0 -> NO_BENCHMARK_CASE)
Usage: python3 docs/board_audit_m17411/verify_m17411.py
"""
import hashlib, json, os, subprocess, sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
M = Path("/root/tp-agent/work/m17411")
X = M / "docs/xval-three-grid-loglog-slope"
out = {}

r = subprocess.run([sys.executable, "-m", "pytest", "tests/test_xval_three_grid_loglog_slope.py", "-q"],
                   cwd=M, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(M / "src")})
out["V1_pytest"] = {"exit": r.returncode, "tail": r.stdout.strip().splitlines()[-1:]}

committed = (X / "xval_results.json").read_bytes()
r = subprocess.run([sys.executable, str(X / "run_xval.py")], cwd=M, capture_output=True, text=True,
                   env={**os.environ, "PYTHONPATH": str(M / "src")})
regen = (X / "xval_results.json").read_bytes()
out["V2_xval"] = {"exit": r.returncode,
                  "summary_line": [l for l in r.stdout.splitlines() if "Overall" in l],
                  "json_sha256_committed": hashlib.sha256(committed).hexdigest(),
                  "json_sha256_regenerated": hashlib.sha256(regen).hexdigest(),
                  "byte_identical": committed == regen}

q = ("SELECT count(*) FROM tensor_benchmark_cases "
     "WHERE anchor_slug='XVAL-FVM-THREE-GRID-LOGLOG-SLOPE-CELLCENTRE-FV-OR'")
r = subprocess.run(["su", "postgres", "-c", f'psql -d tensorplatform -t -A -c "{q}"'],
                   capture_output=True, text=True)
out["V3_benchmark_anchor_rows"] = int(r.stdout.strip() or 0)

ok = (out["V1_pytest"]["exit"] == 0 and out["V2_xval"]["byte_identical"]
      and out["V2_xval"]["exit"] == 0)
out["overall"] = "PASS" if ok else "FAIL"
(OUT / "verify_output.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
print(json.dumps(out, indent=1, ensure_ascii=False))
sys.exit(0 if ok else 1)
