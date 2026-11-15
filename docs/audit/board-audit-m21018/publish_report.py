#!/usr/bin/env python3
"""Publish the board-audit report for m21018 to the platform report API.

Builds /tmp/report_m21105.json (six-section paper-grade markdown + the three
audit figures as base64 png) and POSTs it to $TENSOR_API_BASE/tensor/reports.

Usage: python3 docs/audit/board-audit-m21018/publish_report.py [--dry-run]
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]

FIGS = [
    ("fig1_cpd32_divergence.png",
     "32 cells/D (960x640): the control-volume drag budget grows 1e26 -> 1e171 -> inf between "
     "t=0.20 and t=0.32 and is non-finite for 5986 of 6001 samples; the delivered report calls "
     "this 'GATED: no vortex shedding'."),
    ("fig2_ladder_errors.png",
     "Registered benchmark errors vs the platform tolerance band (cd_mean 5 %, st 3 %): 12 and 20 "
     "cells/D fail both metrics, 32 cells/D produced no finite value."),
    ("fig3_cpd20_signal.png",
     "20 cells/D force record: 10 clean shedding periods, St zero-crossing 0.15508 vs FFT 0.15512 "
     "— the statistics are sound, the force level is 12.6 % high."),
]

MD = """## 1. Case

Board audit (closure check, not a re-do) of mission **m21018**
`GAP-DIFFLBM-EXT-LIT-CYL` — JOUR-DIFFLBM literature anchor 3, FVM side —
merged into TensorFVM as **PR #719** (merge-master `cda4e7f9`, subject
`69311e24`). Physics: 2-D incompressible flow past an unbounded circular
cylinder at Re = UD/ν = 100, laminar von Kármán shedding; D=1, U=1, ν=0.01,
Δt=0.02, t_end=120 tU/D, domain 10D upstream / 20D downstream / ±10D lateral
(5 % blockage). Refer

| metric | reference | band | source |
|---|---|---|---|
| C_d,mean | 1.340 | ±0.07 (5 %) | Braza et al., JFM 165 (1986), doi:10.1017/S0022112086003014 |
| St | 0.164 | ±0.005 (3 %) | Schäfer & Turek, NNFM 52 (1996), doi:10.1007/978-3-322-89849-4_39 |
| C_l amplitude | 0.32 | informational | Braza 1986 |

## 2. Method

Delivered: masked-MAC fractional step (`tensorfvm.immersed`, pre-existing from
m9040) on a staircase cylinder mask, **new** control-volume momentum-budget
force (`immersed/forces.py`: `ControlBox`, `budget_terms`, `ForceRecorder`) and
gated shedding statistics (`verification/cylinder_re100.py:164`: half-amplitude
gate, zero-crossing St + independent zero-padded FFT cross-check); CLI entry
`python -m tensorfvm.verification.cylinder_re100`; grid ladder 12/20/32 cells
per D, 6001 steps each, rendered into `docs/case-cylinder-re100/`.

Audit method: referee = `tensor_benchmark_cases` (postgres `tensorplatform`),
not the delivered report. Checks executed: CLI `--help`; 8 cells/D end-to-end
run (1200 steps, 24.4 s); bit-for-bit re-derivation of the 32 cells/D first
force sample (1552.09585999); CSV forensics on the delivered history; the two
test modules re-run (`16 passed in 24.4 s`); direct SQL read of the referee
table; `maturity_gate.judge` replayed locally before landing the verdict.

## 3. Results

Referee rows: `cylinder-re100-unbounded | cd_mean | 5 % | TensorFVM | pending`
and `… | st | 3 % | TensorFVM | pending` — **the case never passed**.

| grid (cells/D) | steps | C_d,mean | err vs 1.34 | St | err vs 0.164 | C_l amp | referee |
|---|---|---|---|---|---|---|---|
| 12 | 6001 | 1.5193 | +13.38 % (>5 %) | 0.1441 | −12.11 % (>3 %) | 0.113 | FAIL |
| 20 | 6001 | 1.5093 | +12.63 % (>5 %) | 0.1551 | −5.44 % (>3 %) | 0.201 | FAIL |
| 32 | 6001 | nan | – | not reported | – | nan | run diverged at step ≈15 |

Successive differences: 12→20 ΔC_d −0.67 %, ΔSt +7.05 %; 20→32 **NaN** → no
mesh-convergence evidence exists.

## 4. Figures

See the three figures: the diverging 32 cells/D budget, the error ladder against
the registered bands, and the healthy 20 cells/D shedding signal.

## 5. Points to watch

1. **Blocking defect**: `verification/cylinder_re100.py:200` maps a non-finite
   lift record onto the same message as a quiet cylinder (`no vortex shedding …
   nan < gate 0.05`). The 32 cells/D drag budget overflows
   (1.3e26 → 1.8e171 → 6.8e309 → nan, t = 0.20…0.32; 5986/6001 rows non-finite)
   and `README.md`/`ladder.json`/`report.md` all report it as a physics gate;
   the ladder even prints `20 -> 32 … +nan %` as a grid difference. ~2639 s of
   cluster time were spent stepping a NaN field.
2. **Reproducibility**: `runs/` and `artifacts/` were never committed
   (`?? runs/ ?? artifacts/`); the raw histories existed only on a prunable
   worktree. The audit copied the compact payloads + sha256 manifest into
   `docs/audit/board-audit-m21018/artifacts/`.
3. **Test gap**: 16 tests assert good invariants (control-volume independence,
   mask-area convergence, St recovery, St withheld when nothing sheds) but none
   feeds a non-finite history — the only failure that occurred in production.
4. **Integration**: no immersed entry in `backend_registry.py`, no
   `/calculations/*` route; the new modules are called only by the case harness
   and tests → L3 not met.
5. Report shape: per-grid report has 5 of the 6 required sections (no
   limitations section, no explicit "<3 %?" sentence).

## 6. Error decision (explicit)

| quantity | got (best grid 20 cells/D) | reference | error | <3 %? |
|---|---|---|---|---|
| C_d,mean | 1.5093 | 1.3400 | +12.63 % (band 5 %) | **no** |
| St | 0.1551 | 0.1640 | −5.44 % (band 3 %) | **no** |
| C_l amplitude | 0.2005 | 0.3200 | −37.3 % | no (informational) |

Both registered metrics miss tolerance; convergence cannot be assessed because
the finest grid diverged.

**Verdict: L2** (independent functional implementation: real code, executed
entry, real run artifacts, invariant tests; no platform caller → not L3; both
registered metrics out of tolerance and no convergence → not L4). Lane landed in
`tensor_maturity_verdicts` (row 878, scope `m21018 GAP-DIFFLBM-EXT-LIT-CYL`,
mission 21105, `verdict_by='board-audit'`, `maturity_gate` replay
`gate_level=2`). Disposition: **request changes / GAP stays open** — split
`diverged` from `no shedding` and fail fast, commit raw histories, add a NaN
regression test, and buy resolution or switch to body-fitted to get inside
5 %/3 %.
"""


def login() -> str:
    req = urllib.request.Request(BASE + "/auth/login",
                                data=json.dumps({"username": USER, "password": PASS}).encode(),
                                headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["access_token"]


def main(argv=None) -> int:
    token = login()
    figures = []
    for name, caption in FIGS:
        png = (ROOT / "figures" / name).read_bytes()
        figures.append({"name": name, "caption": caption,
                        "png_base64": base64.b64encode(png).decode()})
    payload = {
        "title": "Board audit m21018 GAP-DIFFLBM-EXT-LIT-CYL (TensorFVM, cylinder Re=100): "
                 "honest-FAIL, L2 — finest grid diverged, both registered metrics out of tolerance",
        "summary": "Closure check of m21018 (PR #719). Referee table says cd_mean 5 % / st 3 %, "
                   "status pending; delivered best grid gives Cd +12.63 %, St -5.44 %. The 32 "
                   "cells/D ladder grid overflows to NaN at step ~15 and the gate reports it as "
                   "'no vortex shedding'. Graded L2, GAP stays open.",
        "scenario_key": "board-audit",
        "mission_id": 21105,
        "tags": ["board-audit", "TensorFVM", "GAP-DIFFLBM-EXT-LIT-CYL", "cylinder-Re100",
                 "adversarial-review", "honest-FAIL", "m21018"],
        "markdown": MD,
        "figures": figures,
    }
    out = Path("/tmp/report_m21105.json")
    out.write_text(json.dumps(payload))
    if "--dry-run" in (argv or sys.argv[1:]):
        print("dry-run: wrote", out, out.stat().st_size, "bytes,", len(figures), "figures")
        return 0
    req = urllib.request.Request(BASE + "/tensor/reports", data=out.read_bytes(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + token}, method="POST")
    with urllib.request.urlopen(req, timeout=180) as r:
        print(r.status, r.read()[:400].decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())