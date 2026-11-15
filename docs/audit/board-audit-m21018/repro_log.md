# m21018 board audit — reproduction log

All commands were run on the audit worktree `/root/tp-agent/work/m21105`
(TensorFVM, branch `agent/m21105`) and on the subject worktree
`/root/tp-agent/work/m21018` (branch `agent/m21018`, merged as PR #719 /
commit `69311e24`). Timestamps UTC, 2026-11-15.

## R1 — entry point exists and matches the documented repro command

```
cd /root/tp-agent/work/m21018
PYTHONPATH=src python3 -m tensorfvm.verification.cylinder_re100 --help
```
→ exit 0. Accepts `--backend {immersed,body-fitted} --cells-per-diameter --grid
--dt --t-end --steps --warmup --box --perturbation --log-every --output`, i.e.
exactly the flags used by `docs/case-cylinder-re100/README.md` §Reproduction.
Verdict: repro command is *executable*, not decorative.

## R2 — reduced-fidelity end-to-end run (audit, 8 cells/D)

```
OMP_NUM_THREADS=4 PYTHONPATH=src python3 -m tensorfvm.verification.cylinder_re100 \
    --backend immersed --cells-per-diameter 8 --dt 0.02 --t-end 24 \
    --output /tmp/repro_m21105_cpd8
```
→ 1200 steps in 24.4 s; wrote `result.json`, `benchmark.json`, `errors.json`,
`report.md`, `force_history.csv`, 4 png. Numbers: `cd_mean = 1.5334`
(ref 1.34, +14.43 %), `St` withheld, gate reason
`no vortex shedding: lift half-amplitude 4.150e-03 < gate 0.05`.
So the pipeline input→compute→output is real and the gate is active.

## R3 — bit-for-bit check of the subject run's first step (32 cells/D)

```
OMP_NUM_THREADS=4 PYTHONPATH=src python3 -u -m tensorfvm.verification.cylinder_re100 \
    --backend immersed --cells-per-diameter 32 --dt 0.02 --steps 30 \
    --output /tmp/repro_cpd32_div
```
→ first sample `t=0.02 Cd=+1552.0959 Cl=-2.4126`, identical to row 1 of the
delivered `runs/case-cylinder-re100/cpd32/force_history.csv`
(`0.020000,1552.09585999,-2.41260068`). The code state reproduces the subject
artifacts; the audit then stopped the 32 cells/D local run (>100 s per step at
this resolution locally, no value in re-deriving a divergence already recorded
in the artifacts).

## R4 — divergence evidence in the delivered artifact (no rerun needed)

```
python3 - <<'EOF'
import csv, math
rows = list(csv.DictReader(open('runs/case-cylinder-re100/cpd32/force_history.csv')))
print(next((i, r) for i, r in enumerate(rows)
           if not all(math.isfinite(float(r[k])) for k in ('drag', 'lift'))))
EOF
```
→ first non-finite row index 15 (`t = 0.32`). Rows 11-15 (indices 10-14) are
`1.3e26 → 1.8e171 → 6.8e309 → 6.3e315 → nan`: geometric blow-up then overflow.
5986 of 6001 rows are non-finite, i.e. the finest ladder grid never produced
physics, yet `report.md` and `ladder.json` label it
`GATED: no vortex shedding`.

## R5 — tests

```
cd /root/tp-agent/work/m21018
OMP_NUM_THREADS=4 PYTHONPATH=src python3 -m pytest \
    tests/test_cylinder_re100_case.py tests/test_case_cylinder_re100.py -q
```
→ `16 passed in 24.41 s` (reproduced twice). The tests assert invariants
(control-volume independence of the force, mask-area convergence, St recovery
from a synthetic signal, St withheld when nothing sheds) — but none of them
feeds a non-finite history, so the cpd32 failure mode is untested.

## R6 — benchmark judge (the only referee)

```
psql -U postgres -d tensorplatform -tAc \
 "SELECT case_name,metric,tolerance,repo,status FROM tensor_benchmark_cases \
  WHERE anchor_slug='GAP-DIFFLBM-EXT-LIT-CYL'"
```
→ `cylinder-re100-unbounded | cd_mean | 5% | TensorFVM | pending`
  `cylinder-re100-unbounded | st      | 3% | TensorFVM | pending`
Got values (from `artifacts/cpd20/result.json`, the best grid):
`cd_mean 1.50926` (+12.63 % vs 1.34) → outside 5 %;
`st 0.15508` (-5.44 % vs 0.164) → outside 3 %.
Both registered metrics FAIL and the case was never promoted out of `pending`.

## R7 — independent 12 cells/D rerun (cluster, job cj3192)

```
run_compute submit: OMP_NUM_THREADS=4 PYTHONPATH=src python3 -m \
  tensorfvm.verification.cylinder_re100 --backend immersed \
  --cells-per-diameter 12 --dt 0.02 --t-end 120 --output runs/audit_m21105_cpd12
```
Submitted 2026-11-15 23:56 UTC to the compute gate (4 cores, 2400 s cap).
Status at hand-off: still `queued` behind 33 in-flight jobs, so the audit does
**not** claim an independent numeric reproduction of `cd_mean = 1.5193`; the
claim is carried only by the delivered artifact + the R2/R3 cross-checks.
This is recorded as a check *not completed*, not as a pass.