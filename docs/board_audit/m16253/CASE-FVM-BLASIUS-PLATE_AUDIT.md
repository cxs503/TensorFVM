# CASE-FVM-BLASIUS-PLATE — board audit of m16219 (TensorFVM)

Audit task **m16253** (交差核验单, not a re-delivery). Subject: mission **m16219**,
repo TensorFVM, branch `agent/m16219`, delivery commit **`4f87f691`**
("CASE-FVM-BLASIUS-PLATE: laminar plate c_f/H vs Blasius with machine gate, artefacts
and driver caller"), status `succeeded` at 2026-11-12 11:16:47Z.

Disposition: **approve with notes** — verdict **L4** for the parabolised boundary-layer
rung; the anchor-naming 2-D FVM rung stays WITHHELD (see F3, and the scope limit
statement at the end). Verdict row: `tensor_maturity_verdicts` id **460**
(scope `m16219 CASE-FVM-BLASIUS-PLATE`, gate_state `pending` at insert time,
auditor dry-run of `maturity_gate.judge` re-derives **L4** from the shipped evidence).

## 1. Independence and limitations of this audit

* The auditor ran the delivery **without modifying it**: the m16219 worktree was read
  only; all outputs went to `runs/m16253/`.
* The audited artefacts were frozen into `docs/board_audit/m16253/refs/` with sha256 +
  git provenance (`refs/manifest.json`), because the delivery is **branch-only** and
  the worktree is recyclable (F2).
* Not independent: the reference constants (`0.664`, `1.7208`, `2.5916`) are the
  delivery's own literature transcription. I re-derived them from the ODE
  (`2 f''(0) = 0.664115`, Δ = +0.017 % vs 0.664; `H_ODE = 2.5911`, Δ = +0.019 % vs
  2.5916) and cross-checked against the case sheet, but I did not consult the printed
  Schlichting table myself (no library copy in the sandbox). **External anchor is
  therefore second-hand**: literature DOI/table reference quoted by the delivery, not
  re-read by the auditor.
* The delivering device is the repo's own parabolised boundary-layer march with
  `model='laminar'`. It is not an independent code (no external solver cross-check was
  made), so this is *code-to-analytic* verification, not code-to-code.

## 2. What the run actually does (reconstructed execution path)

`scripts/run_case_blasius_plate.py --configs march` → `case_blasius_plate.run_march_case`
(`src/tensorfvm/case_blasius_plate.py:448`) → `rans_ship_resistance.boundary_layer.march_boundary_layer(x, ue, nu,
model='laminar', trip_x=None, ny=256, stretching=8.0)` (`boundary_layer.py:170`) →
per-station Picard solve of the thin-shear-layer momentum equation → `tau_w = nu·du/dy`
(`boundary_layer.py:370-372`), `theta/delta*` by trapezoid on the wall-normal profile
(`:381-383`) → `compare_against_reference` against `0.664/√Re_x`, `H = 2.5916` with a
3 % gate (`case_blasius_plate.py:169-181`).

The second device (`fvm`) drives the production 2-D SIMPLE solver
(`external_cached.CachedExternalFlowSolver`, AMG/ILU-GMRES) through the same gate, and
is **not** delivering (see F3).

## 3. Checks performed (all re-run today by the auditor)

| check | command | result |
|---|---|---|
| tests | `PYTHONPATH=src python -m pytest tests/test_case_blasius_plate.py -q` | **9 passed** (16.8 s) — `runs/m16253/pytest_audit.log` |
| repro_cmd | `PYTHONPATH=src python scripts/run_case_blasius_plate.py --configs march,march-sensitivity --out runs/m16253/repro` | exit 0, **6.3 s + 45.2 s**; `cf`/`H`/sensitivity rows **bit-identical** to the committed artefacts (`runs/m16253/repro_compare.json`, `identical_numbers: true`) |
| criterion re-score | delivered `march.json` vs literature | `cf` −0.7064 %, `H` −0.4688 %, tolerance 3 % → **PASS** |
| artefact presence | `refs/` snapshot manifest | 16/16 files present (REPORT.md, metrics.csv, march.json, march-sensitivity.json, summary.json, fvm-baseline.trace.jsonl, 4 figures) |
| report completeness | REPORT.md §1–§8 | six required elements present (case/reference, method, parameters, number table, figures, cautions, convergence, verdict) |
| IC independence (added) | `probe_march_ic.py` | control −0.71 %, **uniform (no BL) IC −1.02 % PASS**, linear whole-domain ramp −89.68 % (continuity residual 0.97) |
| startup-layer ladder (added) | `probe_march_consistency.py` | 1×δ99 −0.99 %, 3×δ99 −1.99 %, **10×δ99 −10.30 % FAIL** |
| momentum-integral cross-check (added) | same script | `2·dθ/dx` vs wall-shear `c_f` differ by **+1.8 %/+2.0 %/+2.3 %** (2/5/10-station span) |
| criterion registry | `SELECT … FROM tensor_benchmark_cases WHERE anchor_slug='CASE-FVM-BLASIUS-PLATE'` | **0 rows** (F1); de-facto criterion in `tensor_case_pool` id 109 |
| cluster lever | `cluster_jobs` 1274/1276 | both still **`queued`**, submitted with `ref=main` (F3) |

## 4. Findings, severity-ranked

**F1 — high — the benchmark criterion is not registered.** The mission's own judge
(`tensor_benchmark_cases`) has **0 rows** for this anchor, so the "sole arbiter" of the
case does not exist; the 3 % bar is only in `tensor_case_pool` id 109
(`tol_policy = "<3% | 指标: 局部摩阻 c_f 与形状因子 H vs Blasius 相似解 (Re_x=1e5)"`,
`status='dispatched'`, `linked_anchor=NULL`, `mission_id=16182` ≠ 16219). The delivered
numbers are inside that bar, but the *case bookkeeping was never closed*: the pool row
still looks untaken and unlinked. Same gap as m16206 (F-KMM). Remediation: register the
row (`anchor_slug='CASE-FVM-BLASIUS-PLATE'`, metric/tolerance from the pool row) and
flip the pool row to done+linked on acceptance.

**F2 — medium — the delivery is branch-only.** `4f87f691` is not in the ecosystem
mirror (`/root/tensor-ecosystem/TensorFVM` HEAD `0cd53fa5`, `case_blasius_plate.py`
absent, `grep -r blasius_plate` → nothing). Consequences: (a) the case-level caller
`verification/cases.py::blasius_plate` is invisible to `maturity_gate.callers_verified`
(ECO grep) — the same mechanism that reopened the previous verdict id 436
(`m16182` L3 → gate_level 1, evidence paths unresolvable); (b) users of `main` cannot
run the case. Remediation is merge, not code.

**F3 — medium — the WITHHELD 2-D FVM rung is explained by cost, but the real blocker is
dispatch.** Both cluster jobs submitted by the mission (`cj1274`, `cj1276`) were still
`queued` when the mission finished, and both were submitted with `ref=main` — where
`scripts/run_case_blasius_plate.py` (or `_pilot_blasius_fvm.py`) does not exist — so
even when scheduled they would fail on `FileNotFoundError`. The repo already exposes
`stage_dir` for exactly this. So the honest state is "FVM rung never ran", not "FVM rung
too expensive": the committed trace (200 iters, momentum `3.943e-03` vs `1e-5`,
`c_f = 6.345e-3` vs `2.0998e-3`) is a genuine non-converged record, but it is not
evidence that the elliptic device cannot deliver. **Remediation (next mission):**
resubmit `--configs fvm-baseline,fvm-wall-fine,fvm-stream-fine` with
`stage_dir=/root/tp-agent/work/<id>` (or `ref=agent/m16219` after merge), 2500 iters,
`--threads 6`, and add the missing "developed-BL" stopping criterion the delivery itself
flagged.

**F4 — low — the 3 % agreement is real but IC-conditioned.** The delivered march is
initialised with the *exact* Blasius profile at `x0 = 2.5e-3` (`boundary_layer.py:234-240`).
For ZPG that profile is the exact solution at every station, so the −0.71 % is largely
a *preservation* (manufactured-solution style) check. My probe halves that objection: a
**uniform** start (no boundary layer at all) still lands at **−1.02 % PASS**, and the
resolution ladder converges to −0.15 % at ny=512. But a startup layer 10× too thick
fails at **−10.30 %**, so the gate is met only when the startup layer is of the right
order — a limitation the REPORT does not state (it states only parabolisation and the
`H` truncation bias). Ask: add one sentence + the ladder row.

**F5 — low — the delivery mission left no platform report.** `tensor_dev_missions.result`
records the report POST as blocked by 401 and `report_id` is NULL for m16219, so the
six-element narrative exists only as a repo file (`docs/case-blasius-plate/REPORT.md`).
The repo copy is complete, so this is a packaging defect, not a science defect.

**No defect found in:** the gate arithmetic (`compare_against_reference`, 3 % strict
inequality), the `H`/`delta*` trapezoid integrals (test pins the δ99-truncation bias at
0.991), input rejection (`re_x ≤ 0`, bad arrays), the committed-artefact re-score test,
or the driver wiring test.

## 5. Overstated claims in the delivery

* `headline` says "补齐 L2/L3/L4"; L4 holds **only for the march rung** — the anchor
  name `CASE-FVM-…` advertises the finite-volume device, which remains WITHHELD. The
  REPORT itself is honest about this (§5); the mission headline is not. A stricter
  reading caps the case at **L3** until F3's rerun converges. I declared **L4** because
  the case sheet's metric (`c_f`, `H` vs Blasius at `Re_x=1e5`) is met by a real
  in-repo numerical device with error analysis, resolution study and a verified repro
  path — and I state the boundary here rather than hiding it.
* "ecosystem friction: cost" (§5 of REPORT) under-states F3: the jobs never ran.

## 6. Checks not run (unknown, not passed)

* The 2-D elliptic FVM device to convergence — `not_run` (would need the F3 resubmission;
  ~40 min/grid at 1 s/iteration, 4 grids).
* Cluster nodes — `not_run` from this audit (`cluster_jobs` 1274/1276 still queued;
  no submission made by the auditor).
* Independent reading of Schlichting Table 7.1 — `not_run` (no library copy).
* `tests/test_verification_ladder.py` (57 tests claimed by the mission) — `not_run`
  by me; I ran only the case's own 9 tests plus the repro.
* Turbulent / adverse-pressure-gradient applicability — out of scope by design.

## 7. Artefacts produced by this audit

```
docs/board_audit/m16253/CASE-FVM-BLASIUS-PLATE_AUDIT.md   (this file)
docs/board_audit/m16253/probe_march_ic.py                 IC-independence probe
docs/board_audit/m16253/probe_march_consistency.py        momentum-integral + IC ladder
docs/board_audit/m16253/snapshot_refs.py, refs/           frozen delivery + manifest
docs/board_audit/m16253/repro_compare.py                  delivered vs re-run numbers
docs/board_audit/m16253/insert_verdict.py                 verdict writer (gate dry-run first)
docs/board_audit/m16253/verdict_evidence.json             exact payload of row 460
docs/board_audit/m16253/figures/fig1..fig3.png            figures of the audit report
runs/m16253/repro/                                        independent re-run
runs/m16253/march_ic_probe.json, momentum_integral_probe.json
runs/m16253/repro_compare.json, pytest_audit.log, gate_dryrun.log
```

## 8. Disposition

**Approve with notes.** Verdict **L4** (march rung), gate re-derives L4. Reopen only if
F1/F2 remediation is required for acceptance bookkeeping; the science numbers are
confirmed to −0.71 % / −0.47 % against the literature reference, reproduced
bit-identically from the shipped `repro_cmd`.