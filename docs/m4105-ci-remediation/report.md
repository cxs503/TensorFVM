# m4105 — PR #126 CI remediation (branch `agent/m3884` → `agent/m4105`)

**Scope:** make the PR's `test (3.11)` / `test (3.12)` jobs green without weakening
any verification gate and without editing published reference data.
**Verdict:** the red was a *data* regression, not a solver regression. Reverting the
reference edit restores the suite. No solver, mesh or benchmark file was touched.

## 1. What was red and why

PR #126 (head `48fd44b`) resolved the `#82` merge conflict by taking the
`agent/m3240` side of `src/tensorfvm/lid_cavity_reference.py`, i.e. it replaced the
published Ghia (1982) Table II entry at Re = 400, y = 0.9063
(`-0.23827` → `-0.38760`, the solver's Richardson limit), then rewrote
`tests/test_lid_cavity_re1000_evidence.py` (strict `xfail` → passing test) and the
committed ladder bundle `docs/verification-lid-cavity-ladder/summary.json`
(re-run, 300 float fields changed, with the v-errors rescored against the edited
table).

That edit is rejected by the repository's own reference audit. Local reproduction
on the PR head (Python 3.11, CPU, `python3 -m pytest -q`):

```
FAILED tests/test_ghia_reference_consistency.py::test_the_only_corner_in_all_six_tables_is_the_registered_dispute
FAILED tests/test_ghia_reference_consistency.py::test_the_gap_survives_extrapolation_while_its_neighbours_do_not
FAILED tests/test_ghia_reference_consistency.py::test_the_metric_path_cannot_manufacture_the_gap
3 failed, 5 passed, 4 skipped
E  assert False                       # audit["ok"]
E  assert 1.8130743130839733e-05 > (5 * 0.004308301519374724)
E  assert (1.8130743130839733e-05 / 0.0004887179715446441) > 100
```

## 2. Fix (three files reverted to the `main` content, nothing else)

| File | Change |
|---|---|
| `src/tensorfvm/lid_cavity_reference.py` | restore `(0.9063, -0.23827)`; drop the round-1 `"resolution"` field so the register no longer claims the debt closed |
| `tests/test_lid_cavity_re1000_evidence.py` | restore the strict `xfail` that documents the Re = 400 v-centreline open question |
| `docs/verification-lid-cavity-ladder/summary.json` | restore the committed evidence bundle (scored against the published table; the round-1 bundle re-ran the ladder and rescored it, which is neither needed for a CI repair nor comparable with earlier records) |

Rationale and numbers: `docs/verification-lid-cavity-correction/report.md`. The
register's own policy is the deciding argument — "a dispute is *not* a licence to
edit the transcription" — and the proposed value is the solver's own extrapolation
(0.6 % from the registered limit), so adopting it would make the 3 % check
circular.

New, reproducible evidence in this branch:

* `scripts/audit_m4105_reference_edit_rejection.py` — positive/negative control on
  the reference audit (published table clean and registered; patched table loses
  the fingerprint), writes `docs/m4105-ci-remediation/reference_edit_audit.{json,png}`.
* `scripts/m4105_ci_remediation_report.py` — regenerates the error table
  (`docs/m4105-ci-remediation/metrics_table.md`, `metrics.json`) straight from the bundle, so
  no digit in the reports is hand-copied.

## 3. Verification run in this worktree

```
python3 -m pytest -q -p no:cacheprovider \
    tests/test_ghia_reference_consistency.py \
    tests/test_lid_cavity_re1000_evidence.py \
    tests/test_lid_cavity.py
27 passed, 1 skipped, 1 xfailed
```

The `xfailed` row is the registered dispute (`test_re400_v_centreline_discrepancy_is_still_present_and_unexplained`);
it is expected to stay xfail until the printed table or an independent DNS arrives.
A full `pytest -q --timeout=1800` run of the worktree is recorded in
§ `docs/m4105-ci-remediation/pytest-full.log` (see the accompanying file).

## 4. lint (advisory) — honest status

`ruff check src tests scripts` reports 9822 errors and `ruff format --check`
would reformat 554 files **on `main` as well**: the job is `continue-on-error:
true` on purpose ("the tree carries a large pre-existing ruff backlog"). Fixing
that backlog is a separate, repository-wide cleanup and is explicitly *not*
attempted here (it would be an unrelated, unreviewable diff). The files this
branch touches pass `ruff check` (`src/tensorfvm/lid_cavity_reference.py`,
`tests/test_lid_cavity_re1000_evidence.py` and the two new scripts), and the two
new scripts are `ruff format` clean.

## 5. Not done / limits

* No solver, mesh, benchmark or CI-workflow change; the ladder was not re-run
  (the reverted bundle is the previously committed evidence).
* The Re = 400 v-centreline debt is **not** repaid — it is re-opened as the
  documented open question, which is the honest state of the evidence.
* `test (3.11/3.12)` green is claimed for the local tree; the PR-level CI result
  must be read off the GitHub run after the branch is pushed.