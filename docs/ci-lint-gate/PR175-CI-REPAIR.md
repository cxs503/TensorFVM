# PR #175 CI repair (branch `agent/m4249`)

## Symptom

Job `test (3.11)` on the pull request was red with exactly one failing test out of
1 407 collected items:

```
FAILED tests/test_multi_grid_gci.py::test_reference_agreement_accepts_a_multi_grid_study
       - AttributeError: 'MultiGridStudy' object has no attribute 'fallback_reason'
1 failed, 1386 passed, 18 skipped, 2 xfailed in 934.22s
```

The `lint (advisory)` job already passed, so the red was not produced by the lint
gate this PR introduces; it was a latent defect in the GCI verification helpers
that the branch inherited and that gates the merge either way.

## Root cause

`reference_agreement()` (`src/tensorfvm/verification/richardson.py`) is written
against both study types: its docstring states that `asymptotic` and
`fallback_reason` are echoed into the result so that a coverage verdict computed
on a fall-back band travels with that qualifier.  `RefinementStudy` (three-grid)
carries the field; `MultiGridStudy` (N >= 3 levels, `verification/multi_grid.py`)
decided on the same fall-back — `fallback = not (monotone and finite p and finite L)`,
`convergence = 'inconsistent'`, `safety_factor = 3.0` — but never recorded *which*
rule fired, so the echo raised `AttributeError`.

The caller is committed and intentional: `test_reference_agreement_accepts_a_multi_grid_study`
asserts the band bookkeeping applies unchanged to an N-grid study, and
`scripts/audit_multi_grid_gci.py` reports GCI coverage for the same studies.
The data model was therefore incomplete, not the caller.

## Fix (3 hunks, one production file, one test file)

`src/tensorfvm/verification/multi_grid.py`

* `MultiGridStudy` gains `fallback_reason: str` (next to `convergence`, matching
  the field order of `RefinementStudy`).
* `multi_grid_order()` derives it with the same vocabulary as the three-grid
  routine: `'non-monotone'` when the ladder does not decay one-sided,
  `'implausible-order'` when it decays but the least-squares fit degenerates
  (non-finite `p` or `L`), `'none'` otherwise.  No numerical result changes:
  `order_used`, `gci`, `band`, `safety_factor` and `asymptotic` are untouched.
* The dead store `res = np.full(h.shape, np.nan)` in the fall-back branch is
  removed; it was the only pyflakes finding (`F841`) the changed-file ratchet
  reported for this file.

`tests/test_multi_grid_gci.py`

* `test_fallback_reason_is_echoed_for_multi_grid_studies` pins the new field:
  `'none'` for a converged ladder, `'non-monotone'` for a broken one, and the
  same string echoed through `reference_agreement()`.

## Verification

```
$ python3 -m pytest tests/test_multi_grid_gci.py -q            # was: 1 failed
36 passed
$ python3 -m pytest tests/test_multi_grid_gci.py tests/test_gci_fallback_band.py \
    tests/test_richardson_gci.py tests/test_gci_band_coverage.py tests/test_lint_ratchet.py -q
74 passed
$ ruff check src tests scripts                                  # blocking gate of this PR
All checks passed!
$ ruff check --isolated --select E9,F src/tensorfvm/verification/multi_grid.py \
    tests/test_multi_grid_gci.py                                # changed-file ratchet
All checks passed!
$ python3 -m pytest -q                                          # whole suite, Python 3.11.6
1369 passed, 37 skipped, 2 xfailed, 50 subtests passed, 0 failed (1731s)
```

The local run is Python 3.11.6 with `gmsh` absent, hence 37 skips against the CI
job's 18 (optional mesh runtime); the failing test is reproduced and green under
both.  Final acceptance is the pull-request check suite itself.

## Scope

Only `src/tensorfvm/verification/multi_grid.py` and
`tests/test_multi_grid_gci.py` are modified; the lint-gate changes of PR #175
(`pyproject.toml`, `.github/workflows/ci.yml`, `scripts/lint_ratchet.py`,
`tests/test_lint_ratchet.py`, `docs/ci-lint-gate/`) are left untouched.