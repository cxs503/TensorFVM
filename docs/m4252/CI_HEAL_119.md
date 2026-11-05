# CI heal: why `lint (advisory)` was permanently red on PR #119

Mission m4252 · repository TensorFVM · branch `agent/m3730` (PR #119)
· workflow `.github/workflows/ci.yml`

## 1. Symptom

PR #119 recorded two `lint (advisory)` check-runs (one from the `push` run, one
from the `pull_request` run) with `conclusion = failure`, while both `test
(3.11)` and `test (3.12)` were green **and the workflow runs themselves
reported `conclusion = success`**. The disagreement between the run and its
job is the whole story.

## 2. Root cause

The job was declared advisory at the *job* level:

```yaml
  lint:
    name: lint (advisory)
    continue-on-error: true
```

`jobs.<id>.continue-on-error` only protects the **workflow run**: GitHub still
records the job's own conclusion as `failure`. Every consumer of the per-check
signal therefore saw a red check:

* `tools/pr_triage.py` reads `commits/{sha}/check-runs` and printed
  `lint (advisory)=failure` for every branch in this repository;
* the merge governor's `pr_checks_green()` has to special-case
  `"lint (advisory)"` in `ignore_checks` for TensorFVM, i.e. the check has to
  be ignored rather than satisfied.

So the advisory intent written in the workflow comment was never expressed in
the recorded signal, and no push to `src tests scripts` could ever turn the
check green — the tree carries a large pre-existing backlog:

| whole-tree measurement (unchanged by this PR) | value |
|---|---|
| `ruff check src tests scripts` | Found 9822 errors. |
| files carrying at least one finding | 426 |
| `ruff format --check src tests scripts` | 554 files would be reformatted, 5 files already formatted |

The backlog is dominated by `E701/E702` (semicolon-packed statements), which
is a tree-wide style decision, not something a CI-heal branch may rewrite.

## 3. The red was reproducible on the PR's own files

The two Python files that PR #119 adds were themselves not clean, so even a
correctly-scoped gate would have been red:

```
$ ruff check scripts/gen_suboff_radius_plot.py tests/test_suboff_validation_integration.py   # PR head 3b50c61
/tmp/r119/test_suboff_validation_integration.py:49:5: F841 Local variable `neighbor` is assigned to but never used
$ ruff format --check scripts/gen_suboff_radius_plot.py tests/test_suboff_validation_integration.py
Would reformat: /tmp/r119/gen_suboff_radius_plot.py
Would reformat: /tmp/r119/test_suboff_validation_integration.py
2 files would be reformatted
```

## 4. Fix (two parts)

### 4.1 The PR's own files

* `tests/test_suboff_validation_integration.py`: dropped the unused mock field
  `neighbor` (`F841`) — the mock is only read through `owner`,
  `patch_labels`, `vertices`, `face_vertices`, `face_centers`.
* both files passed through `ruff format` (formatting only, no behaviour
  change).

### 4.2 The workflow (root cause)

`.github/workflows/ci.yml`, `lint` job (job name kept for check-run
continuity):

* the job-level `continue-on-error` is removed;
* *both* whole-tree steps now carry a **step-level** `continue-on-error: true`,
  which is what "advisory" actually means for the recorded signal — same
  pattern the repository already uses for the advisory mypy step in TensorLBM;
* a new **blocking ratchet** step lints exactly the `.py` files the pull
  request touches (`git diff --name-only --diff-filter=ACMR
  "origin/${{ github.base_ref }}...HEAD" -- '*.py'`) with both
  `ruff check` and `ruff format --check`. The legacy backlog cannot fail the
  job, but no new debt can enter the tree;
* `actions/checkout` gets `fetch-depth: 0` so the ratchet can resolve the base
  ref. `push` events are already covered by the pusher-side ruff pre-gate,
  which lints the changed files of the commit it is about to push.

The structure is pinned by `tests/test_ci_lint_ratchet_contract.py` (4 tests)
so the job-level neutralisation cannot come back silently.

## 5. Verification (all numbers measured, none hand-copied)

Post-fix, the exact command the new CI step runs:

```
$ git diff --name-only --diff-filter=ACMR origin/main...HEAD -- '*.py'
scripts/gen_suboff_radius_plot.py
tests/test_suboff_validation_integration.py
$ ruff check $FILES            # rc=0
All checks passed!
$ ruff format --check $FILES   # rc=0
2 files already formatted
```

Contract test, negative control (old `ci.yml` restored into the worktree):

```
$ git show HEAD:.github/workflows/ci.yml > .github/workflows/ci.yml
$ pytest tests/test_ci_lint_ratchet_contract.py -q
FAILED test_lint_job_is_not_neutralised_at_job_level
FAILED test_whole_tree_ruff_steps_stay_advisory_per_step
FAILED test_changed_file_ratchet_is_blocking
FAILED test_ratchet_has_history_to_diff_against
```

and on the fixed workflow:

```
$ pytest tests/test_suboff_validation_integration.py tests/test_ci_lint_ratchet_contract.py -q
13 passed, 2 warnings in 6.99s
```

The workflow file itself parses (`yaml.safe_load`), the job keeps the name
`lint (advisory)`, and the job has no job-level `continue-on-error`.

## 6. Scope / limitations (honest)

* The **whole-tree** ruff backlog (~Found 9822 errors, 554 files would be reformatted, 5 files already formatted)
  is **not** fixed here: rewriting thousands of semicolon-packed statements and
  reformatting 554 files in a CI-heal branch would
  dwarf the PR and collide with every open branch. The advisory steps keep
  that signal visible, and the ratchet stops it from growing.
* Deleted files are excluded from the ratchet (`--diff-filter=ACMR`), so
  deleting a non-conforming file is not blocked.
* Renames are evaluated on the new path only.
* `push` events do not run the ratchet step (the pusher pre-gate covers the
  changed files there); a direct push to a branch therefore gets the advisory
  signal only.
