# Lint gate: root cause, repair and the remaining ratchet

Status: repaired 2026-11-05 (TensorFVM PR #114 repair round, mission m4193).
Rule set in force: ruff defaults (`E4,E7,E9,F`) + `ruff format`, no
`[tool.ruff]` section in `pyproject.toml`.

## Symptom

The `lint (advisory)` job of PR #114 was red while both test jobs passed:

```
127156364702 test (3.11)     success
127156364276 test (3.12)     success
127156363964 lint (advisory) failure
```

## Root cause 1 — the change itself was not ruff-clean (fixed here)

PR #114 added `src/tensorfvm/verification/scalar_transient_diffusion.py` and
`tests/test_scalar_transient_diffusion.py`. Both new files were unformatted, and
the module carried an unused local:

```
$ ruff check src/…/scalar_transient_diffusion.py tests/test_scalar_transient_diffusion.py
F841 Local variable `x` is assigned to but never used
  --> src/tensorfvm/verification/scalar_transient_diffusion.py:96:5
$ ruff format --check <both files>
Would reformat: src/tensorfvm/verification/scalar_transient_diffusion.py
Would reformat: tests/test_scalar_transient_diffusion.py
```

Repair: the unused local `x = mesh['x_centers']` in
`solve_transient_diffusion` was dropped (the cell centres are consumed by the
caller, which builds the reference profile on the same mesh) and both modules
were formatted with `ruff format`. No behavioural change — the module's test
file is 7 passed before and after the edit, and the converged numbers quoted
below were produced by the repaired module.

## Root cause 2 — the whole-tree gate can never be green (reported, not silently patched)

The job ran ruff over the whole tree, and the tree carries a pre-existing
backlog:

```
$ ruff check src tests scripts --statistics
7601 E702  multiple-statements-on-one-line-semicolon
1852 E701  multiple-statements-on-one-line-colon
  98 F401  unused-import
  84 E401  multiple-imports-on-one-line
  69 E731  lambda-assignment
  51 F841  unused-variable
  39 E402  module-import-not-at-top-of-file
  20 E741  ambiguous-variable-name
   6 F541  f-string-missing-placeholders
   3 E703  useless-semicolon
Found 9823 errors.

$ ruff format --check src tests scripts
556 files would be reformatted, 2 files already formatted
```

Consequences:

* the check-run is red for **every** PR, not only #114, so it carries no
  information about a change — which is exactly how one new `F841` got in
  unnoticed;
* it cannot be repaired from an agent worktree:
  - **`.github/` is outside the agent sparse-checkout definition** of this
    worktree (`git sparse-checkout list` → `/src/ /tests/ /scripts/
    /pyproject.toml /README.rst /README.md /docs/ /outputs/`), so `git add`
    refuses the workflow file — the machine-level gate, not an instruction,
    keeps agents out of CI definitions;
  - the only in-scope repair for the `ruff format --check` leg would be to
    reformat all 556 files, i.e. touch every open PR branch (the silent-loss
    hazard `merge_governor` warns about for same-file siblings).

What is shipped instead, inside the writable paths:

1. `scripts/lint_changed_files.sh <base>` — the scoped gate as a runnable,
   testable artifact: `ruff check` + `ruff format --check` over the Python
   files a change touches (`git diff --diff-filter=ACMR <merge-base> --
   src tests scripts` plus untracked files). It reproduces the CI red locally
   and is the piece the owner can wire into CI (below). It mirrors the rule the
   platform already applies before pushing a branch
   (`tp_pusher._ruff_gate`: "只查本次改动的 .py，绝不动整仓").
2. The ready-to-apply workflow patch (owner action, since `.github/` is not
   agent-writable):

   ```yaml
   # .github/workflows/ci.yml — lint job, replacing the two ruff steps
   - uses: actions/checkout@v4
     with:
       fetch-depth: 0          # merge-base diff needs history
   ...
   - name: Lint changed files (ruff check + format)
     env:
       BASE_REF: ${{ github.base_ref }}
       BEFORE_SHA: ${{ github.event.before }}
     run: |
       set -euo pipefail
       base=""
       if [ -n "${BASE_REF}" ]; then
         git fetch --no-tags --quiet origin "+refs/heads/${BASE_REF}:refs/remotes/origin/${BASE_REF}"
         base="origin/${BASE_REF}"
       elif [ -n "${BEFORE_SHA}" ] && [ "${BEFORE_SHA}" != "0000000000000000000000000000000000000000" ]; then
         base="${BEFORE_SHA}"
       fi
       if [ -z "${base}" ] || ! git rev-parse --verify --quiet "${base}^{commit}" >/dev/null; then
         git fetch --no-tags --quiet origin "+refs/heads/main:refs/remotes/origin/main"
         base="origin/main"
       fi
       bash scripts/lint_changed_files.sh "${base}"
   - name: Legacy backlog (informational, non-blocking)
     if: always()
     continue-on-error: true
     run: ruff check src tests scripts --statistics | tail -n 20 || true
   ```

   Ratchet order: backlog → 0, then delete the scoped step, make the whole-tree
   run blocking and drop `continue-on-error`.

## Evidence

Reproduction of the CI red on the pristine PR head, then green after the repair
(exact `run:` block of the intended workflow, executed locally):

```
# 1) pristine PR #114 content, PR-shape base (origin/main == the PR base)
$ git checkout 780f328 -- src/tensorfvm/verification/scalar_transient_diffusion.py \
                           tests/test_scalar_transient_diffusion.py
$ bash scripts/lint_changed_files.sh origin/main
lint base: origin/main -> merge-base db97955 (Merge #78 (m3149) via merge-governor)
linting 2 changed python file(s):
  src/tensorfvm/verification/scalar_transient_diffusion.py
  tests/test_scalar_transient_diffusion.py
F841 Local variable `x` is assigned to but never used
  --> src/tensorfvm/verification/scalar_transient_diffusion.py:96:5
Would reformat: src/tensorfvm/verification/scalar_transient_diffusion.py
Would reformat: tests/test_scalar_transient_diffusion.py
2 files would be reformatted
FAIL: changed files are not ruff-clean                        # exit 1

# 2) repaired tree, PR-shape base
$ bash scripts/lint_changed_files.sh origin/main
All checks passed!
2 files already formatted
PASS: changed files are ruff-clean (check + format)            # exit 0

# 3) repaired tree, push-shape base (PR head 780f328)
$ bash scripts/lint_changed_files.sh 780f328
lint base: 780f328 -> merge-base 780f328 (780f328 m3679: matured mission deliverables)
PASS: changed files are ruff-clean (check + format)            # exit 0
```

Negative control (the gate is not vacuous — probe file added, measured,
deleted):

```
$ printf 'def f():\n    y = 1\n    return 2\n' > tests/_negctl_lint_probe.py
$ bash scripts/lint_changed_files.sh 780f328
F841 Local variable `y` is assigned to but never used --> tests/_negctl_lint_probe.py:2:5
FAIL: changed files are not ruff-clean                         # exit 1
```

Functional regression check after the repair:

```
$ python -m pytest tests/test_scalar_transient_diffusion.py -q
7 passed, 2 warnings in 8.29s
```

## Merge path for the advisory check

The red `lint (advisory)` check-run does not block the merge queue as
configured: the workflow run conclusion is `success` (the job is
`continue-on-error: true`, `CI pull_request ... success`), and
`merge_governor.json` lists `"TensorFVM": ["lint (advisory)"]` under
`ignore_checks`, so `pr_checks_green()` skips that check name and PR #114
satisfies `mergeable=True` + "green" for the governor. The residual red
check-run is reported here rather than silenced: it still mislabels the PR in
the UI, and it stays red for every future PR until root cause 2 is retired.

## Known limits

* The 9823-finding backlog and 556 unformatted files are untouched by design.
* Changed-file scoping (when wired) re-lints only the files a change touches;
  a PR that *moves* legacy code inherits the legacy findings of the moved file.
* The scoped script's base resolution relies on `git merge-base`; it degrades to
  `origin/main` for branch creation / unreachable parent commits (all four event
  shapes were exercised in this round).