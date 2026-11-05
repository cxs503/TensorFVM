# CI lint policy (ruff pin + changed-files ratchet)

## What is red, and why

`lint (advisory)` is a two-command job (`ruff check`, `ruff format --check` over
`src tests scripts`).  It has been red since the ruff backlog entered `main`, and
the first attempt to heal it (PR #177, branch `agent/m4253`) did not land:

| measurement (ruff 0.15.0, tree at `agent/m4253` base) | value |
|---|---|
| `ruff check src tests scripts` | **8517** findings (7651 `E702`, 1852 `E701`, 120 `F401`, 84 `E401`, 71 `E731`, 57 `F841`, …) |
| `ruff format --check src tests scripts` | **532** files would be reformatted, 79 already formatted |

The attempted fix was a whole-tree reformat (611 files) plus a `[tool.ruff]`
ignore list and a pinned linter.  Only **80 files landed** on the branch: the
delivery pipeline builds one commit per mission from at most 80 changed files
(`MAX_FILES`), so the commit message reads
`m4253: matured mission deliverables（536 个文件被门禁滤除）`.  The tree therefore
carries the *policy* half of that change (pin + docs) and none of the
*formatting* half — the gate went from red-for-one-reason to red-for-the-same
reason, and PR #177's CI could never turn green no matter how often it was
re-healed.

Conclusion: **a whole-tree reformat is not deliverable through this pipeline**,
so the gate must be scoped instead of the tree being reformatted.

## What the gate does now

Two steps, both in the `lint` job of `.github/workflows/ci.yml`:

1. **Fatal rules, whole tree** —
   `ruff check --select E9,F63,F7,F82 src tests scripts`.
   These rules (syntax errors, undefined names, broken comparisons, invalid
   escape sequences) are *currently clean* over the entire tree, and their
   findings are real defects, so they stay a whole-tree gate: they cannot be
   dodged by only linting touched files.

2. **Style rules and formatting, changed files only** —
   `python scripts/ci_lint_ratchet.py --base origin/main`
   runs `ruff check` and `ruff format --check` over exactly the Python files the
   change touches (`git diff --diff-filter=ACMR origin/main...HEAD -- '*.py'`,
   minus the evidence trees `docs/`, `artifacts/`, `runs/`).  This is a ratchet:
   **new and modified code cannot add findings**, while the pre-existing backlog
   is frozen and drains as files get edited.  Both new files and rewrites are
   covered, so a file cannot regress by being touched.

The lint job stays `continue-on-error: true` until step 1 and step 2 can be
replaced by a blocking whole-tree `ruff check src tests scripts` (i.e. when the
backlog is gone); at that point drop the flag and delete the ratchet step.

The ratchet mirrors the pusher's own pre-gate, which has the same scope (changed
`.py` files, not the whole repo), so a branch that is green locally is not
surprised in CI.

## The pin

`ruff` is pinned to the same exact release in three places, and a test keeps them
in sync (`tests/test_ci_lint_policy.py`):

- `pyproject.toml` — `dev` extra: `ruff==0.15.0`;
- `.github/workflows/ci.yml` — `pip install "ruff==0.15.0"` after the dev extra;
- `docs/ci-lint-policy.md` — this file.

Why pin at all: `ruff` floats in through `ruff>=0.6`.  `0.17.0` shipped a
documented **breaking** formatter change (trailing pragma comments are no longer
counted when measuring line length), so an identical tree passes
`ruff format --check` on one release and fails on the next.  A formatting gate is
only meaningful against a fixed formatter release.

Bumping the pin: change all three places, run
`ruff format` + `ruff check --fix` on the files you are about to touch, and re-run
`python scripts/ci_lint_ratchet.py --base origin/main` before opening the PR.

No `[tool.ruff]` section is configured on purpose: the gate runs ruff's default
rule set so touched files are linted strictly (a local ignore list would weaken
exactly the files that are supposed to be clean).

## Verification (local, ruff 0.15.0, tree = `main` + this change)

```
$ ruff check --select E9,F63,F7,F82 src tests scripts
All checks passed!

$ python scripts/ci_lint_ratchet.py --base origin/main
ci_lint_ratchet: 2 changed Python file(s) vs origin/main
  scripts/ci_lint_ratchet.py
  tests/test_ci_lint_policy.py
All checks passed!                                  # ruff check -> exit 0
2 files already formatted                           # ruff format --check -> exit 0

# negative control: the ratchet must catch a new finding
$ printf 'import json\n' >> scripts/ci_lint_ratchet.py     # unused import (F401)
$ python scripts/ci_lint_ratchet.py --base origin/main
ci_lint_ratchet: ruff check -> exit 1               # gate fires
$ git checkout -- scripts/ci_lint_ratchet.py
$ python scripts/ci_lint_ratchet.py --base origin/main ; echo $?
0

$ pytest -q tests/test_ci_lint_policy.py
13 passed
$ pytest -q -n 8 --timeout=900        # full suite, 3.11, merged with main
1 failed, 1434 passed, 37 skipped, 2 xfailed, 50 subtests passed in 522.85s
```

`tests/test_ci_lint_policy.py` also exercises the git plumbing and both ruff
sub-commands end to end in a throwaway repository (clean change -> 0, new
finding -> 1, unformatted change -> 1, evidence-only change -> 0), so the gate
cannot regress to a no-op without a red test.

## Known limitations

- The backlog is frozen, not fixed: `ruff check src tests scripts` still reports
  thousands of findings, and untouched files still fail `ruff format --check`.
  That is intentional and visible — the gate does not pretend the debt is gone.
- Because the ratchet is diff-based it needs `fetch-depth: 0` in the lint job;
  a shallow checkout has no merge base and the step would fail loudly (exit 2).
- **Not fixed by this change:** `tests/test_multi_grid_gci.py::test_reference_agreement_accepts_a_multi_grid_study`
  stays red on `main` + this change (`AttributeError: 'MultiGridStudy' object
  has no attribute 'fallback_reason'`, `src/tensorfvm/verification/richardson.py:242`
  reads a field that `src/tensorfvm/verification/multi_grid.py` never declares).
  It is an inherited red, unrelated to the lint gate; the fix is to give
  `MultiGridStudy` the field `RefinementStudy` already has and pass it from
  `multi_grid_order()`.  Left to its own change so this one stays lint-only.
- A branch that deletes and re-adds a file is still linted (the `--diff-filter`
  includes `A`, `C`, `M`, `R`), so renames cannot be used to smuggle findings in.