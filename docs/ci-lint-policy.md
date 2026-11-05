# CI lint policy (ruff pin + formatting baseline)

## Why the lint job went red on PR #120

`lint (advisory)` failed on both runs of PR #120 head `8bbfea1`:

| run | job id | conclusion |
|---|---|---|
| push | 127157441669 | failure |
| pull_request | 126736770749 | failure |

Check-run annotations carry no log text (only `Process completed with exit code 1`),
so the failure was reproduced locally instead:

```
$ ruff check src tests scripts            # ruff 0.15.0 on 8bbfea1
7651  E702  multiple-statements-on-one-line-semicolon
1852  E701  multiple-statements-on-one-line-colon
 120  F401  unused-import
  84  E401  multiple-imports-on-one-line
  71  E731  lambda-assignment
  57  F841  unused-variable
  40  E402  module-import-not-at-top-of-file
  21  E741  ambiguous-variable-name
   6  F541  f-string-missing-placeholders
   3  E703  useless-semicolon
Found 9905 errors.

$ ruff format --check src tests scripts
609 files would be reformatted, 2 files already formatted
```

Both gate commands fail, i.e. the merged head carried neither a `[tool.ruff]`
config nor a formatted tree.  The reformat+ignore fix from `800254d` (round 1)
is absent from the merged head, so the whole backlog was back in main.

## What this change does

1. **Formatting baseline**: `ruff format src tests scripts` + `ruff check --fix`
   on the tree (611 python files), plus the small `[tool.ruff]` ignore list for
   the rules that cannot be fixed mechanically without touching semantics
   (`E402`, `E741`, `F811`).
2. **Hand fixes** for the 13 remaining non-autofixable sites in files that
   landed after the round-1 snapshot: 2× `E731` (lambda assignment → `def`),
   11× `F841` (dead local assignments, each RHS side-effect free).
   24× `F401` unused imports removed with ruff's autofix.
3. **Deterministic linter**: `ruff` is pinned to `0.15.0` in the `dev` extra and
   installed explicitly in the lint job.  `ruff 0.17.0` shipped a *breaking*
   formatter change ("trailing pragma comments are now ignored when computing
   line length"), so an unpinned `ruff>=0.6` makes the formatter gate flap
   between releases on identical source.

## Verification (local, ruff 0.15.0)

```
$ ruff check src tests scripts   -> All checks passed!
$ ruff format --check src tests scripts -> 611 files already formatted
$ pytest -q tests/test_ci_lint_pin.py tests/test_two_phase_projection.py -> passed
```

## Bumping ruff

The pin is the *formatting baseline*; do not bump it casually:

1. `pip install "ruff==<new>"`
2. `ruff check --fix src tests scripts`
3. `ruff format src tests scripts`
4. update the pin in `pyproject.toml` **and** `.github/workflows/ci.yml`
5. `tests/test_ci_lint_pin.py` enforces that the two pins agree and that both
   are exact (`==`) pins — a floating specifier makes CI non-reproducible.
