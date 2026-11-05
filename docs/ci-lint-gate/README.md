# CI lint gate: from a permanently-red advisory job to a green, meaningful ratchet

Status: implemented in this branch (PR #108 follow-up).

## 1. Symptom

PR #108 (`agent/m3620`, head `d1d51be`) reported

```
lint (advisory)   failure
lint (advisory)   failure
```

twice, because `.github/workflows/ci.yml` triggers on both `push` and
`pull_request`. The test job was green.

## 2. Root cause

The lint job ran the ruff **default** rule set over the whole tree
(`ruff check src tests scripts`) plus `ruff format --check`. That tree carries a
large, *pre-existing* backlog, and the job was marked `continue-on-error: true`
precisely because of it (`docs`-free comment in the old workflow: "the tree
carries a large pre-existing ruff backlog ... Fix the backlog, then drop this
continue-on-error and promote it to a gate").

Measured on the PR head tree (`scripts/audit_lint_gate.py --before d1d51be`,
numbers in `docs/ci-lint-gate/lint_gate_audit.json`):

| rule | findings | what it is |
|---|---:|---|
| E702 | 7601 | multiple statements on one line (semicolon) |
| E701 | 1852 | multiple statements on one line (colon) |
| F401 | 99 | unused import |
| E401 | 84 | multiple imports on one line |
| E731 | 69 | lambda assignment |
| F841 | 50 | unused variable |
| E402 | 39 | module import not at top of file |
| E741 | 20 | ambiguous variable name |
| F541 | 6 | f-string without placeholders |
| E703 | 3 | useless semicolon |
| **total** | **9823** | and 555 of 560 files would be reformatted |

So the red had **nothing to do** with the PR's own content: the job was red on
`main` before the PR touched anything. Two consequences:

* the check carried no signal (a permanently red advisory is the same as no
  lint at all);
* it still showed up as `failure` in the PR's check-run list, which is what the
  merge accounting saw.

The only defect the PR *did* contribute was one finding in its new test file:
`tests/test_native_potential_convergence.py:12` — `F401 'pytest' imported but
unused`.

The house style (dense, semicolon-packed kernels and one-line test bodies) is
deliberate and pervasive: 7601 + 1852 = 9453 of the 9823 findings are exactly
that style, and 555 files would be reformatted. A "fix the backlog" commit that
rewrites 555 files in one PR is not reviewable, not safe (removing imports by
machine can drop registration side effects), and not what this PR is about.

## 3. Design of the fix

The gate is split in two, so it is green *today* and can only get stricter:

1. **Tree-wide, blocking — correctness only.** `pyproject.toml` now pins
   `[tool.ruff.lint].select = ["E9", "F63", "F7", "F82"]`: syntax/runtime
   errors (`E9`), invalid comparisons (`F63`), real pyflakes bugs such as
   redefinitions (`F7`), undefined names (`F82`). These already had **zero**
   findings on both `main` and the PR head, so the job can be blocking
   immediately.
2. **Changed files only — the full pyflakes set.** `scripts/lint_ratchet.py`
   holds every `.py` file a change touches to `E9` + all of `F` (unused
   imports/variables, f-string typos, redefinitions, undefined names, ...).
   New code therefore cannot grow the backlog, and every touched file shrinks
   it. `__init__.py` is exempt from `F401` (re-exports).
3. The layout families (`E1/E2/E3/E4/E7`) and `ruff format` stay
   **informational** (`continue-on-error`, plus a backlog report step) until the
   formatting campaign lands; each family is promoted to the blocking set as
   soon as its count reaches zero.

CI job name `lint (advisory)` is intentionally unchanged (check-run accounting
already watches it), but the job-level `continue-on-error` is gone: the job is
now a real gate.

## 4. Changes

| file | change |
|---|---|
| `pyproject.toml` | added `[tool.ruff]` / `[tool.ruff.lint]`: enforced set = `E9,F63,F7,F82`, target-version py310, line-length 120, with the measured backlog and the adoption rule documented in place |
| `.github/workflows/ci.yml` | lint job: dropped job-level `continue-on-error`; blocking `ruff check`; new **ratchet** step (changed files vs. PR base, `fetch-depth: 0`); informational backlog report; format check marked informational |
| `scripts/lint_ratchet.py` | new: "no new backlog" ratchet (git-diff driven, graceful fallback, `--report` mode) |
| `scripts/audit_lint_gate.py` | new: reproducible before/after audit + figure (`git archive` the old tree, run ruff on both) |
| `tests/test_lint_ratchet.py` | new: 6 contract tests pinning both halves of the gate |
| `tests/test_native_potential_convergence.py` | PR's own defect fixed: unused `pytest` import removed, file formatted |
| `docs/ci-lint-gate/README.md` | this document |
| `docs/ci-lint-gate/lint_gate_audit.json`, `docs/ci-lint-gate/lint_gate_backlog.png` | generated evidence |

## 5. Evidence

Reproduce locally (from the branch head):

```bash
pip install -e ".[dev]"

# 1. reproduce the red on the pre-fix tree: 9823 findings, non-zero exit
python scripts/audit_lint_gate.py --before d1d51be --no-figure

# 2. the blocking gate is green: 0 findings
ruff check src tests scripts ; echo "exit=$?"        # -> All checks passed! exit=0

# 3. the ratchet holds on the PR's own file list
python scripts/lint_ratchet.py --base origin/main    # -> clean
python scripts/lint_ratchet.py --files tests/test_native_potential_convergence.py

# 4. contract tests
pytest -q tests/test_lint_ratchet.py                 # -> 6 passed
```

Recorded results (`docs/ci-lint-gate/lint_gate_audit.json`, produced by the script, not
by hand):

| quantity | before (`d1d51be`) | after |
|---|---:|---:|
| default-rule-set findings, `src tests scripts` | 9823 | 9822 |
| findings in the PR's changed python files (`E9,F`) | 1 (`F401`) | **0** |
| enforced gate `{E9,F63,F7,F82}` | 0 | **0** (blocking, green) |
| files `ruff format --check` would rewrite (line length 120) | 555 | 554 |
| `pytest -q tests/test_lint_ratchet.py` | n/a | 6 passed |
| `pytest -q tests/test_native_potential_convergence.py` | n/a | 6 passed (105 s) |

The default-rule-set total barely moves (9823 -> 9822) on purpose: this change
fixes the *gate*, not the 9.4k layout findings that the campaign still owns. The
number that matters for CI is the enforced set: 0 before and 0 after, so the job
can be blocking, and the ratchet keeps it that way.

![backlog by rule family](lint_gate_backlog.png)

## 6. Limitations / what is not done

* The E701/E702 "dense style" debt (9453 findings) and the 555 unformatted
  files are **not** fixed; they are now measured, reported on every CI run, and
  cannot grow. A dedicated formatting campaign (one directory per PR, behaviour
  untouched) is the follow-up.
* `F401/F841/F541` remain in the tree-wide backlog (99/50/6 findings); they are
  enforced only on changed files. Promote each to
  `[tool.ruff.lint].select` once its count is 0.
* The ratchet falls back to the last commit when the base ref cannot be
  resolved (shallow clone, fork, detached HEAD) instead of failing the build; a
  multi-commit PR would then only be checked on its tip commit. CI passes
  `fetch-depth: 0` so the normal path uses the merge base.