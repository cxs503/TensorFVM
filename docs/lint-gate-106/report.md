# PR #106 lint red: attribution, bounded fix, and gate-distance census

Mission m4186 (custody round on TensorFVM PR #106, branch `agent/m3605`).
Worktree branch `agent/m4186`, based on `origin/agent/m3605` @ `038d284`.

## 1. 案例介绍

PR #106 (`agent/m3605`, "matured mission deliverables") adds the 3-D mutation
survival audit of the native finite-volume core. Relative to `main` (`87d4083`)
it touches 7 files:

| file | kind |
| --- | --- |
| `src/tensorfvm/verification/mutation_catalogue3d.py` | new, 99 lines (mutant catalogue) |
| `scripts/audit_mutation_survival3d.py` | new, 190 lines (mutation runner + figure + table) |
| `tests/test_mutation_catalogue3d_contract.py` | new, 36 lines (catalogue contract) |
| `tests/test_native_mesh3d.py` | new, 90 lines (mesh/operator unit tests) |
| `docs/mutation-survival3d/{report.md,mutation-survival-3d.json,.png}` | carried report |

CI on the PR: `test (3.11)` and `test (3.12)` green; the only red checks are
`lint (advisory)` ×2. Both lint steps are **tree-wide** commands
(`ruff check src tests scripts`, `ruff format --check src tests scripts`), so the
job reports the state of the *whole* tree, not of the PR's diff.

## 2. 怎么做的

1. **Reproduce red locally** on a clean worktree of `origin/agent/m3605`
   (ruff 0.15.0, the version resolved by the `dev` extra):
   * `ruff check src tests scripts` → 9812 findings / 425 files;
   * `ruff format --check src tests scripts` → 553 of 560 files would be rewritten;
   * restricted to the 4 Python files the PR adds: `ruff check` → **14 findings**
     (12× `E702` semicolon-packed statements, 2× `E701`), `ruff format --check`
     → **4 files would be reformatted**. That is the part attributable to #106.
2. **Fix the PR-owned files only** (minimal, no unrelated edits):
   `ruff check --fix --unsafe-fixes` (splitting `a=1;b=2` lines) followed by
   `ruff format` on exactly those four paths, plus one surgical rename of a
   shadowing comprehension variable `l` → `line` in `audit_mutation_survival3d.py`
   (`E741`) that ruff cannot autofix.
3. **Prove no semantic drift**: the diff is statement-splitting + reformatting
   only (checked by reading `git diff`); mutant `find=`/`replace=` string literals
   are untouched (ruff never rewrites string contents). Then run the affected
   tests.
4. **Quantify the gate distance** (the residual red is *not* PR-attributable):
   `scripts/audit_ruff_backlog.py` produces a regeneration-safe census
   (`--output …json`, `--markdown …md`) that re-runs both CI steps report-only.
   It is commit-time neutral: `--output-format=json` and `--check` never write
   the tree.

## 3. 计算结果

### 3.1 PR-attributable lint state (the part this round can fix)

| metric | before (origin/agent/m3605) | after | target |
| --- | ---: | ---: | ---: |
| `ruff check` findings in the 4 PR files | 14 | **0** | 0 |
| `ruff format --check` files pending in the 4 PR files | 4 | **0** | 0 |
| `ruff check src tests scripts` (tree-wide) | 9826 | 9812 | 0 |
| `ruff format --check` (tree-wide, files to rewrite) | 557 | 553 | 0 |

The tree-wide improvement (−14 findings, −4 files) is exactly the PR's own debt;
nothing else in the tree was touched.

### 3.2 Tree-wide gate distance (regenerated, see `census.md` / `ruff-census.json`)

Findings 9812 over 425 files, dominated by the project's deliberate dense style
(`E702` 7592, `E701` 1851 = semicolons/colons packing several statements per
line). Real-defect rules are a small remainder: `F401` 98 unused imports,
`F841` 50 unused variables, `F541` 6 f-strings without placeholders, `E401` 84,
`E731` 69, `E402` 39, `E741` 20, `E703` 3.

The formatter step is the harder blocker: **553 of 560 files** are not in
`ruff format` style at all — the tree was never formatted, so this is not a
version skew artefact (spot-checked `--diff`: continuation-line alignment of
dict literals and parenthesised imports is rewritten, not whitespace noise).

### 3.3 Tests after the fix

```
$ PYTHONPATH=src python -m pytest tests/test_native_mesh3d.py \
      tests/test_mutation_catalogue3d_contract.py -q
15 passed, 1 skipped        # skip = gmsh runtime absent, by design
```

## 4. 结果图

`lint-backlog.png` (generated from `ruff-census.json`, see the plotting snippet
below): left panel = findings per rule on a log axis, right panel = formatter
status. It makes the structure of the red obvious: two style rules carry 96 % of
the findings, and the formatter would rewrite essentially every file.

```python
d = json.load(open("docs/lint-gate-106/ruff-census.json"))
# left: horizontal log-scale barh of d["check"]["by_rule"]
# right: [d["format"]["would_reformat"], d["format"]["already_formatted"]]
```

## 5. 注意点

* **The red is main-drift, not PR drift.** `main` ships the identical
  `.github/workflows/ci.yml` (`git show origin/main:.github/workflows/ci.yml`
  equals the PR's), and the tree-wide backlog is pre-existing, so
  `lint (advisory)` is red on `main` too. The merge governor already encodes
  this: `merge_governor.json → ignore_checks.TensorFVM = ["lint (advisory)"]`,
  and `main_red_checks()` exists precisely so that "a PR red only in checks
  main itself fails is not the agent's fault".
* **Why the job cannot simply be made green here.** The two options are
  (a) burn down the whole backlog — mechanically possible
  (`ruff check --fix --unsafe-fixes` + `ruff format` would split ~9.4 k
  semicolon statements and rewrite 553 files, i.e. a repo-wide reformat that
  collides with every one of the ~30 open TensorFVM PRs), or (b) narrow the job
  to changed files / a baseline ratchet. Option (b) lives in
  `.github/workflows/ci.yml`, which the agent sparse checkout deliberately does
  not materialise (`.git/info/sparse-checkout` = `src tests scripts pyproject.toml
  README.rst docs outputs`): CI-gate edits are a governance decision, not an
  agent edit. Both are therefore out of scope for a custody round, and this
  round deliberately kept the change PR-bounded rather than shipping a
  conflicting mega-diff.
* **Version skew caveat.** The census was taken with ruff 0.15.0; CI installs
  `ruff>=0.6` (latest). Rule counts can shift by a few findings between
  versions, but the two dominant classes (`E701`/`E702`) and the formatter
  verdict are version-stable.
* **Recommended path (least-risk, keeps signal):** land the ratchet — CI lints
  only files changed by the push (`git diff --name-only <merge-base>...HEAD`)
  and keeps the tree-wide run as a non-failing report; then burn the backlog
  down per directory (`scripts/audit_ruff_backlog.py --paths src`), biggest
  first. Only after the census reports `green: true` should
  `continue-on-error` be dropped.

## 6. 误差判定

| key quantity | baseline (criterion) | measured | error | within 3 %? |
| --- | ---: | ---: | ---: | --- |
| findings in PR-owned Python files | 0 (CI gate) | 0 | 0 % | **yes** |
| unformatted PR-owned files | 0 (CI gate) | 0 | 0 % | **yes** |
| tree-wide `ruff check` findings | 0 (CI gate) | 9812 | ∞ | no |
| tree-wide files pending `ruff format` | 0 (CI gate) | 553 | ∞ | no |
| regression risk of the fix (tests) | 0 failures | 15 passed / 1 skipped | 0 % | **yes** |

**Honest verdict: the PR-attributable red is fixed and proven, but the
`lint (advisory)` job is NOT green, so the mission's completion criterion
("CI 绿") is not met by this change.** The residual 9812 findings / 553 files
are repo-wide and present on `main`; closing them requires either a
repository-wide reformat decision or a CI-scope change in
`.github/workflows/ci.yml`, both of which are owner/governance decisions rather
than custody-round edits. Consequence: PR #106 can still merge (the governor
ignores this advisory check), but the two advisory check runs will stay red
until the gate is re-scoped or the backlog is retired.