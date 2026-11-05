# Lint gate (`ruff check src tests scripts`)

Status: **blocking** in CI (job `lint`, workflow `.github/workflows/ci.yml`)
since 2026-11-05. Policy lives in `pyproject.toml` → `[tool.ruff]`,
machine-checked by `tests/test_lint_gate.py`.

## Why the job was red (reproduction)

Until this change the lint job ran an unconfigured `ruff check src tests
scripts` over 557 files. Reproduce the pre-fix state on any commit before the
gate commit:

```bash
( cd "$(git rev-parse --show-toplevel)" && git stash list >/dev/null )   # clean tree
git show <pre-gate-commit>:pyproject.toml > /tmp/old-pyproject.toml      # no [tool.ruff]
ruff check --no-cache --statistics --config /tmp/old-pyproject.toml src tests scripts
```

Measured on the PR #128 head tree (`9c52373`) with ruff 0.15.0:

| rule | sites | what it is |
|---|---:|---|
| E702 | 7601 | `a=1;b=2` statement packing |
| E701 | 1852 | `if x: y` / `def f(): return` on one line |
| F401 | 98 | unused import |
| E401 | 84 | `import os,sys` |
| E731 | 69 | `f = lambda ...` |
| F841 | 50 | local assigned but never used |
| E402 | 39 | import after `sys.path` shim / optional-dependency guard |
| E741 | 20 | ambiguous name `l`, `I`, `O` |
| F541 | 6 | f-string without placeholder |
| E703 | 3 | useless trailing semicolon |
| **total** | **9822** | |

The job had to carry `continue-on-error: true`, so its red status was advisory
for humans but *not* for the merge gate: the pr-gate reads per-check-run
conclusions, and `lint (advisory)=failure` therefore blocked every TensorFVM
PR while the backlog existed.

## What the gate enforces now

`select = ["E4", "E7", "E9", "F"]` — defect rules only:

* `E9` syntax errors, `F` undefined/duplicate names, unused imports and
  variables, `E4` malformed import blocks. These find real defects; F841 in
  particular catches "computed and silently dropped" numerics.

Not enforced, on purpose (**house style**, ~9.5k sites in the shipped solver
kernels): `E701`, `E702`, `E703` (statement packing), `E731` (lambda alias),
`E741` (single-letter physics names such as `l` for length). Rewriting these is
a five-digit-line mechanical diff across every solver file — incompatibility
risk with in-flight branches and zero defect value. If someone wants them, do
it as one isolated reformat commit, not as a gate prerequisite.

Two narrow exemptions, both documented in `pyproject.toml`:

* `E402` for `scripts/*`, `tests/test_lbm_reference_audit.py`,
  `src/tensorfvm/external_high_order.py` (late import to break an import cycle),
  `tests/test_two_phase_projection_operator_matrix.py` — the `sys.path` /
  `importorskip` shim pattern is deliberate here.
* `F841` frozen per file for 32 files / 50 sites (dead locals left over from
  derivation scratch work in numeric routines). Frozen per *file*, so any new
  dead local anywhere else fails immediately.

`ruff` is pinned in the `dev` extra as `ruff>=0.6,<0.16`: an unpinned linter
lets a new release move default rules and flip the gate red overnight.

## Verification

```bash
ruff check src tests scripts      # must print "All checks passed!"
ruff format --check src tests scripts   # informational only (see below)
python -m pytest tests/test_lint_gate.py -q
```

`tests/test_lint_gate.py` runs the exact CI command and fails if the policy is
loosened (ignore list widened to a defect code), if a debt-baseline path goes
stale, or if the CI job is switched back to advisory mode. The policy therefore
cannot drift away from the workflow.

## Remaining debt (surfaced, not hidden)

`ruff format --check` still reports 555 of 557 files: the tree predates
`ruff format` and was hand-wrapped. The whole-tree reformat is a separate
mechanical change (it will touch nearly every file and needs to be replayed
after in-flight branches land). The CI step is kept as a **named advisory
step** inside the blocking job, so the debt stays visible in every run's log
without masking the lint gate.

Ratchet order when that lands: reformat → drop the F841 baseline entries as
their files get touched → tighten `line-length` if desired → never add an
entry back to `per-file-ignores`.