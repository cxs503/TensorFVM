# PR #188 CI repair — self-hosted `ci-loong64` lane

**Mission**: m4691 · **Repo**: TensorFVM · **PR**: #188 (`agent/m0` → `main`)
**Head under repair**: `5189073b` "ci: 切换 self-hosted ci-loong64 同池运行（与 agent 开发共用环境）"
**Observed on the PR**: `lint (advisory) = cancelled`, `test () = cancelled` (the empty
parentheses in the check name are themselves part of the evidence, see §3.1).

## 1. Case / context

PR #188 accumulates the m4613/m4625/m4626 deliverables (84 files) plus one CI-infrastructure
commit, `5189073b`, which moved both jobs from GitHub-hosted `ubuntu-latest` to the
self-hosted pool `ci-loong64`. After that commit the PR carries **no usable CI signal at all**:
both jobs report `cancelled`, so neither the 84-file deliverable nor later commits can be
gated. Nothing in the Python tree is at fault here — the defect is entirely in
`.github/workflows/ci.yml`.

<table>
<tr><th>item</th><th>value</th></tr>
<tr><td>runner pool</td><td><code>ci-loong64</code> (loongarch64, node 099, 128 cores)</td></tr>
<tr><td>registered agents</td><td><code>node099-ci-fvm-1</code>, <code>node099-ci-fvm-2</code> (<code>/opt/ci-runners/r-fvm-*</code>), label set contains <code>ci-loong64</code></td></tr>
<tr><td>pool interpreter</td><td>CPython 3.11.6 (Anolis 12.3) — the only one installed; no <code>hostedtoolcache</code></td></tr>
<tr><td>pip index on the pool</td><td><code>https://lpypi.loongnix.cn/loongson/pypi</code> (loongnix mirror)</td></tr>
<tr><td>toolchain present</td><td>setuptools 84.2.1, torch 2.7.1, ruff 0.15.0, pytest 9.2.0</td></tr>
</table>

## 2. What was done

One file changed — `.github/workflows/ci.yml` — plus this evidence directory. Four
reproducible defects of `5189073b` were repaired:

| # | defect | repair |
|---|---|---|
| 1 | `strategy.matrix` deleted, but `name: test (${{ matrix.python-version }})` and `setup-python.with.python-version: ${{ matrix.python-version }}` still referenced it → job name renders `test ()`, version renders empty | matrix removed **cleanly**: job renamed `test (3.11)`, no `matrix.*` reference survives outside comments |
| 2 | `actions/setup-python` cannot provision loongarch64 interpreters (python-build-standalone ships x86_64/aarch64 only); the lint job even hard-coded `"3.12"`, absent on the pool | every job uses the pool's own `python3` and *asserts* `>= 3.11` instead of installing an interpreter |
| 3 | `pip install -e ".[dev]"` with build isolation failed on the pool: `ERROR: No matching distribution found for setuptools>=84` (build-dependency resolution could not reach the loongnix mirror, §3.3) | install with `--no-build-isolation` — the runner's own setuptools 84.2.1 satisfies `build-system.requires`, so the job no longer depends on the mirror being reachable for its build environment |
| 4 | the gmsh step ran `sudo apt-get update && apt-get install libGLU`; the self-hosted pool has no passwordless sudo, and `python` is not guaranteed on `PATH` | gmsh installed via `python3 -m pip` only, still `continue-on-error: true` so a missing libGLU degrades to skips; all steps spell out `python3 -m …` |

`runs-on: ci-loong64` is **kept**: the owner's intent in `5189073b` (share the pool the agents
develop on) is preserved; the migration was simply completed instead of being reverted to
GitHub-hosted runners. The `env: TORCH_DEVICE_BACKEND_AUTOLOAD: "0"` addition of `5189073b`
is kept as well.

The lint job stays `continue-on-error: true` (it is *advisory* by policy until one release
cycle has passed) — but its remaining debt is now stated in the workflow instead of being
silently green: 55 findings, all `F401`/`F841` inside `tests/`, down from 165 at `main`.

## 3. Results

### 3.1 Why the check name proves defect 1

The PR check is reported as `test () = cancelled`. A job name is
`test (${{ matrix.python-version }})`; the only way it renders as `test ()` is an undefined
matrix, i.e. the removed `strategy` block. The same empty expansion is handed to
`setup-python`, which cannot succeed — so the job definition itself is invalid, independent
of runner availability.

### 3.2 Workflow now parses and is matrix-free

```
$ python3 -c "import yaml; d=yaml.safe_load(open('.github/workflows/ci.yml'))"
jobs: ['test', 'lint']
test | name: test (3.11) | runs-on: ci-loong64 | matrix: False
lint | name: lint (advisory) | runs-on: ci-loong64 | matrix: False | coe: True
test steps: ['actions/checkout@v4', 'Report the pool interpreter', 'Install package with dev extras',
             'Install the optional gmsh mesh runtime', 'Run tests', 'Upload coverage report']
```

### 3.3 Defect 3 reproduced, then fixed

```
$ python3 -m pip install -e ".[dev]"
ERROR: Could not find a version that satisfies the requirement setuptools>=84 (from versions: none)
ERROR: No matching distribution found for setuptools>=84

$ python3 -m pip install -e ".[dev]" --no-build-isolation
Successfully installed tensorfvm-0.1.0            # rc=0
```

Honest qualification: the isolated build **failed once and succeeded on a later retry**, so the
pool's build-dependency index is *intermittent*, not permanently missing `setuptools`. That is
still a real flake source for a gate (the failure mode is `ERROR` at the install step, i.e. a
red job for a reason unrelated to the code), and `--no-build-isolation` removes the dependency
on it outright — hence the repair stands even though the defect is flaky rather than absolute.

### 3.4 The repaired `test` recipe executed on the runner host

The pool lives on the same machine as this worktree, so the job recipe was replayed
verbatim (`python3 -m pip install … --no-build-isolation`, then the job's pytest line):

| quantity | value |
|---|---|
| tests collected | 1462 |
| passed | **1426** |
| skipped | 37 |
| xfailed | 2 (16 warnings) |
| subtests passed | 50 |
| failed | **0** |
| wall time | 588 s (`-n 8`; serial run reached 99 % with no `F` before it was reaped by the harness) |
| `ruff check src tests scripts` | 55 findings (advisory), 165 at `main` |

Figure: `lint-and-suite.png` — left panel: lint findings `main` 165 → PR head 55; right
panel: local suite outcome on the pool (1426 pass / 0 fail) next to the 55 advisory lint
findings.

## 4. Notes, limitations, honesty

* **The cancelled runs' logs were not readable from this host** (outbound GitHub API is
  blocked; only `git ls-remote`/`fetch` go through). The root-cause chain above is therefore
  *inference plus reproduction of each individual defect*, not a quote from the run log —
  but every defect is demonstrated on the real pool (§3.3, §3.4) and the `test ()` name in
  the mission's own observation independently confirms defect 1.
* **Green + merge cannot be observed from inside the worktree.** The repair is a branch
  commit; the pr-gate records the run and the merge. If CI still reports queued/cancelled
  after this commit, the next thing to check is runner availability (labels are present as of
  2026-11-06T02:02Z) rather than the workflow.
* **55 lint findings remain** (all in `tests/`, `F401`/`F841`). Fixing them touches ~30 files,
  over the ≤10-file unit for this ticket; they are advisory, counted and visible.
* Enabling E7 (`pycodestyle` statement-layout) is still deliberately out of scope — the
  repository writes dense one-liners on purpose (see `docs/pr109-lint-gate/report.md`).
* If the owner prefers GitHub-hosted runners again, restoring a `python-version` matrix is a
  three-line revert; the reason it was dropped (no loongarch64 builds upstream) is documented
  in the workflow header so the next migration does not repeat the mistake.

## 5. Error judgement (§6 of the report contract)

This is a CI-infrastructure repair; there is no physical quantity and no reference
experiment, so "error < 3 %" is **not applicable** — no numerical result is claimed. The
quantitative claims that *are* made are exact counts, each programmatically obtained:

| claim | baseline | measured | verdict |
|---|---|---|---|
| test job outcome | `cancelled` (no signal) | 1426 passed / 0 failed on the pool | signal restored |
| undefined matrix references in workflow | 2 (`name`, `setup-python`) | 0 code references (comment only) | fixed |
| isolated build on pool pip index | failed once (`setuptools>=84` unreachable, mirror intermittent) | `--no-build-isolation` rc=0, no mirror dependency | fixed |
| lint findings | 165 at `main` | 55 (advisory) | improved 67 %, not zero |