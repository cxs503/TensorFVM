#!/usr/bin/env bash
# Scoped ruff gate for TensorFVM CI (delta ratchet).
#
# Why the gate is scoped instead of whole-tree: the tree carries a large
# pre-existing ruff backlog -- 9822 findings and 554 of 556 Python files that
# `ruff format` would rewrite, measured 2026-11-05 on PR #103's head.  A
# whole-tree `ruff check src tests scripts` is therefore red by construction:
# it failed on a docs-only pull request, where it says nothing about the change
# under test.  That is the failure this script removes.
#
# Why a *delta* ratchet instead of "changed files must be clean": almost no
# Python file in this tree is clean, so the strict form would turn every pull
# request that touches existing code red -- the same dead gate, one step later.
# scripts/ci/ruff_ratchet.py therefore compares each touched file against its
# base revision and fails only on debt this change set adds: new findings for a
# rule, or formatting regressions on files that were format-clean at base.
# New files are compared against an empty baseline, so they must be clean.
#
# Scope resolution order (first one that resolves wins):
#   1. $BASE_SHA      pull_request: github.event.pull_request.base.sha
#   2. $PUSH_BEFORE   push:         github.event.before
#   3. merge-base with origin/main   (local run on a topic branch)
#   4. HEAD~1                        (local run on main / detached HEAD)
# The diff is three-dot (`base...HEAD`), i.e. relative to the merge base, so
# base-branch drift can never be charged to this change set.
# If a base resolves and the change set touches no Python file under src/,
# tests/ or scripts/, there is nothing to lint and the job is green.
# If no base resolves at all, the scope degrades to the full tree (honest red)
# instead of to an empty, silently green scope.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "${REPO_ROOT}" || exit 2

SCOPE="${TMPDIR:-/tmp}/tensorfvm_lint_scope.txt"
RAW="${SCOPE}.raw"
: >"${SCOPE}"
: >"${RAW}"

resolve_base() {
    if [ -n "${BASE_SHA:-}" ] && [ "${BASE_SHA}" != "null" ] \
       && git cat-file -e "${BASE_SHA}^{commit}" 2>/dev/null; then
        printf '%s' "${BASE_SHA}"
        return 0
    fi
    if [ -n "${PUSH_BEFORE:-}" ] && [ "${PUSH_BEFORE}" != "null" ] \
       && [ "${PUSH_BEFORE//0/}" != "" ] \
       && git cat-file -e "${PUSH_BEFORE}^{commit}" 2>/dev/null; then
        printf '%s' "${PUSH_BEFORE}"
        return 0
    fi
    if git rev-parse --verify -q origin/main >/dev/null 2>&1; then
        git merge-base HEAD origin/main 2>/dev/null && return 0
    fi
    git rev-parse --verify -q 'HEAD~1' 2>/dev/null || true
}

base="$(resolve_base)"
if [ -n "${base}" ]; then
    echo "lint scope: three-dot diff ${base}...HEAD"
    git diff --name-only --diff-filter=ACMRT "${base}...HEAD" -- \
        'src/*.py' 'tests/*.py' 'scripts/*.py' | sed '/^$/d' >"${RAW}" || true
    # A sparse local worktree can list tracked files that are not on disk; the
    # CI runner checks out everything, so this filter only affects local runs.
    while IFS= read -r f; do
        [ -f "${f}" ] && printf '%s\n' "${f}"
    done <"${RAW}" >"${SCOPE}"
    if [ ! -s "${SCOPE}" ]; then
        echo "lint scope: the change set touches no Python file under src/, tests/ or scripts/"
        echo "lint scope: nothing to lint -> pass (pre-existing backlog is not re-reported)"
        exit 0
    fi
else
    echo "lint scope: no base could be resolved -> full tree (legacy backlog: expect red)"
    # Unscoped fallback: no baseline to compare against, so report the absolute
    # state honestly instead of pretending the scope were empty.
    rc=0
    ruff check src tests scripts || rc=$?
    ruff format --check src tests scripts || rc=$?
    exit "${rc}"
fi

n_files="$(wc -l <"${SCOPE}" | tr -d ' ')"
echo "lint scope: ${n_files} Python file(s) changed against ${base}"
head -50 "${SCOPE}" | sed 's/^/  /'
if [ "${n_files}" -gt 50 ]; then echo "  ..."; fi

mapfile -t files <"${SCOPE}"
exec python3 "${SCRIPT_DIR}/ruff_ratchet.py" "${base}" "${files[@]}"