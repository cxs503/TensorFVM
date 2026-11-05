#!/usr/bin/env bash
# ruff ratchet for TensorFVM CI.
#
# Why scoped instead of whole-tree: the tree carries a large pre-existing ruff
# backlog (9823 findings and 556 files that `ruff format` would rewrite, counted
# 2026-11-05 on 7f2587d).  A whole-tree gate is therefore red by construction
# and carries no signal.  This script gates instead the Python files that *this*
# change set touches, so new debt cannot land while the historic backlog stays a
# tracked debt item instead of a permanently red advisory job.  Once the sweep
# lands, CI goes back to `ruff check src tests scripts`.
#
# Scope resolution order (first one that resolves wins):
#   1. $BASE_SHA     - pull_request: github.event.pull_request.base.sha
#   2. $PUSH_BEFORE  - push:         github.event.before
#   3. merge-base with origin/main (local runs on a branch)
#   4. HEAD~1        - local runs on main / detached HEAD
# The diff is three-dot (`base...HEAD`), i.e. relative to the merge base: base
# drift must not show up as *this* change's lint debt.
# If a base resolves but the change set has no Python file, there is nothing to
# lint and the job is green.  If no base resolves at all, the scope degrades to
# the full tree (honest red) instead of an empty, silently green scope.
set -uo pipefail

resolve_base() {
    if [ -n "${BASE_SHA:-}" ] && [ "${BASE_SHA}" != "null" ] \
       && git cat-file -e "${BASE_SHA}^{commit}" 2>/dev/null; then
        printf '%s' "${BASE_SHA}"
        return 0
    fi
    if [ -n "${PUSH_BEFORE:-}" ] \
       && [ "${PUSH_BEFORE}" != "0000000000000000000000000000000000000000" ] \
       && git cat-file -e "${PUSH_BEFORE}^{commit}" 2>/dev/null; then
        printf '%s' "${PUSH_BEFORE}"
        return 0
    fi
    if git rev-parse --verify -q origin/main >/dev/null 2>&1; then
        git merge-base HEAD origin/main 2>/dev/null && return 0
    fi
    git rev-parse --verify -q 'HEAD~1' 2>/dev/null || true
}

SCOPE="${TMPDIR:-/tmp}/tensorfvm_lint_scope.txt"
RAW="${SCOPE}.raw"
: >"${SCOPE}"

base="$(resolve_base)"
if [ -n "${base}" ]; then
    echo "lint scope: merge-base diff against ${base}"
    if ! git merge-base "${base}" HEAD >/dev/null 2>&1; then
        echo "lint scope: no merge-base with ${base} (shallow clone?) -> diffing ${base} directly"
    fi
    git diff --name-only --diff-filter=ACMRT "${base}...HEAD" -- '*.py' \
        | sed '/^$/d' >"${RAW}" || true
    # Sparse local worktrees can list tracked files that are not on disk; CI
    # checks out everything, so this filter only affects local runs.
    while IFS= read -r f; do
        [ -f "${f}" ] && printf '%s\n' "${f}"
    done <"${RAW}" >"${SCOPE}"
    if [ ! -s "${SCOPE}" ]; then
        echo "lint scope: the change set touches no Python file -> nothing to lint"
        exit 0
    fi
else
    echo "lint scope: no base could be resolved -> full tree (legacy backlog: expect red)"
    git ls-files '*.py' >"${SCOPE}"
fi

echo "lint scope: $(wc -l <"${SCOPE}" | tr -d ' ') Python file(s)"
head -50 "${SCOPE}" | sed 's/^/  /'
[ "$(wc -l <"${SCOPE}" | tr -d ' ')" -gt 50 ] && echo "  ..."

rc=0
xargs -r -d '\n' ruff check <"${SCOPE}" || rc=$?
xargs -r -d '\n' ruff format --check <"${SCOPE}" || rc=$?
exit "${rc}"