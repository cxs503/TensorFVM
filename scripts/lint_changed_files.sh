#!/usr/bin/env bash
# Lint gate for the Python files a change actually touches.
#
# Why this exists: TensorFVM carries a large pre-existing ruff backlog
# (E701/E702 semicolon-packed statements, unused imports, import order: ~9.8k
# findings) and `ruff format` would rewrite 556 of the 558 files under
# src/tests/scripts.  A whole-tree gate therefore can never be green, so the
# "lint (advisory)" check-run was permanently red and carried no information
# about the change under review.  This script enforces the same rules on the
# changed-file set only, which is the ratchet the platform already applies
# before pushing (tp_pusher `_ruff_gate`); the whole-tree state is still
# reported by CI as non-blocking information.
#
# Usage:
#   scripts/lint_changed_files.sh [BASE_REF]      # default: origin/main
#
# BASE_REF may be any ref/sha; the merge-base against the working tree is used.
# Exit status: 0 = changed files clean, 1 = ruff findings, 2 = cannot resolve base.
set -euo pipefail

base="${1:-origin/main}"
scope=(src tests scripts)

if ! git rev-parse --verify --quiet "${base}^{commit}" >/dev/null; then
    echo "error: cannot resolve lint base '${base}'" >&2
    exit 2
fi

if merge_base="$(git merge-base "${base}" HEAD 2>/dev/null)"; then
    :
else
    merge_base="$(git rev-parse "${base}^{commit}")"
fi
echo "lint base: ${base} -> merge-base ${merge_base} ($(git log -1 --format='%h %s' "${merge_base}"))"

mapfile -t files < <(
    {
        # tracked changes and deletions excluded (ACMR) between the merge-base
        # and the working tree, so uncommitted local edits are gated too.
        git diff --name-only --diff-filter=ACMR "${merge_base}" -- "${scope[@]}"
        git ls-files --others --exclude-standard -- "${scope[@]}"
    } | grep -E '\.pyi?$' | sort -u
)

if [ "${#files[@]}" -eq 0 ]; then
    echo "no changed python files under: ${scope[*]} — nothing to lint"
    exit 0
fi

echo "linting ${#files[@]} changed python file(s):"
printf '  %s\n' "${files[@]}"

status=0
ruff check --no-cache "${files[@]}" || status=1
ruff format --check "${files[@]}" || status=1

if [ "${status}" -ne 0 ]; then
    echo "FAIL: changed files are not ruff-clean (fix with 'ruff check --fix <files>' / 'ruff format <files>')" >&2
else
    echo "PASS: changed files are ruff-clean (check + format)"
fi
exit "${status}"