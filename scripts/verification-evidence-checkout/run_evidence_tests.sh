#!/bin/sh
# Exercise the evidence-dependent tests from a sparse worktree.
#
# The solver worktrees are sparse cones (src/ tests/ scripts/), so the tracked
# evidence under docs/ is absent and the owning tests skip.  `git archive` is a
# read-only export of the same commit, so this reproduces exactly what a full
# checkout would run without touching the sparse configuration.
set -eu
REPO=$(cd "$(dirname "$0")/../.." && pwd)
EVIDENCE=${EVIDENCE:-/tmp/tensorfvm-evidence}
mkdir -p "$EVIDENCE"
cd "$REPO"
git archive HEAD docs/suite-channel docs/periodic-mac docs/verification-suboff-mesh docs/verification-gmsh-cylinder | tar -x -C "$EVIDENCE"
TENSORFVM_EVIDENCE_ROOT="$EVIDENCE" exec python -m pytest -q \
  tests/test_lbm_reference_audit.py tests/test_periodic_mac.py \
  tests/test_suboff_geometry.py tests/test_wall_traction.py \
  tests/test_evidence_checkout_policy.py "$@"
