"""Drift gate for the probe bundle behind ``verdict_qualified``.

``docs/oss_solutions/CASE-FVM-DAMBREAK-STOKER.json`` summarises **two** artifact
bundles, but only one of them was watched:

* ``src/tensorfvm/verification/artifacts/shallow-water-dam-break/metrics.json`` is
  re-run and compared bit-for-bit by
  ``tests/test_dambreak_stoker_repro_verification.py`` and
  ``tests/test_dambreak_anchor_closure.py``;
* ``docs/case-fvm-dambreak-stoker/artifacts/reproduction.json`` -- the probe
  ladder the entry's ``verdict_qualified`` quotes ("PASS at n >= 800 ... at
  n = 200 the fan-head probe fails") -- was read by no test, no audit script and
  no registration gate.  It was produced by ``docs/case-fvm-dambreak-stoker/
  verify_reference.py`` on a machine/commit that nothing pinned, so the entry's
  coarse-grid sentence could have gone stale without a red light.

This file closes that hole: it re-runs the verifier on the accepted grid
(``--quick``, ``n <= 800``, ~12 s) and requires

1. every probe number the entry quotes at ``n <= 800`` to be reproduced to
   ``1e-12`` relative (the same binding strength the headline bundle gets);
2. the *shape* of the ladder -- which rungs fail the 5 % gate and from which rung
   each bed passes -- to be unchanged, because that is the statement
   ``verdict_qualified`` makes, not just the numbers in it;
3. the reference certification of the closed-form solution to stay a pass.

The full `n = 1600` run is not repeated here (the headline bundle's own gate
already covers that grid); the point of this gate is the coarse rungs where the
fan-head probe misses.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VERIFIER_DIR = os.path.join(REPO_ROOT, "docs", "case-fvm-dambreak-stoker")
VERIFIER = os.path.join(VERIFIER_DIR, "verify_reference.py")
BUNDLE = os.path.join(VERIFIER_DIR, "artifacts", "reproduction.json")
POOL_ENTRY = os.path.join(REPO_ROOT, "docs", "oss_solutions",
                          "CASE-FVM-DAMBREAK-STOKER.json")

#: the accepted grid the quick run covers (the bundle's own ``accepted_n``)
QUICK_MAX_N = 800
REL_TOL = 1e-12


def _load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture(scope="module")
def fresh(tmp_path_factory):
    """Re-run the reference verifier on the accepted grid into a temp dir."""
    out = tmp_path_factory.mktemp("dambreak-repro") / "reproduction.json"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [os.path.join(REPO_ROOT, "src"), env.get("PYTHONPATH", "")]).strip(os.pathsep)
    completed = subprocess.run(
        [sys.executable, VERIFIER, "--quick", "--no-inject", "--out", str(out)],
        cwd=VERIFIER_DIR, env=env, capture_output=True, text=True,
        timeout=600)
    assert completed.returncode == 0, completed.stderr
    assert out.is_file(), completed.stdout + completed.stderr
    return _load_json(str(out))


@pytest.fixture(scope="module")
def committed():
    return _load_json(BUNDLE)


@pytest.fixture(scope="module")
def pool():
    return _load_json(POOL_ENTRY)


def _quick_runs(bundle: dict) -> list[tuple[str, int]]:
    return [(case, int(n))
            for case, runs in bundle["runs"].items()
            for n in runs if int(n) <= QUICK_MAX_N]


def test_reference_certification_still_passes(fresh):
    """The closed-form solution itself is certified against the solver's anchors."""
    reference = fresh["reference"]
    assert reference["pass"] is True
    assert reference["worst_relative"] < 1e-12, reference["worst_relative"]
    assert reference["tolerance"] <= 1e-2


def test_every_probe_number_reproduces(fresh, committed):
    """Every (case, n<=800) probe check is reproduced to 1e-12 relative."""
    checked = 0
    for case, n in _quick_runs(fresh):
        fresh_checks = fresh["runs"][case][str(n)]["verdict"]["checks"]
        old_checks = committed["runs"][case][str(n)]["verdict"]["checks"]
        assert set(fresh_checks) == set(old_checks), (
            f"{case}@{n}: probe set changed {sorted(fresh_checks)} != "
            f"{sorted(old_checks)}")
        for probe, value in fresh_checks.items():
            old = old_checks[probe]
            assert value == pytest.approx(old, rel=REL_TOL, abs=REL_TOL), (
                f"{case}@{n}: {probe} {old!r} -> {value!r}")
            checked += 1
    assert checked >= 20, "the quick grid should cover ~5 probes x 3 rungs x 2 cases"


def test_probe_ladder_shape_is_unchanged(fresh, committed):
    """Which rungs fail the gate, and from which rung each bed passes."""
    def shape(bundle):
        tolerance = float(bundle["tolerance"])
        out = {}
        for case, runs in bundle["runs"].items():
            errors = {int(n): float(runs[str(n)]["verdict"]["checks"]["fan_head"])
                      for n in runs if int(n) <= QUICK_MAX_N}
            out[case] = [n for n, err in sorted(errors.items()) if err > tolerance]
        return out

    assert shape(fresh) == shape(committed) == {"ritter-dry": [200],
                                                "stoker-wet": [200, 400]}


def test_entry_quotes_the_reproduced_accepted_grid(fresh, committed, pool):
    """``accepted_n`` and the ladder block still agree with a fresh run."""
    assert fresh["accepted_n"] == committed["accepted_n"] == 800
    assert pool["probe_ladder"]["accepted_n"] == fresh["accepted_n"]
    for case, runs in fresh["runs"].items():
        for n in runs:
            if int(n) > QUICK_MAX_N:
                continue
            assert pool["probe_ladder"]["relative_error"][case][str(int(n))] == (
                pytest.approx(committed["runs"][case][str(n)]["verdict"]["checks"]
                              ["fan_head"], rel=REL_TOL))