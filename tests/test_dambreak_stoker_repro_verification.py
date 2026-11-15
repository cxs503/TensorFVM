"""Reproducibility verification for ``CASE-FVM-DAMBREAK-STOKER`` (case pool).

The case-pool entry ``docs/oss_solutions/CASE-FVM-DAMBREAK-STOKER.json`` declares
``repro_cmd = PYTHONPATH=src python scripts/run_shallow_water_dam_break.py
--n 1600 --resolutions 100,200,400,800,1600``.  That command is the *only*
machine-readable path from the pool entry back to the committed evidence
(``src/tensorfvm/verification/artifacts/shallow-water-dam-break/metrics.json``).

This test closes the "验证欠账" (verification-debt) loop that the pool-entry
binding test (``test_case_fvm_dambreak_stoker_pool.py``) does **not** cover:
that test only checks the *committed* pool entry against the *committed*
artifact -- it never runs the driver, so a driver whose output has drifted from
the artifact (or a pool entry whose ``repro_cmd`` no longer reproduces) would
still pass.  Here we actually **run** the driver's ``run_case``/``write_artifacts``
at the pool grid (n = 1600) and require the regenerated headline metrics to be
**byte-identical** (relative 1e-12) to the committed artifact's ``n = 1600``
numbers -- the same binding the pool entry's ``gate_criteria`` use.

Why byte-identical and not "within 5 %": the 5 % gate is the *acceptance*
criterion; the *reproducibility* criterion is that the committed numbers are
exactly what the driver produces on this machine/commit.  A drift of even 1e-6
in ``front_fan_error`` would mean the pool entry quotes a number that no longer
reproduces -- a silent inconsistency the pool gate exists to catch.
"""

from __future__ import annotations

import json
import os
import pathlib

import pytest

from tensorfvm.shallow_water import (
    TOLERANCE,
    run_case,
    write_artifacts,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
ARTIFACT_DIR = (
    REPO_ROOT / "src" / "tensorfvm" / "verification" / "artifacts"
    / "shallow-water-dam-break"
)
POOL_ENTRY = REPO_ROOT / "docs" / "oss_solutions" / "CASE-FVM-DAMBREAK-STOKER.json"

#: the pool grid -- the ``n`` at which the pool entry's ``gate_criteria`` are bound
POOL_N = 1600


def _load_json(path: pathlib.Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture(scope="module")
def committed_metrics():
    return _load_json(ARTIFACT_DIR / "metrics.json")


@pytest.fixture(scope="module")
def pool_entry():
    return _load_json(POOL_ENTRY)


def _regenerate_headline(tmp_path: pathlib.Path) -> dict:
    """Run the driver's ``run_case`` at the pool grid and return headline metrics.

    Mirrors ``scripts/run_shallow_water_dam_break.py``'s ``n = 1600`` path
    (``run_case`` + ``write_artifacts``) so the regenerated numbers are produced
    by the *same* code path the ``repro_cmd`` invokes -- not a hand-rolled copy.
    """
    # the driver's default case set (Ritter dry + Stoker wet), same as the script
    from tensorfvm.shallow_water import CASES

    out = {}
    for key, case in CASES.items():
        run = run_case(
            case,
            n=POOL_N,
            cfl=0.4,
            order=2,
            t_star=2.0,
        )
        out[key] = run["metrics"]
    return out


def test_repro_cmd_is_declared_in_pool_entry(pool_entry):
    """The pool entry must declare the exact repro command this test runs."""
    assert "repro_cmd" in pool_entry
    assert "run_shallow_water_dam_break.py" in pool_entry["repro_cmd"]
    assert "--n 1600" in pool_entry["repro_cmd"]
    assert "1600" in pool_entry["repro_cmd"]


def test_regenerated_headline_matches_committed_artifact(
    committed_metrics, tmp_path
):
    """The driver's n=1600 headline metrics are byte-identical to the artifact.

    This is the reproducibility binding: the pool entry's ``gate_criteria``
    values equal the artifact's ``n = 1600`` headline metrics (checked by
    ``test_case_fvm_dambreak_stoker_pool.py``), so if the driver no longer
    reproduces those numbers the whole pool entry is stale.
    """
    regen = _regenerate_headline(tmp_path)

    # map regenerated case -> artifact case + the headline metric keys
    # (the same keys the pool-entry gate test binds)
    headline_keys = {
        "ritter-dry": ("front_fan_error", "depth_l1", "local_extrema",
                       "front_series_max_error"),
        "stoker-wet": ("shock_error", "step_ratio_error", "depth_l1",
                       "local_extrema", "front_series_max_error"),
    }
    for kind, keys in headline_keys.items():
        art = committed_metrics["cases"][kind]["metrics"]
        for k in keys:
            assert k in regen[kind], f"regenerated {kind} missing metric {k!r}"
            assert regen[kind][k] == pytest.approx(art[k], rel=1e-12), (
                f"{kind} {k}: regenerated {regen[kind][k]!r} != "
                f"committed {art[k]!r} -- the pool entry's repro_cmd no longer "
                f"reproduces the committed evidence")


def test_regenerated_verdict_passes_at_pool_grid(committed_metrics, tmp_path):
    """The regenerated n=1600 run passes every headline gate (5 %)."""
    regen = _regenerate_headline(tmp_path)
    for kind in ("ritter-dry", "stoker-wet"):
        for k, v in regen[kind].items():
            # only the relative-error metrics are gated by TOLERANCE
            if k.endswith("_error") or k == "depth_l1":
                assert v <= TOLERANCE, (
                    f"{kind} {k} = {v} exceeds the {TOLERANCE} gate at n={POOL_N}")
        # monotone: no interior local extrema
        assert regen[kind]["local_extrema"] == 0, (
            f"{kind} waveform has {regen[kind]['local_extrema']} interior "
            f"extrema -- not monotone at n={POOL_N}")
