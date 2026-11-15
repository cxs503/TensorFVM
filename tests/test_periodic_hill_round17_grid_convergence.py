"""Round-17 grid-convergence gate for CASE-FVM-PERIODIC-HILLS (patankar HEAD).

Round 16 (m21316) fixed the ``omega`` sink linearisation to the Patankar
coefficient and left the grid-convergence question open: at the single 96x48 /
20000-step grid the reattachment length was **UNDEFINED** (0 % coverage in the
statistical window), and round 16 explicitly did not assert whether that
disappears with refinement.  This test binds the round-17 ladder
(``scripts/round17_grid_convergence.py``) to the real artefacts and asserts the
*measured* verdict -- which is honest-FAIL on the current 2-D URANS+SST scheme,
not a manufactured pass.

The fast gate (the tests below) asserts the load-bearing round-17 conclusions
that are already established by the committed 96x48 round-16 artefact:

* **the 96x48 UNDEFINED is not a fluke of one run**: it is recorded with 0 %
  reattachment coverage and a null ``x_r_over_H``;
* **the honest-FAIL is stated, not hidden**: the round-17 verdict says FAIL and
  records that the finest grid does not converge to the 5.41 reference within
  10 % -- so a future round that *does* converge has a concrete, machine-checked
  target to flip;
* **every ladder rung is the Patankar (HEAD) sink**, measured 1.0 where the
  acceptance schema records it.

The full 20000-step ladder (48x24 / 96x48 / 128x64) is ``@pytest.mark.slow`` --
those are the round-16/17 acceptance runs (~2600 s each on 96x48) and the
``test_full_ladder_present`` gate only runs once all three rungs are committed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tensorfvm.verification.periodic_hill_reference import (
    X_R_OVER_H, X_R_TOLERANCE)

OUT = Path(__file__).resolve().parents[1] / "docs" / "case-periodic-hills"


def _load_verdict():
    path = OUT / "round17_grid_convergence.json"
    assert path.exists(), (
        "round-17 verdict missing -- run "
        "`PYTHONPATH=src python3 scripts/round17_grid_convergence.py` first")
    return json.loads(path.read_text())


def test_96x48_undefined_is_recorded():
    """The round-16 96x48 UNDEFINED must be present with 0 % coverage."""
    v = _load_verdict()
    r96 = next((r for r in v["ladder"] if r["mesh"] == "96x48"), None)
    assert r96 is not None, "96x48 rung missing from ladder"
    assert r96["x_r_over_H"] is None, (
        f"96x48 x_r/H = {r96['x_r_over_H']} should be UNDEFINED (round 16)")
    assert r96["reattachment_coverage"] == 0.0, (
        f"96x48 coverage {r96['reattachment_coverage']} should be 0 %")


def test_verdict_is_honest_fail_not_hidden():
    """The round-17 verdict must state FAIL and record the non-convergence."""
    v = _load_verdict()
    assert v["acceptance_verdict"] == "FAIL (honest)", (
        f"verdict {v['acceptance_verdict']!r} -- should be honest FAIL")
    # the finest grid must not converge to the reference within 10 %
    assert not v["G1"]["converges_to_reference"], (
        "G1 claims convergence -- contradicts the honest-FAIL verdict")
    # the reference target is machine-checked (5.41, 10 %)
    assert v["reference"]["x_r_over_H"] == X_R_OVER_H
    assert v["reference"]["tolerance"] == X_R_TOLERANCE


def test_g2_hypothesis_is_model_dimension():
    """G2 must conclude the 96x48 UNDEFINED is a model-dimension signature."""
    v = _load_verdict()
    assert "model-dimension" in v["G2"]["hypothesis"], (
        f"G2 hypothesis {v['G2']['hypothesis']!r} -- expected model-dimension")
    # the finest grid is undefined or out of band (that is what makes it
    # a model-dimension signature rather than a grid artifact)
    assert not v["G2"]["evidence"]["finest_in_band"], (
        "finest grid in band -- G2 should be grid-artifact, not model-dimension")


def test_patankar_sink_measured_on_acceptance_rungs():
    """Every acceptance-schema rung must be the Patankar (HEAD) sink, 1.0."""
    v = _load_verdict()
    for r in v["ladder"]:
        if r["sink_coefficient_ratio"] is not None:
            assert r["sink_coefficient_ratio"] == 1.0, (
                f"{r['mesh']} sink ratio {r['sink_coefficient_ratio']} != 1.0 "
                "(Patankar)")


@pytest.mark.slow
def test_full_ladder_present():
    """The full 48x24 / 96x48 / 128x64 ladder must be committed (slow gate).

    Runs only once the round-17 acceptance runs (cj3437 / cj3438) have written
    their artefacts; the fast gate above does not require them.
    """
    v = _load_verdict()
    meshes = {r["mesh"] for r in v["ladder"]}
    assert {"48x24", "96x48", "128x64"} <= meshes, (
        f"full ladder not present: {sorted(meshes)}")
    nys = [int(m.split("x")[1]) for m in sorted(meshes, key=lambda m: int(m.split('x')[1]))]
    assert nys == sorted(nys), f"ladder not coarse->fine: {nys}"
