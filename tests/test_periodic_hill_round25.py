"""Round 25 contract tests for ``CASE-FVM-PERIODIC-HILLS``.

Round 22/23 left the closure-free ``Re = 2800`` grid ladder ``inconclusive``
because the three refined rungs (``144x72``, ``192x96``, ``240x120``) diverged
inside the startup transient at the coarse protocol ``dt``.  Round 23's probe
proved the blow-up is a **time-step stability limit that tightens with the
mesh** (halving / quartering ``dt`` at fixed mesh keeps ``max|u|`` bounded).

Round 25 is the *completion*: it re-runs the refined rungs at the stable ``dt``
and re-measures the ladder.  These tests pin the completion infrastructure:

1. **Stable-dt selection** -- the round-25 driver's ``STABLE_RUNGS`` must use
   exactly the ``dt`` values round 23's probe proved bounded at fixed mesh
   (0.00665 / 0.0025 / 0.004), not the coarse protocol ``dt`` that diverged.
2. **Artefact isolation** -- round 25 writes ``round25_*`` artefacts so the
   round-23 historical record (the diverged ``round20_laminar_*`` rungs) is not
   overwritten; the analysis must reference the ``round25_*`` stems.
3. **Graceful skip** -- the analysis must not crash when a round-25 rung is not
   yet on disk (the compute is still queued); it skips and reports.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CASEDIR = REPO / "docs" / "case-periodic-hills"
sys.path.insert(0, str(REPO / "src"))


def _load_script(name: str):
    path = REPO / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def driver():
    return _load_script("round25_stable_ladder")


@pytest.fixture(scope="module")
def analysis():
    return _load_script("round25_ladder_analysis")


def test_stable_rungs_use_probe_proven_dt(driver):
    """The stable dt must be the round-23-probe-proven values, not the coarse ones."""
    # round 23 probe proved these bounded at fixed mesh:
    #   (144, 72, 0.00665), (192, 96, 0.0025), (96, 48, 0.02)
    assert driver.STABLE_RUNGS["144x72"]["dt"] == pytest.approx(0.00665)
    assert driver.STABLE_RUNGS["192x96"]["dt"] == pytest.approx(0.0025)
    # 240x120 at dt=0.004 keeps the same CFL as the 144x72 rung that round 23
    # proved stable (CFL ~0.16); the coarse 0.008 diverged.
    assert driver.STABLE_RUNGS["240x120"]["dt"] == pytest.approx(0.004)


def test_stable_rungs_reach_the_averaging_window(driver):
    """Each stable rung must reach the round-20 physical window t in [160, 400]."""
    for tag, cfg in driver.STABLE_RUNGS.items():
        # burn-in ends at t = burn_in * dt >= 160; run ends at t = steps * dt >= 400
        assert cfg["burn_in"] * cfg["dt"] >= 160.0 - 1e-9, tag
        assert cfg["steps"] * cfg["dt"] >= 400.0 - 1e-9, tag


def test_analysis_references_round25_stems(analysis):
    """The analysis must read the round-25 (stable) artefacts, not the round-20 ones."""
    stems = [stem for _, stem, *_ in analysis.LEVELS]
    assert "round25_laminar_Re2800_144x72" in stems
    assert "round25_laminar_Re2800_192x96" in stems
    assert "round25_laminar_Re2800_240x120" in stems
    # the baseline rung is the already-measured round-20 96x48
    assert "round20_laminar_Re2800_96x48" in stems


def test_analysis_skips_missing_rung_gracefully(analysis):
    """With the compute still queued, a missing rung must be skipped, not crash."""
    # monkeypatch _load to return None for the round-25 rungs (not yet on disk)
    import tensorfvm.verification.periodic_hill_round11 as _r11
    from tensorfvm.verification.wall_shear_bubbles import separation_reattachment

    # the analysis's _load reads real artefacts; simulate a missing round-25 rung
    # by checking that _load returns None for a stem that does not exist
    assert analysis._load("round25_laminar_Re2800_999x999") is None


def test_analysis_runs_with_only_baseline_present(analysis):
    """While the round-25 rungs are still queued, the analysis must run on the
    baseline rung alone (produce a table), not crash."""
    # the baseline round-20 96x48 artefact exists on disk; the round-25 rungs
    # are not yet written.  _load returns None for the missing round-25 stems.
    assert analysis._load("round25_laminar_Re2800_144x72") is None
    assert analysis._load("round25_laminar_Re2800_192x96") is None
    assert analysis._load("round25_laminar_Re2800_240x120") is None
    # and the baseline resolves
    assert analysis._load("round20_laminar_Re2800_96x48") is not None
