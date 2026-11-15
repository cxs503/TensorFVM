"""Round-18 grid ladder: artefact/score consistency for ``CASE-FVM-PERIODIC-HILLS``.

The round-17 lesson was that a *derived* number in an artefact (the reattachment
crossing) can silently outlive the operator that produced it, so these tests pin
two different things:

* the grid-law fit's *behaviour* on synthetic sequences where the answer is
  known (order, extrapolant, and the refusal to fit a non-monotone sequence);
* the *cross-consistency* of the round-18 artefacts: the ``x_r/H`` and the
  ``y/H = 0.5`` verdict stored in ``round18_ladder.json`` must be reproducible
  from the stored time means (``mean_tau_b`` / ``mean_u``) with the certified
  operator, and the ``omega`` sink coefficient actually in force must be the
  model-consistent one (``S_P/(beta omega) = 1``).

No PASS/FAIL physics assertion is hard-coded: the acceptance verdict is
recomputed from the numbers and compared with the recorded one, so a hand-edited
verdict fails the test instead of the (honest) FAIL becoming a silent PASS.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import round18_ladder_analysis as r18  # noqa: E402

OUTDIR = REPO / "docs" / "case-periodic-hills"
LADDER = OUTDIR / "round18_ladder.json"


# --------------------------------------------------------------------------- #
# grid law fit
# --------------------------------------------------------------------------- #
def test_fit_recovers_known_order_and_extrapolant():
    x0, C, p = 5.0, 2.0, 1.5
    levels = [(0.1, x0 + C * 0.1 ** p), (0.0625, x0 + C * 0.0625 ** p),
              (0.046875, x0 + C * 0.046875 ** p)]
    law = r18.fit_grid_law(levels)
    assert law["admissible"] is True
    assert law["observed_order_p"] == pytest.approx(p, abs=2e-2)
    assert law["extrapolated_x_r_over_H"] == pytest.approx(x0, abs=1e-2)
    assert law["rms_residual"] < 1e-2


def test_fit_refuses_non_monotone_sequence():
    law = r18.fit_grid_law([(0.1, 7.65), (0.0625, 7.68), (0.046875, 7.20)])
    assert law["admissible"] is False
    assert law["monotone_in_h"] is False
    assert law["reason"]


def test_fit_refuses_two_levels():
    law = r18.fit_grid_law([(0.1, 7.6), (0.05, 7.5)])
    assert law["admissible"] is False
    assert "3 required" in law["reason"]


def test_fit_ignores_non_finite_levels():
    law = r18.fit_grid_law([(0.1, 7.6), (0.05, float("nan")), (0.025, 7.4)])
    assert law["admissible"] is False


# --------------------------------------------------------------------------- #
# artefact consistency
# --------------------------------------------------------------------------- #
def _existing_levels():
    return [lv for lv in r18.LEVELS
            if r18.level_paths(lv[0], lv[1], OUTDIR)[0].exists()]


def test_ladder_artefacts_exist():
    missing = [f"{nx}x{ny}" for nx, ny in r18.LEVELS
               if not r18.level_paths(nx, ny, OUTDIR)[0].exists()]
    if missing:
        pytest.skip(f"ladder artefacts not fetched into the tree: {missing}")
    assert LADDER.exists()


def test_sink_coefficient_is_the_model_consistent_one():
    """An artefact must not claim a variant it did not run (round-16 contract)."""
    levels = _existing_levels()
    if not levels:
        pytest.skip("no ladder artefacts in the tree")
    for nx, ny in levels:
        _, js = r18.level_paths(nx, ny, OUTDIR)
        if not js.exists():
            continue
        meta = json.loads(js.read_text())
        ratio = (meta.get("sink_coefficient_ratio") or {}).get("ratio")
        assert ratio == pytest.approx(1.0, abs=1e-9), f"{nx}x{ny} sink ratio {ratio}"


def test_recorded_verdict_is_reproducible_from_stored_means():
    """Re-score every level from ``mean_tau_b`` / ``mean_u`` and compare."""
    levels = _existing_levels()
    if len(levels) < 3:
        pytest.skip("ladder artefacts incomplete in the tree")
    recorded = json.loads(LADDER.read_text())
    ref = r18.X_R_OVER_H
    for (nx, ny), row in zip(levels, recorded["levels"]):
        npz, js = r18.level_paths(nx, ny, OUTDIR)
        meta = json.loads(js.read_text()) if js.exists() else {}
        scored = r18.score_from_npz(npz, meta.get("config") or {})
        assert row["mesh"] == [nx, ny]
        assert row["x_r_over_H"] == pytest.approx(scored["x_r_over_H"], abs=1e-9)
        assert row["x_s_over_H"] == pytest.approx(scored["x_s_over_H"], abs=1e-9)
        expected_pass = abs(scored["x_r_over_H"] - ref) / ref <= r18.X_R_TOLERANCE
        assert row["x_r_pass"] is expected_pass
        assert row["x_r_error_vs_reference"] == pytest.approx(
            abs(scored["x_r_over_H"] - ref) / ref, rel=1e-12)


def test_ladder_spacing_decreases_monotonically():
    hs = [r18.HILL_LENGTH / nx for nx, _ in r18.LEVELS]
    assert all(a > b for a, b in zip(hs, hs[1:]))


def test_verdict_cannot_disagree_with_levels():
    if not LADDER.exists():
        pytest.skip("no round-18 artefact in the tree")
    recorded = json.loads(LADDER.read_text())
    verdict = recorded["verdict"]
    ref = r18.X_R_OVER_H
    any_pass = any(
        math.isfinite(lv["x_r_over_H"])
        and abs(lv["x_r_over_H"] - ref) / ref <= r18.X_R_TOLERANCE
        for lv in recorded["levels"])
    assert verdict["any_level_inside_band"] is any_pass
    law = recorded["grid_law"]
    if law["admissible"]:
        extrap_pass = abs(law["extrapolated_x_r_over_H"] - ref) / ref <= r18.X_R_TOLERANCE
        assert verdict["extrapolated_inside_band"] is extrap_pass
    else:
        assert verdict["extrapolated_inside_band"] is False
    assert isinstance(verdict["honest_summary"], str) and verdict["honest_summary"]