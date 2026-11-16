"""Round 29: the dt = 0.01 ladder completion for CASE-FVM-PERIODIC-HILLS.

Round 19 re-ran the ``96x48`` dt-control and the ``120x60`` level at the stable
step ``dt = 0.01`` but the two artefacts never landed in the repository (the
round-19 audit classifies both as ``NO_ARTEFACT``).  Round 29 re-submits them
(``r24_<mesh>_dt0.01``) and re-scores every ladder level with the certified
zero-crossing operator.

These tests pin the *invariants* of the round-24 analysis:

* a level whose artefact is absent is ``NO_ARTEFACT`` and carries no ``x_r/H``
  (a missing artefact can never be turned into a measurement);
* a level with a recorded divergence is ``DIVERGED`` and is kept out of the
  grid-law fit (round-19 precedence, re-asserted here so the round-24 ladder
  cannot regress to the round-18 error);
* the dt-control is only ``available`` when the same mesh is measured at two
  time steps, and ``time_step_converged_5pct`` is a pure function of the two
  values;
* the grid law is admissible only with >= 3 admissible levels and a monotone
  sequence (the round-18/19 "2 admissible, 3 required" guard);
* the verdict's two open questions (time-step / grid) are answered only from
  measured data and are ``WITHHELD`` otherwise.

Tests that need the round-24 artefacts are skipped when the artefacts are not
present, so the suite is green both before and after the cluster runs land.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import round29_ladder_completion as r29  # noqa: E402
from round18_ladder_analysis import X_R_OVER_H  # noqa: E402

OUTDIR = REPO / "docs" / "case-periodic-hills"


def _r29_present(tag: str) -> bool:
    return (OUTDIR / f"round11_sst_{tag}.npz").exists()


def _audit() -> dict:
    return r29.analyse(figures=False)


# --------------------------------------------------------------------------- #
# 1. the round-24 levels are present in the ladder with the right metadata
# --------------------------------------------------------------------------- #
def test_r24_levels_are_in_the_ladder():
    tags = {t for (t, *_rest) in r29.LADDER}
    assert "r24_96x48_dt0.01" in tags
    assert "r24_120x60_dt0.01" in tags
    # every entry is a normalised 6-tuple
    for entry in r29.LADDER:
        assert len(entry) == 6, entry
        _tag, nx, ny, dt, _round, window = entry
        assert nx > 0 and ny > 0 and dt > 0
        assert window[0] < window[1]


def test_r24_120x60_window_is_flagged_shortened():
    # the 120x60 level uses a shortened window to fit the job budget; the
    # analysis must record that so a reader cannot mistake it for the protocol
    # t in [160, 400] window.
    spec = [e for e in r29.LADDER_R29 if e[0] == "r24_120x60_dt0.01"][0]
    assert spec[5][1] < 400.0, "120x60 window must be the shortened one"
    out = _audit()
    lv = [l for l in out["levels"] if l["tag"] == "r24_120x60_dt0.01"][0]
    assert lv.get("window_shortened") is True
    assert lv.get("averaging_window") == [160.0, 370.0]


# --------------------------------------------------------------------------- #
# 2. classification invariants (hold regardless of whether the runs have landed)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("tag,expected", [
    ("r18_120x60", "DIVERGED"),
    ("r18_144x72", "DIVERGED"),
])
def test_round18_fine_levels_stay_diverged(tag, expected):
    lv = r29.classify_level(tag, 120 if "120" in tag else 144,
                            60 if "120" in tag else 72, 0.02, 18)
    assert lv["status"] == expected
    assert lv["x_r_over_H"] is None


def test_missing_artefact_is_no_artefact_not_a_measurement():
    for tag in ("r24_96x48_dt0.01", "r24_120x60_dt0.01"):
        if _r29_present(tag):
            continue
        lv = r29.classify_level(tag, 96 if "96" in tag else 120,
                                48 if "96" in tag else 60, 0.01, 24)
        assert lv["status"] == "NO_ARTEFACT"
        assert lv.get("x_r_over_H") is None


def test_measured_level_is_inside_or_outside_band_consistently():
    out = _audit()
    band = out["verdict"]["reference_band"]
    for lv in out["levels"]:
        if lv["status"] != "MEASURED":
            continue
        inside = band[0] <= lv["x_r_over_H"] <= band[1]
        assert lv["x_r_pass"] == inside, lv["tag"]
        # the error is the relative distance to the reference
        assert np.isclose(lv["x_r_error_vs_reference"],
                          abs(lv["x_r_over_H"] - X_R_OVER_H) / X_R_OVER_H)


# --------------------------------------------------------------------------- #
# 3. dt-control invariants
# --------------------------------------------------------------------------- #
def test_dt_control_requires_two_steps_on_one_mesh():
    # with only the dt=0.02 96x48 measured, the control is unavailable
    out = _audit()
    ctrl = out["dt_control"]
    if not _r29_present("r24_96x48_dt0.01"):
        assert ctrl["available"] is False
        return
    # if the control has landed, it must be available and self-consistent
    assert ctrl["available"] is True
    a = ctrl["x_r_over_H"]["0.02"]
    b = ctrl["x_r_over_H"]["0.01"]
    assert np.isclose(ctrl["relative_change"], abs(a - b) / abs(b))
    assert ctrl["time_step_converged_5pct"] == (
        ctrl["relative_change"] <= 0.05)


def test_dt_question_is_withheld_without_the_control():
    out = _audit()
    q = out["verdict"]["question_time_step"]
    if not _r29_present("r24_96x48_dt0.01"):
        assert q["answered"] is False
        assert "WITHHELD" in q["conclusion"]
    else:
        assert q["answered"] is True
        assert q["conclusion"]


# --------------------------------------------------------------------------- #
# 4. grid-law invariants
# --------------------------------------------------------------------------- #
def test_grid_law_needs_three_admissible_levels():
    out = _audit()
    law = out["verdict"]["grid_law"]
    n_measured = out["verdict"]["n_measured_levels"]
    if n_measured < 3:
        assert law["admissible"] is False
        assert "3 required" in str(law.get("reason", ""))
    else:
        # with >= 3 levels the fit is attempted; admissibility then depends on
        # monotonicity, which we do not force either way
        assert isinstance(law["admissible"], bool)


def test_grid_question_matches_law_admissibility():
    out = _audit()
    law = out["verdict"]["grid_law"]
    q = out["verdict"]["question_grid"]
    assert q["answered"] == bool(law["admissible"])


# --------------------------------------------------------------------------- #
# 5. the verdict is honest: no level inside the band unless one truly is
# --------------------------------------------------------------------------- #
def test_acceptance_pass_matches_any_level_inside_band():
    out = _audit()
    v = out["verdict"]
    assert v["acceptance_pass"] == v["any_level_inside_band"]
    # the reference and its band are the single-sourced Breuer/ERCOFTAC values
    assert v["reference_x_r_over_H"] == X_R_OVER_H
    assert np.isclose(v["reference_band"][0], X_R_OVER_H * 0.9)
    assert np.isclose(v["reference_band"][1], X_R_OVER_H * 1.1)


def test_json_artefact_is_written_and_consistent():
    out = _audit()
    path = OUTDIR / "round29_ladder_completion.json"
    assert path.exists()
    on_disk = json.loads(path.read_text())
    assert on_disk["round"] == 24
    assert len(on_disk["levels"]) == len(out["levels"])
    # the on-disk verdict matches the in-memory one on the key fields
    for key in ("n_measured_levels", "any_level_inside_band",
                "acceptance_pass"):
        assert on_disk["verdict"][key] == out["verdict"][key], key
