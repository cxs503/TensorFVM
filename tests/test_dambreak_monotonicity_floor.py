"""The monotonicity gate must be able to say *which* cell failed, and why.

``CASE-FVM-DAMBREAK-STOKER`` grades the waveform with "no spurious oscillation":
``_local_extrema(h, 1e-4 h0) == 0``.  A bare count is not auditable -- the
published honest-FAIL on ``n = 240`` (``docs/case-fvm-dambreak-stoker-oblique/``)
recorded ``local_extrema = 1`` without saying how large the oscillation is or
where it sits relative to the analytic structure.  ``local_extrema_details``
closes that gap.

What is asserted here
---------------------
1. **One predicate.**  The count and the per-extremum report are computed from
   the same mask, so they can never disagree about which cell failed.  Checked
   against an independent re-implementation of the original predicate on
   random profiles (not just on a happy-path fixture).
2. **No gate drift.**  ``EXTREMUM_FRACTION`` is the published acceptance gate
   ``1e-4 h0``.  A test pins it: the point of this work is to *locate* the
   failure, never to relax the criterion.
3. **Oracle.**  The exact Stoker profile, sampled on the same bins, carries
   **zero** beyond-gate extrema.  Whatever the solver's profile carries is
   therefore produced by the solver, not by the detector.
4. **Artifact traceability.**  Every number in
   ``artifacts/shallow-water-dam-break-monotonicity/monotonicity_floor.json``
   is internally consistent (counts match records, the gate recorded equals the
   gate in force, every extremum is classified into an analytic region), and
   the honest-FAIL of the acceptance grid is still recorded as a fail.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from tensorfvm.shallow_water.dam_break import (
    EXTREMUM_FRACTION,
    _extrema_mask,
    _local_extrema,
    local_extrema_details,
)
from tensorfvm.shallow_water.dam_break import STOKER_WET
from tensorfvm.shallow_water.exact import stoker_region_bounds
from tensorfvm.shallow_water.dam_break2d_oblique import grade_profile

ARTIFACTS = os.path.join(
    os.path.dirname(__file__), "..", "src", "tensorfvm", "verification",
    "artifacts", "shallow-water-dam-break-monotonicity")
JSON_PATH = os.path.join(ARTIFACTS, "monotonicity_floor.json")
PNG_PATH = os.path.join(ARTIFACTS, "monotonicity_floor.png")


def _reference_predicate(h, amplitude):
    """The pre-existing predicate, transcribed independently of the source.

    Kept as a literal copy of the vectorised expression the count used before
    the refactor, so a silent change in the shared mask shows up as a
    disagreement instead of as a matching count.
    """
    h = np.asarray(h, dtype=float)
    if h.size < 3:
        return np.array([], dtype=int)
    left = h[1:-1] - h[:-2]
    right = h[2:] - h[1:-1]
    peak = (left > 0) & (right < 0) & (np.abs(left) > amplitude) & (np.abs(right) > amplitude)
    dip = (left < 0) & (right > 0) & (np.abs(left) > amplitude) & (np.abs(right) > amplitude)
    return np.nonzero(peak | dip)[0] + 1


# --------------------------------------------------------------------------- #
# 1. one predicate, two views
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_details_agree_with_count_and_reference(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(3, 200))
    h = 1.0 + 1e-2 * rng.standard_normal(n)
    for amplitude in (0.0, 1e-4, 1e-2, 1.0):
        details = local_extrema_details(h, amplitude)
        assert len(details) == _local_extrema(h, amplitude)
        assert [rec["index"] for rec in details] == list(
            _reference_predicate(h, amplitude))


def test_details_report_kind_amplitude_and_value():
    h = np.array([1.0, 1.001, 0.999, 1.0, 1.0])
    peak, dip = local_extrema_details(h, 1e-4)
    assert peak["index"] == 1 and peak["kind"] == "peak"
    assert peak["value"] == pytest.approx(1.001)
    assert peak["left_slope"] == pytest.approx(0.001)
    assert peak["right_slope"] == pytest.approx(-0.002)
    # amplitude is the *smaller* of the two jumps: a conservative prominence
    assert peak["amplitude"] == pytest.approx(0.001)
    assert dip["index"] == 2 and dip["kind"] == "dip"
    assert dip["amplitude"] == pytest.approx(0.001)


def test_details_amplitudes_respect_the_gate():
    rng = np.random.default_rng(7)
    h = 1.0 + 2e-3 * rng.standard_normal(400)
    gate = 1e-4
    for rec in local_extrema_details(h, gate):
        assert rec["amplitude"] > gate
        assert rec["left_slope"] * rec["right_slope"] < 0.0


def test_details_carry_coordinates_when_given():
    h = np.array([1.0, 1.01, 1.0])
    x = np.array([-2.5, -2.0, -1.5])
    (rec,) = local_extrema_details(h, 1e-4, x=x)
    assert rec["x"] == pytest.approx(-2.0)


def test_details_reject_mismatched_coordinates():
    with pytest.raises(ValueError, match="same shape"):
        local_extrema_details(np.ones(4), 1e-4, x=np.arange(5.0))


def test_degenerate_profiles_are_silent():
    assert local_extrema_details(np.array([]), 1e-4) == []
    assert local_extrema_details(np.array([1.0]), 1e-4) == []
    assert local_extrema_details(np.array([1.0, 2.0]), 1e-4) == []
    assert _extrema_mask(np.array([1.0, 2.0]), 1e-4).sum() == 0


def test_monotone_profiles_have_no_extrema():
    for h in (np.linspace(1.0, 0.5, 201), np.linspace(0.5, 1.0, 201),
              np.ones(201), np.linspace(1.0, 0.5, 201) ** 2):
        assert _local_extrema(h, 1e-4) == 0


# --------------------------------------------------------------------------- #
# 2. the published gate is not to be relaxed by this work
# --------------------------------------------------------------------------- #
def test_extremum_gate_is_the_published_value():
    assert EXTREMUM_FRACTION == 1e-4


# --------------------------------------------------------------------------- #
# 3. oracle: the exact solution trips no beyond-gate extremum
# --------------------------------------------------------------------------- #
def _exact_profile(n=240, domain=12.0):
    ds = domain / float(n)
    s = (np.arange(n) + 0.5) * ds - 0.5 * domain
    h, _ = STOKER_WET.exact(s, STOKER_WET.t_end)
    return s, np.asarray(h, dtype=float)


def test_exact_stoker_profile_is_free_of_beyond_gate_extrema():
    s, h = _exact_profile()
    assert local_extrema_details(h, EXTREMUM_FRACTION * STOKER_WET.h0) == []
    # and the same at the gate the reports quote as an amplitude, for scale
    assert _local_extrema(h, EXTREMUM_FRACTION * STOKER_WET.h0) == 0


def test_exact_stoker_profile_regions_are_ordered():
    """Guard the classifier the artifact relies on (head < tail < shock)."""
    b = stoker_region_bounds(STOKER_WET.h0, STOKER_WET.h1, STOKER_WET.gravity)
    assert b["head"] < b["tail"] < b["shock"]
    assert b["h1"] if "h1" in b else True
    assert b["h2"] > STOKER_WET.h1


# --------------------------------------------------------------------------- #
# 4. the oblique grader now records *what* failed, not only how many
# --------------------------------------------------------------------------- #
def test_grade_profile_reports_matching_detail_count():
    s, h = _exact_profile()
    # a one-cell dip just above the gate, placed inside the plateau
    b = stoker_region_bounds(STOKER_WET.h0, STOKER_WET.h1, STOKER_WET.gravity)
    j = int(np.argmin(np.abs(s - 0.5 * (b["tail"] + b["shock"]) * STOKER_WET.t_end)))
    h = h.copy()
    h[j] -= 1.3 * EXTREMUM_FRACTION * STOKER_WET.h0
    fake = {"s": s, "h": h, "u_n": np.zeros_like(h),
            "keep": np.ones(s.size, dtype=bool), "cells": np.ones(s.size)}
    graded = grade_profile(STOKER_WET, fake, STOKER_WET.t_end)
    assert graded["local_extrema"] == len(graded["local_extrema_details"]) >= 1
    (rec,) = graded["local_extrema_details"]
    assert rec["x"] == pytest.approx(float(s[j]))
    assert rec["amplitude"] > EXTREMUM_FRACTION * STOKER_WET.h0


# --------------------------------------------------------------------------- #
# 5. artifact traceability
# --------------------------------------------------------------------------- #
pytestmark_artifact = pytest.mark.skipif(
    not os.path.exists(JSON_PATH),
    reason="audit artifact not generated (run scripts/audit_dambreak_monotonicity_floor.py)")


@pytest.fixture(scope="module")
def payload():
    if not os.path.exists(JSON_PATH):
        pytest.skip("audit artifact missing")
    with open(JSON_PATH, encoding="utf-8") as fh:
        return json.load(fh)


@pytestmark_artifact
class TestArtifact:
    def test_gate_recorded_equals_gate_in_force(self, payload):
        declared = payload["declared"]
        assert declared["gate_amplitude_fraction_of_h0"] == EXTREMUM_FRACTION
        h0 = (payload.get("case_params") or {}).get("h0", STOKER_WET.h0)
        assert declared["gate_amplitude_absolute"] == pytest.approx(
            EXTREMUM_FRACTION * h0)

    def test_counts_match_records(self, payload):
        for row in payload["runs_2d"] + payload["runs_1d"]:
            assert row["extrema_count"] == len(row["extrema"])
            assert row["oracle_extrema_count"] >= 0
            for rec in row["extrema"]:
                assert rec["amplitude"] > payload["declared"]["gate_amplitude_absolute"]
                assert rec["kind"] in ("peak", "dip")
                assert rec["region"] in ("upstream", "fan", "plateau", "downstream")

    def test_magnitude_metrics_pass_where_waveform_fails(self, payload):
        """The honest-FAIL is a monotonicity-only failure: say so explicitly."""
        assert payload["verdict"]["all_magnitude_metrics_pass"] is True
        for row in payload["runs_2d"]:
            mm = row["magnitude_metrics"]
            for key in ("bore_position_error", "depth_l1", "level_step_error"):
                assert mm[key] <= mm["tolerance"]

    def test_acceptance_grid_still_fails_the_waveform_clause(self, payload):
        """honest-FAIL is preserved, not papered over by this re-analysis."""
        rows = {row["n"]: row for row in payload["runs_2d"]}
        assert 240 in rows, "the anchor's acceptance grid must be in the ladder"
        assert rows[240]["extrema_count"] >= 1

    def test_exact_profile_oracle_is_clean_on_every_grid(self, payload):
        for row in payload["runs_2d"] + payload["runs_1d"]:
            assert row["oracle_extrema_count"] == 0

    def test_extrema_are_localised_in_the_analytic_structure(self, payload):
        for row in payload["runs_2d"]:
            for rec in row["extrema"]:
                assert rec["nearest_wall"] in ("head", "tail", "shock")
                assert rec["nearest_wall_cells"] >= 0.0

    def test_figure_is_committed(self):
        assert os.path.exists(PNG_PATH)
        assert os.path.getsize(PNG_PATH) > 5000