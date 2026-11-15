"""Regression gate for the shallow-water order audit.

Thresholds are the measured values from
``src/tensorfvm/verification/artifacts/shallow-water-order-audit/summary.json``
with an explicit margin; every number below is recomputed here, none is copied
from the artifact.
"""

from __future__ import annotations

import json
import math
import pathlib

import numpy as np
import pytest

from tensorfvm.shallow_water.dam_break import RITTER_DRY, STOKER_WET
from tensorfvm.shallow_water.order_audit import (
    ARTIFACT_DIR,
    ENTRY_PATH,
    FAN_INTERIOR_FRACTION,
    ODX_COLLAPSE_TOLERANCE,
    audit_entry_claim,
    case_fan_window,
    fan_interior_study,
    fan_mask,
    load_case_pool_entry,
    ritter_fan_window,
    smooth_bump_state,
    smooth_bump_study,
    stoker_fan_window,
)

REPO = pathlib.Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# analytic windows and masks (no solver)
# --------------------------------------------------------------------------- #
def test_ritter_window_is_the_trimmed_analytic_fan():
    w = ritter_fan_window(1.0, 9.81)
    c0 = math.sqrt(9.81)
    assert w["lo_full"] == pytest.approx(-c0)
    assert w["hi_full"] == pytest.approx(2.0 * c0)
    # trimming is exactly the declared fraction of the fan width at both ends
    assert w["lo"] - w["lo_full"] == pytest.approx(FAN_INTERIOR_FRACTION * w["width_full"])
    assert w["hi_full"] - w["hi"] == pytest.approx(FAN_INTERIOR_FRACTION * w["width_full"])
    assert w["lo"] < w["hi"]


def test_stoker_window_excludes_bore_and_post_bore_state():
    w = stoker_fan_window(1.0, 0.5, 9.81)
    assert w["lo"] > w["lo_full"] == pytest.approx(-math.sqrt(9.81))
    # the window ends inside the rarefaction, i.e. before the constant state
    # and the bore that follows it
    assert w["hi"] < w["hi_full"] < w["shock"]


def test_case_fan_window_dispatches_on_bed_type():
    assert case_fan_window(RITTER_DRY)["hi_full"] == pytest.approx(
        2.0 * math.sqrt(9.81 * RITTER_DRY.h0))
    assert case_fan_window(STOKER_WET)["shock"] == pytest.approx(
        stoker_fan_window(STOKER_WET.h0, STOKER_WET.h1, STOKER_WET.gravity)["shock"])


def test_fan_mask_is_open_interval():
    xi = np.array([-1.0, 0.0, 0.5, 1.0])
    assert fan_mask(xi, 0.0, 1.0).tolist() == [False, False, True, False]


@pytest.mark.parametrize("fraction", [-0.1, 0.5, 0.75])
def test_window_rejects_illegal_trim_fraction(fraction):
    with pytest.raises(ValueError):
        ritter_fan_window(1.0, 9.81, fraction)


def test_smooth_bump_state_is_a_still_positive_bump():
    x = np.linspace(-20.0, 20.0, 401)
    h = smooth_bump_state(x)
    assert h.min() > 0.9 and h.max() == pytest.approx(1.1, rel=1e-12)
    # symmetric about the basin centre, no discontinuity
    assert np.max(np.abs(np.diff(h))) < 0.01


# --------------------------------------------------------------------------- #
# smooth-regime certification: the operator is second order
# --------------------------------------------------------------------------- #
def test_smooth_regime_is_second_order():
    study = smooth_bump_study((100, 200, 400))
    assert study["gradable"] is True
    # measured 1.82 on this ladder; the design order is 2
    assert study["observed_order_asymptotic"] >= 1.7
    assert 1.7 <= study["observed_order_fit"] <= 2.3


def test_smooth_regime_is_translation_invariant_in_the_bump_amplitude():
    """A weaker bump must not fake second order: order is a rate, not a level."""
    strong = smooth_bump_study((100, 200), config=None)
    weak = smooth_bump_study((100, 200), config={"amplitude": 0.01})
    assert weak["pairs"][-1]["l1"] < strong["pairs"][-1]["l1"]


# --------------------------------------------------------------------------- #
# dam-break fan interior: data-driven vs solution-driven first order
# --------------------------------------------------------------------------- #
def test_dry_bed_from_step_data_is_first_order_with_odx_limit():
    st = fan_interior_study(RITTER_DRY, "step", (100, 200, 400, 800))
    assert st["gradable"] is True
    assert st["asymptotic_order_l1"] <= 1.3          # measured 1.16
    assert st["collapse_finest_relative"] <= ODX_COLLAPSE_TOLERANCE  # measured 0.020


def test_dry_bed_restart_from_continuous_data_recovers_second_order():
    st = fan_interior_study(RITTER_DRY, "smooth", (100, 200, 400))
    assert st["gradable"] is True
    assert st["asymptotic_order_l1"] >= 1.8          # measured 2.22


def test_wet_bed_first_order_is_intrinsic_to_the_bore():
    st = fan_interior_study(STOKER_WET, "step", (100, 200, 400, 800, 1600))
    assert st["gradable"] is True
    assert st["asymptotic_order_l1"] <= 1.3          # measured 1.04
    assert st["collapse_finest_relative"] <= ODX_COLLAPSE_TOLERANCE  # measured 0.012


def test_wet_bed_restart_is_reported_as_non_gradable():
    """The closed form already carries the bore, so the restart cannot be
    regularised: its error sequence is non-monotone and no order is quoted."""
    st = fan_interior_study(STOKER_WET, "smooth", (100, 200, 400))
    assert st["error_monotone"] is False
    assert st["gradable"] is False


def test_fan_window_grows_with_resolution_and_holds_the_graded_cells():
    st = fan_interior_study(RITTER_DRY, "step", (100, 200))
    cells = [row["window_cells"] for row in st["rows"]]
    assert cells == sorted(cells) and cells[1] == pytest.approx(2 * cells[0], rel=0.05)
    assert all(0.2 < row["window_fraction"] < 0.9 for row in st["rows"])


# --------------------------------------------------------------------------- #
# the entry-claim gate must be able to fail
# --------------------------------------------------------------------------- #
def test_committed_artifact_supports_the_committed_entry():
    record = json.loads((ARTIFACT_DIR / "summary.json").read_text(encoding="utf-8"))
    claim = audit_entry_claim(record, load_case_pool_entry())
    entry_order = load_case_pool_entry()["solver"]["order"]
    assert claim["advertised_order"] == entry_order
    assert claim["status"] == "SUPPORTED", claim["reason"]
    assert record["verdict"]["all_gates_pass"] is True


@pytest.mark.parametrize("advertised,status", [(3, "OVERSTATED"), (1, "UNDERSTATED")])
def test_entry_claim_gate_rejects_mismatched_orders(advertised, status):
    record = json.loads((ARTIFACT_DIR / "summary.json").read_text(encoding="utf-8"))
    entry = load_case_pool_entry()
    entry["solver"] = dict(entry["solver"], order=advertised)
    assert audit_entry_claim(record, entry)["status"] == status


def test_entry_claim_withheld_when_entry_has_no_order():
    record = json.loads((ARTIFACT_DIR / "summary.json").read_text(encoding="utf-8"))
    entry = {"solver": {}}
    assert audit_entry_claim(record, entry)["status"] == "WITHHELD"


def test_committed_entry_path_is_the_anchored_case():
    entry = load_case_pool_entry()
    assert entry["slug"] == "CASE-FVM-DAMBREAK-STOKER"
    assert ENTRY_PATH.as_posix() == "docs/case-fvm-dambreak-stoker/case_pool_entry.json"
    assert (REPO / ENTRY_PATH).is_file()