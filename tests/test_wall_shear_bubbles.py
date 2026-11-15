"""Certification of the round-17 wall-shear bubble operator.

The point of these tests is *not* that the code runs: the operator is used to
judge the ``CASE-FVM-PERIODIC-HILLS`` anchor, so its invariants are pinned
against (a) synthetic profiles whose zeros are known analytically, (b) an
independent brute-force quadrature, and (c) the legacy round-9 operator, both to
prove exact agreement where the legacy operator is valid and to reproduce, as a
negative control, the case where the legacy operator returns ``nan`` for a
crossing it had in fact detected.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tensorfvm.verification.periodic_hill_round9 import zero_crossings_periodic
from tensorfvm.verification.wall_shear_bubbles import (
    STATUS_ALL_REVERSED,
    STATUS_MEASURED,
    STATUS_NO_RECIRCULATION,
    STATUS_NOT_FINITE,
    STATUS_OUT_OF_WINDOW,
    bubble_intervals,
    period_of,
    separation_reattachment,
    signed_crossings,
    tanh_shear_profile,
)

DOCS = Path(__file__).resolve().parents[1] / "docs" / "case-periodic-hills"
LEGACY_WINDOW = (1.9, 7.1)


def _grid(n=901):
    return np.linspace(0.0, 9.0, n, endpoint=False)


# --------------------------------------------------------------------------- #
# 1. analytic truth: prescribed zeros are recovered
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("zeros", [[0.5, 7.65], [0.22, 5.41], [0.4, 2.05],
                                   [0.48, 3.144]])
def test_recovers_prescribed_zeros_to_dx_squared(zeros):
    x = _grid()
    tau, _ = tanh_shear_profile(x, zeros)
    out = separation_reattachment(tau, x)
    dx = x[1] - x[0]
    # linear sub-cell interpolation of a smooth zero: error is O(dx^2 * curvature)
    assert out["x_s_over_H"] == pytest.approx(zeros[0], abs=2.0 * dx * dx + 1e-9)
    assert out["x_r_over_H"] == pytest.approx(zeros[1], abs=2.0 * dx * dx + 1e-9)
    assert out["status"] == STATUS_MEASURED
    assert out["n_bubbles"] == 1


def test_crossing_directions_encode_separation_and_reattachment():
    x = _grid()
    tau, _ = tanh_shear_profile(x, [0.5, 5.4])
    cross = signed_crossings(tau, x)
    assert [c["direction"] for c in cross] == [-1, 1]      # falling then rising
    assert cross[0]["x_over_H"] < cross[1]["x_over_H"]
    # a falling crossing means tau > 0 above and tau < 0 below
    probe = cross[0]["index"]
    assert tau[probe] >= 0.0 > tau[probe + 1]


# --------------------------------------------------------------------------- #
# 2. the window must not destroy a measured value (the round-9/16 defect)
# --------------------------------------------------------------------------- #
def test_window_is_a_reporting_filter_not_a_detector():
    x = _grid()
    tau, _ = tanh_shear_profile(x, [0.5, 7.65])          # rea outside 7.1
    out = separation_reattachment(tau, x, window=LEGACY_WINDOW)
    assert out["status"] == STATUS_OUT_OF_WINDOW
    assert out["out_of_window"] is True
    assert out["x_r_over_H"] == pytest.approx(7.65, abs=1e-3)   # number kept
    assert out["x_s_over_H"] == pytest.approx(0.5, abs=1e-3)
    assert out["window"] == [1.9, 7.1]


def test_legacy_operator_returns_nan_for_the_same_profile():
    """Negative control: the defect being closed is real, not hypothetical."""
    x = _grid()
    tau, _ = tanh_shear_profile(x, [0.5, 7.65])
    legacy = zero_crossings_periodic(tau, x, *LEGACY_WINDOW)
    assert legacy["crossings"] and max(legacy["crossings"]) > 7.1
    assert math.isnan(legacy["reattachment"])          # detected, then dropped
    assert math.isnan(legacy["separation"])            # falling crossing unseen


def test_reattachment_agrees_with_legacy_wherever_legacy_is_valid():
    """No silent redefinition of ``x_r/H``: equal to 1e-12 inside the window."""
    x = _grid()
    for zeros in ([0.5, 2.05], [0.48, 3.144], [1.0, 4.2]):
        tau, _ = tanh_shear_profile(x, zeros)
        legacy = zero_crossings_periodic(tau, x, *LEGACY_WINDOW)
        new = separation_reattachment(tau, x, window=LEGACY_WINDOW)
        assert new["status"] == STATUS_MEASURED
        assert new["x_r_over_H"] == pytest.approx(legacy["reattachment"], abs=1e-12)


def test_legacy_separation_is_structurally_unmeasurable():
    """The legacy ``separation`` is a *rising* crossing below 1.9, not the
    separation: with one bubble per period it is always ``nan``."""
    x = _grid()
    for zeros in ([0.5, 2.05], [0.48, 3.144], [0.5, 7.65]):
        tau, _ = tanh_shear_profile(x, zeros)
        legacy = zero_crossings_periodic(tau, x, *LEGACY_WINDOW)
        new = separation_reattachment(tau, x, window=LEGACY_WINDOW)
        assert math.isnan(legacy["separation"])
        assert new["x_s_over_H"] == pytest.approx(zeros[0], abs=1e-3)


# --------------------------------------------------------------------------- #
# 3. periodicity: the seam must not clip a bubble
# --------------------------------------------------------------------------- #
def test_bubble_across_the_periodic_seam_has_positive_length():
    x = _grid()
    tau, _ = tanh_shear_profile(x, [0.3, 8.2], sign=-1.0)   # reversed 8.2 -> 9.3
    out = separation_reattachment(tau, x)
    bubble = out["bubbles"][0]
    assert bubble["separation"] == pytest.approx(8.2, abs=1e-3)
    assert bubble["reattachment"] == pytest.approx(9.3, abs=1e-3)
    assert bubble["length"] == pytest.approx(1.1, abs=2e-3)
    assert bubble["spans_seam"] is True
    assert out["x_r_over_H"] > period_of(x) - 0.1          # unwrapped, not clipped


def test_cyclic_roll_of_the_samples_gives_the_same_bubble_length():
    x = _grid()
    tau, _ = tanh_shear_profile(x, [0.3, 8.2], sign=-1.0)
    base = separation_reattachment(tau, x)["bubbles"][0]["length"]
    rolled = np.roll(tau, 37)
    assert separation_reattachment(rolled, x)["bubbles"][0]["length"] == \
        pytest.approx(base, abs=1e-9)


# --------------------------------------------------------------------------- #
# 4. multiple bubbles: primary selection is defined, not accidental
# --------------------------------------------------------------------------- #
def test_secondary_bubble_does_not_become_the_reattachment():
    x = _grid()
    # main bubble 1.2 -> 2.6, small corner bubble 6.0 -> 6.4
    tau, _ = tanh_shear_profile(x, [1.2, 2.6, 6.0, 6.4])
    out = separation_reattachment(tau, x)
    assert out["n_bubbles"] == 2
    areas = [b["reversed_area"] for b in out["bubbles"]]
    assert out["primary_index"] == int(np.argmax(areas))
    assert out["x_s_over_H"] < out["x_r_over_H"] < 6.0
    assert any(b["separation"] == pytest.approx(6.0, abs=1e-3)
               for b in out["bubbles"])

    first = separation_reattachment(tau, x, primary="first")
    assert first["x_s_over_H"] == pytest.approx(1.2, abs=1e-3)
    with pytest.raises(ValueError):
        separation_reattachment(tau, x, primary="longest")


def test_reversed_area_matches_independent_bruteforce_quadrature():
    x = _grid()
    tau, _ = tanh_shear_profile(x, [0.5, 7.65], sharpness=3.0)
    out = separation_reattachment(tau, x)
    bubble = out["bubbles"][0]
    fine = np.linspace(bubble["separation"], bubble["reattachment"], 400001)
    ref = np.trapezoid(np.clip(-np.interp(fine, x, tau), 0.0, None), fine) \
        if hasattr(np, "trapezoid") else np.trapz(
            np.clip(-np.interp(fine, x, tau), 0.0, None), fine)
    assert bubble["reversed_area"] == pytest.approx(ref, rel=2e-3)


# --------------------------------------------------------------------------- #
# 5. degenerate inputs are reported, not silently converted to nan
# --------------------------------------------------------------------------- #
def test_all_reversed_reports_status_not_nan_only():
    x = _grid(101)
    tau = -np.ones_like(x)
    out = separation_reattachment(tau, x)
    assert out["status"] == STATUS_ALL_REVERSED
    assert out["n_bubbles"] == 0
    assert math.isnan(out["x_r_over_H"])       # nan *with* a reason code


def test_no_recirculation_reports_status():
    x = _grid(101)
    out = separation_reattachment(np.ones_like(x), x)
    assert out["status"] == STATUS_NO_RECIRCULATION


def test_non_finite_shear_reports_status():
    x = _grid()
    tau, _ = tanh_shear_profile(x, [0.5, 5.4])
    tau[17] = np.nan
    out = separation_reattachment(tau, x)
    assert out["status"] == STATUS_NOT_FINITE
    assert out["finite"] is False


def test_exact_zero_plateau_yields_one_crossing_per_sign_change():
    x = np.linspace(0.0, 9.0, 91, endpoint=False)
    tau = np.ones_like(x)
    tau[5:9] = 0.0            # exact-zero plateau on the attached side
    tau[9:30] = -1.0
    cross = signed_crossings(tau, x)
    assert len(cross) == 2
    assert [c["direction"] for c in cross] == [-1, 1]
    out = separation_reattachment(tau, x)
    assert out["n_bubbles"] == 1


def test_input_validation():
    with pytest.raises(ValueError):
        signed_crossings(np.zeros(5), np.zeros(4))
    with pytest.raises(ValueError):
        signed_crossings(np.zeros(2), np.zeros(2))
    with pytest.raises(ValueError):
        period_of(np.array([0.0, 1.0, 2.0, 2.5]))       # non-uniform
    with pytest.raises(ValueError):
        period_of(np.array([0.0, 1.0, 0.5]))            # not increasing
    with pytest.raises(ValueError):
        tanh_shear_profile(np.linspace(0, 1, 5), [])


# --------------------------------------------------------------------------- #
# 6. integration with the stored round-9/11/12 artefacts
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,expected_x_r", [
    ("round9_96x48", 7.651),
    ("round11_sst_96x48", 2.046),
    ("round11_sst_r12-96x48", 3.144),
])
def test_stored_artefacts_recover_the_suppressed_reattachment(name, expected_x_r):
    path = DOCS / f"{name}.npz"
    if not path.exists():
        pytest.skip(f"stored artefact {path} not present")
    data = np.load(path)
    tau = data["mean_tau_b"]
    x = data["x_over_h"]
    legacy = zero_crossings_periodic(tau, x, *LEGACY_WINDOW)
    new = separation_reattachment(tau, x, window=LEGACY_WINDOW)
    assert new["x_r_over_H"] == pytest.approx(expected_x_r, abs=5e-3)
    if expected_x_r > LEGACY_WINDOW[1]:
        assert math.isnan(legacy["reattachment"])          # was reported UNDEFINED
        assert new["status"] == STATUS_OUT_OF_WINDOW
    else:
        assert new["x_r_over_H"] == pytest.approx(
            legacy["reattachment"], abs=1e-12)


def test_round16_patankar_crossing_list_carries_the_answer():
    """Round 16 stored the crossing but no profile -- the number was recoverable."""
    path = DOCS / "r16_patankar_96x48.json"
    if not path.exists():
        pytest.skip("round-16 artefact not present")
    payload = json.loads(path.read_text())
    crossings = payload["time_averaged"]["crossings"]
    assert payload["time_averaged"]["x_r_over_H"] is None      # recorded UNDEFINED
    assert crossings and max(crossings) == pytest.approx(7.6816, abs=1e-3)