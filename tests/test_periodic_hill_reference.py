"""Contract tests for the primary ERCOFTAC UFR 3-30 reference (round 10).

These tests pin the *reference* side of CASE-FVM-PERIODIC-HILLS: the data files
distributed by the reference's author, the reattachment length quoted by the
case's authors on the ERCOFTAC wiki, and the measured size of the error in the
hand-digitised table that rounds 4--9 graded against.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorfvm.verification.periodic_hill_reference import (
    COMPETING_X_R, LEGACY_DIGITISED_Y05, REFERENCE, STATIONS, X_R_OVER_H,
    X_S_OVER_H, legacy_table_discrepancy, load_profile, profile_at,
    reattachment_band, relative_band_ok, station_covers_y05, y05_table)

#: y/H = 0.5 values read off the distributed DNS profiles (regression pins)
DNS_Y05 = {2.0: -0.0912, 3.0: 0.0041, 4.0: 0.1198, 5.0: 0.2183,
           6.0: 0.2876, 7.0: 0.3141, 8.0: 0.2238}


def test_all_ten_stations_load_with_seven_columns():
    assert len(STATIONS) == 10
    for station in STATIONS:
        prof = load_profile(station)
        assert prof["x_over_h"] == pytest.approx(station)
        for key in ("y_over_h", "u_over_ub", "v_over_ub", "uu_over_ub2",
                    "vv_over_ub2", "uv_over_ub2", "k_over_ub2"):
            values = prof[key]
            assert values.shape == prof["y_over_h"].shape
            assert np.all(np.isfinite(values))
        assert np.all(np.diff(prof["y_over_h"]) > 0.0)
        # every profile runs from the hill surface up to the top wall
        assert prof["y_over_h"][-1] == pytest.approx(3.033, abs=1.0e-2)
        surface, _, _ = __import__("tensorfvm.periodic_hill", fromlist=["hill_profile"]).hill_profile(
            np.array([station]), height=1.0, length=9.0)
        assert prof["y_over_h"][0] == pytest.approx(float(surface[0]), abs=5.0e-3)


def test_station_assignment_is_confirmed_by_the_geometry():
    """Station 0.05 starts at the crest (y/H ~ 1), i.e. the file order is the
    canonical x/H = 0.05, 0.5, 1, 2, ..., 8 sequence -- independent evidence."""
    crest = load_profile(0.05)
    assert crest["y_over_h"][0] == pytest.approx(1.0004, abs=1.0e-3)
    assert not station_covers_y05(0.05)
    assert station_covers_y05(2.0)


def test_y05_regression_against_distributed_dns():
    for station, expected in DNS_Y05.items():
        assert profile_at(station, 0.5) == pytest.approx(expected, abs=2.0e-3)


def test_y05_table_is_physical():
    table = y05_table()
    # recirculation at y/H = 0.5 only near the hill, strongest shortly behind it
    assert table[2.0] < 0.0
    assert table[2.0] > -0.3
    # monotone recovery through the stations behind the bubble
    recovery = [table[x] for x in (3.0, 4.0, 5.0, 6.0, 7.0)]
    assert recovery == sorted(recovery)
    # ... and a deceleration on the windward side of the next hill
    assert table[8.0] < table[7.0]


def test_reattachment_reference_and_bands():
    assert X_R_OVER_H == pytest.approx(5.41)
    low, high = reattachment_band()
    assert (low, high) == pytest.approx((4.869, 5.951))
    # only the spurious 4.36 is outside; 5.34 is a 1.3 % near miss inside it
    assert low <= COMPETING_X_R["ledger_rounds3_5"] <= high
    assert not (low <= COMPETING_X_R["test_breuer2009"] <= high)
    assert X_S_OVER_H is None  # not tabulated: reported WITHHELD, not 0.227


def test_legacy_digitised_table_is_measured_to_be_wrong():
    report = legacy_table_discrepancy()
    assert set(report) == set(LEGACY_DIGITISED_Y05)
    worst = max(error for _, _, error in report.values())
    # the hand-digitised table misses every recovery station by > 100 %
    assert worst > 1.0
    for station in (5.0, 6.0, 7.0):
        legacy, dns, error = report[station]
        assert legacy > 2.0 * dns        # systematically too large, not noisy
        assert error > 1.0


def test_error_convention_is_relative_only_where_ref_is_not_small():
    assert relative_band_ok(0.29, 6.0)["band"] == "relative"
    assert relative_band_ok(-0.09, 2.0)["band"] == "absolute"
    # reference is ~0.0: an absolute miss of 0.30 must fail a 10 % band
    assert relative_band_ok(0.30, 3.0)["pass"] is False
    assert relative_band_ok(float("nan"), 4.0)["pass"] is False


def test_unknown_station_is_rejected():
    with pytest.raises(KeyError):
        load_profile(1.25)


def test_reference_record_is_single_sourced():
    assert REFERENCE["x_r_over_H"] == pytest.approx(5.41)
    assert REFERENCE["x_s_over_H"] is None
    assert "kbwiki.ercoftac.org" in REFERENCE["source"] or "ERCOFTAC" in REFERENCE["source"]
