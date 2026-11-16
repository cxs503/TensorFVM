"""Round 32: primary Re = 700 reference data for CASE-FVM-PERIODIC-HILLS.

Pins the newly fetched ERCOFTAC KB case-1 DNS profile files (LESOCC,
Re = 700) so a future change cannot silently move the reference:

* the ten files load, are monotone in ``y``, and carry the seven canonical
  columns;
* the ``y/H = 0.5`` acceptance line is in the fluid at ``x/H = 1 .. 8``
  (measured from the files; note this differs from the Re = 2800 set, where
  stations 1 and 8 are inside the hill body -- the Re = 700 DNS profiles
  start lower at those stations);
* the table values equal a direct read of the raw files (no drift);
* the reattachment reference is single-sourced from
  :data:`tensorfvm.verification.periodic_hill_lower_re.KB_REATTACHMENT_BY_RE`.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorfvm.verification.periodic_hill_re700_reference import (
    DATA_DIR,
    STATIONS,
    X_R_OVER_H,
    X_R_TOLERANCE,
    load_profile,
    profile_at,
    relative_band_ok,
    station_covers_y05,
    y05_table,
)


def test_all_ten_files_load_and_are_monotone():
    for i, x in enumerate(STATIONS, start=1):
        prof = load_profile(x)
        assert prof["source"] == f"UFR3-30_C_700_data_MB-{i:03d}.dat"
        y = prof["y_over_h"]
        assert np.all(np.diff(y) > 0.0)
        assert y[0] >= 0.0 and y[-1] > 3.0  # wall to channel top
        for key in ("u_over_ub", "v_over_ub", "uu_over_ub2", "vv_over_ub2",
                    "uv_over_ub2", "k_over_ub2"):
            assert np.all(np.isfinite(prof[key])), key


def test_y05_admission_matches_geometry():
    covers = {x: station_covers_y05(x) for x in STATIONS}
    assert covers == {0.05: False, 0.5: False, 1.0: True, 2.0: True,
                     3.0: True, 4.0: True, 5.0: True, 6.0: True,
                     7.0: True, 8.0: True}


def test_y05_table_values_match_raw_files():
    """The table must equal a direct read of the raw .dat files (no drift)."""
    table = y05_table()
    assert set(table) == {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}
    expected = {1.0: -0.00344, 2.0: -0.11383, 3.0: -0.01515, 4.0: 0.12983,
                5.0: 0.26104, 6.0: 0.34681, 7.0: 0.37573, 8.0: 0.19790}
    for x, v in expected.items():
        assert table[x] == pytest.approx(v, abs=5e-5), x
        # cross-check against an independent raw-file interpolation
        raw = np.loadtxt(DATA_DIR / f"UFR3-30_C_700_data_MB-{STATIONS.index(x) + 1:03d}.dat",
                          comments="#")
        order = np.argsort(raw[:, 0])
        y, u = raw[order, 0], raw[order, 1]
        assert table[x] == pytest.approx(float(np.interp(0.5, y, u)), abs=1e-6)


def test_reattachment_reference_is_single_sourced():
    from tensorfvm.verification.periodic_hill_lower_re import (
        KB_REATTACHMENT_BY_RE,
    )
    assert X_R_OVER_H == pytest.approx(5.24)
    assert X_R_OVER_H == KB_REATTACHMENT_BY_RE[700.0]
    assert X_R_TOLERANCE == 0.10


def test_relative_band_convention():
    # |ref| >= 0.1 -> relative; below -> absolute (round-6 convention)
    chk = relative_band_ok(0.20, 5.0)  # ref = 0.26104
    assert chk["band"] == "relative"
    assert chk["error"] == pytest.approx(abs(0.20 - 0.26104) / 0.26104, rel=1e-3)
    assert not chk["pass"]
    chk2 = relative_band_ok(-0.01, 3.0)  # ref = -0.01515 -> absolute band
    assert chk2["band"] == "absolute"
    assert chk2["error"] == pytest.approx(abs(-0.01 - (-0.01515)), abs=5e-4)


def test_profile_at_rejects_out_of_range():
    with pytest.raises(ValueError):
        profile_at(2.0, 3.5)  # above the channel top
    with pytest.raises(KeyError):
        profile_at(2.5)  # not a distributed station
