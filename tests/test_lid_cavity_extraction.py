"""Self-verification of the lid-cavity centreline extraction path.

The Re = 400 strict xfail in ``test_lid_cavity_re1000_evidence.py`` names an
"wall-adjacent extraction defect" as the suspect for a discrepancy that grows
under refinement.  These tests give that claim a measurable form: the
extraction path is run on analytic fields of known closed form, so the measured
error is produced by the extraction alone.
"""
import math

import numpy as np
import torch

from tensorfvm.lid_cavity import LidCavityConfig, LidCavitySolver, hermite_interpolate
from tensorfvm.lid_cavity_reference import table
from tensorfvm.verification.lid_cavity_extraction import (
    GHIA_V_ABS_MAX,
    GHIA_V_X,
    OBSERVED_RELATIVE_DEFECT,
    attribution,
    extraction_geometry,
    index_shift_control,
    interpolation_study,
    observed_orders,
    profile,
    synthetic_solver,
)

NS = (32, 64, 128)


def test_mid_planes_are_represented_exactly_on_every_grid():
    """The staggered storage must put the mid-planes on stored faces, not half a cell off."""
    for n in NS:
        geometry = extraction_geometry(n)
        assert geometry["v_mid_plane_is_exact"], (n, geometry["v_mid_plane_y"])
        assert geometry["u_mid_plane_is_exact"], (n, geometry["u_mid_plane_x"])
        assert geometry["v_mid_plane_y"] == 0.5
        assert geometry["point_count"] == n


def test_centreline_ordinates_bracket_the_published_table_and_stay_interior():
    for n in NS:
        solver = LidCavitySolver(LidCavityConfig(n=n, reynolds=400.0))
        x, v = solver.centreline_v()
        x = np.asarray(x)
        assert x.min() > 0.0 and x.max() < 1.0
        # every Ghia ordinate must fall inside the sampled range, so no
        # interpolation target is reached by extrapolation
        assert x.min() <= min(GHIA_V_X) and x.max() >= max(GHIA_V_X), (n, x.max())


def test_extraction_reproduces_a_smooth_profile_to_roundoff():
    """Node values are exact: the profile is sampled where it is read back."""
    for n in NS:
        solver = synthetic_solver(n)
        x, v = solver.centreline_v()
        expected = profile(np.asarray(x, dtype=float))
        np.testing.assert_allclose(np.asarray(v), expected, rtol=0, atol=1e-14)


def test_interpolation_error_is_far_below_the_recorded_discrepancy():
    rows = interpolation_study(NS)
    for row in rows:
        # 1 % of the profile scale would still be 15x below the recorded defect
        assert row["max_relative_error"] < 1e-2, row
    coarsest = rows[0]
    assert (coarsest["max_relative_error"]
            < OBSERVED_RELATIVE_DEFECT / 10.0), coarsest


def test_interpolation_is_at_least_second_order_in_h():
    rows = interpolation_study(NS)
    order = observed_orders(rows)
    assert math.isfinite(order) and order > 2.0, (order, rows)


def test_index_shift_is_detected_so_the_measurement_has_teeth():
    """Positive control: a half-cell sampling error must dwarf the clean error."""
    rows = index_shift_control((32, 64), shifts=(0.5,))
    for row in rows:
        assert row["max_abs_error"] > 50.0 * row["clean_max_abs_error"], row
        assert row["detection_ratio"] > 50.0


def test_attribution_exonerates_the_extraction_path():
    rows = interpolation_study([32])
    verdict = attribution(rows[0]["max_relative_error"])
    assert verdict.ratio > 10.0
    assert "exonerated" in verdict.verdict
    assert verdict.as_dict()["observed_relative_defect"] == OBSERVED_RELATIVE_DEFECT


def test_attribution_flags_an_extraction_budget_that_cannot_be_ignored():
    verdict = attribution(0.05)
    assert verdict.ratio < 10.0
    assert "not exonerated" in verdict.verdict


def test_ghia_peak_used_as_yardstick_is_the_published_one():
    published = max(abs(v) for _, v in table(400.0, "v"))
    assert abs(GHIA_V_ABS_MAX - published) < 1e-15
    assert abs(GHIA_V_ABS_MAX - 0.44993) < 1e-5
    # the analytic stand-in must be the same order of magnitude as the yardstick
    magnitude = float(np.abs(profile(np.linspace(0.0, 1.0, 2001))).max())
    assert 0.2 < magnitude < GHIA_V_ABS_MAX * 1.1, magnitude
    assert all(0.0 < x < 1.0 for x in GHIA_V_X)


def test_wall_faces_are_not_silently_included_in_the_centreline():
    """``v[:, n//2]`` is a y-face slice; the x-wall faces carry no centreline data."""
    for n in NS:
        solver = LidCavitySolver(LidCavityConfig(n=n, reynolds=400.0))
        assert solver.v.shape == (n, n + 1)
        assert solver.u.shape == (n + 1, n)
        x, v = solver.centreline_v()
        assert len(v) == n  # x-cell centres only, no duplicated wall value


def test_hermite_interpolation_is_exact_at_the_sample_points():
    x = np.linspace(0.0, 1.0, 33)[1:-1]
    f = torch.as_tensor(profile(x))
    values = hermite_interpolate(torch.as_tensor(x), f, list(x))
    np.testing.assert_allclose(np.asarray(values, dtype=float), profile(x), atol=1e-12)