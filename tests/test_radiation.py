"""Tests for the radiation view-factor / gray-enclosure module and its verification driver."""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensorfvm import radiation as R
from tensorfvm.verification import radiation as VR


def _cavity(points):
    p = np.asarray(points, float)
    return [R.Strip(tuple(p[i]), tuple(p[(i + 1) % len(p)])) for i in range(len(p))]


def test_degenerate_strip_rejected():
    with pytest.raises(ValueError):
        R.Strip((0.0, 0.0), (0.0, 0.0))


def test_parallel_strips_match_closed_form():
    """F = sqrt(1+(d/w)^2) - d/w for two equal parallel strips (Hottel)."""
    for d in (0.25, 1.0, 4.0):
        si = R.Strip((0.0, 0.0), (1.0, 0.0))
        sj = R.Strip((1.0, d), (0.0, d))       # gap stays on the left of both strips
        ref = math.sqrt(1.0 + d * d) - d
        assert R.strip_view_factor(si, sj) == pytest.approx(ref, rel=1e-13)


def test_perpendicular_strips_shared_edge_match_closed_form():
    w1, w2 = 1.0, 2.0
    si = R.Strip((0.0, 0.0), (w1, 0.0))
    sj = R.Strip((0.0, w2), (0.0, 0.0))
    ref = (w1 + w2 - math.hypot(w1, w2)) / (2.0 * w1)
    assert R.strip_view_factor(si, sj) == pytest.approx(ref, rel=1e-13)


def test_square_cavity_exact_and_rows_sum_to_one():
    sq = R.strip_form_factors(_cavity([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]))
    assert sq["F"][0, 2] == pytest.approx(math.sqrt(2.0) - 1.0, rel=1e-13)
    assert sq["F"][0, 1] == pytest.approx(1.0 - math.sqrt(2.0) / 2.0, rel=1e-13)
    assert np.allclose(sq["F"].sum(axis=1), 1.0, atol=1e-14)
    assert np.allclose(np.diag(sq["F"]), 0.0)
    # reciprocity is structural: the exchange areas are exactly symmetric
    assert np.array_equal(sq["exchange"], sq["exchange"].T)


def test_quadrature_reference_agrees_with_crossed_strings():
    pent = _cavity([(math.cos(t), math.sin(t)) for t in np.linspace(0, 2 * math.pi, 6)[:-1]])
    m = R.strip_form_factors(pent)
    for i in range(5):
        for j in range(5):
            if i == j:
                continue
            ref = R.strip_view_factor_quadrature(pent[i], pent[j], order=200)
            assert m["F"][i, j] == pytest.approx(ref, rel=1e-3)


def test_masked_pairs_are_removed_and_asymmetric_mask_rejected():
    strips = _cavity([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    mask = np.ones((4, 4), dtype=bool)
    np.fill_diagonal(mask, False)
    mask[0, 2] = mask[2, 0] = False
    m = R.strip_form_factors(strips, visible=mask)
    assert m["F"][0, 2] == 0.0 and m["F"][2, 0] == 0.0
    bad = mask.copy()
    bad[0, 3] = True
    bad[3, 0] = False
    with pytest.raises(ValueError):
        R.strip_form_factors(strips, visible=bad)


def test_circle_pair_enclosure_is_exact():
    c = R.circle_pair_form_factors(0.5, 1.0)
    assert c["F"][0, 1] == pytest.approx(1.0)
    assert c["F"][1, 0] == pytest.approx(0.5)
    assert c["F"][1, 1] == pytest.approx(0.5)      # the concave outer wall sees itself
    assert np.allclose(c["F"].sum(axis=1), 1.0)


def test_gray_two_surface_enclosure_matches_closed_form():
    c = R.circle_pair_form_factors(0.5, 1.0)
    enc = R.GrayEnclosure(c["F"], c["lengths"], [0.3, 0.8], [400.0, 300.0])
    power = enc.solve()["power"][0]
    ref = R.two_surface_exchange(float(c["lengths"][0]), float(c["lengths"][1]),
                                 0.3, 0.8, 400.0, 300.0, 1.0)
    assert power == pytest.approx(ref, rel=1e-12)
    assert enc.energy_residual() < 1e-12


def test_reradiating_plate_temperature_and_flux_balance():
    f = np.array([[0.0, 1.0, 0.0], [0.5, 0.0, 0.5], [0.0, 1.0, 0.0]])
    enc = R.GrayEnclosure(f, [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [400.0, np.nan, 300.0],
                          reradiating=[False, True, False])
    res = enc.solve()
    assert res["radiosity"][1] / R.SIGMA == pytest.approx(0.5 * (400.0 ** 4 + 300.0 ** 4), rel=1e-13)
    assert res["power"][0] == pytest.approx(-res["power"][2], rel=1e-13)
    assert enc.energy_residual() < 1e-13


def test_isothermal_enclosure_carries_no_net_flux():
    f = np.array([[0.0, 1.0], [0.5, 0.5]])
    enc = R.GrayEnclosure(f, [1.0, 2.0], [0.9, 0.2], [350.0, 350.0])
    assert np.max(np.abs(enc.solve()["flux"])) < 1e-12 * R.SIGMA * 350.0 ** 4


def test_invalid_inputs_rejected():
    c = R.circle_pair_form_factors(0.5, 1.0)
    with pytest.raises(ValueError):     # emissivity out of range
        R.GrayEnclosure(c["F"], c["lengths"], [0.0, 0.8], [400.0, 300.0])
    with pytest.raises(ValueError):     # rows must sum to one
        R.GrayEnclosure(np.array([[0.0, 0.5], [0.5, 0.5]]), [1.0, 1.0], [0.9, 0.9], [300.0, 300.0])
    with pytest.raises(ValueError):     # finite temperature needed when not reradiating
        R.GrayEnclosure(c["F"], c["lengths"], [0.9, 0.9], [np.nan, 300.0])
    with pytest.raises(ValueError):
        R.circle_pair_form_factors(1.0, 1.0)


def test_sky_depression_is_physical_and_longwave_rate_matches_analytic():
    """The icing balance's long-wave term: clear-sky ratio < 1 and dq/dT = 4 eps sigma T^3."""
    from tensorfvm.cht.ice_conjugate import surface_heat_loss

    for t_air in (-30.0, -10.0, 0.0, 20.0):
        ratio = R.sky_depression(t_air)
        assert 0.4 < ratio < 1.0

    ts, h, eps = -5.0, 1e-3, 0.97
    q = lambda t: surface_heat_loss(t_surface=t, t_air=-10.0, wind=0.0,
                                    emissivity=eps)["q_lw_W_m2"]
    fd = (q(ts + h) - q(ts - h)) / (2.0 * h)
    assert fd == pytest.approx(4.0 * eps * R.SIGMA * (ts + 273.15) ** 3, rel=1e-6)


def test_verification_driver_all_pass():
    """The driver's pre-declared criteria must all pass (model-form C11 excluded)."""
    summary = VR.run()
    graded = {k: v for k, v in summary["criteria"].items() if not k.startswith("C11")}
    failed = [k for k, v in graded.items() if not v["pass"]]
    assert not failed, failed
    assert summary["all_pass"]