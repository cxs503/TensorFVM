"""Tests for occlusion-aware 2-D view factors (shadowing between strips).

The verification driver (``python -m tensorfvm.verification.radiation_occlusion``)
runs the full criterion set with a Monte-Carlo reference; these tests hold the
fast invariants that must never regress: exactness on the unoccluded control,
exact zero for a fully hidden pair, energy closure and reciprocity on obstructed
scenes, the back-to-back rule, and the new non-negativity guard in
``GrayEnclosure``.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensorfvm import radiation as R
from tensorfvm.verification.radiation_occlusion import (
    _beam_scene, _block_scene, _blocked_pair_scene, _cavity_scene,
    monte_carlo_form_factors)


def test_unoccluded_pair_matches_crossed_strings():
    """With no blockers the two different methods must agree."""
    si = R.Strip((0.0, 0.0), (1.0, 0.0))
    sj = R.Strip((1.0, 1.0), (0.0, 1.0))
    ref = R.strip_view_factor(si, sj)
    assert R.strip_view_factor_occluded(si, sj, (), order=64) == pytest.approx(ref, rel=1e-6)


def test_unoccluded_cavity_matches_crossed_strings():
    cavity = _cavity_scene()
    occ = R.strip_form_factors_occluded(cavity, order=64)
    ref = R.strip_form_factors(cavity)
    assert np.max(np.abs(occ["F"] - ref["F"])) < 5e-3     # corner-singularity limited
    assert occ["closure_error"] < 1e-3


def test_fully_blocked_pair_is_exactly_zero():
    """A segment spanning the gap hides the pair completely."""
    strips, i, j = _blocked_pair_scene()
    blockers = [s for k, s in enumerate(strips) if k not in (i, j)]
    f = R.strip_view_factor_occluded(strips[i], strips[j], blockers, order=64)
    assert f == pytest.approx(0.0, abs=1e-12)
    # the obstructed-blind path has no way to see that:
    assert abs(R.strip_view_factor(strips[i], strips[j])) > 0.1


def test_pair_visibility_classifies_blocked_and_partial_pairs():
    strips, i, j = _blocked_pair_scene()
    assert not R.pair_visibility(strips[i], strips[j], [strips[2]])
    beam = _beam_scene()
    # the deck is partially visible from the beam's underside (partial shadow)
    assert R.pair_visibility(beam[4], beam[0], [s for k, s in enumerate(beam)
                                                if k not in (0, 4)])


def test_obstructed_enclosure_closes_and_is_reciprocal():
    for scene in (_beam_scene(), _block_scene()):
        occ = R.strip_form_factors_occluded(scene, order=48)
        assert occ["closure_error"] < 1e-3
        assert occ["reciprocity_error"] < 1e-3
        assert np.allclose(occ["F"].sum(axis=1), 1.0, atol=1e-3)
        assert np.all(np.diag(occ["F"]) == 0.0)
        assert np.all(np.asarray(occ["F"]) >= 0.0)
        assert np.array_equal(occ["exchange"], occ["exchange"].T)


def test_back_to_back_obstacle_faces_do_not_exchange():
    """Faces of a convex obstacle: the chord crosses the solid, so F = 0."""
    block = _block_scene()[4:]
    occ = R.strip_form_factors_occluded(block, order=48)
    assert np.max(np.abs(occ["F"])) == pytest.approx(0.0, abs=1e-12)
    # cavity-style crossed strings invent the exchange between such faces: they
    # read the obstacle as a *cavity*, so opposite faces get sqrt(2) - 1 and
    # adjacent faces get 1 - sqrt(2)/2 although neither pair can radiate at all.
    naive = R.strip_form_factors(block)
    assert naive["F"][0, 2] == pytest.approx(math.sqrt(2.0) - 1.0, rel=1e-12)
    assert naive["F"][0, 1] == pytest.approx(1.0 - math.sqrt(2.0) / 2.0, rel=1e-12)


def test_block_scene_shadowed_pairs_are_zero_and_open_pairs_are_not():
    block = _block_scene()
    f = R.strip_form_factors_occluded(block, order=48)["F"]
    # the floor (0) is shadowed from the obstacle's *upper* face (4) by the
    # obstacle itself, while the obstacle's underside (6) is almost fully in view
    assert f[0, 4] == pytest.approx(0.0, abs=1e-12)
    assert f[0, 6] > 0.1
    # the floor still sees the side walls; the block does not span the hold
    assert f[0, 1] > 0.1 and f[0, 3] > 0.1
    # the obstructed-blind path reports the shadowed pair as an open exchange
    assert abs(R.strip_form_factors(block)["F"][0, 4]) > 1e-3


def test_monte_carlo_reference_agrees_within_its_band():
    """Independent ray tracing, cavity control case (closed, exact values known)."""
    mc = monte_carlo_form_factors(_cavity_scene(), rays=120_000)
    f = mc["F"]
    assert np.max(mc["lost_fraction"]) < 1e-12            # closed scene: nothing escapes
    assert f[0, 2] == pytest.approx(math.sqrt(2.0) - 1.0, abs=5e-3)
    assert f[0, 1] == pytest.approx(1.0 - math.sqrt(2.0) / 2.0, abs=5e-3)
    assert np.allclose(f.sum(axis=1), 1.0, atol=5e-3)


def test_monte_carlo_sees_a_shadow_on_the_beam_scene():
    beam = _beam_scene()
    mc = monte_carlo_form_factors(beam, rays=120_000)
    occ = R.strip_form_factors_occluded(beam, order=48)["F"]
    naive = R.strip_form_factors(beam)["F"]
    band = 4.0 * float(np.max(mc["sigma"]))
    assert np.max(np.abs(occ - mc["F"])) < band
    # the obstructed-blind answer is out of the band by more than an order of magnitude
    assert np.max(np.abs(naive - mc["F"])) > 10.0 * band


def test_gray_enclosure_rejects_negative_view_factors():
    """The row-sum validation alone does not catch a shaded re-entrant scene."""
    beam = _beam_scene()
    f = R.strip_form_factors(beam)["F"]
    assert np.min(f) < -0.1                     # rows still sum to one, but F < 0
    with pytest.raises(ValueError, match="non-negative"):
        R.GrayEnclosure(f, [s.length for s in beam], np.full(8, 0.9), np.full(8, 300.0),
                        row_sum_tol=1e-6)


def test_gray_enclosure_row_sum_tolerance_is_explicit():
    f = np.array([[0.0, 1.0], [0.5, 0.5]])
    with pytest.raises(ValueError, match="rows must sum to one"):
        R.GrayEnclosure(f + 1e-4, [1.0, 1.0], [0.9, 0.9], [300.0, 300.0])
    # a numerically computed matrix may declare its closure accuracy instead
    R.GrayEnclosure(f + 1e-4, [1.0, 1.0], [0.9, 0.9], [300.0, 300.0], row_sum_tol=1e-3)


def test_occluded_enclosure_conserves_energy_and_is_isothermal_clean():
    beam = _beam_scene()
    occ = R.strip_form_factors_occluded(beam, order=64)
    areas = [s.length for s in beam]
    enc = R.GrayEnclosure(occ["F"], areas, np.full(8, 0.9), np.full(8, 300.0),
                          row_sum_tol=1e-3)
    assert np.max(np.abs(enc.solve()["flux"])) < 1e-4 * R.SIGMA * 300.0 ** 4
    temps = np.full(8, 350.0)
    temps[3:6] = 260.0
    enc2 = R.GrayEnclosure(occ["F"], areas, np.full(8, 0.9), temps, row_sum_tol=1e-3)
    assert enc2.energy_residual() < 1e-4
    power = enc2.solve()["power"][3:6]
    assert np.all(power < 0.0)                  # the cold beam loses heat


def test_occluded_quadrature_converges_with_order():
    beam = _beam_scene()
    ref = R.strip_form_factors_occluded(beam, order=192)["F"]
    errs = [np.max(np.abs(R.strip_form_factors_occluded(beam, order=o)["F"] - ref))
            for o in (24, 48, 96)]
    assert errs[0] > errs[1] > errs[2]
    assert errs[-1] < 3e-3