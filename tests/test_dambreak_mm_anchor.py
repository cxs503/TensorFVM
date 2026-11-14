"""The Martin & Moyce anchor convention and the ``CASE-FVM-DAMBREAK-WET`` grading.

Two things are pinned here:

* the *axes* of Martin & Moyce (1952) Fig. 3 -- ``T = t sqrt(2 g / a)``,
  ``X = x_f / a`` with ``X(0) = 1`` -- and the three checkpoints that the
  sibling TensorLBM benchmark froze independently (``1.326 / 2.360 / 3.670``);
* the defect of the legacy ``t sqrt(g/a)`` / displacement axis, quantified by
  :func:`tensorfvm.vof_phase0.anchors_mm.legacy_offset_audit` so it cannot be
  silently reintroduced.
"""

from __future__ import annotations

import math

import pytest

from tensorfvm.vof_phase0 import anchors_mm as A
from tensorfvm.vof_phase0.anchors import (
    MARTIN_MOYCE_FRONT_T,
    MARTIN_MOYCE_FRONT_X,
)
from tensorfvm.vof_threecase.dambreak import (
    DamBreakMMConfig,
    mm_checkpoint_error,
)

A_W = 0.05715
G = 9.81


def test_time_axis_is_the_paper_axis():
    """``T = t sqrt(2 g / a)``: sqrt(2) larger than the legacy ``t sqrt(g/a)``."""
    t, a = 0.1, A_W
    T = A.mm_time(t, a, G)
    assert T == pytest.approx(t * math.sqrt(2.0 * G / a), rel=1e-15)
    legacy = t * math.sqrt(G / a)
    assert T == pytest.approx(math.sqrt(2.0) * legacy, rel=1e-15)
    assert A.physical_time(T, a, G) == pytest.approx(t, rel=1e-15)


def test_front_ordinate_starts_at_one():
    assert A.mm_x(0.0, A_W) == 0.0
    # X(0) = 1 means the front starts at x_f = a, i.e. mm_x(a) == 1
    assert A.mm_x(A_W, A_W) == pytest.approx(1.0, rel=1e-15)
    assert A.mm_front(0.0) == 1.0


def test_frozen_checkpoints_match_the_sibling_repository():
    """TensorLBM froze X = 1.326 / 2.360 / 3.670 at T = 1 / 2 / 2.96."""
    assert A.mm_front(1.0) == pytest.approx(1.326, abs=1e-12)
    assert A.mm_front(2.0) == pytest.approx(2.36, abs=1e-12)
    assert A.mm_front(2.96) == pytest.approx(3.67, abs=1e-12)
    # tabulated abscissae are reproduced exactly, not interpolated
    for T, X in zip(A.MM_T, A.MM_X):
        assert A.mm_front(float(T)) == pytest.approx(float(X), abs=1e-12)
    assert A.REFERENCE_CHECKPOINTS == {
        1.0: 1.326, 1.5: 1.747, 2.0: 2.36, 2.5: 3.00, 2.96: 3.67}


def test_curve_is_monotone_and_support_is_closed():
    xs = [A.mm_front(t) for t in A.MM_T]
    assert all(b > a for a, b in zip(xs, xs[1:])), "front must advance"
    with pytest.raises(ValueError, match="outside the digitised"):
        A.mm_front(A.MM_T[-1] + 0.01)
    with pytest.raises(ValueError, match="outside the digitised"):
        A.mm_front(-0.01)


def test_legacy_axis_is_a_defect_and_not_a_relabelling():
    """Reinterpreting the legacy table as ``(T, X)`` recovers the same curve.

    Read the way the code consumed it (``t sqrt(g/a)`` + displacement) the same
    numbers overstate the experimental front by up to ~47 %, and the implied
    first-segment front acceleration is ~6 g, above the ``2 g = g h0 / a``
    bound a gravity-driven collapse can produce at the toe.
    """
    audit = A.legacy_offset_audit(A_W, G)
    assert audit["max_abs_as_coded_error"] > 0.40
    assert audit["max_abs_relabel_error"] < 0.35
    assert audit["legacy_first_segment_acceleration_g"] > 2.0 * audit[
        "physical_acceleration_bound_g"]
    # the legacy table really is the curve, point by point: within 12 % over
    # 1 <= T <= 2.5 (only its very first segment, T = 0.5, is 34 % off)
    for row in audit["rows"]:
        if 1.0 <= row["t_old"] <= 2.5:
            assert abs(row["relabel_error"]) < 0.12
        if 0.0 < row["t_old"] < 1.0:
            assert abs(row["relabel_error"]) < 0.35
    # ... and the legacy arrays are still importable/unchanged for reproduction
    assert MARTIN_MOYCE_FRONT_T[0] == 0.0 and MARTIN_MOYCE_FRONT_X[0] == 0.0


def test_checkpoint_table_is_in_the_digitised_support():
    rows = A.mm_checkpoint_table(A_W, G)
    assert [r["t_star"] for r in rows] == list(A.MM_CHECKPOINTS_T)
    for r in rows:
        assert A.mm_time(r["t"], A_W, G) == pytest.approx(r["t_star"], rel=1e-12)
        assert r["t_star"] <= A.MM_T[-1] + 1e-12
    times = [r["t"] for r in rows]
    assert all(b > a for a, b in zip(times, times[1:]))


def test_checkpoint_error_sign_and_scale():
    cfg = DamBreakMMConfig()
    t = A.physical_time(2.0, A_W, G)
    x_ref = A.mm_front(2.0) * A_W
    rec = mm_checkpoint_error(t, x_ref, cfg)
    assert rec["T"] == pytest.approx(2.0, abs=1e-12)
    assert rec["relative_error"] == pytest.approx(0.0, abs=1e-12)
    slow = mm_checkpoint_error(t, 0.9 * x_ref, cfg)
    assert slow["relative_error"] == pytest.approx(-0.1, abs=1e-12)
    # outside the digitised support the sample is not graded (nan, never 0)
    far = mm_checkpoint_error(A.physical_time(4.0, A_W, G), x_ref, cfg)
    assert math.isnan(far["relative_error"])