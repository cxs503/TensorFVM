"""Tests for :func:`tensorfvm.verification.periodic_hill_field_metrics.bubble_object_report`.

Round 22 of ``CASE-FVM-PERIODIC-HILLS`` found that the closure-free ``Re=2800``
solution stores **two** reversed-shear regions, and that the default
``primary="largest_area"`` rule selects the second one -- so the published
``x_r/H`` was compared against a reference that names a different bubble.  These
tests pin the detector that makes such a cross-object comparison impossible to
publish unnoticed.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorfvm.verification.periodic_hill_field_metrics import (
    BUBBLE_OBJECT_CONSISTENT,
    BUBBLE_OBJECT_MISMATCH,
    BUBBLE_SINGLE,
    bubble_object_report,
)


def _tau(x, negative_intervals):
    """Wall shear that is +1 everywhere except strictly inside the intervals."""
    tau = np.ones_like(x)
    for lo, hi in negative_intervals:
        tau[(x > lo) & (x < hi)] = -1.0
    return tau


X = np.linspace(0.0, 9.0, 181)


def test_single_bubble_has_nothing_to_disambiguate():
    rep = bubble_object_report(_tau(X, [(0.4, 5.4)]), X)
    assert rep["n_bubbles"] == 1
    assert rep["flag"] == BUBBLE_SINGLE
    assert rep["selected_is_crest"] is True
    assert rep["x_r_over_H"]["largest_area"] == pytest.approx(rep["x_r_over_H"]["first"])


def test_large_second_bubble_is_reported_as_object_mismatch():
    # narrow crest bubble 0.4-1.2, wide mid-valley bubble 3.9-8.2 (the shape the
    # closure-free run actually produces): the largest *area* is the second one.
    rep = bubble_object_report(_tau(X, [(0.4, 1.2), (3.9, 8.2)]), X)
    assert rep["n_bubbles"] == 2
    assert rep["flag"] == BUBBLE_OBJECT_MISMATCH
    assert rep["selected"]["index"] == 1
    assert rep["crest"]["index"] == 0
    assert rep["selected_is_crest"] is False
    # the two candidate numbers must differ by far more than the case tolerance
    a = rep["x_r_over_H"]["largest_area"]
    b = rep["x_r_over_H"]["first"]
    assert b < a
    assert abs(a - b) / a > 0.10
    assert "different physical object" in rep["reason"]


def test_second_bubble_smaller_than_crest_is_consistent():
    # wide crest bubble, narrow downstream bubble: both rules pick the crest one
    rep = bubble_object_report(_tau(X, [(0.4, 5.4), (6.5, 7.0)]), X)
    assert rep["n_bubbles"] == 2
    assert rep["selected_is_crest"] is True
    assert rep["flag"] == BUBBLE_OBJECT_CONSISTENT


def test_report_carries_both_numbers_and_geometry():
    rep = bubble_object_report(_tau(X, [(0.4, 1.2), (3.9, 8.2)]), X)
    assert rep["n_crossings"] == 4
    assert rep["selected"]["length"] == pytest.approx(4.3, abs=0.05)
    assert rep["crest"]["length"] == pytest.approx(0.8, abs=0.05)
    assert rep["selected"]["reversed_area"] > 0.0
    assert set(rep["x_r_over_H"]) == {"largest_area", "first"}


def test_no_reattachment_is_not_reported_as_a_bubble():
    rep = bubble_object_report(np.ones_like(X), X)
    assert rep["n_bubbles"] == 0
    assert rep["flag"] == BUBBLE_SINGLE
    val = rep["x_r_over_H"]["largest_area"]
    assert val is None or (isinstance(val, float) and np.isnan(val))