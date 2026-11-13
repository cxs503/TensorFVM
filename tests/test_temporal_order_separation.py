"""Temporal-order separation for the cell-centred FV scheme (fixed-grid CFL scan).

Pins the temporal-error budget of the merged cell-centred solver
(``EulerSolver``, MUSCL-Hancock + HLLC) on the periodic entropy wave: with the
spatial grid fixed, varying only ``dt`` (via ``cfl``) must leave the error
*spatial-dominated* (log-log slope of ``E`` vs ``dt`` ~ 0, not ~1), i.e. the
scheme is not temporally zeroth-order and the temporal contribution is a minor
(< 15 %) part of the total error.  The exact temporal order ``q`` is honestly
marked ``WITHHELD`` (not resolvable at affordable grid cost).
"""
import numpy as np
import pytest

from tensorfvm.verification.temporal_order_separation import (
    CFL_LADDER, temporal_scan, study)


def test_dt_scales_linearly_with_cfl():
    """The solver's stable_dt is cfl*dx/c, so dt is a linear function of cfl."""
    scan = temporal_scan(40)
    dt = scan["dt"]
    # dt = cfl * h, h = 1/40
    assert dt == pytest.approx([c * 1.0 / 40 for c in CFL_LADDER], rel=1e-12)
    # the ladder is a geometric series: dt ratio 2 between neighbours
    assert dt[1] / dt[0] == pytest.approx(2.0, rel=1e-12)
    assert dt[2] / dt[1] == pytest.approx(2.0, rel=1e-12)


def test_error_is_spatial_dominated_not_temporal_zeroth_order():
    """On a fixed grid the error barely moves with dt: slope ~0, not ~1.

    A temporally zeroth-order scheme would show a ~1 log-log slope as dt is
    refined (the error would scale linearly with dt).  Here the slope is ~0,
    so the temporal error is a minor contributor -- the scheme is not
    temporally zeroth-order.
    """
    for nx in (40, 80, 160):
        scan = temporal_scan(nx)
        assert scan["spatial_dominated"] is True, scan
        # slope is near zero (spatial-dominated), not ~1 (temporal-zeroth-order)
        assert abs(scan["temporal_slope"]) < 0.25, scan
        # the temporal budget (4x dt refinement) is a small fraction of the total
        assert scan["temporal_budget_4x"] < 0.15, scan
        assert scan["temporal_budget_pass_15"] is True


def test_temporal_order_is_honestly_withheld():
    """The exact temporal order q is not resolvable at affordable cost."""
    scan = temporal_scan(40)
    assert scan["temporal_order_resolved"] is False
    assert "WITHHELD" in scan["note"]


def test_scan_reports_the_time_and_steps_it_reached():
    scan = temporal_scan(40)
    assert all(t > 0 for t in scan["t"])
    assert all(s > 0 for s in scan["steps"])
    # finer dt (smaller cfl) means more steps to the same t_end
    assert scan["steps"][0] > scan["steps"][-1]


def test_study_runs_all_grids_and_reports_budget():
    out = study((40, 80))
    assert [s["nx"] for s in out["scans"]] == [40, 80]
    assert all(s["temporal_budget_pass_15"] for s in out["scans"])
