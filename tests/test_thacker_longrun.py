"""Tests for the 2-period Thacker long-window audit (``thacker_longrun``)."""

from __future__ import annotations

import numpy as np
import pytest

from tensorfvm.verification import thacker_basin as tb
from tensorfvm.verification import thacker_longrun as tl


def test_envelope_speed_matches_exact_velocity_field():
    """``|beta| s(t)`` must equal max |u| of the exact linear velocity field."""
    basin = tb.build_basin()
    for frac in (0.25, 0.5, 0.75):
        t = frac * basin.period
        X, Y = tb.grid(basin, 40, 1.5)
        u, v = basin.velocity(X, Y, t)
        grid_max = float(np.max(np.hypot(u, v)))
        envelope = tl.envelope_speed(basin, t)
        # the cell-centre grid stops short of the exact shoreline, so the grid
        # maximum is slightly *below* the envelope (measured 0.19 % at n = 40).
        assert envelope >= grid_max
        assert envelope == pytest.approx(grid_max, rel=1e-2)


def test_half_period_window_is_healthy_and_conservative():
    """The graded 0.5-period window stays physical and conserves volume."""
    m = tl.run(n=24, periods=0.5, samples=21)
    assert m["volume"]["drift"] < 1e-10
    assert m["centre_depth"]["l2_relative"] < 0.10
    assert m["velocity"]["max"] < 4.0
    assert m["health"]["clipped_cells"] == 0


def test_two_periods_expose_the_ledger_defect():
    """Honest current state: 2 periods are *not* physical, and why.

    The assertion is a *guard* on the defect (D16): the day a cure lands the
    numbers move and the report has to be re-graded, so the guard is written
    against the attribution rather than against a bare magnitude.
    """
    m = tl.run(n=24, periods=2.0, samples=21)
    healthy = (m["velocity"]["max"] < 6.0
               and m["centre_depth"]["l2_relative"] < 0.05)
    if healthy:
        pytest.skip("2-period window is now healthy; re-grade the report")
    spike = m["spike"]
    # The offending cell is a thin film at/outside the exact shoreline, and the
    # exact solution has (essentially) no velocity there.
    assert spike["h_over_H"] < 0.1
    assert spike["radius"] >= 0.9 * spike["shoreline"]
    assert spike["speed_exact"] < 1e-6


def test_thin_film_clip_preserves_mass():
    """The cure probe may clip momentum, never mass."""
    m = tl.run(n=24, periods=0.25, thin=1e-2, samples=11)
    assert m["volume"]["drift"] < 1e-10
    assert m["thin_clips"] >= 0