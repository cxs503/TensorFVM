"""Extended tests for Armaly (1983) reproduction: fine grid and convergence.

These tests verify the grid setup and case factory without running the solver
(on fine grid the solver is too slow for CI).
"""

from __future__ import annotations

import pytest

from tensorfvm.inc_bench.cases import BFS_LEVELS, backward_step


def test_fine_grid_geometry():
    """Fine grid must reproduce the expansion ratio to < 0.01 %."""
    setup = backward_step("fine", reynolds_dh=100.0, downstream_step_heights=12.0)
    assert setup.geometry["expansion_ratio_rel_error"] < 0.0001
    assert setup.dx == setup.dy
    assert int(setup.fluid.sum()) > 60000  # ~67500 fluid cells


def test_all_grid_levels_exist():
    """BFS_LEVELS must have coarse, medium, and fine."""
    assert "coarse" in BFS_LEVELS
    assert "medium" in BFS_LEVELS
    assert "fine" in BFS_LEVELS
    # Fine should have more cells per step height than coarse
    assert BFS_LEVELS["fine"][0] > BFS_LEVELS["coarse"][0]


def test_fine_grid_setup_is_consistent():
    """Fine grid setup must be consistent with the case factory conventions."""
    for level in BFS_LEVELS:
        setup = backward_step(level, reynolds_dh=100.0, downstream_step_heights=12.0)
        assert setup.dx == setup.dy, f"{level}: non-uniform grid"
        assert setup.params["h_in"] > 0
        assert setup.params["h_out"] > 0
        assert setup.params["h_out"] / setup.params["h_in"] == pytest.approx(
            setup.geometry["expansion_ratio_mask"], rel=1e-4)


def test_grid_refinement_monotonicity():
    """Grid spacing should decrease and cell count should increase monotonically."""
    results = {}
    for level in ["coarse", "medium", "fine"]:
        setup = backward_step(level, reynolds_dh=100.0, downstream_step_heights=12.0)
        results[level] = {
            "dx": setup.dx,
            "cells": setup.cells,
            "fluid_cells": int(setup.fluid.sum()),
        }
    # Grid spacing should decrease monotonically
    assert results["coarse"]["dx"] > results["medium"]["dx"] > results["fine"]["dx"]
    # Cell count should increase monotonically
    assert results["coarse"]["cells"] < results["medium"]["cells"] < results["fine"]["cells"]


def test_long_domain_setup():
    """40S downstream domain must be constructible at all levels."""
    for level in ["coarse", "medium", "fine"]:
        setup = backward_step(level, reynolds_dh=389.0, downstream_step_heights=40.0)
        assert setup.params["downstream_length"] == 40.0
        assert setup.fluid.shape[1] > setup.fluid.shape[0]  # longer than tall
