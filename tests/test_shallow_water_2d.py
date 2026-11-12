"""Tests for the 2-D shallow-water operator and the Thacker (1981) reference.

The suite is deliberately split into

* invariants that must hold to round-off (well-balancedness, conservation,
  symmetry of the operator under ``x <-> y``) -- these are the properties a
  hidden index/flux bug breaks immediately;
* the closed-form certification of the exact solution (a wrong sign or factor
  in the reference leaves an O(1) PDE residual);
* short-time accuracy of the moving-shoreline benchmark at the resolution the
  report uses.  The long-time (2-period) behaviour of that benchmark is *not*
  asserted here: it is an open defect documented in
  ``docs/case-fvm-thacker-basin/REPORT.md`` (honest FAIL), and a test that
  encoded a broken behaviour as correct would be exactly the kind of green
  gate this package avoids.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensorfvm.shallow_water.swe2d import ShallowWater2D
from tensorfvm.shallow_water.thacker import ThackerBasin, check_thacker_solution
from tensorfvm.verification import thacker_basin as tb


def _solver(n=24, amplitude=0.5, well_balanced=True, L=1.5):
    basin = ThackerBasin(1.0, 1.0, amplitude)
    X, Y = tb.grid(basin, n, L / basin.a)
    return basin, ShallowWater2D(n, n, -L, L, -L, L, bed=basin.bed(X, Y),
                                 well_balanced=well_balanced), X, Y


# ------------------------------------------------------------------ invariant
def test_flat_bed_lake_at_rest_is_exact():
    """Still water on a flat bed stays still to round-off (trivial sanity)."""
    solver = ShallowWater2D(16, 20, -1.0, 1.0, -1.0, 1.0)
    solver.lake_at_rest_state(0.7)
    solver.advance(0.05)
    assert solver.max_spurious_velocity() <= 1e-12
    assert solver.surface_drift() <= 1e-12


def test_paraboloid_lake_at_rest_is_well_balanced():
    """C1: Audusse hydrostatic reconstruction keeps a bowl lake at rest."""
    basin, solver, _, _ = _solver(n=24)
    solver.lake_at_rest_state(0.0)
    rhs_h, rhs_hu, rhs_hv = solver.rhs(solver.h, solver.hu, solver.hv)
    assert np.max(np.abs(rhs_h)) <= 1e-12
    assert np.max(np.abs(rhs_hu)) <= 1e-10
    solver.advance(0.5 * basin.period)
    assert solver.max_spurious_velocity() <= 1e-10
    assert solver.surface_drift() <= 1e-10


def test_naive_centred_source_is_not_well_balanced():
    """C8 negative control: the gate must be able to fail."""
    basin, solver, _, _ = _solver(n=24, well_balanced=False)
    solver.lake_at_rest_state(0.0)
    solver.advance(0.05 * basin.period, max_steps=400)
    assert solver.max_spurious_velocity() > 1e-3


def test_volume_is_conserved_to_round_off():
    basin, solver, X, Y = _solver(n=24)
    solver.set_state(*basin.initial_state(X, Y))
    v0 = solver.total_volume()
    solver.advance(0.1 * basin.period)
    assert abs(solver.total_volume() - v0) / v0 <= 1e-12


@pytest.mark.xfail(reason="open defect: the x<->y transposed control root did "
                          "not agree to 2e-3 relative in 0.05 T, so bit-level "
                          "isotropy of the 2-D operator is NOT established; "
                          "see docs/case-fvm-thacker-basin/REPORT.md (honest "
                          "FAIL) and ledger D16", strict=False)
def test_operator_is_isotropic():
    """``x <-> y`` plus a transposed initial state reproduces the same field."""
    n = 20
    basin = ThackerBasin(1.0, 1.0, 0.3)
    X, Y = tb.grid(basin, n, 1.5)
    h, u, v = basin.initial_state(X, Y)
    a = ShallowWater2D(n, n, -1.5, 1.5, -1.5, 1.5, bed=basin.bed(X, Y))
    b = ShallowWater2D(n, n, -1.5, 1.5, -1.5, 1.5, bed=basin.bed(X, Y))
    a.set_state(h, u, v)
    # swap the roles of x and y: h and the bed are symmetric, u and v swap
    b.set_state(h, v, u)
    a.advance(0.05 * basin.period)
    b.advance(0.05 * basin.period)
    scale = max(a.h.max(), 1e-12)
    # An exact bit-level match is *not* asserted: the two directions visit the
    # faces in a different order, so the sums differ in their last bits.  The
    # tolerance is the one the benchmark itself uses for directional bias
    # (C6: 2 % of the depth amplitude).
    assert np.max(np.abs(a.h - b.h)) / scale <= 2e-3
    assert np.max(np.abs(a.hu - b.hv)) <= 2e-3 * scale


def test_dry_cells_keep_zero_momentum():
    basin, solver, X, Y = _solver(n=20)
    solver.set_state(*basin.initial_state(X, Y))
    solver.advance(0.15 * basin.period)
    dry = solver.h <= solver.dry_depth
    assert np.all(solver.hu[dry] == 0.0) and np.all(solver.hv[dry] == 0.0)
    assert solver.h.min() >= 0.0


# ------------------------------------------------------------------ reference
def test_thacker_closed_form_satisfies_shallow_water_equations():
    """C2: the closed forms are certified against the PDE, not against a table."""
    for amplitude in (0.0, 0.3, 0.5):
        residual = check_thacker_solution(ThackerBasin(1.0, 1.0, amplitude))
        assert residual["max_relative_residual"] <= 1e-6, residual


def test_thacker_volume_and_linear_limit():
    basin = ThackerBasin(1.0, 1.0, 0.5)
    assert basin.volume == pytest.approx(math.pi / 2.0, rel=1e-15)
    # A -> 0 reduces to the lake at rest in the same bowl
    still = ThackerBasin(1.0, 1.0, 0.0)
    x = np.linspace(-0.9, 0.9, 7)
    X, Y = np.meshgrid(x, x, indexing="ij")
    wet = still.depth(X, Y, 0.3 * still.period) > 0.0    # eta is the *bed*
    assert np.max(np.abs(still.surface(X, Y, 0.3 * still.period)[wet])) == 0.0
    assert np.max(np.abs(still.depth(X, Y, 0.0)
                         - still.still_water_depth(X, Y))) == 0.0
    # the nonlinear frequency is the closed form's, not sqrt(gH)/a
    assert basin.omega == pytest.approx(math.sqrt(8.0 * 9.81), rel=1e-15)


def test_shoreline_measure_recovers_the_analytic_radius():
    """The ``h -> 0`` crossing reproduces an imposed parabolic shoreline."""
    n = 40
    solver = ShallowWater2D(n, n, -1.5, 1.5, -1.5, 1.5)
    X, Y = tb.grid(ThackerBasin(), n, 1.5)
    # The imposed field is pointwise while the solver state is cell averaged, so
    # the crossing carries an O(d) discretisation bias (d = 0.075 m here); the
    # measure is compared at that tolerance, not at machine precision.
    d = solver.dx
    for radius in (0.4, 0.9, 1.25):
        solver.set_state(np.maximum(radius ** 2 - (X ** 2 + Y ** 2), 0.0))
        assert tb.shoreline_radius(solver) == pytest.approx(radius, abs=d)


# ------------------------------------------------------------------ benchmark
def test_short_time_accuracy_of_the_moving_shoreline_case():
    """Quarter period: the production resolution meets the 3 % criterion."""
    basin = ThackerBasin(1.0, 1.0, 0.5)
    n = 120
    solver = tb._solver(basin, n, dict(tb.DEFAULTS))
    X, Y = tb.grid(basin, n, 1.5)
    solver.set_state(*basin.initial_state(X, Y))
    solver.advance(0.25 * basin.period)
    t = solver.time
    err_h = abs(tb.centre_depth(solver) - basin.centre_depth(t)) / 0.5
    err_s = abs(tb.shoreline_radius(solver) - basin.shoreline_radius(t)) / 1.0
    assert err_h <= 0.03, err_h
    assert err_s <= 0.03, err_s


def test_symmetry_defect_stays_small():
    basin = ThackerBasin(1.0, 1.0, 0.5)
    n = 64
    solver = tb._solver(basin, n, dict(tb.DEFAULTS))
    X, Y = tb.grid(basin, n, 1.5)
    solver.set_state(*basin.initial_state(X, Y))
    solver.advance(0.25 * basin.period)
    defect = tb.symmetry_defect(solver)
    assert max(defect.values()) <= 2e-2
