"""Tests for the well-balanced shallow-water bed solver (m16596).

The physics under test is an exact property of the shallow-water equations, not
a fitted number: ``h = max(0, eta0 - z), u = 0`` is a steady solution for any
bed.  Two of these tests deliberately assert that a *bad* scheme fails the same
property, so the gate is proven to be able to fail.
"""

import numpy as np
import pytest

from tensorfvm.shallow_water.bed import ShallowWaterBed1D
from tensorfvm.shallow_water.swe1d import ShallowWater1D
from tensorfvm.verification import shallow_water_well_balanced as wb


def _solver(bed_fn, surface, *, n=60, x_min=0.0, x_max=10.0, order=2,
            well_balanced=True):
    dx = (x_max - x_min) / n
    x = x_min + (np.arange(n) + 0.5) * dx
    solver = ShallowWaterBed1D(n, x_min, x_max, bed=bed_fn(x), order=order,
                               well_balanced=well_balanced)
    solver.lake_at_rest_state(surface)
    return solver


def test_bed_shape_and_finiteness_are_validated():
    with pytest.raises(ValueError):
        ShallowWaterBed1D(10, 0.0, 1.0, bed=np.zeros(9))
    with pytest.raises(ValueError):
        ShallowWaterBed1D(10, 0.0, 1.0, bed=np.full(10, np.nan))


def test_flat_bed_reproduces_the_shipped_flat_solver_bit_for_bit():
    payload = wb.flat_reduction(n=200, t_end=0.5)
    assert payload["max_depth_difference"] == 0.0
    assert payload["max_momentum_difference"] == 0.0
    assert payload["steps_flat"] == payload["steps_bed"]


@pytest.mark.parametrize("bed_fn", [wb.smooth_bed, wb.step_bed])
def test_lake_at_rest_is_preserved_to_round_off(bed_fn):
    solver = _solver(bed_fn, 1.0)
    solver.advance(5.0)
    h, u = solver.primitive()
    assert np.max(np.abs(u)) < wb.MAX_REST_VELOCITY
    assert solver.surface_drift() < wb.MAX_SURFACE_DRIFT
    assert solver.clipped_cells == 0
    assert h.min() >= 0.0


def test_lake_at_rest_survives_the_wet_dry_boundary():
    solver = _solver(wb.piercing_bump, 0.45)
    dry0 = solver.h <= solver.dry_depth
    assert dry0.any() and not dry0.all()
    solver.advance(5.0)
    assert np.array_equal(dry0, solver.h <= solver.dry_depth)
    assert np.max(np.abs(solver.primitive()[1])) < wb.MAX_REST_VELOCITY
    assert solver.h.min() >= 0.0


def test_first_order_reconstruction_is_also_well_balanced():
    solver = _solver(wb.step_bed, 1.0, order=1)
    solver.advance(5.0)
    assert np.max(np.abs(solver.primitive()[1])) < wb.MAX_REST_VELOCITY


def test_naive_centred_source_breaks_the_lake_at_rest():
    """Discrimination: the same initial state must *not* stay at rest."""
    solver = _solver(wb.step_bed, 1.0, well_balanced=False)
    solver.advance(1.0)
    assert np.max(np.abs(solver.primitive()[1])) > wb.NAIVE_MIN_VELOCITY


def test_constant_bed_offset_changes_the_solution_only_at_round_off():
    payload = wb.constant_bed_invariance(n=200, t_end=0.5)
    assert payload["max_depth_difference"] < wb.ROUNDOFF_TOLERANCE
    assert payload["max_momentum_difference"] < wb.ROUNDOFF_TOLERANCE


def test_wet_dry_dam_break_stays_positive_and_conserves_volume():
    payload = wb.wet_dry_dam_break(n=200, t_end=0.5)
    assert payload["finite"]
    assert payload["h_min"] >= 0.0
    assert payload["clipped_cells"] == 0
    assert payload["volume_drift"] < wb.ROUNDOFF_TOLERANCE


def test_cfl_number_is_a_dimensionless_courant_number():
    """Regression for the m16596 defect fix in ``ShallowWater1D.cfl_number``.

    The old body returned ``max_wave_speed() * dx / dx`` -- the wave speed in
    m/s.  With ``h = 1 m`` that is 3.13, so a ``cfl < 1`` assertion catches it.
    """
    flat = ShallowWater1D(50, 0.0, 10.0, cfl=0.4)
    x = flat.x
    flat.set_state(np.where(x < 5.0, 1.0, 0.1), None)
    flat.advance(0.01)
    cfl = flat.cfl_number()
    speed = flat.max_wave_speed()
    assert cfl == pytest.approx(speed * flat.dt_last / flat.dx, rel=1e-15)
    assert 0.0 < cfl <= flat.cfl + 1e-12
    assert speed > 1.5  # the old bug returned this, not a Courant number


def test_verification_payload_declares_every_criterion():
    payload = wb.run(n_lake=60, n_break=200, t_end_lake=3.0, t_end_naive=0.5)
    ids = {c["id"] for c in payload["criteria"]}
    for required in ("W1a_smooth_bed_max_velocity", "W2a_step_bed_max_velocity",
                     "W3a_wet_dry_max_velocity", "W4_naive_control_spurious_velocity",
                     "F1a_flat_reduction_depth", "M1b_step_bed_volume_drift"):
        assert required in ids
    # every criterion that is not one of the three declared round-off cases
    # must pass at this resolution
    roundoff_ids = {"F2a_constant_bed_depth", "F2b_constant_bed_momentum",
                    "D1b_sill_volume_drift"}
    failed = {c["id"] for c in payload["criteria"]
              if c["verdict"] == "FAIL"} - roundoff_ids
    assert failed == set()
    assert payload["all_pass_roundoff"]