"""Round-1 free-surface-datum regression tests (``GAP-FVM-TWOPHASE``).

The round-1 fix promotes the ghost-fluid free-surface pressure datum
(``p = 0`` half a cell from the surface cell centre) from an opt-in experiment to
the shipped default, because the historical treatment drops the free-surface
face from the pressure operator -- a homogeneous Neumann condition, i.e. a rigid
lid -- which (a) leaves the lake-at-rest pressure O(h) wrong and (b) delays the
gravity-driven release of the Martin & Moyce dam break.

These tests pin the two properties that motivate the change: the default is the
consistent datum, and on a coarse dam-break leg the shipped treatment releases
measurably earlier than the retained negative control.  Gate D1/D2 themselves
stay unchanged and stay FAILING (the capability gap is not closed by this fix);
nothing here re-tunes a criterion.
"""

import pytest

from tensorfvm.vof_phase0.solver import Phase0TwoPhaseSolver


def test_shipped_default_is_the_consistent_surface_datum():
    solver = Phase0TwoPhaseSolver(8, 8, 1.0, 1.0)
    assert solver.well_balanced_surface is True
    control = Phase0TwoPhaseSolver(8, 8, 1.0, 1.0, well_balanced_surface=False)
    assert control.well_balanced_surface is False


def test_rest_state_pressure_error_collapses_with_the_shipped_datum():
    """Lake at rest: the shipped datum holds p to round-off, the control is O(h)."""
    from tensorfvm.vof_verification.hydrostatic import run_hydrostatic

    shipped = run_hydrostatic(20, 20, steps=8, surface_model="void",
                              well_balanced=True)
    control = run_hydrostatic(20, 20, steps=8, surface_model="void",
                              well_balanced=False)
    assert shipped["pressure_oscillation"] < 1e-10
    assert control["pressure_oscillation"] > 1e-2      # 1/(2n) = 2.5 %
    assert shipped["spurious_velocity"] < 1e-10


@pytest.mark.parametrize("arm", [None, False])
def test_dambreak_leg_records_its_surface_treatment(arm):
    from tensorfvm.twophase_gap.dambreak_gate import run_dambreak_leg

    rec = run_dambreak_leg(40, 16, well_balanced_surface=arm,
                           control_steps=4, max_steps=400)["record"]
    expected = "shipped-default" if arm is None else "centred-control"
    assert rec["surface_treatment"] == expected
    assert rec["well_balanced_surface"] is (True if arm is None else False)
    # the treatment never touches the conservation gates
    assert rec["volume_drift"] < 1e-10
    assert rec["bound_violation"] == 0.0


def test_shipped_datum_releases_earlier_than_the_centred_control():
    """The 60x20 leg: smaller release lag and smaller D2 than the control."""
    from tensorfvm.twophase_gap.dambreak_gate import run_dambreak_leg

    shipped = run_dambreak_leg(60, 20)["record"]
    control = run_dambreak_leg(60, 20, well_balanced_surface=False)["record"]
    lag_s = shipped["diagnostics"]["front"]["phase_lag_t_star"]
    lag_c = control["diagnostics"]["front"]["phase_lag_t_star"]
    assert lag_s < 0.75 * lag_c, (lag_s, lag_c)
    assert (shipped["height_max_rel_error"]
            < 0.95 * control["height_max_rel_error"]), (
        shipped["height_max_rel_error"], control["height_max_rel_error"])
    # honest FAIL is preserved: the gate is not passed by either arm
    assert shipped["height_max_rel_error"] > 0.15