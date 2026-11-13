"""Round-6 tests for ``CASE-FVM-PERIODIC-HILLS`` (fast; no long runs).

The heavy measurements live in ``docs/case-periodic-hills/ROUND6.md`` and its
``round6_*.json`` artefacts; what is tested here are the *instruments* used to
make those measurements, plus the two structural changes of round 6:

1. the SA closure convection order is now a knob
   (``SpalartAllmarasConfig.convection_blend``) and ``blend = 0`` reproduces
   round-5 transport **bit-identically** -- an instrument that silently changed
   the closure could otherwise "explain" any result;
2. the reattachment reference is single-sourced (C5).  The previous state of the
   repo had 4.36 in the acceptance test and 5.34 in the ledger with disjoint
   10 % bands, i.e. the gate's verdict depended on which file you read.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorfvm.periodic_hill_turbulence import SpalartAllmaras, SpalartAllmarasConfig
from tensorfvm.verification import periodic_hill_round6 as r6


# --------------------------------------------------------------------------- #
# C1 instrument: does the plateau detector separate "settling" from "drifting"?
# --------------------------------------------------------------------------- #
def test_plateau_fit_finds_asymptote_of_saturating_signal():
    t = np.arange(0.0, 300.0, 5.0)
    v = 20.0 - 18.0 * np.exp(-t / 40.0)
    out = r6.plateau_fit(t, v)
    assert out["equilibrated"] is True
    assert out["fit"]["ok"] is True
    assert abs(out["fit"]["tau"] - 40.0) / 40.0 < 0.10
    assert abs(out["fit"]["v_inf"] - 20.0) / 20.0 < 0.02
    assert out["asymptote_relative_error"] < 0.05


def test_plateau_fit_rejects_monotone_growth_with_no_asymptote():
    t = np.arange(0.0, 60.0, 2.0)
    v = 2.0 + 0.30 * t          # round-5's measured nu_t growth rate
    out = r6.plateau_fit(t, v)
    assert out["equilibrated"] is False
    assert out["fit"]["tau"] > out["tau_gate"]
    assert out["drift"]["relative"] > 0.02


def test_plateau_fit_accepts_flat_but_noisy_settled_signal():
    # a stationary, flat signal has no trend to resolve: the asymptote is its
    # tail mean and it must *not* be reported as unconverged
    rng = np.random.default_rng(0)
    t = np.arange(0.0, 200.0, 5.0)
    v = 5.0 + 0.01 * rng.standard_normal(t.size)
    out = r6.plateau_fit(t, v)
    assert abs(out["drift"]["relative"]) < 0.02
    assert out["trend_resolved"] is False
    assert out["equilibrated"] is True
    assert out["asymptote"] == pytest.approx(5.0, abs=0.02)


def test_drift_reports_relative_change_over_window():
    t = np.arange(0.0, 101.0, 1.0)
    assert r6.drift(t, np.full_like(t, 3.0))["relative"] == pytest.approx(0.0, abs=1e-12)
    out = r6.drift(t, 1.0 + 0.001 * t)      # +0.05 absolute over the 50-unit window
    assert out["relative"] == pytest.approx(0.05 / (1.0 + 0.001 * 100.0), rel=1e-9)
    assert out["start"] < out["end"]


def test_plateau_fit_handles_degenerate_input():
    out = r6.plateau_fit(np.array([1.0]), np.array([2.0]))
    assert out["equilibrated"] is False
    out = r6.plateau_fit(np.array([]), np.array([]))
    assert out["equilibrated"] is False


# --------------------------------------------------------------------------- #
# C4 instrument: geometric admission of the y/H = 0.5 stations
# --------------------------------------------------------------------------- #
def test_admissible_stations_rejects_lines_inside_the_hill_body():
    out = r6.admissible_stations(ny=24, channel_top=3.036)
    rejected = {e["x_over_H"] for e in out["rejected"]}
    accepted = [e["x_over_H"] for e in out["admissible"]]
    # y/H = 0.5 cuts through the hill for x/H <~ 0.9 and on the windward ramp
    assert 0.05 in rejected and 0.5 in rejected
    assert 5.0 in accepted and 6.0 in accepted
    for entry in out["rejected"]:
        if entry["x_over_H"] < 1.0:
            assert entry["clearance_cells"] < 0.0
    assert accepted == sorted(accepted)
    assert len(accepted) >= 5


def test_profile_error_tolerance_rules():
    reference = {1.0: -0.02, 5.0: 0.95, 0.05: -0.15}
    exact = r6.profile_error(dict(reference), profile=reference)
    assert exact["pass"] is True and exact["worst_error"] == pytest.approx(0.0)
    off = r6.profile_error({1.0: -0.03, 5.0: 0.60, 0.05: -0.15}, profile=reference)
    assert off["pass"] is False
    # near-zero reference switches to an absolute 0.05 band
    near = r6.profile_error({1.0: -0.06}, profile={1.0: -0.02})
    assert near["pass"] is True
    near_bad = r6.profile_error({1.0: 0.10}, profile={1.0: -0.02})
    assert near_bad["pass"] is False
    assert r6.profile_error({}, profile=reference)["pass"] is False


# --------------------------------------------------------------------------- #
# C2 structural: the convection-order knob
# --------------------------------------------------------------------------- #
def _closure(blend=None):
    solver = r6.hill_solver(nx=48, ny=24)
    solver.advance(20)
    config = SpalartAllmarasConfig() if blend is None else \
        SpalartAllmarasConfig(convection_blend=blend)
    closure = SpalartAllmaras(solver, config)
    solver.turbulence = closure
    return closure


def test_convection_blend_default_is_round5_behaviour():
    default = _closure(None)
    explicit = _closure(0.0)
    assert default.config.convection_blend == 0.0
    for _ in range(15):
        default.update(default.solver.time_step_s)
        explicit.update(explicit.solver.time_step_s)
    # bit-identical: the round-5 transport path has not been touched
    assert np.array_equal(default.nu_tilde, explicit.nu_tilde)


def test_second_order_convection_changes_the_transport():
    first = _closure(0.0)
    second = _closure(1.0)
    for _ in range(15):
        first.update(first.solver.time_step_s)
        second.update(second.solver.time_step_s)
    assert np.all(np.isfinite(second.nu_tilde))
    assert np.all(second.nu_tilde >= 0.0)
    assert not np.array_equal(first.nu_tilde, second.nu_tilde)
    relative = np.abs(second.nu_tilde - first.nu_tilde).max() / max(
        first.nu_tilde.max(), 1e-30)
    assert relative > 1e-6


@pytest.mark.parametrize("blend", [-0.1, 1.5, float("nan")])
def test_convection_blend_is_validated(blend):
    with pytest.raises(ValueError):
        SpalartAllmarasConfig(convection_blend=blend)


def test_closure_viscosity_helper_matches_model_definition():
    nu = 1.0 / 2800.0
    nu_t, chi, fv1 = r6.closure_viscosity(np.array([3.0 * nu]), nu)
    assert chi[0] == pytest.approx(3.0)
    assert fv1[0] == pytest.approx(27.0 / (27.0 + 7.1 ** 3))
    assert nu_t[0] < 3.0 * nu                      # fv1 < 1
    capped, _, _ = r6.closure_viscosity(
        np.array([1e8 * nu]), nu,
        SpalartAllmarasConfig(max_viscosity_ratio=100.0))
    assert capped[0] == pytest.approx(100.0 * nu)


# --------------------------------------------------------------------------- #
# C5 structural: one reference, machine-checked; C5b: the round-5 gate was
# unsatisfiable on the grid it was declared for
# --------------------------------------------------------------------------- #
def test_reference_is_single_sourced():
    scan = r6.scan_reference_consistency()
    assert scan["canonical"] == pytest.approx(5.34)
    assert scan["n_offending"] == 0, scan["offending"]


def test_acceptance_test_uses_the_ledger_reference():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).with_name("test_periodic_hill_breuer2009.py")
    spec = importlib.util.spec_from_file_location("_breuer2009_gate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    BREUER_PROFILE_Y05, BREUER_XR_H = module.BREUER_PROFILE_Y05, module.BREUER_XR_H

    assert BREUER_XR_H == pytest.approx(r6.REFERENCE["x_r_over_H"])
    assert BREUER_PROFILE_Y05 == r6.BREUER_PROFILE_Y05


def test_competing_reference_is_recorded_not_deleted():
    assert r6.COMPETING_X_R["value"] == pytest.approx(4.36)
    canonical = r6.REFERENCE["x_r_over_H"]
    lo_a, hi_a = 0.9 * canonical, 1.1 * canonical
    lo_b, hi_b = r6.COMPETING_X_R["band"]
    assert hi_b <= lo_a or hi_a <= lo_b, "the two 10 % bands are not disjoint"


def test_round5_y_plus_gate_was_unsatisfiable_at_ny64():
    req = r6.channel_y_plus_requirement()
    assert req["feasible_at_ny64"] is False
    assert req["ny_for_target"] > 64
    fine = r6.channel_y_plus_requirement(u_tau=1e-3)
    assert fine["feasible_at_ny64"] is True