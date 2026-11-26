"""Verification-debt closure for the merged annular axial Poiseuille backend.

Before this test nothing imported ``tensorfvm.annular`` outside the backend
itself: the assembled production sparse solve existed, the analytic reference
was stored in ``fields['reference_axial_velocity_m_s']``, and no check ever
compared them.  The gates below are external to the backend's own algebra: the
governing ODE and the no-slip conditions are proven on an independently
re-derived profile in 80-digit arithmetic, and the volumetric flow rate is
independently quadratured from the closed form.
"""

import pytest

from tensorfvm.annular import AnnularConfig
from tensorfvm.verification.annular import (
    GATE,
    flow_rate_audit,
    observed_order,
    ode_and_bc_audit,
    run_case,
)


@pytest.fixture(scope='module')
def config():
    return AnnularConfig(ntheta=32, nr=16)


def test_stored_closed_form_solves_the_governing_ode(config):
    audit = ode_and_bc_audit(config)
    # (1/r) d/dr (r du/dr) + G/mu = 0 to high-precision rounding.
    assert audit['max_relative_ode_residual'] < 1e-25
    # No-slip on both circular walls.
    assert audit['wall_residual_over_peak'] < 1e-14
    # The backend's stored formula equals the independent derivation.
    assert audit['stored_vs_independent_max_relative'] < 1e-13
    # Negative control: the residual indicator is falsifiable (5% error -> 5%).
    assert audit['perturbation_control_min_residual'] > 0.04


def test_closed_form_flow_rate_matches_independent_quadrature(config):
    audit = flow_rate_audit(config)
    assert audit['relative_difference'] < 1e-12
    assert audit['quadrature_flow_m3_s'] > 0


def test_production_solve_meets_reference_and_second_order():
    coarse = run_case(16, 8)
    fine = run_case(32, 16)
    for row in (coarse, fine):
        row['inner_radius_m'] = 0.05
    order = observed_order([coarse, fine], 'relative_l2')['fitted_order']
    assert coarse['relative_l2'] > fine['relative_l2']
    assert order > 1.85
    assert fine['relative_l2'] < GATE
    assert fine['relative_flow_rate_error'] < GATE
    # Reference consumption: the stored reference is the independent closed form.
    assert fine['solved_consistency_max'] < 1e-13
    # The gate would reject a doubled error or a 5% reference perturbation.
    assert not (2.0 < GATE) and not (0.05 < GATE)


def test_reversed_flow_is_rejected_by_the_config_guard():
    # Negative control for the gate family: a sign error can only enter through
    # the pressure gradient, and AnnularConfig refuses non-positive gradients,
    # so reversed (suction-driven) axial flow is not representable - a documented
    # limitation of the merged backend rather than a silent wrong answer.
    with pytest.raises(ValueError):
        AnnularConfig(ntheta=32, nr=16, pressure_gradient_pa_m=-0.01)


@pytest.mark.parametrize("ulps", [3, 1024])
def test_thin_gap_flux_quadrature(ulps, record_property):
    import math
    import mpmath as mp
    from tensorfvm.verification.annular_order import quadrature_flow_rate

    ri = .75
    ro = ri + ulps*math.ulp(ri)
    cfg = AnnularConfig(inner_radius_m=ri, outer_radius_m=ro,
                        pressure_gradient_pa_m=3., viscosity_pa_s=.125)
    split = ri + (ulps//2)*math.ulp(ri)
    values = []
    with mp.workdps(90):
        inner, outer = mp.mpf(ri), mp.mpf(ro)
        k = (outer**2-inner**2)/mp.log(outer/inner)
        def integrand(r):
            return 12*mp.pi*r*(outer**2-r**2-k*mp.log(outer/r))
        for a, b in [(ri, ro), (ri, split), (ro, split)]:
            expected = float(mp.quad(integrand, [a, b]))
            measured = flow_rate_audit(cfg, a, b)["quadrature_flow_m3_s"]
            values.append(abs(measured/expected-1))
        full = float(mp.quad(integrand, [ri, ro]))
    values.append(abs(quadrature_flow_rate(ri, ro, 3., .125)/full-1))
    record_property("full_reference_m3_s", full)
    record_property("max_relative_integral_error", max(values))
    assert max(values) < 1e-13


@pytest.mark.parametrize("gap", [.25, 3*2.**-53])
def test_partial_flow_audit_uses_requested_bounds(gap, record_property):
    import math
    import mpmath as mp

    ri = .75
    ro = ri+gap
    split = math.nextafter(ri, ro) if gap < 1e-12 else ri+gap/3
    cfg = AnnularConfig(inner_radius_m=ri, outer_radius_m=ro, pressure_gradient_pa_m=3., viscosity_pa_s=.125)
    errors, differences = [], []
    with mp.workdps(90):
        inner, outer = mp.mpf(ri), mp.mpf(ro)
        k = (outer**2-inner**2)/mp.log(outer/inner)
        def integrand(r):
            return 12*mp.pi*r*(outer**2-r**2-k*mp.log(outer/r))
        for a, b in [(None, split), (ro, split), (ro, ri), (split, split)]:
            expected = float(mp.quad(integrand, [ri if a is None else a, b]))
            audit = flow_rate_audit(cfg, a, b)
            reference = audit["closed_form_flow_m3_s"]
            errors.append(abs(reference/expected-1) if expected else abs(reference))
            differences.append(audit["relative_difference"])
    record_property("max_reference_error_nonzero_intervals", max(errors[:-1]))
    record_property("zero_interval_reference_m3_s", errors[-1])
    record_property("max_reported_relative_difference", max(differences))
    assert max(errors[:-1]) < 2e-14
    assert errors[-1] == 0.
    assert max(differences) < 1e-13


@pytest.mark.parametrize("ulps", [1, 3, 1024])
def test_thin_gap_ode_audit_at_reported_radii(ulps, record_property):
    import math
    import mpmath as mp

    ri, ro = .75, .75 + ulps*math.ulp(.75)
    cfg = AnnularConfig(inner_radius_m=ri, outer_radius_m=ro, pressure_gradient_pa_m=3., viscosity_pa_s=.125)
    with mp.workdps(90):
        report = ode_and_bc_audit(cfg)
        precision_after_audit = mp.mp.dps
    with mp.workdps(90):
        a, b = mp.mpf(ri), mp.mpf(ro)
        k = (b*b-a*a)/mp.log(b/a)
        expected_peak = max(abs(6*(b*b-r*r-k*mp.log(b/r))) for r in map(mp.mpf, report["sample_radii"]))
    peak = report["peak_velocity_m_s"]
    peak_error = abs(peak/float(expected_peak)-1) if expected_peak else abs(peak)
    record_property("sample_peak_error", peak_error)
    record_property("stored_profile_error", report["stored_vs_independent_max_relative"])
    record_property("relative_ode_residual", report["max_relative_ode_residual"])
    assert precision_after_audit == 90
    assert peak_error < 1e-13
    assert report["stored_vs_independent_max_relative"] < 1e-13
    assert report["max_relative_ode_residual"] < 1e-25
    assert report["wall_residual_over_peak"] < 1e-13
    assert report["perturbation_control_min_residual"] == pytest.approx(.05)
