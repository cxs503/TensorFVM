"""Tests for the double-body closure audit (debt D9) and debt D8 falsification."""
import numpy as np
import pytest

from tensorfvm.verification.double_body_added_mass_gate import (
    hemisphere_checks, hemisphere_panels, hull_route_checks, kirchhoff_assembly_checks,
    observed_order, power_law_limit, run, shift_added_mass, skew,
    sphere_kirchhoff_about_origin, spheroid_kirchhoff_about_origin,
    spheroid_kirchhoff_about_origin as _spheroid_ref)
from tensorfvm.panel_added_mass import added_mass_matrix_6dof, sphere_added_mass


def test_hemisphere_rim_closes_the_body():
    panels = hemisphere_panels(1.0, 16, 18)
    assert panels.n_panels > 0
    assert np.max(panels.centroids[:, 2]) < 0.0
    assert np.max(panels.corners[:, :, 2]) < 1e-12   # rim on z = 0, body below


def test_hemisphere_odd_phi_is_rejected():
    with pytest.raises(ValueError):
        hemisphere_panels(1.0, 16, 17)


def test_mirror_route_matches_exact_half_body():
    """Mirror (double-body) hemisphere sway = m_a/2, the anchor C1 gates on."""
    out = hemisphere_checks(((16, 18),))
    row = out["rows"][0]
    assert abs(row["err_sway"]) < 0.03
    assert abs(row["err_yaw"]) < 0.03
    # the sphere's own rotational added inertia about a diameter is exactly zero
    assert row["yaw_own_axis_rel"] < 1e-6


def test_shift_added_mass_round_trip_and_coupling():
    delta = np.array([1.5, -0.4, 0.7])
    base = np.diag([2.0, 3.0, 4.0, 0.1, 0.2, 0.3])
    moved = shift_added_mass(base, delta)
    back = shift_added_mass(moved, -delta)
    assert np.max(np.abs(back - base)) < 1e-12
    # translation-rotation block of a purely translational matrix is -M [c]x
    M = np.diag([2.0, 3.0, 4.0])
    closed = shift_added_mass(np.diag([2.0, 3.0, 4.0, 0.0, 0.0, 0.0]), delta)
    assert np.allclose(closed[:3, 3:], M @ skew(delta), atol=1e-12)
    assert np.allclose(closed[3:, 3:], -skew(delta) @ M @ skew(delta), atol=1e-12)


def test_sphere_closed_form_parallel_axis():
    radius, rho, d = 1.0, 1025.0, 1.5
    m_a = sphere_added_mass(radius, rho)
    ref = sphere_kirchhoff_about_origin(radius, rho, (d, 0.0, 0.0))
    assert ref[5, 5] == pytest.approx(m_a * d * d, rel=1e-12)      # yaw about z
    assert ref[3, 3] == pytest.approx(0.0, abs=1e-12)              # roll about x
    assert ref[1, 5] == pytest.approx(m_a * d, rel=1e-12)          # sway-yaw coupling


def test_spheroid_closed_form_structure():
    ref = spheroid_kirchhoff_about_origin(2.0, 1.0)
    assert ref[0, 0] < ref[1, 1]          # axial added mass below transverse
    assert ref[3, 3] == pytest.approx(0.0, abs=1e-12)   # roll about the symmetry axis
    assert ref[4, 4] == pytest.approx(ref[5, 5], rel=1e-12)
    assert np.allclose(ref[:3, 3:], 0.0, atol=1e-12)    # no coupling about the centre


def test_assembly_matches_closed_form_sphere():
    """All 36 entries of the 6x6 assembly against the parallel-axis closed form."""
    rho, centre = 1.0, (1.5, 0.0, 0.0)
    from tensorfvm.panel_added_mass import sphere_panels_offset
    panels = sphere_panels_offset(1.0, centre, 24, 24)
    assembled = added_mass_matrix_6dof(panels, rho=rho)
    ref = sphere_kirchhoff_about_origin(1.0, rho, centre)
    assert np.max(np.abs(assembled - ref)) / np.max(np.abs(ref)) < 0.03


def test_debt_d8_is_not_reproducible():
    """D8 claimed I_zz = 2.05 for a sphere at (0,0,1.5); 2.05 is the heave entry."""
    out = kirchhoff_assembly_checks()["d8"]
    assert abs(out["yaw_rotational"]) < 1e-6
    assert out["yaw_reference"] == 0.0
    assert out["heave_translational"] == pytest.approx(2.05, abs=0.05)


def test_hull_lid_cap_is_not_the_double_body():
    """The lid closure under-states both M_22 and J_z, systematically (C3)."""
    out = hull_route_checks(((21, 5),))
    row = out["rows"][0]
    assert row["M22_ratio"] < 0.75
    assert row["Jz_ratio"] < 0.75
    assert row["lid_flux_max_sway"] < 1e-12   # D7's premise: the lid carries no flux
    assert row["lid_flux_max_yaw"] < 1e-12
    assert abs(row["volume_rel_error"]) < 0.05


def test_limit_fit_and_order_helpers():
    exact = 5.0
    counts = np.array([100.0, 400.0, 1600.0])
    values = exact - 30.0 * counts ** -1.0
    fit = power_law_limit(counts, values)
    assert fit["limit"] == pytest.approx(exact, rel=1e-3)
    assert fit["exponent"] == pytest.approx(1.0, abs=0.05)
    assert observed_order(0.03, 0.03 / 2.0, 100.0, 200.0) == pytest.approx(1.0, rel=1e-9)
    assert np.isnan(observed_order(0.01, 0.02, 100.0, 200.0))


def test_run_is_serialisable_and_all_pass():
    import json
    summary = run(hemi_ladder=((16, 18),), hull_ladder=((21, 5),))
    json.dumps(summary, default=float)
    assert set(summary["criteria"]) == {"C1", "C2", "C3", "C4", "C5", "C6"}
    assert summary["all_pass"]
