"""Tests for the annular design-order verification harness.

The harness measures the production annular backend against the closed-form
annular Poiseuille solution; these tests protect both halves: the analytic
yardstick must stay self-consistent, and the measured second-order convergence
must hold on a cheap pair of grids so that a regression in the solver (or in the
harness) fails loudly.
"""
import math

import numpy as np
import pytest

from tensorfvm.verification import annular_order as ao


@pytest.fixture(scope="module")
def self_check():
    return ao.reference_self_check()


@pytest.fixture(scope="module")
def two_level_study():
    return ao.convergence(((8, 32), (16, 64)))


def test_analytic_reference_is_self_consistent(self_check):
    assert self_check["passed"], self_check["checks"]
    checks = self_check["checks"]
    assert checks["momentum_equation_relative_residual"] < ao.SELF_CHECK_LIMIT
    assert checks["no_slip_relative_residual"] < ao.SELF_CHECK_LIMIT
    assert checks["closed_form_vs_quadrature_flow_rate_relative"] < 1e-10
    # Analytic identity: pressure force on the cross-section equals the wall drag.
    assert checks["global_equilibrium_identity_relative_residual"] < 1e-12


def test_analytic_max_velocity_matches_numerical_argmax():
    value, radius = ao.analytic_max_velocity()
    grid = np.linspace(ao.RI, ao.RO, 400001)
    profile = ao.analytic_velocity(grid)
    assert radius == pytest.approx(grid[int(np.argmax(profile))], abs=1e-5)
    assert value == pytest.approx(profile.max(), rel=1e-9)
    # du/dr = 0 must hold at that radius.
    h = 1e-9
    slope = (ao.analytic_velocity(radius + h) - ao.analytic_velocity(radius - h)) / (2 * h)
    assert abs(slope) / value * ao.RI < 1e-5


def test_geometry_audit_reports_second_order_area_defect():
    coarse, fine = ao.geometry_audit(8), ao.geometry_audit(16)
    assert coarse["positive_volumes"] and fine["positive_volumes"]
    assert fine["cells"] == 16 * 64
    assert abs(fine["relative_area_error"]) < abs(coarse["relative_area_error"])
    order = math.log(abs(coarse["relative_area_error"]) / abs(fine["relative_area_error"])) / math.log(2)
    assert order == pytest.approx(2.0, abs=0.15)
    # Straight-edge walls always undershoot the true annulus area.
    assert fine["relative_area_error"] < 0


def test_production_solver_matches_integrated_quantities():
    row = ao.run_case(16)
    values = row["values"]
    assert row["cells"] == 1024
    assert row["solver_relative_residual"] < 1e-10
    assert abs(values["flow_rate_error"]) < 0.02
    assert abs(values["l2"]) < ao.ERROR_LIMIT
    assert abs(values["inner_wall_shear_pa"] - abs(values["exact_inner_wall_shear_pa"])) / abs(
        values["exact_inner_wall_shear_pa"]) < ao.ERROR_LIMIT
    assert abs(values["outer_wall_shear_pa"] - abs(values["exact_outer_wall_shear_pa"])) / abs(
        values["exact_outer_wall_shear_pa"]) < ao.ERROR_LIMIT
    # The pressure source and the total wall traction balance on the discrete system.
    assert row["wall_traction_balance_relative"] < 1e-9


def test_observed_second_order_and_acceptance(two_level_study):
    orders = two_level_study["orders"]
    assert orders["velocity_l2"][0] == pytest.approx(2.0, abs=0.1)
    assert orders["velocity_linf"][0] > ao.ORDER_LIMIT
    assert orders["flow_rate"][0] > ao.ORDER_LIMIT
    verdict = ao.acceptance(two_level_study)
    assert verdict["passed"], verdict


def test_order_estimator_negative_control():
    """A first-order error sequence must not pass the second-order gate."""
    first_order = ao._order(1e-3, 1e-3 / 2.1, 1.0 / 8, 1.0 / 16)
    assert first_order < ao.ORDER_LIMIT
    assert ao._order(1e-3, 1e-3 / 4.0, 1.0 / 8, 1.0 / 16) == pytest.approx(2.0, abs=1e-12)
    # Negative radius is outside the closed form's domain: numpy yields nan, not silently
    # a plausible positive velocity.
    assert np.isnan(ao.analytic_velocity(-1.0))
    with pytest.raises(ValueError):
        ao._annular_config(0)

@pytest.mark.parametrize("gap_ratio", [0.2, 1e-4, 1e-8, 2.**-40])
def test_closed_form_flow_across_thin_annuli(gap_ratio, record_property):
    from decimal import Decimal, localcontext
    from tensorfvm.annular import AnnularConfig, analytic_flow

    cfg = AnnularConfig(inner_radius_m=0.75, outer_radius_m=0.75*(1+gap_ratio),
                        pressure_gradient_pa_m=3., viscosity_pa_s=0.125)
    with localcontext() as ctx:
        ctx.prec = 80
        ri, ro = map(Decimal.from_float, (cfg.inner_radius_m, cfg.outer_radius_m))
        bracket = ro**4 - ri**4 - (ro**2 - ri**2)**2/(ro/ri).ln()
        expected = math.pi*3*float(bracket)/(8*.125)
    production = analytic_flow(cfg)
    report = ao.analytic_flow_rate(cfg.inner_radius_m, cfg.outer_radius_m, 3., .125)
    error = max(abs(production/expected-1), abs(report/expected-1))
    record_property("reference_flow_m3_s", expected)
    record_property("max_relative_flow_error", error)
    assert error < 2e-14


@pytest.mark.parametrize("gap_ratio", [0.2, 1e-4, 1e-8, 2.**-40])
def test_thin_gap_velocity_profile(gap_ratio, record_property):
    from decimal import Decimal, localcontext
    from tensorfvm.annular import AnnularConfig, analytic_velocity

    ri, ro = .75, .75*(1+gap_ratio)
    cfg = AnnularConfig(inner_radius_m=ri, outer_radius_m=ro,
                        pressure_gradient_pa_m=3., viscosity_pa_s=.125)
    radii = np.r_[ri, np.nextafter(ri, ro), ri + (ro-ri)*np.array([.125, .5, .875]),
                  np.nextafter(ro, ri), ro]
    with localcontext() as ctx:
        ctx.prec = 80
        a, b = map(Decimal.from_float, (ri, ro))
        k = (b*b-a*a)/(b/a).ln()
        exact = [float(6*(b*b-r*r-k*(b/r).ln())) for r in map(Decimal.from_float, radii)]
    profiles = np.array([analytic_velocity(radii, cfg), ao.analytic_velocity(radii, ri, ro, 3., .125)])
    wall_error = float(np.max(abs(profiles[:, [0, -1]])))
    interior_error = float(np.max(abs(profiles[:, 1:-1]/exact[1:-1]-1)))
    record_property("max_wall_speed_m_s", wall_error)
    record_property("max_relative_interior_error", interior_error)
    assert wall_error == 0.
    assert interior_error < 2e-14


@pytest.mark.parametrize("gap_ratio", [0.2, 1e-4, 1e-8, 2.**-40])
def test_thin_gap_wall_traction(gap_ratio, record_property):
    from decimal import Decimal, localcontext

    ri, ro = .75, .75*(1+gap_ratio)
    gradient = 3.
    with localcontext() as ctx:
        ctx.prec = 80
        a, b = map(Decimal.from_float, (ri, ro))
        k = (b*b-a*a)/(b/a).ln()
        expected = [float(3*(k/r-2*r)/4) for r in (a, b)]
    inner, outer = ao.analytic_wall_shear(ri, ro, gradient)
    shear_error = max(abs(inner/expected[0]-1), abs(outer/expected[1]-1))
    pressure_force = gradient*math.pi*(ro-ri)*(ro+ri)
    wall_drag = 2*math.pi*(ri*inner - ro*outer)
    balance_error = abs(wall_drag/pressure_force-1)
    record_property("reference_inner_Pa", expected[0])
    record_property("reference_outer_Pa", expected[1])
    record_property("max_relative_wall_error", shear_error)
    record_property("relative_force_imbalance", balance_error)
    assert inner > 0 > outer
    assert max(shear_error, balance_error) < 2e-14


@pytest.mark.parametrize("gap", [.15, 7.5e-5, 7.5e-9, 0.75*2.**-40, 3*math.ulp(.75)])
def test_thin_gap_peak_speed_and_location(gap, record_property):
    from decimal import Decimal, localcontext

    ri, ro = .75, .75+gap
    with localcontext() as ctx:
        ctx.prec = 90
        a, b = map(Decimal.from_float, (ri, ro))
        k = (b*b-a*a)/(b/a).ln()
        peak = (k/2).sqrt()
        exact_speed = float(6*(b*b-peak*peak-k*(b/peak).ln()))
    speed, radius = ao.analytic_max_velocity(ri, ro, 3., .125)
    speed_error = abs(speed/exact_speed-1)
    radius_ulps = abs(radius-float(peak))/math.ulp(ri)
    record_property("reference_peak_speed_m_s", exact_speed)
    record_property("relative_speed_error", speed_error)
    record_property("radius_error_ulps", radius_ulps)
    assert ri < radius < ro
    assert radius_ulps <= 1
    assert speed_error < 2e-14


@pytest.mark.parametrize("gap", [2.**-16, 2.**-28])
def test_thin_annulus_cell_geometry(gap, record_property):
    from decimal import Decimal, localcontext
    from tensorfvm.annular import AnnularConfig, AnnularMesh

    cfg = ao._mesh_config(4, 32)
    cfg.annular_config = AnnularConfig(ntheta=32, nr=4, inner_radius_m=.75, outer_radius_m=.75+gap)
    mesh = AnnularMesh(cfg)
    vertices = mesh.vertices.numpy()
    max_area_error = max_center_error = 0.
    with localcontext() as ctx:
        ctx.prec = 70
        for j, i in np.ndindex(mesh.field_shape):
            corners = vertices[[j, j+1, j+1, j], [i, i, i+1, i+1]]
            polygon = [tuple(map(Decimal.from_float, row)) for row in corners]
            edges = list(zip(polygon, polygon[1:] + polygon[:1]))
            crosses = [x*v-y*u for (x,y), (u,v) in edges]
            twice_area = sum(crosses)
            center = [float(sum((a[d]+b[d])*c for (a,b), c in zip(edges, crosses))/(3*twice_area))
                      for d in (0, 1)]
            max_area_error = max(max_area_error, abs(float(mesh.volumes[j,i])/float(twice_area/2)-1))
            max_center_error = max(max_center_error, np.linalg.norm(mesh.centers[j,i].numpy()-center))
    record_property("max_relative_cell_area_error", max_area_error)
    record_property("max_centroid_displacement_m", max_center_error)
    assert max_area_error < 2e-15
    assert max_center_error < 4e-16


@pytest.mark.parametrize("sectors,gap", [(17, .25), (32, 2.**-28)])
def test_annular_periodic_face_chains(sectors, gap, record_property):
    from tensorfvm.annular import AnnularConfig, AnnularMesh

    cfg = ao._mesh_config(4, sectors)
    cfg.annular_config = AnnularConfig(ntheta=sectors, nr=4, inner_radius_m=.75, outer_radius_m=.75+gap)
    mesh = AnnularMesh(cfg)
    faces = mesh.face_vertices.numpy()
    vectors = mesh.face_area_vectors.numpy()
    owners, neighbors = mesh.owner.numpy(), mesh.neighbor.numpy()
    open_cells = 0
    max_defect = 0.
    for cell in range(4*sectors):
        outgoing, incoming = owners == cell, neighbors == cell
        oriented = np.concatenate((faces[outgoing], faces[incoming, ::-1]))
        starts = sorted(map(tuple, oriented[:, 0]))
        ends = sorted(map(tuple, oriented[:, 1]))
        open_cells += starts != ends
        normal = np.concatenate((vectors[outgoing], -vectors[incoming]))
        max_defect = max(max_defect, math.hypot(*(math.fsum(column) for column in normal.T)))
    seam = mesh.vertices[:, -1].numpy() - mesh.vertices[:, 0].numpy()
    mismatch = float(np.max(np.linalg.norm(seam, axis=1)))
    record_property("max_seam_endpoint_mismatch_m", mismatch)
    record_property("open_cell_chains", open_cells)
    record_property("max_cell_normal_sum_m", max_defect)
    assert mismatch == 0.
    assert open_cells == 0
    assert max_defect <= np.finfo(float).eps * cfg.annular_config.outer_radius_m
