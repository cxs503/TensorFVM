"""Tests for the rudder cavitation / erosion face (node p234w2-n09).

Every test pins one of the achievements the node report claims, against an
independent anchor where one exists:

* the panel-method influence kernels against numerical quadrature of the sheet
  integrals (the sign/scale of the source and vortex kernels);
* ``Cp`` of a circle against ``1 - 4 sin^2`` and of a 12 %-thick Joukowski section
  against the exact conformal-mapping solution (this is what validates
  ``Cp_min`` and hence ``sigma_i``);
* the lift slope against the Abbott & von Doenhoff thickness correction;
* the minimum-pressure inception identity ``sigma_i = -Cp_min``, its sign and
  magnitude versus the exact ``d sigma_i/d alpha``;
* the Rayleigh-Plesset integrator against the exact energy-integral collapse time
  and velocity;
* the exactness of the erosion power law ``E ~ (p_wall - p_v)^n``;
* the negative controls (``sigma -> infinity``, nuclei-free limit);
* the whole case run, end to end (all criteria pass, JSON is written).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensorfvm.cavitation import erosion as ER, inception as IC, reference as REF
from tensorfvm.cavitation import report as RPT
from tensorfvm.cavitation.conformal import joukowski_for_thickness
from tensorfvm.cavitation.section import (
    _influence, naca4_contour, naca4_thickness, panel_solve,
)


# --------------------------------------------------------------------------- #
# water properties and the cavitation number itself
# --------------------------------------------------------------------------- #
def test_vapour_pressure_matches_tables():
    # CRC / steam-table values: 1705 Pa at 15 degC, 2339 Pa at 20 degC
    assert IC.water_vapour_pressure(15.0) == pytest.approx(1705.0, rel=0.01)
    assert IC.water_vapour_pressure(20.0) == pytest.approx(2339.0, rel=0.01)
    assert IC.water_vapour_pressure(30.0) == pytest.approx(4246.0, rel=0.02)
    with pytest.raises(ValueError):
        IC.water_vapour_pressure(120.0)


def test_cavitation_number_is_the_definition():
    p_inf, p_v, rho, v = 30_000.0, 1705.0, 1025.0, 8.0
    sigma = IC.cavitation_number(p_inf, p_v, rho, v)
    assert sigma == pytest.approx((p_inf - p_v) / (0.5 * rho * v * v), rel=1e-15)
    # V -> 0 at fixed static pressure drives sigma -> infinity (negative control)
    assert IC.cavitation_number(p_inf, p_v, rho, 1e-9) > 1e18
    # sigma = 0 exactly at the vapour pressure
    assert IC.cavitation_number(p_v, p_v, rho, v) == 0.0
    with pytest.raises(ValueError):
        IC.cavitation_number(p_inf, p_v, -1.0, v)


# --------------------------------------------------------------------------- #
# panel kernels and the panel solution
# --------------------------------------------------------------------------- #
def test_influence_kernels_match_quadrature():
    """Source/vortex sheet kernels vs direct numerical quadrature of the sheets."""
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])
    tangents = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    normals = np.array([[0.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [1.0, 0.0]])
    lengths = np.ones(4)
    centroids = 0.5 * (nodes[1:] + nodes[:-1])
    St, Sn, Vt, Vn = _influence(centroids, nodes, tangents, normals, lengths)
    assert St.shape == Sn.shape == Vt.shape == Vn.shape == (4, 4)
    # the self terms are the one-sided limits: source jump +1/2 in the normal
    # direction and the vortex sheet seen from outside, -1/2 tangentially
    for i in range(4):
        assert Sn[i, i] == pytest.approx(0.5, abs=1e-12)
        assert Vt[i, i] == pytest.approx(-0.5, abs=1e-12)
        assert St[i, i] == pytest.approx(0.0, abs=1e-12)
        assert Vn[i, i] == pytest.approx(0.0, abs=1e-12)

    # off-diagonal entries against Gauss quadrature of the sheet integrals:
    # velocity of a unit source / unit vortex sheet on panel j at field point i
    s, w = np.polynomial.legendre.leggauss(400)
    xs, ws = 0.5 * (s + 1.0), 0.5 * w
    for i, j in ((2, 0), (0, 2), (1, 3), (3, 1)):
        tang_j, nor_j = tangents[j], normals[j]
        d = centroids[i] - nodes[j]
        x0, y0 = float(d @ tang_j), float(d @ nor_j)
        dx, dy = x0 - xs, np.full_like(xs, y0)
        r2 = dx * dx + dy * dy
        # velocity components in panel j's own (t, n) frame ...
        src_local = np.array([np.sum(ws * dx / r2), np.sum(ws * dy / r2)]) / (2.0 * math.pi)
        vor_local = np.array([-np.sum(ws * dy / r2), np.sum(ws * dx / r2)]) / (2.0 * math.pi)
        # ... mapped back to global coordinates
        src = src_local[0] * tang_j + src_local[1] * nor_j
        vor = vor_local[0] * tang_j + vor_local[1] * nor_j
        # projections onto the field panel frame
        tang_i, nor_i = tangents[i], normals[i]
        assert St[i, j] == pytest.approx(float(src @ tang_i), rel=1e-8)
        assert Sn[i, j] == pytest.approx(float(src @ nor_i), rel=1e-8)
        assert Vt[i, j] == pytest.approx(float(vor @ tang_i), rel=1e-8)
        assert Vn[i, j] == pytest.approx(float(vor @ nor_i), rel=1e-8)


def test_circle_cp_min_matches_analytic():
    th = -np.linspace(0.0, 2.0 * math.pi, 401)
    nodes = np.column_stack([0.5 * np.cos(th), 0.5 * np.sin(th)])
    sol = panel_solve(nodes, 0.0, kutta=False)
    # cylinder: Cp = 1 - 4 sin^2 -> Cp_min = -3, Cp_max = 1
    assert sol.cp_min == pytest.approx(-3.0, rel=5e-3)
    assert sol.cp_max == pytest.approx(1.0, rel=5e-3)
    assert sol.tangency_residual < 1e-10


def test_joukowski_exact_solution_agreement():
    sec = joukowski_for_thickness(0.12)
    for alpha in (0.0, 2.0, 5.0, 10.0):
        sol = panel_solve(sec.contour(200), alpha)
        # Cp_min is the quantity the inception criterion consumes
        assert sol.cp_min == pytest.approx(sec.cp_min(alpha), rel=0.02)
        if alpha > 0.0:      # cusped trailing edge: Cl converges as O(h)
            assert sol.cl(chord=sec.chord) == pytest.approx(sec.cl(alpha), rel=0.06)
    assert sec.thickness_ratio == pytest.approx(0.12, rel=1e-6)


def test_naca0012_symmetry_and_lift_slope():
    nodes = naca4_contour(t=0.12, n_half=200)
    sol = panel_solve(nodes, 0.0)
    assert sol.cl() == pytest.approx(0.0, abs=1e-10)          # symmetric section
    upper, lower = sol.cp[:200], sol.cp[200:][::-1]
    assert np.max(np.abs(upper - lower)) < 1e-10              # upper/lower mirror
    # Abbott & von Doenhoff thickness correction
    slope = panel_solve(nodes, 1.0).cl() / math.radians(1.0)
    assert slope == pytest.approx(2.0 * math.pi * (1.0 + 0.77 * 0.12), rel=0.02)


def test_naca_thickness_distribution_closes():
    assert naca4_thickness(1.0, 0.12) == pytest.approx(0.0, abs=1e-15)
    assert naca4_thickness(0.0, 0.12) == pytest.approx(0.0, abs=1e-15)
    assert naca4_thickness(0.3, 0.12) == pytest.approx(0.06, rel=1e-3)


# --------------------------------------------------------------------------- #
# inception criterion
# --------------------------------------------------------------------------- #
def test_inception_is_minus_cpmin():
    for alpha in (0.0, 4.0, 8.0):
        sol = panel_solve(naca4_contour(t=0.18, n_half=200), alpha)
        assert IC.sigma_inception(alpha, thickness_ratio=0.18, n_half=200) == pytest.approx(
            -sol.cp_min, rel=1e-12)


def test_inception_sensitivity_positive_and_exact():
    """d sigma_i/d alpha > 0 (cavitation starts earlier at incidence) and exact."""
    sec = joukowski_for_thickness(0.12)
    h = 0.25
    for alpha in (2.0, 5.0, 8.0):
        d_panel = (-panel_solve(sec.contour(200), alpha + h).cp_min
                   + panel_solve(sec.contour(200), alpha - h).cp_min) / (2 * h)
        d_exact = (-sec.cp_min(alpha + h) + sec.cp_min(alpha - h)) / (2 * h)
        assert d_panel > 0.0
        assert d_panel == pytest.approx(d_exact, rel=0.10)
    curve = IC.sigma_inception_curve(np.arange(0.0, 11.0, 1.0), thickness_ratio=0.18)
    assert np.all(np.diff(curve) > 0.0)


def test_nuclei_free_limit_diverges():
    """sigma_i* -> infinity as the nucleus radius -> 0 (criterion (d))."""
    water = IC.WaterState()
    p_inf, v = 2.0e5, 10.0
    sigmas = [IC.sigma_inception_with_nuclei(0.0, r, water, p_inf, v)
              for r in (1e-4, 1e-5, 1e-6, 1e-7)]
    assert all(b > a for a, b in zip(sigmas, sigmas[1:]))
    assert sigmas[-1] > 10.0 * sigmas[0]
    # Blake threshold asymptote: P_c * r0 -> -4 S / (3 sqrt(3)) as r0 -> 0
    r0 = 1e-9
    limit = ER  # noqa: F841  (keep the import used)
    p_c = IC.blake_critical_pressure(r0, 0.0728, 1.0e5, 1705.0)
    assert p_c * r0 == pytest.approx(-4.0 * 0.0728 / (3.0 * math.sqrt(3.0)), rel=1e-2)


# --------------------------------------------------------------------------- #
# bubble dynamics and erosion
# --------------------------------------------------------------------------- #
def test_rayleigh_plesset_against_exact_collapse():
    rho, r_max, dp = 1025.0, 1.0e-4, 5.0e4
    t_c = ER.rayleigh_collapse_time(r_max, dp, rho)
    assert t_c == pytest.approx(0.915 * r_max * math.sqrt(rho / dp), rel=1e-12)
    t, r, rdot = ER.integrate_rayleigh_plesset(r_max, dp, rho, r_end=0.1 * r_max,
                                               n_steps=60000)
    t_ref = ER.rayleigh_time_to_radius(0.1 * r_max, r_max, dp, rho)
    assert abs(t[-1] - t_ref) / t_c < 5e-3
    v_exact = ER.collapse_velocity(r_max, 0.1 * r_max, dp, rho)
    assert abs(rdot[-1]) == pytest.approx(v_exact, rel=0.05)


def test_impact_pressure_water_hammer():
    rho, c = 1025.0, 1470.0
    p = ER.impact_pressure(5.0e4, rho=rho, sound_speed=c, growth_ratio=12.0, p_v=0.0)
    v = ER.collapse_velocity(1e-4, 1e-4 / 12.0, 5.0e4, rho)
    assert p == pytest.approx(rho * c * v, rel=1e-12)
    assert ER.impact_pressure(0.0) == 0.0        # no pressure deficit -> no impact


def test_erosion_power_law_is_exact():
    energy_slope, r2 = ER.erosion_exponent_fit(2.0, 3.0, 1.0, 10.0, exponent=2.5)
    assert energy_slope == pytest.approx(2.5, abs=1e-9)
    assert r2 == pytest.approx(1.0, abs=1e-12)
    for n in (2.0, 3.0):
        slope, _ = ER.erosion_exponent_fit(2.0, 3.0, 1.0, 10.0, exponent=n)
        assert slope == pytest.approx(n, abs=1e-9)


def test_erosion_zero_above_inception_and_peak_in_band():
    chord, v, sigma_i = 1.0, 10.0, 3.0
    assert ER.erosion_rate(sigma_i, sigma_i, chord, v) == 0.0
    assert ER.erosion_rate(1.5 * sigma_i, sigma_i, chord, v) == 0.0
    ratios = np.linspace(1.001, 4.0, 200)
    rates = np.array([ER.erosion_rate(sigma_i / r, sigma_i, chord, v,
                                      exponent=2.5, attenuation=2.0) for r in ratios])
    peak = float(ratios[int(np.argmax(rates))])
    assert REF.EROSION_PEAK_BAND.contains(peak)
    r2 = REF.r_squared(REF.reference_damage_curve(ratios), REF.scaled(rates))
    assert r2 > 0.8


def test_material_ranking():
    ah36 = ER.MaterialErosionResistance("AH36", 130.0)
    duplex = ER.MaterialErosionResistance("duplex", 250.0)
    e = 1.0e12
    assert duplex.relative_erosion(e) < ah36.relative_erosion(e)


# --------------------------------------------------------------------------- #
# end-to-end case run
# --------------------------------------------------------------------------- #
def test_case_run_all_criteria_pass(tmp_path):
    case = RPT.build_case(n_half=120, alpha_deg=12.0)
    summary = RPT.run_case(case, make_figures=False, out_dir=tmp_path)
    assert summary["operating_point"]["sigma"] == pytest.approx(3.704, abs=1e-2)
    assert summary["operating_point"]["cavitating"] is True
    assert 8.0 < summary["operating_point"]["onset_angle_deg"] < 12.0
    for key, block in summary["criteria"].items():
        if key == "e_pytest":
            continue
        if key == "d_negative_controls":
            assert block["sigma_inf_no_cavitation"]["passed"]
            assert block["nuclei_free_sigma_i_diverges"]["passed"]
            continue
        assert block["passed"], (key, block)
    assert summary["all_passed"] is True
    assert summary["physics_checks"]["rayleigh_plesset"]["passed"] is True
    assert summary["physics_checks"]["erosion_exponent"]["passed"] is True


def test_summary_json_is_written(tmp_path):
    summary = RPT.run_case(RPT.build_case(n_half=80), make_figures=False, out_dir=tmp_path)
    path = tmp_path / "summary.json"
    assert path.is_file()
    import json
    back = json.loads(path.read_text())
    assert back["operating_point"]["sigma"] == pytest.approx(
        summary["operating_point"]["sigma"], rel=1e-12)
    for required in ("case", "operating_point", "criteria", "physics_checks",
                     "sweeps", "bands", "materials"):
        assert required in back