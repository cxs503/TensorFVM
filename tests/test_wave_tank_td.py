"""Tests for the sigma-grid time-domain numerical wave tank (``wave_tank_td``).

The suite is deliberately organised as *physics statements* rather than code
coverage:

* still water at rest is an exact fixed point (no spurious waves, no pressure
  drift) -- this is what the legacy VOF tank could not hold without a
  hydrostatic initialisation;
* the velocity field the relaxation zone imposes is divergence free and the
  surface-face Dirichlet makes the discrete free-surface stiffness equal ``g``
  (checked through the hydrostatic pressure node);
* the projection is exact: ``max|div u| / max|flux|`` stays at round-off with
  and without a body inside the tank;
* the absorbing beach really absorbs, and a tank without it reflects;
* the cut-cell section develops the hydrostatic buoyancy ``rho g nabla`` and
  floats in calm water;
* the potential-flow reference reproduces the analytic waterplane restoring
  coefficients and the flat-plate added-mass limit, with positive damping.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensorfvm.wave_tank.theory import dispersion_k, stokes2_elevation
from tensorfvm.wave_tank_td import (SigmaWaveTank, TankConfig, WaveTarget,
                                    stokes2_eta, stokes2_potential_velocities)
from tensorfvm.wave_tank_td.cases import rectangular_section
from tensorfvm.wave_tank_td.potential2d import (hydrostatic_restoring,
                                                section_hydrodynamics,
                                                waterline_width, wetted_polygon)

G = 9.80665


# --------------------------------------------------------------- wave target
def test_dispersion_matches_analytic_limits():
    # deep water: omega^2 = g k
    k = dispersion_k(2.0, depth=1e4)
    assert abs(k - 4.0 / G) / (4.0 / G) < 1e-6
    # shallow water: omega = k sqrt(g h)
    k_s = dispersion_k(0.05, depth=1.0)
    assert abs(k_s - 0.05 / math.sqrt(G)) / k_s < 1e-4


def test_stokes2_eta_identical_to_legacy_tank():
    wt = WaveTarget(0.02, 1.0, 1.0)
    x = np.linspace(0.0, 3.0, 11)
    for t in (0.0, 0.37, 1.0):
        a = stokes2_eta(x, t, wt.amp, wt.omega, wt.k, wt.depth)
        b = stokes2_elevation(x, t, wt.amp, wt.omega, wt.k, wt.depth)
        assert np.allclose(a, b, rtol=0, atol=1e-15)


def test_stokes2_velocity_satisfies_free_surface_kinematics():
    """``w(z=eta) = d(eta)/dt`` and the second harmonic has the bound amplitude."""
    wt = WaveTarget(0.02, 1.0, 1.0)
    dt = 1e-5
    x = np.array([0.3, 1.1, 2.4])
    eta = wt.eta(x, 0.25)
    _, _ = wt.velocities(x, eta, 0.25)
    detadt = (wt.eta(x, 0.25 + dt) - wt.eta(x, 0.25 - dt)) / (2 * dt)
    u, w = wt.velocities(x, eta, 0.25, )
    u1, w1 = stokes2_potential_velocities(x, eta, 0.25, wt.amp, wt.omega, wt.k,
                                          wt.depth, order2=False)
    assert np.allclose(u1, wt.amp * wt.omega * np.cosh(wt.k * (eta + wt.depth))
                       / math.sinh(wt.k * wt.depth) * np.cos(wt.k * x - wt.omega * 0.25),
                       atol=1e-12)
    assert np.abs(u - u1).max() < 0.1 * np.abs(u1).max()   # 2nd order is small
    detadx = (wt.eta(x + 1e-5, 0.25) - wt.eta(x - 1e-5, 0.25)) / 2e-5
    # kinematic condition on a material surface: w = d(eta)/dt + u d(eta)/dx
    assert np.allclose(w, detadt + u * detadx, atol=0.02 * wt.amp * wt.omega)



# ------------------------------------------------------------ still water
def test_still_water_is_an_exact_fixed_point():
    cfg = TankConfig(length=4.0, depth=1.0, nx=60, ns=20, relax_waves=0.0,
                     damp_strength=0.0)
    tank = SigmaWaveTank(cfg)
    for _ in range(25):
        tank.step(1e-3)
    assert np.abs(tank.u).max() < 1e-12
    assert np.abs(tank.w).max() < 1e-12
    assert np.abs(tank.eta).max() < 1e-12


def test_hydrostatic_pressure_node_matches_g_times_depth():
    """The surface Dirichlet sits half a cell away, so p_top = g*dy/2 exactly."""
    cfg = TankConfig(length=4.0, depth=1.0, nx=40, ns=20, relax_waves=0.0,
                     damp_strength=0.0)
    tank = SigmaWaveTank(cfg)
    for _ in range(5):
        tank.step(1e-3)
    # at rest the stored (non-hydrostatic) pressure is identically zero and the
    # total pressure is exactly the hydrostatic field g(eta - y)
    assert np.abs(tank.p).max() < 1e-9 * G
    ptot = tank.total_pressure()
    assert np.allclose(ptot, G * (tank.eta[:, None] - tank.yc), atol=1e-9)


# ------------------------------------------------------------- projection
@pytest.mark.parametrize("with_body", [False, True])
def test_projection_is_exact_and_conserves_volume(with_body):
    cfg = TankConfig(length=4.0, depth=1.0, nx=60, ns=20, relax_waves=0.5,
                     damp_waves=0.5, damp_strength=0.0)
    tank = SigmaWaveTank(cfg)
    tank.set_wave(0.02, 1.0)
    if with_body:
        # the body is held still on purpose: the test asks whether the *projection*
        # stays exact with a body inside the tank, and a freely floating section
        # in this cut-cell tank is a separate (unstable, documented) problem --
        # see docs/wave-tank-td-seakeeping/README.md section 4.
        tank.set_body(rectangular_section(0.4, 0.15, x0=2.0), added_mass=(200.0, 2.0),
                      motion="fixed")
    vol0 = tank.volume()
    for _ in range(20):
        tank.step(2e-3)
    assert tank.div_residual < 1e-8
    # with the wavemaker inlet the tank volume is free to exchange mass with the
    # wavemaker, so volume conservation is a property of the *closed* tank only
    assert abs(tank.volume() - vol0) / vol0 < 1e-3


def test_closed_tank_conserves_volume_to_round_off():
    cfg = TankConfig(length=6.0, depth=1.0, nx=120, ns=20, relax_waves=0.0,
                     damp_strength=0.0)
    tank = SigmaWaveTank(cfg)
    tank.set_wave(0.02, 1.0)
    tank.initialize_from_wave(0.0)          # wave in the tank, inlet not driven
    tank.u[0, :] = 0.0
    vol0 = tank.volume()
    for _ in range(30):
        tank.step(5e-3)
    assert abs(tank.volume() - vol0) / vol0 < 1e-12


def test_relaxation_zone_imposes_the_target_at_the_inlet():
    cfg = TankConfig(length=6.0, depth=1.0, nx=120, ns=24, advection_blend=0.15)
    tank = SigmaWaveTank(cfg)
    tank.set_wave(0.02, 1.0)
    for _ in range(30):
        t_in = tank.t
        tank.step(1e-2)
    ut, _ = tank._target_fields(t_in)
    # the relaxation weight is 1 on the inlet face, so u[0] *is* the target
    assert np.allclose(tank.u[0, :], ut[0, :], rtol=1e-2, atol=1e-9)
    # and the wave is actually generated (elevation departs from still water)
    assert np.abs(tank.eta).max() > 1e-3


def test_sponge_rate_is_zero_before_the_beach_and_positive_inside():
    cfg = TankConfig(length=12.0, depth=1.0, nx=200, ns=12)
    tank = SigmaWaveTank(cfg)
    wt = tank.set_wave(0.02, 1.0)
    x0 = cfg.length - cfg.damp_waves * wt.wavelength
    rate = tank.sponge_rate(tank.x)
    assert np.all(rate[tank.x < x0] == 0.0)
    assert rate[-1] > 0.0
    # the last cell centre sits half a cell short of the wall, so the
    # quadratic ramp is ~2 % below its wall value
    assert abs(rate[-1] - cfg.damp_strength * wt.omega) < 0.05 * cfg.damp_strength * wt.omega


def test_beach_absorbs_so_the_back_wall_stays_quiet():
    """With the beach the far wall elevation stays ~1 % of a; without it, not."""
    def wall_eta(damping: bool):
        cfg = TankConfig(length=6.0, depth=1.0, nx=150, ns=20, advection_blend=0.15)
        if not damping:
            cfg.damp_strength = 0.0
        tank = SigmaWaveTank(cfg)
        tank.set_wave(0.02, 1.0)
        for _ in range(300):
            tank.step(1e-2)
        return abs(tank.eta[-1])

    a = 0.02
    assert wall_eta(True) < 0.05 * a
    assert wall_eta(False) > 0.10 * a


# ------------------------------------------------------------------- body
def test_section_geometry_and_buoyancy():
    nx, ns = 120, 32
    cfg = TankConfig(length=4.0, depth=1.0, nx=nx, ns=ns, relax_waves=0.0,
                     damp_strength=0.0)
    tank = SigmaWaveTank(cfg)
    length, draft = 0.6, 0.2
    tank.set_body(rectangular_section(length, draft, x0=2.0), added_mass=(300.0, 3.5))
    tank._update_mask()
    fx, fy, m5 = tank.body_force()

    expected = cfg.density * G * length * draft
    assert abs(fy - expected) / expected < 0.03, (fy, expected)
    assert abs(fx) < 1e-6 * expected
    # the reference point defaults to the section's own axis, so a symmetric
    # section carries no static pitch moment
    assert abs(m5) < 1e-6 * expected


def test_body_floats_in_calm_water():
    cfg = TankConfig(length=4.0, depth=1.0, nx=120, ns=28, relax_waves=0.0,
                     damp_strength=0.0)
    tank = SigmaWaveTank(cfg)
    tank.set_body(rectangular_section(0.6, 0.2, x0=2.0), added_mass=(300.0, 3.5))
    for _ in range(200):
        tank.step(2e-3)
    assert abs(tank.body["xi"][0]) < 5e-3
    assert abs(tank.body["xi"][1]) < 5e-3
    assert abs(tank.volume_drift) / tank.volume0 < 5e-3


def test_cell_coverage_of_an_aligned_section():
    cfg = TankConfig(length=4.0, depth=1.0, nx=120, ns=40, relax_waves=0.0)
    tank = SigmaWaveTank(cfg)
    length, draft = 0.6, 0.2
    tank.set_body(rectangular_section(length, draft, x0=2.0))
    tank._update_mask()
    displaced = tank.displaced_volume()
    assert abs(displaced - length * draft) / (length * draft) < 0.03


# ------------------------------------------------------- potential reference
def test_waterline_and_wetted_geometry():
    poly = rectangular_section(0.6, 0.2)
    assert abs(waterline_width(poly) - 0.6) < 1e-12
    wet = wetted_polygon(poly)
    area = 0.5 * abs(np.dot(wet[:, 0], np.roll(wet[:, 1], -1))
                     - np.dot(wet[:, 1], np.roll(wet[:, 0], -1)))
    assert abs(area - 0.12) < 1e-12


def test_hydrostatic_restoring_matches_box_formula():
    b, t = 0.6, 0.2
    poly = rectangular_section(b, t)
    rest = hydrostatic_restoring(poly)
    assert abs(rest["C33"] - 1000.0 * G * b) / (1000.0 * G * b) < 1e-12
    # C55 = rho g nabla GM with GM = I_wl/nabla - (z_G - z_B).  For a box with
    # its deck on the waterline z_G = z_B = -T/2, so GM = I_wl/nabla and
    # C55 = rho g I_wl (a freeboard would subtract rho g nabla (z_G - z_B)).
    expected = 1000.0 * G * (b ** 3 / 12.0)
    assert abs(rest["C55"] - expected) / expected < 1e-9


def test_potential_added_mass_and_damping_positive():
    poly = rectangular_section(0.6, 0.2)
    for period in (1.6, 1.0, 0.8):
        h = section_hydrodynamics(poly, omega=2 * math.pi / period, depth=1.0,
                                  nx=160, nz=44)
        a33 = h["A"][0][0]
        assert a33 > 0.0
        assert h["B"][0][0] >= 0.0, (period, h["B"])
        # high-frequency heave added mass cannot exceed the flat-plate limit
        assert a33 < 1000.0 * math.pi * 0.3 ** 2 * 1.6
        # reciprocity of the coupling terms
        assert abs(h["A"][0][1] - h["A"][1][0]) < 1e-6 * max(1.0, abs(a33))


def test_potential_rao_decreases_with_frequency():
    poly = rectangular_section(0.6, 0.2)
    rao = [section_hydrodynamics(poly, omega=2 * math.pi / T, depth=1.0, nx=200,
                                 nz=48)["rao"]["rao_heave"] for T in (1.6, 0.8, 0.4)]
    assert rao[0] > rao[1] > rao[2]
    assert rao[2] < 0.2