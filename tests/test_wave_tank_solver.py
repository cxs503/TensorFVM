"""Geometry pre-gate and the tank solver's structural guarantees."""
from __future__ import annotations

import unittest

import numpy as np

from tensorfvm.wave_tank.geometry import (BargeSection, KCSSimplifiedHull, WigleyHull,
                                          geometry_gate, polygon_mask)
from tensorfvm.wave_tank.solver import RigidBody2D, TankConfig, WaveTank


class TestGeometryGate(unittest.TestCase):
    def test_wigley_block_coefficient(self):
        rec = geometry_gate(WigleyHull().coefficients())
        self.assertTrue(rec["passed"])
        self.assertAlmostEqual(rec["Cb"], 4.0 / 9.0, places=3)
        self.assertAlmostEqual(rec["nabla_LBT"], rec["Cb"], places=9)

    def test_kcs_simplified_block_coefficient(self):
        rec = geometry_gate(KCSSimplifiedHull().coefficients())
        self.assertTrue(rec["passed"])
        self.assertAlmostEqual(rec["Cb"], 0.651, places=2)
        self.assertLess(abs(rec["L_over_B"] - 230.0 / 32.2), 1e-9)

    def test_gate_rejects_wrong_coefficient(self):
        rec = geometry_gate(dict(WigleyHull().coefficients(), Cb=0.6))
        self.assertFalse(rec["passed"])

    def test_strip_section(self):
        sec = BargeSection(length=2.4, draft=0.3)
        self.assertAlmostEqual(sec.area, 0.72)
        self.assertAlmostEqual(sec.coefficients()["L_over_T"], 8.0)

    def test_polygon_mask_area(self):
        xc = np.linspace(0.05, 1.95, 20)
        yc = np.linspace(0.05, 0.95, 10)
        poly = np.array([[0.5, 0.2], [1.5, 0.2], [1.5, 0.8], [0.5, 0.8]])
        m = polygon_mask(xc, yc, poly)
        self.assertAlmostEqual(float(m.sum()) * 0.1 * 0.1, 1.0 * 0.6, places=9)

    def test_polygon_mask_rotation_is_inside(self):
        xc = np.linspace(-1, 1, 40)
        yc = np.linspace(-1, 1, 40)
        body = RigidBody2D(length=0.8, draft=0.2, x_centre=0.0)
        m0 = polygon_mask(xc, yc, body.polygon(heave=0.0, pitch=0.0))
        m1 = polygon_mask(xc, yc, body.polygon(heave=0.0, pitch=0.20))
        self.assertAlmostEqual(float(m0.sum()), float(m1.sum()), delta=0.05 * float(m0.sum()))


class TestTankStructure(unittest.TestCase):
    def _tank(self, **kw):
        cfg = TankConfig(length=4.0, height=1.0, depth=0.7, nx=48, ny=24,
                         cg_max_iter=120, cg_rtol=1e-5, **kw)
        return WaveTank(cfg)

    def test_initial_state_is_still_water(self):
        tank = self._tank()
        self.assertAlmostEqual(float(tank.surface_elevation().max()), 0.0, places=12)
        self.assertAlmostEqual(tank.liquid_volume(), 4.0 * 0.7, places=10)
        self.assertEqual(float(tank.u.abs().max()), 0.0)

    def test_hydrostatic_pool_stays_at_rest(self):
        """The incremental projection must hold a pool at rest (no free fall)."""
        tank = self._tank()
        for _ in range(12):
            tank.step(2.0e-3)
        self.assertLess(float(tank.u.abs().max()), 1e-6)
        self.assertLess(float(tank.v.abs().max()), 1e-6)
        self.assertLess(abs(float(tank.surface_elevation().max())), 1e-6)

    def test_hydrostatic_pressure_profile(self):
        tank = self._tank()
        tank.step(2.0e-3)
        col = tank.cfg.nx // 2
        for j in (2, 6, 10):
            expected = tank.cfg.rho * tank.cfg.g * (0.0 - tank.yc[j])
            self.assertAlmostEqual(float(tank.p[j, col]) / expected, 1.0, places=3)

    def test_relaxation_zone_imposes_the_target_wave(self):
        tank = self._tank()
        w = tank.set_wave(0.02, 0.8)
        for _ in range(40):
            tank.step(2.0e-3)
        eta = tank.surface_elevation()
        target = tank.target_fields(tank.t)[0]
        # at the wall (w = 1) the tank *is* the target wave
        self.assertAlmostEqual(float(eta[0]), float(target[0]), delta=2e-3)
        self.assertLess(float(eta.abs().max()), 4.0 * w["amp"])

    def test_volume_is_conserved_in_a_closed_tank(self):
        tank = self._tank(damp_strength=0.0, relax_wavelengths=0.0)
        tank.set_wave(0.02, 0.8)
        v0 = tank.liquid_volume()
        for _ in range(20):
            tank.step(2.0e-3)
        self.assertLess(abs(tank.liquid_volume() - v0) / v0, 5e-3)

    def test_body_float_is_in_hydrostatic_equilibrium(self):
        cfg = TankConfig(length=4.0, height=1.0, depth=0.7, nx=60, ny=30,
                         cg_max_iter=150, cg_rtol=1e-5)
        body = RigidBody2D(length=0.8, draft=0.2, x_centre=2.0)
        tank = WaveTank(cfg, body=body)
        self.assertAlmostEqual(body.mass, cfg.rho * body.length * body.draft, places=9)
        _, fy, _ = tank._body_loads(tank.p)
        # stair-step hull: the wetted-face pressure is extrapolated from the cell
        # centre, which leaves a documented O(dy/(2T)) buoyancy bias (8 % here,
        # dy/T = 1/6).  The tolerance is the measured bias, not a claim of 1 %.
        self.assertLess(abs(fy / (body.mass * cfg.g) - 1.0), 0.10)
        for _ in range(10):
            tank.step(2.0e-3)
        self.assertLess(abs(body.heave), 5.0e-2)
        self.assertLess(abs(body.heave_dot), 5e-2)

    def test_time_step_obeys_cfl(self):
        tank = self._tank()
        dt = tank.stable_dt()
        self.assertLessEqual(dt, tank.cfg.cfl * tank.dx / max(float(tank.u.abs().max()), 1e-6))


if __name__ == "__main__":
    unittest.main()
