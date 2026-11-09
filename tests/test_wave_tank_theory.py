"""Closed-form targets and analysis utilities of the numerical wave tank."""
from __future__ import annotations

import math
import unittest

import numpy as np

from tensorfvm.wave_tank import theory


class TestDispersion(unittest.TestCase):
    def test_deep_water_limit(self):
        omega = 2.0
        k = theory.dispersion_k(omega, depth=1000.0)
        self.assertAlmostEqual(k, omega ** 2 / theory.GRAVITY, places=6)

    def test_shallow_water_limit(self):
        omega, h = 0.05, 2.0
        k = theory.dispersion_k(omega, h)
        # kh -> 0: omega^2 = g k^2 h  =>  k = omega / sqrt(g h)
        self.assertAlmostEqual(k, omega / math.sqrt(theory.GRAVITY * h), delta=1e-6)

    def test_residual_is_zero(self):
        omega, h = 3.0, 0.8
        k = theory.dispersion_k(omega, h)
        self.assertAlmostEqual(theory.GRAVITY * k * math.tanh(k * h), omega ** 2, places=10)

    def test_phase_speed_limits(self):
        h = 0.5
        shallow = math.sqrt(theory.GRAVITY * h)
        self.assertAlmostEqual(theory.phase_speed(1e-4, h) / shallow, 1.0, places=5)
        # shallow water is the *fast* limit: c -> sqrt(g h) as kh -> 0
        self.assertLess(theory.phase_speed(20.0, h), shallow)


class TestKinematics(unittest.TestCase):
    def test_deep_water_velocity_decays(self):
        omega, k = 2.0, theory.dispersion_k(2.0, 1000.0)
        _, u0, _ = theory.airy_kinematics(0.0, 0.0, 0.0, 0.01, omega, k, 1000.0)
        _, u1, _ = theory.airy_kinematics(0.0, -1.0 / k, 0.0, 0.01, omega, k, 1000.0)
        self.assertAlmostEqual(float(u1 / u0), math.exp(-1.0), places=3)

    def test_bottom_no_penetration(self):
        omega, h = 2.0, 1.0
        k = theory.dispersion_k(omega, h)
        for t in (0.0, 0.3, 1.1):
            _, u, v = theory.airy_kinematics(0.7, -h, t, 0.02, omega, k, h)
            self.assertAlmostEqual(float(v), 0.0, places=12)

    def test_stokes2_deep_water_second_harmonic(self):
        omega, k, a = 2.0, 1.0, 0.05
        eta = theory.stokes2_elevation(0.0, 0.0, a, omega, k, 1000.0)
        self.assertAlmostEqual(float(eta), a + 0.5 * k * a * a, places=10)

    def test_airy_elevation_range(self):
        omega, k, a = 2.0, 1.0, 0.01
        t = np.linspace(0, 2 * math.pi / omega, 100)
        eta = [float(theory.stokes2_elevation(0.3, tt, a, omega, k, 5.0, order2=False))
               for tt in t]
        self.assertAlmostEqual(max(eta), a, delta=1e-3 * a)
        self.assertAlmostEqual(min(eta), -a, delta=1e-3 * a)


class TestAnalysis(unittest.TestCase):
    def test_harmonic_fit_recovers_amplitude_and_phase(self):
        omega = 2.0
        t = np.linspace(0, 20 * math.pi / omega, 2000)
        s = 0.03 * np.cos(omega * t - 0.7) + 0.004
        amp, phase, mean, resid = theory.harmonic_fit(t, s, omega)
        self.assertAlmostEqual(amp, 0.03, places=6)
        self.assertAlmostEqual(mean, 0.004, places=6)
        self.assertLess(resid, 1e-8)

    def test_goda_recovers_known_reflection(self):
        k = 2.0
        x1, x2 = 0.4, 1.6
        aI, aR = 0.02 + 0j, 0.006j
        h1 = aI * np.exp(1j * k * x1) + aR * np.exp(-1j * k * x1)
        h2 = aI * np.exp(1j * k * x2) + aR * np.exp(-1j * k * x2)
        AI, AR = theory.goda_two_point(x1, x2, complex(h1), complex(h2), k)
        self.assertAlmostEqual(abs(AI), 0.02, places=10)
        self.assertAlmostEqual(abs(AR), 0.006, places=10)

    def test_goda_rejects_singular_spacing(self):
        with self.assertRaises(ValueError):
            theory.goda_two_point(0.0, math.pi, 1 + 0j, 1 + 0j, 1.0)

    def test_three_probe_wavenumber(self):
        k = 3.0
        x = np.array([0.5, 0.9, 1.3])
        phase = k * x + 0.2
        k_fit, _ = theory.three_probe_wavenumber(x, np.ones(3), phase)
        self.assertAlmostEqual(k_fit, k, places=10)

    def test_added_resistance_period_mean(self):
        period, n = 1.0, 40
        t = np.linspace(0, n * period, 4000)
        f = -12.5 + 3.0 * np.cos(2 * math.pi * t / period)
        raw, se = theory.added_resistance_energy(f, t, period)
        self.assertAlmostEqual(raw, 12.5, places=6)
        self.assertLess(se, 1e-6)

    def test_rao_normalisation(self):
        omega = 2.0
        t = np.linspace(0, 20 * math.pi / omega, 4000)
        sig = 0.02 * np.cos(omega * t)
        rao, _, _, _ = theory.rao_from_series(t, sig, omega, 0.01)
        self.assertAlmostEqual(rao, 2.0, places=6)


if __name__ == "__main__":
    unittest.main()
