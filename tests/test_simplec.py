"""SIMPLEC pressure-velocity coupling tests (analytic developed channel)."""

import unittest

import numpy as np
import torch

from tensorfvm import SimpleSolver, SolverConfig
from tensorfvm.verification.cases import channel


class SimplecConfigTests(unittest.TestCase):
    def test_default_is_simple(self):
        self.assertEqual(SolverConfig().pressure_velocity_algorithm, "simple")

    def test_accepts_declared_algorithms(self):
        for name in ("simple", "simplec"):
            c = SolverConfig(pressure_velocity_algorithm=name)
            self.assertEqual(c.pressure_velocity_algorithm, name)

    def test_rejects_other_algorithms(self):
        for name in ("piso", "pimple", "", None, 3):
            with self.subTest(name=name), self.assertRaises(ValueError):
                SolverConfig(pressure_velocity_algorithm=name)


class SimplecMobilityTests(unittest.TestCase):
    """The mobility must be A/max(aP/alpha - sum(aN), 0.1 aP), not A/(aP/alpha)."""

    def solver(self, algorithm):
        return SimpleSolver(SolverConfig(
            nx=24, ny=8, length=.12, height=.02, inlet_velocity=.0005,
            density=1000., reynolds=1., cylinder_radius=None, tolerance=1e-6,
            pressure_velocity_algorithm=algorithm))

    def test_simplec_mobility_uses_neighbour_coefficients(self):
        simple, simplec = self.solver("simple"), self.solver("simplec")
        diagonal, coeffs, _, active, width = simplec._momentum("u")
        relaxed = diagonal / simplec.config.velocity_relaxation
        floor = simplec.simplec_floor_fraction * diagonal
        # The pressure-correction mobility is scaled by the transverse spacing
        # of the component (dy for u), not by the returned momentum width.
        distance_area = simplec.config.height / simplec.config.ny
        expected = torch.where(
            active, distance_area / torch.maximum(relaxed - sum(coeffs), floor), 0.)
        actual = simplec._predict("u")
        delta = float(torch.max(torch.abs(actual - expected)))
        self.assertLess(delta, 1e-15)
        self.assertEqual(simplec.simplec_limited_dofs, 0)
        # The SIMPLEC mobility is strictly larger than the SIMPLE one wherever
        # neighbour coupling is non-zero and the floor does not bind.
        plain = simple._predict("u")
        mask = active & (relaxed - sum(coeffs) > floor)
        ratio = (actual[mask] / plain[mask])
        self.assertTrue(bool(torch.all(ratio > 1.0)))
        reference = ((relaxed / (relaxed - sum(coeffs)))[mask])
        self.assertLess(float(torch.max(torch.abs(ratio - reference))), 1e-12)


class SimplecChannelTests(unittest.TestCase):
    """Analytic Poiseuille gates, fixed point and alpha_p=1 stability."""

    def test_simplec_keeps_every_analytic_gate(self):
        result = channel(12, 2000, algorithm="simplec", velocity_relaxation=0.8,
                         pressure_relaxation=1.0)
        self.assertTrue(result["converged"])
        self.assertTrue(result["passed"], result["metrics"])
        metrics = {m["name"]: m for m in result["metrics"]}
        self.assertLess(metrics["velocity_l2"]["value"], 0.03)
        self.assertLess(metrics["relative_mass_imbalance"]["value"], 1e-6)
        self.assertLess(metrics["scaled_divergence"]["value"], 1e-6)
        self.assertLess(metrics["force_balance"]["value"], 0.03)
        self.assertEqual(result["simplec_limited_dofs"], 0)

    def test_simplec_reaches_the_simple_fixed_point(self):
        plain = channel(12, 2000, algorithm="simple")
        simplec = channel(12, 2000, algorithm="simplec",
                          velocity_relaxation=0.8, pressure_relaxation=1.0)
        self.assertTrue(plain["passed"])
        self.assertTrue(simplec["passed"])
        scale = plain["config"]["inlet_velocity"]
        for name in ("u_m_s", "v_m_s"):
            a = np.asarray(plain["fields"][name])
            b = np.asarray(simplec["fields"][name])
            self.assertLess(float(np.max(np.abs(a - b))) / scale, 1e-4)

    def test_simplec_sustains_pressure_relaxation_one(self):
        with self.assertRaises(RuntimeError):
            channel(8, 400, algorithm="simple", pressure_relaxation=1.0)
        result = channel(8, 400, algorithm="simplec", pressure_relaxation=1.0)
        self.assertTrue(result["converged"])


if __name__ == "__main__":
    unittest.main()