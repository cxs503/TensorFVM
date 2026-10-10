"""NACA 0012 quantitative acceptance regressions."""

import copy
import math
import unittest
from unittest.mock import patch

import torch

from tensorfvm import SimpleSolver, SolverConfig
from tensorfvm.benchmark_airfoil import ERROR_LIMIT, REFERENCE, compare_airfoil, main


class AirfoilBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.result = SimpleSolver(SolverConfig(
            mesh_type="c-grid", airfoil_code="0012", reynolds=1000,
            angle_of_attack=4, nx=32, ny=8, length=20, height=16,
            airfoil_x=6, airfoil_y=8, max_iterations=1,
        )).solve()

    def accepted_result(self):
        result = copy.copy(self.result)
        result.converged = True
        result.history = [{"continuity": 1e-8, "momentum": 1e-8,
                           "mass_imbalance": 1e-8}]
        result.aerodynamic_coefficients = {"lift": REFERENCE["lift_coefficient"], "drag": REFERENCE["drag_coefficient"]}
        return result

    def test_matching_coefficients_pass_independently(self):
        metrics = compare_airfoil(self.accepted_result())
        self.assertTrue(metrics["passed"])
        self.assertEqual(set(metrics["relative_errors"]), {"lift", "drag"})

    def test_each_metric_uses_strict_three_percent_limit(self):
        for name, reference in (("lift", REFERENCE["lift_coefficient"]), ("drag", REFERENCE["drag_coefficient"])):
            result = self.accepted_result()
            result.aerodynamic_coefficients[name] = reference * (1 + ERROR_LIMIT)
            with self.subTest(name=name):
                self.assertFalse(compare_airfoil(result)["passed"])

    def test_nonfinite_or_unconverged_result_cannot_pass(self):
        result = self.accepted_result()
        result.aerodynamic_coefficients["drag"] = math.nan
        self.assertFalse(compare_airfoil(result)["passed"])
        result = self.accepted_result()
        result.converged = False
        self.assertFalse(compare_airfoil(result)["passed"])

    def test_historic_graph_estimate_does_not_pass(self):
        result = self.accepted_result()
        result.aerodynamic_coefficients = {"lift": .2062249835, "drag": .1209791959}
        self.assertFalse(compare_airfoil(result)["passed"])

    def test_digitization_interval_is_used_conservatively(self):
        result = self.accepted_result()
        ref = REFERENCE["drag_coefficient"]
        result.aerodynamic_coefficients["drag"] = ref * 1.0298
        self.assertFalse(compare_airfoil(result)["passed"])

    def test_wrong_case_is_rejected(self):
        result = self.accepted_result()
        result.config = copy.copy(result.config)
        result.config.reynolds = 500
        with self.assertRaises(ValueError):
            compare_airfoil(result)

    def test_cli_returns_two_for_failed_acceptance(self):
        summary = {"passed": False}
        with patch("tensorfvm.benchmark_airfoil.run_benchmark", return_value=summary), \
                patch("builtins.print"):
            self.assertEqual(main([]), 2)


if __name__ == "__main__":
    unittest.main()
