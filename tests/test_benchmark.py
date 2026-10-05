"""Analytical benchmark acceptance and plot regression tests."""

import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import torch

from tensorfvm import SimpleSolver, SolverConfig
from tensorfvm.benchmark import compare_poiseuille, main, run_benchmark, write_contour


class BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.result = SimpleSolver(SolverConfig(
            nx=36, ny=12, length=6, height=1, cylinder_radius=None,
            reynolds=1, tolerance=1e-6, max_iterations=300,
        )).solve()

    def test_computed_solution_passes(self):
        metrics = compare_poiseuille(self.result)
        self.assertTrue(metrics["passed"], metrics)
        for error in metrics["errors"].values():
            self.assertLess(error, 0.03)
        self.assertAlmostEqual(metrics["reference_pressure_gradient"], -12)

    def test_unconverged_result_cannot_pass(self):
        with patch.object(self.result, "converged", False):
            self.assertFalse(compare_poiseuille(self.result)["passed"])

    def test_coarse_pressure_gradient_fails_three_percent(self):
        result = SimpleSolver(SolverConfig(
            nx=24, ny=8, length=6, height=1, cylinder_radius=None,
            reynolds=1, tolerance=1e-6, max_iterations=300,
        )).solve()
        metrics = compare_poiseuille(result)
        self.assertTrue(metrics["converged"])
        self.assertGreater(metrics["errors"]["pressure_gradient"], 0.03)
        self.assertFalse(metrics["passed"])

    def test_threshold_is_strict(self):
        largest_error = max(compare_poiseuille(self.result)["errors"].values())
        with patch("tensorfvm.benchmark.ERROR_LIMIT", largest_error):
            self.assertFalse(compare_poiseuille(self.result)["passed"])
        with patch("tensorfvm.benchmark.ERROR_LIMIT",
                   math.nextafter(largest_error, math.inf)):
            self.assertTrue(compare_poiseuille(self.result)["passed"])

    def test_wrong_geometry_is_rejected(self):
        with patch.object(self.result.config, "cylinder_radius", 0.2):
            with self.assertRaises(ValueError):
                compare_poiseuille(self.result)

    def test_inaccurate_solution_cannot_pass(self):
        with patch.object(self.result, "u", self.result.u * 1.1):
            self.assertFalse(compare_poiseuille(self.result)["passed"])

    def test_nonfinite_solution_cannot_pass(self):
        with patch.object(self.result, "v", torch.full_like(self.result.v, float("nan"))):
            self.assertFalse(compare_poiseuille(self.result)["passed"])

    def test_contour_is_valid_svg(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plot.svg"
            write_contour(self.result, self.result.p, "Computed pressure <p>", path)
            root = ET.parse(path).getroot()
            self.assertEqual(root.tag, "{http://www.w3.org/2000/svg}svg")
            self.assertIn("Computed pressure", path.read_text())
            self.assertIn("&lt;p&gt;", path.read_text())

    def test_failed_run_preserves_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = run_benchmark(tmp, max_iterations=1)
            self.assertFalse(summary["passed"])
            self.assertEqual(len(summary["cases"]), 3)
            stored = json.loads((Path(tmp) / "benchmark.json").read_text())
            self.assertFalse(stored["passed"])
            for name in ("report.rst", "velocity.svg", "pressure.svg", "profiles.csv",
                         "144x48/fields.csv", "144x48/history.json"):
                self.assertTrue((Path(tmp) / name).is_file())

    def test_cli_failure_status(self):
        with tempfile.TemporaryDirectory() as tmp, patch("builtins.print"):
            self.assertEqual(main(["--output", tmp, "--max-iterations", "1"]), 2)


if __name__ == "__main__":
    unittest.main()
