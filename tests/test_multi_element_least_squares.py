"""Verification debt closure: shared least-squares pressure gradient on stretched grids.

These checks are independent of the solver's own diagnostics: the reference is the
analytic gradient of a prescribed field, evaluated at the solver's own cell
centres, so a wrong stencil cannot pass by agreeing with itself.
"""
import tempfile
import unittest

import numpy as np
import torch

from tensorfvm.verification.multi_element_ls import (linear_exactness, operator_consistency,
                                                    order_study, solver_for, solver_step_defect)


class LeastSquaresPressureVerification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        cls.directory = cls.temp.name
        # One strongly clustered mesh (exp(6) wall-to-far-field spacing jump) shared
        # by the operator-level checks; it is cheap (192 cells) and its median cell
        # aspect ratio exceeds four.
        cls.stretched = solver_for(cls.directory, 24, 8, 6.0)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_linear_field_gradient_is_exact_in_interior_cells(self):
        measured = linear_exactness(self.stretched)
        self.assertGreater(measured["median_aspect"], 4.0)
        self.assertGreater(measured["max_aspect"], 20.0)
        self.assertGreaterEqual(measured["interior_cells"], 8)
        self.assertLess(measured["interior_error"], 1e-12, measured)

    def test_assembled_geometry_gradient_equals_least_squares_gradient(self):
        measured = operator_consistency(self.stretched)
        self.assertGreater(measured["operator_scaling"], 1e-2)
        self.assertLess(measured["gradient_difference"], 1e-12, measured)
        self.assertLess(measured["h_row_defect"], 1e-12, measured)
        self.assertLess(measured["i_row_defect"], 1e-12, measured)
        self.assertLess(measured["quadratic_difference"], 1e-12, measured)

    def test_gradient_order_of_accuracy_is_second_order_under_clustering(self):
        families = order_study(self.directory, levels=(8, 16), stretches=(0.0, 6.0))
        for stretch, rows in families.items():
            orders = [row["observed_order"] for row in rows if "observed_order" in row]
            self.assertEqual(len(orders), 1)
            self.assertGreater(orders[0], 1.5, (stretch, rows))
        # Refinement must reduce the error in both families.
        for rows in families.values():
            self.assertLess(rows[-1]["error"], rows[0]["error"])

    def test_coupled_least_squares_solver_keeps_pressure_operator_gate(self):
        measured = solver_step_defect(self.directory)
        self.assertEqual(measured["scheme"], "least-squares")
        self.assertGreaterEqual(measured["pressure_solves"], 1)
        self.assertLess(measured["operator_defect"], 2e-12, measured)
        self.assertLess(measured["mass_imbalance"], 1e-3, measured)
        self.assertTrue(np.isfinite(measured["continuity"]), measured)


if __name__ == "__main__":
    unittest.main()