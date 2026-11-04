"""Verification debt closure: ``gauss-skew-corrected`` conservative pressure reconstruction.

The references are analytic (the exact gradient and the exact face value of a
prescribed linear field, the exact gradient of a smooth field), so the scheme is
never graded against its own diagnostics.

The mesh family carries the point of the audit: a structured quad mesh is made of
*parallelograms*, whose shared-face centroid lies on the segment joining the two
cell centroids -- there the skew displacement and the whole correction are
identically zero.  Genuine skewness needs non-parallelogram cells, produced here
by a deterministic interior-node jitter (25 % of the local spacing).
"""
import tempfile
import unittest

import torch

from tensorfvm.verification.multi_element_skew import (face_reconstruction_exactness,
                                                       force_conservation, gradient_error,
                                                       operator_consistency, order_study,
                                                       solver_for, solver_step_defect,
                                                       wall_normal_removal)


class SkewPressureVerification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        cls.directory = cls.temp.name
        # 192 cells, wall-to-far-field spacing ratio exp(6) and 25 % node jitter:
        # cheap, strongly clustered and genuinely non-parallelogram.
        cls.skewed = solver_for(cls.directory, 24, 8, 6.0, jitter=0.25)
        cls.smooth_mesh = solver_for(cls.directory, 24, 8, 6.0, jitter=0.0)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_linear_pressure_is_reconstructed_exactly_at_face_centroids(self):
        measured = face_reconstruction_exactness(self.skewed)
        self.assertGreater(measured["clean_faces"], 200, measured)
        self.assertGreater(measured["max_interior_skew"], 1e-3, measured)
        # Skew-corrected reconstruction: machine precision.
        self.assertLess(measured["skew_error"], 1e-13, measured)
        # Plain projected interpolation of the same solver: second-order face error.
        self.assertGreater(measured["projected_error"], 1e-3, measured)
        self.assertLess(measured["skew_error"], 1e-8 * measured["projected_error"], measured)

    def test_correction_is_inert_on_parallelogram_cells(self):
        measured = face_reconstruction_exactness(self.smooth_mesh)
        self.assertLess(measured["max_interior_skew"], 1e-12, measured)
        self.assertLess(measured["max_skew"], 1e-12, measured)
        # With a vanishing displacement the two schemes are the same operator.
        self.assertAlmostEqual(measured["skew_error"], measured["projected_error"], places=14)

    def test_boundary_inconsistent_gradient_can_leak_through_the_correction(self):
        """Documents the mechanism, not a rate: the outlet gauge is O(1) wrong for a
        non-vanishing linear test field, and only the correction can import it."""
        measured = face_reconstruction_exactness(self.skewed)
        self.assertGreater(measured["leakage"], measured["projected_error"], measured)
        self.assertLessEqual(measured["all_interior_skew_error"],
                             measured["leakage"] + measured["skew_error"], measured)

    def test_wall_faces_keep_only_the_tangential_skew_displacement(self):
        measured = wall_normal_removal(self.skewed)
        self.assertGreater(measured["wall_faces"], 0, measured)
        self.assertEqual(measured["wall_normal_component"], 0.0, measured)
        self.assertGreater(measured["wall_tangential_component"], 1e-4, measured)
        self.assertEqual(measured["outlet_skew"], 0.0, measured)

    def test_internal_pressure_forces_cancel_to_boundary_flux(self):
        measured = force_conservation(self.skewed)
        self.assertGreater(measured["internal_flux_scale"], 1.0, measured)
        self.assertLess(measured["identity_defect"], 1e-12, measured)
        self.assertLess(measured["relative_defect"], 1e-13, measured)

    def test_assembled_operator_reproduces_the_non_linear_gradient(self):
        measured = operator_consistency(self.skewed)
        self.assertGreater(measured["operator_scaling"], 1e-2, measured)
        self.assertLess(measured["gradient_difference"], 1e-11, measured)
        self.assertLess(measured["quadratic_difference"], 1e-11, measured)
        self.assertEqual(measured["h_row_defect"], 0.0, measured)
        self.assertEqual(measured["i_row_defect"], 0.0, measured)
        self.assertGreater(measured["gauge_cells"], 0, measured)

    def test_scheme_is_convergent_where_the_projected_interpolation_stagnates(self):
        families = order_study(self.directory, levels=(8, 16, 32), jitters=(0.25,))
        rows = families["shear0_stretch6_jitter0.25"]
        self.assertGreaterEqual(len(rows), 3)
        for previous, row in zip(rows, rows[1:]):
            # Skew-corrected: first order on self-similar jittered meshes ...
            self.assertGreater(row["skew_order"], 0.8, row)
            self.assertLess(row["skew"], previous["skew"], row)
            # ... while the projected two-point interpolation does not converge at all.
            self.assertLess(abs(row["projected_order"]), 0.3, row)
        # Parallelogram family: the two schemes coincide because the correction is zero.
        flat = order_study(self.directory, levels=(8, 16), jitters=(0.0,))
        for row in flat["shear0_stretch6_jitter0"]:
            self.assertAlmostEqual(row["skew"], row["projected"], places=14)

    def test_one_coupled_step_keeps_the_pressure_operator_gate(self):
        measured = solver_step_defect(self.directory, nx=6, ny=4, jitter=0.25)
        self.assertEqual(measured["scheme"], "gauss-skew-corrected")
        self.assertLess(measured["operator_defect"], 1e-10, measured)
        self.assertTrue(all(abs(value) < 1e6 for value in measured["force"]), measured)


if __name__ == "__main__":
    unittest.main()