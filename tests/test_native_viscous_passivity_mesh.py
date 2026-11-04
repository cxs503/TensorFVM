"""Mesh-level passivity of the cell-centred viscous operator.

The reference values are closed form: on a uniform Cartesian mesh the scalar
flux operator must reproduce the analytic 7-point stencil (``mu*h`` on internal
faces, ``2*mu*h`` on Dirichlet faces), so the audit tool is graded against
discrete mathematics rather than against its own diagnostics.  The reported
negative margins are the reproducible counterpart of the open SUBOFF audit item
(``u^T A u = -0.2458`` on a mesh whose artifacts were removed).
"""
import unittest

import numpy as np

from tensorfvm.verification.native_viscous_passivity import (
    analytic_anchor, compact, counterexample_sweep, creation_threshold, layered_mesh,
    mosaic_pairs, passivity_margin, seven_point_flux_operator, uniform_cartesian)


class AnalyticAnchor(unittest.TestCase):
    def test_scalar_operator_equals_seven_point_flux_operator(self):
        for n in (2, 3, 4):
            anchor = analytic_anchor(n)
            self.assertLess(anchor['maximum_absolute_matrix_difference'], 1e-13)
            self.assertLess(anchor['eigenvalue_relative_difference'], 1e-13)

    def test_reference_operator_is_a_positive_laplacian(self):
        matrix = seven_point_flux_operator(3).toarray()
        self.assertTrue(np.allclose(matrix, matrix.T))
        self.assertGreater(np.linalg.eigvalsh(matrix)[0], 0.)


class StructuredFamilies(unittest.TestCase):
    """Tetrahedral transitions, prisms, clustering and shear: all passive."""

    def test_layered_families_stay_passive(self):
        for mode in ('hex', 'hybrid', 'wedge'):
            for ratio in (1., .02):
                audit = passivity_margin(layered_mesh(2, 2, 6, ratio, .8, mode))
                self.assertGreater(audit['margin'], 0., f"{mode} ratio={ratio}")
                self.assertLess(audit['eigen_residual'], 1e-8)

    def test_transition_geometry_is_not_the_cause(self):
        # The asymmetry of the operator grows with tetrahedral interfaces ...
        plain = passivity_margin(layered_mesh(2, 2, 6, .1, .3, 'hex'))
        hybrid = passivity_margin(layered_mesh(2, 2, 6, .1, .3, 'hybrid'))
        self.assertGreater(hybrid['asymmetry_ratio'], 4*plain['asymmetry_ratio'])
        # ... but neither creates energy.
        self.assertGreater(hybrid['margin'], 0.)
        self.assertGreater(plain['margin'], 0.)


class CounterexampleFamily(unittest.TestCase):
    """The two-cell counterexample copied into a mesh: sign change is local."""

    @classmethod
    def setUpClass(cls):
        cls.passive = passivity_margin(mosaic_pairs(4, .6))
        cls.creating = passivity_margin(mosaic_pairs(4, 1.))
        cls.worse = passivity_margin(mosaic_pairs(1, 1.1))

    def test_margin_changes_sign_with_family_parameter(self):
        self.assertGreater(self.passive['margin'], 0.)
        self.assertLess(self.creating['margin'], 0.)
        self.assertLess(self.worse['margin'], self.creating['margin'])

    def test_energy_creation_is_local_not_cumulative(self):
        single = passivity_margin(mosaic_pairs(1, 1.))
        many = passivity_margin(mosaic_pairs(8, 1.))
        self.assertAlmostEqual(single['margin'], many['margin'], places=12)

    def test_creating_faces_are_identified_and_explain_the_energy(self):
        audit = self.creating
        contributions = audit['per_face']
        self.assertGreater(audit['creating_faces'], 0)
        self.assertAlmostEqual(float(contributions.sum()),
                               audit['quadratic_form_energy'], places=10)
        # The whole negative energy comes from a minority of the faces.
        self.assertLess(audit['creating_faces'], audit['faces'])

    def test_threshold_matches_the_two_cell_boundary(self):
        threshold = creation_threshold()
        self.assertGreater(threshold['passive_skew'], .8)
        self.assertLess(threshold['creating_skew'], 1.05)
        self.assertGreater(threshold['creating_degrees'], 80.)


class Reporting(unittest.TestCase):
    def test_compact_view_is_json_safe(self):
        import json
        payload = json.dumps(compact(passivity_margin(mosaic_pairs(1, .6))))
        self.assertIn('margin', payload)


if __name__ == '__main__':
    unittest.main()