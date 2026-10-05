"""Numerical and boundary regression tests, using only unittest and torch."""

import math
import unittest
from unittest.mock import patch

import torch

from tensorfvm import SimpleSolver, SolverConfig


class SolverTests(unittest.TestCase):
    def cylinder_config(self, **kwargs):
        options = dict(nx=24, ny=12, cylinder_radius=0.25, reynolds=5,
                       max_iterations=300, tolerance=1e-5)
        options.update(kwargs)
        return SolverConfig(**options)

    def test_invalid_configuration(self):
        invalid = [
            {"nx": 3}, {"ny": 2}, {"nx": 8.5}, {"ny": True},
            {"max_iterations": 0}, {"max_iterations": 2.5},
            {"length": 0}, {"height": -1}, {"inlet_velocity": 0},
            {"reynolds": -1}, {"density": 0}, {"tolerance": 0},
            {"reynolds": math.nan}, {"height": math.inf},
            {"cylinder_x": math.nan}, {"cylinder_y": math.inf},
            {"cylinder_radius": -0.1}, {"cylinder_radius": math.inf},
            {"cylinder_x": 0.2}, {"cylinder_y": 0.2},
            {"cylinder_radius": 1e-8},
            {"velocity_relaxation": 0}, {"velocity_relaxation": 1.01},
            {"pressure_relaxation": math.nan}, {"pressure_relaxation": -0.1},
            {"pseudo_time_step": 0}, {"pseudo_time_step": math.inf},
            {"device": "not-a-device"}, {"density": 1e308, "inlet_velocity": 1e308},
        ]
        for options in invalid:
            with self.subTest(options=options), self.assertRaises(ValueError):
                SolverConfig(**options)

    def test_viscosity_reference_length(self):
        self.assertAlmostEqual(SolverConfig().viscosity, 0.02)
        for radius in (None, 0):
            c = SolverConfig(cylinder_radius=radius)
            self.assertAlmostEqual(c.viscosity, 0.1)

    def test_shapes_dtype_and_result_snapshot(self):
        solver = SimpleSolver(self.cylinder_config(max_iterations=1))
        result = solver.solve()
        self.assertEqual(result.u.shape, (12, 25))
        self.assertEqual(result.v.shape, (13, 24))
        self.assertEqual(result.p.shape, (12, 24))
        self.assertEqual(result.fluid.shape, result.p.shape)
        self.assertEqual(result.fluid.dtype, torch.bool)
        for field in (result.u, result.v, result.p, result.x, result.y):
            self.assertEqual(field.dtype, torch.float64)
            self.assertEqual(field.device.type, "cpu")
        self.assertEqual(result.x.shape, (24,))
        self.assertEqual(result.y.shape, (12,))
        uc, vc = result.cell_center_velocity()
        self.assertEqual(uc.shape, result.p.shape)
        self.assertEqual(vc.shape, result.p.shape)
        self.assertTrue(torch.equal(uc[~result.fluid], torch.zeros_like(uc[~result.fluid])))
        self.assertTrue(torch.equal(vc[~result.fluid], torch.zeros_like(vc[~result.fluid])))
        snapshot = result.u.clone()
        solver.step()
        self.assertTrue(torch.equal(snapshot, result.u))
        self.assertEqual(len(result.history), 1)
        self.assertFalse(result.converged)

    def test_walls_and_solid_normal_faces(self):
        solver = SimpleSolver(self.cylinder_config())
        for _ in range(3):
            solver.step()
        self.assertTrue(torch.equal(solver.u[:, 0], torch.ones(12, dtype=torch.float64)))
        self.assertEqual(torch.count_nonzero(solver.v[0]).item(), 0)
        self.assertEqual(torch.count_nonzero(solver.v[-1]).item(), 0)
        u_blocked = ~(solver.fluid[:, :-1] & solver.fluid[:, 1:])
        v_blocked = ~(solver.fluid[:-1] & solver.fluid[1:])
        self.assertEqual(torch.count_nonzero(solver.u[:, 1:-1][u_blocked]).item(), 0)
        self.assertEqual(torch.count_nonzero(solver.v[1:-1][v_blocked]).item(), 0)
        self.assertEqual(torch.count_nonzero(solver.p[~solver.fluid]).item(), 0)

    def test_tangential_no_slip_uses_half_cell_wall_distance(self):
        solver = SimpleSolver(self.cylinder_config())
        solver.u.zero_()
        diagonal, coeffs, _, active, _ = solver._momentum("u")
        dx, dy, mu = solver.dx, solver.dy, solver.config.viscosity
        opened = solver._u_open
        north, south = torch.zeros_like(opened), torch.zeros_like(opened)
        north[:-1] = opened[1:]
        south[1:] = opened[:-1]
        wall_faces = active & (~north | ~south)
        wall_faces[:, -1] = False
        self.assertGreater(torch.count_nonzero(wall_faces).item(), 0)
        expected = (2 * mu * dy / dx + mu * dx / dy
                    * (2 + (~north).to(torch.float64) + (~south).to(torch.float64)))
        self.assertTrue(torch.allclose(diagonal[wall_faces], expected[wall_faces]))
        self.assertEqual(torch.count_nonzero(coeffs[2][active & ~north]).item(), 0)
        self.assertEqual(torch.count_nonzero(coeffs[3][active & ~south]).item(), 0)

    def test_pressure_correction_retains_outlet_flux_and_conserves_mass(self):
        solver = SimpleSolver(self.cylinder_config())
        solver.u.zero_()
        original = solver._pressure_correction
        captured = {}

        def capture(du, dv):
            correction = original(du, dv)
            captured["predicted"] = solver.u[:, -1].clone()
            captured["delta"] = du[:, -1] * correction[:, -1]
            return correction

        with patch.object(solver, "_pressure_correction", side_effect=capture):
            metrics = solver.step()
        self.assertGreater(torch.max(torch.abs(captured["delta"])).item(), 0.1)
        self.assertTrue(torch.allclose(
            solver.u[:, -1], captured["predicted"] + captured["delta"], atol=1e-12))
        self.assertLess(metrics["continuity"], 1e-8)
        self.assertLess(metrics["mass_imbalance"], 1e-9)
        inlet = solver.u[:, 0].sum() * solver.dy
        outlet = solver.u[:, -1].sum() * solver.dy
        self.assertAlmostEqual(inlet.item(), outlet.item(), places=9)

    def test_zero_divergence_requires_no_pressure_correction(self):
        # Absolute pressure is anchored at the outlet: constant pressure shifts
        # are not a symmetry.  A divergence-free predictor has zero correction.
        solver = SimpleSolver(self.cylinder_config(cylinder_radius=None))
        du = solver._u_active.to(torch.float64)
        dv = solver._v_open.to(torch.float64)
        self.assertEqual(torch.count_nonzero(solver._divergence()).item(), 0)
        q = solver._pressure_correction(du, dv)
        self.assertEqual(torch.count_nonzero(q).item(), 0)

    def test_small_relaxation_cannot_falsely_converge(self):
        solver = SimpleSolver(self.cylinder_config(
            cylinder_radius=None, velocity_relaxation=1e-8))
        metrics = solver.step()
        self.assertLess(metrics["velocity_change"], solver.config.tolerance)
        self.assertGreater(metrics["momentum"], 0.1)
        self.assertFalse(solver.converged)

    def test_channel_profile_refinement(self):
        errors = []
        for ny in (8, 16):
            solver = SimpleSolver(SolverConfig(
                nx=3 * ny, ny=ny, length=6, height=1, cylinder_radius=None,
                reynolds=1, tolerance=1e-6, max_iterations=300))
            result = solver.solve()
            self.assertTrue(result.converged, result.history[-1])
            exact = 6 * result.y * (1 - result.y)
            error = (torch.linalg.vector_norm(result.u[:, -2] - exact)
                     / torch.linalg.vector_norm(exact)).item()
            errors.append(error)
            self.assertGreater(result.p[:, 0].mean().item(),
                               result.p[:, -1].mean().item())
            self.assertLess(torch.max(torch.abs(result.v[:, -2])).item(), 1e-4)
        self.assertLess(errors[0], 0.02)
        self.assertLess(errors[1], errors[0] / 3)

    def test_low_re_cylinder_really_converges(self):
        solver = SimpleSolver(self.cylinder_config())
        result = solver.solve()
        self.assertTrue(result.converged, result.history[-1])
        self.assertGreater(len(result.history), 5)
        for field in (result.u, result.v, result.p):
            self.assertTrue(torch.isfinite(field).all().item())
        metrics = result.history[-1]
        for name in ("continuity", "momentum", "mass_imbalance"):
            self.assertLess(metrics[name], result.config.tolerance)
        self.assertLess(metrics["momentum"], result.history[0]["momentum"] * 1e-3)
        self.assertGreater(result.u.max().item(), result.config.inlet_velocity)
        self.assertGreater(result.v.abs().max().item(), 0.1)

    def test_pseudo_time_reaches_same_steady_solution(self):
        steady = SimpleSolver(self.cylinder_config()).solve()
        transient = SimpleSolver(self.cylinder_config(pseudo_time_step=0.05)).solve()
        self.assertTrue(transient.converged, transient.history[-1])
        self.assertLess(torch.max(torch.abs(steady.u - transient.u)).item(), 2e-4)
        self.assertLess(torch.max(torch.abs(steady.v - transient.v)).item(), 2e-4)


if __name__ == "__main__":
    unittest.main()
