"""Geometry and conservative collocated SIMPLE regressions."""

import unittest
from unittest.mock import patch

import torch

from tensorfvm.solver import SolverConfig
from tensorfvm.body_fitted import BodyFittedMesh, BodyFittedSolver


def config(**kwargs):
    options = dict(mesh_type="body-fitted", nx=16, ny=4, reynolds=1,
                   max_iterations=300, tolerance=1e-5)
    options.update(kwargs)
    return SolverConfig(**options)


class MeshTests(unittest.TestCase):
    def test_exact_rectangle_corners_and_periodic_seam(self):
        c = config(cylinder_x=0.7, cylinder_y=0.8, nx=24, ny=7)
        mesh = BodyFittedMesh(c)
        self.assertEqual(mesh.vertices.shape, (8, 25, 2))
        self.assertEqual(mesh.centers.shape, (7, 24, 2))
        self.assertTrue(torch.equal(mesh.vertices[:, 0], mesh.vertices[:, -1]))
        corners = torch.tensor([[4., 2.], [0., 2.], [0., 0.], [4., 0.]],
                               dtype=torch.float64)
        self.assertTrue(torch.equal(mesh.vertices[-1, ::6][:-1], corners))
        outer = mesh.vertices[-1]
        on_rectangle = ((outer[:, 0].abs() < 1e-12)
                        | ((outer[:, 0] - c.length).abs() < 1e-12)
                        | (outer[:, 1].abs() < 1e-12)
                        | ((outer[:, 1] - c.height).abs() < 1e-12))
        self.assertTrue(on_rectangle.all())
        center = torch.tensor([c.cylinder_x, c.cylinder_y], dtype=torch.float64)
        self.assertTrue(torch.allclose(
            torch.linalg.vector_norm(mesh.vertices[0] - center, dim=-1),
            torch.full((25,), c.cylinder_radius, dtype=torch.float64)))

    def test_polygon_area_positive_and_global_closure(self):
        c = config(cylinder_x=0.8, cylinder_y=1.2)
        m = BodyFittedMesh(c)
        inner = m.vertices[0, :-1]
        following = inner.roll(-1, 0)
        excluded = ((inner[:, 0] * following[:, 1]
                     - inner[:, 1] * following[:, 0]).sum() / 2)
        self.assertTrue((m.volumes > 0).all())
        self.assertAlmostEqual(float(m.volumes.sum()), c.length * c.height
                               - float(excluded), places=12)
        self.assertLess(float(m.face_area_vectors[m.boundary].sum(0).abs().max()), 1e-12)

    def test_cell_closure_face_normals_and_neighbors(self):
        s = BodyFittedSolver(config())
        m = s.mesh
        self.assertLess(float(s._sum(m.face_area_vectors).abs().max()), 1e-12)
        self.assertTrue(torch.allclose(torch.linalg.vector_norm(m.face_normals, dim=-1),
                                       torch.ones_like(m.face_lengths)))
        direction = m.face_centers - m.centers.reshape(-1, 2)[m.owner]
        self.assertTrue(((direction * m.face_normals).sum(-1) > 0).all())
        self.assertTrue((m.neighbor[m.interior] != m.owner[m.interior]).all())
        expected_faces = 2 * s.count + s.config.nx
        self.assertEqual(len(m.owner), expected_faces)
        self.assertEqual(int(m.boundary.sum()), 2 * s.config.nx)

    def test_boundary_partition_and_lengths(self):
        c = config(nx=32)
        m = BodyFittedMesh(c)
        for name, total in (("inlet", c.height), ("outlet", c.height),
                            ("wall", 2 * c.length)):
            self.assertAlmostEqual(float(m.face_lengths[m.masks[name]].sum()),
                                   total, places=12)
        self.assertEqual(int(m.masks["cylinder"].sum()), c.nx)
        self.assertTrue(torch.equal(sum(mask.to(torch.int64) for mask in m.masks.values()),
                                    m.boundary.to(torch.int64)))
        center = torch.tensor([c.cylinder_x, c.cylinder_y], dtype=torch.float64)
        radial = m.face_centers[m.masks["cylinder"]] - center
        self.assertTrue(((radial * m.face_normals[m.masks["cylinder"]]).sum(-1) < 0).all())

    def test_float64_device_geometry(self):
        m = BodyFittedMesh(config())
        for name in ("vertices", "centers", "volumes", "face_centers",
                     "face_lengths", "face_normals", "face_area_vectors"):
            field = getattr(m, name)
            self.assertEqual(field.dtype, torch.float64)
            self.assertEqual(field.device.type, "cpu")


class FittedSolverTests(unittest.TestCase):
    def test_linear_pressure_gradient_and_nonorthogonal_face_derivative(self):
        s = BodyFittedSolver(config(cylinder_x=0.7, cylinder_y=0.8, nx=24))
        # This affine pressure also obeys the zero-pressure outlet reference.
        p = s.mesh.centers[..., 0].flatten() - s.config.length
        g = s._gradient(p, pressure=True)
        exact = torch.zeros_like(g)
        exact[:, 0] = 1
        self.assertTrue(torch.allclose(g, exact, atol=2e-13))
        dp = -p[s.o]
        dp[s.f] = p[s.ni] - p[s.oi]
        derivative = s.k * dp + (s.T * s._interpolate(g)).sum(-1)
        active = s.f | s.mesh.masks["outlet"]
        self.assertTrue(torch.allclose(derivative[active], s.S[active, 0], atol=1e-12))
        self.assertGreater(float(s.T.abs().max()), 0.1)

    def test_pressure_checkerboard_has_nonzero_rhie_chow_response(self):
        s = BodyFittedSolver(config())
        p = torch.tensor([(-1.) ** (j + i) for j in range(4) for i in range(16)],
                         dtype=torch.float64)
        flux, coefficient = s._rhie_chow(torch.zeros_like(s.velocity), p,
                                         torch.ones_like(s.p))
        self.assertGreater(float(flux[s.f].abs().max()), 1)
        self.assertGreater(float(s._sum(flux).abs().max()), 1)
        self.assertTrue((coefficient[s.f] > 0).all())
        self.assertTrue((coefficient[s.mesh.masks["outlet"]] > 0).all())
        self.assertEqual(int(torch.count_nonzero(coefficient[
            s.mesh.boundary & ~s.mesh.masks["outlet"]])), 0)

    def test_pressure_coefficients_use_relaxed_diagonal(self):
        s = BodyFittedSolver(config())
        diagonal, _, _, _ = s._momentum(s.velocity, s.p, s.mass_flux)
        d = s.volume * s.config.velocity_relaxation / diagonal
        _, a = s._rhie_chow(s.velocity, s.p, d)
        expected = s.config.density * s._interpolate(d) * s.k
        mask = s.f | s.mesh.masks["outlet"]
        self.assertTrue(torch.allclose(a[mask], expected[mask]))

    def test_corrected_outlet_flux_retained_and_locally_conservative(self):
        s = BodyFittedSolver(config())
        captured = {}
        correct = s._correct

        def capture(flux, coefficient, d):
            captured["predicted"] = flux.clone()
            correct(flux, coefficient, d)

        with patch.object(s, "_correct", side_effect=capture):
            metrics = s.step()
        outlet = s.mesh.masks["outlet"]
        self.assertGreater(float((s.mass_flux[outlet]
                                  - captured["predicted"][outlet]).abs().max()), 0.01)
        self.assertLess(metrics["continuity"], 1e-9)
        self.assertLess(metrics["mass_imbalance"], 1e-9)
        inlet = -s.mass_flux[s.mesh.masks["inlet"]].sum()
        outflow = s.mass_flux[outlet].sum()
        self.assertAlmostEqual(float(inlet), 2, places=11)
        self.assertAlmostEqual(float(inlet), float(outflow), places=9)

    def test_no_slip_and_uniform_inlet_are_face_conditions(self):
        s = BodyFittedSolver(config())
        s.step()
        walls = s.mesh.masks["wall"] | s.mesh.masks["cylinder"]
        self.assertEqual(int(torch.count_nonzero(s.boundary_velocity[walls])), 0)
        self.assertEqual(int(torch.count_nonzero(s.mass_flux[walls])), 0)
        inlet = s.mesh.masks["inlet"]
        self.assertTrue(torch.equal(s.boundary_velocity[inlet, 0],
                                    torch.ones(int(inlet.sum()), dtype=torch.float64)))
        self.assertEqual(int(torch.count_nonzero(s.boundary_velocity[inlet, 1])), 0)
        # No-slip diffusion contributes to the diagonal, not a solid neighbor.
        zero = torch.zeros_like(s.velocity)
        diagonal, ao, an, _ = s._momentum(zero, s.p * 0, s.mass_flux * 0)
        diffusion = s.config.viscosity * s.k
        diffusion[s.mesh.masks["outlet"]] = 0
        expected = torch.zeros_like(diagonal)
        expected.index_add_(0, s.o, diffusion)
        expected.index_add_(0, s.ni, diffusion[s.f])
        self.assertTrue(torch.allclose(diagonal, expected))
        self.assertTrue(torch.equal(ao, an))
        self.assertTrue((diffusion[walls] > 0).all())

    def test_small_low_re_grid_converges_steady_residual(self):
        r = BodyFittedSolver(config(nx=24, ny=8, reynolds=5)).solve()
        self.assertTrue(r.converged, r.history[-1])
        self.assertGreater(len(r.history), 5)
        for name in ("continuity", "momentum", "mass_imbalance"):
            self.assertLess(r.history[-1][name], r.config.tolerance)
        self.assertLess(r.history[-1]["momentum"], r.history[0]["momentum"] * 1e-3)
        self.assertGreater(float(r.u.max()), r.config.inlet_velocity)
        self.assertGreater(float(r.v.abs().max()), 0.1)
        inlet = r.mesh.owner[r.mesh.masks["inlet"]]
        outlet = r.mesh.owner[r.mesh.masks["outlet"]]
        self.assertGreater(float(r.p.flatten()[inlet].mean()),
                           float(r.p.flatten()[outlet].mean()))

    def test_pseudo_time_reaches_same_steady_state(self):
        steady = BodyFittedSolver(config(tolerance=1e-6)).solve()
        damped = BodyFittedSolver(config(tolerance=1e-6, pseudo_time_step=0.05)).solve()
        self.assertTrue(steady.converged)
        self.assertTrue(damped.converged, damped.history[-1])
        for a, b in ((steady.u, damped.u), (steady.v, damped.v), (steady.p, damped.p)):
            self.assertLess(float((a - b).abs().max()), 2e-4)

    def test_tiny_relaxation_does_not_falsely_converge(self):
        s = BodyFittedSolver(config(velocity_relaxation=1e-8))
        s.velocity.zero_()
        metrics = s.step()
        self.assertGreater(metrics["momentum"], s.config.tolerance)
        self.assertFalse(s.converged)

    def test_result_shapes_snapshots_and_iteration_limit(self):
        s = BodyFittedSolver(config(max_iterations=1))
        r = s.solve()
        self.assertFalse(r.converged)
        self.assertEqual(len(r.history), 1)
        for field in (r.u, r.v, r.p, r.x, r.y):
            self.assertEqual(field.shape, (4, 16))
            self.assertEqual(field.dtype, torch.float64)
            self.assertEqual(field.device.type, "cpu")
            self.assertTrue(torch.isfinite(field).all())
        self.assertEqual(r.fluid.dtype, torch.bool)
        self.assertTrue(r.fluid.all())
        uc, vc = r.cell_center_velocity()
        self.assertTrue(torch.equal(uc, r.u))
        self.assertTrue(torch.equal(vc, r.v))
        snapshots = [field.clone() for field in (r.u, r.v, r.p, r.mass_flux)]
        s.step()
        for old, current in zip(snapshots, (r.u, r.v, r.p, r.mass_flux)):
            self.assertTrue(torch.equal(old, current))
        self.assertEqual(len(r.history), 1)

    def test_nonfinite_iterate_rejected(self):
        s = BodyFittedSolver(config())
        s.velocity[0, 0] = float("nan")
        with self.assertRaisesRegex(RuntimeError, "nonfinite"):
            s.step()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_configured_cuda_device(self):
        r = BodyFittedSolver(config(device="cuda", max_iterations=1)).solve()
        for field in (r.u, r.v, r.p, r.x, r.mass_flux, r.mesh.vertices):
            self.assertEqual(field.device.type, "cuda")
            self.assertEqual(field.dtype, torch.float64)


if __name__ == "__main__":
    unittest.main()
