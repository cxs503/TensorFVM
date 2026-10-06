"""Geometry and conservative collocated SIMPLE regressions."""

import unittest
from unittest.mock import patch

import torch

from tensorfvm.solver import SolverConfig
from tensorfvm.body_fitted import BodyFittedMesh, BodyFittedSolver, CGridMesh
from tensorfvm.benchmark_dfg import compare_dfg


def config(**kwargs):
    options = dict(mesh_type="body-fitted", nx=16, ny=4, reynolds=1,
                   max_iterations=300, tolerance=1e-5)
    options.update(kwargs)
    return SolverConfig(**options)


class MeshTests(unittest.TestCase):
    def test_naca_c_grid_geometry_and_boundary_partition(self):
        c = config(mesh_type="c-grid", nx=32, ny=8, reynolds=100)
        mesh = CGridMesh(c)
        self.assertEqual(mesh.vertices.shape, (9, 33, 2))
        self.assertEqual(mesh.centers.shape, (8, 32, 2))
        self.assertTrue(torch.equal(mesh.vertices[:, 0], mesh.vertices[:, -1]))
        self.assertTrue(torch.isfinite(mesh.vertices).all())
        self.assertTrue(torch.all(mesh.volumes > 0))
        self.assertTrue(torch.isfinite(mesh.centers).all())
        first_layer = torch.linalg.vector_norm(
            mesh.vertices[1, 0] - mesh.vertices[0, 0]
        )
        last_layer = torch.linalg.vector_norm(
            mesh.vertices[-1, 0] - mesh.vertices[-2, 0]
        )
        self.assertLess(first_layer, last_layer)
        self.assertEqual(int(mesh.masks["airfoil"].sum()), c.nx)
        self.assertEqual(int(mesh.masks["wall"].sum()), 0)
        self.assertAlmostEqual(float(mesh.face_lengths[mesh.masks["far-field"]].sum()),
                               2 * c.length, places=12)
        self.assertAlmostEqual(float(mesh.face_lengths[mesh.masks["inlet"]].sum()),
                               c.height, places=12)
        self.assertAlmostEqual(float(mesh.face_lengths[mesh.masks["outlet"]].sum()),
                               c.height, places=12)
        self.assertEqual(int(mesh.boundary.sum()), 2 * c.nx)
        self.assertTrue(torch.equal(
            sum(mask.to(torch.int64) for mask in mesh.masks.values()),
            mesh.boundary.to(torch.int64),
        ))

    def test_cambered_naca_profile_and_closed_wake(self):
        mesh = CGridMesh(config(mesh_type="c-grid", airfoil_code="4412",
                                nx=32, ny=8))
        self.assertTrue(torch.all(mesh.volumes > 0))
        airfoil = mesh.face_centers[mesh.masks["airfoil"]]
        self.assertGreater(float(airfoil[:, 1].max()), 1)
        self.assertTrue(torch.equal(mesh.vertices[:, 0], mesh.vertices[:, -1]))

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
    def test_naca_c_grid_step_produces_finite_aerodynamic_coefficients(self):
        solver = BodyFittedSolver(config(mesh_type="c-grid", nx=32, ny=8,
                                         reynolds=100, max_iterations=1))
        result = solver.solve()
        self.assertFalse(result.converged)
        self.assertEqual(set(result.aerodynamic_coefficients), {"drag", "lift"})
        self.assertTrue(all(torch.isfinite(torch.tensor(value))
                            for value in result.aerodynamic_coefficients.values()))
        self.assertTrue(torch.isfinite(result.mass_flux).all())

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

        def capture(flux, coefficient, d, steady_d=None):
            captured["predicted"] = flux.clone()
            correct(flux, coefficient, d, steady_d)

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

    def test_parabolic_cylinder_inlet_uses_mean_velocity(self):
        c = config(inlet_profile="parabolic", cylinder_x=0.7, cylinder_y=0.8,
                   nx=24, ny=8, height=2, inlet_velocity=0.2)
        s = BodyFittedSolver(c)
        inlet = s.mesh.masks["inlet"]
        y0, y1 = s.mesh.face_vertices[inlet, :, 1].unbind(1)
        expected = (
            6 * c.inlet_velocity
            * ((y1.square() - y0.square()) / (2 * c.height)
               - (y1.pow(3) - y0.pow(3)) / (3 * c.height ** 2))
            / (y1 - y0)
        )
        self.assertTrue(torch.allclose(s.boundary_velocity[inlet, 0], expected))
        self.assertEqual(int(torch.count_nonzero(s.boundary_velocity[inlet, 1])), 0)
        self.assertAlmostEqual(
            float((s.boundary_velocity[inlet, 0] * s.mesh.face_lengths[inlet]).sum()),
            c.inlet_velocity * c.height,
            places=12,
        )

    def test_parabolic_inlet_is_restricted_to_cylinder_o_grid(self):
        with self.assertRaisesRegex(ValueError, "only supported"):
            SolverConfig(inlet_profile="parabolic", mesh_type="cartesian")
        with self.assertRaisesRegex(ValueError, "only supported"):
            SolverConfig(inlet_profile="parabolic", mesh_type="c-grid")

    def test_cylinder_force_coefficients_are_exported(self):
        result = BodyFittedSolver(config(nx=24, ny=8, max_iterations=2)).solve()
        self.assertEqual(set(result.aerodynamic_coefficients), {"drag", "lift"})
        self.assertTrue(all(torch.isfinite(torch.tensor(value))
                            for value in result.aerodynamic_coefficients.values()))

    def test_dfg_acceptance_uses_strict_three_percent_drag_error(self):
        from types import SimpleNamespace

        result = SimpleNamespace(
            aerodynamic_coefficients={"drag": 5.579535 * 1.03, "lift": 0},
            converged=True,
            history=[{"continuity": 0, "momentum": 0, "mass_imbalance": 0}],
            config=SimpleNamespace(tolerance=1e-6, nx=64, ny=24),
        )
        self.assertFalse(compare_dfg(result)["passed"])
        result.aerodynamic_coefficients["drag"] = 5.579535 * 1.02
        self.assertTrue(compare_dfg(result)["passed"])
        result.converged = False
        self.assertFalse(compare_dfg(result)["passed"])

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

    def test_velocity_relaxation_reaches_same_steady_state(self):
        reference = BodyFittedSolver(config(tolerance=1e-6)).solve()
        relaxed = BodyFittedSolver(config(tolerance=1e-6, velocity_relaxation=0.4)).solve()
        self.assertTrue(reference.converged)
        self.assertTrue(relaxed.converged, relaxed.history[-1])
        for a, b in ((reference.u, relaxed.u), (reference.v, relaxed.v),
                     (reference.p, relaxed.p), (reference.mass_flux, relaxed.mass_flux)):
            self.assertLess(float((a - b).abs().max()), 2e-4)

    def test_damped_pressure_response_matches_predictor_sensitivity(self):
        s = BodyFittedSolver(config(pseudo_time_step=0.05))
        diagonal, _, _, _ = s._momentum(s.velocity, s.p, s.mass_flux)
        steady_d = s.volume / diagonal
        d = s.volume * s.config.velocity_relaxation / (
            diagonal + s.config.density * s.volume / s.config.pseudo_time_step)
        old = s.velocity.clone()
        flux, coefficient = s._rhie_chow(s.velocity, s.p, d, steady_d, old)
        pressure = s.p.clone()
        # A unit pressure relaxation exposes the actual full correction.
        s.config.pressure_relaxation = 1
        s._correct(flux, coefficient, d, steady_d)
        correction = s.p - pressure
        # Predictor memory is fixed during this linearization.
        corrected_flux = s.mass_flux.clone()
        s.mass_flux = (s.config.density * (s._interpolate(old) * s.S).sum(-1))
        original_memory = s.mass_flux.clone()
        before, _ = s._rhie_chow(old, pressure, d, steady_d, old)
        s.mass_flux = original_memory
        after, _ = s._rhie_chow(
            old - d[:, None] * s._gradient(correction, pressure=True),
            pressure + correction, d, steady_d, old)
        self.assertTrue(torch.allclose(after - before, corrected_flux - flux,
                                       atol=2e-11, rtol=2e-11))

    def test_tiny_relaxation_does_not_falsely_converge(self):
        s = BodyFittedSolver(config(velocity_relaxation=1e-8))
        s.velocity.zero_()
        metrics = s.step()
        self.assertGreater(metrics["momentum"], s.config.tolerance)
        self.assertFalse(s.converged)

    def test_tiny_pseudo_time_does_not_falsely_converge(self):
        s = BodyFittedSolver(config(pseudo_time_step=1e-8))
        metrics = s.step()
        self.assertGreater(metrics["momentum"], s.config.tolerance)
        self.assertFalse(s.converged)

    def test_linear_solve_checks_true_not_only_recursive_residual(self):
        s = BodyFittedSolver(config())
        one = torch.ones_like(s.p)
        zero = torch.zeros_like(s.p)
        edges = torch.zeros_like(s.k[s.f])
        # Simulate residual drift: the recurrence claims an exact solution,
        # but a fresh operator application does not satisfy the original RHS.
        with patch.object(s, "_matvec", side_effect=[zero, one, zero]):
            with self.assertRaisesRegex(RuntimeError, "did not converge"):
                s._linear(one, edges, edges, one, zero)

    def test_repeated_solve_honors_iteration_budget_and_convergence(self):
        s = BodyFittedSolver(config(max_iterations=1))
        first = s.solve()
        with patch.object(s, "step", side_effect=AssertionError("unexpected iteration")):
            repeated = s.solve()
        self.assertEqual(len(repeated.history), 1)
        self.assertTrue(torch.equal(first.u, repeated.u))
        converged = BodyFittedSolver(config())
        converged.solve()
        with patch.object(converged, "step", side_effect=AssertionError("unexpected iteration")):
            self.assertTrue(converged.solve().converged)

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
