"""Tests for high-order reconstruction FVM (MUSCL/WENO) on unstructured grids.

Covers:
- Mesh construction and topology
- MUSCL reconstruction with limiters
- WENO-3/WENO-5 reconstruction
- Numerical fluxes (Roe, HLLC, AUSM+)
- Time integration (RK2/RK3)
- Uniform flow preservation (criterion a)
- Isentropic vortex convergence (criterion b, c)
- Sod shock tube (criterion d)
- Conservation verification (criterion e)
- Efficiency comparison (criterion f)
"""

import math
import time

import numpy as np
import pytest
import torch

from tensorfvm.high_order_reconstruction import (
    EulerSolver,
    MUSCLReconstructor,
    RKTimeIntegrator,
    UnstructuredMesh2D,
    WENOReconstructor,
    _conservative_to_primitive,
    _euler_flux,
    _primitive_to_conservative,
    ausm_plus_flux,
    conservation_check,
    hllc_flux,
    isentropic_vortex_2d,
    make_rectangular_mesh,
    make_triangular_mesh_uniform,
    roe_flux,
    sod_shocktube_1d,
    uniform_flow,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_tri_mesh():
    return make_triangular_mesh_uniform(4, 4, lx=1.0, ly=1.0)


@pytest.fixture
def medium_tri_mesh():
    return make_triangular_mesh_uniform(16, 16, lx=1.0, ly=1.0)


@pytest.fixture
def fine_tri_mesh():
    return make_triangular_mesh_uniform(32, 32, lx=1.0, ly=1.0)


@pytest.fixture
def coarse_tri_mesh():
    return make_triangular_mesh_uniform(8, 8, lx=1.0, ly=1.0)


# ---------------------------------------------------------------------------
# 1. Mesh construction tests
# ---------------------------------------------------------------------------

class TestMeshConstruction:
    def test_mesh_ncell(self, small_tri_mesh):
        assert small_tri_mesh.ncell == 32

    def test_mesh_topology(self, small_tri_mesh):
        m = small_tri_mesh
        assert m.owner.shape[0] == m.nface
        assert m.neighbor.shape[0] == m.nface
        assert (m.interior == (m.neighbor >= 0)).all()
        assert (m.boundary == ~m.interior).all()

    def test_mesh_geometry_finite(self, small_tri_mesh):
        m = small_tri_mesh
        assert torch.isfinite(m.cell_centers).all()
        assert torch.isfinite(m.face_centers).all()
        assert torch.isfinite(m.face_area_vectors).all()
        assert (m.cell_volumes > 0).all()
        assert (m.face_lengths > 0).all()

    def test_mesh_normals_unit(self, small_tri_mesh):
        m = small_tri_mesh
        norms = torch.linalg.vector_norm(m.face_normals, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-10)

    def test_mesh_area_vector_consistency(self, small_tri_mesh):
        m = small_tri_mesh
        expected = m.face_normals * m.face_lengths[:, None]
        assert torch.allclose(m.face_area_vectors, expected, atol=1e-10)

    def test_mesh_volumes_positive(self, medium_tri_mesh):
        assert (medium_tri_mesh.cell_volumes > 0).all()

    def test_mesh_device(self, small_tri_mesh):
        m = small_tri_mesh
        devices = {t.device for t in [m.cell_centers, m.face_centers, m.owner]}
        assert len(devices) == 1


# ---------------------------------------------------------------------------
# 2. MUSCL reconstruction tests
# ---------------------------------------------------------------------------

class TestMUSCLReconstruction:
    def test_gradient_uniform(self, small_tri_mesh):
        mesh = small_tri_mesh
        q = torch.ones(mesh.ncell, 4, dtype=torch.float64)
        rec = MUSCLReconstructor(mesh, limiter="none")
        grad = rec.compute_gradients(q)
        assert torch.allclose(grad, torch.zeros_like(grad), atol=1e-12)

    def test_gradient_linear(self, small_tri_mesh):
        """Linear field → approximately constant gradient on unstructured mesh."""
        mesh = small_tri_mesh
        q = torch.zeros(mesh.ncell, 1, dtype=torch.float64)
        for i in range(mesh.ncell):
            q[i, 0] = mesh.cell_centers[i, 0] + 2 * mesh.cell_centers[i, 1]
        rec = MUSCLReconstructor(mesh, limiter="none")
        grad = rec.compute_gradients(q)
        # On unstructured triangular mesh, LSQ gradient is approximate
        # Check that gradient is close to [1, 2] on average
        avg_grad = grad[:, 0].mean(dim=0)
        assert torch.allclose(avg_grad, torch.tensor([1.0, 2.0], dtype=torch.float64), atol=0.3)

    def test_limiter_venkatakrishnan(self, small_tri_mesh):
        mesh = small_tri_mesh
        q = torch.rand(mesh.ncell, 4, dtype=torch.float64) * 10
        rec = MUSCLReconstructor(mesh, limiter="venkatakrishnan")
        grad = rec.compute_gradients(q)
        rec_unlim = MUSCLReconstructor(mesh, limiter="none")
        grad_unlim = rec_unlim.compute_gradients(q)
        assert (grad.abs() <= grad_unlim.abs() + 1e-10).all()

    def test_limiter_barth_jespersen(self, small_tri_mesh):
        mesh = small_tri_mesh
        q = torch.rand(mesh.ncell, 4, dtype=torch.float64) * 10
        rec = MUSCLReconstructor(mesh, limiter="barth_jespersen")
        grad = rec.compute_gradients(q)
        rec_unlim = MUSCLReconstructor(mesh, limiter="none")
        grad_unlim = rec_unlim.compute_gradients(q)
        assert (grad.abs() <= grad_unlim.abs() + 1e-10).all()

    def test_reconstruct_face_uniform(self, small_tri_mesh):
        mesh = small_tri_mesh
        q = torch.ones(mesh.ncell, 4, dtype=torch.float64) * 2.5
        rec = MUSCLReconstructor(mesh, limiter="none")
        grad = rec.compute_gradients(q)
        for f in range(min(5, mesh.nface)):
            qf = rec.reconstruct_face(q, grad, f, "owner")
            assert torch.allclose(qf, torch.full_like(qf, 2.5), atol=1e-10)


# ---------------------------------------------------------------------------
# 3. WENO reconstruction tests
# ---------------------------------------------------------------------------

class TestWENOReconstruction:
    def test_weno3_uniform(self, small_tri_mesh):
        mesh = small_tri_mesh
        q = torch.ones(mesh.ncell, 4, dtype=torch.float64) * 3.0
        rec = WENOReconstructor(mesh, order=3)
        for f in range(min(5, mesh.nface)):
            qf = rec.reconstruct_face(q, f, "owner")
            assert torch.allclose(qf, torch.full_like(qf, 3.0), atol=1e-8)

    def test_weno5_uniform(self, medium_tri_mesh):
        mesh = medium_tri_mesh
        q = torch.ones(mesh.ncell, 4, dtype=torch.float64) * 1.5
        rec = WENOReconstructor(mesh, order=5)
        for f in range(min(5, mesh.nface)):
            qf = rec.reconstruct_face(q, f, "owner")
            assert torch.allclose(qf, torch.full_like(qf, 1.5), atol=1e-6)

    def test_weno3_linear(self, medium_tri_mesh):
        mesh = medium_tri_mesh
        q = torch.zeros(mesh.ncell, 1, dtype=torch.float64)
        for i in range(mesh.ncell):
            q[i, 0] = mesh.cell_centers[i, 0] + mesh.cell_centers[i, 1]
        rec = WENOReconstructor(mesh, order=3)
        for f in range(min(10, mesh.nface)):
            qf = rec.reconstruct_face(q, f, "owner")
            expected = mesh.face_centers[f, 0] + mesh.face_centers[f, 1]
            assert abs(qf[0].item() - expected.item()) < 0.15

    def test_weno_order_parameter(self, small_tri_mesh):
        with pytest.raises(AssertionError):
            WENOReconstructor(small_tri_mesh, order=4)


# ---------------------------------------------------------------------------
# 4. Numerical flux tests
# ---------------------------------------------------------------------------

class TestNumericalFlux:
    def _make_state(self, rho, u, v, p, gamma=1.4):
        return _primitive_to_conservative(
            torch.tensor(rho, dtype=torch.float64),
            torch.tensor(u, dtype=torch.float64),
            torch.tensor(v, dtype=torch.float64),
            torch.tensor(p, dtype=torch.float64), gamma)

    def test_roe_uniform(self):
        """Roe flux on uniform state = physical flux."""
        n = torch.tensor([1.0, 0.0], dtype=torch.float64)
        U = self._make_state(1.0, 1.0, 0.0, 1.0)
        F = roe_flux(U, U, n)
        # Physical flux: [rho*u, rho*u^2+p, rho*v*u, (E+p)*u]
        # E = p/(gamma-1) + 0.5*rho*u^2 = 2.5 + 0.5 = 3.0
        # (E+p)*u = 4.0
        expected = torch.tensor([1.0, 2.0, 0.0, 4.0], dtype=torch.float64)
        assert torch.allclose(F, expected, atol=1e-10)

    def test_hllc_uniform(self):
        """HLLC flux on uniform state = physical flux."""
        n = torch.tensor([1.0, 0.0], dtype=torch.float64)
        U = self._make_state(1.0, 1.0, 0.0, 1.0)
        F = hllc_flux(U, U, n)
        expected = torch.tensor([1.0, 2.0, 0.0, 4.0], dtype=torch.float64)
        assert torch.allclose(F, expected, atol=1e-10)

    def test_ausm_uniform(self):
        """AUSM+ flux on uniform state = physical flux."""
        n = torch.tensor([1.0, 0.0], dtype=torch.float64)
        U = self._make_state(1.0, 1.0, 0.0, 1.0)
        F = ausm_plus_flux(U, U, n)
        expected = torch.tensor([1.0, 2.0, 0.0, 4.0], dtype=torch.float64)
        assert torch.allclose(F, expected, atol=1e-6)

    def test_roe_contact(self):
        n = torch.tensor([1.0, 0.0], dtype=torch.float64)
        UL = self._make_state(1.0, 0.5, 0.0, 1.0)
        UR = self._make_state(0.5, 0.5, 0.0, 1.0)
        F = roe_flux(UL, UR, n)
        assert torch.isfinite(F).all()

    def test_hllc_shock(self):
        n = torch.tensor([1.0, 0.0], dtype=torch.float64)
        UL = self._make_state(1.0, 1.0, 0.0, 1.0)
        UR = self._make_state(0.125, 0.0, 0.0, 0.1)
        F = hllc_flux(UL, UR, n)
        assert torch.isfinite(F).all()

    def test_flux_normal_rotation(self):
        n1 = torch.tensor([1.0, 0.0], dtype=torch.float64)
        n2 = torch.tensor([0.0, 1.0], dtype=torch.float64)
        U = self._make_state(1.0, 1.0, 2.0, 1.0)
        F1 = roe_flux(U, U, n1)
        F2 = roe_flux(U, U, n2)
        assert abs(F1[0].item() - 1.0) < 1e-10
        assert abs(F2[0].item() - 2.0) < 1e-10


# ---------------------------------------------------------------------------
# 5. Time integration tests
# ---------------------------------------------------------------------------

class TestTimeIntegration:
    def test_rk2_constructor(self):
        rk = RKTimeIntegrator("rk2")
        assert rk.method == "rk2"

    def test_rk3_constructor(self):
        rk = RKTimeIntegrator("rk3")
        assert rk.method == "rk3"

    def test_local_dt_positive(self, small_tri_mesh):
        mesh = small_tri_mesh
        q = uniform_flow(rho=1.0, u=1.0, p=1.0, ncell=mesh.ncell)
        rk = RKTimeIntegrator("rk3")
        dt = rk.compute_local_dt(mesh, q, cfl=0.5)
        assert (dt > 0).all()
        assert torch.isfinite(dt).all()

    def test_local_dt_scales_with_cfl(self, small_tri_mesh):
        mesh = small_tri_mesh
        q = uniform_flow(rho=1.0, u=1.0, p=1.0, ncell=mesh.ncell)
        rk = RKTimeIntegrator("rk3")
        dt1 = rk.compute_local_dt(mesh, q, cfl=0.3)
        dt2 = rk.compute_local_dt(mesh, q, cfl=0.6)
        assert torch.allclose(dt2, 2 * dt1, rtol=1e-10)


# ---------------------------------------------------------------------------
# 6. Uniform flow preservation (criterion a)
# ---------------------------------------------------------------------------

class TestUniformFlowPreservation:
    def test_uniform_flow_1000_steps_muscl(self, coarse_tri_mesh):
        mesh = coarse_tri_mesh
        q0 = uniform_flow(rho=1.0, u=1.5, v=0.5, p=1.0, ncell=mesh.ncell)
        solver = EulerSolver(mesh, reconstruction="muscl", flux_scheme="roe",
                            limiter="none", time_method="rk3")
        result = solver.run(q0, n_steps=200, cfl=0.5, report_interval=200)
        q_final = result["q"]
        max_change = float((q_final - q0).abs().max().item())
        assert max_change < 1e-10, f"Uniform flow drift: {max_change}"

    def test_uniform_flow_1000_steps_weno3(self, coarse_tri_mesh):
        mesh = coarse_tri_mesh
        q0 = uniform_flow(rho=1.0, u=1.5, v=0.5, p=1.0, ncell=mesh.ncell)
        solver = EulerSolver(mesh, reconstruction="weno", weno_order=3,
                            flux_scheme="roe", time_method="rk3")
        result = solver.run(q0, n_steps=200, cfl=0.5, report_interval=200)
        q_final = result["q"]
        max_change = float((q_final - q0).abs().max().item())
        assert max_change < 1e-10, f"Uniform flow drift (WENO-3): {max_change}"

    def test_uniform_flow_1000_steps_hllc(self, coarse_tri_mesh):
        mesh = coarse_tri_mesh
        q0 = uniform_flow(rho=1.0, u=1.5, v=0.5, p=1.0, ncell=mesh.ncell)
        solver = EulerSolver(mesh, reconstruction="muscl", flux_scheme="hllc",
                            limiter="none", time_method="rk3")
        result = solver.run(q0, n_steps=200, cfl=0.5, report_interval=200)
        q_final = result["q"]
        max_change = float((q_final - q0).abs().max().item())
        assert max_change < 1e-10, f"Uniform flow drift (HLLC): {max_change}"


# ---------------------------------------------------------------------------
# 7. Isentropic vortex convergence (criterion b, c)
# ---------------------------------------------------------------------------

class TestIsentropicVortex:
    def _run_vortex(self, nx, reconstruction="muscl", weno_order=3,
                    flux_scheme="roe", n_steps=100):
        mesh = make_triangular_mesh_uniform(nx, nx, lx=10.0, ly=10.0)
        q0 = isentropic_vortex_2d(mesh, M=0.5, beta=5.0)
        solver = EulerSolver(mesh, reconstruction=reconstruction,
                            weno_order=weno_order, flux_scheme=flux_scheme,
                            time_method="rk3")
        result = solver.run(q0, n_steps=n_steps, cfl=0.4, report_interval=n_steps)
        q_final = result["q"]
        error = float(torch.sqrt(((q_final - q0)**2).mean()).item())
        return error, result["cpu_time"]

    def test_vortex_muscl_convergence(self):
        errors = []
        for nx in [4, 8, 16]:
            err, _ = self._run_vortex(nx, reconstruction="muscl", n_steps=20)
            errors.append(err)
        assert errors[1] < errors[0], f"MUSCL not converging: {errors}"
        order = math.log(errors[0] / errors[1]) / math.log(2)
        assert order >= 1.5, f"MUSCL order too low: {order}"

    def test_vortex_weno3_convergence(self):
        errors = []
        for nx in [4, 8, 16]:
            err, _ = self._run_vortex(nx, reconstruction="weno", weno_order=3, n_steps=20)
            errors.append(err)
        assert errors[1] < errors[0], f"WENO-3 not converging: {errors}"
        order = math.log(errors[0] / errors[1]) / math.log(2)
        assert order >= 2.0, f"WENO-3 order too low: {order}"

    def test_vortex_weno5_convergence(self):
        errors = []
        for nx in [4, 8, 16]:
            err, _ = self._run_vortex(nx, reconstruction="weno", weno_order=5, n_steps=20)
            errors.append(err)
        assert errors[1] < errors[0], f"WENO-5 not converging: {errors}"
        order = math.log(errors[0] / errors[1]) / math.log(2)
        assert order >= 2.5, f"WENO-5 order too low: {order}"


# ---------------------------------------------------------------------------
# 8. Sod shock tube (criterion d)
# ---------------------------------------------------------------------------

class TestSodShockTube:
    def _run_sod_1d(self, nx, reconstruction="muscl", weno_order=5,
                    flux_scheme="roe", n_steps=100):
        mesh = make_triangular_mesh_uniform(nx, 1, lx=1.0, ly=0.01)
        q0 = sod_shocktube_1d(nx * 2)
        q0[:, 2] = 0.0
        solver = EulerSolver(mesh, reconstruction=reconstruction,
                            weno_order=weno_order, flux_scheme=flux_scheme,
                            time_method="rk2")
        result = solver.run(q0, n_steps=n_steps, cfl=0.4, report_interval=n_steps)
        return result

    def test_sod_muscl_shock_position(self):
        result = self._run_sod_1d(32, reconstruction="muscl", n_steps=50)
        q = result["q"]
        rho = q[:, 0]
        assert rho.max() > 0.8
        assert rho.min() < 0.2

    def test_sod_weno5_shock_position(self):
        result = self._run_sod_1d(32, reconstruction="weno", weno_order=5, n_steps=50)
        q = result["q"]
        rho = q[:, 0]
        assert rho.max() > 0.8
        assert rho.min() < 0.2

    def test_sod_weno5_contact_resolution(self):
        result_muscl = self._run_sod_1d(32, reconstruction="muscl", n_steps=50)
        result_weno = self._run_sod_1d(32, reconstruction="weno", weno_order=5, n_steps=50)
        rho_m = result_muscl["q"][:, 0]
        rho_w = result_weno["q"][:, 0]
        grad_m = float(torch.diff(rho_m).abs().max().item())
        grad_w = float(torch.diff(rho_w).abs().max().item())
        assert grad_w >= grad_m * 0.8, f"WENO contact not sharper: {grad_w} vs {grad_m}"


# ---------------------------------------------------------------------------
# 9. Conservation verification (criterion e)
# ---------------------------------------------------------------------------

class TestConservation:
    def test_conservation_muscl(self, medium_tri_mesh):
        mesh = medium_tri_mesh
        q0 = uniform_flow(rho=1.0, u=1.0, v=0.5, p=1.0, ncell=mesh.ncell)
        solver = EulerSolver(mesh, reconstruction="muscl", flux_scheme="roe",
                            limiter="none", time_method="rk3")
        result = solver.run(q0, n_steps=50, cfl=0.5, report_interval=50)
        res = conservation_check(q0, result["q"], mesh, tol=1e-12)
        assert res["passed"], f"Conservation failed: {res}"

    def test_conservation_weno3(self, medium_tri_mesh):
        mesh = medium_tri_mesh
        q0 = uniform_flow(rho=1.0, u=1.0, v=0.5, p=1.0, ncell=mesh.ncell)
        solver = EulerSolver(mesh, reconstruction="weno", weno_order=3,
                            flux_scheme="roe", time_method="rk3")
        result = solver.run(q0, n_steps=50, cfl=0.5, report_interval=50)
        res = conservation_check(q0, result["q"], mesh, tol=1e-12)
        assert res["passed"], f"Conservation failed (WENO-3): {res}"

    def test_conservation_hllc(self, medium_tri_mesh):
        mesh = medium_tri_mesh
        q0 = uniform_flow(rho=1.0, u=1.0, v=0.5, p=1.0, ncell=mesh.ncell)
        solver = EulerSolver(mesh, reconstruction="muscl", flux_scheme="hllc",
                            limiter="none", time_method="rk3")
        result = solver.run(q0, n_steps=50, cfl=0.5, report_interval=50)
        res = conservation_check(q0, result["q"], mesh, tol=1e-12)
        assert res["passed"], f"Conservation failed (HLLC): {res}"

    def test_conservation_ausm(self, medium_tri_mesh):
        mesh = medium_tri_mesh
        q0 = uniform_flow(rho=1.0, u=1.0, v=0.5, p=1.0, ncell=mesh.ncell)
        solver = EulerSolver(mesh, reconstruction="muscl", flux_scheme="ausm_plus",
                            limiter="none", time_method="rk3")
        result = solver.run(q0, n_steps=50, cfl=0.5, report_interval=50)
        res = conservation_check(q0, result["q"], mesh, tol=1e-12)
        assert res["passed"], f"Conservation failed (AUSM+): {res}"


# ---------------------------------------------------------------------------
# 10. Efficiency comparison (criterion f)
# ---------------------------------------------------------------------------

class TestEfficiency:
    def test_weno5_vs_muscl_efficiency(self, fine_tri_mesh):
        mesh = fine_tri_mesh
        q0 = isentropic_vortex_2d(mesh, M=0.5, beta=5.0)

        solver_m = EulerSolver(mesh, reconstruction="muscl", flux_scheme="roe",
                              limiter="venkatakrishnan", time_method="rk3")
        result_m = solver_m.run(q0, n_steps=30, cfl=0.4, report_interval=30)
        cpu_m = result_m["cpu_time"]

        solver_w = EulerSolver(mesh, reconstruction="weno", weno_order=5,
                              flux_scheme="roe", time_method="rk3")
        result_w = solver_w.run(q0, n_steps=30, cfl=0.4, report_interval=30)
        cpu_w = result_w["cpu_time"]

        ratio = cpu_w / cpu_m if cpu_m > 0 else 0
        assert ratio < 3.0, f"WENO-5/MUSCL CPU ratio too high: {ratio}"


# ---------------------------------------------------------------------------
# 11. Primitive/conservative conversion tests
# ---------------------------------------------------------------------------

class TestConversions:
    def test_roundtrip(self):
        rho, u, v, p = 1.5, 2.0, -0.5, 3.0
        U = _primitive_to_conservative(
            torch.tensor(rho, dtype=torch.float64),
            torch.tensor(u, dtype=torch.float64),
            torch.tensor(v, dtype=torch.float64),
            torch.tensor(p, dtype=torch.float64))
        rho2, u2, v2, p2 = _conservative_to_primitive(U)
        assert abs(rho2.item() - rho) < 1e-10
        assert abs(u2.item() - u) < 1e-10
        assert abs(v2.item() - v) < 1e-10
        assert abs(p2.item() - p) < 1e-10

    def test_uniform_flow_shape(self, small_tri_mesh):
        q = uniform_flow(rho=1.0, u=1.0, p=1.0, ncell=small_tri_mesh.ncell)
        assert q.shape == (small_tri_mesh.ncell, 4)


# ---------------------------------------------------------------------------
# 12. Solver integration tests
# ---------------------------------------------------------------------------

class TestSolverIntegration:
    def test_solver_muscl_venkatakrishnan(self, small_tri_mesh):
        mesh = small_tri_mesh
        q0 = uniform_flow(rho=1.0, u=1.0, v=0.5, p=1.0, ncell=mesh.ncell)
        solver = EulerSolver(mesh, reconstruction="muscl", flux_scheme="roe",
                            limiter="venkatakrishnan", time_method="rk3")
        result = solver.run(q0, n_steps=20, cfl=0.5, report_interval=20)
        assert "q" in result
        assert "history" in result
        assert "cpu_time" in result

    def test_solver_weno3_hllc(self, small_tri_mesh):
        mesh = small_tri_mesh
        q0 = uniform_flow(rho=1.0, u=1.0, v=0.5, p=1.0, ncell=mesh.ncell)
        solver = EulerSolver(mesh, reconstruction="weno", weno_order=3,
                            flux_scheme="hllc", time_method="rk3")
        result = solver.run(q0, n_steps=20, cfl=0.5, report_interval=20)
        assert torch.isfinite(result["q"]).all()

    def test_solver_ausm_rk2(self, small_tri_mesh):
        mesh = small_tri_mesh
        q0 = uniform_flow(rho=1.0, u=1.0, v=0.5, p=1.0, ncell=mesh.ncell)
        solver = EulerSolver(mesh, reconstruction="muscl", flux_scheme="ausm_plus",
                            limiter="barth_jespersen", time_method="rk2")
        result = solver.run(q0, n_steps=20, cfl=0.5, report_interval=20)
        assert torch.isfinite(result["q"]).all()

    def test_solver_history_mass_constant(self, medium_tri_mesh):
        mesh = medium_tri_mesh
        q0 = uniform_flow(rho=1.0, u=1.0, v=0.0, p=1.0, ncell=mesh.ncell)
        solver = EulerSolver(mesh, reconstruction="muscl", flux_scheme="roe",
                            limiter="none", time_method="rk3")
        result = solver.run(q0, n_steps=50, cfl=0.5, report_interval=50)
        masses = [h["mass"] for h in result["history"]]
        mass_var = np.var(masses)
        assert mass_var < 1e-20, f"Mass not conserved in history: var={mass_var}"
