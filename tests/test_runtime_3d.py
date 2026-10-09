"""Shared runtime and executable 3-D cylinder kernel regressions."""

import json
import math
from pathlib import Path
import tempfile
import unittest

import torch

from tensorfvm.benchmark_cylinder3d import export_result, run_benchmark
from tensorfvm.runtime import DistributedRuntime, partition_slab
from tensorfvm.solver3d import Cylinder3DConfig, Cylinder3DSolver


def _spanwise_perturbation(solver):
    _, y, z = solver.mesh.centers()
    solver.velocity[..., 1] += .02 * torch.sin(2*math.pi*z/solver.config.span) * torch.sin(math.pi*y/solver.config.height)


def _two_rank_projection_worker(rank: int, init_file: str, output_file: str) -> None:
    """Run a small Gloo z-slab solve and persist a test-only gathered field."""
    torch.distributed.init_process_group(
        backend="gloo", init_method=f"file://{init_file}", rank=rank, world_size=2
    )
    try:
        config = Cylinder3DConfig(nx=32, ny=24, nz=4, max_steps=2,
                                  pressure_iterations=300, time_step=0.001,
                                  pressure_relative_tolerance=1e-12,
                                  pressure_absolute_tolerance=1e-13)
        solver = Cylinder3DSolver(config, runtime=DistributedRuntime.discover("cpu"))
        _spanwise_perturbation(solver)
        result = solver.solve()
        velocity_parts = [torch.empty_like(result.velocity) for _ in range(2)]
        pressure_parts = [torch.empty_like(result.pressure) for _ in range(2)]
        torch.distributed.all_gather(velocity_parts, result.velocity)
        torch.distributed.all_gather(pressure_parts, result.pressure)
        face_parts = []
        for face in result.face_velocity:
            parts = [torch.empty_like(face) for _ in range(2)]
            torch.distributed.all_gather(parts, face)
            face_parts.append(torch.cat(parts, dim=0))
        if rank == 0:
            torch.save({
                "velocity": torch.cat(velocity_parts, dim=0),
                "pressure": torch.cat(pressure_parts, dim=0),
                "history": result.history,
                "force_history": result.force_history,
                "local_shape": tuple(result.velocity.shape),
                "face_velocity": face_parts,
            }, output_file)
        torch.distributed.barrier()
    finally:
        torch.distributed.destroy_process_group()


class RuntimeTests(unittest.TestCase):
    def test_balanced_slab_partitions_cover_domain(self):
        partitions = [partition_slab(10, rank, 3) for rank in range(3)]
        self.assertEqual([(part.start, part.stop) for part in partitions],
                         [(0, 4), (4, 7), (7, 10)])
        self.assertEqual(sum(part.local_size for part in partitions), 10)
        self.assertTrue(partitions[0].owns_lower_boundary)
        self.assertTrue(partitions[-1].owns_upper_boundary)
        with self.assertRaises(ValueError):
            partition_slab(0, 0, 1)
        runtime = DistributedRuntime.discover("cpu")
        self.assertEqual(runtime.world_size, 1)
        self.assertEqual(runtime.partition_z(7).local_size, 7)


class Cylinder3DTests(unittest.TestCase):
    def config(self, **kwargs):
        options = dict(nx=32, ny=24, nz=4, max_steps=2,
                       pressure_iterations=20, time_step=0.001)
        options.update(kwargs)
        return Cylinder3DConfig(**options)

    def test_invalid_configuration(self):
        with self.assertRaises(ValueError):
            self.config(nz=3)
        with self.assertRaises(ValueError):
            self.config(smagorinsky_constant=-0.1)
        with self.assertRaises(ValueError):
            self.config(cylinder_x=0.2)
        with self.assertRaises(ValueError):
            self.config(pressure_relative_tolerance=0)
        with self.assertRaises(ValueError):
            self.config(pressure_absolute_tolerance=float("nan"))

    def test_one_step_has_finite_3d_fields_and_periodic_span(self):
        solver = Cylinder3DSolver(self.config(max_steps=1))
        result = solver.solve()
        self.assertEqual(result.velocity.shape, (4, 24, 32, 3))
        self.assertEqual(result.pressure.shape, (4, 24, 32))
        self.assertEqual(result.partition.start, 0)
        self.assertEqual(result.partition.stop, 4)
        self.assertTrue(torch.isfinite(result.velocity).all())
        self.assertTrue(torch.isfinite(result.pressure).all())
        self.assertGreater(torch.count_nonzero(~result.fluid).item(), 0)
        metric = result.history[-1]
        self.assertTrue(all(math.isfinite(value) for value in metric.values()))
        self.assertGreaterEqual(metric["cfl"], 0)
        self.assertEqual(len(result.force_history), 1)
        self.assertIn("pressure_residual", metric)
        self.assertIn("pressure_converged", metric)

    def test_reduced_pressure_operator_is_symmetric_positive(self):
        solver = Cylinder3DSolver(self.config(max_steps=1))
        generator = torch.Generator(device="cpu").manual_seed(7)
        first = torch.randn(solver.pressure.shape, dtype=torch.float64,
                            generator=generator).masked_fill(~solver._pressure_unknown, 0)
        second = torch.randn(solver.pressure.shape, dtype=torch.float64,
                             generator=generator).masked_fill(~solver._pressure_unknown, 0)
        first_image = solver._pressure_operator(first)
        second_image = solver._pressure_operator(second)
        left = float(solver._global_dot(first, second_image))
        right = float(solver._global_dot(first_image, second))
        energy = float(solver._global_dot(first, first_image))
        self.assertAlmostEqual(left, right, places=12)
        self.assertGreater(energy, 0)

    def test_portable_midspan_export_and_smoke_case(self):
        result = Cylinder3DSolver(self.config(max_steps=1)).solve()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            export_result(result, directory)
            for name in ("midspan.csv", "history.json", "forces.json", "summary.json"):
                self.assertTrue((directory / name).is_file())
            summary = json.loads((directory / "summary.json").read_text())
            self.assertEqual(summary["steps"], 1)
        with tempfile.TemporaryDirectory() as temporary:
            summary = run_benchmark(temporary, steps=2, nx=32, ny=24, nz=4)
            self.assertTrue(summary["passed"], summary)
            self.assertEqual(summary["world_size"], 1)
        with tempfile.TemporaryDirectory() as temporary:
            failed = run_benchmark(temporary, steps=1, nx=32, ny=24, nz=4,
                                   pressure_iterations=1,
                                   pressure_relative_tolerance=1e-12,
                                   pressure_absolute_tolerance=1e-12)
            self.assertFalse(failed["passed"])
            self.assertEqual(failed["final"]["pressure_converged"], 0)

    @unittest.skipUnless(torch.distributed.is_available(), "requires torch.distributed")
    def test_two_rank_z_slab_matches_single_rank_projection(self):
        """Guard real halo/Poisson coupling against a full-domain fallback."""
        config = Cylinder3DConfig(nx=32, ny=24, nz=4, max_steps=2,
                                  pressure_iterations=300, time_step=0.001,
                                  pressure_relative_tolerance=1e-12,
                                  pressure_absolute_tolerance=1e-13)
        solver = Cylinder3DSolver(config)
        _spanwise_perturbation(solver)
        reference = solver.solve()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            init_file = root / "gloo-init"
            output_file = root / "distributed.pt"
            torch.multiprocessing.spawn(_two_rank_projection_worker,
                                        args=(str(init_file), str(output_file)), nprocs=2,
                                        join=True)
            distributed = torch.load(output_file, map_location="cpu", weights_only=True)
        self.assertEqual(distributed["local_shape"], (2, 24, 32, 3))
        self.assertEqual(reference.history[-1]["pressure_converged"], 1)
        self.assertEqual(distributed["history"][-1]["pressure_converged"], 1)
        self.assertLessEqual(reference.history[-1]["pressure_residual"],
                             reference.history[-1]["pressure_target_residual"])
        self.assertLessEqual(float((distributed["velocity"]-reference.velocity).abs().max()),
                             2e-12 * max(1., float(reference.velocity.abs().max())))
        # Pressure has a ~94-unit transient gauge amplitude at dt=.001.
        # Use a global field scale, including near-zero individual cells.
        pressure_error = (distributed["pressure"] - reference.pressure).abs().max()
        self.assertLessEqual(float(pressure_error),
                             2e-12 * max(1., float(reference.pressure.abs().max())))
        for parallel, serial in zip(distributed["face_velocity"], reference.face_velocity):
            self.assertLessEqual(float((parallel - serial).abs().max()),
                                 2e-12 * max(1., float(serial.abs().max())))
        for parallel, serial in zip(distributed["history"], reference.history):
            for key, value in serial.items():
                if key == "pressure_residual":
                    # Residual vectors near the Krylov tolerance need not have the
                    # same direction/norm under different reduction orders. Both
                    # true residuals must satisfy their unchanged solver targets.
                    self.assertLessEqual(parallel[key], parallel["pressure_target_residual"])
                    self.assertLessEqual(value, serial["pressure_target_residual"])
                elif isinstance(value, int):
                    self.assertEqual(parallel[key], value, key)
                else:
                    self.assertTrue(math.isclose(parallel[key], value, rel_tol=2e-12,
                                                 abs_tol=3e-12), key)
        for parallel, serial in zip(distributed["force_history"], reference.force_history):
            for key, value in serial.items():
                if key == "pressure_residual":
                    # Residual vectors near the Krylov tolerance need not have the
                    # same direction/norm under different reduction orders. Both
                    # true residuals must satisfy their unchanged solver targets.
                    self.assertLessEqual(parallel[key], parallel["pressure_target_residual"])
                    self.assertLessEqual(value, serial["pressure_target_residual"])
                elif isinstance(value, int):
                    self.assertEqual(parallel[key], value, key)
                else:
                    self.assertTrue(math.isclose(parallel[key], value, rel_tol=2e-12,
                                                 abs_tol=3e-12), key)


if __name__ == "__main__":
    unittest.main()
