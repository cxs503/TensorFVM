"""Extruded 3-D body-fitted O-grid topology and quality regressions."""

import json
from pathlib import Path
import tempfile
import unittest

import torch

from tensorfvm.body_fitted3d import BodyFittedCylinderMesh3D
from tensorfvm.mesh_cylinder3d import run_mesh_benchmark
from tensorfvm.runtime import DistributedRuntime
from tensorfvm.solver3d import Cylinder3DConfig, Cylinder3DSolver


def _two_rank_mesh_worker(rank: int, init_file: str, output_file: str) -> None:
    """Build independent local O-grid slabs and reduce their geometric invariants."""
    torch.distributed.init_process_group(
        backend="gloo", init_method=f"file://{init_file}", rank=rank, world_size=2
    )
    try:
        runtime = DistributedRuntime.discover("cpu")
        config = Cylinder3DConfig(nx=32, ny=8, nz=6, mesh_type="body-fitted",
                                  body_fitted_stretching=2.5)
        partition = runtime.partition_z(config.nz)
        mesh = BodyFittedCylinderMesh3D.build(config, runtime.device,
                                               partition.start, partition.stop)
        volume = runtime.global_sum(mesh.cell_volumes.sum())
        area = runtime.global_sum(mesh.cylinder_surface_area)
        local_cells = torch.tensor([mesh.cell_count], dtype=torch.int64)
        parts = [torch.empty_like(local_cells) for _ in range(2)]
        torch.distributed.all_gather(parts, local_cells)
        if rank == 0:
            torch.save({"volume": volume, "area": area,
                        "cells_per_rank": torch.cat(parts)}, output_file)
        torch.distributed.barrier()
    finally:
        torch.distributed.destroy_process_group()


class BodyFittedCylinder3DMeshTests(unittest.TestCase):
    def config(self, **kwargs):
        options = dict(nx=32, ny=8, nz=6, mesh_type="body-fitted",
                       body_fitted_stretching=2.5)
        options.update(kwargs)
        return Cylinder3DConfig(**options)

    def test_extruded_o_grid_has_positive_volumes_seam_and_wall_orientation(self):
        config = self.config()
        mesh = BodyFittedCylinderMesh3D.build(config, torch.device("cpu"))
        self.assertEqual(mesh.vertices.shape, (7, 9, 33, 3))
        self.assertEqual(mesh.centers.shape, (6, 8, 32, 3))
        self.assertEqual(mesh.connectivity.shape, (6, 8, 32, 8))
        self.assertTrue(torch.equal(mesh.vertices[:, :, 0], mesh.vertices[:, :, -1]))
        self.assertTrue(torch.all(mesh.cell_volumes > 0))
        self.assertTrue(torch.isfinite(mesh.vertices).all())
        self.assertTrue(torch.isfinite(mesh.centers).all())
        self.assertEqual(int(mesh.connectivity.min()), 0)
        self.assertLess(int(mesh.connectivity.max()), mesh.node_count)
        expected_volume = mesh.cross_section_area * config.span
        self.assertTrue(torch.allclose(mesh.cell_volumes.sum(), expected_volume, atol=1e-12))
        center = torch.tensor([config.cylinder_x, config.cylinder_y], dtype=torch.float64)
        radial = mesh.cylinder_face_centers[..., :2] - center
        normal = mesh.cylinder_face_area_vectors[..., :2]
        self.assertTrue(((radial * normal).sum(-1) < 0).all())
        relative_area_error = abs(float(mesh.cylinder_surface_area)
                                  - mesh.analytic_cylinder_surface_area)
        relative_area_error /= mesh.analytic_cylinder_surface_area
        self.assertLess(relative_area_error, 0.01)

    def test_local_z_slab_preserves_global_coordinates_and_volume(self):
        config = self.config()
        mesh = BodyFittedCylinderMesh3D.build(config, torch.device("cpu"), 2, 5)
        self.assertEqual(mesh.local_nz, 3)
        self.assertEqual(mesh.vertices.shape[0], 4)
        self.assertAlmostEqual(float(mesh.vertices[0, 0, 0, 2]), 1.0)
        self.assertAlmostEqual(float(mesh.vertices[-1, 0, 0, 2]), 2.5)
        expected = mesh.cross_section_area * mesh.local_span
        self.assertTrue(torch.allclose(mesh.cell_volumes.sum(), expected, atol=1e-12))
        self.assertEqual(mesh.lateral_owner.shape[0], 3)
        self.assertEqual(mesh.lateral_neighbor.shape, mesh.lateral_owner.shape)
        self.assertTrue((mesh.lateral_neighbor[~mesh.lateral_boundary] >= 0).all())

    @unittest.skipUnless(torch.distributed.is_available(), "requires torch.distributed")
    def test_two_rank_o_grid_partitions_preserve_global_volume_and_surface_area(self):
        config = self.config()
        reference = BodyFittedCylinderMesh3D.build(config, torch.device("cpu"))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "mesh.pt"
            torch.multiprocessing.spawn(_two_rank_mesh_worker,
                                        args=(str(root / "gloo-init"), str(output)),
                                        nprocs=2, join=True)
            distributed = torch.load(output, map_location="cpu", weights_only=True)
        self.assertTrue(torch.allclose(distributed["volume"], reference.cell_volumes.sum(),
                                       rtol=0, atol=1e-12))
        self.assertTrue(torch.allclose(distributed["area"], reference.cylinder_surface_area,
                                       rtol=0, atol=1e-12))
        self.assertTrue(torch.equal(distributed["cells_per_rank"], torch.tensor([768, 768])))

    def test_cartesian_solver_rejects_curved_mesh_until_curvilinear_kernel_exists(self):
        with self.assertRaises(NotImplementedError):
            Cylinder3DSolver(self.config())

    def test_portable_body_fitted_mesh_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            summary = run_mesh_benchmark(temporary, nx=32, ny=8, nz=6)
            self.assertTrue(summary["passed"], summary)
            directory = Path(temporary)
            for name in ("nodes.csv", "cells.csv", "cylinder-faces.csv", "summary.json"):
                self.assertTrue((directory / name).is_file())
            stored = json.loads((directory / "summary.json").read_text())
            self.assertTrue(stored["passed"])
            self.assertEqual(stored["config"]["mesh_type"], "body-fitted")
            self.assertEqual(stored["quality"]["global_cell_count"], 32 * 8 * 6)
            self.assertLess(stored["quality"]["volume_relative_error"], 1e-12)


if __name__ == "__main__":
    unittest.main()
