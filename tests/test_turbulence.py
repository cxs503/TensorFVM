"""Spalart--Allmaras and high-Re flat-plate regression tests."""

import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch

from tensorfvm import FlatPlateMesh, SimpleSolver, SolverConfig
from tensorfvm.cylinder import export_result
from tensorfvm.benchmark_flat_plate import (
    ERROR_LIMIT,
    compare_flat_plate,
    reference_skin_friction,
)
from tensorfvm.body_fitted import BodyFittedSolver


def flat_plate_config(**kwargs):
    options = dict(
        mesh_type="flat-plate",
        turbulence_model="spalart-allmaras",
        cylinder_radius=None,
        nx=16,
        ny=8,
        length=1,
        height=0.2,
        reynolds=100_000,
        max_iterations=2,
        tolerance=1e-5,
        velocity_relaxation=0.5,
        turbulence_relaxation=0.5,
    )
    options.update(kwargs)
    return SolverConfig(**options)


class FlatPlateMeshTests(unittest.TestCase):
    def test_geometry_boundary_partition_and_closure(self):
        c = flat_plate_config(nx=12, ny=6)
        mesh = FlatPlateMesh(c)
        self.assertEqual(mesh.vertices.shape, (7, 13, 2))
        self.assertEqual(mesh.centers.shape, (6, 12, 2))
        self.assertTrue(torch.all(mesh.volumes > 0))
        self.assertAlmostEqual(float(mesh.face_lengths[mesh.masks["wall"]].sum()),
                               c.length, places=12)
        self.assertAlmostEqual(float(mesh.face_lengths[mesh.masks["far-field"]].sum()),
                               c.length, places=12)
        self.assertAlmostEqual(float(mesh.face_lengths[mesh.masks["inlet"]].sum()),
                               c.height, places=12)
        self.assertAlmostEqual(float(mesh.face_lengths[mesh.masks["outlet"]].sum()),
                               c.height, places=12)
        self.assertEqual(int(mesh.boundary.sum()), 2 * (c.nx + c.ny))
        self.assertTrue(torch.equal(
            sum(mask.to(torch.int64) for mask in mesh.masks.values()),
            mesh.boundary.to(torch.int64),
        ))

    def test_sa_configuration_restrictions_and_reference_length(self):
        with self.assertRaisesRegex(ValueError, "requires"):
            SolverConfig(turbulence_model="spalart-allmaras")
        with self.assertRaisesRegex(ValueError, "does not use cylinder_radius"):
            SolverConfig(mesh_type="flat-plate", cylinder_radius=0.2)
        with self.assertRaises(ValueError):
            flat_plate_config(sa_freestream_ratio=-1)
        c = flat_plate_config()
        self.assertAlmostEqual(c.reference_length, c.length)
        self.assertAlmostEqual(c.viscosity, 1e-5)


class SpalartAllmarasTests(unittest.TestCase):
    def test_one_step_is_finite_and_produces_nonnegative_eddy_viscosity(self):
        solver = BodyFittedSolver(flat_plate_config())
        metric = solver.step()
        self.assertIn("turbulence", metric)
        self.assertIn("turbulence_change", metric)
        self.assertTrue(math.isfinite(metric["turbulence"]))
        self.assertTrue(torch.isfinite(solver.nu_tilde).all())
        self.assertTrue(torch.isfinite(solver.turbulent_kinematic_viscosity).all())
        self.assertTrue((solver.nu_tilde >= 0).all())
        self.assertTrue((solver.turbulent_kinematic_viscosity >= 0).all())
        self.assertGreater(float(solver.wall_distance.min()), 0)
        self.assertIsInstance(SimpleSolver(flat_plate_config()), BodyFittedSolver)

    def test_sa_result_exports_working_and_eddy_viscosities(self):
        result = BodyFittedSolver(flat_plate_config(nx=12, ny=8, max_iterations=1)).solve()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            export_result(result, directory)
            with (directory / "turbulence.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), result.config.nx * result.config.ny)
            self.assertEqual(set(rows[0]), {"x", "y", "nu_tilde", "nu_t"})
            self.assertTrue(all(math.isfinite(float(row["nu_t"])) for row in rows))
            summary = json.loads((directory / "summary.json").read_text())
            self.assertIn("surface_forces", summary)
            self.assertIn("plate", summary["surface_forces"])

    def test_flat_plate_comparison_requires_sa_convergence_and_strict_limit(self):
        c = flat_plate_config(nx=8, ny=4)
        reference = reference_skin_friction(c.reynolds)
        force = reference * 0.5 * c.density * c.inlet_velocity ** 2 * c.length
        result = SimpleNamespace(
            config=c,
            surface_forces={"plate": {"x": force, "y": 0.0}},
            y=torch.full((c.ny, c.nx), 1e-6, dtype=torch.float64),
            converged=True,
            history=[{"continuity": 1e-8, "momentum": 1e-8,
                      "mass_imbalance": 1e-8, "turbulence": 1e-8}],
        )
        self.assertTrue(compare_flat_plate(result)["passed"])
        # Push beyond the strict threshold rather than rely on a rounded product
        # landing microscopically below 20% after force normalization.
        result.surface_forces["plate"]["x"] *= 1 + ERROR_LIMIT + 1e-8
        self.assertFalse(compare_flat_plate(result)["passed"])
        result.surface_forces["plate"]["x"] = force
        result.converged = False
        self.assertFalse(compare_flat_plate(result)["passed"])


if __name__ == "__main__":
    unittest.main()
