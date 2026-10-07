"""External-cylinder transient URANS regressions."""

import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch

from tensorfvm import SimpleSolver, SolverConfig
from tensorfvm.benchmark_cylinder_urans import compare_cylinder_urans
from tensorfvm.body_fitted import BodyFittedMesh
from tensorfvm.cylinder import export_result


def urans_config(**kwargs):
    options = dict(
        mesh_type="body-fitted",
        turbulence_model="spalart-allmaras",
        outer_boundary="far-field",
        body_fitted_stretching=5.0,
        nx=16,
        ny=8,
        length=12.0,
        height=8.0,
        cylinder_x=3.0,
        cylinder_y=4.0,
        cylinder_radius=0.5,
        reynolds=3900,
        time_step=0.05,
        inner_iterations=1,
        initial_perturbation=1e-3,
        max_iterations=4,
        tolerance=1e-2,
    )
    options.update(kwargs)
    return SolverConfig(**options)


class UransTests(unittest.TestCase):
    def test_configuration_rejects_incompatible_time_controls(self):
        with self.assertRaisesRegex(ValueError, "cannot be used together"):
            urans_config(pseudo_time_step=0.1)
        with self.assertRaisesRegex(ValueError, "requires time_step"):
            SolverConfig(mesh_type="body-fitted", initial_perturbation=1e-3)
        with self.assertRaisesRegex(ValueError, "only by the cylinder"):
            SolverConfig(mesh_type="c-grid", outer_boundary="far-field")

    def test_external_mesh_uses_far_field_and_wall_stretching(self):
        c = urans_config()
        mesh = BodyFittedMesh(c)
        self.assertEqual(int(mesh.masks["wall"].sum()), 0)
        self.assertGreater(int(mesh.masks["far-field"].sum()), 0)
        self.assertEqual(int(mesh.masks["cylinder"].sum()), c.nx)
        first = torch.linalg.vector_norm(mesh.vertices[1, 0] - mesh.vertices[0, 0])
        last = torch.linalg.vector_norm(mesh.vertices[-1, 0] - mesh.vertices[-2, 0])
        self.assertLess(first, last)

    def test_transient_solver_tracks_physical_time_and_force_history(self):
        result = SimpleSolver(urans_config()).solve()
        self.assertEqual(len(result.history), 4)
        self.assertEqual(len(result.force_history), 4)
        self.assertAlmostEqual(result.history[-1]["time"], 0.2)
        self.assertTrue(all(item["inner_iterations"] >= 1 for item in result.history))
        self.assertTrue(all(math.isfinite(item["drag"]) and math.isfinite(item["lift"])
                            for item in result.force_history))
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            export_result(result, directory)
            self.assertTrue((directory / "forces.json").is_file())

    def test_comparison_requires_periodic_lift_and_strict_limits(self):
        c = urans_config(max_iterations=100)
        history = [{"continuity": 1e-4, "momentum": 1e-4,
                    "mass_imbalance": 1e-4, "turbulence": 1e-4}]
        force_history = [
            {"step": i + 1, "time": (i + 1) * c.time_step,
             "drag": 1.12, "lift": 0.1 * math.sin(2 * math.pi * 0.2 * (i + 1) * c.time_step)}
            for i in range(100)
        ]
        result = SimpleNamespace(config=c, history=history, force_history=force_history)
        metrics = compare_cylinder_urans(result, warmup_steps=0)
        self.assertTrue(metrics["periodic_lift_resolved"])
        self.assertTrue(metrics["passed"], metrics)
        result.force_history = [dict(item, lift=0.0) for item in force_history]
        self.assertFalse(compare_cylinder_urans(result, warmup_steps=0)["passed"])


if __name__ == "__main__":
    unittest.main()
