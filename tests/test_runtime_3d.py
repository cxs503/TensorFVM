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

    def test_one_step_has_finite_3d_fields_and_periodic_span(self):
        solver = Cylinder3DSolver(self.config(max_steps=1))
        result = solver.solve()
        self.assertEqual(result.velocity.shape, (4, 24, 32, 3))
        self.assertEqual(result.pressure.shape, (4, 24, 32))
        self.assertTrue(torch.isfinite(result.velocity).all())
        self.assertTrue(torch.isfinite(result.pressure).all())
        self.assertGreater(torch.count_nonzero(~result.fluid).item(), 0)
        metric = result.history[-1]
        self.assertTrue(all(math.isfinite(value) for value in metric.values()))
        self.assertGreaterEqual(metric["cfl"], 0)
        self.assertEqual(len(result.force_history), 1)

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


if __name__ == "__main__":
    unittest.main()
