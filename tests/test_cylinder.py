import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from tensorfvm.cylinder import export_result, main
from tensorfvm.solver import SolverConfig


class CylinderOutputTests(unittest.TestCase):
    def result(self, converged=True):
        config = SolverConfig(nx=20, ny=10)
        field = torch.ones((config.ny, config.nx), dtype=torch.float64)
        return SimpleNamespace(
            config=config,
            p=field * 2,
            fluid=field.bool(),
            x=(torch.arange(config.nx, dtype=torch.float64) + 0.5)
              * config.length / config.nx,
            y=(torch.arange(config.ny, dtype=torch.float64) + 0.5)
              * config.height / config.ny,
            cell_center_velocity=lambda: (field, field * 0),
            history=[{"iteration": 1, "continuity": 1e-9, "momentum": 1e-8}],
            converged=converged,
        )

    def test_csv_coordinates_and_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "nested"
            result = self.result()
            export_result(result, directory)
            with (directory / "fields.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 200)
            self.assertAlmostEqual(float(rows[0]["x"]), 0.1)
            self.assertAlmostEqual(float(rows[0]["y"]), 0.1)
            self.assertEqual(float(rows[-1]["p"]), 2)
            self.assertEqual(rows[0]["fluid"], "1")
            summary = json.loads((directory / "summary.json").read_text())
            self.assertTrue(summary["converged"])
            self.assertEqual(summary["iterations"], 1)
            self.assertEqual(
                json.loads((directory / "history.json").read_text()), result.history
            )

    def test_exit_codes_preserve_convergence_status(self):
        for converged, expected in ((True, 0), (False, 2)):
            with self.subTest(converged=converged), tempfile.TemporaryDirectory() as tmp:
                with patch("tensorfvm.cylinder.SimpleSolver") as solver:
                    solver.return_value.solve.return_value = self.result(converged)
                    with contextlib.redirect_stdout(io.StringIO()):
                        code = main(["--nx", "20", "--ny", "10", "--output", tmp])
                self.assertEqual(code, expected)
                self.assertEqual(solver.call_args.args[0].mesh_type, "body-fitted")
                summary = json.loads((Path(tmp) / "summary.json").read_text())
                self.assertEqual(summary["converged"], converged)

    def test_threads_validation(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                main(["--threads", "0"])
        self.assertEqual(error.exception.code, 2)

    def test_cartesian_cli_remains_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("tensorfvm.cylinder.SimpleSolver") as solver:
                solver.return_value.solve.return_value = self.result()
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(main(["--mesh-type", "cartesian",
                                           "--output", tmp]), 0)
                self.assertEqual(solver.call_args.args[0].mesh_type, "cartesian")

    def test_fitted_coordinates_and_connectivity_export(self):
        from tensorfvm import SimpleSolver

        config = SolverConfig(nx=8, ny=4, mesh_type="body-fitted", max_iterations=1)
        result = SimpleSolver(config).solve()
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            export_result(result, directory)
            with (directory / "fields.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), config.nx * config.ny)
            self.assertAlmostEqual(float(rows[0]["x"]), result.x[0, 0].item())
            self.assertAlmostEqual(float(rows[0]["y"]), result.y[0, 0].item())
            self.assertTrue(all(row["fluid"] == "1" for row in rows))
            with (directory / "nodes.csv").open(newline="") as stream:
                nodes = list(csv.DictReader(stream))
            with (directory / "cells.csv").open(newline="") as stream:
                cells = list(csv.DictReader(stream))
            self.assertEqual(len(nodes), (config.nx + 1) * (config.ny + 1))
            self.assertEqual(len(cells), len(rows))
            self.assertTrue((directory / "velocity.svg").is_file())
            self.assertTrue((directory / "pressure.svg").is_file())
            for name in ("velocity.svg", "pressure.svg"):
                self.assertIn("<polygon", (directory / name).read_text())
            for cell in cells:
                for name in ("node0", "node1", "node2", "node3"):
                    self.assertLess(int(cell[name]), len(nodes))
            export_result(self.result(), directory)
            self.assertFalse((directory / "nodes.csv").exists())
            self.assertFalse((directory / "cells.csv").exists())

    def test_c_grid_exports_surface_pressure_and_force_coefficients(self):
        from tensorfvm.body_fitted import CGridMesh

        config = SolverConfig(nx=32, ny=8, mesh_type="c-grid")
        mesh = CGridMesh(config)
        field = torch.zeros((config.ny, config.nx), dtype=torch.float64)
        result = SimpleNamespace(
            config=config, mesh=mesh, p=field, fluid=field.bool(),
            x=mesh.centers[..., 0], y=mesh.centers[..., 1],
            cell_center_velocity=lambda: (field, field),
            history=[], converged=False,
            aerodynamic_coefficients={"drag": 0.1, "lift": 0.2},
        )
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            export_result(result, directory)
            with (directory / "airfoil.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), config.nx)
            self.assertTrue(all(float(row["cp"]) == 0 for row in rows))
            self.assertTrue((directory / "nodes.csv").exists())
            summary = json.loads((directory / "summary.json").read_text())
            self.assertEqual(summary["aerodynamic_coefficients"],
                             {"drag": 0.1, "lift": 0.2})
            self.assertTrue((directory / "velocity.svg").is_file())
            self.assertTrue((directory / "pressure.svg").is_file())

    def test_solver_failure_has_nonzero_exit(self):
        with patch("tensorfvm.cylinder.SimpleSolver", side_effect=ValueError("bad grid")):
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    main([])
        self.assertEqual(error.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
