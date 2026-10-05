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
                summary = json.loads((Path(tmp) / "summary.json").read_text())
                self.assertEqual(summary["converged"], converged)

    def test_threads_validation(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                main(["--threads", "0"])
        self.assertEqual(error.exception.code, 2)

    def test_solver_failure_has_nonzero_exit(self):
        with patch("tensorfvm.cylinder.SimpleSolver", side_effect=ValueError("bad grid")):
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    main([])
        self.assertEqual(error.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
