"""Contract test for the CASE-CYLINDER-DRAG-FVM harness (synthetic data only).

The numbers below are *synthetic*: they exercise the comparison / refinement /
richardson code paths without claiming any physical result.  The physical run of
the case is produced by the cluster job and archived under
``artifacts/case-cylinder-drag-fvm/runs``.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "case_cylinder_drag_fvm.py"
REF = {"values": {"drag": 5.57953523384, "lift": 0.010618948146,
                  "pressure_difference_pa": 0.11752016697},
       "citation": "synthetic test fixture",
       "source_url": "local", "source_sha256": "0" * 64}


def write_run(directory: Path, resolution: str, cd: float) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    nx, ny = (int(v) for v in resolution.split("x"))
    (directory / "result.json").write_text(json.dumps({
        "resolution": resolution, "spacing_m": 2.2 / nx, "iterations": 7, "converged": True,
        "elapsed_s": 0.1, "computed": {"drag": cd, "lift": REF["values"]["lift"],
                                       "pressure_difference_pa": REF["values"]["pressure_difference_pa"]},
        "history": [{"momentum": 1e-2, "continuity": 1e-2}]}))
    (directory / "history.json").write_text(json.dumps(
        [{"momentum": 1e-2 / (i + 1), "continuity": 1e-2 / (i + 1)} for i in range(7)]))
    cells = 10
    fields = {
        "cell_centers_m": np.array([[0.02 + 0.2 * (i % 5), 0.05 + 0.3 * (i // 5)]
                                    for i in range(cells)]),
        "cell_pressure_pa": np.linspace(0.0, 0.12, cells),
        "cell_velocity_m_s": np.zeros((cells, 2)),
        "cylinder_pressure_force_n_m": np.array([[0.5, 0.0]]),
        "cylinder_viscous_force_n_m": np.array([[0.5, 0.0]]),
        "force_scale_n_m": np.array(0.1),
        "cylinder_angle_rad": np.linspace(0, 2 * np.pi, 8, endpoint=False),
        "cylinder_wall_pressure_pa": np.linspace(0.0, 0.1, 8),
    }
    np.savez_compressed(directory / "fields.npz", **fields)


def test_analyze_passes_and_fails_on_tolerance(tmp_path):
    runs = tmp_path / "runs"
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps(REF))
    # monotone refinement converging towards the reference value
    for resolution, cd in (("64x24", 5.70), ("128x48", 5.62), ("256x96", 5.59)):
        write_run(runs / resolution, resolution, cd)
    out = tmp_path / "analysis"
    proc = subprocess.run([sys.executable, str(SCRIPT), "analyze", "--runs", str(runs),
                           "--out", str(out), "--reference", str(reference),
                           "--tolerance", "0.05"], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    summary = json.loads((out / "summary.json").read_text())
    assert summary["case_passed"] is True
    assert summary["grids"][-1]["passed"] == {"drag": True, "lift": True,
                                              "pressure_difference_pa": True}
    assert summary["convergence"]["drag"]["available"] is True
    assert (out / "figures" / "cd_refinement.png").exists()

    # a 10 % drag deviation must fail the 5 % gate
    write_run(runs / "256x96", "256x96", 5.57953523384 * 1.10)
    proc = subprocess.run([sys.executable, str(SCRIPT), "analyze", "--runs", str(runs),
                           "--out", str(out), "--reference", str(reference),
                           "--tolerance", "0.05"], capture_output=True, text=True)
    assert proc.returncode == 2
    summary = json.loads((out / "summary.json").read_text())
    assert summary["case_passed"] is False
    assert summary["grids"][-1]["passed"]["drag"] is False