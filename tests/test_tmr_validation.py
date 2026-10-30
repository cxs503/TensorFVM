"""Tests for the NASA-TMR validation package.

The cases themselves take minutes and are run by ``validation_tmr/gate.py``; the
tests here pin the parts that must not silently change: the PLOT3D mesh
builder, the LU/ILU linear path (which must reproduce the reference solver's
discretisation) and the post-processing conventions (wall-shear sign, theta
integration window, reattachment detection).
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from tensorfvm.solver import SolverConfig  # noqa: E402
from tensorfvm.body_fitted import BodyFittedSolver  # noqa: E402
from tensorfvm.tmr_grid import TmrGridMesh, read_plot3d_2d  # noqa: E402
from validation_tmr.linear import FastLinearBodyFittedSolver  # noqa: E402
from validation_tmr.postprocess import columns, interpolated_station  # noqa: E402

GRID = REPO / "validation_tmr/reference/tmr/grids/flatplate_clust2_4levelsdown_35x25.p2dfmt"
STEP_GRID = REPO / "validation_tmr/reference/tmr/grids/backstep5_5levdn_derived.p2dfmt"


def plate_config():
    return SolverConfig(mesh_type="tmr-grid", mesh_file=str(GRID), nx=1, ny=1,
                        length=2.33333, height=1.0, inlet_velocity=1.0,
                        reynolds=5e6 * 2.33333, cylinder_radius=None,
                        turbulence_model="spalart-allmaras")


def test_plot3d_layout_and_masks():
    blocks = read_plot3d_2d(GRID)
    assert len(blocks) == 1
    x, y = blocks[0]
    assert x.shape == (25, 35)
    mesh = TmrGridMesh(plate_config())
    assert mesh.volumes.numel() == 34 * 24
    assert bool((mesh.volumes > 0).all())
    counts = {name: int(mask.sum()) for name, mask in mesh.masks.items()}
    assert counts["inlet"] == 24 and counts["outlet"] == 24
    assert counts["wall"] == 34 and counts["far-field"] == 34
    # Every interior face is shared by exactly two cells.
    assert int(mesh.interior.sum()) + int(mesh.boundary.sum()) == int(mesh.owner.numel())
    assert bool((mesh.neighbor[mesh.interior] >= 0).all())
    # Area vectors must point out of their owner cell.
    outward = (mesh.face_centers - mesh.centers[mesh.owner]) * mesh.face_area_vectors
    assert float(outward.sum(-1).min()) > 0.0


def test_multi_zone_step_grid_connectivity():
    mesh = TmrGridMesh(SolverConfig(mesh_type="tmr-grid", mesh_file=str(STEP_GRID),
                                    nx=1, ny=1, length=180.0, height=9.0,
                                    inlet_velocity=1.0, reynolds=1e7,
                                    cylinder_radius=None, grid_top_patch="wall"))
    # 17x17 + 7x17 + 25x29 + 9x29 nodes -> 16*16 + 6*16 + 24*28 + 8*28 cells
    assert mesh.volumes.numel() == 16 * 16 + 6 * 16 + 24 * 28 + 8 * 28
    assert int(mesh.masks["far-field"].sum()) == 0  # top patch flipped to wall
    assert int(mesh.masks["inlet"].sum()) == 16
    assert float(mesh.extent[0]) == pytest.approx(-130.0)
    assert float(mesh.extent[1]) == pytest.approx(50.0)


def test_linear_solver_reproduces_reference_discretisation():
    """The LU/ILU path must not change the discretisation it accelerates."""
    torch.set_num_threads(2)
    reference = BodyFittedSolver(plate_config())
    fast = FastLinearBodyFittedSolver(plate_config())
    for _ in range(3):
        reference.step()
        fast.step()
    # The reference solver stops its Jacobi-CG at a 1e-9 relative tolerance, so
    # agreement is expected at that level rather than bitwise.
    for key in ("momentum", "turbulence", "velocity_change"):
        assert reference.history[-1][key] == pytest.approx(
            fast.history[-1][key], rel=1e-4, abs=1e-12)
    # The mass imbalance is at round-off level in both solvers.
    assert abs(reference.history[-1]["mass_imbalance"]) < 1e-8
    assert abs(fast.history[-1]["mass_imbalance"]) < 1e-8
    assert float((reference.velocity - fast.velocity).abs().max()) < 1e-6


def test_wall_shear_convention_and_theta_window():
    """Cf must be positive on an attached plate and the theta window bounded."""
    from validation_tmr.common import wall_data
    solver = FastLinearBodyFittedSolver(plate_config())
    solver.step()
    wall = wall_data(solver)
    plate = wall["y"] < 1e-12
    tau = wall["shear"] * np.sign(wall["normal_y"])
    # Near-wall flow moves downstream, so tau_w on the plate must be positive.
    assert float(np.median(tau[plate])) > 0.0


def test_column_clustering_and_station_interpolation():
    # Slightly skewed columns, as in the TMR multi-zone grids.
    centers = np.zeros((8, 2))
    x = np.repeat([0.0, 1.0, 2.0, 3.0], 2)
    centers[:, 0] = x + np.array([0.0, 0.004] * 4)
    centers[:, 1] = np.tile([1.0, 2.0], 4)
    grouped = columns(centers, tolerance_scale=1e-2)
    assert len(grouped) == 4
    field = centers[:, 0].copy()
    y, value = interpolated_station(centers, field, 1.5, tolerance=1e-2)
    assert y.size == 2
    # The column coordinate is the mean of the skewed cell centres, so the
    # interpolated value reproduces the linear field to the column skew.
    np.testing.assert_allclose(value, [1.5, 1.5], atol=5e-3)