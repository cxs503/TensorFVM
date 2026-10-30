"""Shared helpers for the NASA-TMR validation cases.

Everything that determines a reported number is written to disk: the grid file
hash, the reference-data hash, the solver settings, the residual history, the
cell fields and the per-face wall data.  The gate module recomputes every error
from those files, so no number in the report is transcribed by hand.
"""

import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from tensorfvm.solver import SolverConfig
from validation_tmr.linear import FastLinearBodyFittedSolver

REPO = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment():
    return dict(python=subprocess.run(
        ["python3", "-c", "import sys; print(sys.version.replace(chr(10), ' '))"],
        capture_output=True, text=True).stdout.strip(),
        torch=torch.__version__, numpy=np.__version__,
        threads=torch.get_num_threads(), device="cpu",
        timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


def wall_traction(solver, mask):
    """Per-face wall traction vector, mirroring ``_surface_force`` exactly.

    The molecular wall gradient is reconstructed at the face from the owner
    cell gradient plus an exact no-slip correction, so the result is the same
    discretisation the solver uses for integrated forces.
    """
    owners = solver.o[mask]
    gradient = solver._gradient(solver.velocity)
    wall_d = solver.d[mask]
    wall_gradient = gradient[owners].clone()
    error = -solver.velocity[owners] - torch.einsum("fi,fij->fj", wall_d, wall_gradient)
    wall_gradient += (wall_d[:, :, None] * error[:, None, :]
                      / wall_d.square().sum(-1)[:, None, None])
    stress = solver.config.viscosity * (wall_gradient + wall_gradient.transpose(-1, -2))
    return torch.einsum("fij,fj->fi", stress, solver.S[mask])


def wall_data(solver):
    """Return per-face wall shear, pressure and centre coordinates."""
    mesh = solver.mesh
    mask = mesh.masks["wall"]
    traction = wall_traction(solver, mask)
    area = mesh.face_lengths[mask]
    centers = mesh.face_centers[mask]
    normals = mesh.face_normals[mask]
    surface_stress = traction / area[:, None]
    # Wall shear is the *tangential* part of the surface traction; projecting
    # out the normal component is what separates Cf from the wall pressure.
    normal_stress = (surface_stress * normals).sum(-1)
    tangential = surface_stress - normal_stress[:, None] * normals
    owners = solver.o[mask]
    return dict(area=area.detach().numpy(),
                x=centers[:, 0].detach().numpy(),
                y=centers[:, 1].detach().numpy(),
                shear=tangential[:, 0].detach().numpy(),
                shear_normal=tangential[:, 1].detach().numpy(),
                normal_stress=normal_stress.detach().numpy(),
                pressure=solver.p[owners].detach().numpy(),
                normal_x=normals[:, 0].detach().numpy(),
                normal_y=normals[:, 1].detach().numpy())


def build_solver(grid, viscosity, wall_at_top, device="cpu"):
    """Construct the fast collocated solver on an official TMR PLOT3D grid.

    ``length`` and ``height`` only set the reference scales; the geometry comes
    entirely from the grid file, and the viscosity is passed in directly so the
    Reynolds number of the case does not depend on a bounding box.
    """
    from tensorfvm.tmr_grid import read_plot3d_2d
    blocks = read_plot3d_2d(grid)
    xs = np.concatenate([x.ravel() for x, _ in blocks])
    ys = np.concatenate([y.ravel() for _, y in blocks])
    length = float(xs.max() - xs.min())
    height = float(ys.max() - ys.min())
    reynolds = 1.0 * length / viscosity
    config = SolverConfig(
        mesh_type="tmr-grid", mesh_file=str(grid), nx=1, ny=1,
        length=length, height=height, inlet_velocity=1.0, reynolds=reynolds,
        density=1.0, cylinder_radius=None, turbulence_model="spalart-allmaras",
        sa_freestream_ratio=3.0, velocity_relaxation=0.7, pressure_relaxation=0.3,
        tolerance=1e-5, max_iterations=20000, device=device,
        grid_top_patch="wall" if wall_at_top else "far-field")
    solver = FastLinearBodyFittedSolver(config)
    return solver


def restore(solver, path):
    """Restart the iteration from a checkpointed cell field (same grid only)."""
    import numpy as np
    with np.load(path) as handle:
        velocity = torch.tensor(handle["velocity"], dtype=torch.float64)
        pressure = torch.tensor(handle["pressure"], dtype=torch.float64)
        nu_tilde = torch.tensor(handle["nu_tilde"], dtype=torch.float64)
    if velocity.shape != solver.velocity.shape:
        raise ValueError("checkpoint field does not match the current mesh")
    solver.velocity[:] = velocity
    solver.p[:] = pressure
    solver.nu_tilde[:] = nu_tilde
    solver.turbulent_kinematic_viscosity = solver._sa_eddy_viscosity(solver.nu_tilde)
    return dict(restarted_from=str(path), iterations=0)


def run(solver, output, label, max_iterations, tolerance, log_every=50,
        checkpoint_every=500):
    """Iterate, streaming a residual history and checkpoints to disk.

    Checkpointing every ``checkpoint_every`` iterations means the artefacts of a
    long run stay usable even if the iteration budget is exhausted before the
    solver's own (stricter) convergence flag flips.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = solver.config
    config.max_iterations = max_iterations
    config.tolerance = tolerance
    history = []
    started = time.perf_counter()
    for iteration in range(max_iterations):
        solver.step()
        row = dict(solver.history[-1])
        row["elapsed_seconds"] = time.perf_counter() - started
        history.append(row)
        if iteration % log_every == 0 or solver.converged:
            (output / f"{label}-history.json").write_text(json.dumps(history) + "\n")
            print(f"{label} iter {iteration} momentum {row.get('momentum'):.3e} "
                  f"turbulence {row.get('turbulence', float('nan')):.3e} "
                  f"change {row.get('velocity_change', float('nan')):.2e} "
                  f"{row['elapsed_seconds']:.0f}s", flush=True)
        if iteration and iteration % checkpoint_every == 0:
            save_state(solver, output, label, history)
        if solver.converged:
            break
    (output / f"{label}-history.json").write_text(json.dumps(history) + "\n")
    return history


def save_state(solver, output, label, history):
    output = Path(output)
    mesh = solver.mesh
    field = output / f"{label}-fields.npz"
    np.savez_compressed(
        field,
        centers=mesh.centers.detach().numpy(),
        volumes=mesh.volumes.detach().numpy(),
        velocity=solver.velocity.detach().numpy(),
        pressure=solver.p.detach().numpy(),
        nu_tilde=solver.nu_tilde.detach().numpy(),
        nu_t=solver.turbulent_kinematic_viscosity.detach().numpy(),
        wall_distance=solver.wall_distance.detach().numpy())
    wall_file = output / f"{label}-wall.npz"
    np.savez_compressed(wall_file, **wall_data(solver))
    return dict(fields=str(field.name), wall=str(wall_file.name),
                fields_sha256=sha256(field), wall_sha256=sha256(wall_file),
                iterations=len(history), converged=bool(solver.converged),
                last_residual={k: float(v) for k, v in history[-1].items()},
                runtime_seconds=float(history[-1]["elapsed_seconds"]),
                cells=int(mesh.volumes.numel()),
                faces=int(mesh.owner.numel()),
                zones=[list(map(int, z)) for z in mesh.zone_dims],
                extent=list(map(float, mesh.extent)),
                linear_solves=len(solver.linear_diagnostics))


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")