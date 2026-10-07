"""Generate and audit the extruded body-fitted 3-D cylinder O-grid."""

import argparse
import csv
import json
import math
from dataclasses import asdict
from pathlib import Path

import torch

from .body_fitted3d import BodyFittedCylinderMesh3D
from .runtime import DistributedRuntime
from .solver3d import Cylinder3DConfig


def _node_ids(mesh: BodyFittedCylinderMesh3D) -> torch.Tensor:
    """Return global node ids for the local node array, including shared interfaces."""
    k = torch.arange(mesh.z_start, mesh.z_stop + 1, device=mesh.device)[:, None, None]
    j = torch.arange(mesh.ny + 1, device=mesh.device)[None, :, None]
    i = torch.arange(mesh.nx + 1, device=mesh.device)[None, None, :]
    return ((k * (mesh.ny + 1) + j) * (mesh.nx + 1) + i).to(torch.long)


def _global_connectivity(mesh: BodyFittedCylinderMesh3D) -> torch.Tensor:
    """Translate local hexahedron node ids to the global extruded-grid numbering."""
    local = mesh.connectivity
    stride_theta = mesh.nx + 1
    stride_z = (mesh.ny + 1) * stride_theta
    local_k = local // stride_z
    remainder = local % stride_z
    j = remainder // stride_theta
    i = remainder % stride_theta
    global_k = local_k + mesh.z_start
    return (global_k * stride_z + j * stride_theta + i).to(torch.long)


def _rank_suffix(runtime: DistributedRuntime) -> str:
    return "" if runtime.world_size == 1 else f"-rank{runtime.rank:04d}"


def export_mesh(mesh: BodyFittedCylinderMesh3D, config: Cylinder3DConfig,
                runtime: DistributedRuntime, directory: Path) -> dict:
    """Write rank-local mesh pieces and a root-owned global quality summary."""
    directory.mkdir(parents=True, exist_ok=True)
    suffix = _rank_suffix(runtime)
    node_ids = _node_ids(mesh).detach().cpu()
    nodes = mesh.vertices.detach().cpu()
    with (directory / f"nodes{suffix}.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("node", "x", "y", "z"))
        for k in range(mesh.local_nz + 1):
            for j in range(mesh.ny + 1):
                for i in range(mesh.nx + 1):
                    point = nodes[k, j, i]
                    writer.writerow((int(node_ids[k, j, i]), float(point[0]),
                                     float(point[1]), float(point[2])))
    connectivity = _global_connectivity(mesh).detach().cpu()
    with (directory / f"cells{suffix}.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("cell", "node0", "node1", "node2", "node3",
                         "node4", "node5", "node6", "node7"))
        for k in range(mesh.local_nz):
            for j in range(mesh.ny):
                for i in range(mesh.nx):
                    cell = ((mesh.z_start + k) * mesh.ny + j) * mesh.nx + i
                    writer.writerow((cell, *(int(node) for node in connectivity[k, j, i])))
    surface = mesh.cylinder_face_centers.detach().cpu()
    normal = mesh.cylinder_face_area_vectors.detach().cpu()
    with (directory / f"cylinder-faces{suffix}.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("span_layer", "theta_face", "x", "y", "z", "area_x", "area_y", "area_z"))
        for k in range(mesh.local_nz):
            for i in range(mesh.nx):
                value, area = surface[k, i], normal[k, i]
                writer.writerow((mesh.z_start + k, i, float(value[0]), float(value[1]),
                                 float(value[2]), float(area[0]), float(area[1]), float(area[2])))

    local_volume = mesh.cell_volumes.sum()
    local_area = mesh.cylinder_surface_area
    total_volume = float(runtime.global_sum(local_volume))
    total_area = float(runtime.global_sum(local_area))
    expected_volume = float(mesh.cross_section_area) * mesh.span
    expected_area = 2 * math.pi * mesh.cylinder_radius * mesh.span
    seam_error = float((mesh.vertices[:, :, 0] - mesh.vertices[:, :, -1]).abs().max())
    min_cell_volume = float(runtime.global_max((-mesh.cell_volumes).max()).neg())
    volume_relative_error = abs(total_volume - expected_volume) / expected_volume
    area_relative_error = abs(total_area - expected_area) / expected_area
    quality = {
        "global_cell_count": mesh.nx * mesh.ny * mesh.nz,
        "local_cell_count": mesh.cell_count,
        "global_node_count_with_seam": (mesh.nz + 1) * (mesh.ny + 1) * (mesh.nx + 1),
        "local_node_count_with_seam": mesh.node_count,
        "total_volume": total_volume,
        "expected_extruded_volume": expected_volume,
        "volume_relative_error": volume_relative_error,
        "minimum_cell_volume": min_cell_volume,
        "polygonal_cylinder_surface_area": total_area,
        "analytic_cylinder_surface_area": expected_area,
        "cylinder_surface_area_relative_error": area_relative_error,
        "theta_seam_max_error": seam_error,
    }
    passed = bool(math.isfinite(total_volume) and math.isfinite(total_area)
                  and min_cell_volume > 0 and seam_error == 0
                  and volume_relative_error < 1e-12)
    runtime.barrier()
    if runtime.rank == 0:
        summary = {
            "benchmark": "3d-body-fitted-cylinder-o-grid",
            "passed": passed,
            "runtime": {"device": str(runtime.device), "world_size": runtime.world_size},
            "config": asdict(config),
            "quality": quality,
            "output_partitioning": (
                "rank-local node/cell files share z-interface nodes; cell ids are global"
                if runtime.world_size > 1 else "single global mesh file set"
            ),
        }
        (directory / "summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    runtime.barrier()
    return {"passed": passed, "quality": quality, "world_size": runtime.world_size}


def run_mesh_benchmark(directory, nx: int = 48, ny: int = 24, nz: int = 12,
                       stretching: float = 2.5,
                       runtime: DistributedRuntime | None = None,
                       device: str = "cpu") -> dict:
    """Build, audit, and export a body-fitted O-grid without running a CFD solve."""
    runtime = runtime or DistributedRuntime.discover(device)
    config = Cylinder3DConfig(nx=nx, ny=ny, nz=nz, mesh_type="body-fitted",
                              body_fitted_stretching=stretching,
                              device=str(runtime.device))
    if runtime.world_size > config.nz:
        raise ValueError("world_size cannot exceed the global number of z planes")
    partition = runtime.partition_z(config.nz)
    mesh = BodyFittedCylinderMesh3D.build(config, runtime.device,
                                           partition.start, partition.stop)
    return export_mesh(mesh, config, runtime, Path(directory))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a distributed-ready 3-D body-fitted cylinder O-grid"
    )
    parser.add_argument("--output", type=Path, default=Path("results/cylinder-3d-o-grid"))
    parser.add_argument("--nx", type=int, default=48, help="circumferential cells; divisible by four")
    parser.add_argument("--ny", type=int, default=24, help="radial cells")
    parser.add_argument("--nz", type=int, default=12, help="global spanwise cells")
    parser.add_argument("--stretching", type=float, default=2.5,
                        help="exponential radial clustering toward the cylinder")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    try:
        runtime = DistributedRuntime.initialize_from_environment(args.device)
        summary = run_mesh_benchmark(args.output, args.nx, args.ny, args.nz,
                                     args.stretching, runtime=runtime)
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, f"3-D body-fitted mesh generation failed: {error}\n")
    if runtime.rank == 0:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
