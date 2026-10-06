"""Executable distributed 3-D external-cylinder smoke case."""

import argparse
import csv
import json
import math
from pathlib import Path

import torch

from .runtime import DistributedRuntime
from .solver3d import Cylinder3DConfig, Cylinder3DSolver


def _write_midspan(result, directory: Path) -> None:
    """Write the global midspan plane from the single z rank that owns it."""
    global_k = result.config.nz // 2
    if not result.partition.start <= global_k < result.partition.stop:
        return
    local_k = global_k - result.partition.start
    velocity = result.velocity[local_k].detach().cpu()
    pressure = result.pressure[local_k].detach().cpu()
    fluid = result.fluid[local_k].detach().cpu()
    x, y = result.x[local_k].detach().cpu(), result.y[local_k].detach().cpu()
    with (directory / "midspan.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("x", "y", "u", "v", "w", "p", "fluid"))
        for j in range(result.config.ny):
            for i in range(result.config.nx):
                writer.writerow((float(x[j, i]), float(y[j, i]),
                                 float(velocity[j, i, 0]), float(velocity[j, i, 1]),
                                 float(velocity[j, i, 2]), float(pressure[j, i]),
                                 int(fluid[j, i])))


def export_result(result, directory: Path) -> None:
    """Export one global midspan plane and globally reduced time histories.

    Every rank executes this function.  The rank owning the global midspan
    writes the field plane; rank zero writes JSON histories after a barrier, so
    a ``torchrun`` job creates one coherent portable result directory rather
    than one competing file set per rank.
    """
    directory.mkdir(parents=True, exist_ok=True)
    _write_midspan(result, directory)
    result.runtime.barrier()
    if result.runtime.rank == 0:
        for name, value in (("history.json", result.history), ("forces.json", result.force_history)):
            (directory / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n",
                                          encoding="utf-8")
        final = result.history[-1] if result.history else {}
        summary = {
            "steps": len(result.history),
            "runtime": {"device": str(result.runtime.device), "rank": result.runtime.rank,
                        "world_size": result.runtime.world_size},
            "config": result.config.__dict__,
            "rank_zero_z_partition": {"start": result.partition.start, "stop": result.partition.stop},
            "final": final,
            "force": result.force_history[-1] if result.force_history else {},
        }
        (directory / "summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    result.runtime.barrier()


def run_benchmark(directory, steps: int = 50, nx: int = 48, ny: int = 32, nz: int = 12,
                  runtime: DistributedRuntime | None = None, device: str = "cpu",
                  pressure_iterations: int = 150,
                  pressure_relative_tolerance: float = 1e-8,
                  pressure_absolute_tolerance: float = 1e-11) -> dict:
    """Run a bounded distributed Re=3900 smoke case, not a validated LES benchmark."""
    runtime = runtime or DistributedRuntime.discover(device)
    config = Cylinder3DConfig(
        nx=nx, ny=ny, nz=nz, max_steps=steps, pressure_iterations=pressure_iterations,
        pressure_relative_tolerance=pressure_relative_tolerance,
        pressure_absolute_tolerance=pressure_absolute_tolerance, time_step=0.005,
        smagorinsky_constant=0.1, device=str(runtime.device)
    )
    result = Cylinder3DSolver(config, runtime=runtime).solve()
    export_result(result, Path(directory))
    final = result.history[-1]
    finite = all(math.isfinite(value) for value in final.values())
    # This smoke criterion checks finite low-CFL projection advancement,
    # pressure-residual convergence, and distributed data motion.  It
    # intentionally makes no Cd/St accuracy claim.
    passed = bool(finite and final["cfl"] < 1 and final["pressure_converged"] == 1)
    return {"benchmark": "3d-cylinder-re3900-distributed-smoke", "passed": passed,
            "final": final, "steps": steps, "grid": [nx, ny, nz],
            "world_size": runtime.world_size}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Distributed 3-D Re=3900 cylinder projection/SGS smoke case")
    parser.add_argument("--output", type=Path, default=Path("results/cylinder-3d"))
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--nx", type=int, default=48)
    parser.add_argument("--ny", type=int, default=32)
    parser.add_argument("--nz", type=int, default=12)
    parser.add_argument("--pressure-iterations", type=int, default=150,
                        help="maximum distributed PCG pressure iterations per physical step")
    parser.add_argument("--pressure-relative-tolerance", type=float, default=1e-8)
    parser.add_argument("--pressure-absolute-tolerance", type=float, default=1e-11)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or a CUDA device such as cuda:0")
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    try:
        runtime = DistributedRuntime.initialize_from_environment(args.device)
        summary = run_benchmark(
            args.output, args.steps, args.nx, args.ny, args.nz, runtime=runtime,
            pressure_iterations=args.pressure_iterations,
            pressure_relative_tolerance=args.pressure_relative_tolerance,
            pressure_absolute_tolerance=args.pressure_absolute_tolerance,
        )
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, f"3-D cylinder case failed: {error}\n")
    if runtime.rank == 0:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
