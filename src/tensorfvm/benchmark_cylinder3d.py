"""Executable 3-D external-cylinder smoke case for the common 3-D kernel."""

import argparse
import csv
import json
import math
from pathlib import Path

import torch

from .solver3d import Cylinder3DConfig, Cylinder3DSolver


def export_result(result, directory: Path):
    """Export portable midspan fields and the full time/force histories."""
    directory.mkdir(parents=True, exist_ok=True)
    k = result.config.nz // 2
    velocity = result.velocity[k].detach().cpu()
    pressure = result.pressure[k].detach().cpu()
    fluid = result.fluid[k].detach().cpu()
    x, y = result.x[k].detach().cpu(), result.y[k].detach().cpu()
    with (directory / "midspan.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("x", "y", "u", "v", "w", "p", "fluid"))
        for j in range(result.config.ny):
            for i in range(result.config.nx):
                writer.writerow((float(x[j, i]), float(y[j, i]),
                                 float(velocity[j, i, 0]), float(velocity[j, i, 1]),
                                 float(velocity[j, i, 2]), float(pressure[j, i]),
                                 int(fluid[j, i])))
    for name, value in (("history.json", result.history), ("forces.json", result.force_history)):
        (directory / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n",
                                      encoding="utf-8")
    final = result.history[-1] if result.history else {}
    summary = {
        "steps": len(result.history),
        "runtime": {"device": str(result.runtime.device), "rank": result.runtime.rank,
                    "world_size": result.runtime.world_size},
        "config": result.config.__dict__,
        "final": final,
        "force": result.force_history[-1] if result.force_history else {},
    }
    (directory / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def run_benchmark(directory, steps=50, nx=48, ny=32, nz=12):
    """Run a bounded Re=3900 3-D LES smoke case, not a validated LES benchmark."""
    config = Cylinder3DConfig(nx=nx, ny=ny, nz=nz, max_steps=steps,
                              pressure_iterations=150, time_step=0.005,
                              smagorinsky_constant=0.1)
    result = Cylinder3DSolver(config).solve()
    export_result(result, Path(directory))
    final = result.history[-1]
    finite = all(math.isfinite(value) for value in final.values())
    # This smoke criterion checks a finite, low-CFL 3-D projection advance. It
    # intentionally makes no Cd/St accuracy claim on a stair-step cylinder.
    passed = bool(finite and final["cfl"] < 1)
    return {"benchmark": "3d-cylinder-re3900-smoke", "passed": passed,
            "final": final, "steps": steps, "grid": [nx, ny, nz]}


def main(argv=None):
    parser = argparse.ArgumentParser(description="3-D Re=3900 cylinder projection/SGS smoke case")
    parser.add_argument("--output", type=Path, default=Path("results/cylinder-3d"))
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--nx", type=int, default=48)
    parser.add_argument("--ny", type=int, default=32)
    parser.add_argument("--nz", type=int, default=12)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    try:
        summary = run_benchmark(args.output, args.steps, args.nx, args.ny, args.nz)
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, f"3-D cylinder case failed: {error}\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
