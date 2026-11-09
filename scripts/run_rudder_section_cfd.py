"""2-D body-fitted RANS of the KVLCC2 rudder section at one incidence.

The rudder of KVLCC2 is idealised as a 0.18-thickness symmetric (NACA-4-digit
type) section at the model-scale chord; the flow is solved with the repository's
body-fitted SIMPLE solver (:class:`tensorfvm.body_fitted.BodyFittedSolver`) with
Spalart-Allmaras turbulence and a C-grid, which is the same solver family already
verified against the NACA benchmark cases in ``docs/external-boundary``.

Output (single JSON line prefixed with ``RESULT_JSON``) contains the section
lift/drag coefficients, the iteration count, the convergence state, the final
continuity/momentum residuals and the mesh/configuration record -- i.e. everything
the report needs in order to trace a `c_l` back to a solver run.  Nothing is
estimated: if the solve does not converge, ``converged`` is false and the caller
must handle it (the campaign script drops such points and records the failure).

Usage::

    PYTHONPATH=src python scripts/run_rudder_section_cfd.py \
        --angle-deg 10 --level medium --max-iterations 4000
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import time

LEVELS = {
    "coarse": (96, 32),
    "medium": (160, 48),
    "fine": (256, 72),
}

#: KVLCC2 model rudder chord and design speed (see tensorfvm.maneuvering.hull).
CHORD = 0.1705737535654457
SPEED = 1.1765
NU = 1.139e-6
THICKNESS = "0018"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--angle-deg", type=float, required=True)
    parser.add_argument("--level", choices=sorted(LEVELS), default="medium")
    parser.add_argument("--max-iterations", type=int, default=4000)
    parser.add_argument("--turbulence", default="spalart-allmaras",
                        choices=("spalart-allmaras", "laminar"))
    parser.add_argument("--length", type=float, default=12.0)
    parser.add_argument("--height", type=float, default=8.0)
    parser.add_argument("--reynolds", type=float,
                        default=SPEED * CHORD / NU)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)

    import torch
    torch.set_num_threads(args.threads)
    from tensorfvm.solver import SolverConfig
    from tensorfvm.body_fitted import BodyFittedSolver

    nx, ny = LEVELS[args.level]
    config = SolverConfig(
        nx=nx, ny=ny, length=args.length, height=args.height,
        cylinder_radius=None, airfoil_code=THICKNESS, airfoil_chord=CHORD,
        airfoil_x=0.35 * args.length, airfoil_y=0.5 * args.height,
        angle_of_attack=args.angle_deg, inlet_velocity=SPEED,
        reynolds=args.reynolds, density=998.2, mesh_type="c-grid",
        turbulence_model=args.turbulence, max_iterations=args.max_iterations,
        tolerance=1e-6)
    started = time.time()
    solver = BodyFittedSolver(config)
    result = solver.solve()
    elapsed = time.time() - started

    def coefficient(axis):
        forces = result.surface_forces.get("airfoil", {"x": 0.0, "y": 0.0})
        alpha = math.radians(args.angle_deg)
        scale = 0.5 * config.density * config.inlet_velocity ** 2 * CHORD
        if axis == "drag":
            return (forces["x"] * math.cos(alpha) + forces["y"] * math.sin(alpha)) / scale
        return (-forces["x"] * math.sin(alpha) + forces["y"] * math.cos(alpha)) / scale

    history = result.history[-1] if result.history else {}
    payload = {
        "angle_deg": args.angle_deg,
        "level": args.level, "mesh": {"nx": nx, "ny": ny,
                                      "length": args.length, "height": args.height},
        "cl": float(coefficient("lift")),
        "cd": float(coefficient("drag")),
        "converged": bool(result.converged),
        "iterations": len(result.history),
        "last_metrics": {k: float(v) for k, v in (history or {}).items()
                         if isinstance(v, (int, float))},
        "wall_seconds": elapsed,
        "reynolds": args.reynolds, "speed": SPEED, "chord": CHORD,
        "thickness_ratio": 0.18, "turbulence_model": args.turbulence,
        "python": platform.python_version(),
        "machine": platform.node(),
    }
    print("RESULT_JSON " + json.dumps(payload), flush=True)
    return 0 if result.converged else 3


if __name__ == "__main__":
    raise SystemExit(main())