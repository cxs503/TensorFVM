#!/usr/bin/env python
"""Run a single Rayleigh number case for the heated cavity benchmark."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from tensorfvm.internal_flow.natural_convection import NaturalConvectionCase, NaturalConvectionConfig
from tensorfvm.internal_flow.reference import DE_VAHL_DAVIS_NU, DE_VAHL_DAVIS_VELOCITY

OUT = Path("artifacts/heated-cavity-benchmark")


def run_case(rayleigh: float, meshes: list[int], max_steps: int = 600000,
             tolerance: float = 2e-6, cfl: float = 0.5) -> dict:
    torch.set_num_threads(2)
    started = time.time()
    ledger: dict = {"rayleigh": rayleigh, "prandtl": 0.71, "meshes": {}}
    previous = None
    for n in meshes:
        cfg = NaturalConvectionConfig(
            n=n, rayleigh=rayleigh, prandtl=0.71, cfl=cfl,
            max_steps=max_steps, tolerance=tolerance, check_every=200,
            wall_gradient="second_order",
        )
        case = NaturalConvectionCase(cfg)
        if previous is not None:
            case.solver.prolong_from(previous.solver)
        wall = time.time()
        block = max(2000, 50 * n)
        history: list = []
        stationary = 0
        previous_nu = None
        steps = 0
        while steps < max_steps:
            case.solver.run(max_steps=block, tolerance=1e-30)
            steps += block
            current = case.diagnostics()["nu_mean"]
            history.append((steps, current))
            if previous_nu is not None and abs(current - previous_nu) <= 1e-3 * abs(current):
                stationary += 1
                if stationary >= 2:
                    break
            else:
                stationary = 0
            previous_nu = current
        diag = case.diagnostics()
        diag["wall_seconds"] = time.time() - wall
        diag["steps"] = steps
        diag["converged"] = stationary >= 2
        diag["nu_history"] = history
        ledger["meshes"][str(n)] = diag
        previous = case
        print(f"[cavity] Ra={rayleigh:g} n={n} steps={steps} Nu={diag['nu_mean']:.5f} "
              f"ref={DE_VAHL_DAVIS_NU.get(rayleigh, 'N/A')} "
              f"converged={diag['converged']} ({diag['wall_seconds']:.0f}s)", flush=True)
    ledger["wall_seconds_total"] = time.time() - started
    return ledger


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rayleigh", type=float, required=True)
    parser.add_argument("--meshes", default="32,64,128")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    ra = args.rayleigh
    meshes = [int(x) for x in args.meshes.split(",")]

    ledger = run_case(ra, meshes)

    ref_nu = DE_VAHL_DAVIS_NU.get(ra)
    ref_vel = DE_VAHL_DAVIS_VELOCITY.get(ra, {})
    finest = meshes[-1]
    data = ledger["meshes"][str(finest)]
    nu_err = abs(data["nu_mean"] - ref_nu) / ref_nu if ref_nu else None
    u_err = abs(data["u_max_over_alpha"] - ref_vel.get("u_max", 0)) / ref_vel["u_max"] if ref_vel.get("u_max") else None
    v_err = abs(data["v_max_over_alpha"] - ref_vel.get("v_max", 0)) / ref_vel["v_max"] if ref_vel.get("v_max") else None

    acceptance = {
        "mesh": finest,
        "nu_reference": ref_nu,
        "nu_computed": data["nu_mean"],
        "nu_relative_error": nu_err,
        "nu_within_5_percent": bool(nu_err is not None and nu_err < 0.05),
        "u_max_reference": ref_vel.get("u_max"),
        "u_max_computed": data["u_max_over_alpha"],
        "u_max_relative_error": u_err,
        "v_max_reference": ref_vel.get("v_max"),
        "v_max_computed": data["v_max_over_alpha"],
        "v_max_relative_error": v_err,
        "converged": data["converged"],
    }

    # Save individual case result
    case_path = OUT / f"case_Ra{int(ra)}.json"
    case_path.write_text(json.dumps({"case": ledger, "acceptance": acceptance}, indent=2))
    print(f"\nCase saved to {case_path}")

    status = "PASS" if acceptance["nu_within_5_percent"] else "FAIL"
    print(f"Ra={int(ra)}: Nu_ref={ref_nu:.4f} Nu_got={data['nu_mean']:.4f} err={nu_err*100:.2f}% [{status}]")
    if ref_vel.get("u_max"):
        print(f"  u_max: ref={ref_vel['u_max']:.2f} got={data['u_max_over_alpha']:.2f} err={u_err*100:.2f}%")
        print(f"  v_max: ref={ref_vel['v_max']:.2f} got={data['v_max_over_alpha']:.2f} err={v_err*100:.2f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
