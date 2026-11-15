#!/usr/bin/env python3
"""Round-16 local A/B: the SST ``omega`` sink linearisation on a coarse hill grid.

Same solver, same seed, same budget; the only difference is the implicit sink
coefficient of the ``omega`` destruction:

* ``patankar``  (round 16, HEAD): ``S_P = beta omega``      -> model equilibrium
* ``newton2``   (rounds 10--15):  ``S_P = 2 beta omega``    -> ``omega`` low by sqrt(2)

The coarse 32x16 grid is a *trend* probe, not an acceptance number; the 96x48
runs are the ones quoted in ``docs/case-periodic-hills/ROUND16.md``.

    PYTHONPATH=src python3 scripts/round16_local_ab.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np

from tensorfvm.verification.periodic_hill_round11 import (
    build_solver, gate_summary, run_time_averaged_case)


def _legacy_newton_coefficient(solver):
    """Reproduce rounds 10--15 exactly: double the omega sink coefficient."""
    turb = solver.turbulence
    original = turb.source_terms
    beta = turb.config.beta_star  # unused, kept for clarity

    def source_terms():
        src = original()
        src = dict(src)
        src["omega_destruction_linear"] = 2.0 * np.asarray(src["beta"]) * turb.omega
        return src

    turb.source_terms = source_terms
    return beta


def run(variant, nx=32, ny=16, steps=2000, burn_in=1000):
    solver = build_solver("sst", nx=nx, ny=ny)
    if variant == "newton2":
        _legacy_newton_coefficient(solver)
    payload = run_time_averaged_case(
        f"{variant}_{nx}x{ny}", "sst", nx=nx, ny=ny, steps=steps,
        burn_in_steps=burn_in, sample_every=20, solver=solver, out_dir=None,
        save_fields=False)
    gate = gate_summary(payload)
    return {
        "variant": variant,
        "mesh": f"{nx}x{ny}",
        "steps": steps,
        "x_r_over_H": gate["x_r_over_H"],
        "x_r_over_H_error": gate["x_r_over_H_error"],
        "max_nu_t_ratio": gate["max_nu_t_ratio"],
        "mean_nu_t_ratio": float(np.mean(payload["series"]["mean_nu_t_ratio"])),
        "stationary": gate["stationary"],
        "profile_pass": f"{gate['profile_stations_pass']}/{gate['profile_stations']}",
    }


OUT = Path(__file__).resolve().parents[1] / "docs" / "case-periodic-hills"


def main(argv=None):
    nx = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    ny = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    rows = [run(v, nx=nx, ny=ny) for v in ("patankar", "newton2")]
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "round": 16,
        "kind": "coarse-grid A/B of the omega sink linearisation (trend probe)",
        "mesh": f"{nx}x{ny}",
        "steps": rows[0]["steps"],
        "note": ("trend only: 2000 steps = 40 time units, non-stationary; the "
                 "acceptance numbers come from the 96x48 runs"),
        "rows": rows,
    }
    (OUT / "round16_local_ab.json").write_text(json.dumps(payload, indent=1))
    print(json.dumps(payload, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())