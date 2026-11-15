#!/usr/bin/env python3
"""Round-16 acceptance runs for ``CASE-FVM-PERIODIC-HILLS`` (one config/process).

Round 16 found a real defect in the SST ``omega`` destruction linearisation
(rounds 10--15 used the *Newton* coefficient ``2 beta omega`` on the implicit
diagonal with no compensating ``+beta omega_old^2`` on the right-hand side, so
the model equilibrium was ``omega`` low by ``sqrt(2)`` and ``nu_t`` high by up to
``2x``) and fixed it to the Patankar coefficient ``beta omega``.  The evidence
for the fix, however, stopped at a ``32x16`` / 2000-step trend probe
(``docs/case-periodic-hills/round16_local_ab.json``); the ``96x48`` runs that
``ROUND16.md`` was supposed to quote were never submitted, and no ``ROUND16.md``
exists.  This driver produces them.

    PYTHONPATH=src python3 scripts/round16_acceptance.py \
        --variant patankar --nx 96 --ny 48 --steps 20000 --tag r16_patankar_96x48
    PYTHONPATH=src python3 scripts/round16_acceptance.py \
        --variant newton2  --nx 96 --ny 48 --steps 20000 --tag r16_newton2_96x48

``--variant newton2`` reproduces rounds 10--15 bit-for-bit by restoring the old
sink coefficient through a method override (the HEAD solver is Patankar; there is
no configuration switch, because a switch that is not wired is exactly the round
10--14 failure mode).  Which coefficient was actually in force is *measured*
from ``SSTKOmega.source_terms()`` and written into the artefact as
``sink_coefficient_ratio`` (``1.0`` = Patankar, ``2.0`` = legacy): the artefact
cannot claim a variant it did not run.

Outputs: ``docs/case-periodic-hills/<tag>.json`` (round-11 schema + the round-15
``score_hill`` verdict + the round-11 ``gate_summary`` gate).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from tensorfvm.verification.periodic_hill_round11 import (
    build_solver, gate_summary, run_time_averaged_case)
from tensorfvm.verification.periodic_hill_round15 import score_hill, write_json

OUTDIR = Path(__file__).resolve().parents[1] / "docs" / "case-periodic-hills"


def measure_sink_ratio(solver) -> dict:
    """Measured ``S_P / (beta omega)`` of the ``omega`` destruction term.

    ``1.0`` is the model-consistent Patankar coefficient; ``2.0`` is the
    rounds-10--15 Newton coefficient.  Measured, not assumed.
    """
    turb = solver.turbulence
    src = turb.source_terms()
    beta = np.asarray(src["beta"], dtype=float)
    omega = np.asarray(turb.omega, dtype=float)
    coeff = np.asarray(src["omega_destruction_linear"], dtype=float)
    mask = (np.abs(beta * omega) > 1e-30) & np.isfinite(coeff)
    if not np.any(mask):
        return {"ratio": None, "samples": 0}
    ratio = coeff[mask] / (beta[mask] * omega[mask])
    return {
        "ratio": float(np.median(ratio)),
        "min": float(np.min(ratio)),
        "max": float(np.max(ratio)),
        "samples": int(mask.sum()),
    }


def apply_legacy_newton_sink(solver) -> None:
    """Restore the rounds-10--15 (defective) ``omega`` sink coefficient."""
    turb = solver.turbulence
    original = turb.source_terms

    def source_terms():
        src = dict(original())
        src["omega_destruction_linear"] = 2.0 * np.asarray(src["beta"]) * turb.omega
        return src

    turb.source_terms = source_terms


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", choices=("patankar", "newton2"), default="patankar")
    ap.add_argument("--nx", type=int, default=96)
    ap.add_argument("--ny", type=int, default=48)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--burn-in", type=int, default=8000)
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--reynolds", type=float, default=2800.0)
    ap.add_argument("--sample-every", type=int, default=20)
    ap.add_argument("--momentum-blend", type=float, default=0.75)
    ap.add_argument("--transport-blend", type=float, default=0.0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--outdir", default=str(OUTDIR))
    args = ap.parse_args(argv)

    tag = args.tag or f"r16_{args.variant}_{args.nx}x{args.ny}"
    solver = build_solver(
        "sst", nx=args.nx, ny=args.ny, reynolds=args.reynolds, dt=args.dt,
        convection_blend=args.momentum_blend,
        transport_convection_blend=args.transport_blend)
    if args.variant == "newton2":
        apply_legacy_newton_sink(solver)
    sink = measure_sink_ratio(solver)

    t0 = time.time()
    payload = run_time_averaged_case(
        tag, "sst", nx=args.nx, ny=args.ny, steps=args.steps,
        sample_every=args.sample_every, burn_in_steps=args.burn_in, dt=args.dt,
        reynolds=args.reynolds, solver=solver, out_dir=None,
        save_fields=False, verbose=True, progress_every=max(args.steps // 8, 1))
    wall = time.time() - t0

    payload["round"] = 16
    payload["variant"] = args.variant
    payload["sink_coefficient_ratio"] = sink
    payload["omega_sink"] = ("patankar (beta*omega, model-consistent)"
                             if args.variant == "patankar"
                             else "newton (2*beta*omega, rounds 10-15 legacy)")
    payload["wall_clock_s"] = wall
    payload["gate"] = gate_summary(payload)
    payload["score_hill"] = score_hill(payload)

    out = write_json(payload, Path(args.outdir) / f"{tag}.json")
    summary = {
        "artefact": str(out),
        "variant": args.variant,
        "sink_coefficient_ratio": sink["ratio"],
        "mesh": f"{args.nx}x{args.ny}",
        "steps": args.steps,
        "x_r_over_H": payload["time_averaged"].get("x_r_over_H"),
        "x_r_error": payload["score_hill"].get("x_r_over_H_error"),
        "stationary": payload["stationarity"].get("x_r_over_H_block_spread_rel"),
        "profile_pass_stations": (f"{payload['score_hill']['profile_stations_pass']}"
                                  f"/{payload['score_hill']['profile_stations_admissible']}"),
        "pass": payload["gate"]["pass"],
        "reasons": payload["gate"]["reasons"],
        "wall_clock_s": wall,
    }
    print(json.dumps(summary, indent=1))
    return 0 if payload["gate"]["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())