#!/usr/bin/env python3
"""Round 25 driver: complete the ``CASE-FVM-PERIODIC-HILLS`` grid ladder at
CFL-stable time steps.

Round 22/23 pre-registered a closure-free ``Re = 2800`` grid ladder
(``96x48`` -> ``144x72`` -> ``192x96`` -> ``240x120``) and submitted the three
refined rungs at the *coarse* protocol ``dt`` (0.0133 / 0.01 / 0.008).  All
three diverged inside the startup transient (``diverged_at_step`` = 508 / 538 /
652, ``samples = 0``), so the pre-registered attribution returned
``inconclusive`` -- the ladder was never actually measured.

Round 23's cheap probe (``scripts/round23_divergence_probe.py``) answered the
question that round 22 left open: the blow-up is a **time-step stability limit
that tightens with the mesh**, not a mesh-intrinsic failure.  At fixed mesh the
halved / quartered ``dt`` keeps ``max|u|`` bounded:

* ``144x72`` at ``dt = 0.00665`` (CFL ~0.16) -- bounded, ``max|u|`` ~1.6;
* ``192x96`` at ``dt = 0.0025``  (CFL ~0.08) -- bounded, ``max|u|`` ~1.7;
* ``240x120`` at ``dt = 0.004``   (CFL ~0.16) -- bounded (same CFL as the
  ``144x72`` rung that round 23 proved stable).

This driver therefore re-runs the three refined rungs at those **stable** time
steps, keeping every other knob of the round-20 protocol fixed (terrain-following
mesh family, momentum convection blend 0.75, transport blend 0.0, averaging
window ``t in [160, 400]``, certified round-17 wall-shear operator), and writes
``round25_laminar_Re2800_<tag>`` artefacts so the round-23 historical record
(the diverged ``round20_laminar_*`` rungs) is not overwritten.

Artefact naming
---------------
``round25_<closure>_Re<reynolds>_<tag>.{json,npz}``

Usage
-----
    PYTHONPATH=src python scripts/round25_stable_ladder.py \\
        --closure laminar --reynolds 2800 --tag 144x72 \\
        --nx 144 --ny 72 --steps 60000 --burn-in 24000 --dt 0.00665
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

from tensorfvm.verification.periodic_hill_lower_re import score_case
from tensorfvm.verification.periodic_hill_round11 import (
    run_time_averaged_case,
    strict_finite,
)

OUTDIR = Path(__file__).resolve().parents[1] / "docs" / "case-periodic-hills"

#: the stable-dt rungs this round completes (tag -> (nx, ny, dt, steps, burn-in)).
#: ``dt`` values are the ones round 23's probe proved bounded at fixed mesh.
STABLE_RUNGS = {
    "144x72":  dict(nx=144, ny=72,  dt=0.00665, steps=60151, burn_in=24061),
    "192x96":  dict(nx=192, ny=96,  dt=0.00250, steps=160000, burn_in=64000),
    "240x120": dict(nx=240, ny=120, dt=0.00400, steps=100000, burn_in=40000),
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--closure", default="laminar", choices=("laminar", "sa", "sst"))
    ap.add_argument("--reynolds", type=float, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--nx", type=int, required=True)
    ap.add_argument("--ny", type=int, required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--burn-in", type=int, required=True)
    ap.add_argument("--dt", type=float, required=True)
    ap.add_argument("--sample-every", type=int, default=20)
    ap.add_argument("--convection-blend", type=float, default=None)
    ap.add_argument("--transport-blend", type=float, default=None)
    ap.add_argument("--out-dir", default=str(OUTDIR))
    args = ap.parse_args(argv)

    tmp = Path(tempfile.mkdtemp(prefix="round25_"))
    npz_path = None
    try:
        payload = run_time_averaged_case(
            args.tag, args.closure, nx=args.nx, ny=args.ny, steps=args.steps,
            sample_every=args.sample_every, burn_in_steps=args.burn_in, dt=args.dt,
            reynolds=args.reynolds, convection_blend=args.convection_blend,
            transport_convection_blend=args.transport_blend,
            verbose=True, out_dir=tmp, save_fields=True)
        npz_path = next(tmp.glob("*.npz"), None)
        mean_tau_b = x_over_h = None
        if npz_path is not None:
            with np.load(npz_path) as npz:
                mean_tau_b = np.asarray(npz["mean_tau_b"])
                x_over_h = np.asarray(npz["x_over_h"], dtype=float)
    finally:
        saved_npz = npz_path

    lower_re = score_case(payload, reynolds=args.reynolds,
                          mean_tau_b=mean_tau_b, x_over_h=x_over_h)
    payload["lower_re"] = strict_finite(lower_re)
    payload["gate_reference_reynolds"] = 2800.0
    payload["gate_note"] = (
        "payload['gate'] is the shared gate hard-wired to the Re=2800 reference "
        "(5.41); the verdict for THIS run's Reynolds number is payload['lower_re']")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"round25_{args.closure}_Re{args.reynolds:g}_{args.tag}"
    (out / f"{stem}.json").write_text(
        json.dumps(strict_finite(payload), indent=1, allow_nan=False))
    if saved_npz is not None:
        shutil.copyfile(saved_npz, out / f"{stem}.npz")
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"[round25/{args.closure}/Re{args.reynolds:g}/{args.tag}] "
          f"x_r/H={lower_re['x_r_over_H']} ref={lower_re['reference_x_r_over_H']} "
          f"err={lower_re['error']} verdict={lower_re['verdict']} "
          f"wall={payload['wall_clock_s']:.0f}s", flush=True)
    return 0 if lower_re["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
