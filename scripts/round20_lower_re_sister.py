#!/usr/bin/env python3
"""Round 20 driver: lower-Reynolds sister cases for ``CASE-FVM-PERIODIC-HILLS``.

One configuration per process so the cases shard on the cluster.

The point of the round is a *controlled* comparison, so this driver deliberately
keeps every knob of the existing protocol fixed (terrain-following mesh family,
momentum convection blend 0.75, transport blend 0.0, ``dt``, averaging window
``t in [160, 400]``, certified wall-shear metric) and changes exactly two
things: the Reynolds number and the closure (``laminar`` = no closure provider).

Artefact naming
---------------
``round20_<closure>_Re<reynolds>_<tag>.{json,npz}``

Every artefact carries

* ``gate`` -- the *shared* round-9 round-11 gate, which is hard-wired to the
  ``Re = 2800`` reference.  Preserved for schema compatibility and explicitly
  flagged ``gate_reference_reynolds: 2800`` so nobody quotes it as the verdict
  of a ``Re = 700`` run;
* ``lower_re`` -- the round-20 verdict for the run's **own** Reynolds number,
  from :func:`tensorfvm.verification.periodic_hill_lower_re.score_case`, with
  ``x_r/H`` recomputed by the certified round-17 operator from the stored mean
  wall shear rather than read from the payload's derived field.

Usage
-----
    PYTHONPATH=src python scripts/round20_lower_re_sister.py \\
        --closure laminar --reynolds 700 --tag 96x48 \\
        --nx 96 --ny 48 --steps 20000 --burn-in 8000 --dt 0.02
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tensorfvm.verification.periodic_hill_lower_re import score_case
from tensorfvm.verification.periodic_hill_round11 import (
    run_time_averaged_case,
    strict_finite,
)

OUTDIR = Path(__file__).resolve().parents[1] / "docs" / "case-periodic-hills"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--closure", default="laminar", choices=("laminar", "sa", "sst"))
    ap.add_argument("--reynolds", type=float, required=True)
    ap.add_argument("--tag", default="96x48")
    ap.add_argument("--nx", type=int, default=96)
    ap.add_argument("--ny", type=int, default=48)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--burn-in", type=int, default=8000)
    ap.add_argument("--sample-every", type=int, default=20)
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--convection-blend", type=float, default=None)
    ap.add_argument("--transport-blend", type=float, default=None)
    ap.add_argument("--out-dir", default=str(OUTDIR))
    args = ap.parse_args(argv)

    import shutil
    import tempfile

    import numpy as np

    tmp = Path(tempfile.mkdtemp(prefix="round20_"))
    npz_path = None
    try:
        payload = run_time_averaged_case(
            args.tag, args.closure, nx=args.nx, ny=args.ny, steps=args.steps,
            sample_every=args.sample_every, burn_in_steps=args.burn_in, dt=args.dt,
            reynolds=args.reynolds, convection_blend=args.convection_blend,
            transport_convection_blend=args.transport_blend,
            verbose=True, out_dir=tmp, save_fields=True)

        # Score from the run's own stored mean wall shear with the certified
        # round-17 operator: the payload's x_r_over_H is a derived number and is
        # only used as a cross-check (it is compared below, never preferred).
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
    stem = f"round20_{args.closure}_Re{args.reynolds:g}_{args.tag}"
    (out / f"{stem}.json").write_text(
        json.dumps(strict_finite(payload), indent=1, allow_nan=False))
    if saved_npz is not None:
        shutil.copyfile(saved_npz, out / f"{stem}.npz")
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"[{args.closure}/Re{args.reynolds:g}/{args.tag}] "
          f"x_r/H={lower_re['x_r_over_H']} ref={lower_re['reference_x_r_over_H']} "
          f"err={lower_re['error']} verdict={lower_re['verdict']} "
          f"wall={payload['wall_clock_s']:.0f}s", flush=True)
    return 0 if lower_re["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())