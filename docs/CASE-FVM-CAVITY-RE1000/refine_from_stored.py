"""Refine a stored cavity field to a finer grid (no coarse re-run).

Used by the uncertainty study (``docs/CASE-FVM-CAVITY-RE1000/UNCERTAINTY.md``) to
extend the self-convergence sequence beyond n = 128 without paying for a coarse
re-march: the n = 128 field is read from ``runs/fields.npz``, bilinearly
prolonged with the case module's own transfer operator and marched to steady
state on the finer grid.  Prolongation only supplies an initial guess (the
continuation control in ``runs/control`` shows cold and prolonged starts reach
the same fixed point), so this is a cost trick, not a change of problem.

    PYTHONPATH=src python docs/CASE-FVM-CAVITY-RE1000/refine_from_stored.py \
        --from-n 128 --n 192 --out-dir runs/refine
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tensorfvm.case_cavity_re1000 import prolong, run_resolution, write_json  # noqa: E402
from tensorfvm.lid_cavity import LidCavityConfig, LidCavitySolver  # noqa: E402


def load_field(npz: pathlib.Path, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    data = np.load(npz)
    return (torch.as_tensor(data[f"u_{n}"]).double(), torch.as_tensor(data[f"v_{n}"]).double())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    here = pathlib.Path(__file__).resolve().parent
    ap.add_argument("--fields", default=str(here / "runs" / "fields.npz"))
    ap.add_argument("--from-n", type=int, default=128)
    ap.add_argument("--n", type=int, default=192)
    ap.add_argument("--cfl", type=float, default=0.15)
    ap.add_argument("--tolerance", type=float, default=1e-07)
    ap.add_argument("--max-steps", type=int, default=250_000)
    ap.add_argument("--out-dir", default=str(here / "runs" / "refine"))
    args = ap.parse_args()

    coarse_u, coarse_v = load_field(pathlib.Path(args.fields), args.from_n)
    seed = LidCavitySolver(LidCavityConfig(n=args.from_n, reynolds=1000.0,
                                           lid_velocity=1.0, cfl=args.cfl,
                                           tolerance=args.tolerance,
                                           max_steps=args.max_steps))
    seed.u, seed.v = coarse_u, coarse_v
    u0, v0 = prolong(seed, args.n)

    rec, solver = run_resolution(args.n, init=(u0, v0),
                                 init_from=f"n={args.from_n} bilinear prolongation",
                                 cfl=args.cfl, tolerance=args.tolerance,
                                 max_steps=args.max_steps)
    out = pathlib.Path(args.out_dir)
    write_json(out / f"lid_cavity_Re1000_n{args.n}.json", rec)
    np.savez_compressed(out / f"field_n{args.n}.npz",
                        **{f"u_{args.n}": solver.u.detach().cpu().numpy(),
                           f"v_{args.n}": solver.v.detach().cpu().numpy()})
    print(json.dumps({"n": args.n, "steps": rec["steps"], "verdict": rec["verdict"],
                      "converged": rec["converged"],
                      "u_min_rel": rec["criteria"]["C3_u_min_relative"]["value"],
                      "v_max_rel": rec["criteria"]["C4_v_max_relative"]["value"]}, indent=1))


if __name__ == "__main__":
    main()