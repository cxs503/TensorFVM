"""WENO5 ladder for CASE-WOODWARD-COLELLA (independent-scheme discriminator).

The MUSCL/HLLC solution of the two interacting blast waves is ~14 % above the
digitised published reference inside the shallow low-density region (the
"plateau"), and that number does **not** improve between nx = 400 and 800.  Two
explanations compete:

* our second-order finite-volume scheme is too diffusive there (then an
  independent, less diffusive scheme should land closer to the reference), or
* the metric (a window minimum sitting on the *edge* of the comparison window)
  is ill-conditioned and part of the number is measurement, not physics.

This script runs the independent WENO5 / SSP-RK3 / Rusanov code from
``docs/woodward-colella-double-blast/xcode/wc_crosscode.py`` (which shares no
discretisation with TensorFVM) on the same long ladder, using exactly the same
comparison window and the same metrics as ``wc_reference.py``, so the two
families can be compared cell for cell.  It stores raw profiles + metrics only;
all judgement is in ``wc_plateau_convergence.py``.

    python docs/woodward-colella-double-blast/ladder/wc_ladder_weno5.py \
        --sizes 1600,3200 --outdir <dir>
    # sharded: one size per shard ({i} is 1-based)
    ... --shard 1 --shards 2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from tensorfvm.compressible import wc_reference as wcr
from tensorfvm.compressible.cases import DoubleBlastConfig

_XCODE = Path(__file__).resolve().parents[1] / "xcode"
if str(_XCODE) not in sys.path:
    sys.path.insert(0, str(_XCODE))


def select(sizes: list[int], shard: int | None, shards: int | None) -> list[int]:
    if not shard:
        return sizes
    n = shards or 1
    if not 1 <= shard <= n:
        raise ValueError(f"shard {shard} outside 1..{n}")
    mine = sizes[shard - 1::n]
    if not mine:
        raise ValueError(f"shard {shard}/{n} of {len(sizes)} sizes is empty")
    return mine


def one(nx: int, t_end: float, reference: dict) -> dict:
    from wc_crosscode import weno5_solve  # noqa: E402  (path set above)

    t0 = time.time()
    run = weno5_solve(nx, t_end=t_end)
    x = np.asarray(run["x"], dtype=float)
    rho = np.asarray(run["rho"], dtype=float)
    p = np.asarray(run["p"], dtype=float)
    u = np.asarray(run["u"], dtype=float)
    order = np.argsort(x)
    x, rho, p, u = x[order], rho[order], p[order], u[order]

    m = wcr.metrics_from_profiles(reference["x"], reference["rho"], x, rho)
    m.update({
        "nx": nx,
        "dx": 1.0 / nx,
        "method": run.get("method", "weno5"),
        "cfl": run.get("cfl"),
        "t_end": t_end,
        "steps": run.get("steps"),
        "seconds_solver": run.get("seconds"),
        "seconds_total": round(time.time() - t0, 2),
        "mass_drift": run.get("mass_drift"),
        "positivity_fallbacks": run.get("positivity_fallbacks"),
        "reference": reference["path"],
    })
    return {"metrics": m, "profile": {"x": x, "rho": rho, "p": p, "u": u}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", default="1600,3200")
    ap.add_argument("--shard", type=int, default=None)
    ap.add_argument("--shards", type=int, default=None)
    ap.add_argument("--t-end", type=float, default=DoubleBlastConfig().t_end)
    ap.add_argument("--outdir", type=Path,
                    default=Path("docs/woodward-colella-double-blast/ladder/data"))
    args = ap.parse_args()

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    todo = select(sizes, args.shard, args.shards)
    args.outdir.mkdir(parents=True, exist_ok=True)
    reference = wcr.load_reference()

    for nx in todo:
        got = one(nx, args.t_end, reference)
        m = got["metrics"]
        tag = f"nx{nx}_weno5"
        np.savez_compressed(args.outdir / f"ladder_profile_{tag}.npz",
                            x=got["profile"]["x"], rho=got["profile"]["rho"],
                            p=got["profile"]["p"], u=got["profile"]["u"])
        (args.outdir / f"ladder_metrics_{tag}.json").write_text(
            json.dumps(m, indent=2) + "\n")
        print(json.dumps({
            "tag": tag, "nx": nx,
            "peak_rel_percent": m["peak_density"]["rel_error_percent"],
            "l1_percent": m["l1_relative"]["percent"],
            "low_plateau_percent": m["low_plateau"]["rel_error_percent"],
            "low_plateau_got": m["low_plateau"]["got"],
            "seconds": m["seconds_total"],
        }))


if __name__ == "__main__":
    main()