"""Resolution / scheme ladder for CASE-WOODWARD-COLELLA (two interacting blast waves).

Why this exists
---------------
``docs/CASE-WOODWARD-COLELLA-reference-verification.md`` left exactly one metric
unfinished: the shallow near-vacuum trough (``low_plateau``) sits 14.2 % above the
digitised published reference at nx = 400 **and** at nx = 800, i.e. it does not
converge with the grid.  Two readings are possible and they have opposite
consequences:

* the trough minimum is a *discretisation* error that keeps shrinking, so the
  800-grid value is simply not the continuum value, or
* the trough minimum is *scheme/reference* dependent and will never converge to
  the digitised value at any affordable resolution (then the metric is
  ill-conditioned, not the solver wrong).

This script produces the data that separates the two readings: the same case on
a longer grid ladder (nx = 1600, 3200) plus one off-scheme run (LLF/Rusanov
instead of HLLC) at fixed resolution.  It only runs the solver and stores raw
profiles + metrics; all the judgement lives in
``wc_plateau_convergence.py`` so it can be redone without recomputing.

Usage (repo root, cluster job):

    python docs/woodward-colella-double-blast/ladder/wc_ladder.py \
        --plan 1600:hllc,3200:hllc,1600:rusanov --outdir <dir>

    # sharded form, one plan entry per shard ({i} is 1-based)
    ... --plan ... --shard 1 --shards 3
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from tensorfvm.compressible import wc_reference as wcr
from tensorfvm.compressible.cases import DoubleBlastConfig
from tensorfvm.compressible.double_blast import _run


#: plan entry fields, in order, all but ``nx`` optional
PLAN_FIELDS = ("nx", "flux_mode", "limiter", "cfl", "order")
DEFAULTS = {"flux_mode": "hllc", "limiter": "mc", "cfl": 0.4, "order": 2}
CASTS = {"nx": int, "flux_mode": str, "limiter": str, "cfl": float, "order": int}


def parse_plan(spec: str) -> list[dict]:
    """``"1600:hllc,3200:hllc:minmod:0.2"`` -> list of run configurations.

    Fields are positional: ``nx[:flux_mode[:limiter[:cfl[:order]]]]``; omitted
    fields fall back to :data:`DEFAULTS`.  The point of the extra knobs is a
    *scheme-sensitivity* control at fixed resolution: if a metric moves a lot
    when only the limiter or the time step changes, that metric is measuring
    the scheme, not the solution.
    """
    out: list[dict] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) > len(PLAN_FIELDS):
            raise ValueError(f"too many fields in plan entry: {item!r}")
        cfg: dict = dict(DEFAULTS)
        for name, raw in zip(PLAN_FIELDS, parts):
            cfg[name] = CASTS[name](raw)
        if cfg["nx"] <= 0:
            raise ValueError(f"bad nx in plan: {item!r}")
        if cfg["flux_mode"] not in ("hllc", "gfm"):
            raise ValueError(f"unknown flux mode {cfg['flux_mode']!r}")
        out.append(cfg)
    if not out:
        raise ValueError("empty plan")
    return out


def tag_of(cfg: dict) -> str:
    """File-name tag: ``nx1600_hllc``, plus the non-default knobs."""
    bits = [f"nx{cfg['nx']}", cfg["flux_mode"]]
    for name in ("limiter", "cfl", "order"):
        if cfg[name] != DEFAULTS[name]:
            bits.append(f"{name}{cfg[name]}")
    return "_".join(bits)


def select(plan: list[dict], shard: int | None, shards: int | None):
    """One plan entry per shard (shard numbering is 1-based, as in the queue)."""
    if not shard:
        return plan
    n = shards or 1
    if not 1 <= shard <= n:
        raise ValueError(f"shard {shard} outside 1..{n}")
    mine = plan[shard - 1::n]
    if not mine:
        raise ValueError(f"shard {shard}/{n} of a {len(plan)}-entry plan is empty")
    return mine


def one(cfg: dict, t_end: float, reference: dict) -> dict:
    t0 = time.time()
    nx = cfg["nx"]
    run = _run(nx, t_end, flux_mode=cfg["flux_mode"], limiter=cfg["limiter"],
               cfl=cfg["cfl"], order=cfg["order"])
    x = np.asarray(run["x"], dtype=float).ravel()
    rho = np.asarray(run["rho"], dtype=float).ravel()
    p = np.asarray(run["p"], dtype=float).ravel()
    u = np.asarray(run["u"], dtype=float).ravel()
    order = np.argsort(x)
    x, rho, p, u = x[order], rho[order], p[order], u[order]

    m = wcr.metrics_from_profiles(reference["x"], reference["rho"], x, rho)
    m.update({
        "nx": nx,
        "dx": 1.0 / nx,
        "config": cfg,
        "flux_mode": cfg["flux_mode"],
        "order": run["order"],
        "limiter": run["limiter"],
        "cfl": run["cfl"],
        "t_end": t_end,
        "steps": run["steps"],
        "seconds_solver": run["seconds"],
        "seconds_total": round(time.time() - t0, 2),
        "mass_energy_drift": run["drift"],
        "positivity_fixes": run["positivity_fixes"],
        "reference": reference["path"],
    })
    profile = {"x": x, "rho": rho, "p": p, "u": u}
    return {"metrics": m, "profile": profile}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", default="1600:hllc")
    ap.add_argument("--shard", type=int, default=None)
    ap.add_argument("--shards", type=int, default=None)
    ap.add_argument("--t-end", type=float, default=DoubleBlastConfig().t_end)
    ap.add_argument("--outdir", type=Path,
                    default=Path("docs/woodward-colella-double-blast/ladder/data"))
    args = ap.parse_args()

    plan = select(parse_plan(args.plan), args.shard, args.shards)
    args.outdir.mkdir(parents=True, exist_ok=True)
    reference = wcr.load_reference()

    for cfg in plan:
        got = one(cfg, args.t_end, reference)
        m = got["metrics"]
        tag = tag_of(cfg)
        np.savez_compressed(args.outdir / f"ladder_profile_{tag}.npz",
                            x=got["profile"]["x"], rho=got["profile"]["rho"],
                            p=got["profile"]["p"], u=got["profile"]["u"])
        (args.outdir / f"ladder_metrics_{tag}.json").write_text(
            json.dumps(m, indent=2) + "\n")
        print(json.dumps({
            "tag": tag, "nx": cfg["nx"], "config": cfg,
            "peak_rel_percent": m["peak_density"]["rel_error_percent"],
            "l1_percent": m["l1_relative"]["percent"],
            "low_plateau_percent": m["low_plateau"]["rel_error_percent"],
            "low_plateau_got": m["low_plateau"]["got"],
            "seconds": m["seconds_total"],
            "verdict": m["verdict"],
        }))


if __name__ == "__main__":
    main()