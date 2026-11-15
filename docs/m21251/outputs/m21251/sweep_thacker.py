"""Knob sweep for the Thacker multi-period velocity growth (n=40, 2 periods)."""
import numpy as np
from tensorfvm.verification import thacker_basin as tb

n = 40
base = dict(tb.DEFAULTS)
variants = {
    "baseline (order2, cfl0.4, dry1e-8)": {},
    "order=1": {"order": 1},
    "cfl=0.2": {"cfl": 0.2},
    "dry_depth=1e-4": {"dry_depth": 1e-4},
    "dry_depth=1e-2": {"dry_depth": 1e-2},
    "dry_depth=1e-1": {"dry_depth": 1e-1},
    "not well balanced": {"well_balanced": False},
}

marks = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
for label, over in variants.items():
    params = dict(base)
    params.update(over)
    basin = tb.build_basin(**{k: params[k] for k in ("H", "a", "amplitude", "gravity")})
    wb = params.pop("well_balanced", True)
    solver = tb._solver(basin, n, params, well_balanced=wb)
    X, Y = tb.grid(basin, n, params["domain_factor"])
    solver.set_state(*basin.initial_state(X, Y))
    hist = {}
    ends = [m * basin.period for m in marks]

    def cb(sol, ends=ends, hist=hist):
        for e in ends:
            if e not in hist and sol.time >= e - 1e-12:
                hist[e] = sol.max_spurious_velocity()

    try:
        solver.advance(2.0 * basin.period, max_steps=400000, callback=cb)
        err = ""
    except RuntimeError as exc:
        err = "STOPPED: %s" % exc
    vals = ["%.2f" % hist.get(e, float("nan")) for e in ends]
    print("%-34s |u|max @ T=0.25..2.0: %s  steps=%d flip=%d clip=%d %s"
          % (label, " ".join("%8s" % v for v in vals), solver.steps,
             solver.dry_flips, solver.clipped_cells, err))