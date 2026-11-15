"""Compact knob sweep with early abort at |u|max > 50 m/s."""
import numpy as np
from tensorfvm.verification import thacker_basin as tb

n = 40
base = dict(tb.DEFAULTS)
variants = {
    "baseline": {},
    "order=1": {"order": 1},
    "cfl=0.2": {"cfl": 0.2},
    "dry=1e-4": {"dry_depth": 1e-4},
    "dry=1e-2": {"dry_depth": 1e-2},
    "dry=5e-2": {"dry_depth": 5e-2},
}
basin = tb.build_basin(**{k: base[k] for k in ("H", "a", "amplitude", "gravity")})
marks = (0.5, 1.0, 1.5, 2.0)
for label, over in variants.items():
    p = dict(base)
    p.update(over)
    s = tb._solver(basin, n, p)
    X, Y = tb.grid(basin, n, p["domain_factor"])
    s.set_state(*basin.initial_state(X, Y))
    hist = {}
    stop = {"why": ""}

    def cb(sol, s=s, hist=hist, stop=stop):
        for m in marks:
            if m not in hist and sol.time >= m * basin.period:
                hist[m] = sol.max_spurious_velocity()
        if sol.max_spurious_velocity() > 50.0:
            stop["why"] = "u>50 at t=%.3f" % sol.time
            raise SystemExit

    try:
        s.advance(2.0 * basin.period, max_steps=200000, callback=cb)
    except SystemExit:
        pass
    except RuntimeError as exc:
        stop["why"] = "runtime"
    print("%-10s " % label + " ".join(
        "T%.1f=%7.2f" % (m, hist.get(m, float("nan"))) for m in marks)
        + "  steps=%d dt=%.1e dryflip=%d %s"
        % (s.steps, s.dt_last, s.dry_flips, stop["why"]))