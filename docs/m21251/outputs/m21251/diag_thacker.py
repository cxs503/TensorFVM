"""Diagnose the documented multi-period momentum injection in the Thacker basin."""
import numpy as np
from tensorfvm.verification import thacker_basin as tb

n = 40
params = dict(tb.DEFAULTS)
basin = tb.build_basin(**{k: params[k] for k in ("H", "a", "amplitude", "gravity")})
solver = tb._solver(basin, n, params)
X, Y = tb.grid(basin, n, params["domain_factor"])
h0, u0, v0 = basin.initial_state(X, Y)
solver.set_state(h0, u0, v0)
print("period T =", basin.period, " dx =", solver.dx, "dry_depth =", solver.dry_depth)
print("initial |u| max =", np.max(np.hypot(u0, v0)), " h range", h0.min(), h0.max())

tgrid = np.linspace(0.0, basin.period, 200)
umax_exact = 0.0
for t in tgrid:
    vel = basin.velocity(X, Y, t)
    uu = vel[0] if isinstance(vel, tuple) else vel
    umax_exact = max(umax_exact, float(np.nanmax(np.hypot(*[np.asarray(c) for c in (uu if isinstance(uu, tuple) else (vel[0], vel[1]))]))))
print("analytic max |u| in a period =", umax_exact)

rows = []


def cb(sol):
    h, u, v = sol.primitive()
    sp = np.hypot(u, v)
    rows.append((sol.steps, sol.time, sol.dt_last, float(sp.max()),
                 float(h.min()), float(h.max()),
                 sol.clipped_cells, sol.dry_flips, sol.total_volume()))


t_end = 2.0 * basin.period
try:
    solver.advance(t_end, max_steps=400000, callback=cb)
    print("completed")
except RuntimeError as exc:
    print("advance stopped:", exc)

r = np.array(rows, dtype=float)
print("steps run:", len(r), " t reached:", r[-1, 1], "/", t_end)
flag = r[:, 3] > 3.0 * umax_exact
bad = int(np.argmax(flag)) if flag.any() else len(r) - 1
print("first step with max|u| > 3x analytic bound:", int(r[bad, 0]) if flag.any() else "none")
for i in range(max(0, bad - 6), min(len(r), bad + 6)):
    print("step %5d t=%.6f dt=%.3e |u|max=%.4f hmin=%.3e hmax=%.3f clip=%d flip=%d vol=%.10g"
          % tuple(r[i]))
np.save("/srv/tensor/agent/work/m21251/outputs/m21251/diag_rows.npy", r)