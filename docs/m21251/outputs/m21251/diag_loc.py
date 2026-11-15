import numpy as np
from tensorfvm.verification import thacker_basin as tb
from tensorfvm.shallow_water import swe2d

n = 40
params = dict(tb.DEFAULTS)
basin = tb.build_basin(**{k: params[k] for k in ("H", "a", "amplitude", "gravity")})
solver = tb._solver(basin, n, params)
X, Y = tb.grid(basin, n, params["domain_factor"])
h0, u0, v0 = basin.initial_state(X, Y)
solver.set_state(h0, u0, v0)

print("primitive() ->", open("/srv/tensor/agent/work/m21251/src/tensorfvm/shallow_water/swe2d.py").read().split("def primitive")[1][:400])

target_steps = {80, 120, 200, 300}
grow = []


def cb(sol):
    if sol.steps not in target_steps:
        return
    h, u, v = sol.primitive()
    sp = np.hypot(u, v)
    i, j = np.unravel_index(np.argmax(sp), sp.shape)
    print("\n--- step %d t=%.4f (T=%.2f) max|u|=%.3f at (%d,%d) r=%.3f"
          % (sol.steps, sol.time, sol.time / basin.period, sp.max(), i, j,
             np.hypot(X[i, j], Y[i, j])))
    sl = (slice(max(i - 2, 0), i + 3), slice(max(j - 2, 0), j + 3))
    np.set_printoptions(precision=4, suppress=False, linewidth=200)
    print("h  =\n", h[sl])
    print("u  =\n", u[sl])
    print("bed=\n", solver.bed[sl])
    print("eta=\n", h[sl] + solver.bed[sl])
    print("hu=\n", solver.hu[sl])
    print("eta_analytic max over grid:", np.nanmax(basin.surface(X, Y, sol.time)))
    print("total momentum x:", float(solver.hu.sum() * solver.dx * solver.dy))


solver.advance(2.0 * basin.period, max_steps=400000, callback=cb)