"""Face-state statistics: is the order-2 growth a wet/dry face-extrapolation effect?"""
import numpy as np
from tensorfvm.verification import thacker_basin as tb
from tensorfvm.shallow_water.swe2d import ShallowWater2D

n = 40
p = dict(tb.DEFAULTS)
basin = tb.build_basin(**{k: p[k] for k in ("H", "a", "amplitude", "gravity")})
solver = tb._solver(basin, n, p)
X, Y = tb.grid(basin, n, p["domain_factor"])
solver.set_state(*basin.initial_state(X, Y))

orig = ShallowWater2D._face_states
stats = []


def wrapped(self, axis):
    out = orig(self, axis)
    hL, unL, utL, hR, unR, utR, hsL, hsR = out
    h, u, v = self.primitive()
    uc = np.max(np.abs(u)) if axis == 0 else np.max(np.abs(v))
    uf = max(np.max(np.abs(unL)), np.max(np.abs(unR)))
    # faces where both own-side depths are wet but one star depth is tiny
    both_wet = (hL > self.dry_depth) & (hR > self.dry_depth)
    stats.append((self.steps, self.time, axis, uf, uc, float(np.min(hL)),
                  float(np.min(hR)), int(np.count_nonzero(both_wet)),
                  float(np.max(np.where(both_wet, np.minimum(hsL, hsR), np.inf)))))
    return out


ShallowWater2D._face_states = wrapped
marks = {0.25, 0.5, 0.75, 1.0, 1.5, 2.0}
seen = set()


def cb(sol):
    for m in marks - seen:
        if sol.time >= m * basin.period:
            seen.add(m)


solver.advance(2.0 * basin.period, max_steps=400000, callback=cb)
s = np.array([[x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8]] for x in stats])
for axis in (0, 1):
    sel = s[:, 2] == axis
    print("axis", axis, "n_faces", sel.sum())
    # amplification of face velocity over cell velocity, per recorded step group
    for i in range(0, len(s), max(1, len(s) // 80)):
        if s[i, 2] != axis:
            continue
        print("  step %4d t=%.3f u_face_max=%.3f u_cell_max=%.3f ratio=%.2f hminL=%.2e hminR=%.2e wetfaces=%d min(hsL,hsR)_max=%.2e"
              % (s[i, 0], s[i, 1], s[i, 3], s[i, 4], s[i, 3] / max(s[i, 4], 1e-30),
                 s[i, 5], s[i, 6], s[i, 7], s[i, 8]))