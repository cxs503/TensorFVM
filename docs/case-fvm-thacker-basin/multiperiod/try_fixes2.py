"""Measure the candidate stabilisers over the full 2-period Thacker window."""
import numpy as np
from tensorfvm.verification import thacker_basin as tb
from tensorfvm.shallow_water.swe2d import ShallowWater2D

N = 40
BASE = dict(tb.DEFAULTS)
basin = tb.build_basin(**{k: BASE[k] for k in ("H", "a", "amplitude", "gravity")})
orig = ShallowWater2D._sanitise
ShallowWater2D.h0_ref = 1.0


def make(mode, value):
    def _sanitise(self):
        orig(self)
        h = self.h
        if mode == "cut":
            thin = h < value * self.h0_ref
            self.hu = np.where(thin, 0.0, self.hu)
            self.hv = np.where(thin, 0.0, self.hv)
        else:
            speed = np.hypot(self.hu, self.hv) / np.where(h > self.dry_depth, h, 1.0)
            limit = value * np.sqrt(self.gravity * np.maximum(h, self.dry_depth))
            ex = speed > limit
            if np.any(ex):
                sc = np.where(ex, limit / np.maximum(speed, 1e-300), 1.0)
                self.hu, self.hv = self.hu * sc, self.hv * sc
    return _sanitise


def run(label, mode=None, value=None, **kw):
    if mode is not None:
        ShallowWater2D._sanitise = make(mode, value)
    try:
        out = tb.run_case(n=N, periods=2.0, **kw)
    finally:
        ShallowWater2D._sanitise = orig
    ser = out["series"]
    keys = list(out.keys())
    print("%-12s C4_centre_L2=%.4f%% C5_shore_mean=%.4f%% max=%.4f%% "
          "%s steps=%d flip=%d hmin=%.2e"
          % (label, 100 * out["centre_depth"]["l2_relative"],
             100 * out["shoreline"]["mean_relative"],
             100 * out["shoreline"]["max_relative"],
             {k: (round(v, 6) if isinstance(v, float) else v)
              for k, v in out["volume"].items()},
             out["health"]["steps"], out["health"]["dry_flips"],
             out["health"]["h_min"]))
    np.save("/root/tp-agent/work/m21251/outputs/m21251/series_%s.npy" % label.replace("=", "").replace("%", ""),
            np.vstack([ser["t_over_T"], ser["h_centre"], ser["s"]]))
    return out


run("baseline")
run("order1", order=1)
run("cut1e-2", "cut", 1e-2)
run("cut3e-2", "cut", 3e-2)
run("Fr1", "froude", 1.0)
run("Fr2", "froude", 2.0)