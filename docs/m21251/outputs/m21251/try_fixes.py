"""Candidate stabilisers for the Thacker 2-period thin-film momentum runaway."""
import numpy as np
from tensorfvm.verification import thacker_basin as tb
from tensorfvm.shallow_water.swe2d import ShallowWater2D

N = 40
BASE = dict(tb.DEFAULTS)
basin = tb.build_basin(**{k: BASE[k] for k in ("H", "a", "amplitude", "gravity")})
orig_sanitise = ShallowWater2D._sanitise


def make_sanitise(mode, value):
    def _sanitise(self):
        orig_sanitise(self)
        h = self.h
        if mode == "cut":
            cut = value * self.h0_ref
            thin = h < cut
            self.hu = np.where(thin, 0.0, self.hu)
            self.hv = np.where(thin, 0.0, self.hv)
        elif mode == "froude":
            speed = np.hypot(self.hu, self.hv) / np.where(h > self.dry_depth, h, 1.0)
            limit = value * np.sqrt(self.gravity * np.maximum(h, self.dry_depth))
            excess = speed > limit
            if np.any(excess):
                scale = np.where(excess, limit / np.maximum(speed, 1e-300), 1.0)
                self.hu = self.hu * scale
                self.hv = self.hv * scale
    return _sanitise


def run(mode, value):
    ShallowWater2D._sanitise = make_sanitise(mode, value)
    try:
        out = tb.run_case(n=N, periods=2.0)
        hist = np.asarray(out["series"]["t_over_T"]) if "series" in out else None
        umbrella = np.asarray(out["series"]["t"]) if "series" in out else None
        return out
    finally:
        ShallowWater2D._sanitise = orig_sanitise


def report(label, out):
    c = out["criteria"]
    print("%-18s centre_L2=%.4f shoreline_mean=%.4f max=%.4f vol=%.2e "
          "steps=%d flip=%d C4=%s C5=%s C3=%s"
          % (label, out["centre_depth"]["l2_relative"],
             out["shoreline"]["mean_relative"], out["shoreline"]["max_relative"],
             out["volume"]["drift"], out["health"]["steps"],
             out["health"]["dry_flips"], c["C4_centre_depth"],
             c["C5_shoreline"], c["C3_volume"]))


ShallowWater2D.h0_ref = 1.0
print("=== baseline order-2 (unmodified) ===")
try:
    report("baseline", tb.run_case(n=N, periods=2.0))
except Exception as exc:
    print("baseline failed:", type(exc).__name__, exc)

print("=== order-1 control ===")
try:
    report("order1", tb.run_case(n=N, periods=2.0, order=1))
except Exception as exc:
    print("order1 failed:", type(exc).__name__, exc)

for ratio in (1e-3, 5e-3, 1e-2, 3e-2):
    print("=== thin-film momentum cut h < %.0e H ===" % ratio)
    try:
        report("cut=%.0e" % ratio, run("cut", ratio))
    except Exception as exc:
        print("cut=%.0e failed:" % ratio, type(exc).__name__, exc)

for fr in (1.0, 2.0, 3.0):
    print("=== Froude cap Fr_max = %.1f ===" % fr)
    try:
        report("Fr=%.1f" % fr, run("froude", fr))
    except Exception as exc:
        print("Fr=%.1f failed:" % fr, type(exc).__name__, exc)