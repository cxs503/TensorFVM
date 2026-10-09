"""Launch a labeled overset cylinder run with the rebuilt projection.

Usage:  python3 -u run_probe14.py [nsteps] [out_prefix] [probe_every]
"""
import sys
import overset_cylinder_fvm as M

nsteps = int(sys.argv[1]) if len(sys.argv) > 1 else 14000
out_prefix = sys.argv[2] if len(sys.argv) > 2 else "probe14"
probe_every = int(sys.argv[3]) if len(sys.argv) > 3 else 200

print(f"=== run_probe14: nsteps={nsteps} out_prefix={out_prefix} probe_every={probe_every} ===")
bg, og, hist, summary = M.run_cylinder(
    nsteps=nsteps, probe_every=probe_every, out_prefix=out_prefix,
    ni=120, nj=48, R=0.1, Rh_hole=0.42, Rf=0.6, Rf_og=0.78, beta=2.5,
    aoa_deg=3.0, ramp_steps=200,
)
print("SUMMARY:", summary)
