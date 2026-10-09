import numpy as np, overset_cylinder_fvm as M
bg, og, hist, summary, dump = M.run_cylinder(
    nsteps=15000, probe_every=200, out_prefix='probe_ref',
    ni=120, nj=96, R=0.1, Rh_hole=0.42, Rf=0.6, Rf_og=0.78, beta=1.2,
    aoa_deg=3.0, ramp_steps=300, Rf_sponge=0.5)
print('SUMMARY', summary)
