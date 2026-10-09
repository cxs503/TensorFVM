import numpy as np, overset_cylinder_fvm as M, sys
def one(nj,beta,rfs,steps,tag):
    bg,og,hist,summary,dump = M.run_cylinder(
        nsteps=steps, probe_every=500, out_prefix=tag,
        ni=120, nj=nj, R=0.1, Rh_hole=0.42, Rf=0.6, Rf_og=0.78, beta=beta,
        aoa_deg=3.0, ramp_steps=300, Rf_sponge=rfs)
    print(f'[{tag}] nj={nj} beta={beta} Rf_sponge={rfs} ->', summary); sys.stdout.flush()
one(96, 1.2, 0.6, 10000, 'res96')
one(144,1.0, 0.6, 10000, 'res144')
