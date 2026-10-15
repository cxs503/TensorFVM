#!/usr/bin/env python3
"""compare.py RUNDIR [alpha=10]: upper-surface Cf/Cp of RUNDIR/wall.dat vs TMR CFL3D SA (reference.json)."""
import sys, json, numpy as np
from pathlib import Path
run = Path(sys.argv[1]); alpha = int(float(sys.argv[2])) if len(sys.argv) > 2 else 10
ref = json.load(open(Path(__file__).with_name('reference.json')))
rcf = ref['cf_upper_alpha%d_cfl3d' % alpha]; rcp = ref['cp_upper_alpha%d_cfl3d' % alpha]
d = np.loadtxt(run / 'wall.dat'); x, y, nx, ny, cp, cf = d.T; up = ny > 0 if alpha > 0 else y > 0
o = np.argsort(x[up]); xu = x[up][o]; cfu = cf[up][o]; cpu = cp[up][o]
rev = xu[cfu < 0]
print('upper surface: Cp_min %.3f at x=%.4f (CFL3D %.3f); reversed-flow faces: %d%s' % (
    cpu.min(), xu[cpu.argmin()], min(rcp['cp']), len(rev), ' (x %.4f..%.4f)' % (rev.min(), rev.max()) if len(rev) else ''))
print('%7s %9s %9s %6s | %8s %8s | %7s' % ('x', 'Cf_run', 'Cf_CFL3D', 'ratio', 'Cp_run', 'Cp_CFL3D', 'cf_loc ratio'))
for xs in [0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
    j = np.abs(xu - xs).argmin(); r = np.interp(xs, rcf['x'], rcf['cf']); rp = np.interp(xs, rcp['x'], rcp['cp'])
    loc = (cfu[j] / (1 - cpu[j])) / (r / (1 - rp)) if r else np.nan
    print('%7.3f %9.5f %9.5f %6.2f | %8.3f %8.3f | %7.2f' % (xu[j], cfu[j], r, cfu[j] / r if r else np.nan, cpu[j], rp, loc))
