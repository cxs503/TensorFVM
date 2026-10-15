#!/usr/bin/env python3
"""profiles.py RUNDIR x1 x2 ...: upper-surface wall-normal profiles (u_t, chi, u+ vs y+) and the
von Karman momentum-integral check between consecutive stations:
   d(Ue^2 theta)/dx + delta* Ue dUe/dx = u_tau^2   (exact for a steady 2-D boundary layer)
On the verified flat plate the balance closes within ~4 %."""
import sys, numpy as np, json
from types import SimpleNamespace
from tensorfvm.multi_element import ThreeElementMesh
from scipy.spatial import cKDTree
run = sys.argv[1]; verbose = '-v' in sys.argv; xs = [float(a) for a in sys.argv[2:] if a != '-v']
xs = [x for x in xs]
mesh = ThreeElementMesh(SimpleNamespace(mesh_file=run + '/mesh.msh', device='cpu', single_element=True))
ck = np.load(run + '/checkpoint.npz'); vel = ck['velocity']; nt = ck['nu_tilde']; p = ck['pressure']
nu = 1 / json.load(open(run + '/config.json'))['reynolds']
C = mesh.centers.reshape(-1, 2).numpy(); tree = cKDTree(C)
faces = np.flatnonzero(mesh.masks['main'].numpy()); fc = mesh.face_centers.numpy()[faces]
fn = mesh.face_normals.numpy()[faces]; own = mesh.owner.numpy()[faces]
n = -fn / np.linalg.norm(fn, axis=1, keepdims=True); up = n[:, 1] > 0
print(run, 'it', int(ck['iteration']))
rows = []
for xq in xs:
    k = np.flatnonzero(up); j = k[np.abs(fc[k, 0] - xq).argmin()]; t = np.array([n[j, 1], -n[j, 0]])
    if t[0] < 0: t = -t
    dw = abs(((C[own[j]] - fc[j]) * n[j]).sum()); ut_w = (vel[own[j]] * t).sum(); tau = nu * ut_w / dw; utau = np.sqrt(abs(tau))
    d = np.geomspace(5e-7, 0.2, 200); pts = fc[j] + d[:, None] * n[j]; _, idx = tree.query(pts)
    ut = (vel[idx] * t).sum(1); chi = nt[idx] / nu
    ie = ut.argmax(); ue = ut[ie]; i99 = np.argmax(ut >= 0.99 * ue)
    dd = np.diff(np.r_[0, d]); th = ((ut / ue) * (1 - ut / ue) * dd)[:ie + 1].sum(); ds = ((1 - ut / ue) * dd)[:ie + 1].sum()
    cp_w = 2 * p[own[j]]
    rows.append(dict(x=fc[j, 0], cp=cp_w, cf=2 * tau, ue=ue, ue_bern=np.sqrt(max(1 - cp_w, 0)), th=th, ds=ds, d99=d[i99]))
    print(f'x={fc[j,0]:.4f}: Cp={cp_w:.3f} Cf={2*tau:.5f} u_tau={utau:.4f} U_e={ue:.3f} (Bernoulli {np.sqrt(max(1-cp_w,0)):.3f}) '
          f'delta99={d[i99]:.2e} theta={th:.2e} delta*={ds:.2e} H={ds/th:.2f} Re_theta={ue*th/nu:.0f} chi_max={chi.max():.1f}@{d[chi.argmax()]:.1e}')
    if verbose:
        print('   y+      u+   loglaw   chi   ut/Ue')
        for yy, uu, cc in zip(d, ut, chi):
            yp = yy * utau / nu
            if yp < 1 or yp > 3000: continue
            print(f'{yp:7.1f} {uu/utau:7.2f} {np.log(yp)/0.41+5.0:7.2f} {cc:6.1f} {uu/ue:6.3f}')
if len(rows) > 1:
    print('\nmomentum integral between stations (using profile U_e):  LHS = d(Ue^2 theta) + int delta* Ue dUe ;  RHS = int (Cf/2) dx')
    for a, b in zip(rows[:-1], rows[1:]):
        lhs = b['ue']**2 * b['th'] - a['ue']**2 * a['th'] + 0.5 * (a['ds'] + b['ds']) * 0.5 * (b['ue']**2 - a['ue']**2)
        rhs = 0.5 * (a['cf'] + b['cf']) / 2 * (b['x'] - a['x'])
        print(f'  x {a["x"]:.3f}->{b["x"]:.3f}: LHS {lhs:.3e}  RHS {rhs:.3e}  ratio {lhs/rhs:.2f}')
