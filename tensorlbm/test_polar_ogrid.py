"""test_polar_ogrid.py -- validate the LIVE polar_ogrid operators.

Mirrors test_polar_v2 but imports the production module (polar_ogrid, imported
as POG by overset_cylinder_fvm.py) so the checks run against exactly what the
simulation uses.  Anchors:
  A. harmonic x = r cos th  -> Lap(x) ~ 0 (interior)
  B. p = r^2                -> Lap(p) == +4 (exact constant)
  C. L volume-weighted symmetric & negative semi-definite
  D. projection removes polar_div to machine precision
  E. +div source -> local p MIN (correct sign convention)
  F. cylinder_forces sign on a manufactured front-stagnation pressure
"""
import numpy as np
import polar_ogrid as POG
from scipy.sparse.linalg import eigsh

ni, nj = 60, 30
blk = POG.make_ogrid(ni, nj, 0.0, 0.0, 0.1, 0.78, beta=2.5)
N = ni * nj
x = blk.xc; r2 = blk.r * blk.r
print(f"grid {ni}x{nj}  N={N}")

# ---- A. harmonic field ----
L = POG._lap_face_matrix(blk)          # the exact operator build_poisson uses
La = L.toarray()
Lap_x = (La @ x.reshape(-1)).reshape(ni, nj)
intmask = np.zeros((ni, nj), bool); intmask[:, 2:nj - 2] = True
print(f"[A] Lap(x) interior  max={np.max(np.abs(Lap_x[intmask])):.3e}  (want ~0)")

# ---- B. r^2 -> 4 ----
Lap_r2 = (La @ r2.reshape(-1)).reshape(ni, nj)
print(f"[B] Lap(r^2) interior  mean={Lap_r2[intmask].mean():.4f}  max={Lap_r2[intmask].max():.4f}  (want 4)")

# ---- C. volume-weighted symmetry & definiteness ----
V = blk.J.reshape(-1)
VL = La * V[:, None]
sym = np.max(np.abs(VL - VL.T))
ela = eigsh(La, k=2, which="LA", return_eigenvectors=False)
esa = eigsh(La, k=1, which="SA", return_eigenvectors=False)
print(f"[C] L volume-sym_err={sym:.2e}  lambda_max={ela.max():.3f}  lambda_min={esa[0]:.3f}")
print(f"    NSD (1~0, rest<0): {ela.max()<1e-9 and esa[0]<0}")

# ---- D. projection removes divergence (recv set to div-free freestream, like sim) ----
rng = np.random.default_rng(1)
blk.u = rng.standard_normal((ni, nj)) + 0.3
blk.v = rng.standard_normal((ni, nj)) - 0.2
blk.u += 0.1 * np.sin(blk.xc * 4) * np.cos(blk.yc * 4)
blk.v += 0.1 * np.cos(blk.xc * 4) * np.sin(blk.yc * 4)
recv = blk.r >= 0.55
solved = ~recv.reshape(ni, nj)
# impose recv as uniform freestream (divergence-free) as the caller does each step
blk.u[recv] = 1.0; blk.v[recv] = 0.0
blk.u[:, 0] = 0.0; blk.v[:, 0] = 0.0          # no-slip wall
div0 = POG.polar_div(blk, blk.u, blk.v)
poisson = POG.build_poisson(blk, recv)
dt = 0.02
POG.project(blk, dt, recv, poisson)
# caller re-imposes recv & wall after project()
blk.u[recv] = 1.0; blk.v[recv] = 0.0
blk.u[:, 0] = 0.0; blk.v[:, 0] = 0.0
div1 = POG.polar_div(blk, blk.u, blk.v)
# interior ONLY: exclude wall ring (r<0.15) and recv fringe (r>=0.5), where the
# no-flux / Dirichlet-p=0 boundaries leave a bounded residual by construction.
intm = solved & (blk.r > 0.15) & (blk.r < 0.50)
print(f"[D] polar_div(u*) interior max={np.max(np.abs(div0[intm])):.3e}")
print(f"    polar_div(u_new) interior max={np.max(np.abs(div1[intm])):.3e}")
red = np.max(np.abs(div0[intm])) / max(np.max(np.abs(div1[intm])), 1e-30)
print(f"    reduction factor = {red:.2e}  (want >>1; recv/wall re-imposed)")

# ---- E. Poisson sign convention: a smooth +source at the FRONT must make p a local MIN ----
# projection solves  lap(p) = div(u*)/dt  (this operator: lap(r^2)=+4, standard sign).
# A positive source (expansion, div>0) -> lap(p)>0 -> p locally CONVEX -> local MINIMUM.
blk2 = POG.make_ogrid(ni, nj, 0.0, 0.0, 0.1, 0.78, beta=2.5)
recv2 = blk2.r >= 0.55
poisson2 = POG.build_poisson(blk2, recv2)
src = np.exp(-((blk2.xc + 0.3) ** 2 + blk2.yc ** 2) / 0.04)   # +Gaussian source at front
rhs = np.zeros(ni * nj)
rhs[blk2._solved_idx] = (src.reshape(-1) / dt)[blk2._solved_idx]
p = np.zeros(ni * nj); p[blk2._solved_idx] = poisson2.solve(rhs[blk2._solved_idx])
p = p.reshape(ni, nj)
front_c = (blk2.xc < -0.25) & (np.abs(blk2.yc) < 0.05)
far = (blk2.xc > 0.25) & (np.abs(blk2.yc) < 0.05)             # rear, away from source
print(f"[E] <p> front-source={p[front_c].mean():.4f}  rear={p[far].mean():.4f}  "
      f"(+source->local min: {'OK' if p[front_c].mean() < p[far].mean() else 'BAD'})")

# ---- F. cylinder_forces on manufactured front-high-pressure: expect +drag ----
blk3 = POG.make_ogrid(ni, nj, 0.0, 0.0, 0.1, 0.78, beta=2.5)
# stagnation-style pressure: high at front (th=pi, cos th=-1), low at rear (th=0)
blk3.p = np.where(blk3.xc < 0, 1.0, -1.0) * (1.0 - blk3.r)   # front>0, rear<0
blk3.u[:, 0] = 0.0; blk3.v[:, 0] = 0.0
Cd3, Cl3, Fx3, Fy3 = POG.cylinder_forces(blk3, 1e-3, 1.0)
print(f"[F] manufactured front-high-pressure: Cd={Cd3:.4f} Cl={Cl3:.4f} Fx={Fx3:.4f} "
      f"(expect Fx>0 drag: {'OK' if Fx3 > 0 else 'BAD'})")
print("DONE.")
