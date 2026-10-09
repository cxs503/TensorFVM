"""test_polar_v2.py -- validate function-based projection Laplacian."""
import numpy as np
import polar_v2 as POG
from scipy.sparse.linalg import eigsh

ni, nj = 60, 30
blk = POG.make_ogrid(ni, nj, 0.0, 0.0, 0.1, 0.78, beta=2.5)
N = ni * nj
x = blk.xc; r2 = blk.r * blk.r
print(f"grid {ni}x{nj}  N={N}")

# A. harmonic x = r cos th: interior Lap(x) ~ 0
L = POG.laplace_matrix(blk)
La = L.toarray()
Lap_x = (La @ x.reshape(-1)).reshape(ni, nj)
intmask = np.zeros((ni, nj), bool); intmask[:, 2:nj - 2] = True
print(f"[A] Lap(x) interior  max={np.max(np.abs(Lap_x[intmask])):.3e}  (want ~0)")

# B. Lap(r^2) interior ~ 4
Lap_r2 = (La @ r2.reshape(-1)).reshape(ni, nj)
print(f"[B] Lap(r^2) interior  mean={Lap_r2[intmask].mean():.4f}  max={Lap_r2[intmask].max():.4f}  (want 4)")

# C. symmetry & definiteness (volume-weighted). Build V-weighted symmetric part.
V = blk.J.reshape(-1)
# L is symmetric in <.,.>_V if V L is symmetric: check (V*L) vs its transpose
VL = La * V[:, None]
sym = np.max(np.abs(VL - VL.T))
ela = eigsh(La, k=2, which="LA", return_eigenvectors=False)
esa = eigsh(La, k=1, which="SA", return_eigenvectors=False)
print(f"[C] L volume-sym_err={sym:.2e}  lambda_max={ela.max():.3f}  lambda_min={esa[0]:.3f}")
print(f"    NSD (1~0, rest<0): {ela.max()<1e-9 and esa[0]<0}")

# D. projection removes polar_div exactly (no-slip wall)
rng = np.random.default_rng(1)
blk.u = rng.standard_normal((ni, nj)) + 0.3
blk.v = rng.standard_normal((ni, nj)) - 0.2
blk.u += 0.1 * np.sin(blk.xc * 4) * np.cos(blk.yc * 4)
blk.v += 0.1 * np.cos(blk.xc * 4) * np.sin(blk.yc * 4)
blk.u[:, 0] = 0.0; blk.v[:, 0] = 0.0          # no-slip
recv = blk.r >= 0.55
solved = ~recv.reshape(ni, nj)
div0 = POG.polar_div(blk, blk.u, blk.v)
poisson = POG.build_poisson(blk, recv)
dt = 0.02
POG.project(blk, dt, recv, poisson)
div1 = POG.polar_div(blk, blk.u, blk.v)
intm = solved & np.zeros((ni, nj), bool); intm[:, 2:] = True
print(f"[D] polar_div(u*) interior max={np.max(np.abs(div0[intm])):.3e}")
print(f"    polar_div(u_new) interior max={np.max(np.abs(div1[intm])):.3e}")
red = np.max(np.abs(div0[intm])) / max(np.max(np.abs(div1[intm])), 1e-30)
print(f"    reduction factor = {red:.2e}  (machine precision if >1e10)")

# E. pressure sign: manufactured front source -> p low there
blk2 = POG.make_ogrid(ni, nj, 0.0, 0.0, 0.1, 0.78, beta=2.5)
blk2.u = np.ones((ni, nj)); blk2.v = np.zeros((ni, nj))
blk2.u += 0.2 * np.where(blk2.xc < 0, 1.0, -1.0)
poisson2 = POG.build_poisson(blk2, recv)
POG.project(blk2, dt, recv, poisson2)
front = blk2.xc < -0.15; rear = blk2.xc > 0.15
print(f"[E] <p> front={blk2.p[front].mean():.4f}  rear={blk2.p[rear].mean():.4f}  "
      f"(+div->min: {'OK' if blk2.p[front].mean() < blk2.p[rear].mean() else 'BAD'})")
print("DONE.")
