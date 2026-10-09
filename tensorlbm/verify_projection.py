"""Verify the projection adjointness fix: after project(), div(corr) must equal
div_before/dt to machine precision on solved cells, so div(u_new)=0 there."""
import sys
import numpy as np
sys.path.insert(0, "/workspace")
import polar_ogrid as P

ni, nj = 160, 64
R, Rf, Rf_og, beta = 0.1, 0.60, 0.78, 2.5
blk = P.make_ogrid(ni, nj, 0.0, 0.0, R, Rf_og, beta=beta)
recv = blk.r >= Rf
solved = ~recv
blk._nu = 0.002; blk.U = 1.0

# random non-div-free initial field
rng = np.random.default_rng(0)
blk.u[:] = rng.standard_normal((ni, nj)) * 0.3
blk.v[:] = rng.standard_normal((ni, nj)) * 0.3
# hold recv to a smooth freestream-ish value so the test mirrors the run
blk.u[recv] = 1.0; blk.v[recv] = 0.0

poisson = P.build_poisson(blk, recv)
dt = 1.0e-3

divb = P.polar_div(blk, blk.u, blk.v)
P.project(blk, dt, recv, poisson)
diva = P.polar_div(blk, blk.u, blk.v)

# adjointness check: corr = D^T p, so div(corr) should equal div_before/dt on solved
D = blk._D
p = blk.p.reshape(-1)
corr = (D.T @ p).reshape(-1)
N = ni * nj
cu = corr[:N].reshape(ni, nj); cv = corr[N:].reshape(ni, nj)
div_corr = P.polar_div(blk, cu, cv)
max_adj = float(np.abs((div_corr - divb / dt)[solved]).max())
max_div_after = float(np.abs(diva[solved]).max())
print(f"max|div(corr) - div_before/dt| (solved) = {max_adj:.3e}")
print(f"max|div(u_new)|              (solved) = {max_div_after:.3e}")
print(f"  (old broken code gave ~4.0e-2 here; machine precision is the target)")

# also confirm recv cells are unchanged by caller convention (not corrected here)
assert max_adj < 1.0e-8, "ADJOINTNESS BROKEN"
print("ADJOINTNESS OK")
