"""Focused self-consistency checks for the Rhie-Chow polar projection.

[A] polar_lap(r^2) ~ +4 (continuum Laplacian of x^2+y^2 is 4), plain vs RC.
[B] div of rigid rotation u=(-y,x) ~ 0 (divergence-free preserved).
[C] RC Poisson reduced matrix is symmetric.
[E] CHECKERBOARD DECOUPLING: p_chk=(-1)^j is a null space of the PLAIN face
    Laplacian (-> projection can never remove a radial checkerboard -> spurious
    near-wall jet & ~3x viscous drag), but RC Laplace(p_chk) is NON-zero (the
    null space is broken).  This is the central bug the Rhie-Chow fix targets.
[F] Full Chorin projection removes the divergence of a random non-solenoidal
    velocity; compare RC vs non-RC residual reduction.
"""
import numpy as np
import polar_ogrid as POG

ni, nj = 60, 48
R, Rf_og, beta = 0.1, 0.78, 2.5
og = POG.make_ogrid(ni, nj, 0.25, 0.5, R, Rf_og, beta=beta)
Rf_sponge = 0.5
og.recv = og.r >= Rf_sponge
og.solved = ~og.recv
recv = og.recv.reshape(-1)
solved_idx = np.nonzero(~recv)[0]
dt = 5e-3

# [A] Laplacian of r^2 (continuum = 4)
f = (og.rc**2).copy()
lap_plain = POG.polar_lap(og, f)
lap_rc = POG.polar_lap(og, f, p=f, dt=dt)
print("[A] Lap(r^2)  plain max|..-4| = %.3e   RC max|..-4| = %.3e"
      % (np.abs(lap_plain - 4).max(), np.abs(lap_rc - 4).max()))

# [B] divergence-free rigid rotation
u = -og.yc
v = og.xc
div_rot = POG.polar_div(og, u, v)
print("[B] div(rigid rotation u=(-y,x)) max = %.3e" % np.abs(div_rot).max())

# [C] RC Poisson matrix symmetry
Lfull = POG._lap_face_matrix(og, wall_bc="neumann", dt=dt, rc_poisson=True).toarray()
Lred = Lfull[np.ix_(solved_idx, solved_idx)]
print("[C] RC Poisson reduced-matrix max asymmetry = %.3e" % np.abs(Lred - Lred.T).max())

# [E] checkerboard null space
p_chk = np.ones((ni, nj))
p_chk[:, 1::2] = -1.0          # (-1)^j along radial index
# plain Laplace(p_chk): uses polar_div(grad(p_chk)) with p=None
lap_plain_chk = POG.polar_lap(og, p_chk)            # p=None -> plain
# RC Laplace(p_chk): uses polar_div(grad(p_chk), p=p_chk, dt)
lap_rc_chk = POG.polar_lap(og, p_chk, p=p_chk, dt=dt)
print("[E] checkerboard  plain Lap max = %.3e   RC Lap max = %.3e"
      % (np.abs(lap_plain_chk).max(), np.abs(lap_rc_chk).max()))

# [F] full projection cycle
rng = np.random.default_rng(0)
og.u = rng.standard_normal((ni, nj)) * 0.1
og.v = rng.standard_normal((ni, nj)) * 0.1
og.p = np.zeros((ni, nj))
poisson_rc = POG.build_poisson(og, og.recv, reg_r=0.0, dt=dt, rc=True)
poisson_norc = POG.build_poisson(og, og.recv, reg_r=0.0, dt=dt, rc=False)
for P, tag in ((poisson_rc, "RC"), (poisson_norc, "non-RC")):
    og.u = rng.standard_normal((ni, nj)) * 0.1
    og.v = rng.standard_normal((ni, nj)) * 0.1
    og.p = np.zeros((ni, nj))
    div0 = POG.polar_div(og, og.u, og.v).reshape(-1)
    POG.project(og, dt, og.recv, P)
    div1 = POG.polar_div(og, og.u, og.v).reshape(-1)
    print("[F] %s  div before=%.3e after=%.3e ratio=%.3e"
          % (tag, np.abs(div0).max(), np.abs(div1).max(),
             np.abs(div1).max() / (np.abs(div0).max() + 1e-30)))
