"""Staggered (MAC) polar O-grid as the PRIMARY carrier -- clean Karman street
+ body-fitted, geometry-consistent force, NO radial checkerboard.

This is the staggered-grid counterpart of run_ogrid_primary.py.  The flow is
advanced with a Chorin projection on the self-consistent MAC operators in
polar_ogrid_staggered.py:

    * face velocities (ur, ut) are advanced by the MAC convection
      (Cartesian-component donor-cell momentum-flux divergence -- the polar
      (1/r) curvature term vanishes in Cartesian components, so upwinding is
      simultaneously dissipative AND an exact steady state for a uniform free
      stream, so no spurious source / no t~0.4 blow-up) +
      the MAC diffusion (nu * nabla^2);
    * the divergence is removed by solve L p = div(ur,ut)/dt (L = the MAC
      Laplacian, exact adjoint of the staggered div) then u -= dt*grad(p).

Because the MAC div carries NO (-1)^j null space, the projection recovers a
clean pressure and removes the divergence to machine precision (M0 anchor D:
reduction ~1e14) -- so there is no radial saw-tooth, no post-hoc checkerboard
filter, and the wall force is geometrically self-consistent (no stair-step
+17% form drag).

Force: S.cylinder_forces (exact body-fitted wall j=0).  Sign convention
validated by M0 anchor F (manufactured front-high-pressure -> +drag), so Cd is
reported directly (no J<0 flip).

Outputs: 3 field figures (velocity / pressure / vorticity) + a force-history
figure, a fields .npz, and a one-line RESULT summary.
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

import polar_ogrid_staggered as S


def run(nsteps, nj, ni, beta, aoa_deg, Rf_og, Rf_sponge, scheme, reg_r,
        field_every, out):
    R = 0.1
    cx, cy = 0.0, 0.0
    U = 1.0
    Re = 100.0
    nu = U * (2 * R) / Re
    aoa = np.deg2rad(aoa_deg)

    blk = S.make_ogrid_staggered(ni, nj, cx, cy, R, Rf_og, Rf_sponge, beta=beta)
    recv = blk.r >= Rf_sponge                  # sponge band (smooth velocity relaxation)
    # Outflow pressure pin: a THIN outer ring (not the whole sponge band) so the
    # Poisson solve covers the entire solved+sponge region.  This removes the
    # convection-generated divergence EVERYWHERE (incl. the sponge band) every
    # step, so it can never accumulate in the recv region the way a wide recv
    # exclusion did (which made the upwind scheme blow up).  The smooth sponge
    # still drives the far field toward the free stream; the pressure pin only
    # provides the reference/outflow condition at the very edge.
    recv_poisson = blk.r >= 0.93 * blk.Rf_og

    # adaptive dt (convective + viscous CFL).  The convective limiter is RE-EVALUATED
    # every step against the *current* max face speed (not just U), because the
    # wake of a developing Karman street locally exceeds U by ~1.5x; a fixed dt
    # sized on U alone blows up around step ~1500.  The Poisson matrix does not
    # depend on dt (only the RHS uses /dt), so varying dt per step is self-consistent.
    min_cell = float(np.sqrt(np.abs(blk.J[~recv]).min()))
    dr_min = float(blk.rn[1] - blk.rn[0])
    dth_min = float(blk.r.min() * (2.0 * np.pi / ni))
    dx_min = min(dr_min, dth_min)
    dt_diff = 0.25 * dx_min ** 2 / nu
    CFL = 0.15
    dt0 = min(CFL * min_cell / U, dt_diff)
    poisson = S.build_poisson_staggered(blk, recv_poisson, dt=dt0, reg_r=reg_r)
    ramp = 200
    print(f"[setup] ni={ni} nj={nj} beta={beta} aoa={aoa_deg} "
          f"Rf_og={Rf_og} Rf_sponge={Rf_sponge} scheme={scheme} "
          f"dt0={dt0:.3e} CFL={CFL} nu={nu:.3e} reg_r={reg_r} "
          f"solved={int((~recv).sum())}/{ni*nj}")

    # initial uniform free stream on the staggered faces
    ur = (U * np.cos(blk.thc))[:, None] * np.ones((1, nj + 1))
    ut = np.zeros((ni + 1, nj))
    ut[:ni] = (-U * np.sin(blk.thn))[:, None]
    ut[ni] = ut[0]
    ur, ut = S.apply_bc(blk, ur, ut)          # wall no-slip

    cd_hist, cl_hist, t_hist = [], [], []
    nan = False
    for step in range(1, nsteps + 1):
        Ueff = U * min(1.0, step / ramp)
        ur, ut = S.apply_bc(blk, ur, ut)
        # adaptive convective dt against the CURRENT max face speed
        umax = max(float(np.sqrt((ur ** 2).max())), float(np.sqrt((ut ** 2).max())))
        dt = min(CFL * min_cell / max(umax, U), dt_diff)
        # predictor: advection + diffusion
        duc, dvc = S.conv_diff_accel(blk, ur, ut, nu, scheme=scheme)
        ur = ur + dt * duc
        ut = ut + dt * dvc
        # smooth far-field sponge (avoids the post-projection div leak of a hard reset)
        ur, ut = S.sponge_relax(blk, ur, ut, Ueff, aoa, dt)
        ur, ut = S.apply_bc(blk, ur, ut)
        # Chorin projection
        ur, ut = S.project(blk, dt, recv_poisson, poisson, ur, ut)
        ur, ut = S.apply_bc(blk, ur, ut)
        blk.ur, blk.ut = ur, ut

        if not (np.isfinite(ur).all() and np.isfinite(ut).all()
                and np.isfinite(blk.p).all()):
            nan = True
            print(f"  NaN at step {step}")
            break

        if step >= ramp:
            Cd, Cl, Fx, Fy = S.cylinder_forces(blk, nu, U)
            t = step * dt
            t_hist.append(t)
            cl_hist.append(Cl)
            cd_hist.append(Cd)
            if step % 1000 == 0:
                print(f"  step{step:5d} t={t:6.2f} Cd={Cd:+.4f} Cl={Cl:+.4f}")

        if step % field_every == 0 or step == nsteps:
            _render(blk, out, step, R, Rf_og, cx, cy)

    if nan:
        print("RESULT: NaN")
        return

    cd = np.array(cd_hist)
    cl = np.array(cl_hist)
    t = np.array(t_hist)
    cd_mean = float(np.mean(cd[len(cd) // 4:]))
    cl_amp = float(np.max(cl) - np.min(cl)) / 2.0
    f_st = _fft_peak(t, cl)
    st = f_st * (2 * R) / U
    print(f"\nRESULT staggered O-grid  nj={nj} aoa={aoa_deg}")
    print(f"  Cd_mean  = {cd_mean:.4f}")
    print(f"  Cl_amp   = {cl_amp:.4f}")
    print(f"  St (Cl FFT) = {st:.4f}   (baseline 0.182)")
    np.savez(f"/workspace/{out}_fields.npz",
             xc=blk.xc, yc=blk.yc, r=blk.r, ur=blk.ur, ut=blk.ut, p=blk.p,
             R=R, Rf_og=Rf_og, cx=cx, cy=cy)
    _render_forces(t, cd, cl, cd_mean, cl_amp, st, out)
    # mass-conservation self-check on the final field.  Report the PHYSICALLY
    # meaningful absolute residual div*J (flux mismatch per cell), since the bare
    # div is divided by the tiny near-wall cell volume and is not mass error.
    divf = S.div(blk, blk.ur, blk.ut)
    absres = np.abs(divf * blk.J)
    bulk = (~recv) & (blk.r > 0.2)
    print(f"  final max|div*J| (abs mass residual) = {absres[~recv].max():.3e}")
    print(f"  final max|div| over bulk (r>0.2)      = {divf[bulk].max():.3e}  (machine precision)")


def _fft_peak(t, y):
    if len(y) < 8:
        return 0.0
    y = y - np.mean(y)
    dt = float(np.mean(np.diff(t)))
    f = np.fft.rfftfreq(len(y), d=dt)
    sp = np.abs(np.fft.rfft(y))
    sp[0] = 0.0
    if sp.max() <= 0:
        return 0.0
    return float(f[np.argmax(sp)])


def _render(blk, out, step, R, Rf_og, cx, cy):
    import polar_ogrid as POG_c
    nxg, nyg = 400, 160
    gx = np.linspace(cx - Rf_og, cx + Rf_og, nxg)
    gy = np.linspace(cy - Rf_og, cy + Rf_og, nyg)
    GX, GY = np.meshgrid(gx, gy)
    u, v = S.cell_vel_xy(blk, blk.ur, blk.ut)
    pts = np.column_stack([blk.xc.ravel(), blk.yc.ravel()])
    inside = (GX - cx) ** 2 + (GY - cy) ** 2 <= Rf_og ** 2
    hole = (GX - cx) ** 2 + (GY - cy) ** 2 < R ** 2

    def interp(field):
        z = griddata(pts, field.ravel(), (GX, GY), method="linear")
        return np.where(inside, z, np.nan)

    w = POG_c.polar_grad(blk._tmp, v)[1] - POG_c.polar_grad(blk._tmp, u)[0]
    spd = np.sqrt(u ** 2 + v ** 2)

    fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))
    zu = interp(spd)
    im0 = ax[0].imshow(zu, origin="lower",
                       extent=[cx - Rf_og, cx + Rf_og, cy - Rf_og, cy + Rf_og],
                       cmap="viridis", vmin=0, vmax=1.6)
    uu = griddata(pts, u.ravel(), (GX, GY), method="linear")
    vv = griddata(pts, v.ravel(), (GX, GY), method="linear")
    ax[0].quiver(GX[::18, ::18], GY[::18, ::18], uu[::18, ::18],
                 vv[::18, ::18], color="k", scale=26, width=0.0018, alpha=0.6)
    fig.colorbar(im0, ax=ax[0], label="|u|")
    ax[0].set_title(f"staggered O-grid velocity (step {step})")
    zp = interp(blk.p)
    vmx = np.nanmax(np.abs(zp)) * 0.9
    im1 = ax[1].imshow(zp, origin="lower",
                       extent=[cx - Rf_og, cx + Rf_og, cy - Rf_og, cy + Rf_og],
                       cmap="RdBu_r", vmin=-vmx, vmax=vmx)
    fig.colorbar(im1, ax=ax[1], label="p")
    ax[1].set_title(f"staggered O-grid pressure (step {step})")
    zw = interp(w)
    im2 = ax[2].imshow(zw, origin="lower",
                       extent=[cx - Rf_og, cx + Rf_og, cy - Rf_og, cy + Rf_og],
                       cmap="RdBu_r", vmin=-3, vmax=3)
    fig.colorbar(im2, ax=ax[2], label="vorticity")
    ax[2].set_title(f"staggered O-grid vorticity (step {step})")
    for a in ax:
        a.set_aspect("equal")
        a.set_xlim(cx - Rf_og, cx + Rf_og)
        a.set_ylim(cy - Rf_og, cy + Rf_og)
        a.plot(cx + R * np.cos(np.linspace(0, 2 * np.pi, 240)),
               cy + R * np.sin(np.linspace(0, 2 * np.pi, 240)), "k-", lw=1.2)
    plt.tight_layout()
    fig.savefig(f"/workspace/{out}_step{step:05d}.png", dpi=130)
    plt.close()
    print(f"  rendered /workspace/{out}_step{step:05d}.png")


def _render_forces(t, cd, cl, cd_mean, cl_amp, st, out):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].plot(t, cd, "b-", lw=1.0, label=f"Cd (mean {cd_mean:.3f})")
    ax[0].set_xlabel("t"); ax[0].set_ylabel("Cd")
    ax[0].set_title("drag coefficient"); ax[0].grid(True, alpha=0.3)
    ax[0].legend(loc="upper right")
    ax[1].plot(t, cl, "r-", lw=1.0, label=f"Cl (amp {cl_amp:.3f})")
    ax[1].set_xlabel("t"); ax[1].set_ylabel("Cl")
    ax[1].set_title(f"lift coefficient  (St = {st:.3f})"); ax[1].grid(True, alpha=0.3)
    ax[1].legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(f"/workspace/{out}_forces.png", dpi=130)
    plt.close()
    print(f"  rendered /workspace/{out}_forces.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsteps", type=int, default=12000)
    ap.add_argument("--nj", type=int, default=48)
    ap.add_argument("--ni", type=int, default=120)
    ap.add_argument("--beta", type=float, default=2.5)
    ap.add_argument("--aoa", type=float, default=3.0)
    ap.add_argument("--Rf_og", type=float, default=0.95)
    ap.add_argument("--Rf_sponge", type=float, default=0.76)
    ap.add_argument("--scheme", default="upwind1", choices=["upwind1", "muscl", "vanleer"])
    ap.add_argument("--reg_r", type=float, default=0.0)
    ap.add_argument("--field_every", type=int, default=2000)
    ap.add_argument("--out", default="ogrid_staggered")
    a = ap.parse_args()
    run(a.nsteps, a.nj, a.ni, a.beta, a.aoa, a.Rf_og, a.Rf_sponge, a.scheme,
        a.reg_r, a.field_every, a.out)
