"""Shedding-focused collocated O-grid driver.

Goal: recover the Re=100 Karman vortex street (Cl oscillates, St~0.18, Cd~1.3)
that the over-diffusive default config (fck=1/fp=0.5 or hard recv clamp) locks
into a steady deflected wake.

Changes vs run_ogrid_primary:
  * SMOOTH far-field sponge (relax toward free stream in the outer ring) instead
    of a hard recv clamp -- the hard circular Dirichlet outflow cuts the wake
    short and over-constrains it.
  * Transient asymmetric perturbation (steps [k0,k1]) to trigger the Hopf mode
    if it is subcritical / under-seeded by the aoa alone.
  * No diffusive 3-pt filters; only the spectral (-1)^j checkerboard projection
    (cb=pair) which preserves the shedding.
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

import polar_ogrid as POG


def smooth_sponge(blk, u, v, Ueff, aoa, dt, Rf_sp, sigma_max=25.0):
    ur_fs = Ueff * np.cos(aoa)
    ut_fs = Ueff * np.sin(aoa)
    s = np.where(blk.r >= Rf_sp,
                 sigma_max * ((blk.r - Rf_sp) / (blk.Rf_og - Rf_sp)) ** 2, 0.0)
    u = u + dt * s * (ur_fs - u)
    v = v + dt * s * (ut_fs - v)
    return u, v


def run(nsteps, nj, ni, beta, aoa_deg, Rf_og, Rf_sponge, scheme,
        reg_r, cb, nsteps_per_field, out, pert_amp, pert_k0, pert_k1):
    R = 0.1
    cx, cy = 0.0, 0.0
    U = 1.0
    Re = 100.0
    nu = U * (2 * R) / Re
    aoa = np.deg2rad(aoa_deg)

    og = POG.make_ogrid(ni, nj, cx, cy, R, Rf_og, beta=beta)
    og._nu = nu
    og.U = U
    og.recv = og.r >= Rf_sponge
    og.solved = ~og.recv

    min_cell = float(np.sqrt(np.abs(og.J[og.solved])).min())
    dt_cfl = 0.2 * min_cell / U
    rn = og.rn
    dr_min = float(rn[1] - rn[0])
    dth_min = float(og.r.min() * (2.0 * np.pi / ni))
    dx_min = min(dr_min, dth_min)
    dt_diff = 0.25 * dx_min ** 2 / nu
    dt = min(dt_cfl, dt_diff)
    poisson = POG.build_poisson(og, og.recv, reg_r=reg_r, dt=dt, rc=False)
    ramp = 200
    print(f"[setup] ni={ni} nj={nj} beta={beta} aoa={aoa_deg} "
          f"Rf_og={Rf_og} Rf_sponge={Rf_sponge} scheme={scheme} "
          f"dt={dt:.3e} nu={nu:.3e} solved={og.solved.sum()}/{ni*nj} "
          f"pert=[{pert_k0},{pert_k1}] amp={pert_amp}")

    cd_hist, cl_hist, t_hist = [], [], []
    nan = False
    for step in range(1, nsteps + 1):
        Ueff = U * min(1.0, step / ramp)
        # predictor
        duc, dvc = POG.polar_conv(og, og.u, og.v, Ueff, scheme=scheme)
        diff_u, diff_v = POG.polar_diffusion(og, og.u, og.v)
        ustar = og.u + dt * (-duc + diff_u)
        vstar = og.v + dt * (-dvc + diff_v)
        ustar[:, 0] = 0.0
        vstar[:, 0] = 0.0
        og.u, og.v = ustar, vstar
        # smooth far-field sponge
        og.u, og.v = smooth_sponge(og, og.u, og.v, Ueff, aoa, dt, Rf_sponge)
        og.u[:, 0] = 0.0
        og.v[:, 0] = 0.0
        # project
        POG.project(og, dt, og.recv, poisson, rc=False)
        if cb != "none":
            og.p = POG.polar_checkerboard_project(og, og.p, mode=cb)
            og.u = POG.polar_checkerboard_project(og, og.u, mode=cb)
            og.v = POG.polar_checkerboard_project(og, og.v, mode=cb)
        og.u[:, 0] = 0.0
        og.v[:, 0] = 0.0
        # transient coherent (antisymmetric / lift) perturbation in the near
        # wake to excite the shedding mode without pumping non-physical energy.
        if pert_k0 <= step <= pert_k1:
            vpert = (pert_amp * U * np.sin(og.thn[:, None])
                     * np.exp(-((og.r - 0.4) / 0.2) ** 2))
            og.v += vpert

        if not (np.isfinite(og.u).all() and np.isfinite(og.v).all()
                and np.isfinite(og.p).all()):
            nan = True
            print(f"  NaN at step {step}")
            break

        if step >= ramp:
            Cd, Cl, Fx, Fy = POG.cylinder_forces(og, nu, U)
            t = step * dt
            t_hist.append(t)
            cl_hist.append(Cl)
            cd_hist.append(Cd)
            if step % 1000 == 0:
                print(f"  step{step:5d} t={t:6.2f} Cd={Cd:+.4f} Cl={Cl:+.4f}")

        if step % nsteps_per_field == 0 or step == nsteps:
            _render(og, out, step, R, Rf_og, cx, cy)

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
    print(f"\nRESULT O-grid-shed  nj={nj} aoa={aoa_deg} scheme={scheme}")
    print(f"  Cd_mean  = {cd_mean:.4f}")
    print(f"  Cl_amp   = {cl_amp:.4f}")
    print(f"  St (Cl FFT) = {st:.4f}   (baseline 0.182)")
    np.savez(f"/workspace/{out}_fields.npz",
             xc=og.xc, yc=og.yc, r=og.r, u=og.u, v=og.v, p=og.p,
             R=R, Rf_og=Rf_og, cx=cx, cy=cy)


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


def _render(og, out, step, R, Rf_og, cx, cy):
    nxg, nyg = 400, 160
    gx = np.linspace(cx - Rf_og, cx + Rf_og, nxg)
    gy = np.linspace(cy - Rf_og, cy + Rf_og, nyg)
    GX, GY = np.meshgrid(gx, gy)
    pts = np.column_stack([og.xc.ravel(), og.yc.ravel()])
    inside = (GX - cx) ** 2 + (GY - cy) ** 2 <= Rf_og ** 2
    hole = (GX - cx) ** 2 + (GY - cy) ** 2 < R ** 2

    def interp(field):
        z = griddata(pts, field.ravel(), (GX, GY), method="linear")
        return np.where(inside, z, np.nan)

    w = POG.polar_grad(og, og.v)[1] - POG.polar_grad(og, og.u)[0]
    spd = np.sqrt(og.u ** 2 + og.v ** 2)
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))
    im0 = ax[0].imshow(interp(spd), origin="lower",
                       extent=[cx - Rf_og, cx + Rf_og, cy - Rf_og, cy + Rf_og],
                       cmap="viridis", vmin=0, vmax=1.6)
    fig.colorbar(im0, ax=ax[0], label="|u|")
    ax[0].set_title(f"O-grid velocity (step {step})")
    zp = interp(og.p)
    vmx = np.nanmax(np.abs(zp)) * 0.9
    im1 = ax[1].imshow(zp, origin="lower",
                      extent=[cx - Rf_og, cx + Rf_og, cy - Rf_og, cy + Rf_og],
                      cmap="RdBu_r", vmin=-vmx, vmax=vmx)
    fig.colorbar(im1, ax=ax[1], label="p")
    ax[1].set_title(f"O-grid pressure (step {step})")
    im2 = ax[2].imshow(interp(w), origin="lower",
                      extent=[cx - Rf_og, cx + Rf_og, cy - Rf_og, cy + Rf_og],
                      cmap="RdBu_r", vmin=-3, vmax=3)
    fig.colorbar(im2, ax=ax[2], label="vorticity")
    ax[2].set_title(f"O-grid vorticity (step {step})")
    for a in ax:
        a.set_aspect("equal")
        a.set_xlim(cx - Rf_og, cx + Rf_og)
        a.set_ylim(cy - Rf_og, cy + Rf_og)
        a.plot(cx + R * np.cos(np.linspace(0, 2 * np.pi, 240)),
               cy + R * np.sin(np.linspace(0, 2 * np.pi, 240)), "k-", lw=1.2)
    plt.tight_layout()
    fig.savefig(f"/workspace/{out}_step{step:05d}.png", dpi=130)
    plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsteps", type=int, default=20000)
    ap.add_argument("--nj", type=int, default=48)
    ap.add_argument("--ni", type=int, default=120)
    ap.add_argument("--beta", type=float, default=2.5)
    ap.add_argument("--aoa", type=float, default=3.0)
    ap.add_argument("--Rf_og", type=float, default=0.95)
    ap.add_argument("--Rf_sponge", type=float, default=0.82)
    ap.add_argument("--scheme", default="vanleer",
                    choices=["upwind1", "muscl", "vanleer"])
    ap.add_argument("--reg_r", type=float, default=0.0)
    ap.add_argument("--cb", default="pair", choices=["none", "global", "pair"])
    ap.add_argument("--field_every", type=int, default=2000)
    ap.add_argument("--pert_amp", type=float, default=0.05)
    ap.add_argument("--pert_k0", type=int, default=300)
    ap.add_argument("--pert_k1", type=int, default=2000)
    ap.add_argument("--out", default="ogrid_shed")
    a = ap.parse_args()
    run(a.nsteps, a.nj, a.ni, a.beta, a.aoa, a.Rf_og, a.Rf_sponge, a.scheme,
        a.reg_r, a.cb, a.field_every, a.out, a.pert_amp, a.pert_k0, a.pert_k1)
