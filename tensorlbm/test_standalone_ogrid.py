"""Standalone O-grid cylinder (Re=100) test harness.

Isolates three open questions from the overset debugging:
  (1) 10 Hz spurious mode  -> compare scheme="upwind1" vs "muscl" (and optional
      local 4th-order hyperviscosity).  Pure O-grid, NO _sample_bg, so any
      striping seen here is a REAL field artefact, not overset interpolation.
  (2) Cd over-prediction    -> measure aoa=0 steady Cd (should be ~1.0-1.2 with
      clean physics; upwind1 numerical diffusion raises effective viscosity).
  (3) vorticity striping    -> dump BOTH a native polar pcolormesh AND a
      Cartesian griddata interpolation, to see if striping is a render artefact.

Usage:
  python3 test_standalone_ogrid.py --scheme muscl --hv 0 --nsteps 12000
  python3 test_standalone_ogrid.py --scheme upwind1 --hv 0 --nsteps 12000 -aoa 0
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

import polar_ogrid as POG


def vorticity(blk):
    gxu, gyu = POG.polar_grad(blk, blk.u)
    gxv, gyv = POG.polar_grad(blk, blk.v)
    return gxv - gyu


def run(scheme, hv_cfl, nsteps, nj, beta, aoa_deg, Rf_sponge, dt_fix=None, out_prefix="standalone", fsig=0.0, fr=0.0, fp=0.0, reg_r=0.0, rc=False, fv=0.0):
    ni = 120
    R = 0.1
    Rf_og = 0.78
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

    # dt: same adaptive rule as the main sim (computed BEFORE build_poisson: the
    # Rhie-Chow Poisson operator needs dt).
    if dt_fix is None:
        min_cell = float(np.sqrt(np.abs(og.J[og.solved])).min())
        dt_cfl = 0.2 * min_cell / U
        rn = og.rn
        dr_min = float(rn[1] - rn[0])
        dth_min = float(og.r.min() * (2.0 * np.pi / ni))
        dx_min = min(dr_min, dth_min)
        dt_diff = 0.25 * dx_min ** 2 / nu
        dt = min(dt_cfl, dt_diff)
    else:
        dt = dt_fix
    poisson = POG.build_poisson(og, og.recv, reg_r=reg_r, dt=dt)
    ramp = 200
    print(f"[setup] scheme={scheme} hv_cfl={hv_cfl} ni={ni} nj={nj} beta={beta} "
          f"aoa={aoa_deg} Rf_sponge={Rf_sponge} dt={dt:.3e} nu={nu:.3e} "
          f"solved={og.solved.sum()}/{ni*nj}")

    # probe: near-wall cell at j=7 (where the 10 Hz mode peaks) and a near-wake point
    j_probe = min(7, nj - 2)
    i_probe = ni // 2
    # near-wake velocity probe at x ~ 1.5D downstream (convert to O-grid index)
    px, py = cx + 1.5 * (2 * R), cy + 0.3 * (2 * R)

    def to_ij(x, y):
        th = np.arctan2(y - cy, x - cx)
        r = np.hypot(x - cx, y - cy)
        # nearest cell
        dth = 2.0 * np.pi / ni
        i = int(np.round((th - 0.5 * dth) / dth)) % ni
        j = int(np.argmin(np.abs(og.rc[0, :] - r)))
        return i, j

    ip, jp = to_ij(px, py)

    t_hist = []
    cl_hist = []
    cd_hist = []
    vw_hist = []      # near-wall radial velocity at (i_probe, j_probe)
    vprobe_hist = []  # near-wake v

    field_every = max(1, nsteps // 6)
    nan = False

    for step in range(1, nsteps + 1):
        Ueff = U * min(1.0, step / ramp)
        # far-field donor on recv fringe (freestream at angle aoa, to seed shedding)
        og.u[og.recv] = Ueff * np.cos(aoa)
        og.v[og.recv] = Ueff * np.sin(aoa)
        # predictor
        duc, dvc = POG.polar_conv(og, og.u, og.v, Ueff, scheme=scheme)
        diff_u, diff_v = POG.polar_diffusion(og, og.u, og.v)
        ustar = og.u + dt * (-duc + diff_u)
        vstar = og.v + dt * (-dvc + diff_v)
        if hv_cfl > 0:
            hu, hv = POG.polar_hypervis_local(og, ustar, vstar, cfl=hv_cfl, dt=dt)
            # MINUS sign: Lap^2 has +k^4 eigenvalues -> -dt*a*Lap^2 DAMPS grid-scale modes
            ustar = ustar - dt * hu
            vstar = vstar - dt * hv
        if fsig > 0:
            ustar = POG.polar_filter(og, ustar, sigma_theta=fsig)
            vstar = POG.polar_filter(og, vstar, sigma_theta=fsig)
        if fr > 0:
            ustar = POG.polar_filter(og, ustar, sigma_r=fr)
            vstar = POG.polar_filter(og, vstar, sigma_r=fr)
        ustar[:, 0] = 0.0
        vstar[:, 0] = 0.0
        ustar[og.recv] = og.u[og.recv]
        vstar[og.recv] = og.v[og.recv]
        og.u = ustar
        og.v = vstar
        # project
        POG.project(og, dt, og.recv, poisson, rc=rc)
        # post-projection radial Shapiro filter on the VELOCITY: directly kills the
        # radial (-1)^j checkerboard that plain div cannot see (it is a null space),
        # which would otherwise imprint a spurious near-wall saw-tooth and over-predict
        # the viscous drag ~3x.  reg_r pins the PRESSURE checkerboard; this pins the
        # VELOCITY checkerboard.  Applied before the wall/fringe are re-imposed.
        if fv > 0:
            og.u = POG.polar_filter(og, og.u, sigma_r=fv)
            og.v = POG.polar_filter(og, og.v, sigma_r=fv)
        # post-projection pressure radial filter: the O-grid Poisson operator has
        # the radial checkerboard (−1)^j as a near-null space (face-averaged flux
        # cancels it), so splu injects a pressure saw-tooth that grad(p) then
        # imprints on the velocity.  Filtering p radially each step removes that
        # component, so the corrected velocity stays smooth (no spurious wall jet).
        if fp > 0:
            og.p = POG.polar_filter(og, og.p, sigma_r=fp)
        og.u[:, 0] = 0.0
        og.v[:, 0] = 0.0
        og.u[og.recv] = Ueff * np.cos(aoa)
        og.v[og.recv] = Ueff * np.sin(aoa)

        if not (np.isfinite(og.u).all() and np.isfinite(og.v).all()):
            nan = True
            print(f"  NaN at step {step}")
            break

        if step >= ramp:
            Cd, Cl, Fx, Fy = POG.cylinder_forces(og, nu, U)
            t = step * dt
            t_hist.append(t)
            cl_hist.append(Cl)
            cd_hist.append(Cd)
            vw_hist.append(float(og.v[i_probe, j_probe]))
            vprobe_hist.append(float(og.v[ip, jp]))
            if step % 1000 == 0:
                # radial checkerboard amplitude of the tangential velocity at the
                # near-wall cell: a large value => radial saw-tooth -> wall shear error
                jw = 1
                ut = -og.yc[:, jw] * og.u[:, jw] + og.xc[:, jw] * og.v[:, jw]
                chk = float(np.std(ut[::2] - ut[1::2]) / (np.std(ut) + 1e-12))
                print(f"  step{step:5d} t={t:6.2f} Cd={Cd:+.4f} Cl={Cl:+.4f} "
                      f"vwall={og.v[i_probe,j_probe]:+.4f} vprobe={og.v[ip,jp]:+.4f} "
                      f"chk={chk:.3f}")

        if step % field_every == 0 or step == nsteps:
            _dump_vorticity(og, out_prefix, step, R, Rf_og)

    if nan:
        print("RESULT: NaN")
        return

    # ---- spectral analysis ----
    cl_arr = np.array(cl_hist)
    t_arr = np.array(t_hist)
    cd_mean = float(np.mean(cd_hist[len(cd_hist)//4:]))
    cl_amp = float(np.max(cl_arr) - np.min(cl_arr)) / 2.0 if len(cl_arr) else 0.0
    # dominant frequency of Cl via FFT
    f_st, amp = _fft_peak(t_arr, cl_arr)
    f_wall, amp_wall = _fft_peak(t_arr, np.array(vw_hist))
    print(f"\nRESULT scheme={scheme} hv_cfl={hv_cfl} aoa={aoa_deg}")
    print(f"  Cd_mean={cd_mean:.4f}  Cl_amp={cl_amp:.4f}")
    print(f"  St (from Cl FFT) = {f_st:.4f}  (baseline 0.182)")
    print(f"  10Hz-mode near-wall v FFT peak = {f_wall:.4f}  amp={amp_wall:.4e}")
    # save time series + final field (for offline vorticity/striping analysis)
    np.savez(f"/workspace/{out_prefix}_series.npz",
             t=t_arr, cl=cl_arr, cd=np.array(cd_hist),
             vwall=np.array(vw_hist), vprobe=np.array(vprobe_hist))
    np.savez(f"/workspace/{out_prefix}_field.npz",
             xc=og.xc, yc=og.yc, r=og.r, u=og.u, v=og.v, p=og.p,
             ni=ni, nj=nj, R=R, Rf_og=Rf_og, beta=beta, rc=og.rc, dth=og.dth)


def _fft_peak(t, y):
    if len(y) < 8:
        return float("nan"), 0.0
    y = y - np.mean(y)
    dt = float(np.mean(np.diff(t)))
    n = len(y)
    f = np.fft.rfftfreq(n, d=dt)
    sp = np.abs(np.fft.rfft(y))
    sp[0] = 0.0
    if sp.max() <= 0:
        return 0.0, 0.0
    k = np.argmax(sp)
    return float(f[k]), float(sp[k])


def _dump_vorticity(og, out_prefix, step, R, Rf_og):
    w = vorticity(og)
    wc = np.clip(w, -3, 3)
    # (a) native polar pcolormesh (to check render-artifact striping)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    ax[0].pcolormesh(og.xc, og.yc, wc, cmap="RdBu_r", vmin=-3, vmax=3, shading="auto")
    ax[0].set_title(f"native polar pcolormesh (step {step})")
    ax[0].set_aspect("equal")
    # (b) Cartesian griddata interpolation
    from scipy.interpolate import griddata as gd
    nxg, nyg = 300, 120
    gx = np.linspace(-Rf_og, Rf_og, nxg)
    gy = np.linspace(-Rf_og, Rf_og, nyg)
    GX, GY = np.meshgrid(gx, gy)
    pts = np.column_stack([og.xc.ravel(), og.yc.ravel()])
    val = wc.ravel()
    wcc = gd(pts, val, (GX, GY), method="linear")
    inside = GX ** 2 + GY ** 2 <= Rf_og ** 2
    wcc = np.where(inside, wcc, np.nan)
    wcc = np.where(GX ** 2 + GY ** 2 < R ** 2, np.nan, wcc)
    im = ax[1].imshow(wcc, origin="lower", extent=[-Rf_og, Rf_og, -Rf_og, Rf_og],
                      cmap="RdBu_r", vmin=-3, vmax=3)
    ax[1].set_title(f"Cartesian griddata (step {step})")
    ax[1].set_aspect("equal")
    plt.colorbar(im, ax=ax, label="vorticity")
    plt.tight_layout()
    plt.savefig(f"/workspace/{out_prefix}_vorticity_step{step:05d}.png", dpi=130)
    plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", default="muscl", choices=["upwind1", "muscl"])
    ap.add_argument("--hv", type=float, default=0.0, help="local 4th-order hypervis cfl")
    ap.add_argument("--nsteps", type=int, default=12000)
    ap.add_argument("--nj", type=int, default=48)
    ap.add_argument("--beta", type=float, default=2.5)
    ap.add_argument("--aoa", type=float, default=3.0, help="angle of attack (deg)")
    ap.add_argument("--Rf_sponge", type=float, default=0.6)
    ap.add_argument("--fsig", type=float, default=0.0, help="theta 4th-order filter sigma")
    ap.add_argument("--fr", type=float, default=0.0, help="radial 4th-order filter sigma (velocity, pre-projection)")
    ap.add_argument("--fp", type=float, default=0.0, help="radial 4th-order filter sigma (pressure, post-projection)")
    ap.add_argument("--reg_r", type=float, default=0.0, help="radial 2nd-order Poisson regularisation (breaks checkerboard)")
    ap.add_argument("--rc", action="store_true", help="use Rhie-Chow in projection divergence (unstable on stretched O-grid)")
    ap.add_argument("--fv", type=float, default=0.0, help="radial Shapiro filter sigma on velocity AFTER projection (kills velocity checkerboard)")
    ap.add_argument("--out", default="standalone")
    a = ap.parse_args()
    run(a.scheme, a.hv, a.nsteps, a.nj, a.beta, a.aoa, a.Rf_sponge, out_prefix=a.out,
        fsig=a.fsig, fr=a.fr, fp=a.fp, reg_r=a.reg_r, rc=a.rc, fv=a.fv)
