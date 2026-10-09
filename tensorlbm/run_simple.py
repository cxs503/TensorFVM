"""SIMPLE (Semi-Implicit Method for Pressure-Linked Equations) driver on the
body-fitted polar O-grid -- a SECOND algorithm alongside run_ogrid_primary.py
(Chorin projection).

The two solvers share the SAME underlying operators (make_ogrid / polar_conv /
polar_grad / polar_div / _lap_face_matrix / build_poisson / cylinder_forces), so
their steady-state drag must agree and their shedding behaviour is directly
comparable.  SIMPLE's distinguishing features vs the projection method:
  * implicit diffusion  (I/dt - nu*L) u* = RHS  -> no dt_diff CFL cap, larger dt
  * per-step pressure-correction OUTER iterations + under-relaxation (urf)
  * Rhie-Chow mass-flux divergence drives the pressure-correction equation

NOTE: SIMPLE does NOT change the Re=100 SUBCRITICAL shedding conclusion -- that
is a physical / resolution bottleneck (numerical diffusion pins the effective Re
below the ~47 threshold), independent of the coupling scheme.  SIMPLE's payoff
is the larger stable CFL and the robust pressure-velocity coupling, plus a
cross-check of the drag.
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

import polar_ogrid as POG


def run(nsteps, nj, ni, beta, aoa_deg, Rf_og, Rf_sponge, scheme,
        niters, urf_p, urf_u, rc, nsteps_per_field, out,
        pert_amp=0.0, pert_k0=0, pert_k1=0, weno_alpha=1.0,
        Re=100.0, cfl=0.5):
    R = 0.1
    cx, cy = 0.0, 0.0
    U = 1.0
    nu = U * (2 * R) / Re
    aoa = np.deg2rad(aoa_deg)

    og = POG.make_ogrid(ni, nj, cx, cy, R, Rf_og, beta=beta)
    og._nu = nu
    og.U = U
    og.recv = og.r >= Rf_sponge
    og.solved = ~og.recv

    # ---- SIMPLE dt: implicit diffusion removes the dt_diff = 0.25*dx^2/nu cap.
    # Convection is still explicit, so dt is set by a CFL rule (cfl*dx_min/U),
    # which the implicit diffusion lets us push to ~0.5-1.0 (vs 0.2 for the
    # explicit projection method).
    min_cell = float(np.sqrt(np.abs(og.J[og.solved])).min())
    dr_min = float(og.rn[1] - og.rn[0])
    dth_min = float(og.r.min() * (2.0 * np.pi / ni))
    dx_min = min(dr_min, dth_min)
    dt_cfl = cfl * min_cell / U
    dt_diff_explicit = 0.25 * dx_min ** 2 / nu     # (reference only -- not used)
    dt = dt_cfl

    N = ni * nj
    # pressure-correction Poisson.  rc=False uses the SAME self-adjoint
    # polar_div/polar_grad pair as the projection method, so the SIMPLE velocity
    # correction u = u* - dt*grad(p') EXACTLY cancels the divergence (the Rhie-
    # Chow rc=True variant is NOT self-adjoint with that correction and diverges
    # in the outer iterations).  The plain radial Laplacian carries a (-1)^j
    # checkerboard null space, so reg_r=200 pins it (negligible on the smooth
    # physical pressure, like the projection method's iterative build_poisson).
    if N <= 40000:
        poisson = POG.build_poisson(og, og.recv, reg_r=200.0, dt=dt, rc=False)
    else:
        poisson = POG.build_poisson_iterative(og, og.recv, reg_r=200.0, dt=dt, rc=False)
    # implicit momentum matrix A = I/dt - nu*L
    mom = POG.build_momentum(og, og.recv, dt, nu, reg_r=0.0)
    ramp = 200
    print(f"[setup] SIMPLE ni={ni} nj={nj} beta={beta} aoa={aoa_deg} "
          f"scheme={scheme} rc={rc} niters={niters} urf_p={urf_p} urf_u={urf_u} "
          f"dt={dt:.3e} dt_diff(explicit)={dt_diff_explicit:.3e} "
          f"nu={nu:.3e} solved={og.solved.sum()}/{N}")

    cd_hist, cl_hist, t_hist = [], [], []
    nan = False
    for step in range(1, nsteps + 1):
        Ueff = U * min(1.0, step / ramp)
        # clamped uniform free-stream on the outer fringe (standalone)
        og.u[og.recv] = Ueff * np.cos(aoa)
        og.v[og.recv] = Ueff * np.sin(aoa)
        # one SIMPLE step (implicit momentum + niters pressure-correction iters)
        POG.simple_step(og, dt, mom, poisson, scheme, nu,
                        niters=niters, urf_p=urf_p, urf_u=urf_u, rc=bool(rc))
        # re-impose fringe after the correction (simple_step keeps wall only)
        og.u[og.recv] = Ueff * np.cos(aoa)
        og.v[og.recv] = Ueff * np.sin(aoa)
        # transient coherent (lift) perturbation to trigger the Hopf shedding
        if pert_k0 <= step <= pert_k1:
            edge = min(step - pert_k0, pert_k1 - step)
            ramp_p = min(1.0, edge / 400.0)
            vpert = (pert_amp * U * ramp_p * np.sin(og.thn[:, None])
                     * np.exp(-((og.r - 0.4) / 0.2) ** 2))
            og.v[og.solved] += vpert[og.solved]

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
                print(f"  step{step:5d} t={t:6.2f} Cd_raw={Cd:+.4f} Cl={Cl:+.4f}")

        if step % nsteps_per_field == 0 or step == nsteps:
            _render(og, out, step, R, Rf_og, cx, cy)

    if nan:
        print("RESULT: NaN")
        return

    cd = np.array(cd_hist)
    cl = np.array(cl_hist)
    t = np.array(t_hist)
    n0 = len(cd) // 3
    cd_mean = float(np.mean(cd[n0:]))
    cl_win = cl[n0:]
    cl_amp = float(np.ptp(cl_win)) / 2.0
    cl_rms = float(np.sqrt(np.mean((cl_win - cl_win.mean()) ** 2)))
    f_st = _fft_peak(t[n0:], cl_win - cl_win.mean())
    print(f"\nRESULT SIMPLE  nj={nj} ni={ni} aoa={aoa_deg} scheme={scheme}")
    print(f"  |Cd|_mean = {abs(cd_mean):.4f}   (Cd_raw sign flipped: {cd_mean:+.4f})")
    print(f"  Cl_amp    = {cl_amp:.4f}   Cl_rms = {cl_rms:.4f}  "
          f"(converged window, t>={t[n0]:.2f})")
    print(f"  St (Cl FFT, converged) = {f_st * (2.0 * R) / U:.4f}   (baseline 0.182)")
    print(f"  viscous/pressure decomp (final): {POG.force_decomp(og, nu, U)}")
    np.savez(f"/workspace/{out}_fields.npz",
             xc=og.xc, yc=og.yc, r=og.r, u=og.u, v=og.v, p=og.p,
             R=R, Rf_og=Rf_og, cx=cx, cy=cy)
    np.savez(f"/workspace/{out}_hist.npz",
             t=t, cd=cd, cl=cl, dt=dt, nu=nu, R=R, U=U)


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
        val = field.ravel()
        z = griddata(pts, val, (GX, GY), method="linear")
        z = np.where(inside, z, np.nan)
        z = np.where(hole, np.nan, z)
        return z

    w = POG.polar_grad(og, og.v)[1] - POG.polar_grad(og, og.u)[0]
    spd = np.sqrt(og.u ** 2 + og.v ** 2)
    pphys = -og.p

    fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))
    zu = interp(spd)
    im0 = ax[0].imshow(zu, origin="lower", extent=[cx - Rf_og, cx + Rf_og,
                     cy - Rf_og, cy + Rf_og], cmap="viridis", vmin=0, vmax=1.6)
    sx = gx[::16]; sy = gy[::16]
    SX, SY = np.meshgrid(sx, sy)
    uu = griddata(pts, og.u.ravel(), (SX, SY), method="linear")
    vv = griddata(pts, og.v.ravel(), (SX, SY), method="linear")
    ax[0].quiver(SX, SY, uu, vv, color="k", scale=28, width=0.0018, alpha=0.6)
    fig.colorbar(im0, ax=ax[0], label="|u|")
    ax[0].set_title(f"O-grid velocity (step {step})")
    zp = interp(pphys)
    vmx = np.nanmax(np.abs(zp)) * 0.9
    im1 = ax[1].imshow(zp, origin="lower", extent=[cx - Rf_og, cx + Rf_og,
                     cy - Rf_og, cy + Rf_og], cmap="RdBu_r", vmin=-vmx, vmax=vmx)
    fig.colorbar(im1, ax=ax[1], label="p_phys")
    ax[1].set_title(f"O-grid pressure (step {step})")
    zw = interp(w)
    im2 = ax[2].imshow(zw, origin="lower", extent=[cx - Rf_og, cx + Rf_og,
                     cy - Rf_og, cy + Rf_og], cmap="RdBu_r", vmin=-3, vmax=3)
    fig.colorbar(im2, ax=ax[2], label="vorticity")
    ax[2].set_title(f"O-grid vorticity (step {step})")
    for a in ax:
        a.set_aspect("equal")
        a.set_xlim(cx - Rf_og, cx + Rf_og)
        a.set_ylim(cy - Rf_og, cy + Rf_og)
    th = np.linspace(0, 2 * np.pi, 240)
    for a in ax:
        a.plot(cx + R * np.cos(th), cy + R * np.sin(th), "k-", lw=1.2)
    plt.tight_layout()
    fig.savefig(f"/workspace/{out}_step{step:05d}.png", dpi=130)
    plt.close()
    print(f"  rendered /workspace/{out}_step{step:05d}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsteps", type=int, default=12000)
    ap.add_argument("--nj", type=int, default=48)
    ap.add_argument("--ni", type=int, default=120)
    ap.add_argument("--beta", type=float, default=2.5)
    ap.add_argument("--aoa", type=float, default=3.0)
    ap.add_argument("--Rf_og", type=float, default=0.95)
    ap.add_argument("--Rf_sponge", type=float, default=0.76)
    ap.add_argument("--scheme", default="ppm", choices=["upwind1", "muscl", "vanleer", "central", "weno", "ppm"])
    ap.add_argument("--niters", type=int, default=6,
                    help="SIMPLE outer (pressure-correction) iterations per step")
    ap.add_argument("--urf_p", type=float, default=0.7, help="pressure under-relaxation")
    ap.add_argument("--urf_u", type=float, default=1.0, help="velocity under-relaxation")
    ap.add_argument("--rc", type=int, default=0,
                    help="Rhie-Chow mass-flux divergence in the pressure correction. "
                         "NOTE: rc=1 is NOT self-adjoint with the u=u*-dt*grad(p') "
                         "correction and diverges in the SIMPLE outer iterations; "
                         "keep rc=0 (uses the same self-adjoint polar_div/grad as the "
                         "projection method, reg_r pins the checkerboard null space).")
    ap.add_argument("--cfl", type=float, default=0.5,
                    help="SIMPLE CFL (implicit diffusion -> can exceed the "
                         "projection method's 0.2)")
    ap.add_argument("--field_every", type=int, default=2000)
    ap.add_argument("--pert_amp", type=float, default=0.0)
    ap.add_argument("--pert_k0", type=int, default=0)
    ap.add_argument("--pert_k1", type=int, default=0)
    ap.add_argument("--weno_alpha", type=float, default=1.0,
                    help="Rusanov dissipation scale for the WENO scheme")
    ap.add_argument("--Re", type=float, default=100.0)
    ap.add_argument("--out", default="simple_ogrid")
    a = ap.parse_args()
    run(a.nsteps, a.nj, a.ni, a.beta, a.aoa, a.Rf_og, a.Rf_sponge, a.scheme,
        a.niters, a.urf_p, a.urf_u, a.rc, a.field_every, a.out,
        a.pert_amp, a.pert_k0, a.pert_k1, a.weno_alpha, a.Re, a.cfl)
