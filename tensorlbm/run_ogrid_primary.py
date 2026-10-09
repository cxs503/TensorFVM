"""Body-fitted O-grid as the PRIMARY carrier (standalone, no background).

Demonstrates that the body-fitted polar O-grid sheds a clean Karman vortex
street AND gives the self-consistent force Cd ~ 1.59 (Re=100).  The O-grid is
fully decoupled: its outer fringe is clamped to the uniform free stream every
step, so this is exactly the og2bg driver-side behaviour (standalone-equivalent).

Rendering: Cartesian griddata interpolation of the polar O-grid fields -> 3
clean plots (velocity / pressure / vorticity) with NO radial grid striping.

Force: POG.cylinder_forces (face-vector-consistent on the O-grid wall, j=0).
       The polar projection pins p[fringe]=0; p_phys = sign(J)*p_proj and J<0
       on the O-grid, so the raw returned Cd has the wrong SIGN -- we report
       |Cd| (magnitude is the physical drag coefficient).
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

import polar_ogrid as POG
import overset_cylinder_fvm as M  # only for the Cartesian render helper if needed


def run(nsteps, nj, ni, beta, aoa_deg, Rf_og, Rf_sponge, scheme, fck, fp,
        reg_r, cb, rc, nsteps_per_field, out, pert_amp=0.0,
        pert_k0=0, pert_k1=0, nu_art=0.0, nu_hyp=0.0, weno_alpha=1.0,
        Re=100.0):
    R = 0.1
    cx, cy = 0.0, 0.0
    U = 1.0
    nu = U * (2 * R) / Re
    # allow overriding the physical Reynolds number (diagnostic: a higher Re lifts
    # the effective Re above the subcritical shedding threshold even with the PPM
    # numerical dissipation, so the wake sheds naturally without a strong kick).
    aoa = np.deg2rad(aoa_deg)

    og = POG.make_ogrid(ni, nj, cx, cy, R, Rf_og, beta=beta)
    og._nu = nu
    og.U = U
    og.recv = og.r >= Rf_sponge
    og.solved = ~og.recv

    # adaptive dt (same rule as the main sim)
    min_cell = float(np.sqrt(np.abs(og.J[og.solved])).min())
    dt_cfl = 0.2 * min_cell / U
    rn = og.rn
    dr_min = float(rn[1] - rn[0])
    dth_min = float(og.r.min() * (2.0 * np.pi / ni))
    dx_min = min(dr_min, dth_min)
    dt_diff = 0.25 * dx_min ** 2 / nu
    dt = min(dt_cfl, dt_diff)
    # explicit 4th-order hyperviscosity stability (only for the CENTRAL scheme):
    # dt * nu_hyp * k_max^4 < ~0.1 with k_max ~ 1/dx_min -> dt < 0.1*dx_min^4/nu_hyp.
    # Without this cap the near-wall dx^4 term makes the scheme blow up instantly.
    if scheme == "central" and nu_hyp > 0.0:
        dt_hyp = 0.1 * dx_min ** 4 / nu_hyp
        dt = min(dt, dt_hyp)
    # Poisson solver: the direct SuperLU factorization is robust to the indefinite /
    # near-singular polar Laplacian (tolerates the radial checkerboard null space)
    # and is fast per step (triangular solve).  The earlier OOM was the DENSE matrix
    # assembly in _lap_face_matrix, now fixed to sparse, so SuperLU factors N=32k in
    # ~44 s with modest memory.  We keep the iterative BiCGStab variant only as a
    # fallback for very large grids where SuperLU fill-in would explode.
    N = ni * nj
    if N <= 40000:
        poisson = POG.build_poisson(og, og.recv, reg_r=reg_r, dt=dt, rc=bool(rc))
    else:
        poisson = POG.build_poisson_iterative(og, og.recv, reg_r=reg_r, dt=dt, rc=bool(rc))
    ramp = 200
    print(f"[setup] ni={ni} nj={nj} beta={beta} aoa={aoa_deg} "
          f"Rf_og={Rf_og} Rf_sponge={Rf_sponge} scheme={scheme} "
          f"dt={dt:.3e} nu={nu:.3e} solved={og.solved.sum()}/{ni*nj}")

    cd_hist, cl_hist, t_hist = [], [], []
    nan = False
    for step in range(1, nsteps + 1):
        Ueff = U * min(1.0, step / ramp)
        # clamped uniform free-stream donor on the outer fringe (standalone)
        og.u[og.recv] = Ueff * np.cos(aoa)
        og.v[og.recv] = Ueff * np.sin(aoa)
        # predictor
        duc, dvc = POG.polar_conv(og, og.u, og.v, Ueff, scheme=scheme,
                              alpha_scale=weno_alpha)
        diff_u, diff_v = POG.polar_diffusion(og, og.u, og.v)
        if nu_art > 0.0:
            diff_u = diff_u + nu_art * POG.polar_lap(og, og.u)
            diff_v = diff_v + nu_art * POG.polar_lap(og, og.v)
        # 4th-order hyperviscosity: only damps the grid-scale (checkerboard /
        # odd-even) mode, NOT the large-scale Karman street -- so a low-diffusion
        # CENTRAL convection can stay stable AND shed (effective Re stays ~100).
        if scheme == "central" and nu_hyp > 0.0:
            lap2u = POG.polar_lap(og, POG.polar_lap(og, og.u))
            lap2v = POG.polar_lap(og, POG.polar_lap(og, og.v))
            diff_u = diff_u + nu_hyp * lap2u
            diff_v = diff_v + nu_hyp * lap2v
        ustar = og.u + dt * (-duc + diff_u)
        vstar = og.v + dt * (-dvc + diff_v)
        ustar[:, 0] = 0.0
        vstar[:, 0] = 0.0
        ustar[og.recv] = og.u[og.recv]
        vstar[og.recv] = og.v[og.recv]
        og.u, og.v = ustar, vstar
        # project (Rhie-Chow momentum interpolation when rc=1; stable only on
        # mildly stretched O-grid -- breaks the collocated radial checkerboard at
        # the SOURCE, so no post-hoc filter is needed)
        POG.project(og, dt, og.recv, poisson, rc=bool(rc))
        # targeted (-1)^j checkerboard PROJECTION (spectral: removes only the
        # radial null space, preserves the shedding). replaces the 3-pt average
        # (polar_filter_checkerboard) that low-passed the wake and locked shedding.
        if cb != "none":
            og.p = POG.polar_checkerboard_project(og, og.p, mode=cb)
            og.u = POG.polar_checkerboard_project(og, og.u, mode=cb)
            og.v = POG.polar_checkerboard_project(og, og.v, mode=cb)
        # legacy filters kept as fallback (default off)
        if fck > 0.0:
            og.u = POG.polar_filter_checkerboard(og, og.u, alpha=fck)
            og.v = POG.polar_filter_checkerboard(og, og.v, alpha=fck)
        if fp > 0.0:
            og.p = POG.polar_filter(og, og.p, sigma_r=fp)
        og.u[:, 0] = 0.0
        og.v[:, 0] = 0.0
        og.u[og.recv] = Ueff * np.cos(aoa)
        og.v[og.recv] = Ueff * np.sin(aoa)
        # transient coherent (lift) perturbation in the near wake to trigger the
        # Re=100 Hopf shedding.  Narrow radial envelope + smooth temporal ramp so
        # the kick reaches the asymmetric wake mode without injecting grid-scale
        # energy that the PPM+Rusanov scheme amplifies into a blow-up.  Use amp<=0.02
        # (amp>=0.025 blows up during the forcing window); lowering --weno_alpha
        # reduces numerical dissipation so this gentle kick can cross the subcritical
        # threshold and self-sustain.
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
            cd_hist.append(Cd)  # raw sign (J<0) -> flip when reporting
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
    # Diagnostics MUST use the CONVERGED window only: the first third holds the
    # ramp + the perturbation transient (huge Cl excursion) which inflates
    # cl_amp and drags the FFT peak to a spurious low frequency.
    n0 = len(cd) // 3
    cd_mean = float(np.mean(cd[n0:]))
    cl_win = cl[n0:]
    cl_amp = float(np.ptp(cl_win)) / 2.0
    cl_rms = float(np.sqrt(np.mean((cl_win - cl_win.mean()) ** 2)))
    f_st = _fft_peak(t[n0:], cl_win - cl_win.mean())
    print(f"\nRESULT O-grid-primary  nj={nj} ni={ni} aoa={aoa_deg}")
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
    # Cartesian grid for clean plotting (no radial striping)
    nxg, nyg = 400, 160
    gx = np.linspace(cx - Rf_og, cx + Rf_og, nxg)
    gy = np.linspace(cy - Rf_og, cy + Rf_og, nyg)
    GX, GY = np.meshgrid(gx, gy)
    pts = np.column_stack([og.xc.ravel(), og.yc.ravel()])
    inside = (GX - cx) ** 2 + (GY - cy) ** 2 <= Rf_og ** 2
    hole = (GX - cx) ** 2 + (GY - cy) ** 2 < R ** 2

    def interp(field, vmin=None, vmax=None):
        val = field.ravel()
        z = griddata(pts, val, (GX, GY), method="linear")
        z = np.where(inside, z, np.nan)
        z = np.where(hole, np.nan, z)
        return z

    w = POG.polar_grad(og, og.v)[1] - POG.polar_grad(og, og.u)[0]
    spd = np.sqrt(og.u ** 2 + og.v ** 2)
    pphys = -og.p  # J<0 -> p_phys = -p_proj (reference fringe=0)

    fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))
    # velocity
    zu = interp(spd)
    im0 = ax[0].imshow(zu, origin="lower", extent=[cx - Rf_og, cx + Rf_og,
                     cy - Rf_og, cy + Rf_og], cmap="viridis", vmin=0, vmax=1.6)
    sx = gx[::16]; sy = gy[::16]
    SX, SY = np.meshgrid(sx, sy)
    zu2 = interp(spd)
    uu = griddata(pts, og.u.ravel(), (SX, SY), method="linear")
    vv = griddata(pts, og.v.ravel(), (SX, SY), method="linear")
    ax[0].quiver(SX, SY, uu, vv, color="k", scale=28, width=0.0018, alpha=0.6)
    fig.colorbar(im0, ax=ax[0], label="|u|")
    ax[0].set_title(f"O-grid velocity (step {step})")
    # pressure
    zp = interp(pphys)
    vmx = np.nanmax(np.abs(zp)) * 0.9
    im1 = ax[1].imshow(zp, origin="lower", extent=[cx - Rf_og, cx + Rf_og,
                     cy - Rf_og, cy + Rf_og], cmap="RdBu_r", vmin=-vmx, vmax=vmx)
    fig.colorbar(im1, ax=ax[1], label="p_phys")
    ax[1].set_title(f"O-grid pressure (step {step})")
    # vorticity
    zw = interp(w, )
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
    ap.add_argument("--scheme", default="muscl", choices=["upwind1", "muscl", "vanleer", "central", "weno", "ppm"])
    ap.add_argument("--fck", type=float, default=1.0,
                    help="radial (-1)^j checkerboard removal on velocity")
    ap.add_argument("--fp", type=float, default=0.5,
                    help="radial pressure filter (clean p + correct force)")
    ap.add_argument("--reg_r", type=float, default=0.0)
    ap.add_argument("--cb", default="pair", choices=["none", "global", "pair"],
                    help="(-1)^j checkerboard projection: none / global / pair")
    ap.add_argument("--rc", type=int, default=0,
                    help="Rhie-Chow momentum interpolation (1=on); removes the "
                         "collocated radial checkerboard at the source")
    ap.add_argument("--field_every", type=int, default=2000)
    ap.add_argument("--pert_amp", type=float, default=0.0)
    ap.add_argument("--pert_k0", type=int, default=0)
    ap.add_argument("--pert_k1", type=int, default=0)
    ap.add_argument("--nu_art", type=float, default=0.0,
                    help="2nd-order artificial viscosity for the central scheme")
    ap.add_argument("--nu_hyp", type=float, default=0.0,
                    help="4th-order hyperviscosity (central only); damps the "
                         "grid-scale checkerboard so central can stay stable")
    ap.add_argument("--weno_alpha", type=float, default=1.0,
                    help="Rusanov dissipation scale for the WENO scheme; >1 adds "
                         "robustness against the dispersion instability")
    ap.add_argument("--Re", type=float, default=100.0,
                    help="physical Reynolds number (default 100); raise it to lift "
                         "the effective Re above the subcritical shedding threshold")
    ap.add_argument("--out", default="ogrid_primary")
    a = ap.parse_args()
    run(a.nsteps, a.nj, a.ni, a.beta, a.aoa, a.Rf_og, a.Rf_sponge, a.scheme,
        a.fck, a.fp, a.reg_r, a.cb, a.rc, a.field_every, a.out,
        a.pert_amp, a.pert_k0, a.pert_k1, a.nu_art, a.nu_hyp, a.weno_alpha,
        Re=a.Re)
