"""Generic curvilinear C-grid flow solver driver (Chorin projection method).

Drives either geometry on the SAME generic curve-coordinate FVM layer
(tensorlbm.curve_grid) that was validated for metric consistency:

    * cylinder : full-circle C-grid, j=0 = no-slip wall (whole body), wake
                 trails downstream and is resolved up to the far field; the two
                 cut columns are clamped to the uniform free stream.
    * airfoil  : NACA 4-digit C-grid, j=0 = airfoil arc only (no-slip); the two
                 wake-centreline halves and the far field are free stream.

Boundary conditions
    wall  (no-slip, j=0)        : u = v = 0  (enforced every step)
    recv  (far field + cut cols): u = U cos(aoa), v = U sin(aoa)  (clamped)
The far-field / wake-cut clamp makes the C-grid a closed problem exactly like the
standalone O-grid (its outer fringe is also clamped) -- no background grid needed.

Force: tensorlbm.curve_grid.c_forces integrates pressure + viscous stress on the
wall (j=0) face vectors.  For a non-zero angle of attack the returned (Fx,Fy) are
rotated into the INCOMING-flow frame to give the physical Cd (along the flow) and
Cl (perpendicular).

Usage (steady Cd estimate, cylinder Re=100):
    python3 run_cgrid.py --geometry cylinder --Re 100 --nsteps 8000 --steady

Usage (airfoil sweep, alpha=4 deg):
    python3 run_cgrid.py --geometry airfoil --aoa 4 --Re 100 --nsteps 12000
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

import tensorlbm.curve_grid as CG
import tensorlbm.cgrid_gen as GEN


def _build_block(geometry, ni, nj, Rf, beta, Lw, aoa_deg, n_surf):
    if geometry == "cylinder":
        X, Y, wall, recv, Lref = GEN.make_cylinder_cgrid(
            Rb=1.0, Rf=Rf, ni=ni, nj=nj, beta=beta)
        U = 1.0
    else:
        X, Y, wall, recv, Lref = GEN.make_naca_cgrid(
            m=0.0, p=0.0, t=0.12, ni=ni, nj=nj, Rf=Rf, beta=beta,
            Lw=Lw, n_surf=n_surf)
        U = 1.0
    blk = CG.make_cblock(X, Y, periodic_i=False, U=U, Lref=Lref, wall_mask=wall)
    return blk, X, Y, wall, recv, Lref, U


def run(geometry, nsteps, nj, ni, Rf, beta, Lw, aoa_deg, scheme, cb, reg_r,
        nu_hyp, Re, steady, field_every, out, pert_amp=0.0, pert_k0=0, pert_k1=0,
        n_surf=200, mode="adjoint", restart=None, save_every=0):
    """Checkpoint / restart: pass --restart <ckpt.npz> (produced by
    --save_every) to resume a killed long run; the time loop continues from the
    saved step and force history, so a convergence run can be split into
    survive-able segments instead of one job that hits the background time
    limit.  See run_batch.py for the multi-core strategy (parallel independent
    cases), which is where the 32 cores are actually useful at this grid size.
    """
    pmode = mode
    aoa = np.deg2rad(aoa_deg)
    blk, X, Y, wall, recv, Lref, U = _build_block(
        geometry, ni, nj, Rf, beta, Lw, aoa_deg, n_surf)

    # physical viscosity (cylinder Lref = diameter = 2; airfoil Lref = chord = 1)
    if geometry == "cylinder":
        nu = U * (2.0 * 1.0) / Re
    else:
        nu = U * 1.0 / Re
    blk._nu = nu
    blk.U = U
    blk.recv = recv
    solved = ~recv

    # initialise uniform free stream everywhere, then impose the BCs
    blk.u = np.full((ni, nj), U * np.cos(aoa))
    blk.v = np.full((ni, nj), U * np.sin(aoa))
    blk.u[wall] = 0.0; blk.v[wall] = 0.0
    blk.p = np.zeros((ni, nj))

    # ---- checkpoint / restart ----
    # Long runs are split into survive-able segments: run with --save_every N to
    # dump <out>_ckpt.npz every N steps, then relaunch with
    # --restart <out>_ckpt.npz to continue.  The Poisson operator is rebuilt
    # deterministically from the grid args, so only (u,v,p) + step + force
    # history need to be restored.
    start_step = 0
    cd_hist, cl_hist, t_hist = [], [], []
    if restart is not None:
        ck = np.load(restart)
        blk.u = ck["u"].astype(float).copy()
        blk.v = ck["v"].astype(float).copy()
        blk.p = ck["p"].astype(float).copy()
        start_step = int(ck["step"])
        if "cd" in ck.files and ck["cd"].size:
            cd_hist = list(ck["cd"].astype(float))
            cl_hist = list(ck["cl"].astype(float))
            t_hist = list(ck["t"].astype(float))
        print(f"[restart] from {restart} at step {start_step}, "
              f"hist_len={len(cd_hist)}")

    # adaptive dt: CFL on the smallest cell + viscous stability
    solvedJ = blk.J[solved]
    min_cell = float(np.sqrt(np.abs(solvedJ).min()))
    dt_cfl = 0.2 * min_cell / U
    dx_min = min_cell
    dt_diff = 0.25 * dx_min ** 2 / nu if nu > 0 else 1e9
    dt = min(dt_cfl, dt_diff)
    if scheme == "central" and nu_hyp > 0.0:
        dt = min(dt, 0.1 * dx_min ** 4 / nu_hyp)
    dt = min(dt, 5e-2)  # hard cap for safety

    # Poisson factorisation.  mode="adjoint" (DEFAULT) uses the unweighted
    # c_div o c_grad operator -- stable & accurate on stretched C-grids.
    # mode="rhie" uses the Rhie-Chow rAU=dt/J operator (OpenFOAM-style, suppresses
    # the collocated pressure checkerboard).  Its condition number ~ rAU_ratio*N^2
    # (rAU_ratio up to ~1e7) exceeds double precision, so a naive ILU/BiCGStab
    # solver diverges; instead the SIMPLE+AMG upgrade solves it with a geometric
    # multigrid (_MGSolver) whose convergence is independent of kappa.  The
    # projection loop already carries the SIMPLE structure (pRelax under-relaxation
    # + multiple inner corrections), so Rhie-Chow is now usable on stretched grids
    # exactly as in OpenFOAM.
    N = ni * nj
    if N <= 45000:
        poisson = CG.build_poisson(blk, blk.recv, reg_r=reg_r, dt=dt, mode=pmode)
    else:
        poisson = CG.build_poisson_iterative(blk, blk.recv, reg_r=reg_r, dt=dt, mode=pmode)

    ramp = 200
    print(f"[setup] geometry={geometry} ni={ni} nj={nj} Rf={Rf} aoa={aoa_deg} "
          f"Re={Re} scheme={scheme} cb={cb} reg_r={reg_r}")
    print(f"  Lref={Lref} nu={nu:.4e} dt={dt:.3e} "
          f"solved={solved.sum()}/{N} Jmin={np.abs(blk.J).min():.3e} "
          f"Jmax={blk.J.max():.2e}")

    nan = False
    for step in range(start_step + 1, nsteps + 1):
        Ueff = U * min(1.0, step / ramp)
        cu, cv = Ueff * np.cos(aoa), Ueff * np.sin(aoa)
        # clamp free-stream fringe (far field + wake cut)
        blk.u[recv] = cu; blk.v[recv] = cv

        # ---- predictor (explicit) ----
        duc, dvc = CG.c_conv(blk, blk.u, blk.v, scheme=scheme)
        diff_u, diff_v = CG.c_diffusion(blk, blk.u, blk.v)
        if scheme == "central" and nu_hyp > 0.0:
            diff_u = diff_u + nu_hyp * CG.c_lap(blk, CG.c_lap(blk, blk.u))
            diff_v = diff_v + nu_hyp * CG.c_lap(blk, CG.c_lap(blk, blk.v))
        ustar = blk.u + dt * (-duc + diff_u)
        vstar = blk.v + dt * (-dvc + diff_v)
        # enforce BCs on the predicted field
        ustar[wall] = 0.0; vstar[wall] = 0.0
        ustar[recv] = cu; vstar[recv] = cv
        blk.u, blk.v = ustar, vstar

        # ---- projection (Chorin) ----
        CG.project(blk, dt, poisson)
        # optional checkerboard removal (collocated pressure null space)
        if cb != "none":
            blk.p = CG.c_checkerboard_project(blk, blk.p, mode=cb)
            blk.u = CG.c_checkerboard_project(blk, blk.u, mode=cb)
            blk.v = CG.c_checkerboard_project(blk, blk.v, mode=cb)
        # re-impose hard BCs after the pressure correction
        blk.u[wall] = 0.0; blk.v[wall] = 0.0
        blk.u[recv] = cu; blk.v[recv] = cv

        # perturbation to trigger Re=100 shedding in the unsteady mode
        if pert_k0 <= step <= pert_k1 and pert_amp > 0.0:
            edge = min(step - pert_k0, pert_k1 - step)
            ramp_p = min(1.0, edge / 400.0)
            yc = blk.yc
            vpert = (pert_amp * U * ramp_p * np.sin(3.0 * np.pi * blk.xc / Lref)
                     * np.exp(-((yc) / (0.5 * Lref)) ** 2))
            blk.v[solved] += vpert[solved]

        if not (np.isfinite(blk.u).all() and np.isfinite(blk.v).all()
                and np.isfinite(blk.p).all()):
            nan = True
            print(f"  NaN at step {step}")
            break

        if step >= ramp:
            Cd, Cl, Fx, Fy = CG.c_forces(blk, nu, U)
            # rotate (Fx,Fy) into the incoming-flow frame -> physical Cd/Cl
            Fdrag = Fx * np.cos(aoa) + Fy * np.sin(aoa)
            Flift = -Fx * np.sin(aoa) + Fy * np.cos(aoa)
            q = 0.5 * U * U * Lref
            cd_mean = Fdrag / q; cl_mean = Flift / q
            t = step * dt
            t_hist.append(t); cd_hist.append(cd_mean); cl_hist.append(cl_mean)
            if step % 1000 == 0 or (steady and step % 500 == 0):
                print(f"  step{step:5d} t={t:7.2f} Cd={cd_mean:+.4f} "
                      f"Cl={cl_mean:+.4f}")

        if field_every > 0 and (step % field_every == 0 or step == nsteps):
            _render(blk, out, step, geometry, Lref, aoa)

        # periodic checkpoint so a killed long run can be resumed with --restart
        if save_every > 0 and (step % save_every == 0):
            np.savez(f"/workspace/{out}_ckpt.npz", u=blk.u, v=blk.v, p=blk.p,
                     step=step, t=np.array(t_hist, dtype=float),
                     cd=np.array(cd_hist, dtype=float),
                     cl=np.array(cl_hist, dtype=float),
                     dt=dt, ni=ni, nj=nj)
            print(f"  checkpoint saved @ step {step} -> /workspace/{out}_ckpt.npz")

    if nan:
        print("RESULT: NaN")
        return

    cd = np.array(cd_hist); cl = np.array(cl_hist); t = np.array(t_hist)
    if cd.size == 0:
        print("RESULT: no force history (%d steps < ramp=%d); Cd/Cl not recorded"
              % (nsteps, ramp))
        return
    n0 = max(1, len(cd) // 3)
    cd_mean = float(np.mean(cd[n0:]))
    cl_win = cl[n0:]
    cl_amp = float(np.ptp(cl_win)) / 2.0
    cl_rms = float(np.sqrt(np.mean((cl_win - cl_win.mean()) ** 2)))
    f_st = _fft_peak(t[n0:], cl_win - cl_win.mean()) if len(cl_win) >= 8 else 0.0
    print(f"\nRESULT cgrid {geometry}  ni={ni} nj={nj} aoa={aoa_deg} Re={Re}")
    print(f"  Cd_mean = {cd_mean:.4f}   (converged window t>={t[n0]:.2f})")
    print(f"  Cl_mean = {np.mean(cl_win):+.4f}  Cl_amp = {cl_amp:.4f}  "
          f"Cl_rms = {cl_rms:.4f}")
    if f_st > 0.0:
        print(f"  St (Cl FFT) = {f_st * Lref / U:.4f}")
    np.savez(f"/workspace/{out}_fields.npz", xc=blk.xc, yc=blk.yc, u=blk.u,
             v=blk.v, p=blk.p, Lref=Lref, aoa=aoa_deg,
             # force time-series (from ramp onward) -- essential for GCI / convergence
             t=t, cd=cd, cl=cl, ni=ni, nj=nj, dt=dt, Rf=Rf, beta=beta, n_surf=n_surf,
             step=nsteps)


def _fft_peak(t, y):
    y = y - np.mean(y)
    dt = float(np.mean(np.diff(t))) if len(t) > 1 else 1.0
    f = np.fft.rfftfreq(len(y), d=dt)
    sp = np.abs(np.fft.rfft(y)); sp[0] = 0.0
    return float(f[np.argmax(sp)]) if sp.max() > 0 else 0.0


def run_three(nsteps, aoa_deg, Re, scheme, cb, reg_r, nu_hyp, mode, backend,
              device, dtype, Rhole, n_fringe, ramp, field_every, out,
              ni=121, nj=51):
    """Three-element (slat+main+flap) multi-block overset driver.

    Builds the VALIDATED OPEN-SLOT rig (slat TE ~0.6 above main LE, flap LE ~0.4
    below main TE, both deflected -20 deg; Rhole=0.08, n_fringe=1) and solves it
    with either the numpy driver (run_multiblock) or the PyTorch port
    (run_torch_multiblock).  Forces are normalised by the main chord exactly like
    three_element_validate.py, so this is the command-line entry to the P4 rig.
    """
    from tensorlbm.cgrid_gen import make_three_element_cgrid
    from tensorlbm.multiblock import run_multiblock
    te = make_three_element_cgrid(
        slat=(1.0, 0.0, 0.0, 0.12, -20.0, -0.99, 0.94),
        main=(1.0, 0.02, 0.4, 0.12, 0.0, 0.0, 0.0),
        flap=(1.0, 0.0, 0.0, 0.12, -20.0, 1.10, -0.40),
        ni_m=ni, nj_m=nj, ni_s=ni, nj_s=nj, ni_f=ni, nj_f=nj)
    if backend == "torch":
        from tensorlbm.torch_multiblock import run_torch_multiblock
        import torch
        dt = torch.float64 if dtype == "float64" else torch.float32
        res = run_torch_multiblock(te, nsteps=nsteps, aoa_deg=aoa_deg, Re=Re,
                                   scheme=scheme, cb=cb, reg_r=reg_r,
                                   nu_hyp=nu_hyp, mode=mode, n_fringe=n_fringe,
                                   Rhole=Rhole, ramp=ramp, field_every=field_every,
                                   out=out, device=device, dtype=dt)
    else:
        res = run_multiblock(te, nsteps=nsteps, aoa_deg=aoa_deg, Re=Re,
                             scheme=scheme, cb=cb, reg_r=reg_r, nu_hyp=nu_hyp,
                             mode=mode, n_fringe=n_fringe, Rhole=Rhole, ramp=ramp,
                             field_every=field_every, out=out)
    if res["nan"]:
        print("RESULT: NaN")
        return
    cd = np.array(res["cd"]); cl = np.array(res["cl"]); t = np.array(res["t"])
    if cd.size == 0:
        print("RESULT: no force history (run too short)")
        return
    n0 = max(1, len(cd) // 3)
    print(f"\nRESULT three-element aoa={aoa_deg} Re={Re} backend={backend} "
          f"device={device} dtype={dtype}")
    print(f"  Cd_mean={np.mean(cd[n0:]):.4f}  Cl_mean={np.mean(cl[n0:]):+.4f}"
          f"  (window t>={t[n0]:.2f})")
    np.savez(f"/workspace/{out}_fields.npz",
             t=t, cd=cd, cl=cl, dt=res["dt"],
             geometry="three_element", aoa=aoa_deg, Re=Re, backend=backend)


def _render(blk, out, step, geometry, Lref, aoa):
    xc, yc = blk.xc, blk.yc
    xmin, xmax = xc.min(), xc.max(); ymin, ymax = yc.min(), yc.max()
    nxg, nyg = 360, 200
    gx = np.linspace(xmin, xmax, nxg); gy = np.linspace(ymin, ymax, nyg)
    GX, GY = np.meshgrid(gx, gy)
    pts = np.column_stack([xc.ravel(), yc.ravel()])
    def interp(field):
        return griddata(pts, field.ravel(), (GX, GY), method="linear")
    w = CG.c_grad(blk, blk.v)[0] - CG.c_grad(blk, blk.u)[1]
    spd = np.sqrt(blk.u ** 2 + blk.v ** 2)
    pphys = blk.p
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    zu = interp(spd)
    im0 = ax[0].imshow(zu, origin="lower", extent=[xmin, xmax, ymin, ymax],
                       cmap="viridis", vmin=0, vmax=1.4)
    fig.colorbar(im0, ax=ax[0], label="|u|")
    ax[0].set_title(f"{geometry} velocity (step {step})")
    zp = interp(pphys)
    vmx = np.nanmax(np.abs(zp)) * 0.9 + 1e-9
    im1 = ax[1].imshow(zp, origin="lower", extent=[xmin, xmax, ymin, ymax],
                      cmap="RdBu_r", vmin=-vmx, vmax=vmx)
    fig.colorbar(im1, ax=ax[1], label="p")
    ax[1].set_title(f"{geometry} pressure (step {step})")
    zw = interp(w)
    im2 = ax[2].imshow(zw, origin="lower", extent=[xmin, xmax, ymin, ymax],
                       cmap="RdBu_r", vmin=-3, vmax=3)
    fig.colorbar(im2, ax=ax[2], label="vorticity")
    ax[2].set_title(f"{geometry} vorticity (step {step})")
    for a in ax:
        a.set_aspect("equal")
        # wall outline
        j0 = 0
        a.plot(blk.xc[:, j0], blk.yc[:, j0], "k-", lw=1.2)
    plt.tight_layout()
    fig.savefig(f"/workspace/{out}_step{step:05d}.png", dpi=120)
    plt.close()
    print(f"  rendered /workspace/{out}_step{step:05d}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", default="cylinder",
                    choices=["cylinder", "airfoil", "three_element"],
                    help="cylinder/airfoil = single-block C-grid (validated "
                         "metric layer); three_element = slat+main+flap multi-block "
                         "overset rig (see three_element_validate.py)")
    ap.add_argument("--nsteps", type=int, default=8000)
    ap.add_argument("--nj", type=int, default=71)
    ap.add_argument("--ni", type=int, default=221)
    ap.add_argument("--Rf", type=float, default=15.0)
    ap.add_argument("--beta", type=float, default=3.0)
    ap.add_argument("--Lw", type=float, default=None,
                    help="wake length (airfoil); defaults to Rf")
    ap.add_argument("--aoa", type=float, default=0.0)
    ap.add_argument("--scheme", default="ppm",
                    choices=["upwind1", "muscl", "vanleer", "central", "weno", "ppm"])
    ap.add_argument("--cb", default="pair", choices=["none", "global", "pair"])
    ap.add_argument("--reg_r", type=float, default=0.0)
    ap.add_argument("--mode", default="adjoint", choices=["adjoint", "rhie"],
                    help="pressure operator: adjoint (stable, default) or "
                         "rhie (Rhie-Chow rAU=dt/J; OpenFOAM-style but only "
                         "stable on near-uniform grids)")
    ap.add_argument("--backend", default="cpu", choices=["cpu", "torch"],
                    help="solver backend: 'cpu' = numpy/scipy (default, "
                         "validated); 'torch' = PyTorch (GPU-ready) port in "
                         "tensorlbm.torch_cgrid (device/dtype selected below)")
    ap.add_argument("--device", default="cpu",
                    help="torch device when --backend torch (e.g. cpu, cuda)")
    ap.add_argument("--torch_dtype", default="float64",
                    choices=["float32", "float64"],
                    help="torch dtype when --backend torch")
    ap.add_argument("--Rhole", type=float, default=0.08,
                    help="(three_element) hole distance threshold -- keep small "
                         "so slat/flap walls sit clear of the main's lift surface")
    ap.add_argument("--n_fringe", type=int, default=1,
                    help="(three_element) overset fringe width")
    ap.add_argument("--ramp", type=int, default=50,
                    help="(three_element) freestream ramp steps; the open-slot "
                         "high-lift needs ~0.3 time-units at full speed to develop")
    ap.add_argument("--nu_hyp", type=float, default=0.0)
    ap.add_argument("--Re", type=float, default=100.0)
    ap.add_argument("--steady", action="store_true",
                    help="just report converged Cd/Cl (no shedding diagnostics)")
    ap.add_argument("--field_every", type=int, default=2000)
    ap.add_argument("--pert_amp", type=float, default=0.0)
    ap.add_argument("--pert_k0", type=int, default=0)
    ap.add_argument("--pert_k1", type=int, default=0)
    ap.add_argument("--n_surf", type=int, default=200)
    ap.add_argument("--out", default="cgrid_run")
    ap.add_argument("--restart", default=None,
                    help="resume from a checkpoint npz (u,v,p,step,histories) "
                         "produced by --save_every")
    ap.add_argument("--save_every", type=int, default=0,
                    help="save checkpoint every N steps (split long runs to "
                         "survive background time limits)")
    a = ap.parse_args()
    if a.geometry == "three_element":
        # multi-block overset rig (P4/P5): numpy or torch backend.
        # Pin to the validated 121x51 per-block grid (the run_cgrid defaults
        # 221x71 are for the single-block cylinder/airfoil and would make each
        # three-element block ~15.7k cells -> heavy dense Poisson).  --ni/--nj
        # are ignored for three_element; tune via the element ni_* in cgrid_gen.
        run_three(a.nsteps, a.aoa, a.Re, a.scheme, a.cb, a.reg_r, a.nu_hyp,
                  a.mode, a.backend, a.device, a.torch_dtype, a.Rhole,
                  a.n_fringe, a.ramp, a.field_every, a.out,
                  ni=121, nj=51)
    elif a.backend == "torch":
        import tensorlbm.torch_cgrid as TC
        dt = getattr(__import__("torch"), a.torch_dtype)
        TC.run_torch(a.geometry, a.nsteps, a.nj, a.ni, a.Rf, a.beta, a.Lw,
                     a.aoa, a.scheme, a.cb, a.reg_r, a.nu_hyp, a.Re, a.steady,
                     a.field_every, a.out, a.pert_amp, a.pert_k0, a.pert_k1,
                     a.n_surf, a.mode, a.restart, a.save_every,
                     device=a.device, dtype=dt)
    else:
        run(a.geometry, a.nsteps, a.nj, a.ni, a.Rf, a.beta, a.Lw, a.aoa,
            a.scheme, a.cb, a.reg_r, a.nu_hyp, a.Re, a.steady, a.field_every,
            a.out, a.pert_amp, a.pert_k0, a.pert_k1, a.n_surf, a.mode,
            a.restart, a.save_every)
