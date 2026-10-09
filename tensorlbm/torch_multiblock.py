"""Torch (GPU-ready) multi-block overset driver for the three-element airfoil.

Mirrors tensorlbm.multiblock.run_multiblock but solves each block with the
PyTorch operators in tensorlbm.torch_cgrid (t_conv / t_diffusion / t_forces /
build_torch_poisson / TorchMGSolver / t_project).  The inter-block overset
COUPLING reuses the SAME validated numpy Overset exchange: every step the torch
velocity fields are pulled to numpy mirror buffers, tensorlbm.overset.Overset
fills hole/fringe (donor bilinear) and farfield (free stream), and the result is
pushed back to torch.  This keeps the only backend difference in the per-block
solver, so the numpy<->torch consistency check (P3) isolates exactly that.

Device / dtype are chosen by the caller (CPU float64 for cross-validation, or
CUDA float32 for speed).  Forces are normalised by the main chord, exactly like
the numpy driver, so the two backends are directly comparable.
"""
import numpy as np
import torch
from tensorlbm import curve_grid as CG
from tensorlbm import torch_cgrid as TC
from tensorlbm.cgrid_gen import make_three_element_cgrid
from tensorlbm.overset import Overset, cell_centers


def make_torch_block_from_arrays(X, Y, wall, recv, Lref, device, dtype):
    """Build a TorchBlock from node arrays (so the overset masks match the numpy
    build exactly).  Metric is computed by the validated numpy generator and
    tensorised; _np_block is kept for the MG solver's ILU assembly."""
    nb = CG.make_cblock(X, Y, periodic_i=False, U=1.0, Lref=Lref, wall_mask=wall)
    blk = TC.TorchBlock(nb.ni, nb.nj, device, dtype)
    blk.periodic_i = False
    blk.J = torch.as_tensor(nb.J, device=device, dtype=dtype)
    blk.Axi_x = torch.as_tensor(nb.Axi_x, device=device, dtype=dtype)
    blk.Axi_y = torch.as_tensor(nb.Axi_y, device=device, dtype=dtype)
    blk.Axi_len = torch.as_tensor(nb.Axi_len, device=device, dtype=dtype)
    blk.Aef_x = torch.as_tensor(nb.Aef_x, device=device, dtype=dtype)
    blk.Aef_y = torch.as_tensor(nb.Aef_y, device=device, dtype=dtype)
    blk.Aef_len = torch.as_tensor(nb.Aef_len, device=device, dtype=dtype)
    blk.hxi_f = torch.as_tensor(nb.hxi_f, device=device, dtype=dtype)
    blk.heta_f = torch.as_tensor(nb.heta_f, device=device, dtype=dtype)
    blk.xc = torch.as_tensor(nb.xc, device=device, dtype=dtype)
    blk.yc = torch.as_tensor(nb.yc, device=device, dtype=dtype)
    blk.wall = torch.as_tensor(nb.wall, device=device, dtype=torch.bool)
    blk.recv = torch.as_tensor(np.asarray(recv), device=device, dtype=torch.bool)
    blk.Lref = Lref
    blk._np_block = nb
    return blk


class NPBuf:
    """Plain numpy mirror for Overset.exchange (holds writable u/v plus X/Y/wall/
    recv).  Overset indexes b['u'] etc.; in-place writes hit the numpy arrays."""

    def __init__(self, X, Y, wall, recv, u, v, name):
        self.X = X
        self.Y = Y
        self.wall = wall
        self.recv = recv
        self.u = u
        self.v = v
        self.name = name

    def __getitem__(self, k):
        return getattr(self, k)

    def __setitem__(self, k, v):
        setattr(self, k, v)

    def get(self, k, default=None):
        return getattr(self, k, default)


def make_np_exchange_buffers(geom_dict, aoa_deg, U=1.0, Re=100.0, n_fringe=1,
                             Rhole=0.08):
    """Build the numpy Overset exchange buffers + the Overset system from the
    element geometry.  Returns (bufs, ov, blocks, oblocks) where bufs[i].recv is
    the rebuilt hole|fringe|farfield mask (also used to set the torch block recv)."""
    blocks, oblocks = _numpy_blocks_for_geom(geom_dict, aoa_deg, U, Re)
    cu, cv = U * np.cos(np.deg2rad(aoa_deg)), U * np.sin(np.deg2rad(aoa_deg))
    bufs = []
    for ob in oblocks:
        b = ob.blk
        ni, nj = b.ni, b.nj
        bufs.append(NPBuf(ob.X, ob.Y, b.wall, b.recv.copy(),
                          np.full((ni, nj), cu), np.full((ni, nj), cv), b.name))
    ov = Overset(bufs, n_fringe=n_fringe, U=U, aoa_deg=aoa_deg, Rhole=Rhole)
    return bufs, ov, blocks, oblocks


def _numpy_blocks_for_geom(geom_dict, aoa_deg, U, Re):
    """Lightweight numpy blocks (metric + free stream init) for dt and geometry.
    Returns (blocks, oblocks); OBlock exposes X/Y for the torch build."""
    from tensorlbm.multiblock import make_te_blocks
    return make_te_blocks(geom_dict, aoa_deg, U=U, Re=Re)


def compute_dt_torch(blocks, U=1.0, scheme="ppm", nu_hyp=0.0):
    from tensorlbm.multiblock import compute_dt
    return compute_dt(blocks, U=U, scheme=scheme, nu_hyp=nu_hyp)


def run_torch_multiblock(geom_dict, nsteps=300, aoa_deg=4.0, Re=100.0,
                         scheme="ppm", cb="pair", reg_r=0.0, n_fringe=1,
                         nu_hyp=0.0, mode="adjoint", U=1.0, field_every=0,
                         out="te_torch", ramp=50, record_every=100,
                         device="cpu", dtype=torch.float64, Rhole=0.08):
    aoa = np.deg2rad(aoa_deg)
    cu0, cv0 = U * np.cos(aoa), U * np.sin(aoa)
    # numpy blocks for dt/geometry; Overset exchange buffers (validated numpy)
    np_blocks, np_oblocks = _numpy_blocks_for_geom(geom_dict, aoa_deg, U, Re)
    dt = compute_dt_torch(np_blocks, U=U, scheme=scheme, nu_hyp=nu_hyp)
    bufs, ov, _, _ = make_np_exchange_buffers(geom_dict, aoa_deg, U=U, Re=Re,
                                              n_fringe=n_fringe, Rhole=Rhole)

    # build torch blocks from the SAME X/Y, recv = overset mask
    torch_blocks = []
    for idx, (el, b, ob) in enumerate(zip(geom_dict["elements"], np_blocks, np_oblocks)):
        tb = make_torch_block_from_arrays(ob.X, ob.Y, b.wall, bufs[idx].recv,
                                          el["Lref"], device, dtype)
        tb._nu = b._nu
        tb.name = el["name"]
        tb.u = torch.full((tb.ni, tb.nj), cu0, device=device, dtype=dtype)
        tb.v = torch.full((tb.ni, tb.nj), cv0, device=device, dtype=dtype)
        tb.u[b.wall] = 0.0
        tb.v[b.wall] = 0.0
        tb.p = torch.zeros(tb.ni, tb.nj, device=device, dtype=dtype)
        # farfield mask (hole|fringe kept as donor velocity, never clamped to
        # free stream -- mirrors the numpy driver which clamps ONLY far after
        # the predictor, so the projection sees the same near-seam ustar).
        far = ov.masks[idx]["farfield"]
        tb.farfield = torch.as_tensor(np.asarray(far), device=device, dtype=torch.bool)
        torch_blocks.append(tb)
    poissons = [TC.build_torch_poisson(tb, reg_r=reg_r, dt=dt, mode=mode, nu=tb._nu)
                for tb in torch_blocks]

    print(f"[mb-torch] device={device} dtype={dtype} blocks={len(torch_blocks)} "
          f"aoa={aoa_deg} Re={Re} scheme={scheme} mode={mode} reg_r={reg_r} "
          f"n_fringe={n_fringe}")
    for tb in torch_blocks:
        print(f"  {tb.name}: ni={tb.ni} nj={tb.nj} "
              f"solved={int((~tb.recv).sum())} recv={int(tb.recv.sum())} "
              f"nu={tb._nu:.3e}")
    print(f"  dt={dt:.3e}")

    cd_hist, cl_hist, t_hist = [], [], []
    nan = False
    for step in range(1, nsteps + 1):
        Ueff = U * min(1.0, step / ramp)
        cu, cv = Ueff * np.cos(aoa), Ueff * np.sin(aoa)
        # 1. overset exchange (torch -> numpy mirror -> exchange -> torch)
        for i, tb in enumerate(torch_blocks):
            bufs[i].u = tb.u.detach().cpu().numpy()
            bufs[i].v = tb.v.detach().cpu().numpy()
        ov.exchange(U=Ueff, aoa_deg=aoa_deg)
        # exchanged torch snapshot (donor/free-stream at recv, prior solved elsewhere)
        ex_u = [torch.as_tensor(bufs[i].u, device=device, dtype=dtype)
                for i in range(len(torch_blocks))]
        ex_v = [torch.as_tensor(bufs[i].v, device=device, dtype=dtype)
                for i in range(len(torch_blocks))]
        for i, tb in enumerate(torch_blocks):
            wall = tb.wall
            # 2. predictor (explicit, torch).
            # Mirror the numpy driver EXACTLY: numpy's Overset.exchange() writes
            # the donor/free-stream velocity IN PLACE into nb.u, so the numpy
            # predictor reads the freshly-exchanged field (hole/fringe = donor,
            # farfield = ramped free stream).  We must do the same -- load the
            # exchanged velocity into the recv cells of a clone BEFORE the
            # predictor, otherwise torch reads the STALE tb.u in hole/fringe and
            # the projection sees a different near-seam ustar (the original third-
            # element P3 mismatch).  Then clamp ONLY the far field after the
            # predictor, leaving hole/fringe carrying the predictor's flux.
            u_in = tb.u.clone(); u_in[tb.recv] = ex_u[i][tb.recv]
            v_in = tb.v.clone(); v_in[tb.recv] = ex_v[i][tb.recv]
            duc, dvc = TC.t_conv(tb, u_in, v_in, scheme=scheme)
            du, dv = TC.t_diffusion(tb, u_in, v_in)
            if scheme == "central" and nu_hyp > 0.0:
                du = du + nu_hyp * TC.t_lap(tb, TC.t_lap(tb, u_in))
                dv = dv + nu_hyp * TC.t_lap(tb, TC.t_lap(tb, v_in))
            ustar = u_in + dt * (-duc + du)
            vstar = v_in + dt * (-dvc + dv)
            ustar = ustar.clone()
            vstar = vstar.clone()
            ustar[wall] = 0.0
            vstar[wall] = 0.0
            # ONLY the far field is clamped to the (ramped) free stream, exactly
            # like the numpy driver.  The hole/fringe cells keep the donor-
            # interpolated velocity carrying the predictor's flux, so the Chorin
            # projection sees the same near-seam ustar on both backends (clamping
            # ALL recv here was the third-element P3 backend mismatch: it reset
            # hole/fringe to the pre-predictor donor velocity and made the torch
            # Cl drift ~40% below numpy).
            ustar[tb.farfield] = ex_u[i][tb.farfield]
            vstar[tb.farfield] = ex_v[i][tb.farfield]
            tb.u, tb.v = ustar, vstar
            # 3. projection (torch MG)
            TC.t_project(tb, dt, poissons[i])
            if cb != "none":
                tb.p = TC._t_checkerboard(tb.p)
                tb.u = TC._t_checkerboard(tb.u)
                tb.v = TC._t_checkerboard(tb.v)

        # 4. re-impose overset BCs with the POST-PROJECTION velocity (exact mirror
        #    of the numpy driver, run_multiblock step 4): re-exchange hole/fringe
        #    <- donor/free-stream, re-pin wall + recv, then pin interface pressure
        #    continuity so the two-way Chimera seam does not round-trip a residual
        #    divergence.  Without exchange_pressure() the torch seam drifts exactly
        #    like the numpy one used to, breaking P3 backend consistency.
        for i, tb in enumerate(torch_blocks):
            bufs[i].u = tb.u.detach().cpu().numpy()
            bufs[i].v = tb.v.detach().cpu().numpy()
        ov.exchange(U=Ueff, aoa_deg=aoa_deg)
        for i, tb in enumerate(torch_blocks):
            eu = torch.as_tensor(bufs[i].u, device=device, dtype=dtype)
            ev = torch.as_tensor(bufs[i].v, device=device, dtype=dtype)
            tb.u = tb.u.clone(); tb.u[tb.wall] = 0.0; tb.u[tb.recv] = eu[tb.recv]
            tb.v = tb.v.clone(); tb.v[tb.wall] = 0.0; tb.v[tb.recv] = ev[tb.recv]
            bufs[i].p = tb.p.detach().cpu().numpy()
        ov.exchange_pressure()
        for i, tb in enumerate(torch_blocks):
            tb.p = torch.as_tensor(bufs[i].p, device=device, dtype=dtype)

        if not all(torch.isfinite(tb.u).all() and torch.isfinite(tb.v).all()
                   and torch.isfinite(tb.p).all() for tb in torch_blocks):
            nan = True
            print(f"  NaN at step {step}")
            break

        if step >= ramp and step % record_every == 0:
            Fxt, Fyt = 0.0, 0.0
            for tb in torch_blocks:
                Fx, Fy = TC.t_forces(tb, tb._nu, U)
                Fxt += Fx
                Fyt += Fy
            Lref = max(b.Lref for b in torch_blocks)
            Fdrag = Fxt * np.cos(aoa) + Fyt * np.sin(aoa)
            Flift = -Fxt * np.sin(aoa) + Fyt * np.cos(aoa)
            q = 0.5 * U * U * Lref
            cd, cl = Fdrag / q, Flift / q
            t = step * dt
            cd_hist.append(cd)
            cl_hist.append(cl)
            t_hist.append(t)
            if step % (record_every * 5) == 0 or step == nsteps:
                print(f"  step{step:5d} t={t:7.3f} Cd={cd:+.4f} Cl={cl:+.4f}")
        if field_every > 0 and (step % field_every == 0 or step == nsteps):
            _render_torch_multiblock(torch_blocks, geom_dict, out, step, aoa_deg)

    if nan:
        print("RESULT: NaN")
        return dict(nan=True, cd=cd_hist, cl=cl_hist, t=t_hist, blocks=torch_blocks)
    cd = np.array(cd_hist)
    cl = np.array(cl_hist)
    t = np.array(t_hist)
    n0 = max(1, len(cd) // 3)
    cd_mean = float(np.mean(cd[n0:])) if cd.size else float("nan")
    cl_mean = float(np.mean(cl[n0:])) if cl.size else float("nan")
    print(f"\nRESULT torch multiblock aoa={aoa_deg} Re={Re} mode={mode} "
          f"device={device}")
    print(f"  Cd_mean={cd_mean:.4f}  Cl_mean={cl_mean:+.4f}  "
          f"(window t>={t[n0]:.3f})")
    return dict(nan=False, cd=cd_hist, cl=cl_hist, t=t_hist, cd_mean=cd_mean,
                cl_mean=cl_mean, blocks=torch_blocks, dt=dt)


def _render_torch_multiblock(torch_blocks, geom_dict, out, step, aoa_deg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata
    fig, axes = plt.subplots(1, len(torch_blocks), figsize=(6 * len(torch_blocks), 5))
    if len(torch_blocks) == 1:
        axes = [axes]
    for ax, tb, el in zip(axes, torch_blocks, geom_dict["elements"]):
        xc = tb.xc.detach().cpu().numpy()
        yc = tb.yc.detach().cpu().numpy()
        u = tb.u.detach().cpu().numpy()
        v = tb.v.detach().cpu().numpy()
        xmin, xmax = xc.min(), xc.max()
        ymin, ymax = yc.min(), yc.max()
        gx = np.linspace(xmin, xmax, 220)
        gy = np.linspace(ymin, ymax, 160)
        GX, GY = np.meshgrid(gx, gy)
        pts = np.column_stack([xc.ravel(), yc.ravel()])
        spd = np.sqrt(u ** 2 + v ** 2)
        zu = griddata(pts, spd.ravel(), (GX, GY), method="linear")
        im = ax.imshow(zu, origin="lower", extent=[xmin, xmax, ymin, ymax],
                       cmap="viridis", vmin=0, vmax=1.4)
        fig.colorbar(im, ax=ax, label="|u|")
        recv = tb.recv.detach().cpu().numpy()
        ax.scatter(xc[recv].ravel(), yc[recv].ravel(), s=2, c="red", alpha=0.4,
                   marker=".", label="recv")
        ax.plot(xc[:, 0], yc[:, 0], "k-", lw=1.2, label="wall")
        ax.set_aspect("equal")
        ax.set_title(f"{el['name']} |u| (step {step})")
        ax.legend(loc="upper right", fontsize=7)
    plt.tight_layout()
    fname = f"/workspace/{out}_step{step:05d}.png"
    fig.savefig(fname, dpi=120)
    plt.close()
    print(f"  rendered {fname}")


if __name__ == "__main__":
    te = make_three_element_cgrid()
    run_torch_multiblock(te, nsteps=300, aoa_deg=4.0, Re=100.0, scheme="ppm",
                         cb="pair", reg_r=0.0, n_fringe=1, mode="adjoint",
                         Rhole=0.08, ramp=50,
                         field_every=300, device="cpu", dtype=torch.float64)
